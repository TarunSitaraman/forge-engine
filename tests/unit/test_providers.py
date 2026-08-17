"""Provider abstraction: Ollama, cloud, mock, and the rules between them.

Two rules carry real weight here and are tested as behaviour rather than
documented as intent:

* **No silent downgrade.** An unavailable provider produces an explicit
  unavailability. Forge must never answer "is my knowledge wrong?" with
  whatever model happens to be reachable.
* **No credential leakage.** The API key lives in an environment variable and
  is read at call time. It must not appear in config dumps, health output,
  provenance, or logs.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError as PydanticValidationError

from forge.config import CloudSettings, LLMSettings, OllamaSettings, Settings
from forge.llm import (
    CloudProvider,
    MockProvider,
    OllamaProvider,
    get_provider,
    provider_identity,
    require_provider,
)
from forge.llm.base import (
    CompletionRequest,
    LLMError,
    Message,
    ProviderUnavailable,
    StructuredOutputError,
)
from forge.llm.cloud import ANTHROPIC, OPENAI_COMPATIBLE


def request(text: str = "hello") -> CompletionRequest:
    return CompletionRequest(
        messages=[Message(role="system", content="sys"), Message(role="user", content=text)],
        model_role="analysis",
    )


class Schema(__import__("pydantic").BaseModel):
    answer: str


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


class TestProviderSelection:
    def test_mock_is_selectable(self, fixture_vault, tmp_path):
        settings = Settings(
            vault_path=fixture_vault,
            state_dir=tmp_path / "s",
            llm=LLMSettings(provider="mock"),
        )

        assert isinstance(get_provider(settings), MockProvider)

    def test_ollama_is_selectable(self, fixture_vault, tmp_path):
        settings = Settings(
            vault_path=fixture_vault,
            state_dir=tmp_path / "s",
            llm=LLMSettings(provider="ollama"),
        )

        assert isinstance(get_provider(settings), OllamaProvider)

    def test_cloud_is_selectable(self, fixture_vault, tmp_path):
        settings = Settings(
            vault_path=fixture_vault,
            state_dir=tmp_path / "s",
            llm=LLMSettings(provider="cloud"),
        )

        assert isinstance(get_provider(settings), CloudProvider)

    def test_an_unknown_provider_is_rejected_at_config_time(self):
        """Fail fast: an unknown provider is a startup error, not a surprise later."""
        with pytest.raises(PydanticValidationError):
            LLMSettings(provider="telepathy")  # type: ignore[arg-type]

    def test_remote_ollama_is_configurable(self, fixture_vault, tmp_path):
        """The whole point: Forge runs where the model cannot."""
        settings = Settings(
            vault_path=fixture_vault,
            state_dir=tmp_path / "s",
            llm=LLMSettings(
                provider="ollama",
                ollama=OllamaSettings(base_url="http://192.168.1.50:11434"),
            ),
        )

        assert get_provider(settings).base_url == "http://192.168.1.50:11434"

    def test_the_legacy_base_url_still_wins_when_set(self, fixture_vault, tmp_path):
        """Phases 1-3 and FORGE_OLLAMA_URL configured `llm.base_url` directly."""
        settings = Settings(
            vault_path=fixture_vault,
            state_dir=tmp_path / "s",
            llm=LLMSettings(provider="ollama", base_url="http://legacy:11434"),
        )

        assert get_provider(settings).base_url == "http://legacy:11434"

    def test_provider_identity_names_provider_and_model(self, fixture_vault, tmp_path):
        settings = Settings(
            vault_path=fixture_vault,
            state_dir=tmp_path / "s",
            llm=LLMSettings(
                provider="cloud", cloud=CloudSettings(vendor="anthropic", model="claude-sonnet-5")
            ),
        )

        assert provider_identity(get_provider(settings)) == (
            "cloud:anthropic",
            "claude-sonnet-5",
        )


class TestNoSilentDowngrade:
    def test_require_provider_raises_rather_than_substituting(
        self, fixture_vault, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        settings = Settings(
            vault_path=fixture_vault,
            state_dir=tmp_path / "s",
            llm=LLMSettings(provider="cloud"),
        )

        with pytest.raises(ProviderUnavailable):
            require_provider(settings)

    def test_require_provider_returns_a_healthy_provider(
        self, fixture_vault, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-a-real-key")
        settings = Settings(
            vault_path=fixture_vault,
            state_dir=tmp_path / "s",
            llm=LLMSettings(provider="cloud"),
        )

        assert isinstance(require_provider(settings), CloudProvider)

    def test_an_unbound_role_is_an_explicit_failure(self, fixture_vault, tmp_path):
        """Better an error than quietly answering with the wrong model."""
        settings = Settings(
            vault_path=fixture_vault,
            state_dir=tmp_path / "s",
            llm=LLMSettings(provider="ollama"),
        )
        provider = get_provider(settings)

        with pytest.raises(LLMError):
            provider.resolve_model("nonexistent_role")


# --------------------------------------------------------------------------
# cloud provider
# --------------------------------------------------------------------------


def cloud(monkeypatch, handler, **kw) -> CloudProvider:
    """A CloudProvider whose transport is a stub, so no network is touched."""
    provider = CloudProvider(api_key_env="FORGE_TEST_KEY", **kw)
    monkeypatch.setenv("FORGE_TEST_KEY", "sk-test-value")
    provider._client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url=provider.base_url
    )
    return provider


class TestCloudCredentials:
    def test_missing_credential_is_unavailable_not_an_error(self, monkeypatch):
        monkeypatch.delenv("FORGE_TEST_KEY", raising=False)
        provider = CloudProvider(api_key_env="FORGE_TEST_KEY")

        reachable, detail = provider.health()

        assert reachable is False
        assert "FORGE_TEST_KEY" in detail

    def test_health_never_reveals_the_key(self, monkeypatch):
        monkeypatch.setenv("FORGE_TEST_KEY", "sk-super-secret-value")
        provider = CloudProvider(api_key_env="FORGE_TEST_KEY")

        reachable, detail = provider.health()

        assert reachable is True
        assert "sk-super-secret-value" not in detail

    def test_the_key_is_not_stored_in_configuration(self):
        """Only the variable *name* is configuration. A key in YAML is a key in Git."""
        settings = CloudSettings()

        dumped = json.dumps(settings.model_dump())

        assert "api_key_env" in dumped
        assert "api_key" not in CloudSettings.model_fields

    def test_a_blank_environment_variable_counts_as_missing(self, monkeypatch):
        monkeypatch.setenv("FORGE_TEST_KEY", "   ")
        provider = CloudProvider(api_key_env="FORGE_TEST_KEY")

        assert provider.api_key is None
        assert provider.health()[0] is False

    def test_completing_without_a_credential_raises_unavailable(self, monkeypatch):
        monkeypatch.delenv("FORGE_TEST_KEY", raising=False)
        provider = CloudProvider(api_key_env="FORGE_TEST_KEY")

        with pytest.raises(ProviderUnavailable):
            provider.complete(request())


class TestCloudWireFormats:
    def test_anthropic_request_shape(self, monkeypatch):
        seen: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["url"] = str(req.url)
            seen["headers"] = dict(req.headers)
            seen["body"] = json.loads(req.content)
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "hi"}],
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                },
            )

        provider = cloud(monkeypatch, handler, vendor=ANTHROPIC, model="claude-sonnet-5")
        response = provider.complete(request("question"))

        assert seen["url"].endswith("/v1/messages")
        assert seen["headers"]["x-api-key"] == "sk-test-value"
        assert seen["headers"]["anthropic-version"]
        assert seen["body"]["system"] == "sys"
        assert seen["body"]["messages"] == [{"role": "user", "content": "question"}]
        assert response.text == "hi"
        assert response.prompt_tokens == 5

    def test_openai_compatible_request_shape(self, monkeypatch):
        seen: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["url"] = str(req.url)
            seen["headers"] = dict(req.headers)
            seen["body"] = json.loads(req.content)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "hello"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                },
            )

        provider = cloud(monkeypatch, handler, vendor=OPENAI_COMPATIBLE, model="gpt-x")
        response = provider.complete(request())

        assert seen["url"].endswith("/v1/chat/completions")
        assert seen["headers"]["authorization"] == "Bearer sk-test-value"
        assert seen["body"]["messages"][0]["role"] == "system"
        assert response.text == "hello"

    def test_anthropic_payload_omits_sampling_parameters(self, monkeypatch):
        """The regression that made every cloud call a 400.

        Forge asks for ``temperature=0.0`` everywhere for determinism. Current
        Anthropic models reject non-default sampling parameters outright, so
        forwarding it meant the cloud path could never complete a single call —
        which is why it stayed unmeasured. Assert the whole family, not just
        ``temperature``, so adding one back is a test failure.
        """
        seen: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

        provider = cloud(monkeypatch, handler, vendor=ANTHROPIC, model="claude-sonnet-5")
        provider.complete(request())

        assert "temperature" not in seen["body"]
        assert "top_p" not in seen["body"]
        assert "top_k" not in seen["body"]

    def test_openai_compatible_still_sends_temperature(self, monkeypatch):
        """The removal is Anthropic-specific — OpenAI-shaped gateways accept it."""
        seen: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        provider = cloud(monkeypatch, handler, vendor=OPENAI_COMPATIBLE, model="gpt-x")
        provider.complete(request())

        assert seen["body"]["temperature"] == 0.0

    def test_token_budget_leaves_room_for_thinking(self, monkeypatch):
        """`max_tokens` caps thinking and response text together on current models.

        A budget sized for the JSON alone gets eaten by reasoning and truncates
        the object, which surfaces as a structured-output failure rather than as
        the budget problem it is.
        """
        seen: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

        provider = cloud(monkeypatch, handler, vendor=ANTHROPIC)
        provider.complete(request())

        assert seen["body"]["max_tokens"] >= 16000

    def test_an_explicit_request_budget_still_wins(self, monkeypatch):
        seen: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

        provider = cloud(monkeypatch, handler, vendor=ANTHROPIC)
        provider.complete(
            CompletionRequest(messages=[Message(role="user", content="q")], max_tokens=512)
        )

        assert seen["body"]["max_tokens"] == 512

    def test_openai_compatible_hoists_system_to_the_front(self, monkeypatch):
        """Open-weights chat templates commonly drop a non-leading system turn.

        `structured()` appends the schema instruction as a system message *after*
        the user turn. The Anthropic branch hoists system into its own top-level
        field so order never mattered; served through a gateway the messages hit
        the model's own chat template, and a dropped instruction would strip the
        schema silently — a 200 response that fails to parse, on every call.
        """
        seen: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        provider = cloud(monkeypatch, handler, vendor=OPENAI_COMPATIBLE, model="qwen")
        provider.complete(
            CompletionRequest(
                messages=[
                    Message(role="user", content="q"),
                    Message(role="system", content="schema instruction"),
                ]
            )
        )

        roles = [m["role"] for m in seen["body"]["messages"]]
        assert roles == ["system", "user"]
        assert seen["body"]["messages"][0]["content"] == "schema instruction"

    def test_openai_compatible_collapses_multiple_system_turns(self, monkeypatch):
        seen: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        provider = cloud(monkeypatch, handler, vendor=OPENAI_COMPATIBLE, model="qwen")
        provider.complete(
            CompletionRequest(
                messages=[
                    Message(role="system", content="first"),
                    Message(role="user", content="q"),
                    Message(role="system", content="second"),
                ]
            )
        )

        messages = seen["body"]["messages"]
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[0]["content"] == "first\n\nsecond"

    def test_the_token_budget_is_configurable_for_smaller_models(self):
        """Open-weights models cap far below a frontier model's output ceiling.

        Gateways reject an over-large `max_tokens` rather than clamping it, so
        this has to be tunable without a code change.
        """
        settings = CloudSettings(max_tokens=4096)

        assert settings.max_tokens == 4096

    def test_an_unsupported_vendor_is_rejected(self):
        with pytest.raises(LLMError, match="unsupported cloud vendor"):
            CloudProvider(vendor="mystery-inc")

    def test_the_model_is_recorded_on_the_response(self, monkeypatch):
        provider = cloud(
            monkeypatch,
            lambda r: httpx.Response(200, json={"content": [{"type": "text", "text": "x"}]}),
            model="claude-sonnet-5",
        )

        assert provider.complete(request()).model == "claude-sonnet-5"


class TestCloudFailureModes:
    def test_a_rejected_credential_is_unavailable_not_a_retry(self, monkeypatch):
        calls = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(401, json={"error": "bad key"})

        provider = cloud(monkeypatch, handler)

        with pytest.raises(ProviderUnavailable):
            provider.complete(request())
        assert calls["n"] == 1, "a bad credential must not be retried into a rate limit"

    def test_a_client_error_is_not_retried(self, monkeypatch):
        calls = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(400, json={"error": "bad request"})

        provider = cloud(monkeypatch, handler)

        with pytest.raises(LLMError):
            provider.complete(request())
        assert calls["n"] == 1

    def test_a_server_error_is_retried_then_fails(self, monkeypatch):
        calls = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503, json={"error": "overloaded"})

        provider = cloud(monkeypatch, handler, max_retries=2)

        with pytest.raises(LLMError):
            provider.complete(request())
        assert calls["n"] == 3

    def test_an_unreachable_host_is_unavailable(self, monkeypatch):
        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host")

        provider = cloud(monkeypatch, handler)

        with pytest.raises(ProviderUnavailable, match="cannot reach"):
            provider.complete(request())

    def test_malformed_structured_output_raises_after_one_repair(self, monkeypatch):
        calls = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json={"content": [{"type": "text", "text": "not json"}]})

        provider = cloud(monkeypatch, handler)

        with pytest.raises(StructuredOutputError):
            provider.structured(request(), Schema)
        assert calls["n"] == 2, "exactly one repair attempt, then a hard failure"

    def test_structured_output_recovers_fenced_json(self, monkeypatch):
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "content": [
                        {"type": "text", "text": '```json\n{"answer": "42"}\n```'}
                    ]
                },
            )

        provider = cloud(monkeypatch, handler)

        assert provider.structured(request(), Schema).answer == "42"


class TestOllamaProvider:
    def test_unreachable_ollama_reports_unavailable(self):
        provider = OllamaProvider("http://127.0.0.1:1", models={"analysis": "m"}, max_retries=0)

        reachable, detail = provider.health()

        assert reachable is False
        assert "cannot reach Ollama" in detail

    def test_health_never_raises(self):
        """`forge status` must always print something, never traceback."""
        provider = OllamaProvider("http://127.0.0.1:1", max_retries=0)

        assert provider.health()[0] is False

    def test_capabilities_survive_an_unreachable_server(self):
        provider = OllamaProvider("http://127.0.0.1:1", max_retries=0)

        assert provider.capabilities.name == "ollama"
        assert provider.capabilities.available_models == ()


# --------------------------------------------------------------------------
# ollama reasoning control
# --------------------------------------------------------------------------


def ollama(handler, **kw) -> OllamaProvider:
    """An OllamaProvider whose transport is a stub, so no network is touched."""
    provider = OllamaProvider(models={"extraction": "qwen3:8b"}, max_retries=1, **kw)
    provider._client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url=provider.base_url
    )
    return provider


def _ok(_req: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"message": {"content": "{}"}})


class TestOllamaThinking:
    """Reasoning is a model-behaviour switch, not a tuning knob.

    Qwen3 and friends reason before answering by default. For a bounded
    extraction task that reasoning is generated at full token cost and then
    discarded, which dominates wall time — but turning it off changes what the
    model does, so it is opt-in and separately identified rather than a silent
    default.
    """

    def test_unset_sends_no_think_field(self):
        seen: dict = {}
        provider = ollama(lambda r: (seen.update(json.loads(r.content)), _ok(r))[1])

        provider.complete(CompletionRequest(messages=[Message("user", "hi")]))

        assert "think" not in seen, "an unset toggle must leave the model's default alone"
        assert provider.identity_variant == ""

    @pytest.mark.parametrize("think", [True, False])
    def test_explicit_setting_is_sent_and_changes_identity(self, think):
        seen: dict = {}
        provider = ollama(lambda r: (seen.update(json.loads(r.content)), _ok(r))[1], think=think)

        provider.complete(CompletionRequest(messages=[Message("user", "hi")]))

        assert seen["think"] is think
        assert provider.identity_variant == ("+think" if think else "+nothink")

    def test_model_without_reasoning_falls_back_once_and_says_so(self, caplog):
        """A model with no reasoning mode has none to lose — but say it aloud."""
        attempts: list[bool] = []

        def handler(req: httpx.Request) -> httpx.Response:
            payload = json.loads(req.content)
            attempts.append("think" in payload)
            if "think" in payload:
                return httpx.Response(400, text='"llama3" does not support thinking')
            return _ok(req)

        provider = ollama(handler, think=False)
        provider.complete(CompletionRequest(messages=[Message("user", "hi")]))

        assert attempts == [True, False], "retried exactly once, without the field"

    def test_other_4xx_still_fails_fast(self):
        provider = ollama(lambda r: httpx.Response(400, text="unknown model"), think=False)

        with pytest.raises(LLMError):
            provider.complete(CompletionRequest(messages=[Message("user", "hi")]))

    def test_reasoning_mode_reaches_the_derivation_key(self):
        """Think-on and think-off results must never share a cache entry.

        The derivation key hashes the model id, so the variant has to survive
        the trip from provider to extractor. If it did not, a fast think-off
        run would silently serve its results to a later think-on run.
        """
        from forge.extraction import CandidateExtractor
        from forge.ingestion.derivation import extraction_key

        def key_for(think):
            extractor = CandidateExtractor(
                OllamaProvider(models={"extraction": "qwen3:8b"}, think=think)
            )
            return extraction_key(
                content_hash="h",
                processor_version=extractor.version,
                model_id=extractor.model_id(),
                prompt_version=extractor.prompt_version,
                schema_version=extractor.schema_version,
            ).value()

        assert key_for(False) != key_for(True) != key_for(None)
        assert key_for(False) != key_for(None)
