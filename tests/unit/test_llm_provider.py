"""LLM provider abstraction.

Everything here runs offline. The Ollama adapter is exercised without a server
by checking the failure path it must produce when nothing is listening — which
is the state most developers will first encounter.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from forge.config import LLMSettings, Settings
from forge.llm import (
    CALLS,
    CompletionRequest,
    LLMError,
    LLMProvider,
    Message,
    MockProvider,
    OllamaProvider,
    ProviderUnavailable,
    StructuredOutputError,
    extract_json,
    get_provider,
    malformed_provider,
    unavailable_provider,
)


class Answer(BaseModel):
    name: str
    score: int = 0


def req(text: str = "hello") -> CompletionRequest:
    return CompletionRequest(messages=[Message(role="user", content=text)])


class TestProtocol:
    def test_mock_satisfies_protocol(self):
        assert isinstance(MockProvider(), LLMProvider)

    def test_ollama_satisfies_protocol(self):
        assert isinstance(OllamaProvider(), LLMProvider)

    def test_get_provider_honours_config(self, tmp_path):
        (tmp_path / "v").mkdir()
        settings = Settings(
            vault_path=tmp_path / "v",
            state_dir=tmp_path / "s",
            llm=LLMSettings(provider="mock"),
        )
        assert isinstance(get_provider(settings), MockProvider)

    def test_unknown_provider_is_fatal(self, tmp_path):
        (tmp_path / "v").mkdir()
        settings = Settings(vault_path=tmp_path / "v", state_dir=tmp_path / "s")
        object.__setattr__(settings.llm, "provider", "nope")
        with pytest.raises(LLMError, match="unknown provider"):
            get_provider(settings)


class TestCallCounting:
    def test_calls_are_counted(self):
        p = MockProvider(default_response='{"name": "x"}')
        assert CALLS.count == 0
        p.complete(req())
        p.complete(req())
        assert CALLS.count == 2
        assert CALLS.by_model["mock-1"] == 2

    def test_structured_counts_one_call(self):
        p = MockProvider(default_response='{"name": "x"}')
        p.structured(req(), Answer)
        assert CALLS.count == 1


class TestStructuredOutput:
    def test_valid_json_parses(self):
        p = MockProvider(default_response='{"name": "concept", "score": 3}')
        assert p.structured(req(), Answer) == Answer(name="concept", score=3)

    def test_code_fenced_json_is_recovered(self):
        """Local models wrap JSON in fences constantly; recovering is parsing,
        so software does it rather than spending another model call."""
        p = MockProvider(default_response='Sure!\n```json\n{"name": "x"}\n```\n')
        assert p.structured(req(), Answer).name == "x"

    def test_prose_wrapped_json_is_recovered(self):
        p = MockProvider(default_response='Here you go: {"name": "y"} — hope that helps')
        assert p.structured(req(), Answer).name == "y"

    def test_malformed_output_raises_typed_error(self):
        with pytest.raises(StructuredOutputError, match="not valid JSON"):
            malformed_provider().structured(req(), Answer)

    def test_schema_violation_raises_with_detail(self):
        p = MockProvider(default_response='{"wrong_field": 1}')
        with pytest.raises(StructuredOutputError, match="did not match schema"):
            p.structured(req(), Answer)

    def test_error_carries_raw_output_for_diagnosis(self):
        p = MockProvider(default_response="not json at all")
        try:
            p.structured(req(), Answer)
        except StructuredOutputError as exc:
            assert "not json at all" in exc.raw

    @pytest.mark.parametrize(
        "text,expected",
        [
            ('{"a":1}', '{"a":1}'),
            ('```json\n{"a":1}\n```', '{"a":1}'),
            ('```\n{"a":1}\n```', '{"a":1}'),
            ('prefix {"a":1} suffix', '{"a":1}'),
            ('[{"a":1}]', '[{"a":1}]'),
        ],
    )
    def test_extract_json_shapes(self, text, expected):
        assert extract_json(text) == expected


class TestFailureModes:
    def test_unavailable_provider_is_distinguishable(self):
        """"No model running" and "model answered badly" need different fixes."""
        with pytest.raises(ProviderUnavailable):
            unavailable_provider().complete(req())

    def test_ollama_reports_unavailable_without_a_server(self):
        p = OllamaProvider("http://127.0.0.1:1", models={"extraction": "m"}, timeout=1.0)
        ok, detail = p.health()
        assert ok is False
        assert "cannot reach Ollama" in detail

    def test_ollama_health_never_raises(self):
        assert OllamaProvider("http://127.0.0.1:1", timeout=1.0).health()[0] is False

    def test_ollama_requires_a_configured_model(self):
        with pytest.raises(LLMError, match="no model configured"):
            OllamaProvider(models={}).resolve_model("extraction")

    def test_ollama_complete_raises_provider_unavailable(self):
        p = OllamaProvider("http://127.0.0.1:1", models={"extraction": "m"}, timeout=1.0)
        with pytest.raises(ProviderUnavailable):
            p.complete(req())


class TestDeterminism:
    def test_temperature_defaults_to_zero(self):
        assert req().temperature == 0.0

    def test_mock_is_reproducible(self):
        p = MockProvider(default_response='{"name": "same"}')
        assert p.structured(req(), Answer) == p.structured(req(), Answer)

    def test_mock_records_requests_for_assertion(self):
        p = MockProvider(default_response="{}")
        p.complete(req("first"))
        p.complete(req("second"))
        assert [m.messages[0].content for m in p.requests] == ["first", "second"]


class TestConfigRoles:
    def test_all_roles_must_be_bound(self):
        with pytest.raises(ValueError, match="roles without a configured model"):
            LLMSettings(models={"extraction": "m"})

    def test_model_lookup_by_role(self):
        s = LLMSettings(
            models={
                "extraction": "a",
                "analysis": "b",
                "resolution": "c",
                "synthesis": "d",
            }
        )
        assert s.model_for("analysis") == "b"
