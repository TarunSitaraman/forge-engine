"""Configuration and vault resolution.

The tests that matter here are the negative ones. Forge resolves the vault
implicitly when nothing is passed, and the failure mode of getting that wrong is
silent: an arbitrary directory is treated as a vault, a ``.forge/`` is written
into it, and the run reports success. So "refuses to guess" is asserted as
explicitly as "finds the right one".
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from forge import config as config_module
from forge.config import (
    CLOUD_PRESETS,
    ConfigError,
    Settings,
    _find_vault_root,
    _resolve_vault_root,
    env_value,
    read_env_file,
)


# --------------------------------------------------------------------------
# _find_vault_root
# --------------------------------------------------------------------------


def test_finds_repo_root_from_a_nested_directory(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)

    assert _find_vault_root(nested) == tmp_path.resolve()


def test_finds_repo_root_when_started_at_the_root_itself(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()

    assert _find_vault_root(tmp_path) == tmp_path.resolve()


def test_git_may_be_a_file_not_a_directory(tmp_path: Path) -> None:
    """Worktrees and submodules use a ``.git`` *file*. Both count as a root."""
    (tmp_path / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x")

    assert _find_vault_root(tmp_path) == tmp_path.resolve()


def test_returns_none_rather_than_a_fallback_when_there_is_no_repo(tmp_path: Path) -> None:
    """The signature is ``Path | None`` precisely so callers must handle this."""
    nested = tmp_path / "no" / "repo" / "here"
    nested.mkdir(parents=True)

    assert _find_vault_root(nested) is None


# --------------------------------------------------------------------------
# _resolve_vault_root
# --------------------------------------------------------------------------


def test_the_engines_own_checkout_is_never_taken_as_the_vault(monkeypatch, tmp_path: Path) -> None:
    """Resolution ignores the module's location, so the engine cannot index itself.

    A module-adjacent rule was tried first until 2026-09-01. It made sense while
    the engine lived inside the vault; once the engine had its own repository it
    matched that repository every time under an editable install, so `forge index`
    with no `FORGE_VAULT_PATH` silently indexed the engine's own `docs/` and
    exited 0. Standing in a real vault must win, and standing nowhere must raise.
    """
    module_vault = tmp_path / "engine-checkout"
    cwd_vault = tmp_path / "cwd-vault"
    for p in (module_vault, cwd_vault):
        (p / ".git").mkdir(parents=True)

    monkeypatch.setattr(
        config_module,
        "_find_vault_root",
        lambda start: module_vault if start == Path(config_module.__file__) else cwd_vault,
    )

    assert _resolve_vault_root() == cwd_vault


def test_raises_rather_than_falling_back_to_the_module(monkeypatch, tmp_path: Path) -> None:
    """No vault at the cwd is an error, even when the engine checkout is one."""
    module_vault = tmp_path / "engine-checkout"
    (module_vault / ".git").mkdir(parents=True)

    monkeypatch.setattr(
        config_module,
        "_find_vault_root",
        lambda start: module_vault if start == Path(config_module.__file__) else None,
    )

    with pytest.raises(ConfigError, match="could not locate a Forge vault"):
        _resolve_vault_root()


def test_falls_back_to_the_vault_the_user_is_standing_in(monkeypatch, tmp_path: Path) -> None:
    """A non-editable install has no vault next to it — site-packages is not one."""
    cwd_vault = tmp_path / "cwd-vault"
    (cwd_vault / ".git").mkdir(parents=True)

    monkeypatch.setattr(
        config_module,
        "_find_vault_root",
        lambda start: None if start == Path(config_module.__file__) else cwd_vault,
    )

    assert _resolve_vault_root() == cwd_vault


def test_raises_instead_of_silently_using_the_current_directory(monkeypatch) -> None:
    """The regression this module exists for.

    Falling back to the working directory meant `forge index` in, say,
    ~/Downloads would index that directory, write a `.forge/` into it, and print
    a success line. Failing loudly is the only acceptable behaviour.
    """
    monkeypatch.setattr(config_module, "_find_vault_root", lambda start: None)

    with pytest.raises(ConfigError) as exc:
        _resolve_vault_root()

    message = str(exc.value)
    assert "could not locate a Forge vault" in message
    # The error has to be actionable, not merely correct.
    assert "FORGE_VAULT_PATH" in message


# --------------------------------------------------------------------------
# Settings.load
# --------------------------------------------------------------------------


def test_explicit_argument_wins_over_everything(fixture_vault: Path, monkeypatch) -> None:
    monkeypatch.setenv("FORGE_VAULT_PATH", str(Path.home()))

    settings = Settings.load(fixture_vault)

    assert settings.vault_path == fixture_vault.resolve()


def test_environment_variable_is_used_when_no_argument_is_passed(
    fixture_vault: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FORGE_VAULT_PATH", str(fixture_vault))

    settings = Settings.load()

    assert settings.vault_path == fixture_vault.resolve()


def test_load_propagates_the_resolution_failure(monkeypatch) -> None:
    """`Settings.load` must not convert "no vault found" into a working default."""
    monkeypatch.delenv("FORGE_VAULT_PATH", raising=False)
    monkeypatch.setattr(config_module, "_find_vault_root", lambda start: None)

    with pytest.raises(ConfigError):
        Settings.load()


def test_state_dir_defaults_inside_the_vault(fixture_vault: Path, monkeypatch) -> None:
    monkeypatch.delenv("FORGE_STATE_DIR", raising=False)

    settings = Settings.load(fixture_vault)

    assert settings.state_dir == (fixture_vault / ".forge").resolve()
    assert settings.db_path == settings.state_dir / "forge.db"


def test_a_vault_path_that_is_not_a_directory_fails_at_startup(tmp_path: Path) -> None:
    """Fail fast: a bad path is an error now, not a surprise mid-index."""
    not_a_dir = tmp_path / "file.md"
    not_a_dir.write_text("# not a vault")

    with pytest.raises(ConfigError):
        Settings.load(not_a_dir)


def test_state_dir_may_not_be_the_vault_root(fixture_vault: Path) -> None:
    """`.forge/` is deletable derived state; making it the vault would arm a footgun."""
    with pytest.raises(ConfigError):
        Settings.load(fixture_vault, state_dir=fixture_vault)


def test_environment_is_not_consulted_for_credentials(fixture_vault: Path, monkeypatch) -> None:
    """Only the *name* of the credential variable is configuration."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-never-be-stored")

    settings = Settings.load(fixture_vault)

    assert settings.llm.cloud.api_key_env == "ANTHROPIC_API_KEY"
    assert "sk-should-never-be-stored" not in settings.model_dump_json()


def test_os_environ_untouched_by_load(fixture_vault: Path) -> None:
    """Loading settings reads the environment; it never mutates it."""
    before = dict(os.environ)

    Settings.load(fixture_vault)

    assert dict(os.environ) == before


def test_cloud_token_budget_reads_the_environment(fixture_vault: Path, monkeypatch) -> None:
    """Open-weights models need a lower ceiling than the Anthropic-shaped default."""
    monkeypatch.setenv("FORGE_CLOUD_MAX_TOKENS", "4096")

    settings = Settings.load(fixture_vault)

    assert settings.llm.cloud.max_tokens == 4096


def test_cloud_vendor_and_endpoint_are_configurable(fixture_vault: Path, monkeypatch) -> None:
    """Pointing Forge at an open-weights host is configuration, not a code change."""
    monkeypatch.setenv("FORGE_CLOUD_VENDOR", "openai")
    monkeypatch.setenv("FORGE_CLOUD_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("FORGE_CLOUD_API_KEY_ENV", "GROQ_API_KEY")
    monkeypatch.setenv("FORGE_CLOUD_BASE_URL", "https://api.groq.com/openai")

    cloud = Settings.load(fixture_vault).llm.cloud

    assert (cloud.vendor, cloud.model) == ("openai", "llama-3.3-70b-versatile")
    assert cloud.api_key_env == "GROQ_API_KEY"
    assert cloud.base_url == "https://api.groq.com/openai"


# --------------------------------------------------------------------------
# per-machine settings file
# --------------------------------------------------------------------------


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "forge.env"
    monkeypatch.setenv("FORGE_ENV_FILE", str(path))
    return path


def test_a_missing_settings_file_is_not_an_error(env_file: Path) -> None:
    assert read_env_file() == {}


def test_settings_file_supplies_configuration(env_file: Path, fixture_vault: Path) -> None:
    env_file.write_text("FORGE_LLM_PROVIDER=cloud\nFORGE_CLOUD_MODEL=qwen3-32b\n")

    settings = Settings.load(fixture_vault)

    assert settings.llm.provider == "cloud"
    assert settings.llm.cloud.model == "qwen3-32b"


def test_the_process_environment_wins_over_the_file(
    env_file: Path, fixture_vault: Path, monkeypatch
) -> None:
    """So a single command can be overridden without editing the file."""
    env_file.write_text("FORGE_CLOUD_MODEL=from-file\n")
    monkeypatch.setenv("FORGE_CLOUD_MODEL", "from-environment")

    assert Settings.load(fixture_vault).llm.cloud.model == "from-environment"


def test_loading_the_file_never_mutates_the_environment(env_file: Path, fixture_vault: Path) -> None:
    """A settings file must not leak into every process this one later spawns."""
    env_file.write_text("FORGE_CLOUD_MODEL=qwen3-32b\nSOME_SECRET=shhh\n")
    before = dict(os.environ)

    Settings.load(fixture_vault)

    assert dict(os.environ) == before
    assert "SOME_SECRET" not in os.environ


def test_comments_blanks_quotes_and_export_are_handled(env_file: Path) -> None:
    env_file.write_text(
        "\n"
        "# a comment\n"
        "  \n"
        "export FORGE_CLOUD_MODEL=exported\n"
        'FORGE_CLOUD_BASE_URL="https://example.test"\n'
        "FORGE_CLOUD_API_KEY_ENV='QUOTED_KEY'\n"
    )

    values = read_env_file()

    assert values["FORGE_CLOUD_MODEL"] == "exported"
    assert values["FORGE_CLOUD_BASE_URL"] == "https://example.test"
    assert values["FORGE_CLOUD_API_KEY_ENV"] == "QUOTED_KEY"


def test_a_malformed_line_names_the_file_and_line(env_file: Path) -> None:
    env_file.write_text("FORGE_CLOUD_MODEL=fine\nthis is not an assignment\n")

    with pytest.raises(ConfigError) as exc:
        read_env_file()

    assert "forge.env:2" in str(exc.value)


def test_a_key_may_live_in_the_settings_file(env_file: Path, monkeypatch) -> None:
    """The credential is resolved through the same layered lookup, at call time."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    env_file.write_text("GROQ_API_KEY=gsk-from-file\n")

    assert env_value("GROQ_API_KEY") == "gsk-from-file"
    # ...and it is still not in the environment.
    assert "GROQ_API_KEY" not in os.environ


# --------------------------------------------------------------------------
# cloud presets
# --------------------------------------------------------------------------


def test_a_preset_fills_the_endpoint_and_credential_variable(
    fixture_vault: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FORGE_CLOUD_PRESET", "groq")
    monkeypatch.setenv("FORGE_CLOUD_MODEL", "llama-3.3-70b-versatile")

    cloud = Settings.load(fixture_vault).llm.cloud

    assert cloud.vendor == "openai"
    assert cloud.base_url == "https://api.groq.com/openai"
    assert cloud.api_key_env == "GROQ_API_KEY"
    assert cloud.max_tokens == 4096


def test_explicit_values_override_a_preset(fixture_vault: Path, monkeypatch) -> None:
    """A preset is a default, not a mode — every field stays overridable."""
    monkeypatch.setenv("FORGE_CLOUD_PRESET", "groq")
    monkeypatch.setenv("FORGE_CLOUD_MODEL", "qwen3-32b")
    monkeypatch.setenv("FORGE_CLOUD_API_KEY_ENV", "MY_OWN_KEY")
    monkeypatch.setenv("FORGE_CLOUD_MAX_TOKENS", "2048")

    cloud = Settings.load(fixture_vault).llm.cloud

    assert cloud.api_key_env == "MY_OWN_KEY"
    assert cloud.max_tokens == 2048
    assert cloud.base_url == "https://api.groq.com/openai"  # still from the preset


def test_an_unknown_preset_fails_loudly_with_the_list(fixture_vault: Path, monkeypatch) -> None:
    """Falling back to Anthropic defaults would point the request at the wrong host."""
    monkeypatch.setenv("FORGE_CLOUD_PRESET", "gorq")

    with pytest.raises(ConfigError) as exc:
        Settings.load(fixture_vault)

    message = str(exc.value)
    assert "unknown FORGE_CLOUD_PRESET" in message
    assert "groq" in message


def test_a_preset_without_a_model_says_which_knob_is_missing(
    fixture_vault: Path, monkeypatch
) -> None:
    """A preset supplies an endpoint; choosing the model is always the user's call."""
    monkeypatch.setenv("FORGE_CLOUD_PRESET", "groq")
    monkeypatch.delenv("FORGE_CLOUD_MODEL", raising=False)

    with pytest.raises(ConfigError) as exc:
        Settings.load(fixture_vault)

    assert "FORGE_CLOUD_MODEL" in str(exc.value)


def test_every_preset_is_openai_compatible_and_complete(fixture_vault: Path, monkeypatch) -> None:
    """Guards the table itself: a half-filled preset is a broken endpoint."""
    monkeypatch.setenv("FORGE_CLOUD_MODEL", "some-model")
    for name in CLOUD_PRESETS:
        monkeypatch.setenv("FORGE_CLOUD_PRESET", name)

        cloud = Settings.load(fixture_vault).llm.cloud

        assert cloud.vendor == "openai", name
        assert cloud.base_url.startswith("http"), name
        assert not cloud.base_url.endswith("/v1"), f"{name}: base_url is the root above /v1"
        assert cloud.api_key_env, name
        assert cloud.max_tokens > 0, name


def test_no_preset_keeps_the_anthropic_defaults(fixture_vault: Path, monkeypatch) -> None:
    monkeypatch.delenv("FORGE_CLOUD_PRESET", raising=False)

    cloud = Settings.load(fixture_vault).llm.cloud

    assert (cloud.vendor, cloud.model) == ("anthropic", "claude-sonnet-5")


def test_the_shipped_example_settings_file_parses(monkeypatch) -> None:
    """The template users copy must stay loadable, not just readable."""
    example = Path(__file__).resolve().parents[2] / "config" / "forge.env.example"
    monkeypatch.setenv("FORGE_ENV_FILE", str(example))

    values = read_env_file()

    assert values, "example file parsed to nothing — every line got commented out?"
    # Its active profile is the GPU box; the rest are commented alternatives.
    assert values["FORGE_MODEL_DEFAULT"]
    assert all(k.isupper() for k in values), values


class TestPresetCeilingsFitTheirFreeTier:
    """A preset's max_tokens must fit the host's budget, not the model's capacity.

    Groq counts reserved output against a per-minute limit, so an 8192 ceiling
    on a free tier capped at 8,000 TPM rejected every request before a single
    prompt token was added — measured 2026-08-29, a ~400-token prompt reported
    as 8,595 requested. Cerebras has the same trap through context rather than
    rate, and that one was already reasoned about in the table.

    Only the two hosts whose limits have actually been observed are asserted
    here. The others may well have the same problem, but this repo does not put
    a number on a budget nobody measured — a sweeping assertion over all
    presets would be inventing free-tier limits for hosts never tested.
    """

    def test_groq_stays_under_its_measured_tpm_limit(self):
        """8,000 TPM observed, and Groq counts reserved output against it."""
        from forge.config import CLOUD_PRESETS

        assert CLOUD_PRESETS["groq"]["max_tokens"] <= 4096

    def test_cerebras_stays_under_its_documented_context_cap(self):
        """8K context covering input and output together, per the table's note."""
        from forge.config import CLOUD_PRESETS

        assert CLOUD_PRESETS["cerebras"]["max_tokens"] <= 4096
