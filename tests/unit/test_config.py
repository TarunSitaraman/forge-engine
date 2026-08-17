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
from forge.config import ConfigError, Settings, _find_vault_root, _resolve_vault_root


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


def test_prefers_the_vault_next_to_the_installed_module(monkeypatch, tmp_path: Path) -> None:
    """An editable install pins the CLI to its own checkout from any directory.

    The module location is checked first, so `cd` elsewhere does not repoint the
    vault — which is what makes `pipx install --editable` behave as a
    single-vault personal CLI.
    """
    module_vault = tmp_path / "module-vault"
    cwd_vault = tmp_path / "cwd-vault"
    for p in (module_vault, cwd_vault):
        (p / ".git").mkdir(parents=True)

    monkeypatch.setattr(
        config_module,
        "_find_vault_root",
        lambda start: module_vault if start == Path(config_module.__file__) else cwd_vault,
    )

    assert _resolve_vault_root() == module_vault


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
