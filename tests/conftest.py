"""Shared test fixtures.

Two vaults are available:

* ``fixture_vault`` — a small synthetic vault that deliberately reproduces the
  exact defect shapes found in the real corpus (both malformed ``related:``
  forms, stem collisions, URL-encoded links, duplicate content).
* ``real_vault`` — the actual Forge repository. Integration tests run against
  it so the engine is validated on real material, not only on ideal examples.

No test requires a live LLM. Anything that would is marked ``requires_model``
and skipped by default.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from forge.config import Settings
from forge.corpus.indexer import CorpusIndexer
from forge.llm.base import CALLS
from forge.storage.sqlite_store import SqliteStore

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def reset_call_counter():
    """Every test starts with a clean LLM call count.

    This is what makes "made zero LLM calls" assertable per-test.
    """
    CALLS.reset()
    yield
    CALLS.reset()


@pytest.fixture
def fixture_vault(tmp_path: Path) -> Path:
    """A writable copy of the synthetic vault, so tests may modify files."""
    dest = tmp_path / "vault"
    shutil.copytree(FIXTURE_VAULT, dest)
    (dest / ".git").mkdir()  # make it look like a repo root
    return dest


@pytest.fixture
def settings(fixture_vault: Path, tmp_path: Path) -> Settings:
    return Settings(vault_path=fixture_vault, state_dir=tmp_path / "state")


@pytest.fixture
def indexer(settings: Settings) -> CorpusIndexer:
    return CorpusIndexer(settings)


@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    s = SqliteStore(tmp_path / "state" / "forge.db")
    s.initialize()
    yield s
    s.close()


@pytest.fixture(scope="session")
def real_vault() -> Path:
    """The actual Forge repository."""
    if not (REPO_ROOT / "DSA").is_dir():  # pragma: no cover
        pytest.skip("real corpus not present")
    return REPO_ROOT


@pytest.fixture(scope="session")
def real_settings(real_vault: Path, tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return Settings(
        vault_path=real_vault, state_dir=tmp_path_factory.mktemp("real-state")
    )


@pytest.fixture(scope="session")
def real_index(real_settings: Settings):
    """Index of the real corpus, built once per session (it takes ~0.7s)."""
    return CorpusIndexer(real_settings).build_index()
