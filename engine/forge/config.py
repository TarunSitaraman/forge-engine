"""Configuration for the Forge engine.

Settings come from environment variables prefixed ``FORGE_``, with defaults
suitable for running against this repository with nothing configured.

Two rules this module exists to enforce:

* **No model names in business logic.** Models are bound to *roles*
  (extraction, analysis, resolution, synthesis) here; the rest of the engine
  asks for a role. Swapping a model is a config change, never a code change.
* **Fail fast.** Configuration is validated at construction. A bad vault path
  or an unknown provider is an error at startup, not a surprise mid-index.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ProviderName = Literal["ollama", "cloud", "mock"]
ModelRole = Literal["extraction", "analysis", "resolution", "synthesis"]

#: Directories never walked by the corpus indexer. These are engine-owned or
#: tool-owned, and indexing them would mean the engine indexing itself.
DEFAULT_EXCLUDES: tuple[str, ...] = (
    ".git",
    ".forge",
    ".obsidian",
    ".trash",
    "engine",
    "tests",
    "scripts",
    # Engine configuration, not knowledge. `config/concept-identity.yaml`
    # lives in the vault so it is versioned with the notes, but indexing it
    # would make the engine's own settings a retrievable "source".
    "config",
    "docker",
    "node_modules",
    "__pycache__",
    ".venv",
)


class CloudSettings(BaseModel):
    """Portable inference for machines that cannot run a local model.

    **No credential is ever stored here.** ``api_key_env`` names an environment
    variable; the key itself is read at call time and never written to config,
    logs, the store, or provenance. A key in a YAML file is a key in Git.
    """

    #: Vendor identifier, e.g. "anthropic". Nothing above the provider layer
    #: branches on this — it selects a wire format, not behaviour.
    vendor: str = "anthropic"
    model: str = "claude-sonnet-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: str = "https://api.anthropic.com"
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    max_tokens: int = Field(default=2048, gt=0)
    #: Whether the vendor can be *asked* for schema-conforming JSON. Forge
    #: validates the result regardless; this only selects the request shape.
    supports_structured_output: bool = True


class OllamaSettings(BaseModel):
    """Local or LAN-remote Ollama.

    ``base_url`` is what makes a remote box usable: Forge runs on a laptop that
    cannot host a model, and points at one that can. Nothing here assumes the
    remote host is reachable — an unreachable provider is reported as
    unavailable, never worked around.
    """

    base_url: str = "http://localhost:11434"
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    #: Ollama's reasoning toggle for thinking-capable models (Qwen3, DeepSeek-R1,
    #: gpt-oss, …). ``None`` sends nothing and leaves the model's default alone,
    #: which is what every Forge measurement so far was taken under.
    #:
    #: Setting it to ``False`` is a real change to model behaviour, not a
    #: performance flag: the model stops reasoning before it answers. It is
    #: therefore opt-in, and it changes the extractor's model identity so
    #: think-on and think-off results can never share a derivation cache entry.
    think: bool | None = None


class LLMSettings(BaseModel):
    """Provider configuration. Nothing here is required for Phase 1 indexing."""

    provider: ProviderName = "ollama"
    #: Retained as the Ollama URL for backwards compatibility with Phases 1-3
    #: and `FORGE_OLLAMA_URL`. `ollama.base_url` is the preferred spelling.
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)

    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    cloud: CloudSettings = Field(default_factory=CloudSettings)

    #: Role -> model name. Business logic references the *role*.
    models: dict[str, str] = Field(
        default_factory=lambda: {
            "extraction": "llama3.1:8b",
            "analysis": "llama3.1:8b",
            "resolution": "llama3.1:8b",
            "synthesis": "llama3.1:8b",
        }
    )

    def model_for(self, role: ModelRole) -> str:
        try:
            return self.models[role]
        except KeyError as exc:  # pragma: no cover - guarded by validator
            raise ConfigError(f"No model configured for role {role!r}") from exc

    @model_validator(mode="after")
    def _all_roles_bound(self) -> LLMSettings:
        required = {"extraction", "analysis", "resolution", "synthesis"}
        missing = required - self.models.keys()
        if missing:
            raise ValueError(f"LLM roles without a configured model: {sorted(missing)}")
        return self


class Settings(BaseModel):
    """Top-level engine settings."""

    #: Repository root — the Obsidian vault. Read-only in Phase 1 (ADR-001 D2).
    vault_path: Path

    #: Derived-state directory. Deletable: everything in it rebuilds from the vault.
    state_dir: Path

    #: Directory names skipped when walking the vault.
    exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDES

    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"

    llm: LLMSettings = Field(default_factory=LLMSettings)

    @field_validator("vault_path", "state_dir")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        return v.expanduser().resolve()

    @model_validator(mode="after")
    def _check_vault(self) -> Settings:
        if not self.vault_path.is_dir():
            raise ValueError(f"vault_path is not a directory: {self.vault_path}")
        if self.state_dir == self.vault_path:
            raise ValueError("state_dir must not be the vault root")
        return self

    @property
    def db_path(self) -> Path:
        return self.state_dir / "forge.db"

    @property
    def reports_dir(self) -> Path:
        return self.state_dir / "reports"

    @classmethod
    def load(cls, vault_path: Path | str | None = None, **overrides: object) -> Settings:
        """Build settings from environment with optional explicit overrides."""
        root = Path(vault_path or os.environ.get("FORGE_VAULT_PATH") or _resolve_vault_root())
        state = Path(os.environ.get("FORGE_STATE_DIR") or (root / ".forge"))

        ollama_url = os.environ.get("FORGE_OLLAMA_URL", "http://localhost:11434")
        timeout = float(os.environ.get("FORGE_LLM_TIMEOUT", "120"))
        retries = int(os.environ.get("FORGE_LLM_MAX_RETRIES", "2"))
        llm = LLMSettings(
            provider=os.environ.get("FORGE_LLM_PROVIDER", "ollama"),  # type: ignore[arg-type]
            base_url=ollama_url,
            timeout_seconds=timeout,
            max_retries=retries,
            ollama=OllamaSettings(
                base_url=ollama_url,
                timeout_seconds=timeout,
                max_retries=retries,
                think=_optional_bool(os.environ.get("FORGE_OLLAMA_THINK")),
            ),
            # Only the *name* of the credential variable is configuration. The
            # credential itself is read at call time and never stored here.
            cloud=CloudSettings(
                vendor=os.environ.get("FORGE_CLOUD_VENDOR", "anthropic"),
                model=os.environ.get("FORGE_CLOUD_MODEL", "claude-sonnet-5"),
                api_key_env=os.environ.get("FORGE_CLOUD_API_KEY_ENV", "ANTHROPIC_API_KEY"),
                base_url=os.environ.get("FORGE_CLOUD_BASE_URL", "https://api.anthropic.com"),
                timeout_seconds=timeout,
                max_retries=retries,
            ),
            models={
                role: os.environ.get(
                    f"FORGE_MODEL_{role.upper()}",
                    os.environ.get("FORGE_MODEL_DEFAULT", "llama3.1:8b"),
                )
                for role in ("extraction", "analysis", "resolution", "synthesis")
            },
        )

        # Built as a dict so explicit overrides win rather than colliding with
        # the defaults derived above. Passing both as keyword arguments raised
        # "got multiple values for keyword argument".
        values: dict[str, object] = {
            "vault_path": root,
            "state_dir": state,
            "log_level": os.environ.get("FORGE_LOG_LEVEL", "INFO"),
            "log_format": os.environ.get("FORGE_LOG_FORMAT", "console"),
            "llm": llm,
        }
        values.update(overrides)

        try:
            return cls(**values)  # type: ignore[arg-type]
        except Exception as exc:
            raise ConfigError(str(exc)) from exc


class ConfigError(RuntimeError):
    """Raised when configuration is invalid. Always fatal at startup."""


def _optional_bool(raw: str | None) -> bool | None:
    """Parse a tri-state environment flag: unset means "leave the default alone".

    Unset and "set to something meaningless" are deliberately different from
    "set to false". An unrecognised value raises rather than silently reading as
    False, because a typo in `FORGE_OLLAMA_THINK` would otherwise quietly change
    how the model reasons.
    """
    if raw is None or raw.strip() == "":
        return None
    value = raw.strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"expected a boolean or an unset value, got {raw!r}")


def _find_vault_root(start: Path) -> Path | None:
    """Walk upward from ``start`` for a repository root (a directory containing .git).

    Returns ``None`` rather than a fallback: "no vault here" is a real answer the
    caller has to handle, and the previous silent fallback to the current
    directory is exactly the bug this signature prevents.
    """
    here = start.resolve()
    for candidate in [here, *here.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def _resolve_vault_root() -> Path:
    """Locate the vault when it was not passed explicitly or set in the environment.

    Two locations are tried, in this order:

    1. **Next to the installed module.** A source checkout or an editable install
       puts ``forge/config.py`` inside the vault repository, so this pins the CLI
       to that vault from any working directory — the behaviour a personal vault
       wants, and what ``pipx install --editable`` gives you.
    2. **Upward from the current directory.** For a non-editable install the
       module lives in ``site-packages``, so the only signal left is the vault
       the user is standing in.

    If neither finds a repository root, this raises. It must not fall back to the
    current directory: ``forge index`` would then treat an arbitrary directory as
    a vault, write a ``.forge/`` into it, and report success — silently indexing
    the wrong thing instead of saying it could not find the right thing.
    """
    for start in (Path(__file__), Path.cwd()):
        found = _find_vault_root(start)
        if found is not None:
            return found
    raise ConfigError(
        "could not locate a Forge vault. Forge looks for a directory containing "
        ".git, first next to the installed engine and then upward from the "
        "current directory, and found neither.\n"
        "Fix this by pointing Forge at the vault explicitly:\n"
        "  export FORGE_VAULT_PATH=/path/to/forge   # persist it in ~/.zshrc\n"
        "  forge index --vault /path/to/forge       # or per-command, where supported"
    )
