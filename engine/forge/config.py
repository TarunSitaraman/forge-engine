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

#: Per-machine settings file. This exists because provider configuration is a
#: property of the *machine*, not of the vault: the same checkout is a GPU box
#: on one machine and a laptop borrowing a hosted endpoint on another, and the
#: vault is shared between them by Git. Keeping it out of the repo is the point
#: — it holds an API key.
#:
#: Process environment always wins, so anything here can be overridden for a
#: single command without editing the file.
ENV_FILE_VAR = "FORGE_ENV_FILE"


def default_env_file() -> Path:
    """``~/.config/forge/forge.env``, honouring ``XDG_CONFIG_HOME``."""
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "forge" / "forge.env"


def env_file_path() -> Path:
    return Path(os.environ[ENV_FILE_VAR]) if os.environ.get(ENV_FILE_VAR) else default_env_file()


def read_env_file(path: Path | None = None) -> dict[str, str]:
    """Parse ``KEY=value`` lines. Missing file is not an error.

    Deliberately not a dotenv implementation: no interpolation, no command
    substitution, no multi-line values. A settings file that can execute is a
    settings file that can surprise you, and this one holds a credential.
    """
    target = path or env_file_path()
    try:
        raw = target.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        return {}
    except OSError as exc:  # unreadable is worth saying out loud, not swallowing
        raise ConfigError(f"cannot read {target}: {exc}") from exc

    values: dict[str, str] = {}
    for number, line in enumerate(raw.splitlines(), start=1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("export "):
            entry = entry[len("export ") :].lstrip()
        key, sep, value = entry.partition("=")
        if not sep:
            raise ConfigError(f"{target}:{number}: expected KEY=value, got {line.strip()!r}")
        key = key.strip()
        if not key:
            raise ConfigError(f"{target}:{number}: missing key before '='")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def env_value(name: str, default: str | None = None) -> str | None:
    """One layered lookup: process environment first, then the settings file.

    Used for the API key as well as for configuration, so a key placed in the
    settings file is found at call time without ever being copied into
    :data:`os.environ` — loading settings must not mutate the environment.
    """
    found = os.environ.get(name)
    if found is not None:
        return found
    return read_env_file().get(name, default)


#: Convenience presets for OpenAI-compatible hosts, which is the wire format
#: essentially every open-weights service and local server speaks.
#:
#: A preset only supplies values you have **not** set explicitly — every field
#: remains individually overridable, and any host works without a preset by
#: setting `FORGE_CLOUD_BASE_URL` directly. These are a convenience against the
#: most common setup error (a base URL with the wrong path prefix or a stray
#: `/v1`), not an endorsement or an integration: third-party endpoints can
#: change, and the explicit variables are always authoritative.
#:
#: `base_url` is the root that has `/v1/chat/completions` beneath it.
CLOUD_PRESETS: dict[str, dict[str, object]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai",
        "api_key_env": "GROQ_API_KEY",
        "max_tokens": 8192,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api",
        "api_key_env": "OPENROUTER_API_KEY",
        "max_tokens": 8192,
    },
    "together": {
        "base_url": "https://api.together.xyz",
        "api_key_env": "TOGETHER_API_KEY",
        "max_tokens": 8192,
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai",
        "api_key_env": "CEREBRAS_API_KEY",
        "max_tokens": 8192,
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference",
        "api_key_env": "FIREWORKS_API_KEY",
        "max_tokens": 8192,
    },
    # Local OpenAI-compatible servers. They ignore the credential, but one must
    # still be present, so these name a variable you can set to anything.
    "lmstudio": {
        "base_url": "http://localhost:1234",
        "api_key_env": "FORGE_LOCAL_API_KEY",
        "max_tokens": 4096,
    },
    "llama-cpp": {
        "base_url": "http://localhost:8080",
        "api_key_env": "FORGE_LOCAL_API_KEY",
        "max_tokens": 4096,
    },
    "vllm": {
        "base_url": "http://localhost:8000",
        "api_key_env": "FORGE_LOCAL_API_KEY",
        "max_tokens": 4096,
    },
}

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
    #: Caps thinking *and* response text together on current Anthropic models,
    #: which think by default — so this has to leave room for both. See
    #: `CloudProvider.__init__`.
    #:
    #: **Lower it for open-weights models.** The default is sized for a frontier
    #: model with a 128K output ceiling; a served Llama or Qwen usually caps at
    #: 4096-8192, and gateways reject a request that asks for more rather than
    #: clamping it. `FORGE_CLOUD_MAX_TOKENS` exists for exactly this.
    max_tokens: int = Field(default=16000, gt=0)
    #: Whether the vendor can be *asked* for schema-conforming JSON. Forge
    #: validates the result regardless; this only selects the request shape.
    supports_structured_output: bool = True

    @field_validator("model")
    @classmethod
    def _model_is_named(cls, v: str) -> str:
        """A preset supplies an endpoint, never a model — that choice is yours.

        Catching it here turns "which model?" into a startup error with an
        instruction, instead of an opaque rejection from the host.
        """
        if not v.strip():
            raise ValueError(
                "no cloud model configured. A preset supplies the endpoint and "
                "credential variable but not the model — set FORGE_CLOUD_MODEL "
                "to one the host serves."
            )
        return v.strip()


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
        """Build settings from the environment, with optional explicit overrides.

        Three layers, highest first: an explicit argument, the process
        environment, then the per-machine settings file. :data:`os.environ` is
        read but never written — a settings file must not leak into the
        environment of everything this process later spawns.
        """
        env = _environment()
        root = Path(vault_path or env.get("FORGE_VAULT_PATH") or _resolve_vault_root())
        state = Path(env.get("FORGE_STATE_DIR") or (root / ".forge"))

        ollama_url = env.get("FORGE_OLLAMA_URL", "http://localhost:11434")
        timeout = float(env.get("FORGE_LLM_TIMEOUT", "120"))
        retries = int(env.get("FORGE_LLM_MAX_RETRIES", "2"))
        cloud_defaults = _cloud_preset(env.get("FORGE_CLOUD_PRESET"))
        # Wrapped so a bad provider knob is a ConfigError like every other
        # configuration failure — the CLI maps that to a clean exit 2, whereas a
        # raw pydantic ValidationError would surface as a traceback.
        try:
            llm = LLMSettings(
                provider=env.get("FORGE_LLM_PROVIDER", "ollama"),  # type: ignore[arg-type]
                base_url=ollama_url,
                timeout_seconds=timeout,
                max_retries=retries,
                ollama=OllamaSettings(
                    base_url=ollama_url,
                    timeout_seconds=timeout,
                    max_retries=retries,
                    think=_optional_bool(env.get("FORGE_OLLAMA_THINK")),
                ),
                # Only the *name* of the credential variable is configuration.
                # The credential itself is read at call time, never stored here.
                cloud=CloudSettings(
                    vendor=env.get("FORGE_CLOUD_VENDOR", cloud_defaults["vendor"]),
                    model=env.get("FORGE_CLOUD_MODEL", cloud_defaults["model"]),
                    api_key_env=env.get("FORGE_CLOUD_API_KEY_ENV", cloud_defaults["api_key_env"]),
                    base_url=env.get("FORGE_CLOUD_BASE_URL", cloud_defaults["base_url"]),
                    max_tokens=int(
                        env.get("FORGE_CLOUD_MAX_TOKENS", cloud_defaults["max_tokens"])
                    ),
                    timeout_seconds=timeout,
                    max_retries=retries,
                ),
                models={
                    role: env.get(
                        f"FORGE_MODEL_{role.upper()}",
                        env.get("FORGE_MODEL_DEFAULT", "llama3.1:8b"),
                    )
                    for role in ("extraction", "analysis", "resolution", "synthesis")
                },
            )
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(_first_message(exc)) from exc

        # Built as a dict so explicit overrides win rather than colliding with
        # the defaults derived above. Passing both as keyword arguments raised
        # "got multiple values for keyword argument".
        values: dict[str, object] = {
            "vault_path": root,
            "state_dir": state,
            "log_level": env.get("FORGE_LOG_LEVEL", "INFO"),
            "log_format": env.get("FORGE_LOG_FORMAT", "console"),
            "llm": llm,
        }
        values.update(overrides)

        try:
            return cls(**values)  # type: ignore[arg-type]
        except Exception as exc:
            raise ConfigError(str(exc)) from exc


class ConfigError(RuntimeError):
    """Raised when configuration is invalid. Always fatal at startup."""


def _first_message(exc: Exception) -> str:
    """The human sentence out of a pydantic error, without the URL and traceback.

    Configuration errors are read by a person fixing a shell profile, not by a
    developer reading a stack trace.
    """
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            first = errors()[0]
            message = str(first.get("msg", "")).removeprefix("Value error, ")
            location = ".".join(str(p) for p in first.get("loc", ()))
            return f"{location}: {message}" if location else message
        except Exception:  # pragma: no cover - defensive
            pass
    return str(exc)


def _environment() -> dict[str, str]:
    """Settings-file values with the process environment layered on top."""
    merged = read_env_file()
    merged.update(os.environ)
    return merged


def _cloud_preset(name: str | None) -> dict[str, object]:
    """Resolve `FORGE_CLOUD_PRESET` to defaults, or the Anthropic defaults.

    A preset supplies *defaults only*; every field it fills is still
    individually overridable, which is what keeps it a convenience rather than a
    mode. An unknown name fails loudly with the list — silently falling back to
    Anthropic defaults would point a request at the wrong endpoint entirely.
    """
    base: dict[str, object] = {
        "vendor": "anthropic",
        "model": "claude-sonnet-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com",
        "max_tokens": 16000,
    }
    if not name:
        return base
    key = name.strip().casefold()
    if key not in CLOUD_PRESETS:
        raise ConfigError(
            f"unknown FORGE_CLOUD_PRESET {name!r}. Known presets: "
            f"{', '.join(sorted(CLOUD_PRESETS))}.\n"
            "Any other OpenAI-compatible host works without a preset — set "
            "FORGE_CLOUD_VENDOR=openai and FORGE_CLOUD_BASE_URL yourself."
        )
    # Presets describe OpenAI-compatible hosts; none of them are Anthropic.
    base.update({"vendor": "openai", "model": ""})
    base.update(CLOUD_PRESETS[key])
    return base


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
