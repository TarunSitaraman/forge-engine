"""An interactive shell for Forge: `forge shell`.

The one design rule here: **a slash command *is* a Forge command.** This module
keeps no registry of its own. `/index` runs the same `index` the CLI runs, with
the same options and the same output, because it dispatches into the very same
typer group. A second list of commands would drift from the first one the day
somebody adds a command and forgets this file, and a shell that silently lacks
a command is worse than no shell.

Three consequences worth knowing:

* Every command gains a slash form for free, including ones added later.
* Anything you can pass on the command line works here — `/graph path A B`,
  `/diagnostics links --limit 100`, `/index --json`.
* Bare text with no leading slash is routed to `ask`, which is the common case
  and the reason to sit in a shell rather than retyping `forge` each time.

The header deliberately does **not** probe the provider. Reachability costs a
network round trip, and a header that stalls for a timeout on every redraw is
worse than one that says only what it knows for free. `/status` is the command
that actually asks.

Read-only, like the rest of the engine: this module adds no write path to the
vault. It only reaches commands that already exist.
"""

from __future__ import annotations

import atexit
import shlex
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

import typer

from ..config import Settings, default_env_file

#: Commands that belong to the shell rather than to the CLI. `shell` itself is
#: refused: nesting a REPL inside a REPL gives two prompts reading one stdin,
#: and the inner one wins in a way nobody expects.
BUILTIN_HELP = ("help", "?")
BUILTIN_QUIT = ("quit", "exit", "q")
BUILTIN_CLEAR = ("clear", "cls")
REFUSED = ("shell",)

#: Readline history, kept beside the settings file rather than in the vault —
#: the vault is content, and a shell history is machine state.
HISTORY_LIMIT = 1000


def history_path() -> Path:
    """`~/.config/forge/shell_history`, honouring `XDG_CONFIG_HOME`."""
    return default_env_file().parent / "shell_history"


class Kind(Enum):
    EMPTY = "empty"
    QUIT = "quit"
    HELP = "help"
    CLEAR = "clear"
    REFUSED = "refused"
    COMMAND = "command"


@dataclass(frozen=True)
class Action:
    """What a line of input means. Parsing is separated from doing so that it
    can be tested without a terminal, which is the only part with real logic."""

    kind: Kind
    argv: tuple[str, ...] = ()
    message: str = ""


def parse(line: str, known: Sequence[str] = ()) -> Action:
    """Turn one line of input into an :class:`Action`.

    `known` is the set of real command names. It is only used to give a better
    message for an unknown slash command — dispatch itself is left to the CLI,
    which already reports usage errors properly and does it in one place.
    """
    text = line.strip()
    if not text:
        return Action(Kind.EMPTY)

    if not text.startswith("/"):
        # The whole line, not a shlex split: a question is prose and contains
        # apostrophes and quotes that shlex would either eat or choke on.
        return Action(Kind.COMMAND, ("ask", text))

    body = text[1:].strip()
    if not body:
        return Action(Kind.EMPTY)

    try:
        argv = shlex.split(body)
    except ValueError as exc:  # unbalanced quote
        return Action(Kind.HELP, message=f"could not parse: {exc}")
    if not argv:
        return Action(Kind.EMPTY)

    head = argv[0].lower()
    if head in BUILTIN_QUIT:
        return Action(Kind.QUIT)
    if head in BUILTIN_HELP:
        return Action(Kind.HELP)
    if head in BUILTIN_CLEAR:
        return Action(Kind.CLEAR)
    if head in REFUSED:
        return Action(
            Kind.REFUSED,
            message=f"/{head} is not available inside the shell; you are already in it",
        )
    if known and head not in known:
        close = [n for n in known if n.startswith(head)]
        hint = f" Did you mean /{close[0]}?" if len(close) == 1 else ""
        return Action(Kind.REFUSED, message=f"unknown command /{head}.{hint} Try /help")
    return Action(Kind.COMMAND, tuple(argv))


def command_names(app: typer.Typer) -> list[str]:
    """Every command the CLI exposes, sorted. The single source of truth."""
    group = typer.main.get_command(app)
    return sorted(getattr(group, "commands", {}).keys())


def visible_names(names: Sequence[str]) -> list[str]:
    """What to offer in help and completion.

    `REFUSED` commands stay in `known` so that typing one gets the explanation
    rather than "unknown command", but listing something the shell will not run
    is a promise it breaks. Offer only what works.
    """
    return [n for n in names if n not in REFUSED]


def _ansi(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def render_header(settings: Settings, files: int, indexed: int, width: int = 78) -> str:
    """The coloured bar. Plain ANSI rather than rich, so it composes with the
    readline prompt below it without either of them miscounting the other."""
    provider = settings.llm.provider
    model = settings.llm.models.get("extraction") or "?"
    if provider == "cloud":
        model = settings.llm.cloud.model or model

    bar = "─" * width
    name = _ansi(" FORGE ", "1;97;44")
    stale = "" if indexed == files else _ansi(f"  (indexed {indexed})", "33")
    lines = [
        _ansi(bar, "36"),
        f"{name} {_ansi(str(settings.vault_path), '96')}",
        f"  {_ansi(f'{files} files', '92')}{stale}"
        f"   {_ansi(f'{provider}:{model}', '95')}",
        _ansi(bar, "36"),
        f"  {_ansi('/help', '93')} for commands · "
        f"{_ansi('/quit', '93')} to leave · plain text asks a question",
        "",
    ]
    return "\n".join(lines)


def render_help(names: Sequence[str], width: int = 78) -> str:
    """Commands in columns, plus the shell's own verbs."""
    shown = visible_names(names)
    out = [_ansi("Commands", "1;97"), ""]
    per_row = max(1, width // 18)
    for i in range(0, len(shown), per_row):
        row = shown[i : i + per_row]
        out.append("  " + "".join(f"/{n:<17}" for n in row).rstrip())
    out += [
        "",
        _ansi("Shell", "1;97"),
        "  /help /quit /clear",
        "",
        "  Anything without a leading slash is passed to " + _ansi("ask", "93") + ".",
        "  Options work as on the command line: " + _ansi("/diagnostics links --limit 100", "90"),
        "",
    ]
    return "\n".join(out)


def _install_readline(names: Sequence[str]) -> None:
    """History and tab completion, best-effort.

    Wrapped because `readline` is not guaranteed on every platform and a missing
    line editor must degrade to a plain prompt rather than refuse to start.
    """
    try:
        import readline
    except ImportError:  # pragma: no cover - platform dependent
        return

    path = history_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            readline.read_history_file(str(path))
        readline.set_history_length(HISTORY_LIMIT)
        atexit.register(_save_history, str(path))
    except OSError:  # pragma: no cover - unwritable home
        pass

    options = [f"/{n}" for n in visible_names(names)] + [
        f"/{n}" for n in BUILTIN_HELP + BUILTIN_QUIT + BUILTIN_CLEAR
    ]

    def complete(text: str, state: int) -> str | None:
        matches = [o for o in options if o.startswith(text)]
        return matches[state] if state < len(matches) else None

    readline.set_completer(complete)
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")


def _save_history(path: str) -> None:  # pragma: no cover - atexit
    try:
        import readline

        readline.write_history_file(path)
    except Exception:
        pass


#: The prompt. `\001`/`\002` mark the non-printing bytes so readline computes
#: the visible width correctly — without them, editing a line longer than the
#: terminal wraps into the prompt and the cursor lands in the wrong column.
PROMPT = "\001\033[1;36m\002forge\001\033[0m\002 › "


def dispatch(app: typer.Typer, argv: Sequence[str]) -> int:
    """Run one CLI command in-process and return its exit code.

    `standalone_mode=False` stops the CLI from calling `sys.exit`, which would
    take the shell down with it. Everything is caught: a command that raises
    must end the command, not the session.
    """
    group = typer.main.get_command(app)
    try:
        group.main(list(argv), prog_name="forge", standalone_mode=False)
        return 0
    except SystemExit as exc:  # typer.Exit and friends still raise this
        return int(exc.code or 0)
    except typer.Exit as exc:
        return int(exc.exit_code or 0)
    except typer.Abort:
        return 130
    except KeyboardInterrupt:
        print()
        return 130
    except Exception as exc:
        # Usage errors from the CLI land here too. Print and carry on; the
        # alternative is a REPL that dies on a typo.
        print(_ansi(f"{type(exc).__name__}: {exc}", "91"))
        return 1


def run(
    app: typer.Typer,
    settings: Settings,
    files: int,
    indexed: int,
    reader: Callable[[str], str] = input,
    writer: Callable[[str], None] = print,
) -> int:
    """The loop. `reader`/`writer` are injected so this is testable headlessly."""
    names = command_names(app)
    _install_readline(names)
    writer(render_header(settings, files, indexed))

    while True:
        try:
            line = reader(PROMPT)
        except (EOFError, KeyboardInterrupt):
            writer("")
            return 0

        action = parse(line, names)
        if action.kind is Kind.EMPTY:
            continue
        if action.kind is Kind.QUIT:
            return 0
        if action.kind is Kind.CLEAR:
            writer("\033[2J\033[H" + render_header(settings, files, indexed))
            continue
        if action.kind is Kind.HELP:
            if action.message:
                writer(_ansi(action.message, "91"))
            writer(render_help(names))
            continue
        if action.kind is Kind.REFUSED:
            writer(_ansi(action.message, "91"))
            continue
        dispatch(app, action.argv)
