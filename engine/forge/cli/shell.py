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
import os
import re
import shlex
import sys
import time
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


#: What a bare `/` offers before anything is typed. Ordered by how often a
#: session actually reaches for them, not alphabetically — a quick bar sorted
#: A-Z puts `activate` first, which nobody wants and nobody needs.
FAVOURITES = (
    "ask",
    "index",
    "status",
    "diagnostics",
    "search",
    "concept",
    "graph",
    "corpus-stats",
)

#: How many suggestions to show. Enough to be useful, few enough to scan.
SUGGESTION_LIMIT = 8


def command_help(app: typer.Typer) -> dict[str, str]:
    """Command name -> its one-line description.

    Read off the typer group like the names are, so a command's description is
    its own docstring and there is nothing to keep in sync. A hand-written table
    here would be wrong the first time somebody rewords a docstring.
    """
    group = typer.main.get_command(app)
    out: dict[str, str] = {}
    for name, cmd in getattr(group, "commands", {}).items():
        text = (getattr(cmd, "help", None) or getattr(cmd, "short_help", None) or "").strip()
        out[name] = text.splitlines()[0].strip() if text else ""
    return out


def suggestions(
    prefix: str,
    names: Sequence[str],
    limit: int = SUGGESTION_LIMIT,
) -> list[str]:
    """Commands to offer for a partially typed slash command.

    A bare `/` gets the favourites, because an alphabetical list of everything
    is a worse answer to "what can I do" than a short list of what is actually
    used. Once something is typed, prefix matches come first and substring
    matches follow — typing `eval` should find `retrieval-eval`, which a
    prefix-only match would hide.
    """
    available = visible_names(names)
    prefix = prefix.strip().lower()
    if not prefix:
        return [n for n in FAVOURITES if n in available][:limit]

    starts = [n for n in available if n.startswith(prefix)]
    contains = [n for n in available if prefix in n and n not in starts]
    return (starts + contains)[:limit]


def visible_names(names: Sequence[str]) -> list[str]:
    """What to offer in help and completion.

    `REFUSED` commands stay in `known` so that typing one gets the explanation
    rather than "unknown command", but listing something the shell will not run
    is a promise it breaks. Offer only what works.
    """
    return [n for n in names if n not in REFUSED]


#: The wordmark, one string per row so it can be revealed a row at a time.
BANNER = (
    "███████╗ ██████╗ ██████╗  ██████╗ ███████╗",
    "██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝",
    "█████╗  ██║   ██║██████╔╝██║  ███╗█████╗  ",
    "██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  ",
    "██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗",
    "╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝",
)

#: Top-to-bottom gradient over the wordmark. Six rows, six shades.
BANNER_COLOURS = ("38;5;39", "38;5;38", "38;5;44", "38;5;43", "38;5;37", "38;5;30")

#: Per-row delay for the reveal. Six rows plus the panel is under a fifth of a
#: second in total — enough to read as motion, short enough that nobody waits.
FRAME_SECONDS = 0.028

#: Commands that print nothing until they finish, so a spinner cannot collide
#: with their output. `ask` is the whole reason the shell exists; the others
#: print progress as they go and are deliberately left alone.
QUIET_COMMANDS = frozenset({"ask"})

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _ansi(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def _visible_width(text: str) -> int:
    """Printable width, ignoring colour escapes.

    Box drawing needs the width the terminal will show, not `len`. Getting this
    wrong is invisible until a coloured value lands in a padded cell and the
    right-hand border walks off by the length of the escape sequence.
    """
    return len(_ANSI_RE.sub("", text))


def wants_colour(stream=None) -> bool:
    """Colour only when it can be seen and has not been refused.

    `NO_COLOR` is honoured because it is the convention, and a pipe gets plain
    text because escape codes in a log file are noise. This is the first place
    in the engine to emit colour, so the rule is set here.
    """
    if os.environ.get("NO_COLOR"):
        return False
    stream = stream or sys.stdout
    try:
        return bool(stream.isatty())
    except Exception:  # pragma: no cover - exotic stream
        return False


def wants_animation(stream=None) -> bool:
    """Motion needs a terminal, and never runs when colour is refused."""
    return wants_colour(stream)


def render_banner(colour: bool = True) -> str:
    rows = [
        _ansi(row, BANNER_COLOURS[i]) if colour else row
        for i, row in enumerate(BANNER)
    ]
    return "\n".join("  " + r for r in rows)


#: `  ` + a 9-column label + `  ` + the two borders and their inner padding.
_ROW_OVERHEAD = 15


def _fit(value: str, room: int) -> str:
    """Trim a value to the space available, keeping the informative end.

    Paths are the reason this exists: a vault under a long home directory is
    wider than the panel, and without trimming the row simply grows past the
    border it is supposed to sit inside. The tail is kept because the last
    segments identify the vault; the head is a prefix every row would share.
    """
    if room <= 1 or len(value) <= room:
        return value
    return "…" + value[-(room - 1) :]


def _row(label: str, value: str, colour: bool, value_code: str, width: int) -> str:
    """One line inside the panel, padded to exactly `width` visible columns."""
    value = _fit(value, width - _ROW_OVERHEAD)
    lab = _ansi(f"{label:<9}", "38;5;245") if colour else f"{label:<9}"
    val = _ansi(value, value_code) if colour else value
    body = f"  {lab} {val}"
    pad = " " * max(0, width - _visible_width(body) - 3)
    edge = _ansi("│", "38;5;30") if colour else "│"
    return f"{edge}{body}{pad} {edge}"


def render_header(
    settings: Settings, files: int, indexed: int, width: int = 62, colour: bool | None = None
) -> str:
    """The panel under the wordmark.

    Drawn by hand rather than with rich: the widths here are computed against
    `_visible_width`, and hand-drawing keeps the panel and the readline prompt
    below it from disagreeing about how wide anything is.
    """
    if colour is None:
        colour = wants_colour()

    provider = settings.llm.provider
    model = settings.llm.models.get("extraction") or "?"
    if provider == "cloud":
        model = settings.llm.cloud.model or model

    corpus = f"{files} files"
    if indexed != files:
        corpus += f"  · indexed {indexed}"

    def edge(left: str, right: str) -> str:
        line = left + "─" * (width - 2) + right
        return _ansi(line, "38;5;30") if colour else line

    rows = [
        edge("╭", "╮"),
        _row("vault", str(settings.vault_path), colour, "38;5;51", width),
        _row("corpus", corpus, colour, "38;5;84" if indexed == files else "38;5;214", width),
        _row("model", f"{provider} · {model}", colour, "38;5;177", width),
        edge("╰", "╯"),
    ]
    hint_keys = ("/help", "/quit")
    if colour:
        hint = (
            f"  {_ansi(hint_keys[0], '38;5;220')} for commands · "
            f"{_ansi(hint_keys[1], '38;5;220')} to leave · "
            f"{_ansi('plain text asks a question', '38;5;245')}"
        )
    else:
        hint = f"  {hint_keys[0]} for commands · {hint_keys[1]} to leave · plain text asks a question"
    rows += ["", hint, ""]
    return "\n".join(rows)


def show_intro(
    settings: Settings,
    files: int,
    indexed: int,
    writer: Callable[[str], None] = print,
    animate: bool | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Wordmark then panel, revealed a row at a time when there is a terminal.

    `animate` and `sleep` are injected so the timing can be tested without
    actually waiting, and so a pipe gets the same text with no delay at all.
    """
    if animate is None:
        animate = wants_animation()
    colour = wants_colour()

    if animate:
        for i, row in enumerate(BANNER):
            writer("  " + _ansi(row, BANNER_COLOURS[i]))
            sleep(FRAME_SECONDS)
    else:
        writer(render_banner(colour))
    writer("")
    writer(render_header(settings, files, indexed, colour=colour))


def render_help(
    names: Sequence[str], width: int = 78, helps: dict[str, str] | None = None
) -> str:
    """Commands with what each one does.

    A bare list of names answers "what exists" but not "which one do I want",
    which is the question anyone opening help is actually asking.
    """
    shown = visible_names(names)
    out = [_ansi("Commands", "1;97"), ""]
    if helps:
        room = width - 20
        for n in shown:
            desc = helps.get(n, "")
            if len(desc) > room:
                desc = desc[: room - 1].rstrip() + "…"
            out.append(f"  {_ansi(f'/{n:<16}', '38;5;110')} {_ansi(desc, '38;5;245')}")
    else:
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


def _nullctx():
    from contextlib import nullcontext

    return nullcontext()


def _spinner(label: str):
    """A rich status for commands that print nothing until they finish.

    Returned as a context manager so the caller does not care whether one is
    actually running. rich ships with typer, so this costs no new dependency.
    """
    from contextlib import nullcontext

    if not wants_animation():
        return nullcontext()
    try:
        from rich.console import Console

        return Console().status(f"[dim]{label}[/dim]", spinner="dots")
    except Exception:  # pragma: no cover - rich absent or non-capable terminal
        return nullcontext()


def dispatch(app: typer.Typer, argv: Sequence[str]) -> int:
    """Run one CLI command in-process and return its exit code.

    `standalone_mode=False` stops the CLI from calling `sys.exit`, which would
    take the shell down with it. Everything is caught: a command that raises
    must end the command, not the session.
    """
    group = typer.main.get_command(app)
    quiet = bool(argv) and argv[0] in QUIET_COMMANDS
    try:
        with _spinner("thinking") if quiet else _nullctx():
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
    animate: bool | None = None,
) -> int:
    """The loop. `reader`/`writer` are injected so this is testable headlessly."""
    names = command_names(app)
    helps = command_help(app)
    _install_readline(names)
    show_intro(settings, files, indexed, writer=writer, animate=animate)

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
            writer("\033[2J\033[H")
            show_intro(settings, files, indexed, writer=writer, animate=False)
            continue
        if action.kind is Kind.HELP:
            if action.message:
                writer(_ansi(action.message, "91"))
            writer(render_help(names, helps=helps))
            continue
        if action.kind is Kind.REFUSED:
            writer(_ansi(action.message, "91"))
            continue
        dispatch(app, action.argv)
