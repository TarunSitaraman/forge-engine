"""A full-screen terminal UI for Forge: `forge tui`.

`forge shell` is a line-oriented REPL — it prints and scrolls like any other
command. This is the other thing: an alternate-screen application with a title
bar, a scrolling transcript, a persistent input box and a footer of key hints,
in the shape the current generation of agent CLIs has settled on.

**Textual is an optional dependency**, in the `tui` extra, alongside the `agent`
extra that carries LangGraph. The rule in `pyproject.toml` is that the core
install stays minimal and everything load-bearing is justified; a full-screen UI
is not load-bearing for indexing a vault, and `forge shell` covers the same
ground with nothing beyond the standard library and what typer already brings.
So `forge tui` asks for the extra rather than the wheel carrying it for
everybody.

Two things are shared with the shell rather than reimplemented, because a second
copy is a second thing to drift:

* `shell.parse` decides what a line means — slash command, question, or builtin.
* `shell.command_names` is the command list, read off the typer group.

What is *not* shared is dispatch. The shell calls the CLI in-process and lets it
print to stdout. Here stdout is captured line by line and posted to the
transcript from a worker thread, so a long command streams as it goes instead of
freezing the interface and then dumping.
"""

from __future__ import annotations

import io
import threading
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable

import typer

from ..config import Settings
from .shell import Kind, command_help, command_names, parse, suggestions, visible_names

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


REQUIRED = "textual"

INSTALL_HINT = (
    "forge tui needs the optional TUI extra.\n"
    "  pip install 'forge-engine[tui]'\n"
    "or, for a pipx install:\n"
    "  pipx inject forge-engine textual\n"
    "\n"
    "`forge shell` needs nothing extra and offers the same commands."
)


def textual_available() -> bool:
    """Whether the optional extra is installed."""
    try:  # pragma: no cover - trivial
        import textual  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass(frozen=True)
class Stats:
    """What the title bar shows on the right.

    `llm_calls` is Forge's analogue of the token counter these interfaces
    usually carry. It is the more honest number for this engine: the whole
    design asserts that deterministic work makes zero model calls, so a counter
    that stays at 0 through an index is the system telling the truth about
    itself.
    """

    files: int
    indexed: int
    spans: int
    llm_calls: int

    def render(self) -> str:
        stale = "" if self.indexed == self.files else f" [#d7875f]({self.indexed} indexed)[/]"
        return (
            f"[#5f8787]{self.files} files[/]{stale}   "
            f"[#5f8787]{self.spans:,} spans[/]   "
            f"[#5f8787]{self.llm_calls} llm calls[/]"
        )


class LineWriter(io.TextIOBase):
    """A stdout stand-in that hands finished lines to a callback.

    Commands print with `print`, which arrives here in fragments — a line's text
    and its newline are separate writes. Buffering to the newline is what makes
    the transcript show whole lines rather than a word at a time.
    """

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self._buffer = ""
        self._lock = threading.Lock()

    def write(self, text: str) -> int:  # type: ignore[override]
        with self._lock:
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self._emit(line)
        return len(text)

    def flush(self) -> None:  # type: ignore[override]
        with self._lock:
            if self._buffer:
                self._emit(self._buffer)
                self._buffer = ""

    def writable(self) -> bool:  # type: ignore[override]
        return True


def run_command(app_typer: typer.Typer, argv: Iterable[str], emit: Callable[[str], None]) -> int:
    """Run one CLI command, streaming its output through `emit`.

    Both streams are captured: structlog writes warnings to stderr, and a retry
    or a provider warning belongs in the transcript rather than scribbled over
    the alternate screen, where it would corrupt the layout until a redraw.
    """
    writer = LineWriter(emit)
    group = typer.main.get_command(app_typer)
    try:
        with redirect_stdout(writer), redirect_stderr(writer):  # type: ignore[arg-type]
            try:
                group.main(list(argv), prog_name="forge", standalone_mode=False)
                code = 0
            except SystemExit as exc:
                code = int(exc.code or 0)
            except typer.Exit as exc:
                code = int(exc.exit_code or 0)
            except typer.Abort:
                code = 130
            except Exception as exc:
                print(f"{type(exc).__name__}: {exc}")
                code = 1
    finally:
        writer.flush()
    return code


def build_app(app_typer: typer.Typer, settings: Settings, stats: Stats):
    """Construct the Textual application.

    A factory rather than a module-level class so importing this module does not
    require Textual — `forge tui` can then fail with an instruction instead of
    an ImportError traceback, and the rest of the CLI is unaffected.
    """
    from textual import work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical
    from textual.widgets import Input, RichLog, Static

    names = command_names(app_typer)
    shown = visible_names(names)
    helps = command_help(app_typer)

    class ForgeTUI(App):
        CSS = """
        Screen { background: #0b0e14; }

        #titlebar {
            height: 1;
            background: #0b0e14;
            color: #c8d3f5;
            padding: 0 1;
        }
        #rule { height: 1; color: #1f2430; }

        #transcript {
            border-left: solid #1f2430;
            padding: 0 1;
            background: #0b0e14;
            scrollbar-size-vertical: 1;
        }

        #palette {
            height: auto;
            max-height: 10;
            background: #11151c;
            color: #8f9aae;
            padding: 0 1;
            display: none;
        }
        #palette.open { display: block; }

        #promptwrap {
            height: auto;
            border-left: solid #4d9de0;
            background: #11151c;
            padding: 0 1;
        }
        Input {
            background: #11151c;
            border: none;
            padding: 0;
            color: #c8d3f5;
        }
        Input:focus { border: none; }

        #statusline {
            height: 1;
            background: #11151c;
            color: #4d5566;
            padding: 0 1;
        }
        #keys {
            height: 1;
            background: #0b0e14;
            color: #3b4252;
            padding: 0 1;
        }
        """

        BINDINGS = [
            ("ctrl+c", "quit", "quit"),
            ("ctrl+l", "clear", "clear"),
            ("escape", "interrupt", "interrupt"),
            # priority, because the Input below has focus and would otherwise
            # swallow these before the quick bar ever sees them. Each action
            # falls through to normal behaviour when the bar is closed.
            Binding("down", "suggest_next", "", show=False, priority=True),
            Binding("up", "suggest_prev", "", show=False, priority=True),
            Binding("tab", "suggest_accept", "", show=False, priority=True),
        ]

        def __init__(self) -> None:
            super().__init__()
            self._stats = stats
            self._busy = False
            self._suggested: list[str] = []
            self._cursor = 0
            self._running = ""
            self._elapsed = 0
            self._timer = None

        # -- layout --------------------------------------------------------

        def compose(self) -> ComposeResult:
            yield Static(self._titlebar(), id="titlebar")
            yield Static("─" * 200, id="rule")
            yield RichLog(id="transcript", markup=True, wrap=True, auto_scroll=True)
            yield Static("", id="palette")
            with Vertical(id="promptwrap"):
                yield Input(placeholder="ask a question, or /command", id="prompt")
            yield Static(self._statusline(), id="statusline")
            yield Static(self._keys(), id="keys")

        def _titlebar(self) -> str:
            name = str(settings.vault_path).rsplit("/", 1)[-1] or "vault"
            return f"[b #c8d3f5]# {name}[/]   {self._stats.render()}"

        def _statusline(self) -> str:
            provider = settings.llm.provider
            model = settings.llm.models.get("extraction") or "?"
            if provider == "cloud":
                model = settings.llm.cloud.model or model
            return f"[#4d9de0]Forge[/]  [#8f9aae]{provider}[/]  [#4d5566]{model}[/]"

        def _keys(self) -> str:
            return (
                "[#3b4252]esc[/] interrupt    "
                "[#3b4252]tab[/] complete    "
                "[#3b4252]ctrl+l[/] clear    "
                "[#3b4252]ctrl+c[/] quit"
            )

        # -- transcript ----------------------------------------------------

        def on_mount(self) -> None:
            log = self.query_one("#transcript", RichLog)
            log.write("[#4d9de0]Forge[/] — knowledge OS, read-only with respect to the vault")
            log.write(f"[#4d5566]{settings.vault_path}[/]")
            log.write("")
            log.write("[#4d5566]Type a question, or /help for commands.[/]")
            log.write("")
            self.query_one("#prompt", Input).focus()

        def say(self, markup: str) -> None:
            self.query_one("#transcript", RichLog).write(markup)

        def refresh_titlebar(self) -> None:
            self.query_one("#titlebar", Static).update(self._titlebar())

        # -- the quick bar --------------------------------------------------

        def on_input_changed(self, event: Input.Submitted | object) -> None:
            """Offer commands while a slash command is still being typed.

            Only before the first space: once there are arguments, the user has
            chosen the command and a list of other commands is in the way.
            """
            value = self.query_one("#prompt", Input).value
            if value.startswith("/") and " " not in value:
                self._suggested = suggestions(value[1:], names)
                self._cursor = 0
            else:
                self._suggested = []
            self._draw_palette()

        def _draw_palette(self) -> None:
            bar = self.query_one("#palette", Static)
            if not self._suggested:
                bar.remove_class("open")
                bar.update("")
                return
            rows = []
            for i, name in enumerate(self._suggested):
                desc = helps.get(name, "")
                if i == self._cursor:
                    rows.append(
                        f"[#4d9de0]▸[/] [b #c8d3f5]/{name:<16}[/] [#8f9aae]{desc}[/]"
                    )
                else:
                    rows.append(f"  [#6c7889]/{name:<16}[/] [#4d5566]{desc}[/]")
            bar.update("\n".join(rows))
            bar.add_class("open")

        def _move(self, delta: int) -> bool:
            if not self._suggested:
                return False
            self._cursor = (self._cursor + delta) % len(self._suggested)
            self._draw_palette()
            return True

        def action_suggest_next(self) -> None:
            self._move(1)

        def action_suggest_prev(self) -> None:
            self._move(-1)

        def action_suggest_accept(self) -> None:
            """Tab completes; Enter still submits whatever is typed.

            Keeping them separate means Enter never runs something other than
            what is on screen, which is the surprise worth avoiding.
            """
            if not self._suggested:
                return
            box = self.query_one("#prompt", Input)
            box.value = f"/{self._suggested[self._cursor]} "
            box.cursor_position = len(box.value)
            self._suggested = []
            self._draw_palette()

        # -- actions -------------------------------------------------------

        def action_clear(self) -> None:
            self.query_one("#transcript", RichLog).clear()

        def action_interrupt(self) -> None:
            if self._busy:
                self.workers.cancel_all()
                self.say(f"[#d7875f]interrupted {self._running}[/]")
                self._busy = False
                self._stop_timer()
            else:
                self.query_one("#prompt", Input).value = ""

        def on_input_submitted(self, event: Input.Submitted) -> None:
            line = event.value
            event.input.value = ""
            self._suggested = []
            self._draw_palette()
            if not line.strip():
                return

            action = parse(line, names)
            if action.kind is Kind.QUIT:
                self.exit()
                return
            if action.kind is Kind.EMPTY:
                return

            # The worker is `exclusive`, so starting a second command would
            # cancel the first — silently, and with nothing on screen to say
            # either had been running. Refusing is the honest answer: the user
            # can wait or press esc, and either way knows what is happening.
            if self._busy:
                self.say(
                    f"[#d7875f]still running {self._running} "
                    f"({self._elapsed}s) — esc to interrupt[/]"
                )
                return

            # Echo what was asked, so the transcript reads as a conversation.
            self.say(f"[#4d9de0]›[/] [#c8d3f5]{line}[/]")

            if action.kind is Kind.HELP:
                if action.message:
                    self.say(f"[#d75f5f]{action.message}[/]")
                self._show_help()
                return
            if action.kind is Kind.REFUSED:
                self.say(f"[#d75f5f]{action.message}[/]")
                return
            if action.kind is Kind.CLEAR:
                self.action_clear()
                return

            self._run(list(action.argv))

        def _show_help(self) -> None:
            """Names alone answer "what exists"; the question being asked is
            "which one do I want"."""
            self.say("")
            for name in shown:
                self.say(f"[#6c7889]/{name:<16}[/] [#4d5566]{helps.get(name, '')}[/]")
            self.say("")
            self.say(
                "[#4d5566]Text without a slash is a question. "
                "Type / for suggestions, tab to complete. /quit to leave.[/]"
            )
            self.say("")

        # -- running a command ---------------------------------------------

        @work(thread=True, exclusive=True)
        def _run(self, argv: list[str]) -> None:
            """Commands run off the UI thread so the interface stays alive.

            Output is posted line by line rather than collected and dumped: a
            long index or an extraction run should show progress, which is the
            whole reason for a streaming writer rather than a StringIO.
            """
            from ..llm.base import CALLS

            self.call_from_thread(self._set_busy, True, argv)

            produced = 0

            def emit(line: str) -> None:
                nonlocal produced
                produced += 1
                # `call_from_thread` raises if it is ever reached from the loop
                # thread. Nothing should redirect stdout there, but a stray
                # print must not take the worker down with it.
                try:
                    self.call_from_thread(self.say, _escape(line))
                except RuntimeError:
                    pass

            code = run_command(app_typer, argv, emit)

            try:
                calls = CALLS.count
            except Exception:
                calls = self._stats.llm_calls
            self.call_from_thread(self._finish, code, calls, produced)

        def _set_busy(self, busy: bool, argv: list[str] | None = None) -> None:
            """Show that something is happening, and for how long.

            A command that reaches a provider can take many seconds — the cloud
            timeout defaults to minutes — and with no indicator a slow command
            and a dead one look identical. That ambiguity is the whole problem
            this solves.
            """
            self._busy = busy
            if busy and argv:
                self._running = " ".join(argv)
                self._elapsed = 0
                self.say(f"[#4d5566]▸ {self._running}[/]")
                self._tick()
                self._timer = self.set_interval(1.0, self._tick)

        def _tick(self) -> None:
            self._elapsed += 1
            self.query_one("#statusline", Static).update(
                f"[#d7875f]● running[/] [#8f9aae]{self._running}[/] "
                f"[#4d5566]· {self._elapsed}s · esc to interrupt[/]"
            )

        def _stop_timer(self) -> None:
            if self._timer is not None:
                self._timer.stop()
                self._timer = None
            self._running = ""
            self.query_one("#statusline", Static).update(self._statusline())

        def _finish(self, code: int, calls: int, produced: int = 0) -> None:
            self._busy = False
            self._stop_timer()
            if code != 0:
                self.say(f"[#d75f5f]exit {code}[/]")
            elif produced == 0:
                # Silence used to be indistinguishable from a hang. Say so.
                self.say("[#4d5566](no output)[/]")
            self.say("")
            self._stats = Stats(
                self._stats.files, self._stats.indexed, self._stats.spans, calls
            )
            self.refresh_titlebar()

    return ForgeTUI()


def _escape(text: str) -> str:
    """Command output is data, not markup.

    A vault path or a diagnostic can legitimately contain square brackets, and
    an unescaped `[dsa/pattern]` would be swallowed as a style tag — the tag
    disappears and takes the text with it.
    """
    return text.replace("[", r"\[")


def run_tui(app_typer: typer.Typer, settings: Settings, stats: Stats) -> int:
    """Entry point. Returns a process exit code."""
    if not textual_available():
        print(INSTALL_HINT)
        return 2
    build_app(app_typer, settings, stats).run()
    return 0
