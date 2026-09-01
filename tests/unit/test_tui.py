"""Tests for the full-screen TUI.

Textual is an optional extra, so everything that needs it skips when it is
absent — the core install must not be made to depend on a terminal UI by the
back door of a failing test.

What is tested is the part with logic: output capture, markup safety, the stats
line, and that driving the interface actually runs a command and shows it.
"""

from __future__ import annotations

import asyncio

import pytest

from forge.cli.main import app as cli
from forge.cli.tui import (
    INSTALL_HINT,
    LineWriter,
    Stats,
    _escape,
    run_command,
    run_tui,
    textual_available,
)
from forge.config import Settings

textual = pytest.importorskip("textual", reason="TUI extra not installed")


@pytest.fixture
def settings(tmp_path):
    vault = tmp_path / "vault"
    (vault / ".git").mkdir(parents=True)
    return Settings(vault_path=vault, state_dir=tmp_path / "state")


class TestLineWriter:
    def test_lines_are_emitted_whole(self):
        """`print` arrives in fragments — the text and its newline are separate
        writes. Buffering to the newline is what stops the transcript showing a
        word at a time."""
        got: list[str] = []
        w = LineWriter(got.append)

        w.write("hello")
        assert got == []
        w.write(" world\n")

        assert got == ["hello world"]

    def test_several_lines_in_one_write(self):
        got: list[str] = []
        LineWriter(got.append).write("a\nb\nc\n")
        assert got == ["a", "b", "c"]

    def test_a_trailing_fragment_survives_flush(self):
        got: list[str] = []
        w = LineWriter(got.append)
        w.write("no newline")
        w.flush()
        assert got == ["no newline"]

    def test_flush_is_idempotent(self):
        got: list[str] = []
        w = LineWriter(got.append)
        w.write("x\n")
        w.flush()
        w.flush()
        assert got == ["x"]


class TestRunCommand:
    def test_output_is_captured_rather_than_printed(self, capsys):
        got: list[str] = []

        code = run_command(cli, ["--help"], got.append)

        assert code == 0
        assert any("Usage" in line for line in got)
        assert capsys.readouterr().out == "", "output must not escape to the real stdout"

    def test_a_failing_command_returns_non_zero_without_raising(self):
        got: list[str] = []
        assert run_command(cli, ["definitely-not-a-command"], got.append) != 0

    def test_a_usage_error_is_captured_too(self):
        got: list[str] = []
        run_command(cli, ["diagnostics", "--nonsense-flag"], got.append)
        assert got, "the error belongs in the transcript, not on the raw screen"


class TestMarkupSafety:
    def test_brackets_in_output_are_escaped(self):
        """Command output is data. `tags: ['dsa/pattern']` is a real line from
        `forge inspect`, and unescaped it would be read as a style tag — the tag
        vanishes and takes the text with it."""
        assert _escape("tags : ['dsa/pattern']") == r"tags : \['dsa/pattern']"

    def test_plain_text_is_untouched(self):
        assert _escape("634 files") == "634 files"


class TestStats:
    def test_it_reports_the_counters(self):
        got = Stats(files=634, indexed=634, spans=7016, llm_calls=0).render()
        assert "634 files" in got
        assert "7,016 spans" in got
        assert "0 llm calls" in got

    def test_a_stale_index_is_flagged(self):
        got = Stats(files=634, indexed=600, spans=7016, llm_calls=0).render()
        assert "600 indexed" in got

    def test_a_current_index_is_not(self):
        assert "indexed" not in Stats(634, 634, 7016, 0).render()


class TestTheExtraIsOptional:
    def test_the_hint_names_both_install_paths_and_the_fallback(self):
        assert "forge-engine[tui]" in INSTALL_HINT
        assert "pipx inject" in INSTALL_HINT
        assert "forge shell" in INSTALL_HINT

    def test_a_missing_extra_is_an_instruction_not_a_traceback(
        self, monkeypatch, settings, capsys
    ):
        monkeypatch.setattr("forge.cli.tui.textual_available", lambda: False)

        code = run_tui(cli, settings, Stats(1, 1, 1, 0))

        assert code == 2
        assert "pip install" in capsys.readouterr().out

    def test_it_is_installed_here(self):
        assert textual_available() is True


class TestDrivingTheInterface:
    """Headless runs through Textual's pilot. Sync wrappers around asyncio so
    the suite needs no async plugin."""

    def _drive(self, settings, keys, stats=None):
        from textual.widgets import RichLog

        from forge.cli.tui import build_app

        async def go():
            app = build_app(cli, settings, stats or Stats(10, 10, 100, 0))
            async with app.run_test(size=(100, 30)) as pilot:
                for key in keys:
                    if key == "\n":
                        await pilot.press("enter")
                    else:
                        await pilot.press(*key)
                for _ in range(80):
                    await pilot.pause()
                    if not app._busy:
                        break
                await asyncio.sleep(0.2)
                await pilot.pause()
                log = app.query_one("#transcript", RichLog)
                return app, len(log.lines)

        return asyncio.run(go())

    def _transcript(self, settings, keys) -> str:
        """The transcript as plain text, for asserting on what was shown."""
        from textual.widgets import RichLog

        from forge.cli.tui import build_app

        async def go():
            app = build_app(cli, settings, Stats(10, 10, 100, 0))
            async with app.run_test(size=(100, 30)) as pilot:
                for key in keys:
                    if key == "\n":
                        await pilot.press("enter")
                    else:
                        await pilot.press(*key)
                for _ in range(80):
                    await pilot.pause()
                    if not app._busy:
                        break
                await asyncio.sleep(0.2)
                await pilot.pause()
                log = app.query_one("#transcript", RichLog)
                return "\n".join(strip.text for strip in log.lines)

        return asyncio.run(go())

    def test_it_starts_and_shows_the_vault(self, settings):
        _, lines = self._drive(settings, [])
        assert lines > 0

    def test_a_command_runs_and_its_output_lands_in_the_transcript(self, settings):
        _, before = self._drive(settings, [])
        _, after = self._drive(settings, ["/corpus-stats", "\n"])
        assert after > before, "running a command should add transcript lines"

    def test_an_unknown_command_is_named_in_the_transcript(self, settings):
        text = self._transcript(settings, ["/nonsense", "\n"])
        assert "nonsense" in text
        assert "unknown command" in text

    def test_help_lists_commands(self, settings):
        _, plain = self._drive(settings, [])
        _, helped = self._drive(settings, ["/help", "\n"])
        assert helped > plain
