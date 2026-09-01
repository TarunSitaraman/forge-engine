"""Tests for the interactive shell.

Only the parsing and the loop's control flow are tested. Rendering is ANSI text
with no branching worth pinning, and dispatch is the CLI's own code, already
covered where it lives. What matters here is that a line of input means what the
user thinks it means, and that nothing takes the session down.
"""

from __future__ import annotations

import pytest

from forge.cli.main import app
from forge.cli.shell import (
    Action,
    Kind,
    command_names,
    dispatch,
    history_path,
    parse,
    render_header,
    render_help,
    run,
    visible_names,
)
from forge.config import Settings

NAMES = command_names(app)


class TestParsing:
    @pytest.mark.parametrize("line", ["", "   ", "\t", "/", "/   "])
    def test_blank_input_does_nothing(self, line):
        assert parse(line).kind is Kind.EMPTY

    @pytest.mark.parametrize("word", ["quit", "exit", "q", "QUIT"])
    def test_quit_words(self, word):
        assert parse(f"/{word}").kind is Kind.QUIT

    @pytest.mark.parametrize("word", ["help", "?"])
    def test_help_words(self, word):
        assert parse(f"/{word}").kind is Kind.HELP

    def test_a_slash_command_becomes_argv(self):
        got = parse("/graph path A B", NAMES)
        assert got.kind is Kind.COMMAND
        assert got.argv == ("graph", "path", "A", "B")

    def test_options_survive(self):
        got = parse("/diagnostics links --limit 100", NAMES)
        assert got.argv == ("diagnostics", "links", "--limit", "100")

    def test_quoted_arguments_stay_one_token(self):
        got = parse('/inspect "DSA/01_Patterns/Binary Search.md"', NAMES)
        assert got.argv == ("inspect", "DSA/01_Patterns/Binary Search.md")

    def test_bare_text_is_routed_to_ask(self):
        got = parse("how does BFS differ from DFS", NAMES)
        assert got.argv == ("ask", "how does BFS differ from DFS")

    def test_a_question_keeps_its_punctuation(self):
        """Prose is passed whole. shlex would eat the apostrophe and could
        raise on an unbalanced quote, which is normal in a question."""
        got = parse("what's the point of Kadane's algorithm?", NAMES)
        assert got.argv == ("ask", "what's the point of Kadane's algorithm?")

    def test_an_unbalanced_quote_in_a_slash_command_is_reported_not_raised(self):
        got = parse('/search "unclosed', NAMES)
        assert got.kind is Kind.HELP
        assert "could not parse" in got.message

    def test_an_unknown_command_is_named(self):
        got = parse("/nonsense", NAMES)
        assert got.kind is Kind.REFUSED
        assert "/nonsense" in got.message

    def test_a_single_near_match_is_suggested(self):
        got = parse("/diagno", NAMES)
        assert "/diagnostics" in got.message

    def test_without_a_known_set_nothing_is_rejected(self):
        """The known set is optional so `parse` stays usable on its own."""
        assert parse("/nonsense").kind is Kind.COMMAND


class TestShellIsNotNested:
    def test_slash_shell_is_refused(self):
        got = parse("/shell", NAMES)
        assert got.kind is Kind.REFUSED
        assert "already in it" in got.message

    def test_but_it_is_not_offered(self):
        """Listing a command the shell refuses would be a promise it breaks."""
        assert "shell" in NAMES
        assert "shell" not in visible_names(NAMES)
        assert "/shell" not in render_help(NAMES)


class TestDispatchKeepsTheSessionAlive:
    def test_a_usage_error_returns_non_zero_rather_than_raising(self):
        assert dispatch(app, ["diagnostics", "--nonsense-flag"]) != 0

    def test_an_unknown_command_does_not_raise(self):
        assert dispatch(app, ["definitely-not-a-command"]) != 0

    def test_a_working_command_returns_zero(self):
        assert dispatch(app, ["--help"]) == 0


class TestTheLoop:
    def _settings(self, tmp_path):
        vault = tmp_path / "vault"
        (vault / ".git").mkdir(parents=True)
        return Settings(vault_path=vault, state_dir=tmp_path / "state")

    def _run(self, settings, lines):
        written: list[str] = []
        supply = iter(lines)

        def reader(_prompt):
            return next(supply)

        return run(app, settings, 10, 10, reader=reader, writer=written.append), written

    def test_quit_leaves_cleanly(self, tmp_path):
        code, out = self._run(self._settings(tmp_path), ["/quit"])
        assert code == 0
        assert "FORGE" in out[0]

    def test_end_of_input_leaves_cleanly(self, tmp_path):
        """Ctrl-D must end the session, not raise into the caller."""

        def reader(_prompt):
            raise EOFError

        code = run(app, self._settings(tmp_path), 1, 1, reader=reader, writer=lambda _: None)
        assert code == 0

    def test_interrupt_leaves_cleanly(self, tmp_path):
        def reader(_prompt):
            raise KeyboardInterrupt

        code = run(app, self._settings(tmp_path), 1, 1, reader=reader, writer=lambda _: None)
        assert code == 0

    def test_blank_lines_do_not_end_the_session(self, tmp_path):
        code, _ = self._run(self._settings(tmp_path), ["", "   ", "/quit"])
        assert code == 0

    def test_an_unknown_command_does_not_end_the_session(self, tmp_path):
        code, out = self._run(self._settings(tmp_path), ["/nonsense", "/quit"])
        assert code == 0
        assert any("nonsense" in line for line in out)

    def test_help_is_printed_without_leaving(self, tmp_path):
        code, out = self._run(self._settings(tmp_path), ["/help", "/quit"])
        assert code == 0
        assert any("/index" in line for line in out)


class TestHeader:
    def test_it_names_the_vault_and_the_provider(self, tmp_path):
        vault = tmp_path / "vault"
        (vault / ".git").mkdir(parents=True)
        settings = Settings(vault_path=vault, state_dir=tmp_path / "state")

        got = render_header(settings, files=634, indexed=634)

        assert str(vault) in got
        assert "634 files" in got
        assert settings.llm.provider in got

    def test_a_stale_index_is_flagged(self, tmp_path):
        vault = tmp_path / "vault"
        (vault / ".git").mkdir(parents=True)
        settings = Settings(vault_path=vault, state_dir=tmp_path / "state")

        assert "indexed 600" in render_header(settings, files=634, indexed=600)
        assert "indexed" not in render_header(settings, files=634, indexed=634)


class TestHistoryLocation:
    def test_history_sits_beside_the_settings_file_not_in_the_vault(self, tmp_path, monkeypatch):
        """The vault is content. Shell history is machine state and must not
        land in it — the engine is read-only with respect to the vault."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

        got = history_path()

        assert got == tmp_path / "forge" / "shell_history"


class TestActionIsInert:
    def test_actions_are_frozen(self):
        with pytest.raises(Exception):
            Action(Kind.EMPTY).kind = Kind.QUIT  # type: ignore[misc]
