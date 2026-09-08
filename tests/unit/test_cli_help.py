"""`forge --help`: is every command findable?

Thirty-eight commands is past the point where a flat list is usable, so they are
grouped. The grouping is only worth anything if it stays complete, and the way
it rots is the ordinary one: someone adds a command and does not think about
where it belongs, so it lands in an unlabelled box under everything else and is
never seen again.

The first test here is what prevents that. It fails on the new command, in the
suite, before the help is ever printed.
"""

from __future__ import annotations

from typer.testing import CliRunner

from forge.cli.main import HELP_PANELS, app, assign_help_panels

runner = CliRunner()


def test_every_command_has_been_given_a_panel() -> None:
    unplaced = assign_help_panels(app, HELP_PANELS)
    assert not unplaced, (
        f"these commands are in no help panel: {unplaced}. "
        f"Add each to HELP_PANELS in forge/cli/main.py."
    )


def test_no_command_is_listed_in_two_panels() -> None:
    seen: dict[str, str] = {}
    for panel, names in HELP_PANELS.items():
        for name in names:
            assert name not in seen, f"{name} is in both {seen[name]!r} and {panel!r}"
            seen[name] = panel


def test_the_help_opens_on_where_to_start() -> None:
    """Panel order is the reading order, and it is not the registration order.

    Typer draws panels in the order it first meets one, so this held `Measure`
    third until the command list was sorted. That is the whole reason
    `assign_help_panels` reorders rather than only labelling.
    """
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    positions = [result.stdout.find(panel) for panel in HELP_PANELS]
    assert all(p > 0 for p in positions), "a panel is missing from the help"
    assert positions == sorted(positions), "panels are not in the order of HELP_PANELS"


def test_demo_is_the_first_thing_offered() -> None:
    """Someone with no vault has exactly one command they can run. Say so first."""
    assert next(iter(HELP_PANELS)) == "Start here"
    assert HELP_PANELS["Start here"][0] == "demo"
