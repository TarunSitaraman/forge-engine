"""Drift detection for documented external repositories.

`Projects/` documents repos that live elsewhere and change without the vault
noticing. Detection is deterministic — `git ls-remote` against a recorded
commit — so it needs no model, no API token, and no rate limit budget.
"""

from __future__ import annotations

import pytest

from forge.upstream import UpstreamError, UpstreamStatus, check


def _fetch(mapping):
    def fetch(url):
        value = mapping[url]
        if isinstance(value, Exception):
            raise value
        return value

    return fetch


class TestStates:
    def test_matching_commit_is_current(self):
        [s] = check([("p.md", "u", "abc")], _fetch({"u": "abc"}))
        assert s.state == "current"
        assert s.drifted is False

    def test_a_moved_head_is_drift(self):
        [s] = check([("p.md", "u", "abc")], _fetch({"u": "def"}))
        assert s.state == "drifted"
        assert s.drifted is True

    def test_never_pinned_is_not_reported_as_drift(self):
        """Unpinned means nobody has claimed to have reviewed it — not stale."""
        [s] = check([("p.md", "u", None)], _fetch({"u": "abc"}))
        assert s.state == "unpinned"
        assert s.drifted is False

    def test_an_unreachable_remote_is_reported_not_swallowed(self):
        [s] = check([("p.md", "u", "abc")], _fetch({"u": UpstreamError("boom")}))
        assert s.state == "unreachable"
        assert s.drifted is False
        assert "boom" in s.error

    def test_one_bad_remote_does_not_stop_the_others(self):
        statuses = check(
            [("a.md", "ua", "x"), ("b.md", "ub", "y")],
            _fetch({"ua": UpstreamError("down"), "ub": "y"}),
        )
        assert [s.state for s in statuses] == ["unreachable", "current"]


class TestNoModelInvolved:
    def test_checking_makes_no_model_calls(self):
        from forge.llm.base import CALLS

        CALLS.reset()
        check([("p.md", "u", "abc")], _fetch({"u": "def"}))
        assert CALLS.count == 0


class TestFrontmatterPinning:
    def test_an_existing_key_is_replaced_not_duplicated(self):
        from forge.cli.phase3 import _rewrite_frontmatter_value

        text = "---\ntype: project\nupstream_commit: old\n---\n\n# Title\n"
        out = _rewrite_frontmatter_value(text, "upstream_commit", "new")
        assert out.count("upstream_commit") == 1
        assert "upstream_commit: new" in out

    def test_a_missing_key_is_appended(self):
        from forge.cli.phase3 import _rewrite_frontmatter_value

        text = "---\ntype: project\n---\n\n# Title\n"
        out = _rewrite_frontmatter_value(text, "upstream_commit", "abc")
        assert "upstream_commit: abc" in out

    def test_the_body_is_untouched(self):
        from forge.cli.phase3 import _rewrite_frontmatter_value

        text = "---\ntype: project\n---\n\n# Title\n\nBody with: colons and --- dashes\n"
        out = _rewrite_frontmatter_value(text, "upstream_commit", "abc")
        assert out.endswith("# Title\n\nBody with: colons and --- dashes\n")

    def test_other_keys_keep_their_order_and_formatting(self):
        """A pin must be a one-line diff, not a reformatted frontmatter block."""
        from forge.cli.phase3 import _rewrite_frontmatter_value

        text = "---\nzeta: 1\nalpha: 2\nupstream_commit: old\n---\n\nbody\n"
        out = _rewrite_frontmatter_value(text, "upstream_commit", "new")
        assert out.index("zeta") < out.index("alpha")

    def test_a_file_without_frontmatter_is_left_alone(self):
        from forge.cli.phase3 import _rewrite_frontmatter_value

        text = "# Title\n\nbody\n"
        assert _rewrite_frontmatter_value(text, "upstream_commit", "abc") == text
