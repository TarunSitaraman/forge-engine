"""`forge demo`: a ten-file vault that carries the defects this engine exists for.

Every other command needs a vault. Somebody evaluating the tool does not have
one, and pointing it at a stranger's notes to see whether it is any good is a
poor first move. So this writes a small vault whose defects are known, runs the
real pipeline over it, and reports what came back.

Two rules govern this module, and they are the difference between a demo and a
brochure:

1. **Nothing here is canned.** The findings printed are read out of the same
   `link_report`, `frontmatter_report` and `build_plan` that `forge diagnostics`
   and `forge bootstrap` use. If the engine stops finding one of these, the
   demo says MISSED, in front of whoever is watching.
2. **The vault is stated, not hidden.** The narration says what was planted.
   A demo that pretends to discover what its author placed there is a magic
   trick, and this is supposed to be evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

# The sample vault. Deliberately small enough to read in a minute, because the
# claim being made is "you can check this yourself" and a corpus nobody reads is
# not checkable.
#
# The shape is copied from the real finding, not invented: two sibling pattern
# pages, each with its own cheat sheet, where one page links the *other* page's
# sheet under its own heading. That is what 24 of 32 pages did in the vault this
# was built against, and it is why the sheet nobody links is the tell.
FILES: dict[str, str] = {
    "index.md": """---
title: Notebook
type: index
---

# Notebook

Patterns: [[binary-search]], [[two-pointers]]

Data structures: [[Heap]]
""",
    "patterns/binary-search.md": """---
title: Binary Search
type: pattern
tags: [pattern]
---

# Binary Search

Halve the search space on every comparison.

## Cheat sheet

[[Binary Search Cheat Sheet]]

## See also

[[Monotonic Stack]]
""",
    # The planted defect. The link resolves, so no link checker complains, and
    # the heading above it says whose sheet it is meant to be.
    "patterns/two-pointers.md": """---
title: Two Pointers
type: pattern
tags: [pattern]
---

# Two Pointers

Walk two indices toward each other over sorted input.

## Cheat sheet

[[Binary Search Cheat Sheet]]
""",
    "patterns/sliding-window.md": """---
title: Sliding Window
type: pattern
tags: [pattern]
---

# Sliding Window

Keep a window over the sequence and move its edges.

Nothing in this vault links to this page.
""",
    # Same filename in two folders, so a bare `[[Heap]]` cannot be resolved
    # without a human deciding which page it means.
    "patterns/Heap.md": """---
title: Heap
type: pattern
---

# Heap

The pattern: keep the k best seen so far.
""",
    "data-structures/Heap.md": """---
title: Heap
type: data-structure
---

# Heap

The structure: a complete binary tree with the heap property.
""",
    "cheatsheets/Binary Search Cheat Sheet.md": """---
title: Binary Search Cheat Sheet
type: cheatsheet
---

# Binary Search Cheat Sheet

`lo, hi = 0, len(a) - 1` and the invariant that the answer is in `[lo, hi]`.
""",
    "cheatsheets/Two Pointers Cheat Sheet.md": """---
title: Two Pointers Cheat Sheet
type: cheatsheet
---

# Two Pointers Cheat Sheet

`lo, hi = 0, len(a) - 1`, move the pointer that cannot improve the answer.
""",
    # No frontmatter at all. This is here to be reported as INFO and nothing
    # more: plenty of good notes carry no metadata, and a tool that calls every
    # one of them a defect produces a backlog nobody will ever work through.
    "drafts/scratch.md": """# Scratch

Half a thought about monotonic stacks. No frontmatter, and that is fine.
""",
    # Frontmatter that does not parse: the list is never closed.
    "drafts/broken-frontmatter.md": """---
title: Broken
tags: [a, b
---

# Broken

The frontmatter above is not valid YAML.
""",
}


@dataclass(frozen=True)
class Check:
    """One planted defect and whether the engine reported it."""

    title: str
    planted: str
    survives: str
    found: bool
    evidence: str


def write_vault(path: Path) -> Path:
    """Write the sample vault under `path`. Returns the path."""
    for name, body in FILES.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    return path


def is_occupied(path: Path) -> bool:
    """True if `path` holds anything. The demo never writes over someone's notes."""
    return path.exists() and any(path.iterdir())


def build(settings: Any) -> tuple[list[Check], dict[str, int]]:
    """Index the demo vault, seed its graph, and check for each planted defect.

    Returns the checks and the headline counts. Deterministic: no model is
    called, and the caller asserts that by reading `CALLS.count`.
    """
    from ..bootstrap import build_plan
    from ..corpus import IndexPipeline, load_store
    from ..corpus.diagnostics import frontmatter_report, link_report
    from ..corpus.indexer import CorpusIndexer

    store = load_store(settings)
    index = CorpusIndexer(settings).build_index()
    IndexPipeline(settings, store).run(persist=True, write_reports=False)

    links = link_report(index)
    frontmatter = frontmatter_report(index)
    plan = build_plan(index)

    # Seeding is what gives `forge dash` something to open on afterwards. The
    # pruning branch `forge bootstrap` carries is not needed here: this store
    # was created seconds ago, so it holds no edge from an older reading.
    for concept in plan.concepts:
        store.put_concept(concept)
    for link in plan.links:
        store.put_link(link)
    store.set_concept_inbound(plan.inbound_links)
    store.close()

    inbound = {
        c.vault_path: plan.inbound_links.get(c.id, (0, ()))[0]
        for c in plan.concepts
        if c.vault_path
    }
    wrong = "cheatsheets/Two Pointers Cheat Sheet.md"
    right = "cheatsheets/Binary Search Cheat Sheet.md"
    orphan = "patterns/sliding-window.md"

    checks = [
        Check(
            title="A link that resolves, and is wrong",
            planted=(
                "patterns/two-pointers.md links [[Binary Search Cheat Sheet]] "
                "under its own '## Cheat sheet' heading."
            ),
            survives=(
                "The link resolves. Obsidian follows it, the graph view draws "
                "an edge, and every link checker passes. Nothing is malformed."
            ),
            found=inbound.get(wrong) == 0 and (inbound.get(right) or 0) >= 2,
            evidence=(
                f"'Two Pointers Cheat Sheet' has {inbound.get(wrong)} inbound links; "
                f"'Binary Search Cheat Sheet' has {inbound.get(right)}. "
                "One sheet is linked twice and its sibling never."
            ),
        ),
        Check(
            title="A page nothing links to",
            planted="patterns/sliding-window.md is never linked from anywhere.",
            survives=(
                "It parses, it resolves its own links, and it appears in every "
                "file listing. Only asking what points *at* it finds it."
            ),
            found=inbound.get(orphan) == 0,
            evidence=(
                f"'sliding-window' has {inbound.get(orphan)} inbound links, counted "
                f"over every page in the vault rather than over the graph "
                f"({plan.unreferenced()} of {len(plan.concepts)} pages are unreferenced here)."
            ),
        ),
        Check(
            title="A link that goes nowhere",
            planted="patterns/binary-search.md links [[Monotonic Stack]], which does not exist.",
            survives="Nothing. This is the one a link checker also finds.",
            found="Monotonic Stack" in links.unresolved_targets,
            evidence=f"{links.by_status.get('missing', 0)} unresolved of {links.total_links} links.",
        ),
        Check(
            title="A link two pages could answer",
            planted="index.md links [[Heap]], and two files are named Heap.md.",
            survives=(
                "Editors pick one silently, usually the first match, and the "
                "index then describes one page while linking another."
            ),
            found="Heap" in links.ambiguous_targets,
            evidence=(
                "[[Heap]] -> "
                + ", ".join(links.ambiguous_targets.get("Heap", []))
                + ". Both are held out of the graph until `forge identity decide` "
                "records which one a bare [[Heap]] means."
            ),
        ),
        Check(
            title="Frontmatter that does not parse, told apart from frontmatter that is absent",
            planted=(
                "drafts/broken-frontmatter.md has an unclosed YAML list. "
                "drafts/scratch.md has no frontmatter at all."
            ),
            survives=(
                "A tool that reports both as problems hands you a backlog of "
                "every informal note you own, and the real defect sits in it."
            ),
            found=frontmatter.by_severity.get("error", 0) == 1
            and frontmatter.without_frontmatter >= 1,
            evidence=(
                f"{frontmatter.by_severity.get('error', 0)} error, "
                f"{frontmatter.by_severity.get('info', 0)} info. "
                f"{frontmatter.without_frontmatter} file carries no frontmatter and is "
                "not counted as a defect."
            ),
        ),
    ]

    counts = {
        "files": index.file_count,
        "links": links.total_links,
        "broken": links.by_status.get("missing", 0),
        "concepts": len(plan.concepts),
        "found": sum(1 for c in checks if c.found),
        "total": len(checks),
    }
    return checks, counts


def render(checks: list[Check], counts: dict[str, int], path: Path, calls: int) -> Iterator[str]:
    """The walkthrough, line by line."""
    yield f"Wrote a {counts['files']}-file vault to {path}"
    yield ""
    yield (
        f"A link checker reads it and reports {counts['broken']} broken link "
        f"out of {counts['links']}. Here is what else is wrong with it."
    )
    yield ""

    for n, check in enumerate(checks, start=1):
        yield f"{'FOUND ' if check.found else 'MISSED'}  {n}. {check.title}"
        for line in _wrap(check.planted, "          "):
            yield line
        yield ""
        for line in _wrap(f"Why it survives other tools: {check.survives}", "          "):
            yield line
        yield ""
        for line in _wrap(f"Reported as: {check.evidence}", "          "):
            yield line
        yield ""

    yield f"{counts['found']} of {counts['total']} planted defects reported."
    yield f"Model calls: {calls}. No network, no API key, nothing written outside the vault."
    yield ""
    yield "Look around:"
    yield f"  forge dash        --vault {path}"
    yield f"  forge diagnostics --vault {path}"
    yield f"  forge gaps        --vault {path}"
    yield ""
    yield "Then point the same commands at your own notes."


def _wrap(text: str, indent: str, width: int = 78) -> Iterator[str]:
    """Wrap to `width` columns including the indent, with no trailing spaces."""
    words: list[str] = []
    length = len(indent)
    for word in text.split():
        if words and length + 1 + len(word) > width:
            yield indent + " ".join(words)
            words, length = [], len(indent)
        length += len(word) + (1 if words else 0)
        words.append(word)
    if words:
        yield indent + " ".join(words)
