"""Deterministic Markdown parsing.

No LLM touches this file. Same bytes in, same structure out, always.

Two hazards discovered during the Phase 0 audit are handled here explicitly,
because getting either wrong silently corrupts the index:

1. **Fenced and inline code must be stripped before link extraction.** The
   corpus contains 240 Python code blocks, many holding matrix literals like
   ``[[1,2],[3,4]]``. A naive ``\\[\\[...\\]\\]`` regex reads those as
   wikilinks and invents hundreds of false links.
2. **Frontmatter must be excluded from body parsing**, but is itself a source
   of wikilinks (the ``related:`` field), so it is parsed separately rather
   than discarded.

Implementation note: this uses stdlib ``re`` rather than a Markdown library.
That is a deliberate choice, not laziness — Forge needs *source locations*
(line numbers) for every heading and link so spans can be built, plus
wikilink and tag syntax that no CommonMark parser implements. A full parser
would have to be post-processed for all of it anyway.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Fence opener/closer: ``` or ~~~, optionally indented up to 3 spaces.
_FENCE_RE = re.compile(r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.*?)\s*#*\s*$")
_WIKILINK_RE = re.compile(r"\[\[(?P<body>[^\[\]]+?)\]\]")
_MDLINK_RE = re.compile(r"(?<!\!)\[(?P<text>[^\]]*)\]\((?P<target>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
_TAG_RE = re.compile(r"(?<![\w/#])#(?P<tag>[A-Za-z][\w/-]*)")
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")


@dataclass(frozen=True)
class Heading:
    level: int
    text: str
    line: int  # 1-based


@dataclass(frozen=True)
class WikiLink:
    """A parsed ``[[target|alias#anchor]]`` reference."""

    raw: str
    target: str
    anchor: str | None
    alias: str | None
    line: int
    in_frontmatter: bool = False


@dataclass(frozen=True)
class MarkdownLink:
    """A parsed ``[text](target)`` reference."""

    text: str
    target: str
    line: int


@dataclass
class ParsedMarkdown:
    """Everything the indexer needs from one Markdown file."""

    raw: str
    #: Body with frontmatter removed, code fences/inline code blanked out.
    #: Line numbering is preserved so locations stay accurate.
    body_masked: str
    frontmatter_raw: str | None
    frontmatter_end_line: int  # 0 when absent
    headings: list[Heading] = field(default_factory=list)
    wikilinks: list[WikiLink] = field(default_factory=list)
    markdown_links: list[MarkdownLink] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    code_block_count: int = 0
    code_languages: list[str] = field(default_factory=list)
    line_count: int = 0

    @property
    def title(self) -> str | None:
        """First H1, which the vault's conventions require on every file."""
        for h in self.headings:
            if h.level == 1:
                return h.text
        return None


def split_frontmatter(text: str) -> tuple[str | None, int]:
    """Return ``(frontmatter_text, end_line)``; ``(None, 0)`` when absent.

    ``end_line`` is the 1-based line number of the closing ``---``.
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None, 0
    for i in range(1, len(lines)):
        stripped = lines[i].rstrip()
        if stripped in ("---", "..."):
            return "\n".join(lines[1:i]), i + 1
    return None, 0  # unterminated: not frontmatter


def mask_code(text: str) -> tuple[str, int, list[str]]:
    """Blank out fenced blocks and inline code, preserving line structure.

    Returns ``(masked_text, fenced_block_count, languages)``. Masked lines
    become empty strings, so every subsequent line number is still correct.
    """
    out: list[str] = []
    fence: str | None = None
    count = 0
    languages: list[str] = []

    for line in text.split("\n"):
        m = _FENCE_RE.match(line)
        if fence is None:
            if m and m.group("fence"):
                fence = m.group("fence")[0] * 3
                info = m.group("info").strip()
                languages.append(info.split()[0] if info else "")
                count += 1
                out.append("")
                continue
            out.append(_INLINE_CODE_RE.sub(lambda mo: " " * len(mo.group(0)), line))
        else:
            # Inside a fence: a closing fence is the same char, >= 3 long.
            if m and m.group("fence") and m.group("fence")[0] * 3 == fence:
                fence = None
            out.append("")
    return "\n".join(out), count, languages


def parse_wikilink(body: str, line: int, *, in_frontmatter: bool = False) -> WikiLink | None:
    """Parse the inside of ``[[...]]``.

    Handles ``target``, ``target|alias``, ``target#anchor``, and
    ``target#anchor|alias``.
    """
    raw = body
    alias: str | None = None
    if "|" in body:
        body, alias = body.split("|", 1)
        alias = alias.strip() or None
    anchor: str | None = None
    if "#" in body:
        body, anchor = body.split("#", 1)
        anchor = anchor.strip() or None
    target = body.strip()
    if not target:
        return None
    return WikiLink(
        raw=raw, target=target, anchor=anchor, alias=alias, line=line, in_frontmatter=in_frontmatter
    )


def parse_markdown(text: str) -> ParsedMarkdown:
    """Parse one Markdown document deterministically."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    fm_raw, fm_end = split_frontmatter(normalized)

    lines = normalized.split("\n")
    # Replace frontmatter lines with blanks so line numbers stay true.
    body_lines = ([""] * fm_end) + lines[fm_end:] if fm_end else lines
    body = "\n".join(body_lines)

    masked, fence_count, languages = mask_code(body)

    headings: list[Heading] = []
    wikilinks: list[WikiLink] = []
    md_links: list[MarkdownLink] = []
    tags: list[str] = []

    for idx, line in enumerate(masked.split("\n"), start=1):
        if not line.strip():
            continue
        if (hm := _HEADING_RE.match(line)) is not None:
            headings.append(
                Heading(level=len(hm.group("hashes")), text=hm.group("text").strip(), line=idx)
            )
        for wm in _WIKILINK_RE.finditer(line):
            if (wl := parse_wikilink(wm.group("body"), idx)) is not None:
                wikilinks.append(wl)
        for mm in _MDLINK_RE.finditer(line):
            md_links.append(
                MarkdownLink(text=mm.group("text"), target=mm.group("target"), line=idx)
            )
        for tm in _TAG_RE.finditer(line):
            tags.append(tm.group("tag"))

    # Frontmatter carries wikilinks too (the corpus's `related:` field), and
    # they are real links even though the field's YAML is malformed corpus-wide.
    if fm_raw is not None:
        for offset, fline in enumerate(fm_raw.split("\n"), start=2):
            for wm in _WIKILINK_RE.finditer(fline):
                if (wl := parse_wikilink(wm.group("body"), offset, in_frontmatter=True)) is not None:
                    wikilinks.append(wl)

    return ParsedMarkdown(
        raw=normalized,
        body_masked=masked,
        frontmatter_raw=fm_raw,
        frontmatter_end_line=fm_end,
        headings=headings,
        wikilinks=wikilinks,
        markdown_links=md_links,
        tags=_dedupe(tags),
        code_block_count=fence_count,
        code_languages=_dedupe([lang for lang in languages if lang]),
        line_count=len(lines),
    )


def _dedupe(items: list[str]) -> list[str]:
    """Order-preserving dedupe — keeps output deterministic."""
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out
