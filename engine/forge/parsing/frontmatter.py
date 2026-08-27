"""Robust frontmatter parsing with structured diagnostics and *proposed* repairs.

Phase 1 policy (ADR-001 D2, segregated write-back): this module **never writes
to the corpus**. It parses, diagnoses, and emits machine-readable repair
proposals. Applying them is a separate, explicitly-approved human action.

The audit found frontmatter to be the corpus's largest machine-readability
defect: 283 ``related:`` fields malformed, of which 68 fail YAML parsing
outright and 215 parse into nested lists rather than links. Both shapes come
from the same authoring mistake — writing Obsidian wikilinks into a YAML
value without quoting:

    related: [[Pattern Index]], [[Template Index]]   # -> YAML ParserError
    related: [[[DFS]], [[BFS]]]                      # -> [[['DFS']], [['BFS']]]

Neither is usable as relationships. The proposed repair for both is the same
and is deterministic:

    related: ["Pattern Index", "Template Index"]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import yaml

_WIKILINK_INNER_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")
#: A wikilink whose closing ``]]`` was truncated to a single ``]`` at end of
#: value — e.g. ``related: [[A]], [[B]``. Discovered in 18 corpus files during
#: Phase 1; the Phase 0 audit characterized only the two shapes above it.
_TRUNCATED_WIKILINK_RE = re.compile(r"\[\[([^\[\]]+?)\](?!\])\s*$")
_KEY_VALUE_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z_][\w-]*)\s*:\s*(?P<value>.*)$")


class Severity(str, Enum):
    ERROR = "error"  # unusable as metadata
    WARNING = "warning"  # parses, but semantically wrong
    INFO = "info"  # absent or incomplete, not broken


class DiagnosticCode(str, Enum):
    YAML_PARSE_ERROR = "FM001"
    NESTED_LIST_WIKILINKS = "FM002"
    NO_FRONTMATTER = "FM003"
    NOT_A_MAPPING = "FM004"
    DUPLICATE_KEY = "FM005"
    UNQUOTED_WIKILINK = "FM006"
    EMPTY_FRONTMATTER = "FM007"
    TRUNCATED_WIKILINK = "FM008"


#: Human-readable explanations, emitted into reports so a diagnostic is
#: actionable without reading this source file.
CODE_DESCRIPTIONS: dict[DiagnosticCode, str] = {
    DiagnosticCode.YAML_PARSE_ERROR: "Frontmatter is not valid YAML and cannot be loaded at all.",
    DiagnosticCode.NESTED_LIST_WIKILINKS: (
        "Field parses as a list of lists rather than a list of link names, because "
        "unquoted [[wikilinks]] collide with YAML flow-sequence syntax."
    ),
    DiagnosticCode.NO_FRONTMATTER: "File has no YAML frontmatter block.",
    DiagnosticCode.NOT_A_MAPPING: "Frontmatter parses but is not a key/value mapping.",
    DiagnosticCode.DUPLICATE_KEY: "Key appears more than once; YAML silently keeps the last value.",
    DiagnosticCode.UNQUOTED_WIKILINK: "Value contains unquoted [[wikilinks]] that YAML misreads.",
    DiagnosticCode.EMPTY_FRONTMATTER: "Frontmatter block is present but empty.",
    DiagnosticCode.TRUNCATED_WIKILINK: (
        "The final [[wikilink]] in the value is missing one closing bracket, e.g. "
        "'related: [[A]], [[B]'. Deterministically recoverable, but a distinct "
        "defect from the unquoted-wikilink case."
    ),
}


@dataclass(frozen=True)
class Diagnostic:
    code: DiagnosticCode
    severity: Severity
    message: str
    key: str | None = None
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "key": self.key,
            "line": self.line,
            "description": CODE_DESCRIPTIONS[self.code],
        }


@dataclass
class RepairProposal:
    """A deterministic, reviewable repair. Never applied automatically."""

    key: str
    line: int
    original: str
    proposed: str
    reason: str
    #: True only when the repaired frontmatter was re-parsed and verified.
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "line": self.line,
            "original": self.original,
            "proposed": self.proposed,
            "reason": self.reason,
            "verified": self.verified,
        }


@dataclass
class FrontmatterResult:
    present: bool
    valid: bool
    data: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    repairs: list[RepairProposal] = field(default_factory=list)
    #: Fully repaired frontmatter text, when every problem has a verified fix.
    repaired_text: str | None = None

    @property
    def has_errors(self) -> bool:
        return any(d.severity is Severity.ERROR for d in self.diagnostics)


class _DuplicateDetectingLoader(yaml.SafeLoader):
    """SafeLoader that records duplicate mapping keys instead of silently
    keeping the last one."""

    def __init__(self, stream: Any) -> None:
        super().__init__(stream)
        self.duplicate_keys: list[str] = []

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            try:
                key = self.construct_object(key_node, deep=deep)
            except Exception:  # pragma: no cover - malformed key
                continue
            if isinstance(key, str):
                if key in seen:
                    self.duplicate_keys.append(key)
                seen.add(key)
        return super().construct_mapping(node, deep)


def parse_frontmatter(raw: str | None) -> FrontmatterResult:
    """Parse a frontmatter block, diagnosing and proposing repairs.

    ``raw`` is the text *between* the ``---`` fences (as returned by
    :func:`forge.parsing.markdown.split_frontmatter`), or ``None``.
    """
    if raw is None:
        return FrontmatterResult(
            present=False,
            valid=False,
            diagnostics=[
                Diagnostic(
                    code=DiagnosticCode.NO_FRONTMATTER,
                    severity=Severity.INFO,
                    message="No YAML frontmatter block present.",
                )
            ],
        )

    if not raw.strip():
        return FrontmatterResult(
            present=True,
            valid=False,
            diagnostics=[
                Diagnostic(
                    code=DiagnosticCode.EMPTY_FRONTMATTER,
                    severity=Severity.INFO,
                    message="Frontmatter block is present but empty.",
                )
            ],
        )

    repairs, diagnostics = _propose_wikilink_repairs(raw)

    data, parse_error, duplicates = _safe_load(raw)

    if parse_error is not None:
        line = getattr(getattr(parse_error, "problem_mark", None), "line", None)
        diagnostics.append(
            Diagnostic(
                code=DiagnosticCode.YAML_PARSE_ERROR,
                severity=Severity.ERROR,
                message=f"YAML parse failed: {_clean_yaml_error(parse_error)}",
                line=(line + 2) if isinstance(line, int) else None,  # +2: 1-based, past '---'
            )
        )
        repaired_text, verified = _verify_repairs(raw, repairs)
        for r in repairs:
            r.verified = verified
        return FrontmatterResult(
            present=True,
            valid=False,
            data={},
            diagnostics=diagnostics,
            repairs=repairs,
            repaired_text=repaired_text if verified else None,
        )

    if not isinstance(data, dict):
        diagnostics.append(
            Diagnostic(
                code=DiagnosticCode.NOT_A_MAPPING,
                severity=Severity.ERROR,
                message=f"Frontmatter parsed as {type(data).__name__}, expected a mapping.",
            )
        )
        return FrontmatterResult(present=True, valid=False, diagnostics=diagnostics, repairs=repairs)

    for key in duplicates:
        diagnostics.append(
            Diagnostic(
                code=DiagnosticCode.DUPLICATE_KEY,
                severity=Severity.WARNING,
                message=f"Key {key!r} defined more than once; only the last value survives.",
                key=key,
            )
        )

    # Parsed cleanly, but nested-list values mean the wikilinks were misread.
    for key, value in data.items():
        if _is_nested_list(value):
            diagnostics.append(
                Diagnostic(
                    code=DiagnosticCode.NESTED_LIST_WIKILINKS,
                    severity=Severity.WARNING,
                    message=(
                        f"Field {key!r} parsed as a list of lists "
                        f"({_preview(value)}); wikilinks were misread as nested "
                        f"YAML sequences and are unusable as relationships."
                    ),
                    key=str(key),
                )
            )

    repaired_text, verified = _verify_repairs(raw, repairs)
    for r in repairs:
        r.verified = verified

    return FrontmatterResult(
        present=True,
        valid=not any(d.severity is Severity.ERROR for d in diagnostics),
        data=data,
        diagnostics=diagnostics,
        repairs=repairs,
        repaired_text=repaired_text if (verified and repairs) else None,
    )


def extract_wikilink_values(raw: str, key: str) -> list[str]:
    """Pull link names out of a frontmatter field regardless of YAML validity.

    This is how the indexer recovers the ``related:`` graph today: the field is
    unparseable as YAML corpus-wide, but the *links themselves* are perfectly
    recoverable by text extraction. Reading them does not require repairing the
    file, which is what keeps the corpus untouched.
    """
    for line in raw.split("\n"):
        m = _KEY_VALUE_RE.match(line)
        if m and m.group("key") == key:
            value = m.group("value")
            names = [t.strip() for t in _WIKILINK_INNER_RE.findall(value) if t.strip()]
            # Include a final wikilink truncated to one closing bracket.
            if (t := _TRUNCATED_WIKILINK_RE.search(_WIKILINK_INNER_RE.sub("", value))) is not None:
                names.append(t.group(1).strip())
            return names
    return []


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------


def _safe_load(raw: str) -> tuple[Any, Exception | None, list[str]]:
    loader = _DuplicateDetectingLoader(raw)
    try:
        data = loader.get_single_data()
        return data, None, loader.duplicate_keys
    except Exception as exc:
        return None, exc, []
    finally:
        loader.dispose()


def _propose_wikilink_repairs(raw: str) -> tuple[list[RepairProposal], list[Diagnostic]]:
    """Propose quoted-list replacements for any line whose value holds wikilinks.

    Returns proposals plus any diagnostics raised while inspecting the values.
    """
    proposals: list[RepairProposal] = []
    extra: list[Diagnostic] = []

    for idx, line in enumerate(raw.split("\n"), start=2):  # +2: 1-based, past '---'
        m = _KEY_VALUE_RE.match(line)
        if not m:
            continue
        value = m.group("value")
        if "[[" not in value:
            continue

        names = [t.strip() for t in _WIKILINK_INNER_RE.findall(value) if t.strip()]
        residue = _WIKILINK_INNER_RE.sub("", value)

        # Recover a final wikilink whose closing ']]' was truncated to ']'.
        truncated = _TRUNCATED_WIKILINK_RE.search(residue)
        if truncated is not None:
            names.append(truncated.group(1).strip())
            residue = residue[: truncated.start()]
            extra.append(
                Diagnostic(
                    code=DiagnosticCode.TRUNCATED_WIKILINK,
                    severity=Severity.WARNING,
                    message=(
                        f"Field {m.group('key')!r} ends with a truncated wikilink "
                        f"'[[{truncated.group(1).strip()}]' (one closing bracket)."
                    ),
                    key=m.group("key"),
                    line=idx,
                )
            )

        if not names:
            continue

        # Only propose when wikilinks account for the whole value. If there is
        # other content mixed in, a mechanical rewrite could lose it.
        if residue.strip(" ,[]"):
            continue

        # Quote the wikilink *including its brackets*, not just the name.
        #
        # Emitting `related: ["A", "B"]` is valid YAML and was the original
        # repair, but it silently destroys the links. Two consumers read these
        # fields by text-extracting `[[...]]` from the raw frontmatter —
        # `extract_wikilink_values`, which is how `CorpusIndexer` builds the
        # `related` graph, and `parse_markdown`, which counts frontmatter
        # wikilinks. Strip the brackets and both return nothing: measured on
        # the corpus, that repair would have dropped 746 `related:` edges.
        # Obsidian likewise renders a bare string as text and a quoted
        # wikilink as a link.
        #
        # `["[[A]]", "[[B]]"]` satisfies every reader: valid YAML, a real
        # list once parsed, still text-extractable, still an Obsidian link.
        rendered = ", ".join(_yaml_quote(f"[[{n}]]") for n in names)
        proposals.append(
            RepairProposal(
                key=m.group("key"),
                line=idx,
                original=line,
                proposed=f"{m.group('indent')}{m.group('key')}: [{rendered}]",
                reason=(
                    "Unquoted [[wikilinks]] collide with YAML flow-sequence syntax; "
                    "quoting each wikilink makes the field valid YAML while keeping "
                    "the links readable by the indexer and by Obsidian."
                ),
            )
        )
    return proposals, extra


def _verify_repairs(raw: str, repairs: list[RepairProposal]) -> tuple[str | None, bool]:
    """Apply proposals in memory and confirm the result parses to a mapping.

    A repair is only ever offered as ``verified`` if this succeeds. Nothing is
    written to disk here.
    """
    if not repairs:
        return None, False
    by_line = {r.line: r.proposed for r in repairs}
    out = [by_line.get(idx, line) for idx, line in enumerate(raw.split("\n"), start=2)]
    candidate = "\n".join(out)
    data, err, _ = _safe_load(candidate)
    if err is not None or not isinstance(data, dict):
        return None, False
    if any(_is_nested_list(v) for v in data.values()):
        return None, False
    return candidate, True


def _is_nested_list(value: Any) -> bool:
    return isinstance(value, list) and any(isinstance(item, list) for item in value)


def _yaml_quote(name: str) -> str:
    escaped = name.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _preview(value: Any, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _clean_yaml_error(exc: Exception) -> str:
    return " ".join(str(exc).split())[:200]
