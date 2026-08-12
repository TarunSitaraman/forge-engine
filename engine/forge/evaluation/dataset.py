"""Retrieval evaluation dataset.

Loads the versioned label set in ``tests/fixtures/eval/``. Kept in the
repository rather than generated, because a labelled set that regenerates
itself measures nothing — the whole value is that it is fixed while the
retrieval implementation changes underneath it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

DEFAULT_DATASET = Path("tests") / "fixtures" / "eval" / "retrieval-v1.yaml"


class DatasetError(Exception):
    """Raised when the evaluation set is malformed or its labels have rotted."""


@dataclass(frozen=True)
class EvalQuery:
    id: str
    category: str
    query: str
    relevant: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "query": self.query,
            "relevant": list(self.relevant),
        }


@dataclass
class EvalDataset:
    version: int
    unit: str
    queries: list[EvalQuery] = field(default_factory=list)
    path: Path | None = None

    def __iter__(self) -> Iterator[EvalQuery]:
        return iter(self.queries)

    def __len__(self) -> int:
        return len(self.queries)

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for query in self.queries:
            counts[query.category] = counts.get(query.category, 0) + 1
        return dict(sorted(counts.items()))

    def by_category(self, category: str) -> list[EvalQuery]:
        return [q for q in self.queries if q.category == category]

    def label_count(self) -> int:
        return sum(len(q.relevant) for q in self.queries)

    def verify_labels(self, vault_path: Path) -> list[str]:
        """Return labels that no longer point at a real file.

        A reorganized corpus must not be able to silently turn this set into
        unreachable ground truth — which would show up as a mysterious drop in
        recall rather than as the data problem it is.
        """
        missing: list[str] = []
        for query in self.queries:
            for path in query.relevant:
                if not (vault_path / path).is_file():
                    missing.append(f"{query.id}: {path}")
        return missing

    @classmethod
    def load(cls, path: Path | None = None) -> EvalDataset:
        target = Path(path or DEFAULT_DATASET)
        if not target.is_file():
            raise DatasetError(f"evaluation set not found: {target}")

        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict) or "queries" not in raw:
            raise DatasetError(f"{target}: expected a mapping with a 'queries' key")

        queries: list[EvalQuery] = []
        seen: set[str] = set()
        for entry in raw["queries"]:
            for required in ("id", "category", "query", "relevant"):
                if required not in entry:
                    raise DatasetError(f"{target}: query missing {required!r}: {entry}")
            if entry["id"] in seen:
                raise DatasetError(f"{target}: duplicate query id {entry['id']!r}")
            seen.add(entry["id"])
            if not entry["relevant"]:
                raise DatasetError(f"{target}: query {entry['id']!r} has no relevant documents")
            queries.append(
                EvalQuery(
                    id=str(entry["id"]),
                    category=str(entry["category"]),
                    query=str(entry["query"]),
                    relevant=tuple(str(r) for r in entry["relevant"]),
                )
            )

        return cls(
            version=int(raw.get("version", 1)),
            unit=str(raw.get("unit", "source")),
            queries=queries,
            path=target,
        )
