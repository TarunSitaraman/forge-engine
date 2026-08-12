"""Integration tests against the **actual Forge corpus**.

These are the tests that matter most: they validate the engine on the real
material it exists to process, including its genuine defects. Synthetic
fixtures cannot demonstrate that the indexer survives 629 files of
inconsistently-formatted Markdown containing 555 code blocks.
"""

from __future__ import annotations

import subprocess

import pytest

from forge.corpus import IndexPipeline, analyze_conventions, compute_stats
from forge.corpus.diagnostics import frontmatter_report, link_report
from forge.corpus.indexer import CorpusIndexer, detect_changes
from forge.domain import ChangeStatus
from forge.llm.base import CALLS
from forge.parsing.frontmatter import DiagnosticCode
from forge.parsing.links import LinkStatus
from forge.storage import SqliteStore


class TestCorpusIsUntouched:
    def test_indexing_does_not_modify_the_vault(self, real_settings, real_vault):
        """Exit criterion 1: the corpus remains unchanged."""
        before = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=real_vault,
            capture_output=True,
            text=True,
        ).stdout

        CorpusIndexer(real_settings).build_index()

        after = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=real_vault,
            capture_output=True,
            text=True,
        ).stdout
        assert before == after, "indexing must not change any tracked file"

    def test_no_markdown_file_mtime_changes(self, real_settings, real_vault):
        sample = sorted((real_vault / "DSA" / "01_Patterns").glob("*.md"))[:20]
        before = {p: p.stat().st_mtime_ns for p in sample}
        CorpusIndexer(real_settings).build_index()
        assert {p: p.stat().st_mtime_ns for p in sample} == before


class TestDeterminism:
    def test_same_corpus_produces_same_index(self, real_settings):
        """Exit criterion 2: deterministic indexing."""
        indexer = CorpusIndexer(real_settings)
        a = indexer.build_index()
        b = indexer.build_index()
        assert a.fingerprint() == b.fingerprint()
        assert a.file_count == b.file_count

    def test_discovery_order_is_stable(self, real_settings):
        indexer = CorpusIndexer(real_settings)
        assert indexer.discover() == indexer.discover()
        assert indexer.discover() == sorted(indexer.discover())

    def test_per_file_fields_are_reproducible(self, real_settings):
        indexer = CorpusIndexer(real_settings)
        a = {f.path: f.content_hash for f in indexer.build_index().files}
        b = {f.path: f.content_hash for f in indexer.build_index().files}
        assert a == b


class TestIndexingMakesNoModelCalls:
    def test_full_index_costs_zero_llm_calls(self, real_settings):
        """Exit criterion 5, first half."""
        CALLS.reset()
        CorpusIndexer(real_settings).build_index()
        assert CALLS.count == 0

    def test_reindexing_unchanged_corpus_costs_zero_llm_calls(self, real_settings, tmp_path):
        """Exit criterion 5: the headline incremental-processing guarantee."""
        store = SqliteStore(tmp_path / "zero.db")
        store.initialize()
        pipeline = IndexPipeline(real_settings, store)

        first = pipeline.run(write_reports=False)
        assert first.changes.summary()["new"] == first.index.file_count

        CALLS.reset()
        second = pipeline.run(write_reports=False)

        assert CALLS.count == 0
        assert second.changes.requires_processing == []
        assert second.persisted_sources == 0
        assert second.changes.summary()["unchanged"] == second.index.file_count
        store.close()


class TestSingleFileChangeIsolation:
    def test_editing_one_file_marks_only_that_file(self, real_settings, tmp_path):
        """Exit criterion 6, using the real corpus's own hashes.

        The vault is not modified: the edit is simulated by perturbing the
        stored hash, which exercises the same comparison.
        """
        index = CorpusIndexer(real_settings).build_index()
        previous = {f.path: f.content_hash for f in index.files}

        target = "DSA/01_Patterns/DFS.md"
        assert target in previous
        previous[target] = "0" * 64  # pretend the stored copy was older

        changes = detect_changes(index.files, previous)
        assert [c.path for c in changes.modified] == [target]
        assert len(changes.unchanged) == index.file_count - 1
        assert changes.new == [] and changes.deleted == []


class TestKnownHardCases:
    """Graph Traversal / DFS / BFS and the stem collisions, on real data."""

    @pytest.mark.parametrize(
        "path",
        [
            "DSA/01_Patterns/Graph Traversal.md",
            "DSA/01_Patterns/DFS.md",
            "DSA/01_Patterns/BFS.md",
        ],
    )
    def test_hard_case_files_are_indexed_distinctly(self, real_index, path):
        by_path = real_index.by_path()
        assert path in by_path
        assert by_path[path].content_hash

    def test_hard_case_files_have_distinct_hashes(self, real_index):
        by_path = real_index.by_path()
        hashes = {
            by_path[p].content_hash
            for p in (
                "DSA/01_Patterns/Graph Traversal.md",
                "DSA/01_Patterns/DFS.md",
                "DSA/01_Patterns/BFS.md",
            )
        }
        assert len(hashes) == 3

    def test_dfs_and_bfs_resolve_to_their_own_files(self, real_index):
        gt = real_index.by_path()["DSA/01_Patterns/Graph Traversal.md"]
        resolved = {
            link.target: link.resolved_path
            for link in gt.links
            if link.target in ("DFS", "BFS")
        }
        assert resolved.get("DFS") == "DSA/01_Patterns/DFS.md"
        assert resolved.get("BFS") == "DSA/01_Patterns/BFS.md"

    def test_real_stem_collisions_are_flagged_ambiguous(self, real_index):
        """The corpus violates its own one-canonical-home rule in three places."""
        report = link_report(real_index)
        ambiguous = report.ambiguous_targets
        assert "Heap" in ambiguous
        assert "Binary Search" in ambiguous
        for candidates in ambiguous.values():
            assert len(candidates) > 1

    def test_ambiguous_links_are_never_silently_resolved(self, real_index):
        for f in real_index.files:
            for link in f.links:
                if link.status is LinkStatus.AMBIGUOUS:
                    assert link.resolved_path is None
                    assert len(link.candidates) > 1


class TestCodeFenceHazardOnRealData:
    def test_python_literals_do_not_become_links(self, real_index):
        """The corpus has hundreds of `[[...]]` matrix literals in code blocks."""
        by_path = real_index.by_path()
        heavy = [f for f in by_path.values() if f.code_block_count > 0]
        assert len(heavy) > 100, "sanity: the corpus really is code-heavy"

        numeric_targets = [
            link.target
            for f in by_path.values()
            for link in f.links
            if link.target.replace(",", "").replace(" ", "").isdigit()
        ]
        assert numeric_targets == [], f"code literals leaked into links: {numeric_targets[:5]}"


class TestDiagnosticsOnRealData:
    def test_frontmatter_defects_are_found(self, real_index):
        """Exit criterion 4: malformed frontmatter is reported."""
        report = frontmatter_report(real_index)
        assert report.by_code.get(DiagnosticCode.YAML_PARSE_ERROR.value, 0) > 0
        assert report.by_code.get(DiagnosticCode.NESTED_LIST_WIKILINKS.value, 0) > 0
        assert report.repairable_files > 0

    def test_every_parse_error_has_a_verified_repair(self, real_index):
        """All 68 hard failures are mechanically repairable — but not applied."""
        unrepairable = [
            f.path
            for f in real_index.files
            if any(d.code is DiagnosticCode.YAML_PARSE_ERROR for d in f.diagnostics)
            and not any(r.verified for r in f.repairs)
        ]
        assert unrepairable == []

    def test_related_links_recoverable_despite_broken_yaml(self, real_index):
        """Reading the related: graph needs no repair to any file."""
        broken_with_related = [
            f
            for f in real_index.files
            if any(d.code is DiagnosticCode.YAML_PARSE_ERROR for d in f.diagnostics)
            and f.related
        ]
        assert len(broken_with_related) > 0

    def test_unresolved_links_are_reported(self, real_index):
        """Exit criterion 3."""
        report = link_report(real_index)
        assert report.unresolved_total > 0
        assert report.unresolved_distinct > 0
        for target, info in report.unresolved_targets.items():
            assert info["count"] >= 1
            assert info["sources"], f"{target} must name the files that link to it"

    def test_reports_serialize_to_json(self, real_index):
        import json

        for payload in (
            frontmatter_report(real_index).to_dict(),
            link_report(real_index).to_dict(),
            compute_stats(real_index).to_dict(),
            analyze_conventions(real_index).to_dict(),
        ):
            assert json.loads(json.dumps(payload)) == payload


class TestConventionAnalysis:
    def test_both_systems_are_reported_and_neither_is_chosen(self, real_index):
        report = analyze_conventions(real_index)
        assert {s["id"] for s in report.systems} == {"repo-wide", "dsa-local"}
        assert "UNRESOLVED" in report.resolution_status
        assert len(report.conflicts) >= 3

    def test_conformance_shows_the_conflict_is_real(self, real_index):
        report = analyze_conventions(real_index)
        # DSA files overwhelmingly follow Title Case, which the repo-wide rule
        # forbids. Both systems are genuinely in use.
        assert report.conformance["dsa-local"]["filename_pct"] > 90
        assert report.conformance["repo-wide"]["filename_pct"] < 60


class TestSpansOnRealData:
    def test_spans_cover_documents_and_are_ordered(self, real_settings, real_index):
        indexer = CorpusIndexer(real_settings)
        sources = {s.locator: s for s in indexer.to_sources(real_index)}
        by_path = real_index.by_path()

        for path in (
            "DSA/01_Patterns/Graph Traversal.md",
            "Technologies/Docs/rag.md",
            "README.md",
        ):
            indexed = by_path[path]
            _, spans = indexer.to_document_and_spans(indexed, sources[path])
            assert spans, f"{path} produced no spans"
            assert [s.ordinal for s in spans] == list(range(len(spans)))
            for s in spans:
                assert s.start_line <= s.end_line
                assert s.text.strip()

    def test_spans_carry_heading_paths(self, real_settings, real_index):
        indexer = CorpusIndexer(real_settings)
        source = next(
            s for s in indexer.to_sources(real_index) if s.locator == "Technologies/Docs/rag.md"
        )
        indexed = real_index.by_path()["Technologies/Docs/rag.md"]
        _, spans = indexer.to_document_and_spans(indexed, source)
        assert any(s.heading_path for s in spans)

    def test_span_ids_are_deterministic(self, real_settings, real_index):
        indexer = CorpusIndexer(real_settings)
        source = next(
            s for s in indexer.to_sources(real_index) if s.locator == "DSA/01_Patterns/DFS.md"
        )
        indexed = real_index.by_path()["DSA/01_Patterns/DFS.md"]
        first = indexer.to_document_and_spans(indexed, source)[1]
        second = indexer.to_document_and_spans(indexed, source)[1]
        assert [s.id for s in first] == [s.id for s in second]


class TestCorpusImportSemantics:
    def test_existing_corpus_is_user_authored_not_source_fact(self, real_settings, real_index):
        """The corpus is authoritative for belief, not evidence for truth."""
        from forge.domain import TrustTier

        sources = CorpusIndexer(real_settings).to_sources(real_index)
        assert sources
        assert all(s.trust_tier is TrustTier.USER_AUTHORED for s in sources)

    def test_no_claims_are_created_during_indexing(self, real_settings, tmp_path):
        """Phase 1 indexes; it does not assert. Claim extraction is Phase 2+."""
        store = SqliteStore(tmp_path / "c.db")
        store.initialize()
        IndexPipeline(real_settings, store).run(write_reports=False)
        counts = store.counts()
        assert counts["claims"] == 0
        assert counts["concepts"] == 0
        assert counts["sources"] > 0 and counts["spans"] > 0
        store.close()


class TestStatsMatchReality:
    def test_counts_are_computed_not_asserted(self, real_index, real_vault):
        stats = compute_stats(real_index)
        on_disk = len(
            [
                p
                for p in real_vault.rglob("*.md")
                if not any(
                    part.startswith(".") or part in ("engine", "tests", "scripts", "docker")
                    for part in p.relative_to(real_vault).parts
                )
            ]
        )
        assert stats.file_count == on_disk

    def test_folder_totals_sum_to_the_whole(self, real_index):
        stats = compute_stats(real_index)
        assert sum(v["files"] for v in stats.by_folder.values()) == stats.file_count
        assert sum(v["lines"] for v in stats.by_folder.values()) == stats.total_lines
