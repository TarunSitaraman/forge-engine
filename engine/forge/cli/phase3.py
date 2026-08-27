"""Phase 3 CLI: activation, identity, graph, evaluation, embeddings.

Registered onto the existing app, so Phase 1 and Phase 2 commands are
unchanged. Every command supports ``--json`` and returns a non-zero exit code
on failure, so the whole workflow is scriptable.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from typing import Any, Optional

import typer

from ..activation import ProposalActivator, RelationshipActivator
from ..domain import LinkType, ProposalStatus, ProposalType, SafetyClass
from ..embeddings import HashingEmbeddingProvider, NullEmbeddingProvider, OllamaEmbeddingProvider
from ..evaluation import DEFAULT_DATASET, EvalDataset, RetrievalEvaluator
from ..graph import KnowledgeGraph, check_integrity
from ..llm.base import CALLS
from ..identity import IdentityConfig, IdentityService
from ..matching import build_ambiguity_index
from ..proposals import ProposalService
from ..retrieval import SearchService
from ..storage.sqlite_store import SqliteStore

graph_app = typer.Typer(no_args_is_help=True, help="Traverse and inspect the knowledge graph.")
identity_app = typer.Typer(
    no_args_is_help=True, help="Record explicit decisions about concept naming."
)
embeddings_app = typer.Typer(no_args_is_help=True, help="Optional local embeddings.")


def _emit(payload: Any, as_json: bool) -> bool:
    if as_json:
        typer.echo(jsonlib.dumps(payload, indent=2, sort_keys=True, default=str))
        return True
    return False


def _identity_service(settings: Any) -> IdentityService:
    return IdentityService(IdentityConfig.load(settings.vault_path / "config" / "concept-identity.yaml"))


def _embedding_provider(settings: Any, name: str):
    if name == "hashing":
        return HashingEmbeddingProvider()
    if name == "ollama":
        return OllamaEmbeddingProvider(settings.llm.base_url)
    return NullEmbeddingProvider()


def register(app: typer.Typer, settings_factory: Any) -> None:
    """Attach Phase 3 commands."""

    # -- activation --------------------------------------------------------

    @app.command()
    def activate(
        proposal_id: Optional[str] = typer.Argument(
            None, help="Proposal to activate. Omit to activate every approved proposal."
        ),
        vault: Optional[Path] = typer.Option(None),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Turn approved proposals into canonical Concepts and Claims."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        activator = ProposalActivator(store, identity=_identity_service(settings))

        if proposal_id:
            proposal, ambiguous = ProposalService(store).resolve(proposal_id)
            if proposal is None:
                typer.echo(
                    f"{proposal_id!r} matches {len(ambiguous)} proposals"
                    if ambiguous
                    else f"no proposal {proposal_id!r}",
                    err=True,
                )
                store.close()
                raise typer.Exit(code=1)
            report = activator.activate_all([proposal])
        else:
            report = activator.activate_approved()

        if _emit(report.to_dict(), json_out):
            store.close()
            return

        if not report.results:
            typer.echo("nothing to activate (no approved proposals awaiting activation)")
        for result in report.results:
            marker = {"created": "+", "already_active": "=", "refused": "-", "failed": "!"}[
                result.outcome.value
            ]
            typer.echo(f"[{marker}] {result.proposal_id[:12]}  {result.outcome.value}")
            typer.echo(f"      {result.reason}")
        typer.echo(f"\ncounts: {report.counts()}")
        store.close()

        if report.failed:
            raise typer.Exit(code=1)

    @app.command()
    def relationships(
        vault: Optional[Path] = typer.Option(None),
        min_cooccurrence: int = typer.Option(2, help="Shared spans required for RELATED_TO."),
        apply: bool = typer.Option(False, "--apply", help="Create the relationships."),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Discover evidence-backed concept relationships. Dry-run by default."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        activator = RelationshipActivator(store, min_cooccurrence=min_cooccurrence)

        candidates = activator.discover_cooccurrence()
        names = {c.id: c.qualified_name for c in store.list_concepts()}

        if not apply:
            payload = {
                "dry_run": True,
                "candidates": [
                    {**c.to_dict(), "from_name": names.get(c.from_concept_id), "to_name": names.get(c.to_concept_id)}
                    for c in candidates
                ],
            }
            if not _emit(payload, json_out):
                typer.echo(f"{len(candidates)} candidate relationship(s) — nothing created (pass --apply)")
                for c in candidates:
                    typer.echo(
                        f"  {names.get(c.from_concept_id, c.from_concept_id)} <-> "
                        f"{names.get(c.to_concept_id, c.to_concept_id)}  "
                        f"[{c.type.value}] {c.rationale}"
                    )
            store.close()
            return

        report = activator.activate(candidates)
        if not _emit(report.to_dict(), json_out):
            typer.echo(
                f"considered {report.candidates_considered}, created {report.created}, "
                f"already present {report.already_present}, rejected {len(report.rejected)}"
            )
            for rejection in report.rejected[:10]:
                typer.echo(f"  rejected: {rejection['reason']}")
        store.close()

    # -- knowledge lookups -------------------------------------------------

    @app.command()
    def concept(
        name: str = typer.Argument(..., help="Concept name, optionally namespace/Name."),
        vault: Optional[Path] = typer.Option(None),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Show a canonical concept: origin, evidence, claims, relationships."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        graph = KnowledgeGraph(store)

        namespace, _, bare = name.rpartition("/")
        matches = store.concepts_named(bare)
        if namespace:
            matches = [c for c in matches if c.namespace == namespace]
        if not matches:
            matches = [c for c in SearchService(store).concepts(bare, limit=5)]

        if not matches:
            typer.echo(f"no concept named {name!r}", err=True)
            store.close()
            raise typer.Exit(code=1)

        if len(matches) > 1 and not namespace:
            # Two concepts share this bare name. Show both rather than picking.
            payload = {
                "ambiguous": True,
                "candidates": [c.qualified_name for c in matches],
            }
            if not _emit(payload, json_out):
                typer.echo(f"{name!r} names {len(matches)} distinct concepts — specify one:")
                for c in matches:
                    typer.echo(f"  {c.qualified_name}   ({c.kind.value})")
            store.close()
            return

        detail = graph.explain_concept(matches[0].id)
        if _emit(detail, json_out):
            store.close()
            return

        info = detail["concept"]
        typer.echo(f"concept   : {info['qualified_name']}  [{info['kind']}]")
        if info["aliases"]:
            typer.echo(f"aliases   : {', '.join(info['aliases'])}")
        if info["vault_path"]:
            typer.echo(f"vault     : {info['vault_path']}")
        prov = detail["provenance"]
        typer.echo(
            f"provenance: {prov['tier']} via {prov['derivation']}"
            + (f" ({prov['model_id']})" if prov["model_id"] else "")
        )
        if origin := detail["origin_proposal"]:
            typer.echo(
                f"origin    : proposal {origin['id'][:12]} [{origin['status']}] "
                f"decided by {origin['decided_by']}"
            )
            typer.echo(f"            {origin['reason']}")
        if detail["origin_spans"]:
            typer.echo("evidence  :")
            for span in detail["origin_spans"]:
                typer.echo(f"  {span['citation']}")
                typer.echo(f"    {span['text'][:100]}")
        if detail["claims"]:
            typer.echo("claims    :")
            for claim in detail["claims"]:
                typer.echo(f"  [{claim['tier']}] {claim['statement']}")
        if detail["relationships"]:
            typer.echo("related   :")
            for rel in detail["relationships"]:
                typer.echo(f"  -[{rel['type']}]- {rel['label']}  ({rel['rationale']})")
        store.close()

    @app.command()
    def claim(
        claim_id: str = typer.Argument(..., help="Claim id (may be abbreviated)."),
        vault: Optional[Path] = typer.Option(None),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Show a claim and walk its evidence back to the source page."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()

        found = store.get_claim(claim_id) or next(
            (c for c in store.list_claims() if c.id.startswith(claim_id)), None
        )
        if found is None:
            typer.echo(f"no claim {claim_id!r}", err=True)
            store.close()
            raise typer.Exit(code=1)

        evidence = KnowledgeGraph(store).get_claim_evidence(found.id)
        payload = {
            "id": found.id,
            "statement": found.statement,
            "status": found.status.value,
            "tier": found.provenance.tier.value,
            "derivation": found.provenance.derivation.value,
            "model_id": found.provenance.model_id,
            "origin_proposal_id": found.origin_proposal_id,
            "subject_concept_id": found.subject_concept_id,
            "evidence": evidence,
        }
        if _emit(payload, json_out):
            store.close()
            return

        typer.echo(f"claim     : {found.statement}")
        typer.echo(f"tier      : {found.provenance.tier.value} ({found.provenance.derivation.value})")
        typer.echo(f"status    : {found.status.value}")
        if found.origin_proposal_id:
            typer.echo(f"origin    : proposal {found.origin_proposal_id[:12]}")
        typer.echo("evidence  :")
        for item in evidence:
            typer.echo(f"  [{item['relation']}] {item['citation']}")
            typer.echo(f"      source : {item['source_id'][:12] if item['source_id'] else '?'} "
                       f"({item['source_kind']}, {item['trust_tier']})")
            if item["text"]:
                typer.echo(f"      text   : {item['text'][:120]}")
        store.close()

    # -- graph -------------------------------------------------------------

    app.add_typer(graph_app, name="graph")

    @graph_app.command("show")
    def graph_show(
        entity: str = typer.Argument(..., help="Concept name or entity id."),
        vault: Optional[Path] = typer.Option(None),
        depth: int = typer.Option(2, help="Traversal depth (bounded)."),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Show a concept's neighbourhood, bounded by depth."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        graph = KnowledgeGraph(store)

        target = _resolve_entity(store, entity)
        if target is None:
            typer.echo(f"no concept or entity {entity!r}", err=True)
            store.close()
            raise typer.Exit(code=1)

        neighbors = graph.get_neighbors(target)
        related = graph.get_related_concepts(target, max_depth=depth)
        payload = {
            "entity_id": target,
            "neighbors": [n.to_dict() for n in neighbors],
            "related_within_depth": [
                {"concept": c.qualified_name, "distance": d} for c, d in related
            ],
        }
        if _emit(payload, json_out):
            store.close()
            return

        concept_obj = store.get_concept(target)
        typer.echo(f"entity: {concept_obj.qualified_name if concept_obj else target}")
        typer.echo(f"direct neighbours ({len(neighbors)}):")
        for n in neighbors:
            typer.echo(f"  -[{n.link.type.value}]-> {n.label}   {n.link.rationale or ''}")
        if related:
            typer.echo(f"reachable within depth {depth}:")
            for c, d in related:
                typer.echo(f"  {d} hop(s): {c.qualified_name}")
        store.close()

    @graph_app.command("path")
    def graph_path(
        source: str = typer.Argument(...),
        target: str = typer.Argument(...),
        vault: Optional[Path] = typer.Option(None),
        max_depth: int = typer.Option(3),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Find the shortest path between two entities, within a depth bound."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        graph = KnowledgeGraph(store)

        src, tgt = _resolve_entity(store, source), _resolve_entity(store, target)
        if src is None or tgt is None:
            typer.echo(f"could not resolve {'source' if src is None else 'target'}", err=True)
            store.close()
            raise typer.Exit(code=1)

        path = graph.find_path(src, tgt, max_depth=max_depth)
        payload = {
            "found": path is not None,
            "max_depth": max_depth,
            "path": path.to_dict() if path else None,
        }
        if not _emit(payload, json_out):
            if path is None:
                typer.echo(f"no path within {max_depth} hops (this does not prove none exists)")
            else:
                labels = graph.node_labels(path.nodes)
                typer.echo(path.describe(labels))
        store.close()

    @graph_app.command("stats")
    def graph_stats(
        vault: Optional[Path] = typer.Option(None),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Measure the graph. These numbers decide whether Neo4j is ever needed."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        metrics = KnowledgeGraph(store).metrics()

        if not _emit(metrics.to_dict(), json_out):
            data = metrics.to_dict()
            for key in (
                "nodes",
                "edges",
                "max_degree",
                "mean_degree",
                "branching_factor",
                "isolated_nodes",
                "neighbor_query_ms",
                "path_query_ms",
            ):
                typer.echo(f"{key:20}: {data[key]}")
            typer.echo(f"{'by_type':20}: {data['by_type']}")
        store.close()

    @app.command()
    def upstream(
        vault: Optional[Path] = typer.Option(None),
        pin: bool = typer.Option(
            False, "--pin", help="Record current commits, marking packs reviewed."
        ),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Check documented external repositories for drift. Zero model calls.

        `Projects/` describes repositories that live elsewhere and change
        without the vault noticing. A page declares its upstream in
        frontmatter:

            upstream_repo: https://github.com/you/project
            upstream_commit: 4e9355d6...

        Detection is `git ls-remote` against the recorded commit — no API
        token, no rate limit, and private repos work through the git
        credentials you already have.

        `--pin` records the current commit, which is how you say "I have
        reviewed this pack against the repo as it stands now". Rewriting a
        drifted pack is a separate step that needs judgement; this only
        reports. Exits 1 when anything has drifted.
        """
        from ..corpus.indexer import CorpusIndexer
        from ..upstream import CHECKED_KEY, COMMIT_KEY, check, declared_upstreams

        settings = settings_factory(vault)
        CALLS.reset()
        index = CorpusIndexer(settings).build_index()
        declared = declared_upstreams(index, settings.vault_path)

        if not declared:
            msg = (
                "no page declares an upstream. Add to a project pack's frontmatter:\n"
                "  upstream_repo: https://github.com/you/project"
            )
            if not _emit({"checked": 0, "detail": msg}, json_out):
                typer.echo(msg)
            return

        statuses = check(declared)
        drifted = [s for s in statuses if s.drifted]

        pinned: list[str] = []
        if pin:
            from datetime import date

            for status in statuses:
                if status.current is None:
                    continue
                target = settings.vault_path / status.path
                text = target.read_text(encoding="utf-8")
                updated = _rewrite_frontmatter_value(text, COMMIT_KEY, status.current)
                updated = _rewrite_frontmatter_value(
                    updated, CHECKED_KEY, date.today().isoformat()
                )
                if updated != text:
                    target.write_text(updated, encoding="utf-8")
                    pinned.append(status.path)

        payload = {
            "checked": len(statuses),
            "drifted": len(drifted),
            "pinned": pinned,
            "llm_calls": CALLS.count,
            "statuses": [s.to_dict() for s in statuses],
        }
        if not _emit(payload, json_out):
            for status in statuses:
                mark = {
                    "current": "ok      ",
                    "drifted": "DRIFTED ",
                    "unpinned": "unpinned",
                    "unreachable": "ERROR   ",
                }[status.state]
                typer.echo(f"{mark} {status.path}")
                if status.state == "drifted":
                    typer.echo(
                        f"         documented {status.recorded[:12]} -> now {status.current[:12]}"
                    )
                elif status.state == "unpinned":
                    typer.echo(f"         upstream at {status.current[:12]}; never pinned")
                elif status.state == "unreachable":
                    typer.echo(f"         {status.error}")
            if pinned:
                typer.echo(f"\npinned {len(pinned)} page(s) to the current commit")
            typer.echo(f"\n{len(statuses)} checked, {len(drifted)} drifted, {CALLS.count} llm calls")

        if drifted and not pin:
            raise typer.Exit(code=1)

    @app.command()
    def ask(
        question: str = typer.Argument(..., help="What you want to know."),
        vault: Optional[Path] = typer.Option(None),
        passages: int = typer.Option(8, help="Spans to put in front of the model."),
        semantic: bool = typer.Option(False, "--semantic", help="Re-rank with embeddings."),
        show_passages: bool = typer.Option(False, "--show-passages"),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Answer a question from the vault, citing every statement.

        One model call. Retrieval does the work; the model only writes up what
        was retrieved, and every citation is checked against the passages
        actually supplied. If retrieval finds nothing, no call is made and you
        are told the vault has nothing — the model never falls back on its own
        knowledge, which would make the answer untraceable.

        Exits 1 when the vault cannot answer, or when a citation does not
        resolve, so it composes in a script.
        """
        from ..answering import Answerer
        from ..llm import get_provider
        from ..llm.base import ProviderUnavailable
        from ..retrieval import SearchService

        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()

        provider = None
        try:
            provider = get_provider(settings)
        except ProviderUnavailable as exc:
            typer.echo(f"no model available: {exc}", err=True)
        except Exception as exc:
            typer.echo(f"provider error: {type(exc).__name__}: {exc}", err=True)

        answer = Answerer(
            SearchService(store, embeddings=_embedding_provider(settings, "hashing")),
            provider,
            passages=passages,
        ).ask(question, semantic=semantic)
        store.close()

        if _emit(answer.to_dict(), json_out):
            raise typer.Exit(code=0 if (answer.answered and answer.grounded) else 1)

        typer.echo(answer.text)
        if answer.sources():
            typer.echo("\nsources:")
            for n, src in zip(answer.cited, answer.sources()):
                typer.echo(f"  [{n}] {src}")
        if answer.invalid_citations:
            typer.echo(
                f"\nWARNING: the answer cited {answer.invalid_citations}, which were "
                f"never supplied — only {len(answer.passages)} passages were given. "
                f"Treat those statements as unsupported.",
                err=True,
            )
        if show_passages:
            typer.echo("\npassages considered:")
            for i, hit in enumerate(answer.passages, 1):
                typer.echo(f"  [{i}] {hit.citation}  (score {hit.score:.3f})")
        typer.echo(f"\nllm calls: {answer.llm_calls}")

        if not answer.answered or not answer.grounded:
            raise typer.Exit(code=1)

    @app.command()
    def bootstrap(
        vault: Optional[Path] = typer.Option(None),
        apply: bool = typer.Option(
            False, "--apply", help="Write the concepts and edges to the store."
        ),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Seed the knowledge graph from vault structure. Zero model calls.

        Filenames become concepts and links become MENTIONS edges. The vault
        already states what its concepts are — a human decided `Binary Search`
        deserves one canonical home and created the page. Reading that is
        deterministic; inferring it from prose is what extraction did badly.

        Nothing is written to the vault. This populates the derived store only,
        and it is a preview until `--apply`.
        """
        from ..bootstrap import build_plan
        from ..corpus.indexer import CorpusIndexer

        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()

        CALLS.reset()
        indexer = CorpusIndexer(settings)
        plan = build_plan(indexer.build_index(), decided=indexer._decided_targets())

        written = 0
        if apply:
            for concept in plan.concepts:
                store.put_concept(concept)
            for link in plan.links:
                store.put_link(link)
            written = len(plan.concepts) + len(plan.links)

        payload = {**plan.to_dict(), "applied": apply, "written": written, "llm_calls": CALLS.count}
        if not _emit(payload, json_out):
            typer.echo(f"concepts     : {len(plan.concepts)}")
            typer.echo(f"edges        : {len(plan.links)} (RELATED_TO, score 1.0)")
            typer.echo(f"pages skipped: {len(plan.skipped_pages)} (navigation, chapters, artifacts)")
            typer.echo(f"llm calls    : {CALLS.count}")
            typer.echo("\nby kind:")
            for kind, n in plan.by_kind().items():
                typer.echo(f"  {n:5}  {kind}")
            if plan.undecided_collisions:
                typer.echo(
                    f"\n{len(plan.undecided_collisions)} undecided name collision(s) left out "
                    f"of the graph — the engine will not pick one:"
                )
                for name, paths in sorted(plan.undecided_collisions.items())[:10]:
                    typer.echo(f"  {name}: {', '.join(paths)}")
                typer.echo("  Resolve with `forge identity decide \"<name>\" <qualified-name>`.")
            typer.echo(
                f"\n{'written to the store' if apply else 'preview only — re-run with --apply to write'}"
            )
        store.close()

    # -- identity ----------------------------------------------------------

    app.add_typer(identity_app, name="identity")

    @identity_app.command("scaffold")
    def identity_scaffold(
        vault: Optional[Path] = typer.Option(None),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Record the vault's name collisions in the identity file, undecided."""
        from ..corpus.indexer import CorpusIndexer

        settings = settings_factory(vault)
        service = _identity_service(settings)
        index = build_ambiguity_index(CorpusIndexer(settings).discover())
        added, skipped = service.scaffold(index)
        path = service.config.save(settings.vault_path / "config" / "concept-identity.yaml")

        payload = {
            "added": added,
            "skipped_existing": skipped,
            "path": str(path),
            "unresolved": [r.name for r in service.unresolved()],
        }
        if not _emit(payload, json_out):
            typer.echo(f"documented {added} collision(s), preserved {skipped} existing decision(s)")
            typer.echo(f"written to {path}")
            typer.echo("\nNothing was decided. Resolve one with:")
            typer.echo("  forge identity decide <name> <namespace>/<Name>")

    @identity_app.command("list")
    def identity_list(
        vault: Optional[Path] = typer.Option(None),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """List known collisions and how they are resolved."""
        settings = settings_factory(vault)
        service = _identity_service(settings)
        payload = {
            "collisions": [r.to_dict() for r in service.config.collisions.values()],
            "aliases": service.alias_map(),
        }
        if _emit(payload, json_out):
            return
        if not service.config.collisions:
            typer.echo("no collisions recorded — run `forge identity scaffold`")
        for resolution in service.config.collisions.values():
            status = resolution.default or "UNDECIDED"
            typer.echo(f"{resolution.name:<20} -> {status}")
            for identity in resolution.identities:
                marker = "*" if identity.qualified_name == resolution.default else " "
                typer.echo(f"   {marker} {identity.qualified_name:<28} {identity.vault_path or ''}")

    @identity_app.command("decide")
    def identity_decide(
        name: str = typer.Argument(..., help="The colliding name, e.g. Heap."),
        qualified: str = typer.Argument(..., help="Chosen identity, e.g. data-structure/Heap."),
        vault: Optional[Path] = typer.Option(None),
        by: str = typer.Option("cli"),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Decide which identity a bare name means."""
        settings = settings_factory(vault)
        service = _identity_service(settings)
        try:
            resolution = service.decide(name, qualified, by=by)
        except (KeyError, ValueError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1)
        path = service.config.save(settings.vault_path / "config" / "concept-identity.yaml")

        if not _emit({**resolution.to_dict(), "path": str(path)}, json_out):
            typer.echo(f"{name!r} now resolves to {qualified!r}")
            typer.echo(f"recorded in {path}")

    @identity_app.command("clear")
    def identity_clear(
        name: str = typer.Argument(...),
        vault: Optional[Path] = typer.Option(None),
    ) -> None:
        """Return a decided collision to the undecided state."""
        settings = settings_factory(vault)
        service = _identity_service(settings)
        try:
            service.clear(name)
        except KeyError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1)
        service.config.save(settings.vault_path / "config" / "concept-identity.yaml")
        typer.echo(f"{name!r} is undecided again")

    # -- embeddings --------------------------------------------------------

    app.add_typer(embeddings_app, name="embeddings")

    @embeddings_app.command("status")
    def embeddings_status(
        vault: Optional[Path] = typer.Option(None),
        provider: str = typer.Option("ollama", help="ollama | hashing | none"),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Report embedding availability and the retrieval degradation mode."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        embedder = _embedding_provider(settings, provider)
        service = SearchService(store, embeddings=embedder)

        payload = {
            "provider": provider,
            "model_id": embedder.model_id,
            "dimensions": embedder.dimensions,
            "available": embedder.available,
            "stored_vectors": store.count_embeddings(),
            "semantic_available": service.semantic_available,
            "degradation": service.degradation_note(),
        }
        if not _emit(payload, json_out):
            for key, value in payload.items():
                typer.echo(f"{key:20}: {value}")
        store.close()

    @embeddings_app.command("build")
    def embeddings_build(
        vault: Optional[Path] = typer.Option(None),
        provider: str = typer.Option("hashing", help="ollama | hashing"),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Embed every stored span. Optional — lexical retrieval works without it."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()
        embedder = _embedding_provider(settings, provider)
        service = SearchService(store, embeddings=embedder)

        if not embedder.available:
            typer.echo(
                f"provider {provider!r} unavailable; lexical retrieval continues unaffected",
                err=True,
            )
            store.close()
            raise typer.Exit(code=1)

        spans = []
        for source in store.list_sources():
            for document in store.documents_for_source(source.id):
                spans.extend(store.spans_for_document(document.id))
        embedded = service.index_embeddings(spans)

        payload = {"provider": provider, "model_id": embedder.model_id, "embedded": embedded}
        if not _emit(payload, json_out):
            typer.echo(f"embedded {embedded} span(s) with {embedder.model_id}")
        store.close()

    # -- evaluation --------------------------------------------------------

    @app.command(name="retrieval-eval")
    def retrieval_eval(
        vault: Optional[Path] = typer.Option(None),
        dataset: Optional[Path] = typer.Option(None, help="Labelled query set."),
        methods: str = typer.Option("lexical", help="Comma-separated: lexical,semantic,hybrid"),
        provider: str = typer.Option("hashing", help="Embedding provider for semantic/hybrid."),
        detail: bool = typer.Option(False, help="Include per-query scores."),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Measure retrieval against the labelled evaluation set."""
        settings = settings_factory(vault)
        store = SqliteStore(settings.db_path)
        store.initialize()

        # The default set lives in the repository, so resolve it against the
        # vault rather than the working directory — otherwise the command only
        # works when run from the repository root.
        target = dataset or (settings.vault_path / DEFAULT_DATASET)
        try:
            data = EvalDataset.load(target)
        except Exception as exc:
            typer.echo(str(exc), err=True)
            store.close()
            raise typer.Exit(code=2)

        rotted = data.verify_labels(settings.vault_path)
        wanted = tuple(m.strip() for m in methods.split(",") if m.strip())
        embedder = _embedding_provider(settings, provider) if wanted != ("lexical",) else None

        run = RetrievalEvaluator(store, embeddings=embedder).run(data, methods=wanted)
        payload = run.to_dict(include_scores=detail)
        payload["label_rot"] = rotted

        if _emit(payload, json_out):
            store.close()
            return

        typer.echo(f"dataset: {data.path} (v{data.version}, {len(data)} queries, {data.label_count()} labels)")
        typer.echo(f"categories: {data.categories()}")
        if rotted:
            typer.echo(f"WARNING: {len(rotted)} label(s) no longer resolve: {rotted[:3]}")
        typer.echo("")
        for summary in run.summaries:
            typer.echo("  " + summary.headline())
        if run.comparisons:
            typer.echo("\nvs lexical baseline:")
            for comparison in run.comparisons:
                typer.echo(f"  {comparison['candidate']:<20} {comparison['verdict']:<24} {comparison['deltas']}")
        for note in run.notes:
            typer.echo(f"\nnote: {note}")
        if detail:
            typer.echo("\nper-category (best method):")
            best = run.best()
            if best:
                for category, values in best.by_category.items():
                    typer.echo(
                        f"  {category:18} R@5={values['recall@5']:.3f} "
                        f"R@10={values['recall@10']:.3f} MRR={values['mrr']:.3f}"
                    )
        store.close()


def _resolve_entity(store: SqliteStore, token: str) -> str | None:
    """Resolve a concept name, qualified name, or raw id to an entity id."""
    if store.get_concept(token) is not None:
        return token
    namespace, _, bare = token.rpartition("/")
    matches = store.concepts_named(bare)
    if namespace:
        matches = [c for c in matches if c.namespace == namespace]
    if len(matches) == 1:
        return matches[0].id
    if matches:
        return matches[0].id
    if store.get_claim(token) is not None:
        return token
    return None


def _rewrite_frontmatter_value(text: str, key: str, value: str) -> str:
    """Set one frontmatter key, adding it if absent.

    Deliberately line-based rather than a YAML round-trip: re-emitting the
    document's frontmatter would reformat keys the user hand-wrote and reorder
    them, turning a one-line pin into a noisy diff.
    """
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---", 4)
    if end == -1:
        return text

    head, rest = text[4:end], text[end:]
    lines = head.split("\n")
    for i, line in enumerate(lines):
        if line.split(":", 1)[0].strip() == key:
            lines[i] = f"{key}: {value}"
            break
    else:
        lines.append(f"{key}: {value}")
    return "---\n" + "\n".join(lines) + rest
