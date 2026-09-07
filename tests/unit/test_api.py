"""The Phase 6 read API, and the three gates it exists to satisfy.

Gate assertions are named for the gate they close:

* `test_gate_one_...` — from any claim, reach the exact source span in one
  interaction.
* `test_gate_two_...` — generated content is distinguishable from source
  evidence, which for an API means the tier travels with every payload and the
  explorer keys its styling on it.
* `test_gate_three_...` — the model is comprehensible with no chat interface
  present.

The rest guard properties that are easy to lose quietly: read-only, zero model
calls, and one SQLite connection per request.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi", reason="the API is an optional extra")
from fastapi.testclient import TestClient
from forge.api import API_VERSION
from forge.api.app import STATIC_DIR, create_app
from forge.config import Settings
from forge.domain import (
    Claim,
    Concept,
    ConceptKind,
    Derivation,
    Document,
    EntityType,
    EvidenceLink,
    EvidenceRelation,
    Provenance,
    ProvenanceTier,
    Source,
    SourceKind,
    Span,
    TrustTier,
    record_create,
)
from forge.llm.base import CALLS
from forge.storage import SqliteStore

SPAN_TEXT = (
    "A B-tree keeps every one of its leaves at exactly one depth below the root, "
    "which is what bounds a lookup to the height of the tree."
)


def _prov(tier: ProvenanceTier, derivation: Derivation, **kw) -> Provenance:
    return Provenance(tier=tier, derivation=derivation, **kw)


@pytest.fixture
def seeded(tmp_path: Path) -> Path:
    """A small but complete store: source -> document -> span -> claim -> concept.

    Deliberately mixes tiers. A fixture where everything is USER_ASSERTION
    could not show that generated content is marked differently, which is one
    of the gates.
    """
    db = tmp_path / "forge.db"
    store = SqliteStore(db)
    store.initialize()

    human = _prov(ProvenanceTier.USER_ASSERTION, Derivation.DETERMINISTIC, agent="bootstrap/0.1.0")
    model = _prov(
        ProvenanceTier.EXTRACTED_CLAIM,
        Derivation.MODEL,
        confidence=0.8,
        agent="extractor/0.2.0",
        model_id="test-model",
    )

    concept = Concept(
        id=Concept.make_id("B-tree Index"),
        canonical_name="B-tree Index",
        kind=ConceptKind.TECHNOLOGY,
        vault_path="Technologies/Docs/btree.md",
        provenance=human,
    )
    other = Concept(
        id=Concept.make_id("Vector Databases"),
        canonical_name="Vector Databases",
        kind=ConceptKind.TECHNOLOGY,
        provenance=human,
    )
    store.put_concept(concept)
    store.put_concept(other)
    store.append_revision(
        record_create(EntityType.CONCEPT, concept.id, {"canonical_name": "B-tree Index"})
    )

    source = Source.for_path(
        "Technologies/Docs/btree.md",
        kind=SourceKind.MARKDOWN,
        content_hash="hash-1",
        title="B-trees",
        # What `CorpusIndexer.to_sources` assigns to every vault file. The
        # default is `unverified`, which would make this fixture unlike the
        # corpus it stands in for.
        trust_tier=TrustTier.USER_AUTHORED,
    )
    store.put_source(source)
    document = Document(
        id=Document.make_id(source.id, "hash-1"),
        source_id=source.id,
        parser="test",
        parser_version="1",
        content_hash="hash-1",
    )
    store.put_document(document)
    span = Span(
        id=Span.make_id(document.id, 0, "L1"),
        document_id=document.id,
        ordinal=0,
        locator="L1-L2",
        heading_path=("Structure",),
        start_line=1,
        end_line=2,
        text=SPAN_TEXT,
        content_hash="span-1",
    )
    store.put_spans([span])
    store.rebuild_search_index()

    claim = Claim(
        id=Claim.make_id("B-tree leaves sit at one depth", span.id),
        statement="B-tree leaves all sit at the same depth.",
        subject_concept_id=concept.id,
        provenance=model,
    )
    store.put_claim(
        claim,
        [
            EvidenceLink(
                id=EvidenceLink.make_id(claim.id, span.id, EvidenceRelation.PARAPHRASES),
                claim_id=claim.id,
                span_id=span.id,
                relation=EvidenceRelation.PARAPHRASES,
                provenance=model,
            )
        ],
    )
    store.close()
    return db


@pytest.fixture
def client(seeded: Path, tmp_path: Path) -> TestClient:
    settings = Settings.load(state_dir=tmp_path / "state")
    return TestClient(create_app(settings, db_path=seeded))


def _first_claim(client: TestClient) -> dict:
    page = client.get("/claims").json()
    assert page["items"], "the fixture stored no claims"
    return page["items"][0]


# -- gate one --------------------------------------------------------------


def test_gate_one_a_claim_reaches_its_exact_source_span_in_one_request(client):
    """One HTTP request from a claim id yields the span's own words.

    "One interaction" is the gate, so the assertion is on the number of
    requests, not merely that the data is reachable somehow. A design that
    returned span *ids* and made the client fetch each one would satisfy a
    weaker reading of this and fail the gate.
    """
    claim_id = _first_claim(client)["id"]

    response = client.get(f"/claims/{claim_id}")

    assert response.status_code == 200
    evidence = response.json()["evidence"]
    assert evidence, "the claim came back with no evidence chain"
    first = evidence[0]
    assert first["text"] == SPAN_TEXT, "the span's verbatim text is not in the response"
    assert first["source_locator"] == "Technologies/Docs/btree.md"
    assert first["citation"], "no human-readable citation"
    assert first["span_id"]


def test_the_evidence_route_agrees_with_the_claim_route(client):
    """Two ways to the same chain must not drift apart."""
    claim_id = _first_claim(client)["id"]
    embedded = client.get(f"/claims/{claim_id}").json()["evidence"]
    standalone = client.get(f"/claims/{claim_id}/evidence").json()
    assert embedded == standalone


def test_the_span_route_returns_the_same_text_the_evidence_quoted(client):
    claim_id = _first_claim(client)["id"]
    span_id = client.get(f"/claims/{claim_id}/evidence").json()[0]["span_id"]
    span = client.get(f"/spans/{span_id}").json()
    assert span["text"] == SPAN_TEXT
    assert span["source_locator"] == "Technologies/Docs/btree.md"


# -- gate two --------------------------------------------------------------


def test_gate_two_every_entity_payload_carries_its_provenance_tier(client):
    """A UI can only distinguish generated content if the tier travels with it.

    Asserted across every route that returns an assertable object, because the
    one route that forgot would render model output as though a human wrote it.
    """
    concept = client.get("/concepts").json()["items"][0]
    claim = _first_claim(client)
    detail = client.get(f"/concepts/{concept['id']}").json()

    for payload, where in [
        (concept, "/concepts"),
        (claim, "/claims"),
        (detail["concept"], "/concepts/{id}.concept"),
        (detail["claims"][0], "/concepts/{id}.claims[]"),
    ]:
        assert payload["provenance"]["tier"], f"{where} published no tier"
        assert payload["provenance"]["derivation"], f"{where} published no derivation"


def test_gate_two_a_model_derived_claim_is_distinguishable_from_its_evidence(client):
    """The distinction the gate is about, on real data.

    The claim is `EXTRACTED_CLAIM` from a `MODEL` derivation; the span it cites
    is the document's own text under a `USER_AUTHORED` trust tier. A client
    that keys on these fields can style them differently; if they were equal
    here, no client could.
    """
    claim_id = _first_claim(client)["id"]
    detail = client.get(f"/claims/{claim_id}").json()

    # Lower-case: `Derivation` and `ProvenanceTier` serialize with different
    # casing, and a client that assumes one convention marks the wrong things.
    assert detail["provenance"]["derivation"] == "model"
    assert detail["provenance"]["tier"] == "EXTRACTED_CLAIM"
    assert detail["provenance"]["model_id"] == "test-model"

    evidence = detail["evidence"][0]
    assert evidence["trust_tier"] == "user_authored"
    assert evidence["trust_tier"] != detail["provenance"]["tier"]


def test_gate_two_the_explorer_styles_every_tier_and_not_by_colour_alone(client):
    """Colour is not an accessible signal on its own, so a shape is required.

    Checks the served page rather than the file on disk: a page that exists but
    is not reachable satisfies nothing.
    """
    page = client.get("/").text

    for tier in ProvenanceTier:
        assert f"tier-{tier.value}" in page, f"no styling for {tier.value}"

    # The structural markers that carry the distinction without colour.
    assert "border-left: 3px dashed" in page, "model-derived content has no dashed edge"
    assert "border-left: 3px solid" in page, "asserted content has no solid edge"
    assert "MODEL_TIERS" in page, "the page does not classify tiers as model-derived"


# -- gate three ------------------------------------------------------------


def test_gate_three_the_explorer_has_no_chat_interface(client):
    """The model must be comprehensible without one, so there is not one.

    Guards against a later "just add a little ask box": the gate is that the
    knowledge model reads on its own, and a prompt would let it stop doing so
    without anyone noticing.

    Asserted structurally rather than by searching for the word "chat", which
    the page's own comment uses to explain its absence. A word search would
    fail on documentation and pass on a prompt box named something else.
    """
    page = client.get("/").text
    body = re.sub(r"<!--.*?-->", "", page, flags=re.DOTALL)
    body = re.sub(r"<style>.*?</style>", "", body, flags=re.DOTALL)

    assert "<textarea" not in body.lower(), "the explorer has a free-text box"
    assert "<form" not in body.lower(), "the explorer has a form to submit"

    # The one input is the concept name filter, which queries a GET listing.
    inputs = re.findall(r"<input[^>]*>", body, flags=re.IGNORECASE)
    assert len(inputs) == 1, f"expected only the name filter, found {inputs}"
    assert 'type="search"' in inputs[0]

    # And nothing on the page sends anything anywhere.
    assert "method:" not in body, "a fetch specifies a method other than GET"
    assert "POST" not in body.upper().replace("POSTGRES", "")


def test_gate_three_every_entity_is_reachable_by_navigation(client):
    """Concept to claim to evidence to span, following only published links.

    Walks the chain the way a reader would, so a break anywhere in it fails
    here rather than in someone's browser.
    """
    concept_id = client.get("/concepts").json()["items"][0]["id"]
    detail = client.get(f"/concepts/{concept_id}").json()

    claim_id = detail["claims"][0]["id"]
    evidence = client.get(f"/claims/{claim_id}").json()["evidence"][0]
    span = client.get(f"/spans/{evidence['span_id']}").json()
    source = client.get(f"/sources/{span['source_id']}").json()

    assert source["locator"] == "Technologies/Docs/btree.md"


def test_the_explorer_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert (STATIC_DIR / "explorer.html").is_file()


# -- read-only -------------------------------------------------------------


def test_every_route_is_a_get(client):
    """Enforced from the schema, not from a promise in a docstring.

    Knowledge changes through proposal and activation, which need a human
    decision. An HTTP write path would be a second way in without that gate.
    """
    schema = client.get("/openapi.json").json()
    offenders = [
        f"{method.upper()} {path}"
        for path, methods in schema["paths"].items()
        for method in methods
        if method.lower() not in {"get", "head", "options", "parameters"}
    ]
    assert offenders == [], f"non-GET routes exist: {offenders}"


def test_the_api_makes_no_model_calls(client):
    """Deterministic by construction, and asserted the way the rest of Forge does."""
    CALLS.reset()
    for path in ["/health", "/stats", "/concepts", "/claims", "/sources", "/revisions"]:
        assert client.get(path).status_code == 200
    concept_id = client.get("/concepts").json()["items"][0]["id"]
    claim_id = _first_claim(client)["id"]
    client.get(f"/concepts/{concept_id}")
    client.get(f"/claims/{claim_id}")
    client.get("/search?q=leaves")

    assert CALLS.count == 0
    assert client.get("/stats").json()["llm_calls"] == 0


# -- the connection-per-request rule ---------------------------------------


def test_routes_work_from_the_threadpool(client):
    """The defect the first smoke run hit, pinned.

    `sqlite3` connections are bound to their creating thread and FastAPI runs
    sync endpoints in a threadpool, so a single shared store raised
    `ProgrammingError` on the first route that touched the database. Every
    route below runs on a worker thread; before the fix, `/stats` was a 500.
    """
    for path in ["/stats", "/concepts", "/claims", "/sources", "/revisions"]:
        assert client.get(path).status_code == 200, f"{path} failed from a worker thread"


def test_concurrent_requests_do_not_share_a_connection(client):
    """Two requests in flight at once, which is what a browser does."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: client.get("/stats").status_code, range(8)))
    assert results == [200] * 8


# -- listings, lookups, and their edges ------------------------------------


def test_a_missing_entity_is_a_404_not_a_500(client):
    for path in ["/concepts/nope", "/claims/nope", "/spans/nope", "/sources/nope"]:
        assert client.get(path).status_code == 404, path


def test_an_unknown_entity_type_names_the_valid_ones(client):
    response = client.get("/revisions/Nonsense/whatever")
    assert response.status_code == 400
    assert "Concept" in response.json()["detail"]


def test_concepts_can_be_filtered_by_name(client):
    page = client.get("/concepts", params={"q": "b-tree"}).json()
    assert page["total"] == 1
    assert page["items"][0]["canonical_name"] == "B-tree Index"


def test_paging_reports_the_full_total_not_the_window(client):
    """A client cannot page correctly if `total` counts only what it received."""
    page = client.get("/concepts", params={"limit": 1}).json()
    assert page["returned"] == 1
    assert page["total"] == 2
    assert page["limit"] == 1


def test_a_limit_beyond_the_ceiling_is_refused(client):
    """An unbounded listing is a way to make the server hold the whole store."""
    assert client.get("/concepts", params={"limit": 10_000}).status_code == 422


def test_search_finds_the_span_and_cites_its_source(client):
    hits = client.get("/search", params={"q": "leaves"}).json()
    assert hits, "lexical search returned nothing for a word in the span"
    assert hits[0]["text"] == SPAN_TEXT
    assert hits[0]["source_locator"] == "Technologies/Docs/btree.md"


def test_the_revision_timeline_reports_what_changed(client):
    concept_id = client.get("/concepts").json()["items"][0]["id"]
    rows = client.get(f"/revisions/Concept/{concept_id}").json()
    assert rows, "no revisions recorded for a concept that was created"
    assert rows[0]["op"] == "create"
    assert "canonical_name" in rows[0]["changed_fields"]


def test_a_path_that_does_not_exist_says_so_without_claiming_none_exists(client):
    """`found: false` means "not within max_depth", which is all a bounded search knows."""
    ids = [c["id"] for c in client.get("/concepts").json()["items"]]
    response = client.get("/path", params={"source": ids[0], "target": ids[1]})
    assert response.status_code == 200
    assert response.json()["found"] is False


def test_stats_reports_the_store_and_its_graph(client):
    stats = client.get("/stats").json()
    assert stats["api_version"] == API_VERSION
    assert stats["counts"]["concepts"] == 2
    assert stats["counts"]["claims"] == 1


def test_the_explorer_reads_the_graph_metric_names_the_api_publishes(client):
    """A renamed metric shows as a dash, not an error, so nothing else catches it.

    The first version of the overview read `g.isolated`, which
    `GraphMetrics.to_dict` calls `isolated_nodes`; the row rendered "-" against
    a graph with 545 nodes and looked plausible.
    """
    page = client.get("/").text
    published = set(client.get("/stats").json()["graph"] or {})
    assert published, "the fixture has no graph to check against"

    read = set(re.findall(r"\bg\.([a-z_]+)", page))
    unknown = read - published
    assert unknown == set(), f"the explorer reads metrics the API does not publish: {unknown}"


def test_the_openapi_schema_describes_every_route(client):
    """The schema is the API's documentation; an undescribed route is undiscoverable."""
    schema = client.get("/openapi.json").json()
    for path in ["/concepts", "/claims/{claim_id}", "/claims/{claim_id}/evidence", "/search"]:
        assert path in schema["paths"], f"{path} is missing from the schema"


def test_the_casing_of_the_fields_a_client_styles_on_is_pinned(client):
    """Three enums, and they do not agree with each other.

    `ProvenanceTier` serializes upper-case, `Derivation` and `TrustTier`
    lower-case. The explorer's first version compared `derivation === "MODEL"`
    and never matched, so a model-derived object was marked only when its tier
    happened to give it away. Pinned here because the failure is silent and
    renders generated content as though a human wrote it.
    """
    claim = client.get(f"/claims/{_first_claim(client)['id']}").json()

    assert claim["provenance"]["tier"] == "EXTRACTED_CLAIM"
    assert claim["provenance"]["derivation"] == "model"
    assert claim["evidence"][0]["trust_tier"] == "user_authored"


def test_the_explorer_escapes_values_it_renders(client):
    """Span text is arbitrary document content and reaches the page as data.

    Without escaping, a vault page containing markup would execute in the
    reader's browser. Checked structurally: the page defines an escaper and
    the raw-insertion sites go through it.
    """
    page = client.get("/").text
    assert re.search(r"const esc\s*=", page), "the explorer defines no escaper"
    assert page.count("esc(") > 20, "values are being interpolated without escaping"
