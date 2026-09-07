"""The Phase 8 MCP server, and the two gates it exists to satisfy.

* `test_gate_one_...` — an external agent can query the model and receive
  provenance with every result.
* `test_gate_two_...` — identical semantics to the HTTP API, and no capability
  lives in only one interface.

Gate two is asserted twice over, because the two halves fail differently. The
registry comparison catches a capability added to one interface and forgotten
in the other; the payload comparison catches two interfaces that expose the
same names and disagree about what they mean.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

mcp_sdk = pytest.importorskip("mcp", reason="the MCP server is an optional extra")
fastapi = pytest.importorskip("fastapi", reason="parity needs the HTTP API too")

from fastapi.testclient import TestClient
from forge.api import queries
from forge.api.app import create_app
from forge.config import Settings
from forge.llm.base import CALLS
from forge.mcp.server import create_server
from mcp.server.mcpserver.exceptions import ToolError

# The Phase 6 fixture builds exactly the store both interfaces need, and
# sharing it is the point: a parity test on data one interface never sees
# proves nothing.
from tests.unit.test_api import SPAN_TEXT, seeded  # noqa: F401


@pytest.fixture
def server(seeded: Path, tmp_path: Path):  # noqa: F811
    settings = Settings.load(state_dir=tmp_path / "state")
    return create_server(settings, db_path=seeded)


@pytest.fixture
def linked(seeded: Path):  # noqa: F811
    """The seeded store plus one edge, and the ids it joins.

    Separate from `seeded` on purpose: the Phase 6 suite asserts that no path
    exists between its two concepts, so adding an edge to the shared fixture
    would break a test that is correct.
    """
    from forge.domain import ClaimLink, Derivation, LinkType, Provenance, ProvenanceTier
    from forge.storage import SqliteStore

    store = SqliteStore(seeded)
    a, b = (c.id for c in store.list_concepts())
    store.put_link(
        ClaimLink(
            id=ClaimLink.make_id(a, b, LinkType.RELATED_TO),
            from_id=a,
            to_id=b,
            type=LinkType.RELATED_TO,
            provenance=Provenance(
                tier=ProvenanceTier.USER_ASSERTION,
                derivation=Derivation.DETERMINISTIC,
                agent="bootstrap/0.1.0",
            ),
            score=1.0,
            rationale="human-authored link; not a computed similarity",
        )
    )
    store.close()
    return seeded, a, b


@pytest.fixture
def http(seeded: Path, tmp_path: Path) -> TestClient:  # noqa: F811
    settings = Settings.load(state_dir=tmp_path / "state")
    return TestClient(create_app(settings, db_path=seeded))


def call(server, name: str, arguments: dict | None = None):
    """Invoke a tool and return its structured content."""
    result = asyncio.run(server.call_tool(name, arguments or {}))
    assert not result.is_error, f"{name} errored: {result.content}"
    return result.structured_content


#: `GraphMetrics` times its own probe queries, so these differ between any two
#: calls including two calls to the same interface. They measure the call, not
#: the knowledge, and comparing them would make the parity test flaky rather
#: than strict.
MEASURED_AT_CALL_TIME = ("neighbor_query_ms", "path_query_ms")


def without_timings(payload):
    if isinstance(payload, dict) and isinstance(payload.get("graph"), dict):
        graph = {k: v for k, v in payload["graph"].items() if k not in MEASURED_AT_CALL_TIME}
        return {**payload, "graph": graph}
    return payload


def unwrap(payload):
    """MCP wraps a bare list return in `{"result": [...]}`; the HTTP body is the list.

    That is a transport difference, not a semantic one, so parity comparisons
    strip it rather than declaring the interfaces different.
    """
    if isinstance(payload, dict) and set(payload) == {"result"}:
        return payload["result"]
    return payload


def tool_names(server) -> set[str]:
    return {t.name for t in asyncio.run(server.list_tools())}


def ids(http: TestClient) -> tuple[str, str, str]:
    concept_id = http.get("/concepts").json()["items"][0]["id"]
    claim_id = http.get("/claims").json()["items"][0]["id"]
    span_id = http.get(f"/claims/{claim_id}/evidence").json()[0]["span_id"]
    return concept_id, claim_id, span_id


# -- gate two: one implementation, two interfaces --------------------------


def test_gate_two_every_capability_is_exposed_by_both_interfaces(server, http):
    """The registry is the contract; both interfaces are checked against it.

    Catches the failure that actually happens over time: a query added to the
    HTTP API and forgotten in MCP, or the reverse. Neither interface's own test
    suite would notice.
    """
    registry = set(queries.CAPABILITIES)
    schema = http.get("/openapi.json").json()
    http_ids = {
        methods[m]["operationId"]
        for methods in schema["paths"].values()
        for m in methods
        if "operationId" in methods[m]
    }
    # `health` is interface plumbing, not a knowledge capability: MCP has its
    # own liveness semantics and needs no equivalent.
    http_capabilities = http_ids - {"health"}

    assert tool_names(server) == registry, "MCP tools do not match the registry"
    assert http_capabilities == registry, "HTTP operation ids do not match the registry"


def test_gate_two_both_interfaces_return_the_same_payload(server, http):
    """Identical semantics, asserted on the bytes rather than on the intent.

    Same store, same arguments, both interfaces: the JSON must be equal. Two
    implementations that merely *look* alike would pass the registry check
    above and fail here the first time one of them rounded a score or renamed
    a field.
    """
    concept_id, claim_id, span_id = ids(http)
    cases = [
        ("get_stats", {}, "/stats", None),
        ("list_concepts", {}, "/concepts", None),
        ("get_concept", {"concept_id": concept_id}, f"/concepts/{concept_id}", None),
        (
            "list_concept_neighbors",
            {"concept_id": concept_id},
            f"/concepts/{concept_id}/neighbors",
            None,
        ),
        ("list_claims", {}, "/claims", None),
        ("get_claim", {"claim_id": claim_id}, f"/claims/{claim_id}", None),
        ("get_claim_evidence", {"claim_id": claim_id}, f"/claims/{claim_id}/evidence", None),
        ("get_span", {"span_id": span_id}, f"/spans/{span_id}", None),
        ("list_sources", {}, "/sources", None),
        ("list_recent_revisions", {}, "/revisions", None),
        (
            "list_entity_revisions",
            {"entity_type": "Concept", "entity_id": concept_id},
            f"/revisions/Concept/{concept_id}",
            None,
        ),
        ("search_spans", {"q": "leaves"}, "/search", {"q": "leaves"}),
    ]

    for tool, arguments, path, params in cases:
        over_mcp = without_timings(unwrap(call(server, tool, arguments)))
        over_http = without_timings(http.get(path, params=params).json())
        assert over_mcp == over_http, f"{tool} disagrees between the two interfaces"


def test_gate_two_a_source_lookup_agrees(server, http):
    """Split out because it needs an id the other cases do not fetch."""
    source_id = http.get("/sources").json()["items"][0]["id"]
    assert unwrap(call(server, "get_source", {"source_id": source_id})) == http.get(
        f"/sources/{source_id}"
    ).json()


def test_gate_two_a_bounded_path_search_agrees(server, http):
    concept_ids = [c["id"] for c in http.get("/concepts").json()["items"]]
    arguments = {"source": concept_ids[0], "target": concept_ids[1]}
    over_http = http.get("/path", params=arguments).json()
    assert unwrap(call(server, "find_path", arguments)) == over_http
    assert over_http["found"] is False


def test_gate_two_paging_arguments_mean_the_same_thing(server, http):
    """A shared implementation should make this true; asserted anyway.

    Paging is where two interfaces most easily diverge, because each has its
    own idiom for defaults.
    """
    over_mcp = unwrap(call(server, "list_concepts", {"limit": 1, "offset": 1}))
    over_http = http.get("/concepts", params={"limit": 1, "offset": 1}).json()
    assert over_mcp == over_http
    assert over_mcp["total"] == 2
    assert over_mcp["returned"] == 1


# -- gate one: provenance with every result --------------------------------


def test_gate_one_a_concept_result_carries_its_provenance(server):
    concept_id = unwrap(call(server, "list_concepts", {}))["items"][0]["id"]
    detail = call(server, "get_concept", {"concept_id": concept_id})

    assert detail["concept"]["provenance"]["tier"]
    assert detail["concept"]["provenance"]["derivation"]


def test_gate_one_a_claim_result_carries_provenance_and_its_evidence_carries_trust(server):
    """The two kinds of provenance, and why both are published.

    A claim has a provenance *tier*, which says how warranted it is. The span
    evidencing it has a *trust tier*, which says how much the source is worth.
    They are different questions and an agent needs both: a faithful extraction
    from a weak source is still a faithful extraction.
    """
    claim_id = unwrap(call(server, "list_claims", {}))["items"][0]["id"]
    detail = call(server, "get_claim", {"claim_id": claim_id})

    assert detail["provenance"]["tier"] == "EXTRACTED_CLAIM"
    assert detail["provenance"]["derivation"] == "model"
    assert detail["provenance"]["model_id"] == "test-model"
    assert detail["evidence"][0]["trust_tier"] == "user_authored"
    assert detail["evidence"][0]["text"] == SPAN_TEXT


def test_gate_one_search_hits_carry_their_source(server):
    """A hit is quoted source text, so it must say which source it was quoted from."""
    hits = unwrap(call(server, "search_spans", {"q": "leaves"}))
    assert hits, "search returned nothing for a word in the span"
    assert hits[0]["source_locator"] == "Technologies/Docs/btree.md"
    assert hits[0]["trust_tier"] == "user_authored"
    assert hits[0]["citation"]


def test_gate_one_every_tool_result_carries_some_provenance(server, http):
    """Swept across every capability, not spot-checked on the convenient ones.

    "Provenance" differs by kind: derived knowledge has a provenance tier,
    quoted source material has a trust tier and a locator, a revision has the
    cause that triggered it. What no result may do is arrive unattributable.
    """
    concept_id, claim_id, span_id = ids(http)
    source_id = http.get("/sources").json()["items"][0]["id"]
    # What counts as attribution differs by kind of result, and each of these
    # is a real answer to "where did this come from": derived knowledge has a
    # provenance tier, quoted source material has a trust tier and a locator, a
    # revision has the cause that triggered it, and a derived *report* names the
    # deterministic procedure that produced it.
    attribution = {
        "provenance",
        "trust_tier",
        "source_locator",
        "source_id",
        "citation",
        "cause",
        "op",
        "detected_by",
        "derived_by",
        "supporting_sources",
    }

    invocations = {
        "list_concepts": {},
        "get_concept": {"concept_id": concept_id},
        "list_claims": {},
        "get_claim": {"claim_id": claim_id},
        "get_claim_evidence": {"claim_id": claim_id},
        "get_span": {"span_id": span_id},
        "list_sources": {},
        "get_source": {"source_id": source_id},
        "list_entity_revisions": {"entity_type": "Concept", "entity_id": concept_id},
        "list_recent_revisions": {},
        "search_spans": {"q": "leaves"},
        # Phase 9.
        "get_belief": {"concept_id": concept_id},
        "list_gaps": {},
        "get_changes": {},
    }
    # Three capabilities cannot be swept on this fixture, and each has its own
    # test rather than being skipped silently: `get_stats` returns counts
    # rather than entities, and `find_path` and `list_concept_neighbors` need
    # an edge, which the `linked` fixture supplies.
    # Phase 9 adds five more that this fixture cannot sweep: it has no
    # questions and no syntheses, so those capabilities correctly return
    # nothing. Each has its own test rather than being skipped silently.
    uncovered = {
        "get_stats",
        "find_path",
        "list_concept_neighbors",
        "list_questions",
        "get_question",
        "get_question_evidence",
        "list_syntheses",
        "get_synthesis",
    }
    assert set(invocations) | uncovered == set(queries.CAPABILITIES)

    def attributed(record: dict) -> bool:
        """On the record, or on the entity it wraps.

        `get_concept` returns a detail envelope whose attribution sits on its
        `concept`. One level, deliberately: recursing further would let a
        response pass because something buried in it happened to carry a
        citation.
        """
        if attribution & set(record):
            return True
        return any(
            isinstance(value, dict) and attribution & set(value) for value in record.values()
        )

    for tool, arguments in invocations.items():
        payload = unwrap(call(server, tool, arguments))
        records = payload if isinstance(payload, list) else payload.get("items", [payload])
        assert records, f"{tool} returned nothing to check"
        for record in records:
            assert attributed(record), f"{tool} returned a record with no attribution"


def test_gate_one_a_neighbour_says_who_asserted_the_link(linked, tmp_path: Path):
    """Being told two concepts are related is useless without knowing on whose say-so.

    `RELATED_TO` alone reads like a measurement. The rationale is what says it
    came from a human writing a wikilink.
    """
    db, a, _ = linked
    server = create_server(Settings.load(state_dir=tmp_path / "state"), db_path=db)

    neighbors = unwrap(call(server, "list_concept_neighbors", {"concept_id": a}))

    assert neighbors, "the linked fixture produced no neighbours"
    assert neighbors[0]["provenance"]["tier"] == "USER_ASSERTION"
    assert "not a computed similarity" in neighbors[0]["rationale"]


def test_gate_two_neighbours_agree_between_the_interfaces(linked, tmp_path: Path):
    """The parity sweep runs on a store with no edges, so this covers the gap."""
    db, a, _ = linked
    settings = Settings.load(state_dir=tmp_path / "state")
    server = create_server(settings, db_path=db)
    client = TestClient(create_app(settings, db_path=db))

    over_mcp = unwrap(call(server, "list_concept_neighbors", {"concept_id": a}))
    assert over_mcp == client.get(f"/concepts/{a}/neighbors").json()


def test_gate_one_a_path_edge_says_who_asserted_it(linked, tmp_path: Path):
    """A path is a chain of assertions, so each hop needs its own provenance.

    Without this an agent is told two concepts are RELATED_TO and cannot tell
    a human-authored wikilink from a model's guess, which is the distinction
    the whole system is built to keep.
    """
    db, a, b = linked
    server = create_server(Settings.load(state_dir=tmp_path / "state"), db_path=db)

    path = call(server, "find_path", {"source": a, "target": b})

    assert path["found"] is True
    edge = path["edges"][0]
    assert edge["provenance"]["tier"] == "USER_ASSERTION"
    assert edge["provenance"]["derivation"] == "deterministic"
    assert "not a computed similarity" in edge["rationale"]


# -- read-only, and deterministic ------------------------------------------


def test_no_tool_can_write(server):
    """Read-only is the design, so the tool list must contain nothing else.

    An agent is the caller you least want holding a write path: knowledge
    changes through proposal and activation, which need a human decision.
    """
    forbidden = ("create", "update", "delete", "put", "post", "write", "activate", "approve")
    offenders = [n for n in tool_names(server) if n.startswith(forbidden)]
    assert offenders == [], f"tools that sound like writes: {offenders}"


def test_the_tools_make_no_model_calls(server, http):
    """Asserted the way the rest of the engine asserts it."""
    concept_id, claim_id, span_id = ids(http)
    CALLS.reset()
    for tool, arguments in [
        ("get_stats", {}),
        ("list_concepts", {}),
        ("get_concept", {"concept_id": concept_id}),
        ("get_claim", {"claim_id": claim_id}),
        ("get_span", {"span_id": span_id}),
        ("search_spans", {"q": "leaves"}),
    ]:
        call(server, tool, arguments)
    assert CALLS.count == 0


# -- errors an agent can act on --------------------------------------------


def test_a_missing_entity_tells_the_agent_which_id_was_not_found(server):
    """`ToolError`'s message reaches the agent; other exceptions do not.

    The SDK turns an unexpected exception into an opaque "error executing
    tool", which is right for a bug and useless for a bad id. A missing
    concept is not a bug and the agent can recover from being told so.
    """
    with pytest.raises(ToolError) as excinfo:
        asyncio.run(server.call_tool("get_concept", {"concept_id": "nope"}))
    assert "nope" in str(excinfo.value)
    assert "concept" in str(excinfo.value)


def test_an_unknown_entity_type_lists_the_valid_ones(server):
    with pytest.raises(ToolError) as excinfo:
        asyncio.run(server.call_tool(
            "list_entity_revisions", {"entity_type": "Nonsense", "entity_id": "x"}
        ))
    assert "Concept" in str(excinfo.value)


# -- what an agent is told about the tools ---------------------------------


def test_every_tool_is_described(server):
    """A tool an agent cannot understand is a tool it will use wrongly."""
    for tool in asyncio.run(server.list_tools()):
        assert tool.description, f"{tool.name} has no description"
        assert len(tool.description) > 30, f"{tool.name}'s description is too thin to act on"


def test_the_output_schemas_publish_provenance(server):
    """Provenance is part of the contract, not a convention to be discovered.

    An agent reading the tool list should be able to see that results carry
    attribution before it calls anything.
    """
    schemas = {t.name: json.dumps(t.output_schema or {}) for t in asyncio.run(server.list_tools())}
    for tool in ["get_concept", "get_claim", "list_concepts", "list_concept_neighbors"]:
        assert "provenance" in schemas[tool], f"{tool} does not publish provenance in its schema"
    for tool in ["get_span", "search_spans", "list_sources"]:
        assert "trust_tier" in schemas[tool], f"{tool} does not publish a trust tier"


def test_the_instructions_tell_an_agent_how_to_read_provenance(server):
    """The distinction is worthless if the agent flattens it when reporting.

    The server's instructions are the one place to say so before any tool is
    called.
    """
    instructions = server.instructions or ""
    assert "USER_ASSERTION" in instructions
    assert "MODEL_INFERENCE" in instructions
    assert "read-only" in instructions.lower()


# -- the real transport ----------------------------------------------------


def test_the_cli_serves_over_stdio_without_polluting_the_protocol(seeded: Path, tmp_path: Path):  # noqa: F811
    """Spawns `forge mcp` and speaks the real protocol to it.

    In-process tests exercise the tools but not the thing most likely to break
    an agent integration silently: **stdout is the protocol channel**, so a
    single stray `print` anywhere in startup corrupts the stream and every
    client sees a parse error rather than a Forge problem. The CLI writes its
    diagnostics to stderr for that reason, and this is what checks it.

    Also covers the CLI wiring itself, which no in-process test touches.
    """
    import os
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "forge.db").write_bytes(seeded.read_bytes())

    env = {
        **os.environ,
        "FORGE_VAULT_PATH": str(tmp_path),
        "FORGE_STATE_DIR": str(state),
    }
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "forge.cli.main", "mcp"],
        env=env,
        cwd=str(Path(__file__).resolve().parents[2] / "engine"),
    )

    async def drive():
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            init = await session.initialize()
            tools = await session.list_tools()
            stats = await session.call_tool("get_stats", {})
            missing = await session.call_tool("get_concept", {"concept_id": "nope"})
            return init, tools, stats, missing

    init, tools, stats, missing = asyncio.run(asyncio.wait_for(drive(), timeout=60))

    assert init.server_info.name == "forge"
    assert "USER_ASSERTION" in (init.instructions or ""), (
        "an agent is told nothing about how to read provenance"
    )
    assert {t.name for t in tools.tools} == set(queries.CAPABILITIES)

    counts = json.loads(stats.content[0].text)
    assert counts["counts"]["concepts"] == 2
    assert counts["llm_calls"] == 0

    assert missing.is_error, "a missing concept came back as a success"
    assert "nope" in missing.content[0].text
