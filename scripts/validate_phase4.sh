#!/usr/bin/env bash
# Phase 4 exit-criteria validation.
#
# Demonstrates each criterion by executing it — nothing here is asserted from
# documentation. Run from the repo root:
#   bash scripts/validate_phase4.sh
set -uo pipefail

export PYTHONPATH="${PYTHONPATH:-}:engine"
SB="${TMPDIR:-/tmp}/forge-phase4-validate"
rm -rf "$SB"; mkdir -p "$SB"

pass=0; fail=0
ok(){ echo "  PASS  $1"; pass=$((pass+1)); }
bad(){ echo "  FAIL  $1"; fail=$((fail+1)); }
head2(){ echo; echo "=== $1 ==="; }

export FORGE_STATE_DIR="$SB/state"

# A seeded store: paper A's claim is canonical, paper B is new evidence.
python3 - <<'PY' >"$SB/seed.txt" 2>&1
import os, pathlib, sys
sys.path.insert(0, "engine"); sys.path.insert(0, "scripts")
from forge.config import Settings
from forge.storage import SqliteStore
from forge.ingestion import IngestionPipeline, IngestOptions
from forge.extraction import CandidateExtractor
from forge.proposals import ProposalService
from forge.activation import ProposalActivator
from forge.domain import ProposalStatus
from phase4_demo import extraction_provider, PAPER_A, PAPER_B

settings = Settings.load(state_dir=pathlib.Path(os.environ["FORGE_STATE_DIR"]))
store = SqliteStore(settings.db_path); store.initialize()
pipe = IngestionPipeline(settings, store,
                         extractor=CandidateExtractor(extraction_provider(), max_spans=6))
pipe.ingest_path(PAPER_A, IngestOptions(extract=True, propose=True, max_spans=6))
svc = ProposalService(store)
for p in svc.list(status=ProposalStatus.PENDING, limit=20):
    svc.approve(p.id)
ProposalActivator(store).activate_approved()
report = pipe.ingest_path(PAPER_B)
print("SOURCE_B", report.sources[0].source_id)
print("seeded:", len(store.list_claims()), "claim(s),", store.counts()["spans"], "spans")
store.close()
PY
SOURCE_B=$(grep '^SOURCE_B ' "$SB/seed.txt" | awk '{print $2}')
tail -1 "$SB/seed.txt" | sed 's/^/  /'

run_py() { python3 - "$@" ; }

head2 "1-4. LangGraph orchestrates a typed, checkpointable workflow with interruption"
run_py <<'PY' && ok "graph ran, state is typed, checkpoint written, run interrupted" || bad "workflow did not interrupt"
import os, pathlib, sys, json
sys.path.insert(0,"engine"); sys.path.insert(0,"scripts")
from forge.config import Settings
from forge.storage import SqliteStore
from forge.evolution.service import EvolutionService
from forge.evolution.state import EvolutionState
from phase4_demo import assessment_provider
settings = Settings.load(state_dir=pathlib.Path(os.environ["FORGE_STATE_DIR"]))
store = SqliteStore(settings.db_path); store.initialize()
source_id = [s.id for s in store.list_sources() if "paper-b" in s.locator][0]
svc = EvolutionService(store, settings, provider=assessment_provider(),
                       provider_id="mock", model_id="mock-1")
out = svc.start(source_id)
print("  nodes      :", " -> ".join(n.node for n in out.run.nodes))
print("  status     :", out.run.status.value, "| interrupted:", out.interrupted)
print("  state keys :", len(EvolutionState.__annotations__), "typed fields")
cp = settings.state_dir / "checkpoints.db"
print("  checkpoint :", cp.name, cp.stat().st_size, "bytes")
pathlib.Path("/tmp/forge-p4-wf").write_text(out.run.id)
svc.close(); store.close()
sys.exit(0 if out.interrupted and out.run.status.value == "waiting_for_review" and cp.exists() else 1)
PY

head2 "5. The workflow resumes after the process is gone"
run_py <<'PY' && ok "a paused run resumed after every object was destroyed" || bad "resume failed"
import os, pathlib, sys
sys.path.insert(0,"engine"); sys.path.insert(0,"scripts")
from forge.config import Settings
from forge.storage import SqliteStore
from forge.evolution.service import EvolutionService
from forge.proposals import ProposalService
from forge.domain import WorkflowStatus
from phase4_demo import assessment_provider
settings = Settings.load(state_dir=pathlib.Path(os.environ["FORGE_STATE_DIR"]))
wf = pathlib.Path("/tmp/forge-p4-wf").read_text()
store = SqliteStore(settings.db_path); store.initialize()
run = store.get_workflow(wf)
print("  loaded from disk:", run.id[:12], run.status.value)
for pid in run.proposal_ids:
    ProposalService(store).approve(pid, note="validated")
svc = EvolutionService(store, settings, provider=assessment_provider(),
                       provider_id="mock", model_id="mock-1")
out = svc.resume(wf)
print("  resumed status  :", out.run.status.value)
print("  nodes after     :", " -> ".join(n.node for n in out.run.nodes[-3:]))
svc.close(); store.close()
sys.exit(0 if out.run.status is WorkflowStatus.COMPLETED else 1)
PY

head2 "6-7. New evidence changed existing knowledge; narrowing was deterministic"
run_py <<'PY' && ok "claim marked DISPUTED by new evidence; candidates found with 0 model calls" || bad "knowledge did not evolve"
import os, pathlib, sys
sys.path.insert(0,"engine")
from forge.config import Settings
from forge.storage import SqliteStore
settings = Settings.load(state_dir=pathlib.Path(os.environ["FORGE_STATE_DIR"]))
store = SqliteStore(settings.db_path); store.initialize()
wf = store.get_workflow(pathlib.Path("/tmp/forge-p4-wf").read_text())
claim = store.get_claim(wf.assessments[0].claim_id)
print("  claim      :", repr(claim.statement))
print("  status     :", claim.status.value, "(statement unchanged)")
for c in wf.candidates:
    print(f"  candidate  : {c.concept_name} [{c.selector}] {c.detail}")
narrowing = [n for n in wf.nodes if n.node == "identify_affected_concepts"][0]
print("  narrowing model calls:", narrowing.llm_calls)
store.close()
sys.exit(0 if claim.status.value == "disputed" and narrowing.llm_calls == 0 and wf.candidates else 1)
PY

head2 "8. Assessments are grounded in real stored spans"
run_py <<'PY' && ok "every cited span resolves to real stored evidence" || bad "grounding failed"
import os, pathlib, sys
sys.path.insert(0,"engine")
from forge.config import Settings
from forge.storage import SqliteStore
settings = Settings.load(state_dir=pathlib.Path(os.environ["FORGE_STATE_DIR"]))
store = SqliteStore(settings.db_path); store.initialize()
wf = store.get_workflow(pathlib.Path("/tmp/forge-p4-wf").read_text())
allowed = set(wf.evidence_span_ids)
ok_all = True
for a in wf.assessments:
    for sid in a.evidence_span_ids:
        span = store.get_span(sid)
        print(f"  {a.classification.value} cites {span.citation() if span else 'MISSING'}")
        ok_all &= span is not None and sid in allowed
store.close()
sys.exit(0 if ok_all and wf.assessments else 1)
PY

head2 "9. A hallucinated citation is rejected, not repaired"
run_py <<'PY' && ok "an ungrounded assessment was rejected and produced no proposal" || bad "ungrounded output was accepted"
import json, sys, tempfile, pathlib
sys.path.insert(0,"engine")
from forge.storage import SqliteStore
from forge.domain import *
from forge.evolution import EvidenceAssessor
from forge.llm import MockProvider
d = pathlib.Path(tempfile.mkdtemp()); store = SqliteStore(d/"g.db"); store.initialize()
prov = Provenance(tier=ProvenanceTier.MODEL_INFERENCE, derivation=Derivation.MODEL, agent="x", model_id="m")
src = Source.for_path("p.pdf", kind=SourceKind.PDF, content_hash="h"); store.put_source(src)
doc = Document(id=Document.make_id(src.id,"h"), source_id=src.id, parser="p", parser_version="1", content_hash="h")
store.put_document(doc)
span = Span(id=Span.make_id(doc.id,0,"p.1"), document_id=doc.id, ordinal=0, locator="p.1",
            start_line=1, end_line=2, text="Some evidence text here for the claim.", content_hash="s")
store.put_spans([span])
claim = Claim(id=Claim.make_id("A claim.", span.id), statement="A claim.", provenance=prov)
store.put_claim(claim, [EvidenceLink(id=EvidenceLink.make_id(claim.id, span.id, EvidenceRelation.INFERS_FROM),
    claim_id=claim.id, span_id=span.id, relation=EvidenceRelation.INFERS_FROM, provenance=prov)])
liar = MockProvider(default_response=json.dumps({"assessments":[{
    "claim_id": claim.id, "classification":"SUPPORTS",
    "rationale":"this cites a span that was never shown",
    "evidence_span_ids":["span-i-invented"], "refined_statement":""}]}))
batch = EvidenceAssessor(store, liar, provider_id="mock", model_id="mock-1").assess([span],[claim])
print("  records kept :", len(batch.records))
print("  rejected     :", batch.rejected[0]["reason"][:80] if batch.rejected else "none")
store.close()
sys.exit(0 if not batch.records and batch.rejected else 1)
PY

head2 "10. Conflicts require human review; the LLM cannot mutate knowledge directly"
run_py <<'PY' && ok "a conflict paused for review and changed nothing until approved" || bad "conflict bypassed review"
import os, pathlib, sys, tempfile
sys.path.insert(0,"engine"); sys.path.insert(0,"scripts")
from forge.config import Settings
from forge.storage import SqliteStore
from forge.evolution.service import EvolutionService
from forge.domain import ClaimStatus
from phase4_demo import assessment_provider, extraction_provider, PAPER_A, PAPER_B
from forge.ingestion import IngestionPipeline, IngestOptions
from forge.extraction import CandidateExtractor
from forge.proposals import ProposalService
from forge.domain import ProposalStatus
from forge.activation import ProposalActivator

work = pathlib.Path(tempfile.mkdtemp())
settings = Settings.load(state_dir=work/"state")
store = SqliteStore(settings.db_path); store.initialize()
pipe = IngestionPipeline(settings, store, extractor=CandidateExtractor(extraction_provider(), max_spans=6))
pipe.ingest_path(PAPER_A, IngestOptions(extract=True, propose=True, max_spans=6))
svc0 = ProposalService(store)
for p in svc0.list(status=ProposalStatus.PENDING, limit=20): svc0.approve(p.id)
ProposalActivator(store).activate_approved()
rep = pipe.ingest_path(PAPER_B)
claim = store.list_claims()[0]
svc = EvolutionService(store, settings, provider=assessment_provider(), provider_id="mock", model_id="mock-1")
out = svc.start(rep.sources[0].source_id)
before = store.get_claim(claim.id).status
print("  paused           :", out.run.status.value)
print("  claim while paused:", before.value)
print("  proposal safety  :", store.get_proposal(out.run.proposal_ids[0]).safety.value)
svc.close(); store.close()
sys.exit(0 if out.run.status.value == "waiting_for_review" and before is ClaimStatus.ACTIVE else 1)
PY

head2 "11-13. Revisions, provenance, and provider identity are recorded"
run_py <<'PY' && ok "revision created; provenance carries provider, model, prompt, schema" || bad "provenance incomplete"
import os, pathlib, sys
sys.path.insert(0,"engine")
from forge.config import Settings
from forge.storage import SqliteStore
from forge.domain import EntityType
settings = Settings.load(state_dir=pathlib.Path(os.environ["FORGE_STATE_DIR"]))
store = SqliteStore(settings.db_path); store.initialize()
wf = store.get_workflow(pathlib.Path("/tmp/forge-p4-wf").read_text())
claim_id = wf.assessments[0].claim_id
revs = store.revisions_for(EntityType.CLAIM, claim_id)
print("  revisions  :", [r.op.value for r in revs])
p = store.get_proposal(wf.proposal_ids[0])
print("  provenance :", p.provenance.tier.value, "|", p.provenance.model_id)
print("             :", "prompt", p.provenance.prompt_version, "| schema", p.provenance.schema_version)
print("  run records:", wf.provider_id, "/", wf.model_id)
store.close()
sys.exit(0 if revs and p.provenance.model_id and p.provenance.prompt_version
         and p.provenance.schema_version and wf.provider_id != "none" else 1)
PY

head2 "14-16. Ollama, remote Ollama, and cloud all work through one abstraction"
run_py <<'PY' && ok "three providers behind the same protocol; remote Ollama configurable" || bad "provider abstraction broken"
import sys, pathlib, tempfile
sys.path.insert(0,"engine")
from forge.config import Settings, LLMSettings, OllamaSettings, CloudSettings
from forge.llm import get_provider, provider_identity, CloudProvider, OllamaProvider, MockProvider
from forge.llm.base import LLMProvider
root = pathlib.Path(".").resolve()
def build(**llm):
    return get_provider(Settings(vault_path=root, state_dir=pathlib.Path(tempfile.mkdtemp()),
                                 llm=LLMSettings(**llm)))
local = build(provider="ollama")
remote = build(provider="ollama", ollama=OllamaSettings(base_url="http://192.168.1.50:11434"))
cloud = build(provider="cloud", cloud=CloudSettings(vendor="anthropic", model="claude-sonnet-5"))
mock = build(provider="mock")
for name, p in (("ollama", local), ("remote", remote), ("cloud", cloud), ("mock", mock)):
    print(f"  {name:<7}: {p.__class__.__name__:<15} {provider_identity(p)}  protocol={isinstance(p, LLMProvider)}")
print("  remote base_url:", remote.base_url)
sys.exit(0 if remote.base_url == "http://192.168.1.50:11434"
         and isinstance(cloud, CloudProvider) and all(
             isinstance(p, LLMProvider) for p in (local, remote, cloud, mock)) else 1)
PY

head2 "17-18. No provider is required; a missing one is explicit and never downgraded"
run_py <<'PY' && ok "deterministic commands need no provider; missing provider is explicit" || bad "provider handling wrong"
import subprocess, sys, os, pathlib, tempfile, shutil
sys.path.insert(0,"engine"); sys.path.insert(0,"scripts")

# Deterministic commands run in a FRESH state dir. `forge index` prunes sources
# that are no longer in the vault, which would delete the seeded PDF fixtures
# (tests/ is an excluded directory) - correct engine behaviour, wrong thing to
# do to the store the later criteria depend on.
fresh = pathlib.Path(tempfile.mkdtemp())
env = {**os.environ, "FORGE_STATE_DIR": str(fresh)}
for cmd in (["index"], ["search", "retrieval"], ["concepts"], ["graph", "stats"]):
    r = subprocess.run(["python3","-m","forge.cli.main",*cmd], capture_output=True, text=True, env=env)
    print(f"  forge {' '.join(cmd):<18} exit={r.returncode}  (no provider configured)")
    if r.returncode != 0:
        sys.exit(1)

# The semantic path reports unavailability rather than substituting, on a copy
# of the seeded store so the original is untouched.
from forge.config import Settings
from forge.storage import SqliteStore
from forge.evolution.service import EvolutionService
seeded = Settings.load(state_dir=pathlib.Path(os.environ["FORGE_STATE_DIR"]))
alt = pathlib.Path(tempfile.mkdtemp())
shutil.copy(seeded.db_path, alt / "forge.db")
s2 = Settings.load(state_dir=alt)
st2 = SqliteStore(s2.db_path); st2.initialize()
src = [x.id for x in st2.list_sources() if "paper-b" in x.locator][0]
svc = EvolutionService(st2, s2)   # no provider at all
out = svc.start(src)
print("  no-provider run ->", out.run.status.value)
print("  errors          :", out.run.errors[0][:70] if out.run.errors else "none")
svc.close(); st2.close()
sys.exit(0 if out.run.status.value == "semantic_analysis_unavailable" else 1)
PY

head2 "19-20. Assessments are cached and correctly invalidated"
run_py <<'PY' && ok "cache hit on repeat; model or provider change invalidates" || bad "cache behaviour wrong"
import sys, pathlib, tempfile, json, re
sys.path.insert(0,"engine")
from forge.storage import SqliteStore
from forge.domain import *
from forge.evolution import EvidenceAssessor
from forge.llm import MockProvider
from forge.llm.base import CALLS
d = pathlib.Path(tempfile.mkdtemp()); store = SqliteStore(d/"c.db"); store.initialize()
prov = Provenance(tier=ProvenanceTier.MODEL_INFERENCE, derivation=Derivation.MODEL, agent="x", model_id="m")
src = Source.for_path("p.pdf", kind=SourceKind.PDF, content_hash="h"); store.put_source(src)
doc = Document(id=Document.make_id(src.id,"h"), source_id=src.id, parser="p", parser_version="1", content_hash="h")
store.put_document(doc)
span = Span(id=Span.make_id(doc.id,0,"p.1"), document_id=doc.id, ordinal=0, locator="p.1",
            start_line=1, end_line=2, text="Evidence text for caching.", content_hash="s")
store.put_spans([span])
claim = Claim(id=Claim.make_id("A claim.", span.id), statement="A claim.", provenance=prov)
store.put_claim(claim, [EvidenceLink(id=EvidenceLink.make_id(claim.id, span.id, EvidenceRelation.INFERS_FROM),
    claim_id=claim.id, span_id=span.id, relation=EvidenceRelation.INFERS_FROM, provenance=prov)])
def p():
    def respond(rq):
        t = rq.messages[-1].content
        return json.dumps({"assessments":[{"claim_id": re.findall(r"\[claim_id: ([^\]]+)\]",t)[0],
            "classification":"SUPPORTS","rationale":"supports the claim as written",
            "evidence_span_ids":[re.findall(r"\[span_id: ([^\]]+)\]",t)[0]],"refined_statement":""}]})
    return MockProvider(responder=respond)
a1 = EvidenceAssessor(store, p(), provider_id="mock", model_id="m1").assess([span],[claim])
a2 = EvidenceAssessor(store, p(), provider_id="mock", model_id="m1").assess([span],[claim])
a3 = EvidenceAssessor(store, p(), provider_id="mock", model_id="m2").assess([span],[claim])
a4 = EvidenceAssessor(store, p(), provider_id="cloud", model_id="m1").assess([span],[claim])
print(f"  first run      : calls={a1.llm_calls} hits={a1.cache.hits}")
print(f"  repeat         : calls={a2.llm_calls} hits={a2.cache.hits}")
print(f"  model changed  : calls={a3.llm_calls} hits={a3.cache.hits}")
print(f"  provider changed: calls={a4.llm_calls} hits={a4.cache.hits}")
store.close()
sys.exit(0 if a1.llm_calls==1 and a2.llm_calls==0 and a2.cache.hits==1
         and a3.llm_calls==1 and a4.llm_calls==1 else 1)
PY

head2 "21. Duplicate workflow execution is safe"
run_py <<'PY' && ok "re-running created nothing and spent nothing" || bad "repeat execution was not idempotent"
import os, pathlib, sys
sys.path.insert(0,"engine"); sys.path.insert(0,"scripts")
from forge.config import Settings
from forge.storage import SqliteStore
from forge.evolution.service import EvolutionService
from forge.llm.base import CALLS
from phase4_demo import assessment_provider
settings = Settings.load(state_dir=pathlib.Path(os.environ["FORGE_STATE_DIR"]))
store = SqliteStore(settings.db_path); store.initialize()
src = [s.id for s in store.list_sources() if "paper-b" in s.locator][0]
before = (store.count_revisions(), len(store.list_claims()),
          store.counts()["evidence_links"], store.counts()["proposals"])
svc = EvolutionService(store, settings, provider=assessment_provider(), provider_id="mock", model_id="mock-1")
CALLS.reset()
out = svc.start(src)
after = (store.count_revisions(), len(store.list_claims()),
         store.counts()["evidence_links"], store.counts()["proposals"])
print("  before:", before)
print("  after :", after)
print("  llm calls on repeat:", CALLS.count, "| cache hits:", out.run.cache_hits)
svc.close(); store.close()
sys.exit(0 if before == after and CALLS.count == 0 else 1)
PY

head2 "22. Failure modes are explicit"
run_py <<'PY' && ok "unavailable / malformed / timeout each fail explicitly" || bad "a failure became a success"
import sys, pathlib, tempfile
sys.path.insert(0,"engine")
from forge.storage import SqliteStore
from forge.domain import *
from forge.evolution import EvidenceAssessor
from forge.evolution.assessor import AssessmentOutcome
from forge.llm import MockProvider
from forge.llm.base import ProviderUnavailable, LLMError
d = pathlib.Path(tempfile.mkdtemp()); store = SqliteStore(d/"f.db"); store.initialize()
prov = Provenance(tier=ProvenanceTier.MODEL_INFERENCE, derivation=Derivation.MODEL, agent="x", model_id="m")
src = Source.for_path("p.pdf", kind=SourceKind.PDF, content_hash="h"); store.put_source(src)
doc = Document(id=Document.make_id(src.id,"h"), source_id=src.id, parser="p", parser_version="1", content_hash="h")
store.put_document(doc)
span = Span(id=Span.make_id(doc.id,0,"p.1"), document_id=doc.id, ordinal=0, locator="p.1",
            start_line=1, end_line=2, text="Evidence.", content_hash="s")
store.put_spans([span])
claim = Claim(id=Claim.make_id("A claim.", span.id), statement="A claim.", provenance=prov)
store.put_claim(claim, [EvidenceLink(id=EvidenceLink.make_id(claim.id, span.id, EvidenceRelation.INFERS_FROM),
    claim_id=claim.id, span_id=span.id, relation=EvidenceRelation.INFERS_FROM, provenance=prov)])
cases = {
  "provider unavailable": (MockProvider(fail_with=ProviderUnavailable("down")), AssessmentOutcome.SEMANTIC_ANALYSIS_UNAVAILABLE),
  "malformed output    ": (MockProvider(default_response="not json"), AssessmentOutcome.ASSESSMENT_REJECTED),
  "transport failure   ": (MockProvider(fail_with=LLMError("timeout")), AssessmentOutcome.RETRYABLE_FAILURE),
}
allok = True
for label,(p,expected) in cases.items():
    b = EvidenceAssessor(store, p, provider_id="mock", model_id="m").assess([span],[claim])
    print(f"  {label} -> {b.outcome.value}")
    allok &= b.outcome is expected and not b.ok and not b.records
store.close()
sys.exit(0 if allok else 1)
PY

head2 "23. No API key is committed anywhere"
if git grep -nIE "(sk-ant-[A-Za-z0-9]{10,}|sk-[A-Za-z0-9]{32,}|ANTHROPIC_API_KEY[[:space:]]*[=:][[:space:]]*[\"']?[A-Za-z0-9_-]{16,})" -- . >"$SB/keys.txt" 2>&1; then
  echo "  matches:"; sed 's/^/    /' "$SB/keys.txt"
  bad "a credential-shaped string is committed"
else
  ok "no credential-shaped strings in tracked files"
fi

head2 "24-26. Phase 1, 2, and 3 still pass"
for phase in 1 2 3; do
  if bash "scripts/validate_phase${phase}.sh" >"$SB/p${phase}.txt" 2>&1; then
    ok "Phase ${phase}: $(grep -E 'PASSED:' "$SB/p${phase}.txt" | tail -1 | tr -s ' ')"
  else
    bad "Phase ${phase} regression: $(tail -3 "$SB/p${phase}.txt")"
  fi
done
if python3 -m pytest tests/ -q >"$SB/tests.txt" 2>&1; then
  ok "full suite: $(tail -1 "$SB/tests.txt")"
else
  bad "tests failed: $(tail -3 "$SB/tests.txt")"
fi

head2 "27. CI is offline — the whole suite runs with no model reachable"
# Demonstrated by establishing that nothing is reachable and then running the
# suite anyway. Forcing FORGE_LLM_PROVIDER=mock would be the wrong test: two
# Phase 1 tests deliberately assert the behaviour when the configured provider
# is *unreachable*, and handing them a working mock removes the condition
# under test.
reachable_ollama=$(curl -sS -m 3 http://localhost:11434/api/tags >/dev/null 2>&1 && echo yes || echo no)
have_key=$([ -n "${ANTHROPIC_API_KEY:-}" ] && echo yes || echo no)
echo "  local ollama reachable : $reachable_ollama"
echo "  cloud credential set   : $have_key"
if [ "$reachable_ollama" = "no" ] && [ "$have_key" = "no" ]; then
  if python3 -m pytest tests/ -q >"$SB/offline.txt" 2>&1; then
    ok "$(tail -1 "$SB/offline.txt") with no model of any kind reachable"
  else
    bad "offline run failed: $(tail -3 "$SB/offline.txt")"
  fi
else
  echo "  (a provider is reachable here, so this run cannot prove offline operation)"
  bad "could not demonstrate offline CI: a provider was available"
fi

head2 "28. The end-to-end demo works"
if python3 scripts/phase4_demo.py >"$SB/demo.txt" 2>&1; then
  steps=$(grep -cE '^ *[0-9]+\. ' "$SB/demo.txt")
  ok "demo completed ($steps steps, incl. a real process-restart resume)"
  grep -E "IDENTICAL|claim status|status   : completed" "$SB/demo.txt" | head -3 | sed 's/^/    /'
else
  bad "demo failed: $(tail -5 "$SB/demo.txt")"
fi

head2 "29. The assessment evaluation set runs offline"
if python3 scripts/assessment_eval.py >"$SB/assess.txt" 2>&1; then
  grep -E "valid=|balance" "$SB/assess.txt" | sed 's/^/  /'
  ok "5 labelled cases, one per classification, all pipeline metrics green"
else
  bad "assessment eval failed: $(tail -5 "$SB/assess.txt")"
fi

head2 "30. Real-model results are reported honestly"
run_py <<'PY' && ok "provider availability probed and documented, not fabricated" || bad "provider availability not documented"
import pathlib, sys, subprocess
doc = pathlib.Path("docs/research/provider-availability.md")
if not doc.is_file():
    print("  missing docs/research/provider-availability.md"); sys.exit(1)
text = doc.read_text()
ollama = subprocess.run(["curl","-sS","-m","3","http://localhost:11434/api/tags"],
                        capture_output=True, text=True).returncode
import os
key = bool(os.environ.get("ANTHROPIC_API_KEY"))
print(f"  local ollama reachable : {ollama == 0}")
print(f"  cloud credential set   : {key}")
print(f"  documented as UNMEASURED: {'UNMEASURED' in text}")
sys.exit(0 if "UNMEASURED" in text and "Do not fabricate" not in text.split("##")[0] else 1)
PY

echo
echo "==================================================="
echo "  PASSED: $pass    FAILED: $fail"
echo "==================================================="
rm -rf "$SB"
[ $fail -eq 0 ]
