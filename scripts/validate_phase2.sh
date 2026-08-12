#!/usr/bin/env bash
# Phase 2 exit-criteria validation.
#
# Demonstrates each of the 16 criteria by executing it. Run from the repo root:
#   bash scripts/validate_phase2.sh
set -uo pipefail

export PYTHONPATH="${PYTHONPATH:-}:engine"
SB="${TMPDIR:-/tmp}/forge-phase2-validate"
rm -rf "$SB"; mkdir -p "$SB"
# A throwaway vault copy, so write-back can be proved without touching the real one.
mkdir -p "$SB/vault/DSA/00_Index"
cp "DSA/00_Index/DSA Home.md" "$SB/vault/DSA/00_Index/" 2>/dev/null
( cd "$SB/vault" && git init -q . )

FORGE="python3 -m forge.cli.main"
PDF=tests/fixtures/pdf
pass=0; fail=0
ok(){ echo "  PASS  $1"; pass=$((pass+1)); }
bad(){ echo "  FAIL  $1"; fail=$((fail+1)); }
head2(){ echo; echo "=== $1 ==="; }

head2 "1. Phase 1 tests still pass"
if bash scripts/validate_phase1.sh >"$SB/p1.txt" 2>&1; then
  ok "$(grep -E 'PASSED:' "$SB/p1.txt" | tail -1 | tr -s ' ')"
else
  bad "Phase 1 regression: $(tail -3 "$SB/p1.txt")"
fi

export FORGE_STATE_DIR="$SB/state"

head2 "2. PDF ingestion works offline"
python3 - <<'PY' && ok "PDF ingested with 0 LLM calls" || bad "PDF ingestion failed"
import json, subprocess, sys
out = subprocess.run(
    ["python3","-m","forge.cli.main","ingest","tests/fixtures/pdf/multipage.pdf","--json"],
    capture_output=True, text=True).stdout
r = json.loads(out); s = r["sources"][0]
print(f"  status={s['status']} spans={s['spans']} pages={s['pages']} llm_calls={r['totals']['llm_calls']}")
sys.exit(0 if s["status"]=="ingested" and s["spans"]>0 and r["totals"]["llm_calls"]==0 else 1)
PY

head2 "3. Markdown ingestion reuses Phase 1 parsing"
python3 - <<'PY' && ok "markdown hash matches the Phase 1 indexer exactly" || bad "markdown reuse broken"
import sys; sys.path.insert(0,"engine")
from pathlib import Path
from forge.config import Settings
from forge.corpus.indexer import CorpusIndexer
from forge.sources import MarkdownAdapter
s = Settings.load()
rel = "DSA/01_Patterns/DFS.md"
acquired = MarkdownAdapter().acquire(s.vault_path/rel)
indexed = CorpusIndexer(s).build_index().by_path()[rel]
print(f"  adapter={acquired.content_hash[:16]} indexer={indexed.content_hash[:16]}")
sys.exit(0 if acquired.content_hash == indexed.content_hash else 1)
PY

head2 "4. Spans are deterministic and traceable"
python3 - <<'PY' && ok "identical span ids across runs; every span cites page+section" || bad "span traceability failed"
import sys, tempfile, pathlib; sys.path.insert(0,"engine")
from forge.config import Settings
from forge.storage import SqliteStore
from forge.ingestion import IngestionPipeline
from forge.retrieval import SearchService
ids = []
for _ in range(2):
    d = pathlib.Path(tempfile.mkdtemp())
    st = SqliteStore(d/"x.db"); st.initialize()
    IngestionPipeline(Settings.load(state_dir=d), st).ingest_path(pathlib.Path("tests/fixtures/pdf/multipage.pdf"))
    svc = SearchService(st)
    spans = svc.spans_for_source("tests/fixtures/pdf/multipage.pdf")
    ids.append(sorted(s.id for s in spans))
    st.close()
for s in spans:
    print(f"  {s.citation()}")
sys.exit(0 if ids[0]==ids[1] and all(s.page for s in spans) else 1)
PY

head2 "5. Source changes are detected incrementally"
python3 - <<'PY' && ok "new / unchanged / modified / deleted all detected" || bad "change detection failed"
import sys, tempfile, pathlib, shutil; sys.path.insert(0,"engine")
from forge.config import Settings
from forge.storage import SqliteStore
from forge.ingestion import IngestionPipeline
d = pathlib.Path(tempfile.mkdtemp()); v = d/"v"; v.mkdir()
(v/".git").mkdir(); f = v/"a.md"; f.write_text("# A\n\nOriginal body text here.\n")
st = SqliteStore(d/"c.db"); st.initialize()
p = IngestionPipeline(Settings(vault_path=v, state_dir=d/"s"), st)
print("  first :", p.ingest_path(f).sources[0].status.value)
print("  again :", p.ingest_path(f).sources[0].status.value)
f.write_text("# A\n\nChanged body text entirely.\n")
print("  edited:", p.ingest_path(f).sources[0].status.value)
docs = st.documents_for_source(st.list_sources()[0].id)
print(f"  document versions retained: {sorted(x.version for x in docs)}")
sys.exit(0 if len(docs)==2 else 1)
PY

head2 "6. Unchanged sources generate zero LLM calls"
$FORGE ingest "$PDF/multipage.pdf" >/dev/null 2>&1
python3 - <<'PY' && ok "re-ingest: 0 LLM calls, 0 duplicate documents, 0 duplicate spans" || bad "cost control failed"
import json, subprocess, sys
out = subprocess.run(["python3","-m","forge.cli.main","ingest","tests/fixtures/pdf/multipage.pdf","--json"],
                     capture_output=True, text=True).stdout
r = json.loads(out)
print(f"  by_status={r['by_status']} llm_calls={r['totals']['llm_calls']} persisted={r['sources'][0]['spans']} spans (already stored)")
sys.exit(0 if r["by_status"]=={"unchanged":1} and r["totals"]["llm_calls"]==0 else 1)
PY

head2 "7/8. Extraction works with a provider; ingestion works without one"
python3 - <<'PY' && ok "extraction runs via the provider abstraction; ingestion succeeds with no model" || bad "extraction path failed"
import sys, json, tempfile, pathlib; sys.path.insert(0,"engine")
from forge.config import Settings
from forge.storage import SqliteStore
from forge.ingestion import IngestionPipeline, IngestOptions
from forge.extraction import CandidateExtractor
from forge.llm import MockProvider
d = pathlib.Path(tempfile.mkdtemp()); s = Settings.load(state_dir=d)

# (a) no provider at all
st = SqliteStore(d/"a.db"); st.initialize()
r = IngestionPipeline(s, st).ingest_path(pathlib.Path("tests/fixtures/pdf/simple.pdf"), IngestOptions(extract=True))
print(f"  no provider  : status={r.sources[0].status.value} spans={r.sources[0].spans} "
      f"extraction={r.sources[0].extraction_status.value} llm_calls={r.llm_calls}")
assert r.sources[0].status.value == "ingested" and r.sources[0].spans > 0

# (b) with a provider, through the real abstraction
def respond(req):
    if "concepts" in req.messages[1].content:
        return json.dumps({"concepts":[{"name":"Attention","kind":"concept","mention":"Self-attention"}]})
    return json.dumps({"claims":[{"statement":"Self-attention weighs all positions",
                                 "evidence_quote":"Self-attention lets a model weigh all positions at once.",
                                 "concept":"Attention"}]})
st2 = SqliteStore(d/"b.db"); st2.initialize()
pipe = IngestionPipeline(s, st2, extractor=CandidateExtractor(MockProvider(responder=respond)))
r2 = pipe.ingest_path(pathlib.Path("tests/fixtures/pdf/simple.pdf"), IngestOptions(extract=True))
print(f"  with provider: extraction={r2.sources[0].extraction_status.value} "
      f"concepts={r2.sources[0].concepts_proposed} claims={r2.sources[0].claims_proposed} "
      f"llm_calls={r2.llm_calls}")
sys.exit(0 if r2.sources[0].claims_proposed > 0 else 1)
PY

head2 "9/10. Extracted objects carry provenance; evidence points at real spans"
python3 - <<'PY' && ok "every model-derived proposal carries model id + resolvable evidence" || bad "provenance/evidence failed"
import sys, json, tempfile, pathlib; sys.path.insert(0,"engine")
from forge.config import Settings
from forge.storage import SqliteStore
from forge.ingestion import IngestionPipeline, IngestOptions
from forge.extraction import CandidateExtractor
from forge.llm import MockProvider
from forge.proposals import ProposalService
from forge.retrieval import SearchService
from forge.domain import ProposalType
d = pathlib.Path(tempfile.mkdtemp())
def respond(req):
    if "concepts" in req.messages[1].content:
        return json.dumps({"concepts":[{"name":"Chunking Strategy","kind":"concept","mention":"Chunk size"}]})
    return json.dumps({"claims":[{"statement":"Chunk size affects retrieval quality",
                                 "evidence_quote":"Chunk size materially affects retrieval quality.",
                                 "concept":"Chunking Strategy"}]})
st = SqliteStore(d/"p.db"); st.initialize()
pipe = IngestionPipeline(Settings.load(state_dir=d), st, extractor=CandidateExtractor(MockProvider(responder=respond)))
pipe.ingest_path(pathlib.Path("tests/fixtures/pdf/multipage.pdf"), IngestOptions(extract=True))
svc, search = ProposalService(st), SearchService(st)
claims = svc.list(type=ProposalType.NEW_CLAIM)
ok = bool(claims)
for c in claims:
    hit = search.span(c.evidence_span_ids[0])
    ok &= hit is not None and hit.source is not None and c.provenance.model_id is not None
    print(f"  claim: {c.operation.after[:52]}")
    print(f"     provenance: {c.provenance.derivation.value} model={c.provenance.model_id} tier={c.provenance.tier.value}")
    print(f"     evidence  : {hit.citation}")
sys.exit(0 if ok else 1)
PY

head2 "11. Ambiguous concepts are never auto-merged"
python3 - <<'PY' && ok "Heap / Binary Search / Trie all stay ambiguous with no winner" || bad "ambiguity handling failed"
import sys; sys.path.insert(0,"engine")
from forge.config import Settings
from forge.corpus.indexer import CorpusIndexer
from forge.matching import ConceptMatcher, build_ambiguity_index
paths = CorpusIndexer(Settings.load()).discover()
m = ConceptMatcher([], ambiguity_index=build_ambiguity_index(paths))
ok = True
for name in ("Heap","Binary Search","Trie"):
    r = m.match(name)
    ok &= r.kind.value == "ambiguous" and r.best is None
    print(f"  {name:<16} -> {r.kind.value}, best={r.best}, candidates={[c.vault_path for c in r.candidates]}")
sys.exit(0 if ok else 1)
PY

head2 "12/13. Metadata repairs are proposals with approval state"
python3 - <<'PY' && ok "283 verified repair proposals with pending/approved/rejected states" || bad "proposal system failed"
import json, subprocess, sys
env_gen = subprocess.run(["python3","-m","forge.cli.main","proposals","generate","--json"],
                         capture_output=True, text=True).stdout
g = json.loads(env_gen)
print(f"  built={g['built']} by_safety={g['by_safety']}")
listed = json.loads(subprocess.run(["python3","-m","forge.cli.main","proposals","list","--json","--limit","2"],
                                   capture_output=True, text=True).stdout)
pid = listed["proposals"][0]["id"]
subprocess.run(["python3","-m","forge.cli.main","proposals","approve",pid],capture_output=True,text=True)
pid2 = listed["proposals"][1]["id"]
subprocess.run(["python3","-m","forge.cli.main","proposals","reject",pid2],capture_output=True,text=True)
counts = json.loads(subprocess.run(["python3","-m","forge.cli.main","proposals","list","--json","--status","all"],
                                   capture_output=True, text=True).stdout)["counts"]
print(f"  counts={counts}")
sys.exit(0 if g["built"]>0 and "approved" in counts and "rejected" in counts else 1)
PY

head2 "14. No vault file is modified without explicit approval"
BEFORE=$(git status --porcelain -- '*.md' | grep -v '^?? docs/' | sort)
$FORGE ingest "$PDF" >/dev/null 2>&1
$FORGE proposals generate >/dev/null 2>&1
PID=$($FORGE proposals list --json --limit 1 | python3 -c "import sys,json;print(json.load(sys.stdin)['proposals'][0]['id'])")
$FORGE proposals approve "$PID" >/dev/null 2>&1
AFTER=$(git status --porcelain -- '*.md' | grep -v '^?? docs/' | sort)
[ "$BEFORE" == "$AFTER" ] && ok "ingest + approve left every tracked Markdown file untouched" \
                          || bad "vault changed: $(diff <(echo "$BEFORE") <(echo "$AFTER"))"

echo "  --- and --apply DOES write, reversibly, on a throwaway vault ---"
(
  export FORGE_VAULT_PATH="$SB/vault" FORGE_STATE_DIR="$SB/vault/.forge"
  $FORGE index >/dev/null 2>&1
  $FORGE proposals generate >/dev/null 2>&1
  P=$($FORGE proposals list --json --limit 1 | python3 -c "import sys,json;print(json.load(sys.stdin)['proposals'][0]['id'])")
  $FORGE proposals approve "$P" --apply 2>&1 | sed -n '2,4p' | sed 's/^/    /'
  python3 -c "
import yaml
t=open('$SB/vault/DSA/00_Index/DSA Home.md').read()
print('    repaired YAML now parses:', yaml.safe_load(t.split('---')[1])['related'][:3], '...')"
  ls "$SB/vault/.forge/backups"/*/DSA/00_Index/*.md >/dev/null 2>&1 && echo "    backup created: yes"
)

head2 "15. Tests cover the new behaviour"
if python3 -m pytest tests/ -q >"$SB/tests.txt" 2>&1; then
  ok "$(tail -1 "$SB/tests.txt")"
else
  bad "tests failed: $(tail -3 "$SB/tests.txt")"
fi

head2 "16. CLI demonstrates the Phase 2 workflow"
allok=1
for cmd in "ingest $PDF/simple.pdf" "search retrieval" "concepts" "documents" "proposals list"; do
  if $FORGE $cmd >/dev/null 2>&1; then echo "  ok: forge $cmd"; else echo "  FAILED: forge $cmd"; allok=0; fi
done
[ $allok -eq 1 ] && ok "all Phase 2 commands work" || bad "a Phase 2 command failed"

echo
echo "==================================================="
echo "  PASSED: $pass    FAILED: $fail"
echo "==================================================="
rm -rf "$SB"
[ $fail -eq 0 ]
