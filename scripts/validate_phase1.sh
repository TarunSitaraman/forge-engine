#!/usr/bin/env bash
# Phase 1 exit-criteria validation.
#
# Demonstrates each criterion by executing it, rather than asserting it in
# prose. Run from the repository root:  bash scripts/validate_phase1.sh
set -uo pipefail

export PYTHONPATH="${PYTHONPATH:-}:engine"
FORGE="python3 -m forge.cli.main"
pass=0; fail=0

ok()   { echo "  PASS  $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $1"; fail=$((fail+1)); }
head2(){ echo; echo "=== $1 ==="; }

head2 "1. Existing corpus remains unchanged"
before=$(git status --porcelain -- '*.md' | grep -v '^?? docs/' | sort)
$FORGE index >/dev/null 2>&1
after=$(git status --porcelain -- '*.md' | grep -v '^?? docs/' | sort)
[ "$before" == "$after" ] && ok "no tracked Markdown file changed by indexing" \
                          || bad "indexing modified Markdown: $(diff <(echo "$before") <(echo "$after"))"

head2 "2. Corpus is deterministically indexed"
f1=$($FORGE index --json --no-persist --no-reports 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["fingerprint"])')
f2=$($FORGE index --json --no-persist --no-reports 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["fingerprint"])')
[ "$f1" == "$f2" ] && ok "identical fingerprint across runs: ${f1:0:16}..." || bad "fingerprints differ: $f1 vs $f2"

head2 "3. Unresolved links are reported"
python3 - <<'PY' && ok "unresolved links reported with categories and candidates" || bad "no unresolved links reported"
import json,subprocess,sys
out=subprocess.run(["python3","-m","forge.cli.main","diagnostics","links","--json"],capture_output=True,text=True).stdout
r=json.loads(out)["links"]
s=r["summary"]
print(f"  {s['unresolved_occurrences']} occurrences / {s['unresolved_distinct_targets']} distinct targets")
print(f"  by status: {r['by_status']}")
sys.exit(0 if s["unresolved_occurrences"]>0 else 1)
PY

head2 "4. Malformed frontmatter is reported"
python3 - <<'PY' && ok "frontmatter defects reported with repair proposals" || bad "no frontmatter defects reported"
import json,subprocess,sys
out=subprocess.run(["python3","-m","forge.cli.main","diagnostics","frontmatter","--json"],capture_output=True,text=True).stdout
r=json.loads(out)["frontmatter"]
print(f"  by code: {r['by_code']}")
print(f"  repairable (NOT applied): {r['summary']['repairable_files']}")
sys.exit(0 if r["by_code"] else 1)
PY

head2 "5. Re-indexing unchanged content produces no LLM calls"
$FORGE index >/dev/null 2>&1
python3 - <<'PY' && ok "second index: 0 LLM calls, 0 persisted, all unchanged" || bad "re-index did work it should not have"
import json,subprocess,sys
out=subprocess.run(["python3","-m","forge.cli.main","index","--json"],capture_output=True,text=True).stdout
r=json.loads(out)
print(f"  llm_calls={r['llm_calls']} persisted={r['persisted']['sources']} changes={r['changes']}")
sys.exit(0 if r["llm_calls"]==0 and r["persisted"]["sources"]==0
         and r["changes"]["unchanged"]==r["files_indexed"] else 1)
PY

head2 "6. Modifying one file marks only that source changed"
python3 -m pytest tests/integration/test_real_corpus.py::TestSingleFileChangeIsolation \
  tests/integration/test_pipeline_and_cli.py::TestPipelineLifecycle::test_editing_a_file_reprocesses_only_it \
  -q >/dev/null 2>&1 && ok "single-file change isolation verified on real + fixture vaults" \
                     || bad "change isolation failed"

head2 "7. Provenance floor rules are enforced"
python3 -m pytest tests/unit/test_provenance.py -q >/dev/null 2>&1 \
  && ok "provenance rules enforced at the domain boundary" || bad "provenance tests failed"
python3 - <<'PY'
import sys; sys.path.insert(0,"engine")
from forge.domain import *
from forge.domain.provenance import ProvenanceInput
try:
    Provenance(tier=ProvenanceTier.SOURCE_FACT, derivation=Derivation.MODEL, agent="x", model_id="m",
               inputs=(ProvenanceInput(entity_type=EntityType.CLAIM, entity_id="c", tier=ProvenanceTier.SYNTHESIS),))
    print("  UNEXPECTED: violation allowed")
except ProvenanceViolation as e:
    print(f"  demonstrated: {str(e)[:88]}...")
PY

head2 "8. Revision history works"
python3 -m pytest tests/unit/test_revision_and_model.py tests/unit/test_storage.py -q >/dev/null 2>&1 \
  && ok "revision + non-destructive supersession verified" || bad "revision tests failed"
python3 - <<'PY'
import sys,tempfile,pathlib; sys.path.insert(0,"engine")
from forge.storage import SqliteStore
from forge.domain import *
s=SqliteStore(pathlib.Path(tempfile.mkdtemp())/"d.db"); s.initialize()
mk=lambda i,t: Claim(id=i, statement=t, provenance=deterministic_provenance("demo", ProvenanceTier.USER_ASSERTION))
s.put_claim(mk("c1","chunk size does not matter"))
s.supersede_claim("c1", mk("c2","chunk size materially affects retrieval"))
old=s.get_claim("c1")
print(f"  old claim retained : {old.statement!r} status={old.status.value} -> {old.superseded_by}")
print(f"  revisions for c1   : {[r.op.value for r in s.revisions_for(EntityType.CLAIM,'c1')]}")
PY

head2 "9. Local model is invokable through the provider abstraction"
python3 -m pytest tests/unit/test_llm_provider.py tests/unit/test_spike.py -q >/dev/null 2>&1 \
  && ok "provider abstraction verified (mock + ollama failure path)" || bad "provider tests failed"
$FORGE status --json 2>/dev/null | python3 -c '
import sys, json
d = json.load(sys.stdin)["llm"]
print("  configured: {} reachable={}".format(d["provider"], d["reachable"]))
print("  detail: {}".format(d["detail"][:80]))'

head2 "10. Local model limitations are documented"
[ -f docs/research/local-model-capability-spike.md ] \
  && grep -q "NOT RUN" docs/research/local-model-capability-spike.md \
  && ok "spike report exists and states honestly that it did not run" \
  || bad "spike report missing or misleading"

head2 "11. Engine runs without a paid API"
# Phase 4 deliberately added an optional cloud provider, so "the string
# 'anthropic' appears in the source" is no longer the right test. What must
# stay true — and is what this criterion always meant — is stricter:
#   a) no literal credential is committed anywhere in the repo, and
#   b) the default provider is local, so a fresh clone needs no paid account.
credential_hits=$(git grep -nIE "(sk-ant-[A-Za-z0-9]{10,}|sk-[A-Za-z0-9]{32,})" -- . || true)
default_provider=$(python3 -c "import sys; sys.path.insert(0,'engine'); from forge.config import LLMSettings; print(LLMSettings().provider)")
if [ -n "$credential_hits" ]; then
  echo "$credential_hits" | sed 's/^/    /'
  bad "a literal credential is committed"
elif [ "$default_provider" != "ollama" ]; then
  bad "default provider is '$default_provider', not the local one"
else
  ok "no committed credentials; default provider is local ('$default_provider')"
fi

head2 "12. Tests pass"
if python3 -m pytest tests/ -q >/tmp/forge-tests.txt 2>&1; then
  ok "$(tail -1 /tmp/forge-tests.txt)"
else
  bad "test suite failed: $(tail -3 /tmp/forge-tests.txt)"
fi

head2 "13. CLI demonstrates Phase 1 functionality"
allok=1
for cmd in "status" "corpus-stats" "diagnostics all" "index" "inspect README.md"; do
  if $FORGE $cmd >/dev/null 2>&1; then echo "  ok: forge $cmd"; else echo "  FAILED: forge $cmd"; allok=0; fi
done
$FORGE model-test --no-write >/dev/null 2>&1
[ $? -eq 1 ] && echo "  ok: forge model-test (exit 1, no model — correct)" || { echo "  FAILED: model-test exit code"; allok=0; }
[ $allok -eq 1 ] && ok "all six CLI commands work" || bad "a CLI command failed"

echo
echo "==================================================="
echo "  PASSED: $pass    FAILED: $fail"
echo "==================================================="
[ $fail -eq 0 ]
