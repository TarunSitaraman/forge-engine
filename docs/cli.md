# Forge CLI

*Every command, what it actually does, and what it will not do.*

**Commands are read-only with respect to the Markdown vault**, with one
explicit exception: `forge proposals approve --apply`. Everything else writes
only to `.forge/`, which is derived state and can be deleted at any time —
`forge index` rebuilds it.

---

## Install

```bash
pip install -e ".[dev]"     # from the repository root
forge --help
```

Python 3.10+. No model, no database server, and no API key is required for
anything below except `forge model-test`.

### macOS — `forge` as a global command

macOS ships Python 3.9.6, which is **below Forge's floor** and cannot load the
domain models (they use PEP 604 unions that pydantic evaluates at runtime). You
need a newer interpreter, and you want `forge` on your `PATH` without activating
a virtualenv first.

**Get Python from python.org, not Homebrew.** Download the *macOS 64-bit
universal2 installer* for 3.12 or 3.13 from
<https://www.python.org/downloads/macos/> and run the `.pkg` — a prebuilt
binary, about two minutes, no build step.

Once a Python version goes security-only its later patches are **source-only**,
so the newest 3.12.x may have no installer at all. To find the newest one that
does, without leaving the terminal:

```bash
MINOR=3.12; BASE=https://www.python.org/ftp/python
VER=$(for V in $(curl -s $BASE/ | grep -oE "${MINOR}\.[0-9]+" | sort -u -t. -k3 -nr | head -10); do
  curl -sfI "$BASE/$V/python-$V-macos11.pkg" >/dev/null 2>&1 && echo "$V" && break
done)
cd ~ && curl -fLO "$BASE/$VER/python-$VER-macos11.pkg"
open ~/"python-$VER-macos11.pkg"          # NOT `sudo installer` — see below
```

Then, from a directory that is **not** `~/Downloads`, `~/Desktop`, or
`~/Documents`:

```bash
cd ~
/Applications/Python\ 3.12/Install\ Certificates.command
```

Skipping that last step is the most common cause of `pip` failing with SSL
errors later: python.org builds ship without CA certificates wired up.

> **Verified on macOS 10.15.8 Catalina (Intel) with Python 3.12.10** — the
> oldest configuration this has been run on. Homebrew on that machine had no
> bottles and fell back to source builds; the python.org path took minutes.

```bash
python3.12 -m pip install --user pipx
python3.12 -m pipx ensurepath      # puts ~/.local/bin on PATH
```

Reload the shell so that `PATH` takes effect. Run this **on its own** — `exec`
replaces the shell process and silently discards anything pasted after it:

```bash
exec $SHELL -l
```

```bash
cd ~/forge                         # wherever you cloned it
pipx install --editable ".[dev]"
forge --help
```

**Why not Homebrew.** It is fine on a current machine, and if you already run it
`brew install python@3.12 pipx` works. But on an older Intel Mac there is often
no prebuilt bottle for the OS version, and Homebrew silently falls back to
compiling from source — including chains like `git → cmake → …`, each built
locally. That can run for hours on modest hardware, with no error to tell you
something went wrong. It is not worth adopting Homebrew just for this.

**Why `pipx` and not `pip install`.** pipx gives the CLI its own isolated
environment and links only the `forge` executable onto your `PATH`. It is also
the way around PEP 668: Homebrew and most Linux distributions mark their Python
"externally managed" and refuse a plain `pip install` with
`error: externally-managed-environment`. python.org builds do not set that flag,
which is why `pip install --user pipx` above works.

**Why `--editable`.** It keeps the installed command pointed at your checkout,
so edits to `engine/` take effect immediately with no reinstall — and it makes
Forge resolve the vault to that checkout from *any* working directory, which is
what you want for a single personal vault. Confirm with:

```bash
cd ~ && forge status | head -1      # -> vault : /Users/you/forge
```

**Tab completion** (macOS defaults to zsh):

```bash
forge --install-completion
```

Then, on its own line again, `exec $SHELL -l` — after which `forge <TAB>`
completes subcommands and flags.

**A model, if you want the LLM-backed commands.** Everything except
`forge model-test`, `forge evolve`, and extraction runs without one. The
provider is per-machine configuration — see the next section.

### Where settings live

Provider configuration is a property of the **machine**, not of the vault: the
same checkout is a GPU box on one machine and a laptop borrowing a hosted
endpoint on another, and the vault is shared between them by Git. So settings
live in a per-machine file rather than in the repository or a shell profile:

```bash
mkdir -p ~/.config/forge
cp config/forge.env.example ~/.config/forge/forge.env
chmod 600 ~/.config/forge/forge.env      # it can hold an API key
$EDITOR ~/.config/forge/forge.env
forge status                              # shows which file was loaded
```

On Windows the same path resolves under your user profile —
`%USERPROFILE%\.config\forge\forge.env`:

```powershell
mkdir "$env:USERPROFILE\.config\forge" -Force
copy config\forge.env.example "$env:USERPROFILE\.config\forge\forge.env"
notepad "$env:USERPROFILE\.config\forge\forge.env"
forge status
```

`~/.config/forge/forge.env` (or `$XDG_CONFIG_HOME/forge/forge.env`; override
with `FORGE_ENV_FILE`) is read for every setting on this page, including the API
key. Three layers resolve each value, highest first: an explicit CLI option, the
process environment, then this file. So a single command can always be
overridden without editing anything:

```bash
FORGE_LLM_PROVIDER=mock forge status
```

The format is `KEY=value`, one per line, with `#` comments, blank lines, a
leading `export `, and surrounding quotes all accepted. There is deliberately no
interpolation and no command substitution — a settings file that can execute is
a settings file that can surprise you, and this one holds a credential. A
malformed line fails at startup naming the file and line number.

**Loading the file never mutates the environment.** Values are resolved through
it, not exported into it, so nothing here leaks into processes Forge spawns —
and the key in particular is fetched at call time and never lands in
`os.environ`. `.gitignore` covers `forge.env` so a stray copy inside the repo
cannot be committed.

### Per-machine provider: the ASUS primary, a hosted open-weights fallback

**No paid API is required, and none is assumed.** Nothing in the engine branches
on which provider answered; only the recorded provenance differs, and it always
records *which* provider and model produced a result — so results from two
different models are never silently compared.

The intended setup is two tiers: the GPU box does the real work, and a hosted
open-weights endpoint covers the Mac when that box is off.

#### Tier 1 — the ASUS, from anywhere

The ASUS is the default provider and needs no configuration at all beyond the
model:

```bash
ollama pull qwen3:8b
export FORGE_MODEL_DEFAULT=qwen3:8b
export FORGE_LLM_TIMEOUT=300
```

`FORGE_OLLAMA_URL` points at any reachable host — nothing assumes the model runs
locally — so the Mac can drive it over the LAN:

```bash
export FORGE_OLLAMA_URL=http://<asus-hostname>:11434
```

To reach it off the LAN, put both machines on a private network (Tailscale or
equivalent) and use the private hostname. **Do not expose port 11434 to the
public internet** — Ollama has no authentication, so anything that can reach it
can use it.

Ollama binds to loopback by default, so the GPU box must be told to listen on
the private interface before anything else can connect. **The reference machine
here is Windows** (ASUS laptop, RTX 4050 ~6 GB VRAM, 16 GB RAM):

```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0:11434", "User")
```

Then quit Ollama from the system tray and relaunch it — the variable is read at
startup. On Linux the equivalent is `OLLAMA_HOST=0.0.0.0:11434 ollama serve`, or
the same variable in the systemd unit.

Verify from the box itself, then from the other machine:

```powershell
curl.exe http://localhost:11434            # -> Ollama is running
```

This tier is the one with a measurement behind it (Qwen3 8B, 5/5 — see below),
which is a reason to prefer it, not just a convenience.

#### Tier 2 — a hosted open-weights endpoint, when the ASUS is off

The cloud provider speaks two wire formats, and the second one —
`FORGE_CLOUD_VENDOR=openai` — is the de-facto shape for essentially every hosted
open-weights service (Groq, OpenRouter, Together, Cerebras, Fireworks) as well
as self-hosted servers (vLLM, llama.cpp, LM Studio). Pointing Forge at one is
configuration, not a code change.

**Presets** collapse the endpoint, the credential-variable name, and a safe
token ceiling into one value, because a base URL with the wrong path prefix is
the single most common way to get this wrong:

```bash
FORGE_LLM_PROVIDER=cloud
FORGE_CLOUD_PRESET=groq
FORGE_CLOUD_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_...
```

Known presets: `groq`, `openrouter`, `together`, `cerebras`, `fireworks`,
`lmstudio`, `llama-cpp`, `vllm`. An unknown name fails at startup with the list
rather than silently falling back to a different host.

**You always choose the model.** A preset supplies an endpoint, never a model —
omitting `FORGE_CLOUD_MODEL` is a startup error naming that variable, not a
guess. And a preset is a *default*, not a mode: every field it fills stays
individually overridable, and any other host works without one by setting
`FORGE_CLOUD_VENDOR=openai` plus `FORGE_CLOUD_BASE_URL` (the root that has
`/v1/chat/completions` beneath it — include any vendor path prefix, no trailing
`/v1`).

Presets are a convenience against a typo, not an integration: third-party
endpoints can change, and the explicit variables are always authoritative.

Three things worth knowing about this path:

- **Output ceilings are lower than a frontier model's.** Presets set 4096–8192;
  the bare default is 16000, sized for a 128K-output model. Gateways reject an
  over-large request rather than clamping it, and the 400 body is surfaced in
  the error. Override with `FORGE_CLOUD_MAX_TOKENS`.
- **Only the *name* of the credential variable is configuration.** The key is
  read at call time, never written to config, the store, provenance, or logs,
  and never copied into the environment. `forge status` reports whether a key is
  present without echoing it.
- **JSON mode is requested where the schema is known**
  (`response_format: {"type": "json_object"}`), and Forge validates the result
  regardless. A response that will not validate against the schema raises rather
  than becoming a degraded write — the same contract as every other provider.

Anthropic remains supported as a third option (`FORGE_CLOUD_VENDOR=anthropic`,
the default when no preset is set) if a key ever exists; nothing requires it.

#### Re-measure after any provider change

**A quality result belongs to a model, not to Forge.** The one real-model
measurement on record — 5/5 on the assessment set, 2026-08-14 — is Qwen3 8B via
Ollama and describes *only* that. Moving the Mac to a hosted open-weights model
does not inherit it, and the two must not be pooled.

The evaluation is reproducible, so re-run it rather than estimating:

```bash
# Whatever provider the environment is currently configured for
python3 scripts/assessment_eval.py --provider cloud --json    # or --provider ollama
forge model-test --repetitions 3 --note "host: <name>, model: <id>"
```

`assessment_eval.py` drives the production assessor and proposer — only the
provider changes between modes — and `forge model-test` writes its results to
[`research/local-model-capability-spike.md`](./research/local-model-capability-spike.md),
recording which provider and model produced them. Both refuse to invent a number
when the provider is unreachable; they report the unavailability instead.

Five cases were never enough to establish a rate even for Qwen3. Treat any new
provider's first run as a smoke test too, and read
[provider availability](./research/provider-availability.md) §6 before quoting
either as a rate.

> **Status of the Anthropic path.** It has still never completed a call. Until
> recently it could not: the request forwarded `temperature=0.0`, which current
> Anthropic models reject with a 400. The shape is now correct and asserted by
> tests, but a correct shape is not a measurement.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `zsh: command not found: forge` | `~/.local/bin` is not on `PATH`. Run `pipx ensurepath`, then `exec $SHELL -l`. |
| `error: externally-managed-environment` | You ran `pip install` against Homebrew Python. Use pipx as above, or a venv. |
| `configuration error: could not locate a Forge vault` | A non-editable install run outside any git repository. Either reinstall with `--editable`, or `export FORGE_VAULT_PATH=~/forge`. |
| `forge status` reports the wrong vault | A non-editable install resolves the vault from your current directory. Set `FORGE_VAULT_PATH` to pin it. |
| `pydantic` / `TypeError` on import | Python 3.9 or older. Check with `python3 -V`; reinstall against a newer one: `pipx install --python "$(python3.12 -c 'import sys; print(sys.executable)')" --editable ".[dev]"`. |
| `installer: ... NSInvalidArgumentException ... nil string parameter` | macOS privacy (TCC) is blocking access to the folder the `.pkg` is in — `~/Downloads` is protected, and `sudo installer` running as root gets denied. Use `open <pkg>` and click through the GUI installer instead. |
| `FileNotFoundError` from `os.getcwd()` in *any* Python command | Your shell's working directory is a TCC-protected folder (`~/Downloads`, `~/Desktop`, `~/Documents`) the interpreter has no permission for, so it cannot resolve its own cwd and dies before running anything. `cd ~` and retry. Grant Terminal Full Disk Access only if you actually need to work from those folders. |
| `WARNING: Install Certificates failed` | Same cause as the row above — run it from `~`. Leaving it unfixed makes every later `pip` call fail on SSL. |
| A pasted block stops silently after `exec $SHELL -l` | `exec` replaces the shell process and discards the rest of the buffered input, so the following lines never ran — no error, just a prompt. Paste it on its own. |
| `No module named pytest` | pipx installed the `[dev]` extras into its own venv, not your system Python. Run `"$(pipx environment --value PIPX_LOCAL_VENVS)/forge-engine/bin/python" -m pytest tests -q`. |
| `brew install` sits on `./bootstrap --prefix=...` for a very long time | No prebuilt bottle for your macOS version, so Homebrew is compiling from source. Not hung, but it can take hours on older hardware. Interrupting is safe — partial builds are discarded. Use the python.org installer instead. |
| Ollama `UNAVAILABLE` in `forge status` | Expected before a model host is configured — every deterministic command still works without one. If you did configure one, check it is running and reachable: `curl <host>:11434`. |

To upgrade after pulling new commits, an editable install needs nothing. To
rebuild it anyway: `pipx reinstall forge-engine`. To remove it entirely:
`pipx uninstall forge-engine`.

---

## Configuration

Environment variables, all optional:

| Variable | Default | Purpose |
|---|---|---|
| `FORGE_ENV_FILE` | `~/.config/forge/forge.env` | Per-machine settings file (see above) |
| `FORGE_VAULT_PATH` | see below | Vault to index |
| `FORGE_STATE_DIR` | `<vault>/.forge` | Derived state |
| `FORGE_LLM_PROVIDER` | `ollama` | `ollama`, `cloud`, or `mock` |
| `FORGE_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint — local or LAN |
| `FORGE_OLLAMA_THINK` | unset | Tri-state reasoning toggle; unset leaves the model's default |
| `FORGE_MODEL_DEFAULT` | `llama3.1:8b` | Model for all roles (Ollama) |
| `FORGE_MODEL_EXTRACTION` / `_ANALYSIS` / `_RESOLUTION` / `_SYNTHESIS` | — | Per-role override (Ollama) |
| `FORGE_CLOUD_PRESET` | — | `groq`, `openrouter`, `together`, `cerebras`, `fireworks`, `lmstudio`, `llama-cpp`, `vllm`. Fills the four fields below; you still set the model |
| `FORGE_CLOUD_VENDOR` | `anthropic` | `anthropic` or `openai` — a wire format, not a company. `openai` is the shape every open-weights host speaks |
| `FORGE_CLOUD_MODEL` | `claude-sonnet-5` | Model for every role on the cloud provider |
| `FORGE_CLOUD_API_KEY_ENV` | `ANTHROPIC_API_KEY` | **Name** of the variable holding the key |
| `FORGE_CLOUD_BASE_URL` | `https://api.anthropic.com` | API root; for `openai`, the path above `/v1/chat/completions` |
| `FORGE_CLOUD_MAX_TOKENS` | `16000` | Output cap. **Lower to 4096–8192 for open-weights models** |
| `FORGE_LLM_TIMEOUT` / `FORGE_LLM_MAX_RETRIES` | `120` / `2` | Per-call timeout and retries |
| `FORGE_LOG_LEVEL` / `FORGE_LOG_FORMAT` | `INFO` / `console` | `console` or `json` |

**No API key is ever configuration.** `FORGE_CLOUD_API_KEY_ENV` names the
variable to read; the key itself is read at call time and never written to
config, the store, provenance, or logs — a key in a YAML file is a key in Git.
Configuration is validated at startup: a bad vault path or an unbound model role
fails immediately rather than mid-run.

The cloud provider binds one model to every role, so the per-role
`FORGE_MODEL_*` variables apply to Ollama only — which is deliberate, since
their `llama3.1:8b` default is not a valid cloud model.

### How the vault is located

When neither a `--vault` option nor `FORGE_VAULT_PATH` is given, Forge looks for
a directory containing `.git`, in this order:

1. **Next to the installed engine.** A source checkout or an editable install
   puts `forge/config.py` inside the vault repository, so the command stays
   pinned to that vault from any working directory.
2. **Upward from the current directory.** For a non-editable install the engine
   lives in `site-packages`, so the only remaining signal is the vault you are
   standing in.

If neither finds one, Forge **fails with exit code 2** and tells you to set
`FORGE_VAULT_PATH`. It does not fall back to the current directory: doing so
meant `forge index` in an arbitrary directory would index that directory, write
a `.forge/` into it, and print a success line — silently operating on the wrong
thing instead of reporting that it could not find the right one.

---

## `forge index`

Walk the vault, hash every file, parse structure and metadata, resolve links,
detect changes, persist what changed, write reports.

```bash
forge index                 # normal run
forge index --json          # machine-readable
forge index --reset         # drop derived state first (safe: it rebuilds)
forge index --no-persist    # analyse without writing derived state
forge index --no-reports    # skip .forge/reports/
forge index -v              # verbose logging
```

```
Indexed 630 files in 0.7s
  fingerprint : 76ef7f11a21f5db7...
  changes     : 0 new, 0 modified, 630 unchanged, 0 deleted
  persisted   : 0 sources, 0 documents, 0 spans
  LLM calls   : 0
```

**`LLM calls: 0` is reported on every run, not only under test.** Indexing is
entirely deterministic; the `corpus` package cannot import `llm`.

Re-running with no edits persists nothing. Editing one file marks exactly that
file modified and reprocesses only it.

Writes to `.forge/reports/`: `corpus-stats.json`, `frontmatter-report.json`,
`link-report.json`, `convention-report.json`.

---

## `forge status`

Engine, derived-state, and provider status. Safe to run before anything is
indexed.

```
vault          : /home/user/forge
derived state  : /home/user/forge/.forge (exists: True)
markdown files : 630
indexed sources: 630
  spans=7003 documents=630 concepts=0 claims=0 revisions=630
llm provider   : ollama (UNAVAILABLE)
  cannot reach Ollama at http://localhost:11434: [Errno 111] Connection refused
```

`concepts=0 claims=0` is expected in Phase 1. Indexing describes the corpus; it
does not assert anything about it. Extraction is Phase 2+.

An unreachable provider is reported, never raised — status must work when
nothing is installed.

---

## `forge corpus-stats`

Statistics computed from the filesystem on every run. Nothing here is
maintained by hand, because the Phase 0 audit found stale counts in three
separate files.

```bash
forge corpus-stats
forge corpus-stats --json --top 20
```

Reports file/line/byte counts, frontmatter coverage, `canonical: true` count,
headings, code blocks, wikilink and Markdown-link totals, duplicate content
hashes, per-folder breakdown, and filename-style distribution.

---

## `forge diagnostics`

Report metadata, link, and convention problems. **Reports only — nothing is
modified.**

```bash
forge diagnostics                      # all
forge diagnostics frontmatter
forge diagnostics links --limit 40
forge diagnostics conventions
forge diagnostics --json               # full detail, including repair proposals
```

```
FRONTMATTER
  362/630 files have frontmatter; 294 valid, 68 invalid
    FM001: 68     YAML parse failure
    FM002: 215    nested-list wikilinks
    FM003: 268    no frontmatter
    FM008: 18     truncated final wikilink
  283 file(s) have verified repair proposals (NOT applied — approval required)

LINKS
  4656 total (4113 wiki, 589 markdown)
    ambiguous: 180   case_mismatch: 1   missing: 102   resolved: 4373
  unresolved: 282 occurrences across 89 distinct targets
     74x [ambiguous] 'Heap'  candidates=[...Patterns/Heap.md, ...DataStructures/Heap.md]

CONVENTIONS — UNRESOLVED — requires human decision (ADR-001 D3)
```

"Verified" means the proposed repair was applied in memory and re-parsed
successfully. It does **not** mean it was written. Applying repairs is a future
human-approved workflow (ADR-001 D2).

---

## `forge inspect <path>`

Everything deterministically known about one file.

```bash
forge inspect "DSA/01_Patterns/DFS.md"
forge inspect "Technologies/Docs/rag.md" --spans
forge inspect README.md --json
```

Shows hash, size, frontmatter state and keys, tags, recovered `related:` links,
heading count, code blocks, link counts, diagnostics, repair proposals as a
diff, unresolved links with candidates, and — with `--spans` — the derived
span breakdown with heading paths.

Useful for seeing a defect and its proposed fix side by side:

```
diagnostics    :
  [warning] FM008: Field 'related' ends with a truncated wikilink '[[Tree Traversal]'
  [error]   FM001: YAML parse failed: ...
repair proposals (NOT applied):
  line 6 verified=True
    - related: [[Binary Search Tree]], [[Tree Traversal]
    + related: ["Binary Search Tree", "Tree Traversal"]
```

---

## `forge model-test`

Run the local-model capability spike: structured concept extraction, claim
extraction, relationship extraction, and a small synthesis task, each repeated
so reliability is measured rather than sampled once.

```bash
forge model-test                          # writes docs/research/...
forge model-test --repetitions 5
forge model-test --no-write --json
forge model-test --note "M2 Pro, 32GB"    # repeatable
```

Writes `docs/research/local-model-capability-spike.md`.

**Exits non-zero when no model is reachable**, and the generated document says
so plainly rather than reporting an empty success. Contradiction detection is
deliberately not tested — it is not a required Phase 1 capability.

Requires Ollama:

```bash
ollama serve
ollama pull llama3.1:8b
export FORGE_MODEL_DEFAULT=llama3.1:8b
forge model-test
```

---

## `forge ingest <path>`

Ingest a PDF or Markdown file — or every supported file under a directory —
into the canonical knowledge model.

```bash
forge ingest paper.pdf
forge ingest ~/papers/                # directory; per-file outcomes
forge ingest paper.pdf --extract      # + LLM concept/claim candidates
forge ingest paper.pdf --force        # re-process even if unchanged
forge ingest paper.pdf --embed        # store embeddings if available
forge ingest paper.pdf --json
```

```
[ok] papers/rag-survey.pdf
    source 6l776o8b149t | document 6rtul9pel1lu
    3 spans, 3 pages, 329 chars

1 source(s) in 0.03s | 3 spans | 0 concepts | 0 claims | 0 proposals
LLM calls: 0  cache: {'hits': 0, 'misses': 0, 'writes': 0}
```

Deterministic by default — **no model is required**. `--extract` adds concept
and claim candidates and needs a local Ollama; without one, ingestion still
succeeds and reports `skipped_no_provider`.

Re-running on unchanged content reports `unchanged`, does no work, and makes
zero LLM calls. Statuses: `ingested`, `unchanged`, `ocr_required`,
`parse_failed`, `not_found`, `unsupported`. An image-only PDF reports
**`OCR_REQUIRED`** rather than pretending extraction succeeded.

---

## `forge search <query>`

Lexical search over spans. Returns **evidence with provenance**, not generated
prose.

```bash
forge search "chunking strategy"
forge search rag --source paper.pdf --page 2
forge search heap --kind pdf --heading "Data Structures"
forge search rag --semantic          # re-rank with embeddings if available
forge search rag --json
```

```
   1.892  papers/rag-survey.pdf :: p.1 | Retrieval Augmented Generation
          Retrieval Augmented Generation. RAG grounds generation in retrieved…
```

Filters: `--source`, `--kind`, `--page`, `--heading`, `--limit`. With
`--semantic` and no embeddings available, the command says so and returns
lexical results rather than failing.

---

## `forge concepts` / `forge documents`

```bash
forge concepts rag
forge documents --json
```

`concepts` is empty in Phase 2 by design — extraction produces *proposals*, not
concepts. The command says so rather than looking broken.

---

## `forge proposals`

Review and decide proposed changes. **Nothing is applied by default.**

```bash
forge proposals generate               # build metadata-repair proposals
forge proposals list                   # pending by default
forge proposals list --status all --type new_claim
forge proposals show <id>              # full detail incl. evidence
forge proposals approve <id>           # records the decision only
forge proposals approve <id> --apply   # ALSO writes to the vault
forge proposals reject <id> --note "wrong"
forge proposals approve-all --safety deterministic_verified          # dry run
forge proposals approve-all --safety deterministic_verified --no-dry-run
```

**Batch approval is guarded twice.** `approve-all` is a dry run by default —
it prints what *would* be approved and decides nothing until `--no-dry-run`.
And it refuses ambiguous proposals outright:

```
$ forge proposals approve-all --safety ambiguous
refusing to bulk-approve ambiguous proposals; pass --include-ambiguous
if that is genuinely what you want
$ echo $?
2
```

Bulk-approving an ambiguous semantic proposal is approving a decision nobody
made. Safety class stays derived from provenance and evidence — a model cannot
assert it about its own output.

`show` prints the change as a diff, the reason, its origin (deterministic vs
which model), the evidence spans with citations, and — for ambiguous concepts —
every candidate with no selection made.

**Approval is not application.** By default `approve` records the decision and
prints exactly which files *would* change:

```
approved 4t5k5g9us655 (metadata_repair)

vault write-back NOT performed (pass --apply). 1 file(s) would change:
  DSA/00_Index/DSA Home.md:6
    - related: [[Pattern Index]], [[Algorithm Index]]
    + related: ["Pattern Index", "Algorithm Index"]
```

With `--apply`, Forge backs the file up to `.forge/backups/<timestamp>/`,
records a revision holding both states, and writes only the named line. It
refuses if the proposal is unapproved, not `deterministic_verified`, or if the
target line changed since the proposal was generated.

Ids may be abbreviated. An abbreviation matching several proposals resolves to
none of them and lists the candidates.

---

## `forge activate [proposal]`

Turn approved proposals into canonical Concepts and Claims. With no argument,
activates every approved proposal awaiting activation.

```bash
forge activate                    # all approved proposals
forge activate 4t5k5g9us655       # one (ids may be abbreviated)
forge activate --json
```

Four outcomes, none of them silent:

```
[+] 4t5k5g9us65  created
      created concept 'Retrieval Augmented Generation'
[=] 9e3b45dcuk63  already_active
      concept 'Chunking Strategy' already exists
[-] 71fb28cc0a11  refused
      'Heap' is an unresolved collision and cannot be activated;
      decide it first: forge identity decide 'Heap' <one of [...]>
[!] 33ad91bb7e02  failed
      OperationalError: database is locked
```

`failed` leaves the proposal `APPROVED` so the same command retries it, and
exits non-zero. Activation is idempotent: running it twice creates nothing the
second time. **Nothing is written to Markdown** — canonical knowledge lives in
the derived store.

---

## `forge concept <name>` / `forge claim <id>`

Ask what Forge knows, and why.

```bash
forge concept "Retrieval Augmented Generation"
forge concept data-structure/Heap        # qualified, when the name collides
forge claim 3umn0uf7g0hd --json
```

`concept` prints the origin proposal, the evidence spans that caused it, its
claims, and its relationships. If a bare name matches several concepts it
**lists them and picks none**:

```
$ forge concept Heap
'Heap' names 2 distinct concepts — specify one:
  data-structure/Heap   (data_structure)
  pattern/Heap          (pattern)
```

`claim` walks the chain the other way — claim → evidence → span → page →
document → source — with the citation and trust tier at each step.

---

## `forge relationships`

Discover evidence-backed relationships between concepts. **Dry run by default.**

```bash
forge relationships                       # show candidates, create nothing
forge relationships --apply
forge relationships --min-cooccurrence 3  # raise the evidence bar
```

Rejections are printed with their reason, because the refusals are the point:

```
considered 3, created 1, rejected 2
  rejected: only 1 shared span(s); RELATED_TO requires at least 2
```

---

## `forge graph`

```bash
forge graph show "Retrieval Augmented Generation" --depth 2
forge graph path "Chunking Strategy" "Vector Database" --max-depth 3
forge graph stats --json
```

Every traversal is bounded by depth **and** a node budget. `path` reports
absence honestly:

```
no path within 3 hops (this does not prove none exists)
```

`stats` prints the measurements that decide whether a graph database is ever
justified — node and edge counts, branching factor, and query latency in
milliseconds.

---

## `forge identity`

Record explicit decisions about colliding concept names. No LLM is involved.

```bash
forge identity scaffold                          # document collisions, decide none
forge identity list
forge identity decide Heap data-structure/Heap
forge identity clear Heap                        # back to undecided
```

`scaffold` writes `config/concept-identity.yaml` with every collision found in
the vault and **no defaults set**. Re-running it preserves decisions already
made. `decide` refuses a qualified name that is not one of the collision's
actual identities.

---

## `forge embeddings`

Optional, off by default, and never required.

```bash
forge embeddings status
forge embeddings build --provider hashing   # deterministic, no download
forge embeddings build --provider ollama    # requires a local Ollama
```

`hashing` is a lexical-statistical vectorizer, **not** a neural embedding — it
exists so the embedding pathway can be measured without a model download. See
[the retrieval baseline](./research/retrieval-baseline.md).

---

## `forge retrieval-eval`

Measure retrieval against the labelled evaluation set.

```bash
forge retrieval-eval
forge retrieval-eval --methods lexical,semantic,hybrid --detail
forge retrieval-eval --json
```

```
dataset: tests/fixtures/eval/retrieval-v1.yaml (v1, 24 queries, 48 labels)

  lexical                R@5=0.406  R@10=0.608  P@5=0.158  MRR=0.471  misses=6  18.7ms/q
  semantic               R@5=0.301  R@10=0.581  P@5=0.133  MRR=0.342  misses=6  244.6ms/q
  hybrid(w=0.25)         R@5=0.378  R@10=0.544  P@5=0.158  MRR=0.337  misses=7  259.0ms/q

vs lexical baseline:
  semantic             regression   {...}
  hybrid(w=0.25)       regression   {...}
```

Labels are verified against the filesystem on every run; any that no longer
resolve are reported as `label_rot` rather than silently lowering recall.

---

## `forge diagnostics graph`

Structural integrity of the knowledge graph — nine codes, **report only**.

```bash
forge diagnostics graph --json
```

Nothing is repaired automatically. An automatic repair to a knowledge graph is
an unreviewed change to what the user believes.

---

## `forge evolve <source>`

Evaluate how an ingested source's evidence affects existing knowledge. This is
the Phase 4 command — the one that makes Forge evaluate rather than only store.

```bash
forge evolve paper-b.pdf          # id, vault locator, or bare filename
forge evolve paper-b.pdf --json
```

```
Forge Knowledge Evolution
──────────────────────────────

Source:
  papers/paper-b.pdf

Concepts affected:
  ✓ Retrieval Augmented Generation   [exact_name]

Claims examined:
  1

Semantic assessments:
  Potential Conflict: 1

Proposals:
  1

Status:
  WAITING_FOR_REVIEW

Workflow:
  6cjiiqmffd5fcdtdtkcrq24136

Waiting for review. Decide the proposals, then resume:
  forge proposals list --status pending
  forge workflow resume 6cjiiqmffd5f
```

**Nothing is applied.** The workflow pauses whenever it has produced a
proposal, and knowledge changes only after you approve it and resume.

Exit codes: `1` the source is not ingested or the run failed, `2` the semantic
provider is unavailable, `3` LangGraph is not installed (`pip install -e '.[agent]'`).

---

## `forge workflow`

```bash
forge workflow list --status waiting_for_review
forge workflow status <id>
forge workflow inspect <id>          # "why did Forge propose this?"
forge workflow resume <id>
```

`inspect` is the accountability command. It prints which concepts were
considered **and the selector that found each one**, which claims were
examined, what the model concluded and on which spans, the proposals that
resulted, what a human decided, the revisions that followed, and the cost:

```
concepts considered, and why:
  Retrieval Augmented Generation   [exact_name]  'Retrieval Augmented Generation' appears in the evidence

assessments:
  POTENTIAL_CONFLICT     RAG can improve factual accuracy.
      The new source reports RAG introducing errors with irrelevant context...
      evidence: 3xk2m9...  (cached)

proposals:
  51njqpsoa7uc  claim_conflict     [activated] ambiguous

nodes     : register_evidence -> identify_affected_concepts -> ... -> finalize_workflow
cost      : 1 llm call(s), 0 cache hit(s), 13.67ms
```

`resume` continues a paused run after you have decided its proposals. Resuming
with nothing decided pauses again — a resume is not consent. If the configured
provider differs from the one that assessed the run, resume refuses until you
pass `--allow-provider-change`, because mixing judgements from two models with
no way to tell them apart is exactly the ambiguity provenance exists to
prevent.

---

## Choosing a provider

Forge is provider-agnostic; selection is configuration, and no paid API is ever
required.

```bash
# Self-hosted and free — the default deployment path.
export FORGE_LLM_PROVIDER=ollama
export FORGE_MODEL_DEFAULT=qwen3:8b

# Model on another machine (Forge on a laptop, GPU on a desktop).
export FORGE_LLM_PROVIDER=ollama
export FORGE_OLLAMA_URL=http://192.168.1.50:11434

# Hosted open-weights, for machines that cannot host a model. The `openai`
# vendor is a wire format — Groq, OpenRouter, Together, vLLM, llama.cpp and
# most other servers speak it.
export FORGE_LLM_PROVIDER=cloud
export FORGE_CLOUD_VENDOR=openai
export FORGE_CLOUD_BASE_URL=https://api.groq.com/openai
export FORGE_CLOUD_MODEL=llama-3.3-70b-versatile
export FORGE_CLOUD_API_KEY_ENV=GROQ_API_KEY
export FORGE_CLOUD_MAX_TOKENS=8192  # open-weights models cap well below 16000

# Hosted proprietary, if a key happens to exist. Never required.
export FORGE_LLM_PROVIDER=cloud
export FORGE_CLOUD_VENDOR=anthropic
export FORGE_CLOUD_MODEL=claude-sonnet-5
export ANTHROPIC_API_KEY=...        # read at call time; never stored

# Deterministic, for CI.
export FORGE_LLM_PROVIDER=mock
```

**Credentials live in the environment, never in configuration.** Forge stores
only the *name* of the variable to read. `forge status` reports whether a
credential is present without printing it.

If the configured provider is unavailable, Forge says so and stops. It never
substitutes a different model for a knowledge-mutation decision.

---

## What the CLI will not do

No command *requires* a paid API, deletes a note, or writes to the vault — with
exactly one exception: `forge proposals approve --apply`, which requires an
approved, deterministically-verified proposal, backs the file up first, records
a revision, and touches only the single line named in the proposal.

`test_cli_never_writes_to_the_vault` byte-compares every Markdown file before
and after running the read commands.

---

## Related

- [Phase 1 implementation architecture](./architecture/phase-1-implementation.md)
- [Phase 2 implementation architecture](./architecture/phase-2-implementation.md)
- [Phase 3 implementation architecture](./architecture/phase-3-implementation.md)
- [Phase 4 implementation architecture](./architecture/phase-4-implementation.md)
- [Provider availability](./research/provider-availability.md)
- [Retrieval baseline](./research/retrieval-baseline.md)
- [Test strategy](./test-strategy.md)
- [ADR-001](./decisions/001-forge-knowledge-os.md)
