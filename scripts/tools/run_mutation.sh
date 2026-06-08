#!/usr/bin/env bash
# run_mutation.sh — portable, scoped mutation testing for the capital-critical core.
#
# WHY cosmic-ray, not mutmut: mutmut 3.x hard-refuses native Windows on EVERY
# subcommand ("To run mutmut on Windows, please use the WSL" — boxed/mutmut#397),
# which would pin mutation testing to a WSL/.venv-wsl/Python-3.12 island that no
# clean checkout or GitHub-Actions runner could reproduce. cosmic-ray runs
# natively on Windows + Linux + CI on this repo's Python 3.11.9 and mutates
# in-place (no `mutants/` source-copy tree → no collision with check_drift.py's
# rglob("*.py") scans).
#
# PORTABILITY: every path here is repo-relative; the only machine assumption is a
# working `python -m pytest` (resolved via PATH / the active venv). No absolute
# paths, no WSL, no /mnt/c. A fresh clone runs this unchanged.
#
# Usage (invoke via `uv run bash` so the venv's cosmic-ray/python are on PATH):
#   uv run bash scripts/tools/run_mutation.sh <module_src> "<test_command>" [timeout_s] [workers]
# Example (parallel, default 8 workers):
#   uv run bash scripts/tools/run_mutation.sh pipeline/cost_model.py \
#     "python -m pytest tests/test_pipeline/test_cost_model.py -q" 30
# Example (serial debug — 1 worker):
#   uv run bash scripts/tools/run_mutation.sh pipeline/cost_model.py "<cmd>" 30 1
#
# PARALLELISM (added 2026-06-07): cosmic-ray's `local` distributor runs ONE mutant
# at a time. The `http` distributor runs N independent workers, each with its own
# git clone of the repo, and "mutation testing time scales down linearly with the
# number of workers" (official distributed tutorial). Default WORKERS=8 → ~8x.
# Workers default to 8 (≈25% of a 32-core box) to leave thermal/instability
# headroom — raise only on a machine that tolerates it. WORKERS=1 falls back to
# the `local` serial distributor (no clones, simplest path for debugging).
#
# (cosmic-ray cannot read a process-substitution `<(...)` config on Windows —
# this script writes a REAL temp .toml under .mutation/, which works everywhere.)
#
# Output: a survival-rate report + the session sqlite under .mutation/ (gitignored).
# The survivor report is ALSO tee'd to .mutation_run.log (repo root, NOT trap-
# cleaned) so results survive the EXIT cleanup of .mutation/.
set -euo pipefail

MODULE_SRC="${1:?usage: run_mutation.sh <module_src> <test_command> [timeout_s] [workers]}"
TEST_CMD="${2:?need a test command, e.g. 'python -m pytest tests/...'}"
TIMEOUT="${3:-30}"
WORKERS="${4:-8}"

if [[ ! -f "$MODULE_SRC" ]]; then
  echo "ERROR: module source not found: $MODULE_SRC" >&2
  exit 1
fi

if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [[ "$WORKERS" -lt 1 ]]; then
  echo "ERROR: workers must be a positive integer, got: $WORKERS" >&2
  exit 1
fi

# Ephemeral session dir — gitignored, removed on exit so it never reaches a peer's
# working tree or a commit (multi-terminal-shared-file-hygiene).
SESS_DIR=".mutation"
mkdir -p "$SESS_DIR"
slug="$(echo "$MODULE_SRC" | tr '/.' '__')"
CFG="$SESS_DIR/${slug}.toml"
SESSION="$SESS_DIR/${slug}.sqlite"

WORKERS_PID=""
cleanup() {
  # Tear down http workers first (they block forever serving requests), then
  # remove the ephemeral session dir.
  if [[ -n "$WORKERS_PID" ]]; then
    kill "$WORKERS_PID" 2>/dev/null || true
    # cr-http-workers spawns child `cosmic-ray http-worker` procs; reap them too.
    pkill -f "cosmic-ray.* http-worker" 2>/dev/null || true
  fi
  rm -f "$CFG" "$SESSION" 2>/dev/null || true
}
trap cleanup EXIT

# 1. Write the config deterministically (avoids cosmic-ray's interactive new-config).
#    WORKERS=1 → serial `local` distributor. WORKERS>1 → parallel `http` distributor
#    with one worker-url per worker (consecutive ports from 9876).
if [[ "$WORKERS" -le 1 ]]; then
  DIST_BLOCK=$'[cosmic-ray.distributor]\nname = "local"'
else
  base_port=9876
  urls=""
  for ((i = 0; i < WORKERS; i++)); do
    [[ -n "$urls" ]] && urls+=", "
    urls+="\"http://localhost:$((base_port + i))\""
  done
  DIST_BLOCK=$'[cosmic-ray.distributor]\nname = "http"\n\n[cosmic-ray.distributor.http]\nworker-urls = ['"$urls"']'
fi

cat > "$CFG" <<EOF
[cosmic-ray]
module-path = "$MODULE_SRC"
timeout = ${TIMEOUT}.0
excluded-modules = []
test-command = "$TEST_CMD"

$DIST_BLOCK
EOF

echo "=== baseline (unmutated tests must pass) ==="
cosmic-ray --verbosity INFO baseline "$CFG"

echo "=== init session (plan mutations) ==="
cosmic-ray init "$CFG" "$SESSION" --force

if [[ "$WORKERS" -le 1 ]]; then
  echo "=== exec (serial local distributor) ==="
  cosmic-ray exec "$CFG" "$SESSION"
else
  # http distributor: cr-http-workers (per cosmic_ray/tools/http_workers.py) ONLY
  # starts the workers — each is a `cosmic-ray http-worker` serving requests from a
  # shallow (depth=1) git clone of REPO_URL, and it BLOCKS forever awaiting them.
  # So we background it, wait for every port to LISTEN (condition-based, not a
  # sleep guess — per .claude/rules/condition-based-waiting.md), then run the exec
  # client locally; exec distributes mutants over HTTP and writes results to OUR
  # $SESSION. cleanup() tears the workers down on EXIT.
  #
  # WARNING: workers clone COMMITTED HEAD only. Any uncommitted test-suite speedup
  # is INVISIBLE to them — commit test changes BEFORE a parallel run or every
  # worker runs the slow committed suite.
  REPO_URL="$(git rev-parse --show-toplevel)"
  echo "=== starting $WORKERS http workers (shallow clone of committed $REPO_URL) ==="
  cr-http-workers "$CFG" "$REPO_URL" &
  WORKERS_PID=$!

  echo "=== waiting for $WORKERS worker ports to listen ==="
  deadline=$(( $(date +%s) + 120 ))
  for ((i = 0; i < WORKERS; i++)); do
    port=$((9876 + i))
    until netstat -ano 2>/dev/null | grep -q "[:.]$port .*LISTENING"; do
      if [[ $(date +%s) -gt $deadline ]]; then
        echo "ERROR: worker on port $port did not start within 120s" >&2
        exit 1
      fi
      sleep 1
    done
  done
  echo "=== all $WORKERS workers listening — running distributed exec ==="
  cosmic-ray exec "$CFG" "$SESSION"
fi

echo "=== survival rate ==="
cr-rate --estimate --confidence 95.0 "$SESSION"

# --- Durable mutation evidence (MUST outlive the EXIT trap) -------------------
# The trap deletes .mutation/ (the session sqlite) on exit, so any triage run
# AFTER this script has no structured record of which mutant survived on which
# source line. A prior 1236-mutant strategy_fitness run lost its survivor detail
# exactly this way — recoverable only by deterministic re-init + log-join.
#
# Primary artifact: `cosmic-ray dump` — the OFFICIAL machine-readable export.
# Per the cosmic-ray Commands reference it "writes a detailed JSON representation
# of a session to stdout". Verified-against-real-output 2026-06-07: the format is
# JSON-LINES (NDJSON) — one work item per line, NOT a single array. Each line is
#   [ {job_id, mutations:[{module_path, operator_name, occurrence,
#                          start_pos:[line,col], end_pos:[line,col]}]},
#     {worker_outcome, test_outcome, output, diff} ]
# so a triage tool extracts survivors with a one-line filter:
#   [json.loads(l) for l in open(dump) if json.loads(l)[1].get("test_outcome")=="survived"]
# start_pos[0] is the 1-based source line. No text-scraping, no awk, no
# re-init+log-join (the painful recovery this run needed before the fix existed).
# We persist the FULL dump (all outcomes), so a triage tool filters
# test_outcome=="survived" itself rather than trusting a pre-filtered text view.
#
# Companion: cr-report --show-diff (human-readable) for quick eyeballing, and a
# copy of the sqlite so cr-rate / cr-report / cr-html can be re-queried without a
# full re-run. All three land at gitignored repo-root paths BEFORE the trap.
DURABLE_LOG="$(pwd)/.mutation_run.log"
DUMP_JSON="$(pwd)/.mutation_dump_${slug}.json"
SURV_REPORT="$(pwd)/.mutation_survivors_${slug}.txt"
DURABLE_SESSION="$(pwd)/.mutation_session_${slug}.sqlite"
HEAD_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)"
NOW_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

echo "=== survivor dump (official JSON, line-anchored) -> $DUMP_JSON ==="
# cosmic-ray dump is the canonical machine-readable export (see comment above).
# Tolerate failure: if dump errors, the human-readable report below + the
# preserved sqlite still give triage a path. Fail-open, never abort the run tail.
if cosmic-ray dump "$SESSION" > "$DUMP_JSON" 2>/dev/null && [[ -s "$DUMP_JSON" ]]; then
  echo "wrote $(wc -c < "$DUMP_JSON") bytes of dump JSON"
else
  echo "WARN: cosmic-ray dump produced no output (triage can re-init deterministically)" >&2
fi

echo "=== survivor report (human-readable diffs) -> $SURV_REPORT ==="
{
  echo "# mutation survivors — $MODULE_SRC"
  echo "# commit: $HEAD_SHA"
  echo "# generated: $NOW_UTC"
  echo "# rate: $(cr-rate "$SESSION" 2>/dev/null)"
  echo "# machine-readable source of truth: $(basename "$DUMP_JSON")"
  echo
} > "$SURV_REPORT"
cr-report --show-diff "$SESSION" 2>/dev/null >> "$SURV_REPORT" || true

# Preserve the sqlite so cr-rate / cr-report / cr-html re-query without a re-run.
# Copied (not moved) — the original stays under .mutation/ for the trap to clean,
# keeping .mutation/ the single ephemeral-session location.
cp -f "$SESSION" "$DURABLE_SESSION" 2>/dev/null \
  && echo "preserved sqlite -> $DURABLE_SESSION" \
  || echo "WARN: could not preserve sqlite (triage can re-init deterministically)" >&2

# Keep the legacy human-readable tee for backward-compat / quick eyeballing.
echo "=== survivors (human-readable, appended to $DURABLE_LOG) ==="
cr-report --no-show-output "$SESSION" | tee -a "$DURABLE_LOG"
