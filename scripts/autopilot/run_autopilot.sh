#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# canompx3_autopilot_v1 — headless self-driving task runner
# ═══════════════════════════════════════════════════════════════
# Completes ONE task unattended: run -> review -> one repair pass ->
# branch commit -> report. Never touches capital/live/schema paths
# (Tier-B actions are blocked + journalled by the PreToolUse guard).
#
# Usage:
#   bash scripts/autopilot/run_autopilot.sh "<task description>"
#   bash scripts/autopilot/run_autopilot.sh --worktree "<task>"   # fresh worktree
#
# Patterns reused from scripts/tools/ralph_headless.sh: claude/python
# resolution, CLAUDECODE unset, JSON output parse, retry-once, --max-turns.
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Parse args ────────────────────────────────────────────────
USE_WORKTREE=0
TASK=""
for arg in "$@"; do
    case "$arg" in
        --worktree) USE_WORKTREE=1 ;;
        *) TASK="$arg" ;;
    esac
done
if [[ -z "$TASK" ]]; then
    echo "ERROR: no task given. Usage: run_autopilot.sh [--worktree] \"<task>\"" >&2
    exit 2
fi

# ── Resolve claude CLI (same probe order as ralph_headless.sh) ──
CLAUDE=""
for p in "claude" \
         "/c/Users/joshd/.local/bin/claude.exe" \
         "/mnt/c/Users/joshd/.local/bin/claude.exe" \
         "$HOME/.local/bin/claude"; do
    if command -v "$p" &>/dev/null || [[ -x "$p" ]]; then CLAUDE="$p"; break; fi
done
[[ -z "$CLAUDE" ]] && { echo "ERROR: claude CLI not found" >&2; exit 2; }

PY=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "")
[[ -z "$PY" ]] && { echo "ERROR: python not found" >&2; exit 2; }

# ── Optional fresh worktree (isolation layer) ─────────────────
if [[ "$USE_WORKTREE" -eq 1 ]]; then
    DESC="autopilot-$(date +%H%M%S)"
    if [[ -x "scripts/tools/new_session.sh" ]]; then
        echo "Creating isolated worktree '$DESC'..."
        bash scripts/tools/new_session.sh "$DESC" || {
            echo "ERROR: new_session.sh failed; aborting (fail closed)." >&2; exit 2; }
        echo "NOTE: worktree created. Re-run this script INSIDE the new worktree." >&2
        exit 0
    else
        echo "ERROR: --worktree requested but new_session.sh missing." >&2; exit 2
    fi
fi

# ── Refuse to run on main (fail closed) ───────────────────────
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"
if [[ "$BRANCH" == "main" || "$BRANCH" == "master" || -z "$BRANCH" ]]; then
    echo "ERROR: autopilot refuses to run on '$BRANCH'. Switch to a feature branch." >&2
    exit 2
fi

# ── Run identity + journal ────────────────────────────────────
unset CLAUDECODE 2>/dev/null || true
unset CLAUDE_CODE_ENTRYPOINT 2>/dev/null || true

RUN_ID="autopilot-$(date +%Y%m%d_%H%M%S)"
export AUTOPILOT_RUN=1
export AUTOPILOT_RUN_ID="$RUN_ID"

JOURNAL_DIR="docs/runtime/autopilot"
mkdir -p "$JOURNAL_DIR"
JOURNAL="$JOURNAL_DIR/${RUN_ID}.jsonl"
SEEN_HASHES="$JOURNAL_DIR/${RUN_ID}.seen.json"
RUN_JSON="$JOURNAL_DIR/${RUN_ID}.run.json"
REPAIR_JSON="$JOURNAL_DIR/${RUN_ID}.repair.json"
: > "$JOURNAL"

journal() { echo "{\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"event\":\"$1\",\"msg\":\"$2\"}" >> "$JOURNAL"; }

START_SHA="$(git rev-parse HEAD)"
journal "RUN_START" "branch=$BRANCH task=$(echo "$TASK" | tr '"' "'")"

echo "════════════════════════════════════════════"
echo "  AUTOPILOT v1 — $RUN_ID"
echo "  Branch: $BRANCH"
echo "  Task:   $TASK"
echo "  Journal: $JOURNAL"
echo "════════════════════════════════════════════"

SYSTEM_APPEND="$(cat docs/prompts/autopilot-task-template.md)"
MAX_TURNS=40

# ── Run one claude -p pass; $1=prompt $2=out.json ─────────────
run_claude() {
    local prompt="$1" out="$2"
    "$CLAUDE" -p "$prompt" \
        --dangerously-skip-permissions \
        --allowedTools "Edit,Read,Write,Bash,Grep,Glob" \
        --max-turns "$MAX_TURNS" \
        --no-session-persistence \
        --output-format json \
        --append-system-prompt "$SYSTEM_APPEND" \
        > "$out" 2>"${out%.json}.err" || true
    # retry once on empty output
    if [[ ! -s "$out" ]] || [[ "$(wc -c < "$out")" -lt 20 ]]; then
        sleep 5
        "$CLAUDE" -p "$prompt" \
            --dangerously-skip-permissions \
            --allowedTools "Edit,Read,Write,Bash,Grep,Glob" \
            --max-turns "$MAX_TURNS" \
            --no-session-persistence \
            --output-format json \
            --append-system-prompt "$SYSTEM_APPEND" \
            > "$out" 2>"${out%.json}.err" || true
    fi
}

extract_result() {
    $PY -c "
import json,sys
try:
    d=json.load(open(sys.argv[1],encoding='utf-8'))
    print(d.get('result','') or '')
except Exception:
    pass
" "$1" 2>/dev/null || true
}

# ── 1) Build pass ─────────────────────────────────────────────
echo "→ Build pass..."
journal "BUILD_START" ""
run_claude "$TASK" "$RUN_JSON"
BUILD_RESULT="$(extract_result "$RUN_JSON")"
journal "BUILD_DONE" ""

# ── 2) Review pass ────────────────────────────────────────────
echo "→ Review (review_diff.py)..."
REVIEW_OUT="$($PY scripts/autopilot/review_diff.py --diff-base HEAD --seen-hashes "$SEEN_HASHES" 2>/dev/null || true)"
COMMIT_SAFE="$(echo "$REVIEW_OUT" | $PY -c "import json,sys;print(json.load(sys.stdin).get('commit_safe',False))" 2>/dev/null || echo "False")"
NEW_HUNKS="$(echo "$REVIEW_OUT" | $PY -c "import json,sys;print(json.load(sys.stdin).get('new_hunks',0))" 2>/dev/null || echo "0")"
HIGH_RISK="$(echo "$REVIEW_OUT" | $PY -c "import json,sys;print(','.join(json.load(sys.stdin).get('high_risk',[])))" 2>/dev/null || echo "")"
journal "REVIEW_DONE" "commit_safe=$COMMIT_SAFE new_hunks=$NEW_HUNKS high_risk=$HIGH_RISK"

# ── 3) One repair pass (only if there ARE new findings) ───────
FINDINGS="$(echo "$REVIEW_OUT" | $PY -c "import json,sys;print(chr(10).join(json.load(sys.stdin).get('findings',[])))" 2>/dev/null || true)"
if [[ -n "$FINDINGS" && "$NEW_HUNKS" != "0" ]]; then
    echo "→ Repair pass (one) for review findings..."
    journal "REPAIR_START" ""
    REPAIR_PROMPT="A post-build review of your changes produced these findings. Fix ONLY these findings — do not add new features or expand scope. Then re-run targeted tests.

FINDINGS:
$FINDINGS

After fixing, output the === AUTOPILOT REPORT === block."
    run_claude "$REPAIR_PROMPT" "$REPAIR_JSON"
    journal "REPAIR_DONE" ""
    # Re-review (dedupe spans passes via the same seen-hashes file)
    REVIEW_OUT="$($PY scripts/autopilot/review_diff.py --diff-base HEAD --seen-hashes "$SEEN_HASHES" 2>/dev/null || true)"
    COMMIT_SAFE="$(echo "$REVIEW_OUT" | $PY -c "import json,sys;print(json.load(sys.stdin).get('commit_safe',False))" 2>/dev/null || echo "False")"
    HIGH_RISK="$(echo "$REVIEW_OUT" | $PY -c "import json,sys;print(','.join(json.load(sys.stdin).get('high_risk',[])))" 2>/dev/null || echo "")"
fi

# ── 4) Gates: drift + commit (branch only, never push) ────────
COMMIT_SHA=""
if [[ -z "$(git status --porcelain)" ]]; then
    echo "→ No file changes produced; nothing to commit."
    journal "NO_CHANGES" ""
elif [[ "$COMMIT_SAFE" != "True" ]]; then
    echo "⚠ NOT commit-safe (high-risk files: ${HIGH_RISK:-none}). Leaving changes uncommitted for operator review."
    journal "COMMIT_SKIPPED" "high_risk=$HIGH_RISK"
else
    echo "→ Running drift gate..."
    if $PY pipeline/check_drift.py > "$JOURNAL_DIR/${RUN_ID}.drift.log" 2>&1; then
        DRIFT_OK=1
    else
        DRIFT_OK=0
    fi
    if [[ "$DRIFT_OK" -eq 1 ]]; then
        git add -A
        git commit -m "autopilot($RUN_ID): $TASK" >/dev/null 2>&1 || true
        COMMIT_SHA="$(git rev-parse --short HEAD)"
        if [[ "$COMMIT_SHA" == "$(git rev-parse --short "$START_SHA")" ]]; then
            COMMIT_SHA=""  # commit hook may have blocked; no new SHA
            echo "⚠ Commit did not advance HEAD (hook block?). Changes left staged."
            journal "COMMIT_NOOP" ""
        else
            echo "→ Committed: $COMMIT_SHA"
            journal "COMMITTED" "$COMMIT_SHA"
        fi
    else
        echo "⚠ Drift gate FAILED — not committing. See ${RUN_ID}.drift.log"
        journal "DRIFT_FAILED" ""
    fi
fi

# ── 5) Final report ───────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "  AUTOPILOT REPORT — $RUN_ID"
echo "════════════════════════════════════════════"
echo "  Branch:      $BRANCH"
echo "  Start SHA:   $(git rev-parse --short "$START_SHA")"
echo "  Commit SHA:  ${COMMIT_SHA:-<none — uncommitted>}"
echo "  Commit-safe: $COMMIT_SAFE"
echo "  High-risk:   ${HIGH_RISK:-none}"
echo "  Pushed:      NO (autopilot never pushes)"
echo "  Armed live:  NO"
echo ""
echo "  Tier-B blockers logged this run:"
grep '"event":"BLOCKED_TIER_B"' "$JOURNAL" 2>/dev/null | sed 's/^/    /' || echo "    none"
echo ""
echo "  --- model report block ---"
echo "$BUILD_RESULT" | sed -n '/=== AUTOPILOT REPORT ===/,/=== END AUTOPILOT REPORT ===/p' || true
echo "════════════════════════════════════════════"
echo "  Journal: $JOURNAL"

journal "RUN_END" "commit=${COMMIT_SHA:-none}"
