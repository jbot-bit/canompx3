#!/usr/bin/env bash
# Ralph Loop Runner — Continuous audit → fix → verify cycle
#
# Usage:
#   bash scripts/ralph_loop_runner.sh              # Run forever
#   bash scripts/ralph_loop_runner.sh --once       # Single iteration
#   bash scripts/ralph_loop_runner.sh --audit-only # Audit only, no fixes
#
# Stop gracefully: touch ralph_loop.stop
#
# Requires: claude CLI in PATH

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

STOP_FILE="$REPO_ROOT/ralph_loop.stop"
HISTORY_FILE="$REPO_ROOT/docs/ralph-loop/ralph-loop-history.md"
AUDIT_FILE="$REPO_ROOT/docs/ralph-loop/ralph-loop-audit.md"
PLAN_FILE="$REPO_ROOT/docs/ralph-loop/ralph-loop-plan.md"
LOG_DIR="$REPO_ROOT/docs/ralph-loop/logs"

# Parse args
RUN_MODE="loop"
if [[ "${1:-}" == "--once" ]]; then
    RUN_MODE="once"
elif [[ "${1:-}" == "--audit-only" ]]; then
    RUN_MODE="audit-only"
fi

# Ensure directories exist
mkdir -p "$LOG_DIR"

# Get current iteration number from history
get_iteration() {
    local last
    last=$(grep -c "^## Iteration:" "$HISTORY_FILE" 2>/dev/null || echo "0")
    echo $((last + 1))
}

# Phase 1: Audit
run_audit() {
    local iter=$1
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_file="$LOG_DIR/iteration-${iter}-audit.log"

    echo "[$timestamp] Ralph Loop — Iteration $iter — AUDIT PHASE"

    claude --print \
        "You are the Ralph Loop Auditor (read .claude/agents/ralph-auditor.md for your full instructions).

This is iteration $iter of the Ralph Loop.

Read the previous audit state from docs/ralph-loop/ralph-loop-audit.md
Read the history from docs/ralph-loop/ralph-loop-history.md

Then run a full audit:
1. Run infrastructure gates: drift check, behavioral audit, test suite, lint
2. Scan trading_app/live/ for the Seven Sins (silent failures, fail-open, phantom state, etc.)
3. Check canonical integrity (hardcoded lists, magic numbers, dependency direction)
4. Check test coverage for recently changed files

Write your structured findings to docs/ralph-loop/ralph-loop-audit.md using the format from your agent prompt.

IMPORTANT: Do NOT write any production code. Only update the audit file." \
        2>&1 | tee "$log_file"

    echo "[$timestamp] Audit phase complete. Log: $log_file"
}

# Phase 2: Plan + Implement
run_implement() {
    local iter=$1
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_file="$LOG_DIR/iteration-${iter}-implement.log"

    echo "[$timestamp] Ralph Loop — Iteration $iter — IMPLEMENT PHASE"

    claude --print \
        "You are the Ralph Loop Architect (read .claude/agents/ralph-architect.md for your full instructions).

This is iteration $iter of the Ralph Loop.

1. Read the current audit from docs/ralph-loop/ralph-loop-audit.md
2. Select the highest-priority finding that is safe to fix autonomously
3. Write the plan to docs/ralph-loop/ralph-loop-plan.md
4. Then become the Implementer (read .claude/agents/ralph-implementer.md):
   - Follow the 2-pass method: Discovery first, then Implementation
   - Apply the minimal fix
   - Run tests and drift check
5. Report what was done

SAFETY: If the top finding requires schema changes, entry model changes,
or touches 5+ files — SKIP implementation and flag for human review.
Update the plan file with your decision.

Do NOT commit. The Verifier handles that gate." \
        2>&1 | tee "$log_file"

    echo "[$timestamp] Implement phase complete. Log: $log_file"
}

# Phase 3: Verify
run_verify() {
    local iter=$1
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_file="$LOG_DIR/iteration-${iter}-verify.log"

    echo "[$timestamp] Ralph Loop — Iteration $iter — VERIFY PHASE"

    claude --print \
        "You are the Ralph Loop Verifier (read .claude/agents/ralph-verifier.md for your full instructions).

This is iteration $iter of the Ralph Loop.

1. Read the plan from docs/ralph-loop/ralph-loop-plan.md
2. Read the audit from docs/ralph-loop/ralph-loop-audit.md
3. Run ALL 6 verification gates:
   - Gate 1: python pipeline/check_drift.py
   - Gate 2: python scripts/tools/audit_behavioral.py
   - Gate 3: python -m pytest tests/ -x -q
   - Gate 4: ruff check pipeline/ trading_app/ scripts/
   - Gate 5: Blast radius verification (read callers of changed functions)
   - Gate 6: Regression scan (verify the specific fix)
4. Write your verdict to the plan file
5. If ACCEPT: append the full iteration record to docs/ralph-loop/ralph-loop-history.md
6. If REJECT: document why in the plan file and flag for next iteration

Use the structured output format from your agent prompt." \
        2>&1 | tee "$log_file"

    echo "[$timestamp] Verify phase complete. Log: $log_file"
}

# Main loop
echo "=== Ralph Loop Starting ==="
echo "Mode: $RUN_MODE"
echo "Stop file: $STOP_FILE"
echo ""

while true; do
    # Check for stop signal
    if [[ -f "$STOP_FILE" ]]; then
        echo "Stop file detected — shutting down Ralph Loop gracefully."
        rm -f "$STOP_FILE"
        exit 0
    fi

    ITER=$(get_iteration)
    echo "=========================================="
    echo "  Ralph Loop — Iteration $ITER"
    echo "=========================================="

    # Phase 1: Audit
    run_audit "$ITER"

    # Check stop between phases
    if [[ -f "$STOP_FILE" ]]; then
        echo "Stop file detected after audit — shutting down."
        rm -f "$STOP_FILE"
        exit 0
    fi

    # Phase 2: Implement (unless audit-only mode)
    if [[ "$RUN_MODE" != "audit-only" ]]; then
        run_implement "$ITER"

        # Check stop between phases
        if [[ -f "$STOP_FILE" ]]; then
            echo "Stop file detected after implement — shutting down."
            rm -f "$STOP_FILE"
            exit 0
        fi

        # Phase 3: Verify
        run_verify "$ITER"
    fi

    echo ""
    echo "Iteration $ITER complete."
    echo ""

    # Exit if single-run mode
    if [[ "$RUN_MODE" == "once" || "$RUN_MODE" == "audit-only" ]]; then
        echo "Single-run mode — exiting."
        exit 0
    fi

    # Brief pause between iterations
    echo "Sleeping 10s before next iteration... (touch ralph_loop.stop to exit)"
    sleep 10
done
