#!/usr/bin/env bash
# Ralph Loop — headless batch runner
# Runs N iterations of the Ralph Loop autonomously via claude -p
# Each iteration gets a fresh context window (no babysitting needed)
#
# Usage:
#   bash scripts/tools/ralph_headless.sh          # 5 iterations (default)
#   bash scripts/tools/ralph_headless.sh 10        # 10 iterations
#   bash scripts/tools/ralph_headless.sh 3 "live_config.py"  # 3 iterations, scoped
set -e
cd /c/Users/joshd/canompx3

MAX_ITERS="${1:-5}"
SCOPE="${2:-}"
LOG_DIR="docs/ralph-loop/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M)
MASTER_LOG="$LOG_DIR/headless-${TIMESTAMP}.log"

echo "════════════════════════════════════════════" | tee "$MASTER_LOG"
echo "  RALPH HEADLESS — Seven Sins Auditor" | tee -a "$MASTER_LOG"
echo "════════════════════════════════════════════" | tee -a "$MASTER_LOG"
echo "  Started:    $(date)" | tee -a "$MASTER_LOG"
echo "  Iterations: $MAX_ITERS" | tee -a "$MASTER_LOG"
echo "  Scope:      ${SCOPE:-auto (Next Targets from audit file)}" | tee -a "$MASTER_LOG"
echo "  Log:        $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "════════════════════════════════════════════" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

SCOPE_LINE="use Next Targets from audit file"
if [ -n "$SCOPE" ]; then
    SCOPE_LINE="$SCOPE"
fi

ACCEPT_COUNT=0
REJECT_COUNT=0
ERROR_COUNT=0
FIXES=""

for i in $(seq 1 "$MAX_ITERS"); do
    ITER_LOG="$LOG_DIR/headless-${TIMESTAMP}-iter${i}.log"
    STARTED=$(date +%s)
    echo "────────────────────────────────────────" | tee -a "$MASTER_LOG"
    echo "  [$i/$MAX_ITERS] Starting — $(date '+%H:%M:%S')" | tee -a "$MASTER_LOG"

    PROMPT="Run one Ralph Loop iteration.
Scope: ${SCOPE_LINE}
Today's date: $(date +%Y-%m-%d)

IMPORTANT: You are running headless (no user interaction). After completing the iteration:
1. If ACCEPT: commit with a descriptive message, then output the report
2. If REJECT or escalation needed: output the report with REJECT status, do NOT commit
3. Output ONLY the structured === RALPH LOOP ITER ... === report block at the end"

    # Run claude in pipe mode with restricted tools
    if claude -p "$PROMPT" \
        --allowedTools "Edit,Read,Write,Bash,Grep,Glob,Agent" \
        > "$ITER_LOG" 2>&1; then

        ENDED=$(date +%s)
        ELAPSED=$(( ENDED - STARTED ))
        MINS=$(( ELAPSED / 60 ))
        SECS=$(( ELAPSED % 60 ))

        # Extract target and verdict from report block
        TARGET=$(grep -oP 'Scope: \K.*' "$ITER_LOG" | tail -1)
        VERDICT=""

        # Parse verdict from structured report block (Verdict: ACCEPT|REJECT|SKIPPED)
        VERDICT=$(grep -oP 'Verdict: \K\w+' "$ITER_LOG" | tail -1)
        if [ -z "$VERDICT" ]; then
            # Fallback: scan for keywords anywhere
            if grep -q "ACCEPT" "$ITER_LOG"; then VERDICT="ACCEPT"
            elif grep -q "REJECT" "$ITER_LOG"; then VERDICT="REJECT"
            else VERDICT="UNKNOWN"; fi
        fi

        case "$VERDICT" in
            ACCEPT|SKIPPED) ACCEPT_COUNT=$((ACCEPT_COUNT + 1)) ;;
            REJECT)         REJECT_COUNT=$((REJECT_COUNT + 1)) ;;
            *)              ERROR_COUNT=$((ERROR_COUNT + 1)) ;;
        esac

        # Extract commit hash if present
        COMMIT=$(grep -oP 'Commit: \K[a-f0-9]+' "$ITER_LOG" | tail -1)
        ACTION=$(grep -oP 'Action: \K.*' "$ITER_LOG" | tail -1)

        echo "  [$i/$MAX_ITERS] ${VERDICT} — ${MINS}m${SECS}s — ${TARGET:-unknown target}" | tee -a "$MASTER_LOG"
        if [ -n "$ACTION" ]; then
            echo "           ${ACTION}" | tee -a "$MASTER_LOG"
        fi
        if [ -n "$COMMIT" ]; then
            echo "           commit: ${COMMIT}" | tee -a "$MASTER_LOG"
            FIXES="${FIXES}  ${COMMIT} ${TARGET}\n"
        fi
    else
        ENDED=$(date +%s)
        ELAPSED=$(( ENDED - STARTED ))
        MINS=$(( ELAPSED / 60 ))
        SECS=$(( ELAPSED % 60 ))
        echo "  [$i/$MAX_ITERS] ERROR (exit $?) — ${MINS}m${SECS}s" | tee -a "$MASTER_LOG"
        ERROR_COUNT=$((ERROR_COUNT + 1))
    fi

    # Extract the report block for the master log
    if grep -q "=== RALPH LOOP" "$ITER_LOG"; then
        echo "" >> "$MASTER_LOG"
        sed -n '/=== RALPH LOOP/,/===/p' "$ITER_LOG" >> "$MASTER_LOG"
    fi

    # Running score
    TOTAL=$((ACCEPT_COUNT + REJECT_COUNT + ERROR_COUNT))
    echo "           Score: ${ACCEPT_COUNT}✓ ${REJECT_COUNT}✗ ${ERROR_COUNT}? (${TOTAL}/${MAX_ITERS})" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"
done

echo "════════════════════════════════════════════" | tee -a "$MASTER_LOG"
echo "  RALPH BATCH COMPLETE" | tee -a "$MASTER_LOG"
echo "════════════════════════════════════════════" | tee -a "$MASTER_LOG"
echo "  Finished:  $(date)" | tee -a "$MASTER_LOG"
echo "  Accept: $ACCEPT_COUNT | Reject: $REJECT_COUNT | Error: $ERROR_COUNT" | tee -a "$MASTER_LOG"
if [ -n "$FIXES" ]; then
    echo "" | tee -a "$MASTER_LOG"
    echo "  Commits:" | tee -a "$MASTER_LOG"
    echo -e "$FIXES" | tee -a "$MASTER_LOG"
fi
echo "  Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "════════════════════════════════════════════" | tee -a "$MASTER_LOG"
