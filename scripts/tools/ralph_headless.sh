#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Ralph Headless v3 — Autonomous batch audit + fix runner
# ═══════════════════════════════════════════════════════════════
# Runs N iterations of the Ralph Loop via `claude -p` (non-interactive).
# Each iteration is a fresh context window with full project context.
#
# Usage:
#   bash scripts/tools/ralph_headless.sh              # 5 iterations (default)
#   bash scripts/tools/ralph_headless.sh 10           # 10 iterations
#   bash scripts/tools/ralph_headless.sh 3 "live_config.py"  # 3 iters, scoped
#
# Stop gracefully: touch ralph_loop.stop
#
# v3 changes (2026-03-17):
#   - JSON output for structured parsing + cost tracking
#   - --dangerously-skip-permissions (fixes ~40% failure rate from blocked prompts)
#   - --max-turns 50 (prevents runaway but leaves room for full audit cycles)
#   - --no-session-persistence (keeps session list clean)
#   - Retry once on empty output before counting as error
#   - Separate stderr from stdout (prevents JSON corruption)
#   - Git safety checks between iterations
#   - Agent tool removed from allowed tools (prevents background noise)
#   - --append-system-prompt for headless-specific instructions
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

# ── Resolve project root ──────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Resolve claude CLI ────────────────────────────────────────
CLAUDE=""
for p in "claude" \
         "/c/Users/joshd/.local/bin/claude.exe" \
         "/mnt/c/Users/joshd/.local/bin/claude.exe" \
         "$HOME/.local/bin/claude"; do
    if command -v "$p" &>/dev/null || [[ -x "$p" ]]; then
        CLAUDE="$p"
        break
    fi
done
if [[ -z "$CLAUDE" ]]; then
    echo "ERROR: claude CLI not found" >&2
    exit 2
fi

# ── Resolve python ────────────────────────────────────────────
PY=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "")
if [[ -z "$PY" ]]; then
    echo "ERROR: python not found" >&2
    exit 2
fi

# ── Config ────────────────────────────────────────────────────
MAX_ITERS="${1:-5}"
SCOPE="${2:-}"
LOG_DIR="docs/ralph-loop/logs"
STOP_FILE="$PROJECT_ROOT/ralph_loop.stop"
TIMESTAMP=$(date +%Y%m%d_%H%M)
MASTER_LOG="$LOG_DIR/headless-${TIMESTAMP}.log"
MAX_TURNS=50
MAX_RETRIES=1

mkdir -p "$LOG_DIR"

# ── Clear nesting detection (allows spawning from within Claude Code) ──
unset CLAUDECODE 2>/dev/null || true
unset CLAUDE_CODE_ENTRYPOINT 2>/dev/null || true

# ── Scope instruction ─────────────────────────────────────────
SCOPE_LINE="Read the Next Targets from docs/ralph-loop/ralph-loop-audit.md and pick the highest-centrality unscanned file."
if [[ -n "$SCOPE" ]]; then
    SCOPE_LINE="Scope: $SCOPE"
fi

# ── Counters ──────────────────────────────────────────────────
ACCEPT=0
REJECT=0
SKIP=0
ERROR=0
TOTAL_COST="0.0000"
FIXES=""

# ── Helpers ───────────────────────────────────────────────────
banner() { echo "$1" | tee -a "$MASTER_LOG"; }

# Parse JSON output from claude -p --output-format json
# Writes result text to .txt file, prints "cost_usd subtype" to stdout
parse_iter_json() {
    local json_file="$1"
    local txt_file="${json_file%.json}.txt"

    $PY -c "
import json, sys
try:
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        d = json.load(f)
    result = d.get('result', '') or ''
    cost = d.get('total_cost_usd', 0) or 0
    subtype = d.get('subtype', '') or ''
    with open(sys.argv[2], 'w', encoding='utf-8') as f:
        f.write(result)
    print(f'{cost:.4f} {subtype}')
except Exception as e:
    try:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            raw = f.read()
        with open(sys.argv[2], 'w', encoding='utf-8') as f:
            f.write(raw)
    except:
        pass
    print('0.0000 parse_error')
" "$json_file" "$txt_file" 2>/dev/null || echo "0.0000 error"
}

# Extract verdict from result text
extract_verdict() {
    local txt_file="$1"
    local verdict=""

    if [[ ! -s "$txt_file" ]]; then
        echo "EMPTY"
        return
    fi

    # Primary: structured report field
    verdict=$(grep -oP 'Verdict: \K\w+' "$txt_file" 2>/dev/null | tail -1) || true

    # Fallback: keyword scan
    if [[ -z "$verdict" ]]; then
        if grep -q "DIMINISHING_RETURNS" "$txt_file" 2>/dev/null; then verdict="DIMINISHING_RETURNS"
        elif grep -q "NEEDS_REVIEW" "$txt_file" 2>/dev/null; then verdict="NEEDS_REVIEW"
        elif grep -q "ACCEPT" "$txt_file" 2>/dev/null; then verdict="ACCEPT"
        elif grep -q "REJECT" "$txt_file" 2>/dev/null; then verdict="REJECT"
        elif grep -q "SKIPPED" "$txt_file" 2>/dev/null; then verdict="SKIPPED"
        else verdict="UNKNOWN"
        fi
    fi

    echo "$verdict"
}

# ── Banner ────────────────────────────────────────────────────
banner "════════════════════════════════════════════"
banner "  RALPH HEADLESS v3 — Autonomous Auditor"
banner "════════════════════════════════════════════"
banner "  Started:    $(date)"
banner "  Iterations: $MAX_ITERS"
banner "  Scope:      ${SCOPE:-auto (Next Targets from audit file)}"
banner "  Model:      haiku (no MCP) | Max turns: $MAX_TURNS"
banner "  Log:        $MASTER_LOG"
banner "════════════════════════════════════════════"
banner ""

# ── Headless system prompt appendix ───────────────────────────
# This goes into the system prompt (high authority position) to enforce
# structured output and prevent tool sprawl.
SYSTEM_APPEND="HEADLESS MODE ACTIVE — You are running non-interactively in a batch.
Rules for headless operation:
1. You MUST output the === RALPH LOOP ITER [N] COMPLETE === report block as your FINAL output.
   Without it the batch runner cannot track your progress and the iteration counts as ERROR.
2. Do NOT spawn Agent subagents or background tasks — do everything inline.
3. Do NOT ask questions or wait for user input — make autonomous decisions per your agent prompt.
4. Combine bash calls aggressively (use && chains) to conserve your $MAX_TURNS turn budget.
5. If you hit your turn limit before finishing, output the report block with what you have."

# ── Run one claude invocation ─────────────────────────────────
run_one() {
    local iter=$1
    local attempt=$2
    local iter_json="$LOG_DIR/headless-${TIMESTAMP}-iter${iter}.json"
    local iter_err="$LOG_DIR/headless-${TIMESTAMP}-iter${iter}.err"

    local PROMPT="Run one Ralph Loop iteration.
Read your full instructions from .claude/agents/ralph-loop.md FIRST.
Iteration number: $iter (batch run, attempt $attempt)
Date: $(date +%Y-%m-%d)
$SCOPE_LINE

Execute ALL steps from ralph-loop.md (Steps 0 through 5).
Your FINAL output MUST be the structured === RALPH LOOP ITER [N] COMPLETE === report block."

    $CLAUDE -p "$PROMPT" \
        --model haiku \
        --dangerously-skip-permissions \
        --allowedTools "Edit,Read,Write,Bash,Grep,Glob" \
        --max-turns "$MAX_TURNS" \
        --no-session-persistence \
        --output-format json \
        --append-system-prompt "$SYSTEM_APPEND" \
        --mcp-config "$PROJECT_ROOT/docs/ralph-loop/ralph-mcp.json" \
        --strict-mcp-config \
        > "$iter_json" 2>"$iter_err" || true
}

# ── Run one iteration with retry and parsing ──────────────────
run_iteration() {
    local iter=$1
    local started=$(date +%s)
    local head_sha
    head_sha=$(git rev-parse HEAD 2>/dev/null) || true

    local iter_json="$LOG_DIR/headless-${TIMESTAMP}-iter${iter}.json"
    local iter_txt="$LOG_DIR/headless-${TIMESTAMP}-iter${iter}.txt"
    local iter_err="$LOG_DIR/headless-${TIMESTAMP}-iter${iter}.err"

    # Attempt 1
    run_one "$iter" 1

    # Retry on empty output
    if [[ ! -s "$iter_json" ]] || [[ "$(wc -c < "$iter_json" 2>/dev/null)" -lt 20 ]]; then
        if [[ $MAX_RETRIES -gt 0 ]]; then
            banner "  [$iter/$MAX_ITERS] Empty output — retrying (attempt 2)..."
            sleep 5
            run_one "$iter" 2
        fi
    fi

    local ended=$(date +%s)
    local elapsed=$(( ended - started ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    # Parse JSON → text file + cost + subtype
    local parse_result="0.0000 unknown"
    if [[ -s "$iter_json" ]]; then
        parse_result=$(parse_iter_json "$iter_json")
    fi

    local iter_cost="${parse_result%% *}"
    local subtype="${parse_result##* }"

    # Accumulate cost
    TOTAL_COST=$($PY -c "print(f'{float(\"$TOTAL_COST\") + float(\"$iter_cost\"):.4f}')" 2>/dev/null) || true

    # Extract verdict from result text
    local verdict
    verdict=$(extract_verdict "$iter_txt")

    # Handle max_turns: result may be empty but work may have been committed
    if [[ "$verdict" == "EMPTY" && "$subtype" == "error_max_turns" ]]; then
        local new_sha
        new_sha=$(git rev-parse HEAD 2>/dev/null) || true
        if [[ "$new_sha" != "$head_sha" ]]; then
            verdict="ACCEPT"
            banner "  [$iter/$MAX_ITERS] Hit max turns but committed work — treating as ACCEPT"
        else
            verdict="MAX_TURNS"
        fi
    fi

    # Extract metadata from result text
    local target="" action="" commit=""
    if [[ -s "$iter_txt" ]]; then
        target=$(grep -oP 'Scope: \K.*' "$iter_txt" 2>/dev/null | tail -1) || true
        action=$(grep -oP 'Action: \K.*' "$iter_txt" 2>/dev/null | tail -1) || true
        commit=$(grep -oP 'Commit: \K[a-f0-9]+' "$iter_txt" 2>/dev/null | tail -1) || true
    fi

    # If no commit from report, check git for new commits
    if [[ -z "$commit" ]]; then
        local new_sha
        new_sha=$(git rev-parse HEAD 2>/dev/null) || true
        if [[ "$new_sha" != "$head_sha" ]]; then
            commit=$(git rev-parse --short HEAD 2>/dev/null) || true
        fi
    fi

    # Classify verdict
    case "$verdict" in
        ACCEPT)                          ACCEPT=$((ACCEPT + 1)) ;;
        SKIPPED|DIMINISHING_RETURNS)     SKIP=$((SKIP + 1)) ;;
        REJECT|NEEDS_REVIEW)             REJECT=$((REJECT + 1)) ;;
        *)                               ERROR=$((ERROR + 1)); verdict="ERROR($subtype)" ;;
    esac

    # Log to master
    banner "  [$iter/$MAX_ITERS] ${verdict} — ${mins}m${secs}s — ${target:-unknown target} [\$${iter_cost}]"
    if [[ -n "$action" ]]; then
        banner "           ${action}"
    fi
    if [[ -n "$commit" ]]; then
        banner "           commit: ${commit}"
        FIXES="${FIXES}  ${commit} ${target}\n"
    fi

    # Append report block to master log
    if [[ -s "$iter_txt" ]] && grep -q "=== RALPH LOOP" "$iter_txt" 2>/dev/null; then
        echo "" >> "$MASTER_LOG"
        sed -n '/=== RALPH LOOP/,/================================/p' "$iter_txt" >> "$MASTER_LOG"
    fi

    # Note stderr if present
    if [[ -s "$iter_err" ]]; then
        banner "           stderr: $(head -1 "$iter_err")"
    fi

    # Running score
    local total=$((ACCEPT + REJECT + SKIP + ERROR))
    banner "           Score: ${ACCEPT}✓ ${REJECT}✗ ${SKIP}⊘ ${ERROR}? (${total}/${MAX_ITERS})"
    banner ""
}

# ── Main loop ─────────────────────────────────────────────────
for i in $(seq 1 "$MAX_ITERS"); do
    # Check stop file
    if [[ -f "$STOP_FILE" ]]; then
        banner "Stop file detected — shutting down."
        rm -f "$STOP_FILE"
        break
    fi

    run_iteration "$i"

    # Brief pause between iterations (prevents API rate limiting)
    if [[ $i -lt $MAX_ITERS ]]; then
        sleep 3
    fi
done

# ── Summary ───────────────────────────────────────────────────
banner "════════════════════════════════════════════"
banner "  RALPH BATCH COMPLETE"
banner "════════════════════════════════════════════"
banner "  Finished:  $(date)"
banner "  Accept: $ACCEPT | Skip: $SKIP | Reject: $REJECT | Error: $ERROR"
banner "  Cost:    \$${TOTAL_COST}"
if [[ -n "$FIXES" ]]; then
    banner ""
    banner "  Commits:"
    echo -e "$FIXES" | tee -a "$MASTER_LOG"
fi
banner ""
banner "  Master log: $MASTER_LOG"
banner "════════════════════════════════════════════"
