#!/usr/bin/env bash
# Ralph Review — post-batch Opus quality gate
# Reviews [judgment] commits from the last ralph batch
# Skips review if all commits were [mechanical]
#
# Usage:
#   bash scripts/tools/ralph_review.sh                    # review last 5 commits
#   bash scripts/tools/ralph_review.sh 3                  # review last 3 commits
#   bash scripts/tools/ralph_review.sh 10 abc1234         # review 10 commits since abc1234
set -e
cd /c/Users/joshd/canompx3

LOOK_BACK="${1:-5}"
SINCE_COMMIT="${2:-}"
LOG_DIR="docs/ralph-loop/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M)
REVIEW_LOG="$LOG_DIR/review-${TIMESTAMP}.log"

# Ensure claude CLI is available
export PATH="/c/Users/joshd/.local/bin:$HOME/.local/bin:$PATH"
unset CLAUDECODE 2>/dev/null || true

CLAUDE_BIN=$(command -v claude 2>/dev/null || echo "")
if [[ -z "$CLAUDE_BIN" ]]; then
    echo "ERROR: claude CLI not found on PATH"
    exit 1
fi

echo "════════════════════════════════════════════" | tee "$REVIEW_LOG"
echo "  RALPH REVIEW — Opus Quality Gate" | tee -a "$REVIEW_LOG"
echo "════════════════════════════════════════════" | tee -a "$REVIEW_LOG"

# Find ralph commits
if [ -n "$SINCE_COMMIT" ]; then
    RALPH_COMMITS=$(git log --oneline "$SINCE_COMMIT"..HEAD --grep="Ralph Loop" --format="%H" 2>/dev/null)
else
    RALPH_COMMITS=$(git log --oneline -"$LOOK_BACK" --grep="Ralph Loop" --format="%H" 2>/dev/null)
fi

if [ -z "$RALPH_COMMITS" ]; then
    echo "  No Ralph Loop commits found in last $LOOK_BACK commits." | tee -a "$REVIEW_LOG"
    echo "  Nothing to review." | tee -a "$REVIEW_LOG"
    echo "════════════════════════════════════════════" | tee -a "$REVIEW_LOG"
    exit 0
fi

# Separate judgment vs mechanical
JUDGMENT_COMMITS=""
MECHANICAL_COUNT=0
JUDGMENT_COUNT=0

for COMMIT in $RALPH_COMMITS; do
    MSG=$(git log --format="%s" -1 "$COMMIT")
    if echo "$MSG" | grep -q "^\[judgment\]"; then
        JUDGMENT_COMMITS="$JUDGMENT_COMMITS $COMMIT"
        JUDGMENT_COUNT=$((JUDGMENT_COUNT + 1))
        echo "  [judgment] $COMMIT — $MSG" | tee -a "$REVIEW_LOG"
    elif echo "$MSG" | grep -q "^\[mechanical\]"; then
        MECHANICAL_COUNT=$((MECHANICAL_COUNT + 1))
        echo "  [mechanical] $COMMIT — $MSG (skip)" | tee -a "$REVIEW_LOG"
    else
        # Untagged ralph commits treated as judgment (conservative)
        JUDGMENT_COMMITS="$JUDGMENT_COMMITS $COMMIT"
        JUDGMENT_COUNT=$((JUDGMENT_COUNT + 1))
        echo "  [untagged→judgment] $COMMIT — $MSG" | tee -a "$REVIEW_LOG"
    fi
done

echo "" | tee -a "$REVIEW_LOG"
echo "  Mechanical: $MECHANICAL_COUNT (skipped)" | tee -a "$REVIEW_LOG"
echo "  Judgment:   $JUDGMENT_COUNT (reviewing)" | tee -a "$REVIEW_LOG"
echo "" | tee -a "$REVIEW_LOG"

if [ "$JUDGMENT_COUNT" -eq 0 ]; then
    echo "  All commits mechanical — no Opus review needed." | tee -a "$REVIEW_LOG"
    echo "  RESULT: PASS (all mechanical)" | tee -a "$REVIEW_LOG"
    echo "════════════════════════════════════════════" | tee -a "$REVIEW_LOG"
    exit 0
fi

# Build diff payload for Opus
DIFF_PAYLOAD=""
for COMMIT in $JUDGMENT_COMMITS; do
    MSG=$(git log --format="%s" -1 "$COMMIT")
    DIFF=$(git diff "$COMMIT"~1 "$COMMIT" -- "*.py")
    DIFF_PAYLOAD="$DIFF_PAYLOAD
--- COMMIT: $COMMIT ---
Message: $MSG
$DIFF
"
done

# Single Opus review call
echo "  Sending $JUDGMENT_COUNT judgment commit(s) to Opus for review..." | tee -a "$REVIEW_LOG"
echo "" | tee -a "$REVIEW_LOG"

REVIEW_PROMPT="You are reviewing code changes made by an autonomous Sonnet-based auditor (Ralph Loop) on an ORB breakout trading pipeline.

These are ONLY the [judgment] commits — behavior changes, exception handling, logic fixes, new guards. Mechanical commits (dead code, imports, annotations) have been filtered out.

Review each diff for:
1. **Architectural mistakes** — does the fix make sense in context? Is it the right approach?
2. **Unnecessary changes** — is this fixing a real problem or creating busywork?
3. **Scope creep** — did it touch more than needed? Any 'while I was here' cleanup?
4. **Bad judgment calls** — wrong exception type, incorrect guard logic, misunderstanding of the codebase
5. **Subtle regressions** — could this change break something the tests don't cover?

Project context:
- 4-instrument ORB breakout pipeline (MGC, MNQ, MES, M2K)
- One-way dep: pipeline/ → trading_app/
- Entry models: E1+E2 active. E0 purged. E3 soft-retired.
- Fail-closed is the design principle — unknown state = block, not pass.
- Cost model from pipeline.cost_model.COST_SPECS. Sessions from pipeline.dst.SESSION_CATALOG.

For each commit, output:
  COMMIT: <hash>
  VERDICT: GOOD | SUSPECT | REVERT
  REASON: <1-2 sentences>

End with a summary line:
  REVIEW RESULT: PASS (all good) | FLAG (N suspect) | REVERT (N should revert)

Here are the diffs:

$DIFF_PAYLOAD"

REVIEW_OUTPUT=$("$CLAUDE_BIN" --print --model opus "$REVIEW_PROMPT" 2>&1) || true

echo "$REVIEW_OUTPUT" | tee -a "$REVIEW_LOG"
echo "" | tee -a "$REVIEW_LOG"

# Parse result
if echo "$REVIEW_OUTPUT" | grep -q "REVERT"; then
    echo "  ⚠ REVIEW RESULT: REVERT RECOMMENDED — check $REVIEW_LOG" | tee -a "$REVIEW_LOG"
elif echo "$REVIEW_OUTPUT" | grep -q "FLAG"; then
    echo "  ⚠ REVIEW RESULT: SUSPECT COMMITS FLAGGED — check $REVIEW_LOG" | tee -a "$REVIEW_LOG"
else
    echo "  REVIEW RESULT: PASS" | tee -a "$REVIEW_LOG"
fi

echo "  Review log: $REVIEW_LOG" | tee -a "$REVIEW_LOG"
echo "════════════════════════════════════════════" | tee -a "$REVIEW_LOG"
