#!/bin/bash
# Ralph Code Review Loop — reviews each commit with fresh context
#
# Usage:
#   bash scripts/infra/ralph-review.sh [base_commit] [max_iterations]
#
# Examples:
#   bash scripts/infra/ralph-review.sh 47b6730 20    # Review all commits since 47b6730
#   bash scripts/infra/ralph-review.sh HEAD~5 10     # Review last 5 commits

set -euo pipefail

export PATH="/c/Users/joshd/.local/bin:$HOME/.local/bin:$PATH"
unset CLAUDECODE 2>/dev/null || true

CLAUDE_BIN=$(command -v claude 2>/dev/null || echo "")
if [[ -z "$CLAUDE_BIN" ]]; then
  echo "ERROR: claude CLI not found on PATH"
  exit 1
fi

BASE_COMMIT="${1:-47b6730}"
MAX_ITERATIONS="${2:-30}"
REVIEW_OUTPUT="scripts/infra/ralph/ralph-review-results.md"

# Get commits to review (oldest first)
mapfile -t COMMITS < <(git log --reverse --oneline "$BASE_COMMIT..HEAD")

TOTAL=${#COMMITS[@]}
if [[ $TOTAL -eq 0 ]]; then
  echo "No commits found between $BASE_COMMIT and HEAD"
  exit 0
fi

echo "========================================"
echo "  Ralph Code Review Loop"
echo "========================================"
echo "  Commits to review: $TOTAL"
echo "  Base: $BASE_COMMIT"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Output: $REVIEW_OUTPUT"
echo "========================================"

# Initialize output file
cat > "$REVIEW_OUTPUT" <<EOF
# Ralph Code Review Results

Base: $BASE_COMMIT
Generated: $(date '+%Y-%m-%d %H:%M:%S')
Commits reviewed: 0 / $TOTAL

---

EOF

ITERATION=0
for COMMIT_LINE in "${COMMITS[@]}"; do
  ITERATION=$((ITERATION + 1))
  if [[ $ITERATION -gt $MAX_ITERATIONS ]]; then
    echo "Max iterations ($MAX_ITERATIONS) reached"
    break
  fi

  SHA=$(echo "$COMMIT_LINE" | cut -d' ' -f1)
  MSG=$(echo "$COMMIT_LINE" | cut -d' ' -f2-)

  echo ""
  echo "========================================"
  echo "  REVIEW $ITERATION / $TOTAL: $SHA"
  echo "  $MSG"
  echo "  $(date '+%H:%M:%S')"
  echo "========================================"

  # Get the diff for this commit
  DIFF=$(git diff "$SHA~1..$SHA" --stat && echo "---FULL DIFF---" && git diff "$SHA~1..$SHA")

  REVIEW_PROMPT="You are a code reviewer for a futures trading data pipeline project.

Review this git commit and provide a focused, actionable review.

## Commit
$COMMIT_LINE

## Diff
\`\`\`
$DIFF
\`\`\`

## Review Checklist
For each item, mark PASS or FAIL with a one-line reason:

1. **Correctness**: Does the code do what the commit message says?
2. **Safety**: Any risk of data corruption, race conditions, or security issues?
3. **Test coverage**: Are changes covered by tests? Any missing assertions?
4. **CLAUDE.md compliance**: Does it follow project rules (fail-closed, idempotent, one-way deps)?
5. **TRADING_RULES.md compliance**: Any trading logic violations?
6. **Hardcoded values**: Any magic numbers, hardcoded paths, or symbols that should be configurable?
7. **Schema sync**: If schema changed, are all references updated (tests, config, sql_adapter)?

## Output Format
Start with a one-line verdict: LGTM / MINOR ISSUES / NEEDS ATTENTION / CRITICAL
Then the checklist results.
Then 0-3 specific findings (file:line, what's wrong, suggested fix).
Keep it under 40 lines total."

  # Run review with fresh context
  OUTPUT=$("$CLAUDE_BIN" --print "$REVIEW_PROMPT" 2>&1) || true

  # Append to results file
  cat >> "$REVIEW_OUTPUT" <<EOF
## $SHA — $MSG

$OUTPUT

---

EOF

  # Update count in header
  sed -i "s/Commits reviewed: [0-9]* \/ $TOTAL/Commits reviewed: $ITERATION \/ $TOTAL/" "$REVIEW_OUTPUT"

  echo ""
  echo "$OUTPUT" | head -5
  echo "  [Full review saved to $REVIEW_OUTPUT]"
done

echo ""
echo "========================================"
echo "  RALPH REVIEW COMPLETE"
echo "  Reviewed $ITERATION / $TOTAL commits"
echo "  Results: $REVIEW_OUTPUT"
echo "========================================"
