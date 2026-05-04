#!/bin/bash
# Ralph Wiggum Bash Loop — fresh context window per iteration
#
# Usage:
#   bash scripts/infra/ralph.sh <max_iterations> [plan_file]
#
# Examples:
#   bash scripts/infra/ralph.sh 15                    # Use default scripts/infra/ralph/ralph-plan.md
#   bash scripts/infra/ralph.sh 20 my-task-plan.md    # Custom plan file

set -euo pipefail

# Ensure claude CLI is on PATH (needed when bash is launched from CMD on Windows)
export PATH="/c/Users/joshd/.local/bin:$HOME/.local/bin:$PATH"
unset CLAUDECODE 2>/dev/null || true

# Resolve claude binary
CLAUDE_BIN=$(command -v claude 2>/dev/null || echo "")
if [[ -z "$CLAUDE_BIN" ]]; then
  echo "ERROR: claude CLI not found on PATH"
  echo "  Checked: /c/Users/joshd/.local/bin and \$HOME/.local/bin"
  echo "  Install with: npm install -g @anthropic-ai/claude-code"
  exit 1
fi
echo "Using claude at: $CLAUDE_BIN"

MAX_ITERATIONS="${1:-10}"
PLAN_FILE="${2:-scripts/infra/ralph/ralph-plan.md}"
ACTIVITY_FILE="scripts/infra/ralph/ralph-activity.md"
PROMPT_TEMPLATE="scripts/infra/ralph-prompt.md"
COMPLETION_KEYWORD="RALPH_COMPLETE"

# Validate
if [[ ! -f "$PROMPT_TEMPLATE" ]]; then
  echo "ERROR: Prompt template not found: $PROMPT_TEMPLATE"
  echo "  Run from project root: cd /c/Users/joshd/canompx3"
  exit 1
fi

if [[ ! -f "$PLAN_FILE" ]]; then
  echo "ERROR: Plan file not found: $PLAN_FILE"
  echo "  Create one with: bash scripts/infra/ralph-new-plan.sh 'Your task description'"
  echo "  Or copy the template: cp scripts/infra/ralph-plan-template.md scripts/infra/ralph/ralph-plan.md"
  exit 1
fi

# Create activity file if missing
if [[ ! -f "$ACTIVITY_FILE" ]]; then
  echo "# Ralph Activity Log" > "$ACTIVITY_FILE"
  echo "" >> "$ACTIVITY_FILE"
  echo "Auto-created $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ACTIVITY_FILE"
  echo "" >> "$ACTIVITY_FILE"
fi

# Build the prompt by injecting plan file name into template
PROMPT=$(cat "$PROMPT_TEMPLATE")
PROMPT="${PROMPT//\{PLAN_FILE\}/$PLAN_FILE}"
PROMPT="${PROMPT//\{ACTIVITY_FILE\}/$ACTIVITY_FILE}"
PROMPT="${PROMPT//\{COMPLETION_KEYWORD\}/$COMPLETION_KEYWORD}"

echo "========================================"
echo "  Ralph Wiggum Bash Loop"
echo "========================================"
echo "  Plan:           $PLAN_FILE"
echo "  Activity log:   $ACTIVITY_FILE"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Completion:     $COMPLETION_KEYWORD"
echo "========================================"
echo ""

for (( i=1; i<=MAX_ITERATIONS; i++ )); do
  echo ""
  echo "========================================"
  echo "  RALPH ITERATION $i / $MAX_ITERATIONS"
  echo "  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "========================================"
  echo ""

  # Run claude with the prompt, capture output
  OUTPUT=$("$CLAUDE_BIN" --print --dangerously-skip-permissions "$PROMPT" 2>&1) || true

  echo "$OUTPUT"

  # Check for completion keyword in output
  if echo "$OUTPUT" | grep -q "$COMPLETION_KEYWORD"; then
    echo ""
    echo "========================================"
    echo "  RALPH COMPLETE after $i iterations"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    exit 0
  fi

  echo ""
  echo "--- Iteration $i done, continuing... ---"
done

echo ""
echo "========================================"
echo "  RALPH: Max iterations ($MAX_ITERATIONS) reached"
echo "  Check $PLAN_FILE for remaining tasks"
echo "  Check $ACTIVITY_FILE for what was done"
echo "========================================"
exit 1
