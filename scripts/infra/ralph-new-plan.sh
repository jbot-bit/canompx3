#!/bin/bash
# Generate a new ralph-plan.md from a task description using Claude
#
# Usage:
#   ./scripts/infra/ralph-new-plan.sh "Rebuild US_POST_EQUITY pipeline for all instruments"
#   ./scripts/infra/ralph-new-plan.sh "Fix DST split bug and revalidate strategies"
#
# This calls Claude ONCE to generate a structured plan, then you review it
# before starting the Ralph loop.

set -euo pipefail

DESCRIPTION="${1:-}"

if [[ -z "$DESCRIPTION" ]]; then
  echo "Usage: ./scripts/infra/ralph-new-plan.sh 'Your task description'"
  echo ""
  echo "Examples:"
  echo "  ./scripts/infra/ralph-new-plan.sh 'Rebuild daily_features and outcomes for US_POST_EQUITY'"
  echo "  ./scripts/infra/ralph-new-plan.sh 'Add walk-forward validation for MES strategies'"
  exit 1
fi

PLAN_FILE="scripts/infra/ralph/ralph-plan.md"
ACTIVITY_FILE="scripts/infra/ralph/ralph-activity.md"

if [[ -f "$PLAN_FILE" ]]; then
  echo "WARNING: $PLAN_FILE already exists."
  echo "  Rename or delete it first, or use a different file name."
  echo "  Existing plan will NOT be overwritten."
  exit 1
fi

echo "Generating plan for: $DESCRIPTION"
echo ""

PLAN_PROMPT="Read CLAUDE.md and TRADING_RULES.md for project context.

Break this task into 3-8 sequential steps with clear acceptance criteria:

$DESCRIPTION

Output ONLY a markdown file in this EXACT format (no other text):

# Ralph Plan

## Context
[1-2 sentences about what this plan achieves]

## Tasks

\`\`\`json
[
  {
    \"id\": 1,
    \"category\": \"category_name\",
    \"description\": \"Clear task description\",
    \"passes\": false,
    \"steps\": [\"Step 1: ...\", \"Step 2: ...\"],
    \"acceptance\": \"What must be true for this to pass\"
  }
]
\`\`\`

## Notes
[Any relevant file paths, constraints, or gotchas]

Rules:
- Each task should be completable in one iteration (~5-10 min of agent work)
- Include verification steps (run tests, check output, query DB)
- Order tasks so each builds on the previous
- Category should be one of: setup, schema, implementation, rebuild, verification, cleanup"

# Generate the plan (unset CLAUDECODE to allow nested invocation)
unset CLAUDECODE 2>/dev/null || true
claude --print "$PLAN_PROMPT" > "$PLAN_FILE" 2>&1

# Create fresh activity file
echo "# Ralph Activity Log" > "$ACTIVITY_FILE"
echo "" >> "$ACTIVITY_FILE"
echo "Plan generated $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ACTIVITY_FILE"
echo "Task: $DESCRIPTION" >> "$ACTIVITY_FILE"
echo "" >> "$ACTIVITY_FILE"

echo ""
echo "Created:"
echo "  $PLAN_FILE    <- Review and edit this before running Ralph"
echo "  $ACTIVITY_FILE <- Activity log (auto-populated during loop)"
echo ""
echo "Next steps:"
echo "  1. Review $PLAN_FILE — edit tasks if needed"
echo "  2. Run: ./scripts/infra/ralph.sh 15"
