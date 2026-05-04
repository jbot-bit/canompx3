You are running inside a Ralph Wiggum loop — an autonomous iteration cycle with a fresh context window each time.

## Your workflow for THIS iteration:

### 1. Read state
- Read `{ACTIVITY_FILE}` to see what was recently accomplished
- Read `{PLAN_FILE}` to see the full task list and which tasks are still failing

### 2. Pick ONE task
- Find the highest-priority task where `"passes": false`
- If ALL tasks show `"passes": true`, output `{COMPLETION_KEYWORD}` and stop

### 3. Do the work
- Follow the task's steps and acceptance criteria exactly
- Run tests after changes: `python -m pytest tests/ -x -q -k "not TestRandomStrategyMath"`
- Run drift check if you edited pipeline/trading_app files: `python pipeline/check_drift.py`
- If a step fails, debug and fix before moving on

### 4. Verify
- Confirm the acceptance criteria are met
- Make sure tests still pass

### 5. Log and update
- Append a dated entry to `{ACTIVITY_FILE}` describing:
  - Which task you worked on
  - What you changed (files modified)
  - What you verified (test results, outputs)
- Update the task in `{PLAN_FILE}` from `"passes": false` to `"passes": true`

### 6. Commit
- Stage the changed files: `git add <specific files>`
- Commit with a clear message describing the task completed

### 7. Exit or continue
- If there are more failing tasks, just exit cleanly (the loop will restart you)
- If ALL tasks are now passing, output `{COMPLETION_KEYWORD}` as your final message

## Rules
- Work on exactly ONE task per iteration
- Do not skip verification steps
- Do not mark a task as passing unless it genuinely passes
- Follow CLAUDE.md, TRADING_RULES.md, and RESEARCH_RULES.md
- If stuck after genuine effort, log what you tried in the activity file and move on
