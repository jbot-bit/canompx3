task: Add /check institutional-audit command (config-only, 3 files) + unblock pre-commit by fixing a pre-existing B007 lint error in trading_app/live/instance_lock.py:49 (committed in f1413178). The B007 rename is a no-op: line-49 loop does not use the control var; line-119 loop DOES use `attempt` and is left untouched.
mode: IMPLEMENTATION

## Scope Lock
- .claude/commands/check.md
- .claude/hooks/intent-router.py
- .claude/rules/auto-skill-routing.md
- trading_app/live/instance_lock.py

## Blast Radius
- .claude/commands/check.md — new slash command, zero callers; harness reads frontmatter for the skill list.
- .claude/hooks/intent-router.py — one INTENT_RULES tuple added; routing-parity drift check binds it to the rule doc.
- .claude/rules/auto-skill-routing.md — one Intent Map bullet (backticked target, required by the parity extractor).
- trading_app/live/instance_lock.py — line-49 ONLY: rename loop var `attempt`→`_attempt` (B007). Pure no-op: that loop never reads the var. Line-119 loop in acquire_instance_lock() DOES use `attempt` (lines 148/150/153) and is deliberately NOT renamed (would be F821 into a capital path). No behavior change; unblocks the whole-tree ruff pre-commit gate.
- Reads: none (read-only on gold.db not involved). Writes: none.
