# Auto-Skill Routing — Proactive Invocation

The user does NOT type `/skill` commands. Claude MUST invoke skills automatically based on context.

## Mandatory Auto-Triggers (no user action needed)

| Context | Skill to invoke | How to detect |
|---------|----------------|---------------|
| Session start / first message | `/orient` | No prior messages in conversation |
| User says "commit" / "push" / "done" | `/verify quick` FIRST, then commit | Trigger words in message |
| After editing 2+ production files | `/verify quick` before responding | PostToolUse tracking |
| Before editing pipeline/ or trading_app/ core files | `/blast-radius` | File in NEVER_TRIVIAL list |
| User asks about strategy/portfolio/fitness | `/trade-book` or `/regime-check` | "trade", "portfolio", "fitness", "what do I trade" |
| User asks about project state/status | `/orient` | "status", "where are we", "what's broken" |
| User describes a bug | `/quant-debug` | "bug", "broken", "wrong", "error in" |
| User wants to plan/design | `/design` | "plan", "design", "how should we", "think through" |
| User asks about past research | `/pinecone-assistant` routing | "what did we find", "why did we", "history of" |
| Research claim needs validation | `/audit hypothesis` | "is this real", "validate", "stress test" |

## Post-Work Auto-Checks

After completing any non-trivial code change, Claude MUST:
1. Run drift check: `python pipeline/check_drift.py`
2. Run targeted tests for changed files (check TEST_MAP)
3. Report results — do NOT claim "done" without evidence

This replaces the need for the user to remember `/verify`.

## Rules

- NEVER wait for the user to type a slash command if the context clearly matches
- Skills with `disable-model-invocation: true` (audit, rebuild-outcomes, post-rebuild, validate-instrument, ralph) still require explicit `/name` — they are destructive operations
- When in doubt, invoke the skill — false positives (unnecessary invocation) are cheaper than false negatives (missing a check)
