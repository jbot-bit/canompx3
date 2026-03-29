# Auto-Skill Routing — Proactive Invocation

The user does NOT type `/skill` commands. Claude should invoke skills proactively based on context.

## Auto-Triggers

| Context | Action | Detection |
|---------|--------|-----------|
| Session start / first message | Run `/orient` | No prior messages |
| "commit" / "push" / "done" | Run `/verify quick` FIRST, then commit | Trigger words |
| After editing 2+ production files | Run drift + targeted tests before responding | File count tracking |
| Before editing NEVER_TRIVIAL files | Run `/blast-radius` on the target | File in core list |
| Strategy/portfolio/fitness question | Run `/trade-book` or `/regime-check` | "trade", "portfolio", "fitness" |
| Project state question | Run `/orient` | "status", "where are we", "what's broken" |
| Bug description | Run `/quant-debug` | "bug", "broken", "wrong", "error" |
| Plan/design request | Run `/design` | "plan", "design", "how should we" |
| Past research question | Route via `/pinecone-assistant` | "what did we find", "why did we" |
| Hypothesis validation | Run `/audit hypothesis` | "is this real", "validate", "stress test" |
| Schema/init_db edit | Require full stage-gate (complex change) | File is init_db.py or cost_model.py |
| config.py entry model edit | Require full stage-gate (breaking change) | File is trading_app/config.py |
| User says "done"/"complete" for a stage | Run `/verify done` before closing | Trigger words + active STAGE_STATE |

## Post-Work Auto-Checks

After any non-trivial code change:
1. `python pipeline/check_drift.py`
2. Targeted tests for changed files (check TEST_MAP in `.claude/hooks/post-edit-pipeline.py`)
3. Report results — do NOT claim "done" without evidence

## Rules

- Proactively invoke when context clearly matches — don't wait for slash commands
- Destructive skills (audit full, rebuild-outcomes, post-rebuild, validate-instrument, ralph) require explicit `/name`
- When in doubt, invoke — false positives are cheaper than missed checks
