# Auto-Skill Routing — Proactive Invocation

The user does NOT type `/skill` commands. Claude invokes skills proactively based on context.

## Auto-Triggers

| Context | Action |
|---------|--------|
| Session start | `/orient` |
| "next" / "keep going" / "continue" | `/next` |
| "commit" / "push" | Just execute (no blocking) |
| "done" / "complete" (with active stage) | `/verify done` |
| Strategy/portfolio/fitness question | `/trade-book` or `/regime-check` |
| Project state question | `/orient` |
| Bug description | `/quant-debug` |
| Plan/design request | `/design` |
| Past research question | `/pinecone-assistant` |
| Schema/init_db/config.py edit | Full stage-gate |
| "improve skill" | `/skill-improve` |

Proactively invoke when context matches. Destructive skills require explicit `/name`.
Post-work auto-checks (drift + tests) are handled by hooks — no need to duplicate.
