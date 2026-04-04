# Auto-Skill Routing — Proactive Invocation

The user does NOT type `/skill` commands. Claude invokes skills proactively based on context.

## Auto-Triggers

Match by INTENT, not exact words. ADHD-friendly — user phrases things casually.

| Intent | Example phrases | Action |
|--------|----------------|--------|
| Session start | (first message) | `/orient` |
| Keep going | "next", "keep going", "continue", "what now", "more" | `/next` |
| Git ops | "commit", "push", "pusdh", "comit" | Just execute |
| Stage done | "done", "complete", "finished", "that's it" | `/verify done` |
| What do I trade | "book", "portfolio", "what's live", "tonight", "strats", "playbook", "what am I trading", "show me my stuff" | `/trade-book` |
| Health/fitness | "how's it going", "performing", "decay", "regime", "fitness", "healthy", "anything dying" | `/regime-check` |
| Project state | "status", "where are we", "what's broken", "catch me up", "orient" | `/orient` |
| Something wrong | "off", "wrong", "broken", "doesn't add up", "weird", "failing", "bug", "numbers look wrong", "this doesn't make sense" | `/quant-debug` |
| Plan/design | "plan", "design", "brainstorm", "think about", "how should we", "4t", "approach" | `/design` |
| Past findings | "didn't we test", "wasn't that dead", "what did we find", "remind me", "history of", "NO-GO?" | `/pinecone-assistant` |
| Test a hypothesis | "is this real", "does X work", "test this", "research", "investigate edge", "deep dive" | `/research` |
| Code review | "review", "check my work", "bloomey", "seven sins", "before I commit", "anything wrong" | `/code-review` |
| Schema/config edit | (editing init_db, config.py, cost_model) | Full stage-gate |
| Improve skill | "improve skill", "skill loop", "optimize skill" | `/skill-improve` |

Proactively invoke when context matches. Destructive skills require explicit `/name`.
Post-work auto-checks (drift + tests) are handled by hooks — no need to duplicate.
