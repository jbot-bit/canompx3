# Workflow Preferences

## Data First — MANDATORY
Query data before reading code when the user is asking what happened, how bad, or why numbers differ. `data-first-guard.py` enforces this after repeated reads.

## Implementation Gating — MANDATORY
Do NOT write code until the user explicitly says to implement.
- Design: "plan", "design", "brainstorm", "iterate", "4t"
- Implement: "build it", "do it", "implement", "go", "ship it"

## Git Operations — Just Execute
"commit", "push", and "merge" mean execute, not discuss. Warn only for real risk like secrets or destructive scope.

## Trivial-Change Tier — Skip Ceremony
Trivial means all of:
- no `pipeline/` or `trading_app/` production logic touched
- no schema or canonical-source change
- under 100 net diff lines
- verification lands in the same change

Trivial work skips branch/design/stage ceremony. Non-trivial work keeps the full stage-gate.

## Response Style — Concise, No Extras
- No CLI docs, no docstrings on unchanged code, no unsolicited background tasks
- No summaries >3 lines, no unnecessary type annotations or "improvements"
- Direct questions → direct answers. Tasks → do it, report result.

## No Performative Self-Correction — MANDATORY
Do the right thing without narrating obvious internal process. Guardrail rules apply to code changes, not casual discussion.

## Trading Queries — Exact Format
- Return EXACTLY the count requested. "Top 2" = 2 rows. Not 3.
- ALWAYS include: instrument, session name, session time (Brisbane TZ), orb_minutes, entry_model, confirm_bars, filter_type, rr_target, direction, sample_size, win_rate, ExpR, Sharpe, fitness status
- Sort by ExpR, NEVER Sharpe alone
- Use `pipeline.paths.GOLD_DB_PATH` — never hardcode

## Session Start
Ambiguous first message → ask ONE question: "Design or implement?" Then follow strictly.

## Subagents And Teams
Prefer subagents only when a side task would flood the main conversation with logs, search results, or file dumps. Keep prompts narrow and returned summaries short. Do not use agent teams by default; they multiply context and cost.
