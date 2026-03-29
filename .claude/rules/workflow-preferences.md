# Workflow Preferences

## Data First — MANDATORY
When investigating ANY question about data, behavior, counts, mismatches, or pipeline correctness:
1. **QUERY THE DATA FIRST.** Write and run the SQL/Python query immediately.
2. Look at the actual numbers.
3. THEN read code only if the numbers don't explain themselves.

Do NOT theorize about what the code "probably does." Do NOT read 10 files to build a mental model before checking the data. 10 minutes of querying beats hours of code reading. The data shows WHAT is happening and HOW BAD it is. Code reading only tells you WHY — and you don't need WHY until you know WHAT.

Trigger words: "check", "investigate", "why is X", "what's happening", "real data", "actual numbers", "empirical" → QUERY FIRST, no exceptions.

## Implementation Gating — MANDATORY
Do NOT start writing code, editing files, or running implementation commands until the user explicitly says to implement.
Words that mean DESIGN MODE (no code): "plan", "design", "think about", "what if", "how would", "explore", "iterate", "4t", "brainstorm".
Words that mean IMPLEMENT NOW: "build it", "do it", "implement", "go", "ship it", "make it happen".

If the user says "plan this feature" and you start writing code, you have failed. Stay in design mode. Iterate on the plan. Present options. Wait for the green light. This has been the #1 source of friction — Claude jumping to implementation during design sessions. Do not do it.

If the user pushes back with "I said design" or "stop implementing" — you went too far. Apologize briefly and return to design mode immediately.

## Git Operations — Just Execute
When the user says "commit", "commit all", "push", "merge", or any variant (including typos like "pusdh", "comit", "vcommit"):
1. Do NOT explain that there's nothing to commit
2. Do NOT ask "are you sure?" or "which files?"
3. Do NOT describe what you're about to do
4. Just check git status, stage relevant files, and execute

The user has been forced to repeat commit/push commands multiple times across sessions because Claude hesitated, explained, or asked unnecessary questions. Stop doing that. The user knows what they want. Execute.

Exception: warn (but still execute) if staging files that look like secrets (.env, credentials, tokens).

## Response Style — Concise, No Extras
- Do NOT add CLI usage documentation unless asked
- Do NOT add docstrings or comments to code you didn't change
- Do NOT start background tasks the user didn't request
- Do NOT add "here's what I did" summaries longer than 3 lines
- Do NOT add unnecessary type annotations, error handling, or "improvements" beyond scope

When the user asks a direct question, give a direct answer. Not a paragraph. Not a tutorial. The answer.

When the user asks you to do something, do it and report the result. Don't explain your reasoning unless it's non-obvious or risky.

The user has explicitly said they find verbose AI responses frustrating. Respect that.

## Trading Queries — Exact Format
When querying trading strategies or trade data:
- Return EXACTLY the number requested. "Top 2 per instrument" = 2 rows per instrument. Not 3. Not 5. Not "here are some extras that might interest you."
- ALWAYS include ALL of these fields: instrument, session name, session time (Brisbane TZ), orb_minutes (5/15/30), entry_model, confirm_bars, filter_type, rr_target, direction, sample_size, win_rate, ExpR, Sharpe, fitness status (FIT/WATCH/DECAY)
- Sort by ExpR or edge ratio, NEVER by Sharpe alone (Sharpe is biased under multiple testing — see RESEARCH_RULES.md)
- Use the canonical DB path from `pipeline.paths.GOLD_DB_PATH` — never hardcode paths
- Include data freshness (when was the strategy last validated/promoted)
- The user has corrected missing rr_target, missing session times, wrong sort order, and incomplete field sets across 10+ sessions. Get it right the first time.

## Stage-Gate System — Workflow Entry Point

For non-trivial work, `/stage-gate` is the canonical entry point. It classifies, then routes:

| User intent | Route |
|-------------|-------|
| "orient" / session start | `/orient` → checks STAGE_STATE → routes |
| "plan" / "design" / "4t" | `/stage-gate` → DESIGN → `/design` |
| "build" / "go" / "implement" | `/stage-gate` → IMPLEMENTATION → preflight → execute |
| "verify" / "done?" | `/verify done` (reads acceptance from STAGE_STATE if available) |
| "review" | `/code-review` (reads scope from STAGE_STATE if available) |
| "where was I" / "resume" | `/resume-rebase` → drift check → continue or reclassify |
| Quick fix / typo | `/stage-gate trivial [file]` → minimal STAGE_STATE → edit |

**Verification skill routing (which one when):**
- Pre-commit quick check → `/verify` (or `/verify quick`)
- "Is this stage done?" → `/verify done` (reads STAGE_STATE acceptance criteria)
- Post-major-milestone deep audit → `/verify full` (impact map + gates)
- Institutional code review → `/bloomey-review` or `/code-review`

The stage-gate-guard hook hard-blocks production edits without an active STAGE_STATE.md.
See `.claude/rules/stage-gate-protocol.md` for the always-loaded awareness rules.

## Session Start — Intent Framing
If the user's first message is ambiguous about whether they want design or implementation, ask ONE question: "Design or implement?" Then follow their answer strictly.

Do not assume. Do not guess. Ask once, then execute.
