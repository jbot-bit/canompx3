# Workflow Preferences

## Data First — MANDATORY
Query data BEFORE reading code. `data-first-guard.py` hook enforces this with blocking after 7 consecutive reads. Trigger words: "check", "investigate", "why is X", "what's happening" → QUERY FIRST.

## Implementation Gating — MANDATORY
Do NOT write code until the user explicitly says to implement.
- DESIGN MODE: "plan", "design", "think about", "brainstorm", "iterate", "4t"
- IMPLEMENT NOW: "build it", "do it", "implement", "go", "ship it"

If the user says "plan" and you write code, you have failed. #1 source of friction.

## Git Operations — Just Execute
"commit", "push", "merge" (including typos) → check status, stage, execute. No explaining, no "are you sure?", no describing what you're about to do. Exception: warn if staging secrets.

## Response Style — Concise, No Extras
- No CLI docs, no docstrings on unchanged code, no unsolicited background tasks
- No summaries >3 lines, no unnecessary type annotations or "improvements"
- Direct questions → direct answers. Tasks → do it, report result.

## No Performative Self-Correction — MANDATORY
NEVER produce self-correcting statements like:
- "You're right — I'm jumping into code without proper grounding"
- "Let me stop and orient in the project context first"
- "I should read the existing code before writing"
- "Let me step back and think about this"

These are WASTED TOKENS. The 2-pass method, design proposal gate, and data-first rules apply to CODE CHANGES ONLY — not to planning, discussion, answering questions, or running scripts. During conversation, just talk. If you need to read a file, read it. Don't announce your self-correction. Nobody cares about your internal process — just do the right thing silently.

## Trading Queries — Exact Format
- Return EXACTLY the count requested. "Top 2" = 2 rows. Not 3.
- ALWAYS include: instrument, session name, session time (Brisbane TZ), orb_minutes, entry_model, confirm_bars, filter_type, rr_target, direction, sample_size, win_rate, ExpR, Sharpe, fitness status
- Sort by ExpR, NEVER Sharpe alone
- Use `pipeline.paths.GOLD_DB_PATH` — never hardcode

## Session Start
Ambiguous first message → ask ONE question: "Design or implement?" Then follow strictly.
