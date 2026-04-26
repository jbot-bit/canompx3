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

## Trivial-Change Tier — Skip Ceremony
A change is **trivial** when ALL hold: (a) no `pipeline/` or `trading_app/` files touched, (b) no schema/canonical-source change, (c) <100 lines of net diff, (d) tests or verification land in the same diff. Trivial = edit, run tests, commit, push on whatever branch is currently checked out. NO branch cut, NO design doc in `docs/plans/`, NO stage file in `docs/runtime/stages/`. The full ceremony (stage-gate, design doc, branch-from-origin/main, scope_lock) applies to NON-trivial work — production code, schema edits, multi-stage features. Hook helpers, frontmatter additions, gitignore tweaks, single-rule edits, drift-check additions are trivial. Do NOT invent stage files for them.

## Response Style — Concise, No Extras
- No CLI docs, no docstrings on unchanged code, no unsolicited background tasks
- No summaries >3 lines, no unnecessary type annotations or "improvements"
- Direct questions → direct answers. Tasks → do it, report result.

## No Performative Self-Correction — MANDATORY
Never narrate your internal process ("let me stop and orient", "I should read first"). Just do the right thing silently. Guardrail rules (2-pass, design gate) apply to CODE CHANGES, not discussion.

## Trading Queries — Exact Format
- Return EXACTLY the count requested. "Top 2" = 2 rows. Not 3.
- ALWAYS include: instrument, session name, session time (Brisbane TZ), orb_minutes, entry_model, confirm_bars, filter_type, rr_target, direction, sample_size, win_rate, ExpR, Sharpe, fitness status
- Sort by ExpR, NEVER Sharpe alone
- Use `pipeline.paths.GOLD_DB_PATH` — never hardcode

## Session Start
Ambiguous first message → ask ONE question: "Design or implement?" Then follow strictly.
