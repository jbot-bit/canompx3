---
name: orient
description: Project awareness — synthesize project state and recommend next action
allowed-tools: Bash, Read, Grep, Glob
---
Orient me on the current project state: $ARGUMENTS

Use when: "orient", "what's going on", "what's broken", "what should I work on", "status", "where are we", "project state", "session start", "what happened", "catch me up"

## Step 1: Run the pulse

If `$ARGUMENTS` contains `--full`:
```bash
python scripts/tools/project_pulse.py --format json
```

Otherwise (default — fast mode, serves cached drift/tests):
```bash
python scripts/tools/project_pulse.py --fast --format json
```

Read the JSON output. It contains:
- `counts`: how many items in each category (broken, decaying, ready, unactioned, paused)
- `handoff`: last session context + next steps
- `fitness_summary`: strategy counts per instrument
- `items`: categorized findings with severity

## Step 2: Narrate priorities

Present a concise briefing to the user. Follow this priority order — stop at the first non-empty category:

1. **BROKEN (fix now):** If `counts.broken > 0`, lead with these. Name each broken item and what to do about it. Do NOT suggest other work until broken items are addressed.

2. **DECAYING (act soon):** If strategies are in WATCH/DECAY or pipeline steps are stale, flag them. Include the instrument and what's stale.

3. **CONTEXT (what was happening):** Always include the handoff summary + next steps. This orients the user on where they left off.

4. **READY (on deck):** List action queue items as options the user can pick from.

## Step 3: Recommend one action

End with ONE specific recommendation: the single most impactful thing to do right now. Not a list. One thing.

Format: `>>> Recommended: [action] <<<`

## Rules
- Be concise. The whole briefing should be <20 lines.
- Do NOT re-read HANDOFF.md, MEMORY.md, or other files — the pulse JSON already has everything.
- If the user says `/orient --full`, run without `--fast` to include drift checks and tests (slower).
- Strategy counts are from `fitness_summary` — do not query the DB separately.
