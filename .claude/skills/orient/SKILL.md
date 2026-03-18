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
- `recommendation`: the single most impactful next action (USE THIS — don't compute your own)
- `counts`: how many items in each category (broken, decaying, ready, unactioned, paused)
- `handoff`: last session context + next steps
- `fitness_summary`: strategy counts per instrument
- `upcoming_sessions`: trading sessions starting in the next 6 hours with strategy counts
- `time_since_green`: how long since the system was fully clean
- `session_delta`: what changed since last session (commits by other tools)
- `items`: categorized findings with severity + `action` field (suggested skill/command)

## Step 2: Narrate the briefing

Present a concise briefing. Include these sections IN ORDER, skip empty ones:

1. **Session delta** (if present): "Since your last session: [commits]" — critical for multi-tool awareness
2. **BROKEN (fix now):** Name each broken item + its `action` field (e.g., "Drift failed -> /health-check")
3. **DECAYING (act soon):** Stale pipelines, WATCH/DECAY strategies, stale handoff
4. **Upcoming sessions:** Trading sessions in the next 6h with per-instrument strategy counts
5. **CONTEXT:** Handoff summary + top 3 next steps
6. **Time since green:** One line if not "now"

## Step 3: Deliver the recommendation

The pulse JSON has a `recommendation` field. Use it directly — do not recompute.

Format: `>>> [recommendation] <<<`

## Rules
- Be concise. The whole briefing should be <20 lines.
- Do NOT re-read HANDOFF.md, MEMORY.md, or other files — the pulse JSON already has everything.
- If the user says `/orient --full`, run without `--fast` to include drift checks and tests (slower).
- Strategy counts are from `fitness_summary` — do not query the DB separately.
- Each item has an `action` field — always show it so the user can copy-paste the command.
