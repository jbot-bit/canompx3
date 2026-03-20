---
name: orient
description: Project awareness — synthesize project state, blueprint routing, and recommend next action
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

## Step 2: Load strategic context

Read the active research threads from the blueprint:
```bash
python -c "
# Quick state check — what's active, what's next
import duckdb
from pipeline.paths import GOLD_DB_PATH
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
vs = con.sql('SELECT instrument, COUNT(*) as n FROM validated_setups WHERE status = \'active\' GROUP BY instrument').fetchall()
ef = con.sql('SELECT COUNT(*) FROM edge_families').fetchone()[0]
print(f'Active instruments: {ACTIVE_ORB_INSTRUMENTS}')
print(f'Validated setups: {dict(vs) if vs else \"NONE\"}')
print(f'Edge families: {ef} rows')
# Check model on disk
import os
model_path = 'models/ml/meta_label_MNQ_hybrid.joblib'
print(f'ML model: {\"EXISTS\" if os.path.exists(model_path) else \"NONE\"}')
con.close()
"
```

Also check `docs/STRATEGY_BLUEPRINT.md §9` (Active Research Threads) — but treat it as STALE. The pulse JSON and live queries above are truth.

## Step 3: Narrate the briefing

Present a concise briefing. Include these sections IN ORDER, skip empty ones:

1. **Session delta** (if present): "Since your last session: [commits]" — critical for multi-tool awareness
2. **BROKEN (fix now):** Name each broken item + its `action` field (e.g., "Drift failed -> /health-check")
3. **DECAYING (act soon):** Stale pipelines, WATCH/DECAY strategies, stale handoff
4. **Strategic state:** Active instruments, validated setup count, ML model status, edge families status. One-liner per item.
5. **Upcoming sessions:** Trading sessions in the next 6h with per-instrument strategy counts
6. **Active threads:** What's in progress from blueprint §9 (paper trading ready? ML status? 2026 holdout?)
7. **CONTEXT:** Handoff summary + top 3 next steps
8. **Time since green:** One line if not "now"

## Step 4: Route to next action

The pulse JSON has a `recommendation` field. Use it directly — do not recompute.

But ALSO consider:
- If user seems to want strategy work → "Check `docs/STRATEGY_BLUEPRINT.md` — route to the right section"
- If user asks "what should I research" → point to blueprint §9 Active Threads or §5 NO-GO gaps
- If user asks "what's broken" → focus on BROKEN items only
- If user asks "catch me up" → focus on session delta + handoff

Format recommendation: `>>> [recommendation] <<<`

## Rules
- Be concise. The whole briefing should be <25 lines.
- Do NOT re-read HANDOFF.md, MEMORY.md, or other files — the pulse JSON already has everything.
- If the user says `/orient --full`, run without `--fast` to include drift checks and tests (slower).
- Strategy counts are from `fitness_summary` + the live query — do not cite stale memory.
- Each item has an `action` field — always show it so the user can copy-paste the command.
- When in doubt about a number, QUERY. Never cite from memory for volatile data.

## Next → Route the User

Based on the briefing, suggest the appropriate next skill:
- Something broken? → `/health-check` or `/quant-debug`
- Strategy research needed? → `/discover [instrument]` or `/research [topic]`
- Portfolio health? → `/regime-check`
- Want to know what to trade? → `/trade-book`
- Planning a change? → `/brainstorm [topic]`
- ML question? → `/ml-verify`
