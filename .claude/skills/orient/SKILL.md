---
name: orient
description: Project awareness — synthesize project state, blueprint routing, and recommend next action
allowed-tools: Bash, Read, Grep, Glob
---
Orient me on the current project state: $ARGUMENTS

Use when: "orient", "what's going on", "what's broken", "status", "where are we", "session start", "catch me up"

## Step 0: Check active stage

Read `docs/runtime/STAGE_STATE.md` (or `docs/runtime/stages/*.md`) if it exists.
- Active stage → surface at TOP: "**ACTIVE STAGE:** [task] — [mode]"
- IMPLEMENTATION mid-flight → recommend `/resume-rebase`
- Stale (git log shows scope files changed since `updated`) → flag STALE

## Step 1: Run the pulse

Default (fast): `python scripts/tools/project_pulse.py --fast --format json`
With `--full`: `python scripts/tools/project_pulse.py --format json`

The JSON has: `recommendation`, `counts`, `handoff`, `fitness_summary`, `upcoming_sessions`, `time_since_green`, `session_delta`, `items` (with `action` fields).

## Step 2: Load strategic context

```bash
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
vs = con.sql(\"SELECT instrument, COUNT(*) as n FROM validated_setups WHERE status = 'active' GROUP BY instrument\").fetchall()
ef = con.sql('SELECT COUNT(*) FROM edge_families').fetchone()[0]
print(f'Active: {ACTIVE_ORB_INSTRUMENTS}')
print(f'Validated: {dict(vs) if vs else \"NONE\"}')
print(f'Edge families: {ef}')
con.close()
"
```

## Step 3: Narrate (<25 lines)

In order (skip empty):
1. **Session delta** — commits since last session
2. **BROKEN** — each item + its `action` field
3. **DECAYING** — stale pipelines, WATCH/DECAY strategies
4. **Strategic state** — active instruments, validated count, edge families (one-liner each). List dead instruments on a SEPARATE line, never in the same sentence as "active"/"deployed"/"live".
5. **Upcoming sessions** — next 6h with strategy counts
6. **Context** — handoff summary + top 3 next steps
7. **Time since green** — one line if not "now"

## Step 4: Route

Use pulse JSON `recommendation` directly. Format: `>>> [recommendation] <<<`

## Rules

- Be concise. <25 lines total.
- Do NOT re-read HANDOFF.md — pulse JSON already has it.
- Strategy counts from live queries, NEVER from memory.
- Each item has an `action` field — always show it.
