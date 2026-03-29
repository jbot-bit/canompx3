---
name: discover
description: Research edge discovery for instrument and session — follows Blueprint test sequence
allowed-tools: Read, Grep, Glob, Bash
---
Research edge discovery for instrument and session: $ARGUMENTS

Use when: "discover", "scan for edges", "research [instrument]", "find strategies", "edge discovery"

## Step 0: Blueprint Pre-Check (MANDATORY)

Before ANY research, check `docs/STRATEGY_BLUEPRINT.md`:
1. **NO-GO Registry (SS5):** Already dead? STOP.
2. **Variable Space (SS4):** MNQ E2 = only positive unfiltered baseline. MGC/MES need size filters.
3. **Assumptions (SS10):** Flag relevant risks.

Verify instrument is active:
```bash
python -c "
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, ASSET_CONFIGS
inst = '$ARGUMENTS'.split()[0]
print(f'{inst} active: {inst in ACTIVE_ORB_INSTRUMENTS}')
print(f'Sessions: {ASSET_CONFIGS.get(inst, {}).get(\"enabled_sessions\", [])}')
"
```

## Step 1: Parse Arguments

Parse $ARGUMENTS for instrument (required) and session (optional).
Default entry model: E2. E0 is PURGED. E3 is in SKIP_ENTRY_MODELS.

## Step 2: Baseline Viability (Gate 2)

Test >=3 RR values and >=3 apertures before declaring dead:
```bash
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
for rr in [1.0, 1.5, 2.0]:
    for om in [5, 15, 30]:
        r = con.execute('SELECT COUNT(*), ROUND(AVG(pnl_r),4) FROM orb_outcomes WHERE symbol=? AND entry_model=? AND rr_target=? AND orb_minutes=? AND confirm_bars=1', ['INST', 'E2', rr, om]).fetchone()
        print(f'  {\"+ \" if r[1] and r[1]>0 else \"- \"}RR{rr} O{om}: N={r[0]:,} ExpR={r[1]}')
con.close()
"
```

## Step 3: Previous Research + Current State

Check memory files and Blueprint SS5 NO-GO table for prior findings. Query validated_setups for current state.

## Step 4: Per-Session Scan

Query orb_outcomes grouped by orb_label for the instrument. If specific session requested, run detailed analysis.

## Step 5: Interpret (Gates 1+3)

For each finding:
1. **Mechanism (Gate 1):** WHY should this work? If "numbers show it" — suspicious.
2. **Statistics (Gate 3):** BH FDR at honest K. Labels: p<0.005 "validated", p<0.05 "promising", else "observation".

## Step 6: Report

| Session | RR | O | ExpR | WR | N | p-value | BH-sig? | Mechanism | Label |
|---------|-----|---|------|-----|---|---------|---------|-----------|-------|

Include: variable coverage (RR/apertures tested vs missing), key findings, NO-GO paths hit.

## Rules

- NEVER say "significant" without p-value or "edge" without BH FDR
- Sample size: <30 INVALID, 30-99 REGIME, 100+ CORE
- Before declaring dead: test >=3 RR, >=3 apertures, E1+E2
- Check NO-GO registry FIRST
