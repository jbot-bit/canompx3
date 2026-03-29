---
name: discover
description: Research edge discovery for instrument and session — follows Blueprint test sequence
allowed-tools: Read, Grep, Glob, Bash
---
Research edge discovery for instrument and session: $ARGUMENTS

Use when: "discover", "scan for edges", "research [instrument]", "find strategies", "edge discovery", "what works for [instrument]", "test [session]"

## Step 0: Blueprint Pre-Check (MANDATORY)

Before ANY research, check `docs/STRATEGY_BLUEPRINT.md`:

1. **NO-GO Registry (§5):** Is this instrument/session/approach already dead? If yes, tell the user immediately and STOP. Don't rediscover dead paths.
2. **Variable Space (§4):** What's the baseline for this instrument? MNQ E2 is the only positive unfiltered baseline. MGC/MES need size filters.
3. **What We Might Be Wrong About (§10):** Flag relevant assumptions.
4. **Active Threads (§9):** Is someone already working on this?

```bash
python -c "
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, ASSET_CONFIGS
inst = '{INSTRUMENT}'
if inst not in ACTIVE_ORB_INSTRUMENTS:
    print(f'WARNING: {inst} is NOT in ACTIVE_ORB_INSTRUMENTS. Check if dead.')
else:
    cfg = ASSET_CONFIGS.get(inst, {})
    sessions = cfg.get('enabled_sessions', [])
    print(f'{inst} enabled sessions: {sessions}')
"
```

## Step 1: Parse Arguments

Parse $ARGUMENTS for instrument (required) and session (optional).
Examples: "MGC CME_REOPEN", "MES NYSE_OPEN", "MNQ", "MGC all"

Session names from `SESSION_CATALOG`: CME_REOPEN, TOKYO_OPEN, SINGAPORE_OPEN, LONDON_METALS, EUROPE_FLOW, US_DATA_830, NYSE_OPEN, US_DATA_1000, COMEX_SETTLE, CME_PRECLOSE, NYSE_CLOSE, BRISBANE_1025.

Default entry model: E2 (stop-market). E0 is PURGED. E3 is in SKIP_ENTRY_MODELS.

## Step 2: Baseline Viability (Blueprint Gate 2)

Before running discovery, check if a baseline exists. This determines whether to proceed or stop.

```bash
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
# Check across RR targets — don't just test one
for rr in [1.0, 1.5, 2.0]:
    for om in [5, 15, 30]:
        r = con.execute('''
            SELECT COUNT(*) as n, ROUND(AVG(pnl_r), 4) as expr
            FROM orb_outcomes
            WHERE symbol=? AND entry_model='E2' AND rr_target=? AND orb_minutes=?
              AND confirm_bars=1
        ''', ['{INSTRUMENT}', rr, om]).fetchone()
        tag = '+' if r[1] and r[1] > 0 else '-'
        print(f'  {tag} RR{rr} O{om}: N={r[0]:,} ExpR={r[1]}')
con.close()
"
```

**CRITICAL RULE (from Blueprint §3 Gate 2):** Test ≥3 RR values and ≥3 apertures before declaring dead. ONE negative point doesn't kill the space.

If ALL combinations are negative → report "NO-GO: negative baseline across all tested RR/aperture combinations" and recommend checking with size filters (G4+/G5+).

## Step 3: Check Previous Research

Check memory files AND the NO-GO registry for previous findings:
- Blueprint §5: consolidated NO-GO table
- Memory files: `regime_findings.md`, `m2k_findings.md`, `aperture_comparison.md`, topic files in MEMORY.md

If previous findings exist, summarize them. Don't repeat dead-end research.

## Step 4: Check Current Validated State

```bash
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
r = con.execute(\"SELECT strategy_id, orb_label, entry_model, rr_target, orb_minutes, filter_type, sample_size, ROUND(expectancy_r,4) as expr FROM validated_setups WHERE instrument='{INSTRUMENT}' AND LOWER(status)='active' ORDER BY expectancy_r DESC\").fetchall()
print(f'Validated strategies for {\"INSTRUMENT\"}: {len(r)}')
for row in r:
    print(f'  {row[0]} | {row[1]} {row[2]} RR{row[3]} O{row[4]} {row[5]} N={row[6]} ExpR={row[7]}')
con.close()
"
```

## Step 5: Per-Session Scan

For each session (or the specified session), query the baseline:

```bash
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
print(con.sql('''
    SELECT orb_label, COUNT(*) as n, ROUND(AVG(pnl_r), 4) as expr,
           ROUND(COUNT(CASE WHEN pnl_r > 0 THEN 1 END)*100.0/COUNT(*), 1) as wr
    FROM orb_outcomes
    WHERE symbol='{INSTRUMENT}' AND entry_model='E2' AND rr_target=1.0
      AND confirm_bars=1 AND orb_minutes=5
    GROUP BY orb_label ORDER BY expr DESC
''').fetchdf().to_string(index=False))
con.close()
"
```

If a specific session was requested, also run the discovery scanner:
```bash
python research/discover.py --instrument {INSTRUMENT} --session {SESSION} --entry-model E2 --json
```

## Step 6: Interpret Results (Blueprint Gates 1+3)

For each finding:

1. **Mechanism check (Gate 1):** WHY should this work? Plausible structural reason? If "the numbers show it" — flag as suspicious.
2. **Statistical test (Gate 3):** BH FDR at honest test count. NEVER present counts or eyeball comparisons.
   - p < 0.005: "validated finding"
   - p < 0.05: "promising hypothesis"
   - Not BH-significant: "statistical observation"
   - ExpR ≤ 0: "NO-GO — negative baseline"
3. **Variable coverage:** Did you test enough of the space? Mark what's untested.

## Step 7: Report

**{INSTRUMENT} {SESSION} Discovery Report**

| Session | RR | O | ExpR | WR | N | p-value | BH-sig? | Mechanism | Label |
|---------|-----|---|------|-----|---|---------|---------|-----------|-------|

**Variable coverage:**
- RR tested: [list] — missing: [list]
- Apertures tested: [list] — missing: [list]
- Entry models: E2 ☑ E1 ☐

**Key findings:** [2-3 bullets]
**Recommended next steps:** [specific, with Blueprint gate reference]
**NO-GO paths hit:** [list any dead ends found]

## Rules (from RESEARCH_RULES.md + Blueprint)

- NEVER say "significant" without p-value
- NEVER say "edge" without BH FDR confirmation
- Sample size: <30 INVALID, 30-99 REGIME, 100+ CORE
- RSI/MACD/Bollinger are "guilty until proven"
- Always include year-by-year breakdown for BH-significant findings
- All sessions are dynamic/event-based from SESSION_CATALOG
- Before declaring dead: tested ≥3 RR, ≥3 apertures, E1+E2?
- Check NO-GO registry FIRST — don't rediscover dead paths

## Next → After Discovery

- BH-significant findings? → `/research [finding]` for deeper hypothesis testing
- Ready to implement? → `/design [feature]` (or `/design auto` to proceed immediately)
- Want review? → `/bloomey-review`
- Everything NO-GO? → Update Blueprint §5, move on
