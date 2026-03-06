Research edge discovery for instrument and session: $ARGUMENTS

Use when: "discover", "scan for edges", "research [instrument]", "find strategies", "edge discovery", "what works for [instrument]", "test [session]"

## Instructions

You are running an AI-assisted strategy research workflow. Follow these steps exactly.

### Step 1: Parse Arguments

Parse $ARGUMENTS for instrument (required) and session (optional).
Examples: "MGC CME_REOPEN", "MES NYSE_OPEN", "MNQ", "MGC all"

Session names use the dynamic catalog: CME_REOPEN, TOKYO_OPEN, SINGAPORE_OPEN, LONDON_METALS, US_DATA_830, NYSE_OPEN, US_DATA_1000, COMEX_SETTLE, CME_PRECLOSE, NYSE_CLOSE, BRISBANE_1025.

Default entry model: E2 (stop-market, industry standard). Use E1 for conservative baseline comparison.
E0 is DEAD (purged Feb 2026) -- never use E0.

### Step 2: Check Research Memory

Check memory files in the project memory directory for previous findings on this (instrument, session).
Relevant files: `regime_findings.md`, `m2k_findings.md`, `mgc_regime_analysis.md`, `aperture_comparison.md`, and other topic files in MEMORY.md index.
If previous findings exist, summarize them before running new scans.

### Step 3: Check Current Validated State

```bash
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
r = con.execute(\"SELECT COUNT(*) FROM validated_setups WHERE symbol='{INSTRUMENT}'\").fetchone()
print(f'Validated strategies for {INSTRUMENT}: {r[0]}')
con.close()
"
```

### Step 4: Run the Discovery Scanner

```bash
python research/discover.py --instrument {INSTRUMENT} --session {SESSION} --entry-model E2 --json
```
If no session specified, add `--all-sessions`.

### Step 5: Interpret Results

For each scan result, apply RESEARCH_RULES.md labels:
- BH-significant with p<0.005: "validated finding"
- BH-significant with p<0.05: "promising hypothesis"
- Not BH-significant: "statistical observation" (mention but don't recommend action)
- Baseline ExpR <= 0: "NO-GO -- negative baseline"

CRITICAL: Run the actual statistical test. NEVER present counts or eyeball comparisons as analysis. Every quantitative claim MUST have a p-value from an actual test.

### Step 6: Report

**{INSTRUMENT} {SESSION} Discovery Report**

| Predictor | Delta | p-value | BH-sig? | N | Label |
|-----------|-------|---------|---------|---|-------|
| ... | ... | ... | ... | ... | ... |

Key findings: [2-3 bullet summary]
Recommended actions: [specific next steps, if any survive FDR]

### Step 7: Save Findings

Save findings to the appropriate memory file:
- MGC findings -> `regime_findings.md` or `mgc_regime_analysis.md`
- MNQ findings -> update existing topic file or create new one
- New instrument -> create `{instrument}_findings.md`

Include: n_tests, n_significant, top findings, recommended actions, date.

### Rules (from RESEARCH_RULES.md)

- NEVER say "significant" without p-value
- NEVER say "edge" without BH FDR confirmation
- Sample size labels: <30 INVALID, 30-99 REGIME, 100+ CORE
- RSI/MACD/Bollinger are "guilty until proven" -- flag if they appear significant
- Always include year-by-year breakdown for any BH-significant finding
- All sessions are dynamic/event-based from SESSION_CATALOG -- DST is fully resolved
