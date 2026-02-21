Research edge discovery for instrument and session: $ARGUMENTS

## Instructions

You are running an AI-assisted strategy research workflow. Follow these steps exactly.

### Step 1: Parse arguments
Parse $ARGUMENTS for instrument (required) and session (optional).
Examples: "MGC 1000", "MES 0030", "MNQ", "MGC all"
Default entry model: E0 for 1000, E1 for all others.

### Step 2: Check research memory
Use claude-mem:mem-search to check if this (instrument, session) combination has been researched before.
Search query: "discover {instrument} {session} edge research"
If previous findings exist, summarize them before running new scans.

### Step 3: Check current validated state
Run a quick query to see how many validated strategies already exist for this instrument/session:
```bash
python -c "
import duckdb
con = duckdb.connect('gold.db', read_only=True)
r = con.execute(\"SELECT COUNT(*) FROM validated_setups WHERE symbol='{INSTRUMENT}' AND orb_label='{SESSION}'\").fetchone()
print(f'Validated strategies for {INSTRUMENT} {SESSION}: {r[0]}')
con.close()
"
```

### Step 4: Run the discovery scanner
Execute the scanner and capture JSON output:
```bash
python research/discover.py --instrument {INSTRUMENT} --session {SESSION} --entry-model {EM} --json
```
If no session specified, add `--all-sessions`.

### Step 5: Interpret results
For each scan result, apply RESEARCH_RULES.md labels:
- BH-significant with p<0.005: "validated finding"
- BH-significant with p<0.05: "promising hypothesis"
- Not BH-significant: "statistical observation" (mention but don't recommend action)
- Baseline ExpR <= 0: "NO-GO — negative baseline"

CRITICAL: For DST-affected sessions (0900, 1800, 0030, 2300), always report winter AND summer splits separately.

### Step 6: Report
Format findings as:

**{INSTRUMENT} {SESSION} Discovery Report**

| Predictor | Delta | p-value | BH-sig? | N | Label |
|-----------|-------|---------|---------|---|-------|
| ... | ... | ... | ... | ... | ... |

Key findings: [2-3 bullet summary]
Recommended actions: [specific next steps, if any survive FDR]

### Step 7: Save to memory
Save findings using claude-mem:mem-search save_memory with key "discover_{instrument}_{session}_{date}".
Include: n_tests, n_significant, top findings, recommended actions.

### Rules (from RESEARCH_RULES.md)
- NEVER say "significant" without p-value
- NEVER say "edge" without BH FDR confirmation
- Sample size labels: <30 INVALID, 30-99 REGIME, 100+ CORE
- RSI/MACD/Bollinger are "guilty until proven" — flag if they appear significant
- Always include year-by-year breakdown for any BH-significant finding
