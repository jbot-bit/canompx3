# Untapped Session Candidates — Session Discovery Scan (2026-03-01)

## Scan Parameters
- Script: `research/research_session_discovery.py`
- Grid: 288 times x 5 RR x 4 G-filters (G0/G4/G5/G6) = 5,760 combos/instrument
- Break detection: close-based (NOT E2 stop-market)
- BH FDR q=0.05 across 20,180 tested combos
- 49 FDR survivors total

## KEY FINDING: G-Filter Bias
Adding G0 (no ORB size minimum) revealed that:
- **MGC**: ZERO edges at G0 at any time (p<0.05). G4+ "edges" are selection bias.
- **MES**: ZERO edges at G0 at any time (p<0.05). Same issue.
- **MNQ**: G0 barely changes N (+1-3 trades). Edge is real at all filter levels.
- **M2K**: G0 unlocks massive hidden signal (N jumps from 202 to 1,287 at 09:25).

## TIER 1: Pipeline Candidates (FDR survivors, N>=700)

### MNQ 10:25 Brisbane
- N=1,272-1,289 (G0-G6), avgR=+0.221 to +0.247 (RR2.5-3.0)
- FDR significant at all G-filters, Sharpe_ann=2.09-2.21
- Positive 6/6 years. Both seasons positive (Rw=+0.305, Rs=+0.189)
- **Not near any existing session** (25 min from nearest)
- STATUS: Ready for pipeline

### MNQ 09:40 Brisbane
- N=1,270-1,290 (G0-G6), avgR=+0.185-0.203 (RR2.5-3.0)
- FDR significant, Sharpe_ann=1.78-1.81
- Positive 6/6 years. Both seasons positive (Rw=+0.099, Rs=+0.231)
- 20 min from CME_REOPEN (09:00) — may overlap
- STATUS: Ready for pipeline (check overlap with CME_REOPEN)

### MNQ 19:55 Brisbane
- N=724-1,291 (G6-G0), avgR=+0.167-0.238 (RR2.0-2.5)
- FDR significant at G5/G6, Sharpe_ann=1.68-1.73
- Positive 5-6/6 years. Both seasons positive (Rw=+0.197, Rs=+0.192)
- 115 min from nearest session — completely independent liquidity pool
- STATUS: Ready for pipeline

### M2K 09:25 Brisbane (G0 ONLY)
- N=1,287 at G0, avgR=+0.310 (RR3.0), Sharpe_ann=2.54
- **FDR significant** — 2 of 49 total survivors
- Positive 5/6 years
- CAUTION: Winter avgR=-0.020 at RR2.0, summer=+0.257. Summer-dominant.
- At G4: N drops to 202 (85% of data lost). G4 filter destroys this signal.
- STATUS: Ready for pipeline — test with G0 (no ORB size filter)

## TIER 2: Worth Testing (N>=300, p<0.01, both seasons positive)

### MNQ 09:35 Brisbane
- N=1,268-1,290, avgR=+0.174-0.175 (RR3.0)
- Both seasons positive but winter weak (Rw=+0.074-0.084)

### MNQ 03:55 Brisbane
- N=1,078-1,290, avgR=+0.180-0.192 (RR3.0)
- Both seasons positive. Near COMEX_SETTLE? No — that's 04:25 area.

### MNQ 20:05 Brisbane
- N=1,060-1,243, avgR=+0.186 (RR3.0 G6)
- Both seasons positive (Rw=+0.145, Rs=+0.207)

### MNQ 16:50-16:55 Brisbane
- N=771-960, avgR=+0.150-0.191
- Near LONDON_METALS session. May be redundant.

### M2K 09:35 Brisbane (G6)
- N=504, avgR=+0.148 (RR1.0), Sharpe_ann=1.54
- Both seasons positive (Rw=+0.295, Rs=+0.129)

## TIER 3: Monitor Only (interesting but thin)

### MGC — ALL edges are filter-dependent
- MGC 10:05 (N=274 at G5), MGC 10:20 (N=372 at G4)
- These ONLY appear with G4+ filter. Unfiltered population is negative.
- Interpretation: "wide gold ORBs tend to continue" — volatility filter, not time edge.

### MES — ALL edges are filter-dependent
- MES 11:05 (N=696 at G6), MES 16:45 (N=452 at G4)
- Same issue as MGC. Unfiltered population has no edge.

## Robustness Results (pre-pipeline)
- [x] Year-by-year leave-one-out: MNQ 09:25 stable (all years OK). M2K 09:25 stable (all years OK).
- [x] Direction split: M2K 09:25 positive BOTH long (+0.268) and short (+0.354)
- [x] Multiple comparisons: 49 FDR from 20,180 tests. BH guarantees <5% false discovery. ~5 distinct edges.
- [x] Effect size: avgR +0.17 to +0.31 — not micro-edges inflated by big N
- [x] Pipeline verification (E2 stop-market entry) — COMPLETE

## Pipeline Verification Results (2026-03-02)

### MNQ — BRISBANE_1025 CONFIRMED
- 684,432 outcomes, 5,376 strategies discovered, 287 validated, 97 FDR survivors
- **BRISBANE_1025: 17 validated, 4 FDR survivors** — real edge survives E2 entry
  - Best: E2 G6 RR3.0 CB1, avgR=+0.141, N=861, FDR significant
  - All E2 with G5/G6/G8 or VOL_RV12 filters, RR1.0-4.0
- BRISBANE_1955: 0 validated — best experimental Sharpe=1.05 (RR1.0, vol filter). Edge too weak for E2.
- BRISBANE_0925: 0 validated in this rebuild (was 1 previously — marginal)
- Edge families: 240 families, 24 ROBUST, 44 WHITELISTED

### M2K — BRISBANE SESSIONS ALL NEGATIVE
- 631,764 outcomes, 5,082 strategies discovered, 86 validated (non-BRISBANE), 29 FDR
- **BRISBANE_0925: ALL strategies negative** (avgR = -0.32 to -0.47). Scan showed +0.310R.
- **BRISBANE_1025: ALL strategies negative**
- **Root cause:** Scan used close-based break detection. Pipeline uses E2 stop-market entry. M2K's small ORBs at BRISBANE times get stopped out before bar close — the scan's "edge" was an entry-model artifact.
- M2K's validated strategies are all existing sessions (LONDON_METALS, SINGAPORE_OPEN, NYSE_OPEN, etc.) with G5/G6/G8 filters.

### Lesson: Close-Based Scan ≠ E2 Pipeline
The session discovery scan is a cheap screen, not a trading signal. It identifies *candidate* times worth testing in the pipeline. For instruments with small ORBs (M2K), the difference between "price closed beyond ORB" and "stop-market entry" is fatal. For instruments with larger ORBs (MNQ), results align.

## Sessions Added to Pipeline (2026-03-01)
| Session | Brisbane Time | Instruments | break_group |
|---------|-------------|-------------|-------------|
| BRISBANE_1025 | 10:25 | MNQ, M2K | asia |
| BRISBANE_1955 | 19:55 | MNQ only | london |

Note: M2K BRISBANE sessions have zero validated strategies but remain in the pipeline for ongoing monitoring. BRISBANE_1955 likewise for MNQ.

Files modified:
- `pipeline/dst.py` — resolver functions + SESSION_CATALOG + DST_CLEAN + DOW_ALIGNED
- `pipeline/init_db.py` — ORB_LABELS_DYNAMIC
- `pipeline/asset_configs.py` — enabled_sessions for MNQ, M2K, NQ
- `trading_app/config.py` — ORB_WINDOWS_MINUTES, EARLY_EXIT_MINUTES, SESSION_EXIT_MODE
- `trading_app/ai/sql_adapter.py` — VALID_ORB_LABELS
- `trading_app/ai/grounding.py` — session list in prompt
- `tests/test_app_sync.py` — EXPECTED_ORB_LABELS, grid size (2772→3276)
- `tests/test_pipeline/test_dst.py` — session count (11→13), resolver set
- `tests/test_pipeline/test_init_db.py` — session list
- `tests/test_trading_app/test_early_exits.py` — expected sessions set

All 1,850 tests passing, all 41 drift checks clean.
