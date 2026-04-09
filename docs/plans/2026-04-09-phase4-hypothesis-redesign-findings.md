# Phase 4 Hypothesis Redesign — Raw Findings (Low Effort, NEEDS HIGH-EFFORT AUDIT)

**WARNING: This file was produced during a low-effort session. All findings need verification at high effort before acting on them. Do not trust metadata — verify against code and data.**

## Session Context (Apr 9 2026)

Ran Phase 4 discovery with pre-registered hypothesis files. Results exposed gaps in the hypothesis design. This document captures findings for the next high-effort session to audit and redesign.

## Discovery Results (verified — these are from actual runs)

| Instrument | Hypotheses | BH FDR Pass | Validated | Notes |
|---|---|---|---|---|
| MNQ | 16 | 8 | 5 | NYSE_OPEN (4) + EUROPE_FLOW (1) |
| MES | 16 | 0 | 0 | NYSE_OPEN near-miss (FDR_p=0.089 vs 0.05 threshold) |
| MGC | 7 | 0 | 0 | Best: LONDON_METALS G5 (p=0.35, N=72) |

## Money-on-Table Gaps (verified from DB queries)

### Gap 1: MES NYSE_OPEN diluted by negative sessions
- MES NYSE_OPEN G4-G8: all positive ExpR (0.069-0.091), raw p 0.026-0.049
- MES US_DATA_830: ALL NEGATIVE (ExpR -0.053 to -0.133)
- MES EUROPE_FLOW: ALL NEGATIVE (ExpR -0.098 to -0.165)
- Family K=16 diluted the 4 NYSE_OPEN positives with 8 strong negatives
- At K=4 (NYSE_OPEN only), these would likely pass BH FDR

### Gap 2: MNQ OVNRNG_50 missed FDR by 0.002
- FDR_p=0.052, threshold=0.05. N=1192, ExpR=0.089, 5/7 years positive

### Gap 3: COST_LT filters untested in Mode A
- Deployed strategies use COST_LT10/LT12 at SINGAPORE/TOKYO/COMEX
- Legacy p-values: 0.0001 to 0.012 (strong but contaminated)
- Contaminated by: pre-2019 NQ parent data, 2026 holdout, 35K trial brute-force
- Zero Mode A evidence for deployed portfolio dimensions

### Gap 4: RR targets locked to 2.0
- Deployed COMEX uses RR1.5 (ExpR=0.194), EUROPE uses RR3.0 (ExpR=0.138)
- RR2.0 is NOT canonical from Crabel (used EOD exit, not fixed RR)

### Gap 5: OVNRNG thresholds wrong for MES
- MNQ median overnight range = 75.5pts (OVNRNG_50 passes 75%)
- MES median = 17.5pts (OVNRNG_50 passes 7%, OVNRNG_100 = 0.9%)
- Pipeline has OVNRNG_10 and OVNRNG_25 available (not tested)

## Deployed vs Mode A — Zero Overlap (verified)

Deployed `topstep_50k_mnq_auto`:
1. MGC CME_REOPEN G6 RR2.5
2. MNQ SINGAPORE_OPEN COST_LT12 RR2.0
3. MNQ COMEX_SETTLE OVNRNG_100 RR1.5
4. MNQ EUROPE_FLOW COST_LT10 RR3.0
5. MNQ TOKYO_OPEN COST_LT10 RR2.0

Mode A survivors: all NYSE_OPEN + EUROPE_FLOW with G-filters at RR2.0.

## Regime Infrastructure Audit (from low-effort agent — NEEDS VERIFICATION)

### Claimed WIRED IN:
- ATR percentile filters (ATR_P30/P50/P70) — in ALL_FILTERS, used in grid
- Cross-asset ATR (X_MES_ATR70, X_MES_ATR60, X_MGC_ATR70) — MNQ only
- Regime waivers in validator — waives DORMANT years (ATR<20, N<=5)
- DST split metrics — stability verdict per strategy
- regime.discovery + regime.validator — rolling window research tool
- check_regime.py — pre-session ATR dashboard

### Claimed DEAD/DORMANT:
- atr_vel_regime — computed, no filter depends on it
- overnight_range_pct — filter exists but not routed via get_filters_for_grid()
- CUSUM monitor — implemented, not wired to bot
- Shiryaev-Roberts — NOT IMPLEMENTED (referenced in criteria #12)

## Filter Universe (from low-effort agent — NEEDS VERIFICATION)

Claimed 67 total filters in ALL_FILTERS:
- 22 BASE_GRID_FILTERS (always included)
- 45 session-specific composites (routed by get_filters_for_grid())
- ~30 E2-excluded (FAST/CONT variants use break bar timing)
- Key untested dimensions: OVNRNG_10, OVNRNG_25, COST_LT variants at various sessions

## Literature Findings (from training memory — NOT VERIFIED against local PDFs)

### Critical finding: RR2.0 is NOT Crabel canonical
- Crabel used EOD exit, not fixed RR. RR2.0 is practitioner convention.
- Chan (2009): optimal exits are strategy-specific, not assumed.
- Multi-RR testing is more defensible than locking to 2.0.

### What lacks academic grounding:
- Gold session-specific ORB — no academic citation found
- MES multi-session ORB — no academic citation found
- DOW filter on ORB — no specific citation
- Cost-friction as pre-trade filter — project-specific operationalization of Chan Ch1

## NEXT SESSION TODO (HIGH EFFORT REQUIRED)

1. **Audit all of today's work** at high effort — verify code changes, data operations, hypothesis files
2. **Read the actual local PDFs** in resources/ for Crabel, Chan, Aronson, LdP — extract real citations, don't rely on training memory
3. **Verify the regime infrastructure audit** — read the actual code, don't trust the agent's summary
4. **Verify the filter universe** — read config.py, confirm the 67 filter count, confirm routing logic
5. **Design the hypothesis framework properly** — with real lit grounding, proper session/filter/RR matrix per instrument
6. **Consider gold as REGIME strategy** — MGC may need different approach (not unconditional ORB)
7. **Address the deployed portfolio gap** — the 5 deployed lanes have zero Mode A evidence
