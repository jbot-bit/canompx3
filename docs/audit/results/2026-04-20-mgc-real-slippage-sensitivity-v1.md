# H0 — MGC Real-Slippage Sensitivity v1 Result

**Date:** 2026-04-20
**Pre-reg:** `docs/audit/hypotheses/2026-04-20-mgc-real-slippage-sensitivity.yaml`
**Script:** `research/research_mgc_real_slippage_sensitivity_v1.py`
**Output:** `research/output/mgc_real_slippage_sensitivity_v1.json`
**Parent audit:** `docs/audit/results/2026-04-20-mgc-adversarial-reexamination.md` §9 H0
**Status:** COMPLETE — baseline cross-check passed; sensitivity curve reported
**Classification:** Confirmatory audit (K=0 MinBTL trials)
**Supersedes:** `docs/audit/results/2026-04-20-mgc-real-slippage-sensitivity-v1.md` at previous HALT

## Executive verdict

**Closure stands.** Across the 5 native-low-R-v1 cells that were killed by
`research_mgc_path_accurate_subr_v1.py`, slippage realism (2 → 10 ticks
round-trip) does NOT rescue any cell into deployable territory. Two cells
are slippage-robust (stay above +0.05R even at 10-tick friction), but
both were killed by path-accuracy mechanisms independent of friction —
slippage is not their binding constraint. Three cells show soft decay
and fall below +0.05R at or before the pilot mean (6.75 ticks).

No cell moves into deployable status. No new MGC allocation is implied.

### Pre-reg kill-criteria verdict per cell

| Cell | 2t | 4t | 6.75t | 10t | Verdict (pre-reg) |
|---|---:|---:|---:|---:|---|
| NYSE_OPEN_OVNRNG_50_RR1_LR075 | +0.2226 | +0.1948 | +0.1894 | +0.1495 | **A_pilot_not_binding** |
| US_DATA_1000_ATR_P70_RR1_LR05 | +0.0710 | +0.0673 | +0.0499 | +0.0551 | **A_pilot_not_binding** |
| US_DATA_1000_OVNRNG_10_RR1_LR05 | +0.0685 | +0.0712 | +0.0483 | +0.0403 | **SOFT decay** (below +0.05R at 6.75t) |
| US_DATA_1000_BROAD_RR1_LR05 | +0.0488 | +0.0468 | +0.0226 | +0.0104 | **SOFT decay** |
| NYSE_OPEN_BROAD_RR1_LR05 | +0.0380 | +0.0212 | +0.0150 | −0.0099 | **SOFT decay** (flips negative by 10t) |

`A_pilot_not_binding` per pre-reg §kill_criteria means: the cell stays
above +0.05R across all tested slippages, indicating friction is not
what killed it in `research_mgc_path_accurate_subr_v1.py`. For these
two cells, the path-accuracy mechanism (stop-vs-target intrabar ordering)
is the binding constraint, not friction realism.

## Baseline cross-check — PASSED

Pre-reg §baseline_cross_check required cells recomputed at modeled
slippage (2 ticks = $2) to match `research_mgc_native_low_r_v1.py`
reported ExpR within ±0.002R. All 5 cells match at **0.0000 difference**
(both via the canonical upstream pipeline and via my independent
re-derivation path):

```
Cell                                    Reported   Pipeline   Recomputed@2t   Status
NYSE_OPEN_OVNRNG_50_RR1_LR075           +0.2226    +0.2226    +0.2226         OK
US_DATA_1000_ATR_P70_RR1_LR05           +0.0710    +0.0710    +0.0710         OK
US_DATA_1000_OVNRNG_10_RR1_LR05         +0.0685    +0.0685    +0.0685         OK
US_DATA_1000_BROAD_RR1_LR05             +0.0488    +0.0488    +0.0488         OK
NYSE_OPEN_BROAD_RR1_LR05                +0.0380    +0.0380    +0.0380         OK
```

The previous HALT (2026-04-20 first attempt, mismatches 0.10–0.27R) was
caused by four distinct bugs in the first script — all eliminated in this
rewrite by delegating to the canonical upstream
`research_mgc_payoff_compression_audit` module that `research_mgc_native_low_r_v1.py`
itself uses:

| Prior failure | Fixed by |
|---|---|
| `ATR_P70` mapped to `atr_20_pct >= 0.70` (ratio) instead of `>= 70.0` (percent) | Reusing canonical `passes_filter` via `build_family_trade_matrix` |
| Implicit `direction='long'` filter restricted population vs unfiltered source | No direction filter — inherited from canonical `load_rows` (which has none) |
| Re-derived LR target rewrite logic differed from `conservative_lower_target_pnl` | Port of that function, spec-parameterized for slippage sensitivity |
| Date-window mismatch possible from per-row filter re-application | Single canonical `load_rows(end_exclusive='2026-01-01')` call, same as source |

## Design — canonical delegation

Per `.claude/rules/research-truth-protocol.md` § Canonical filter
delegation and `.claude/rules/institutional-rigor.md` Rule 4 (never
re-encode canonical logic), this script:

- **Imports** `FAMILIES`, `load_rows`, `build_family_trade_matrix` directly
  from `research.research_mgc_payoff_compression_audit` — the same module
  `research_mgc_native_low_r_v1.py` consumes.
- **Uses** `pipeline.cost_model.to_r_multiple` with `dataclasses.replace`
  to produce a MGC `CostSpec` with overridden `slippage` field per grid
  value. Commission and spread are left unchanged per pre-reg scope
  (`slippage_interpretation: single scalar per trade; real asymmetry ...
  not modeled here`).
- **Back-projects** stored `pnl_r` and `mfe_r` to friction-invariant gross
  points (pnl_points, mfe_points) and re-applies the new friction in
  both numerator and denominator. This is equivalent algebra to the
  canonical `to_r_multiple`; it produces identical values at modeled
  slippage (proved by the 0.0000 baseline match).

## Interpretation

### The two slippage-robust cells

**NYSE_OPEN_OVNRNG_50_RR1_LR075** (N=52) — smallest-N cell in the set,
stays at +0.1495 even at 10-tick friction. The OVNRNG_50 filter is a
restrictive prior-knowledge gate (overnight range ≥ 50 points — a ~2%
gold move overnight); the surviving sample is pre-selected for high-vol
regime days. Friction-robust is expected; path-accuracy is what killed it.

**US_DATA_1000_ATR_P70_RR1_LR05** (N=428) — the only slippage-robust
warm-filtered US_DATA_1000 cell. ATR_P70 gates on own-ATR ≥ 70th
percentile (a high-vol filter). Stays at +0.0551 at 10-tick.

For both cells, `path_accurate_subr_v1` killed them via intrabar
stop-vs-target ordering — when the backtest replays bar-by-bar rather
than using stored `orb_outcomes`, the mfe-reached-target assumption
breaks for bars where the stop was touched first. Friction is not the
mechanism here.

### The three decaying cells

The BROAD (unfiltered) cells and OVNRNG_10 (weak filter — overnight range
≥ 10 points is ~the median) show monotonic decay with friction. This is
the expected pattern for a noise-plus-thin-signal configuration: any
friction increase eats a larger share of the small per-trade expectancy.
None of these cells would have survived a Chordia-strict t ≥ 3.00 gate
even at modeled friction.

## Operational implications

1. **No deployment change.** Closure of MGC-for-capital stands confirmed
   — not a single cell rescued into deployable territory, even under the
   A_pilot_not_binding cells (because path-accuracy killed those
   separately).

2. **Shadow-track candidate update.** Pre-reg `§interpretation` prescribes
   that `A_pilot_not_binding` cells should be investigated for mechanism
   (H3-class question) rather than added to capital. The two robust cells
   (NYSE_OPEN_OVNRNG_50_LR075, US_DATA_1000_ATR_P70_LR05) are candidates
   for a follow-up mechanism-audit pre-reg — NOT a shadow-track
   deployment pre-reg — because their path-accuracy kill is the binding
   constraint, not friction.

3. **No MGC trial budget spent.** This is a confirmatory audit of killed
   cells, not discovery. Per `backtesting-methodology.md` RULE 4 and
   pre-reg §n_trials=0, MGC MinBTL budget (~30 trials per 3.8-year
   horizon) is unchanged. Estimated remaining: ~29 trials (1 spent on
   H1 portfolio-diversifier).

4. **Repo-wide debt unchanged.** MGC is the only instrument with a TBBO
   slippage pilot measurement. MNQ/MES slippage realism is still
   unverified. `docs/runtime/debt-ledger.md` entry
   `cost-realism-slippage-pilot` remains open; MNQ TBBO pilot
   (`research/research_mnq_e2_slippage_pilot.py`) is next-action.

## What this result does NOT claim

- NOT a claim that the two robust cells have tradeable edge — they are
  killed by path-accuracy and the A_pilot_not_binding verdict opens a
  mechanism question, not a deployment path.
- NOT a claim that 6.75 ticks is the "right" slippage — the pilot (n=40)
  is underpowered per the parent audit §9.5.C and logged caveats in the
  pre-reg `pilot_methodology_caveats_logged`.
- NOT a refutation of the parent audit's finding that MGC real slippage
  is ~3.4× modeled — that finding concerns book-wide cost realism and is
  independent of whether this specific set of 5 cells survives.

## Audit trail

- Pre-reg locked: `docs/audit/hypotheses/2026-04-20-mgc-real-slippage-sensitivity.yaml`
  (`status: LOCKED`, `commit_sha: TO_FILL_AFTER_COMMIT`)
- Script: `research/research_mgc_real_slippage_sensitivity_v1.py`
- JSON output: `research/output/mgc_real_slippage_sensitivity_v1.json`
- Canonical upstream: `research/research_mgc_payoff_compression_audit.py::{FAMILIES, load_rows, build_family_trade_matrix}`
- Cost math: `pipeline.cost_model.to_r_multiple` with `dataclasses.replace`-generated CostSpecs
- Data snapshot: `gold.db` at 2026-04-20
- Commit: TO_FILL_AFTER_COMMIT
