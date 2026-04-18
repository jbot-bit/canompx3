# Research filter-delegation audit — 2026-04-19

**Generated:** 2026-04-19 (overnight session)
**Source script:** ad-hoc Python audit run inline (not committed — audit is one-shot)
**Canonical rule:** `.claude/rules/research-truth-protocol.md` § Canonical filter delegation (added 2026-04-18)

## Goal

Find every Python file under `research/` that re-implements canonical filter logic from `trading_app.config.ALL_FILTERS` rather than delegating to `research.filter_utils.filter_signal`. Fix the offenders in Phase 2.

## Method

Scanned 266 `research/**/*.py` files (including `archive/` and `__pycache__`). Flagged files matching any of:

- Inline ratio-gate patterns on `overnight_range / atr_20 >= 1.0` (classic OVNRNG_100 mis-impl)
- Inline `if filter_key == "<canonical>":` branches for OVNRNG_100 / VWAP_MID_ALIGNED / ORB_G5 / ATR_P50 / COST_LT
- Helper function declarations `compute_deployed_filter`, `vwap_signal`, `deployed_filter_signal`

Cross-checked each hit against whether the file already imports `research.filter_utils.filter_signal` (canonical delegation).

## Raw findings

| File | Last modified | Hits |
|---|---|---|
| `research/comprehensive_deployed_lane_scan.py` | 2026-04-15 | 5 inline branches + ratio gate + compute_deployed_filter declaration |
| `research/garch_partner_state_provenance_audit.py` | 2026-04-16 | 1 ratio-pattern match (FALSE POSITIVE — see classification below) |
| `research/research_trend_day_mfe.py` | 2026-03-16 | 1 ratio-pattern match (FALSE POSITIVE — see classification below) |

## Per-file classification

### `research/comprehensive_deployed_lane_scan.py` — **GENUINE OFFENDER**

`compute_deployed_filter(df, filter_key)` at line ~226 has inline branches for:
- `OVNRNG_100`: `overnight_range / atr_20 >= 1.0` — **WRONG**. Canonical `OvernightRangeAbsFilter(min_range=100.0)` at `trading_app/config.py:1384` is absolute `overnight_range >= 100.0`. Different semantics, different fire-mask, different results.
- `ATR_P50`: `atr_20 >= np.nanpercentile(atr_20, 50)` — structurally plausible but look-ahead-contaminated (percentile over full sample).
- `OVNRNG_100`, `ATR_P50`, `VWAP_MID_ALIGNED`, `ORB_G5` — four canonical filters re-implemented inline.

**Verdict:** FIX in Phase 2. Replace `compute_deployed_filter` body with delegation to `filter_signal(df, filter_key, orb_label)`.

**Downstream contamination:** every scan output that invokes `compute_deployed_filter(..., "OVNRNG_100")` is wrong on the OVNRNG_100 cells. The 2026-04-15 comprehensive scan output (`docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md`) did not apply overlay filtering for `OVNRNG_100` explicitly in the BH-global survivor list — it tested rel_vol/bb_volume/break_delay etc. as NEW feature overlays on unfiltered or deployed-filter populations. Grep confirmed the survivor list cells do not filter by OVNRNG_100; the only `compute_deployed_filter` overlay cells use the `deployed_filter` key of the Alpha-deployed lane (which for MNQ COMEX_SETTLE IS OVNRNG_100). So at least one cell IS contaminated: MNQ COMEX_SETTLE short RR1.5 marked `deployed` in the scan's survivor table used OVNRNG_100-as-ratio as its gate — wrong. Flag: any "deployed" survivor in the 2026-04-15 scan is suspect until re-run.

### `research/garch_partner_state_provenance_audit.py` — **FALSE POSITIVE**

File defines custom research candidates at line 64-75:
```python
Candidate("M1", "OVN_NOT_HIGH_60", "overnight_range_pct < 60"),
Candidate("M2", "ATRVEL_GE_100", "atr_vel_ratio >= 1.00"),
...
```

Each is a NEW research-provisional filter candidate being audited for partner-state mechanism. These are NOT re-implementations of `ALL_FILTERS` entries. They have distinct names (`OVN_NOT_HIGH_60` vs canonical `OVNRNG_100`), distinct semantics (`<` upper bound vs `>=` lower bound), and serve a different purpose (partner-state decomposition).

My grep regex hit `ratio >= 1.0` at line 70 (`atr_vel_ratio >= 1.00`) which is unrelated to OVNRNG canonical filter — it's an ATR-velocity-ratio candidate.

**Verdict:** no fix needed.

### `research/research_trend_day_mfe.py` — **FALSE POSITIVE**

File declares derived feature `overnight_expansion = overnight_range / atr_20` at line 794-799 and includes it in `PREDICTOR_COLS` for an MFE regression. The derived column is NEW (not in `daily_features`), created only inside this script, used as a continuous regressor, NOT as a filter fire-mask.

This is legitimate feature engineering for regression analysis, not filter re-implementation.

**Verdict:** no fix needed.

## Findings outside my regex scope — detected via manual review

### Quantile-over-full-sample look-ahead in `bucket_high` / `bucket_low`

`research/comprehensive_deployed_lane_scan.py:275-288`:
```python
def bucket_high(vals: pd.Series, pct: float) -> np.ndarray:
    vv = vals.astype(float)
    thresh = np.nanpercentile(vv, pct)
    return (vv > thresh).fillna(False).astype(int).values
```

When called on a full cell's data (IS + OOS combined), this computes the quantile across BOTH windows and applies it to gate IS fires. IS fires therefore depend on the OOS distribution — a subtle look-ahead.

This is NOT a filter-delegation violation (it's used for feature-binning inside `_build_filters`, not filter application). But it IS a look-ahead-bias issue discovered during Phase 4 and should be documented as a separate fix class. Covered by Phase 4 (rel_vol IS-only sensitivity) and by the new historical-failure-log entry under "Quantile-over-full-sample" class.

### `compute_deployed_filter(df, "rel_vol_HIGH_Q3")` — not flagged but relevant

The scan's `compute_deployed_filter` does not have a `rel_vol_HIGH_Q3` branch. Instead, `rel_vol_HIGH_Q3` is a FEATURE built via `bucket_high` (above), and `compute_deployed_filter` applies the DEPLOYED LANE's filter (OVNRNG_100 etc.) to the rows the rel_vol feature operates on. The rel_vol feature quantile bug is the look-ahead concern; the compute_deployed_filter bug contaminates the deployed-filter population used for two-pass testing. Both are real and separate.

## Summary

- **1 genuine delegation offender** (`comprehensive_deployed_lane_scan.py`), fix in Phase 2.
- **2 false positives** (regex picked up unrelated patterns in `research_trend_day_mfe.py` and `garch_partner_state_provenance_audit.py`).
- **1 adjacent look-ahead class** (`bucket_high` on full-sample quantile) documented separately and addressed by Phase 4.

## Phase 2 scope (precise)

Edit `research/comprehensive_deployed_lane_scan.py::compute_deployed_filter` (lines ~226-267) to:

1. Import `research.filter_utils.filter_signal`.
2. Replace the function body with: `return np.asarray(filter_signal(df, filter_key, orb_label)).astype(int)`.
3. Handle `filter_key is None` (current behavior: return ones) — preserve.
4. `orb_label` parameter: currently the function doesn't take `orb_label`. `filter_signal` needs it for VWAP_MID_ALIGNED's per-session column lookup. Add `orb_label` parameter; all ~4 call sites in the file need updating.
5. Run canonical equivalence on ONE test cell to verify fire-mask matches per-filter for all 4 impacted filters (OVNRNG_100, VWAP_MID_ALIGNED, ORB_G5, ATR_P50).

## Downstream doc markers

Scan's result doc at `docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md` needs a top-of-doc WARN header: "Survivor cells using OVNRNG_100 / VWAP_MID_ALIGNED / ORB_G5 / ATR_P50 as the deployed filter are based on inline re-implementations flagged for correction on 2026-04-19. Rel_vol_HIGH_Q3 / bb_volume_ratio / break_delay survivor cells are not impacted (those use feature-building, not filter-delegation). The rel_vol BH-global survivor list is separately conditional on the quantile look-ahead noted in `docs/audit/results/2026-04-19-rel-vol-cross-scan-overlap-decomposition.md`."

## Audit trail

Commit-of-record for this audit: embedded in Phase 2 fix commit.
