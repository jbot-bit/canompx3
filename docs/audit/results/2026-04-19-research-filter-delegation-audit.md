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

---

## Addendum — 2026-04-19 broader-regex sweep (campaign Phase 13)

**Motivation:** the original audit's grep patterns were narrow (`overnight_range / atr_20`, `def compute_deployed_filter`, `vwap_signal`, `deployed_filter_signal`). A broader sweep using `if filter_type == "<CANONICAL>":` + hardcoded SQL fragment patterns (`d.atr_20_pct >=`, `d.overnight_range >=`, `"d\..* >= \d"`) found **3 additional genuine offenders** the original sweep missed, plus 1 older file.

### Broader sweep patterns used

- `grep -rn "if filter_type ==" research/ --include="*.py"` — captures per-filter inline branches
- `grep -rn "d\.atr_20_pct >=\|d\.overnight_range >=\|d\.orb_.*_size" research/ --include="*.py"` — captures inline SQL gate literals
- `grep -rn 'filter_sql.*:\|AND d\.' research/ --include="*.py"` — captures dict-of-SQL-strings patterns

### Additional genuine offenders

#### `research/garch_broad_exact_role_exhaustion.py` (last modified 2026-04-16, commit 7aaf256c) — **GENUINE OFFENDER**

Lines 73-118 re-implement canonical filter logic inline as Python helper returning SQL strings. Hits 6 canonical filter families: `ATR_P*`, `OVNRNG_*`, `X_MES_ATR60`, `VWAP_MID_ALIGNED`, `VWAP_BP_ALIGNED`. Each is a re-implementation of `ALL_FILTERS[key].matches_row`. Output is raw SQL (design-wise different from `compute_deployed_filter`), but semantically equivalent drift hazard: if inline SQL differs from canonical by 1 edge-case, every downstream result is contaminated.

**Verdict:** FIX. Restructure to load outcomes+features via `strategy_fitness._load_strategy_outcomes` which already handles all filter delegation canonically, OR load daily_features to DataFrame, apply `filter_signal(df, filter_type, orb_label)`, materialize fire-day set, inject via temp table.

#### `research/garch_validated_role_exhaustion.py` (last modified 2026-04-16, commit 7aaf256c) — **GENUINE OFFENDER**

Lines 73-97+ similarly re-implement filter SQL branches inline. Hits 8 canonical filter families: `ORB_G5`, `COST_LT12`, `OVNRNG_100`, `ATR_P50`, `ATR_P70`, `VWAP_MID_ALIGNED`, `VWAP_BP_ALIGNED`, `X_MES_ATR60`. The `OVNRNG_100` entry is `overnight_range >= 100.0` (correct absolute by luck of author choice, not by delegation). Still fragile.

**Verdict:** FIX. Same approach.

#### `research/garch_comex_settle_institutional_battery.py` (last modified 2026-04-15, commit 553c8089 "research: garch COMEX_SETTLE — institutional trader discipline, RULE 4 K_lane PASS") — **GENUINE OFFENDER**

Lines 92-99 hardcode filter SQL as tuple data embedded in a list of candidates:
```python
("MNQ", "SINGAPORE_OPEN", 30, 1.5, "long", "d.atr_20_pct >= 50.0"),
...
("MNQ", "COMEX_SETTLE", 5, 1.5, "long", "d.overnight_range >= 100.0"),
```

The commit message claims "institutional trader discipline" while violating the filter-delegation rule — ironic.

**Verdict:** FIX OR DEPRECATE. If one-shot and results already captured, add header note + retire. If reusable, refactor to delegation.

#### `research/break_delay_filtered.py` (last modified 2026-03-30, commit 3edde387 "research: break delay TRIPLY DEAD") — **GENUINE OFFENDER (LOW PRIORITY)**

Lines 26-47 hardcode filter SQL like `atr_20_pct >= 60`, `rel_vol_COMEX_SETTLE >= 1.2`. Pre-2026-04-18 (before delegation rule landed). Commit message confirms "TRIPLY DEAD — zero signal". Mis-implemented filter can only inflate survival rates; DEAD verdict is resilient to filter drift.

**Verdict:** LOW-PRIORITY. Accept "negative-verdict resilient" argument OR add deprecation header.

### Classification summary (updated after addendum)

| File | Status | Priority | Action |
|---|---|---|---|
| `research/comprehensive_deployed_lane_scan.py` | GENUINE OFFENDER | HIGH | FIX (original audit) |
| `research/garch_broad_exact_role_exhaustion.py` | GENUINE OFFENDER | HIGH — 6 filters | FIX |
| `research/garch_validated_role_exhaustion.py` | GENUINE OFFENDER | HIGH — 8 filters | FIX |
| `research/garch_comex_settle_institutional_battery.py` | GENUINE OFFENDER | MEDIUM — likely one-shot | FIX or DEPRECATE |
| `research/break_delay_filtered.py` | GENUINE OFFENDER | LOW — negative verdict resilient | DEPRECATE |
| `research/garch_partner_state_provenance_audit.py` | false positive | — | no-op |
| `research/research_trend_day_mfe.py` | false positive | — | no-op |

### Downstream contamination check

Each of the 3 NEW high/medium offenders produced outputs during their last run. Their output docs must carry a WARN header indicating potential inline-filter contamination until the fix lands. Specific output-doc enumeration is a follow-up task, not done in this addendum.

### Methodology lesson

The original sweep's narrow regex missed the `if filter_type == "<name>":` pattern because it was looking for function declarations or specific bug patterns. Future filter-delegation sweeps must include:
- Per-branch inline pattern: `grep -rn "if filter_type ==" research/`
- SQL-literal pattern: `grep -rn "d\.atr_20_pct >=\|d\.overnight_range >=" research/`
- Dict-of-SQL pattern: `grep -rn '".*AND d\.' research/`

### X_MES_ATR60 sub-finding (relates to v3.2 amendment I-4)

The I-4 correction added `DATA_PIPELINE_GAP` sub-classification for lanes firing at 0% due to missing pipeline columns. However, canonical `trading_app/strategy_fitness.py::_load_strategy_outcomes` lines 430-431 DO enrich `CrossAssetATRFilter` at runtime:
```python
if isinstance(filt, CrossAssetATRFilter):
    _enrich_cross_asset_atr(con, feat_dicts, filt.source_instrument)
```

So X_MES_ATR60 fires correctly when measured via the canonical `_load_strategy_outcomes` path (used by SR monitor, fitness tracker, live bot). The fire-rate audit's 0% finding for the 5 X_MES_ATR60 lanes was likely measured via direct `SELECT * FROM daily_features` without runtime enrichment — an **audit methodology artifact**, not a live pipeline gap.

**Implication for amendment I-4:** the `DATA_PIPELINE_GAP` sub-label should be applied ONLY when fire-rate-via-canonical-`_load_strategy_outcomes` is 0, NOT when direct SQL against `daily_features` is 0. The amendment should clarify that fire-rate measurement MUST use the canonical runtime-enriched path, not naive SQL.

**Action:** either tighten I-4 wording (campaign Phase follow-up), OR verify the `fire_rate_audit.md` was done via canonical path. If via canonical → 0% is real pipeline gap. If via naive SQL → re-measure required.

### Follow-up tasks

1. Apply fixes to 3 HIGH/MEDIUM-priority offenders (garch_broad_exact_role_exhaustion, garch_validated_role_exhaustion, garch_comex_settle_institutional_battery) OR mark deprecated if single-use. Each requires design-proposal-gate per `.claude/rules/institutional-rigor.md`.
2. Enumerate contaminated output docs across `docs/audit/results/2026-04-1[5-6]-garch-*.md` and add WARN headers.
3. Verify `fire_rate_audit.md` measurement path (canonical vs naive SQL) and reconcile I-4 sub-classification accordingly.
4. Add broader-regex methodology pattern to future filter-delegation sweep SOPs.

**End of broader-regex addendum.**

---

## Addendum 2 — 2026-04-19 hardcoded-commission sweep (campaign follow-up)

Triggered by real finding in PR #14 (File 2 filter-delegation fix) — inline SQL `(2.74 / risk_dollars) < 0.12` violated the canonical cost model in two ways:

1. **Stale commission constant.** `2.74` = pre-Rithmic MNQ round-trip commission. Current canonical `pipeline.cost_model.COST_SPECS['MNQ'].commission_rt = $1.42`.
2. **Missing slippage component.** Canonical `CostRatioFilter.matches_row` uses `100 × total_friction / (raw_risk + total_friction)` — TOTAL friction (commission + slippage). Inline used commission alone.

### Broader-sweep results — additional offenders

Grep pattern: `"2\.74\|2\.44\|commission.*= *[0-9]\."` against `research/**/*.py`:

| File | Line | Violation | Priority | Proposed fix |
|---|---|---|---|---|
| `research/garch_validated_scope_honest_test.py` | 55-56 | Identical `(2.74 / NULLIF(o.risk_dollars, 0)) < 0.12` COST_LT12 inline to PR #14's target | **HIGH** (same bug class, active research file) | Delegate via `filter_signal(df, 'COST_LT12', orb_label)` — same pattern as PR #14 |
| `research/research_prop_firm_fit.py` | 57 | Hardcoded dict `{"MNQ": 2.74, ...}` with comment `# 1.24 + 0.50 + 1.00` (pre-Rithmic breakdown) | MED (prop-firm fit calculator — may be single-use or active) | Source from `pipeline.cost_model.COST_SPECS[sym].commission_rt + slippage`; do NOT replicate the constant |
| `research/tmp_dd_anatomy.py` | 23 | `FRICTION = {"MGC": 5.74, "MNQ": 2.74, "MES": 3.74, "M2K": 3.24}` — pre-Rithmic friction dict across 4 instruments | LOW (`tmp_` prefix = scratch/temp) | Deprecate or fix: either delete if unused, or replace with `COST_SPECS[sym].total_friction` per instrument |
| `research/tmp_dd_budget_configs.py` | 23 | Same `FRICTION` dict as `tmp_dd_anatomy.py` | LOW (`tmp_` prefix = scratch) | Same treatment |

### Canonical source for commission / friction

```python
from pipeline.cost_model import COST_SPECS
mnq_commission_rt = COST_SPECS['MNQ'].commission_rt  # 1.42 (Rithmic, current)
mnq_total_friction = COST_SPECS['MNQ'].total_friction  # commission_rt + slippage
mnq_point_value = COST_SPECS['MNQ'].point_value
```

**Any research file hardcoding these values without the above source is drift-prone.** Acceptable exception: a historical/archive script explicitly dated and scoped to a bygone commission regime, with a top-of-file comment declaring the frozen value.

### Downstream-contamination assessment

Docs potentially affected (outputs produced by these 4 scripts) must be inspected for WARN headers:

- `garch_validated_scope_honest_test.py` → produces docs matching `docs/audit/results/2026-04-*-garch-validated-scope-honest*.md` — any cells using COST_LT12 may over-admit trades that fail the canonical cost screen. Magnitude: small (~0.3% per PR #14 empirical data).
- `research_prop_firm_fit.py` → may produce capital-allocation / prop-firm projections. Using `2.74` instead of `1.42` overstates per-trade cost by ~2× on MNQ. **Material for scaling-decision docs.**
- `tmp_dd_anatomy.py`, `tmp_dd_budget_configs.py` → `tmp_` scratch scripts — likely ad-hoc, outputs (if any) should be verified / retired.

### Plan for follow-up fixes

1. **Next filter-delegation PR:** apply the PR #14 pattern to `garch_validated_scope_honest_test.py`. Parity test expected to surface the SAME 2-row COST_LT12 divergence class — this is a feature of the fix, not a defect.
2. **`research_prop_firm_fit.py`:** design-proposal-gate — before fixing, verify whether the file is still consumed (output docs, CI, other scripts). If yes, fix. If no, deprecate with a top-of-file note.
3. **`tmp_*` scratch scripts:** propose deletion. Their outputs (if any) are already captured elsewhere; `tmp_` prefix is the project convention for "not canonical." If any output doc references them as source-of-truth, migrate + retire.

### Historical-failure-log entry (backtesting-methodology.md RULE 11)

Entry to append:

> **2026-04-19: Commission-value drift (pre-Rithmic hardcoded `2.74` in research scripts).** Inline commission filter SQL in multiple research files hardcoded the pre-Rithmic MNQ commission (`2.74`) instead of sourcing from `pipeline.cost_model.COST_SPECS`. After Rithmic integration (Q1 2026), canonical commission dropped to `1.42` but these scripts retained the stale value. Compounded by a structural error — inline formula `(commission / risk_dollars)` ignored slippage, while canonical `CostRatioFilter` uses `total_friction / (raw_risk + total_friction)`. Net effect: stale inline admitted trades the canonical screen correctly rejects. Caught by PR #14 parity test (2 divergent rows on MNQ NYSE_OPEN RR1.5 long IS; fix commit 9baa3b50). **Lesson:** any hardcoded commission/friction constant in research/ is drift bait. Source from COST_SPECS, or declare the frozen-historical value at top-of-file with date scope.

**End of addendum 2.**
