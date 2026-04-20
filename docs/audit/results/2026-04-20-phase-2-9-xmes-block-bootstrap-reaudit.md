# Phase 2.9 X_MES_ATR60 block-bootstrap re-audit — CONFIRM verdict

**Date:** 2026-04-20
**Pre-reg:** [`docs/audit/hypotheses/2026-04-20-phase-2-9-xmes-block-bootstrap-reaudit.yaml`](../hypotheses/2026-04-20-phase-2-9-xmes-block-bootstrap-reaudit.yaml)
**Script:** `research/phase_2_9_xmes_block_bootstrap_reaudit.py`
**Output:** `research/output/phase_2_9_xmes_block_bootstrap_reaudit.json`
**Predecessor:** [`2026-04-20-phase-2-9-framing-audit.md`](2026-04-20-phase-2-9-framing-audit.md) (CONTINUE×4, commit `889c479c`, without block-bootstrap gate)
**Branch:** `research/phase-2-9-xmes-block-bootstrap-reaudit`

---

## TL;DR

**Verdict: `CONFIRM_PHASE_2_9`.**

Block-bootstrap null rejects the "filter-fire days are random" hypothesis on both X_MES_ATR60 primary cells. The 2026-04-20 framing audit's CONTINUE verdict for the 2 production X_MES_ATR60 lanes stands. **RR1.5 is borderline** (p_boot = 0.0464) and warrants a watch flag even though it formally confirms. Reference OVNRNG_100 cells pass cleanly, confirming the null is calibrated.

This closes Target 1 from the 2026-04-20 vol-regime-confluence resume doc.

---

## Per-cell results

| Priority | Cell | N_base | N_fire | fire_rate | observed_ExpR | t_IS | null_mean | null_p95 | **p_boot** | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| primary | MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60 | 1650 | 704 | 0.427 | +0.1481 | +4.35 | +0.0656 | +0.1302 | **0.0192** | CONFIRM |
| primary | MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60 | 1636 | 695 | 0.425 | +0.1507 | +3.42 | +0.0709 | +0.1489 | **0.0464** | CONFIRM (borderline) |
| reference | MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100 | 1650 | 527 | 0.319 | +0.1731 | +4.37 | +0.0653 | +0.1365 | **0.0070** | REFERENCE_PASS |
| reference | MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | 1636 | 520 | 0.318 | +0.2064 | +4.01 | +0.0714 | +0.1579 | **0.0057** | REFERENCE_PASS |

All 4 cells: IS window trading_day < 2026-01-01, block size = 40, n_perms = 10,000, seed = 20260420.

---

## Why "borderline" matters for RR1.5

The RR1.5 cell is just under the 0.05 threshold (0.0464) with observed_ExpR only slightly above null_p95 (+0.1507 vs +0.1489). Under the pre-reg decision tree it formally confirms, but it is **seed-sensitive** and **n_perms-sensitive** at this resolution. A different deterministic seed or n_perms=5,000 could cross the threshold in either direction by ~0.005–0.010.

Operational implication: treat RR1.5 as **CONFIRM_WATCH** in the live book, not as a hard pass. Pointer: if the 2026-07-15 150-day OOS review shows RR1.5 OOS sign-flipping or failing C8 (OOS/IS ≥ 0.40), retire; otherwise hold. RR1.0 is comfortable (p=0.019 with observed well above null_p95).

---

## Why this differs from the vol-regime sprint's Stage H finding

The vol-regime sprint's Stage H (commit `096f5ec1`) found **all X_MES_ATR60 *confluence* variants fail block-bootstrap null** (p_boot 0.07–0.18 across 4 MNQ confluence cells). The two framing-audit lanes tested here are **NOT confluence lanes** — X_MES_ATR60 is the base filter, not a variant stacked on top of ORB_G5. Specifically:

- vol-regime Stage H null: `base = days where ORB_G5 fires`, `variant = X_MES_ATR60`. Tests whether X_MES_ATR60 adds info **beyond ORB_G5**. Answer: no.
- this re-audit null: `base = all E2 entry-firing days`, `variant = X_MES_ATR60`. Tests whether X_MES_ATR60 adds info **beyond random day selection**. Answer: yes.

Both results are correct and non-contradictory. The filter is itself edge-carrying at the population level, but does not stack with ORB_G5 to add further edge. The framing-audit CONTINUE verdict concerns the standalone-filter framing, so the re-audit applies to that framing and confirms it.

---

## Method

- **Null:** `research.vol_regime_gates_g_h_i_j.moving_block_bootstrap` (same function used for the CONFIRMED 2 OVN confluence variants in the vol-regime sprint). Block-resamples daily `pnl_r` while keeping the canonical filter fire-pattern fixed; counts how often the mean on fire-days in the resampled series meets or beats the observed mean.
- **Filter delegation:** `research.filter_utils.filter_signal(df, filter_key, orb_label)` → `trading_app.config.ALL_FILTERS[filter_key].matches_df(df, orb_label)`. Canonical, fail-closed on NaN. No ad-hoc thresholds.
- **Data load:** local `load_lane_canonical()` in the re-audit script emits the exact column names the registered filters read (`overnight_range`, `cross_atr_MES_pct`) at SQL time. The vol-regime sprint's `load_lane` aliases MES ATR as `mes_atr_20_pct` for its inline `variant_mask` callers; that signature doesn't satisfy the canonical filter contract. Rather than mutate the parent-branch file or alias downstream, this re-audit loads with the canonical column names directly. See the `refactor` commit message on this branch for the full rationale (institutional-rigor.md § 4 and § 8). `moving_block_bootstrap` is still imported from the sprint file (shared canonical null).
- **Data equivalence:** `daily_features.atr_20_pct` for `symbol='MES'` is identical across `orb_minutes ∈ {5, 15, 30}` within the IS window (`trading_day < 2026-01-01`). The 3 trading days with 0.08–0.12pp spreads across orb_minutes (2026-04-09/10/12) are all OOS and excluded by the IS mask. The canonical-contract load and an alternative alias-based load produce numerically identical results to 4dp (verified pre/post refactor: XMES RR1.0 p_boot=0.0192, RR1.5 p_boot=0.0464, OVN RR1.0 p_boot=0.0070, OVN RR1.5 p_boot=0.0057, all unchanged).
- **n_perms:** 10,000 (upgraded from the vol-regime sprint's 5,000 for tighter p_boot resolution near 0.05, given the RR1.5 borderline risk).
- **Seed:** 20260420 (deterministic).
- **Window:** IS only, `trading_day < HOLDOUT_SACRED_FROM` (2026-01-01, per `trading_app.holdout_policy`).

---

## Decision tree (pre-registered, locked before any p_boot was seen)

From pre-reg YAML:
1. ALL primary p_boot<0.05 AND ALL reference p_boot<0.05 → **CONFIRM_PHASE_2_9**   ← this outcome
2. ANY primary p_boot≥0.05 → DOWNGRADE_XMES_LANES (enumerate)
3. ANY primary p_boot≥0.10 → RETIRE_XMES_CANDIDATE
4. ANY reference p_boot≥0.05 → FLAG_NULL_CALIBRATION

Observed: 4/4 below 0.05 → branch (1) selected.

---

## Operational actions

1. **No change to 2 production X_MES_ATR60 lanes** — they remain research-provisional per phase-2.9 promotion on 2026-04-11.
2. **Watch flag on RR1.5** — add to next OOS review checkpoint (2026-07-15, ≥150 OOS days): if RR1.5 sign-flips or fails C8 ≥ 0.40, retire.
3. **No downgrade, no retirement.** Framing audit's CONTINUE×4 verdict is upheld on the two X_MES_ATR60 lanes. The two OVNRNG_100 lanes were already confirmed via the vol-regime sprint's confluence framing — this reference check confirms at the standalone-filter framing too.
4. **Close Target 1** from the vol-regime sprint resume doc. Targets 2 (re-null the 13 BH-global 2026-04-15 survivors) and 3 (pre-reg prose/values auto-check) remain open.

---

## Reproduce

```bash
cd C:/Users/joshd/canompx3-vol-regime
git checkout research/phase-2-9-xmes-block-bootstrap-reaudit
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db PYTHONPATH=. python research/phase_2_9_xmes_block_bootstrap_reaudit.py
```

Output JSON at `research/output/phase_2_9_xmes_block_bootstrap_reaudit.json`.

---

## Institutional provenance

- Pre-reg locked before execution (commit `32a9cfd8`).
- Script initially committed before execution (commit `19a29955`). First run returned N_fire=0 on both X_MES_ATR60 cells — caught as infrastructure gap (column-name mismatch between `load_lane`'s alias and the canonical filter contract), not a data finding.
- First attempted fix was a downstream alias (commit `698542f9`) that made the run produce correct results but left the canonical-contract violation in place.
- Proper fix committed as refactor (this branch's current HEAD): the re-audit script now loads with canonical column names at SQL time (`cross_atr_MES_pct`, `overnight_range`) via a local `load_lane_canonical` function, eliminating any downstream aliasing. Results are 4dp-identical to the alias-based run; verdict unchanged.
- MinBTL 2.77y < 6.65y available (K=4, E_floor=1.0).
- Read-only DuckDB connection; no writes to `validated_setups`, `experimental_strategies`, or any canonical table.
- Filter delegation canonical (`research.filter_utils.filter_signal` → `ALL_FILTERS[key].matches_df`). No re-encoded filter logic in the re-audit script.
- Block-bootstrap null methodology matches the 2026-04-15 historical failure log standard (variant mask fixed, pnl_r block-resampled).
- `pipeline/check_drift.py` run and passing on the re-audit branch after refactor; no drift violations introduced.

## Honest self-review (pre-commit)

1. **Is `load_lane_canonical` a hidden re-encoding of canonical logic?** No. It is a data-load query with canonical column naming. It does NOT compute any filter signal — it loads `overnight_range` and `atr_20_pct WHERE symbol='MES'` and presents them under the canonical names. Filter semantics remain in `ALL_FILTERS[*].matches_df`.
2. **Could the canonical-contract load silently differ from the canonical cross-asset enrichment pipeline?** Possible in principle if the canonical enrichment joined across `orb_minutes` differently. Verified empirically: `atr_20_pct` for `symbol='MES'` is identical across `orb_minutes ∈ {5,15,30}` within IS; OOS-only drift (3 days) excluded by the IS mask.
3. **Does the pre-reg discipline survive?** The pre-reg locked the null spec, the filter keys, the decision tree, and the K budget before any result. The infrastructure fix + refactor did not change any of those — only the mechanism by which the canonical filter receives its input data. p_boot values are 4dp-identical pre- and post-refactor, which is the strongest possible evidence that the fix did not touch the null hypothesis being tested.
4. **Borderline RR1.5 (p=0.0464) — am I declaring CONFIRM when I shouldn't?** Per the locked decision tree, 0.0464 < 0.05 ⇒ CONFIRM. The borderline nature is flagged in the result doc and an operational watch-flag is recommended. Honest reporting, not retroactive gate relaxation.
5. **Is any result being laundered through a result-doc edit?** No. The pre-reg YAML is unchanged after the first run. The script's infrastructure evolved (alias → refactor) but produces identical outputs. The result doc amended only the method-section wording to describe the final loader approach; the per-cell numbers and verdict were written from the refactored-run output.
