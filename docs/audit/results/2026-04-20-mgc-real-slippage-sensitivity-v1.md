# H0 — MGC Real-Slippage Sensitivity v1 — **HALTED**

**Date:** 2026-04-20
**Status:** HALTED (baseline cross-check failed)
**Pre-reg:** `docs/audit/hypotheses/2026-04-20-mgc-real-slippage-sensitivity.yaml`
**Script:** `research/research_mgc_real_slippage_sensitivity_v1.py`
**Output:** `research/output/mgc_real_slippage_sensitivity_v1.json`
**Parent audit:** `docs/audit/results/2026-04-20-mgc-adversarial-reexamination.md` §9 H0

## Verdict: HALT

The pre-reg's §baseline_cross_check required cells recomputed at
modeled slippage (2 ticks) to match the `research_mgc_native_low_r_v1.py`
reported ExpR values within ±0.005R. **All 5 cells mismatched by
0.10 – 0.27R.** Per institutional-rigor rule #8 (verify before claiming)
and the pre-reg's own halt condition, no conclusion about slippage
sensitivity can be drawn from this run.

```
Cross-check at slippage=2 ticks (modeled):
  NYSE_OPEN_OVNRNG_50_RR1_LR075: reported +0.2226, recomputed -0.0474, |diff|=0.2700
  US_DATA_1000_ATR_P70_RR1_LR05: reported +0.0710, recomputed -0.0528, |diff|=0.1238
  US_DATA_1000_OVNRNG_10_RR1_LR05: reported +0.0685, recomputed -0.0528, |diff|=0.1213
  US_DATA_1000_BROAD_RR1_LR05:   reported +0.0488, recomputed -0.0528, |diff|=0.1016
  NYSE_OPEN_BROAD_RR1_LR05:      reported +0.0380, recomputed -0.0748, |diff|=0.1128
```

Additional anomalies:
- `US_DATA_1000_OVNRNG_10_RR1_LR05` and `US_DATA_1000_BROAD_RR1_LR05`
  recomputed to IDENTICAL ExpR (-0.0528 at slippage=2). BROAD must have
  more trades than OVNRNG_10 by construction; identical ExpR across
  two different filter scopes is a strong signal of filter-mapping bug.
- N per cell differs from reported (e.g., ATR_P70 reported N=414, I got
  a different count).

## Likely root causes (prioritized for next-session investigation)

1. **Filter threshold definitions are wrong.** My script maps:
   - `OVNRNG_50` → `daily_features.overnight_range >= 50.0`
   - `OVNRNG_10` → `daily_features.overnight_range >= 10.0`
   - `ATR_P70`   → `daily_features.atr_20_pct >= 0.70`
   These may not match canonical `trading_app.config.ALL_FILTERS`. The
   canonical filter must be used via `research.filter_utils.filter_signal`
   per `.claude/rules/research-truth-protocol.md` § Canonical filter
   delegation — I re-encoded instead, which is exactly the failure
   class the 2026-04-19 filter-delegation audit documented.

2. **Low-R target rewrite logic differs.** `research_mgc_native_low_r_v1.py`
   applies the LR target ONLY if mfe_r exceeds the target (via mfe column
   from stored `orb_outcomes.mfe_r` at modeled friction). My reconstruction
   via `mfe_points = stored_mfe_r × risk_modeled / point_value` may not
   recover the same target-hit logic.

3. **MGC direction filter.** I inferred `long = entry > stop`, but the
   stored orb_outcomes rows may include multiple directions per trading
   day; my direction filter may be pulling shorts unintentionally.

4. **Row count discrepancy.** Reported `ATR_P70` N=414; my script returned
   a different count. Could be:
   - Different date range (I used 2022-06-13 → 2026-01-01; source may
     have used 2023-09-11 → 2026-01-01, the prior `minimum_start_date`
     per `pipeline/asset_configs.py:78-81` before the 2022-06-13
     expansion)
   - Different holdout semantics

## What this HALT preserves

- **Zero false audit conclusions.** The pre-reg's baseline cross-check
  worked as designed — caught the mismatch BEFORE anyone acted on the
  sensitivity grid.
- **The closure conclusion stands unchanged** — this test did not
  confirm it under realistic slippage, but also did not refute it.
- **Trial budget intact**. Per the pre-reg, this audit costs 0 trials
  from MGC's MinBTL budget (confirmatory audit is not discovery). The
  halt costs nothing on the statistical side.

## What needs to change before H0 can be retried

1. **Delegate filter logic** to `research.filter_utils.filter_signal(df, filter_key, orb_label)`
   per `research-truth-protocol.md` § Canonical filter delegation. No
   inline threshold decoding.
2. **Delegate LR rewrite** to whatever helper `research_mgc_native_low_r_v1.py`
   actually uses, OR inline-reproduce it with line-level citation from
   that file.
3. **Run baseline cross-check first**; only proceed to the sensitivity
   grid if all 5 cells match within ±0.005R.
4. **Confirm date window.** Query `research_mgc_native_low_r_v1.py`
   directly to see which IS window it uses, and match exactly.

## Honesty over completion

This result is a HALT, not a failure. The cost-realism sensitivity
question remains formally open. Per institutional-rigor rule #2
("after any fix, review the fix") the next attempt must:
- Read `research_mgc_native_low_r_v1.py` line-by-line before writing
  the replay script
- Use `filter_utils.filter_signal` directly
- Pass baseline cross-check before any slippage-grid evaluation

Tracked in task queue as follow-up. Not in this session's shipping
scope.

## Output artifacts

- Script: `research/research_mgc_real_slippage_sensitivity_v1.py`
  (retained for post-mortem; DO NOT use its current outputs as evidence)
- JSON: `research/output/mgc_real_slippage_sensitivity_v1.json`
  (flagged as pre-cross-check, do not cite)
- This result doc: lock the status as HALT until cross-check passes
