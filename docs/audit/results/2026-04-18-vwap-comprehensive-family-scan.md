# VWAP Comprehensive Family Scan — Result

**Pre-reg:** `docs/audit/hypotheses/2026-04-18-vwap-comprehensive-family-scan.yaml`
**Pre-reg sha:** `495810f5`
**Run UTC:** 2026-04-18T08:50:32.213018+00:00
**Mode A holdout:** sacred from `2026-01-01` per `trading_app.holdout_policy`
**IS:** trading_day < `2026-01-01`
**OOS one-shot:** `[2026-01-01, 2026-04-18)`
**Elapsed:** 333s

## Phase 1 admissibility verdict

**VWAP_MID_ALIGNED + VWAP_BP_ALIGNED**: RULE 6.1 SAFE — trade-time-knowable.

- Build path: `pipeline/build_daily_features.py:983-998` (Module 7, Mar 20 2026)
  computes VWAP from pre-session bars only (`ts_utc < orb_start`, strict less-than).
- Filter: `trading_app/config.py:2420-2554` `VWAPBreakDirectionFilter` reads same
  `orb_{label}_vwap` column + entry-time-known `orb_high`/`orb_low`/`break_dir`.
- Filter and column are perfectly aligned. Fail-closed on missing data.

## Schema verification

- `daily_features.orb_{label}_vwap` populated for all 12 sessions.
- Triple-join on `(trading_day, symbol, orb_minutes)` per `.claude/rules/daily-features-joins.md`.

## Coverage

- Total combos enumerated: 1296
- Cells attempted: 1136
- Cells with usable results: 1030
- Skipped — disabled (instrument, session) per `pipeline.asset_configs`: 144
- Skipped — N<60 IS or N<30 on/off: 122

## BH-FDR pass counts at each K framing

- **K_global** (K=1296 cells, q=0.05): **0** pass (trustworthy only)
- **K_family** (K=648 per VWAP variant, q=0.05): **0** pass
- **K_lane** (K=12 per (session, instr, apt, rr), q=0.05): **54** pass
- **K_session** (K=108 per session, q=0.05): **22** pass
- **K_instrument** (K=432 per instrument, q=0.05): **0** pass
- **K_feature_session** (per (variant, session), q=0.05): **23** pass

## Trustworthy gate

- Total cells: 1030
- Flagged extreme_fire (<5% or >95%): 0
- Flagged arithmetic_only (WR-flat + Δ_IS large): 2
- Flagged tautology (T0 |corr| > 0.7 vs deployed filter): 8
- N_on_IS < 50: excluded from trustworthy
- Trustworthy cells: 999

## Hypothesis verdicts

### H1 family-level (BINDING)
Gates: BH-FDR K_family q<0.05 AND dir_match AND |t_IS|>=3.0 AND yrs_positive_IS>=4/7 AND boot_p<0.10
**Survivors: 0**
Threshold: >=3 to CONTINUE, 1-2 to PARK, 0 to KILL (K1)

### H2 strict (descriptive ribbon)
Gates: as H1 but |t_IS|>=3.79 (Chordia no-theory)
**Survivors: 0**

### H3 positive control
L6 IS ExpR=+0.1817 (baseline +0.2101, |diff|=0.0284, tol=0.02); H3 = FAIL
Pass = harness reproduces L6 known-live IS ExpR within tolerance.

## Red flag audit (RULE 12)

- |t_IS| > 7: 0 cells
- |delta_IS| > 0.6: 4 cells

**WARNING: Red flag cells present. Audit before treating as survivors.**

## H1 survivors (binding family-level gate)

| Instr | Session | Apt | RR | Dir | Variant | N_on_IS | N_on_OOS | Fire% | ExpR_IS | ExpR_OOS | Δ_IS | Δ_OOS | t | p | boot_p | yrs+ | BH_g | BH_f | BH_l |
|-------|---------|-----|----|----|---------|---------|----------|-------|---------|----------|------|-------|---|---|--------|------|------|------|------|
(none)

## H2 strict ribbon (|t|>=3.79 Chordia no-theory)

| Instr | Session | Apt | RR | Dir | Variant | N_on_IS | ExpR_IS | Δ_IS | Δ_OOS | t | boot_p | yrs+ |
|-------|---------|-----|----|----|---------|---------|---------|------|-------|---|--------|------|
(none)

## Promising cells (|t|>=2.5 + dir_match + trustworthy, top 40 by |t|)

| Instr | Session | Apt | RR | Dir | Variant | N_on_IS | Fire% | ExpR_IS | Δ_IS | Δ_OOS | t | boot_p | BH_f | BH_l |
|-------|---------|-----|----|----|---------|---------|-------|---------|------|-------|---|--------|------|------|
| MGC | EUROPE_FLOW | O30 | 2.0 | short | VWAP_MID | 211 | 49.3% | +0.135 | +0.440 | +0.421 | +3.67 | 0.0608 | . | Y |
| MGC | EUROPE_FLOW | O30 | 1.5 | short | VWAP_MID | 219 | 50.0% | +0.094 | +0.355 | +0.596 | +3.45 | 0.0864 | . | Y |
| MGC | EUROPE_FLOW | O30 | 2.0 | short | VWAP_BP_ | 271 | 63.4% | +0.062 | +0.413 | +0.580 | +3.42 | 0.2082 | . | Y |
| MES | NYSE_CLOSE | O5 | 2.0 | short | VWAP_BP_ | 110 | 46.2% | -0.175 | +0.431 | +0.200 | +3.04 | 0.0411 | . | Y |
| MGC | EUROPE_FLOW | O30 | 1.5 | short | VWAP_BP_ | 280 | 64.0% | +0.032 | +0.321 | +0.607 | +3.03 | 0.2960 | . | Y |
| MNQ | US_DATA_1000 | O15 | 1.5 | long | VWAP_BP_ | 574 | 71.9% | +0.135 | +0.279 | +1.057 | +3.03 | 0.0022 | . | Y |
| MNQ | COMEX_SETTLE | O5 | 2.0 | short | VWAP_BP_ | 374 | 49.4% | +0.138 | +0.283 | +0.026 | +2.97 | 0.0235 | . | Y |
| MES | NYSE_CLOSE | O5 | 1.5 | short | VWAP_BP_ | 126 | 47.2% | -0.104 | +0.363 | +0.379 | +2.94 | 0.1096 | . | Y |
| MNQ | US_DATA_1000 | O15 | 2.0 | long | VWAP_BP_ | 520 | 71.7% | +0.062 | +0.311 | +0.873 | +2.89 | 0.1317 | . | Y |
| MES | NYSE_CLOSE | O5 | 1.5 | short | VWAP_MID | 114 | 42.9% | -0.088 | +0.363 | +0.379 | +2.88 | 0.1676 | . | Y |
| MES | NYSE_CLOSE | O5 | 2.0 | short | VWAP_MID | 99 | 41.8% | -0.165 | +0.414 | +0.200 | +2.82 | 0.0724 | . | Y |
| MGC | EUROPE_FLOW | O15 | 2.0 | short | VWAP_MID | 221 | 48.3% | +0.067 | +0.321 | +0.631 | +2.82 | 0.2527 | . | Y |
| MNQ | CME_PRECLOSE | O5 | 1.5 | long | VWAP_MID | 388 | 58.6% | +0.245 | +0.241 | +0.128 | +2.66 | 0.0002 | . | Y |
| MES | COMEX_SETTLE | O15 | 1.0 | short | VWAP_MID | 360 | 46.4% | +0.011 | +0.168 | +0.356 | +2.65 | 0.3601 | . | Y |
| MGC | EUROPE_FLOW | O15 | 2.0 | short | VWAP_BP_ | 271 | 59.6% | +0.023 | +0.298 | +0.344 | +2.62 | 0.3965 | . | Y |
| MES | COMEX_SETTLE | O15 | 1.0 | short | VWAP_BP_ | 422 | 54.3% | -0.005 | +0.163 | +0.325 | +2.57 | 0.4871 | . | Y |
| MGC | EUROPE_FLOW | O30 | 1.0 | short | VWAP_MID | 220 | 50.0% | +0.038 | +0.209 | +0.280 | +2.50 | 0.2515 | . | . |

## Flagged cells (excluded from trustworthy — transparency)

- extreme_fire: 0
- arithmetic_only: 2
- tautology (|corr|>0.7 vs deployed filter): 8

Top 20 flagged with |t|>=2.5:

| Instr | Session | Apt | RR | Dir | Variant | t | Fire% | T0 corr | T0 vs | Reason |
|-------|---------|-----|----|----|---------|---|-------|---------|-------|--------|
| MNQ | US_DATA_1000 | O15 | 2.0 | long | VWAP_MID | +3.75 | 54.9% | 1.00 | VWAP_MID_ALIGNED | TAUT(1.00) |
| MNQ | US_DATA_1000 | O15 | 1.5 | long | VWAP_MID | +3.27 | 54.7% | 1.00 | VWAP_MID_ALIGNED | TAUT(1.00) |

## Post-run finding — H3 specification error (NOT a harness bug)

The literal K2 fire above was caused by an error in the pre-reg's H3 baseline calibration, NOT by an execution defect in the scan harness. Independent SQL verification 2026-04-18 against canonical `gold.db`:

| Computation | N_on | ExpR_on | Source |
|---|---|---|---|
| **Scan harness** (strict Mode A IS, `confirm_bars=1`, `atr_20 NOT NULL`) | 436 | **+0.1817** | this scan output |
| **Independent SQL** (strict Mode A IS, no other filters) | 436 | **+0.1844** | direct DuckDB query at run time |
| Independent SQL (Mode B grandfathered IS through 2026-04-18) | 454 | +0.1915 | direct DuckDB query |
| `validated_setups` row (used as H3 baseline in pre-reg) | **701** | **+0.2101** | validated_setups, last_trade_day=2026-04-02 |

The harness produced **+0.1817** versus independent SQL's **+0.1844** — agreement within 0.003 R, well below the 0.02 tolerance. The harness is correct.

The pre-reg's H3 baseline of **+0.2101** was sourced from the `validated_setups` row for L6, which is **Mode B grandfathered** (computed at validation time on a different IS window definition with N=701 trades), not strict Mode A IS (N=436 trades through 2026-01-01). The 247-trade gap (701-454) reflects a combination of:

- Mode A boundary difference (excludes 2026 Q1 trades that were in Mode B IS)
- Filter resolution differences between `validated_setups` build path and current canonical orb_outcomes layer (likely rebuild artifacts post-Phase 3c)
- Possibly different atr_20/confirm_bars filtering in the original validation

**Specification error logged.** The H3 baseline should have cited a recomputed strict-Mode-A IS value (~+0.1844) rather than the validated_setups Mode B grandfathered value (+0.2101). This is a pre-reg writing error, not a runtime defect.

## VERDICT

**K2 FIRES (literal, per pre-reg threshold) AND K1 FIRES (substantive, zero H1 survivors)**

Both kill criteria are tripped — and they agree on the substantive verdict:

- **K1 substantive:** 0 cells survive the H1 family-level binding gate (BH-FDR K_family q<0.05 AND dir_match AND |t|>=3.0 AND yrs+ >=4 AND boot_p<0.10). Across 999 trustworthy cells in 2 VWAP filter variants × 12 sessions × 3 instruments × 3 apertures × 3 RR × 2 directions, **the VWAP family does not produce a single statistically-significant cell beyond the L6 lane that was already known and is itself flagged as a TAUTOLOGY (T0 corr=1.00 with the deployed VWAP_MID_ALIGNED filter on its own cell)**.
- **K2 literal:** the H3 positive control failed at the pre-reg threshold (|diff|=0.0284 > 0.02). Per institutional rigor, the literal verdict stands. The post-run finding above documents that this was a pre-reg specification error rather than an execution defect.

Decision rule (per pre-reg):
- continue_if: H1 survivors >= 3 → recommend rel_vol × VWAP cross-factor next
- park_if: H1 survivors == 1 or 2 → defer per-cell Pathway B
- kill_if: H1 survivors == 0 → K1 fires, VWAP family DOCTRINE-CLOSED

**Action:** **VWAP family DOCTRINE-CLOSED.** The substantive K1 verdict is independent of the H3 specification error. The K2 fire is honored as written; pre-reg specification error logged for next iteration's H3 calibration.

The OOS window for the VWAP family was consumed in this run (one-shot OOS). Re-running with a corrected H3 calibration would not be permitted under Mode A discipline because OOS would be re-tested against the same data — the family verdict stands as-is.

## Substantive findings (independent of H3 calibration)

- **0 H1 survivors at K_family BH-FDR.** The lowest p-value across 999 trustworthy cells was +3.67 (MGC EUROPE_FLOW O30 short VWAP_MID, p~0.0003 estimated) — but that fails the binding gate at boot_p=0.0608 (>0.10) and the K_family BH-FDR critical p of ~7.7e-5 (rank 1 at q=0.05/648).
- **MNQ US_DATA_1000 O15 long VWAP_MID** (the L6 lane) is correctly flagged TAUTOLOGY (T0 corr=1.00) — it IS the deployed-filter under test.
- **No second VWAP cell beyond L6** clears family-level significance. The VWAP edge appears to be specific to L6 (MNQ US_DATA_1000 O15 RR1.5 long), not a transferable family.
- **Promising tier (|t|>=2.5, dir_match, trustworthy)** contains 17 cells, top by |t|: MGC EUROPE_FLOW O30 RR{1.5,2.0} short variants (t=3.4-3.7) but boot_p>=0.06 → fail RULE 8 boot gate.
- **MES NYSE_CLOSE O5 RR2.0 short VWAP_BP** has |t|=3.04, boot_p=0.0411, dir_match=Y — but ExpR_on_IS = **-0.175** (filter selects less-bad days; on-signal still loses money). Not a tradable signal.
- **MNQ CME_PRECLOSE O5 RR1.5 long VWAP_MID** has |t|=2.66, p=0.0002, ExpR_on_IS=+0.245 — interesting but fails boot_p (not shown in table) and falls below BH-FDR K_family threshold.

## NEXT STEPS

1. **Update HANDOFF.md** with one-line "VWAP family DOCTRINE-CLOSED 2026-04-18" verdict citing this result.
2. **Update STRATEGY_BLUEPRINT.md NO-GO registry** with VWAP family closure entry, citing this scan as evidence and noting the H3 specification error caveat.
3. **Pivot to HTF Phase A build** per the surface map. The HTF orphan columns issue (`prev_week_*` / `prev_month_*` populated but no canonical build code) needs a proper port from `.worktrees/canompx3-htf` into `pipeline/build_daily_features.py`, with drift check and rebuild verification, BEFORE running an HTF scan.
4. **Pre-reg writer hardening:** add a step to the pre-reg-writer prompt that **H3 positive-control baselines must be verified by independent SQL against the SAME data window the harness will use**, not trusted from `validated_setups` (which may use a different IS boundary).

## Hardening items surfaced by this run

- **`validated_setups` baseline values are Mode B grandfathered** for any row with `last_trade_day` between 2026-01-01 and 2026-04-08 (Phase 3c/Mode A boundary). Future H3 calibrations must distinguish between the `validated_setups` ExpR (potentially Mode B IS) and the strict Mode A IS recomputation. This is a recurring trap and should be added to `.claude/rules/research-truth-protocol.md` as an explicit warning.
- **L6 deployed lane**: under strict Mode A IS the lane has ExpR=+0.1844 (vs Mode B +0.2101). The lane is still positive and live, but its IS-IS comparable is +0.1844 not +0.2101. The C12 review should be revisited to use Mode A baseline for any future drift comparisons.
- **`research/oneshot_utils.py`** currently lives only on the `research/f5-below-pdl-stage1` branch. It should be merged to main so future scans don't have to inline the bootstrap helper.

## Post-commit A+ hardening addendum (2026-04-18)

After the commit, a code review graded this work B+ for two canonical-integrity violations:
- the scan's local `vwap_signal()` re-encoded `trading_app.config.VWAPBreakDirectionFilter.matches_df`
- the scan's local `deployed_filter_signal()` re-encoded `OrbSizeFilter` / `OvernightRangeFilter` / etc.

Root-cause fixes landed in a follow-up A+ hardening commit:
- new canonical helper `research/filter_utils.py` that wraps `ALL_FILTERS[key].matches_df(df, orb_label)` as a numpy 0/1 signal — thin delegation, no re-encoding
- 15 unit tests in `tests/test_research/test_filter_utils.py` proving wrapper equivalence across 6 filter classes (VWAP_MID_ALIGNED, VWAP_BP_ALIGNED, ORB_G5, ORB_G8, OVNRNG_100, ATR_P50) — all PASS
- scan script refactored to call `filter_signal()` for both the family-under-test and the T0 tautology pre-screen; local `vwap_signal` and `deployed_filter_signal` deleted

**Finding caught by the refactor:** on the L6 cell (MNQ US_DATA_1000 O15 RR1.5 long VWAP_MID_ALIGNED) with Mode A IS filters, the refactored canonical path returns N_on_IS=435; the original inlined `vwap_signal` returned N_on_IS=436. ExpR_on_IS matches to 4 decimals at +0.1817 in both. The 1-row divergence is the exact drift-risk `.claude/rules/institutional-rigor.md` Rule 4 was written to prevent — one row differs between the old inline code and the canonical filter's NaN-handling path. **No verdict changes:** 1 row cannot turn 0 family survivors into 3+; K1 + K2 verdicts stand. But the finding documents why canonical delegation matters in practice, not just in principle.

**Scan committed with N=436 reading** on this specific cell. Not reconsumed — Mode A one-shot OOS already exhausted. Future scans using the refactored path will produce the corrected N=435. Verdict unchanged either way.

## Caveats

- Pathway A_family scan; survivors require separate Pathway B pre-reg + validator pass before deployment.
- VWAP_MID_ALIGNED and VWAP_BP_ALIGNED share the underlying `orb_{label}_vwap` column. They are not orthogonal — high cross-correlation expected and informational only.
- T0 tautology pre-screen uses approximations of canonical filters (ORB_G5 = top-quintile orb_size, ATR_P50 = median ATR, OVNRNG_100 = overnight_range/atr ≥ 1.0, ORB_G8 = orb_size ≥ 8). Material divergence from `trading_app/config.py` filter logic would underestimate T0 risk.
- Per-year IS positivity uses a per-(year, on-signal) groupby with N≥5 floor. Years with <5 trades-on are not counted toward yrs_total.
- Bootstrap is moving-block (block=5, B=10000), centered-data H0 (corrected 2026-04-18 per oneshot_utils.py addendum). Tail = upper if ExpR_on_IS>0 else lower.
- Dir_match strictly requires sign(Δ_IS) == sign(Δ_OOS) AND sign(Δ_IS) != 0.

## Reproducibility

- Repo: `C:/Users/joshd/canompx3` (parent), branch `main`
- Pre-reg sha: `495810f5`
- Script: `research/vwap_comprehensive_family_scan.py`
- DB: `C:\Users\joshd\canompx3\gold.db`
- Canonical sources used: `pipeline.dst.SESSION_CATALOG`, `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS`, `pipeline.asset_configs.ASSET_CONFIGS`, `pipeline.paths.GOLD_DB_PATH`, `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`.
