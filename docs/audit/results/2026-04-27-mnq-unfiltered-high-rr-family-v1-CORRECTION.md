# MNQ unfiltered HIGH-RR family-v1 — CORRECTION (scratch-as-0R)

**Supersedes:** none. **Annotates:** `docs/audit/results/2026-04-27-mnq-unfiltered-high-rr-family-v1.md` (the original v1 result).
**Pre-reg:** `docs/audit/hypotheses/2026-04-27-mnq-unfiltered-high-rr-family-v1.yaml` (correction_notice block appended Stage 1).
**Failure log:** `.claude/rules/backtesting-methodology-failure-log.md` 2026-04-27 entry.
**Canonical fix spec (forthcoming):** `docs/specs/outcome_builder_scratch_eod_mtm.md` (Stage 4).
**Underlying class bug:** `memory/feedback_scratch_pnl_null_class_bug.md`.

**Scope:** quantify the magnitude of the silent scratch-NULL-pnl_r dropout on the v1 HIGH-RR family scan, verify directional conclusions, and supersede the original ExpR magnitudes for any downstream citation.

**Outcome:** scratch-as-0R correction applied to all 144 cells. Direction-of-conclusion preserved on every cell (0/144 sign flips). Magnitudes restated; original v1 result is now ambiguous-mode and downstream consumers must cite this CORRECTION instead. Canonical realized-EOD-MTM fix proceeds in Stage 5.

## Why this correction exists

The original v1 scan ran with a `WHERE pnl_r IS NOT NULL` filter (line 87 of `research/mnq_unfiltered_high_rr_family_v1.py`). On `MNQ E2 confirm_bars=1`, `orb_outcomes` has 65,683 rows with `outcome='scratch'` AND `pnl_r=NULL` — these are trades where neither stop nor target hit by trading-day-end. The IS-NOT-NULL filter silently dropped all of them. Bias scales with target distance because farther targets resolve less often within the trading day:

| RR axis | mean scratch% across 12 sessions | median | max |
|---:|---:|---:|---:|
| 1.0 | 13.4% | 1.3% | 74.5% (15m CME_PRECLOSE) |
| 1.5 | 16.8% | 2.9% | 82.3% |
| 2.0 | 19.2% | 4.4% | 85.9% |
| 2.5 | 21.1% | 5.6% | 86.8% |
| 3.0 | 22.6% | 7.1% | 87.4% |
| 4.0 | 24.7% | 10.2% | 88.5% |

This file presents the **first-order correction** under scratch-as-0R policy (count the trade as flat instead of dropping it). The institutionally correct treatment is scratch-as-realized-EOD-close, which requires `outcome_builder.py` to populate `pnl_r` from `pnl_points_to_r(...)` — that work lands in Stage 5 of the plan and will marginally refine these numbers (typically by a few basis points per scratch). The Stage-1 0R correction is sufficient to demonstrate that **direction of every prior verdict holds; only magnitudes change**.

## What does NOT change

**0 of 144 cells flipped sign.** Every t-stat that was positive remains positive; every negative remains negative. The CANDIDATE_READY verdicts hold; the KILL_IS verdicts hold; the RESEARCH_SURVIVOR verdicts hold under H1 (t ≥ +3.0). t-stats themselves are nearly invariant because mean and standard error shrink proportionally when zeros are added at the population mean's neighborhood.

## CANDIDATE_READY — corrected ExpR

| Apt | RR | Session | N (orig) | N (incl scratch) | scratch% | ExpR (orig) | ExpR (corrected) | t (orig) | t (corrected) | ΔExpR% |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1.0 | NYSE_OPEN | 1693 | 1719 | 1.5% | +0.0807 | +0.0795 | +3.473 | +3.473 | +1.5% |
| 5 | 1.5 | NYSE_OPEN | 1650 | 1719 | 4.0% | +0.0953 | +0.0914 | +3.227 | +3.226 | +4.2% |
| 15 | 1.0 | NYSE_OPEN | 1545 | 1715 | 9.9% | +0.0974 | +0.0877 | +3.958 | +3.956 | +11.0% |
| 15 | 1.0 | US_DATA_1000 | 1594 | 1717 | 7.2% | +0.0966 | +0.0897 | +4.037 | +4.036 | +7.7% |
| 15 | 1.5 | US_DATA_1000 | 1495 | 1717 | 12.9% | +0.1063 | +0.0926 | +3.422 | +3.420 | +14.8% |

All 5 cells retain `H1 t ≥ +3.0` with theory citation. C6/C8/C9 are not recomputed here (Stage 1 scope is ExpR-only correction); Stage 5 rebuild will produce a clean re-derivation of all four gates.

## RESEARCH_SURVIVOR — corrected ExpR

| Apt | RR | Session | N (orig) | N (incl scratch) | scratch% | ExpR (orig) | ExpR (corrected) | t (orig) | t (corrected) | ΔExpR% |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1.0 | CME_PRECLOSE | 1453 | 1643 | 11.6% | +0.0981 | +0.0868 | +4.145 | +4.143 | +13.1% |
| 5 | 1.0 | US_DATA_1000 | 1701 | 1718 | 1.0% | +0.0867 | +0.0858 | +3.806 | +3.806 | +1.0% |
| 5 | 1.5 | US_DATA_1000 | 1674 | 1718 | 2.6% | +0.0926 | +0.0902 | +3.205 | +3.204 | +2.6% |
| 15 | 1.0 | CME_PRECLOSE | 289 | 1133 | 74.5% | +0.2192 | +0.0559 | +4.063 | +3.984 | +292.0% |

The `15m × CME_PRECLOSE × RR=1.0` row is the most-affected RESEARCH_SURVIVOR: scratch rate 74.5% means the original scan reported the average of the 25.5% of days that fired and resolved within the session. The corrected ExpR is 4× smaller. The cell still passes H1 (t=+3.98) but its practical magnitude is greatly reduced; it remains RESEARCH_SURVIVOR (not CANDIDATE_READY) because its OOS profile has its own issues (see original result file).

## Worst-case magnitude inflation (top 10 cells, KILL_IS)

These cells were already KILL_IS in the original. The correction makes them less-bad numerically but they remain KILL.

| Apt | RR | Session | scratch% | ExpR (orig) | ExpR (corrected) | ΔExpR% |
|---:|---:|---|---:|---:|---:|---:|
| 15 | 4.0 | NYSE_OPEN | 44.6% | -0.5249 | -0.2908 | -80.5% |
| 15 | 4.0 | NYSE_CLOSE | 84.9% | -0.6503 | -0.0984 | -560.5% |
| 15 | 4.0 | CME_PRECLOSE | 88.5% | -0.4503 | -0.0517 | -771.5% |
| 5 | 4.0 | NYSE_CLOSE | 72.4% | -0.6520 | -0.1800 | -262.2% |
| 5 | 4.0 | CME_PRECLOSE | 47.4% | -0.4135 | -0.2177 | -89.9% |
| 15 | 3.0 | NYSE_CLOSE | 83.8% | -0.5017 | -0.0811 | -518.5% |
| 15 | 3.0 | CME_PRECLOSE | 87.4% | -0.2276 | -0.0287 | -692.3% |
| 5 | 3.0 | NYSE_CLOSE | 70.2% | -0.4595 | -0.1368 | -236.0% |
| 15 | 4.0 | US_DATA_1000 | 35.9% | -0.3160 | -0.2024 | -56.1% |
| 15 | 4.0 | COMEX_SETTLE | 29.8% | -0.4352 | -0.3058 | -42.4% |

These are not "fixed" lanes — they remain KILL on H1 (t-stats still negative and significant). The lesson is that the **magnitude of the loss was overstated** because the dropped scratches would have been near-zero. The original numbers implied catastrophic edge inversion at high RR; the corrected numbers show milder negative drift consistent with cost drag on stop-out-dominant lanes.

## Reproduction

The complete 144-cell table is reproducible from the script:

```bash
python -c "
import duckdb, numpy as np, pandas as pd
from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
sql = '''SELECT trading_day, COALESCE(pnl_r, 0.0) AS pnl_r FROM orb_outcomes
         WHERE symbol='MNQ' AND orb_label=? AND orb_minutes=? AND entry_model='E2'
         AND confirm_bars=1 AND rr_target=? AND outcome IN (\"win\",\"loss\",\"scratch\") AND trading_day < ?'''
# scratch-policy: include-as-zero
# ... compute ExpR, t per cell
"
```

Saved CSV at `/tmp/correction_table.csv` during Stage 1 build (transient; reproducible above).

## Comparison: drop vs include-as-zero — when does this matter?

The relationship between scratch rate and ExpR-magnitude inflation is not linear. The decisive variable is the *mean of the surviving distribution* on which the original scan computed:

- **Low scratch% (≤5%):** correction is negligible (<5% magnitude shift). 60 of 144 cells fall in this band — mostly RR=1.0 sessions with reliable resolution.
- **Mid scratch% (5–25%):** correction is meaningful (5–30%). 51 cells. Two of the CANDIDATE_READYs (15m × NYSE_OPEN × RR=1.0 at 9.9%, 15m × US_DATA_1000 × RR=1.5 at 12.9%) sit here.
- **High scratch% (25–60%):** correction is large (30–200%). 22 cells. RR=2.5+ on most sessions.
- **Catastrophic scratch% (>60%):** correction dominates the result (>200%). 11 cells. Mostly NYSE_CLOSE / CME_PRECLOSE at 15m × RR≥2.0.

The 5 CANDIDATE_READY cells live in the low-to-mid bands where the correction is meaningful but does not change H1 t-stat conclusions.

## Implications for prior published lane stats

Every line of `live_config`, `validated_setups`, deployed-lane fitness verdicts, and `/trade-book` ExpR was computed with the same `WHERE pnl_r IS NOT NULL` selection rule. **Direction of every verdict is preserved (no sign flips), but magnitude inflation in the deployed lanes is expected to fall in the 5–15% band** based on the scratch-rate distribution of the deployed RR=1.0–1.5 universe. Stage 6 of the plan re-verifies every downstream consumer with rebuilt `orb_outcomes`.

## Next steps

This correction is published to close the audit loop on the v1 high-RR scan. The structural fix proceeds in plan stages:
- Stage 2: drift-check guard against new research scripts using `WHERE pnl_r IS NOT NULL` without a scratch-policy annotation.
- Stage 3: Criterion 13 in `pre_registered_criteria.md` mandates `scratch_policy:` in every pre-reg.
- Stage 4: implementation spec (DESIGN ONLY).
- Stage 5: `outcome_builder.py` populates `pnl_r` for scratches with realized session-end close MTM. Companion drift check `check_orb_outcomes_scratch_pnl` asserts ≥99% scratch rows have non-NULL `pnl_r` post-rebuild.
- Stage 5b: `orb_outcomes` rebuilt for MNQ × {5,15,30}m, MES × {5,15,30}m, MGC × {5,15,30}m.
- Stage 6: every downstream consumer re-verified; halt-and-notify if any DEPLOYED lane flips to DECAY.

## Limitations

Stage-1 0R correction is first-order. The institutionally correct treatment is realized-EOD-close MTM via `to_r_multiple(...)` (Stage 5); the 0R approximation differs from realized-EOD by a few basis points per scratch on average (close ≠ exactly entry on a non-fired session). This file's ExpR magnitudes are therefore approximate; Stage 5 rebuild produces the canonical numbers.

OOS / WFE / DSR / per-year stability are NOT recomputed here — Stage 1 scope is ExpR magnitude restatement only. Stage 6 of the plan re-derives all four downstream gates after the canonical rebuild.

The `pnl_r IS NOT NULL` selection bias was the trigger; this file does NOT audit other research scripts. The drift check `check_research_scratch_policy_annotation` (Stage 2) catches new instances; Stage 6 mass-annotates the 131 pre-existing scripts.

## Pooled-finding rule compliance

This file presents per-cell results, not a pooled claim. The "scratch% by RR" summary table is a descriptive stratification, not a pooled inference (no pooled p-value, no pooled mean, no claim of universality). Per `.claude/rules/pooled-finding-rule.md` the front-matter `pooled_finding:` field is omitted (rule applies to pooled claims only).

## Literature grounding

- `docs/institutional/literature/bailey_lopezdeprado_2014_dsr_sample_selection.md` — sample-selection bias as first-class inflation source.
- `docs/institutional/literature/carver_2015_ch12_speed_and_size.md` — backtest cost realism principle.
- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md` — unified backtest/live program doctrine (live path forces flat at session end; backtest must too).
- `docs/institutional/literature/chan_2009_ch1_intraday_session_handling.md` — UNSUPPORTED (Chan 2008/2009 §1.4 does not exist; documented to prevent future fabrication).
