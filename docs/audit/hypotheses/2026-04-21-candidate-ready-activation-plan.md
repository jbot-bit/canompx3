# PR #51 CANDIDATE_READY Activation Plan (2026-04-21)

**Status:** PLAN ONLY — not executed this session. Use as the entry point for the next clean session.

**Parent:** `docs/audit/results/2026-04-20-mnq-unfiltered-baseline-cross-family-v1.md` (PR #51, 5 CANDIDATE_READYs).
**Authority:** `docs/institutional/pre_registered_criteria.md` Amendment 3.0 (Pathway B individual-hypothesis discipline) + Criterion 11 (account death MC) + Criterion 12 (Shiryaev-Roberts drift monitor).

## The 5 candidates

From `docs/audit/results/2026-04-20-mnq-unfiltered-baseline-cross-family-v1.md`, all `CANDIDATE_READY` cells after 4 gates:

| # | orb_min | RR | session | IS_N | IS_ExpR | IS_t | 2026_N | 2026_ExpR |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| C1 | 5  | 1.0 | NYSE_OPEN     | 1693 | +0.0807 | +3.473 | 71 | +0.1360 |
| C2 | 5  | 1.5 | NYSE_OPEN     | 1650 | +0.0953 | +3.227 | 68 | +0.0846 |
| C3 | 15 | 1.0 | NYSE_OPEN     | 1545 | +0.0974 | +3.958 | 55 | +0.2574 |
| C4 | 15 | 1.0 | US_DATA_1000  | 1594 | +0.0966 | +4.037 | 60 | +0.1799 |
| C5 | 15 | 1.5 | US_DATA_1000  | 1495 | +0.1063 | +3.422 | 55 | +0.1609 |

All five are **UNFILTERED** lanes — no `filter_type` in the spec, just session/aperture/RR/entry_model/direction. This is a **new structural category** for this project; no current DEPLOYED lane is UNFILTERED.

## Why they are not yet activated

Verified 2026-04-21: none of these `strategy_id`s exist in `validated_setups`. The `allocator → lane_allocation.json` pipeline only promotes from `validated_setups`, so there is no code path by which a `CANDIDATE_READY` mark in a results MD becomes a live lane. Activation requires running the full validator on each candidate.

Additionally, `validated_setups.filter_type` has no `UNFILTERED` entry (existing values: ATR_P50, ATR_P70, COST_LT08, COST_LT12, CROSS_SGP_MOMENTUM, GAP_R015, ORB_G5, ORB_G5_NOFRI, ORB_G8, OVNRNG_10, OVNRNG_100, OVNRNG_50, PDR_R080, VWAP_MID_ALIGNED, X_MES_ATR60). The schema likely permits it but has never been populated — a pre-activation schema check is required.

## Pre-registered activation sequence (for the next session)

### Step 0 — Verify schema tolerance for UNFILTERED

```sql
-- Inspect validated_setups filter_type column
DESCRIBE validated_setups;
-- Does ALL_FILTERS include an UNFILTERED (no-op) entry?
-- If not, either add a null-pass filter to trading_app/config.py, or
-- the validator needs to handle filter_type IS NULL / UNFILTERED at the
-- matches_row site.
```

If `ALL_FILTERS` does not have an `UNFILTERED` key, pause here. Adding one is a production-code change that needs its own pre-reg + blast-radius scan (affects `strategy_validator._check_criterion_8_oos`, `rebalance_lanes`, and any consumer iterating `ALL_FILTERS`).

### Step 1 — Write a Pathway B individual pre-reg per candidate

Per Amendment 3.0, UNFILTERED activation is Pathway B K=1 per candidate. One YAML per candidate (5 total) at `docs/audit/hypotheses/2026-04-22-pr51-candidate-C{n}.yaml`. Each must include:

- `testing_mode: individual`
- `theory_citation` (justifying why THIS session × aperture × RR × entry is expected to have unfiltered edge — e.g., Crabel 1990 opening range for NYSE_OPEN)
- Exact scope (instrument, orb_label, orb_minutes, rr_target, entry_model, confirm_bars, direction)
- Criterion-8 `require_power_floor: false` for the initial promotion (activate under Tier 2 directional semantics per Amendment 3.2)
- Kill criteria: OOS ExpR < 0; 2026 sign flip; negative tail exceeding stored max_drawdown_r.

### Step 2 — Run strategy_validator against each candidate

```bash
python -m trading_app.strategy_validator \
  --instrument MNQ \
  --min-sample 100 \
  --no-regime-waivers \
  --hypotheses docs/audit/hypotheses/2026-04-22-pr51-candidate-C{n}.yaml \
  --holdout-date 2026-01-01
```

Each candidate must pass **all 12 criteria** per `pre_registered_criteria.md` acceptance matrix to enter `validated_setups`. Under Amendment 3.2, Criterion 8 is the 3-tier gate — with 55-71 2026 paper trades per candidate, all are Tier 2 directional, and they will pass the sign gate unless OOS ExpR < 0.

### Step 3 — Account death MC (Criterion 11)

Before any candidate is added to `prop_profiles.ACCOUNT_PROFILES` as a PROVISIONAL lane:

```bash
python scripts/tools/account_death_mc.py \
  --profile topstep_50k_mnq_auto \
  --candidate-lanes C1,C2,C3,C4,C5 \
  --paths 10000 --horizon-days 90
```

Require 90-day survival ≥ 70% (Criterion 11). Correlation-adjusted: the 5 candidates are all MNQ, so they correlate with the existing 6 lanes — the MC must include the full 11-lane portfolio.

### Step 4 — Shiryaev-Roberts monitor (Criterion 12)

Each new lane needs an SR drift monitor active at deployment. Depending on the current infra, this may or may not be automatic — verify before promotion.

### Step 5 — Allocator rebalance

```bash
python scripts/tools/rebalance_lanes.py --profile topstep_50k_mnq_auto
```

The rebalance reads `validated_setups` (now containing the newly-activated candidates) and regenerates `docs/runtime/lane_allocation.json`. Session-regime gating may still filter some out.

## Expected outcome

- 2–4 of 5 candidates likely pass all 12 criteria. The NYSE_OPEN 15m RR1.0 (C3, IS t=+3.96) is the strongest by IS stats and has the cleanest 2026 shadow (+0.26 R).
- Lane count grows from 6 to 8–10 on the topstep_50k_mnq_auto profile.
- Portfolio N grows proportionally; Tier 1 portfolio timing accelerates.

## Explicit non-goals of this plan file

- Does NOT include a "if the candidate fails Criterion 5 DSR, here's how to pass it" branch. No post-hoc tuning. If a candidate fails, it fails.
- Does NOT authorize skipping Criterion 11 account-death MC for speed.
- Does NOT authorize hand-editing `lane_allocation.json` to include a candidate that has not been through the validator pipeline.

## Next-session checklist

1. Read this file and `docs/audit/results/2026-04-20-mnq-unfiltered-baseline-cross-family-v1.md` (the parent PR #51 results).
2. Execute Step 0 (schema check) — if it fails, pause and design the UNFILTERED plumbing first.
3. Execute Steps 1–5 sequentially. Do not parallelize — each gate is sequential.
4. If any candidate fails Criterion 1 or 2, park that candidate (do not modify pre-reg to get it to pass).
5. Rerun `research/portfolio_bootstrap_v1.py` after activation to measure portfolio-level Tier 1 proximity.
