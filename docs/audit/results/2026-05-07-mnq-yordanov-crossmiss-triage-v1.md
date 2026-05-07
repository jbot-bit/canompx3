---
pooled_finding: true
per_cell_breakdown_path: docs/audit/results/2026-05-07-mnq-yordanov-crossmiss-triage-v1.csv
flip_rate_pct: 0.0
---

# Yordanov 2026 § 3.8 Cross+Miss veto-signal triage on 3 deployed MNQ lanes

**Pre-reg:** `docs/audit/hypotheses/2026-05-07-mnq-yordanov-crossmiss-triage-v1.yaml`
**Pre-reg commit lock:** `fd7c5073`
**Companion CSV:** `docs/audit/results/2026-05-07-mnq-yordanov-crossmiss-triage-v1.csv`
**Canonical DB:** `C:\Users\joshd\canompx3\gold.db`

## Scope

Pathway B K=1 individual-mechanism IS-only triage probe over 3 deployed
MNQ lanes (rebalance 2026-05-03 per `docs/runtime/lane_allocation.json`).
IS window: `trading_day < 2026-01-01` (`trading_app.holdout_policy.HOLDOUT_SACRED_FROM`).
Scope frozen at pre-reg commit `fd7c5073`; identical to
`docs/audit/hypotheses/2026-05-07-mnq-yordanov-crossmiss-triage-v1.yaml § scope`.
3 lanes: COMEX_SETTLE / US_DATA_1000 (O15) / NYSE_OPEN. OOS not consumed.

## Question

Does the Yordanov 2026 § 3.8 "Cross + Dev Miss" veto-signal pattern
(NQ futures, n=44, 22.7% hit-rate at 0.5× dev vs 71.1% baseline, 48.4pp
gap) appear on MNQ when ORB high/low is substituted for Yordanov's Volume
Profile Value Area as the filter range?

## Verdict

**HALT — DESIGN_FLAWED. AMENDMENT REQUIRED before any second run.**

The pre-registered hit-rate metric is structurally degenerate against the
pre-registered bucket definitions. The probe ran successfully against
canonical layers (3,103 IS trades classified across the 3 deployed MNQ
lanes), but the resulting numbers are uninformative by mathematical
construction, NOT by edge:

- pooled `hit_rate(NO_CROSS) − hit_rate(CROSS_MISS)` = **99.40pp**, z = 45.47, two-sided p ≈ 0
- per-lane gaps: 100.0pp / 98.6pp / 99.6pp on (COMEX_SETTLE / US_DATA_1000 / NYSE_OPEN)

These numbers far exceed Yordanov's NQ-published 48.4pp gap and exceed every
plausible MNQ-edge magnitude. Per `backtesting-methodology.md` § RULE 12
("|t| > 7 → STOP and investigate") the run is treated as a discovery-loop
failure, NOT an edge claim. **No promotion. No annotation of the Yordanov
extract. No confirmatory pre-reg.**

## Why the metric is degenerate

The pre-reg bucket definitions are:

- `NO_CROSS`: no `bar.close` re-cross of `orb_mid` in the post-entry window
- `CROSS_HIT`: re-cross occurred AND favourable excursion >= 0.5×ORB-range
- `CROSS_MISS`: re-cross occurred AND favourable excursion never reached 0.5×ORB-range

The pre-reg primary metric is `hit_rate(bucket) at 0.5x dev`, defined as
"P(post-entry favourable excursion >= 0.5×ORB-range)". This is exactly the
quantity used to *split* `CROSS_HIT` from `CROSS_MISS`. So:

- `hit_rate(CROSS_HIT)` = 1.0 by construction
- `hit_rate(CROSS_MISS)` = 0.0 by construction
- `hit_rate(NO_CROSS)` is the only freely-measurable quantity

The pooled gap collapses to `hit_rate(NO_CROSS) − 0`, which on the deployed
lanes (where most non-stopped LONG trades extend 0.5×ORB without retracing
to mid) is ≈100%. **The 99.40pp gap reports a tautology, not a Yordanov
analogue.**

## Why this happened — Yordanov's measurement frame is multi-episode

Re-reading Yordanov 2026 § 2.7 + § 3.8 carefully:

> "After Episode 1 (initial breakout), the session is monitored for a Filter
> Mid cross. ... If a cross occurs, the remaining session is scanned for
> Episode 2 (bo2) — a fresh breakout through either filter boundary."

His hit-rate is reported per-bucket as a DESCRIPTIVE statistic of the
within-episode follow-through depth. The session-level bucket label is
applied to Episode 1, but `hit_rate(0.5x)` is measured relative to
breakout reference, BEFORE any cross — so for `CROSS_MISS`, the hit-rate
of 22.7% is "the proportion of CROSS_MISS sessions where Episode 1 did
reach 0.5×". His CROSS_MISS bucket allows in-episode hit-then-cross
ordering ambiguity that our trade-event ledger does not preserve.

A faithful single-trade analogue would need EITHER:

1. **Reconstruct session episodes** independent of the lane's E2 entry
   (treat the entry as one of many possible breakouts within the session;
   measure per-episode mid-cross + dev-hit ordering on bars_1m without
   reference to `orb_outcomes` at all). This is closer to a feature-build
   exercise than a triage.
2. **Re-define the veto signal operationally**: use mid-cross occurrence as
   the bucket, and measure something orthogonal to bucket membership as
   the outcome (e.g., subsequent `pnl_r`, time-to-stop, or terminal
   excursion AFTER the cross event). This makes the veto signal a
   regressor on PnL, not a follow-through-depth classifier.

Either route requires an amendment to the pre-reg, NOT a same-session
re-run. Same-session metric tuning would be post-hoc threshold drift
(`pre_registered_criteria.md` Forbidden, `backtesting-methodology.md`
RULE 12 red flag).

## What was correctly implemented

- Canonical layers only (`orb_outcomes` triple-joined to `daily_features`).
- Sacred holdout enforced (`trading_day < 2026-01-01` from `HOLDOUT_SACRED_FROM`).
- Canonical filter delegation via `research.filter_utils.filter_signal`.
- Strict look-ahead boundary (`bars_1m.ts_utc > entry_ts`).
- Triple-join on `(trading_day, symbol, orb_minutes)`.
- scratch-policy: drop, declared in script header.
- Post-entry window bounded by `exit_ts` per pre-reg (corrected mid-run from
  an initial 23-hour day boundary that further inflated hit-rates).

The probe and self-review machinery work; the bug is in the conceptual
design of the pre-registered metric, not the implementation.

## Why I am writing this as HALT, not as FALSIFIED / SURVIVES

Per `institutional-rigor.md` Rule 3 ("Refactor when you see a pattern of
bugs") and `quant-audit-protocol.md` ("bias rule: Do NOT call anything
'promising'..."), publishing a SURVIVES verdict on a tautological metric
would be a Rule-12 violation and a `feedback_storytelling_bias`-class
failure. The numbers technically pass the locked K2 SURVIVES criteria
(pooled gap ≥ 35pp AND ≥ 2/3 lanes ≥ 25pp), but the criteria themselves
were predicated on the metric being *informative*. They are not.

The decision-ledger entry below records this honestly. The pre-reg
remains valid as a record of what was attempted; the verdict is HALT,
the next-session decision is whether to amend or shelve.

## Per-lane raw numbers (FOR AUDIT TRAIL ONLY — DO NOT INTERPRET AS EDGE)

| lane | N_total | N_NO_CROSS | hit_NO_CROSS | N_CROSS_HIT | hit_CROSS_HIT_05x | N_CROSS_MISS | hit_CROSS_MISS_05x | gap_pp | tier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | 524 | 217 | 100.0% | 205 | 100.0% | 102 | 0.0% | 100.0pp | DEGENERATE |
| MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15 | 889 | 366 | 98.6% | 303 | 100.0% | 220 | 0.0% | 98.6pp | DEGENERATE |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | 1690 | 760 | 99.6% | 493 | 100.0% | 437 | 0.0% | 99.6pp | DEGENERATE |
| **POOLED (3 lanes)** | **3103** | **1343** | **99.4%** | **1001** | **100.0%** | **759** | **0.0%** | **99.4pp** | **DEGENERATE** |

The 100.0% on `CROSS_HIT` and 0.0% on `CROSS_MISS` are tautological — the
hit-rate metric IS the bucket label.

## Method notes

- IS window: `trading_day < 2026-01-01` from `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`.
- Triple-join: `o.trading_day = d.trading_day AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes`.
- Filter delegation: `research.filter_utils.filter_signal` (canonical wrapper around `ALL_FILTERS[key].matches_df`).
- Post-entry bars window: `bars_1m.ts_utc > entry_ts AND ts_utc <= exit_ts` (per pre-reg bucket_definitions).
- Bucket assignment: presence-based (cross_seen, dev_hit) over the in-trade window. Side from `entry_price` vs `(orb_high+orb_low)/2`.
- scratch-policy: drop (`WHERE pnl_r IS NOT NULL`).
- No writes to `validated_setups`, `experimental_strategies`, `lane_allocation.json`, `paper_trades`.

## Reproduction

```
python research/yordanov_crossmiss_triage_v1.py
```

(self-review intentionally not invoked because the verdict is HALT, not
PASS — drift checks pass independently and were verified pre-pre-reg-commit
at fd7c5073.)

## Next-session decision (USER GATE)

Two options for next session — explicit gate to user, no automatic action:

1. **AMENDMENT**: Author `2026-05-08-mnq-yordanov-crossmiss-triage-v2.yaml`
   with an orthogonal-to-bucket metric. Two candidate metric reframes:
   - Veto-as-PnL-regressor: cross_seen as a binary feature, regress
     subsequent `pnl_r`. Tests "does pre-exit mid-cross predict bad
     outcomes?". Cleanly trade-time-knowable as a proxy if cross detection
     is real-time.
   - Multi-episode session reconstruction (bars_1m only, no
     `orb_outcomes` reference): faithful Yordanov replication. Larger
     scope, longer write.
2. **SHELVE**: Park the Yordanov mechanism for MNQ; mark the literature
   extract § 3.8 as "MNQ replication attempted under pre-reg
   `2026-05-07-mnq-yordanov-crossmiss-triage-v1` HALTED on metric design
   flaw; no published verdict on edge presence". Cite this result file in
   the literature extract footnote.

Either choice requires user direction. The cheap-falsification path the
v1 pre-reg attempted is closed.

## Caveats

- HALT is NOT a falsification of Yordanov's NQ finding. It is a falsification
  of the proposed cheap-replication metric on MNQ.
- HALT is NOT a deployment claim either way. The 3 deployed MNQ lanes are
  unaffected.
- Yordanov's NQ result (n=44, 22.7% vs 71.1%, 48.4pp gap) is not refuted by
  this halt; it is also not corroborated.
- This document supersedes nothing. The pre-reg `fd7c5073` remains valid as
  the historical record of what was attempted.
