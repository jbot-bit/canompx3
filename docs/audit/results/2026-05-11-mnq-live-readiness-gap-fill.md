# MNQ Live Readiness Gap Fill

**Date:** 2026-05-11
**Mode:** deployability / live-readiness audit
**Live execution changed:** no
**DB/schema writes:** no

## Scope / Question

Are existing MNQ trades being blocked from live-readiness classification by a
metadata gap rather than by missing canonical proof?

## Exact Claim

The project was too strict in one metadata layer: `validated_setups.slippage_validation_status`
is blank for most active MNQ rows, but canonical MNQ E2 TBBO evidence already covers routine
slippage for the deployed MNQ E2 session set. A blank row-level slippage field should not hard
block an MNQ E2 strategy when its session is one of the covered routine TBBO sessions. It should
remain a controlled-live-pilot warning because event-day tail risk is still open.

## Canonical Data

- Candidate source: `validated_setups` as candidate list only.
- Replay source: canonical `orb_outcomes` joined to `daily_features` through
  `trading_app.strategy_fitness._load_strategy_outcomes`.
- Slippage basis:
  - `docs/runtime/debt-ledger.md` records MNQ TBBO pilot v2 N=142, 100% of days within
    the modeled two-tick routine threshold across all 9 deployed MNQ sessions.
  - `docs/audit/results/2026-04-20-mnq-e2-slippage-pilot-v2-gap-fill.md` lists covered
    sessions: `CME_PRECLOSE`, `COMEX_SETTLE`, `EUROPE_FLOW`, `LONDON_METALS`,
    `NYSE_OPEN`, `SINGAPORE_OPEN`, `TOKYO_OPEN`, `US_DATA_1000`, `US_DATA_830`.
- Scope of inference: MNQ + E2 + one of those covered sessions only.

## Anti-Tunnel Check

This is not an E2-only doctrine claim. Fresh active-shelf counts are:

| Instrument | Entry model | Active rows |
|---|---:|---:|
| MES | E2 | 48 |
| MGC | E1 | 3 |
| MGC | E2 | 10 |
| MNQ | E1 | 44 |
| MNQ | E2 | 742 |

E2 is dominant in the current MNQ inventory, and the slippage evidence is specifically MNQ E2
stop-market evidence. The fix therefore applies to MNQ E2 only. MNQ E1, MES, and MGC must keep
their own evidence gates.

## Implementation

`trading_app/deployability.py` now maps blank slippage status to a controlled
`slippage_event_tail_pending` warning only when all of these are true:

1. `instrument == MNQ`
2. `entry_model == E2`
3. `orb_label` is one of the nine covered MNQ TBBO routine sessions

Everything outside that scope still fails closed as `slippage_missing`.

The broad all-active audit also hit a native DuckDB/Python segfault while replaying hundreds of
rows through one read-only connection. The audit now refreshes the replay connection every 50 rows.
This is still read-only and does not change replay logic.

## Results

Profile audit:

- File: `docs/audit/results/2026-05-11-topstep-50k-mnq-profile-deployability.json`
- Total selected profile candidates: 2
- Deployable candidates: 2
- Verdict: both `CONTROLLED_LIVE_PILOT_CANDIDATE`
- Hard blockers: none
- Institutional language allowed: 0

Selected profile candidates:

- `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`
- `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`

All-active MNQ audit:

- File: `docs/audit/results/2026-05-11-mnq-all-active-deployability.json`
- Total active MNQ candidates: 786
- Controlled live pilot candidates: 177
- Institutional language allowed: 0
- Remaining verdicts:
  - `BLOCKED_FAMILY_FRAGILE`: 476
  - `BLOCKED_OOS_UNDERPOWERED`: 77
  - `BLOCKED_REPLAY_MISMATCH`: 36
  - `NO_GO_BIAS_OR_DATA`: 19
  - `BLOCKED_SLIPPAGE`: 1

## Classification

**VALID, scoped.**

The correction is valid for MNQ E2 covered sessions as a routine-slippage metadata gap fill. It
is conditional for live use because event-day tail remains unmeasured and allocator/profile
selection still has to prove additivity, correlation, runtime controls, and account-risk fit.

## Verdict / Decision

Adopt the scoped deployability inference. Covered MNQ E2 blank slippage rows
are controlled-live-pilot candidates when no other hard blocker exists; they
are not direct live routes.

## What This Can Safely Do

- Classify covered MNQ E2 rows with blank slippage metadata as controlled live pilot candidates
  when no other hard blocker exists.
- Expose a real allocator/profile candidate pool instead of burying it under false slippage gaps.
- Keep event-tail risk visible as a warning.
- Keep non-covered sessions, non-E2, non-MNQ, family-fragile, replay-mismatched, OOS-underpowered,
  and lookahead-unsafe rows blocked.

## What This Must Never Do

- Treat TradingView or any external backtest as proof.
- Borrow MNQ E2 TBBO evidence for MNQ E1, MES, MGC, or uncovered sessions.
- Promote any all-active shelf row directly to live routing without profile-scope account risk,
  allocator add/replace math, correlation checks, runtime/SR gates, and execution constraints.
- Remove the `slippage_event_tail_pending` warning until event-day tail is measured.
- Enable live trading, mutate broker state, mutate schema, or write deployment DB state from this pass.

## Next Production-Readiness Work

The highest-leverage next step is not another signal search. It is a profile-construction gate over
the 177 controlled MNQ candidates:

1. Deduplicate families and same-session variants.
2. Run add/replace/correlation against the current selected profile.
3. Enforce account risk, one-position/session, SR, and allocator Chordia gates.
4. Produce a proposed profile change set, if any, as paper/sandbox only until reviewed.

No live allocation was changed in this pass.

## Reproduction / Outputs

```bash
./.venv-wsl/bin/python scripts/tools/full_shelf_deployability_audit.py --scope profile --profile topstep_50k_mnq_auto --instrument MNQ --format json --output docs/audit/results/2026-05-11-topstep-50k-mnq-profile-deployability.json --fail-policy report-only
./.venv-wsl/bin/python scripts/tools/full_shelf_deployability_audit.py --scope all-active --profile topstep_50k_mnq_auto --instrument MNQ --format json --output docs/audit/results/2026-05-11-mnq-all-active-deployability.json --fail-policy report-only
```

Outputs:

- `docs/audit/results/2026-05-11-topstep-50k-mnq-profile-deployability.json`
- `docs/audit/results/2026-05-11-mnq-all-active-deployability.json`

## Caveats / Limitations

- The inference is not valid for MNQ E1, MES, MGC, or uncovered MNQ sessions.
- Event-day tail is still open, so institutional-language clearance remains
  false.
- The 177 all-active controlled candidates require a separate
  profile-construction gate before any paper/sandbox proposal.
