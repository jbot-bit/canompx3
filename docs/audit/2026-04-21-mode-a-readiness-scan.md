# 2026-04-21 Mode A Readiness Scan

Scope: PCC-6 additive posture-clearing evidence for the Fork D orthogonal hunt.

Question:
- If Terminal 1 clears the ONC / baseline-cleanliness ambiguity, which of the six live lanes are structurally closest to a credible Mode A re-validation path, and which are furthest?

## Inputs

- Empirical-edge and fragility read from `docs/audit/2026-04-21-phase-b-institutional-reeval.md` on `origin/research/pr48-sizer-rule-oos-backtest`
- Gate-state and live-SR evidence read from `docs/audit/2026-04-21-reset-snapshot.md` on the same branch
- Grandfathering map from this branch's PCC-4 audit

## Mode A gate view today

Common blocker set across all six:
- `holdout_clean = FAIL`
- `grandfathered research-provisional only`
- `DSR > 0.95 = FAIL` under the current pre-ONC cross-check implementation
- `t >= 3.79 = FAIL`

Differentiators:
- empirical verdict (`KEEP` vs `CONDITIONAL`)
- negative-year fragility
- SR alarm state
- whether the lane already clears `t >= 3.00` even before any theory-state argument

## Lane ranking

| Lane | Empirical state | Key strengths | Key blockers beyond common provenance gap | Readiness tier |
| --- | --- | --- | --- | --- |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | `KEEP` | `ExpR +0.1296`, `Sharpe_ann +1.7852`, `WFE 0.8225`, no negative year with `N>=50`, no SR alarm | `t≈3.400` still below 3.79; DSR cross-check fail; holdout contamination | `CLOSEST` |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | `KEEP` | `ExpR +0.0938`, `Sharpe_ann +1.3054`, `WFE 1.4222`, no negative year with `N>=50`, no SR alarm | `t≈2.928` below 3.00 and 3.79; DSR cross-check fail; holdout contamination | `CLOSE` |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | `KEEP` | `ExpR +0.1256`, `Sharpe_ann +1.6618`, `WFE 0.7008`, no negative year with `N>=50`, no SR alarm | `t≈2.831` below 3.00 and 3.79; DSR cross-check fail; holdout contamination | `CLOSE` |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | `CONDITIONAL` | `t≈3.717`, `WFE 2.6151`, no SR alarm | Two negative years with `N>=50`; DSR fail; holdout contamination | `MID` |
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | `CONDITIONAL` | `ExpR +0.0770`, `WFE 2.8551`, no SR alarm | `t≈2.528`, one negative year with `N=98`, DSR fail, holdout contamination | `MID-LOW` |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | `CONDITIONAL` | `ExpR +0.0850`, `t≈3.511`, `WFE 1.9835` | Current SR `ALARM`, one negative year with `N=247`, DSR fail, holdout contamination | `BLOCKED-PENDING-REVIEW` |

## Practical interpretation

Closest lanes for a future clean rediscovery / Mode A rerun:
1. `TOKYO_OPEN`
2. `SINGAPORE_OPEN`
3. `US_DATA_1000`

Reason:
- all three are empirical `KEEP`
- all three avoid the current SR alarm
- all three avoid negative years with `N >= 50`

Least attractive lane to prioritize before the alarm is resolved:
- `NYSE_OPEN`

Reason:
- it is the only lane currently in live SR `ALARM`
- even after provenance is fixed, it would still carry an immediate runtime-control question

## Verdict

Mode A readiness today is **relative**, not absolute:
- `3` lanes are credible clean-rediscovery candidates once provenance is fixed
- `2` lanes are empirically alive but fragile
- `1` lane is temporarily subordinated by live SR alarm

This scan does **not** relabel any lane as Mode A-ready now. It only identifies where clean rediscovery effort would have the highest expected payoff once Terminal 1 resolves the posture blocker.
