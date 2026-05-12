---
date: 2026-05-12
mode: adversarial backlog audit
status: COMPLETE
verdict: CONDITIONAL
live_impact: none
---

# Chordia Prereg Backlog Audit

## Verdict

**CONDITIONAL.** There is no justified MNQ same-session sibling audit batch
right now. The only MGC allocator-paused exact lane was audited first and
failed. MES remains blocked unless the specific non-CME_PRECLOSE question is
outside a cold session and has a real deployment role.

No production state changed. No `chordia_audit_log.yaml`, `validated_setups`,
`lane_allocation.json`, profile, or broker config mutation is authorized by
this audit.

## Evidence Read

- Current allocator view: `strategy-lab.get_lane_allocation_summary` for
  `topstep_50k_mnq_auto`, rebalance `2026-05-11`, 3 active, 51 paused.
- MGC candidate view: `strategy-lab.list_promotable_candidates(MGC)` returned
  12 FIT rows, but the allocator-paused MGC row is only
  `MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4`.
- MGC MinBTL: `research-catalog.estimate_k_budget(MGC, n_trials=1, e_max=1.0)`
  passed; `n_trials=4` failed against the 2.70-year clean horizon.
- LONA check: zero saved LONA strategies, zero completed LONA reports, zero
  uploaded symbols. LONA cannot rank these preregs today; it has no canonical
  MNQ/MES/MGC ORB datasets or repo Chordia semantics.

## MGC First

The right first prereg was the lone allocator-paused MGC row, not a broad MGC
scan:

- Prereg: `docs/audit/hypotheses/2026-05-12-mgc-cmereopen-orbg4-chordia-criteria-repair-v1.yaml`
- Runner: `research/chordia_strict_unlock_v1.py`
- Result: `docs/audit/results/2026-05-12-mgc-cmereopen-orbg4-chordia-criteria-repair-v1.md`
- Verdict: `FAIL_STRICT_CHORDIA`
- Measured: IS N=168, ExpR=+0.1110, Sharpe=0.1669, t=2.163, threshold=3.00.

This kills direct Chordia repair for `MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4`.
It also reinforces the older quarantine evidence: MGC still cannot get an
active profile until Criteria 1-10 plus deployment gates are clean.

The broader MGC FIT list is not a license to run a batch. Two sampled top
LONDON_METALS `ORB_VOL_8K` rows fail the criterion ladder on C1, C2, C7, C9,
and C13, with volume-era warnings. With MGC's short clean horizon, K must stay
tight; no MGC Chordia batch is justified from this pass.

## MES

Do not bypass `REGIME_COLD`.

Current allocator state has MES CME_PRECLOSE paused because the session regime
is cold. The same file also shows MES COMEX_SETTLE, SINGAPORE_OPEN, and
US_DATA_830 examples paused by cold regime. Strategy-lab can report many MES
rows as FIT, but that is not the live allocator question when the session gate
is cold.

One non-CME example, `MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10`, is explicitly
paused by `Session regime COLD (-0.0157)`. Its criterion ladder also fails
Chordia t at 2.25. No MES Chordia prereg is justified until the target is
non-CME_PRECLOSE, not cold under allocator regime logic, and has a concrete
profile/deployment role.

## MNQ

Stop same-session sibling audits until allocator residual EV/correlation says
what it would actually deploy.

Canonical proof-gate docs already close the old MNQ `PRIORITY_A` framing:

- `PD_*` E2 rows are wrong as tested for production because the raw selector
  used close-confirmed direction; clean long-stop replay fails.
- `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60` and
  `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5` are not direct unlocks; their
  replacement/additivity gate parked both.
- `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60` failed strict Chordia.

The next MNQ prereg, if any, must be an allocator residual EV/correlation
question first, not another same-session Chordia sibling replay.

## Best Next Step

Highest EV, shortest path to truth:

1. Keep the allocator session-regime cache fix; it removes repeated regime
   computation without changing gate semantics.
2. Treat `MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4` as Chordia-dead for profile
   repair unless a new, pre-registered object changes the question.
3. Do not run MES CME_PRECLOSE audits while CME_PRECLOSE is cold. For MES,
   first find a non-cold, non-CME candidate with a real profile role.
4. Do not run MNQ same-session sibling audits until residual EV/correlation
   identifies an actual add/replace target.
5. Preserve Mode A: no threshold, profile, session, or eligibility tuning off
   2026 OOS.
