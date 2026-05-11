# PR48 MES `q45_exec` bridge rejection

**Date:** 2026-05-11  
**Queue item:** `mes_q45_exec_bridge`  
**Verdict:** `REJECT_CURRENT_BRIDGE_TAINTED_E2_RELVOL`

## Scope

Define whether the alive PR48 `MES:q45_exec` research branch can be carried into
a bounded runtime surface, or explicitly reject it as not yet expressible.

## Outputs

The bounded PR48 result scripts were rerun against the current canonical DB.
They updated the result docs from latest canonical trading day `2026-04-16` to
`2026-05-07`:

```bash
./.venv-wsl/bin/python research/pr48_role_followthrough_v1.py
./.venv-wsl/bin/python research/pr48_promotion_shortlist_v1.py
```

The numerical rerun still shows `MES:q45_exec` as positive versus its parent:

- IS shortlist test: `+$16.19/day`, `t=+8.219`, BH survives.
- OOS direction check: positive, `+$20.48/day`, direction matches.
- Candidate metrics:
  - IS `q45_exec`: policy EV `-0.0015R/opportunity`, daily total `+$13,502`.
  - OOS `q45_exec`: policy EV `+0.0040R/opportunity`, daily total `+$323`.

So the old branch is not lost numerically. The issue is representation and
research validity.

## Caveats

This rejection applies to the current PR48 `MES:q45_exec` bridge object, which
depends on the tainted E2 `rel_vol_<SESSION>` predictor. It does not reject the
broader role question forever, and it does not evaluate a future re-derived
pre-entry-safe substitute feature. The rerun confirms the old numbers still
exist; the blocker is representation and integrity, not disappearance of the
historical effect.

## Blocking Authority

The later integrity authority supersedes any deployment-bound use of this
feature on E2:

- `docs/runtime/decision-ledger.md` entry
  `rel-vol-banned-on-e2-2026-04-28`
- `docs/audit/results/2026-04-28-e2-lookahead-contamination-registry.md`
- `pipeline/build_daily_features.py` relative-volume construction

The key rule is that `rel_vol_<SESSION>` is banned as an E2 predictor because
its numerator is break-bar volume. For E2, a material share of entries occur
before the canonical close-outside-ORB break bar, making the feature post-entry
data on those rows.

`MES:q45_exec` is exactly a `rel_vol_{session}` Q4+Q5 filter on `MES O5 E2 CB1
RR1.5`, so it cannot be carried into runtime honestly unless the predictor is
cleanly re-derived with a pre-entry-safe substitute or the canonical
`rel_vol_<SESSION>` computation changes and is re-audited.

## Runtime Fit Check

Even ignoring the feature-taint blocker, the current runtime is not a clean
direct carrier:

1. The only active execution profile is `topstep_50k_mnq_auto`, instrument
   `MNQ` only.
2. There is no active MES profile-local parent surface.
3. `MES:q45_exec` is not a concrete validated standalone `strategy_id`; it is a
   conditional filter over all `MES O5 E2 CB1 RR1.5` session-direction parents.
4. The current conditional-overlay infrastructure can expose shadow context, but
   using it here would preserve a banned E2 `rel_vol` predictor as operator
   state.

## Decision

Reject the current `MES:q45_exec` bridge.

Do not implement:

- a MES q45 runtime overlay,
- a MES q45 profile-local filter sleeve,
- a `validated_setups` promotion,
- a `lane_allocation.json` route,
- or an execution-engine hook for this exact `rel_vol` q45 object.

The only honest continuation is a clean re-derivation stage that keeps the PR48
role question but replaces the E2-tainted predictor with a pre-entry-safe
feature. Candidate feature families must come from the safe-list in the E2
look-ahead registry, not from break-bar state.

## Next Action

Close `mes_q45_exec_bridge` as rejected for the current object. The next queue
item is `track_d_mnq_comex_settle_gate0_runner_design`.
