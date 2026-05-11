# PR48 MGC shadow-only observation closeout

**Date:** 2026-05-11  
**Queue item:** `pr48_mgc_shadow_observation`  
**Verdict:** `PASS_OPERATOR_VISIBILITY_UNSCORED`

## Scope

Observe whether the `MGC:cont_exec` PR48 shadow-only overlay is visible through
the intended runtime/operator surfaces without turning into live sizing,
allocator state, `validated_setups`, or a synthetic trade carrier.

This is not a promotion decision and not a live sizing authorization.

## Grounding

- Design: `docs/audit/results/2026-04-23-pr48-mgc-shadow-only-overlay-design.md`
- Translation blocker: `docs/audit/results/2026-04-23-pr48-mgc-cont-exec-bounded-translation.md`
- Research source: `docs/audit/results/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.md`
- Decision ledger:
  - `pr48-mgc-shadow-overlay-phase1-implemented`
  - `pr48-phase2-conditional-role-native-surface-implemented`

## Outputs

### Lifecycle / derived state

Command:

```bash
./.venv-wsl/bin/python -c "import json; from trading_app.lifecycle_state import read_lifecycle_state; s=read_lifecycle_state('topstep_50k'); print(json.dumps(s['conditional_overlays'], indent=2, sort_keys=True, default=str))"
```

Observed:

- `conditional_overlays.available = true`
- `conditional_overlays.valid = true`
- overlay id: `pr48_mgc_cont_exec_v1`
- mode: `shadow_only`
- role: `allocator`
- state date: `2026-05-11`
- summary:
  - `row_count = 18`
  - `ready_count = 0`
  - `unscored_count = 18`
  - `invalid_count = 0`
  - `status = unscored`

The state envelope was written at:

- `data/state/conditional_overlay_topstep_50k_pr48_mgc_cont_exec_v1.json`

The envelope is structurally valid:

- `state_type = conditional_overlay_shadow`
- `schema_version = 1`
- freshness `as_of_date = 2026-05-11`
- freshness `max_age_days = 1`

### Dashboard payload

Command:

```bash
./.venv-wsl/bin/python -c "import json; from trading_app.live.bot_dashboard import _build_operator_payload; p=_build_operator_payload('topstep_50k'); print(json.dumps({'profile': p.get('profile'), 'conditional_overlays': p.get('conditional_overlays'), 'overlay_checks': [c for c in p.get('checks', []) if c.get('name') == 'Conditional overlays']}, indent=2, sort_keys=True, default=str))"
```

Observed dashboard check:

```json
{
  "name": "Conditional overlays",
  "status": "info",
  "detail": "pr48_mgc_cont_exec_v1 unscored"
}
```

The dashboard therefore surfaces the overlay state explicitly and does not hide
the lack of a current score.

### Live log path

Log/code inspection found the live wiring path intact:

- `trading_app/live/session_orchestrator.py` constructs `RoleResolver(profile_id, today=self.trading_day)` for profile sessions.
- `trading_app/execution_engine.py` records overlay context only in `ActiveTrade.overlay_context`.
- Existing logs show the RoleResolver startup hook being invoked:
  - `logs/live/live_20260425_1033.log.err`
  - `logs/live/live_20260425_0228.log.err`
  - `logs/live/live_20260425_0225.log.err`
  - `logs/live/demo_20260425_0205.log.err`

No `CONDITIONAL_OVERLAY_SHADOW` trade-context event was found for this closeout.
That is not treated as a failure because the observed 2026-05-11 overlay state is
`unscored`, and this stage is visibility-only. It does mean there is no empirical
trade-context evidence to use for promotion.

## Interpretation

The observation path behaves as designed:

1. The static PR48 MGC overlay spec exists and is profile scoped.
2. The derived-state envelope self-validates against profile, lane, DB, and code
   identity.
3. Missing current-day MGC feature data degrades to `unscored` instead of
   guessing buckets.
4. The dashboard exposes the exact `unscored` state as an operator-visible
   check.
5. Runtime wiring records shadow context only; it does not mutate contracts,
   allocator state, `validated_setups`, or execution sizing.

## Caveats

The visibility verdict does not rescue the underlying PR48 `rel_vol` research
object. A later authority, `docs/runtime/decision-ledger.md`
`rel-vol-banned-on-e2-2026-04-28`, bans `rel_vol_<SESSION>` as an E2 predictor
unless the canonical computation changes and is re-audited. The observed MGC
overlay is therefore acceptable only as a visibility-path observation. It must
not be used as promotion, sizing, or deployment evidence.

## Verdict

`PASS_OPERATOR_VISIBILITY_UNSCORED`

Close the observation item as complete for operator visibility. Do not treat
this as:

- live deployment readiness
- evidence that the overlay produced a current-day bucket
- evidence that any MGC trade should be resized
- a reason to promote `MGC:cont_exec` into `validated_setups` or
  `lane_allocation.json`
- a clean research-valid rel-vol-on-E2 finding

## Next Action

With the MGC shadow-observation blocker closed, the next honest queue action is
`mes_q45_exec_bridge`: define whether `MES:q45_exec` can be expressed as a
bounded runtime surface, or explicitly reject it as not yet expressible.
