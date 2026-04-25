# Post Stale-Lock Action Queue

**Date:** 2026-04-21
**Context:** broad stale-lock audit + adjacent-edge ranking for ORB
**Purpose:** convert the audit outcome into an explicit action queue instead of leaving it as narrative-only research interpretation.

## Binding rules

- No sacred holdout tuning.
- No holdout leakage.
- No hand-picked subset rediscovery dressed up as confirmation.
- No generic pooled ML smoothing.
- Old kills stay binding unless the reopen condition is explicit and mechanism-correct.

## Ranked queue

### 1. GARCH `R3` session-clipped forward shadow

- Status: `FIRST PASS DONE -> READY_FOR_FORWARD_MONITORING`
- Why: strongest surviving evidence stack; adds at the meta-layer; not killed by A4 router failure.
- Exact action:
  - freeze `SESSION_CLIPPED`
  - freeze session-support table from 2026-04-16 normalized sizing audit
  - freeze normalization factor
  - emit trade ledger + daily ledger + markdown report from canonical forward outcomes for `topstep_50k_mnq_auto`
- What actually happened:
  - scaffold built at `research/garch_r3_shadow_ledger.py`
  - first artifacts emitted to `data/forward_monitoring/` and `docs/audit/results/2026-04-21-garch-r3-session-clipped-shadow.md`
  - root cause was not strategy logic but a rolling post-pass builder bug:
    narrow incremental `daily_features` rebuilds lost prior seed history and
    computed `garch_forecast_vol_pct` before the current row's
    `garch_forecast_vol`
  - canonical fix landed in `pipeline/build_daily_features.py`, guarded by
    `pipeline/check_drift.py` and regression tests
  - recent late-history GARCH gaps were repaired for `MNQ`, `MES`, and `MGC`
    on the affected window and stale `validated_setups` trade-window
    provenance was refreshed via
    `scripts/migrations/backfill_validated_trade_windows.py`
  - rerun verdict: `READY_FOR_FORWARD_MONITORING` with `0` missing-state
    fallback trades and `0` raw feature-gap rows on the emitted shadow ledger
- Stop condition:
  - any need to retune map, cutoffs, or support table
  - provenance failure for `garch_forecast_vol_pct`

### 2. L1 `EUROPE_FLOW` pre-break-context overlay

- Status: `ACTIONED -> KILL`
- Why: the restored frozen prereg was executed on 2026-04-23 and no admissible
  feature survived honestly.
- What actually happened:
  - frozen prereg restored at
    `docs/audit/hypotheses/2026-04-21-l1-europe-flow-pre-break-context-prereg.yaml`
  - executed via `research/l1_europe_flow_pre_break_context_scan.py`
  - result written to
    `docs/audit/results/2026-04-21-l1-europe-flow-pre-break-context-prereg.md`
  - `pre_velocity_HIGH_Q3` did not reach raw significance
  - `rel_vol_HIGH_Q3` showed IS lift but failed `K=2` BH-FDR and flipped OOS sign
- Stop condition:
  - do not reopen banned `break_*` or ATR-normalized replacement variants from this path

### 3. Prior-day level Pathway-B on one strongest hot cell

- Status: `LOCKS_WRITTEN -> EXECUTION TRIAGE NEXT`
- Why: family not dead; the repo now contains locked exact bridge hypotheses,
  but the next executed path has not yet been chosen and run.
- Exact action:
  - triage the already-locked exact hypotheses
  - execute one confirm-or-kill path only
  - no renewed mega scan
- Stop condition:
  - broad rescan over the whole prior-day family
  - multiple lock-shopping after seeing forward behavior

### 4. Cross-asset earlier-session context for later ORB quality

- Status: `KEEP, NOT YET`
- Why: plausible regime-conditioning role remains open, but timing discipline is easy to get wrong.
- Exact action:
  - prereg a single paired-session transmission path only
  - condition later ORB quality, not direction prediction
- Blocker:
  - needs a stricter chronology spec before any scan

### 5. HTF freshness / distance / outside-range conditioning

- Status: `PARK`
- Why: simple break-aligned HTF family is dead; only a truly different feature class is open.
- Exact action:
  - feature-build prereg first
  - no scan until canonical HTF freshness/distance fields exist and are drift-guarded
- Blocker:
  - canonical feature layer does not exist yet

## Dead and not to be reopened casually

- OVNRNG router / session-ranking framing
- simple HTF prev-week / prev-month break-aligned family
- generic pooled ML smoothing / meta-model rescue attempts

## Immediate execution order

1. Treat the GARCH `R3` shadow as an active forward-monitoring artifact, not a blocked hypothesis stub.
2. Leave the frozen policy untouched; monitor ledger deltas instead of retuning.
3. Start the Prior-day Pathway-B execution triage as the next open research path.
