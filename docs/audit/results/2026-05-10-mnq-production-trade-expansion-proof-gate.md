# MNQ Production Trade Expansion Proof Gate

**Date:** 2026-05-10
**Scope:** `topstep_50k_mnq_auto` candidate expansion, no live-state mutation.

## Verdict

**SANDBOX / RESEARCH ONLY for new trades.** No new lane is production-ready from this pass.

The real production work is not "find more trades"; it is to make sure any
extra trades are live-placeable, additive to the current book, and not selected
by post-entry data. The prior-day candidates failed that standard once the
question was reframed to the live-placeable long-stop question.

## Exact Claims And Classifications

| Candidate / idea | Exact claim being tested | Canonical proof | Classification |
|---|---|---|---|
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60` | Cross-asset prior-close MES ATR context improves COMEX_SETTLE enough to be an allocator candidate. | `docs/runtime/chordia_audit_log.yaml`; `docs/audit/results/2026-05-04-mnq-comexsettle-xmesatr60-chordia-unlock-v1.md`; prior-close ATR derivation cleared in `docs/audit/results/2026-04-28-e2-lookahead-contamination-registry.md` row 30. | **CONDITIONAL**. Statistically admissible, but still needs replacement/additivity, correlation, SR/survival, and same-session runtime review before production expansion. |
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5` | A smaller RR1.0 COMEX_SETTLE broad-size lane is deployable. | `docs/audit/results/2026-05-10-mnq-comexsettle-orbg5-rr10-chordia-unlock-v1.md`: IS N=1478, t=3.802, OOS N=75. | **CONDITIONAL**, not distinct edge. Fire rate is 98.93% IS / 100% OOS, so this is basically parent/session exposure and must be tested as replacement/allocator role, not a new filter edge. |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60` | Cross-asset ATR unlocks another NYSE_OPEN lane. | `docs/audit/results/2026-05-10-mnq-nyseopen-xmesatr60-rr10-chordia-unlock-v1.md`: IS t=3.690 < 3.79. | **UNVERIFIED**. Fails strict no-theory Chordia. Do not promote. |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG` as currently encoded | Prior-day geometry long context proves a deployable E2 lane. | Raw strict replay: `docs/audit/results/2026-05-10-mnq-usdata1000-pdgolong-rr10-chordia-unlock-v1.md`. Clean live-placeable replay: `docs/audit/results/2026-05-10-clean-long-stop-mnq-us-data-1000-e2-rr1.0-cb1-pd-go-long.md`. | **WRONG as tested**. Current filter uses close-confirmed `break_dir`. Clean long-stop replay drops to IS t=1.885 and OOS ExpR=+0.0049. |
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG` as currently encoded | Prior-day clear-of-congestion long context proves a deployable E2 lane. | Raw strict replay: `docs/audit/results/2026-05-10-mnq-comexsettle-pdclearlong-rr10-chordia-unlock-v1.md`. Clean live-placeable replay: `docs/audit/results/2026-05-10-clean-long-stop-mnq-comex-settle-e2-rr1.0-cb1-pd-clear-long.md`. | **WRONG as tested**. Clean long-stop replay is IS negative: ExpR=-0.0192, t=-0.480. |

## Bias / Leakage Findings

- Current `PD_*` E2 filters are not deployable selectors because they require
  `orb_<session>_break_dir == "long"`. That close-confirmed direction can be
  post-entry for E2 stop-market touch fills.
- The correct live question is not "which historical E2 outcomes have
  `break_dir=long`?" It is "if a trader places a long stop at ORB high after ORB
  formation under this prior-day context, what is the realized distribution?"
- Clean replay answered that question for the two tempting prior-day branches:
  neither survives.

## Alternative Framings Checked

| Framing | Actually tested? | Result |
|---|---:|---|
| Standalone new lane | Partly. Strict Chordia raw replays run; clean long-stop replays added for PD branches. | No new standalone production lane. |
| Allocator / replacement role | Tested in follow-up on 2026-05-11. | `X_MES_ATR60` and `ORB_G5` both PARK in `docs/audit/results/2026-05-11-mnq-comex-additivity-replacement-gate.md`; no live allocation change. |
| Signal / execution layer | Tested for PD by clean side-specific E2 long stop. | PD long idea not strong enough clean; do not rescue via same contaminated selector. |

## Safe Use

- Use `X_MES_ATR60` and `ORB_G5` only as candidates for a bounded allocator
  replacement/additivity audit.
- Use `PD_*` only as research history and as a cautionary example of a
  direction-selector contamination pattern on E2.
- Use `research/mnq_e2_long_stop_replay_v1.py` for future clean side-specific
  E2 long-stop falsification of prior-day geometry ideas.

## Must Never Do

- Do not add `PD_*` E2 rows to `docs/runtime/chordia_audit_log.yaml` as
  deployment-cleared.
- Do not let `PD_*` E2 rows into live allocation or execution without a clean
  side-specific replay that passes the normal research and deployment gates.
- Do not treat TradingView, generated backtests, raw strict replay of a
  contaminated selector, or OOS N<30 as proof.
- Do not weaken filters or collapse runtime constraints just to increase trade
  count.

## Controls Implemented

- `trading_app.config.is_e2_deployment_unsafe_filter()` now marks direct
  break-bar E2 filters and `PD_*` close-confirmed direction selectors as unsafe
  for deployment-bound E2 use.
- `trading_app.strategy_discovery` skips those filters for future E2 discovery.
- `trading_app.execution_engine` refuses to arm those filters for E2.
- `trading_app.deployability` classifies those rows as `NO_GO_BIAS_OR_DATA`.
- `trading_app.lane_allocator` pauses those rows before scoring/allocation.

## Follow-Up Result

The allocator replacement/additivity follow-up ran on 2026-05-11:
`docs/audit/results/2026-05-11-mnq-comex-additivity-replacement-gate.md`.

Both admissible COMEX candidates classified `PARK`. No PASS_ADD /
PASS_REPLACE candidate exists from this branch, and the current E2 `PD_*`
evidence remains inadmissible for production.

## Reproduction / Outputs

- Strict replay preregs:
  `docs/audit/hypotheses/2026-05-10-mnq-comexsettle-orbg5-rr10-chordia-unlock-v1.yaml`,
  `docs/audit/hypotheses/2026-05-10-mnq-comexsettle-pdclearlong-rr10-chordia-unlock-v1.yaml`,
  `docs/audit/hypotheses/2026-05-10-mnq-nyseopen-xmesatr60-rr10-chordia-unlock-v1.yaml`,
  `docs/audit/hypotheses/2026-05-10-mnq-usdata1000-pdgolong-rr10-chordia-unlock-v1.yaml`.
- Clean live-placeable long-stop runner:
  `research/mnq_e2_long_stop_replay_v1.py`.
- Additivity / replacement runner:
  `research/mnq_comex_additivity_replacement_gate_2026_05_11.py`.
- Result outputs live under `docs/audit/results/2026-05-10-*` and
  `docs/audit/results/2026-05-11-mnq-comex-additivity-replacement-gate.*`.

## Caveats / Limitations

- This pass did not search for new strategies; it classified the candidate
  expansion set already under review.
- The COMEX candidates are not killed as research facts, but they are parked
  for live expansion under current profile/runtime constraints.
- The `PD_*` finding is narrower: it blocks the current close-confirmed E2
  direction-selector framing, not every possible prior-day geometry idea.
