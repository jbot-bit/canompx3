# MNQ Live-Tradeability Lens Reset

**Date:** 2026-05-11
**Scope:** Wider-lens follow-up to the MNQ production trade expansion proof gate.
**Live impact:** None. No DB, schema, broker, or `lane_allocation.json` mutation.

## Verdict

**Do not widen by weakening proof. Widen by changing the role being tested.**

The current strongest path to more valid live-tradeable MNQ exposure is not a
new prior-day E2 direction filter. The real remaining edge, if it exists, is
most likely one of:

1. same-session COMEX replacement/additivity against the current book,
2. conditional overlay / sizing role on an already-deployed parent,
3. portfolio-slot diversification from a broad family that is not claiming a
   novel filter edge.

No new lane is production-ready from the 2026-05-10 proof pass.

## Exact Claim Discipline

| Claim | Status | Canonical proof | Action |
|---|---|---|---|
| `PD_GO_LONG` / `PD_CLEAR_LONG` E2 filters produce live-placeable extra trades. | **WRONG as tested** | Clean side-specific long-stop replay in `2026-05-10-clean-long-stop-*`; deployment-bound blocks in `trading_app/config.py`, `strategy_discovery.py`, `execution_engine.py`, `deployability.py`, `lane_allocator.py`. | Closed for production. Reopen only with a new pre-registered live-placeable side-specific model. |
| `COMEX_SETTLE X_MES_ATR60` can improve the live book. | **PARK after role test** | Existing Chordia row and `2026-05-04-mnq-comexsettle-xmesatr60-chordia-unlock-v1.md`; role gate `2026-05-11-mnq-comex-additivity-replacement-gate.md`. | Do not promote. Add math is same-session/runtime-unclean; replacement loses annualized R and slightly loses Sharpe vs current book. |
| `COMEX_SETTLE ORB_G5 RR1.0` can improve the live book. | **PARK after role test** | `2026-05-10-mnq-comexsettle-orbg5-rr10-chordia-unlock-v1.md`; role gate `2026-05-11-mnq-comex-additivity-replacement-gate.md`. | Do not promote. Add math loses Sharpe; replacement gains annualized R but loses Sharpe; same-session correlation/subset reject. |
| `NYSE_OPEN X_MES_ATR60` unlocks another live lane. | **UNVERIFIED** | `2026-05-10-mnq-nyseopen-xmesatr60-rr10-chordia-unlock-v1.md`, IS t below 3.79. | Do not promote. |

## Alternative Framing Check

| Alternative | Was it actually tested? | Result |
|---|---:|---|
| **Different framing:** "more trades" as live-tradeable inventory, not raw backtest rows. | Partly. Proof gate tested strict replay plus clean long-stop replay where E2 direction could leak. | This killed the tempting prior-day long branches for production. |
| **Different role:** standalone lane vs replacement vs additive allocator candidate. | Yes for the two admissible COMEX candidates. | Both PARK in `2026-05-11-mnq-comex-additivity-replacement-gate.md`. |
| **Different layer:** signal vs execution vs portfolio. | Signal and execution were tested for PD; portfolio layer tested for the two admissible COMEX candidates. | No new production lane from this branch. |
| **Different model family:** E1, NYSE_CLOSE, D6 overlay, PR48/Track D. | Previously tested in separate bounded artifacts, not re-tested in this pass. | These remain separate branches; none can be silently promoted into the current MNQ live book. |

## Where The Real Edge Might Live

**Best current edge, if real:** still appears to be the already-deployed COMEX
parent, not the two tested RR1.0 alternatives. The evidence is not that a fancy
filter is magic; it is that the COMEX E2 parent family has repeatedly survived
more scrutiny than the prior-day E2 direction branches, and the current
replacement/additivity test did not justify swapping it out.

**Where it lives:** likely in parent/session structure and portfolio fit, not in
post-entry direction labels or TradingView-style chart proof.

**How it should be used:** allocator replacement/additivity first. Only after a
positive portfolio result should it go through correlation, same-session runtime
collision, SR/watch, event-tail/slippage, and live preflight gates.

## Silences Closed

- The stale `allocator_paused_pool_priority_a_audits` queue item is not a live
  todo anymore. All five candidates now have proof dispositions.
- `PD_*` E2 is not merely "noted as risky"; it is blocked from deployment-bound
  discovery, execution arming, deployability, and allocation.
- `COMEX_SETTLE X_MES_ATR60` and `COMEX_SETTLE ORB_G5 RR1.0` were not left as
  vague ideas; the replacement/additivity gate ran and both are PARK.
- NYSE_OPEN cross-ATR failed the strict gate and is not carried forward.

## Must Not Do

- Do not promote a row because it increases trade count.
- Do not treat `break_dir`-selected E2 historical outcomes as live entry proof.
- Do not append PD E2 rows to `chordia_audit_log.yaml` as deployment-cleared.
- Do not use TradingView, YouTube, or generated backtests as canonical proof.
- Do not mutate `lane_allocation.json` from this pass.
- Do not call anything production-ready until live-placeability, additivity,
  runtime expressibility, monitoring, and cost/risk gates are all proven.

## Next Executable Gate

The bounded allocator replacement/additivity audit has now run:
`docs/audit/results/2026-05-11-mnq-comex-additivity-replacement-gate.md`.

Result:

- `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60` -> `PARK`
- `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5` -> `PARK`

There is no PASS_ADD / PASS_REPLACE candidate from this widened-lens branch.

## Reproduction / Outputs

- Proof-gate summary:
  `docs/audit/results/2026-05-10-mnq-production-trade-expansion-proof-gate.md`.
- Clean long-stop outputs:
  `docs/audit/results/2026-05-10-clean-long-stop-mnq-us-data-1000-e2-rr1.0-cb1-pd-go-long.*`,
  `docs/audit/results/2026-05-10-clean-long-stop-mnq-comex-settle-e2-rr1.0-cb1-pd-clear-long.*`.
- Additivity / replacement output:
  `docs/audit/results/2026-05-11-mnq-comex-additivity-replacement-gate.md`.

## Caveats / Limitations

- This reset does not claim the entire tradeable universe is exhausted.
- It only closes the current MNQ candidate-expansion branch under the tested
  roles.
- Other layers, including profile construction from controlled candidates,
  E1/MES/MGC evidence, and execution overlays, require separate bounded gates.
