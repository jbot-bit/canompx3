# MNQ US_DATA_1000 F5_BELOW_PDL RR1.5 Prereg Result

**Date:** 2026-04-24
**Route:** `standalone_discovery`
**Hypothesis:** `docs/audit/hypotheses/2026-04-24-mnq-usdata1000-f5-below-pdl-rr15-v1.yaml`
**SHA:** `1fff43df829075e6601e2094945fc6318303a56b97fb9aed3028cc811c768117`
**Verdict:** `KILL_FOR_VALIDATION_NOW`
**Outcome:** rejected by validator on Criterion 8 (`N_oos=8 < 30`); F5_BELOW_PDL filter not deployed.

## Scope

Exact locked cell:

- `MNQ`
- `US_DATA_1000`
- `O5`
- `E2`
- `RR1.5`
- `CB1`
- `stop_multiplier=1.0`
- `F5_BELOW_PDL`

This was a standalone-candidate test only. It was never a live-routing claim
and never bypassed `experimental_strategies -> strategy_validator ->
validated_setups`.

## Reproduction

Front-door dry run:

- Command: `scripts/tools/prereg_front_door.py --hypothesis-file ... --db /mnt/c/users/joshd/canompx3/gold.db --execute --dry-run`
- Result: `Phase 4 safety net: 1/1 raw trials accepted by scope predicate`
- Result: `Discovered 1 strategies (1 canonical, 0 aliases)`
- DB write: none

Front-door write run:

- Command: `scripts/tools/prereg_front_door.py --hypothesis-file ... --db /mnt/c/users/joshd/canompx3/gold.db --execute`
- Result: `Phase 4 safety net: 1/1 raw trials accepted by scope predicate`
- Result: `Discovered 1 strategies (1 canonical, 0 aliases)`
- Destination: `experimental_strategies`

Validator run:

- Command: `python -m trading_app.strategy_validator --instrument MNQ --db /mnt/c/users/joshd/canompx3/gold.db --testing-mode individual --workers 1`
- Pre-check: exactly one unvalidated `MNQ` row existed, and it was this hypothesis SHA.
- Result: `0 PASSED, 1 REJECTED`
- No `validated_setups` row was created.

## Measured Row

`experimental_strategies` row:

| Strategy | N | ExpR | Sharpe Ann | p | Status |
|---|---:|---:|---:|---:|---|
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_F5_BELOW_PDL` | 125 | +0.4033 | 1.5732 | 0.000152 | `REJECTED` |

Rejection reason:

`criterion_8: N_oos=8 < 30 (Amendment 3.0 condition 4: no insufficient-OOS-data exemptions for Pathway B individual testing mode)`

## Interpretation

[MEASURED] The pre-2026 in-sample candidate hit the target shape: `N=125`,
`ExpR=+0.4033`, and `p=0.000152`.

[MEASURED] The exact prereg and Phase 4 scope did not widen: the safety net
accepted `1/1` raw trials and wrote one canonical experimental strategy.

[MEASURED] The candidate failed validation because the sacred 2026 holdout has
only `N_oos=8` matching trades. Under Pathway B, the repo explicitly forbids
insufficient-OOS-data exemptions.

[INFERRED] This is not evidence that the mechanism is dead forever. It is
evidence that the exact `.4R` claim is not validatable now. More forward
samples may make it testable later, but editing the scope today would be
post-hoc rescue.

## Actions Taken

- Registered `F5_BELOW_PDL` for hypothesis-file injection only.
- Locked a `K=1` prereg for the exact `.4R` frontier cell.
- Executed through `prereg_front_door`, not an ad hoc script.
- Ran the validator once it was measured safe to do so.
- Left the result as rejected research evidence, not live/deployment truth.

## Limitations

- Did not promote to `validated_setups`.
- Did not change profiles, lane allocation, deployment scope, sizing, or
  `paper_trades`.
- Did not rerun consumed `PD_*` hypotheses.
- Did not broaden the prior-day geometry grid.
- Did not implement the short-side `F5_BELOW_PDL` path; the current validation
  failure shows the immediate bottleneck is OOS sample count, not short-side
  predicate plumbing.

## Next Decision

Do not run another exact `F5_BELOW_PDL` `.4R+` Pathway-B cell tonight unless it
has at least 30 held-out trades or a separately justified family-mode design.
The visible top alternatives in the triage note also have `N_oos=8`, so they
are likely to hit the same Criterion 8 wall.

The next highest-EV branch is to work on deployable expectancy improvement
from already-validated prior-day geometry shelf rows, or wait for more 2026
forward samples before retrying exact `.4R+` standalone validation.
