# 2026-04-21 Grandfathering Eligibility Audit

Scope: PCC-4 additive posture-clearing evidence for the Fork D orthogonal hunt.

Question:
- Which of the six live lanes cleanly fit the current `research-provisional + operationally deployable` grandfathering path, and what specific evidence would be required to close the remaining gap to Mode A cleanliness?

## Governing clauses

From `docs/institutional/pre_registered_criteria.md` Amendment 2.7:
- Existing validated rows discovered with 2026 data in scope are `Research-provisional` and `NOT OOS-clean`.
- Existing deployed lanes may remain `operationally deployable`, but they inherit the provisional status and cannot be relabeled production-grade without clean rediscovery under `--holdout-date 2026-01-01`.
- Walk-forward `wf_passed = True` remains useful as in-sample discipline only; it does not substitute for clean forward OOS.

## Lane map

| Lane | discovery_date | provenance | wf_passed | current grandfathering fit | missing evidence to close gap |
| --- | --- | --- | --- | --- | --- |
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | `2026-04-11` | `LEGACY` | `True` | Fits `research-provisional + operationally deployable` | Clean rediscovery with `--holdout-date 2026-01-01`, then fresh 2026 OOS scoring on the clean lineage |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | `2026-04-13` | `VALIDATOR_NATIVE` | `True` | Fits `research-provisional + operationally deployable` | Same as above; validator-native provenance does not rescue post-holdout discovery |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | `2026-04-11` | `LEGACY` | `True` | Fits `research-provisional + operationally deployable` | Clean rediscovery under Mode A plus clean OOS scoring |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | `2026-04-11` | `LEGACY` | `True` | Fits `research-provisional + operationally deployable` | Clean rediscovery under Mode A plus clean OOS scoring |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | `2026-04-11` | `LEGACY` | `True` | Fits `research-provisional + operationally deployable` | Clean rediscovery under Mode A plus clean OOS scoring |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | `2026-04-13` | `VALIDATOR_NATIVE` | `True` | Fits `research-provisional + operationally deployable` | Clean rediscovery under Mode A plus clean OOS scoring |

## Findings

1. None of the six fail the grandfathering path itself.
   - All six are compatible with the current repo stance that deployed lanes may remain operationally active while still being only research-provisional.

2. None of the six are close to escaping grandfathering on existing evidence.
   - Every lane has `discovery_date >= 2026-04-11`, well after the sacred holdout boundary `2026-01-01`.
   - That alone is enough to block OOS-clean status under the current rules.

3. `VALIDATOR_NATIVE` provenance is not a special escape hatch.
   - The two validator-native lanes still fail the same clean-holdout provenance requirement because the issue is discovery timing, not only the discovery engine.

4. `wf_passed = True` helps operational comfort, not institutional promotion.
   - Amendment 2.7 is explicit that walk-forward remains in-sample-only for these grandfathered rows.

## Practical map

Grandfathering status today:
- `6/6` fit the current provisional deployability carve-out
- `0/6` fit Mode A clean-proof status

Gap-closing evidence required for any lane:
1. re-run discovery with `--holdout-date 2026-01-01`
2. verify the same lane is rediscovered on the clean protocol
3. score 2026-01-01 onward as genuine forward OOS on that clean lineage
4. then re-open the institutional gate stack

## Verdict

`ALL_SIX_ARE_PROVISIONALLY_ELIGIBLE_FOR_GRANDFATHERING`

`ZERO_OF_SIX_ARE_CURRENTLY_ELIGIBLE_FOR_MODE_A_CLEAN_STATUS`
