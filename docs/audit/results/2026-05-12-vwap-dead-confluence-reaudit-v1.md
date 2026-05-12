# VWAP dead-confluence verdict procedural re-audit v1

**Pre-reg:** `docs/audit/hypotheses/2026-05-12-vwap-dead-confluence-reaudit-v1.yaml`
**Original result:** `docs/audit/results/2026-04-18-vwap-comprehensive-family-scan.md`
**Original runner:** `research/vwap_comprehensive_family_scan.py`
**Mode A holdout:** `2026-01-01`

## Verdict

**PASS_STANDS**

The 2026-04-18 VWAP family DOCTRINE-CLOSED verdict stands as a procedural research verdict. This audit did not rerun the consumed family scan.

## Checklist

| Check | Status | Evidence |
|---|---|---|
| primary_artifacts_exist | PASS | Found current prereg plus original VWAP prereg/result/runner files. |
| canonical_layers_only | PASS | Original runner loads orb_outcomes JOIN daily_features through GOLD_DB_PATH. |
| temporal_validity | PASS | VWAP source uses strict pre-ORB bars and result marks VWAP filters RULE 6.1 SAFE. |
| mode_a_holdout | PASS | Current prereg holdout=2026-01-01; original result reports IS trading_day < 2026-01-01. |
| k_family_bh_fdr | PASS | Original result reports K_family BH-FDR with 0 pass. |
| oos_dir_match | PASS | Original result states OOS one-shot consumption and dir_match requirement. |
| tautology_or_overlap_screen | PASS | Original result reports T0 tautology flags; runner delegates filters through research.filter_utils. |
| original_dead_verdict_present | PASS | Original result has 0 H1 survivors and VWAP family DOCTRINE-CLOSED verdict. |
| no_rerun_of_consumed_oos | PASS | Original runner refuses rerun when result exists due to one-shot OOS consumption. |
| h3_positive_control_caveat | CAVEATED_PASS | Original result documents H3 baseline specification error while preserving independent K1 death verdict. |

## Canonical Sanity Query

Bounded read-only sanity query on the known L6 VWAP cell:

| Cell | N total IS | N on-signal IS | ExpR on-signal IS |
|---|---:|---:|---:|
| MNQ US_DATA_1000 O15 E2 RR1.5 CB1 VWAP_MID_ALIGNED | 933 | 497 | +0.1919 |

This is a sanity check only. It is not a family rescan and does not change the original K-family verdict.

## Scope Discipline

- No write to `gold.db`.
- No write to `experimental_strategies` or `validated_setups`.
- No rerun of `research/vwap_comprehensive_family_scan.py`.
- No OOS retuning.

## Decision

VWAP remains closed as a broad confluence family from this primary scan.
The known live/shelf VWAP exact lanes remain exact-lane facts; they do not reopen the family.
Next priority in the user's list should be volume-at-break or prior-day geometry, using the same prereg + bounded-runner pattern.
