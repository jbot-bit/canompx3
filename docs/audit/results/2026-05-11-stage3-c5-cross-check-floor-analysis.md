# Stage 3 — family_singleton floor under C5-as-cross-check

**Author:** Claude Code (read-only analysis; no DB or production-code writes)
**Date:** 2026-05-11
**Stage doc:** `docs/runtime/stages/stage3-c5-cross-check-floor.md`
**Prior:** Stage 2 disposition C locked + 3rd-pass corrections
(`2026-05-11-family-singleton-doctrine-analysis.md`).
**Decision-owner:** User. This doc enumerates the corrected floor and
empirical reality; it does not implement code.

## Scope

This document enumerates the BINDING criteria a Disposition C
"family_singleton conditional downgrade" floor must honour, given that
`pre_registered_criteria.md` Amendment 2.1 already downgraded C5 (DSR)
from binding to cross-check. It then audits the 5 MES candidates and the
276-row singleton universe against those binding criteria and surfaces an
intra-doc inconsistency in the locked criteria file itself.

## Verdict

A literature- and code-grounded Disposition C floor for family_singleton
is enumerated in § 4. It binds on **8 BINDING criteria + 2 CROSS-CHECK
criteria reported but not gating**. Empirical pass-count on the singleton
universe **today** is **zero** — but the limiting blocker is C8 OOS
status uncomputed across the shelf, not DSR. This is broader than
Stage 3 alone can fix; recommended decomposition is in § 6.

## 1. The BINDING criteria — verbatim from pre_registered_criteria.md

Per the enforcement-summary table at `pre_registered_criteria.md:480-494`
(post-amendments authoritative version), the BINDING criteria are:

| # | Criterion | Threshold | Enforcement | validated_setups column |
|---|---|---|---|---|
| C1 | Pre-registration | file exists at `docs/audit/hypotheses/` | BINDING | (provenance: not a column; checked via hypothesis-file lookup) |
| C2 | MinBTL | N_trials ≤ 300 (clean) or 2000 (proxy) | BINDING | `discovery_k` (per-row trial budget at promotion time) |
| C3 | BH FDR | q < 0.05 on pre-registered family K | BINDING | `fdr_significant` + `fdr_adjusted_p` |
| C4 | Chordia t-stat | banded per Amendment 2.2 | BINDING via banding | `sharpe_ratio` → derive t-stat via `compute_t_from_sharpe()` |
| C5 | DSR | computed + reported | **CROSS-CHECK ONLY** (Amendment 2.1) | `dsr_score` (informational) |
| C6 | WFE | WFE ≥ 0.50 | BINDING | `wfe` |
| C7 | Sample size | N ≥ 100 trades | BINDING | `sample_size` |
| C8 | 2026 OOS / forward | depends on holdout policy | **CONTINGENT** (Amendment 2.3) | `c8_oos_status` (status string) |
| C9 | Era stability | no era ExpR < -0.05 (N ≥ 50) | BINDING | `era_dependent` + per-era stats |
| C10 | Data era compat | volume filters MICRO-only | BINDING | inferred from `filter_type` |
| C11 | Account death MC | 90-day survival ≥ 70% | BINDING (at deployment) | (deployment-time check; not in validated_setups) |
| C12 | Shiryaev-Roberts | active drift monitor | BINDING (post-deployment) | (lifecycle / sr_review_registry) |

### Why "BINDING criteria count" varies depending on framing

- For a strategy that is *promotable to research-provisional*, the
  binding set is C1, C2, C3, C4, C6, C7, C9, C10 — eight criteria.
  This is `pre_registered_criteria.md:495` directly: "A strategy passing
  1-4, 6-7, 9-10 is **research-provisional**."
- For a strategy that is *operationally deployable*, add C11. Total: nine.
- For *production-grade institutional proof*, add C8 (contingent on
  holdout policy per Amendment 2.7 — currently Mode A) and C12 active.
  Total: eleven.

For Disposition C's family_singleton floor, the relevant tier is at
minimum **operationally deployable** (C1-C4, C6-C7, C9-C11) plus C5
computed-and-reported. C8 binds *for deployment* under Mode A holdout
policy when the row's `last_trade_day` crosses the holdout boundary.

## 2. Per-row evidence on the 5 MES candidates

Audited 2026-05-11 against `validated_setups` directly.

| Field | row 1 | row 2 | row 3 | row 4 | row 5 |
|---|---|---|---|---|---|
| strategy_id | MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10 | MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_S075 | MES_US_DATA_1000_E2_RR1.5_CB1_COST_LT08_O15 | MES_US_DATA_1000_E2_RR1.5_CB1_COST_LT10_O15 | MES_US_DATA_1000_E2_RR1.5_CB1_ORB_G8_O15 |
| sample_size | 266 | 269 | 905 | 1109 | 1033 |
| ExpR (IS) | 0.1258 | 0.1120 | 0.1017 | 0.0866 | 0.0896 |
| OOS ExpR | 0.1103 | 0.0914 | 0.1157 | 0.0936 | 0.1018 |
| WFE | 0.613 | 0.5545 | 1.1441 | 1.2624 | 1.2778 |
| fdr_adjusted_p | 0.040 | 0.037 | 0.023 | 0.032 | 0.033 |
| fdr_significant | TRUE | TRUE | TRUE | TRUE | TRUE |
| sharpe_ratio | 0.138 | 0.139 | 0.086 | 0.074 | 0.076 |
| dsr_score | 8.3e-05 | 6.0e-05 | 0 | 0 | 0 |
| **c8_oos_status** | **NULL** | **NULL** | **NULL** | **NULL** | **NULL** |
| slippage_validation_status | NULL | NULL | NULL | NULL | NULL |
| validation_pathway | family | family | family | family | family |
| discovery_k | 8856 | 8856 | 9568 | 9568 | 9568 |
| years_tested | 6 | 6 | 6 | 6 | 6 |
| trade_day_count | 308 | 308 | 1127 | 1363 | 1271 |
| era_dependent | (not surfaced this query) |

### Pass/Fail against the BINDING set

| Criterion | rows passing | rows failing | rows unknown/missing |
|---|---|---|---|
| C1 pre-registration | 0 verified | 0 verified | 5 (not checked — VALIDATOR_NATIVE provenance means promoted before pre-reg discipline) |
| C2 MinBTL N ≤ 300 | 0 | 5 (discovery_k 8856-9568 far exceeds 300) | 0 |
| C3 BH FDR q < 0.05 | 5 | 0 | 0 |
| C4 Chordia t-stat banded | requires computation from per-trade SR + N; rough check: t ≈ SR × √N. Row 1: 0.138 × √266 ≈ 2.25 (BAND C per Amendment 2.2); rows 3-5: 0.086 × √905 ≈ 2.59 (BAND C). All below t≥3.00. | | |
| C5 DSR cross-check | computed (values stored) | n/a (cross-check) | 0 |
| C6 WFE ≥ 0.50 | 5 | 0 | 0 |
| C7 N ≥ 100 | 5 | 0 | 0 |
| C8 OOS | 0 PASSED | 0 FAILED | 5 NULL (status uncomputed) |
| C9 era stability | (not surfaced this query — would need per-era ExpR) | | |
| C10 MICRO-only volume filter | 4 (filter_types are COST_LT* and ORB_G8 — non-volume) + 1 (ORB_G8 is not a volume filter) → 5 PASS | 0 | 0 |
| C11 account MC | (deployment-time check, not pre-screened) | | |

### What blocks each row right now

- **C2 MinBTL** — all 5 rows have discovery_k > 8000. Pre-Phase-0
  promotion provenance means they were promoted under brute-force
  discovery that violates Bailey 2013 MinBTL by ~30×. Whether this is
  a binding gate or a grandfathered-pre-Phase-0 condition is a
  doctrine question, not a row-level question.
- **C4 Chordia t-stat banded** — all 5 rows fail t ≥ 3.00 by rough
  computation. Exact t-stat re-derivation per Amendment 2.2 may give a
  different banded verdict; this is a per-row audit not yet run.
- **C8 OOS** — all 5 rows have `c8_oos_status IS NULL`. Stage 3.5 (C8
  backfill) addresses this; outcomes unknown a priori.

So Disposition C's floor under the BINDING criteria correctly enforced
unblocks **zero** of the 5 rows immediately. Even after Stage 3.5 (C8
backfill), the C2 + C4 blockers stand. Without a C2 grandfather decision,
the 5 rows cannot deploy regardless of family_singleton disposition.

## 3. 276-singleton universe pass-count

`SELECT vs.instrument, COUNT(*), SUM(c3_pass), SUM(c6_pass), SUM(c7_pass), SUM(c8_pass), SUM(c8_null), SUM(all_4_pass)`:

| Instrument | total | C3 | C6 | C7 | C8_PASSED | C8_NULL | C3+C6+C7+C8_PASSED |
|---|---|---|---|---|---|---|---|
| MES | 22 | 22 | 22 | 22 | 0 | 22 | 0 |
| MNQ | 254 | 254 | 254 | 254 | 0 | 254 | 0 |

Interpretation: **every singleton row passes C3+C6+C7 individually**
(they had to, to clear `status=active`). The 100% pass-rate reflects that
`strategy_validator` already enforces these. **Zero** singleton rows have
C8 PASSED — C8 is universally uncomputed on singletons.

## 4. The corrected Disposition C floor — concrete spec for Stage 4

Stage 4 (code change to `trading_app/deployability.py`) consumes this
spec. The conditional downgrade rule:

> `family_singleton` remains a HARD blocker UNLESS the strategy clears
> ALL of the following AS QUERIED FROM `validated_setups` AT GATE-EVAL
> TIME:
>
> 1. **C3 BH FDR:** `fdr_significant = TRUE AND fdr_adjusted_p < 0.05`.
> 2. **C4 Chordia t-stat:** computed t-stat must fall in BAND A or B per
>    Amendment 2.2 in `pre_registered_criteria.md`. (Stage 4 must
>    delegate t-stat computation to a canonical helper; see § 5 doctrine
>    fix.)
> 3. **C6 WFE:** `wfe IS NOT NULL AND wfe >= 0.50`.
> 4. **C7 sample size:** `sample_size >= 100`.
> 5. **C9 era stability:** `era_dependent = FALSE` OR explicit per-era
>    no-drag check.
> 6. **C10 data era compat:** `filter_type` does not match the
>    pre-MICRO-era volume-filter prefix list (non-MICRO volume filters
>    are pre-2024 contaminated per the data-era compatibility rule).
> 7. **C5 DSR cross-check:** `dsr_score IS NOT NULL` (computed and
>    stored) — reported but not gating per Amendment 2.1.
>
> Criteria NOT enforced at the family_singleton-floor layer because they
> are enforced elsewhere by deployability.py existing logic:
> - C2 MinBTL — pre-Phase-0 grandfather doctrine separate workstream.
> - C8 OOS — already a hard blocker via `c8_missing`/`c8_not_passed`.
> - C11 account MC — deployment-time check, not pre-screened on the
>   `validated_setups` shelf.
> - C12 SR monitor — post-deployment lifecycle gate.

If the user picks a different framing for the floor, Stage 3 closes with
that framing.

## 5. Intra-doc inconsistency in pre_registered_criteria.md

The locked file carries two tables that disagree on C5:

- **Line 290-303 "Acceptance matrix"** — column "Required" = "YES" for
  C5 with threshold "DSR > 0.95".
- **Line 480-494 "Enforcement summary"** — Enforcement = "CROSS-CHECK
  ONLY (Amendment 2.1)" for C5.

The Enforcement summary is the post-amendments-authoritative version;
the Acceptance matrix is pre-amendment historical. Per
`docs/governance/document_authority.md` (presumed) the post-amendment
table wins, but the unamended early table is a citation trap: anyone
reading top-to-bottom will hit line 296 "C5 DSR > 0.95 — YES required"
before they reach the amendments at line 358+, and may make a doctrine
decision on the wrong line.

Recommended doctrine fix (SEPARATE WORKSTREAM, not Stage 3 itself):
amend `pre_registered_criteria.md` so the Acceptance matrix at line
290-303 reflects the post-amendment enforcement state — either by
inlining the amendment notes per-row or by removing the matrix and
referring forward to the Enforcement summary. File:
`pre_registered_criteria.md:290-303`. Severity: MEDIUM (audit-trail
clarity, not capital-impacting because Stage 2's third-pass caught it).

A second smaller inconsistency: line 294 says "C8 2026 OOS: OOS ExpR ≥ 0
AND ≥ 0.40 × IS ExpR — YES required", while line 489 says "C8 2026 OOS:
depends on holdout policy — CONTINGENT (Amendment 2.3)". Same
post-amendment-supersedes-acceptance-matrix pattern.

## 6. Broader Stage 3 finding — C8 uncomputed across the active shelf

Universe-level: only **3 of 847** active rows have C8_PASSED; **844** have
`c8_oos_status IS NULL`. The 3 are all currently-deployed MNQ lanes — and
one of those (`MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12`) has c8=None despite
being live. This is **doctrine drift larger than Stage 3** — but it
constrains the practical effect of Stage 4's family_singleton change:

- Without shelf-wide C8 backfill, Disposition C unlocks 0 rows even if
  family_singleton becomes conditional and C2 grandfather is resolved.
- C8 backfill is a separate workstream (Stage 3.5 + parallel for the
  broader shelf).
- Stage 4 should still proceed because it codifies the doctrine; the
  empirical unlock count depends on the parallel C8 backfill.

## 7. What this analysis does NOT do

- Does NOT modify `trading_app/deployability.py`, `pre_registered_criteria.md`,
  `validated_setups`, or any other production file.
- Does NOT compute exact Chordia t-stats per Amendment 2.2 — the rough
  t = SR × √N approximation is used for indicative pass/fail framing only.
- Does NOT decide the C2 MinBTL grandfather question for pre-Phase-0
  rows. That is a separate doctrine workstream.
- Does NOT run C8 backfill on any row. Stage 3.5 + broader workstream.
- Does NOT fix the intra-doc inconsistency in `pre_registered_criteria.md`.
  Flagged for separate workstream.
- Does NOT consider lane-correlation effects for any unlocked singletons.
  That is a pre-deployment check, not a floor enumeration.

## Limitations

Stage 3 is read-only and does not enumerate every edge case. The
mapping from C1-C12 to `validated_setups` columns assumes the current
schema; schema drift would invalidate the spec. C4 t-stat computation
is approximated; Stage 4 implementer must call the canonical helper.
C9 era-stability not surfaced this query.

## Bailey-LdP MinBTL: this audit ran 0 trials. No brute force.

Read-only against `gold.db`, scoped to `validated_setups` joined to
`edge_families`.

## Reproducibility

```python
import duckdb
con = duckdb.connect('C:/Users/joshd/canompx3/gold.db', read_only=True)

# § 2 — 5 MES candidates row-level evidence
con.execute("""
    SELECT vs.strategy_id, vs.sample_size, vs.expectancy_r, vs.oos_exp_r, vs.wfe,
           vs.fdr_significant, vs.fdr_adjusted_p, vs.dsr_score, vs.sharpe_ratio,
           vs.c8_oos_status, vs.slippage_validation_status, vs.validation_pathway,
           vs.years_tested, vs.discovery_k, vs.promotion_provenance,
           vs.first_trade_day, vs.last_trade_day, vs.trade_day_count
    FROM validated_setups vs
    WHERE strategy_id IN (
        'MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10',
        'MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_S075',
        'MES_US_DATA_1000_E2_RR1.5_CB1_COST_LT08_O15',
        'MES_US_DATA_1000_E2_RR1.5_CB1_COST_LT10_O15',
        'MES_US_DATA_1000_E2_RR1.5_CB1_ORB_G8_O15'
    )
    ORDER BY vs.strategy_id
""").fetchall()

# § 3 — 276 singleton universe pass-count
con.execute("""
    SELECT vs.instrument, COUNT(*) AS total,
           SUM(CASE WHEN vs.fdr_significant AND vs.fdr_adjusted_p < 0.05 THEN 1 ELSE 0 END) AS c3_pass,
           SUM(CASE WHEN vs.wfe >= 0.50 THEN 1 ELSE 0 END) AS c6_pass,
           SUM(CASE WHEN vs.sample_size >= 100 THEN 1 ELSE 0 END) AS c7_pass,
           SUM(CASE WHEN vs.c8_oos_status = 'PASSED' THEN 1 ELSE 0 END) AS c8_pass,
           SUM(CASE WHEN vs.c8_oos_status IS NULL THEN 1 ELSE 0 END) AS c8_null,
           SUM(CASE WHEN vs.fdr_significant AND vs.fdr_adjusted_p < 0.05
                          AND vs.wfe >= 0.50 AND vs.sample_size >= 100
                          AND vs.c8_oos_status = 'PASSED' THEN 1 ELSE 0 END) AS all_4_pass
    FROM validated_setups vs JOIN edge_families ef ON vs.family_hash=ef.family_hash
    WHERE vs.status='active' AND ef.robustness_status='SINGLETON'
    GROUP BY 1 ORDER BY 1
""").fetchall()

# § 6 — C8 universe-wide coverage
con.execute("""
    SELECT vs.instrument, COUNT(*) AS total,
           SUM(CASE WHEN vs.c8_oos_status IS NULL THEN 1 ELSE 0 END) AS c8_null,
           SUM(CASE WHEN vs.c8_oos_status = 'PASSED' THEN 1 ELSE 0 END) AS c8_pass
    FROM validated_setups vs WHERE vs.status='active'
    GROUP BY 1 ORDER BY 1
""").fetchall()
```

## Decision asked of user

1. Accept § 4's BINDING-criteria spec as Stage 4's input? Or amend.
2. Approve § 5's recommendation (separate workstream) to amend the
   acceptance matrix in `pre_registered_criteria.md` so it matches the
   enforcement summary?
3. Approve § 6's framing that Stage 4 proceeds despite C8 backfill being
   a parallel workstream, OR hold Stage 4 until shelf-wide C8 backfill
   completes?
