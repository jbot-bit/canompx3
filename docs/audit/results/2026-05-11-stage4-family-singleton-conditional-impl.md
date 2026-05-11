# Stage 4 — family_singleton conditional downgrade (implementation)

**Author:** Claude Code
**Date:** 2026-05-11
**Stage doc:** `docs/runtime/stages/stage4-family-singleton-conditional.md`
**Prior:** Stage 3 § 4 floor spec locked; Stage 3.5 shelf-wide C8 backfill
APPLIED (commit `c91377fc`).
**Branch:** `stage4/family-singleton-conditional` (worktree
`.worktrees/stage4-family-singleton-conditional`).
**Base:** origin/main @ 62732518 (post-PR #257).
**Decision-owner:** User. Three pre-code decisions locked 2026-05-11.

## Scope

Truth-layer code change to `trading_app/deployability.py` implementing the
Disposition C family_singleton conditional downgrade. SINGLETON-status
rows now emit `family_singleton` as a `warning` (routed to
`CONTROLLED_LIVE_PILOT_CANDIDATE`) when they clear the binding criteria
from `pre_registered_criteria.md` Enforcement summary (line 480-494
post-amendments), and as `hard` otherwise.

## Verdict

**Implemented + tested + empirically regressed against gold.db.**
Pre-merge adversarial-audit gate per
`.claude/rules/adversarial-audit-gate.md` still pending.

## Changes

### `trading_app/deployability.py`

- **Imports (lines 19-26):** added `pipeline.data_era.is_micro`,
  `trading_app.chordia.chordia_verdict_label` +
  `chordia_verdict_allows_deploy`, and `trading_app.config.ALL_FILTERS`.
  All canonical sources (no parallel implementations).
- **`RETIRE_OR_PURGE_ISSUES` (line 97):** removed `family_singleton`
  from the set. PURGED stays. Comment in code documents the doctrine
  reason inline.
- **`CONTROLLED_PILOT_WARNINGS` (NEW, after line 113):** module-level
  set lifted from the inline allowlist that previously lived at
  line 618-621. Now includes `slippage_event_tail_pending`,
  `sr_alarm_watch_reviewed`, AND `family_singleton`. Each member is
  documented with the doctrine reason for membership.
- **`_singleton_clears_binding_criteria(row)` (NEW, after
  `_active_in_sql`):** pure function that evaluates the 6 binding
  criteria + C5 reported. Returns `(passes, failed_criterion_ids)`.
  C4 delegates to `chordia_verdict_label`; C10 delegates to
  `ALL_FILTERS[filter_type].requires_micro_data` + `is_micro(instrument)`.
  No re-encoding of canonical logic.
- **Family branch (now at lines 648-687):** previously emitted
  unconditional hard for SINGLETON; now branches on
  `_singleton_clears_binding_criteria()`. Hard issues carry a detail
  dict listing the failed criteria; warning issues carry a detail dict
  with the doctrine reference.
- **`controlled_warning_ids` (now at line 769):** replaced inline
  allowlist literal with reference to `CONTROLLED_PILOT_WARNINGS`.
- **SELECT widening (lines 174-175):** added `vs.sharpe_ratio` and
  `vs.era_dependent`. Both columns already exist on `validated_setups`
  (no migration needed).

### `tests/test_trading_app/test_deployability.py`

- `_row()` fixture extended with `sharpe_ratio=0.30` (clears Chordia
  BAND A at N=200), `era_dependent=False`, `fdr_significant=True`,
  `fdr_adjusted_p=0.01`. Existing 26 tests pass unchanged.
- 7 new SINGLETON tests added:
  1. `test_singleton_clearing_all_binding_criteria_is_controlled_pilot_warning`
     — passes all 6 binding criteria → verdict = CONTROLLED_LIVE_PILOT_CANDIDATE,
     issue severity = warning.
  2. `test_singleton_failing_chordia_band_is_still_hard_block` — mirrors
     the 5 MES Stage-2 candidates (low sharpe → low t-stat) → hard
     block with `C4_Chordia` in the failed-criteria detail.
  3. `test_singleton_with_era_dependent_true_is_hard_block` — C9 failure.
  4. `test_singleton_with_low_sample_size_is_hard_block` — C7 failure.
  5. `test_purged_family_remains_unconditional_hard_block` — sanity:
     Stage 4 change is SINGLETON-only; PURGED unchanged.
  6. `test_singleton_passing_routes_to_nearest_to_deployable_not_retire`
     — promotion-bucket routing assertion (consequence of removing
     `family_singleton` from `RETIRE_OR_PURGE_ISSUES`).
  7. `test_singleton_passing_emits_doctrine_reference_in_detail` —
     audit-trail completeness; warning detail dict carries the doctrine
     reason for downstream readers.

## Empirical regression (real gold.db, 2026-05-11)

`_singleton_clears_binding_criteria()` run against all 276 active
SINGLETONs:

| Outcome | Count |
|---|---|
| Pass all binding criteria → CONTROLLED_LIVE_PILOT_CANDIDATE | **33** |
| Hard-blocked, failing at least one criterion | **243** |

Failure-reason distribution on the 243 hard-blocked rows:

| Failed criterion | Count |
|---|---|
| C4_Chordia[FAIL_BOTH] (t < 3.00) | 174 |
| C4_Chordia[FAIL_CHORDIA] (3.00 ≤ t < 3.79, no theory) | 68 |
| C9_era_dependent (era_dependent=TRUE) | 19 |

C4 (Chordia banded t-stat) is doing the heavy lifting — 242 of 243
hard-block decisions fail on Chordia. This matches the Stage 3 § 2
finding that the 5 MES Stage-2 candidates have t ≈ 2.2-2.6 (BAND C).

### 5 MES Stage-2 candidates — actual verdicts under Stage 4

| strategy_id | passes | failed criteria |
|---|---|---|
| MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10 | False | `['C4_Chordia[FAIL_BOTH]']` |
| MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_S075 | False | `['C4_Chordia[FAIL_BOTH]']` |
| MES_US_DATA_1000_E2_RR1.5_CB1_COST_LT08_O15 | False | `['C4_Chordia[FAIL_BOTH]']` |
| MES_US_DATA_1000_E2_RR1.5_CB1_COST_LT10_O15 | False | `['C4_Chordia[FAIL_BOTH]']` |
| MES_US_DATA_1000_E2_RR1.5_CB1_ORB_G8_O15 | False | `['C4_Chordia[FAIL_BOTH]']` |

**The 5 MES Stage-2 candidates remain HARD-BLOCKED** under Stage 4
because they fail C4 Chordia (t too low). Stage 4 doctrine is now
codified; the 5 rows are still parked pending either (a) a per-strategy
PASS_PROTOCOL_A unlock with explicit theory citation, or (b) a
distinct workstream addressing the t-stat shortfall. Disposition C
unlock count for the original 5 MES targets is **0**, consistent with
Stage 3 § 4 / Stage 2 § 3.5 empirical predictions.

### 33 MNQ-singleton-pass rows — the new "controlled pilot" surface

All 33 are MNQ active rows. Sample (first 5):

- MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_O15 (sharpe=0.31, N=173)
- MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P50 (sharpe=0.17, N=731)
- MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08 (sharpe=0.21, N=759)
- MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08_O15 (sharpe=0.27, N=232)
- MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10 (sharpe=0.17, N=1019)

These rows now route to `CONTROLLED_LIVE_PILOT_CANDIDATE` instead of
`BLOCKED_FAMILY_FRAGILE`. **Capital impact is zero** because the lane
allocator reads `validated_setups.status`, not the deployability verdict
string (per Stage 2 blast-radius audit; Stage 4 blast-radius re-confirmed).
Lane-correlation gates downstream would collapse most of the 33 against
existing deployed CME_PRECLOSE-cluster siblings before any real
deployment.

## Capital safety

- Lane allocator (`scripts/tools/rebalance_lanes.py:109`) reads
  `validated_setups.status` — unchanged.
- `lane_allocation.json` — untouched.
- Broker state, schema, validated_setups membership — untouched.
- The 3 currently-deployed MNQ lanes (all WHITELISTED, not SINGLETON)
  are unaffected by the SINGLETON-only conditional.
- Adversarial-stress gate (`scripts/tools/adversarial_stress_gate.py`)
  and full-shelf audit (`scripts/tools/full_shelf_deployability_audit.py`)
  consume the verdict surface unchanged.

## Verification

- `pytest tests/test_trading_app/test_deployability.py -q` →
  **33 passed in 0.44s** (was 26 before; +7 new tests).
- `pytest tests/test_trading_app/test_deployability.py
  tests/test_tools/test_adversarial_stress_gate.py
  tests/test_tools/test_backfill_deployability_evidence.py -q` →
  **48 passed in 0.41s**.
- `python pipeline/check_drift.py` → **123 checks passed, 0 failures,
  0 skipped, 19 advisory**.
- Dead code sweep: `grep -rn "controlled_warning_ids = {"` —
  exactly one match, in the file just edited.
- Canonical-delegation sweep: C4 calls `chordia_verdict_label`; C10
  calls `is_micro` + `ALL_FILTERS[...].requires_micro_data`; no
  parallel logic.

## Doctrine grounding

- `docs/institutional/pre_registered_criteria.md` Enforcement summary
  (line 480-494): the binding criteria list comes from here, NOT the
  pre-amendment Acceptance matrix at line 290-303 (which has a separate
  doctrine-fix follow-up per Stage 3 § 5).
- Amendment 2.1 (2026-04-07): C5 DSR downgraded to CROSS-CHECK ONLY.
  Honoured: `_singleton_clears_binding_criteria` requires `dsr_score
  IS NOT NULL` (computed-and-reported) but does NOT enforce ≥ 0.95.
- Amendment 2.2 (Chordia banding): delegated to canonical
  `chordia_verdict_label` helper. SINGLETONs get `has_theory=False`
  by construction (Pathway A family promotion), so they need BAND A
  (t ≥ 3.79) to pass.
- Disposition C (Stage 2 user pick): floor enforces the locked
  criteria; CONTROLLED_LIVE_PILOT_CANDIDATE route honours the
  Bailey/Carver family-corroboration asymmetry.

## Bloomberg-grade provenance

Every load-bearing claim in this document has:
- Timestamp: 2026-05-11.
- Source layer: `validated_setups.{c8_oos_status, fdr_significant,
  fdr_adjusted_p, sharpe_ratio, era_dependent, wfe, sample_size,
  filter_type, dsr_score}` + `edge_families.robustness_status`.
- Canonical helpers: `chordia_verdict_label`, `is_micro`,
  `ALL_FILTERS[...].requires_micro_data`.
- Doctrine refs: `pre_registered_criteria.md:480-494` (Enforcement
  summary, post-amendments authoritative).
- Tool: `_singleton_clears_binding_criteria` (new pure function
  in this commit; tested via 7 new fixtures).

## No pigeon-holing

Stage 4 was scoped to the family_singleton question. Empirical
regression intentionally evaluated **all 276 active SINGLETONs**, not
just the 5 MES Stage-2 candidates. Surfaced 33 MNQ rows that newly
route to CONTROLLED_LIVE_PILOT_CANDIDATE — a universe-wide finding.

## Bailey-LdP MinBTL: this stage ran 0 new trials.

The empirical regression is a pure read against existing
`validated_setups` rows; no new backtests, no new strategies.

## Reproducibility

```python
import logging
logging.disable(logging.CRITICAL)
import duckdb
from pipeline.paths import GOLD_DB_PATH
from trading_app.deployability import (
    _load_candidate_rows,
    _singleton_clears_binding_criteria,
)

with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
    rows = _load_candidate_rows(con, scope='all-active', profile_id=None, instruments=None)

singletons = [r for r in rows if r.get('robustness_status') == 'SINGLETON']
print(f'Active SINGLETONs: {len(singletons)}')

pass_count = sum(1 for r in singletons if _singleton_clears_binding_criteria(r)[0])
print(f'Pass binding criteria: {pass_count}')

# Per-instrument breakdown
from collections import Counter
inst_pass = Counter(r['instrument'] for r in singletons if _singleton_clears_binding_criteria(r)[0])
print(f'Pass by instrument: {dict(inst_pass)}')
```

## Limitations

- The `has_theory=False` default for SINGLETONs disqualifies BAND B
  (3.00 ≤ t < 3.79 with theory). A future per-strategy capital-review
  workflow could supply `has_theory=True` per row to unlock specific
  BAND B candidates. Out of scope for Stage 4.
- C9 era stability uses the stored `era_dependent` flag rather than
  re-running per-era ExpR computation. If `era_dependent` is stale,
  the floor may admit a row that would now fail era stability under
  fresh data. Cross-check: Stage 3.5 backfilled C8 freshly; C9 has no
  analogous backfill workflow yet (possible debt-ledger entry).
- C5 enforces "dsr_score IS NOT NULL" (reported). Per Amendment 2.1,
  this is the minimum operationalisation; once N_eff is formally
  solved (ONC algorithm, deferred), C5 may become binding and the
  floor would tighten.
- Doctrine fix for `pre_registered_criteria.md:290-303` intra-doc
  inconsistency (Stage 3 § 5) is still pending and remains a separate
  workstream.

## Adversarial-audit gate (2026-05-11, CONDITIONAL → fix → PASS)

Per `.claude/rules/adversarial-audit-gate.md` the `evidence-auditor`
subagent was dispatched on commit `6fd3bb0b` in an independent context.

**Audit verdict: CONDITIONAL.** One critical finding closed in the
follow-up commit:

### Finding C5-GATING-CONTRADICTION (CONDITIONAL)

The auditor caught a docstring/code contradiction in
`_singleton_clears_binding_criteria`:

- Docstring at line 189-194: "C5 is included in the 'reported' sense
  only — its absence does not fail the floor."
- Inline comment at line 223-226: "reported, NOT gating per Amendment
  2.1."
- BUT code at line 227-228:
  ```python
  if row.get("dsr_score") is None:
      failed.append("C5_DSR_uncomputed")
  ```
  with `passes = len(failed) == 0` at line 260.

So a NULL `dsr_score` was actually gating the floor — a latent bug
contrary to `pre_registered_criteria.md` Amendment 2.1 (line 367-370,
"CROSS-CHECK ONLY"). The bug had zero current empirical impact (all
276 active SINGLETONs have non-NULL `dsr_score`), but would have
hard-blocked any future SINGLETON with NULL DSR — exactly the kind of
silent doctrine violation the audit gate exists to catch.

### Fix applied (follow-up commit)

- `_singleton_clears_binding_criteria` now returns a 3-tuple:
  `(passes, failed_criterion_ids, dsr_reported)`. C5 NULL no longer
  appends to `failed`; instead it surfaces as the `dsr_reported`
  boolean for audit-trail visibility.
- Family-branch caller updated to consume the 3-tuple and emit
  `c5_dsr_reported` + `c5_status` in both the warning-branch and
  hard-branch detail dicts.
- Docstring rewritten to explicitly state C5 is NOT in the gating
  set, with a note referencing the audit incident so the fix cannot
  silently regress.
- 2 new tests added:
  - `test_singleton_with_null_dsr_still_passes_when_binding_criteria_cleared`
  - `test_singleton_with_null_dsr_AND_failing_chordia_is_still_hard`

### Other audit findings (resolved)

- **Lane allocator independence (INFERRED → VERIFIED):** the auditor
  flagged that the implementer's "capital impact zero" claim was
  INFERRED rather than traced. Post-audit grep across
  `scripts/tools/rebalance_lanes.py` and `trading_app/lane_allocator.py`
  for `build_deployability_audit`, `StrategyDeployability`,
  deployability verdict strings, and `CONTROLLED_LIVE_PILOT_CANDIDATE`
  returned **zero hits**. Capital independence is now VERIFIED.
- **NULL handling for sharpe_ratio, fdr_significant, era_dependent:**
  reviewed; all fail-closed, intentional, doctrine-aligned.
- **Multi-warning coexistence:** `CONTROLLED_PILOT_WARNINGS` set
  membership correctly handles multiple coexisting warnings; verdict
  defaults to `CONTROLLED_LIVE_PILOT_CANDIDATE` when any controlled
  warning present and no hard.
- **SELECT widening compatibility:** the new `sharpe_ratio` +
  `era_dependent` columns are additive; no destructure consumers
  depend on positional ordering.

### Post-fix verification

- `pytest tests/test_trading_app/test_deployability.py` — **35 passed**
  (was 33 before; +2 new C5-NULL tests).
- `pytest tests/test_trading_app/test_deployability.py
  tests/test_tools/test_adversarial_stress_gate.py
  tests/test_tools/test_backfill_deployability_evidence.py` —
  **50 passed**.
- `python pipeline/check_drift.py` — **123 checks pass, 0 fail**.
- Re-ran empirical regression: **33 / 276 SINGLETONs pass** (unchanged;
  all MNQ; 0 MES). The 5 MES candidates still fail on C4 Chordia. The
  fix is backward-compatible because no current row has NULL DSR.

**Audit verdict post-fix: PASS.** Ready to open PR.

## Next actions

1. ~~Adversarial-audit gate~~ **DONE (CONDITIONAL → PASS).**
2. **Open PR** — single PR scope per Stage 4 stage doc; full body
   matches Stage 1 PR #258 pattern.
3. **Stage 2 + Stage 3 docs PRs** — separate docs-only PRs awaiting
   bundling decision (can merge with Stage 4 or separately).
4. **Stage 1 PR #258** — still awaiting merge.

## Decision asked of user

After adversarial audit returns: review the audit verdict and approve
PR open. If audit returns CONDITIONAL or FAIL, address findings before
proceeding.
