---
task: DSR reference-universe lock drift check — mechanize Criterion 5 Amendment 3.5 (declared-schema + check)
mode: IMPLEMENTATION
updated: 2026-05-29
scope_lock:
  - docs/institutional/pre_registered_criteria.md
  - docs/institutional/hypothesis_registry_template.md
  - pipeline/check_drift.py
  - tests/test_tools/test_dsr_universe_lock_drift_check.py
blast_radius: |
  ## Blast Radius
  - docs/institutional/pre_registered_criteria.md — ADD "Declared schema" subsection under Amendment 3.5 naming the `criterion_5` prereg block (4 fields + allowed derivation labels). Doctrine authority the check cites. Existing text UNCHANGED.
  - docs/institutional/hypothesis_registry_template.md — ADD optional `criterion_5` block with comments (required ONLY when claiming DSR-clearance). Authors already routed here by loader error messages.
  - pipeline/check_drift.py — ADD one check `check_dsr_universe_lock_declared`, modeled on `check_chordia_result_threshold_matches_prereg` / Amendment-3.4 gate: iterate active preregs, EXCLUDE drafts/, trigger = presence of a `criterion_5` block, validate completeness (family non-empty str; pre_2026_only + failures_and_siblings_included explicit-bool true; effective_trials positive int; derivation in allowed-set). Missing/malformed → BLOCK. `hypotheses_dir` test seam. Register in check registry.
  - tests/test_tools/test_dsr_universe_lock_drift_check.py — NEW. Injection cases proving the check fires per-field.
  - Reads: prereg YAML files only. NO DB connection. Writes: NONE.
  - Consumers NOT edited: hypothesis_loader.py (no loader echo this stage — YAGNI; NYSE_PREOPEN runner pulls it in when needed). trading_app/dsr.py (ONC helper, separate uncommitted stage). deployability.py (DSR cross-check, unaffected).
acceptance:
  - "python pipeline/check_drift.py passes (expect 0 NEW violations — no current prereg carries a criterion_5 block, so the check is dormant on the existing corpus)"
  - "Injection: a prereg with a complete valid criterion_5 block → no violation"
  - "Injection: criterion_5 block present but each individual field missing/wrong-typed/false → BLOCK (one case per field — proves the check tests each dimension)"
  - "Injection: prereg with NO criterion_5 block → no violation (dormant trigger)"
  - "Injection: a drafts/ file with a malformed criterion_5 block → no violation (drafts excluded)"
  - "pytest tests/test_tools/test_dsr_universe_lock_drift_check.py passes"
  - "grep confirms the check is registered in the check registry"
---

# DSR universe-lock drift check

## Purpose
Amendment 3.5 locked the DSR V[SR] universe (pre-2026, family-scoped, all-siblings+failures,
no winner-filter) after the same candidate scored 0.000↔0.982 by universe choice alone.
Today it is REVIEW-ENFORCED ONLY — no prereg field declares the universe, nothing verifies it.

## Honest grounding (corrected from first design pass)
- `DSR_FIXED_UNIV_CLEAR` is a STATUS-TAXONOMY string living ONLY in pre_registered_criteria.md
  (verified: grep finds it in no prereg YAML). Keying the trigger on it = dead-on-arrival check.
- Therefore the declared schema (`criterion_5` block) IS the trigger. The block is greenfield;
  defined here together with the check so the check is provably LIVE (injection forces a BLOCK).
- A registered check that can never fire is WORSE than no check (false "covered" signal) —
  integrity-guardian § 3 / § 7. This design avoids that by injection-proving every field.

## Why declaration-discipline, NOT recompute-and-compare (Take 2 over Take 3)
The amendment assigns V[SR] COMPUTATION to the runner via the canonical helper. The drift
check's job is to enforce the universe is DECLARED + pinned, not to re-run the science
(slow, DB-bound, duplicates runner responsibility). Recompute-compare is future hardening.

## Grounding
- docs/institutional/pre_registered_criteria.md Amendment 3.5 (the lock)
- pipeline/check_drift.py check_chordia_result_threshold_matches_prereg + Amendment-3.4 gate (idiom precedent)
- trading_app/hypothesis_loader.py (fail-closed / explicit-bool / allowed-set validation idiom)
- memory feedback_dsr_reference_universe_free_parameter_2026_05_29.md

## NOT in scope
- Loader metadata echo (YAGNI — runner pulls it when built).
- Recompute V[SR] in the check.
- Flipping any prereg to claim clearance (research decision).
- The ONC dsr.py helper (separate uncommitted stage — will commit alongside).
