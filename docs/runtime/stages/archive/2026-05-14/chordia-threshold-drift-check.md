task: Add drift check `check_chordia_result_threshold_matches_prereg` + register it; update prereg-writer prompt with theory_citation field-presence semantics note.
mode: IMPLEMENTATION
slug: chordia-threshold-drift-check
classification: judgment

## Scope Lock
- pipeline/check_drift.py
- docs/prompts/prereg-writer-prompt.md

## Blast Radius
- `pipeline/check_drift.py` (additive only — new function + 1 registry tuple; no behaviour change to existing 125 checks; advisory for pre-2026-05-12 files, binding for >= sentinel).
- `docs/prompts/prereg-writer-prompt.md` (single line addition under § OUTPUT RULES; no schema removals).
- Reads: filesystem (`docs/audit/hypotheses/*.yaml`, `docs/audit/results/*.md`); no DB.
- Writes: none beyond the two scope-locked files.
- Tests: optional smoke (drift run shows the new check passing).

## Acceptance
1. `python pipeline/check_drift.py` exits 0 and prints the new Check line with PASS or ADVISORY (non-blocking) on the existing MGC L_M prereg/result pair (advisory only because that pair is dated 2026-05-12 sentinel — borderline; the date-sentinel comparison uses `>=` so it IS binding, but the MGC L_M case will surface as a real violation per acceptance #2).
2. The MGC L_M prereg/result mismatch SHOULD surface — the check exists specifically to catch this class. Resolution path: either amend the prereg's `chordia_threshold_basis` to reflect the actual t>=3.00 applied (loader correctly inferred theory from non-empty `theory_citation` field), OR strip the prose from `theory_citation` and re-run. **Out of scope for this stage** — recording as a real first-firing of the check is institutionally correct; the misclassification was already retracted in commit `5cafcb90`.
3. Drift check sentinel-date logic mirrors `check_hypothesis_minbtl_compliance` (2026-05-12 binding cutoff, advisory below).
4. No loader code change. No DB mutation. No allocator state change.
