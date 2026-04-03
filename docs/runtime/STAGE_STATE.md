---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Governance tools — trace system, research claim validator, stale-doc scanner
updated: 2026-04-03T14:00:00Z
scope_lock:
  - pipeline/trace.py
  - pipeline/paths.py
  - scripts/tools/research_claim_validator.py
  - scripts/tools/stale_doc_scanner.py
  - tests/test_governance_tools.py
  - logs/traces/.gitkeep
  - .gitignore
blast_radius:
  - pipeline/paths.py: add 1 constant (TRACES_DIR). No callers affected.
  - All other files are NEW — zero blast radius on existing code.
  - No schema changes, no entry model changes, no pipeline logic changes.
acceptance:
  - python scripts/tools/research_claim_validator.py --n 150 --p 0.003 --mechanism "cost-gated" exits 0 (VALID)
  - python scripts/tools/research_claim_validator.py --n 25 --p 0.003 --mechanism "cost-gated" exits 1 (INVALID)
  - python scripts/tools/research_claim_validator.py --n 50 --p 0.003 --mechanism "cost-gated" exits 0 (REGIME_ONLY)
  - python scripts/tools/stale_doc_scanner.py runs without error
  - python -m pytest tests/test_governance_tools.py -x -q passes
  - python pipeline/check_drift.py passes
---
