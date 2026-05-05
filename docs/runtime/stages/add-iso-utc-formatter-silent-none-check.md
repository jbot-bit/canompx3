---
task: add-iso-utc-formatter-silent-none-check
mode: IMPLEMENTATION
status: IN-PROGRESS-2026-05-06
agent: claude
updated: 2026-05-06
scope_lock:
  - pipeline/check_drift.py
  - tests/test_pipeline/test_check_drift.py
  - docs/governance/class-bug-coverage.md
  - trading_app/live/session_orchestrator.py
  - trading_app/live/bot_dashboard.py
  - docs/runtime/stages/add-iso-utc-formatter-silent-none-check.md
blast_radius: "Adds ONE ADVISORY drift check (check_iso_utc_formatter_silent_none) + 7-test fixture class + governance note (6-family matrix). Two comment-only annotations on session_orchestrator.py:1203 and bot_dashboard.py:353 carry semantic intent for known-safe silent-None tails. Reads 7 scan files via AST. Writes new function in check_drift.py registered as ADVISORY. No DB writes. No callers outside check_drift CLI. Annotations are comment-only — zero behavior change."
acceptance:
  - tests/test_pipeline/test_check_drift.py::TestIsoUtcFormatterSilentNone passes
  - check_iso_utc_formatter_silent_none() returns [] against canonical files
  - python pipeline/check_drift.py exits 0 with new check listed as ADVISORY
  - ruff check + format pass on all touched files
---

## Plan source

User-supplied detailed plan executed under Auto Mode. Plan encodes 2 audit
rounds: blast-radius corrections (3 wrong file paths, predicate missing
`log.critical`/`log.error`) + Explore parallel pass (2 try/except-pass
false-positive sites in `session_orchestrator.py`/`bot_dashboard.py`,
6-family governance count, no docs/governance precedent for the matrix
format).

## What ships

1. `_ISO_UTC_FORMATTER_SCAN_FILES` constant (7 corrected paths)
2. `_function_has_isinstance_then_silent_none` AST helper (predicate)
3. `check_iso_utc_formatter_silent_none` check function
4. Registration in `CHECKS` list as ADVISORY (after E2 lookahead entry)
5. `TestIsoUtcFormatterSilentNone` test class (7 cases)
6. `# silent-none-policy: ...` annotations on the 2 known-safe sites
7. `docs/governance/class-bug-coverage.md` (6-family matrix + rule for new)

## Out of scope (deferred per plan)

- `_iso_utc` → `iso_utc` rename
- Generalized class-bug fingerprint scanner (Stage B — killed)
- Updating `feedback_iso_utc_silent_none_class_pattern.md` to remove F6
- Predicate sophistication for try/except-pass detection (chose annotation
  rollout instead)
