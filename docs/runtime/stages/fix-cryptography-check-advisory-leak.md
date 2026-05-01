# Stage: Fix cryptography-pin-check Phase 2 advisory leak

**Status:** IMPLEMENTATION
**Branch:** fix/cryptography-check-advisory-leak
**Triggered by:** code-review (PR #174 + #175 review) — Section D, Critical D1
**Risk tier:** medium (guardrail bug, time-bombed for 2026-10-29)

---

## Problem

`check_cryptography_pin_holds` was registered with `is_advisory=False` (Phase 1 must
fail-closed on `cryptography>=47` regression). But its Phase 2 staleness branch
appends an `"  ADVISORY: ..."` string to the same `violations` list returned to the
executor. With `is_advisory=False`, that string becomes a blocking violation —
every commit on/after `2026-10-29` (the constraints.txt revisit-by date) will
fail pre-commit drift with text saying "ADVISORY: ... non-blocking".

Verified by REPL: `check_cryptography_pin_holds()` with mocked
`datetime.date.today() == 2027-01-01` and clean Phase 1 state returns
`['  ADVISORY: constraints.txt revisit-by:2026-10-29 has passed (64 day(s) overdue)...']`.

## Scope lock

- `pipeline/check_drift.py` — change Phase 2 branch from `violations.append(...)`
  to `print(...)` + `return []`. Mirrors how D1-D5 advisories already surface.
- `tests/test_pipeline/test_check_drift.py` (or new `test_check_drift_crypto.py`)
  — add 3 unit tests: clean-state, regression (cryptography>=47 + fastmcp),
  staleness (date past revisit-by).
- `pipeline/check_drift.py` SLOW_CHECK_LABELS — add D3 (D3 measures 486ms,
  inconsistent with D2 1.2s being marked slow).
- (Out of scope this stage) Triage the 3 real D2 hits in research/ — separate
  follow-up since they are pre-existing canonical-import violations.

## Blast radius

- Pre-commit drift gate behavior on/after 2026-10-29 (currently broken).
- Pre-commit drift gate behavior in --fast mode (D3 reclassification).
- No live-trading path; no production strategy logic.

## Acceptance

1. Three new tests pass (clean / regression / staleness).
2. All 1253 existing tests still pass.
3. `python pipeline/check_drift.py` exits 0.
4. Synthetic call with date past 2026-10-29 returns `[]` not a violations list.

## Authority

- Code review: this session.
- Original PR #175: `f2fb88f1` (squash-merged 2026-04-29).
