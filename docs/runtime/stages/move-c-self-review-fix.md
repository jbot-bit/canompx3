---
task: Move C self-review fix — remove dead guard + add non-canonical pattern test
mode: IMPLEMENTATION
slug: move-c-self-review-fix
agent: claude-code-terminal-b
created: 2026-04-08T03:00:00Z
updated: 2026-04-08T03:00:00Z
parent_commit: 838ab85
---

## Purpose

Bloomey self-review of Move C (commit `838ab85`) caught one MEDIUM finding I
introduced. Per institutional rigor rule 2 ("after any fix, review the fix —
fixes introduce new bugs"), this is a same-session follow-up to close the gap
before the work moves further forward.

## Findings

**MEDIUM — dead-code fail-closed guard at `scripts/databento_daily.py:82-91`**

PREMISE: The fail-closed guard is unreachable. By construction
`set({k: f(k) for k in S}) == S`, so `S - set(comprehension) = ∅` always.

EVIDENCE: `_DATABENTO_SYMBOLS` is built from `for instrument in
ACTIVE_ORB_INSTRUMENTS` (line 79), and the guard checks
`set(ACTIVE_ORB_INSTRUMENTS) - set(_DATABENTO_SYMBOLS)` (line 86) — always
empty set, `if _missing` always False, RuntimeError unreachable.

VERDICT: Self-caught dead code. Per rule 5 (no dead code), must remove or
replace with a guard that actually fires.

**GAP — explicit test for non-canonical pattern path missing**

`get_outright_root` has TWO failure paths: unknown instrument (covered by
`test_unknown_instrument_raises`) and non-canonical pattern (no test). Adding
a `monkeypatch` test closes the gap without inflating real ASSET_CONFIGS.

## Scope Lock

- scripts/databento_daily.py

(Test files not listed — `tests/` is in `SAFE_DIRS`.)

## Blast Radius

Single-file edit. Removes 10 lines of dead code from `databento_daily.py`.
The removed guard is unreachable (proven by construction in self-review),
so no observable behavior change to runtime — pure cleanup. Real fail-closed
remains: any unresolvable instrument raises ValueError from line 79 at module
load, which is strictly stronger than the deleted guard. No external
importers of `databento_daily.py` (verified via grep), so no caller updates.

Test addition: `tests/test_pipeline/test_asset_configs.py` gains one new
`monkeypatch`-based test in `TestGetOutrightRoot` for the non-canonical
pattern branch. Pure additive coverage.

## Done Criteria

- [ ] `scripts/databento_daily.py` — guard at lines 82-91 removed, comment
  rewritten to reflect that module-load comprehension IS the fail-closed gate
- [ ] `tests/test_pipeline/test_asset_configs.py::TestGetOutrightRoot::test_non_canonical_pattern_raises` — passes
- [ ] All TestGetOutrightRoot tests still pass (8/8)
- [ ] `pytest tests/test_pipeline/` — full suite green (1019+1)
- [ ] `python pipeline/check_drift.py` — 84/0/7 unchanged
- [ ] One commit referencing 838ab85 as the parent
