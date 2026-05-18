---
task: "STAGE 2 — Canonical-inline-copy meta-registry (Layer 2 of 3-layer hardening). Add CANONICAL_INLINE_COPIES registry + meta-check asserting every registered canonical→inline pair is covered by a dedicated parity check. Closes the class-level enforcement gap per the n=3+ doctrine."
mode: IMPLEMENTATION
scope_lock:
  - pipeline/canonical_inline_copies.py
  - pipeline/check_drift.py
  - tests/test_pipeline/test_canonical_inline_copies_registry.py
  - docs/runtime/decision-ledger.md
  - docs/runtime/stages/2026-05-19-stage-2-canonical-inline-copies-meta-registry.md
---

## Blast Radius

- `pipeline/canonical_inline_copies.py` (NEW) — frozen dataclass `InlineCopyPair` + module-level `CANONICAL_INLINE_COPIES: list[InlineCopyPair]`. Pure doctrine surface. Imported by check_drift.py. ~80 lines.
- `pipeline/check_drift.py` — add ONE function `check_canonical_inline_copies_have_parity_check` (Check #159) + register. Imports from `canonical_inline_copies`. Pure additive, no existing check touched. Drift count 158→159.
- `tests/test_pipeline/test_canonical_inline_copies_registry.py` (NEW) — registry-integrity tests + meta-check mutation-probe tests (deliberately deregister a pair, missing parity-check function, missing injection-test file).
- `docs/runtime/decision-ledger.md` — one entry.
- Reads: imports from check_drift's parity-check functions (in-process), reads `tests/test_pipeline/` directory listing to verify injection-test convention.
- Trading logic / DB / lane allocator: zero touch. No capital-class blast radius.

## Pre-decided design (carry-over from d88a5465 stage-2 proposal)

Three open questions from Stage 1's proposal — resolved as recommended:

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | Seed with 4 known or grep-audit codebase first? | **BOTH** — seed with the 4 known, then run a `# mirrors\|# from canonical\|# cite:` grep audit and add any survivors as discovery work in the same stage. | Seeded entries land the doctrine; audit closes blind-spot risk. ~30 min extra. |
| 2 | Meta-check enforces injection-test naming convention? | **YES** — registry entry MUST point to a `tests/test_pipeline/<slug>.py` file that exists, AND that test file must contain ≥1 test function per gated constant (sibling-coverage doctrine). | Matches `regex-alternation-sibling-coverage` doctrine + Stage 1's own pattern. Without this, registry becomes dead doctrine. |
| 3 | Registry location: `pipeline/check_drift.py` or sibling? | **Sibling file `pipeline/canonical_inline_copies.py`** | check_drift.py is already >10k lines (see `large-file-reads.md` rule). Sibling file keeps registry importable + greppable without dragging the check engine. |

## Canonical-source map for the seed entries

These four are CONFIRMED instances of the bug class — sources verified during Stage 1 session:

| # | Inline site | Canonical source | Parity check | Bug-class anchor | Status |
|---|---|---|---|---|---|
| 1 | `scripts/research/fast_lane_promote_queue.py:65-70` (T_KILL_FLOOR, T_PROMOTE_FLOOR, EXPR_FLOOR, N_FLOOR, FIRE_MIN, FIRE_MAX) | `docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml` `screen:` block lines 102-115 | `check_fast_lane_promote_threshold_parity` (Check #158) | `memory/feedback_canonical_inline_copy_parity_bug_class.md` | LANDED d88a5465 |
| 2 | Cost-specs class (sites TBD-audit) | `pipeline/cost_model.COST_SPECS` | TBD — likely needs new parity check OR `cost_model.py` already enforces | `memory/feedback_doctrine_drift_cost_specs_2026_05_01.md` | NEEDS AUDIT |
| 3 | Allocator-gate class (sites TBD-audit) | TBD — review `feedback_allocator_gate_class_pattern_fail_open.md` | `apply_chordia_gate` + `apply_c8_oos_gate` pattern | `memory/feedback_allocator_gate_class_pattern_fail_open.md` | NEEDS AUDIT |
| 4 | Chordia threshold class | `pre_registered_criteria.md` Criterion 4 (t≥3.79 / t≥3.00) | `check_chordia_result_threshold_matches_prereg` (Check ~141) | `memory/feedback_chordia_theory_citation_field_presence_trap.md` | ALREADY HAS PARITY CHECK |

Audit step (~30 min, IN this stage): grep the codebase for `# mirrors\|# from canonical\|# cite:\|# canonical source:\|# see also.*\.yaml` patterns and triage each hit as (a) genuine inline copy → register, (b) doc-only comment → skip, (c) already covered by existing parity check → register with pointer.

## Acceptance

1. `pipeline/canonical_inline_copies.py` lands with `InlineCopyPair` dataclass + `CANONICAL_INLINE_COPIES` list containing at minimum the 4 known seeds + any survivors from the grep audit.
2. `check_canonical_inline_copies_have_parity_check` (Check #159) registered in `CHECKS`. For each registry entry asserts: (a) parity-check function exists in `pipeline.check_drift` module globals, (b) is callable, (c) named test file exists at `tests/test_pipeline/<test_slug>.py`, (d) that test file contains ≥N test functions where N = count of gated constants in the entry.
3. ≥5 mutation-probe tests in the new test file: clean-state pass, deregistered-pair-must-stay-covered (negative), missing-parity-check-function (negative), missing-test-file (negative), insufficient-test-count (negative).
4. `python pipeline/check_drift.py` count goes 158→159 and Check 159 PASSES. Existing 11 parity tests still PASS.
5. evidence-auditor pass after landing (advisory not compulsory — pure doctrine surface, no truth-layer mutation — but doing it anyway per Stage 1 precedent).
6. Commit message cites `[[canonical-inline-copy-parity-bug-class]]` + `[[n3-same-class-doctrine-threshold]]`.

## Self-check (simulated, pre-implementation)

- **Happy path:** registry has 4+ entries, all parity checks exist, all test files exist with ≥N tests each → Check 159 returns `[]`.
- **Drift path (a):** add a registry entry pointing to nonexistent `check_foo` → meta-check returns "parity check `check_foo` not found in pipeline.check_drift module globals (registry entry orphaned)".
- **Drift path (b):** add entry pointing to missing test file → meta-check returns "injection test file `tests/test_pipeline/test_foo.py` not found".
- **Drift path (c):** test file exists but has only 2 test functions for an entry claiming 6 gated constants → meta-check returns "expected ≥6 test functions in `test_foo.py`, found 2 — sibling-coverage doctrine violation".
- **Failure mode:** if `pipeline.check_drift` import fails inside meta-check → return that as the violation (don't crash silently). Same fail-closed pattern as Stage 1.

## Risk

- **Risk:** registry entries become stale (parity check renamed, test file moved). Mitigation: meta-check IS the staleness detector. By design.
- **Risk:** grep audit surfaces 10+ candidate pairs and stage scope blows up. Mitigation: triage time-boxed at 30 min. Any genuine inline copy not registered in this stage gets a follow-up stage file, NOT silent skip.
- **Risk:** sibling file `pipeline/canonical_inline_copies.py` becomes the next-largest unmaintained registry. Mitigation: kept under 150 lines via dataclass-per-entry discipline; if it grows past that, split by canonical-source class.

## Stage 2.5 (bundled OR follow-up, operator call)

Pre-existing Pyright tech debt in `pipeline/check_drift.py` flagged during Stage 1 audit:
- ~10 `reportOptionalSubscript` warnings on `m.group(...)` lines (1914, 2488, 3032, 3897, 3905, 3913, 4944, 4948, 5179, 6183) — missing `if m is None` guard before subscripting.
- 3 `"object" is not iterable` warnings (8914, 9139, 9189) — missing cast/guard on dict/list values typed as `object`.

Pure annotation hygiene; no runtime impact (regex matches succeed on their inputs in practice). Mechanical fix pattern:
```python
m = re.search(...)
if m is None:
    continue   # or violations.append(...) for fail-closed paths
# now m.group(...) is safe
```

**Operator decision:** bundle into Stage 2 commit (adds ~15 trivial edits, no behavior change, ~20 min) OR separate trivial-tier stage. Recommended **bundle** — same file, same session, mechanical, and matches the "who's gonna fix this if we always leave it" callout from 2026-05-19.

## Next session opener

```
/clear
Read this stage file: docs/runtime/stages/2026-05-19-stage-2-canonical-inline-copies-meta-registry.md
Read memory/feedback_canonical_inline_copy_parity_bug_class.md + feedback_n3_same_class_doctrine_threshold.md.
Then: (1) grep audit per the table above, (2) implement, (3) run evidence-auditor, (4) commit + push, (5) close stage file.
Bundle Stage 2.5 (Pyright cleanup) unless told otherwise.
```
