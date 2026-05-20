---
task: Relocate `## Hash Schema` and `## Suppression Status Enum` canonical blocks from two stage files into docs/specs/fast_lane_state_graph.md; update Checks #167 + #173 + canonical_inline_copies.py to the new paths; add invariant (d) to existing Check #159 forbidding docs/runtime/ canonical sources; document precedent.
mode: IMPLEMENTATION
scope_lock:
  - docs/specs/fast_lane_state_graph.md
  - docs/runtime/stages/2026-05-20-fast-lane-anti-fp-trial-provenance.md
  - docs/runtime/stages/2026-05-20-fast-lane-anti-fp-2a3-scanner-bridge-wiring.md
  - pipeline/check_drift.py
  - pipeline/canonical_inline_copies.py
  - tests/test_pipeline/test_check_canonical_inline_copies_have_parity_check.py
  - .claude/rules/stage-gate-protocol.md
  - memory/feedback_canonical_block_in_stage_file_anti_pattern_n1_2026_05_21.md
---

## Blast Radius

- docs/specs/fast_lane_state_graph.md — append §§ 9 (Hash Schema) and 10 (Suppression Status Enum). Spec already AUTHORITATIVE; already parsed by `check_fast_lane_state_graph_node_parity`. Adding sections does not alter that check's scope (it parses § 2 Node Inventory). New sections are read by Checks #167 and #173 after the path swap.
- docs/runtime/stages/2026-05-20-fast-lane-anti-fp-trial-provenance.md — replace the body of the `## Hash Schema (`structural_hash`)` section (~22 lines) with a single backlink line. mode:CLOSED frontmatter retained. File becomes safely sweepable in future close-outs.
- docs/runtime/stages/2026-05-20-fast-lane-anti-fp-2a3-scanner-bridge-wiring.md — same shape on the `## Suppression Status Enum` block.
- pipeline/check_drift.py — Check #167 path literal at line ~10555 + Check #173 path literal at line ~11020 + invariant (d) inside `check_canonical_inline_copies_have_parity_check` body. ~3 small edits in one file.
- pipeline/canonical_inline_copies.py — two `InlineCopyPair.canonical_source` strings updated to spec paths.
- tests/test_pipeline/test_check_canonical_inline_copies_have_parity_check.py — append one injection test for invariant (d).
- .claude/rules/stage-gate-protocol.md — append 3 lines to § Stage Completion.
- memory/feedback_canonical_block_in_stage_file_anti_pattern_n1_2026_05_21.md — new feedback file.

Reads: drift checks read the new spec sections. Writes: zero capital-class. Zero schema changes. Zero entry-model changes. Zero production-behavior changes (constants in scripts/research/* unchanged; only their canonical citation path changes).

## Companion design doc

`docs/plans/active/2026-05/2026-05-21-canonical-blocks-out-of-stage-files-design.md` (multi-take deliberation, doctrine citations, three approaches compared, Take 2 chosen and defended).

## Acceptance Criteria (mirrors design § Acceptance)

1. `python pipeline/check_drift.py` returns all-passed including Checks #167, #173, #159.
2. Injection test on Check #159 invariant (d) passes (synthetic `docs/runtime/foo.md` entry → violation).
3. `grep -nE "canonical_source\s*=" pipeline/canonical_inline_copies.py | grep "docs/runtime/"` returns zero matches.
4. Both stage files still on disk with `mode: CLOSED`; bodies replaced with backlinks; no parser surface remaining inside them.
5. Existing test files for Checks #167 and #173 still pass post-relocation.

## Order of operations

1. Append §§ 9 and 10 to `docs/specs/fast_lane_state_graph.md` (verbatim copies + attribution).
2. Replace block bodies in two stage files with backlinks.
3. Edit Check #167 path literal in `check_drift.py`. Verify regex anchor tolerates numbered heading.
4. Edit Check #173 path literal in `check_drift.py`. Verify regex anchor tolerates numbered heading.
5. Edit two `canonical_source` strings in `canonical_inline_copies.py`.
6. Run `python pipeline/check_drift.py` — expect all passed (Checks #167 and #173 now reading the spec).
7. Add invariant (d) to `check_canonical_inline_copies_have_parity_check` body + docstring.
8. Append injection test for invariant (d).
9. Run drift again; run the relevant pytest files.
10. Append doctrine line to `stage-gate-protocol.md`.
11. Write feedback file.
12. Final drift run; commit.

## Stale-detection

Scope_lock files are not currently being modified by any sibling terminal (working tree shows only `docs/runtime/fast_lane_trial_ledger.yaml` dirty + a staged deletion of `2026-05-21-fast-lane-ledger-append-commit.md` — both from the other terminal earlier today, neither in scope_lock above). Safe to proceed.

## Hard Constraints (carried from design)

- NO new drift check function.
- NO new PreToolUse hook.
- NO frontmatter marker on stage files.
- NO new spec file (relocation target is the existing AUTHORITATIVE spec).
- Stage files retain `mode: CLOSED` — audit trail preserved.
- Citations: project-doctrine-only. No literature stretch (Bailey/LdP/Aronson do not ground governance).
