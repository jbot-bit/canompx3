---
status: active
owner: joshdlees
last_reviewed: 2026-05-21
superseded_by: ""
---

# Governance fix — canonical parser-surface blocks must leave `docs/runtime/stages/`

**Date:** 2026-05-21
**Author:** main session, design iteration 3
**Scope class:** governance / class-bug defense extension
**Companion stage file:** `docs/runtime/stages/2026-05-21-canonical-blocks-out-of-stage-files.md`

---

## Problem (one paragraph)

Two stage files under `docs/runtime/stages/` carry markdown blocks that are parsed by drift checks — `## Hash Schema` in `2026-05-20-fast-lane-anti-fp-trial-provenance.md` (parsed by Check #167) and `## Suppression Status Enum` in `2026-05-20-fast-lane-anti-fp-2a3-scanner-bridge-wiring.md` (parsed by Check #173). The directory's documented semantic contract (per `.claude/rules/stage-gate-protocol.md` and `docs/governance/system_authority_map.md` surface taxonomy) is "Plans / history / baton — Never cited as live runtime truth." That collision triggered an n=1 protocol-noncompliance event on 2026-05-20: commit `ef4f0f29` deleted the trial-provenance file as a routine close-out, breaking Check #167; commit `4bd288c4` restored it as Bug 2 of a two-bug fix. Same author, same operator-day, opposite mental models four hours apart. The architectural smell is precise: the canonical-inline-copy registry permits any `canonical_source` path, including ephemeral directories — every other registered pair points at `docs/audit/hypotheses/`, `docs/institutional/`, or `.claude/rules/`, but these two outliers were never refused at landing time.

## Why this matters

Today the protection is "Check #167 / #173 fail-closed on next CI run after the file is deleted." That worked for Bug 2 — broke CI, got fixed within hours, no capital affected. But the latency is post-commit. A future close-out following the same false reasoning will land the deletion before drift catches it; recovery is a revert. The deeper issue is that `docs/runtime/stages/` has now been promoted, by accident, to a polymorphic location: most files are ephemeral, two are load-bearing canonical sources, the difference is not machine-readable. Anyone (operator or future Claude) reasoning over the directory contract will encounter this trap again.

## Doctrine cited (verbatim authorities)

- `docs/governance/document_authority.md` — Document Authority Registry. Stage files are *plans/history/baton* class. They "are context and design surfaces, not canonical research truth."
- `docs/governance/system_authority_map.md` — Surface Taxonomy table. Plans/history/baton row: **"Never cited as live runtime truth."** Canonical registries row: stable code truth, "one owned source per concept; no duplicate literals downstream." Parser surface is canonical truth by definition.
- `docs/governance/class-bug-coverage.md` — class-bug family taxonomy. Two rules apply: (i) "Generalized fingerprint scanners… were considered and rejected" — forbids the iteration-1 Path B PreToolUse hook approach; (ii) per-class checks ship as targeted < 80 LOC additions to existing class defenses.
- `memory/feedback_canonical_inline_copy_parity_bug_class.md` — n≥10 instance class. This is instance 11 by lineage. The class is documented; mechanical enforcement is doctrine-correct.
- `memory/feedback_n3_same_class_doctrine_threshold.md` — at n≥3 of same structural class, mechanical enforcement is doctrine-correct, not meta-tooling-on-n=1 violation.
- `memory/feedback_meta_tooling_n1_tunnel_2026_05_01.md` — forbids forcing-functions on n=1. Applies to the specific symptom (canonical-block-in-stage-file deleted on close-out, n=1). Reconciled with the prior bullet: the *class* is n=11; the *specific symptom* is n=1. Mechanical enforcement targets the class via the existing meta-check; no new hook is authored for the n=1 symptom.
- `memory/feedback_next_acceptance_check_load_bearing_design_stage_outlive.md` — documents the specific incident shape (DESIGN stage outliving sibling sub-stages because downstream cites it as design grounding).
- `.claude/rules/institutional-rigor.md` § 10 — Canonical Sources Authority Table. "Import from the single source of truth. Never inline lists or magic numbers." Generalizes to: parser surfaces have one canonical location, and that location is class-appropriate.

**Explicit honesty:** This is project-doctrine grounded. The trading literature in `docs/institutional/literature/` does NOT ground governance protocol decisions. Bailey, López de Prado, Aronson, Harvey-Liu, Chordia, Harris, Carver, Chan ground statistical methodology and trading microstructure. Citing them as authority for document-canonical-source governance would be a citation stretch. The authority chain runs through `docs/governance/*` and the canonical-inline-copy-parity feedback class.

## Three approaches considered (multi-take deliberation)

### Take 1 — Pure relocation, no new check
Move the two blocks to `docs/specs/fast_lane_state_graph.md`. Update parser paths. Update registry. Document the precedent in `stage-gate-protocol.md`. NO new check.

- **Pros:** Smallest diff. n=1 symptom → no forcing function (doctrine-pure on `feedback_meta_tooling_n1_tunnel_2026_05_01.md`). Removes the precondition.
- **Cons:** Convention-only protection for future canonical blocks added to future stage files. The convention failed once already in the exact incident we have evidence of (the close-stage commit message had warned about this very pattern; the operator broke it 4 hours later).

### Take 2 — Relocation + one-invariant extension to existing Check #159 (CHOSEN)
Take 1 PLUS: extend `check_canonical_inline_copies_have_parity_check` (the existing Layer-2 meta-check for the canonical-inline-copy-parity family) with invariant (d): `entry.canonical_source` must NOT start with `docs/runtime/`. ~10 LOC added to an existing check.

- **Pros:** Mechanical enforcement of the class invariant ("canonical sources don't live in ephemeral directories") at the registry layer. Zero new checks, zero new hooks, zero new artifacts — one invariant added to an existing class-defense check. Doctrine-aligned on all three competing rules: (i) class-bug-coverage allows tightening existing class defenses; (ii) n=3 doctrine endorses mechanical enforcement at instance 11; (iii) meta-tooling-on-n=1 ban doesn't apply because nothing new is built. P3 (future canonical blocks default-protected) mechanically enforced.
- **Cons:** Slightly larger one-time diff than Take 1 (~10 extra LOC). Adds one mutation-proof test.

### Take 3 — Frontmatter marker + new PreToolUse hook + new drift check (rejected)
Iteration 1 Path B. `canonical_role:` marker on each stage file; new drift check; new PreToolUse hook; pre-commit guard.

- **Cons:** ~120 LOC of new infrastructure on an n=1 symptom. Directly violates `feedback_meta_tooling_n1_tunnel_2026_05_01.md`. Leaves the surface-taxonomy violation in place and guards around it instead of removing it. Adds permanent maintenance burden (new hook, new check, marker convention) for a problem that disappears under Take 2's structural fix.

**Verdict:** Take 2. Wins on doctrine, on minimum-viable-fix grounds, and on architectural cleanliness. Take 1 was the runner-up; rejected because the convention already proved insufficient.

## Approach (Take 2 detail)

**Stage 1 — Relocate the two canonical blocks.**

1. Append `## 9. Hash Schema` to `docs/specs/fast_lane_state_graph.md`. Verbatim copy of the `## Hash Schema (`structural_hash`)` block from `2026-05-20-fast-lane-anti-fp-trial-provenance.md` lines 133–155. Add a one-line attribution: *"Originally landed in commit `5c6040c3` (Stage 2A.1, 2026-05-20). Relocated from `docs/runtime/stages/2026-05-20-fast-lane-anti-fp-trial-provenance.md` per surface-taxonomy doctrine, 2026-05-21."*
2. Append `## 10. Suppression Status Enum` to the same spec. Verbatim copy of the canonical block from `2026-05-20-fast-lane-anti-fp-2a3-scanner-bridge-wiring.md` lines 42–55 (or whatever the exact line range is). Same attribution shape, commit `9b13d0d1` (Stage 2A.3).
3. Replace the body of `## Hash Schema (`structural_hash`)` in the trial-provenance stage file with a single backlink line: `> Canonical: docs/specs/fast_lane_state_graph.md § 9 Hash Schema. This stage file is now design rationale only.`
4. Same backlink replacement in the 2a3 file pointing at § 10.

**Stage 2 — Update parsers and registry to new locations.**

5. `pipeline/check_drift.py` Check #167 (`check_fast_lane_structural_hash_schema_parity`): change the `stage_path = …` literal at line ~10555 from the runtime/stages path to `docs/specs/fast_lane_state_graph.md`. Confirm the parser locates the `## 9. Hash Schema` heading (anchor may need to broaden to match `^## (?:\d+\. )?Hash Schema` to tolerate the section-numbered version).
6. `pipeline/check_drift.py` Check #173 (`check_fast_lane_promote_queue_provenance_present`): same shape, line ~11020. Change literal to spec path; tolerate `## 10. Suppression Status Enum` heading.
7. `pipeline/canonical_inline_copies.py`: two `InlineCopyPair.canonical_source` strings updated to point at the new spec sections.

**Stage 3 — One invariant added to existing meta-check.**

8. `pipeline/check_drift.py` `check_canonical_inline_copies_have_parity_check`: add invariant (d) inside the existing per-entry loop. Pseudocode shape: `if entry.canonical_source.startswith("docs/runtime/"): violations.append(...)`. Reason text: *"canonical_source path may not live under docs/runtime/ (ephemeral surface per system_authority_map.md). Relocate to docs/specs/, docs/institutional/, or .claude/rules/."* Update the function docstring to list invariant (d).
9. Add one injection test to the existing test file for the meta-check. Pattern: build a synthetic `CANONICAL_INLINE_COPIES` list with one entry whose `canonical_source` is `docs/runtime/foo.md`; assert the check returns a non-empty violation list with the expected message fragment.

**Stage 4 — Doc the precedent + write n=1 feedback file.**

10. Append 3 lines to `.claude/rules/stage-gate-protocol.md` § Stage Completion: *"Stage files are decision-class artifacts (per `docs/governance/system_authority_map.md` surface taxonomy). Never embed parser surface (drift-check-parsed YAML/table blocks) in stage files. Canonical blocks belong in `docs/specs/`, `docs/institutional/`, or `.claude/rules/`. Enforced by `check_canonical_inline_copies_have_parity_check` invariant (d)."*
11. Write `memory/feedback_canonical_block_in_stage_file_anti_pattern_n1_2026_05_21.md`: ~30 lines documenting the Bug 2 incident chain, the class lineage (instance 11 of canonical-inline-copy-parity), and the structural fix.

## Blast radius

| File | Edit type | Class |
|---|---|---|
| `docs/specs/fast_lane_state_graph.md` | append §§ 9–10 | spec / canonical |
| `docs/runtime/stages/2026-05-20-fast-lane-anti-fp-trial-provenance.md` | replace block body with backlink | history (now de-fanged) |
| `docs/runtime/stages/2026-05-20-fast-lane-anti-fp-2a3-scanner-bridge-wiring.md` | replace block body with backlink | history (now de-fanged) |
| `pipeline/check_drift.py` | 2 path literals + 1 new invariant in existing function | code (verified by drift run) |
| `pipeline/canonical_inline_copies.py` | 2 string values updated | code (verified by Check #159) |
| `tests/test_pipeline/test_check_canonical_inline_copies_have_parity_check.py` (or sibling) | append one injection test | test |
| `.claude/rules/stage-gate-protocol.md` | append 3 lines to § Stage Completion | rule |
| `memory/feedback_canonical_block_in_stage_file_anti_pattern_n1_2026_05_21.md` | new file | feedback |

Total: ~8 files. Net diff ~+150 lines (mostly the verbatim relocations and the feedback file), ~−40 lines (block bodies replaced with backlinks). One PR.

**Capital-class touched:** zero.
**Schema changes:** zero.
**Entry-model changes:** zero.
**Production behavior changes:** zero — `fast_lane_structural_hash.py` and `fast_lane_promote_queue.py` still read the same `HASH_SCHEMA_*` and `STATUS_VALUES` constants from their own modules; only the canonical *citation path* changes.

## Acceptance criteria (testable)

1. `python pipeline/check_drift.py` returns "all checks passed" — including Check #167, Check #173, and Check #159 (the meta-check, now with invariant (d) active over the updated registry).
2. Injection test on Check #159: synthetic entry with `canonical_source="docs/runtime/foo.md"` → check returns non-empty violations list containing the path-rejection message.
3. `grep -nE "canonical_source\s*=" pipeline/canonical_inline_copies.py | grep "docs/runtime/"` returns zero matches.
4. Both stage files still exist on disk (`mode: CLOSED` frontmatter retained) but their bodies contain backlinks, not parser surface. Future close-out commits may now safely `git rm` them without breaking drift.
5. Companion test files for Check #167 and Check #173 still pass — relocations were verbatim, parser anchor regex tolerated.

## Rollback plan

Five logical edits across 8 files; each edit is a small contained change.
- If Check #167 or #173 fails post-relocation → bisect to the parser-path commit; broaden the section-heading regex or correct the spec section name.
- If Check #159 injection test fails → check the invariant-(d) wording matches what the test asserts.
- If the registry edit lands without the relocation → Check #167/#173 fail-closed pointing at a non-existent block; revert the registry edit.
- Total rollback time: < 5 minutes of `git revert` per stage.

## What this rule forbids going forward

- New `InlineCopyPair` entries with `canonical_source` paths under `docs/runtime/` (fails Check #159 invariant (d) at landing time).
- Stage files holding parser-surface blocks (rule in `stage-gate-protocol.md` + invariant (d) at the registry layer).
- Building a new PreToolUse hook or a new dedicated drift check for this n=1 symptom (would violate `feedback_meta_tooling_n1_tunnel_2026_05_01.md`).

## What this rule does NOT cover (silences acknowledged)

- **HANDOFF baton lying** (Bug 2 had two failure modes: (a) the close-out deletion, (b) the next-session HANDOFF baton falsely claiming the file was restored). This design addresses (a) only. (b) is a separate documented class (`feedback_closeout_verify_against_canonical.md`) with its own defense — not in scope here.
- **Canonical block moved within a spec file** (someone refactors `fast_lane_state_graph.md` and removes the parsed section). Existing Check #167 / #173 fail-closed on next drift run; that is the same latency as today. Not made worse by this change.
- **Pre-merge hook coverage** outside Claude. The class-bug-coverage doctrine forbids generalizing to a directory-content scanner. CI runs `check_drift.py`; that is the project's chosen defense layer.

## Self-review before claim-of-done (per institutional-rigor § 1)

- Happy path: append two spec sections, swap two paths in `check_drift.py`, swap two strings in `canonical_inline_copies.py`, run drift → all pass.
- Edge: spec heading uses numbered prefix (`## 9. Hash Schema`) while old parser expects bare `## Hash Schema`. Probe: broaden regex to `^## (?:\d+\.\s+)?Hash Schema\b`. Mutation-test the broadened regex by injecting a misnamed heading.
- Failure: someone in the future adds an entry to `CANONICAL_INLINE_COPIES` with a `docs/runtime/` path. Invariant (d) catches at next drift run. Confirmed by injection test.
- Operator-state-drift failure (the original incident class): future close-out tries to `git rm` one of these stage files. Now safe — bodies hold only backlinks, no parser surface. Drift unaffected by the deletion.
