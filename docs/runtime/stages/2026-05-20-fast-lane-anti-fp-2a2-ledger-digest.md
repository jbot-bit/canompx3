---
task: Stage 2A.2 IMPLEMENTATION — Trial Ledger + Graveyard Digest. Append-only universe-of-trials ledger + NO-GO graveyard digest as derived state. Consumes the structural_hash module from 2A.1 (shipped). Predecessor for 2A.3 (scanner/ranker/bridge wiring).
mode: IMPLEMENTATION
scope_lock:
  - scripts/research/fast_lane_trial_ledger.py
  - scripts/research/fast_lane_graveyard_digest.py
  - docs/runtime/fast_lane_trial_ledger.yaml
  - docs/runtime/fast_lane_graveyard_digest.yaml
  - docs/runtime/fast_lane_structural_hashes.yaml
  - pipeline/check_drift.py
  - tests/test_pipeline/test_check_drift_fast_lane_trial_ledger_append_only.py
  - tests/test_pipeline/test_check_drift_fast_lane_graveyard_digest_parity.py
design_grounding: docs/runtime/stages/2026-05-20-fast-lane-anti-fp-trial-provenance.md
parent_split: docs/runtime/stages/2026-05-20-fast-lane-anti-fp-implementation.md (Sub-Stage 2A.2 section)
predecessor_substage: 2A.1 (structural_hash + Check #167) — shipped, 14 tests pass, drift count = 168
---

## Predecessor and Context

- **Design grounding:** `docs/runtime/stages/2026-05-20-fast-lane-anti-fp-trial-provenance.md` (DESIGN mode, locked). All architectural decisions live there.
- **Parent split file:** `docs/runtime/stages/2026-05-20-fast-lane-anti-fp-implementation.md` (3-sub-stage IMPLEMENTATION split). 2A.1 has shipped; this stage opens 2A.2.
- **Predecessor verified at stage-open:**
  - `python pipeline/check_drift.py` reports 168 checks (2A.1's Check #167 plus pre-existing context-budget Check #168 from another branch).
  - `pytest tests/test_research/test_fast_lane_structural_hash.py tests/test_pipeline/test_check_drift_fast_lane_structural_hash_schema_parity.py -q` → 14 passed.
  - `pipeline/canonical_inline_copies.py` carries the `HASH_SCHEMA_VERSION` InlineCopyPair.

## Goal

Land the append-only trial ledger + graveyard digest + their two drift checks. Both files are derived-state; neither is hand-editable.

## Files to Touch

### Production code (2 new modules + 1 drift-check file edit)

1. `scripts/research/fast_lane_trial_ledger.py` — writer + reader for `fast_lane_trial_ledger.yaml`. Helper `append_trial_ledger_entry(run_id, structural_hash, k_lineage_dict, n_hat, holdout_policy='mode_A', holdout_sacred_from='2026-01-01', …)`. Pure module (no main); imported by 2A.3's scanner. **Capital-class refusal:** writer raises if `prereg_path` resolves to anything under `validated_setups`/`chordia_audit_log.yaml`/`lane_allocation.json`.
2. `scripts/research/fast_lane_graveyard_digest.py` — builder for `fast_lane_graveyard_digest.yaml`. Reads `chatgpt_bundle/06_RD_GRAVEYARD.md` + `docs/STRATEGY_BLUEPRINT.md` § NO-GO + `docs/runtime/action-queue.yaml` entries with `status: park` or `status: kill`. Emits hash-set keyed by `structural_hash`.
3. `pipeline/check_drift.py` — add 2 check functions + CHECKS entries:
   - **#169 `check_fast_lane_trial_ledger_append_only`** — fails if any entry's `run_timestamp_utc` decreases, any prior entry is mutated, or `run_id` duplicates. Also fails if any entry lacks `holdout_policy: mode_A` or `holdout_sacred_from: 2026-01-01`. (Numbering: parent split's table assigned #168; current main carries Check #168 from another branch, so this stage uses #169 to avoid collision. Both numbers reference the same check function per the parent split's "Numbering note" pattern.)
   - **#170 `check_fast_lane_graveyard_digest_parity`** — fails if any graveyard MD entry is missing from the digest hashset (or vice versa), or if the digest file lacks the `do_not_hand_edit: true` banner.

### Derived-state files (new, with `do_not_hand_edit: true` banner)

4. `docs/runtime/fast_lane_trial_ledger.yaml` — initial state = empty `entries: []` list with `schema_version: 1` and the banner. Populated by 2A.3 when scanner wires in.
5. `docs/runtime/fast_lane_graveyard_digest.yaml` — initial state = built by running the digest module against current canonical sources. Populated at stage close.
6. `docs/runtime/fast_lane_structural_hashes.yaml` — initial state = empty `index: {}` with `schema_version: 1` and the banner. Rebuilt deterministically from the ledger; 2A.2 lands the skeleton, 2A.3 populates.

### Tests (2 new files)

7. `tests/test_pipeline/test_check_drift_fast_lane_trial_ledger_append_only.py` — 4 injection probes:
   - mutate `run_timestamp_utc` backwards on an existing entry → fail.
   - mutate a prior entry's `structural_hash` → fail.
   - duplicate `run_id` across entries → fail.
   - malformed YAML / missing schema_version / missing banner → fail.
   - Plus 2 holdout-sentinel injections: strip `holdout_policy`; mutate `holdout_sacred_from` to `2025-01-01`.
8. `tests/test_pipeline/test_check_drift_fast_lane_graveyard_digest_parity.py` — 4 injection probes:
   - add ghost graveyard entry to digest (not in any source MD) → fail.
   - remove real graveyard entry from digest (still in source MD) → fail.
   - hash-collide a graveyard entry with an active prereg → fail.
   - strip `do_not_hand_edit: true` banner → fail.

## Files NOT to Touch

- `scripts/research/fast_lane_structural_hash.py` — shipped in 2A.1; any change requires a `HASH_SCHEMA_VERSION` bump and a new stage.
- `scripts/research/fast_lane_promote_queue.py`, `cherry_pick_ranker.py`, `fast_lane_to_heavyweight_bridge.py` — 2A.3 scope.
- `trading_app/live/**`, `validated_setups`, `lane_allocation.json`, `chordia_audit_log.yaml` — capital-class boundary inviolate. Ledger writer must REFUSE to write entries naming these.

## Acceptance Criteria

1. **Drift count: 168 → 170.** `python pipeline/check_drift.py` reports +2 checks net; all pass modulo the pre-existing MGC `MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4` trade-window staleness.
2. **#169 fail-closed injections** (6 probes): all assert fail-closed.
3. **#170 fail-closed injections** (4 probes): all assert fail-closed.
4. **Capital-class refusal test:** unit test against ledger writer asserts raise on attempt to write any entry whose `prereg_path` points to a capital-class file. (Test lives in `test_check_drift_fast_lane_trial_ledger_append_only.py` even though it's behavioral — keeps the boundary in the same suite.)
5. **Graveyard digest content check:** the live-built `fast_lane_graveyard_digest.yaml` contains every entry from `06_RD_GRAVEYARD.md` + `STRATEGY_BLUEPRINT.md` § NO-GO + `action-queue.yaml` parked entries. Spot-check by reading the digest after build and grep-confirming a known NO-GO entry's `structural_hash` is present.
6. **Empty-ledger sanity:** with `entries: []`, Check #169 passes (vacuous). With one synthetic entry, check passes. With two entries where the second's timestamp < the first's, check fails.
7. **`structural_hashes.yaml` skeleton:** Check #169 does NOT enforce parity between ledger and hashes-index (2A.3 builds that); this stage only lands the skeleton file with the banner + empty `index: {}`.

## Pre-Implementation Probes (run BEFORE any edit)

1. `python pipeline/check_drift.py` — baseline = 168.
2. `ls docs/runtime/fast_lane_*.yaml` — confirm NONE of the three derived-state files exist yet.
3. `grep -nE "^def check_fast_lane_(trial_ledger|graveyard_digest)" pipeline/check_drift.py` — confirm zero hits.
4. `ls scripts/research/fast_lane_trial_ledger.py scripts/research/fast_lane_graveyard_digest.py` — confirm both modules do not exist.

If any probe reports unexpected state, STOP and re-scope; the prior session may have advanced the work.

## Execution Sequence

1. **Pre-flight probes (above).**
2. **Author `fast_lane_trial_ledger.py`** — schema dataclass, writer with capital-class refusal, reader with append-only invariant. No I/O against the YAML yet.
3. **Author `fast_lane_graveyard_digest.py`** — source-MD parsing + `structural_hash` computation via 2A.1's `compute_structural_hash`. No I/O against the YAML yet.
4. **Land skeletons:**
   - `docs/runtime/fast_lane_trial_ledger.yaml` with banner + `schema_version: 1` + `entries: []`.
   - `docs/runtime/fast_lane_structural_hashes.yaml` with banner + `schema_version: 1` + `index: {}`.
   - `docs/runtime/fast_lane_graveyard_digest.yaml` built by running `python -m scripts.research.fast_lane_graveyard_digest --build` once.
5. **Add Check #169 + tests** — land + verify all 6 injection probes fail-closed; check passes against the skeleton ledger.
6. **Add Check #170 + tests** — land + verify all 4 injection probes fail-closed; check passes against the live-built digest.
7. **Drift count audit:** `python pipeline/check_drift.py` → 170 checks; +2 net vs 2A.1 baseline.
8. **Stage close** with a commit citing parent split file + design grounding + Bailey-López de Prado 2014 § 3 (universe-of-trials) + `06_RD_GRAVEYARD.md` re-opening rule.

## Risks / Blind Spots (carry-over from design grounding § "Risks / Blind Spots")

- **Ledger truthfulness depends on disciplined writes** — 2A.3 wires the actual call sites; until then the ledger remains `entries: []` and Check #170 (provenance present, ships in 2A.3) is the bypass-class closer.
- **Graveyard digest staleness** — digest rebuilds on every invocation in the module (`fast_lane_graveyard_digest.py --build`). 2A.3's scanner calls this on every walk; until then the digest only updates when manually rebuilt. Document this in the stage-close commit so the operator knows when to rebuild.
- **Hash brittleness on canonical-source change** — already protected by Check #167 (2A.1). No new exposure in this stage.
- **n=1 meta-tooling on capital-class refusal:** the writer's refusal to write entries naming `validated_setups`/`chordia_audit_log.yaml`/`lane_allocation.json` is a forcing function on an n=0 surface (no documented bypass attempt). Justification: capital-class boundary is a project invariant (CLAUDE.md § Database Location, § Source-of-Truth Chain Rule); refusal is mechanical because the boundary is mechanical. Aligns with `feedback_n3_same_class_doctrine_threshold.md` rationale at the class level, not the incident level.

## Out of Scope (handled by 2A.3 or later)

- Scanner integration: emitting `structural_hash`, `K_global_observed`, `K_family_observed`, `K_lane_observed`, `N_hat` per PROMOTE/QUEUED entry.
- Bridge refusal logic for sibling-retest / dup-active / banned-entry / E2-lookahead / K-overrun.
- Ranker propagation refusal.
- Empirical ρ̂ estimation (Stage 2B, post N≥50 ledger entries).
- Status roll-up wiring (Stage 2B).

## Stage Acceptance

When this stage closes:

- `git diff main --stat` shows only files in the `scope_lock` above.
- `python pipeline/check_drift.py` → 170 checks, 169 pass + 1 pre-existing MGC violation.
- `pytest tests/test_pipeline/test_check_drift_fast_lane_trial_ledger_append_only.py tests/test_pipeline/test_check_drift_fast_lane_graveyard_digest_parity.py -q` → all pass.
- `docs/runtime/fast_lane_graveyard_digest.yaml` exists, has banner, and contains every source NO-GO entry.
- Commit cites parent split + design grounding + Bailey-López de Prado 2014 § 3 + `06_RD_GRAVEYARD.md`.
- Parent split file (`2026-05-20-fast-lane-anti-fp-implementation.md`) gets its Sub-Stage 2A.2 section annotated `[CLOSED <commit-sha>]` in a follow-up edit (deferred to stage-close commit, not opened here).

## Provenance

- **2026-05-19** prior session: parent design grounding authored.
- **2026-05-19** prior session: parent split file authored (3-sub-stage IMPLEMENTATION).
- **2026-05-19** prior session: Sub-Stage 2A.1 shipped (`compute_structural_hash` + Check #167 + 14 tests).
- **2026-05-20** (this session): Sub-Stage 2A.2 stage file opened with locked `scope_lock`. No production code written this session.
- **Next session:** open this stage, run § Pre-Implementation Probes, execute § Execution Sequence steps 1–8.
