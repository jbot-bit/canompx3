---
task: Stage 2A IMPLEMENTATION — Fast-Lane Anti-FP Trial Provenance Layer. Split into 3 sub-stages (2A.1/2A.2/2A.3) so each fits a single /clear-window. Reads design grounding at docs/runtime/stages/2026-05-20-fast-lane-anti-fp-trial-provenance.md.
mode: IMPLEMENTATION
scope_lock:
  - pipeline/canonical_inline_copies.py
  - pipeline/check_drift.py
  - scripts/research/fast_lane_structural_hash.py
  - tests/test_research/test_fast_lane_structural_hash.py
  - tests/test_pipeline/test_check_drift_fast_lane_structural_hash_schema_parity.py
---

## Blast Radius

Stage 2A is SPLIT. Only Sub-stage 2A.1 is in active scope_lock above. Sub-stages 2A.2 and 2A.3 are queued — their scope_locks are documented below but NOT active until their own stage files open.

**Why split:** prior session's Stage 1 was interrupted by Tier-4 /clear at this exact scope size. Repeating the scope = repeating the bug. Each sub-stage caps at ≤5 production files + tests, fits a single /clear-window, ships independently.

---

## Sub-Stage 2A.1 — Structural Hash (ACTIVE)

**Goal:** Single source of truth for the 16-hex `structural_hash`. Foundational — every later sub-stage consumes this module.

**Files:**
- `pipeline/canonical_inline_copies.py` (+1 InlineCopyPair: HASH_SCHEMA_VERSION)
- `pipeline/check_drift.py` (+Check #167 hash-schema parity + CHECKS entry)
- `scripts/research/fast_lane_structural_hash.py` (NEW; ~120 lines)
- `tests/test_research/test_fast_lane_structural_hash.py` (NEW; 8 tests — 6 unit + 2 property)
- `tests/test_pipeline/test_check_drift_fast_lane_structural_hash_schema_parity.py` (NEW; 4 injection tests)

**Numbering note:** original design assigned hash parity to Check #168. Split swaps to Check #167 (foundational first). Original #167 (ledger append-only) moves to 2A.2 as Check #168.

**Reads:** `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS`, `pipeline.dst.SESSION_CATALOG`, `trading_app.config.ALL_FILTERS` (canonical delegation).

**Writes:** zero derived-state files (the hash function is pure; the ledger that uses it ships in 2A.2).

**Acceptance for 2A.1:**
1. `python pipeline/check_drift.py` count rises 166 → 167; all pass modulo pre-existing MGC carry-over.
2. 12 new tests pass.
3. Canonical-inline-copy meta-check #159 still passes (HASH_SCHEMA_VERSION has a live parity check + sibling tests).
4. Hash is reproducible: same inputs → same hash across runs.
5. Property test: order-stable JSON serialisation regardless of dict insertion order.

---

## Sub-Stage 2A.2 — Trial Ledger + Graveyard Digest (CLOSED 2026-05-20)

**Stage file:** `docs/runtime/stages/2026-05-20-fast-lane-anti-fp-2a2-ledger-digest.md`
**Drift target hit:** 168 → 170 (+2 net). Final count: 149 passed + 20 advisory + 1 pre-existing MGC violation = 170 checks total.
**Tests landed:** 24 (16 ledger + 8 digest) — all pass.
**Numbering note:** the original split assigned #168 + #169, but #168 was claimed by `check_fast_lane_status_rollup_reconstruction_parity` (landed independently in another branch); 2A.2 renumbered to #169 + #170 to avoid collision. The check functions and tests carry the renumbered identifiers throughout.



**Goal:** Append-only universe-of-trials ledger + NO-GO graveyard digest as derived state.

**Files (proposed scope_lock; not active until stage file opens):**
- `scripts/research/fast_lane_trial_ledger.py` (NEW; ~180 lines)
- `scripts/research/fast_lane_graveyard_digest.py` (NEW; ~150 lines)
- `docs/runtime/fast_lane_trial_ledger.yaml` (NEW derived-state)
- `docs/runtime/fast_lane_graveyard_digest.yaml` (NEW derived-state)
- `docs/runtime/fast_lane_structural_hashes.yaml` (NEW derived-state index)
- `pipeline/check_drift.py` (+Check #168 ledger append-only + Check #169 graveyard digest parity)
- `tests/test_pipeline/test_check_drift_fast_lane_trial_ledger_append_only.py` (NEW; 4 injection)
- `tests/test_pipeline/test_check_drift_fast_lane_graveyard_digest_parity.py` (NEW; 4 injection)

**Acceptance for 2A.2:** check count 167 → 169; ledger writer refuses to write to capital-class files (mock-fs test); digest contains every NO-GO entry from `06_RD_GRAVEYARD.md` + `STRATEGY_BLUEPRINT.md` § NO-GO.

---

## Sub-Stage 2A.3 — Scanner / Ranker / Bridge Wiring (QUEUED)

**Goal:** Connect 2A.1+2A.2 substrate into the live chain. Suppression rules become enforceable.

**Files (proposed scope_lock; not active until stage file opens):**
- `scripts/research/fast_lane_promote_queue.py` (emit structural_hash + k_lineage + N_hat + 6 suppression statuses)
- `scripts/research/cherry_pick_ranker.py` (require provenance fields)
- `scripts/research/fast_lane_to_heavyweight_bridge.py` (pre-flight refusal)
- `pipeline/check_drift.py` (+Check #170 promote-queue provenance present)
- `tests/test_pipeline/test_check_drift_fast_lane_promote_queue_provenance_present.py` (NEW; 4 injection)
- `tests/test_research/test_fast_lane_promote_queue_suppression.py` (NEW; 6 integration)
- `tests/test_research/test_fast_lane_to_heavyweight_bridge_refusal.py` (NEW; 4 refusal-path)

**Acceptance for 2A.3:** check count 169 → 170; bridge refuses synthetic K_lane≥2 fixture; self-review pass on live QUEUED `MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30`; capital-class greppable check clean.

---

## Predecessor

- Design grounding: `docs/runtime/stages/2026-05-20-fast-lane-anti-fp-trial-provenance.md` (commit `f6b65420`).
- Stage 1: commit `ded02d2a` (state-graph spec + Check #166).

## Stage-Gate Discipline

- Each sub-stage gets its own stage file when it opens.
- 2A.2 cannot open until 2A.1 is committed + pushed + drift check passes.
- 2A.3 cannot open until 2A.2 is committed + pushed.
- Self-review pass after every sub-stage; no batching.
