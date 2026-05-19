---
task: Stage 2A.3 IMPLEMENTATION — Fast-Lane Anti-FP Scanner / Ranker / Bridge Wiring. Wires Stage 2A.1 (structural_hash) + Stage 2A.2 (trial ledger + graveyard digest) into the live fast-lane chain. After this stage the scanner emits 5 provenance fields per PROMOTE candidate, the ranker refuses to rank provenance-less entries, the bridge refuses to author drafts on suppression-rule matches, and `promote_queue.yaml` carries enforceable trial-accounting state. Closes Bailey-Lopez de Prado 2014 § 3 "universe of trials" gap on the live path. Parent split file: docs/runtime/stages/2026-05-20-fast-lane-anti-fp-implementation.md § "Sub-Stage 2A.3". Design grounding: docs/runtime/stages/2026-05-20-fast-lane-anti-fp-trial-provenance.md.
mode: IMPLEMENTATION
scope_lock:
  - scripts/research/fast_lane_promote_queue.py
  - scripts/research/cherry_pick_ranker.py
  - scripts/research/fast_lane_to_heavyweight_bridge.py
  - pipeline/check_drift.py
  - pipeline/canonical_inline_copies.py
  - tests/test_pipeline/test_check_drift_fast_lane_promote_queue_provenance_present.py
  - tests/test_research/test_fast_lane_promote_queue_suppression.py
  - tests/test_research/test_fast_lane_to_heavyweight_bridge_refusal.py
---

## Blast Radius

- `scripts/research/fast_lane_promote_queue.py` — scanner. Extends `PromoteEntry` dataclass with 5 provenance fields (`structural_hash`, `k_lineage`, `N_hat`, `upstream_k_role`, `upstream_k_value`); adds 6 suppression statuses to `STATUS_VALUES`; calls `fast_lane_trial_ledger.append_entry()` + `fast_lane_structural_hash.structural_hash()` + reads `fast_lane_graveyard_digest.yaml` per scan; emits enriched cache to `promote_queue.yaml`. Existing Check #157 (orphan reconstruction) re-runs unchanged.
- `scripts/research/cherry_pick_ranker.py` — ranker. Fail-closed guard: refuses to rank entries lacking `structural_hash` / `k_lineage` / `N_hat`. Propagates fields into the journal append-only row. Existing Check #160 (`HEAVYWEIGHT_T_THRESHOLD` parity) unchanged.
- `scripts/research/fast_lane_to_heavyweight_bridge.py` — bridge. Adds ONE pre-flight refusal step before draft authoring. Refusal triggers: `K_lane >= 2` (sibling-retest), graveyard structural-hash match, banned entry_model (E0/E3), E2 look-ahead filter prefix/substring. Delegates E2 detection to `trading_app.config.E2_EXCLUDED_FILTER_PREFIXES` / `E2_EXCLUDED_FILTER_SUBSTRINGS` (canonical; never re-encode per institutional-rigor § 10). Existing Check #161 (bridge methodology parity) unchanged because suppression rules are upstream of methodology.
- `pipeline/check_drift.py` — adds Check #171 `check_fast_lane_promote_queue_provenance_present`. Fail-closed on any PROMOTE/QUEUED entry in `promote_queue.yaml` lacking the 5 provenance fields.
- `pipeline/canonical_inline_copies.py` — adds one `InlineCopyPair` entry: 6 suppression-status string values inlined in `fast_lane_promote_queue.STATUS_VALUES` must mirror the canonical enum defined in this stage file's § "Suppression Status Enum" below. Meta-check #159 enforces.
- Reads: `docs/runtime/promote_queue.yaml`, `fast_lane_trial_ledger.yaml`, `fast_lane_graveyard_digest.yaml`, `fast_lane_structural_hashes.yaml`, the scanner's existing inputs (result MDs, revocation sidecars, action-queue).
- Writes (additive only): `promote_queue.yaml` (new fields per entry), `fast_lane_trial_ledger.yaml` (one new ledger entry per scan), `cherry_pick_ranking_<date>.csv` (new columns), `cherry_pick_journal.yaml` (new fields), `docs/audit/hypotheses/drafts/` (unchanged behaviour; only refusal-path added).
- Capital-class? **No.** Bridge writes only to `docs/audit/hypotheses/drafts/` exactly as today; `chordia_audit_log.yaml` / `validated_setups` / `lane_allocation.json` / `trading_app/live/` boundary untouched. Stage 2A.2's `CapitalClassWriteRefused` continues to gate the ledger.

## Predecessor and Gate Status

- **Stage 2A.1** (`b643bfc0` + `5c6040c3`): structural_hash module + Check #167 hash-schema parity + 14 tests. **Landed on main.**
- **Stage 2A.2** (`cec5a8d5`): trial ledger + graveyard digest writers + Checks #169 + #170 + 24 tests. **Landed on main 2026-05-20 via merge commit `b8aba2f1`** (pushed to origin); drift count 147 PASSED on main → 149 PASSED post-merge (verified pre-stage); both new checks active and passing.
- **Gate condition met** per parent split file § Stage-Gate Discipline ("2A.3 cannot open until 2A.2 is committed + pushed + drift check passes").

## Suppression Status Enum (canonical — mirrored by `canonical_inline_copies.py`)

Six new values added to `fast_lane_promote_queue.STATUS_VALUES`. Existing values (`QUEUED`, `ESCALATED`, `REVOKED`, `PARKED`, `REJECTED_OOS_UNPOWERED`, `ERROR`) preserved.

| Status | Trigger | Bridge action |
|---|---|---|
| `SUPPRESSED_GRAVEYARD` | `structural_hash` matches an entry in `fast_lane_graveyard_digest.yaml` | refuse |
| `SUPPRESSED_DUPLICATE_ACTIVE` | `structural_hash` matches an existing prereg under `docs/audit/hypotheses/` (NOT `drafts/`) that has not yet produced a result MD with the same hash | refuse |
| `SUPPRESSED_SIBLING_RETEST` | `K_lane >= 2` (ledger contains >=2 entries sharing this structural_hash) | refuse |
| `SUPPRESSED_BANNED_ENTRY_MODEL` | `entry_model in {E0, E3}` | refuse |
| `SUPPRESSED_E2_LOOKAHEAD` | `entry_model == E2` AND `filter_type` matches `E2_EXCLUDED_FILTER_PREFIXES` / `E2_EXCLUDED_FILTER_SUBSTRINGS` (canonical from `trading_app.config`) | refuse |
| `SUPPRESSED_K_OVERRUN` | `N_hat >= K_declared_in_prereg * 2` (effective-N exceeds budget per Bailey-Lopez de Prado 2014 Eq. 9) | refuse |

OOS-power gate (Check #161) stays unchanged and remains upstream of these suppression statuses on the verdict-priority order. If a result MD already carries `REJECTED_OOS_UNPOWERED`, that wins; suppression statuses below only fire on results that would otherwise reach `QUEUED`.

## K-Lineage Schema (per PROMOTE/QUEUED entry)

```yaml
k_lineage:
  K_global: <int>          # total ledger entries at scan time
  K_family: <int>          # ledger entries sharing (instrument, orb_label, orb_minutes)
  K_lane: <int>            # ledger entries sharing full structural_hash
  K_declared_in_prereg: <int>
  K_effective_minBTL: <float>   # 2 * ln(K_global) / E[max_N]^2 per Bailey 2013 Thm 1
  bh_fdr_passes:
    K_family: bool
    K_lane: bool
    K_global: bool         # informational; not a hard gate per backtesting-methodology RULE 4
  correlation_haircut_N_hat: <int>   # via DSR Eq. 9: rho_hat + (1 - rho_hat) * M_correlated
  rho_hat_assumed: 0.5     # conservative prior; documented per design grounding § Risks
```

`rho_hat = 0.5` is the conservative prior per design grounding § "Effective-N Accounting". Empirical fit deferred to Stage 2B once ledger >= 50 entries (currently 0; this stage seeds the first batch). Inlining the prior is intentional — it is the gate value that Check #171 enforces is present and identical across entries.

## Acceptance Criteria

1. `python pipeline/check_drift.py` count rises **170 -> 171**; all PASSED modulo pre-existing MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4 trade-window carry-over (orthogonal).
2. **Check #171** `check_fast_lane_promote_queue_provenance_present` PASSED. Fail-closed on (a) any PROMOTE / QUEUED entry in `promote_queue.yaml` lacking `structural_hash` / `k_lineage` / `N_hat`, (b) `rho_hat_assumed != 0.5` (locked prior), (c) banner scrub on the cache file.
3. **6 scanner suppression integration tests** pass — one per row in the enum table above. Each uses a synthetic fixture (ledger + digest + active-prereg state) that triggers exactly one suppression status; asserts the scanner emits that status and writes the entry to `promote_queue.yaml` with the provenance fields.
4. **4 bridge refusal-path tests** pass — one per non-OOS refusal trigger (K_lane>=2, graveyard match, banned entry, E2 look-ahead). Each constructs a QUEUED entry with the suppression match and asserts the bridge raises `BridgeRefused` (or equivalent) WITHOUT writing to `docs/audit/hypotheses/drafts/`. Verified by file-state assertion before/after.
5. **4 Check #171 injection tests** pass — (a) strip `structural_hash` from one entry, (b) NULL one `k_lineage` field, (c) missing `N_hat`, (d) hand-edit cache to mutate `rho_hat_assumed`. All fail-closed.
6. **Self-review pass on live QUEUED entry** `MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30` — running the scanner against the live repo emits the entry with: `K_lane = 1` (no sibling retest yet), `structural_hash` present and reproducible, no graveyard match (verified against current 57-entry digest), no E2 look-ahead (entry is E1), `N_hat` sane (a positive integer). If any of these diverge, the design has a bug — not the entry.
7. **Capital-class greppable check clean**: `grep -nE "(validated_setups|chordia_audit_log\.yaml|lane_allocation\.json|trading_app/live/)" scripts/research/fast_lane_promote_queue.py scripts/research/cherry_pick_ranker.py scripts/research/fast_lane_to_heavyweight_bridge.py` returns only docstring / banner mentions of the boundary, never an active write or import path that targets these surfaces.
8. **Canonical delegation verified**: scanner reads `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS` / `pipeline.dst.SESSION_CATALOG` / `trading_app.config.ALL_FILTERS` via the existing 2A.1 `fast_lane_structural_hash` module (never re-encodes). Bridge reads `trading_app.config.E2_EXCLUDED_FILTER_PREFIXES` / `E2_EXCLUDED_FILTER_SUBSTRINGS` directly (never re-encodes). `grep -nE "E2_EXCLUDED_FILTER_(PREFIXES|SUBSTRINGS)\s*=" scripts/research/` returns ZERO hits.
9. **Holdout sentinel sanity**: every ledger entry the scanner appends carries `holdout_policy: mode_A` and `holdout_sacred_from: 2026-01-01` (Stage 2A.2's append_entry already enforces this; the scanner just passes through).

## Files NOT to Touch

- `research/chordia_strict_unlock_v1.py` — bounded runner; behaviour locked.
- `scripts/research/fast_lane_structural_hash.py` (2A.1), `scripts/research/fast_lane_trial_ledger.py` (2A.2), `scripts/research/fast_lane_graveyard_digest.py` (2A.2) — consumed via import only; no edits.
- `scripts/research/cherry_pick_grounder.py`, `cherry_pick_journal_enricher.py`, `llm_hypothesis_proposer.py` — orthogonal.
- Anything under `trading_app/live/`, `validated_setups`, `lane_allocation.json`, `chordia_audit_log.yaml` — capital-class boundary inviolate.
- `pipeline/cost_model.py`, `pipeline/dst.py`, `trading_app/holdout_policy.py`, `trading_app/config.py` — canonical sources; delegation only, never re-encode.

## Approach (execution order)

1. Open this stage file (this commit only — establishes scope_lock + parent gate).
2. Register suppression-status enum in `pipeline/canonical_inline_copies.py` (new `InlineCopyPair`).
3. Extend scanner: add 5 provenance fields to `PromoteEntry`, add 6 STATUS_VALUES, wire ledger.append + structural_hash + graveyard-digest read per entry, emit enriched cache.
4. Add ranker guard: refuse to rank entries lacking provenance; propagate to journal.
5. Add bridge pre-flight refusal: 4 triggers (K_lane>=2, graveyard, banned entry, E2 look-ahead).
6. Add Check #171 + 4 injection tests.
7. Add 6 scanner suppression integration + 4 bridge refusal tests.
8. Verify (drift + tests + self-review on live QUEUED entry).
9. Commit + push.

## Risks / Blind Spots

1. **`rho_hat = 0.5` is a prior, not a measurement.** Inherits from design grounding § Risks #1. Mitigation: documented, locked, parity-checked. Empirical fit deferred to Stage 2B once N >= 50 ledger entries.
2. **Scanner now writes the ledger.** Every scanner invocation appends one entry per PROMOTE/QUEUED result MD it processes. First run after 2A.3 lands will write the universe-of-trials baseline (currently 0 entries). This is intentional; the ledger is supposed to be append-only and reflect the full history of scanned results.
3. **Bridge refusal is a forcing function.** Per design grounding § Risks #6 — justification = the n=3+ doctrine threshold (E2 lookahead 24 TAINTED + canonical-inline-copy 5 instances + Chordia threshold parity 4 instances) plus structural design need. Mechanical because the bug class is mechanical.
4. **Stage-numbering ambiguity (open meta-finding from session start).** Two unrelated commits both labelled "Stage 2A.3" exist (older `bbd1e479` connective-tissue orchestrator, this anti-FP wiring). Out-of-scope here; tracked separately as a hardening proposal.

## Stage-Gate Transition Plan

| Step | Mode | Action |
|---|---|---|
| 0 | IMPLEMENTATION (this file) | Stage file opened; scope_lock locked. |
| 1 | IMPLEMENTATION | Implement steps 2-7 in Approach above. |
| 2 | VERIFICATION | Drift 170 -> 171; 14 new tests pass; self-review on live QUEUED entry. |
| 3 | COMMIT | Single commit citing this stage file + Bailey-Lopez de Prado 2014 § 3 + feedback_n3_same_class_doctrine_threshold.md + parent split file. Push to origin/main. |
| 4 | STAGE 2B handoff | Status roll-up surface (formerly Stage 2) can now layer atop a guarded funnel. Ledger has its first entries; empirical `rho_hat` estimation activates once N >= 50. |

## Authoring Provenance

- **Author:** Claude (Opus 4.7, explanatory mode).
- **Authoring session:** 2026-05-20. Prior session merged anti-FP 2A.2 onto main (merge commit `b8aba2f1`), satisfying the parent split file's gate condition.
- **Approval:** user approved the design proposal in the same session; this stage file is the first commit of the implementation arc.
