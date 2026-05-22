# Fast-Lane Prereg Pipeline — State Graph (Canonical Reference)

**Status:** AUTHORITATIVE — single source of truth for the fast-lane prereg pipeline's derived-state nodes, edges, and schema contracts. Stage scripts reference this file from their module docstrings rather than re-inlining the graph.

**Created:** 2026-05-19 (Stage 1 of `docs/plans/2026-05-19-fast-lane-pipeline-connective-tissue-design.md`, Approach A approved).

**Parity enforcement:** `pipeline/check_drift.py::check_fast_lane_state_graph_node_parity` parses the `## Node Inventory` block below and verifies that:
- every `path:` entry exists on disk OR carries `proposed: true` (Stage 2/3 reserved schemas).
- every derived-state file in the chain's known glob roots is named in this doc.

Schema changes to any active node require an amendment commit to this file.

---

## 1. Chain Overview

The fast-lane prereg pipeline is a 9-stage chain from idea to live deployment:

```
idea
  → prereg YAML (active or draft)
  → fast-lane scan result MD
  → PROMOTE queue (derived state)
  → cherry-pick rank (derived state)
  → heavyweight Chordia draft
  → grounded draft (optional theory upgrade)
  → heavyweight result MD
  → journal closure (derived state)
  → operator deployment (capital-class — outside this graph's mutation scope)
```

Every stage is fail-closed and idempotent. No stage in this graph mutates capital-class files (`docs/runtime/chordia_audit_log.yaml`, `docs/runtime/lane_allocation.json`, `validated_setups` DB table, anything under `trading_app/live/`). Operator deployment at the terminal node is explicitly outside this graph.

---

## 2. Node Inventory

Each node is a derived-state file or directory written by exactly one script in the chain. Hand-edits are forbidden; the writer rebuilds from upstream sources on every run.

```yaml
# PARSED BY check_fast_lane_state_graph_node_parity — do not change format
# without updating the parser and its injection tests.
nodes:
  - id: promote_queue
    path: docs/runtime/promote_queue.yaml
    writer: scripts/research/fast_lane_promote_queue.py
    schema_version: 1
    description: |
      PROMOTE-queue cache reconstructed from result MDs, revocation sidecars,
      heavyweight preregs (active only — drafts/ excluded), action-queue park
      entries, and the canonical OOS window from holdout_policy.
    parity_drift_check: check_fast_lane_promote_orphans

  - id: cherry_pick_journal
    path: docs/runtime/cherry_pick_journal.yaml
    writer: scripts/research/cherry_pick_ranker.py
    schema_version: 1
    description: |
      Append-only journal of cherry-pick decisions on PROMOTE survivors.
      Records ranking score components, deflation_headroom, oos_power_tier,
      and (after enricher run) heavyweight_verdict + lesson_label.
    parity_drift_check: check_cherry_pick_journal_integrity

  - id: cherry_pick_ranking_csv
    path: docs/runtime/cherry_pick_ranking_*.csv
    writer: scripts/research/cherry_pick_ranker.py
    schema_version: 1
    description: |
      Per-run ranking CSV (one file per invocation, dated). Snapshot of the
      ranker's view at run time. Glob pattern: cherry_pick_ranking_<YYYY-MM-DD>.csv.
    parity_drift_check: null  # snapshot files; staleness tolerated by design

  - id: heavyweight_drafts_dir
    path: docs/audit/hypotheses/drafts/
    writer: scripts/research/fast_lane_to_heavyweight_bridge.py
    schema_version: 1
    description: |
      Quarantine directory for bridge-generated heavyweight Chordia DRAFTs.
      Loader does NOT walk this directory — operator hand-promotes a draft
      to docs/audit/hypotheses/ after theory-grounding or strict-t acceptance.
    parity_drift_check: check_triage_provenance_completeness

  - id: fast_lane_status_rollup
    path: docs/runtime/fast_lane_status.yaml
    writer: scripts/tools/fast_lane_status.py
    schema_version: 2
    description: |
      Per-strategy_id status roll-up: current_stage, age_days, next_action_token,
      lineage_class, blocker_class, primary_blocker, blocker_evidence, and
      artifact provenance. Rebuilt on every orchestrator run.
    parity_drift_check: check_fast_lane_status_rollup_reconstruction_parity

  - id: fast_lane_age_staleness
    path: docs/runtime/fast_lane_age_staleness.yaml
    writer: scripts/tools/fast_lane_age_staleness.py
    schema_version: 1
    description: |
      Per-strategy_id age signal at each stage transition. Operator-visible
      staleness view ("strategy X has been QUEUED for 12 days").
    parity_drift_check: null  # to be added in Stage 3
    proposed: true  # Stage 3 of fast-lane-pipeline-connective-tissue plan; not yet implemented
```

---

## 3. Edge Inventory

Edges are reads between stage scripts (not capital-class writes). All edges are read-only over upstream nodes.

```yaml
edges:
  - from: llm_hypothesis_proposer
    to: prereg_yaml_draft
    via: scripts/research/llm_hypothesis_proposer.py
    artifact: docs/audit/hypotheses/<stem>.draft.yaml
    note: Track A LLM proposer; writes draft, never active prereg.

  - from: prereg_yaml_active
    to: fast_lane_scan_result
    via: research/chordia_strict_unlock_v1.py
    artifact: docs/audit/results/<stem>.md
    note: Runner branches on metadata.template_version (fast_lane_v5.1 vs heavyweight).

  - from: fast_lane_scan_result
    to: promote_queue
    via: scripts/research/fast_lane_promote_queue.py
    note: Scanner reads result MDs + revocation sidecars + heavyweight preregs.

  - from: promote_queue
    to: cherry_pick_ranking_csv
    via: scripts/research/cherry_pick_ranker.py
    note: Ranker scores QUEUED entries by heavyweight-Chordia pass probability.

  - from: cherry_pick_ranking_csv
    to: cherry_pick_journal
    via: scripts/research/cherry_pick_ranker.py
    note: Same script writes both per-run CSV and append-only journal.

  - from: cherry_pick_journal
    to: heavyweight_drafts_dir
    via: scripts/research/fast_lane_to_heavyweight_bridge.py
    note: Bridge generates Chordia DRAFT per top-ranked QUEUED entry (omits theory_citation).

  - from: heavyweight_drafts_dir
    to: heavyweight_drafts_dir
    via: scripts/research/cherry_pick_grounder.py
    artifact: <slug>.grounded.yaml or <slug>.grounded.rejected.txt
    note: Optional theory-upgrade attaches literature citation; same directory.

  - from: heavyweight_scan_result
    to: cherry_pick_journal
    via: scripts/research/cherry_pick_journal_enricher.py
    note: Enricher backfills heavyweight_verdict, t_observed_post_clustered_se, lesson_label.
```

---

## 4. Schema Contracts (active nodes)

### 4.1 `promote_queue.yaml`

Top-level keys: `schema_version: 1`, `generated_at`, `entries[]`.
Entry fields: `strategy_id`, `status` (QUEUED | REVOKED | ERROR | PARKED), `pooled_t`, `pooled_n`, `oos_n`, `oos_power_tier`, `dir_match`, `pooling_artifact`, `result_md_path`, `revocation_md_path`.
Hand-edit detection: `check_fast_lane_promote_orphans` (Check #157) reconstructs independently and diffs.

### 4.2 `cherry_pick_journal.yaml`

Top-level keys: `schema_version: 1`, `entries[]` (append-only — never reordered, never deleted).
Entry fields: `strategy_id`, `decision_date`, `score`, `skip_recommended`, `oos_power_tier`, `deflation_headroom`, `dir_match`, `heavyweight_verdict` (null until enricher fills), `t_observed_post_clustered_se` (null until enricher fills), `lesson_label` (null until enricher fills).
Integrity: monotonic dates, allowed power-tier enum, escalation parity vs `promote_queue.yaml` — `check_cherry_pick_journal_integrity`.

### 4.3 `cherry_pick_ranking_<date>.csv`

Per-run snapshot. Columns: `strategy_id`, `score`, `deflation_headroom`, `n_adequacy`, `oos_power_readiness`, `dir_match`, `non_artifact`, `era_stability_proxy`, `skip_recommended`. No drift check — snapshot files accumulate by design.

### 4.4 `docs/audit/hypotheses/drafts/`

Directory of `*.draft.yaml`, `*.grounded.yaml`, and `*.rejected.txt`. Loader skips this directory; operator hand-promotes by `git mv` into `docs/audit/hypotheses/`. Triage-provenance integrity: `check_triage_provenance_completeness`.

---

## 5. Reserved Schemas (proposed, not yet enforced)

### 5.1 `fast_lane_status.yaml` — Stage 2 / V2 report contract (SHIPPED 2026-05-19; amended 2026-05-22)

Top-level keys: `schema_version: 2`, `generated_at`, `do_not_hand_edit: true`, `source`, `warning`, `entries[]`.
Entry fields: `strategy_id`, `current_stage` (one of: ACTIVE_PREREG, FAST_LANE_RUN, PROMOTE_QUEUED, RANKED, BRIDGED, GROUNDED, HEAVYWEIGHT_PENDING, HEAVYWEIGHT_COMPLETE, ENRICHED, plus terminal stages REVOKED, PARKED, REJECTED_OOS_UNPOWERED, the six `SUPPRESSED_*` statuses from §10, ERROR), `age_days`, `next_action_token`, `upstream_artifact_path`, `downstream_artifact_path`, `observed_at` (per-source provenance dict), `lineage_class`, `blocker_class`, `primary_blocker`, `blocker_evidence`.

`lineage_class` is one of `FAST_LANE`, `DIRECT_HEAVYWEIGHT`, or `UNKNOWN`. Direct heavyweight rows are visible backlog, but they are not selected as Fast Lane next actions.

`blocker_class` is one of `NONE`, `UNDERPOWERED_OOS`, `INVALID_ARTIFACT`, `PROVENANCE_SUPPRESSED`, `PARKED`, or `ERROR`. `primary_blocker` is a stable token such as `oos_power_below_floor`, `pooling_artifact_revoked`, or the lowercase suppression token. `blocker_evidence` carries compact source evidence from `promote_queue.yaml` including status, error reason, structural hash, K lineage, result path, revocation sidecar, or park entry when present.

Drift-checked by `check_fast_lane_status_rollup_reconstruction_parity` (Check #168) — three failure classes: hand-edit drift, tampered banner (`schema_version` / `do_not_hand_edit` / `source`), and capital-class write attempt (greppable static check on writer source). The check imports `scripts.tools.fast_lane_status.SCHEMA_VERSION` rather than freezing the schema integer inline.

**`next_action_token` for HEAVYWEIGHT_COMPLETE is lineage-qualified** (amended 2026-05-20):

- HEAVYWEIGHT_COMPLETE WITH `observed_at.journal_iter` populated (fast-lane lineage present — the ranker scored the strategy and wrote a journal row before the heavyweight ran) → `run_cherry_pick_journal_enricher`. The enricher backfills `heavyweight_verdict + t_observed_post_clustered_se + lesson_label` on the existing journal row.
- HEAVYWEIGHT_COMPLETE WITHOUT a journal entry (no fast-lane lineage — heavyweight Chordia prereg authored directly, predating the 2026-05-19 cherry-pick loop) → `operator_deployment_decision`. The enricher is structurally update-only and cannot create journal rows; the heavyweight result MD already carries the operator-actionable verdict.

This qualifier is enforced by `scripts/tools/fast_lane_status._next_action_for()` and its companion tests in `tests/test_tools/test_fast_lane_status.py`. Without it, the rollup emits a stage script as the next action when running that script would silently no-op — a fail-quiet misclassification of 38 entries was caught and fixed 2026-05-20.

### 5.2 `fast_lane_age_staleness.yaml` — Stage 3 deliverable (DEFERRED)

Top-level keys: `schema_version: 1`, `generated_at`, `entries[]`.
Entry fields: `strategy_id`, `age_at_queued_days`, `age_at_ranked_days`, `age_at_bridged_days`, `age_at_grounded_days`, `age_at_enriched_days` (each null if stage not reached).

NOT yet drift-checked. Deferred on 2026-05-22 because the schema-v2 roll-up and `fast_lane_walk.py` report now expose blocker evidence, lineage separation, and next-action routing directly. Do not add this artifact until there is a distinct operator consumer.

---

## 6. Capital-Class Boundary

This graph terminates at the journal-closure node. Anything downstream of that — `chordia_audit_log.yaml`, `lane_allocation.json`, `validated_setups`, `trading_app/live/*` — is operator-controlled and outside any script in this chain. No connector listed here may write to those paths under any circumstance. The connector design (`docs/plans/2026-05-19-fast-lane-pipeline-connective-tissue-design.md` § 4.1 "Capital-class file mutation") names this boundary explicitly as the highest-severity invariant.

---

## 7. Authority

- This doc is the **single source of truth** for the fast-lane chain's node list, edge list, and schema contracts.
- Stage scripts MUST NOT re-inline the graph in their docstrings; instead, each stage script's module docstring carries one back-reference line: `See docs/specs/fast_lane_state_graph.md for the canonical chain definition.`
- Schema changes to any active node (sections 4.1–4.4) require an amendment commit that updates this doc AND the relevant parity drift check.
- Proposed-schema entries (section 5) are exempt from parity enforcement until the corresponding stage ships.

---

## 8. Related Doctrine

- `docs/plans/2026-05-19-fast-lane-pipeline-connective-tissue-design.md` — design proposal that authored this spec (Stage 1 of 5).
- `.claude/rules/backtesting-methodology.md` — RULE 3.3 OOS power floor, RULE 4 K framing, RULE 6.3 E2 banned columns.
- `.claude/rules/research-truth-protocol.md` — canonical-source delegation rule.
- `.claude/rules/institutional-rigor.md` § 4 — canonical-source authority table (delegation, not re-encode).
- `memory/feedback_canonical_inline_copy_parity_bug_class.md` — class pattern this doc + drift check defend against (6th confirmed instance defense).

---

## 9. Hash Schema (`structural_hash`)

Originally landed in commit `5c6040c3` (Stage 2A.1, 2026-05-20). Relocated here from `docs/runtime/stages/2026-05-20-fast-lane-anti-fp-trial-provenance.md` on 2026-05-21 per the surface-taxonomy doctrine in `docs/governance/system_authority_map.md` (parser-surface blocks may not live under `docs/runtime/`). Parsed by `pipeline/check_drift.py::check_fast_lane_structural_hash_schema_parity` (Check #167).

```yaml
# Canonical inputs to the hash — order-stable, normalised, version-tagged
hash_schema_version: 1
inputs:
  instrument: str           # MNQ | MES | MGC (delegated from pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS)
  orb_label: str            # session catalog key (delegated from pipeline.dst.SESSION_CATALOG)
  orb_minutes: int          # 5 | 15 | 30
  rr_target: float          # 1.0 | 1.5 | 2.0 (rounded to 1 dp)
  entry_model: str          # E1 | E2 | E3  (E3 graveyard-banned — suppression rule fires)
  confirm_bars: int         # 1 | 2
  filter_type: str          # canonical key from ALL_FILTERS, normalised
  direction: str            # LONG | SHORT | BOTH (normalised case)
  filter_threshold: str     # canonical-form string of bound values, e.g. "ATR_P50_ge_0.50"
formula: sha256(canonical_json(inputs))[:16]  # 16-hex-char structural id
```

Hash deliberately **excludes** data window, K framing, t-stat — those are test instances, not structural identity. Two preregs with the same hash test the same lane; that's the de-dup criterion.

The hash is registered in `pipeline.canonical_inline_copies` so any future field addition triggers Check #159's meta-parity + a sibling injection test.

---

## 10. Suppression Status Enum

Originally landed in commit `9b13d0d1` (Stage 2A.3, 2026-05-20). Relocated here from `docs/runtime/stages/2026-05-20-fast-lane-anti-fp-2a3-scanner-bridge-wiring.md` on 2026-05-21 per the same surface-taxonomy doctrine. Parsed by `pipeline/check_drift.py::check_fast_lane_promote_queue_provenance_present` (Check #173). Mirrored by `fast_lane_promote_queue.STATUS_VALUES` (registered in `pipeline/canonical_inline_copies.py`).

Six values added to `fast_lane_promote_queue.STATUS_VALUES`. Existing values (`QUEUED`, `ESCALATED`, `REVOKED`, `PARKED`, `REJECTED_OOS_UNPOWERED`, `ERROR`) preserved.

| Status | Trigger | Bridge action |
|---|---|---|
| `SUPPRESSED_GRAVEYARD` | `structural_hash` matches an entry in `fast_lane_graveyard_digest.yaml` | refuse |
| `SUPPRESSED_DUPLICATE_ACTIVE` | `structural_hash` matches an existing prereg under `docs/audit/hypotheses/` (NOT `drafts/`) that has not yet produced a result MD with the same hash | refuse |
| `SUPPRESSED_SIBLING_RETEST` | `K_declared_in_prereg <= 1` AND `K_lane >= 2` (legacy K=1 lane has repeated prior trials sharing this structural_hash) | refuse |
| `SUPPRESSED_BANNED_ENTRY_MODEL` | `entry_model in {E0, E3}` | refuse |
| `SUPPRESSED_E2_LOOKAHEAD` | `entry_model == E2` AND `filter_type` matches `E2_EXCLUDED_FILTER_PREFIXES` / `E2_EXCLUDED_FILTER_SUBSTRINGS` (canonical from `trading_app.config`) | refuse |
| `SUPPRESSED_K_OVERRUN` | `K_lane > K_declared_in_prereg` (observed trial count exceeds the preregistered lane-level trial budget; `N_hat` remains sample adequacy/DSR input, not the K-budget comparator) | refuse |

OOS-power gate (Check #161) stays unchanged and remains upstream of these suppression statuses on the verdict-priority order. If a result MD already carries `REJECTED_OOS_UNPOWERED`, that wins; suppression statuses below only fire on results that would otherwise reach `QUEUED`.
