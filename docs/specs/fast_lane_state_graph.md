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
    schema_version: 1
    description: |
      Per-strategy_id status roll-up: current_stage, age_days, next_action_token,
      upstream_artifact_path, downstream_artifact_path. Rebuilt on every
      orchestrator run.
    parity_drift_check: null  # to be added in Stage 2
    proposed: true  # Stage 2 of fast-lane-pipeline-connective-tissue plan; not yet implemented

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

### 5.1 `fast_lane_status.yaml` — Stage 2 deliverable (PROPOSED)

Top-level keys: `schema_version: 1`, `generated_at`, `do_not_hand_edit: true`, `entries[]`.
Entry fields: `strategy_id`, `current_stage` (one of: ACTIVE_PREREG, FAST_LANE_RUN, PROMOTE_QUEUED, RANKED, BRIDGED, GROUNDED, HEAVYWEIGHT_PENDING, HEAVYWEIGHT_COMPLETE, ENRICHED), `age_days`, `next_action_token`, `upstream_artifact_path`, `downstream_artifact_path`.

NOT yet drift-checked. Will be enforced when Stage 2 lands.

### 5.2 `fast_lane_age_staleness.yaml` — Stage 3 deliverable (PROPOSED)

Top-level keys: `schema_version: 1`, `generated_at`, `entries[]`.
Entry fields: `strategy_id`, `age_at_queued_days`, `age_at_ranked_days`, `age_at_bridged_days`, `age_at_grounded_days`, `age_at_enriched_days` (each null if stage not reached).

NOT yet drift-checked. Will be enforced when Stage 3 lands.

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
