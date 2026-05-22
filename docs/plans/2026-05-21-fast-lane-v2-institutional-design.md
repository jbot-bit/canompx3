# Fast Lane V2 Institutional Design

**Date:** 2026-05-21  
**Status:** DESIGN / PLAN — no code changes approved by this document alone  
**Supersedes:** `docs/plans/2026-05-19-fast-lane-pipeline-connective-tissue-design.md` where this document is stricter about provenance and trial-ledger writes.  
**Risk tier:** critical research-control surface, capital-adjacent. The terminal output may inform a separate human capital review, but V2 must not mutate capital-class state.

---

## 1. Purpose

Fast Lane V2 is an institutional research-control loop for turning a candidate trading idea into an evidence-backed research review without collapsing discovery, validation, and capital-control boundaries.

The target operator experience is:

```text
idea -> prereg -> fast-lane run -> triage verdict -> heavyweight verification -> research review packet
```

The forbidden shortcut is:

```text
idea -> strategy -> live lane
```

Fast Lane V2 should make research faster by removing clerical stalls, stale caches, and unclear next actions. It must not make research easier by weakening K accounting, OOS discipline, Chordia thresholds, role mapping, or capital boundaries.

---

## 2. Design Axioms

### Axiom 1 — trial history is execution history

Only a real research execution may create a trial-ledger event.

A real research execution is a run that consumes a pre-registered hypothesis and produces a result artifact from canonical research data. Queue scans, status rebuilds, report rendering, ranker scoring, dry-runs, bridge preflights, and dashboard views are not research executions.

### Axiom 2 — derived views are rebuildable and non-authoritative

`promote_queue.yaml`, `fast_lane_status.yaml`, ranking CSVs, graveyard digests, and walk reports are derived views. They may cache current state for operator convenience, but they are not proof and must be rebuildable from canonical artifacts.

### Axiom 3 — no capital-class mutation

Fast Lane V2 terminates at a report-only research review. It must not write:

- `docs/runtime/chordia_audit_log.yaml`
- `docs/runtime/lane_allocation.json`
- `validated_setups` or other deployment DB tables
- any `trading_app/live/` runtime-control file
- broker state or account routing state

Any downstream capital action remains operator-controlled and separately reviewed.

### Axiom 4 — role before score

Every candidate is classified before evaluation as one of:

- `standalone`
- `filter`
- `conditioner`
- `allocator`
- `confluence`
- `execution`
- `diagnostic`

The role determines the parent population, comparator, primary metric, and promotion target. A candidate that fails as standalone is not automatically dead in a conditional role.

### Axiom 5 — speed comes from automation of ceremony, not relaxation of standards

Fast Lane V2 automates:

- source discovery
- structural duplicate detection
- prereg linting
- status rollups
- next-action routing
- result/journal cross-linking
- stale-hand-off detection

Fast Lane V2 does not automate:

- theory-grant decisions
- active hypothesis promotion from drafts
- heavyweight acceptance
- deployment selection
- live allocation changes

---

## 3. Current Defect Being Fixed

The current fast-lane scanner appends rows to `docs/runtime/fast_lane_trial_ledger.yaml` during scans. The appended `run_id` is timestamp-based, so repeated scanner invocations create repeated ledger entries for the same result files.

Measured consequence on 2026-05-21:

- `docs/runtime/promote_queue.yaml` reports `K_lane=33` for `MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30`.
- A no-append scanner dry-run recomputed the same lane as `K_lane=36`, proving repeated scans had already inflated the ledger beyond the cached value.
- That inflation suppresses candidates as sibling retests even when no new research trial occurred.

This is a research-validity bug. It does not create live-order risk by itself, but it can distort the candidate funnel and make the system reject or misclassify candidates for false provenance reasons.

---

## 4. V2 Architecture

### 4.1 Components

#### `trial_event_log`

An append-only event log for real research executions only.

Event identity:

```text
trial_id = sha256(prereg_sha + runner_id + result_artifact_sha + canonical_data_fingerprint)[:16]
```

Properties:

- idempotent on `trial_id`
- one event per actual execution result
- no timestamp-only identity
- stores source prereg path and SHA
- stores result MD/CSV paths and SHAs
- stores runner id and git SHA
- stores holdout sentinel
- stores K lineage declared at execution time
- never updated by queue scanners or report writers

#### `trial_index`

A rebuildable derived index over the append-only trial ledger plus correction records.
Implementation note (2026-05-22): the current shipped surface is the pure
module `scripts/research/fast_lane_trial_index.py`, derived from
`docs/runtime/fast_lane_trial_ledger.yaml` and
`docs/runtime/fast_lane_trial_corrections.yaml`. There is no authoritative
tracked `fast_lane_trial_index.yaml` cache yet; if a cache is added later, it
must remain rebuildable and drift-checked.

Responsibilities:

- compute `K_global`
- compute `K_family`
- compute `K_lane`
- compute duplicate structural hashes
- expose provenance to scanners/rankers/bridges

It may be cached for speed, but the cache is not authoritative.

#### `promote_queue`

A derived classifier over result artifacts plus trial index.

Responsibilities:

- parse fast-lane result artifacts
- reject parse failures
- classify `PROMOTE`, `KILL`, `NEEDS_MORE`
- enforce per-direction sanity gates
- enforce banned entry models
- enforce E2 lookahead exclusions by delegating to `trading_app.config`
- enforce graveyard lane suppression
- enforce OOS power gate
- read K-lineage from `trial_index`

It must not append to the trial event log.

#### `candidate_ranker`

A derived prioritization layer over queue survivors.

Responsibilities:

- score candidates by heavyweight pass likelihood
- expose score components transparently
- never hide low scores
- never treat score as validation
- write ranking snapshots and journal entries only when explicitly requested

The score is operator triage, not proof.

#### `heavyweight_bridge`

A draft generator from a fast-lane survivor to a heavyweight Chordia prereg draft.

Responsibilities:

- write only under `docs/audit/hypotheses/drafts/`
- preserve upstream provenance
- refuse banned/duplicate/graveyard/K-overrun inputs
- omit theory-citation fields unless explicitly grounded
- default `theory_grant: false`
- set `execution_gate.allowed_now: false`

#### `verifier`

The heavyweight research runner and result/journal closure path.

Responsibilities:

- run the actual heavyweight test from active prereg files
- append trial event only when a real run produces result artifacts
- produce result docs
- enrich journal state from actual result docs

#### `research_review_reporter`

A report-only surface.

Responsibilities:

- join heavyweight verdict, role, OOS power tier, current strategy-lab context, lane allocation context, and capital-boundary constraints
- emit `KILL`, `PARK`, `BULLPEN`, `RECOMMEND_RESEARCH_REVIEW`, or `ESCALATE_CAPITAL_REVIEW`
- never write capital-class state

---

## 5. State Model

### 5.1 Canonical events

Fast Lane V2 uses explicit event types:

| Event | Created by | Meaning |
| --- | --- | --- |
| `IDEA_CAPTURED` | idea intake | A candidate mechanism was recorded. |
| `PREREG_DRAFTED` | prereg writer | A draft prereg exists; not active. |
| `PREREG_ACTIVATED` | operator / prereg gate | An active prereg exists and may be run. |
| `TRIAL_EXECUTED` | research runner only | A real run produced result artifacts. |
| `TRIAGE_CLASSIFIED` | derived classifier | Queue status derived from result + provenance. |
| `RANKED` | ranker | Candidate received a transparent priority score. |
| `HEAVYWEIGHT_DRAFTED` | bridge | A heavyweight prereg draft was written. |
| `HEAVYWEIGHT_ACTIVATED` | operator | Draft was moved to active hypotheses. |
| `HEAVYWEIGHT_EXECUTED` | research runner only | Heavyweight run produced result artifacts. |
| `JOURNAL_ENRICHED` | enricher | Journal was updated from heavyweight result. |
| `CAPITAL_REVIEW_PACKET_EMITTED` | research review reporter | Report-only research review packet was emitted. |

Only `TRIAL_EXECUTED` and `HEAVYWEIGHT_EXECUTED` count toward K lineage.

### 5.2 Derived statuses

Derived queue/status files may report:

- `ACTIVE_PREREG`
- `FAST_LANE_RUN`
- `PROMOTE_QUEUED`
- `RANKED`
- `BRIDGED`
- `GROUNDED`
- `HEAVYWEIGHT_PENDING`
- `HEAVYWEIGHT_COMPLETE`
- `ENRICHED`
- `REVOKED`
- `PARKED`
- `REJECTED_OOS_UNPOWERED`
- `SUPPRESSED_BANNED_ENTRY_MODEL`
- `SUPPRESSED_E2_LOOKAHEAD`
- `SUPPRESSED_GRAVEYARD`
- `SUPPRESSED_DUPLICATE_ACTIVE`
- `SUPPRESSED_SIBLING_RETEST`
- `SUPPRESSED_K_OVERRUN`
- `ERROR`

Derived status names must stay synchronized with `docs/specs/fast_lane_state_graph.md` and drift checks.

---

## 6. Research Standards

Fast Lane V2 must preserve the current project standards:

- Mode A holdout from 2026-01-01.
- Pre-registration before any discovery or confirmation run.
- Canonical discovery truth only from `bars_1m`, `daily_features`, and `orb_outcomes`.
- MinBTL / K-budget before writing or activating preregs.
- BH/FDR where family selection is present.
- Chordia / Harvey-Liu threshold treatment per Criterion 4.
- OOS power disclosure before binary OOS conclusions.
- Role-specific metrics per `conditional-edge-framework.md`.
- No E2 break-bar lookahead filters.
- No post-hoc theory-citation unlock.
- No live/deployment claim from execution records alone.

The funnel must report weak or underpowered evidence honestly. It should prefer `PARK`, `UNVERIFIED_INSUFFICIENT_POWER`, or `BULLPEN` over false certainty.

---

## 7. Efficiency Design

Fast Lane V2 should be efficient in these concrete ways:

1. **Single operator command for status**
   - Rebuild derived views.
   - Print the next action.
   - Print why each blocked candidate is blocked.
   - Never mutate trial history.

2. **Stable artifact identities**
   - Use content hashes and git SHA, not timestamps alone.
   - Make repeated runs idempotent.

3. **Small scoped checks**
   - Fast status checks should not run full drift.
   - Full verification gates run only before accepting or merging code changes.

4. **One authoritative provenance path**
   - Runners emit execution events.
   - Scanners read them.
   - Rankers and bridges read scanners.
   - No component re-derives K independently unless it is a drift check.

5. **Operator-focused output**
   - One recommended next action.
   - One reason a candidate is blocked.
   - One artifact to inspect.
   - No giant unprioritized report as the primary interface.

6. **Cache invalidation by reconstruction**
   - Derived files are rebuilt from source artifacts.
   - Drift checks compare derived files to reconstruction.
   - Hand edits fail closed.

---

## 8. Safety Boundaries

### 8.1 Capital-class write denylist

V2 code must refuse writes whose destination path contains:

- `docs/runtime/chordia_audit_log.yaml`
- `docs/runtime/lane_allocation.json`
- `trading_app/live/`
- `validated_setups`
- broker journals or account-routing state

This denylist belongs in a shared helper so every fast-lane writer uses the same refusal rule.

### 8.2 Draft quarantine

Generated preregs must land under:

```text
docs/audit/hypotheses/drafts/
```

The hypothesis loader must not walk this directory. Promotion to active hypotheses is an operator action and should be visible in git history.

### 8.3 Theory-grant protection

No generated draft may include:

- empty `theory_citation`
- placeholder `theory_citation`
- implied `theory_grant: true`

Theory grant requires local-source grounding and explicit operator approval.

---

## 9. Data Contracts

### 9.1 `fast_lane_trial_event_log.yaml`

Proposed replacement or successor to the current scanner-populated ledger.

```yaml
do_not_hand_edit: true
schema_version: 2
events:
  - trial_id: "16hex"
    event_type: "TRIAL_EXECUTED"
    run_timestamp_utc: "2026-05-21T00:00:00Z"
    git_sha: "abcdef1"
    runner_id: "research/chordia_strict_unlock_v1.py"
    prereg_path: "docs/audit/hypotheses/example.yaml"
    prereg_sha256: "64hex"
    result_md_path: "docs/audit/results/example.md"
    result_md_sha256: "64hex"
    result_csv_path: "docs/audit/results/example.csv"
    result_csv_sha256: "64hex"
    structural_hash: "16hex"
    template_version: "fast_lane_v5.1"
    research_mode: "confirmation"
    question_role: "standalone"
    testing_mode: "individual"
    K_declared: 1
    holdout_policy: "mode_A"
    holdout_sacred_from: "2026-01-01"
    canonical_data_fingerprint:
      source: "gold.db"
      orb_outcomes_max_trading_day: "2026-05-21"
      daily_features_max_trading_day: "2026-05-21"
    outcome_summary:
      verdict: "PROMOTE"
      pooled_t: 3.064
      pooled_n: 226
      pooled_fire: 0.1468
```

### 9.2 `fast_lane_trial_index` derived view

Derived from the trial ledger after V2 corrections. The shipped contract is
the in-memory payload returned by `scripts/research/fast_lane_trial_index.py`.
Do not treat a future YAML cache as authoritative.

```yaml
schema_version: 1
source: "scripts/research/fast_lane_trial_index.py"
total_v2_trials: 28
by_structural_hash:
  "16hex":
    K_structural: 1
    trial_ids: ["16hex"]
by_lane:
  "MNQ|US_DATA_1000|30":
    K_lane: 1
    trial_ids: ["16hex"]
```

### 9.3 `promote_queue.yaml`

Derived from result artifacts and the V2 trial-index view. It must not contain fresh trial events. Its K fields must cite ledger/index provenance and trial IDs once a persisted index cache exists.

---

## 10. Implementation Roadmap

### Phase 0 — freeze current risk

Goal: stop further provenance pollution.

Acceptance:

- `fast_lane_promote_queue.py --dry-run` does not mutate `fast_lane_trial_ledger.yaml`.
- `fast_lane_walk.py --dry-run` does not mutate `fast_lane_trial_ledger.yaml`.
- Back-to-back dry-runs leave ledger bytes unchanged.
- Tests monkeypatch runtime paths or assert real ledger bytes unchanged.

### Phase 1 — split execution events from derived scans

Goal: introduce the V2 event/index model without changing candidate economics.

Acceptance:

- Research runner writes execution events, or a temporary adapter imports existing result artifacts exactly once by stable content identity.
- Scanner reads the event/index model and never appends to it.
- Repeated scanner runs produce identical K fields.
- Existing polluted duplicate rows are preserved as historical artifacts but excluded from V2 K counts via a documented correction file.

### Phase 2 — repair K and suppression semantics

Goal: ensure K gates compare the right quantities.

Acceptance:

- Trial-count gates use trial counts.
- Sample-size and OOS-power gates use sample counts.
- No gate compares `n_hat` trade count to `K_declared` trial count.
- K-overrun language is updated to distinguish `K_global`, `K_family`, `K_lane`, and sample-size adequacy.

### Phase 3 — operator-grade status surface

Goal: make the end-to-end loop efficient.

Acceptance:

- One command prints current status and one next action.
- Every blocked candidate reports exactly one primary blocker plus supporting evidence.
- Status rebuild is read-only over trial history and capital-class files.
- The report distinguishes `candidate is weak`, `candidate is underpowered`, `candidate is invalid`, and `candidate is waiting on operator action`.

### Phase 4 — bridge and verifier hardening

Goal: keep heavyweight transition honest.

Acceptance:

- Bridge refuses any candidate without V2 provenance.
- Bridge refuses stale or polluted K lineage.
- Heavyweight execution creates a new execution event only when a result artifact is produced.
- Journal enrichment is update-only from result docs.
- Theory grant remains explicit and locally grounded.

### Phase 5 — report-only research review

Goal: produce a useful research review without mutating capital-class state.

Acceptance:

- Report joins heavyweight verdict, OOS power, role, current `strategy-lab` context, and allocation context.
- Report emits one of: `KILL`, `PARK`, `BULLPEN`, `RECOMMEND_RESEARCH_REVIEW`, `ESCALATE_CAPITAL_REVIEW`.
- Report includes capital-boundary disclaimer and exact next operator action.
- No writes to `chordia_audit_log.yaml`, `lane_allocation.json`, `validated_setups`, or live runtime paths.

---

## 11. Verification Strategy

Minimum tests before V2 can be trusted:

- Dry-run mutation tests for scanner and walk.
- Idempotency tests for repeated scanner runs.
- Event identity tests proving same prereg/result does not create duplicate trial events.
- Append-only tests for the V2 event log.
- Derived-index reconstruction parity tests.
- Capital-class write refusal tests.
- E2 lookahead suppression tests delegated to canonical config.
- Graveyard suppression tests.
- Duplicate active prereg tests.
- OOS-power gate tests.
- Role-metric tests for at least one standalone and one conditional-role candidate.
- Bridge refusal tests for missing provenance and polluted K lineage.
- End-to-end fixture test:

```text
draft prereg -> active prereg -> mocked result -> execution event -> queue -> rank -> bridge draft -> heavyweight result -> journal enrich -> research review packet
```

The end-to-end fixture must prove no capital-class file is touched.

---

## 12. Hardening Requirements

### 12.1 Fail-closed behavior

Every V2 component must fail closed on:

- missing source prereg
- malformed YAML
- missing result artifact
- missing result hash
- missing trial event for a result artifact
- mismatched prereg SHA
- mismatched result SHA
- missing canonical data fingerprint
- unknown status enum
- unknown research role
- unsupported template version
- unreadable canonical DB
- unavailable OOS-window computation
- malformed strategy_id
- duplicate active prereg for the same structural hash

Fail-closed means the candidate moves to `ERROR`, `PARK`, or a specific `SUPPRESSED_*` state with a human-readable reason. It must not quietly pass, silently disappear, or be rewritten into a weaker blocker.

### 12.2 Idempotency

All rebuild paths must be idempotent:

- running the scanner twice produces the same queue
- running status twice produces the same status
- running ranker twice without new inputs either produces identical output or an explicitly versioned snapshot
- running a dry-run never writes the event log
- replaying the same execution event does not duplicate K lineage

The test suite must include byte-for-byte unchanged assertions for the event log on dry-run paths.

### 12.3 Atomic writes

Derived YAML writers should write to a temporary file and then atomically replace the target. This prevents half-written caches when a process is interrupted.

Append-only event-log writes should validate the full resulting file before replacing it. If validation fails, the previous event log remains intact.

### 12.4 Concurrency guard

V2 should assume Claude, Codex, and terminal scripts may run near each other.

Required behavior:

- one writer lock for event-log append
- no lock needed for read-only scanners
- derived writers may use target-specific lock files
- lock failure reports a clear retry instruction
- no process may hold a lock while running heavyweight research

### 12.5 Schema evolution

Every persisted V2 artifact must carry `schema_version`.

Rules:

- readers accept only known schema versions
- migrations are explicit scripts, never implicit reader rewrites
- new enum values require drift-check parity updates
- deleted fields require an amendment in this design or successor spec
- unknown fields are preserved where safe, but not trusted

### 12.6 Observability

Every operator report should include:

- source artifact path
- current stage
- primary blocker
- next action
- K lineage summary
- OOS power tier
- role and parent/comparator
- whether the candidate is actionable, parked, rejected, or waiting

The report should be compact by default and allow a verbose mode for audit detail.

### 12.7 Audit trail

V2 must prefer correction records over destructive cleanup.

If a prior artifact is wrong, polluted, stale, or superseded:

- keep the original artifact
- write a correction/supersession record
- make derived readers exclude or downgrade the bad artifact based on that record
- add a drift/test case preventing the class from recurring

---

## 13. Edge Cases To Handle

### 13.1 Duplicate and repeated work

- same prereg run twice with identical output
- same prereg run twice with different output
- two preregs with same structural hash
- same result MD copied under a different filename
- result MD exists but source prereg was deleted or renamed
- source prereg exists but result MD has a different strategy_id
- bridge draft exists for a candidate that later becomes suppressed

Expected policy:

- identical execution identity is idempotent
- different output for same prereg is a new event only if the runner and data fingerprint explain the difference
- same structural hash increments lineage only for real executions
- copied result artifacts are detected by hash and refused as duplicates

### 13.2 Bad or partial files

- interrupted YAML write
- empty YAML file
- YAML top-level list where dict expected
- missing `scope`
- missing `metadata.total_expected_trials`
- malformed `orb_minutes`, `rr_target`, or `confirm_bars`
- non-ASCII artifact names
- Windows path separators in repo-relative paths

Expected policy:

- fail closed with exact parse reason
- normalize path separators in serialized artifacts
- avoid relying on filename alone when content has a canonical id

### 13.3 Research-boundary traps

- E2 filter uses break-bar or post-entry information
- theory citation field exists but is empty
- theory grant is inferred from a draft
- 2026 OOS is used to select a rule
- conditional-role finding is promoted as standalone
- deployment analytics result is treated as discovery proof
- live/paper trade record is treated as validation proof

Expected policy:

- refuse or park with a doctrine citation
- surface the correct route: discovery, confirmation, deployment readiness, or operations

### 13.4 Statistical traps

- pooled result hides both directions failing
- low N appears strong due to one-cell luck
- OOS sign matches but is underpowered
- OOS sign flips but is underpowered
- broad family search is reported as K=1
- sample count is compared to trial count
- repeated scanner rebuild inflates K

Expected policy:

- pooled results require per-direction disclosure
- OOS power tier controls language
- K counts are event-derived only
- sample-size gates and K-budget gates remain separate

### 13.5 Operational traps

- stale cache after revocation sidecar lands
- active heavyweight prereg exists but no result
- result exists but journal not enriched
- draft has sat in quarantine too long
- graveyard digest is stale
- action queue says park but promote queue says queued
- status file hand-edited

Expected policy:

- rebuild derived state before reporting
- show primary inconsistency as next action
- drift checks fail hand-edits
- never resolve inconsistency by mutating capital-class state

---

## 14. Future-Proofing

### 14.1 Plugin and model independence

V2 should not depend on any single LLM, plugin, or app connector. LLMs may propose ideas or draft text, but the repo decides through:

- prereg lint
- canonical data
- execution events
- research result artifacts
- drift checks
- operator review

LONA, external backtest tools, or generic strategy generators may feed `IDEA_CAPTURED` only. They must not feed validation, K lineage, or capital-review recommendations as proof.

### 14.2 New data support

The event model must allow richer data later without changing the research boundary:

- depth / order book data
- slippage probes
- broker execution records
- additional futures instruments
- self-funded versus prop-account profiles

New data can add fields to `canonical_data_fingerprint`, but cannot bypass prereg, holdout, or role mapping.

### 14.3 New research roles

The role taxonomy may grow, but each new role must define:

- parent population
- comparator
- primary metric
- promotion target
- failure language
- deployment boundary

Until then, unknown roles fail closed.

### 14.4 Multi-agent safety

The design must remain safe when multiple agents help:

- disjoint write scopes
- explicit stage files for mutating work
- event-log lock for append operations
- generated artifacts tagged with source and git SHA
- no hidden state in agent memory as source of truth

### 14.5 Degradation and recovery

If V2 state becomes inconsistent:

- status command should still run in degraded mode
- it should report which artifact is unreadable
- it should recommend the smallest repair action
- it should not attempt automatic repair unless explicitly invoked
- automatic repair must be report-first and diff-visible

---

## 15. Success Criteria

Fast Lane V2 is successful when:

1. The operator can run one command and know the next honest action.
2. Rebuilding status never changes trial history.
3. K-lineage is stable under repeated scans.
4. Every candidate has visible role, parent/comparator, K, OOS power, and blocker state.
5. Drafts cannot silently become active preregs.
6. Heavyweight results cannot silently become deployment state.
7. The fastest path is also the most auditable path.

---

## 16. Non-Goals

Fast Lane V2 does not:

- pick live lanes automatically
- write broker configuration
- weaken Chordia thresholds
- use LONA or external sandbox results as evidence
- treat LLM-generated ideas as preregistered hypotheses without checks
- collapse conditional findings into standalone strategy claims
- erase polluted historical ledger rows without an explicit correction record

---

## 17. Final Recommendation

Proceed with Fast Lane V2, starting with Phase 0 and Phase 1.

Do not build more idea-generation automation until the provenance boundary is fixed. A faster idea maker on top of polluted K accounting will only create faster false confidence.

The highest-EV first implementation is:

1. make all dry-runs truly non-mutating;
2. move trial append authority from scanners to real execution runners;
3. add a V2 trial index derived from stable event identity;
4. correct the current polluted ledger with an exclusion/correction artifact rather than deleting history.
