# Fast-Lane Prereg Pipeline — Connective-Tissue Audit and Design

**Date:** 2026-05-19
**Mode:** /design explore (no code; design only, awaiting approval)
**Author:** Claude (Opus 4.7)
**Risk tier:** critical (capital-adjacent — terminal node feeds lane_allocation.json)

---

## Scope

Audit the end-to-end fast-lane prereg pipeline and design the connective tissue
so it runs start-to-finish with awareness at each stage. The chain in question:

  idea → prereg YAML → fast-lane scan → result MD → PROMOTE queue → cherry-pick
  rank → heavyweight Chordia draft → (heavyweight replay) → lane_allocation.json

Find every break, manual handoff, stale-cache risk, and silent-failure path.
Do not write code in this pass. Produce a structured design.

---

## Turn 1 — ORIENT

### 1.1 Files actually involved (verified by Glob/Grep/Read this session)

**Stage 1 — Idea → prereg YAML**
- `scripts/research/llm_hypothesis_proposer.py` — Track A LLM proposer entry point.
- `scripts/research/lhp/literature_index.py` — local corpus search.
- `scripts/research/lhp/adjacency.py` — Mode A + validated_setups screen.
- `scripts/research/lhp/graveyard.py` — NO-GO / kill-verdict registry filter.
- `scripts/research/lhp/neighbor_scan.py` — adjacency family check.
- `scripts/research/lhp/static_checks.py` — fatal checks (banned columns, citations).
- `scripts/research/lhp/yaml_emitter.py` — writes `.draft.yaml` or
  `.draft.yaml.rejected`.
- `scripts/research/lhp/llm_client.py` — LLM call abstraction.
- Front door: `scripts/infra/prereg-loop.sh` → `scripts/tools/prereg_front_door.py`.

**Stage 2 — prereg YAML → fast-lane scan result**
- Active prereg lives in `docs/audit/hypotheses/<stem>.yaml`.
- Template: `docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml`.
- Runner: `research/chordia_strict_unlock_v1.py` (note: this is the same
  bounded runner used for both fast-lane v5.1 and heavyweight strict-unlock
  — the runner branches on `metadata.template_version`).
- Output: `docs/audit/results/<stem>.md` and `<stem>.csv`.
- Hypothesis loader: `trading_app/hypothesis_loader.py`.

**Stage 3 — result MD → PROMOTE queue (derived state)**
- Scanner: `scripts/research/fast_lane_promote_queue.py`.
- Cache: `docs/runtime/promote_queue.yaml`.
- Inputs read at scan time: result MDs, revocation sidecars, heavyweight
  preregs (active dir only — `drafts/` excluded), `action-queue.yaml` park
  entries, the canonical DB for OOS window via
  `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`.
- Drift check #157 reconstructs and diffs the cache.

**Stage 4 — PROMOTE queue → cherry-pick rank**
- Ranker: `scripts/research/cherry_pick_ranker.py`.
- Output: `docs/runtime/cherry_pick_ranking_<date>.csv` and append-only
  `docs/runtime/cherry_pick_journal.yaml`.
- Score components: deflation_headroom, n_adequacy, oos_power_readiness,
  dir_match, non_artifact, era_stability_proxy.
- Threshold mirror: `HEAVYWEIGHT_T_THRESHOLD` mirrors Criterion 4 strict
  no-theory hurdle (parity enforced by drift check #160).

**Stage 5 — Ranked candidate → heavyweight Chordia DRAFT**
- Bridge: `scripts/research/fast_lane_to_heavyweight_bridge.py`.
- Output: `docs/audit/hypotheses/drafts/<date>-<slug>-chordia-heavyweight-v1.draft.yaml`.
- Methodology parity drift check #161.
- Deliberately omits `theory_citation` (field-presence trap defense).

**Stage 6 — DRAFT → grounded DRAFT (optional theory upgrade)**
- Grounder: `scripts/research/cherry_pick_grounder.py`.
- Output: `<slug>.grounded.yaml` or `<slug>.grounded.rejected.txt`.
- Lowers the strict t hurdle from 3.79 to 3.00 if a real literature citation
  passes content verification.

**Stage 7 — Active prereg → heavyweight replay → result MD**
- Same runner as Stage 2 (`research/chordia_strict_unlock_v1.py`), but
  picked up via the heavyweight branch (no `template_version: fast_lane_v5.1`).
- Output: heavyweight `docs/audit/results/*-chordia-unlock-v1.md` or
  `*-chordia-heavyweight-v1.md`.

**Stage 8 — Heavyweight verdict → journal closure**
- Enricher: `scripts/research/cherry_pick_journal_enricher.py`.
- Backfills `heavyweight_verdict`, `t_observed_post_clustered_se`,
  `lesson_label` on the matching journal entry.

**Stage 9 — Verdict → live deployment**
- Operator hand-edit of `docs/runtime/chordia_audit_log.yaml` (capital-class).
- Operator hand-edit of `docs/runtime/lane_allocation.json` via
  `trading_app/lane_allocator.py` ceremony.
- Allocator gate functions in `trading_app/lane_allocator.py` and
  `trading_app/deployability.py` consult `validated_setups` /
  `chordia_audit_log` columns at runtime.

### 1.2 Authority documents read or referenced

- `docs/STRATEGY_BLUEPRINT.md` (Section 3 test sequence; Section 5 NO-GO).
- `docs/institutional/pre_registered_criteria.md` (12 criteria).
- `.claude/rules/backtesting-methodology.md` (RULE 3.3 OOS power floor; RULE 4
  K framing; RULE 6.3 E2 look-ahead banned columns; RULE 10 pre-registration).
- `.claude/rules/research-truth-protocol.md` (Layer classification;
  Mode A baselines; canonical filter delegation).
- `.claude/rules/institutional-rigor.md` (canonical-source authority; no dead
  code; verify before claiming).
- `.claude/rules/integrity-guardian.md` (fail-closed; never trust metadata).
- `.claude/rules/pooled-finding-rule.md` (per-cell breakdown when pooled).
- `.claude/rules/stage-gate-protocol.md` (scope_lock + blast_radius format).
- Memory files cited inline by the stage scripts:
  `feedback_chordia_theory_citation_field_presence_trap.md`,
  `feedback_lhp_validator_vs_field_presence_trap_n1.md`,
  `feedback_canonical_inline_copy_parity_bug_class.md`,
  `feedback_meta_tooling_n1_tunnel_2026_05_01.md`,
  `feedback_n3_same_class_doctrine_threshold.md`,
  `feedback_chordia_oos_park_vs_unverified_power_floor.md`.

### 1.3 Purpose (why this matters)

Today every stage is well-instrumented and fail-closed in isolation, but the
chain runs by operator typing. The cost shows up three ways:

1. **Latent stalls.** A PROMOTE landed yesterday; nobody ran the ranker, so the
   draft was never proposed; the heavyweight slot stays unfilled. The journal
   has zero entry for that result.
2. **Stale-cache risk.** `promote_queue.yaml` is derived but operator-triggered
   to rebuild. A landed revocation sidecar can sit hours before the cache
   reflects it; downstream consumers (ranker, journal) read stale state.
3. **Silent gaps.** A bridge draft sits in `drafts/`; nobody grounds it, the
   draft never makes it to active; lane_allocation.json never sees the
   candidate. Today there is no view that says "draft is N days old, still in
   drafts/, no grounded sibling on disk."

The connective tissue is not new code that mutates capital state. It is a
pipeline-awareness surface: one orchestrator command that walks every stage
and reports what needs to happen next, plus a lightweight age-and-staleness
view that surfaces the latent stalls.

---

## Turn 2 — DESIGN (multi-take deliberation)

### 2.1 Take 1 — What went wrong before in this domain

From hard-lessons memory files and the audit registry, the recurring failure
modes when wiring derived-state pipelines in this project are:

- **Meta-tooling on n=1.** Building an enforcement hook before three
  recurrences exist (closed: PR #200 Tier 3 read-budget;
  closed: Stage 4 stale-PR-merge guard). The right tier for a connector that
  spans 7 stages with one observed handoff failure is documentation and a
  read-only status view — not a writer or a hook.
- **Inline-copy parity bugs.** Five confirmed instances of canonical-source-
  to-inline-copy drift. Any new connector that hardcodes thresholds, status
  enums, or path patterns becomes the 6th. Defense: every numeric/string
  inline must register in `pipeline.canonical_inline_copies` AND have a
  drift check.
- **Bootstrap-runtime-control in-band state mutation.** When one process
  writes registry/audit-log state in the same run that evaluates a gate, the
  result MD and the JSON state can contradict each other (PR #257 incident).
  Defense: every stage of this pipeline must remain a read-only consumer
  except for its own single derived-state file. No connector touches
  `chordia_audit_log.yaml`, `lane_allocation.json`, or `validated_setups`.
- **Field-presence traps.** An empty string in `theory_citation` silently
  unlocks a softer threshold. Defense: any new connector that synthesizes
  fields MUST omit empty fields rather than emit them as empty strings.
- **Stale-cache vs hand-edit divergence.** `promote_queue.yaml` is rebuilt
  on every run — a hand-edit shows up only as a diff line on the next scan.
  Defense: the orchestrator must always rebuild derived caches before
  reading them, never read first.
- **Pre-reg writer drift.** Pathway B confirmatory tests must NOT inherit
  upstream K as their own; upstream K belongs under `upstream_discovery_provenance`
  with `role: PROVENANCE_ONLY`. The bridge already does this correctly;
  the orchestrator must not undo it.

### 2.2 Take 2 — Bottom-up design from failure prevention

Working backward from the failure modes:

- The orchestrator is a **read-mostly reporter**. It walks every stage,
  reports state, and emits one structured action list. It does NOT advance
  state automatically.
- Two new derived-state files (one age-staleness view; one
  status-by-strategy roll-up); both are rebuilt on every run; neither is
  hand-editable; both have drift-check parity with their sources.
- One new shell front-door command that runs the read steps in order:
  rebuild derived caches → emit the roll-up → print the action queue.
- No new writes to capital-class files.
- No automation of operator decisions; the orchestrator surfaces
  "draft X is N days old without a grounded sibling" — it does NOT
  promote, ground, or run anything.
- A single canonical state-graph document (this design's appendix) becomes
  the source of truth for the chain, replacing the implicit graph in stage
  files and in operator memory.

### 2.3 Take 3 — Challenge: too complex? too simple? right ordering?

**Risk of over-engineering.** A full DAG executor is wrong here. The
existing model — each script is fail-closed, idempotent, read-only over its
upstream — is exactly the institutional pattern this project rewards. The
gap is not "we need automation"; the gap is "we need a single command that
walks the chain and tells the operator what to do next."

**Risk of under-engineering.** A bash one-liner that chains the scripts is
also wrong. It hides the staleness signal (which is the actual problem):
"how old is the oldest unattended PROMOTE result?" needs to be a first-class
output, not implicit in the diff lines.

**Right scope.** One orchestrator command + one age-staleness view + one
roll-up file. No automation past the read step.

**Ordering.** The orchestrator runs the stages in upstream-before-downstream
order, but executes each as a read-only refresh, not an action. Read order:

1. Rebuild `promote_queue.yaml` from result MDs / revocation sidecars /
   heavyweight preregs / action-queue / canonical DB.
2. Rebuild the ranker CSV from `promote_queue.yaml` and the result MDs (for
   OOS row parsing).
3. Walk `cherry_pick_journal.yaml` and `docs/audit/hypotheses/drafts/`,
   build the staleness view (next-action per strategy_id).
4. Walk active hypotheses in `docs/audit/hypotheses/` (excluding `drafts/`),
   detect orphaned active preregs with no matching result MD (= scheduled,
   not yet run).
5. Walk heavyweight result MDs and run the enricher's read-only path to
   detect journal entries whose verdict is still null but a result MD
   exists.
6. Emit the structured action queue: one row per strategy_id with current
   stage, age, and next-action.

### 2.4 Pressure-test against past failures

| Past failure class                                            | Connector's defense                                                                  |
|----------------------------------------------------------------|---------------------------------------------------------------------------------------|
| Inline-copy parity bug (5 confirmed)                          | All thresholds delegated to existing canonical sources; no new inlines.              |
| Bootstrap-runtime-control in-band mutation (PR #257)          | Orchestrator is read-only over capital-class files; no writes outside its own caches. |
| Field-presence trap (theory_citation; Chordia threshold flip) | Orchestrator never synthesizes optional fields; reads only.                          |
| Stale-cache vs hand-edit (n=1)                                | Always rebuild derived caches before reading; diff lines shown but not acted on.     |
| Meta-tooling on n=1                                           | No automatic edits; surface-only; no hooks added until n=3 in same class.            |
| OOS power floor bypass                                        | Re-uses scanner's existing OOS gate; orchestrator does not re-implement.             |
| Pooled-finding without per-cell breakdown                     | Scanner already enforces; orchestrator surfaces the pooling_artifact flag verbatim. |
| Chordia PARK ≠ UNVERIFIED (power floor missing)               | Enricher already maps PARK to UNVERIFIED when oos_power_tier signals; orchestrator surfaces enricher output verbatim. |

### 2.5 Proposed connectors

**Connector 1 — Status roll-up file.**
A new derived-state file at `docs/runtime/fast_lane_status.yaml`. Schema:
one entry per strategy_id observed anywhere in the chain, with current
stage, age in days since last advancement, next-action token, and pointers
to the upstream artifact and the downstream artifact (when known).
Rebuilt on every orchestrator run. Drift check parity with sources
(promote_queue.yaml, cherry_pick_journal.yaml, hypotheses/, hypotheses/drafts/,
results/). No hand-edit allowed (drift check fails if hand-edited).

**Connector 2 — Age-staleness view.**
A new derived-state file at `docs/runtime/fast_lane_age_staleness.yaml`.
For each strategy_id, the age in days at each stage transition (e.g., "queued
12 days, ranked 11 days, no bridge draft"). Operator-visible signal:
"strategy X has been QUEUED for 12 days; bridge draft authoring is the
next operator action."

**Connector 3 — Orchestrator command.**
A new front-door shell wrapper at `scripts/infra/fast-lane-walk.sh` →
`scripts/tools/fast_lane_walk.py`. Read-only end-to-end walk:
rebuild caches → produce status roll-up → produce staleness view → emit
the action queue to stdout (and optionally to a CSV). Returns non-zero if
any stage cache is inconsistent (matches the existing scanner's fail-closed
ERROR pattern).

**Connector 4 — Awareness output contract.**
Stdout output is a fixed-format Markdown report so the operator can read
the action queue at a glance. Top of the report carries:
- counts per stage (active prereg N; PROMOTE N; QUEUED N; ranked N;
  bridged-but-not-grounded N; grounded N; heavyweight-pending N; closed N)
- top-3 stalled strategies by age
- any ERROR status from any stage cache
- a "next operator action" footer that names exactly one strategy_id and
  one stage to act on (= highest-rank QUEUED with no bridge draft, OR
  oldest ungrounded draft, OR oldest verdict-pending journal entry)

**Connector 5 — State-graph reference doc.**
A new doc at `docs/specs/fast_lane_state_graph.md` (this design's appendix
formalized) — a single source of truth for the chain. Every stage script
references it in its module docstring (no inlined graph descriptions); any
schema change to a derived-state file requires an amendment to this doc.

### 2.6 Three approaches, with trade-offs

| Approach                                                                                       | Pros                                                                                                                                                                  | Cons                                                                                                                                                              |
|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **A. Status roll-up + orchestrator walk (recommended).**                                       | Adds one read-only command and two derived-state files. Matches the project's institutional pattern (everything else is derived state + drift check). Easy to roll back. | Operator still must trigger; no automatic state advancement.                                                                                                      |
| **B. Approach A + auto-advance for low-risk stages (bridge from PROMOTE; enricher backfill).** | Removes two manual transitions.                                                                                                                                       | Adds writers; needs careful idempotency tests; bridge "auto-advance" creates draft files even when the operator may prefer to PARK. Risk of meta-tooling on n=1. |
| **C. Full DAG executor (Airflow / make / luigi).**                                              | Future-proof, declarative.                                                                                                                                            | Heavyweight tooling for a 7-stage chain; introduces a new dependency; reduces transparency.                                                                       |

**Recommendation:** **Approach A.** Get the awareness surface right first;
ship for two weeks; only then consider auto-advancing the two safest
transitions (B). Never C.

---

## Turn 3 — DETAIL (ordered stage plan, no code)

Five upstream-before-downstream stages, each independently shippable.

### Stage 1 — Author the state-graph reference doc

Files to create:
- `docs/specs/fast_lane_state_graph.md`

What it contains:
- Node list: every derived-state file and its single writer.
- Edge list: every read between scripts.
- Schema contracts for the four current derived files
  (`promote_queue.yaml`, `cherry_pick_journal.yaml`, ranking CSV pattern,
  drafts directory).
- The two new schemas (status roll-up, age-staleness view) — proposed
  here; sealed when Stage 3 ships.
- Authority block: this doc is the single source of truth for the chain;
  stage scripts reference it from their module docstrings instead of
  re-inlining the graph.

Acceptance criteria:
- Doc exists; the eight stage scripts each have a one-line pointer added
  to their module docstring referencing the doc.
- A new drift check verifies that the in-doc node list matches the on-disk
  set of derived-state files (no orphan nodes; no orphan files).

Reads:
- Every stage script docstring (read-only).
- Every derived-state file glob in `docs/runtime/` (read-only).

Writes:
- One new spec file.
- Module-docstring one-liners in seven stage scripts (no logic change).

Blast radius:
- Low — documentation + one-line docstring touches. The drift check is the
  only mechanical guardrail; no behavioral change.

### Stage 2 — Build the status roll-up file

Files to create:
- `scripts/tools/fast_lane_status.py` (writer of the roll-up; read-only over
  the chain inputs).
- `docs/runtime/fast_lane_status.yaml` (initial on-disk derived state).
- `tests/test_tools/test_fast_lane_status.py` (unit tests covering the
  walk logic, idempotency, and the "hand-edit detection" drift check).
- `pipeline/check_drift.py` adds one check function for status-roll-up
  reconstruction parity (mirrors the existing #157 pattern).

What it does:
- Reads every active prereg in `docs/audit/hypotheses/` (excluding `drafts/`).
- Reads every result MD pair in `docs/audit/results/`.
- Reads the existing `promote_queue.yaml`, `cherry_pick_journal.yaml`,
  and the latest `cherry_pick_ranking_<date>.csv`.
- Reads every file under `docs/audit/hypotheses/drafts/`.
- Builds one entry per observed strategy_id with: current_stage, age_days,
  next_action_token, upstream_artifact_path, downstream_artifact_path
  (when known).
- Writes the roll-up file with a derived-state warning banner identical
  to `promote_queue.yaml`'s "do not hand-edit" notice.

Acceptance criteria:
- Drift check rises by one; all checks pass.
- The tool runs end-to-end on the current state and produces a non-empty
  roll-up.
- Injection test: a hand-edited entry in the roll-up triggers the new
  parity drift check.

Reads (read-only):
- All upstream derived-state files; the canonical DB only for OOS window
  if and only if the scanner is also rerun (see Stage 4).

Writes:
- One new derived-state file.

Blast radius:
- Low — new writer of a new file. No production-code touched.

### Stage 3 — Build the age-staleness view

Files to create:
- `scripts/tools/fast_lane_age_staleness.py`.
- `docs/runtime/fast_lane_age_staleness.yaml`.
- `tests/test_tools/test_fast_lane_age_staleness.py`.
- One new drift check.

What it does:
- For each strategy_id in the status roll-up, computes age_days at every
  prior stage transition using filesystem mtimes of the upstream artifacts
  (result MD mtime for "promoted", journal entry date for "ranked",
  drafts/ mtime for "bridged", grounded.yaml mtime for "grounded").
- Filesystem mtime is the canonical age signal — git-history is the
  fallback when mtime is unreliable (e.g., post-clone).
- Output format: one row per strategy_id with age_at_each_stage.

Acceptance criteria:
- All four new files behave per spec.
- Drift check enforces parity between view and sources.

Reads:
- The status roll-up + filesystem mtimes.

Writes:
- One new derived-state file.

Blast radius:
- Low — same shape as Stage 2.

### Stage 4 — Build the orchestrator command

Files to create:
- `scripts/tools/fast_lane_walk.py` — read-only orchestrator.
- `scripts/infra/fast-lane-walk.sh` — front-door wrapper.
- `tests/test_tools/test_fast_lane_walk.py`.

What it does, in order:
1. Run the existing `fast_lane_promote_queue.py --write` (rebuild
   `promote_queue.yaml`).
2. Run the existing `cherry_pick_ranker.py --write --write-journal`
   (rebuild ranking + append journal if applicable).
3. Run the existing `cherry_pick_journal_enricher.py` read-only path
   (backfill verdicts for resolved entries).
4. Run the new `fast_lane_status.py` (rebuild status roll-up).
5. Run the new `fast_lane_age_staleness.py` (rebuild staleness view).
6. Render the awareness Markdown report to stdout (and optionally to
   `docs/runtime/fast_lane_walk_<date>.md`).
7. Return non-zero if any upstream stage emitted ERROR.

Acceptance criteria:
- The shell wrapper resolves the python venv via the same logic as
  `prereg-loop.sh`.
- End-to-end run on the current repo produces a non-empty report with
  at least the existing one QUEUED entry and one REVOKED entry.
- Idempotent: running twice in a row produces the same report (modulo
  the date stamp).
- Returns exit code 2 if any cache rebuild emits ERROR (matches
  `fast_lane_promote_queue.py`'s convention).

Reads (read-only):
- Every upstream derived-state file.
- The canonical DB (read-only, for the OOS window rebuild).

Writes (read-only over everything except its own caches):
- Refreshes `promote_queue.yaml`, ranking CSV, journal entries (via
  existing scripts), status roll-up, staleness view, and optionally a
  walk report.

Blast radius:
- Medium — orchestrates writers that already exist + the two new ones.
- All writers remain idempotent; no new state mutation. The orchestrator
  itself writes only the walk report (optional) and is otherwise a
  composer.

### Stage 5 — Wire awareness into existing operator entry points

Files to touch:
- `HANDOFF.md` — add a one-line pointer to `fast-lane-walk.sh` under
  the "Quick commands" or session-start section.
- `docs/STRATEGY_BLUEPRINT.md` — add a one-line pointer to the state
  graph doc under the relevant routing section.
- `CLAUDE.md` § Quick Commands — add the orchestrator command alongside
  `check_drift.py` and `context_resolver.py`.

What it does:
- Surfaces the walk command at the three places an operator looks at
  session start.
- No code changes; no logic changes.

Acceptance criteria:
- The three docs reference the walk command.
- `check_drift.py` continues to pass.

Blast radius:
- Zero — documentation-only edits.

---

## Turn 4 — VALIDATE

### 4.1 Failure modes per connector

**Status roll-up reconstruction failure.** If one upstream file is malformed
(unparsed YAML), the status roll-up could either fail-closed (correct) or
silently drop the strategy_id (wrong). Defense: every load helper in
`fast_lane_status.py` must propagate parse failures as a status="ERROR"
entry, mirroring `fast_lane_promote_queue.py`'s pattern. Drift check
verifies that the count of ERROR entries in the roll-up matches the count
in `promote_queue.yaml` plus newly detected parse failures.

**Age-staleness mtime spoofing.** Filesystem mtimes can drift on cross-
worktree edits or after a clean clone. Defense: when mtime is older than
the file's first git-commit timestamp, fall back to the git-log timestamp.
Tested via fixtures with mocked mtimes.

**Orchestrator double-write.** If the orchestrator runs while another
session is mid-write on `promote_queue.yaml`, the cache could be
half-rebuilt. Defense: existing branch-flip + session-lock guards
already cover this; the orchestrator inherits the protection automatically.
No new lock file needed.

**Idempotency violation in journal append.** Ranker already has the
"(strategy_id, today)" idempotency guard. Orchestrator inherits it. Add
one regression test: orchestrator runs twice in a row, journal size is
unchanged on the second run.

**Capital-class file mutation.** Any change introduced here that writes
to `chordia_audit_log.yaml`, `lane_allocation.json`, `validated_setups`,
or anything under `trading_app/live/` is a critical regression. Defense:
new drift check that asserts the orchestrator's process tree never opens
any of those paths in write mode. Mock-based unit test for this guard.

**Inline-copy parity creep.** The new status roll-up could grow inlined
thresholds (e.g., "if age > 7 days, flag as stalled"). Defense: every
threshold goes through the canonical-inlines registry; drift check #159
meta-check ensures the new entry has live parity.

**Schema drift on derived-state files.** If status roll-up or staleness
view schemas change, downstream readers (none today, but operator-visible)
could silently misread. Defense: schema_version field is bumped on every
schema change; drift check enforces the schema_version match.

**Stale-cache vs hand-edit.** Already covered by the existing #157
pattern plus the two new checks.

**Pre-existing test failure surfacing.** When the orchestrator runs the
existing scanners, any latent ERROR (e.g., MGC trade-window orthogonal)
becomes visible in the report. This is desirable, not a regression.

### 4.2 Behavioral tests (not just "it runs")

- Idempotency: two consecutive walks produce the same report content
  (modulo date stamp).
- Staleness signal: a synthetic 14-day-old QUEUED result surfaces as the
  top-1 stalled strategy in the report.
- Error propagation: a malformed prereg in `docs/audit/hypotheses/`
  surfaces as an ERROR row in both `promote_queue.yaml` and the new
  status roll-up.
- Read-only over capital-class files: with `chordia_audit_log.yaml`,
  `lane_allocation.json`, and `validated_setups` set to read-only on the
  test fixture filesystem, the entire walk completes successfully.
- Hand-edit detection: a hand-edited entry in the new roll-up triggers
  the new drift check on the next walk.
- Backwards compatibility: every existing scanner and ranker test
  continues to pass.

### 4.3 Rollback plan

Each stage is independently revertible:

- Stage 1: delete the spec file and the docstring one-liners; the drift
  check that depends on it is removed in the same revert commit.
- Stage 2: delete the new writer, the new derived file, the test, and
  the drift check. The chain's runtime behavior is unaffected (status
  roll-up is purely additive).
- Stage 3: same shape as Stage 2.
- Stage 4: delete the orchestrator, shell wrapper, and test. The existing
  scripts continue to work standalone.
- Stage 5: revert the doc edits.

Total revert is one commit per stage if needed. Nothing in this design
mutates an existing derived-state schema, so no migration is required.

### 4.4 Guardian prompts needed?

- **ENTRY_MODEL_GUARDIAN:** not needed — no entry-model logic touched.
- **PIPELINE_DATA_GUARDIAN:** not needed — no canonical-layer logic
  touched.
- **Capital-class read-only guardian (new):** worth adding as a
  drift check at Stage 4 — assert the orchestrator never opens any
  capital-class file in write mode (greppable static check on the
  orchestrator's source plus a runtime mock-fs test).

### 4.5 Open questions for operator

- **Do we want the orchestrator to ground bridge drafts automatically?**
  Default: NO. The grounder is already cheap and operator-driven; auto-
  grounding risks creating noisy `.grounded.rejected.txt` files. Recommend
  surfacing "draft N days old, no grounded sibling" in the report and
  leaving the call to the operator. Revisit after two weeks of usage.
- **Do we want the orchestrator to run the heavyweight Chordia replay
  automatically once a draft is moved out of `drafts/`?**
  Default: NO. Heavyweight replay is capital-class authoring and
  deliberately operator-typed (per the bridge's checklist printout).
  Same recommendation: surface, do not act.
- **Markdown report or YAML report or both?**
  Recommendation: stdout Markdown for human read; optional CSV companion
  for tooling; no YAML report (would be a third derived file with no
  downstream reader today — YAGNI).
- **Should the orchestrator be wired into a hook?**
  Default: NO. Per the meta-tooling-on-n=1 rule, build the surface first,
  observe operator usage, only then consider hook integration.

---

## Summary — current-state map and gap list

### Current-state map (one line per stage)

| # | Stage                    | Writer                                 | Derived artifact                                       | Trigger        |
|---|--------------------------|----------------------------------------|--------------------------------------------------------|----------------|
| 1 | Idea -> prereg DRAFT     | `llm_hypothesis_proposer.py`           | `docs/audit/hypotheses/*.draft.yaml`                   | operator       |
| 2 | DRAFT -> active prereg   | operator hand-move                     | `docs/audit/hypotheses/*.yaml`                         | operator       |
| 3 | active prereg -> result  | `research/chordia_strict_unlock_v1.py` | `docs/audit/results/*.md` + CSVs                       | operator       |
| 4 | results -> PROMOTE queue | `fast_lane_promote_queue.py`           | `docs/runtime/promote_queue.yaml`                      | operator       |
| 5 | queue -> rank + journal  | `cherry_pick_ranker.py`                | `docs/runtime/cherry_pick_ranking_<date>.csv` + journal | operator       |
| 6 | rank -> heavyweight DRAFT| `fast_lane_to_heavyweight_bridge.py`   | `docs/audit/hypotheses/drafts/*.draft.yaml`            | operator       |
| 7 | DRAFT -> grounded DRAFT  | `cherry_pick_grounder.py`              | `.grounded.yaml` or `.grounded.rejected.txt`           | operator       |
| 8 | grounded -> active       | operator hand-move                     | `docs/audit/hypotheses/*.yaml`                         | operator       |
| 9 | heavyweight verdict      | `research/chordia_strict_unlock_v1.py` | new `docs/audit/results/*.md`                          | operator       |
|10 | verdict -> journal       | `cherry_pick_journal_enricher.py`      | `docs/runtime/cherry_pick_journal.yaml`                | operator       |
|11 | verdict -> live          | operator ceremony                      | `chordia_audit_log.yaml` + `lane_allocation.json`      | operator only  |

### Gap list

1. **No single command walks the chain.** Operator must remember the
   correct invocation order across 6 different scripts.
2. **No staleness signal.** No surface tells the operator "X has been
   QUEUED for 12 days; bridge it or PARK it." The signal must be derived
   per-session by hand-grepping.
3. **No status-per-strategy roll-up.** To answer "what stage is strategy
   Y in?" the operator opens 4 files.
4. **Cache freshness is operator-typed.** `promote_queue.yaml` rebuilds
   only when the operator runs the scanner. A landed revocation sidecar
   can sit hours before the cache reflects it.
5. **No formal state-graph doc.** The chain's structure is implicit in
   stage scripts and operator memory; a future refactor can drift one
   stage out of compatibility without anyone noticing until the next
   run fails.
6. **No "next operator action" surface.** The operator has to synthesize
   "what to do next" from across the chain rather than reading it.

### Proposed connections (mapped to gaps)

| Gap | Connector                                     | Stage |
|-----|-----------------------------------------------|-------|
| 1   | `fast-lane-walk.sh` orchestrator             | 4     |
| 2   | `fast_lane_age_staleness.yaml` view          | 3     |
| 3   | `fast_lane_status.yaml` roll-up              | 2     |
| 4   | Orchestrator always rebuilds caches first    | 4     |
| 5   | `docs/specs/fast_lane_state_graph.md`        | 1     |
| 6   | Awareness report footer                      | 4     |

### Blast radius (whole design)

- **New files:** 5 (state-graph spec; status roll-up writer; staleness
  view writer; orchestrator; shell wrapper).
- **New derived-state files:** 2 (status roll-up; staleness view) +
  optional walk report.
- **New tests:** 4 test files.
- **New drift checks:** 3 (status roll-up parity; staleness view parity;
  read-only-over-capital-class assertion).
- **Existing files touched:** module-docstring one-liner in 7 stage
  scripts (no logic change), 3 doc files for the front-door surface
  (HANDOFF.md, STRATEGY_BLUEPRINT.md, CLAUDE.md).
- **Capital-class files touched:** 0.
- **Production-code logic changed:** 0.

### Stage plan (one-line summary)

1. **Stage 1 — Spec.** Write `docs/specs/fast_lane_state_graph.md`; add
   docstring pointer in the 7 stage scripts; drift-check for node parity.
2. **Stage 2 — Status roll-up.** New writer + new derived file + tests +
   drift check.
3. **Stage 3 — Staleness view.** New writer + new derived file + tests +
   drift check.
4. **Stage 4 — Orchestrator.** New walk script + shell wrapper + tests
   + read-only-over-capital-class drift check.
5. **Stage 5 — Surface.** Three documentation pointers added at the
   operator entry points.

Each stage is shippable independently. Hard gate: do NOT proceed to Stage 2
until Stage 1 is in `docs/specs/` and the docstring pointers are committed.
Do NOT proceed to Stage 4 until Stages 2 and 3 ship.

---

## Awaiting approval

This is plan output. No code has been written. To proceed, reply with
"go" or "approved" and I will write Stage 1 to
`docs/runtime/stages/2026-05-19-fast-lane-state-graph-spec.md` and begin
implementation. To iterate, reply with the change you want and I will
revise this design in place.

Open questions worth a decision before approval (defaults proposed in
§ 4.5): auto-grounding behavior; auto-heavyweight-replay behavior; report
format; hook integration.
