---
task: Stage 2A — Fast-Lane Anti-False-Positive Trial Provenance Layer (universe-of-trials ledger + structural hash + K-lineage + graveyard suppression). Reorders the connective-tissue plan: 2A ships BEFORE 2B (status roll-up) so awareness plumbing is not layered on top of an unvalidated false-positive surface.
mode: CLOSED
closed_commit: ef4f0f29
closed_date: 2026-05-20
closed_note: |
  Stage 2A design grounding shipped across 2A.1 (b643bfc0 + 5c6040c3), 2A.2 (cec5a8d5),
  and 2A.3 (9b13d0d1 + bdb04ca6). All four downstream stages closed, all drift checks green
  on baseline run.

  Superseded 2026-05-21 by the canonical parser-surface relocation:
  Check #167 now parses `docs/specs/fast_lane_state_graph.md` § 9 Hash Schema.
  This stage file is retained only as design/audit history. Its old `## Hash Schema`
  body is intentionally just a backlink so future close-out sweeps do not have to
  preserve parser surface under `docs/runtime/stages/`.
original_mode: DESIGN
scope_lock:
  - docs/runtime/stages/2026-05-20-fast-lane-anti-fp-trial-provenance.md
---

## Blast Radius

This stage file is the DESIGN GROUNDING. Mode = DESIGN; this file is the only artifact written by the authoring step. Next session converts to mode: IMPLEMENTATION and reads the implementation scope below into a new scope_lock.

- This file (new) — design grounding survives /clear; locks scope_lock + acceptance criteria for the implementation stage that follows.
- Zero production-code touched in the design-authoring step.
- Implementation step's blast radius (informational, not active scope_lock): touches pipeline/check_drift.py (+4 checks), pipeline/canonical_inline_copies.py (+1 entry), three derived-state YAML files under docs/runtime/, three scripts under scripts/research/ (provenance fields only, no behavior change in ranker or queue); zero capital-class touched.

---

## Predecessor and Context

- **Plan parent:** `docs/plans/2026-05-19-fast-lane-pipeline-connective-tissue-design.md` (Approach A approved).
- **Stage 1 (shipped):** commit `ded02d2a` — `docs/specs/fast_lane_state_graph.md` + Check #166 + 7 docstring pointers.
- **Reorder rationale:** the connective-tissue plan's original Stage 2 was "status roll-up" (`fast_lane_status.yaml`). The reorder injects 2A (this stage) ahead of 2B (the original Stage 2) because:
  1. Status roll-up is awareness plumbing — it surfaces what is in the chain, not whether what is in the chain is real.
  2. Fast-lane is a Pathway-A funnel: PROMOTE candidates flow into heavyweight Chordia DRAFTs via the bridge. Today the chain has no enforced trial accounting (no `K_global`, `K_family`, `K_lane`), no de-dup gate before bridge, no NO-GO graveyard suppression. Layering awareness on top of an unguarded funnel makes the false-positive factory more efficient, not safer.
  3. Bailey–López de Prado 2014 § 3 is unambiguous: "a backtest where the researcher has not controlled for the extent of the search involved in his or her finding is worthless." The universe-of-trials ledger is the institutional answer.

---

## Verdict

**MODIFY the connective-tissue plan: ship Stage 2A first; the original Stage 2 (status roll-up) becomes Stage 2B.**

The current Stage 2 (`fast_lane_status.yaml` + age-staleness view) is awareness plumbing — visibility, not truth. Without the substrate below, the rollup amplifies a false-positive surface rather than constraining it.

---

## Reference Sources (read at design time)

- `docs/specs/fast_lane_state_graph.md` — canonical node inventory (Stage 1 output).
- `scripts/research/llm_hypothesis_proposer.py` — Track A LLM proposer (upstream of funnel).
- `scripts/research/fast_lane_promote_queue.py` — scanner; emits PROMOTE/QUEUED/REVOKED/ERROR.
- `scripts/research/cherry_pick_ranker.py` — ranks PROMOTE survivors for heavyweight bridge.
- `scripts/research/fast_lane_to_heavyweight_bridge.py` — bridge to heavyweight Chordia DRAFT.
- `scripts/research/cherry_pick_grounder.py` — optional theory-citation upgrade path.
- `scripts/research/cherry_pick_journal_enricher.py` — read-only verdict backfill.
- `research/chordia_strict_unlock_v1.py` — bounded runner shared by fast-lane and heavyweight branches.
- `pipeline/check_drift.py` — guardrail surface (current count 166; this stage targets 170).
- `chatgpt_bundle/06_RD_GRAVEYARD.md` — NO-GO / PARK / KILL registry.
- `docs/STRATEGY_BLUEPRINT.md` § NO-GO — strategy-class graveyard list.
- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` — DSR Eq. 1 + Eq. 9; canonical citation for effective-N accounting.
- `docs/institutional/pre_registered_criteria.md` — Criterion 4 K framing; Criterion 6 OOS power floor.
- `.claude/rules/backtesting-methodology.md` — RULE 3.3 OOS power floor; RULE 4 K framing; RULE 6.3 E2 banned columns; RULE 10 pre-registration.

---

## Hard Constraints (verbatim from user)

- No discovery scan.
- No OOS tuning.
- No promotion logic changes yet.
- No reimplementation of canonical costs/sessions/holdout.
- No new ML.
- Must preserve Mode A holdout.
- Must treat fast-lane as candidate triage only, not validation.

---

## Files to TOUCH (implementation-step scope_lock — informational here)

When the next session converts this design to `mode: IMPLEMENTATION`, the `scope_lock` becomes:

1. `pipeline/canonical_inline_copies.py` — register 1 new InlineCopyPair for the hash-schema version.
2. `pipeline/check_drift.py` — add 4 check functions (#167–#170) plus CHECKS entries.
3. `scripts/research/fast_lane_promote_queue.py` — emit `structural_hash`, `K_global_observed`, `K_family_observed`, `K_lane_observed`, `N_hat` per entry. No promotion-logic change.
4. `scripts/research/cherry_pick_ranker.py` — consume + propagate provenance fields into the journal; refuse to rank entries lacking them (guard only).
5. `scripts/research/fast_lane_to_heavyweight_bridge.py` — refuse to bridge on suppression-rule match. One pre-flight check, no behavior change otherwise.
6. Tests: 7 new test files (see Acceptance Criteria § Tests).
7. Stage file: a new `docs/runtime/stages/<date>-fast-lane-anti-fp-implementation.md` for the IMPLEMENTATION step.

This is 5 production-code files + tests + stage file. The stage-gate rule cap is ≤5 production files in scope_lock; this fits.

---

## Files NOT to TOUCH

- `research/chordia_strict_unlock_v1.py` — bounded runner; behavior locked.
- `scripts/research/cherry_pick_grounder.py` — theory-citation surface; orthogonal.
- `scripts/research/cherry_pick_journal_enricher.py` — backfill-only; no new write fields.
- `scripts/research/llm_hypothesis_proposer.py` — upstream of the funnel; out of scope.
- Anything under `trading_app/live/`, `validated_setups` table, `lane_allocation.json`, `chordia_audit_log.yaml` — capital-class boundary inviolate.
- `pipeline/cost_model.py`, `pipeline/dst.py`, `trading_app/holdout_policy.py` — canonical sources; delegation only, never re-encode.

---

## New Derived-State Files (3)

All three carry the `do_not_hand_edit: true` banner; drift checks fail closed on hand-edit.

### 1. `docs/runtime/fast_lane_trial_ledger.yaml`

Append-only universe-of-trials ledger. One entry per pre-reg execution (active or fast-lane). Never deleted. Mirrors the audit-log immutability contract.

### 2. `docs/runtime/fast_lane_structural_hashes.yaml`

Derived index `{structural_hash: [list of prereg paths + run timestamps]}`. Rebuilt on every scanner run from the ledger.

### 3. `docs/runtime/fast_lane_graveyard_digest.yaml`

Derived hash-set of all NO-GO / PARKED / KILL entries from `chatgpt_bundle/06_RD_GRAVEYARD.md` + `docs/STRATEGY_BLUEPRINT.md` § NO-GO + the action-queue PARK entries. Rebuilt on every run.

---

## Hash Schema (`structural_hash`)

> **Canonical:** `docs/specs/fast_lane_state_graph.md` § 9 Hash Schema. Relocated 2026-05-21 per surface-taxonomy doctrine in `docs/governance/system_authority_map.md` (parser-surface blocks may not live under `docs/runtime/`). This stage file is now design rationale only — the canonical YAML block lives in the spec, parsed by `check_fast_lane_structural_hash_schema_parity` (Check #167).

---

## Trial Ledger Schema (`fast_lane_trial_ledger.yaml`)

Append-only. One entry per pre-reg execution:

```yaml
schema_version: 1
do_not_hand_edit: true
entries:
  - run_id: <UUID>
    run_timestamp_utc: <ISO8601>
    prereg_path: docs/audit/hypotheses/<stem>.yaml
    prereg_sha: <git sha of the prereg at run time>
    structural_hash: <16-hex>
    template_version: fast_lane_v5.1 | chordia_unlock_v1 | …
    testing_mode: family | individual
    pathway: A | B
    K_declared: int            # K committed in the prereg
    K_global_observed: int     # cumulative count across ledger at run time
    K_family_observed: int     # count within (instrument, session) at run time
    K_lane_observed: int       # count within full hash at run time (= sibling re-tests)
    upstream_provenance:
      role: PROVENANCE_ONLY | INPUT_K   # per backtesting-methodology RULE 4
      upstream_K: int                   # if upstream scan supplied this prereg
    holdout_policy: mode_A
    holdout_sacred_from: 2026-01-01
    outcome:
      verdict: PROMOTE | REVOKE | KILL | UNVERIFIED_OOS_POWER | …
      result_md_path: docs/audit/results/<stem>.md
      t_observed: <float|null>
      t_clustered: <float|null>
      oos_n: <int|null>
      oos_power_tier: CAN_REFUTE | DIRECTIONAL_ONLY | STATISTICALLY_USELESS | null
```

Ledger is the universe-of-trials per Bailey–López de Prado Eq. 1 (E[max ŜR_n] needs N = independent trial count). The scanner reads the ledger before reporting PROMOTE, computes effective-N, and refuses to emit PROMOTE if N_eff exceeds the prereg's declared K_budget × an inflation tolerance.

---

## K-Lineage Schema (per candidate)

Every PROMOTE / QUEUED / REVOKE entry in `promote_queue.yaml` gains:

```yaml
k_lineage:
  K_global: <int>     # total ledger entries at the time PROMOTE was emitted
  K_family: <int>     # ledger entries sharing (instrument, session, orb_minutes)
  K_lane: <int>       # ledger entries sharing full structural_hash (= sibling-retest count)
  K_declared_in_prereg: <int>
  K_effective_minBTL: <float>   # 2·ln(K_global) / E[max_N]² per Bailey 2013
  bh_fdr_passes:
    K_family: true | false
    K_lane: true | false
    K_global: true | false      # informational; not a hard gate per RULE 4
  correlation_haircut_N_hat: <int>  # via DSR Eq. 9: ρ̂ + (1-ρ̂)·M
  ρ̂_assumed: 0.5    # conservative; documented prior pending empirical fit
```

Bridge refuses to author a heavyweight Chordia DRAFT if `K_lane ≥ 2` — sibling-retest = repeated grasp at the same lane; treat as PARK candidate, not PROMOTE.

---

## Suppression Rules

1. **NO-GO graveyard match.** If `structural_hash` matches any entry in `fast_lane_graveyard_digest.yaml`, scanner emits status `SUPPRESSED_GRAVEYARD` with the citation. Bridge refuses. Operator reopen path requires explicit critique per `06_RD_GRAVEYARD.md` "Rule for re-opening" — implemented as a `graveyard_reopen_justification` field the bridge checks for.
2. **Structural-hash collision with active prereg.** If hash matches any prereg currently in `docs/audit/hypotheses/` (not `drafts/`) that hasn't already produced a result MD with the same hash, emit `SUPPRESSED_DUPLICATE_ACTIVE`.
3. **Sibling-retest (`K_lane ≥ 2`).** Emit `SUPPRESSED_SIBLING_RETEST` with pointer to prior ledger entries. Forces operator to acknowledge the lane was already tested.
4. **Banned entry model.** `entry_model ∈ {E0, E3}` → `SUPPRESSED_BANNED_ENTRY_MODEL` (E0 purged, E3 in `SKIP_ENTRY_MODELS`).
5. **Look-ahead filter on E2.** Cross-check `filter_type` against `E2_EXCLUDED_FILTER_PREFIXES` / `E2_EXCLUDED_FILTER_SUBSTRINGS` from `trading_app/config.py` (canonical delegation; never re-encode). Emit `SUPPRESSED_E2_LOOKAHEAD`.
6. **OOS power floor.** Existing Check #161 stays; no change. Suppression layer is upstream of OOS, so an N_eff-failing entry never even reaches the OOS gate.

---

## Effective-N Accounting for Correlated Variants

For the PROMOTE entry's BH-FDR claim, compute:

```python
# Per Bailey-Lopez de Prado 2014 Eq. 9:
N_hat = rho_hat + (1 - rho_hat) * M_correlated
# M_correlated = K_family (siblings within same instrument+session)
# rho_hat = 0.5 (conservative prior; lit cites ρ̂ for related family typically 0.3–0.7)
```

`N_hat` is the input to the DSR rejection threshold `ŜR_0` (Eq. 1). Scanner emits this number per candidate. If `N_hat ≥ K_declared_in_prereg × 2`, scanner emits `SUPPRESSED_K_OVERRUN` — the universe-of-trials inflated the effective N beyond what the prereg budgeted.

Conservative `ρ̂ = 0.5` is documented as a prior, not a measurement. Empirical fit deferred to Stage 2B (the rollup) once the ledger has enough entries (`N ≥ 50`) to estimate `ρ̂`. Until then `ρ̂` is treated as a worst-case-realistic placeholder, not a "fact."

---

## Acceptance Criteria

1. `python pipeline/check_drift.py` count rises by **exactly 4** (166 → 170); all pass modulo pre-existing MGC carry-over.
2. **New checks:**
   - **#167 `check_fast_lane_trial_ledger_append_only`** — fails if any entry's `run_timestamp_utc` decreases or any prior entry is mutated.
   - **#168 `check_fast_lane_structural_hash_schema_parity`** — fails if the hash schema in code diverges from the schema doc-spec (canonical-inline-copy parity).
   - **#169 `check_fast_lane_graveyard_digest_parity`** — fails if any graveyard MD entry is missing from the digest hashset, or vice versa.
   - **#170 `check_fast_lane_promote_queue_provenance_present`** — fails if any PROMOTE / QUEUED entry in `promote_queue.yaml` lacks `structural_hash`, `k_lineage`, or `N_hat`.
3. **Injection tests** (≥4 per check = 16 total minimum):
   - **#167:** mutate timestamp backwards; mutate prior-entry field; duplicate `run_id`; malformed YAML — all fail-closed.
   - **#168:** drop a hash-input field in code; change formula constant; alter hash-output length; scrub `hash_schema_version` — all fail-closed.
   - **#169:** add ghost graveyard entry to digest; remove real graveyard entry; hash-collide a graveyard with active; malformed digest — all fail-closed.
   - **#170:** strip provenance from one entry; NULL one `k_lineage` field; missing `N_hat`; hand-edit cache to remove suppression — all fail-closed.
4. Bridge refuses to author a DRAFT for any QUEUED entry with `K_lane ≥ 2`, `graveyard_match`, `e2_lookahead_match`, or `banned_entry_model` — verified by integration test with synthetic fixtures.
5. Ledger contains zero entries from `validated_setups` or `chordia_audit_log.yaml` — fail-closed if a write attempts that table (capital-class boundary check).
6. **Holdout sentinel:** every ledger entry carries `holdout_policy: mode_A` and `holdout_sacred_from: 2026-01-01`; check fails if either is missing or mutated.
7. **Self-review pass** executes the chain end-to-end on the live repo (with current QUEUED entry `MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30`) and reports its `structural_hash`, `K_lineage`, `N_hat`, suppression status. **Expected behaviour:** entry stays QUEUED with `K_lane = 1` (no sibling retest yet), `N_hat` sane, no graveyard match. If any of those diverge, the design has a bug — not the entry.

---

## Exact Tests to Add

| Test file | Purpose | Min count |
|---|---|---|
| `tests/test_pipeline/test_check_drift_fast_lane_trial_ledger_append_only.py` | Check #167 injection probes | 4 |
| `tests/test_pipeline/test_check_drift_fast_lane_structural_hash_schema_parity.py` | Check #168 injection probes | 4 |
| `tests/test_pipeline/test_check_drift_fast_lane_graveyard_digest_parity.py` | Check #169 injection probes | 4 |
| `tests/test_pipeline/test_check_drift_fast_lane_promote_queue_provenance_present.py` | Check #170 injection probes | 4 |
| `tests/test_research/test_fast_lane_structural_hash.py` | Canonical normalisation: 6 unit + 2 property tests | 8 |
| `tests/test_research/test_fast_lane_promote_queue_suppression.py` | 1 integration per suppression rule (graveyard, dup-active, sibling-retest, banned-entry, E2-lookahead, K-overrun) | 6 |
| `tests/test_research/test_fast_lane_to_heavyweight_bridge_refusal.py` | Refusal paths: K_lane, graveyard, dup-active, E2-lookahead | 4 |

**Total new tests: ≥34.** Mirrors the existing parity-check pattern: function-level injection (fail-closed mutation probes) + integration-level end-to-end on synthetic fixtures. No live DB needed.

---

## Risks / Blind Spots

1. **`ρ̂ = 0.5` is a prior, not a measurement.** The conservative correlation assumption could understate or overstate N_hat for any specific instrument-session family. Mitigation: documented as a prior in both ledger schema and result MD; empirical fit deferred to Stage 2B once `N ≥ 50` ledger entries accrue. Risk = under-suppression (missed false-positives) if real ρ̂ < 0.5; risk = over-suppression (missed real edges) if real ρ̂ > 0.5. Bias is conservative on capital (false-positive-side risk is worse than missed-edge risk for this funnel).
2. **Ledger truthfulness depends on disciplined writes.** Every prereg execution must call the ledger writer. If a runner is bypassed (e.g., manual python invocation), the ledger silently undercounts. Mitigation: Check #170 fails if `promote_queue.yaml` carries a strategy without a ledger entry — closes the bypass class.
3. **Graveyard digest staleness.** If `06_RD_GRAVEYARD.md` is edited after the digest is built but before the next scan, suppression lags. Mitigation: orchestrator (Stage 2B / Stage 4) rebuilds digest on every walk; in the meantime the scanner itself rebuilds on every invocation per Stage 2A's writer contract.
4. **Hash brittleness on canonical-source change.** If `ALL_FILTERS` adds a new normalisation rule (e.g., a synonym mapping), old hashes silently diverge from new hashes for the same lane. Mitigation: `hash_schema_version` bump on any normalisation change; Check #168 enforces; old-hash → new-hash migration documented in the schema doc.
5. **Mode A holdout boundary.** Ledger should NEVER ingest validated_setups or any production capital-class table. Acceptance criterion 5 covers this; bears restating because the temptation to "enrich the ledger with deployment status" is exactly the boundary the rule forbids.
6. **Meta-tooling-on-n=1 trap.** The bridge-refusal logic (acceptance criterion 4) is a forcing function. Justification: this is the n=3+ doctrine threshold case — multiple confirmed false-positive surfaces (E2 look-ahead 24 TAINTED; canonical-inline-copy 5 instances; Chordia threshold parity 4 instances) plus the structural design need to prevent the funnel from becoming a factory. Refusal is mechanical because the bug class is mechanical. See `feedback_n3_same_class_doctrine_threshold.md`.
7. **Author drift on conservative prior.** Future authors may relax `ρ̂` "because the new data shows it's lower" without re-fitting on the full ledger. Mitigation: `ρ̂_assumed` is a registered canonical-inline-copy with Check #168 parity; any change requires a stage and a deflation re-derivation. Same defense that protects the Chordia thresholds.

---

## Smallest Safe Next Command

Next session, after `/clear` and stage-gate classification of this design as `mode: IMPLEMENTATION`:

```bash
python pipeline/check_drift.py
```

Establishes the baseline check count (expected 166) before any new file lands. The implementation step's first edit is then adding the new `InlineCopyPair` to `pipeline/canonical_inline_copies.py` for the hash-schema version — smallest pre-flight write before the four new check functions land.

---

## Stage-Gate Transition Plan

| Step | Mode | Authoring action |
|---|---|---|
| 0 | DESIGN (this file) | Design grounding written; no production code touched. |
| 1 | IMPLEMENTATION | New stage file `docs/runtime/stages/<date>-fast-lane-anti-fp-implementation.md` opens with the implementation `scope_lock` from § "Files to TOUCH". |
| 2 | IMPLEMENTATION | Land 4 drift checks + ledger writer + suppression logic + tests. |
| 3 | VERIFICATION | `check_drift.py` count 166 → 170; 34+ new tests pass; bridge-refusal integration test passes on synthetic fixtures. |
| 4 | VERIFICATION | Self-review pass on the live QUEUED entry `MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30`; expected verdict unchanged. |
| 5 | COMMIT | Commit cites this stage file + `feedback_n3_same_class_doctrine_threshold.md` + Bailey-López de Prado 2014 § 3 / Eq. 9 + `06_RD_GRAVEYARD.md` re-opening rule. |
| 6 | STAGE 2B handoff | Original "status roll-up" stage re-opens as Stage 2B; ledger entries now feed empirical `ρ̂` estimation once `N ≥ 50`. |

---

## Authoring Provenance

- **Author:** Claude (Opus 4.7, explanatory mode).
- **Authoring session:** 2026-05-19 (continued from prior session's design conversation; user `/clear` + redirect kept Stage 1 unmodified and pivoted from generic Stage 2 to anti-FP Stage 2A).
- **Approval:** user typed "Stage 2A authoring (the design above) is the first task of the next session, written to docs/runtime/stages/2026-05-20-fast-lane-anti-fp-trial-provenance.md with this brief as its design grounding."
- **Stage file dated 2026-05-20** intentionally: the design grounding lands today; implementation lands in the next session (operator's working day = tomorrow Brisbane time).
