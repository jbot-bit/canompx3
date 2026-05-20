# Stage 2A.3 — Handoff (worktree-local)

**Worktree:** `C:/Users/joshd/canompx3-fastlane-2a3-scanner-wiring`
**Branch:** `session/joshd-fastlane-2a3-scanner-wiring` (from origin/main @ `a0490bbd`)
**Stage file:** `docs/runtime/stages/2026-05-20-fast-lane-anti-fp-2a3-scanner-bridge-wiring.md` (already on branch via commit `5cc63cc7`)
**Mode:** IMPLEMENTATION. Approach Step 1 done (stage file open). Steps 2–9 remain.

---

## Why this worktree exists

The original session was operating in `C:/Users/joshd/canompx3` (main tree). That tree had ~5 uncommitted files from another active terminal implementing **Check #172 graveyard status tokens parity** (9th canonical-inline-copy instance). Those files overlap Stage 2A.3 scope_lock (`pipeline/check_drift.py`, `pipeline/canonical_inline_copies.py`). Per `parallel-session-isolation.md`, I spawned this isolated worktree before any edit.

**Do NOT touch the main tree from this session.** Other terminal is mid-flight there.

---

## Stage-file corrections (apply during implementation, before commit)

The stage file at `docs/runtime/stages/2026-05-20-fast-lane-anti-fp-2a3-scanner-bridge-wiring.md` was authored against a stale drift count + live queue snapshot. Two corrections needed:

1. **Acceptance Criterion #1:** drift count is `171 → 172`, not `170 → 171`.
   - Reason: Check #171 (`check_holdout_sentinel_inline_copy_parity`) already landed in commit `a0490bbd` (Stage 2A.2 follow-up, 8th canonical-inline-copy instance).
   - Verified: `python pipeline/check_drift.py` reports "150 checks passed [OK], 0 skipped, 20 advisory" + 1 pre-existing violation = 171 total checks.
   - The new check this stage adds is therefore `#172` — but the other terminal also plans to land `#172` (graveyard status tokens parity). **Whichever lands on main first wins #172; the other becomes #173 via rebase.** Plan to handle at merge time.

2. **Acceptance Criterion #6 (live self-review entry):** `MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30` is now `PARKED` (action-queue#20), not `QUEUED`.
   - Reason: scanner already classified it `PARKED` on its last run; entry sits in `docs/runtime/promote_queue.yaml` lines 31–54 with `status: PARKED`.
   - **Self-review still valid** against this entry: provenance fields must be emitted regardless of terminal status (PARKED entries still get `structural_hash` + `k_lineage` + `N_hat`; suppression statuses only override `QUEUED` → `SUPPRESSED_*`). The design test for AC#6 becomes: "scanner emits a PARKED entry with full provenance fields populated and reproducible structural_hash."
   - Today the cache has only 2 entries: 1 REVOKED + 1 PARKED. No live QUEUED. Acceptance proceeds against either status — provenance must populate either way.

3. **Update the stage file `## Approach` Step 1 status** to reflect "DONE (commit 5cc63cc7)" before final commit, so the next reader doesn't re-open it.

---

## Approach (from stage file, abbreviated — steps remaining)

2. **Register suppression-status enum in `pipeline/canonical_inline_copies.py`** — append one `InlineCopyPair`:
   - `name="fast_lane_promote_suppression_status_values"`
   - `inline_site="scripts/research/fast_lane_promote_queue.py"`
   - `canonical_source="docs/runtime/stages/2026-05-20-fast-lane-anti-fp-2a3-scanner-bridge-wiring.md § Suppression Status Enum"`
   - `gated_constants=("STATUS_VALUES",)`
   - `parity_check="check_fast_lane_promote_queue_provenance_present"` *(combines suppression-enum parity + provenance presence in one check per stage file AC#2)*
   - `test_file="tests/test_pipeline/test_check_drift_fast_lane_promote_queue_provenance_present.py"`
   - `bug_class_anchor="memory/feedback_canonical_inline_copy_parity_bug_class.md (10th confirmed instance, 2026-05-20 — Stage 2A.3)"` *(or 9th if other terminal hasn't landed yet — verify at commit time)*
   - **Note:** the stage file's § "Suppression Status Enum" table IS the canonical source for the 6 new tokens. Add to that file's heading verbatim so the parity check can parse it.

3. **Scanner extension** (`scripts/research/fast_lane_promote_queue.py`):
   - Add 5 fields to `PromoteEntry` dataclass: `structural_hash: str`, `k_lineage: dict[str, Any]`, `N_hat: int`, `upstream_k_role: str`, `upstream_k_value: int | None`.
   - Extend `STATUS_VALUES` with 6 new tokens (exact strings from stage file § Suppression Status Enum table).
   - Wire `fast_lane_structural_hash.compute_structural_hash(inputs)` per entry. Inputs come from the parsed `strategy_id` + the source YAML's `scope` block; use the existing `find_heavyweight_prereg` glob to locate the source YAML (note: source YAML lives at `docs/audit/hypotheses/<stem>.yaml`, NOT under drafts/).
   - Read `docs/runtime/fast_lane_graveyard_digest.yaml` once per scan; build `{structural_hash → entry}` map for O(1) suppression lookup.
   - Read `docs/runtime/fast_lane_trial_ledger.yaml` once per scan; compute `K_global / K_family / K_lane` per entry. `K_family = entries sharing (instrument, orb_label, orb_minutes)`. `K_lane = entries sharing full structural_hash`.
   - Compute `K_effective_minBTL = 2 * ln(max(K_global, 2)) / E[max_N]^2` per Bailey 2013 Thm 1; use `max(...,2)` to avoid `ln(1)=0` div-by-zero. `E[max_N]` proxy: `pooled_n` for the cell.
   - `correlation_haircut_N_hat = pooled_n * (rho_hat + (1 - rho_hat) * M_correlated)` where `rho_hat = 0.5` (locked prior, see stage file § Risks #1) and `M_correlated = K_family`. This IS the `N_hat` field.
   - `bh_fdr_passes` dict: compute BH at each K framing; `K_global` is informational only (RULE 4 of backtesting-methodology).
   - **Suppression priority order** (apply BEFORE classify(), so it sits between `pooling_artifact` check and `OOS_UNPOWERED` gate, but AFTER existing REVOKED/ESCALATED/PARKED short-circuits):
     - If `revocation_sidecar / heavyweight_prereg / park_entry` exists → existing classify() returns REVOKED/ESCALATED/PARKED. Stop.
     - `SUPPRESSED_BANNED_ENTRY_MODEL` if `entry_model in {E0, E3}`.
     - `SUPPRESSED_E2_LOOKAHEAD` if `entry_model == "E2"` AND filter matches canonical `trading_app.config.E2_EXCLUDED_FILTER_PREFIXES` (startswith) OR `E2_EXCLUDED_FILTER_SUBSTRINGS` (any sub in filter_type). **Import canonically; never re-encode.**
     - `SUPPRESSED_GRAVEYARD` if `structural_hash` matches a lane-class digest entry (`hash_kind == "lane"` in the digest). Class-level hash matches are informational only — record as `error_reason` cite, don't suppress.
     - `SUPPRESSED_DUPLICATE_ACTIVE` if scanning the hypotheses dir finds another `.yaml` (not in `drafts/`) with the same structural_hash and no result MD yet.
     - `SUPPRESSED_SIBLING_RETEST` if `K_lane >= 2`.
     - `SUPPRESSED_K_OVERRUN` if `N_hat >= K_declared_in_prereg * 2`.
     - Then existing OOS-power gate → QUEUED.
   - **Ledger append per entry:** for every PROMOTE result MD the scanner processes, append one `LedgerEntry` via `fast_lane_trial_ledger.append_trial_ledger_entry`. The ledger writer is idempotent on `run_id` — use `run_id = f"scanner-{prereg_sha}-{run_timestamp_utc}"`.
   - Cache emit: `_entry_to_dict` already converts NaN→None; just extend it to handle the new dict fields (`k_lineage`, `upstream_provenance`).

4. **Ranker guard** (`scripts/research/cherry_pick_ranker.py`):
   - In `rank_queue_entries`, immediately after `status != "QUEUED"` filter, add a second filter:
     ```python
     required = ("structural_hash", "k_lineage", "N_hat")
     if any(entry.get(k) is None for k in required):
         raise ValueError(f"Ranker refusal: QUEUED entry {strategy_id} lacks provenance fields {required}; rerun scanner before ranking")
     ```
   - Propagate `structural_hash`, `k_lineage`, `N_hat` into the journal entry built by `build_journal_entry` — extend the dict with three new keys near `bridge_draft_path`.
   - Extend `RankedCandidate` dataclass with `structural_hash: str`, `k_lineage: dict`, `N_hat: int`.

5. **Bridge pre-flight refusal** (`scripts/research/fast_lane_to_heavyweight_bridge.py`):
   - Add `class BridgeRefused(Exception)` at module top.
   - In `build_heavyweight_prereg`, BEFORE the `missing_required` check, add a pre-flight that reads the matching cache entry from `promote_queue.yaml` (lookup by `source.scope["strategy_id"]`):
     - Refuse if `K_lane >= 2` (sibling retest)
     - Refuse if structural_hash matches a `hash_kind=="lane"` entry in graveyard digest
     - Refuse if `entry_model in {E0, E3}` (banned)
     - Refuse if `entry_model == "E2"` AND filter triggers `trading_app.config.E2_EXCLUDED_FILTER_*` (import canonically — note `from trading_app.config import E2_EXCLUDED_FILTER_PREFIXES, E2_EXCLUDED_FILTER_SUBSTRINGS`)
   - Critical: **never re-encode E2 prefixes/substrings**. Acceptance criterion #8: `grep -nE "E2_EXCLUDED_FILTER_(PREFIXES|SUBSTRINGS)\s*=" scripts/research/` returns ZERO hits.
   - Each refusal raises `BridgeRefused` BEFORE `write_draft` runs. The `main()` CLI catches `BridgeRefused` and exits 3 with the refusal reason on stderr.

6. **Drift Check** (`pipeline/check_drift.py`):
   - Add `def check_fast_lane_promote_queue_provenance_present() -> list[str]`:
     - Banner scrub on the cache file (`DERIVED STATE - do not hand-edit`).
     - For every PROMOTE/QUEUED/SUPPRESSED_* entry: assert `structural_hash` is 16-hex, `k_lineage` is a dict with all required keys, `N_hat` is a positive int, `k_lineage.rho_hat_assumed == 0.5`.
     - For entries that pass: confirm `STATUS_VALUES` constant in `fast_lane_promote_queue.py` contains the 6 suppression tokens exactly as the stage file § Suppression Status Enum table (parity).
   - Append to `CHECKS` list with the description string referencing Stage 2A.3.
   - **Number choice:** Use the next available integer. At commit time, `python pipeline/check_drift.py 2>&1 | tail -2` shows the count. If it's still 171, this is #172. If the other terminal landed Check #172 first, this becomes #173.

7. **Test files** (3 new):

   a. `tests/test_pipeline/test_check_drift_fast_lane_promote_queue_provenance_present.py` — 4 injection tests:
      - Strip `structural_hash` from one entry → fails
      - NULL one `k_lineage` field → fails
      - Missing `N_hat` → fails
      - Mutate `rho_hat_assumed` to 0.6 → fails
      - PLUS the sibling-coverage requirement from Check #159: one test per gated constant (`STATUS_VALUES` parity — mutate one of the 6 suppression tokens in `fast_lane_promote_queue` and assert the check catches it).

   b. `tests/test_research/test_fast_lane_promote_queue_suppression.py` — 6 scanner integration tests:
      - One per row in stage file § Suppression Status Enum. Each builds a tmp_path fixture with synthetic result MD + synthetic ledger (preloaded entries to force K_lane>=2 or K_overrun) + synthetic graveyard digest, runs `scan(...)`, asserts emitted entry has the expected suppression status AND populated provenance fields.

   c. `tests/test_research/test_fast_lane_to_heavyweight_bridge_refusal.py` — 4 bridge refusal-path tests:
      - One per non-OOS refusal trigger. Each builds a synthetic cache entry + source YAML matching the refusal class, calls `build_heavyweight_prereg`, asserts `BridgeRefused`, asserts `DRAFTS_DIR / <slug>.draft.yaml` does NOT exist on disk after the call (file-state before/after).

8. **Verify + commit:**
   - `cd C:/Users/joshd/canompx3-fastlane-2a3-scanner-wiring`
   - `python pipeline/check_drift.py` — count goes 171 → 172 (or 173), all PASSED modulo pre-existing MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4 trade-window carry-over (orthogonal).
   - Run new test files individually: `python -m pytest tests/test_pipeline/test_check_drift_fast_lane_promote_queue_provenance_present.py tests/test_research/test_fast_lane_promote_queue_suppression.py tests/test_research/test_fast_lane_to_heavyweight_bridge_refusal.py -v`
   - Self-review on live cache entry: `python scripts/research/fast_lane_promote_queue.py --write` then re-read `docs/runtime/promote_queue.yaml`; confirm PARKED entry `MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30` now carries `structural_hash` + `k_lineage` + `N_hat` + `rho_hat_assumed: 0.5`. Verify K_lane=1, no graveyard match, no E2 lookahead (E1 entry), N_hat sane.
   - Capital-class grep: `grep -nE "(validated_setups|chordia_audit_log\.yaml|lane_allocation\.json|trading_app/live/)" scripts/research/fast_lane_promote_queue.py scripts/research/cherry_pick_ranker.py scripts/research/fast_lane_to_heavyweight_bridge.py` — only docstring/banner mentions allowed.
   - Canonical-delegation grep: `grep -nE "E2_EXCLUDED_FILTER_(PREFIXES|SUBSTRINGS)\s*=" scripts/research/` — must return ZERO hits (only imports).
   - Single commit citing this stage file + Bailey-López de Prado 2014 § 3 + `feedback_n3_same_class_doctrine_threshold.md`.
   - Push to origin: `git -C C:/Users/joshd/canompx3-fastlane-2a3-scanner-wiring push origin session/joshd-fastlane-2a3-scanner-wiring`
   - Open PR via `gh pr create` — capital-class? **No** (bridge writes only to drafts/; ledger writer's `CapitalClassWriteRefused` gates the boundary). Per `feedback_direct_push_vs_pr_flow_token_cost.md`, direct-push-to-main is the default — but this branch was spawned off origin/main into its own session branch, so a PR is the natural flow. Confirm with user before pushing.

---

## Context already in memory (don't re-read)

The previous session loaded these files into context and ran the relevant greps:

- `scripts/research/fast_lane_promote_queue.py` (full)
- `scripts/research/fast_lane_structural_hash.py` (full)
- `scripts/research/fast_lane_trial_ledger.py` (full)
- `scripts/research/fast_lane_graveyard_digest.py` (full)
- `scripts/research/cherry_pick_ranker.py` (full)
- `scripts/research/fast_lane_to_heavyweight_bridge.py` (full)
- `pipeline/canonical_inline_copies.py` (full)
- Stage file (full)
- `docs/runtime/promote_queue.yaml` lines 1–55 (only 2 entries: REVOKED + PARKED)
- `pipeline/check_drift.py`: locations of CHECKS list end (lines 13164–13170), check #168–172 descriptions, last InlineCopyPair name (`fast_lane_trial_ledger_holdout_sentinel`)
- `trading_app/config.py`: `E2_EXCLUDED_FILTER_PREFIXES` at line 4045, `E2_EXCLUDED_FILTER_SUBSTRINGS` at line 4049

If you `/clear` first (which the user requested), the next session re-reads these. Budget: each upstream module is 200–800 lines.

---

## Risks / blind spots inherited from stage file

1. **`rho_hat = 0.5` is a prior, not a measurement.** Locked; documented; parity-checked. Empirical fit deferred to Stage 2B.
2. **First scanner run after 2A.3 lands writes the universe-of-trials baseline** (currently 0 entries, will become 2 after first run — REVOKED + PARKED today).
3. **Bridge refusal is a forcing function** justified by n=3+ doctrine threshold.
4. **Stage-numbering ambiguity** ("Stage 2A.3" collides with prior `bbd1e479` connective-tissue orchestrator — out of scope here).
5. **Cross-terminal collision:** the other terminal is closing `check_graveyard_status_tokens_parity` as Check #172 against `06_RD_GRAVEYARD.md` + `canonical_inline_copies.py` (9th instance). **My InlineCopyPair will be the 10th instance** if their work lands first; their changes will appear as a merge conflict in `canonical_inline_copies.py` and `check_drift.py` if mine lands first. Mitigation: at PR time, run `git fetch origin && git rebase origin/main` BEFORE pushing.
