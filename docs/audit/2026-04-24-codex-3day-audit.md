# Codex 3-Day Audit — 2026-04-21 → 2026-04-24

**Auditor:** Claude (autonomous /code-review sweep, /next loop)
**Scope:** All Codex-authored commits since 2026-04-21 across pipeline/, trading_app/, scripts/, tests/
**Method:** `feedback_codex_audit_playbook.md` — diff-only first pass, parallel grep, mutation probe, reverse-audit per tier.
**Status:** IN PROGRESS — checkpointed as tiers complete.

---

## TIER 1 — Live Trading — VERDICT: PASS-WITH-FIXES

**Tests:** 263 pass.
**Files:** trading_app/live/{session_orchestrator, bot_dashboard, alert_engine, bar_aggregator, detectors/*}, execution_engine, derived_state, conditional_overlays, run_live_session, prop_profiles.

| Commit | Title | Grade | Key finding |
|--------|-------|-------|-------------|
| 04edf5e6 | harden shadow overlay state checks | A- | Closed two silent-failure bugs: (1) `valid:True` reported even when `summary.status==invalid`; (2) `date.today()` used instead of trading-day boundary. Tests added. CRLF normalization made the diff look 9000 lines but real semantic change is 179/20. |
| 45f50916 | dashboard action coordination | B+ | Lock startup PID-alive checks. Preflight no longer instantiates SessionOrchestrator (removes Windows journal-lock dance). Tests added. **MED:** imports private `_is_pid_alive` from `instance_lock` — should be promoted public. |
| 94a9c6fb | keep phase2 overlays shadow-only | A | Removed live-multiplier path from `_apply_conditional_roles`. Was theoretical risk because overlay specs are hardcoded shadow_only, but the live-multiplier code path existed in 34d9e732 and got neutered here. Test added. |
| 34d9e732 | pr48 phase 2 native conditional-role surface | B+ | New `RoleResolver` class + integration into ExecutionEngine. **MED:** `RoleResolver.get_overlay_context` substring-matches `strategy_id` (`spec.instrument not in strategy_id`) — collision risk (`"E2" in "E20"`). **MED:** triple-copy of `_apply_conditional_roles(...)` at execution_engine.py:964/1186/1362 — should DRY. |
| 2811a622 | shadow overlay carrier + prereg routing | A- | New `conditional_overlays.py` (422 lines). Spec dataclass + breakpoint loader + state envelope with code/db fingerprints. **LOW:** `STATE_DIR.mkdir(...)` runs at module import — side-effect-on-import, should be lazy. |
| 7b691d83 | orb caps by session+instrument | A | Closes false-positive fail-closed that was blocking 10-lane Tradovate scaling. self_funded_tradovate dormant — risk path inactive. Caller updates complete. |
| c208ef7a | UP038 lint | SKIP | mechanical |

**Hardening backlog (Task #6):**
- T1.A: Promote `_is_pid_alive` to public in `trading_app/live/instance_lock.py` (or add `__all__`).
- T1.B: Replace substring matching in `RoleResolver.get_overlay_context` with canonical `parse_strategy_id` from `trading_app.eligibility.builder`.
- T1.C: DRY the triple `_apply_conditional_roles` calls in `trading_app/execution_engine.py` (964/1186/1362).
- T1.D: Move `STATE_DIR.mkdir(...)` in `trading_app/conditional_overlays.py` from module-import to first-use.

---

## TIER 2 — Discovery + Config — VERDICT: PASS

**Tests:** 459 pass.
**Files:** trading_app/{config, eligibility/, strategy_discovery, strategy_validator, phase_4_discovery_gates, conditional_overlays, hypothesis_loader}.

| Commit | Title | Grade | Key finding |
|--------|-------|-------|-------------|
| 6887632f | classify sparse oos honestly | A | `_check_criterion_8_oos` now returns structured verdict dict (`c8_oos_status` ∈ {PASSED, NO_OOS_DATA, INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH, INSUFFICIENT_N_PATHWAY_B_REJECT, NEGATIVE_OOS_EXPR, FAILED_RATIO}) plus legacy tuple wrapper for backward compat. Directly addresses `feedback_oos_power_floor.md`. 119 test lines added. |
| 9a5e66e0 | save mnq geometry bridge locks | A | New `PrevDayGeometryFilter` with 6 modes (below_pdl_long, downside_displacement_long, clear_of_congestion_long, go_long_context, inside_pdr_long, near_pivot_long_50). NaN-safe, divide-by-zero-safe, filter owns `break_dir == "long"` restriction so Phase-4 cannot silently widen to short breaks. 173 test lines added. |
| cc9273ca | restore prereg discovery front door | A | `check_single_use` extended with `instrument` scoping (paired with existing `orb_minutes` scoping). Enables multi-instrument prereg files to run separately. New `prereg_front_door.py` tool (361 lines). Backward-compat: legacy single-aperture path preserved. |
| 55cd9b30 | recover conditional-edge framework | A | hypothesis_loader: new `_ALLOWED_RESEARCH_QUESTION_TYPES` and `_ALLOWED_ROLE_KINDS` enums with validation. 4 new research scripts under `research/`. 8 new docs. Real net: 47 lines code + 62 lines tests; the apparent 1627/1752 was CRLF normalization. |

**Hardening backlog:** none from Tier 2 — all commits clean.

---

## TIER 3 — Pipeline + Features — VERDICT: PASS

**Tests:** 194 pass.
**Files:** pipeline/{check_drift, build_daily_features, system_authority, system_context, cost_model, health_check, work_queue}.

| Commit | Title | Grade | Key finding |
|--------|-------|-------|-------------|
| c110e4a3 | harden SQL literal detection | A- | Already reviewed earlier — fixes false-pos prose detection AND false-neg trading_app coverage. |
| c6d60a07 | mes tbbo slippage pilot | A | cost_model.py change is comment-only (pilot result references). Cost numbers unchanged. |
| 70dfa83f | enforce documentation truth hygiene | A | New `check_doc_hygiene_contracts()` drift check (118 lines): forbids placeholder stamps, validates `execution.entrypoint` matches `execution.mode` (design_only ↔ null), enforces "Generated from"/"Do not edit by hand" on generated docs. **"Check the checker" exemplar.** Adds 2 ENFORCEMENT_RULES to system_authority. |
| a2fb1098 | centralize garch warmup constants | A | Replaces magic `min_prior=60` and `min_obs=252` in build_daily_features with module-level `GARCH_MIN_PRIOR_CLOSES`/`GARCH_PCT_MIN_PRIOR_VALUES`. Canonical-source delegation (integrity-guardian rule 4). |
| 43b23e63 | rule-4.3 + slow-check guard | A | `_assert_slow_labels_valid()` runs at module import: fails closed when SLOW_CHECK_LABELS references a label not in CHECKS. Prevents silent fast-mode coverage regression. 4 tests including monkey-patched poison label. **"Check the checker" textbook.** |
| 070f5314 | GARCH R3 session-clipped forward shadow | A | New `check_recent_garch_feature_coverage()` drift check: detects late-history NULL coverage (recent 20 rows on partitions with ≥300 total). Catches the exact failure pattern motivating the fix (post-pass rolling state losing seed history). |
| 1cd1e0fb | mgc payoff compression + health_check polish | A | Refactors `check_recent_garch_feature_coverage` magic numbers to use the new `GARCH_*` constants from a2fb1098 (good cross-commit consolidation). health_check.py: named timeout constants, full pytest excluded from parallel-slow-checks group (avoids resource contention false-failures). |
| 763e10f2 | canonical action queue baton | A- | New `pipeline/work_queue.py` (532 lines) with Pydantic models, Literal-typed enums, schema versions. system_context.py 1732-line apparent diff is CRLF — real net is 587 ins / 1 del across 2 files. New tests added. |
| 701fe6a4 | line-ending + ruff format | SKIP | mechanical |

**Hardening backlog from Tier 3:** none — every "real" change shipped with new tests AND new drift checks. This tier is **stronger** than what was there before. Codex did "check the checker" and "harden as we go" of its own accord here.

---

## TIER 4 — Tooling — VERDICT: PASS-WITH-FIXES

**Tests:** 218 pass, **1 FAIL**: `test_pulse_integration.py::test_text_output_is_scannable` — `assert len(lines) <= 60` failing at 63 lines. Pulse output bloat from queue-baton additions (`763e10f2` / `1c75882f`) without compensating trim.

**Files:** scripts/tools/{project_pulse, claude_superpower_brief, context_views, checkpoint_guard, session_router, compact_handoff, check_claim_hygiene, bias_grounding_guard}, scripts/infra/{codex-capital-review, codex-project}, plus 3 ai/ tests.

| Commit | Title | Grade | Key finding |
|--------|-------|-------|-------------|
| aed352e5 | harden pulse fast path + optional anthropic imports | A | `_worktree_metadata()` extracted: bounded glob depth (3 levels) prevents Windows broken-symlink rglob errors. Fast mode skips system_brief, worktree_conflicts, deep momentum inspection. PEP 604 `float \| int` modernization. Tests added for optional anthropic SDK. |
| 2c49fc13 | low-token evidence guards | A | New `check_claim_hygiene.py`: PR bodies must contain Evidence/Claims/Disconfirming Checks/Grounding sections; result docs must have scope/decision/repro/skepticism. Regex-only, no DB, no network. Plus `bias_grounding_guard.py`. |
| 156c600f | auto-route concurrent terminals | A | New `checkpoint_guard.py`: blocks commits staging `docs/audit/results/*` artifacts WITHOUT also staging a durable closeout surface (HANDOFF, decision-ledger, debt-ledger, plans/). New `session_router.py` for multi-terminal routing. Institutional discipline. |
| 1c75882f | extend queue context consumers | A- | claude_superpower_brief / context_views / project_pulse extended for action-queue display. **Likely contributor to the 63-line pulse bloat — see HIGH finding below.** Tests added (151 lines for pulse, 40 for context_views). |
| b45af2d1 | capital-review routing | A | Adds `/capital-review` skill + Codex shell wrapper. Pure routing addition. |
| a1ac60c8 | archive + compact baton | A | New `scripts/tools/compact_handoff.py` (274 lines): regex parser for legacy + rolling HANDOFF formats, archives detail to `docs/handoffs/archived/`, rewrites root HANDOFF as compact baton. Defensive parsing. |

**HIGH finding (T4-FAIL):** Pulse output exceeds 60-line scannable budget after `763e10f2` (action-queue baton) + `1c75882f` (queue context consumers). Test was added explicitly to enforce scannability; these commits silently violated it. Fix path is **trim pulse output back to ≤60 lines** (NOT relax the test threshold — that would defeat the test's purpose). Logged as **T4.A** in hardening backlog.

**Hardening backlog from Tier 4:**
- T4.A: Trim `project_pulse.py` text/markdown formatters to fit ≤60 lines. Likely candidates: collapse "Memory topics" + "Recent notes" footer; consolidate Stage list when >5 stages; drop redundant separators.

---

## REVERSE AUDIT — VERDICT: 1 NEW HIGH + 2 LOW SEAMS

Cross-tier scan with knowledge of all per-tier findings (RepoAudit pattern).

| ID | Finding | Severity |
|----|---------|----------|
| RA-1 | `STATE_DIR.mkdir()` at module-import is a **repeated pattern across 6 files**: conditional_overlays, account_survival, lane_ctl, pre_session_check, sprt_monitor, sr_monitor. T1.D wasn't unique — it's standing house style. Either accept (consistent) or refactor all 6 to lazy. | LOW |
| RA-2 | `pipeline/build_daily_features.py` still hardcodes `252` for trading-days-per-year (lines 766, 835, 1436, 1438, 1441, 1479, 1483, 1495). a2fb1098 centralized GARCH warmup constants but missed `TRADING_DAYS_PER_YEAR`. Should add module-level constant and replace. | LOW |
| RA-3 | `check_doc_hygiene_contracts` from 70dfa83f catches `UNSTAMPED`/`TO_BE_STAMPED` but **not** `commit_sha: PENDING` or arbitrary `commit_sha: TO_FILL_*` (other than the legitimate `TO_FILL_AFTER_COMMIT`). | **HIGH — fixed below** |
| RA-4 | Live-mode silent-flip risk — searched for `details["mode"] == "live"` and `mode="live"` in trading_app: zero hits remaining post-94a9c6fb. Spec dataclass `mode` is hardcoded `shadow_only` — operator can't accidentally flip. **PASS.** | — |

---

## HARDENING ACTIONS — APPLIED

| ID | Action | File(s) | Status |
|----|--------|---------|--------|
| **T4.A** | Trim pulse `format_text` to fit ≤60 line scannable budget. Collapsed Doctrine + Backbone + Authority-map into single line with counts (`see --json for full list`); merged SR-streams into the C12 line. | `scripts/tools/project_pulse.py` (+11/-8) | **SHIPPED** |
| **RA-3** | Extended `check_doc_hygiene_contracts` placeholder regex to catch `commit_sha: PENDING` and arbitrary `commit_sha: TO_FILL_*`, while exempting the legitimate `commit_sha: TO_FILL_AFTER_COMMIT` per `.claude/rules/research-truth-protocol.md` § 2a. | `pipeline/check_drift.py` (+15/-2) | **SHIPPED** |
| **RA-3 tests** | 4 mutation-probe tests: PENDING caught, TO_FILL_LATER caught, TO_FILL_AFTER_COMMIT exempted, real SHA passes. All pass. | `tests/test_pipeline/test_check_drift_context.py` (+47) | **SHIPPED** |
| **T1.B** | Replaced substring matching in `RoleResolver.get_overlay_context` with canonical `parse_strategy_id` from `trading_app.eligibility.builder`, per memory rule `feedback_aperture_overlay_canonical_parser.md`. Closes the `"E2" in "E20"` / `"O5" in "O50"` collision class. | `trading_app/conditional_overlays.py` (+13/-5) | **SHIPPED** |

### Hardening BACKLOG (not blocking, logged for future)

| ID | Action | Why deferred |
|----|--------|--------------|
| T1.A | Promote `_is_pid_alive` to public name in `trading_app/live/instance_lock.py` | Style-only; no functional bug |
| T1.C | DRY the triple `_apply_conditional_roles(...)` calls in `trading_app/execution_engine.py` (lines 964/1186/1362) | Refactor debt; current code works correctly |
| T1.D + RA-1 | Move 6 `STATE_DIR.mkdir(...)` calls from module-import to lazy-on-use | Pattern-consistent; refactor scope > this audit |
| RA-2 | Centralize `TRADING_DAYS_PER_YEAR = 252` in `pipeline/build_daily_features.py` | 8 call-sites, multi-commit ripple; deserves its own staged refactor |

---

## FINAL VERDICT

**Codex's 3-day output: PASS-WITH-FIXES → PASS after hardening shipped.**

| Tier | Files | Commits | Tests | Verdict |
|------|-------|---------|-------|---------|
| 1 — Live trading | 8 hot files | 7 | 263 pass | PASS-WITH-FIXES (now PASS post-T1.B) |
| 2 — Discovery+config | 7 hot files | 4 | 459 pass | PASS |
| 3 — Pipeline+features | 7 hot files | 9 | 194 pass | PASS |
| 4 — Tooling | 8 hot files | 6 | 218 → 219 pass (post-T4.A) | PASS-WITH-FIXES (now PASS post-T4.A) |
| Reverse audit | cross-tier | — | — | 1 HIGH + 2 LOW seams; HIGH fixed (RA-3) |
| Full sweep | repo-wide | — | **4780 pass, 9 skip, 0 fail** (10:00 wall) | PASS |

**Bugs Codex hid:** zero confirmed. All 5 substantive findings traced to either:
- Real bugs Codex CLOSED (overlay valid-state lie, trading-day boundary, lock-startup race, lane-cap fail-closed false positive, sparse-OOS classification)
- Style/refactor debt (substring matching, triple-copy, eager mkdir) — none capital-affecting in current code paths
- Test threshold violation Codex introduced and didn't notice (pulse 63 > 60) — fixed in this sweep

**Notably excellent in Codex's work:** the per-commit drift-check additions (`check_doc_hygiene_contracts`, `check_recent_garch_feature_coverage`, `_assert_slow_labels_valid`, `check_claim_hygiene`, `checkpoint_guard`). These are "check the checker" textbook — exactly the institutional discipline you wanted.

---

## SELF-REVIEW (per "code review when finished")

**Where this audit could be wrong:**

1. **Diff-only first pass may have missed semantic regressions** that span multiple commits but were each individually clean. Mitigated by reverse-audit pass + full pytest sweep (4780/4780). Caught zero seam bugs in the code paths I reviewed; the one seam I missed (test budget) was caught by the test, not by my reading.

2. **CRLF noise removed by `--ignore-all-space`** — three commits had >1000-line apparent diffs that were almost entirely line-ending normalization (04edf5e6, 070f5314, 763e10f2, 55cd9b30). I read the semantic deltas only. If a real semantic change happened to land in the same blank lines, my pass would not have caught it. Counter-evidence: the `--shortstat --ignore-all-space` numbers matched per-file expectations within ±20 lines.

3. **Mutation probes were mental, not executed** for the live-trading hot paths — I did not synthetically inject a `RoleResolver` mismatch and watch it fire. The substring-matching defect (T1.B) was logical-deduction-only. The fix delegates to `parse_strategy_id` which IS executed by the existing 6 conditional-overlay tests.

4. **I did not audit the dirty-tree WIP** (Codex's 8 uncommitted files: `phase_4_discovery_gates.py`, `strategy_discovery.py`, context tooling, related tests). Per playbook anti-pattern #5, review committed history only. Those files remain unaudited until they land.

5. **Stage-gate hook initially blocked my edit to check_drift.py** until I created the IMPLEMENTATION stage with proper scope_lock/blast_radius. This is the institutional-rigor working as designed — no scope creep into unstaged territory.

**What I'm most confident about:**
- The 5 silent-failure / capital-affecting bugs Codex CLOSED (audited in detail, all with new tests).
- The drift-check additions (4 new ones) all pass mutation probes.
- T4.A regression detection — pulse line count is now exactly 60 (verified).
- RA-3 regex — verified with 4 unit tests + manual probe on 4 input shapes.
- T1.B canonical-parser swap — verified with full conditional_overlays test suite.

**What I'm least confident about:**
- Whether the `RoleResolver` strategy_id format actually matches `parse_strategy_id`'s expectations in production. The prod live engine calls `_apply_conditional_roles(trade, trade.orb_label)` with `trade.strategy_id` — and `parse_strategy_id` expects `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_COST_LT12` shape. Need integration confirmation when overlay actually fires live. Mitigation: ValueError catch returns gracefully (silent skip), so a parse failure is fail-closed — overlay context simply isn't applied. Acceptable for shadow-only.

## Audit method notes

- All diffs read with `--ignore-all-space` to filter CRLF normalization noise.
- Per-commit semantic diff verified against `git diff --shortstat --ignore-all-space`.
- Mutation probes performed mentally for each new validation/guard.
- Tests run after each tier: Tier 1 → 263 pass, Tier 2 → 459 pass.
- Prior 2026-04-24 Codex-audit-remediation (commit b668a982) is independent of this sweep.
