# Full-System Correctness Audit — Findings Ledger

**Date:** 2026-05-31
**Worktree:** `canompx3-correctness-audit` (branch `session/joshd-correctness-audit`, off `origin/main` @ `8a289cb7`)
**Method:** falsifiable-claim-first. Every finding states a claim, a **probe set defined up front**, a pass/fail condition, evidence (execution output only), a classification, and a decision with capital impact. A "DEAD" verdict requires *every* probe negative. UNCERTAIN is reported, never guessed.
**Authority for method:** plan §"The audit method (anti-bias, falsifiable)". Corrects the v1 `rolling_correlation.py` one-probe miss.

**Classifications:** WORKS / BROKEN / HALF-WORKS / DEAD / SUPERSEDED / UNCERTAIN.
**Capital impact tiers:** none / research / capital-path. Capital-path findings are **Tier-B: surfaced, never auto-fixed**; CRIT/HIGH truth-layer fixes trigger the `evidence-auditor` adversarial gate before the next phase.

---

## Phase 0 — Ground-truth snapshot (read-only)

### gold.db health & freshness (via gold-db MCP, read-only)
| Table | Rows | Horizon |
|---|---|---|
| `daily_features` | 35,406 | max trading_day 2026-05-29 |
| `orb_outcomes` | 8,949,498 | max trading_day 2026-05-28; max entry_ts 2026-05-28T22:27Z |
| `validated_setups` | 871 | — |
| `edge_families` | 528 | max created 2026-05-23 |

- DB path: `C:\Users\joshd\canompx3\gold.db` (canonical, `path_source=canonical`), size 7.63 GB, mtime 2026-05-31T03:17Z.
- Access policy: read-only, no raw SQL writes, no live DB writes, no GitHub live access. **Status: OK.**

### check_drift.py baseline
- **173 checks PASSED, 0 skipped (DB available), 22 advisory. NO DRIFT.** (highest check index 195; plan cited "198/28 parity" — volatile stat, live count is authoritative.)
- **Finding F0-A (meta):** a fully-green static-invariant suite coexists with the documented real bugs (Rapid sim-cap leak, `--strict-live-clean` stash, COMEX_SETTLE provenance scare). This is the empirical proof of the plan's thesis — *static "wired/consistent" checks do not see behavioral/semantic correctness.* Class: WORKS (the suite does what it claims); the gap is **coverage**, addressed in Phases 1–2. Capital impact: none (meta-observation).

### Active capital paths (Phase 1 target set) — via strategy-lab MCP
Profile `topstep_50k_mnq_auto`, rebalance 2026-05-30 (1 day old, staleness OK), `active_count=3`, `paused_count=845`, `all_scores_count=848`. Allocation file: `docs/runtime/lane_allocation.json`.

| # | strategy_id | instrument | orb_label | O | RR | filter | status |
|---|---|---|---|---|---|---|---|
| L1 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | MNQ | COMEX_SETTLE | 5 | 1.5 | OVNRNG_100 | DEPLOY |
| L2 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | MNQ | US_DATA_1000 | 15 | 1.5 | VWAP_MID_ALIGNED | DEPLOY |
| L3 | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08` | MNQ | TOKYO_OPEN | 5 | 1.5 | COST_LT08 | DEPLOY |

All 3 are E2 entry model, CB1, RR1.5, MNQ. This is the highest-capital-risk surface — swept end-to-end in Phase 1.

### Repo / cross-AI state (Phase 5 target set)
- **19 worktrees** (incl. a live Codex task on `codex/start-bot-dashboard-live-pilot-ux` in the main checkout — DO NOT TOUCH).
- **41 local branches, 51 remote branches.**
- **2 open PRs:** #348 (grounding-integrity), #345 (post-compaction hook-fix).
- Coordination note: this audit runs in an isolated worktree precisely because the main checkout is being mutated by a live peer. Pruning is a separate reviewed step (Phase 5).

### Learning-loop primitives present (Phase 4 targets)
`docs/runtime/debt-ledger.md`, `docs/runtime/decision-ledger.md`, `docs/runtime/action-queue.yaml`, `memory/*.md` + `MEMORY.md`, Ralph loop, auto-memory-capture hooks — all present. Phase 4 wires audit output into these, not into new gates.

---

## Findings ledger

> Format per finding:
> **ID** · **Claim** · **Capital impact** · **Probe set (defined up front)** · **Pass/fail condition** · **Evidence** · **Classification** · **Decision**

### Phase 1 — Active-capital-path data-integrity sweep (READ-ONLY, Tier-B)

**Probe set (defined up front)** per active lane, end-to-end chain vs **live gold.db** (`duckdb read_only=True`, canonical `GOLD_DB_PATH`):
- **L1 validating row** — a `validated_setups` row matching all dims (instrument/orb_label/orb_minutes/entry_model/filter_type/rr_target/confirm_bars) with `status='active'` AND `retired_at IS NULL`.
- **L2 live-routable filter** — `filter_type ∈ trading_app.config.ALL_FILTERS` (canonical, 99 filters).
- **L3 canonical trade-window provenance** — `pipeline.dst.orb_utc_window(day, orb_label, orb_minutes)` resolves without raise; orb_label ∈ `SESSION_CATALOG` (13 labels). No `break_ts` fallback.
- **L4 entry model supported** — `entry_model ∈ config.ENTRY_MODELS` AND `∉ SKIP_ENTRY_MODELS`.
- **Orphan sweep** — every `status='active'` validated_setups row (N=848) checked for non-routable filter or retired/unsupported entry model.

**Pass condition:** all 4 links OK for every active lane AND 0 orphan violations. **Fail:** any missing/retired/non-routable link, or any active row whose filter/entry-model no longer routes.

**Evidence** (`.claude/scratch/phase1_integrity_sweep.py`, executed against gold.db @ 2026-05-31):
| Lane | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | active, N=513, ExpR=0.2151 ✓ | OVNRNG_100 ✓ | 17:30–17:35Z ✓ | E2 ✓ |
| MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15 | active, N=701, ExpR=0.2101 ✓ | VWAP_MID_ALIGNED ✓ | 14:00–14:15Z ✓ | E2 ✓ |
| MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08 | active, N=427, ExpR=0.2037 ✓ | COST_LT08 ✓ | 00:00–00:05Z ✓ | E2 ✓ |

Orphan sweep: **848 active rows scanned, 0 integrity violations.**

**F1-A · Active lanes route end-to-end · capital-path.**
- Classification: **WORKS.** All 3 active lanes pass L1–L4; 0 orphans.
- Decision: no action. Tier-B confirmed clean (read-only; nothing changed).

**F1-B · Two `status='active'` rows per active lane (apparent duplicate) · capital-path.**
- Probe-deeper result: the second row is an **intentional stop-multiplier sibling** (`_S075` suffix, `family_hash` prefix `sm075` vs `sm100`, same root hash). Distinct `strategy_id`, distinct `stop_multiplier`.
- Cross-check vs `lane_allocation.json`: all 3 active lanes bind the **base sm100** `strategy_id`; **0** active lanes carry `_S075`; the 264 `_S075` rows are all in the **paused** pool. No double-routing.
- Classification: **WORKS** (not a defect). Decision: no action. *Discipline note:* the raw 2-rows-per-lane reading would mis-classify as a duplicate-routing bug without the family_hash mechanism check — logged as a recurrence-guard candidate for Phase 4 (an active-lane → exactly-one-deployed-variant assertion would make this mechanically self-evident).

### Phase 2 — Behavioral correctness gap report ("does it actually work")

**Method:** per feature area — *what observable output proves it works, and does a test assert it?* The ABSENCE of such a test is the finding. Highest-capital-risk gaps get a behavioral test (RED→GREEN; capital paths xfail-pinned, fix gated).

**Existing behavioral spine (verified present & strong):**
- `tests/test_integration/test_backtest_live_convergence.py` (549 L) — asserts backtest (`outcome_builder.compute_single_outcome` E2) == live (`ExecutionEngine`) produce BYTE-IDENTICAL E2 entry ts+price on a deterministic DST-spanning fixture corpus. Grounded in Chan 2013 Ch1 p4. Covers **entry-timing** correctness. Class: WORKS.
- `scripts/tests/canary_suite.py` (649 L) — contamination-trap suite (every guard catches its trap). Class: WORKS.

**Coverage-gap map** (capital-path module → observable output → is it asserted?):
| Module | Observable output that proves correctness | Behavioral test asserting it? |
|---|---|---|
| `outcome_builder`/`execution_engine` (E2 entry) | backtest entry == live entry (ts+price) | ✅ `test_backtest_live_convergence` |
| `account_survival` | sim survival/DD/MLL/contract-cap modelling | ✅ `test_account_survival.py` |
| `prop_portfolio.select_for_profile` (prop sizing) | prop contract cap binds prop book | ✅ `test_contract_cap` (tradeify) |
| **`prop_portfolio.select_for_profile` (self_funded sizing)** | **self_funded book bound by RISK, not prop cap** | ❌ **GAP → F2-A (defect, test added)** |
| **`rebalance_lanes --strict-live-clean`** | **non-CONTINUE SR status → PAUSE before live alloc** | ❌ **GAP → F2-B (no defect, coverage gap)** |
| `lane_allocator` | rebalance/feature-cache correctness | partial (7 test refs; no strict-live behavioral assert) |

**F2-A · `select_for_profile` (doctrine's "build_book") binds a `self_funded` book on the prop-style contract cap · capital-path · HALF-WORKS.**
- **Symbol note (self-review correction):** there is no `build_book` function — the doctrine's shorthand "build_book" is the actual symbol **`select_for_profile`** (`prop_portfolio.py:518`); the contract-budget logic is inside it (line 558). `_load_strategies_and_build_books` is a separate higher-level wrapper. The test and reproduction call `select_for_profile` (correct).
- **Probe set:** (1) does `select_for_profile` have a firm branch on `contract_budget`? (2) is there a live `self_funded` profile? (3) does building its book hit the prop micro-cap? (4) does any existing test assert the doctrine-correct behavior?
- **Evidence (execution-traced):**
  - `prop_portfolio.py:558` (inside `select_for_profile`, line 518) — `contract_budget = tier.max_contracts_micro` **unconditional**; the "Contract cap reached (N/{budget} micro)" exclusion (line 585/591) fires regardless of `firm`. No `self_funded` branch (grep: only firm logic is `banned_instruments`).
  - Live profile `self_funded_tradovate` (firm=self_funded, $30k) → tier `max_contracts_micro=12`.
  - Reproduced: `select_for_profile(AccountProfile('sf','self_funded',30000,max_slots=30), 30 strats)` → **total_contracts=12, 18× "Contract cap reached (12/12 micro)"**. The prop cap bound a personal-capital book.
  - No existing test asserts self_funded contract-cap behavior (`test_contract_cap` uses `tradeify`; self_funded tests assert slot/cognitive caps only).
- **Doctrine violated:** `.claude/rules/self-funded-sizing-doctrine.md` — "prop caps NEVER bound personal-capital earnings; self_funded `max_contracts_*` is a margin/sanity guard, not an earnings ceiling." The marker drift-guard (`check_prop_caps_do_not_leak_into_self_funded`) is marker-presence only; the doctrine itself states the structural book-builder branch is an unbuilt "Tier-B follow-up."
- **Decision: Tier-B — SURFACED, NOT FIXED.** Fix = a firm-aware sizing branch in `build_book` (capital-allocation path; adversarial-audit gated; needs operator sign-off). **Deliverable:** behavioral test `test_self_funded_book_not_bound_by_prop_contract_cap` added as `xfail(strict=True)` — it PINS the gap as an executable assertion and auto-fails (XPASS) the moment the fix lands, forcing conversion to a live regression guard. Verified: `48 passed, 1 xfailed`.
- *Capital-impact nuance:* latent — binds only when a self_funded book has >12 deployable lanes; the deployed `topstep_50k_mnq_auto` profile (3 lanes) is unaffected. But `self_funded_tradovate` exists and would be throttled.

**F2-B · `--strict-live-clean` live-safety gate has no behavioral test · capital-path · coverage-gap (no defect found).**
- **Probe:** does the strict gate (force non-`CONTINUE` SR-status strategies to PAUSE before fresh live allocation, `rebalance_lanes.py:128-138`) have a test asserting that behavior?
- **Evidence:** zero test files reference `strict_live_clean`/`strict-live-clean`. Traced the code: strict list is correctly threaded into both `apply_*_gate` (line 144) and `build_allocation` (line 156) via the reassigned `scoring_input` — **no logic hole found**. The gate behaves correctly; it is simply **unasserted**.
- **Classification: WORKS (untested).** Decision: log as a coverage gap → Phase 4 recurrence candidate (this is a live-safety control on a recent-incident path; a behavioral test belongs here but writing it requires SR-scoring fixtures — queued, not a defect). No fix needed.

### Phase 3 — Verified dead/half-wired resolution

**F3-A · `trading_app/rolling_correlation.py` re-adjudicated (v1 said DEAD) · none · WORKS.**
- **Full probe set (every probe must be negative for DEAD):**
  | Probe | Result |
  |---|---|
  | 1. Python import (non-test) | none — BUT module is CLI-invoked, not imported |
  | 1b. Python import (test) | ✅ `test_rolling_correlation.py` imports `compute_rolling_correlation` — **12 tests** |
  | 2. allowlist/manifest | ✅ `check_drift.py:1885` — listed in the read-only-consumers allowlist (the v1 MISS) |
  | 4. CLI entrypoint | ✅ `__main__` (line 360) + argparse `--db-path/--instrument/--window-days/...` (line 13 usage doc) |
  | 5. graph/centrality | ✅ `docs/ralph-loop/import_centrality.json`, ralph audit reports |
- **Verdict: NOT DEAD.** It is a standalone CLI risk-analysis tool (`python -m trading_app.rolling_correlation`). Zero importers ≠ dead when the caller is the operator/shell. **v1's one-grep "DEAD" verdict is OVERTURNED.** Decision: keep; no deletion. *This is the canonical proof of why the method requires the full probe set.*

**F3-B · `lane_allocation` single-file → per-profile-dir migration (plan's "half-wired" example) · capital-path · WORKS (intentional staged migration).**
- **Probe:** are both `docs/runtime/lane_allocation.json` (legacy) and `docs/runtime/lane_allocation/<profile>.json` (new) live? Is it silent half-life or an annotated migration? Are they in sync? Which does the reader prefer?
- **Evidence:**
  - `lane_allocator.py:73-77` — explicit comment: *"Stage 1a (2026-05-21): per-profile directory introduced alongside the legacy single-profile file. Writer emits to BOTH paths; reader prefers new, falls back to legacy. Stage 1d removes the legacy path."* Schema spec: `docs/specs/lane_allocation_schema.md`.
  - Reader `prop_profiles.load_allocation_lanes` (line 1458+) — single-owner resolver, "prefers the per-profile file first, falling back to legacy" (documented removal point line 1333).
  - **Files byte-identical:** both SHA `294955e5...`, 260,514 bytes — dual-write in sync.
- **Verdict: WORKS — intentional, spec'd, staged dual-write migration, NOT silent half-life.** It is annotated at the constant, the resolver, and the schema spec, with an explicit teardown stage (1d). Decision: no action; this is the plan's "explicit intentional-fallback annotation" case, already satisfied by the existing comments.

**F3-C · `walk_forward.py` vs `walkforward.py` (ralph-flagged) · none · DEFERRED-UNCERTAIN.**
- Ralph audit (`ralph-audit-report.md:84,367`) flagged two similarly-named files; `strategy_validator.py` imports `walkforward.py` (active). `walk_forward.py` *may* be an older copy. **Not re-probed in this pass** (out of capital-path scope; needs its own full probe set before any delete). Classification: UNCERTAIN — reported, not guessed, not deleted. Queued as a Phase-4 follow-up.

**Phase 3 net: zero deletions.** Both flagship candidates (rolling_correlation, lane_allocation) are alive/intentional. This is a correct outcome — the method's job is to PREVENT the v1-style wrong deletion, and it did.

### Phase 5 — Cross-AI coordination (audit + classify; PRUNE is a separate reviewed step)

**Inventory:** 19 worktrees, 41 local + 51 remote branches, 3 open PRs (was 2 at Phase 0 — Codex opened **#349** on `codex/start-bot-dashboard-live-pilot-ux` during this audit, the live task I isolated from. Isolation decision validated: its work is now in the proper review channel).

**Open PRs (active — DO NOT prune):**
| PR | Branch | Status |
|---|---|---|
| #349 | codex/start-bot-dashboard-live-pilot-ux | ACTIVE (Codex live task, dashboard START_BOT) |
| #348 | session/joshd-grounding-integrity | ACTIVE (grounding-integrity, ahead=3) |
| #345 | session/push-hookfix | ACTIVE (post-compaction hook, ahead=2) |

**Branch classification (`git merge-base --is-ancestor origin/$b origin/main`):**
- **MERGED (ahead=0) — prunable remote branches (9):** `codex/chordia-lane-bench-automation`, `session/joshd-allocation-intel-cli`, `session/joshd-auto-memory-capture`, `session/joshd-canary-harness`, `session/joshd-ehr-validation-mode`, `session/joshd-lane-allocator-strict-live-clean`, `session/joshd-live-readiness-gates-clean`, `session/joshd-mffu-layerc`, `session/self-funded-sizing-guard-clean`. These are fully in main → safe to delete remote+local+worktree.
- **UNMERGED, ahead≥1 (≈28):** mix of (a) open-PR heads (#348/#345 above — keep), (b) genuine in-flight work (e.g. `ralph-10x-iterations` ahead=11, `ci-format-fix` ahead=6), and (c) **possible superseded-by-content** — memory warns `merge-base` falsely flags content-superseded branches as "unmerged" (a cherry-picked/squashed delta shows ahead≥1 but is already in main by content). **Per-branch content-diff adjudication is the PRUNE step — deferred, NOT done here** (sweeping in-flight peer work has bitten before; `feedback_orphan_merged_by_content_not_commit`).
- **Worktrees on MERGED branches → prunable (8):** `canompx3-{adaptive-stops-audit,adaptive-stops-h0,adaptive-stops-h4,canary-harness,heartbeat,self-funded-guard,worktree-lease-fix,grounding}` — their branches are ahead but most map to merged/PR'd work. Worktree pruning requires per-tree dirty-state check (`git -C <wt> status`) before removal — also deferred to the reviewed prune step.

**MEMORY.md vs .codex/MEMORY.md duplication:** flagged for the "link don't mirror" resolution in Phase 4 (checked below).

**Decision:** classification complete; **pruning surfaced as a separate reviewed action** (see Phase 6 / surfaced decisions). No branches/worktrees deleted in this pass.

### Phase 4 — The self-improving loop (the "keeps learning" deliverable)

**4a · Findings → memory wiring (architecture confirmed).** Repo `memory/` is **gitignored** (`.gitignore:110`); the durable corpus is **user-private** at `~/.claude/projects/.../memory/` (**221 feedback files**, 108KB `MEMORY.md` index). `.codex/MEMORY.md` is a **policy pointer** ("Codex uses the shared workspace memory files; long-term = `MEMORY.md` in main session") — **not a mirror**, so the plan's "MEMORY.md vs .codex duplication" concern is already resolved by the existing link-don't-mirror design. **WORKS.** The loop = finding → user-private `memory/*.md` → `MEMORY.md` index → auto-loaded next session (survives `/clear` via auto-memory-capture hook). This audit writes recurrence-risk findings into that store (done in 4d).

**4b · Recurrence mining (evidence-driven, n≥2 only).** Clustered all 221 feedback files by failure class:
| n | Failure class |
|---|---|
| **13** | **worktree / lease / branch-flip / concurrent-session** ← dominant |
| 7 | stale-summary / SHA / baton-drift |
| 7 | MCP / process-accumulation |
| 6 | pre-commit / lock / gold.db-contention |
| 5 | CRLF / encoding / diff-renderer |
| 1 | wired-but-dead-until-merged |

The **worktree/branch-flip/concurrent-session class (n=13)** is the clear recurrence winner — and it **fired 3× live in this very session**: (1) stale-lease force-release nearly colliding with the live Codex task, (2) shared-state-guard self-reference false positive, (3) **branch-flip-guard false-positive on every Bash call** (~20×). This is the n≥2 trigger for a targeted improvement.

**F4-A · branch-flip-guard false-fires across the AI/worktree boundary · none (friction, not capital) · HALF-WORKS.**
- **Mechanism (traced):** `settings.json` hardcodes the hook command to the **main-checkout** path (`feedback_worktree_hooks_resolve_to_main_checkout_path`). So a Bash call in *my isolated audit worktree* invokes the **main checkout's** `branch-flip-guard.py`, which reads the **main checkout's** `.claude.pid` (records `main`) and the **main checkout's** current branch (= `codex/start-bot-...`, because the live Codex task flipped it). They differ → BLOCK fires, **falsely attributing Codex's legitimate branch to my session.** Advisory-only (never blocked a commit), but ~20× noise/session and it masks any *real* flip.
- **Proposed fix (NOT applied — guardrail path, and the main-checkout hook is in use by the live Codex session):** the guard should compare the branch of the **worktree the Bash command actually ran in** (resolvable from `tool_input.command`'s `-C <path>` / cwd or `CLAUDE_PROJECT_DIR`) against *that worktree's* `.claude.pid`, not blindly the main-checkout lock. Alternatively, scope the guard to no-op when the invoking cwd's git-common-dir ≠ the lock's worktree. **Surfaced for sign-off** (modifying a safety hook the live peer depends on is Tier-B-adjacent; do it deliberately, not mid-audit).
- Decision: log to memory + action-queue (done 4d); fix gated on operator review.

**F4-B · stale-MCP-restart warning hook is NOT in `origin/main` (wired-but-dead-until-merged) · none · DEAD-until-merged.**
- **Evidence:** the "warn when a running MCP server's committed code is newer than its process" hook (memory: committed `92e71d7b`) is **absent from main** (`grep "newer than" .claude/hooks/session-start.py` → empty in this main-based worktree); it lives only on `session/joshd-grounding-integrity` (**PR #348, OPEN**). The auto-memory-capture loop **IS** in main (3 hook registrations in `settings.json`). So of the two hooks the plan asked to confirm live: auto-memory-capture = **LIVE**, stale-MCP-restart-warn = **DEAD until #348 merges.**
- Decision: surfaced — merging #348 activates it. No code change here.

**4c · Wire audit output into action-queue (done).** Added a tracked item for the F2-A capital fix + F4-A guard fix so they don't fall through between assistants (see `docs/runtime/action-queue.yaml`).

**4d · Findings → durable memory (done).** Wrote `project_full_system_correctness_audit_2026_05_31.md` to the user-private memory store + indexed in `MEMORY.md`, capturing: the audit verdict, F2-A (capital, gated), F4-A (guard false-fire + fix), F4-B (#348 activates stale-MCP-warn), and the method-win (rolling_correlation DEAD-verdict overturn = full-probe-set discipline).

**Phase 4 net:** the loop is wired to *accumulate* (memory) and *propose* (action-queue + this ledger), evidence-driven (n=13 recurrence → one targeted guard fix proposed, not speculative new gates). No new blocking checks added — per the plan, "keeps learning ≠ more gates."

---

## Surfaced decisions (Tier-B — require operator sign-off)

1. **F2-A — self_funded prop-cap leak (capital).** Add a firm-aware sizing branch to `prop_portfolio.build_book` so `self_funded` books are bound by risk, not `tier.max_contracts_micro`. Behavioral test already pins the gap (`xfail(strict=True)`). Adversarial-audit gated.
2. **F4-A — branch-flip-guard false-fire fix (guardrail hook).** Scope the guard to the invoking worktree, not the main-checkout lock. Touches a safety hook the live Codex session uses — do deliberately.
3. **Phase 5 pruning (cross-AI).** 9 MERGED remote branches + ~8 worktrees on merged/PR'd work are prunable; execute as a **separate reviewed step** with per-tree dirty checks (sweeping in-flight peer work has bitten before).
4. **PR #348 merge** activates the stale-MCP-restart warning (F4-B) on main.
