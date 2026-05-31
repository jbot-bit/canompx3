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

_(Phase 2+ findings appended below.)_
