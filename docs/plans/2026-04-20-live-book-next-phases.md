# Live Book Next Phases

**Date:** 2026-04-20  
**Prerequisite:** [2026-04-20-live-book-truth-status-reaudit-v2.md](/tmp/canompx3-live-book-reaudit/docs/audit/results/2026-04-20-live-book-truth-status-reaudit-v2.md)  
**Status:** LOCKED  
**Rule:** No later phase begins until the gate for the current phase is satisfied.

## Phase 1 — Truth-Surface Repair

**Goal:** Restore trustworthy truth surfaces without inventing runtime failures.

### Deliverables
- correct the audit / handoff claim set so it no longer asserts a `prop_portfolio.py` failure that fresh verification disproved
- update `TRADING_RULES.md` live-book section to current reality
- update the active `topstep_50k_mnq_auto` profile notes in `trading_app/prop_profiles.py` so they do not claim a stale fixed lane count
- add explicit language distinguishing:
  - `operationally deployable`
  - `research-provisional`
  - `production-grade proof`

### Gate
- `python3 -m py_compile trading_app/prop_portfolio.py` passes
- `tests/test_trading_app/test_prop_portfolio.py` passes
- `TRADING_RULES.md` and active profile notes no longer contradict current code / allocation JSON
- audit artifacts no longer claim a runtime failure disproved by fresh command evidence

### Why first
- False audit claims corrupt every later judgment.

**Progress update (2026-04-20):** implemented and verified.

## Phase 2 — Live Attribution Restoration

**Goal:** Make current-lane evidence real.

### Deliverables
- lane-level logging for the current 6 strategy IDs:
  - fired
  - skipped
  - filled
  - missed
  - slippage
  - realized `pnl_r`
- report surface comparing realized vs modeled by `strategy_id`

### Gate
- current 6 strategy IDs produce non-zero rows in `paper_trades`
- enough rows exist to support a first mechanism audit

### Why second
- Without this, live mechanism claims remain speculative.

**Progress update (2026-04-20):**
- instrumentation path implemented and verified:
  - shared `paper_trades` write helper
  - backfill hardening against live-row overwrite
  - manual live logger routed through shared helper
  - durable `live_signal_events` table
  - orchestrator event writes and profile-backed exit bridge
  - read-only operator report:
    - `scripts/tools/live_attribution_report.py`
    - current-book report merges:
      - `lane_allocation.json`
      - `validated_setups` modeled comparator
      - `paper_trades`
      - `live_signal_events`
  - first live mechanism-audit pre-reg locked:
    - `docs/audit/hypotheses/2026-04-20-first-live-mechanism-audit.yaml`
  - canonical operator warning added:
    - `trading_app/pre_session_check.py`
    - warns when the current session lane(s) still have zero live/shadow completed rows and zero event rows
- gate still pending:
  - no live runtime rows yet for the current 6 strategy IDs
  - first real mechanism audit remains blocked until those rows exist

### First Runtime Collection Command

Once the live path starts producing rows, the first admissible audit surface is:

```bash
/mnt/c/Users/joshd/canompx3/.venv-wsl/bin/python scripts/tools/live_attribution_report.py \
  --allocation-path docs/runtime/lane_allocation.json \
  --db-path /mnt/c/Users/joshd/canompx3/gold.db \
  --journal-path /mnt/c/Users/joshd/canompx3/live_journal.db
```

Use `--json` for machine-readable output during the first mechanism audit.

## Phase 3 — Scale-Ready Gate

**Goal:** Replace hand-waving with a formal go / no-go scale gate.

### Deliverables
- canonical `scale_ready` definition
- explicit proving-loop rule
- pass / fail output for `topstep_50k_mnq_auto`

### Required checks
- attribution coverage exists
- operator surfaces healthy
- no unresolved scale-blocking slippage debt
- correlation control populated / tested
- doctrine allows promotion beyond research-provisional language

### Gate
- all checks green
- explicit decision:
  - `NOT READY`
  - `CONDITIONALLY READY`
  - `READY`

### Why third
- Scale should not be discussed again until this exists.

## Phase 4 — Cost / Risk Closure

**Goal:** Close the remaining scale-blocking realism debt.

### Deliverables
- deployed-session slippage completion:
  - `EUROPE_FLOW`
  - `COMEX_SETTLE`
  - `US_DATA_1000`
- event-tail assessment for MNQ
- `corr_lookup` population and verification

### Gate
- cost debt reduced from partial to adequate for current live book
- correlation guard no longer a no-op for scaling discussions

## Phase 5 — Clean Mode A Rediscovery of Active Families

**Goal:** Restore evidence legitimacy for the actual live families.

### Deliverables
- pre-registered clean rediscovery run
- same family definitions only:
  - current live session / entry / RR / filter families
- survival report:
  - rediscovered unchanged
  - degraded
  - not rediscovered

### Gate
- active families reclassified honestly under clean Mode A

### Why fifth
- This is the step that upgrades current live families from provisional toward validated.

## Order Constraint

This order is not cosmetic:
- Phase 1 fixes broken proof surfaces
- Phase 2 creates real live evidence
- Phase 3 prevents premature scale
- Phase 4 closes realism / exposure debt
- Phase 5 repairs research legitimacy

Skipping ahead creates the exact same failure mode the re-audit just killed.
