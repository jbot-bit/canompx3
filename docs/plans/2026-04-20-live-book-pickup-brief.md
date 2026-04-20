# Live Book Pickup Brief — 2026-04-20

## Scope

This brief is for continuing the live-book audit/hardening workflow from branch:

- `codex/live-book-reaudit-fix`
- head commit: `2da48312`

This is **not** a scale branch. It is an audit/hardening branch.

## What Is Already Done

### Audit/truth-state work

- Re-audited the earlier live-book diagnosis.
- Killed the claims:
  - "scale is the highest-confidence lever"
  - "discovery is not the main issue"
- Restored the correct framing:
  - current deployed book is operationally real
  - current evidence is still provisional
  - no honest scale call yet

### Runtime/operator hardening

- `paper_trades` shared write surface exists
- `live_signal_events` event journal exists
- allocator-scoped attribution report exists
- pre-session check now surfaces live-attribution gaps

### Code-review hardening

Fixed the concrete review failures:

1. Backfill rows no longer count as completed live evidence in the attribution report
2. Realized PnL/slippage aggregates now exclude backfill rows
3. Evidence is scoped to allocator `rebalance_date`
4. Orchestrator no longer writes fabricated completed live rows when entry context is missing
5. Orchestrator bridged rows now use real entry timestamps

## Verified State

### Branch verification

- targeted compile/lint/test pass completed
- broader regression suite completed
- smoke report and pre-session commands run against canonical DBs

### Current runtime truth

For current active 6-lane book, since allocator rebalance date `2026-04-18`:

- `live_signal_events`: `0`
- completed `paper_trades` with `execution_source IN ('live', 'shadow')`: `0`
- report status for all 6 lanes: `NO_EVIDENCE`

So the branch is code-ready, but the live mechanism is still **evidence-empty**.

## What Is Still Left

### Real next work

1. Claude audit the branch
2. Refresh runtime prerequisites
   - MNQ data freshness
   - Criterion 11 survival report
3. Collect real current-window runtime rows
4. Run first live mechanism audit using those rows

### Not honest to do yet

Do **not**:

- make a scale recommendation
- promote to a scale-ready gate verdict
- claim live slippage / skip / reject behavior is known
- claim lane-by-lane live quality is known

Those need real rows first.

## Exact Commands

### Current attribution report

```bash
/mnt/c/Users/joshd/canompx3/.venv-wsl/bin/python scripts/tools/live_attribution_report.py \
  --allocation-path /mnt/c/Users/joshd/canompx3/docs/runtime/lane_allocation.json \
  --db-path /mnt/c/Users/joshd/canompx3/gold.db \
  --journal-path /mnt/c/Users/joshd/canompx3/live_journal.db
```

### Current pre-session truth surface

```bash
DUCKDB_PATH=/mnt/c/Users/joshd/canompx3/gold.db \
/mnt/c/Users/joshd/canompx3/.venv-wsl/bin/python -m trading_app.pre_session_check \
  --session NYSE_OPEN --profile topstep_50k_mnq_auto
```

### Core branch verification

```bash
python3 -m py_compile \
  trading_app/prop_profiles.py \
  trading_app/pre_session_check.py \
  trading_app/live/session_orchestrator.py \
  scripts/tools/live_attribution_report.py \
  tests/test_tools/test_live_attribution_report.py \
  tests/test_trading_app/test_pre_session_check.py \
  tests/test_trading_app/test_session_orchestrator.py
```

```bash
/mnt/c/Users/joshd/canompx3/.venv-wsl/bin/python -m pytest \
  tests/test_tools/test_live_attribution_report.py \
  tests/test_trading_app/test_pre_session_check.py \
  tests/test_trading_app/test_session_orchestrator.py -q
```

## Files To Read First

- `HANDOFF.md`
- `docs/audit/results/2026-04-20-live-book-truth-status-reaudit-v2.md`
- `docs/audit/hypotheses/2026-04-20-first-live-mechanism-audit.yaml`
- `docs/plans/2026-04-20-live-book-next-phases.md`
- `scripts/tools/live_attribution_report.py`
- `trading_app/pre_session_check.py`
- `trading_app/live/session_orchestrator.py`

## Known Repo Hygiene Leftovers

These are **not** part of this branch’s content, but still exist in the base repo state:

- unrelated dirty files on base branch `live-book-rehab`
- two stale Git worktree admin entries under `.git/worktrees`

Those stale worktree admin dirs could not be deleted from this shell because the filesystem returned `Read-only file system`.

## Bottom Line

This branch is ready for audit and pickup.

It is **not** ready for scale conclusions.

The next honest step is to get real current-window runtime rows, then run the first live mechanism audit.
