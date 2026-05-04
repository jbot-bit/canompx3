---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Multi-Instrument Live Session — Design

## Date: 2026-03-10
## Status: IMPLEMENTING

## Problem

The live trading system runs one instrument at a time. The user trades MGC, MNQ,
MES, and M2K simultaneously. To get signals for all instruments, they'd need to
manually start 4 separate processes. The dashboard only manages one.

## Architecture Decision

**Single-process, multi-orchestrator via `asyncio.gather()`.**

Each instrument gets its own `SessionOrchestrator` (proven, tested pattern).
All run concurrently in one async event loop. Broker auth is per-orchestrator
(4 token fetches — negligible overhead for a 23-hour session).

### Why NOT multi-process

- Process isolation is overkill for signal-only mode (no orders)
- Single log file is easier to monitor than 4
- Dashboard manages one subprocess, not four
- Shared stop-file works naturally in one process

### Component Analysis

| Component | Shared? | Notes |
|-----------|---------|-------|
| BrokerAuth | No (1/inst) | Each orchestrator creates its own — simple, isolated |
| DataFeed | No (1/inst) | One WebSocket per symbol (Tradovate limitation) |
| ExecutionEngine | No (1/inst) | Portfolio + cost spec are per-instrument |
| LiveORBBuilder | No (1/inst) | Tracks bars per instrument |
| PerformanceMonitor | No (1/inst) | Daily reset is instance-scoped |
| PositionTracker | No (1/inst) | Isolation prevents cross-instrument contamination |
| Signal file | Shared | `live_signals.jsonl` — append-mode, instrument-tagged |
| Stop file | Shared | `live_session.stop` — checked by all feeds |

## Bug Fix: Stop-File Race Condition

**Current:** Each feed deletes the stop file when it detects it. First feed to
see it deletes it → other feeds never stop.

**Fix:** Feeds no longer delete the stop file. The runner deletes it after all
feeds have exited. Each feed independently detects the file and sets
`_stop_requested = True`.

## Changes

### New: `trading_app/live/multi_runner.py`

```python
class MultiInstrumentRunner:
    """Run signal-only sessions for multiple instruments concurrently."""

    def __init__(self, instruments, broker, demo, signal_only, ...):
        self.orchestrators = {
            inst: SessionOrchestrator(instrument=inst, ...)
            for inst in instruments
        }

    async def run(self):
        tasks = {
            inst: asyncio.create_task(orch.run())
            for inst, orch in self.orchestrators.items()
        }
        # Wait for all to complete (or one to fail)
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        # Log per-instrument results

    def post_session(self):
        for orch in self.orchestrators.values():
            orch.post_session()
```

### Modified: `scripts/run_live_session.py`

- Accept `--instrument MGC` (single, existing) or `--all` (all active instruments)
- `--all` uses `ACTIVE_ORB_INSTRUMENTS` from `pipeline.asset_configs`
- `--all` creates `MultiInstrumentRunner` instead of single `SessionOrchestrator`

### Modified: `trading_app/live/tradovate/data_feed.py`

- Remove `_STOP_FILE.unlink(missing_ok=True)` from heartbeat (line 151)
- Feed detects stop file and exits, but doesn't delete it

### Modified: `trading_app/live/projectx/data_feed.py`

- Same stop-file fix (lines 203, 319)

### Modified: `ui/copilot.py`

- Add "Start All Instruments" button alongside per-instrument selector
- Uses `--all --signal-only` flag

### New: `tests/test_trading_app/test_multi_runner.py`

- Test multi-orchestrator creation
- Test concurrent run (mock feeds)
- Test stop-file handling (no deletion race)
- Test post_session called for all instruments

## Constraints

- One-way dependency respected: no pipeline imports in multi_runner
- No schema changes
- No entry model changes
- Existing single-instrument mode unchanged (backward compatible)
- Signal file format unchanged (already has instrument field)
