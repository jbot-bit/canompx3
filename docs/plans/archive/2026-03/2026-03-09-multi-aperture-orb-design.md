---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Multi-Aperture ORB Support in ExecutionEngine

## Date: 2026-03-09
## Status: Approved (4TP)

## Problem

ExecutionEngine creates ONE ORB per session using the session's default duration
from `ORB_DURATION_MINUTES`. But strategies are validated on specific apertures
(5m, 15m, or 30m). A 5m TOKYO_OPEN strategy validated on a 5m ORB trades against
a 15m ORB in live — wrong signal, invalidates all backtested validation.

## Evidence

- `validated_setups` has 967 O5, 573 O15, 256 O30 strategies
- TOKYO_OPEN alone: 130 O5 + 221 O15 strategies (MGC)
- The two loaded MGC TOKYO_OPEN strategies: one O5, one O15
- Orchestrator logs APERTURE_MISMATCH warning but proceeds anyway

## Solution

Key `self.orbs` by `(session_label, orb_minutes)` instead of `session_label`.
Create one ORB per unique aperture needed by loaded strategies. Each ORB shares
the same session start time but has a different end time.

## Data Model Changes

- `self.orbs`: `dict[str, LiveORB]` → `dict[tuple[str, int], LiveORB]`
- `ActiveTrade`: add `orb_minutes: int = 5`
- `LiveORB`: add `orb_minutes: int = 5`

## Key Behavior

- 5m, 15m, 30m ORBs share the same start time for a given session
- 5m ORB completes first → triggers 5m strategies → 15m ORB still forming
- Each aperture's ORB has different high/low (more bars = wider range)
- Stop prices, entry prices (E2), G-filters all use the correct aperture's values
- Matches backtesting exactly

## Fail-Safe

- Strategy with missing ORB key → log ERROR, skip (never KeyError crash)
- `orb_minutes` defaults to 5 for backward compatibility
- Startup validation: every loaded strategy must have a matching ORB

## Blast Radius

Contained to `execution_engine.py` (7 touch points) + test updates (4 assertions).
No changes to orchestrator, pipeline, portfolio, or DB schema.

## Touch Points

| File | Line | Change |
|------|------|--------|
| execution_engine.py:138 | ActiveTrade | Add orb_minutes field |
| execution_engine.py:74 | LiveORB | Add orb_minutes field |
| execution_engine.py:210 | self.orbs type | dict key → tuple |
| execution_engine.py:274-292 | on_trading_day_start | Multi-aperture loop |
| execution_engine.py:441 | _arm_strategies | Match orb_minutes |
| execution_engine.py:523 | ActiveTrade construction | Set orb_minutes |
| execution_engine.py:562 | _process_confirming | Tuple key lookup |
| execution_engine.py:942 | _check_exits (E3) | Tuple key lookup |
| session_orchestrator.py:100-113 | F23 guard | Remove (no longer needed) |
| test_execution_engine.py:82,98,112,124 | orbs access | Tuple key |
