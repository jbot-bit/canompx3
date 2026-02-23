# MCP Data Exploration Tool — Design

**Date:** 2026-02-23
**Status:** Approved

## Problem

The existing MCP server (`trading_app/mcp_server.py`) has 13 query templates but they mostly operate on `validated_setups`. Researchers need ad-hoc queries against raw `orb_outcomes` + `daily_features` for stats, comparisons, and splits — without writing a full research script each time.

## Solution

Wire up the existing MCP server and add 5 new query templates for raw outcomes analysis.

## Configuration

Add `gold-db` MCP server to `.claude/settings.json`:
```json
{
  "mcpServers": {
    "gold-db": {
      "command": "python",
      "args": ["trading_app/mcp_server.py"],
      "cwd": "<project_root>"
    }
  }
}
```

## New Query Templates

All use SAFE_JOIN (`orb_outcomes JOIN daily_features ON trading_day + symbol + orb_minutes`), read-only, parameterized.

### 1. OUTCOMES_STATS
Raw outcomes stats for any instrument/session/entry_model/filter slice.
Returns: N, win_rate, avg_pnl_r (ExpR), sharpe, max_drawdown, avg_mae, avg_mfe.

### 2. ENTRY_MODEL_COMPARE
Side-by-side E0 vs E1 vs E3 for same instrument + session.
Returns: entry_model, N, win_rate, avg_pnl_r, sharpe per model.

### 3. DOW_BREAKDOWN
Day-of-week performance splits.
Returns: day_of_week, N, win_rate, avg_pnl_r per DOW.

### 4. DST_SPLIT
DST on vs off performance (mandatory per CLAUDE.md for DST-sensitive sessions).
Returns: dst_regime (on/off), N, win_rate, avg_pnl_r, sharpe per regime.

### 5. FILTER_COMPARE
Compare filter types for same session (e.g., G4 vs G6 vs NO_FILTER).
Returns: filter_applied, N, win_rate, avg_pnl_r, sharpe per filter bucket.

## Parameters (all templates)

- `instrument` (required): MGC, MNQ, MES, M2K
- `orb_label` (required): 0900, 1000, 1100, etc.
- `entry_model` (optional for most, required for OUTCOMES_STATS): E0, E1, E3
- `rr_target` (optional): 1.0, 1.5, 2.0, 2.5, 3.0
- `confirm_bars` (optional): 1, 2, 3

## Out of Scope

- Interactive REPL
- Visualization/charts
- research/lib/ bridge (add later if templates prove insufficient)
- Write operations
- Feature distribution queries (YAGNI)

## Implementation Steps

1. Add 5 SQL templates to `sql_adapter.py`
2. Add execution methods to `SQLAdapter` class
3. Update `QueryTemplate` enum
4. Wire MCP server in `.claude/settings.json`
5. Add tests for new templates
6. Verify MCP server starts and responds
