# Stage: gold-db MCP `get_db_health` fail-closed on empty/husk DB

task: Make gold-db MCP `get_db_health` fail-closed (DEGRADED/EMPTY) when canonical tables are absent/empty, and flag worktree-husk DB resolution. NOT YET STARTED — design-first.
mode: DESIGN

## Scope Lock
- trading_app/mcp_server.py  (the get_db_health implementation)
- tests/test_trading_app/  (companion test — exact file TBD at design time)

## Blast Radius
- `trading_app/mcp_server.py` — modifies the `get_db_health` return contract: add a
  status tier (DEGRADED/EMPTY) when `daily_features`/`orb_outcomes`/`validated_setups`
  report `exists:false` or `row_count:0`. Read-only MCP surface; no DB writes, no
  schema change, no capital path. Callers: any agent/remote consumer reading
  health JSON — they currently see `status:"OK"` on an empty husk (the bug).
- Worktree-husk detection: `get_db_health` should additionally flag when
  `size_bytes` is tiny (~12 KB) and rows are 0, naming the expected canonical
  path `C:\Users\joshd\canompx3\gold.db`. Reads `pipeline.paths.GOLD_DB_PATH`
  (canonical) — no path-resolution change, just a diagnostic.
- Reads: gold.db (read-only health probe); Writes: none.

## Why (evidence, 2026-06-11)
Live `get_db_health` from this worktree returned:
- `db_path` = the WORKTREE husk (`…canompx3-wt-06Thu11-20261023\gold.db`), `size_bytes: 12288`
- `horizon`: daily_features / orb_outcomes / validated_setups all `exists:false, row_count:0`
- `status: "OK"`  ← FALSE-PASS: reports OK while every canonical table is empty.

This caused `get_strategy_fitness` to fail with `Table validated_setups does not exist`.
Two defects:
  A. status:OK on empty DB = fail-OPEN (the must-fix direction per
     `feedback_capital_guard_fail_direction_matters`).
  B. per-worktree DB resolution silently lands on an empty husk; health doesn't say so.

## Fail-direction note
This is a fail-OPEN bug (false-PASS), so it is the must-fix severity tier. The fix
flips it to fail-closed: an empty/husk DB must NOT report OK.

## Next (fresh context)
1. Read `trading_app/mcp_server.py` get_db_health; confirm the status-assignment line.
2. Design the DEGRADED/EMPTY tier + husk diagnostic; present 4-point proposal.
3. Implement + companion test (inject empty-DB → assert status != OK).
4. drift + tests + self-review.

## Related
- `docs/audit/results/2026-06-11-qwen-overlay-claims-reaudit.md` — the task that surfaced this.
- `memory/feedback_capital_guard_fail_direction_matters_2026_06_07.md` — false-PASS = must-fix.
