# Stage: gold-db MCP `get_db_health` fail-closed on empty/husk DB

task: Make gold-db MCP `get_db_health` fail-closed (DEGRADED/EMPTY) when canonical tables are absent/empty, and flag worktree-husk DB resolution. DONE (2026-06-11).
mode: CLOSED

## Verification (2026-06-11) — DONE
- Targeted: `pytest test_mcp_server.py -k "db_health or db_freshness"` → 6/6 PASS
  (incl. new test_db_health_reports_empty_on_husk_db, _degraded_on_partial_core,
  _ok_when_core_present_even_if_derived_empty = GAP-1 guard, and the renamed
  db_freshness DEGRADED test).
- Full file: `pytest test_mcp_server.py -q` → 29/29 PASS, no regressions.
- Live end-to-end: real canonical DB (35,496 daily_features / 8.95M orb_outcomes)
  → status OK; hermetic empty husk → EMPTY + husk_suspected + expected_canonical_db,
  read_only_open_ok True (opened fine, just empty).
- Drift: `check_drift.py` (DUCKDB_PATH=real DB) → 187 passed, 0 violations. The ~23
  husk-induced false DB violations did NOT reappear under a real DB (the target).
- Dead-code sweep: _classify_table_health referenced by both db_health + db_freshness;
  CANONICAL_CORE_TABLES / HUSK_SIZE_BYTES_HINT / husk_suspected all referenced.
- Mitigation: worktree husk gold.db (12 KB) deleted → resolver falls through to shared
  7.6 GB canonical DB.
- Scope/branch hygiene: production edit done on feature/mcp-db-health-fail-closed
  (NOT the session/* branch — branch-context guard sanctioned, no override left in
  settings). Follow-up resolver fix filed as paths_resolver_skips_husk_sized_gold_db.

## Scope Lock
- trading_app/db_access.py
- tests/test_trading_app/test_mcp_server.py
- docs/runtime/stages/2026-06-11-mcp-db-health-fail-closed.md
- docs/runtime/action-queue.yaml

Scope notes: db_access.py holds db_health / db_freshness (mcp_server.py:214/219
delegate here); the companion tests live in TestDbOperationalTools.

## Chosen design (2026-06-11, after adversarial pass)
Status tiers key on a CANONICAL-CORE subset = `daily_features` + `orb_outcomes`
(both in APPROVED_SNAPSHOT_TABLES), NOT all four tables. `validated_setups` /
`edge_families` are research-derived and legitimately empty on a real DB —
gating OK on them would falsely mark a healthy DB DEGRADED (GAP 1).
- EMPTY (husk): BOTH core tables `exists:false`. Adds `husk_suspected: true`,
  `expected_canonical_db` (= pipeline.paths.GOLD_DB_PATH), `size_bytes`.
- DEGRADED: core partially present (one missing, or present with row_count==0).
- OK: both core tables exist with row_count > 0.
RESOLVED CONTRADICTION (2026-06-11): status is gated SOLELY by canonical-core.
Research-derived tables (validated_setups/edge_families) being absent/empty NEVER
changes the status word — a healthy-core DB with unbuilt derived tables is NORMAL
(the GAP-1 intent). The plan's DEGRADED bullet originally also said "OR a derived
table empty while core healthy", which contradicts its own OK bullet AND the
GAP-1 regression test (which asserts OK when derived tables are absent). The test
is authoritative; the implementation follows it. Derived state is still surfaced
in horizon/tables for the caller, it just doesn't gate the status.
A shared pure helper `_classify_table_health(horizon)` is the single source of
truth for BOTH db_health and db_freshness (no re-encoded logic — rigor §4).
db_freshness gets the same classification; its misnamed test
(`..._without_silent_success`, which asserted OK on a core-partial DB) is fixed
to assert DEGRADED.

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
