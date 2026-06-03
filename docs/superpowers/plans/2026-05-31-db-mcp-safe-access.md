# DB MCP Safe Access Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `gold-db` safe and useful for local/phone/remote-agent DB visibility without exposing raw writes or direct live `gold.db` access to GitHub/remote jobs.

**Architecture:** Keep `trading_app/mcp_server.py` as a thin MCP wrapper. Put DB health, freshness, snapshot-manifest, and snapshot-export logic in `trading_app/db_access.py`, using read-only DuckDB connections configured through `pipeline.db_config.configure_connection`. GitHub-style consumers read stamped snapshots under the canonical runtime root, not the live DB file.

**Tech Stack:** Python 3.11+, DuckDB, FastMCP, pytest, ruff.

---

## File Structure

- Create `trading_app/db_access.py`
  - Owns read-only DB health/freshness/policy helpers.
  - Owns approved snapshot manifest validation and snapshot export.
  - Opens DuckDB only with `read_only=True`, `configure_connection(con)`, and explicit `finally: con.close()`.
- Modify `trading_app/mcp_server.py`
  - Adds thin MCP tools: `get_db_health`, `get_db_freshness`, `get_db_snapshot_manifest`, `get_db_access_policy`.
  - Keeps existing template-query and fitness behavior unchanged.
- Create `scripts/tools/export_gold_db_snapshot.py`
  - CLI wrapper around `trading_app.db_access.export_snapshot`.
  - Direct-script import safe via repo-root `sys.path` insertion.
- Modify `tests/test_trading_app/test_mcp_server.py`
  - Tests MCP helper behavior without requiring canonical `gold.db`.
- Create `tests/test_tools/test_export_gold_db_snapshot.py`
  - Tests snapshot export, stale refusal, approved-root enforcement, deprecated scratch DB refusal, and remote-style snapshot readability.
- Create `docs/plans/active/2026-05/2026-05-30-db-mcp-safe-access.md`
  - Durable design record and future write-broker boundary.
- Create `docs/superpowers/plans/2026-05-31-db-mcp-safe-access.md`
  - This implementation-grade plan.
- Update `HANDOFF.md`
  - Compact cross-tool baton only.

---

### Task 1: Add MCP Operational Tests

**Files:**
- Modify: `tests/test_trading_app/test_mcp_server.py`

- [ ] **Step 1: Write failing tests for DB health and access policy**

Add imports:

```python
import json
import duckdb
```

Add MCP helper imports:

```python
from trading_app.mcp_server import (
    _get_db_access_policy,
    _get_db_freshness,
    _get_db_health,
    _get_db_snapshot_manifest,
)
```

Add tests:

```python
def _make_health_db(path):
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE daily_features (trading_day DATE, symbol VARCHAR)")
    con.execute("INSERT INTO daily_features VALUES ('2026-05-29', 'MNQ')")
    con.execute("CREATE TABLE validated_setups (strategy_id VARCHAR, instrument VARCHAR)")
    con.execute("INSERT INTO validated_setups VALUES ('S1', 'MNQ')")
    con.close()


class TestDbOperationalTools:
    def test_db_health_reports_read_only_open_and_horizon(self, tmp_path):
        db_path = tmp_path / "gold.db"
        _make_health_db(db_path)

        result = _get_db_health(db_path=db_path)

        assert result["status"] == "OK"
        assert result["db_path"] == str(db_path)
        assert result["exists"] is True
        assert result["read_only_open_ok"] is True
        assert result["access"]["write_enabled"] is False
        assert result["horizon"]["daily_features"]["max_trading_day"] == "2026-05-29"

    def test_db_health_fails_closed_when_missing(self, tmp_path):
        result = _get_db_health(db_path=tmp_path / "missing.db")

        assert result["status"] == "MISSING"
        assert result["exists"] is False
        assert result["read_only_open_ok"] is False
        assert "missing" in result["open_error"].lower()

    def test_db_access_policy_is_local_read_only_and_write_disabled(self):
        policy = _get_db_access_policy()

        assert policy["default_transport"] == "stdio"
        assert policy["http_enabled"] is False
        assert policy["write_enabled"] is False
        assert policy["raw_sql_writes_enabled"] is False
        assert policy["github_live_db_access"] == "forbidden"
```

- [ ] **Step 2: Write failing tests for freshness and snapshot manifest validation**

```python
def test_db_freshness_reports_missing_tables_without_silent_success(self, tmp_path):
    db_path = tmp_path / "gold.db"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE daily_features (trading_day DATE, symbol VARCHAR)")
    con.execute("INSERT INTO daily_features VALUES ('2026-05-29', 'MNQ')")
    con.close()

    result = _get_db_freshness(db_path=db_path)

    assert result["status"] == "OK"
    assert result["tables"]["daily_features"]["exists"] is True
    assert result["tables"]["daily_features"]["row_count"] == 1
    assert result["tables"]["orb_outcomes"]["exists"] is False


def test_snapshot_manifest_lists_only_valid_approved_manifests(self, tmp_path):
    root = tmp_path / "snapshots"
    good = root / "snap-a"
    good.mkdir(parents=True)
    (good / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "snapshot_id": "snap-a",
                "generated_at_utc": "2026-05-30T00:00:00+00:00",
                "source_db": {"path": "C:/repo/gold.db", "mtime_utc": "2026-05-29T00:00:00+00:00"},
                "tables": {"daily_features": {"row_count": 1}},
                "horizon": {"daily_features": {"max_trading_day": "2026-05-29"}},
            }
        ),
        encoding="utf-8",
    )
    bad = root / "snap-b"
    bad.mkdir()
    (bad / "manifest.json").write_text(json.dumps({"snapshot_id": "snap-b"}), encoding="utf-8")

    result = _get_db_snapshot_manifest(snapshot_root=root)

    assert result["status"] == "OK_WITH_ERRORS"
    assert [snap["snapshot_id"] for snap in result["snapshots"]] == ["snap-a"]
    assert result["errors"][0]["snapshot_id"] == "snap-b"
```

- [ ] **Step 3: Run red test**

Run:

```bash
python -m pytest tests/test_trading_app/test_mcp_server.py -q
```

Expected before implementation: import error or failures because `_get_db_health`, `_get_db_freshness`, `_get_db_snapshot_manifest`, and `_get_db_access_policy` do not exist.

---

### Task 2: Implement Read-Only DB Access Helpers

**Files:**
- Create: `trading_app/db_access.py`
- Modify: `trading_app/mcp_server.py`

- [ ] **Step 1: Create `trading_app/db_access.py` with read-only helpers**

Implement:

```python
APPROVED_SNAPSHOT_TABLES = ("daily_features", "orb_outcomes", "validated_setups", "edge_families")
DEFAULT_SNAPSHOT_ROOT = CANONICAL_RUNTIME_ROOT / "data" / "snapshots" / "gold_db"
DEPRECATED_SCRATCH_DB = Path("C:/db/gold.db")
MANIFEST_VERSION = 1
DEFAULT_MAX_DB_AGE_HOURS = 168
```

Implement `_read_only_connection(db_path)` as a context manager:

```python
@contextmanager
def _read_only_connection(db_path: Path):
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        configure_connection(con)
        yield con
    finally:
        con.close()
```

Implement:

```python
db_access_policy() -> dict
db_freshness(db_path: Path | None = None) -> dict
db_health(db_path: Path | None = None) -> dict
validate_snapshot_manifest(manifest: dict[str, Any]) -> list[str]
snapshot_manifest(snapshot_root: Path | None = None) -> dict
```

Required behavior:
- missing DB returns `status: "MISSING"` and `read_only_open_ok: False`;
- deprecated `C:/db/gold.db` returns blocked status;
- read-open errors return `status: "ERROR"` and include `open_error`;
- all table checks tolerate missing approved tables explicitly;
- access policy states stdio/local/read-only/no raw SQL writes/no live GitHub DB access.

- [ ] **Step 2: Add thin MCP wrappers**

In `trading_app/mcp_server.py`, import:

```python
from trading_app.db_access import db_access_policy, db_freshness, db_health, snapshot_manifest
```

Add core wrappers:

```python
def _get_db_health(db_path: Path | None = None) -> dict:
    return db_health(db_path=db_path)


def _get_db_freshness(db_path: Path | None = None) -> dict:
    return db_freshness(db_path=db_path)


def _get_db_snapshot_manifest(snapshot_root: Path | None = None) -> dict:
    return snapshot_manifest(snapshot_root=snapshot_root)


def _get_db_access_policy() -> dict:
    return db_access_policy()
```

Register MCP tools inside `_build_server()` with no parameters:

```python
@mcp.tool()
def get_db_health() -> dict:
    return _get_db_health()

@mcp.tool()
def get_db_freshness() -> dict:
    return _get_db_freshness()

@mcp.tool()
def get_db_snapshot_manifest() -> dict:
    return _get_db_snapshot_manifest()

@mcp.tool()
def get_db_access_policy() -> dict:
    return _get_db_access_policy()
```

- [ ] **Step 3: Run green test**

Run:

```bash
python -m pytest tests/test_trading_app/test_mcp_server.py -q
```

Expected: all tests in `test_mcp_server.py` pass.

---

### Task 3: Add Snapshot Export CLI and Tests

**Files:**
- Create: `scripts/tools/export_gold_db_snapshot.py`
- Create: `tests/test_tools/test_export_gold_db_snapshot.py`
- Modify: `trading_app/db_access.py`

- [ ] **Step 1: Write failing snapshot export tests**

Create `tests/test_tools/test_export_gold_db_snapshot.py` with:

```python
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from scripts.tools.export_gold_db_snapshot import export_snapshot
```

Test cases:
- writes Parquet and manifest;
- exported Parquet can be read by a new DuckDB connection without opening the source DB;
- refuses output outside approved snapshot root;
- refuses unapproved tables such as `paper_trades`;
- refuses stale DB unless `max_age_hours=None`;
- refuses deprecated `C:/db/gold.db`.

Remote-read acceptance test:

```python
def test_export_snapshot_can_be_consumed_without_source_db(snapshot_db: Path, tmp_path: Path) -> None:
    root = tmp_path / "approved_snapshots"
    now = datetime.fromtimestamp(snapshot_db.stat().st_mtime, tz=UTC)
    manifest = export_snapshot(
        db_path=snapshot_db,
        output_dir=root / "run-remote",
        snapshot_root=root,
        tables=["daily_features"],
        now=now,
    )

    con = duckdb.connect(":memory:")
    try:
        rows = con.execute(
            "SELECT trading_day, symbol FROM read_parquet(?)",
            [manifest["tables"]["daily_features"]["path"]],
        ).fetchall()
    finally:
        con.close()

    assert rows == [(datetime(2026, 5, 29).date(), "MNQ")]
```

- [ ] **Step 2: Implement `export_snapshot`**

In `trading_app/db_access.py`, implement:

```python
export_snapshot(
    *,
    db_path: Path | None = None,
    output_dir: Path | None = None,
    snapshot_root: Path | None = None,
    tables: list[str] | None = None,
    now: datetime | None = None,
    max_age_hours: int | None = DEFAULT_MAX_DB_AGE_HOURS,
) -> dict[str, Any]
```

Required behavior:
- source DB must exist;
- source DB cannot be deprecated scratch DB;
- output dir must resolve under `snapshot_root`;
- selected tables must be subset of approved tables;
- source DB age must be <= `max_age_hours` unless `max_age_hours is None`;
- existing destination directory is replaced;
- exported files are Parquet;
- manifest includes `manifest_version`, `snapshot_id`, `generated_at_utc`, `source_db`, `access_policy`, `tables`, and `horizon`;
- manifest validates before writing.

- [ ] **Step 3: Implement direct CLI wrapper**

Create `scripts/tools/export_gold_db_snapshot.py` with repo-root import setup:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

CLI arguments:
- `--db-path`;
- `--snapshot-root`;
- `--output-dir`;
- `--tables`;
- `--max-age-hours`;
- `--allow-stale`.

Exit behavior:
- success prints manifest JSON and returns `0`;
- failure prints `ERROR: ...` to stderr and returns `1`.

- [ ] **Step 4: Run green snapshot tests**

Run:

```bash
python -m pytest tests/test_tools/test_export_gold_db_snapshot.py -q
```

Expected: all snapshot tests pass.

---

### Task 4: Formalize Design and Handoff

**Files:**
- Create: `docs/plans/active/2026-05/2026-05-30-db-mcp-safe-access.md`
- Create: `docs/superpowers/plans/2026-05-31-db-mcp-safe-access.md`
- Modify: `HANDOFF.md`

- [ ] **Step 1: Add active design note**

The active design note must state:
- `gold-db` remains local/stdin/read-only;
- remote and GitHub workflows use snapshots, not live DB access;
- V1 has no write broker;
- future write broker, if built, is single-writer, named-job only;
- raw SQL, allocation edits, order routing, `paper_trades` mutation, and live state writes are forbidden.

- [ ] **Step 2: Add this formal implementation plan**

Save this file as:

```text
docs/superpowers/plans/2026-05-31-db-mcp-safe-access.md
```

- [ ] **Step 3: Update handoff compactly**

`HANDOFF.md` must mention:
- branch `codex/db-mcp-safe-access`;
- PR #327;
- implemented MCP tools and snapshot exporter;
- no live DB writes or trading-state mutation;
- full drift blocker, if still present, is unrelated shared DB row-integrity state.

---

### Task 5: Verification, Review, Rebase, and PR Update

**Files:**
- No new files unless verification finds a real gap.

- [ ] **Step 1: Run focused tests**

Run:

```bash
python -m pytest tests/test_trading_app/test_mcp_server.py tests/test_tools/test_export_gold_db_snapshot.py -q
```

Expected:

```text
31 passed
```

- [ ] **Step 2: Run ruff checks**

Run:

```bash
python -m ruff check trading_app/db_access.py trading_app/mcp_server.py scripts/tools/export_gold_db_snapshot.py tests/test_trading_app/test_mcp_server.py tests/test_tools/test_export_gold_db_snapshot.py
python -m ruff format --check trading_app/db_access.py trading_app/mcp_server.py scripts/tools/export_gold_db_snapshot.py tests/test_trading_app/test_mcp_server.py tests/test_tools/test_export_gold_db_snapshot.py
```

Expected:

```text
All checks passed!
5 files already formatted
```

- [ ] **Step 3: Run CLI smoke**

Run:

```bash
python scripts/tools/export_gold_db_snapshot.py --help
```

Expected:
- command exits `0`;
- output lists `--db-path`, `--snapshot-root`, `--output-dir`, `--tables`, `--max-age-hours`, and `--allow-stale`.

- [ ] **Step 4: Run targeted drift checks**

Run:

```powershell
@'
from pipeline import check_drift
checks = [
    ("trading_app connection leaks", lambda: check_drift.check_trading_app_connection_leaks(check_drift.TRADING_APP_DIR)),
    ("db config usage", check_drift.check_db_config_usage),
]
failed = False
for name, fn in checks:
    errors = fn()
    print(f"{name}: {'PASS' if not errors else 'FAIL'}")
    for err in errors:
        print(err)
    failed = failed or bool(errors)
raise SystemExit(1 if failed else 0)
'@ | python -
```

Expected:

```text
trading_app connection leaks: PASS
db config usage: PASS
```

- [ ] **Step 5: Run full drift and document exact result**

Run:

```bash
python pipeline/check_drift.py --quiet
```

Expected acceptable outcomes:
- full pass; or
- only the known unrelated shared `gold.db` violation:

```text
FAIL: Daily features row integrity (one row per aperture per trading_day x symbol) (count=1)
```

If any new failure names appear, stop and fix before pushing.

- [ ] **Step 6: Rebase and push**

Run:

```bash
git fetch origin
git rebase origin/main
python -m pytest tests/test_trading_app/test_mcp_server.py tests/test_tools/test_export_gold_db_snapshot.py -q
git push --force-with-lease
```

Expected:
- branch is based on latest `origin/main`;
- tests still pass after rebase;
- PR #327 updates.

- [ ] **Step 7: Update PR body**

Update PR #327 body to include:
- link to `docs/superpowers/plans/2026-05-31-db-mcp-safe-access.md`;
- implemented behavior;
- non-goals;
- verification evidence;
- full drift result and unrelated DB blocker if still present.

- [ ] **Step 8: Check PR state**

Run:

```bash
gh pr view 327 --repo jbot-bit/canompx3 --json number,url,headRefOid,mergeStateStatus,statusCheckRollup
```

Expected:
- report head SHA;
- if checks are pending, say pending;
- if checks fail, summarize failing job;
- if checks pass and merge state is clean/mergeable, merge according to current repo policy.

---

## Self-Review

- Spec coverage: all requested items are represented by Tasks 1-5.
- Placeholder scan: no placeholder markers or undefined future code paths remain.
- Type consistency: MCP helper names, test imports, and `db_access.py` function names match across tasks.
- Non-goals preserved: no MotherDuck, Quack, DuckLake, recurring data, live feed purchase, live DB write, allocation edit, order route, or `paper_trades` mutation.
