#!/usr/bin/env python3
"""
Pipeline Dashboard — generates a self-contained HTML report.

Panels:
  1. Ingestion Progress (checkpoints + file inventory)
  2. Database Metrics (bars_1m, bars_5m counts, date range)
  3. Data Quality (OHLCV sanity, gap analysis)
  4. Contract History (roll dates, volume by contract)
  5. Guardrails Status (drift checks, test results)
  6. Development Roadmap (from ROADMAP.md)
  7. System Info (Python, DB path, disk)

Usage:
    python pipeline/dashboard.py
    python pipeline/dashboard.py --output report.html
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import duckdb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.paths import GOLD_DB_PATH, DAILY_DBN_DIR

PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
ROADMAP_PATH = PROJECT_ROOT / "ROADMAP.md"


# =============================================================================
# DATA COLLECTORS
# =============================================================================

def collect_db_metrics(db_path: Path) -> dict:
    """Query gold.db for row counts, date ranges, DB size."""
    result = {
        "exists": db_path.exists(),
        "size_mb": 0,
        "bars_1m_count": 0,
        "bars_5m_count": 0,
        "bars_1m_min_date": None,
        "bars_1m_max_date": None,
        "bars_5m_min_date": None,
        "bars_5m_max_date": None,
        "symbols": [],
        "tables": [],
    }

    if not db_path.exists():
        return result

    result["size_mb"] = round(db_path.stat().st_size / (1024 * 1024), 2)

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()
        result["tables"] = [t[0] for t in tables]

        if "bars_1m" in result["tables"]:
            result["bars_1m_count"] = con.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
            if result["bars_1m_count"] > 0:
                dr = con.execute("SELECT MIN(DATE(ts_utc)), MAX(DATE(ts_utc)) FROM bars_1m").fetchone()
                result["bars_1m_min_date"] = str(dr[0])
                result["bars_1m_max_date"] = str(dr[1])
                result["symbols"] = [
                    r[0] for r in con.execute("SELECT DISTINCT symbol FROM bars_1m").fetchall()
                ]

        if "bars_5m" in result["tables"]:
            result["bars_5m_count"] = con.execute("SELECT COUNT(*) FROM bars_5m").fetchone()[0]
            if result["bars_5m_count"] > 0:
                dr = con.execute("SELECT MIN(DATE(ts_utc)), MAX(DATE(ts_utc)) FROM bars_5m").fetchone()
                result["bars_5m_min_date"] = str(dr[0])
                result["bars_5m_max_date"] = str(dr[1])
    finally:
        con.close()

    return result


def collect_checkpoint_progress(cp_dir: Path) -> dict:
    """Parse JSONL checkpoint files for ingestion progress."""
    result = {
        "checkpoint_files": 0,
        "chunks_done": 0,
        "chunks_failed": 0,
        "chunks_in_progress": 0,
        "total_rows_written": 0,
        "earliest_chunk": None,
        "latest_chunk": None,
    }

    if not cp_dir.exists():
        return result

    for cp_file in cp_dir.glob("checkpoint_*.jsonl"):
        result["checkpoint_files"] += 1
        seen = {}  # key -> latest record

        with open(cp_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (record["chunk_start"], record["chunk_end"])
                if key not in seen or record["attempt_id"] >= seen[key]["attempt_id"]:
                    seen[key] = record

        for key, rec in seen.items():
            if rec["status"] == "done":
                result["chunks_done"] += 1
                result["total_rows_written"] += rec.get("rows_written", 0)
            elif rec["status"] == "failed":
                result["chunks_failed"] += 1
            elif rec["status"] == "in_progress":
                result["chunks_in_progress"] += 1

            if result["earliest_chunk"] is None or rec["chunk_start"] < result["earliest_chunk"]:
                result["earliest_chunk"] = rec["chunk_start"]
            if result["latest_chunk"] is None or rec["chunk_end"] > result["latest_chunk"]:
                result["latest_chunk"] = rec["chunk_end"]

    return result


def collect_file_inventory(dbn_dir: Path) -> dict:
    """Scan .dbn.zst files in the data directory."""
    result = {
        "dir_exists": dbn_dir.exists(),
        "total_files": 0,
        "total_size_mb": 0,
        "first_file": None,
        "last_file": None,
    }

    if not dbn_dir.exists():
        return result

    files = sorted(dbn_dir.glob("glbx-mdp3-*.ohlcv-1m.dbn.zst"))
    result["total_files"] = len(files)

    if files:
        result["first_file"] = files[0].name
        result["last_file"] = files[-1].name
        result["total_size_mb"] = round(
            sum(f.stat().st_size for f in files) / (1024 * 1024), 1
        )

    return result


def collect_guardrail_status() -> dict:
    """Run drift check and collect test results."""
    result = {
        "drift_passed": None,
        "drift_output": "",
        "tests_passed": None,
        "tests_output": "",
        "test_count": 0,
        "test_failures": 0,
    }

    # Run drift check
    try:
        proc = subprocess.run(
            [sys.executable, "pipeline/check_drift.py"],
            capture_output=True, text=True, timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        result["drift_passed"] = proc.returncode == 0
        result["drift_output"] = proc.stdout
    except Exception as e:
        result["drift_output"] = str(e)

    # Run tests
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=no", "-q"],
            capture_output=True, text=True, timeout=60,
            cwd=str(PROJECT_ROOT),
        )
        result["tests_passed"] = proc.returncode == 0
        result["tests_output"] = proc.stdout
        # Parse test count from output like "69 passed"
        for line in proc.stdout.splitlines():
            if "passed" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "passed" and i > 0:
                        try:
                            result["test_count"] = int(parts[i - 1])
                        except ValueError:
                            pass
                    if p == "failed" and i > 0:
                        try:
                            result["test_failures"] = int(parts[i - 1])
                        except ValueError:
                            pass
    except Exception as e:
        result["tests_output"] = str(e)

    return result


def collect_contract_history(db_path: Path) -> list[dict]:
    """Get contract usage history from the database."""
    if not db_path.exists():
        return []

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute("""
            SELECT source_symbol,
                   COUNT(*) as bar_count,
                   MIN(DATE(ts_utc)) as first_date,
                   MAX(DATE(ts_utc)) as last_date,
                   SUM(volume) as total_volume
            FROM bars_1m
            WHERE symbol = 'MGC'
            GROUP BY source_symbol
            ORDER BY first_date
        """).fetchall()
    finally:
        con.close()

    return [
        {
            "contract": r[0],
            "bars": r[1],
            "first_date": str(r[2]),
            "last_date": str(r[3]),
            "volume": r[4],
        }
        for r in rows
    ]


def collect_data_quality(db_path: Path) -> dict:
    """Run data quality checks on bars_1m."""
    result = {
        "has_data": False,
        "bars_per_day_avg": 0,
        "bars_per_day_min": 0,
        "bars_per_day_max": 0,
        "gap_days": 0,
        "ohlcv_issues": 0,
        "duplicate_count": 0,
        "null_source_count": 0,
    }

    if not db_path.exists():
        return result

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        count = con.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
        if count == 0:
            return result
        result["has_data"] = True

        # Bars per day stats
        bpd = con.execute("""
            SELECT AVG(cnt), MIN(cnt), MAX(cnt) FROM (
                SELECT DATE(ts_utc) as d, COUNT(*) as cnt
                FROM bars_1m GROUP BY d
            )
        """).fetchone()
        result["bars_per_day_avg"] = round(bpd[0], 1) if bpd[0] else 0
        result["bars_per_day_min"] = int(bpd[1]) if bpd[1] else 0
        result["bars_per_day_max"] = int(bpd[2]) if bpd[2] else 0

        # Gap days (weekdays with no data)
        date_range = con.execute(
            "SELECT MIN(DATE(ts_utc)), MAX(DATE(ts_utc)) FROM bars_1m"
        ).fetchone()
        actual_days = con.execute(
            "SELECT COUNT(DISTINCT DATE(ts_utc)) FROM bars_1m"
        ).fetchone()[0]
        if date_range[0] and date_range[1]:
            total_span = (date_range[1] - date_range[0]).days + 1
            # Rough weekday estimate: ~5/7 of span
            expected_weekdays = int(total_span * 5 / 7)
            result["gap_days"] = max(0, expected_weekdays - actual_days)

        # OHLCV sanity: high < low
        result["ohlcv_issues"] = con.execute(
            "SELECT COUNT(*) FROM bars_1m WHERE high < low"
        ).fetchone()[0]

        # Duplicates
        result["duplicate_count"] = con.execute("""
            SELECT COUNT(*) FROM (
                SELECT symbol, ts_utc FROM bars_1m
                GROUP BY symbol, ts_utc HAVING COUNT(*) > 1
            )
        """).fetchone()[0]

        # NULL source_symbol
        result["null_source_count"] = con.execute(
            "SELECT COUNT(*) FROM bars_1m WHERE source_symbol IS NULL"
        ).fetchone()[0]
    finally:
        con.close()

    return result


def collect_roadmap_status(roadmap_path: Path) -> list[dict]:
    """Parse ROADMAP.md for phase checklist."""
    phases = []
    if not roadmap_path.exists():
        return phases

    content = roadmap_path.read_text(encoding="utf-8")
    current_phase = None

    for line in content.splitlines():
        if line.startswith("## Phase"):
            current_phase = {"name": line.lstrip("# ").strip(), "items": [], "done": False}
            phases.append(current_phase)
        elif current_phase and line.startswith("- "):
            item = line[2:].strip()
            done = item.startswith("[x]") or item.startswith("[X]")
            current_phase["items"].append({"text": item, "done": done})

    return phases


# =============================================================================
# HTML RENDERERS
# =============================================================================

def status_badge(ok: bool | None, text: str = "") -> str:
    if ok is None:
        return f'<span class="badge badge-unknown">{text or "UNKNOWN"}</span>'
    if ok:
        return f'<span class="badge badge-ok">{text or "PASS"}</span>'
    return f'<span class="badge badge-fail">{text or "FAIL"}</span>'


def render_ingestion_panel(cp: dict, inv: dict, db: dict) -> str:
    total_files = inv["total_files"]
    # Estimate processed files from checkpoint progress
    pct = 0
    if total_files > 0 and db["bars_1m_count"] > 0:
        # ~1400 bars/day, ~1 file/day
        estimated_days = db["bars_1m_count"] / 1400
        pct = min(100, round(estimated_days / total_files * 100, 1))

    return f"""
    <div class="panel">
      <h2>Ingestion Progress</h2>
      <div class="progress-bar-container">
        <div class="progress-bar" style="width:{pct}%">{pct}%</div>
      </div>
      <table>
        <tr><td>DBN files in directory</td><td><strong>{total_files:,}</strong></td></tr>
        <tr><td>DBN total size</td><td>{inv['total_size_mb']:,.1f} MB</td></tr>
        <tr><td>First file</td><td>{inv['first_file'] or 'N/A'}</td></tr>
        <tr><td>Last file</td><td>{inv['last_file'] or 'N/A'}</td></tr>
        <tr><td>Checkpoint files</td><td>{cp['checkpoint_files']}</td></tr>
        <tr><td>Chunks done</td><td>{cp['chunks_done']}</td></tr>
        <tr><td>Chunks failed</td><td>{cp['chunks_failed']}</td></tr>
        <tr><td>Rows written (checkpoints)</td><td>{cp['total_rows_written']:,}</td></tr>
        <tr><td>Database rows (bars_1m)</td><td><strong>{db['bars_1m_count']:,}</strong></td></tr>
      </table>
    </div>
    """


def render_db_panel(db: dict) -> str:
    exists_badge = status_badge(db["exists"], "EXISTS" if db["exists"] else "MISSING")
    return f"""
    <div class="panel">
      <h2>Database Metrics</h2>
      <table>
        <tr><td>Status</td><td>{exists_badge}</td></tr>
        <tr><td>Size</td><td>{db['size_mb']} MB</td></tr>
        <tr><td>Tables</td><td>{', '.join(db['tables']) or 'None'}</td></tr>
        <tr><td>bars_1m rows</td><td><strong>{db['bars_1m_count']:,}</strong></td></tr>
        <tr><td>bars_1m range</td><td>{db['bars_1m_min_date'] or 'N/A'} &rarr; {db['bars_1m_max_date'] or 'N/A'}</td></tr>
        <tr><td>bars_5m rows</td><td><strong>{db['bars_5m_count']:,}</strong></td></tr>
        <tr><td>bars_5m range</td><td>{db['bars_5m_min_date'] or 'N/A'} &rarr; {db['bars_5m_max_date'] or 'N/A'}</td></tr>
        <tr><td>Symbols</td><td>{', '.join(db['symbols']) or 'None'}</td></tr>
      </table>
    </div>
    """


def render_quality_panel(quality: dict) -> str:
    if not quality["has_data"]:
        return '<div class="panel"><h2>Data Quality</h2><p>No data in database.</p></div>'

    issues_badge = status_badge(
        quality["ohlcv_issues"] == 0 and quality["duplicate_count"] == 0 and quality["null_source_count"] == 0,
        "CLEAN" if (quality["ohlcv_issues"] == 0 and quality["duplicate_count"] == 0) else "ISSUES"
    )

    return f"""
    <div class="panel">
      <h2>Data Quality {issues_badge}</h2>
      <table>
        <tr><td>Bars/day average</td><td>{quality['bars_per_day_avg']}</td></tr>
        <tr><td>Bars/day min</td><td>{quality['bars_per_day_min']}</td></tr>
        <tr><td>Bars/day max</td><td>{quality['bars_per_day_max']}</td></tr>
        <tr><td>Estimated gap days</td><td>{quality['gap_days']}</td></tr>
        <tr><td>OHLCV issues (high &lt; low)</td><td>{quality['ohlcv_issues']}</td></tr>
        <tr><td>Duplicate (symbol, ts_utc)</td><td>{quality['duplicate_count']}</td></tr>
        <tr><td>NULL source_symbol</td><td>{quality['null_source_count']}</td></tr>
      </table>
    </div>
    """


def render_contract_panel(contracts: list[dict]) -> str:
    if not contracts:
        return '<div class="panel"><h2>Contract History</h2><p>No contract data.</p></div>'

    rows = ""
    for c in contracts:
        rows += f"""
        <tr>
          <td>{c['contract']}</td>
          <td>{c['bars']:,}</td>
          <td>{c['first_date']}</td>
          <td>{c['last_date']}</td>
          <td>{c['volume']:,}</td>
        </tr>"""

    return f"""
    <div class="panel wide">
      <h2>Contract History ({len(contracts)} contracts)</h2>
      <table>
        <tr><th>Contract</th><th>Bars</th><th>First Date</th><th>Last Date</th><th>Total Volume</th></tr>
        {rows}
      </table>
    </div>
    """


def render_guardrails_panel(g: dict) -> str:
    drift_badge = status_badge(g["drift_passed"])
    test_badge = status_badge(g["tests_passed"], f"{g['test_count']} passed" if g["tests_passed"] else "FAIL")

    # Parse drift output for individual checks
    drift_lines = ""
    for line in g["drift_output"].splitlines():
        if "PASSED" in line:
            drift_lines += f'<div class="check-ok">{line.strip()}</div>'
        elif "FAILED" in line:
            drift_lines += f'<div class="check-fail">{line.strip()}</div>'

    return f"""
    <div class="panel">
      <h2>Guardrails</h2>
      <table>
        <tr><td>Drift Detection</td><td>{drift_badge}</td></tr>
        <tr><td>Test Suite</td><td>{test_badge}</td></tr>
      </table>
      <details>
        <summary>Drift check details</summary>
        <div class="details-content">{drift_lines or 'No output'}</div>
      </details>
    </div>
    """


def render_roadmap_panel(phases: list[dict]) -> str:
    if not phases:
        return '<div class="panel"><h2>Development Roadmap</h2><p>ROADMAP.md not found.</p></div>'

    items_html = ""
    for phase in phases:
        done_count = sum(1 for i in phase["items"] if i["done"])
        total = len(phase["items"])
        items_html += f'<h3>{phase["name"]} ({done_count}/{total})</h3><ul>'
        for item in phase["items"]:
            check = "&#x2705;" if item["done"] else "&#x2B1C;"
            items_html += f'<li>{check} {item["text"]}</li>'
        items_html += "</ul>"

    return f"""
    <div class="panel">
      <h2>Development Roadmap</h2>
      {items_html}
    </div>
    """


def render_system_panel() -> str:
    return f"""
    <div class="panel">
      <h2>System Info</h2>
      <table>
        <tr><td>Python</td><td>{sys.version.split()[0]}</td></tr>
        <tr><td>Platform</td><td>{sys.platform}</td></tr>
        <tr><td>DB path</td><td>{GOLD_DB_PATH}</td></tr>
        <tr><td>Data dir</td><td>{DAILY_DBN_DIR}</td></tr>
        <tr><td>Generated</td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
      </table>
    </div>
    """


# =============================================================================
# FULL DASHBOARD
# =============================================================================

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0d1117; color: #c9d1d9; padding: 20px; }
h1 { color: #58a6ff; margin-bottom: 20px; }
h2 { color: #58a6ff; margin-bottom: 12px; font-size: 1.1em; }
h3 { color: #8b949e; margin: 10px 0 6px; font-size: 0.95em; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
        gap: 16px; }
.panel { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
         padding: 16px; }
.panel.wide { grid-column: 1 / -1; }
table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
td, th { padding: 6px 10px; text-align: left; border-bottom: 1px solid #21262d; }
th { color: #8b949e; font-weight: 600; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
         font-size: 0.8em; font-weight: 600; }
.badge-ok { background: #238636; color: #fff; }
.badge-fail { background: #da3633; color: #fff; }
.badge-unknown { background: #6e7681; color: #fff; }
.progress-bar-container { background: #21262d; border-radius: 8px; height: 24px;
                          margin-bottom: 12px; overflow: hidden; }
.progress-bar { background: linear-gradient(90deg, #238636, #2ea043);
                height: 100%; line-height: 24px; text-align: center;
                color: #fff; font-size: 0.8em; font-weight: 600;
                min-width: 40px; transition: width 0.3s; }
details { margin-top: 10px; }
summary { cursor: pointer; color: #58a6ff; font-size: 0.85em; }
.details-content { margin-top: 8px; font-size: 0.8em; font-family: monospace; }
.check-ok { color: #3fb950; }
.check-fail { color: #f85149; }
ul { list-style: none; padding-left: 8px; font-size: 0.9em; }
li { padding: 2px 0; }
.footer { margin-top: 20px; text-align: center; color: #484f58; font-size: 0.8em; }
"""


def render_dashboard(db: dict, cp: dict, inv: dict, quality: dict,
                     contracts: list, guardrails: dict, roadmap: list) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MGC Pipeline Dashboard</title>
  <style>{CSS}</style>
</head>
<body>
  <h1>MGC Pipeline Dashboard</h1>
  <div class="grid">
    {render_ingestion_panel(cp, inv, db)}
    {render_db_panel(db)}
    {render_quality_panel(quality)}
    {render_guardrails_panel(guardrails)}
    {render_contract_panel(contracts)}
    {render_roadmap_panel(roadmap)}
    {render_system_panel()}
  </div>
  <div class="footer">
    Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &mdash; MGC Data Pipeline
  </div>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate MGC Pipeline Dashboard")
    parser.add_argument("--output", type=str, default="dashboard.html",
                        help="Output HTML file (default: dashboard.html)")
    args = parser.parse_args()

    print("Collecting data for dashboard...")

    db = collect_db_metrics(GOLD_DB_PATH)
    print(f"  DB metrics: {db['bars_1m_count']:,} bars_1m rows")

    cp = collect_checkpoint_progress(CHECKPOINT_DIR)
    print(f"  Checkpoints: {cp['chunks_done']} done, {cp['chunks_failed']} failed")

    inv = collect_file_inventory(DAILY_DBN_DIR)
    print(f"  File inventory: {inv['total_files']} files")

    quality = collect_data_quality(GOLD_DB_PATH)
    print(f"  Data quality: {'clean' if quality.get('ohlcv_issues', 0) == 0 else 'issues'}")

    contracts = collect_contract_history(GOLD_DB_PATH)
    print(f"  Contracts: {len(contracts)}")

    guardrails = collect_guardrail_status()
    print(f"  Drift: {'PASS' if guardrails['drift_passed'] else 'FAIL'}")
    print(f"  Tests: {'PASS' if guardrails['tests_passed'] else 'FAIL'}")

    roadmap = collect_roadmap_status(ROADMAP_PATH)
    print(f"  Roadmap phases: {len(roadmap)}")

    html = render_dashboard(db, cp, inv, quality, contracts, guardrails, roadmap)

    output_path = PROJECT_ROOT / args.output
    output_path.write_text(html, encoding="utf-8")
    print(f"\nDashboard written to: {output_path}")
    print(f"  Size: {len(html):,} bytes")


if __name__ == "__main__":
    main()
