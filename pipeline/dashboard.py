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
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import duckdb

# Add project root to path
from pipeline.db_contracts import deployable_validated_relation
from pipeline.paths import DAILY_DBN_DIR, GOLD_DB_PATH

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
        "daily_features_count": 0,
        "bars_1m_min_date": None,
        "bars_1m_max_date": None,
        "bars_5m_min_date": None,
        "bars_5m_max_date": None,
        "daily_features_min_date": None,
        "daily_features_max_date": None,
        "symbols": [],
        "tables": [],
    }

    if not db_path.exists():
        return result

    result["size_mb"] = round(db_path.stat().st_size / (1024 * 1024), 2)

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()
        result["tables"] = [t[0] for t in tables]

        if "bars_1m" in result["tables"]:
            result["bars_1m_count"] = con.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
            if result["bars_1m_count"] > 0:
                dr = con.execute("SELECT MIN(DATE(ts_utc)), MAX(DATE(ts_utc)) FROM bars_1m").fetchone()
                result["bars_1m_min_date"] = str(dr[0])
                result["bars_1m_max_date"] = str(dr[1])
                result["symbols"] = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM bars_1m").fetchall()]

        if "bars_5m" in result["tables"]:
            result["bars_5m_count"] = con.execute("SELECT COUNT(*) FROM bars_5m").fetchone()[0]
            if result["bars_5m_count"] > 0:
                dr = con.execute("SELECT MIN(DATE(ts_utc)), MAX(DATE(ts_utc)) FROM bars_5m").fetchone()
                result["bars_5m_min_date"] = str(dr[0])
                result["bars_5m_max_date"] = str(dr[1])

        if "daily_features" in result["tables"]:
            result["daily_features_count"] = con.execute("SELECT COUNT(*) FROM daily_features").fetchone()[0]
            if result["daily_features_count"] > 0:
                dr = con.execute("SELECT MIN(trading_day), MAX(trading_day) FROM daily_features").fetchone()
                result["daily_features_min_date"] = str(dr[0])
                result["daily_features_max_date"] = str(dr[1])
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

        for _key, rec in seen.items():
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
        result["total_size_mb"] = round(sum(f.stat().st_size for f in files) / (1024 * 1024), 1)

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
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        result["drift_passed"] = proc.returncode == 0
        result["drift_output"] = proc.stdout
    except Exception as e:
        result["drift_output"] = str(e)

    # Run tests
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-x", "-q", "--tb=no"],
            capture_output=True,
            text=True,
            timeout=300,
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


def collect_contract_history(db_path: Path, instrument: str = "MGC") -> list[dict]:
    """Get contract usage history from the database."""
    if not db_path.exists():
        return []

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT source_symbol,
                   COUNT(*) as bar_count,
                   MIN(DATE(ts_utc)) as first_date,
                   MAX(DATE(ts_utc)) as last_date,
                   SUM(volume) as total_volume
            FROM bars_1m
            WHERE symbol = ?
            GROUP BY source_symbol
            ORDER BY first_date
        """,
            [instrument],
        ).fetchall()
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
        date_range = con.execute("SELECT MIN(DATE(ts_utc)), MAX(DATE(ts_utc)) FROM bars_1m").fetchone()
        actual_days = con.execute("SELECT COUNT(DISTINCT DATE(ts_utc)) FROM bars_1m").fetchone()[0]
        if date_range[0] and date_range[1]:
            total_span = (date_range[1] - date_range[0]).days + 1
            # Rough weekday estimate: ~5/7 of span
            expected_weekdays = int(total_span * 5 / 7)
            result["gap_days"] = max(0, expected_weekdays - actual_days)

        # OHLCV sanity: high < low
        result["ohlcv_issues"] = con.execute("SELECT COUNT(*) FROM bars_1m WHERE high < low").fetchone()[0]

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


def collect_strategy_metrics(db_path: Path) -> dict:
    """Query validated_setups + experimental_strategies for strategy panel."""
    result = {
        "has_data": False,
        "validated_count": 0,
        "experimental_count": 0,
        "top_strategies": [],
        "session_breakdown": [],
        "best_expr": 0,
        "best_sharpe": 0,
    }

    if not db_path.exists():
        return result

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = [
            t[0]
            for t in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        ]

        if "validated_setups" not in tables:
            return result

        shelf_relation = deployable_validated_relation(con, alias="vs")
        count = con.execute(f"SELECT COUNT(*) FROM {shelf_relation}").fetchone()[0]
        if count == 0:
            return result

        result["has_data"] = True
        result["validated_count"] = count

        if "experimental_strategies" in tables:
            result["experimental_count"] = con.execute("SELECT COUNT(*) FROM experimental_strategies").fetchone()[0]

        # Top 10 by ExpR
        top = con.execute(f"""
            SELECT orb_label, entry_model, confirm_bars, rr_target,
                   filter_type, sample_size, win_rate, expectancy_r,
                   sharpe_ratio, max_drawdown_r
            FROM {shelf_relation}
            ORDER BY expectancy_r DESC
            LIMIT 10
        """).fetchall()
        result["top_strategies"] = [
            {
                "orb": r[0],
                "em": r[1],
                "cb": r[2],
                "rr": r[3],
                "filter": r[4],
                "n": r[5],
                "wr": r[6],
                "expr": r[7],
                "sharpe": r[8],
                "maxdd": r[9],
            }
            for r in top
        ]

        # Session breakdown
        sessions = con.execute(f"""
            SELECT orb_label,
                   COUNT(*) as count,
                   ROUND(AVG(expectancy_r), 3) as avg_expr,
                   ROUND(MAX(expectancy_r), 3) as best_expr,
                   ROUND(AVG(sharpe_ratio), 3) as avg_sharpe
            FROM {shelf_relation}
            GROUP BY orb_label
            ORDER BY count DESC
        """).fetchall()
        result["session_breakdown"] = [
            {"orb": r[0], "count": r[1], "avg_expr": r[2], "best_expr": r[3], "avg_sharpe": r[4]} for r in sessions
        ]

        # Global bests
        bests = con.execute(f"""
            SELECT MAX(expectancy_r), MAX(sharpe_ratio)
            FROM {shelf_relation}
        """).fetchone()
        result["best_expr"] = round(bests[0], 3) if bests[0] else 0
        result["best_sharpe"] = round(bests[1], 3) if bests[1] else 0

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


def _fmt_count(n: int) -> str:
    """Format large numbers with M/K suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def status_badge(ok: bool | None, text: str = "") -> str:
    if ok is None:
        return f'<span class="badge badge-muted">{text or "UNKNOWN"}</span>'
    if ok:
        return f'<span class="badge badge-ok">{text or "PASS"}</span>'
    return f'<span class="badge badge-fail">{text or "FAIL"}</span>'


def render_header(db: dict, strategies: dict | None) -> str:
    bars = _fmt_count(db["bars_1m_count"])
    strat_count = strategies["validated_count"] if strategies and strategies["has_data"] else 0
    db_size = f"{db['size_mb'] / 1024:.1f} GB" if db["size_mb"] >= 1024 else f"{db['size_mb']} MB"
    exp_count = strategies["experimental_count"] if strategies and strategies["has_data"] else 0

    return f"""
    <header class="header">
      <div class="header-inner">
        <div class="header-top">
          <h1>ORB PIPELINE<span class="header-sub">Futures Research &amp; Backtesting</span></h1>
        </div>
        <div class="hero-row">
          <div class="hero-stat">
            <div class="hero-value">{bars}</div>
            <div class="hero-label">1-Min Bars</div>
          </div>
          <div class="hero-stat">
            <div class="hero-value gold">{strat_count}</div>
            <div class="hero-label">Validated</div>
          </div>
          <div class="hero-stat">
            <div class="hero-value">{_fmt_count(exp_count)}</div>
            <div class="hero-label">Experimental</div>
          </div>
          <div class="hero-stat">
            <div class="hero-value">{len(db["symbols"])}</div>
            <div class="hero-label">Instruments</div>
          </div>
          <div class="hero-stat">
            <div class="hero-value">{db_size}</div>
            <div class="hero-label">Database</div>
          </div>
        </div>
      </div>
    </header>"""


def render_status_strip(guardrails: dict, quality: dict, db: dict) -> str:
    drift_cls = "green" if guardrails["drift_passed"] else ("red" if guardrails["drift_passed"] is False else "muted")
    drift_text = (
        "Drift PASS"
        if guardrails["drift_passed"]
        else ("Drift FAIL" if guardrails["drift_passed"] is False else "Drift ?")
    )

    test_cls = "green" if guardrails["tests_passed"] else ("red" if guardrails["tests_passed"] is False else "muted")
    test_text = f"Tests {guardrails['test_count']}" if guardrails["tests_passed"] else "Tests FAIL"

    q_ok = quality["ohlcv_issues"] == 0 and quality["duplicate_count"] == 0 and quality["null_source_count"] == 0
    q_cls = "green" if q_ok else "red"
    q_text = "Quality CLEAN" if q_ok else "Quality ISSUES"

    db_cls = "green" if db["exists"] else "red"
    db_text = "DB Online" if db["exists"] else "DB Missing"

    return f"""
    <div class="strip">
      <div class="strip-inner">
        <span class="strip-item"><span class="dot {drift_cls}"></span>{drift_text}</span>
        <span class="strip-item"><span class="dot {test_cls}"></span>{test_text}</span>
        <span class="strip-item"><span class="dot {q_cls}"></span>{q_text}</span>
        <span class="strip-item"><span class="dot {db_cls}"></span>{db_text}</span>
        <span class="strip-item strip-right">{datetime.now().strftime("%Y-%m-%d %H:%M")}</span>
      </div>
    </div>"""


def render_strategy_panel(strats: dict) -> str:
    if not strats["has_data"]:
        return '<div class="card wide"><div class="card-title">Strategies</div><p class="muted">No validated strategies yet.</p></div>'

    # Top 10 table
    top_rows = ""
    for i, s in enumerate(strats["top_strategies"], 1):
        wr_pct = f"{s['wr'] * 100:.0f}%"
        sharpe = f"{s['sharpe']:+.2f}" if s["sharpe"] else "—"
        maxdd = f"{s['maxdd']:.1f}R" if s["maxdd"] else "—"
        top_rows += f"""
        <tr>
          <td class="muted">{i}</td>
          <td class="session-name">{s["orb"]}</td>
          <td>{s["em"]}</td>
          <td class="num">{s["cb"]}</td>
          <td class="num">{s["rr"]:.1f}</td>
          <td class="filter-name">{s["filter"]}</td>
          <td class="num">{s["n"]:,}</td>
          <td class="num">{wr_pct}</td>
          <td class="num positive">+{s["expr"]:.3f}</td>
          <td class="num">{sharpe}</td>
          <td class="num">{maxdd}</td>
        </tr>"""

    # Session bars
    max_count = max((s["count"] for s in strats["session_breakdown"]), default=1)
    session_bars = ""
    for s in strats["session_breakdown"]:
        w = round((s["count"] / max_count) * 100)
        session_bars += f"""
        <div class="sbar-row">
          <span class="sbar-label">{s["orb"]}</span>
          <div class="sbar-track"><div class="sbar-fill" style="width:{w}%"></div></div>
          <span class="sbar-num">{s["count"]}</span>
          <span class="sbar-expr">+{s["avg_expr"]:.3f}</span>
        </div>"""

    return f"""
    <div class="card wide gold-border">
      <div class="card-title">
        <span>Validated Strategies</span>
        <span class="card-badges">
          <span class="badge badge-gold">{strats["validated_count"]} validated</span>
          <span class="badge badge-ok">ExpR +{strats["best_expr"]:.3f}</span>
          <span class="badge badge-ok">Sharpe +{strats["best_sharpe"]:.3f}</span>
        </span>
      </div>
      <div class="strat-grid">
        <div class="strat-table-wrap">
          <div class="section-label">Top 10 by Expectancy</div>
          <table>
            <tr><th>#</th><th>Session</th><th>EM</th><th>CB</th><th>RR</th><th>Filter</th>
                <th>N</th><th>WR</th><th>ExpR</th><th>Sharpe</th><th>MaxDD</th></tr>
            {top_rows}
          </table>
        </div>
        <div class="strat-bars-wrap">
          <div class="section-label">By Session</div>
          {session_bars}
        </div>
      </div>
    </div>"""


def render_db_panel(db: dict) -> str:
    return f"""
    <div class="card">
      <div class="card-title">Database</div>
      <table>
        <tr><td>bars_1m</td><td class="num"><strong>{db["bars_1m_count"]:,}</strong></td></tr>
        <tr><td>bars_5m</td><td class="num"><strong>{db["bars_5m_count"]:,}</strong></td></tr>
        <tr><td>daily_features</td><td class="num"><strong>{db["daily_features_count"]:,}</strong></td></tr>
        <tr><td>1m range</td><td>{db["bars_1m_min_date"] or "—"} &rarr; {db["bars_1m_max_date"] or "—"}</td></tr>
        <tr><td>features range</td><td>{db["daily_features_min_date"] or "—"} &rarr; {db["daily_features_max_date"] or "—"}</td></tr>
        <tr><td>symbols</td><td>{", ".join(db["symbols"]) or "—"}</td></tr>
        <tr><td>tables</td><td>{len(db["tables"])}</td></tr>
      </table>
    </div>"""


def render_quality_guardrails_panel(quality: dict, guardrails: dict) -> str:
    if not quality["has_data"]:
        return '<div class="card"><div class="card-title">Quality &amp; Guards</div><p class="muted">No data.</p></div>'

    q_ok = quality["ohlcv_issues"] == 0 and quality["duplicate_count"] == 0

    # Drift details
    drift_lines = ""
    for line in guardrails["drift_output"].splitlines():
        if "PASSED" in line:
            drift_lines += f'<div class="chk-ok">{line.strip()}</div>'
        elif "FAILED" in line:
            drift_lines += f'<div class="chk-fail">{line.strip()}</div>'

    return f"""
    <div class="card">
      <div class="card-title">
        <span>Quality &amp; Guards</span>
        {status_badge(q_ok, "CLEAN" if q_ok else "ISSUES")}
      </div>
      <table>
        <tr><td>Bars/day avg</td><td class="num">{quality["bars_per_day_avg"]}</td></tr>
        <tr><td>Bars/day range</td><td class="num">{quality["bars_per_day_min"]} — {quality["bars_per_day_max"]}</td></tr>
        <tr><td>Gap days</td><td class="num">{quality["gap_days"]}</td></tr>
        <tr><td>OHLCV issues</td><td class="num">{quality["ohlcv_issues"]}</td></tr>
        <tr><td>Duplicates</td><td class="num">{quality["duplicate_count"]}</td></tr>
        <tr><td>Drift detection</td><td>{status_badge(guardrails["drift_passed"])}</td></tr>
        <tr><td>Test suite</td><td>{status_badge(guardrails["tests_passed"], f"{guardrails['test_count']} passed" if guardrails["tests_passed"] else "FAIL")}</td></tr>
      </table>
      <details>
        <summary>Drift check details</summary>
        <div class="details-box">{drift_lines or "No output"}</div>
      </details>
    </div>"""


def render_ingestion_panel(cp: dict, inv: dict, db: dict) -> str:
    total_files = inv["total_files"]
    pct = 0
    if total_files > 0 and db["bars_1m_count"] > 0:
        estimated_days = db["bars_1m_count"] / 1400
        pct = min(100, round(estimated_days / total_files * 100, 1))

    return f"""
    <div class="card">
      <div class="card-title">Ingestion</div>
      <div class="pbar-header">
        <span class="muted">Progress</span>
        <span class="gold">{pct}%</span>
      </div>
      <div class="pbar-track"><div class="pbar-fill" style="width:{pct}%"></div></div>
      <table>
        <tr><td>DBN files</td><td class="num"><strong>{total_files:,}</strong></td></tr>
        <tr><td>Compressed</td><td class="num">{inv["total_size_mb"]:,.1f} MB</td></tr>
        <tr><td>Chunks done</td><td class="num">{cp["chunks_done"]:,}</td></tr>
        <tr><td>Chunks failed</td><td class="num">{cp["chunks_failed"]}</td></tr>
        <tr><td>DB rows</td><td class="num"><strong>{db["bars_1m_count"]:,}</strong></td></tr>
      </table>
    </div>"""


def render_system_panel() -> str:
    return f"""
    <div class="card">
      <div class="card-title">System</div>
      <table>
        <tr><td>Python</td><td>{sys.version.split()[0]}</td></tr>
        <tr><td>Platform</td><td>{sys.platform}</td></tr>
        <tr><td>DB path</td><td class="path-cell">{GOLD_DB_PATH}</td></tr>
        <tr><td>Data dir</td><td class="path-cell">{DAILY_DBN_DIR}</td></tr>
        <tr><td>Generated</td><td>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</td></tr>
      </table>
    </div>"""


def render_contract_panel(contracts: list[dict]) -> str:
    if not contracts:
        return ""

    rows = ""
    for c in contracts:
        rows += f"""
        <tr>
          <td class="mono">{c["contract"]}</td>
          <td class="num">{c["bars"]:,}</td>
          <td>{c["first_date"]}</td>
          <td>{c["last_date"]}</td>
          <td class="num">{c["volume"]:,}</td>
        </tr>"""

    return f"""
    <div class="card wide">
      <details>
        <summary class="collapse-title">Contract History ({len(contracts)})</summary>
        <div class="collapse-body">
          <table>
            <tr><th>Contract</th><th>Bars</th><th>First</th><th>Last</th><th>Volume</th></tr>
            {rows}
          </table>
        </div>
      </details>
    </div>"""


def render_roadmap_panel(phases: list[dict]) -> str:
    if not phases:
        return ""

    phase_html = ""
    for phase in phases:
        name = phase["name"]
        is_done = "DONE" in name.upper()
        total = len(phase["items"])
        icon = '<span class="phase-done">&#10003;</span>' if is_done else '<span class="phase-todo">&#9675;</span>'

        items_html = ""
        for item in phase["items"]:
            items_html += f'<div class="roadmap-item">{item["text"]}</div>'

        phase_html += f"""
        <details {"" if not is_done or total == 0 else ""}>
          <summary class="phase-row">{icon} {name}</summary>
          <div class="phase-items">{items_html}</div>
        </details>"""

    return f"""
    <div class="card wide">
      <details>
        <summary class="collapse-title">Development Roadmap ({len(phases)} phases)</summary>
        <div class="collapse-body">{phase_html}</div>
      </details>
    </div>"""


# =============================================================================
# CSS
# =============================================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Syne:wght@400;500;600;700;800&display=swap');

:root {
  --bg:        #07080c;
  --bg-card:   #0f1117;
  --bg-hover:  #151820;
  --border:    #1a1d27;
  --border-g:  #2a2210;
  --gold:      #c9a227;
  --gold-b:    #e8bf3b;
  --gold-d:    #6b5819;
  --gold-glow: rgba(201,162,39,0.07);
  --text:      #d1cdc7;
  --text-dim:  #5e616a;
  --text-b:    #f0ede8;
  --green:     #22c55e;
  --red:       #ef4444;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'JetBrains Mono', 'Cascadia Code', 'Fira Code', monospace;
  background: var(--bg);
  color: var(--text);
  font-size: 13px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}

/* ── Header ──────────────────────────────────────────── */

.header {
  background: linear-gradient(180deg, #0d0e14, var(--bg));
  border-bottom: 1px solid var(--border);
  padding: 28px 32px 24px;
  position: relative;
}
.header::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent 5%, var(--gold-d), var(--gold), var(--gold-b), var(--gold), var(--gold-d), transparent 95%);
}
.header-inner { max-width: 1440px; margin: 0 auto; }
.header-top { display: flex; align-items: baseline; gap: 16px; margin-bottom: 22px; }
.header h1 {
  font-family: 'Syne', sans-serif;
  font-size: 1.5em;
  font-weight: 800;
  color: var(--gold);
  letter-spacing: 3px;
  text-transform: uppercase;
}
.header-sub {
  color: var(--text-dim);
  font-weight: 400;
  font-size: 0.5em;
  letter-spacing: 0;
  text-transform: none;
  margin-left: 14px;
}
.hero-row { display: flex; gap: 40px; }
.hero-stat { text-align: center; }
.hero-value {
  font-family: 'Syne', sans-serif;
  font-size: 2em;
  font-weight: 700;
  color: var(--text-b);
  line-height: 1.1;
}
.hero-value.gold { color: var(--gold-b); }
.hero-label {
  font-size: 0.7em;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-top: 4px;
}

/* ── Status strip ────────────────────────────────────── */

.strip {
  background: var(--bg-card);
  border-bottom: 1px solid var(--border);
  padding: 8px 32px;
  overflow-x: auto;
}
.strip-inner {
  display: flex;
  gap: 20px;
  max-width: 1440px;
  margin: 0 auto;
  align-items: center;
}
.strip-item {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 0.85em;
  white-space: nowrap;
  color: var(--text);
}
.strip-right { margin-left: auto; color: var(--text-dim); }
.dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}
.dot.green  { background: var(--green); box-shadow: 0 0 5px var(--green); }
.dot.red    { background: var(--red);   box-shadow: 0 0 5px var(--red);   }
.dot.muted  { background: var(--text-dim); }

/* ── Layout ──────────────────────────────────────────── */

.main {
  max-width: 1440px;
  margin: 0 auto;
  padding: 20px 32px 48px;
}
.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}

/* ── Cards ───────────────────────────────────────────── */

.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 18px 20px;
  position: relative;
}
.card::after {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 3px; height: 100%;
  background: var(--gold-d);
  border-radius: 6px 0 0 6px;
  opacity: 0.4;
}
.card.wide { grid-column: 1 / -1; }
.card.gold-border::after { background: var(--gold); opacity: 1; }

.card-title {
  font-family: 'Syne', sans-serif;
  font-size: 0.78em;
  font-weight: 600;
  color: var(--gold);
  text-transform: uppercase;
  letter-spacing: 2px;
  margin-bottom: 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
}
.card-badges { display: flex; gap: 6px; flex-wrap: wrap; }

/* ── Tables ──────────────────────────────────────────── */

table { width: 100%; border-collapse: collapse; font-size: 0.92em; }
td, th { padding: 6px 10px; text-align: left; }
th {
  color: var(--text-dim);
  font-weight: 500;
  font-size: 0.75em;
  text-transform: uppercase;
  letter-spacing: 1px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 9px;
}
tr:not(:last-child) td { border-bottom: 1px solid #0c0d12; }
tr:hover td { background: var(--bg-hover); }
td:first-child { color: var(--text-dim); }
td strong { color: var(--text-b); font-weight: 600; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.positive { color: var(--green); font-weight: 600; }
.negative { color: var(--red); }
.muted { color: var(--text-dim); }
.gold { color: var(--gold-b); }
.mono { font-family: inherit; letter-spacing: 0.5px; }
.session-name { color: var(--gold); font-weight: 500; }
.filter-name { font-size: 0.85em; color: var(--text-dim); }
.path-cell { font-size: 0.78em; word-break: break-all; color: var(--text-dim); }
.section-label {
  font-size: 0.72em;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: 10px;
  font-weight: 500;
}

/* ── Badges ──────────────────────────────────────────── */

.badge {
  display: inline-flex;
  align-items: center;
  padding: 3px 9px;
  border-radius: 4px;
  font-size: 0.76em;
  font-weight: 600;
  letter-spacing: 0.3px;
}
.badge-ok   { background: rgba(34,197,94,0.1);  color: var(--green); border: 1px solid rgba(34,197,94,0.18); }
.badge-fail { background: rgba(239,68,68,0.1);  color: var(--red);   border: 1px solid rgba(239,68,68,0.18); }
.badge-muted{ background: rgba(94,97,106,0.1);  color: var(--text-dim); border: 1px solid rgba(94,97,106,0.18); }
.badge-gold { background: var(--gold-glow);      color: var(--gold-b);   border: 1px solid var(--border-g); }

/* ── Progress bar ────────────────────────────────────── */

.pbar-header { display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 0.8em; }
.pbar-track { background: #0c0d12; border-radius: 3px; height: 5px; margin-bottom: 14px; overflow: hidden; }
.pbar-fill { height: 100%; background: linear-gradient(90deg, var(--gold-d), var(--gold), var(--gold-b)); border-radius: 3px; }

/* ── Session bars ────────────────────────────────────── */

.strat-grid { display: grid; grid-template-columns: 1fr 320px; gap: 24px; }
.strat-table-wrap { overflow-x: auto; }
.strat-bars-wrap { padding-top: 2px; }

.sbar-row { display: flex; align-items: center; gap: 8px; padding: 3px 0; }
.sbar-label { width: 130px; font-size: 0.82em; color: var(--text); flex-shrink: 0; }
.sbar-track { flex: 1; height: 14px; background: #0c0d12; border-radius: 2px; overflow: hidden; }
.sbar-fill { height: 100%; background: linear-gradient(90deg, var(--gold-d), var(--gold)); border-radius: 2px; }
.sbar-num { font-size: 0.75em; color: var(--text-dim); min-width: 28px; text-align: right; }
.sbar-expr { font-size: 0.75em; color: var(--green); min-width: 46px; text-align: right; }

/* ── Collapsible sections ────────────────────────────── */

details { margin-top: 10px; }
summary { cursor: pointer; user-select: none; }
summary::-webkit-details-marker { color: var(--gold-d); }
.collapse-title {
  font-family: 'Syne', sans-serif;
  font-size: 0.78em;
  font-weight: 600;
  color: var(--gold);
  text-transform: uppercase;
  letter-spacing: 2px;
  padding: 4px 0;
}
.collapse-title:hover { color: var(--gold-b); }
.collapse-body { margin-top: 12px; }
.details-box {
  margin-top: 8px;
  font-size: 0.82em;
  max-height: 360px;
  overflow-y: auto;
  padding: 10px 12px;
  background: #090a0e;
  border-radius: 4px;
  border: 1px solid var(--border);
}
.chk-ok   { color: var(--green); padding: 1px 0; }
.chk-fail { color: var(--red);   padding: 1px 0; }

/* ── Roadmap ─────────────────────────────────────────── */

.phase-row { padding: 5px 0; font-size: 0.88em; }
.phase-row:hover { color: var(--gold); }
.phase-done { color: var(--green); margin-right: 4px; }
.phase-todo { color: var(--text-dim); margin-right: 4px; }
.phase-items { padding-left: 20px; margin-bottom: 6px; }
.roadmap-item { padding: 2px 0; font-size: 0.82em; color: var(--text-dim); }

/* ── Footer ──────────────────────────────────────────── */

.footer {
  text-align: center;
  padding: 20px 32px;
  color: var(--text-dim);
  font-size: 0.72em;
  letter-spacing: 1px;
  border-top: 1px solid var(--border);
  max-width: 1440px;
  margin: 0 auto;
}

/* ── Scrollbar ───────────────────────────────────────── */

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold-d); }

/* ── Responsive ──────────────────────────────────────── */

@media (max-width: 1100px) {
  .strat-grid { grid-template-columns: 1fr; }
  .strat-bars-wrap { margin-top: 12px; }
}
@media (max-width: 900px) {
  .grid { grid-template-columns: 1fr; }
  .card.wide { grid-column: 1; }
  .hero-row { flex-wrap: wrap; gap: 24px; }
  .header, .main, .strip { padding-left: 16px; padding-right: 16px; }
}
"""


# =============================================================================
# FULL DASHBOARD
# =============================================================================


def render_dashboard(
    db: dict,
    cp: dict,
    inv: dict,
    quality: dict,
    contracts: list,
    guardrails: dict,
    roadmap: list,
    strategies: dict | None = None,
) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ORB Pipeline Dashboard</title>
  <style>{CSS}</style>
</head>
<body>
  {render_header(db, strategies)}
  {render_status_strip(guardrails, quality, db)}
  <div class="main">
    <div class="grid">
      {render_strategy_panel(strategies) if strategies else ""}
      {render_db_panel(db)}
      {render_quality_guardrails_panel(quality, guardrails)}
      {render_ingestion_panel(cp, inv, db)}
      {render_system_panel()}
      {render_contract_panel(contracts)}
      {render_roadmap_panel(roadmap)}
    </div>
  </div>
  <div class="footer">
    ORB PIPELINE &middot; Multi-Instrument Futures Research &middot; {datetime.now().strftime("%Y-%m-%d %H:%M")}
  </div>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate ORB Pipeline Dashboard")
    parser.add_argument(
        "--output", type=str, default="dashboard.html", help="Output HTML file (default: dashboard.html)"
    )
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

    strategies = collect_strategy_metrics(GOLD_DB_PATH)
    print(f"  Strategies: {strategies['validated_count']} validated")

    html = render_dashboard(db, cp, inv, quality, contracts, guardrails, roadmap, strategies)

    output_path = PROJECT_ROOT / args.output
    output_path.write_text(html, encoding="utf-8")
    print(f"\nDashboard written to: {output_path}")
    print(f"  Size: {len(html):,} bytes")


if __name__ == "__main__":
    main()
