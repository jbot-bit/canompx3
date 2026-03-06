#!/usr/bin/env python3
"""
Daily Trade Sheet Generator.

Builds the resolved live portfolio for all active instruments, resolves
today's session times via dst.py, and generates a self-contained HTML
file showing exactly what to trade.

ONLY shows strategies that passed the dollar gate. Nothing else.

Usage:
    python scripts/tools/generate_trade_sheet.py
    python scripts/tools/generate_trade_sheet.py --date 2026-03-04
    python scripts/tools/generate_trade_sheet.py --output my_sheet.html
    python scripts/tools/generate_trade_sheet.py --no-open
"""

import argparse
import os
import sys
import webbrowser
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.asset_configs import get_active_instruments
from pipeline.cost_model import get_cost_spec
from pipeline.dst import SESSION_CATALOG
from pipeline.paths import GOLD_DB_PATH
from trading_app.live_config import (
    LIVE_PORTFOLIO,
    LIVE_MIN_EXPECTANCY_R,
    LIVE_MIN_EXPECTANCY_DOLLARS_MULT,
)
from trading_app.strategy_fitness import compute_fitness


# ── Filter → plain English ────────────────────────────────────────────


def _filter_description(filter_type: str) -> str:
    """Convert filter_type to plain English for the trade sheet."""
    # Exact matches first
    exact = {
        "NO_FILTER": "Any ORB size",
        "VOL_RV12_N20": "Volume >= 1.2x median",
        "DIR_LONG": "LONG ONLY",
        "DIR_SHORT": "SHORT ONLY",
    }
    if filter_type in exact:
        return exact[filter_type]

    ft = filter_type

    # Composite filters: parse components
    parts = []

    # ORB size component
    for g in ["G2", "G3", "G4", "G5", "G6", "G8"]:
        if f"ORB_{g}" in ft or ft.startswith(g):
            pts = g[1:]
            parts.append(f"ORB >= {pts} pts")
            break

    # Break quality composites
    if "CONT" in ft:
        parts.append("continuation only")
    if "FAST5" in ft:
        parts.append("break within 5 min")
    if "FAST10" in ft:
        parts.append("break within 10 min")

    # DOW composites
    if "NOMON" in ft:
        parts.append("skip Monday")
    if "NOFRI" in ft:
        parts.append("skip Friday")
    if "NOTUE" in ft:
        parts.append("skip Tuesday")

    # Volume composite
    if "VOL_RV12_N20" in ft:
        parts.append("vol >= 1.2x")

    if parts:
        return " + ".join(parts)

    return filter_type  # fallback: show raw


def _direction_rule(filter_type: str) -> str:
    """Determine direction constraint from filter_type."""
    if "DIR_LONG" in filter_type:
        return "LONG ONLY"
    if "DIR_SHORT" in filter_type:
        return "SHORT ONLY"
    if "CONT" in filter_type:
        return "CONT"
    return "ANY"


def _parse_aperture(strategy_id: str) -> int:
    """Extract ORB aperture minutes from strategy_id suffix."""
    if "_O30" in strategy_id:
        return 30
    if "_O15" in strategy_id:
        return 15
    return 5


def _exp_dollars_from_row(row: dict, instrument: str) -> float | None:
    """Compute expected dollars per trade from a DB row dict."""
    median_risk_pts = row.get("median_risk_points")
    exp_r = row.get("expectancy_r")
    if median_risk_pts is None or exp_r is None:
        return None
    try:
        spec = get_cost_spec(instrument)
        one_r = median_risk_pts * spec.point_value + spec.total_friction
        return exp_r * one_r
    except Exception:
        return None


def _passes_dollar_gate(row: dict, instrument: str) -> tuple[bool, float | None]:
    """Check if expected $/trade >= LIVE_MIN_EXPECTANCY_DOLLARS_MULT * RT cost.

    Returns (passes, exp_dollars).
    """
    exp_d = _exp_dollars_from_row(row, instrument)
    if exp_d is None:
        return True, None  # skip gate if data missing (fail-open on missing data)
    try:
        spec = get_cost_spec(instrument)
        min_dollars = LIVE_MIN_EXPECTANCY_DOLLARS_MULT * spec.total_friction
        return exp_d >= min_dollars, exp_d
    except Exception:
        return True, exp_d


def _check_fitness(strategy_id: str, db_path: Path) -> str:
    """Quick fitness check. Returns status string."""
    try:
        f = compute_fitness(strategy_id, db_path=db_path)
        return f.fitness_status
    except Exception:
        return "UNKNOWN"


# ── Session time resolution ───────────────────────────────────────────


def _resolve_session_times(trading_day: date) -> dict[str, tuple[int, int]]:
    """Resolve all session start times in Brisbane for a given date."""
    times = {}
    for label, entry in SESSION_CATALOG.items():
        if entry["type"] == "dynamic":
            resolver = entry["resolver"]
            h, m = resolver(trading_day)
            times[label] = (h, m)
    return times


def _format_time(h: int, m: int) -> str:
    """Format (hour, minute) as HH:MM with AM/PM."""
    period = "AM" if h < 12 else "PM"
    display_h = h % 12
    if display_h == 0:
        display_h = 12
    return f"{display_h}:{m:02d} {period}"


def _sort_key(h: int, m: int) -> int:
    """Sort sessions by Brisbane time, starting from 8 AM (trading day start)."""
    # Shift so 8 AM = 0, wrapping midnight sessions to after evening
    shifted = (h * 60 + m - 8 * 60) % (24 * 60)
    return shifted


# ── Data collection ───────────────────────────────────────────────────


def _load_best_by_expr(
    db_path: Path,
    instrument: str,
    orb_label: str,
    entry_model: str,
    filter_type: str,
    min_expectancy_r: float,
) -> dict | None:
    """Load the best locked-RR variant by ExpR.

    Joins family_rr_locks to restrict each family to its SharpeDD-locked RR.
    Falls back to best ExpR if no lock exists (graceful degradation).
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT vs.strategy_id, vs.instrument, vs.orb_label, vs.entry_model,
                   vs.rr_target, vs.confirm_bars, vs.filter_type,
                   vs.expectancy_r, vs.win_rate, vs.sample_size,
                   vs.sharpe_ratio, vs.max_drawdown_r,
                   es.median_risk_points
            FROM validated_setups vs
            LEFT JOIN experimental_strategies es
              ON vs.strategy_id = es.strategy_id
            LEFT JOIN family_rr_locks frl
              ON vs.instrument = frl.instrument
              AND vs.orb_label = frl.orb_label
              AND vs.filter_type = frl.filter_type
              AND vs.entry_model = frl.entry_model
              AND vs.orb_minutes = frl.orb_minutes
              AND vs.confirm_bars = frl.confirm_bars
            WHERE vs.instrument = ?
              AND vs.orb_label = ?
              AND vs.entry_model = ?
              AND vs.filter_type = ?
              AND LOWER(vs.status) = 'active'
              AND vs.expectancy_r >= ?
              AND (frl.locked_rr IS NULL OR vs.rr_target = frl.locked_rr)
            ORDER BY vs.expectancy_r DESC
            LIMIT 1
        """,
            [instrument, orb_label, entry_model, filter_type, min_expectancy_r],
        ).fetchall()

        if not rows:
            return None

        cols = [desc[0] for desc in con.description]
        return dict(zip(cols, rows[0]))
    finally:
        con.close()


def collect_trades(trading_day: date, db_path: Path) -> list[dict]:
    """Resolve best-ExpR variant for each live spec, per instrument.

    Queries validated_setups directly, sorted by ExpR (not Sharpe).
    Applies dollar gate. Only returns cost-positive trades.
    """
    instruments = get_active_instruments()
    trades = []

    for instrument in instruments:
        for spec in LIVE_PORTFOLIO:
            variant = _load_best_by_expr(
                db_path,
                instrument,
                spec.orb_label,
                spec.entry_model,
                spec.filter_type,
                min_expectancy_r=LIVE_MIN_EXPECTANCY_R,
            )
            if variant is None:
                continue

            # Dollar gate
            passes, exp_d = _passes_dollar_gate(variant, instrument)
            if not passes:
                continue

            # Fitness check (regime gate for REGIME tier)
            fitness = _check_fitness(variant["strategy_id"], db_path)
            if spec.tier == "regime" and spec.regime_gate == "high_vol":
                if fitness not in ("FIT",):
                    continue  # gated off

            trades.append(
                {
                    "session": variant["orb_label"],
                    "instrument": instrument,
                    "strategy_id": variant["strategy_id"],
                    "aperture": _parse_aperture(variant["strategy_id"]),
                    "direction": _direction_rule(variant["filter_type"]),
                    "filter_desc": _filter_description(variant["filter_type"]),
                    "filter_type": variant["filter_type"],
                    "rr": variant["rr_target"],
                    "win_rate": variant["win_rate"],
                    "exp_r": variant["expectancy_r"],
                    "exp_dollars": exp_d if exp_d is not None else _exp_dollars_from_row(variant, instrument),
                    "sample_size": variant["sample_size"],
                    "fitness": fitness,
                }
            )

    return trades


# ── HTML generation ───────────────────────────────────────────────────


def _direction_badge(direction: str) -> str:
    """HTML badge for direction constraint."""
    if direction == "LONG ONLY":
        return '<span class="badge badge-long">LONG ONLY</span>'
    if direction == "SHORT ONLY":
        return '<span class="badge badge-short">SHORT ONLY</span>'
    if direction == "CONT":
        return '<span class="badge badge-cont">CONT ONLY</span>'
    return ""


def _fitness_badge(fitness: str) -> str:
    """HTML badge for non-FIT fitness."""
    if fitness == "FIT":
        return ""
    cls = {
        "WATCH": "badge-watch",
        "DECAY": "badge-decay",
        "STALE": "badge-stale",
    }.get(fitness, "badge-decay")
    return f' <span class="badge {cls}">{fitness}</span>'


def generate_html(trades: list[dict], session_times: dict, trading_day: date) -> str:
    """Generate self-contained HTML trade sheet."""

    # Group trades by session
    sessions_used = sorted(set(t["session"] for t in trades), key=lambda s: _sort_key(*session_times.get(s, (0, 0))))

    day_name = trading_day.strftime("%A")
    date_str = trading_day.strftime("%d %b %Y")
    now_str = datetime.now().strftime("%H:%M")

    # Build session cards
    cards_html = ""
    trade_num = 0
    for session in sessions_used:
        h, m = session_times.get(session, (0, 0))
        time_str = _format_time(h, m)
        event = SESSION_CATALOG.get(session, {}).get("event", "")
        session_trades = [t for t in trades if t["session"] == session]

        rows_html = ""
        for t in session_trades:
            trade_num += 1
            exp_d_str = f"${t['exp_dollars']:+.2f}" if t["exp_dollars"] is not None else "n/a"
            dir_badge = _direction_badge(t["direction"])
            fit_badge = _fitness_badge(t["fitness"])

            exp_r_class = "expr-high" if t["exp_r"] >= 0.20 else ""

            rows_html += f"""
            <tr>
                <td class="instrument-cell">{t["instrument"]}</td>
                <td>{t["aperture"]}m</td>
                <td>{dir_badge if dir_badge else "ANY"}</td>
                <td class="filter-cell">{t["filter_desc"]}</td>
                <td>{t["rr"]:.1f} : 1</td>
                <td>{t["win_rate"]:.0%}</td>
                <td class="{exp_r_class}">{t["exp_r"]:+.3f}</td>
                <td class="dollars-cell">{exp_d_str}</td>
                <td>{fit_badge if fit_badge else '<span class="fit-ok">FIT</span>'}</td>
            </tr>"""

        cards_html += f"""
        <div class="session-card">
            <div class="session-header">
                <div class="session-time">{time_str} BRIS</div>
                <div class="session-name">{session}</div>
                <div class="session-event">{event}</div>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Instrument</th>
                        <th>ORB</th>
                        <th>Direction</th>
                        <th>Filter</th>
                        <th>RR</th>
                        <th>WR</th>
                        <th>ExpR</th>
                        <th>Exp$/trade</th>
                        <th>Fitness</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>"""

    # Instrument summary
    instr_counts = {}
    instr_total_exp = {}
    for t in trades:
        instr_counts[t["instrument"]] = instr_counts.get(t["instrument"], 0) + 1
        if t["exp_dollars"] is not None:
            instr_total_exp[t["instrument"]] = instr_total_exp.get(t["instrument"], 0) + t["exp_dollars"]

    summary_html = ""
    for instr in sorted(instr_counts.keys()):
        total = instr_total_exp.get(instr, 0)
        summary_html += f"""
        <div class="summary-card">
            <div class="summary-instrument">{instr}</div>
            <div class="summary-count">{instr_counts[instr]} trades</div>
            <div class="summary-dollars">${total:.2f}/day edge</div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trade Sheet — {day_name} {date_str}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #0d1117;
        color: #e6edf3;
        padding: 20px;
        max-width: 1200px;
        margin: 0 auto;
    }}
    .header {{
        text-align: center;
        margin-bottom: 30px;
        padding: 20px;
        border-bottom: 2px solid #30363d;
    }}
    .header h1 {{
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 5px;
    }}
    .header .subtitle {{
        color: #8b949e;
        font-size: 14px;
    }}
    .header .date {{
        font-size: 20px;
        color: #58a6ff;
        margin-top: 5px;
    }}
    .entry-model-note {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 24px;
        font-size: 14px;
        color: #8b949e;
        text-align: center;
    }}
    .entry-model-note strong {{
        color: #58a6ff;
    }}
    .summary-row {{
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
        flex-wrap: wrap;
    }}
    .summary-card {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        flex: 1;
        min-width: 140px;
        text-align: center;
    }}
    .summary-instrument {{
        font-size: 18px;
        font-weight: 700;
        color: #58a6ff;
    }}
    .summary-count {{
        font-size: 14px;
        color: #8b949e;
        margin-top: 4px;
    }}
    .summary-dollars {{
        font-size: 16px;
        font-weight: 600;
        color: #3fb950;
        margin-top: 4px;
    }}
    .session-card {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        margin-bottom: 16px;
        overflow: hidden;
    }}
    .session-header {{
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 12px 16px;
        background: #1c2128;
        border-bottom: 1px solid #30363d;
    }}
    .session-time {{
        font-size: 22px;
        font-weight: 700;
        color: #f0883e;
        min-width: 100px;
    }}
    .session-name {{
        font-size: 16px;
        font-weight: 600;
        color: #e6edf3;
    }}
    .session-event {{
        font-size: 12px;
        color: #8b949e;
        margin-left: auto;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
    }}
    thead th {{
        text-align: left;
        padding: 8px 12px;
        font-size: 11px;
        text-transform: uppercase;
        color: #8b949e;
        border-bottom: 1px solid #30363d;
        font-weight: 600;
    }}
    tbody td {{
        padding: 10px 12px;
        font-size: 14px;
        border-bottom: 1px solid #21262d;
    }}
    tbody tr:last-child td {{
        border-bottom: none;
    }}
    tbody tr:hover {{
        background: #1c2128;
    }}
    .instrument-cell {{
        font-weight: 700;
        color: #58a6ff;
    }}
    .filter-cell {{
        color: #e6edf3;
    }}
    .dollars-cell {{
        font-weight: 600;
        color: #3fb950;
    }}
    .expr-high {{
        color: #3fb950;
        font-weight: 600;
    }}
    .fit-ok {{
        color: #3fb950;
        font-size: 12px;
    }}
    .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }}
    .badge-long {{
        background: #1f3a2a;
        color: #3fb950;
        border: 1px solid #3fb950;
    }}
    .badge-short {{
        background: #3d1f20;
        color: #f85149;
        border: 1px solid #f85149;
    }}
    .badge-cont {{
        background: #2a2f1f;
        color: #d29922;
        border: 1px solid #d29922;
    }}
    .badge-watch {{
        background: #3d2e1f;
        color: #d29922;
        border: 1px solid #d29922;
    }}
    .badge-decay {{
        background: #3d1f20;
        color: #f85149;
        border: 1px solid #f85149;
    }}
    .badge-stale {{
        background: #2d2d2d;
        color: #8b949e;
        border: 1px solid #8b949e;
    }}
    .footer {{
        text-align: center;
        padding: 20px;
        color: #484f58;
        font-size: 12px;
        border-top: 1px solid #30363d;
        margin-top: 20px;
    }}
    @media print {{
        body {{ background: white; color: black; padding: 10px; }}
        .session-card {{ border: 1px solid #ccc; }}
        .session-header {{ background: #f0f0f0; }}
        .session-time {{ color: #d35400; }}
        .instrument-cell {{ color: #2980b9; }}
        .dollars-cell {{ color: #27ae60; }}
        .summary-card {{ border: 1px solid #ccc; }}
    }}
    @media (max-width: 768px) {{
        .session-header {{ flex-direction: column; gap: 4px; }}
        .session-event {{ margin-left: 0; }}
        table {{ font-size: 12px; }}
        thead th, tbody td {{ padding: 6px 8px; }}
    }}
</style>
</head>
<body>
    <div class="header">
        <h1>TRADE SHEET</h1>
        <div class="date">{day_name} {date_str}</div>
        <div class="subtitle">Generated {now_str} &mdash; {trade_num} active trades &mdash; All times Brisbane (AEST UTC+10)</div>
    </div>

    <div class="entry-model-note">
        All entries are <strong>E2 (stop-market)</strong> &mdash; place stop orders at ORB high/low.
        They trigger automatically on breakout. ORB = first N minutes after session start.
    </div>

    <div class="summary-row">
        {summary_html}
    </div>

    {cards_html}

    <div class="footer">
        Source: live_config.py resolved portfolio &mdash; dollar gate applied &mdash;
        only cost-positive trades shown
    </div>
</body>
</html>"""
    return html


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate daily trade sheet HTML")
    parser.add_argument("--date", type=str, default=None, help="Trading date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path. Default: trade_sheet.html")
    parser.add_argument("--db-path", type=Path, default=None, help="Path to gold.db")
    parser.add_argument("--no-open", action="store_true", help="Don't open in browser after generating")
    args = parser.parse_args()

    db_path = args.db_path or GOLD_DB_PATH
    trading_day = date.fromisoformat(args.date) if args.date else date.today()
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "trade_sheet.html"

    print(f"Trade Sheet Generator")
    print(f"  Date:   {trading_day}")
    print(f"  DB:     {db_path}")
    print(f"  Output: {output_path}")
    print()

    # Resolve session times
    print("Resolving session times...")
    session_times = _resolve_session_times(trading_day)
    for label in sorted(session_times, key=lambda s: _sort_key(*session_times[s])):
        h, m = session_times[label]
        print(f"  {_format_time(h, m):>10}  {label}")
    print()

    # Collect trades
    print("Building resolved portfolios...")
    trades = collect_trades(trading_day, db_path)
    print(f"  {len(trades)} active trades across {len(set(t['instrument'] for t in trades))} instruments")
    print()

    if not trades:
        print("ERROR: No active trades found. Check DB and live_config.py.")
        sys.exit(1)

    # Generate HTML
    html = generate_html(trades, session_times, trading_day)
    output_path.write_text(html, encoding="utf-8")
    print(f"Written to {output_path}")

    # Open in browser
    if not args.no_open:
        webbrowser.open(str(output_path))
        print("Opened in browser.")


if __name__ == "__main__":
    main()
