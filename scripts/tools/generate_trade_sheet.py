#!/usr/bin/env python3
"""
Daily Trade Sheet Generator V2.

Two sections:
  1. DEPLOYED — lanes from active prop_profiles (what you're committed to)
  2. OPPORTUNITIES — all other validated strategies that pass gates (manual pickup)

Usage:
    python scripts/tools/generate_trade_sheet.py              # both sections
    python scripts/tools/generate_trade_sheet.py --deployed-only  # deployed only (V1 behavior)
    python scripts/tools/generate_trade_sheet.py --date 2026-03-04
    python scripts/tools/generate_trade_sheet.py --no-open
"""

import argparse
import sys
import webbrowser
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.cost_model import get_cost_spec
from pipeline.dst import SESSION_CATALOG
from pipeline.paths import GOLD_DB_PATH
from trading_app.prop_profiles import ACCOUNT_PROFILES
from trading_app.strategy_fitness import compute_fitness

# Dollar gate: expected $/trade must be >= this multiplier * RT friction.
# Was LIVE_MIN_EXPECTANCY_DOLLARS_MULT in live_config.py (1.3).
_DOLLAR_GATE_MULT = 1.3


@dataclass(frozen=True)
class FitnessCheckResult:
    """Resolved fitness status for a strategy, including lookup failures."""

    status: str
    error: str | None = None


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


def _passes_dollar_gate(row: dict, instrument: str) -> tuple[bool, float | None]:
    """Check if expected $/trade >= _DOLLAR_GATE_MULT * RT cost.

    Returns (passes, exp_dollars). Fail-closed: returns (False, None) when
    median_risk_points is missing or cost spec is unavailable.
    """
    median_risk_pts = row.get("median_risk_points")
    exp_r = row.get("expectancy_r")
    if median_risk_pts is None or exp_r is None:
        return False, None
    try:
        spec = get_cost_spec(instrument)
    except Exception as exc:
        print(f"  WARNING: cost spec lookup failed for {instrument}: {exc}", flush=True)
        return False, None
    exp_d = exp_r * median_risk_pts * spec.point_value
    min_dollars = _DOLLAR_GATE_MULT * spec.total_friction
    return exp_d >= min_dollars, exp_d


def _check_fitness(
    strategy_id: str,
    db_path: Path,
    cache: dict[str, FitnessCheckResult] | None = None,
) -> FitnessCheckResult:
    """Quick fitness check. Returns cached status and surfaces lookup failures."""
    if cache is not None and strategy_id in cache:
        return cache[strategy_id]

    try:
        f = compute_fitness(strategy_id, db_path=db_path)
        result = FitnessCheckResult(status=f.fitness_status)
    except Exception as exc:
        result = FitnessCheckResult(status="UNKNOWN", error=f"{type(exc).__name__}: {exc}")

    if cache is not None:
        cache[strategy_id] = result
    return result


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


def collect_trades(trading_day: date, db_path: Path, profile_filter: str | None = None) -> list[dict]:
    """Collect trades from prop_profiles deployed lanes.

    For each active profile's daily_lanes, looks up the exact strategy_id
    in validated_setups. Applies dollar gate. Only returns cost-positive trades.

    Args:
        profile_filter: if set, only show this profile (e.g. "apex_50k_manual").

    Source of truth: trading_app.prop_profiles.ACCOUNT_PROFILES (not live_config).
    """
    trades = []
    fitness_cache: dict[str, FitnessCheckResult] = {}

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        for pid, profile in ACCOUNT_PROFILES.items():
            if profile_filter and pid != profile_filter:
                continue
            if not profile.active or not profile.daily_lanes:
                continue

            for lane in profile.daily_lanes:
                sid = lane.strategy_id
                instrument = lane.instrument

                # Direct lookup — lane specifies exact strategy_id
                row = con.execute(
                    """
                    SELECT vs.strategy_id, vs.orb_label, vs.orb_minutes,
                           vs.filter_type, vs.rr_target, vs.win_rate,
                           vs.expectancy_r, vs.sample_size,
                           es.median_risk_points
                    FROM validated_setups vs
                    LEFT JOIN experimental_strategies es
                      ON vs.strategy_id = es.strategy_id
                    WHERE vs.strategy_id = ?
                """,
                    [sid],
                ).fetchone()

                if row is None:
                    print(f"  WARNING: {sid} not in validated_setups — skipping", flush=True)
                    continue

                cols = [d[0] for d in con.description]
                variant = dict(zip(cols, row, strict=False))

                # Dollar gate
                passes, exp_d = _passes_dollar_gate(variant, instrument)
                if not passes:
                    print(f"  WARNING: {sid} failed dollar gate — skipping", flush=True)
                    continue

                # Fitness check
                fitness = _check_fitness(sid, db_path, fitness_cache)

                sm = lane.planned_stop_multiplier or profile.stop_multiplier

                trades.append(
                    {
                        "session": variant["orb_label"],
                        "instrument": instrument,
                        "strategy_id": sid,
                        "aperture": variant.get("orb_minutes", 5),
                        "direction": _direction_rule(variant["filter_type"]),
                        "filter_desc": _filter_description(variant["filter_type"]),
                        "filter_type": variant["filter_type"],
                        "rr": variant["rr_target"],
                        "win_rate": variant["win_rate"],
                        "exp_r": variant["expectancy_r"],
                        "exp_dollars": exp_d,
                        "sample_size": variant["sample_size"],
                        "fitness": fitness.status,
                        "fitness_error": fitness.error,
                        "profile": pid,
                        "stop_mult": sm,
                        "orb_cap": lane.max_orb_size_pts,
                        "notes": lane.execution_notes,
                    }
                )
    finally:
        con.close()

    return trades


def collect_opportunities(
    db_path: Path,
    deployed_sids: set[str],
) -> list[dict]:
    """Collect all validated strategies that pass gates but aren't deployed.

    Best per session x instrument (highest ExpR). Applies dollar gate.
    Skips PURGED/DECAY fitness. No look-ahead — uses only validated_setups
    and experimental_strategies (pre-computed, no future data).
    """
    active_instruments = tuple(sorted(ACTIVE_ORB_INSTRUMENTS))
    opportunities = []
    fitness_cache: dict[str, FitnessCheckResult] = {}

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        # Best strategy per session x instrument, respecting family_rr_locks.
        # Join locks to pick the locked RR target per family (no RR snooping).
        rows = con.execute(
            """
            WITH locked AS (
                SELECT vs.strategy_id, vs.instrument, vs.orb_label, vs.orb_minutes,
                       vs.filter_type, vs.rr_target, vs.stop_multiplier,
                       vs.win_rate, vs.expectancy_r, vs.sample_size,
                       es.median_risk_points,
                       ROW_NUMBER() OVER (
                           PARTITION BY vs.instrument, vs.orb_label
                           ORDER BY vs.expectancy_r DESC
                       ) as rn
                FROM validated_setups vs
                INNER JOIN family_rr_locks frl
                  ON vs.instrument = frl.instrument
                  AND vs.orb_label = frl.orb_label
                  AND vs.filter_type = frl.filter_type
                  AND vs.entry_model = frl.entry_model
                  AND vs.orb_minutes = frl.orb_minutes
                  AND vs.confirm_bars = frl.confirm_bars
                  AND vs.rr_target = frl.locked_rr
                LEFT JOIN experimental_strategies es
                  ON vs.strategy_id = es.strategy_id
                WHERE LOWER(vs.status) = 'active'
                  AND vs.expectancy_r > 0
                  AND vs.sample_size >= 100
                  AND vs.instrument IN (SELECT UNNEST(?::VARCHAR[]))
            )
            SELECT * FROM locked WHERE rn = 1
            ORDER BY orb_label, instrument
        """,
            [list(active_instruments)],
        ).fetchall()

        cols = [d[0] for d in con.description]

        for row in rows:
            variant = dict(zip(cols, row, strict=False))
            sid = variant["strategy_id"]
            instrument = variant["instrument"]

            # Skip already-deployed strategies
            if sid in deployed_sids:
                continue

            # Dollar gate
            passes, exp_d = _passes_dollar_gate(variant, instrument)
            if not passes:
                continue

            # Fitness check — skip PURGED/DECAY
            fitness = _check_fitness(sid, db_path, fitness_cache)
            if fitness.status in ("PURGED", "DECAY"):
                continue

            opportunities.append(
                {
                    "session": variant["orb_label"],
                    "instrument": instrument,
                    "strategy_id": sid,
                    "aperture": variant.get("orb_minutes", 5),
                    "direction": _direction_rule(variant["filter_type"]),
                    "filter_desc": _filter_description(variant["filter_type"]),
                    "filter_type": variant["filter_type"],
                    "rr": variant["rr_target"],
                    "win_rate": variant["win_rate"],
                    "exp_r": variant["expectancy_r"],
                    "exp_dollars": exp_d,
                    "sample_size": variant["sample_size"],
                    "fitness": fitness.status,
                    "fitness_error": fitness.error,
                    "profile": "opportunity",
                    "stop_mult": variant.get("stop_multiplier", 1.0),
                    "orb_cap": None,
                    "notes": "",
                }
            )
    finally:
        con.close()

    return opportunities


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
        "UNKNOWN": "badge-unknown",
    }.get(fitness, "badge-unknown")
    return f' <span class="badge {cls}">{fitness}</span>'


def _build_session_cards(
    trades: list[dict],
    session_times: dict,
    profiles_used: dict,
    css_class: str = "",
) -> tuple[str, int]:
    """Build session card HTML from a list of trades. Returns (html, trade_count)."""
    sessions_used = sorted(
        set(t["session"] for t in trades),
        key=lambda s: _sort_key(*session_times.get(s, (0, 0))),
    )

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
            fitness_title = ""
            if t.get("fitness_error"):
                fitness_title = f' title="{t["fitness_error"]}"'

            exp_r_class = "expr-high" if t["exp_r"] >= 0.20 else ""

            notes_parts = []
            if t.get("orb_cap"):
                notes_parts.append(f"Cap {t['orb_cap']:.0f}pts")
            if t.get("stop_mult") and t["stop_mult"] != 0.75:
                notes_parts.append(f"Stop {t['stop_mult']}x")
            if t.get("notes"):
                notes_parts.append(t["notes"][:80])
            notes_html = f'<div class="lane-notes">{" | ".join(notes_parts)}</div>' if notes_parts else ""

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
                <td>N={t["sample_size"]}</td>
                <td{fitness_title}>{fit_badge if fit_badge else '<span class="fit-ok">FIT</span>'}</td>
            </tr>"""
            if notes_html:
                rows_html += f"""
            <tr class="notes-row"><td colspan="10">{notes_html}</td></tr>"""

        # Session header with firm badge
        firm_badges = set()
        for t in session_trades:
            pi = profiles_used.get(t.get("profile", ""), {})
            if pi:
                firm_badges.add(f'<span class="badge badge-firm">{pi.get("firm", "?")} {pi.get("mode", "?")}</span>')
        firm_badges_html = " ".join(firm_badges)

        card_class = f"session-card {css_class}" if css_class else "session-card"
        cards_html += f"""
        <div class="{card_class}">
            <div class="session-header">
                <div class="session-time">{time_str} BRIS</div>
                <div class="session-name">{session} {firm_badges_html}</div>
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
                        <th>N</th>
                        <th>Fitness</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>"""

    return cards_html, trade_num


def generate_html(
    trades: list[dict],
    session_times: dict,
    trading_day: date,
    opportunities: list[dict] | None = None,
) -> str:
    """Generate self-contained HTML trade sheet with deployed + opportunities."""

    day_name = trading_day.strftime("%A")
    date_str = trading_day.strftime("%d %b %Y")
    now_str = datetime.now().strftime("%H:%M")

    # Profile summary bar
    profiles_used = {}
    for t in trades:
        pid = t.get("profile", "unknown")
        if pid not in profiles_used:
            prof = ACCOUNT_PROFILES.get(pid)
            if prof:
                from trading_app.prop_profiles import PROP_FIRM_SPECS, get_account_tier

                spec = PROP_FIRM_SPECS.get(prof.firm, None)
                tier = get_account_tier(prof.firm, prof.account_size)
                auto_label = {"none": "MANUAL", "full": "AUTO", "semi": "SEMI"}.get(
                    spec.auto_trading if spec else "none", "?"
                )
                profiles_used[pid] = {
                    "firm": prof.firm.upper(),
                    "size": f"${prof.account_size // 1000}K",
                    "dd": f"${tier.max_dd:,}",
                    "stop": f"{prof.stop_multiplier}x",
                    "lanes": len(prof.daily_lanes),
                    "mode": auto_label,
                    "copies": prof.copies,
                }

    profile_bar_html = ""
    for _pid, info in profiles_used.items():
        copies_note = f" x{info['copies']}" if info["copies"] > 1 else ""
        profile_bar_html += f"""
        <div class="profile-card profile-{info["mode"].lower()}">
            <strong>{info["firm"]} {info["size"]}{copies_note}</strong>
            <span class="profile-mode">{info["mode"]}</span>
            <div class="profile-detail">DD {info["dd"]} | Stop {info["stop"]} | {info["lanes"]} lanes</div>
        </div>"""

    # Build deployed session cards
    cards_html, trade_num = _build_session_cards(trades, session_times, profiles_used)

    # Build opportunities section
    opps_html = ""
    opps_count = 0
    if opportunities:
        opps_html, opps_count = _build_session_cards(opportunities, session_times, profiles_used, css_class="opp-card")

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

    fitness_errors = [t for t in trades if t.get("fitness_error")]
    fitness_warning_html = ""
    if fitness_errors:
        error_items = ""
        for t in fitness_errors[:10]:
            error_items += f"<li><code>{t['strategy_id']}</code>: {t['fitness_error']}</li>"
        more_note = ""
        if len(fitness_errors) > 10:
            more_note = f"<p>+ {len(fitness_errors) - 10} more fitness lookup errors.</p>"
        fitness_warning_html = f"""
    <div class="warning-box">
        <strong>Fitness lookup errors:</strong> {len(fitness_errors)} row(s) rendered as <code>UNKNOWN</code>.
        <ul>{error_items}</ul>
        {more_note}
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
    .badge-unknown {{
        background: #5c4712;
        color: #ffd58a;
        border: 1px solid #d29922;
    }}
    .profile-bar {{
        display: flex;
        gap: 12px;
        margin-bottom: 24px;
        flex-wrap: wrap;
    }}
    .profile-card {{
        padding: 10px 16px;
        border-radius: 8px;
        border: 1px solid #30363d;
        background: #161b22;
        flex: 1;
        min-width: 200px;
    }}
    .profile-card strong {{ font-size: 14px; }}
    .profile-mode {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        margin-left: 8px;
    }}
    .profile-manual .profile-mode {{
        background: #1a3a1a;
        color: #3fb950;
        border: 1px solid #3fb950;
    }}
    .profile-auto .profile-mode {{
        background: #1a2a3a;
        color: #58a6ff;
        border: 1px solid #58a6ff;
    }}
    .profile-shadow .profile-mode {{
        background: #2d2d2d;
        color: #8b949e;
        border: 1px solid #8b949e;
    }}
    .profile-detail {{
        font-size: 12px;
        color: #8b949e;
        margin-top: 4px;
    }}
    .badge-firm {{
        font-size: 10px;
        padding: 2px 6px;
        border-radius: 4px;
        background: #21262d;
        color: #8b949e;
        border: 1px solid #30363d;
        margin-left: 8px;
    }}
    .lane-notes {{
        font-size: 11px;
        color: #d29922;
        padding: 2px 8px 6px;
    }}
    .notes-row td {{
        padding: 0 !important;
        border: none !important;
    }}
    .warning-box {{
        background: #271b05;
        border: 1px solid #d29922;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 24px;
        color: #ffd58a;
        line-height: 1.5;
    }}
    .warning-box ul {{
        margin: 8px 0 0 20px;
    }}
    .warning-box code {{
        background: #3d2e1f;
        padding: 1px 4px;
        border-radius: 4px;
    }}
    .section-divider {{
        text-align: center;
        margin: 32px 0 24px;
        padding: 16px;
        border-top: 2px solid #30363d;
    }}
    .section-divider h2 {{
        font-size: 20px;
        font-weight: 700;
        color: #8b949e;
    }}
    .section-divider .section-sub {{
        font-size: 13px;
        color: #484f58;
        margin-top: 4px;
    }}
    .opp-card {{
        border-color: #2d333b;
        opacity: 0.85;
    }}
    .opp-card .session-header {{
        background: #13171d;
    }}
    .opp-card .session-time {{
        color: #8b949e;
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
        <div class="subtitle">Generated {now_str} &mdash; {trade_num} deployed + {
        opps_count
    } opportunities &mdash; All times Brisbane (AEST UTC+10)</div>
    </div>

    <div class="profile-bar">
        {profile_bar_html}
    </div>

    <div class="entry-model-note">
        All entries are <strong>E2 (stop-market)</strong> &mdash; place stop orders at ORB high/low.
        They trigger automatically on breakout. ORB = first N minutes after session start.
    </div>

    {fitness_warning_html}

    <div class="summary-row">
        {summary_html}
    </div>

    {cards_html}

    {
        f'''
    <div class="section-divider">
        <h2>OPPORTUNITIES &mdash; {opps_count} additional validated strategies</h2>
        <div class="section-sub">Best per session x instrument &mdash; not deployed, pass all gates &mdash; manual pickup</div>
    </div>
    {opps_html}
    '''
        if opps_html
        else ""
    }

    <div class="footer">
        Source: prop_profiles.py deployed lanes + validated_setups opportunities &mdash;
        dollar gate applied &mdash; only cost-positive, non-PURGED trades shown
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
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Filter to one profile (e.g. apex_50k_manual). Default: all active.",
    )
    parser.add_argument(
        "--deployed-only",
        action="store_true",
        help="Only show deployed lanes (V1 behavior). Skip opportunities section.",
    )
    args = parser.parse_args()

    db_path = args.db_path or GOLD_DB_PATH
    trading_day = date.fromisoformat(args.date) if args.date else date.today()
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "trade_sheet.html"

    print("Trade Sheet Generator")
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

    # Collect deployed trades
    if args.profile:
        print(f"Filtering to profile: {args.profile}")
    print("Building deployed lanes...")
    trades = collect_trades(trading_day, db_path, profile_filter=args.profile)
    print(f"  {len(trades)} deployed trades across {len(set(t['instrument'] for t in trades))} instruments")

    # Collect opportunities (unless --deployed-only)
    opportunities = []
    if not args.deployed_only:
        print("Scanning validated opportunities...")
        deployed_sids = {t["strategy_id"] for t in trades}
        opportunities = collect_opportunities(db_path, deployed_sids)
        opp_instruments = set(t["instrument"] for t in opportunities) if opportunities else set()
        print(f"  {len(opportunities)} opportunities across {len(opp_instruments)} instruments")
    print()

    if not trades and not opportunities:
        print("ERROR: No trades or opportunities found. Check DB and prop_profiles.py.")
        sys.exit(1)

    # Generate HTML
    html = generate_html(trades, session_times, trading_day, opportunities=opportunities or None)
    output_path.write_text(html, encoding="utf-8")
    print(f"Written to {output_path}")

    # Open in browser
    if not args.no_open:
        webbrowser.open(str(output_path))
        print("Opened in browser.")


if __name__ == "__main__":
    main()
