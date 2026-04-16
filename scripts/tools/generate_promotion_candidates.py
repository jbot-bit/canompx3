#!/usr/bin/env python3
"""
Strategy Promotion Candidate Report.

Surfaces validated strategies that passed full institutional gates
(FDR, walk-forward, ROBUST edge family) but are NOT yet in the live portfolio.
Generates a scorecard HTML for PM review.

Institutional grounding: Lopez de Prado AFML Ch.18 strategy lifecycle.
Scorecard pattern: auto-generated metrics, manual go/no-go decision.

Usage:
    python scripts/tools/generate_promotion_candidates.py
    python scripts/tools/generate_promotion_candidates.py --format terminal
    python scripts/tools/generate_promotion_candidates.py --no-open
"""

import argparse
import json
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.asset_configs import get_active_instruments
from pipeline.cost_model import get_cost_spec
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS, CORE_MIN_SAMPLES, OrbSizeFilter
from trading_app.live_config import (
    LIVE_MIN_EXPECTANCY_DOLLARS_MULT,
    LIVE_MIN_EXPECTANCY_R,
    LIVE_PORTFOLIO,
)
from trading_app.validated_shelf import deployable_validated_relation

# ── Core query ────────────────────────────────────────────────────────


def find_uncovered_candidates(db_path: Path) -> list[dict]:
    """Find FDR+WF+ROBUST strategies not covered by LIVE_PORTFOLIO.

    Returns list of candidate dicts sorted by expectancy_r DESC.
    Each candidate is the best-ExpR strategy per uncovered
    (instrument, orb_label, entry_model, filter_type) combo.
    """
    covered = {(s.orb_label, s.entry_model, s.filter_type) for s in LIVE_PORTFOLIO}

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        shelf_relation = deployable_validated_relation(con, alias="vs")
        rows = con.execute(
            f"""
            SELECT vs.strategy_id, vs.instrument, vs.orb_label, vs.entry_model,
                   vs.filter_type, vs.orb_minutes, vs.rr_target, vs.confirm_bars,
                   vs.sample_size, vs.win_rate, vs.expectancy_r, vs.sharpe_ann,
                   vs.max_drawdown_r, vs.years_tested, vs.all_years_positive,
                   vs.yearly_results, vs.fdr_significant, vs.fdr_adjusted_p,
                   vs.wf_passed, vs.wf_windows, vs.wfe, vs.skewness,
                   vs.kurtosis_excess, vs.stop_multiplier,
                   ef.robustness_status, ef.member_count, ef.pbo,
                   ef.cv_expectancy, ef.trade_tier,
                   es.median_risk_points
            FROM {shelf_relation}
            INNER JOIN edge_families ef
              ON vs.strategy_id = ef.head_strategy_id
            LEFT JOIN experimental_strategies es
              ON vs.strategy_id = es.strategy_id AND es.is_canonical = TRUE
            WHERE vs.fdr_significant = TRUE
              AND vs.wf_passed = TRUE
              AND ef.robustness_status = 'ROBUST'
              AND vs.expectancy_r >= ?
            ORDER BY vs.expectancy_r DESC
            """,
            [LIVE_MIN_EXPECTANCY_R],
        ).fetchall()
        cols = [desc[0] for desc in con.description]
    finally:
        con.close()

    seen: set[tuple] = set()
    candidates = []
    for row in rows:
        d = dict(zip(cols, row, strict=False))
        combo_key = (d["orb_label"], d["entry_model"], d["filter_type"])
        if combo_key in covered:
            continue
        inst_key = (d["instrument"], d["orb_label"], d["entry_model"], d["filter_type"])
        if inst_key in seen:
            continue
        seen.add(inst_key)
        candidates.append(d)

    return candidates


# ── Day-overlap computation ───────────────────────────────────────────


def build_day_sets(
    con: duckdb.DuckDBPyConnection,
    candidates: list[dict],
) -> dict[tuple, frozenset[str]]:
    """Build trade-day sets for candidates and same-session LIVE_PORTFOLIO specs.

    Used to compute day-overlap before promotion. Only OrbSizeFilter types are
    supported — VolumeFilter/DirectionFilter require separate pre-computation and
    are skipped here (cross-class comparisons have structurally low overlap).

    Key: (instrument, orb_label, entry_model, filter_type, orb_minutes)
    Value: frozenset of trading_day strings
    """
    specs_to_load: set[tuple] = set()
    for c in candidates:
        inst, session, em, ft, om = (
            c["instrument"],
            c["orb_label"],
            c["entry_model"],
            c["filter_type"],
            c["orb_minutes"],
        )
        if ft in ALL_FILTERS and isinstance(ALL_FILTERS[ft], OrbSizeFilter):
            specs_to_load.add((inst, session, em, ft, om))
        for spec in LIVE_PORTFOLIO:
            if spec.orb_label != session or spec.entry_model != em:
                continue
            if spec.filter_type not in ALL_FILTERS:
                continue
            if not isinstance(ALL_FILTERS[spec.filter_type], OrbSizeFilter):
                continue
            specs_to_load.add((inst, session, em, spec.filter_type, om))

    result: dict[tuple, frozenset[str]] = {}
    for key in specs_to_load:
        inst, session, em, ft, om = key
        f = ALL_FILTERS[ft]
        size_col = f"orb_{session}_size"
        conds = [
            "oo.symbol = ?",
            "oo.orb_label = ?",
            "oo.entry_model = ?",
            "oo.orb_minutes = ?",
            f"df.{size_col} IS NOT NULL",
        ]
        params: list = [inst, session, em, om]
        if f.min_size is not None:
            conds.append(f"df.{size_col} >= ?")
            params.append(f.min_size)
        if f.max_size is not None:
            conds.append(f"df.{size_col} < ?")
            params.append(f.max_size)
        rows = con.execute(
            f"SELECT DISTINCT CAST(oo.trading_day AS VARCHAR) "  # noqa: S608
            f"FROM orb_outcomes oo "
            f"JOIN daily_features df ON oo.symbol = df.symbol "
            f"  AND oo.trading_day = df.trading_day "
            f"WHERE {' AND '.join(conds)}",
            params,
        ).fetchall()
        result[key] = frozenset(r[0] for r in rows)

    return result


# ── Enrichment ────────────────────────────────────────────────────────


def enrich_candidate(candidate: dict, day_sets: dict | None = None) -> dict:
    """Add year-by-year breakdown, decay trend, dollar gate results, and overlap."""
    yearly_raw = candidate.get("yearly_results")
    if yearly_raw:
        yearly = json.loads(yearly_raw) if isinstance(yearly_raw, str) else yearly_raw
        years = []
        avg_rs = []
        for yr in sorted(yearly.keys()):
            y = yearly[yr]
            n = y.get("n", y.get("trades", 0))
            wr = y.get("win_rate", 0)
            avg_r = y.get("avg_r", y.get("expectancy_r", 0))
            years.append(
                {
                    "year": yr,
                    "n": n,
                    "win_rate": wr,
                    "avg_r": round(avg_r, 4),
                    "total_r": round(avg_r * n, 2),
                }
            )
            avg_rs.append(avg_r)
        candidate["year_by_year"] = years

        if len(avg_rs) >= 3:
            x = list(range(len(avg_rs)))
            x_mean = sum(x) / len(x)
            y_mean = sum(avg_rs) / len(avg_rs)
            num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, avg_rs, strict=True))
            den = sum((xi - x_mean) ** 2 for xi in x)
            candidate["decay_slope"] = round(num / den, 4) if den > 0 else 0.0
        else:
            candidate["decay_slope"] = 0.0
    else:
        candidate["year_by_year"] = []
        candidate["decay_slope"] = 0.0

    instruments = get_active_instruments()
    median_risk = candidate.get("median_risk_points")
    expr = candidate.get("expectancy_r", 0)
    dollar_results = {}
    for inst in instruments:
        try:
            spec = get_cost_spec(inst)
            min_dollars = LIVE_MIN_EXPECTANCY_DOLLARS_MULT * spec.total_friction
            result: dict = {
                "rt_cost": round(spec.total_friction, 2),
                "min_required": round(min_dollars, 2),
                "point_value": spec.point_value,
            }
            # Compute actual Exp$ and pass/fail when median_risk is available
            # and this is the candidate's native instrument
            if median_risk is not None and inst == candidate.get("instrument"):
                exp_dollars = expr * median_risk * spec.point_value
                result["exp_dollars"] = round(exp_dollars, 2)
                result["passes"] = exp_dollars >= min_dollars
            dollar_results[inst] = result
        except Exception as exc:
            print(f"  WARNING: cost spec unavailable for {inst}: {exc}")
            dollar_results[inst] = {"rt_cost": None, "min_required": None, "point_value": None}
    candidate["dollar_gate_results"] = dollar_results

    # --- Day-overlap against existing LIVE_PORTFOLIO specs ---
    candidate["overlap_pct"] = None
    candidate["overlap_with"] = None
    candidate["marginal_days"] = None
    if day_sets is not None:
        ft = candidate["filter_type"]
        if ft in ALL_FILTERS and isinstance(ALL_FILTERS[ft], OrbSizeFilter):
            inst = candidate["instrument"]
            session = candidate["orb_label"]
            em = candidate["entry_model"]
            om = candidate["orb_minutes"]
            cand_key = (inst, session, em, ft, om)
            cand_days = day_sets.get(cand_key, frozenset())
            max_pct = 0.0
            worst_ft: str | None = None
            union_portfolio: set[str] = set()
            for spec in LIVE_PORTFOLIO:
                if spec.orb_label != session or spec.entry_model != em:
                    continue
                if spec.filter_type not in ALL_FILTERS or not isinstance(ALL_FILTERS[spec.filter_type], OrbSizeFilter):
                    continue
                spec_key = (inst, session, em, spec.filter_type, om)
                spec_days = day_sets.get(spec_key, frozenset())
                union_portfolio |= spec_days
                if cand_days:
                    pct = len(cand_days & spec_days) / len(cand_days)
                    if pct > max_pct:
                        max_pct = pct
                        worst_ft = spec.filter_type
            candidate["overlap_pct"] = round(max_pct, 4)
            candidate["overlap_with"] = worst_ft
            candidate["marginal_days"] = len(cand_days - union_portfolio)

    return candidate


# ── Spec code generator ──────────────────────────────────────────────


def generate_spec_code(
    orb_label: str,
    entry_model: str,
    filter_type: str,
    sample_size: int = CORE_MIN_SAMPLES,
) -> str:
    """Generate copy-paste LiveStrategySpec Python code.

    Tier is derived from sample_size vs CORE_MIN_SAMPLES (config.py).
    """
    family_id = f"{orb_label}_{entry_model}_{filter_type}"
    if sample_size >= CORE_MIN_SAMPLES:
        return f'LiveStrategySpec("{family_id}", "core", "{orb_label}", "{entry_model}", "{filter_type}", None)'
    return f'LiveStrategySpec("{family_id}", "regime", "{orb_label}", "{entry_model}", "{filter_type}", "high_vol")'


# ── Terminal format ──────────────────────────────────────────────────


def format_terminal(candidates: list[dict]) -> str:
    """Compact CLI table output."""
    sessions = sorted(set(c["orb_label"] for c in candidates))
    lines = []
    lines.append(f"=== PROMOTION CANDIDATES: {len(candidates)} strategies across {len(sessions)} sessions ===")
    lines.append("")
    lines.append(
        f"{'Strategy ID':<55} {'Inst':<5} {'ORB':<4} {'ExpR':>6} "
        f"{'N':>5} {'WR%':>5} {'Fam':>4} {'PBO':>5} {'WFE':>5} {'Decay':>7} {'Overlap':>10}"
    )
    lines.append("-" * 122)

    for c in candidates:
        pbo_str = f"{c['pbo']:.2f}" if c.get("pbo") is not None else "n/a"
        wfe_str = f"{c['wfe']:.1%}" if c.get("wfe") is not None else "n/a"
        decay_str = f"{c['decay_slope']:+.4f}" if c.get("decay_slope") is not None else "n/a"
        op = c.get("overlap_pct")
        if op is None:
            overlap_str = "n/a"
            overlap_flag = ""
        elif op >= 0.8:
            overlap_str = f"{op:.0%} vs {c.get('overlap_with', '?')}"
            overlap_flag = " [WARN]"
        elif op >= 0.5:
            overlap_str = f"{op:.0%} vs {c.get('overlap_with', '?')}"
            overlap_flag = " [NOTE]"
        else:
            overlap_str = f"{op:.0%}"
            overlap_flag = ""
        lines.append(
            f"{c['strategy_id']:<55} {c['instrument']:<5} {c['orb_minutes']:>3}m "
            f"{c['expectancy_r']:>+.3f} {c['sample_size']:>5} "
            f"{c['win_rate']:>4.0%} {c['member_count']:>4} "
            f"{pbo_str:>5} {wfe_str:>5} {decay_str:>7} {overlap_str:>10}{overlap_flag}"
        )

    lines.append("")
    lines.append("To promote: add LiveStrategySpec to trading_app/live_config.py")
    lines.append("Suggested code per candidate:")
    lines.append("")

    seen_combos: set[tuple] = set()
    for c in candidates:
        combo = (c["orb_label"], c["entry_model"], c["filter_type"])
        if combo in seen_combos:
            continue
        seen_combos.add(combo)
        lines.append(f"    {generate_spec_code(c['orb_label'], c['entry_model'], c['filter_type'], c['sample_size'])},")

    lines.append("")
    return "\n".join(lines)


# ── HTML report ──────────────────────────────────────────────────────


def generate_html(candidates: list[dict]) -> str:
    """Generate self-contained HTML scorecard report."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    sessions = sorted(set(c["orb_label"] for c in candidates))

    # Build candidate cards grouped by session
    cards_html = ""
    for session in sessions:
        session_candidates = [c for c in candidates if c["orb_label"] == session]
        if not session_candidates:
            continue

        rows_html = ""
        for c in session_candidates:
            pbo_str = f"{c['pbo']:.2f}" if c.get("pbo") is not None else "n/a"
            wfe_str = f"{c['wfe']:.1%}" if c.get("wfe") is not None else "n/a"
            decay = c.get("decay_slope", 0)
            decay_cls = "decay-warn" if decay < -0.05 else ""
            decay_str = f"{decay:+.4f}"

            # Year-by-year mini table
            yby_rows = ""
            for y in c.get("year_by_year", []):
                yr_cls = "yr-negative" if y["avg_r"] < 0 else ""
                yby_rows += (
                    f'<tr class="{yr_cls}">'
                    f"<td>{y['year']}</td><td>{y['n']}</td>"
                    f"<td>{y['win_rate']:.0%}</td><td>{y['avg_r']:+.4f}</td>"
                    f"<td>{y['total_r']:+.2f}</td></tr>"
                )

            # Dollar gate per instrument
            dg_rows = ""
            for inst, dg in c.get("dollar_gate_results", {}).items():
                if dg.get("rt_cost") is None:
                    continue
                exp_str = f"${dg['exp_dollars']:.2f}" if "exp_dollars" in dg else "-"
                if "passes" in dg:
                    pass_cls = "badge-ayp" if dg["passes"] else "badge-decay-warn"
                    pass_str = f"<span class='{pass_cls}'>{'PASS' if dg['passes'] else 'FAIL'}</span>"
                else:
                    pass_str = "-"
                dg_rows += (
                    f"<tr><td>{inst}</td>"
                    f"<td>${dg['rt_cost']:.2f}</td>"
                    f"<td>${dg['min_required']:.2f}</td>"
                    f"<td>{exp_str}</td>"
                    f"<td>{pass_str}</td></tr>"
                )

            spec_code = generate_spec_code(c["orb_label"], c["entry_model"], c["filter_type"], c["sample_size"])

            op = c.get("overlap_pct")
            overlap_badge = ""
            if op is not None:
                wf_label = c.get("overlap_with") or "?"
                marginal = c.get("marginal_days", "?")
                if op >= 0.8:
                    overlap_badge = (
                        f"<span class='badge badge-overlap-warn'>"
                        f"OVERLAP {op:.0%} vs {wf_label} ({marginal} new days)</span>"
                    )
                elif op >= 0.5:
                    overlap_badge = (
                        f"<span class='badge badge-overlap-note'>"
                        f"OVERLAP {op:.0%} vs {wf_label} ({marginal} new days)</span>"
                    )

            rows_html += f"""
            <div class="candidate-card">
                <div class="candidate-header">
                    <div class="candidate-id">{c["strategy_id"]}</div>
                    <div class="candidate-badges">
                        <span class="badge badge-robust">ROBUST</span>
                        <span class="badge badge-fdr">FDR</span>
                        <span class="badge badge-wf">WF {c.get("wf_windows", "?")} win</span>
                        {"<span class='badge badge-ayp'>ALL YEARS +</span>" if c.get("all_years_positive") else ""}
                        {"<span class='badge badge-decay-warn'>DECAYING</span>" if decay < -0.05 else ""}
                        {"<span class='badge badge-decay-warn'>0.75x STOP</span>" if c.get("stop_multiplier", 1.0) < 1.0 else ""}
                        {overlap_badge}
                    </div>
                </div>

                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-label">ExpR</div>
                        <div class="metric-value metric-expr">{c["expectancy_r"]:+.4f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Sharpe</div>
                        <div class="metric-value">{c.get("sharpe_ann", 0):.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Sample</div>
                        <div class="metric-value">{c["sample_size"]}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value">{c["win_rate"]:.0%}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Family</div>
                        <div class="metric-value">{c.get("member_count", "?")} members</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">PBO</div>
                        <div class="metric-value">{pbo_str}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">WFE</div>
                        <div class="metric-value">{wfe_str}</div>
                    </div>
                    <div class="metric {decay_cls}">
                        <div class="metric-label">Decay</div>
                        <div class="metric-value">{decay_str}/yr</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Years</div>
                        <div class="metric-value">{c.get("years_tested", "?")}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">RR Target</div>
                        <div class="metric-value">{c["rr_target"]:.1f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Stop Mult</div>
                        <div class="metric-value">{c.get("stop_multiplier", 1.0):.2f}x</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Aperture</div>
                        <div class="metric-value">{c["orb_minutes"]}m</div>
                    </div>
                </div>

                <div class="section-row">
                    <div class="section-half">
                        <div class="section-title">Year-by-Year</div>
                        <table class="inner-table">
                            <thead><tr><th>Year</th><th>N</th><th>WR</th><th>AvgR</th><th>TotalR</th></tr></thead>
                            <tbody>{yby_rows}</tbody>
                        </table>
                    </div>
                    <div class="section-half">
                        <div class="section-title">Dollar Gate (per instrument)</div>
                        <table class="inner-table">
                            <thead><tr><th>Inst</th><th>RT Cost</th><th>Gate</th><th>Exp$</th><th>Result</th></tr></thead>
                            <tbody>{dg_rows}</tbody>
                        </table>
                    </div>
                </div>

                <div class="spec-code">
                    <div class="section-title">Add to live_config.py</div>
                    <pre><code>{spec_code},</code></pre>
                    {"<div style='color:#d29922;font-size:12px;margin-top:4px;'>&#9888; 0.75x stop strategy &mdash; LiveStrategySpec does not yet support stop_multiplier. Will trade at 1.0x stops.</div>" if c.get("stop_multiplier", 1.0) < 1.0 else ""}
                </div>
            </div>"""

        cards_html += f"""
        <div class="session-group">
            <div class="session-group-header">{session}</div>
            {rows_html}
        </div>"""

    # Summary banner
    total = len(candidates)
    instruments_found = sorted(set(c["instrument"] for c in candidates))
    top_expr = candidates[0]["expectancy_r"] if candidates else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Promotion Candidates — {now_str}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #0d1117;
        color: #e6edf3;
        padding: 20px;
        max-width: 1400px;
        margin: 0 auto;
    }}
    .header {{
        text-align: center;
        margin-bottom: 30px;
        padding: 20px;
        border-bottom: 2px solid #30363d;
    }}
    .header h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 5px; }}
    .header .subtitle {{ color: #8b949e; font-size: 14px; }}
    .header .highlight {{ color: #58a6ff; font-size: 20px; margin-top: 8px; }}
    .banner {{
        background: #1c2128;
        border: 1px solid #d29922;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 24px;
        text-align: center;
        font-size: 16px;
    }}
    .banner strong {{ color: #d29922; }}
    .session-group {{
        margin-bottom: 32px;
    }}
    .session-group-header {{
        font-size: 20px;
        font-weight: 700;
        color: #58a6ff;
        padding: 8px 0;
        border-bottom: 2px solid #30363d;
        margin-bottom: 16px;
    }}
    .candidate-card {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        margin-bottom: 16px;
        padding: 16px;
    }}
    .candidate-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        flex-wrap: wrap;
        gap: 8px;
    }}
    .candidate-id {{
        font-family: 'Cascadia Code', 'Fira Code', monospace;
        font-size: 14px;
        color: #e6edf3;
        font-weight: 600;
    }}
    .candidate-badges {{ display: flex; gap: 6px; flex-wrap: wrap; }}
    .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }}
    .badge-robust {{ background: #1f3a2a; color: #3fb950; border: 1px solid #3fb950; }}
    .badge-fdr {{ background: #1f2a3a; color: #58a6ff; border: 1px solid #58a6ff; }}
    .badge-wf {{ background: #1f2a3a; color: #58a6ff; border: 1px solid #58a6ff; }}
    .badge-ayp {{ background: #1f3a2a; color: #3fb950; border: 1px solid #3fb950; }}
    .badge-decay-warn {{ background: #3d1f20; color: #f85149; border: 1px solid #f85149; }}
    .badge-overlap-warn {{ background: #3d1f20; color: #f85149; border: 1px solid #f85149; }}
    .badge-overlap-note {{ background: #2d2210; color: #d29922; border: 1px solid #d29922; }}
    .metrics-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
        gap: 8px;
        margin-bottom: 16px;
    }}
    .metric {{
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 8px;
        text-align: center;
    }}
    .metric-label {{ font-size: 11px; color: #8b949e; text-transform: uppercase; }}
    .metric-value {{ font-size: 16px; font-weight: 600; margin-top: 2px; }}
    .metric-expr {{ color: #3fb950; font-size: 18px; }}
    .decay-warn .metric-value {{ color: #f85149; }}
    .section-row {{
        display: flex;
        gap: 16px;
        margin-bottom: 12px;
    }}
    .section-half {{ flex: 1; min-width: 250px; }}
    .section-title {{
        font-size: 13px;
        color: #8b949e;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 6px;
    }}
    .inner-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }}
    .inner-table th {{
        background: #0d1117;
        padding: 4px 8px;
        text-align: left;
        color: #8b949e;
        font-weight: 600;
        border-bottom: 1px solid #21262d;
    }}
    .inner-table td {{
        padding: 4px 8px;
        border-bottom: 1px solid #21262d;
    }}
    .yr-negative td {{ color: #f85149; }}
    .spec-code {{
        margin-top: 8px;
    }}
    .spec-code pre {{
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 10px 12px;
        font-family: 'Cascadia Code', 'Fira Code', monospace;
        font-size: 13px;
        color: #3fb950;
        overflow-x: auto;
        cursor: pointer;
    }}
    .spec-code pre:hover {{
        border-color: #58a6ff;
    }}
    .footer {{
        text-align: center;
        padding: 20px;
        color: #484f58;
        font-size: 12px;
        border-top: 1px solid #30363d;
        margin-top: 20px;
    }}
    @media (max-width: 768px) {{
        .section-row {{ flex-direction: column; }}
        .metrics-grid {{ grid-template-columns: repeat(3, 1fr); }}
    }}
</style>
</head>
<body>
    <div class="header">
        <h1>PROMOTION CANDIDATES</h1>
        <div class="highlight">{total} candidates across {len(sessions)} sessions</div>
        <div class="subtitle">Generated {now_str} &mdash; Instruments: {", ".join(instruments_found)} &mdash; Top ExpR: {top_expr:+.3f}</div>
    </div>

    <div class="banner">
        <strong>{total} new strategies</strong> passed FDR + Walk-Forward + ROBUST edge family gates
        but are not yet in <code>live_config.py</code>. Review each scorecard and decide.
    </div>

    {cards_html}

    <div class="footer">
        Gate criteria: FDR-significant (BH) + Walk-Forward passed + ROBUST family (5+ members, PBO computed) + ExpR &ge; {LIVE_MIN_EXPECTANCY_R}
        &mdash; Source: gold.db edge_families &times; validated_setups
    </div>
</body>
</html>"""
    return html


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate promotion candidate scorecard")
    parser.add_argument(
        "--format",
        choices=["html", "terminal"],
        default="html",
        help="Output format (default: html)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output HTML path")
    parser.add_argument("--db-path", type=Path, default=None, help="Path to gold.db")
    parser.add_argument("--no-open", action="store_true", help="Don't open in browser")
    args = parser.parse_args()

    db_path = args.db_path or GOLD_DB_PATH
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "promotion_candidates.html"

    print("Promotion Candidate Report")
    print(f"  DB: {db_path}")

    candidates = find_uncovered_candidates(db_path)

    day_sets: dict = {}
    if candidates:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            day_sets = build_day_sets(con, candidates)
        finally:
            con.close()

    enriched = [enrich_candidate(c, day_sets=day_sets) for c in candidates]

    if not enriched:
        print("\n  No uncovered ROBUST candidates found. Portfolio is fully covered.")
        return

    print(f"  Found {len(enriched)} promotion candidates")
    print()

    if args.format == "terminal":
        print(format_terminal(enriched))
        return

    html = generate_html(enriched)
    output_path.write_text(html, encoding="utf-8")
    print(f"  Written to {output_path}")

    if not args.no_open:
        webbrowser.open(str(output_path))
        print("  Opened in browser.")


if __name__ == "__main__":
    main()
