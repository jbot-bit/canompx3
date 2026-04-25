#!/usr/bin/env python3
"""Prior-day geometry routing/additivity audit for promoted MNQ shelf rows.

Question locked in `docs/runtime/stages/prior-day-geometry-routing-audit.md`:
after the exact prior-day bridge hypotheses were consumed and five broader
geometry rows reached the validated shelf, which role do they actually belong
in: route live, keep on shelf, or park as non-additive?

Truth:
- canonical trade outcomes: `orb_outcomes`
- canonical feature/filter state: `daily_features`

Comparison-only context:
- `docs/runtime/lane_allocation.json`
- `validated_setups`
- `trading_app/prop_profiles.py`

No writes to validated_setups / live config / lane_allocation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from research.portfolio_additivity_engine import (
    CandidateAdditivityAudit,
    CorrelationRow,
    CorrelationSummary,
    PortfolioSnapshot,
    fmt_num,
    run_additivity_audit,
)

PROFILE_ID = "topstep_50k_mnq_auto"
RESULT_PATH = Path("docs/audit/results/2026-04-23-prior-day-geometry-routing-audit.md")
CANDIDATE_IDS = [
    "MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG",
    "MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG",
    "MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG",
    "MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG",
    "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG",
]


def _delta(a: float | None, b: float | None) -> float:
    if a is None or b is None or (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
        return float("nan")
    return float(a - b)


def _snapshot_table_row(snapshot: PortfolioSnapshot) -> str:
    return (
        f"| {snapshot.label} | {snapshot.start_date} | {snapshot.end_date} | "
        f"{snapshot.business_days} | {snapshot.total_trades} | {snapshot.trade_days} | "
        f"{fmt_num(snapshot.total_r, 1, signed=True)} | {fmt_num(snapshot.annual_r, 1, signed=True)} | "
        f"{fmt_num(snapshot.sharpe_ann, 3, signed=True)} | {fmt_num(snapshot.max_dd_r, 1)} | "
        f"{fmt_num(snapshot.worst_single_day_r, 1, signed=True)} | {fmt_num(snapshot.avg_trades_per_day, 3)} |"
    )


def _candidate_score(row: CandidateAdditivityAudit, base_is: PortfolioSnapshot) -> tuple[float, float]:
    return (
        _delta(row.add_is.annual_r, base_is.annual_r),
        _delta(row.add_is.sharpe_ann, base_is.sharpe_ann),
    )


def _qualifies_for_add_route(row: CandidateAdditivityAudit, base_is: PortfolioSnapshot) -> bool:
    delta_annual, delta_sharpe = _candidate_score(row, base_is)
    return (
        row.profile_fit.current_profile_allowed
        and
        row.replaced_lane is None
        and
        row.standalone_is.annual_r > 0
        and delta_annual > 0
        and delta_sharpe > 0
    )


def _qualifies_for_replace_route(row: CandidateAdditivityAudit, base_is: PortfolioSnapshot) -> bool:
    if row.replace_is is None:
        return False
    delta_annual = _delta(row.replace_is.annual_r, base_is.annual_r)
    delta_sharpe = _delta(row.replace_is.sharpe_ann, base_is.sharpe_ann)
    return row.standalone_is.annual_r > 0 and delta_annual > 0 and delta_sharpe > 0


def decide_rows(
    rows: list[CandidateAdditivityAudit],
    base_is: PortfolioSnapshot,
    max_slots: int,
    live_count: int,
) -> dict[str, dict[str, Any]]:
    free_slots = max(0, max_slots - live_count)
    add_candidates = [row for row in rows if _qualifies_for_add_route(row, base_is)]
    add_candidates.sort(key=lambda row: _candidate_score(row, base_is), reverse=True)
    route_ids = {row.candidate.strategy_id for row in add_candidates[:free_slots]}

    decisions: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = row.candidate.strategy_id
        delta_annual = _delta(row.add_is.annual_r, base_is.annual_r)
        delta_sharpe = _delta(row.add_is.sharpe_ann, base_is.sharpe_ann)
        repl_annual = _delta(row.replace_is.annual_r, base_is.annual_r) if row.replace_is else float("nan")
        repl_sharpe = _delta(row.replace_is.sharpe_ann, base_is.sharpe_ann) if row.replace_is else float("nan")
        if sid in route_ids:
            decision = "ROUTE_LIVE"
            reason = (
                "Clears additive free-slot test on the common IS window: annualized R and honest Sharpe improve, "
                "the candidate fits the current auto profile's static routing gates, and it ranks first among current add candidates."
            )
        elif row.replaced_lane is not None and row.standalone_is.annual_r > 0 and delta_annual > 0 and delta_sharpe > 0:
            decision = "KEEP_ON_SHELF"
            reason = (
                "Research-only additive math is positive, but this is a same-session live-book collision. Under the current runtime, same-session "
                "adds are not a clean free-slot route: same-aperture adds are blocked and different-aperture adds require execution translation / "
                "size-down handling. Keep it on shelf until that translation is modeled or shadow-tested."
            )
        elif _qualifies_for_replace_route(row, base_is):
            decision = "KEEP_ON_SHELF"
            reason = (
                "Replacement math is positive, but the current profile has a free slot and this candidate did not win the additive ranking. "
                "Keep it as a shelf substitute candidate, not an immediate live route."
            )
        elif row.standalone_is.annual_r > 0:
            decision = "KEEP_ON_SHELF"
            reason = (
                "Standalone edge remains positive, but additive portfolio math is weaker than the live-route winner or does not clear the "
                "clean add test. Valid shelf row, not the first routing candidate. Do not treat this as a kill; it stays visible for other-profile "
                "or manual-review consideration."
            )
        else:
            decision = "PARK_NON_ADDITIVE"
            reason = "Does not retain positive standalone contribution on the common IS window."
        decisions[sid] = {
            "decision": decision,
            "reason": reason,
            "delta_annual_add": delta_annual,
            "delta_sharpe_add": delta_sharpe,
            "delta_annual_replace": repl_annual,
            "delta_sharpe_replace": repl_sharpe,
        }
    return decisions


def _corr_row_for_lane(rows: list[CorrelationRow], strategy_id: str) -> CorrelationRow | None:
    for row in rows:
        if row.strategy_id == strategy_id:
            return row
    return None


def _render_corr_summary(summary: CorrelationSummary) -> list[str]:
    return [
        f"- Corr to aggregate live-book daily R: `{fmt_num(summary.corr_to_base_portfolio, 3, signed=True)}`",
        f"- Candidate trade days: `{summary.candidate_trade_days}`",
        f"- Candidate days overlapping any live lane: `{summary.days_with_any_live_overlap}` "
        f"(`{fmt_num(summary.pct_candidate_days_with_live_overlap * 100 if not np.isnan(summary.pct_candidate_days_with_live_overlap) else float('nan'), 1)}%`)",
        f"- Avg live lanes active on candidate days: `{fmt_num(summary.avg_live_lanes_active_on_candidate_days, 3)}`",
    ]


def _render_profile_fit(row: CandidateAdditivityAudit) -> list[str]:
    active_other = [p for p in row.profile_fit.active_profile_fits if p != PROFILE_ID]
    inactive = list(row.profile_fit.inactive_profile_fits)
    lines = [
        f"- Fits current auto profile static gates: `{'YES' if row.profile_fit.current_profile_allowed else 'NO'}`",
        f"- Other active profile fits: `{', '.join(active_other) if active_other else 'none'}`",
        f"- Inactive configured profile fits: `{', '.join(inactive) if inactive else 'none'}`",
    ]
    if row.standalone_is.annual_r > 0:
        lines.append(
            "- Watch status if not auto-routed: `KEEP_VISIBLE` "
            "(positive shelf row; preserve for other-profile or manual review rather than collapsing to dead)"
        )
    else:
        lines.append("- Watch status if not auto-routed: `NO`")
    return lines


def render_doc(payload: dict[str, Any], decisions: dict[str, dict[str, Any]]) -> str:
    base_is: PortfolioSnapshot = payload["base_is"]
    base_oos: PortfolioSnapshot | None = payload["base_oos"]
    live_specs = payload["live_specs"]
    rows: list[CandidateAdditivityAudit] = payload["candidates"]
    max_slots = payload["max_slots"]
    live_count = len(live_specs)
    route_rows = [row for row in rows if decisions[row.candidate.strategy_id]["decision"] == "ROUTE_LIVE"]

    parts = [
        "# Prior-Day Geometry Routing Audit",
        "",
        "Date: 2026-04-23",
        "",
        "## Scope",
        "",
        "Resolve the role of the five promoted MNQ prior-day geometry shelf survivors against the current live MNQ book.",
        "",
        "Truth used for this audit:",
        "",
        "- `gold.db::orb_outcomes`",
        "- `gold.db::daily_features`",
        "",
        "Comparison-only deployment context:",
        "",
        "- `docs/runtime/lane_allocation.json`",
        "- `validated_setups` for exact live-lane and candidate parameters",
        "- `trading_app/prop_profiles.py`",
        "",
        "## Live Book Context",
        "",
        f"- Profile: `{PROFILE_ID}`",
        f"- Current live lane count: `{live_count}`",
        f"- `max_slots={max_slots}`",
        f"- Free slots before any replacement question: `{max(0, max_slots - live_count)}`",
        "",
        "Live lanes reconstructed in this audit:",
        "",
    ]
    for spec in live_specs:
        parts.append(
            f"- `{spec.strategy_id}` -> {spec.orb_label} O{spec.orb_minutes} RR{spec.rr_target} {spec.filter_type}"
        )

    parts.extend(
        [
            "",
            "## Current Live Book Baseline (IS)",
            "",
            "| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            _snapshot_table_row(base_is),
        ]
    )
    if base_oos is not None:
        parts.extend(
            [
                "",
                "## Current Live Book Baseline (2026 OOS Monitor Only)",
                "",
                "| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                _snapshot_table_row(base_oos),
            ]
        )

    parts.extend(
        [
            "",
            "## Decision Summary",
            "",
            "| Candidate | Session | RR | Standalone Annual R IS | Add Δ Annual R IS | Add Δ Sharpe IS | Replacement target | Replace Δ Annual R IS | Replace Δ Sharpe IS | Decision |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|---|",
        ]
    )
    for row in rows:
        d = decisions[row.candidate.strategy_id]
        replaced = row.replaced_lane.strategy_id if row.replaced_lane else "-"
        parts.append(
            f"| `{row.candidate.strategy_id}` | {row.candidate.orb_label} | {row.candidate.rr_target:.1f} | "
            f"{fmt_num(row.standalone_is.annual_r, 1, signed=True)} | "
            f"{fmt_num(d['delta_annual_add'], 1, signed=True)} | "
            f"{fmt_num(d['delta_sharpe_add'], 3, signed=True)} | "
            f"`{replaced}` | "
            f"{fmt_num(d['delta_annual_replace'], 1, signed=True)} | "
            f"{fmt_num(d['delta_sharpe_replace'], 3, signed=True)} | "
            f"`{d['decision']}` |"
        )

    if route_rows:
        winner = route_rows[0]
        parts.extend(
            [
                "",
                "## Primary Routing Winner",
                "",
                f"- Winner: `{winner.candidate.strategy_id}`",
                f"- Reason: {decisions[winner.candidate.strategy_id]['reason']}",
            ]
        )
    else:
        parts.extend(
            [
                "",
                "## Primary Routing Winner",
                "",
                "- No candidate cleared the additive free-slot route test.",
            ]
        )

    for row in rows:
        d = decisions[row.candidate.strategy_id]
        parts.extend(
            [
                "",
                f"## {row.candidate.strategy_id}",
                "",
                f"- Decision: `{d['decision']}`",
                f"- Reason: {d['reason']}",
                "",
                "### Standalone / Add / Replace (IS)",
                "",
                "| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                _snapshot_table_row(row.standalone_is),
                _snapshot_table_row(row.add_is),
            ]
        )
        if row.replace_is is not None:
            parts.append(_snapshot_table_row(row.replace_is))
        parts.extend(["", "### Profile / Routing Fit", "", *_render_profile_fit(row)])
        parts.extend(
            [
                "",
                f"- Additive annualized R delta vs live book: `{fmt_num(d['delta_annual_add'], 1, signed=True)}`",
                f"- Additive honest Sharpe delta vs live book: `{fmt_num(d['delta_sharpe_add'], 3, signed=True)}`",
            ]
        )
        if row.replace_is is not None and row.replaced_lane is not None:
            parts.extend(
                [
                    f"- Same-session replacement target: `{row.replaced_lane.strategy_id}`",
                    f"- Replacement annualized R delta vs live book: `{fmt_num(d['delta_annual_replace'], 1, signed=True)}`",
                    f"- Replacement honest Sharpe delta vs live book: `{fmt_num(d['delta_sharpe_replace'], 3, signed=True)}`",
                ]
            )
        parts.extend(
            [
                "",
                "### Diversification vs Live Book (IS)",
                "",
                "| Live lane | Daily corr | Overlap days | Overlap % of candidate days |",
                "|---|---:|---:|---:|",
            ]
        )
        for corr_row in row.corr_rows_is:
            parts.append(
                f"| `{corr_row.strategy_id}` | {fmt_num(corr_row.corr_daily_r, 3, signed=True)} | "
                f"{corr_row.overlap_days} | "
                f"{fmt_num(corr_row.overlap_pct_of_candidate_days * 100 if not np.isnan(corr_row.overlap_pct_of_candidate_days) else float('nan'), 1)}% |"
            )
        parts.extend(["", *_render_corr_summary(row.corr_summary_is)])

        if row.replaced_lane is not None:
            session_corr = _corr_row_for_lane(row.corr_rows_is, row.replaced_lane.strategy_id)
            if session_corr is not None:
                parts.extend(
                    [
                        "",
                        "### Same-Session Substitution View",
                        "",
                        f"- Current live same-session lane: `{row.replaced_lane.strategy_id}`",
                        f"- Corr vs current same-session lane: `{fmt_num(session_corr.corr_daily_r, 3, signed=True)}`",
                        f"- Overlap vs current same-session lane: `{session_corr.overlap_days}` days "
                        f"(`{fmt_num(session_corr.overlap_pct_of_candidate_days * 100 if not np.isnan(session_corr.overlap_pct_of_candidate_days) else float('nan'), 1)}%` of candidate days)",
                    ]
                )

        parts.extend(
            [
                "",
                "### 2026 OOS Monitor Only",
                "",
                "| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                _snapshot_table_row(row.standalone_oos),
                _snapshot_table_row(row.add_oos),
            ]
        )
        if row.replace_oos is not None:
            parts.append(_snapshot_table_row(row.replace_oos))

    parts.extend(
        [
            "",
            "## Verdict",
            "",
            "The prior-day geometry branch is no longer a discovery problem.",
            "It is a routing problem.",
            "",
            "Selection rule used here:",
            "",
            "- free-slot additive test first",
            "- same-session candidates are not treated as direct free-slot adds under the current runtime",
            "- current auto-profile static gates must allow the candidate",
            "- same-session replacement view second",
            "- positive shelf rows that are not first-route winners remain visible for other-profile / manual-review use",
            "- 2026 OOS monitor-only, not selection proof",
            "",
            "## Reproduction",
            "",
            "```bash",
            "./.venv-wsl/bin/python research/prior_day_geometry_routing_audit.py",
            "```",
            "",
            "No randomness. Read-only DB. No writes to `validated_setups` / `experimental_strategies` / `live_config` / `lane_allocation.json`.",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prior-day geometry routing/additivity audit")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_PATH,
        help="Write markdown result to this path",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print markdown result to stdout instead of writing a file",
    )
    args = parser.parse_args()

    payload = run_additivity_audit(PROFILE_ID, CANDIDATE_IDS)
    decisions = decide_rows(
        rows=payload["candidates"],
        base_is=payload["base_is"],
        max_slots=payload["max_slots"],
        live_count=len(payload["live_specs"]),
    )
    rendered = render_doc(payload, decisions)
    if args.stdout:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    route_live = [sid for sid, d in decisions.items() if d["decision"] == "ROUTE_LIVE"]
    if not args.stdout:
        print(f"Wrote {args.output}")
    print(f"ROUTE_LIVE: {route_live if route_live else 'none'}")
    for sid, d in decisions.items():
        print(f"{sid}: {d['decision']} | {d['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
