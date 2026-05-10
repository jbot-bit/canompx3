#!/usr/bin/env python3
"""MNQ COMEX additivity/replacement gate for live-tradeability expansion.

Bounded follow-up to:
- docs/audit/results/2026-05-10-mnq-production-trade-expansion-proof-gate.md
- docs/audit/results/2026-05-11-mnq-live-tradeability-lens-reset.md

Read-only against canonical DB/runtime state. Writes only result artifacts under
docs/audit/results/. No live allocation, broker, schema, or validated_setups
mutation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from research.portfolio_additivity_engine import (
    LaneSpec,
    PortfolioSnapshot,
    compute_snapshot,
    correlation_rows,
    daily_return_frame,
    fmt_num,
    matching_live_lane,
)
from trading_app.config import WF_START_OVERRIDE
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.lane_correlation import RHO_REJECT_THRESHOLD, SUBSET_REJECT_THRESHOLD, _pearson
from trading_app.prop_profiles import get_profile_lane_definitions
from trading_app.strategy_fitness import _load_strategy_outcomes

PROFILE_ID = "topstep_50k_mnq_auto"
RESULT_MD = Path("docs/audit/results/2026-05-11-mnq-comex-additivity-replacement-gate.md")
RESULT_CSV = Path("docs/audit/results/2026-05-11-mnq-comex-additivity-replacement-gate.csv")

CANDIDATES = [
    LaneSpec(
        strategy_id="MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60",
        instrument="MNQ",
        orb_label="COMEX_SETTLE",
        entry_model="E2",
        rr_target=1.0,
        confirm_bars=1,
        filter_type="X_MES_ATR60",
        orb_minutes=5,
    ),
    LaneSpec(
        strategy_id="MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5",
        instrument="MNQ",
        orb_label="COMEX_SETTLE",
        entry_model="E2",
        rr_target=1.0,
        confirm_bars=1,
        filter_type="ORB_G5",
        orb_minutes=5,
    ),
]


@dataclass(frozen=True)
class CandidateGate:
    candidate: LaneSpec
    standalone_is: PortfolioSnapshot
    standalone_oos: PortfolioSnapshot
    add_is: PortfolioSnapshot
    add_oos: PortfolioSnapshot
    replace_is: PortfolioSnapshot | None
    replace_oos: PortfolioSnapshot | None
    replaced_lane: LaneSpec | None
    corr_report_gate_pass: bool
    corr_report_worst_rho: float
    corr_report_worst_subset: float
    corr_report_reject_reasons: tuple[str, ...]
    corr_to_book: float | None
    overlap_pct: float
    decision: str
    reason: str


def _delta(a: float | None, b: float | None) -> float:
    if a is None or b is None or (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
        return float("nan")
    return float(a - b)


def _snapshot_row(snapshot: PortfolioSnapshot) -> str:
    return (
        f"| {snapshot.label} | {snapshot.start_date} | {snapshot.end_date} | "
        f"{snapshot.business_days} | {snapshot.total_trades} | {snapshot.trade_days} | "
        f"{fmt_num(snapshot.total_r, 1, signed=True)} | {fmt_num(snapshot.annual_r, 1, signed=True)} | "
        f"{fmt_num(snapshot.sharpe_ann, 3, signed=True)} | {fmt_num(snapshot.max_dd_r, 1)} | "
        f"{fmt_num(snapshot.worst_single_day_r, 1, signed=True)} | {fmt_num(snapshot.avg_trades_per_day, 3)} |"
    )


def _live_specs() -> list[LaneSpec]:
    specs: list[LaneSpec] = []
    for lane in get_profile_lane_definitions(PROFILE_ID):
        specs.append(
            LaneSpec(
                strategy_id=str(lane["strategy_id"]),
                instrument=str(lane["instrument"]),
                orb_label=str(lane["orb_label"]),
                entry_model=str(lane["entry_model"]),
                rr_target=float(lane["rr_target"]),
                confirm_bars=int(lane["confirm_bars"]),
                filter_type=str(lane["filter_type"]),
                orb_minutes=int(lane["orb_minutes"]),
            )
        )
    if not specs:
        raise RuntimeError(f"Profile {PROFILE_ID!r} has no live lane definitions")
    return specs


def _load_trades(con: duckdb.DuckDBPyConnection, spec: LaneSpec) -> list[dict[str, Any]]:
    start_date = WF_START_OVERRIDE.get(spec.instrument)
    outcomes = _load_strategy_outcomes(
        con,
        instrument=spec.instrument,
        orb_label=spec.orb_label,
        orb_minutes=spec.orb_minutes,
        entry_model=spec.entry_model,
        rr_target=spec.rr_target,
        confirm_bars=spec.confirm_bars,
        filter_type=spec.filter_type,
        start_date=start_date,
    )
    trades: list[dict[str, Any]] = []
    for row in outcomes:
        pnl_r = row.get("pnl_r")
        if pnl_r is None:
            continue
        td = row["trading_day"]
        trades.append(
            {
                "trading_day": td.date() if hasattr(td, "date") else td,
                "outcome": row.get("outcome"),
                "pnl_r": float(pnl_r),
                "instrument": spec.instrument,
                "session": spec.orb_label,
                "strategy_id": spec.strategy_id,
            }
        )
    if not trades:
        raise RuntimeError(f"FAIL-CLOSED: zero trades loaded for {spec.strategy_id}")
    return trades


def _common_window(trades_by_strategy: dict[str, list[dict[str, Any]]]) -> tuple[Any, Any]:
    starts = []
    ends = []
    for trades in trades_by_strategy.values():
        days = [t["trading_day"] for t in trades]
        if days:
            starts.append(min(days))
            ends.append(max(days))
    if not starts or not ends:
        raise RuntimeError("No trades loaded for gate")
    return max(starts), min(ends)


def _daily_pnl(trades: list[dict[str, Any]]) -> dict[Any, float]:
    by_day: dict[Any, float] = {}
    for trade in trades:
        day = trade["trading_day"]
        by_day[day] = by_day.get(day, 0.0) + float(trade["pnl_r"])
    return by_day


def _pair_correlation_gate(
    candidate: LaneSpec,
    live_specs: list[LaneSpec],
    trades_all: dict[str, list[dict[str, Any]]],
    *,
    start_date: Any,
    end_date: Any,
) -> tuple[bool, float, float, tuple[str, ...]]:
    candidate_pnl = {
        day: pnl
        for day, pnl in _daily_pnl(trades_all[candidate.strategy_id]).items()
        if start_date <= day <= end_date
    }
    reject_reasons: list[str] = []
    worst_rho = 0.0
    worst_subset = 0.0
    for live in live_specs:
        live_pnl = {
            day: pnl
            for day, pnl in _daily_pnl(trades_all[live.strategy_id]).items()
            if start_date <= day <= end_date
        }
        shared = sorted(set(candidate_pnl) & set(live_pnl))
        smaller = min(len(candidate_pnl), len(live_pnl))
        subset = len(shared) / smaller if smaller else 0.0
        rho = 0.0
        if len(shared) >= 5:
            rho = _pearson([candidate_pnl[d] for d in shared], [live_pnl[d] for d in shared])
        worst_rho = max(worst_rho, rho)
        worst_subset = max(worst_subset, subset)

        same_session = candidate.orb_label == live.orb_label and candidate.instrument == live.instrument
        reasons = []
        if rho > RHO_REJECT_THRESHOLD:
            reasons.append(f"rho={rho:.3f}>{RHO_REJECT_THRESHOLD}")
        if same_session and subset > SUBSET_REJECT_THRESHOLD:
            reasons.append(f"subset={subset:.1%}>{SUBSET_REJECT_THRESHOLD:.0%}")
        if reasons:
            reject_reasons.append(f"{live.strategy_id}: {'; '.join(reasons)}")
    return len(reject_reasons) == 0, worst_rho, worst_subset, tuple(reject_reasons)


def _decide(
    *,
    candidate: LaneSpec,
    base_is: PortfolioSnapshot,
    standalone_is: PortfolioSnapshot,
    add_is: PortfolioSnapshot,
    replace_is: PortfolioSnapshot | None,
    replaced_lane: LaneSpec | None,
    corr_gate_pass: bool,
    corr_reasons: tuple[str, ...],
) -> tuple[str, str]:
    add_annual = _delta(add_is.annual_r, base_is.annual_r)
    add_sharpe = _delta(add_is.sharpe_ann, base_is.sharpe_ann)
    repl_annual = _delta(replace_is.annual_r, base_is.annual_r) if replace_is else float("nan")
    repl_sharpe = _delta(replace_is.sharpe_ann, base_is.sharpe_ann) if replace_is else float("nan")

    if standalone_is.annual_r <= 0:
        return "FAIL", "Standalone common-window IS annualized R is not positive."

    if replaced_lane is not None:
        if repl_annual > 0 and repl_sharpe > 0:
            return (
                "PASS_REPLACE",
                "Same-session add is not runtime-clean, but replacement improves common-window IS annualized R and honest Sharpe.",
            )
        return (
            "PARK",
            "Candidate is same-session with the live COMEX lane and replacement does not improve both common-window IS annualized R and honest Sharpe.",
        )

    if add_annual > 0 and add_sharpe > 0 and corr_gate_pass:
        return "PASS_ADD", "Additive route improves common-window IS annualized R and honest Sharpe and passes correlation gate."

    if not corr_gate_pass:
        return "PARK", "Additive route is blocked by canonical correlation gate: " + "; ".join(corr_reasons)

    return "PARK", "Additive route does not improve both common-window IS annualized R and honest Sharpe."


def _run() -> tuple[list[LaneSpec], PortfolioSnapshot, PortfolioSnapshot | None, list[CandidateGate], dict[str, Any]]:
    live_specs = _live_specs()
    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        configure_connection(con)

        all_specs = live_specs + CANDIDATES
        trades_all = {spec.strategy_id: _load_trades(con, spec) for spec in all_specs}
        start_all, end_all = _common_window(trades_all)
        is_end = min(end_all, HOLDOUT_SACRED_FROM - timedelta(days=1))
        if start_all > is_end:
            raise RuntimeError("No in-sample window available")
        oos_start = HOLDOUT_SACRED_FROM
        has_oos = oos_start <= end_all

        base_trades = {spec.strategy_id: trades_all[spec.strategy_id] for spec in live_specs}
        base_is = compute_snapshot("Current live book", base_trades, start_all, is_end)
        base_oos = compute_snapshot("Current live book", base_trades, oos_start, end_all) if has_oos else None

        gates: list[CandidateGate] = []
        for candidate in CANDIDATES:
            standalone_trades = {candidate.strategy_id: trades_all[candidate.strategy_id]}
            add_trades = dict(base_trades)
            add_trades[candidate.strategy_id] = trades_all[candidate.strategy_id]

            standalone_is = compute_snapshot(f"{candidate.strategy_id} standalone", standalone_trades, start_all, is_end)
            add_is = compute_snapshot(f"Current live book + {candidate.strategy_id}", add_trades, start_all, is_end)
            standalone_oos = compute_snapshot(
                f"{candidate.strategy_id} standalone",
                standalone_trades,
                oos_start,
                end_all,
            )
            add_oos = compute_snapshot(f"Current live book + {candidate.strategy_id}", add_trades, oos_start, end_all)

            frame_is = daily_return_frame(add_trades, start_all, is_end)
            _corr_rows, corr_summary = correlation_rows(frame_is, candidate.strategy_id)

            replaced_lane = matching_live_lane(candidate, live_specs)
            replace_is = None
            replace_oos = None
            if replaced_lane is not None:
                replace_trades = dict(base_trades)
                replace_trades.pop(replaced_lane.strategy_id, None)
                replace_trades[candidate.strategy_id] = trades_all[candidate.strategy_id]
                replace_is = compute_snapshot(
                    f"Replace {replaced_lane.strategy_id} with {candidate.strategy_id}",
                    replace_trades,
                    start_all,
                    is_end,
                )
                replace_oos = compute_snapshot(
                    f"Replace {replaced_lane.strategy_id} with {candidate.strategy_id}",
                    replace_trades,
                    oos_start,
                    end_all,
                )

            corr_gate_pass, worst_rho, worst_subset, reject_reasons = _pair_correlation_gate(
                candidate,
                live_specs,
                trades_all,
                start_date=start_all,
                end_date=is_end,
            )
            decision, reason = _decide(
                candidate=candidate,
                base_is=base_is,
                standalone_is=standalone_is,
                add_is=add_is,
                replace_is=replace_is,
                replaced_lane=replaced_lane,
                corr_gate_pass=corr_gate_pass,
                corr_reasons=reject_reasons,
            )

            gates.append(
                CandidateGate(
                    candidate=candidate,
                    standalone_is=standalone_is,
                    standalone_oos=standalone_oos,
                    add_is=add_is,
                    add_oos=add_oos,
                    replace_is=replace_is,
                    replace_oos=replace_oos,
                    replaced_lane=replaced_lane,
                    corr_report_gate_pass=corr_gate_pass,
                    corr_report_worst_rho=worst_rho,
                    corr_report_worst_subset=worst_subset,
                    corr_report_reject_reasons=reject_reasons,
                    corr_to_book=corr_summary.corr_to_base_portfolio,
                    overlap_pct=corr_summary.pct_candidate_days_with_live_overlap,
                    decision=decision,
                    reason=reason,
                )
            )

    meta = {
        "common_start": start_all,
        "common_end": end_all,
        "is_end": is_end,
        "oos_start": oos_start,
        "has_oos": has_oos,
    }
    return live_specs, base_is, base_oos, gates, meta


def _render(live_specs: list[LaneSpec], base_is: PortfolioSnapshot, base_oos: PortfolioSnapshot | None, gates: list[CandidateGate], meta: dict[str, Any]) -> str:
    parts = [
        "# MNQ COMEX Additivity / Replacement Gate",
        "",
        "**Date:** 2026-05-11",
        f"**Profile:** `{PROFILE_ID}`",
        "**Live impact:** None. No DB, schema, broker, validated-setups, or lane-allocation mutation.",
        "",
        "## Scope",
        "",
        "This is the close-first gate opened by the widened-lens reset. It tests the only two admissible COMEX candidates left by the 2026-05-10 proof gate:",
        "",
        "- `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60`",
        "- `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5`",
        "",
        "Truth sources:",
        "",
        "- canonical `gold.db::orb_outcomes`",
        "- canonical `gold.db::daily_features`",
        "- current profile lane definitions / `docs/runtime/lane_allocation.json` context",
        "- canonical filter application via `trading_app.strategy_fitness._load_strategy_outcomes`",
        "- canonical correlation gate algorithm/thresholds from `trading_app.lane_correlation`, applied on the same WF-eligible sample",
        "- instrument eligibility starts from `trading_app.config.WF_START_OVERRIDE`",
        "",
        "## Current Book",
        "",
    ]
    for spec in live_specs:
        parts.append(f"- `{spec.strategy_id}`")
    parts.extend(
        [
            "",
            "## Common Window",
            "",
            f"- Common start: `{meta['common_start']}`",
            f"- Common end: `{meta['common_end']}`",
            f"- IS end: `{meta['is_end']}`",
            f"- OOS monitor start: `{meta['oos_start']}`",
            "",
            "## Current Book Baseline",
            "",
            "| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            _snapshot_row(base_is),
        ]
    )
    if base_oos is not None:
        parts.extend(["", "### 2026 OOS Monitor", "", _snapshot_row(base_oos)])

    parts.extend(
        [
            "",
            "## Decision Matrix",
            "",
            "| Candidate | Decision | Add Delta Annual R IS | Add Delta Sharpe IS | Replacement Target | Replace Delta Annual R IS | Replace Delta Sharpe IS | Corr Gate | Worst Rho | Worst Subset |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for gate in gates:
        repl = gate.replaced_lane.strategy_id if gate.replaced_lane else "-"
        repl_ann = _delta(gate.replace_is.annual_r, base_is.annual_r) if gate.replace_is else float("nan")
        repl_sharpe = _delta(gate.replace_is.sharpe_ann, base_is.sharpe_ann) if gate.replace_is else float("nan")
        parts.append(
            f"| `{gate.candidate.strategy_id}` | `{gate.decision}` | "
            f"{fmt_num(_delta(gate.add_is.annual_r, base_is.annual_r), 1, signed=True)} | "
            f"{fmt_num(_delta(gate.add_is.sharpe_ann, base_is.sharpe_ann), 3, signed=True)} | "
            f"`{repl}` | {fmt_num(repl_ann, 1, signed=True)} | {fmt_num(repl_sharpe, 3, signed=True)} | "
            f"`{gate.corr_report_gate_pass}` | {fmt_num(gate.corr_report_worst_rho, 3, signed=True)} | "
            f"{fmt_num(gate.corr_report_worst_subset * 100, 1)}% |"
        )

    for gate in gates:
        parts.extend(
            [
                "",
                f"## {gate.candidate.strategy_id}",
                "",
                f"- Decision: `{gate.decision}`",
                f"- Reason: {gate.reason}",
                f"- Corr to aggregate current-book daily R: `{fmt_num(gate.corr_to_book, 3, signed=True)}`",
                f"- Candidate days overlapping any live lane: `{fmt_num(gate.overlap_pct * 100, 1)}%`",
                f"- Canonical correlation gate pass: `{gate.corr_report_gate_pass}`",
                f"- Correlation reject reasons: `{'; '.join(gate.corr_report_reject_reasons) if gate.corr_report_reject_reasons else 'none'}`",
                "",
                "### IS Snapshots",
                "",
                "| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                _snapshot_row(gate.standalone_is),
                _snapshot_row(gate.add_is),
            ]
        )
        if gate.replace_is is not None:
            parts.append(_snapshot_row(gate.replace_is))
        parts.extend(
            [
                "",
                "### 2026 OOS Monitor Only",
                "",
                "| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                _snapshot_row(gate.standalone_oos),
                _snapshot_row(gate.add_oos),
            ]
        )
        if gate.replace_oos is not None:
            parts.append(_snapshot_row(gate.replace_oos))

    parts.extend(
        [
            "",
            "## Verdict",
            "",
            f"Final classifications: {', '.join(f'`{gate.candidate.strategy_id}` = `{gate.decision}`' for gate in gates)}.",
            "",
            "No candidate is automatically production-live from this gate. `PASS_REPLACE` means the candidate earns a separate operator preflight / allocation-change review; it is not permission to mutate live state in this pass.",
            "",
            "Multiple-testing accounting: fixed K=2 from the proof gate. This pass is role classification against current book, not fresh discovery or threshold search.",
            "",
            "Runtime interpretation: both candidates are same-session COMEX O5 alternatives to the deployed COMEX lane. Same-session ADD is not a clean current-stack route; the only admissible live-tradeability role here is replacement, not additive coexistence.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "./.venv-wsl/bin/python research/mnq_comex_additivity_replacement_gate_2026_05_11.py",
            "```",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> int:
    live_specs, base_is, base_oos, gates, meta = _run()
    RESULT_MD.write_text(_render(live_specs, base_is, base_oos, gates, meta), encoding="utf-8")
    with RESULT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "strategy_id",
                "decision",
                "reason",
                "add_delta_annual_r_is",
                "add_delta_sharpe_is",
                "replacement_target",
                "replace_delta_annual_r_is",
                "replace_delta_sharpe_is",
                "corr_gate_pass",
                "worst_rho",
                "worst_subset",
            ]
        )
        for gate in gates:
            writer.writerow(
                [
                    gate.candidate.strategy_id,
                    gate.decision,
                    gate.reason,
                    _delta(gate.add_is.annual_r, base_is.annual_r),
                    _delta(gate.add_is.sharpe_ann, base_is.sharpe_ann),
                    gate.replaced_lane.strategy_id if gate.replaced_lane else "",
                    _delta(gate.replace_is.annual_r, base_is.annual_r) if gate.replace_is else "",
                    _delta(gate.replace_is.sharpe_ann, base_is.sharpe_ann) if gate.replace_is else "",
                    gate.corr_report_gate_pass,
                    gate.corr_report_worst_rho,
                    gate.corr_report_worst_subset,
                ]
            )
    print(f"Wrote {RESULT_MD}")
    print(f"Wrote {RESULT_CSV}")
    for gate in gates:
        print(f"{gate.candidate.strategy_id}: {gate.decision} | {gate.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
