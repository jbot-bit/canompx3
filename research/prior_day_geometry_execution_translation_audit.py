#!/usr/bin/env python3
"""Prior-day geometry execution translation audit for MNQ US_DATA_1000.

Question locked in:
  docs/runtime/stages/prior-day-geometry-execution-translation-audit.md

This is not another discovery pass. It answers a narrower runtime question:

  if a positive same-session O5 prior-day geometry row is added beside the live
  MNQ US_DATA_1000 O15 lane, what actually happens under the current runtime?

Truth:
- canonical trade outcomes: orb_outcomes
- canonical filter state: daily_features

Comparison/runtime context:
- validated_setups
- trading_app/prop_profiles.py
- trading_app/risk_manager.py
- trading_app/execution_engine.py

No writes to validated_setups / live config / lane_allocation.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from research.comprehensive_deployed_lane_scan import compute_deployed_filter, load_lane
from research.portfolio_additivity_engine import load_candidate_specs, load_live_lane_specs
from research.research_portfolio_assembly import build_daily_equity, compute_drawdown, compute_honest_sharpe
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.risk_manager import RiskLimits, RiskManager

logging.getLogger("trading_app.risk_manager").setLevel(logging.ERROR)

PROFILE_ID = "topstep_50k_mnq_auto"
RESULT_PATH = Path("docs/audit/results/2026-04-23-prior-day-geometry-execution-translation-audit.md")
US_DATA_1000_CANDIDATE_IDS = [
    "MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG",
    "MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG",
    "MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG",
    "MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG",
]
INCUMBENT_ID = "MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15"
OOS_START = HOLDOUT_SACRED_FROM
OOS_END = pd.Timestamp("2026-04-07").date()


@dataclass(frozen=True)
class TimedTrade:
    strategy_id: str
    instrument: str
    orb_label: str
    orb_minutes: int
    rr_target: float
    filter_type: str
    trading_day: date
    direction: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    pnl_r: float


@dataclass
class ActiveTrade:
    timed: TimedTrade
    contracts: int = 1

    def __post_init__(self) -> None:
        self.strategy_id = self.timed.strategy_id
        self.orb_label = self.timed.orb_label
        self.orb_minutes = self.timed.orb_minutes
        self.direction = self.timed.direction
        self.instrument = self.timed.instrument
        self.strategy = SimpleNamespace(instrument=self.timed.instrument)
        self.state = SimpleNamespace(value="ENTERED")


@dataclass(frozen=True)
class Snapshot:
    label: str
    start_date: date
    end_date: date
    total_r: float
    annual_r: float
    sharpe_ann: float | None
    max_dd_r: float
    worst_single_day_r: float
    total_trades: int
    business_days: int
    avg_trades_per_day: float


@dataclass(frozen=True)
class PairOverlapSummary:
    candidate_days: int
    incumbent_days: int
    shared_days: int
    time_overlap_days: int
    time_overlap_same_dir_days: int
    time_overlap_opp_dir_days: int
    candidate_active_at_incumbent_entry_days: int
    incumbent_active_at_candidate_entry_days: int
    median_entry_gap_min: float | None
    median_overlap_min: float | None


@dataclass(frozen=True)
class TranslationOutcome:
    candidate_id: str
    candidate_filter: str
    standalone_is: Snapshot
    standalone_oos: Snapshot | None
    translated_is: Snapshot
    translated_oos: Snapshot | None
    delta_annual_is: float
    delta_sharpe_is: float
    delta_annual_oos: float | None
    delta_sharpe_oos: float | None
    pair_overlap: PairOverlapSummary
    candidate_accepted: int
    candidate_blocked: int
    incumbent_block_delta: int
    other_live_block_delta: int
    same_session_halfsize_suggested: int
    same_session_halfsize_effective_noop: int
    max_open_positions: int
    max_open_us_data_positions: int
    block_reasons: tuple[tuple[str, int], ...]
    verdict: str
    next_step: str


def fmt_num(value: float | None, digits: int = 1, signed: bool = False) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    spec = f"+.{digits}f" if signed else f".{digits}f"
    return format(float(value), spec)


def load_timed_trades(spec) -> list[TimedTrade]:
    frame = load_lane(spec.orb_label, spec.orb_minutes, spec.rr_target, spec.instrument)
    if frame.empty:
        return []
    if spec.filter_type == "NO_FILTER":
        active = np.ones(len(frame), dtype=bool)
    else:
        active = compute_deployed_filter(frame, spec.filter_type, spec.orb_label).astype(bool)

    selected = frame.loc[active, ["trading_day", "break_dir", "pnl_r"]].copy()
    selected["trading_day"] = pd.to_datetime(selected["trading_day"]).dt.date

    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        timed = con.execute(
            """
            SELECT trading_day, entry_ts, exit_ts
            FROM orb_outcomes
            WHERE symbol = ?
              AND orb_label = ?
              AND orb_minutes = ?
              AND rr_target = ?
              AND confirm_bars = ?
              AND entry_model = ?
              AND pnl_r IS NOT NULL
            """,
            [
                spec.instrument,
                spec.orb_label,
                spec.orb_minutes,
                spec.rr_target,
                spec.confirm_bars,
                spec.entry_model,
            ],
        ).fetchdf()

    timed["trading_day"] = pd.to_datetime(timed["trading_day"]).dt.date
    merged = selected.merge(timed, on="trading_day", how="left", validate="one_to_one")
    if merged["entry_ts"].isna().any() or merged["exit_ts"].isna().any():
        missing = merged.loc[merged["entry_ts"].isna() | merged["exit_ts"].isna(), "trading_day"].tolist()
        raise RuntimeError(f"Missing timed rows for {spec.strategy_id}: {missing[:5]}")

    trades: list[TimedTrade] = []
    for row in merged.itertuples(index=False):
        trades.append(
            TimedTrade(
                strategy_id=spec.strategy_id,
                instrument=spec.instrument,
                orb_label=spec.orb_label,
                orb_minutes=int(spec.orb_minutes),
                rr_target=float(spec.rr_target),
                filter_type=spec.filter_type,
                trading_day=row.trading_day,
                direction=row.break_dir,
                entry_ts=pd.Timestamp(row.entry_ts),
                exit_ts=pd.Timestamp(row.exit_ts),
                pnl_r=float(row.pnl_r),
            )
        )
    return trades


def compute_snapshot(label: str, trades_by_strategy: dict[str, list[dict[str, Any]]], start_date: date, end_date: date) -> Snapshot:
    daily_returns, all_trades, _counts = build_daily_equity(trades_by_strategy)
    filtered_daily = [(day, pnl) for day, pnl in daily_returns if start_date <= day <= end_date]
    filtered_trades = [t for t in all_trades if start_date <= t["trading_day"] <= end_date]
    _sharpe_d, sharpe_ann, business_days = compute_honest_sharpe(filtered_daily, start_date, end_date)
    dd = compute_drawdown(filtered_daily, start_date, end_date)
    total_r = float(sum(pnl for _, pnl in filtered_daily))
    annual_r = float(total_r / business_days * 252) if business_days else float("nan")
    avg_trades_per_day = float(len(filtered_trades) / business_days) if business_days else float("nan")
    return Snapshot(
        label=label,
        start_date=start_date,
        end_date=end_date,
        total_r=total_r,
        annual_r=annual_r,
        sharpe_ann=sharpe_ann,
        max_dd_r=float(dd["max_dd_r"]),
        worst_single_day_r=float(dd["worst_single_day"]),
        total_trades=len(filtered_trades),
        business_days=business_days,
        avg_trades_per_day=avg_trades_per_day,
    )


def trade_window(trades_by_strategy: dict[str, list[TimedTrade]]) -> tuple[date, date]:
    starts: list[date] = []
    ends: list[date] = []
    for trades in trades_by_strategy.values():
        if not trades:
            continue
        days = [t.trading_day for t in trades]
        starts.append(min(days))
        ends.append(max(days))
    if not starts or not ends:
        raise RuntimeError("No trades loaded for execution translation audit")
    return max(starts), min(ends)


def simulate_runtime(specs, trades_map: dict[str, list[TimedTrade]]) -> dict[str, Any]:
    risk_mgr = RiskManager(RiskLimits())
    accepted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    max_open_positions = 0
    max_open_us_data_positions = 0
    halfsize_suggested = 0
    halfsize_noop = 0

    all_days = sorted(set().union(*(set(t.trading_day for t in trades_map[s.strategy_id]) for s in specs)))
    for trading_day in all_days:
        risk_mgr.daily_reset(trading_day)
        active: list[ActiveTrade] = []
        realized_pnl = 0.0
        day_trades = sorted(
            [t for s in specs for t in trades_map[s.strategy_id] if t.trading_day == trading_day],
            key=lambda t: (t.entry_ts, t.orb_minutes, t.strategy_id),
        )

        def settle_until(ts: pd.Timestamp) -> None:
            nonlocal active, realized_pnl
            still_active: list[ActiveTrade] = []
            exiting = sorted([a for a in active if a.timed.exit_ts <= ts], key=lambda a: a.timed.exit_ts)
            for a in exiting:
                realized_pnl += a.timed.pnl_r
            still_active = [a for a in active if a.timed.exit_ts > ts]
            active = still_active

        for trade in day_trades:
            settle_until(trade.entry_ts)
            can_enter, reason, suggested_factor = risk_mgr.can_enter(
                strategy_id=trade.strategy_id,
                orb_label=trade.orb_label,
                active_trades=active,
                daily_pnl_r=realized_pnl,
                orb_minutes=trade.orb_minutes,
                instrument=trade.instrument,
                direction=trade.direction,
            )
            if not can_enter:
                rejected.append(
                    {
                        "strategy_id": trade.strategy_id,
                        "trading_day": trade.trading_day,
                        "reason": reason,
                    }
                )
                continue

            if suggested_factor < 1.0:
                halfsize_suggested += 1
                effective_contracts = max(1, int(1 * suggested_factor))
                if effective_contracts == 1:
                    halfsize_noop += 1
            active.append(ActiveTrade(trade, contracts=1))
            accepted[trade.strategy_id].append(
                {
                    "trading_day": trade.trading_day,
                    "outcome": "accepted",
                    "pnl_r": trade.pnl_r,
                    "instrument": trade.instrument,
                    "session": trade.orb_label,
                    "strategy_id": trade.strategy_id,
                }
            )
            risk_mgr.on_trade_entry()
            max_open_positions = max(max_open_positions, len(active))
            us_data_open = sum(1 for a in active if a.orb_label == "US_DATA_1000")
            max_open_us_data_positions = max(max_open_us_data_positions, us_data_open)

        settle_until(pd.Timestamp("2100-01-01", tz="UTC"))

    return {
        "accepted": accepted,
        "rejected": rejected,
        "max_open_positions": max_open_positions,
        "max_open_us_data_positions": max_open_us_data_positions,
        "halfsize_suggested": halfsize_suggested,
        "halfsize_noop": halfsize_noop,
    }


def _overlap_minutes(a: TimedTrade, b: TimedTrade) -> float:
    start = max(a.entry_ts, b.entry_ts)
    end = min(a.exit_ts, b.exit_ts)
    if start >= end:
        return 0.0
    return float((end - start).total_seconds() / 60.0)


def pair_overlap_summary(candidate_trades: list[TimedTrade], incumbent_trades: list[TimedTrade]) -> PairOverlapSummary:
    cand_df = pd.DataFrame(
        [
            {
                "trading_day": t.trading_day,
                "direction": t.direction,
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
            }
            for t in candidate_trades
        ]
    )
    inc_df = pd.DataFrame(
        [
            {
                "trading_day": t.trading_day,
                "direction": t.direction,
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
            }
            for t in incumbent_trades
        ]
    )
    merged = cand_df.merge(inc_df, on="trading_day", suffixes=("_c", "_i"), how="inner")
    if merged.empty:
        return PairOverlapSummary(len(cand_df), len(inc_df), 0, 0, 0, 0, 0, 0, None, None)

    entry_gaps: list[float] = []
    overlap_mins: list[float] = []
    time_overlap_days = 0
    same_dir = 0
    opp_dir = 0
    cand_active_at_inc_entry = 0
    inc_active_at_cand_entry = 0
    for row in merged.itertuples(index=False):
        entry_gaps.append(float((row.entry_ts_i - row.entry_ts_c).total_seconds() / 60.0))
        overlap_min = _overlap_minutes(
            TimedTrade("", "", "", 0, 0.0, "", row.trading_day, row.direction_c, row.entry_ts_c, row.exit_ts_c, 0.0),
            TimedTrade("", "", "", 0, 0.0, "", row.trading_day, row.direction_i, row.entry_ts_i, row.exit_ts_i, 0.0),
        )
        if overlap_min > 0:
            time_overlap_days += 1
            overlap_mins.append(overlap_min)
            if row.direction_c == row.direction_i:
                same_dir += 1
            else:
                opp_dir += 1
        if row.entry_ts_c <= row.entry_ts_i < row.exit_ts_c:
            cand_active_at_inc_entry += 1
        if row.entry_ts_i <= row.entry_ts_c < row.exit_ts_i:
            inc_active_at_cand_entry += 1

    return PairOverlapSummary(
        candidate_days=len(cand_df),
        incumbent_days=len(inc_df),
        shared_days=len(merged),
        time_overlap_days=time_overlap_days,
        time_overlap_same_dir_days=same_dir,
        time_overlap_opp_dir_days=opp_dir,
        candidate_active_at_incumbent_entry_days=cand_active_at_inc_entry,
        incumbent_active_at_candidate_entry_days=inc_active_at_cand_entry,
        median_entry_gap_min=float(np.median(entry_gaps)) if entry_gaps else None,
        median_overlap_min=float(np.median(overlap_mins)) if overlap_mins else None,
    )


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if isinstance(a, float) and np.isnan(a):
        return None
    if isinstance(b, float) and np.isnan(b):
        return None
    return float(a - b)


def _snapshot_from_records(label: str, records: dict[str, list[dict[str, Any]]], start_date: date, end_date: date) -> Snapshot:
    return compute_snapshot(label=label, trades_by_strategy=records, start_date=start_date, end_date=end_date)


def _verdict_for(outcome: dict[str, Any], pair: PairOverlapSummary, translated_is: Snapshot, base_is: Snapshot) -> tuple[str, str]:
    if outcome["candidate_blocked"] > 0 or outcome["incumbent_block_delta"] > 0 or outcome["other_live_block_delta"] > 0:
        return (
            "ARCHITECTURE_CHANGE_REQUIRED",
            "Do not auto-route under current runtime; model a dedicated shadow translation because live blocking already appears in replay.",
        )
    if outcome["same_session_halfsize_suggested"] > 0 and outcome["same_session_halfsize_effective_noop"] == outcome["same_session_halfsize_suggested"]:
        return (
            "CURRENT_RUNTIME_FEASIBLE_BUT_NOT_DERISKED",
            "If promoted, treat it as a full-size same-session duplicate under current runtime; any true size-down translation requires code/policy change first.",
        )
    if translated_is.annual_r > base_is.annual_r:
        return (
            "CURRENT_RUNTIME_FEASIBLE",
            "Eligible for a narrow shadow / policy review as a same-session cross-aperture add.",
        )
    return (
        "PARK",
        "Keep on shelf/manual review only; translated runtime replay does not improve on the current live book.",
    )


def render_doc(
    base_is: Snapshot,
    base_oos: Snapshot | None,
    live_specs,
    outcomes: list[TranslationOutcome],
) -> str:
    parts = [
        "# Prior-Day Geometry Execution Translation Audit",
        "",
        "Date: 2026-04-23",
        "",
        "## Scope",
        "",
        "Resolve the only open Path 1 question left on the prior-day geometry branch:",
        "",
        "> can the positive `MNQ US_DATA_1000 O5` prior-day geometry rows coexist honestly with the live `MNQ US_DATA_1000 O15` lane under the current runtime?",
        "",
        "This is a runtime replay, not another discovery pass and not another additivity memo.",
        "",
        "## Truth Surfaces",
        "",
        "Canonical trade truth:",
        "",
        "- `orb_outcomes`",
        "- `daily_features`",
        "",
        "Runtime / deployment context:",
        "",
        "- `validated_setups` for exact live/candidate row parameters",
        "- `trading_app/prop_profiles.py`",
        "- `trading_app/risk_manager.py`",
        "- `trading_app/execution_engine.py`",
        "",
        "## MEASURED Runtime Facts Used",
        "",
        "- Current live profile rows resolve with `max_contracts=1` from `build_portfolio_from_profile(...)` in `trading_app/portfolio.py`.",
        "- Same-session different-aperture overlap triggers `suggested_contract_factor=0.5` in `RiskManager.can_enter(...)`.",
        "- `ExecutionEngine` applies that factor with `max(1, int(...))`, so for 1-contract rows the half-size translation is a no-op.",
        "- The current live book has no same-session duplicate session rows; this branch is a special runtime path, not the normal portfolio-builder path.",
        "",
        "## Live Book Replayed Under Current Runtime Rules",
        "",
        "Live lanes included in the replay:",
        "",
    ]
    for spec in live_specs:
        parts.append(f"- `{spec.strategy_id}`")
    parts.extend(
        [
            "",
            "| Surface | Start | End | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Trades | Avg Trades/Day |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| Base live book IS | {base_is.start_date} | {base_is.end_date} | {fmt_num(base_is.total_r, 1, signed=True)} | {fmt_num(base_is.annual_r, 1, signed=True)} | {fmt_num(base_is.sharpe_ann, 3, signed=True)} | {fmt_num(base_is.max_dd_r, 1)} | {fmt_num(base_is.worst_single_day_r, 1, signed=True)} | {base_is.total_trades} | {fmt_num(base_is.avg_trades_per_day, 3)} |",
        ]
    )
    if base_oos is not None:
        parts.append(
            f"| Base live book OOS monitor | {base_oos.start_date} | {base_oos.end_date} | {fmt_num(base_oos.total_r, 1, signed=True)} | {fmt_num(base_oos.annual_r, 1, signed=True)} | {fmt_num(base_oos.sharpe_ann, 3, signed=True)} | {fmt_num(base_oos.max_dd_r, 1)} | {fmt_num(base_oos.worst_single_day_r, 1, signed=True)} | {base_oos.total_trades} | {fmt_num(base_oos.avg_trades_per_day, 3)} |"
        )

    parts.extend(
        [
            "",
            "## Candidate Outcome Summary",
            "",
            "| Candidate | Shared Days w/ O15 | Time Overlap Days | Same-Dir Overlap | Opp-Dir Overlap | Candidate Blocked | Incumbent Block Δ | Other Live Block Δ | Half-Size Suggested | Half-Size No-Op | Δ Annual R IS | Δ Sharpe IS | Verdict |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in outcomes:
        parts.append(
            f"| `{row.candidate_id}` | {row.pair_overlap.shared_days} | {row.pair_overlap.time_overlap_days} | "
            f"{row.pair_overlap.time_overlap_same_dir_days} | {row.pair_overlap.time_overlap_opp_dir_days} | "
            f"{row.candidate_blocked} | {row.incumbent_block_delta} | {row.other_live_block_delta} | "
            f"{row.same_session_halfsize_suggested} | {row.same_session_halfsize_effective_noop} | "
            f"{fmt_num(row.delta_annual_is, 1, signed=True)} | {fmt_num(row.delta_sharpe_is, 3, signed=True)} | `{row.verdict}` |"
        )

    for row in outcomes:
        parts.extend(
            [
                "",
                f"## {row.candidate_id}",
                "",
                f"- Candidate filter: `{row.candidate_filter}`",
                f"- Pairwise overlap with live `US_DATA_1000 O15` incumbent: `{row.pair_overlap.shared_days}` shared trade days out of `{row.pair_overlap.candidate_days}` candidate days.",
                f"- Time overlap days: `{row.pair_overlap.time_overlap_days}` "
                f"(same-direction `{row.pair_overlap.time_overlap_same_dir_days}`, opposite-direction `{row.pair_overlap.time_overlap_opp_dir_days}`).",
                f"- Candidate active when O15 tried to enter: `{row.pair_overlap.candidate_active_at_incumbent_entry_days}` days.",
                f"- Median O15 minus O5 entry gap: `{fmt_num(row.pair_overlap.median_entry_gap_min, 1, signed=True)}` minutes.",
                f"- Median overlapping hold time on overlap days: `{fmt_num(row.pair_overlap.median_overlap_min, 1)}` minutes.",
                f"- Runtime half-size suggestions: `{row.same_session_halfsize_suggested}`; effective no-op due to 1-contract floor: `{row.same_session_halfsize_effective_noop}`.",
                f"- Candidate blocked by runtime: `{row.candidate_blocked}`.",
                f"- Incremental incumbent blocks vs current live book: `{row.incumbent_block_delta}`.",
                f"- Incremental other-live blocks vs current live book: `{row.other_live_block_delta}`.",
                f"- Max concurrent open positions in translated replay: `{row.max_open_positions}` total, `{row.max_open_us_data_positions}` on `US_DATA_1000`.",
                "",
                "| Surface | Start | End | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Trades | Avg Trades/Day |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
                f"| Standalone candidate IS | {row.standalone_is.start_date} | {row.standalone_is.end_date} | {fmt_num(row.standalone_is.total_r, 1, signed=True)} | {fmt_num(row.standalone_is.annual_r, 1, signed=True)} | {fmt_num(row.standalone_is.sharpe_ann, 3, signed=True)} | {fmt_num(row.standalone_is.max_dd_r, 1)} | {fmt_num(row.standalone_is.worst_single_day_r, 1, signed=True)} | {row.standalone_is.total_trades} | {fmt_num(row.standalone_is.avg_trades_per_day, 3)} |",
            ]
        )
        if row.standalone_oos is not None:
            parts.append(
                f"| Standalone candidate OOS monitor | {row.standalone_oos.start_date} | {row.standalone_oos.end_date} | {fmt_num(row.standalone_oos.total_r, 1, signed=True)} | {fmt_num(row.standalone_oos.annual_r, 1, signed=True)} | {fmt_num(row.standalone_oos.sharpe_ann, 3, signed=True)} | {fmt_num(row.standalone_oos.max_dd_r, 1)} | {fmt_num(row.standalone_oos.worst_single_day_r, 1, signed=True)} | {row.standalone_oos.total_trades} | {fmt_num(row.standalone_oos.avg_trades_per_day, 3)} |"
            )
        parts.append(
            f"| Translated live+candidate IS | {row.translated_is.start_date} | {row.translated_is.end_date} | {fmt_num(row.translated_is.total_r, 1, signed=True)} | {fmt_num(row.translated_is.annual_r, 1, signed=True)} | {fmt_num(row.translated_is.sharpe_ann, 3, signed=True)} | {fmt_num(row.translated_is.max_dd_r, 1)} | {fmt_num(row.translated_is.worst_single_day_r, 1, signed=True)} | {row.translated_is.total_trades} | {fmt_num(row.translated_is.avg_trades_per_day, 3)} |"
        )
        if row.translated_oos is not None:
            parts.append(
                f"| Translated live+candidate OOS monitor | {row.translated_oos.start_date} | {row.translated_oos.end_date} | {fmt_num(row.translated_oos.total_r, 1, signed=True)} | {fmt_num(row.translated_oos.annual_r, 1, signed=True)} | {fmt_num(row.translated_oos.sharpe_ann, 3, signed=True)} | {fmt_num(row.translated_oos.max_dd_r, 1)} | {fmt_num(row.translated_oos.worst_single_day_r, 1, signed=True)} | {row.translated_oos.total_trades} | {fmt_num(row.translated_oos.avg_trades_per_day, 3)} |"
            )
        parts.extend(
            [
                "",
                f"- IS delta vs base live book: annualized R `{fmt_num(row.delta_annual_is, 1, signed=True)}`, honest Sharpe `{fmt_num(row.delta_sharpe_is, 3, signed=True)}`.",
                f"- OOS monitor delta vs base: annualized R `{fmt_num(row.delta_annual_oos, 1, signed=True)}`, honest Sharpe `{fmt_num(row.delta_sharpe_oos, 3, signed=True)}`.",
                f"- Runtime block reasons: `{', '.join(f'{k}={v}' for k, v in row.block_reasons) if row.block_reasons else 'none'}`.",
                f"- Verdict: `{row.verdict}`.",
                f"- Next step: {row.next_step}",
            ]
        )

    parts.extend(
        [
            "",
            "## Bottom Line",
            "",
            "- `US_DATA_1000 O5` prior-day geometry is not blocked by same-session runtime rules in the same way as `COMEX_SETTLE O5`.",
            "- But the current runtime does **not** implement a real same-session size-down for these rows: the half-size suggestion collapses to `1` contract because every live/candidate row is clamped to `max_contracts=1`.",
            "- So any promotion here would be an explicit policy decision to allow full-size same-session duplicate exposure, not a quiet reuse of an already-safe translation path.",
            "- `COMEX_SETTLE PD_CLEAR_LONG` remains outside this branch; same-aperture coexistence is still blocked and replacement remains negative.",
        ]
    )
    return "\n".join(parts) + "\n"


def main(output_path: Path = RESULT_PATH) -> None:
    live_specs = load_live_lane_specs(PROFILE_ID)
    candidate_specs = load_candidate_specs(US_DATA_1000_CANDIDATE_IDS)
    next(spec for spec in live_specs if spec.strategy_id == INCUMBENT_ID)

    timed_trades = {spec.strategy_id: load_timed_trades(spec) for spec in [*live_specs, *candidate_specs]}

    base_window_start, base_window_end = trade_window(
        {sid: trades for sid, trades in timed_trades.items() if sid in {s.strategy_id for s in live_specs}}
    )
    base_sim = simulate_runtime(live_specs, timed_trades)
    base_rejected_by_strategy = Counter(r["strategy_id"] for r in base_sim["rejected"])
    base_existing_reason_counts = Counter(r["reason"].split(":")[0] for r in base_sim["rejected"])
    base_is_end = min(base_window_end, date(2025, 12, 31))
    base_is = _snapshot_from_records("base_live_is", base_sim["accepted"], base_window_start, base_is_end)
    base_oos = None
    if base_window_end >= OOS_START:
        base_oos = _snapshot_from_records("base_live_oos", base_sim["accepted"], OOS_START, min(base_window_end, OOS_END))

    outcomes: list[TranslationOutcome] = []
    for candidate_spec in candidate_specs:
        combo_specs = [*live_specs, candidate_spec]
        combo_window_start, combo_window_end = trade_window({s.strategy_id: timed_trades[s.strategy_id] for s in combo_specs})
        combo_is_end = min(combo_window_end, date(2025, 12, 31))
        base_is_local = _snapshot_from_records("base_live_combo_is", base_sim["accepted"], combo_window_start, combo_is_end)
        base_oos_local = None
        if combo_window_end >= OOS_START:
            base_oos_local = _snapshot_from_records("base_live_combo_oos", base_sim["accepted"], OOS_START, min(combo_window_end, OOS_END))

        standalone_records = {candidate_spec.strategy_id: [
            {
                "trading_day": t.trading_day,
                "outcome": "accepted",
                "pnl_r": t.pnl_r,
                "instrument": t.instrument,
                "session": t.orb_label,
                "strategy_id": t.strategy_id,
            }
            for t in timed_trades[candidate_spec.strategy_id]
        ]}
        standalone_is = _snapshot_from_records("standalone_is", standalone_records, combo_window_start, combo_is_end)
        standalone_oos = None
        if combo_window_end >= OOS_START:
            standalone_oos = _snapshot_from_records("standalone_oos", standalone_records, OOS_START, min(combo_window_end, OOS_END))

        combo_sim = simulate_runtime(combo_specs, timed_trades)
        combo_rejected_by_strategy = Counter(r["strategy_id"] for r in combo_sim["rejected"])
        combo_existing_reason_counts = Counter(
            r["reason"].split(":")[0] for r in combo_sim["rejected"] if r["strategy_id"] != candidate_spec.strategy_id
        )
        translated_is = _snapshot_from_records("translated_is", combo_sim["accepted"], combo_window_start, combo_is_end)
        translated_oos = None
        if combo_window_end >= OOS_START:
            translated_oos = _snapshot_from_records("translated_oos", combo_sim["accepted"], OOS_START, min(combo_window_end, OOS_END))

        candidate_rejected = [r for r in combo_sim["rejected"] if r["strategy_id"] == candidate_spec.strategy_id]
        incumbent_block_delta = max(0, combo_rejected_by_strategy.get(INCUMBENT_ID, 0) - base_rejected_by_strategy.get(INCUMBENT_ID, 0))
        other_live_ids = {s.strategy_id for s in live_specs if s.strategy_id != INCUMBENT_ID}
        other_live_block_delta = sum(
            max(0, combo_rejected_by_strategy.get(strategy_id, 0) - base_rejected_by_strategy.get(strategy_id, 0))
            for strategy_id in other_live_ids
        )
        blocks = Counter(r["reason"].split(":")[0] for r in candidate_rejected)
        for reason, count in combo_existing_reason_counts.items():
            delta = count - base_existing_reason_counts.get(reason, 0)
            if delta > 0:
                blocks[reason] += delta
        pair = pair_overlap_summary(timed_trades[candidate_spec.strategy_id], timed_trades[INCUMBENT_ID])
        verdict, next_step = _verdict_for(
            {
                "candidate_blocked": len(candidate_rejected),
                "incumbent_block_delta": incumbent_block_delta,
                "other_live_block_delta": other_live_block_delta,
                "same_session_halfsize_suggested": combo_sim["halfsize_suggested"],
                "same_session_halfsize_effective_noop": combo_sim["halfsize_noop"],
            },
            pair,
            translated_is,
            base_is_local,
        )
        outcomes.append(
            TranslationOutcome(
                candidate_id=candidate_spec.strategy_id,
                candidate_filter=candidate_spec.filter_type,
                standalone_is=standalone_is,
                standalone_oos=standalone_oos,
                translated_is=translated_is,
                translated_oos=translated_oos,
                delta_annual_is=float(translated_is.annual_r - base_is_local.annual_r),
                delta_sharpe_is=float((translated_is.sharpe_ann or 0.0) - (base_is_local.sharpe_ann or 0.0)),
                delta_annual_oos=_delta(
                    translated_oos.annual_r if translated_oos else None,
                    base_oos_local.annual_r if base_oos_local else None,
                ),
                delta_sharpe_oos=_delta(
                    translated_oos.sharpe_ann if translated_oos else None,
                    base_oos_local.sharpe_ann if base_oos_local else None,
                ),
                pair_overlap=pair,
                candidate_accepted=len(combo_sim["accepted"].get(candidate_spec.strategy_id, [])),
                candidate_blocked=len(candidate_rejected),
                incumbent_block_delta=incumbent_block_delta,
                other_live_block_delta=other_live_block_delta,
                same_session_halfsize_suggested=combo_sim["halfsize_suggested"],
                same_session_halfsize_effective_noop=combo_sim["halfsize_noop"],
                max_open_positions=combo_sim["max_open_positions"],
                max_open_us_data_positions=combo_sim["max_open_us_data_positions"],
                block_reasons=tuple(sorted(blocks.items())),
                verdict=verdict,
                next_step=next_step,
            )
        )

    output_path.write_text(render_doc(base_is, base_oos, live_specs, outcomes), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prior-day geometry execution translation audit.")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    args = parser.parse_args()
    main(output_path=args.output)
