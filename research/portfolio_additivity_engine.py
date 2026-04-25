#!/usr/bin/env python3
"""Reusable additivity audit engine for shelf candidates vs a live profile.

Purpose:
- reconstruct the current live book from canonical trade outcomes + canonical
  filter logic
- load candidate shelf rows from `validated_setups`
- compare candidate standalone / additive / same-session replacement roles
- keep the result as research-only evidence; no writes to live config

Truth sources:
- canonical trade outcomes: `orb_outcomes`
- canonical feature/filter state: `daily_features`

Comparison-only deployment context:
- `docs/runtime/lane_allocation.json`
- `validated_setups`
- `trading_app/prop_profiles.py`
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.db_contracts import deployable_validated_relation
from pipeline.paths import GOLD_DB_PATH
from research.comprehensive_deployed_lane_scan import compute_deployed_filter, load_lane
from research.research_portfolio_assembly import build_daily_equity, compute_drawdown, compute_honest_sharpe
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.prop_portfolio import profile_static_gate_reason
from trading_app.prop_profiles import ACCOUNT_PROFILES, get_profile_lane_definitions

ALLOCATION_PATH = PROJECT_ROOT / "docs/runtime/lane_allocation.json"


@dataclass(frozen=True)
class LaneSpec:
    strategy_id: str
    instrument: str
    orb_label: str
    entry_model: str
    rr_target: float
    confirm_bars: int
    filter_type: str
    orb_minutes: int


@dataclass(frozen=True)
class PortfolioSnapshot:
    label: str
    start_date: date
    end_date: date
    total_r: float
    annual_r: float
    sharpe_ann: float | None
    max_dd_r: float
    worst_single_day_r: float
    total_trades: int
    trade_days: int
    business_days: int
    avg_trades_per_day: float


@dataclass(frozen=True)
class CorrelationRow:
    strategy_id: str
    corr_daily_r: float | None
    overlap_days: int
    overlap_pct_of_candidate_days: float


@dataclass(frozen=True)
class CorrelationSummary:
    corr_to_base_portfolio: float | None
    candidate_trade_days: int
    days_with_any_live_overlap: int
    pct_candidate_days_with_live_overlap: float
    avg_live_lanes_active_on_candidate_days: float


@dataclass(frozen=True)
class ProfileFitSummary:
    current_profile_allowed: bool
    active_profile_fits: tuple[str, ...]
    inactive_profile_fits: tuple[str, ...]


@dataclass(frozen=True)
class CandidateAdditivityAudit:
    candidate: LaneSpec
    standalone_is: PortfolioSnapshot
    standalone_oos: PortfolioSnapshot
    add_is: PortfolioSnapshot
    add_oos: PortfolioSnapshot
    replace_is: PortfolioSnapshot | None
    replace_oos: PortfolioSnapshot | None
    replaced_lane: LaneSpec | None
    corr_rows_is: list[CorrelationRow]
    corr_summary_is: CorrelationSummary
    profile_fit: ProfileFitSummary


def _load_allocation_payload(profile_id: str) -> dict[str, Any]:
    payload = json.loads(ALLOCATION_PATH.read_text(encoding="utf-8"))
    if payload.get("profile_id") != profile_id:
        raise RuntimeError(f"Expected profile_id={profile_id!r}, found {payload.get('profile_id')!r}")
    return payload


def _rows_to_specs(rows: list[tuple[Any, ...]]) -> dict[str, LaneSpec]:
    return {
        row[0]: LaneSpec(
            strategy_id=row[0],
            instrument=row[1],
            orb_label=row[2],
            entry_model=row[3],
            rr_target=float(row[4]),
            confirm_bars=int(row[5]),
            filter_type=row[6],
            orb_minutes=int(row[7]),
        )
        for row in rows
    }


def _load_specs_from_strategy_ids(strategy_ids: list[str]) -> list[LaneSpec]:
    if not strategy_ids:
        raise RuntimeError("No strategy IDs supplied")
    placeholders = ", ".join(["?"] * len(strategy_ids))
    query = f"""
        SELECT strategy_id, instrument, orb_label, entry_model,
               rr_target, confirm_bars, filter_type, orb_minutes
        FROM validated_setups
        WHERE strategy_id IN ({placeholders})
    """
    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        rows = con.execute(query, strategy_ids).fetchall()
    found = _rows_to_specs(rows)
    missing = [sid for sid in strategy_ids if sid not in found]
    if missing:
        raise RuntimeError(f"Missing live lane params in validated_setups: {missing}")
    return [found[sid] for sid in strategy_ids]


def load_live_lane_specs(profile_id: str) -> list[LaneSpec]:
    lane_defs = get_profile_lane_definitions(profile_id)
    strategy_ids = [lane["strategy_id"] for lane in lane_defs]
    if not strategy_ids:
        raise RuntimeError(f"Profile {profile_id!r} has no lane definitions")
    return _load_specs_from_strategy_ids(strategy_ids)


def load_candidate_specs(strategy_ids: list[str]) -> list[LaneSpec]:
    if len(strategy_ids) != len(set(strategy_ids)):
        raise RuntimeError(f"Duplicate candidate strategy IDs supplied: {strategy_ids}")
    if not strategy_ids:
        raise RuntimeError("No candidate strategy IDs supplied")

    placeholders = ", ".join(["?"] * len(strategy_ids))
    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        shelf_relation = deployable_validated_relation(con, alias="vs")
        rows = con.execute(
            f"""
            SELECT strategy_id, instrument, orb_label, entry_model,
                   rr_target, confirm_bars, filter_type, orb_minutes
            FROM {shelf_relation}
            WHERE strategy_id IN ({placeholders})
            """,
            strategy_ids,
        ).fetchall()

    found = _rows_to_specs(rows)
    missing = [sid for sid in strategy_ids if sid not in found]
    if missing:
        raise RuntimeError(f"Candidate strategy IDs not found on deployable shelf: {missing}")
    return [found[sid] for sid in strategy_ids]


def load_trade_records(spec: LaneSpec) -> list[dict[str, Any]]:
    frame = load_lane(spec.orb_label, spec.orb_minutes, spec.rr_target, spec.instrument)
    if frame.empty:
        return []
    if spec.entry_model != "E2" or spec.confirm_bars != 1:
        raise RuntimeError(f"Unexpected params outside additivity engine scope: {spec}")

    if spec.filter_type == "NO_FILTER":
        active = np.ones(len(frame), dtype=bool)
    else:
        active = compute_deployed_filter(frame, spec.filter_type, spec.orb_label).astype(bool)

    selected = frame.loc[active].copy()
    selected["trading_day"] = pd.to_datetime(selected["trading_day"]).dt.date
    return [
        {
            "trading_day": row.trading_day,
            "outcome": row.outcome,
            "pnl_r": float(row.pnl_r),
            "instrument": spec.instrument,
            "session": spec.orb_label,
            "strategy_id": spec.strategy_id,
        }
        for row in selected.itertuples(index=False)
    ]


def common_window(trades_by_strategy: dict[str, list[dict[str, Any]]]) -> tuple[date, date]:
    starts: list[date] = []
    ends: list[date] = []
    for trades in trades_by_strategy.values():
        if not trades:
            continue
        days = [t["trading_day"] for t in trades]
        starts.append(min(days))
        ends.append(max(days))
    if not starts or not ends:
        raise RuntimeError("No trades loaded for additivity audit")
    return max(starts), min(ends)


def compute_snapshot(
    label: str,
    trades_by_strategy: dict[str, list[dict[str, Any]]],
    start_date: date,
    end_date: date,
) -> PortfolioSnapshot:
    daily_returns, all_trades, _counts = build_daily_equity(trades_by_strategy)
    filtered_daily = [(day, pnl) for day, pnl in daily_returns if start_date <= day <= end_date]
    filtered_trades = [t for t in all_trades if start_date <= t["trading_day"] <= end_date]
    _sharpe_d, sharpe_ann, business_days = compute_honest_sharpe(filtered_daily, start_date, end_date)
    dd = compute_drawdown(filtered_daily, start_date, end_date)
    total_r = float(sum(pnl for _, pnl in filtered_daily))
    trade_days = len(filtered_daily)
    annual_r = float(total_r / business_days * 252) if business_days else float("nan")
    avg_trades_per_day = float(len(filtered_trades) / business_days) if business_days else float("nan")
    return PortfolioSnapshot(
        label=label,
        start_date=start_date,
        end_date=end_date,
        total_r=total_r,
        annual_r=annual_r,
        sharpe_ann=sharpe_ann,
        max_dd_r=float(dd["max_dd_r"]),
        worst_single_day_r=float(dd["worst_single_day"]),
        total_trades=len(filtered_trades),
        trade_days=trade_days,
        business_days=business_days,
        avg_trades_per_day=avg_trades_per_day,
    )


def daily_return_frame(
    trades_by_strategy: dict[str, list[dict[str, Any]]],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    index = pd.bdate_range(start=start_date, end=end_date).date
    payload: dict[str, list[float]] = {}
    for strategy_id, trades in trades_by_strategy.items():
        daily_map: dict[date, float] = {}
        for trade in trades:
            day = trade["trading_day"]
            if not (start_date <= day <= end_date):
                continue
            daily_map[day] = daily_map.get(day, 0.0) + float(trade["pnl_r"])
        payload[strategy_id] = [daily_map.get(day, 0.0) for day in index]
    return pd.DataFrame(payload, index=pd.Index(index, name="trading_day"))


def safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    if len(a) < 2 or float(a.std(ddof=1)) == 0.0 or float(b.std(ddof=1)) == 0.0:
        return None
    return float(a.corr(b))


def correlation_rows(frame: pd.DataFrame, candidate_id: str) -> tuple[list[CorrelationRow], CorrelationSummary]:
    candidate = frame[candidate_id]
    other_cols = [c for c in frame.columns if c != candidate_id]
    rows: list[CorrelationRow] = []
    base_portfolio = frame[other_cols].sum(axis=1) if other_cols else pd.Series(0.0, index=frame.index)
    candidate_active = candidate != 0.0
    other_active = (frame[other_cols] != 0.0).sum(axis=1) if other_cols else pd.Series(0, index=frame.index)
    candidate_days = int(candidate_active.sum())
    for column in other_cols:
        lane = frame[column]
        overlap_mask = candidate_active & (lane != 0.0)
        rows.append(
            CorrelationRow(
                strategy_id=column,
                corr_daily_r=safe_corr(candidate, lane),
                overlap_days=int(overlap_mask.sum()),
                overlap_pct_of_candidate_days=float(overlap_mask.sum() / candidate_days) if candidate_days else float("nan"),
            )
        )
    portfolio_overlap_days = int((candidate_active & (other_active > 0)).sum())
    summary = CorrelationSummary(
        corr_to_base_portfolio=safe_corr(candidate, base_portfolio),
        candidate_trade_days=candidate_days,
        days_with_any_live_overlap=portfolio_overlap_days,
        pct_candidate_days_with_live_overlap=(
            float(portfolio_overlap_days / candidate_days) if candidate_days else float("nan")
        ),
        avg_live_lanes_active_on_candidate_days=(
            float(other_active.loc[candidate_active].mean()) if candidate_days else float("nan")
        ),
    )
    return rows, summary


def fmt_num(value: float | None, digits: int = 3, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    spec = f"+.{digits}f" if signed else f".{digits}f"
    return format(float(value), spec)


def matching_live_lane(candidate: LaneSpec, live_specs: list[LaneSpec]) -> LaneSpec | None:
    same_session = [spec for spec in live_specs if spec.orb_label == candidate.orb_label and spec.instrument == candidate.instrument]
    if len(same_session) > 1:
        raise RuntimeError(f"Expected at most one live lane per same instrument/session for replacement audit, got {same_session}")
    return same_session[0] if same_session else None


def summarize_profile_fit(current_profile_id: str, candidate: LaneSpec) -> ProfileFitSummary:
    active_fits: list[str] = []
    inactive_fits: list[str] = []
    for profile_id, profile in ACCOUNT_PROFILES.items():
        reason = profile_static_gate_reason(profile, candidate.instrument, candidate.orb_label)
        if reason is not None:
            continue
        if profile.active:
            active_fits.append(profile_id)
        else:
            inactive_fits.append(profile_id)
    return ProfileFitSummary(
        current_profile_allowed=(
            profile_static_gate_reason(
                ACCOUNT_PROFILES[current_profile_id],
                candidate.instrument,
                candidate.orb_label,
            )
            is None
        ),
        active_profile_fits=tuple(sorted(active_fits)),
        inactive_profile_fits=tuple(sorted(inactive_fits)),
    )


def run_additivity_audit(profile_id: str, candidate_ids: list[str]) -> dict[str, Any]:
    live_specs = load_live_lane_specs(profile_id)
    candidates = load_candidate_specs(candidate_ids)
    all_specs = live_specs + candidates
    trades_all = {spec.strategy_id: load_trade_records(spec) for spec in all_specs}

    start_all, end_all = common_window(trades_all)
    is_end = min(end_all, HOLDOUT_SACRED_FROM - timedelta(days=1))
    if start_all > is_end:
        raise RuntimeError("No in-sample window available for additivity audit")

    oos_start = HOLDOUT_SACRED_FROM
    has_oos = oos_start <= end_all

    base_trades = {spec.strategy_id: trades_all[spec.strategy_id] for spec in live_specs}
    base_is = compute_snapshot("Live book", base_trades, start_all, is_end)
    base_oos = compute_snapshot("Live book", base_trades, oos_start, end_all) if has_oos else None

    audits: list[CandidateAdditivityAudit] = []
    for candidate in candidates:
        standalone_trades = {candidate.strategy_id: trades_all[candidate.strategy_id]}
        add_trades = dict(base_trades)
        add_trades[candidate.strategy_id] = trades_all[candidate.strategy_id]

        standalone_is = compute_snapshot(f"{candidate.strategy_id} standalone", standalone_trades, start_all, is_end)
        add_is = compute_snapshot(f"Live book + {candidate.strategy_id}", add_trades, start_all, is_end)

        if has_oos:
            standalone_oos = compute_snapshot(
                f"{candidate.strategy_id} standalone",
                standalone_trades,
                oos_start,
                end_all,
            )
            add_oos = compute_snapshot(f"Live book + {candidate.strategy_id}", add_trades, oos_start, end_all)
        else:
            standalone_oos = compute_snapshot(
                f"{candidate.strategy_id} standalone",
                {},
                oos_start,
                oos_start,
            )
            add_oos = compute_snapshot(
                f"Live book + {candidate.strategy_id}",
                {},
                oos_start,
                oos_start,
            )

        frame_is = daily_return_frame(add_trades, start_all, is_end)
        corr_rows_is, corr_summary_is = correlation_rows(frame_is, candidate.strategy_id)

        replaced_lane = matching_live_lane(candidate, live_specs)
        replace_is: PortfolioSnapshot | None = None
        replace_oos: PortfolioSnapshot | None = None
        if replaced_lane is not None:
            replace_trades = dict(base_trades)
            replace_trades.pop(replaced_lane.strategy_id, None)
            replace_trades[candidate.strategy_id] = trades_all[candidate.strategy_id]
            replace_is = compute_snapshot(
                f"Live book with {replaced_lane.strategy_id} replaced by {candidate.strategy_id}",
                replace_trades,
                start_all,
                is_end,
            )
            if has_oos:
                replace_oos = compute_snapshot(
                    f"Live book with {replaced_lane.strategy_id} replaced by {candidate.strategy_id}",
                    replace_trades,
                    oos_start,
                    end_all,
                )

        audits.append(
            CandidateAdditivityAudit(
                candidate=candidate,
                standalone_is=standalone_is,
                standalone_oos=standalone_oos,
                add_is=add_is,
                add_oos=add_oos,
                replace_is=replace_is,
                replace_oos=replace_oos,
                replaced_lane=replaced_lane,
                corr_rows_is=corr_rows_is,
                corr_summary_is=corr_summary_is,
                profile_fit=summarize_profile_fit(profile_id, candidate),
            )
        )

    return {
        "profile_id": profile_id,
        "live_specs": live_specs,
        "candidates": audits,
        "base_is": base_is,
        "base_oos": base_oos,
        "common_start": start_all,
        "common_end": end_all,
        "is_end": is_end,
        "oos_start": oos_start,
        "max_slots": ACCOUNT_PROFILES[profile_id].max_slots,
    }


__all__ = [
    "ALLOCATION_PATH",
    "CandidateAdditivityAudit",
    "CorrelationRow",
    "CorrelationSummary",
    "LaneSpec",
    "PortfolioSnapshot",
    "ProfileFitSummary",
    "fmt_num",
    "load_candidate_specs",
    "load_live_lane_specs",
    "run_additivity_audit",
    "summarize_profile_fit",
]
