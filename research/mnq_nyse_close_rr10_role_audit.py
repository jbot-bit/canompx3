#!/usr/bin/env python3
"""MNQ NYSE_CLOSE RR1.0 role audit against the live MNQ allocator book.

Question locked in `docs/runtime/stages/mnq-nyse-close-rr10-role-audit.md`:
after the exact ORB_G8 filter path was killed, does broad unfiltered
MNQ NYSE_CLOSE RR1.0 belong as a standalone / allocator candidate, or nowhere?

Truth sources:
- canonical trade outcomes: `orb_outcomes`
- canonical feature/filter state: `daily_features`

Comparison-only deployment context:
- `docs/runtime/lane_allocation.json`
- `validated_setups` for the exact live-lane parameters
- `trading_app/prop_profiles.py` for slot/session policy

Method:
- Reconstruct the 6 current live MNQ lanes from canonical `orb_outcomes` and
  canonical filter logic (`compute_deployed_filter` -> `ALL_FILTERS`).
- Reconstruct raw `MNQ NYSE_CLOSE E2 RR1.0 CB1 O5 NO_FILTER`.
- Compare the current 6-lane book vs the same book plus NYSE_CLOSE using
  honest business-day Sharpe and drawdown on a common calendar.
- Keep 2026 OOS as monitor-only; selection verdict is based on pre-2026 IS.

No writes to validated_setups / live config / lane_allocation.
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

from pipeline.paths import GOLD_DB_PATH
from research.comprehensive_deployed_lane_scan import compute_deployed_filter, load_lane
from research.research_portfolio_assembly import build_daily_equity, compute_drawdown, compute_honest_sharpe
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.prop_profiles import ACCOUNT_PROFILES

PROFILE_ID = "topstep_50k_mnq_auto"
ALLOCATION_PATH = PROJECT_ROOT / "docs/runtime/lane_allocation.json"
RESULT_PATH = PROJECT_ROOT / "docs/audit/results/2026-04-23-mnq-nyse-close-rr10-role-audit.md"
HOLDOUT_DATE = HOLDOUT_SACRED_FROM


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


def load_live_lane_specs() -> list[LaneSpec]:
    payload = json.loads(ALLOCATION_PATH.read_text(encoding="utf-8"))
    if payload.get("profile_id") != PROFILE_ID:
        raise RuntimeError(f"Expected profile_id={PROFILE_ID!r}, found {payload.get('profile_id')!r}")

    strategy_ids = [lane["strategy_id"] for lane in payload.get("lanes", [])]
    if not strategy_ids:
        raise RuntimeError(f"{ALLOCATION_PATH} contains no lanes")

    placeholders = ", ".join(["?"] * len(strategy_ids))
    query = f"""
        SELECT strategy_id, instrument, orb_label, entry_model,
               rr_target, confirm_bars, filter_type, orb_minutes
        FROM validated_setups
        WHERE strategy_id IN ({placeholders})
    """
    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        rows = con.execute(query, strategy_ids).fetchall()

    found = {
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
    missing = [sid for sid in strategy_ids if sid not in found]
    if missing:
        raise RuntimeError(f"Missing live lane params in validated_setups: {missing}")
    return [found[sid] for sid in strategy_ids]


def nyse_close_candidate() -> LaneSpec:
    return LaneSpec(
        strategy_id="MNQ_NYSE_CLOSE_E2_RR1.0_CB1_NO_FILTER",
        instrument="MNQ",
        orb_label="NYSE_CLOSE",
        entry_model="E2",
        rr_target=1.0,
        confirm_bars=1,
        filter_type="NO_FILTER",
        orb_minutes=5,
    )


def load_trade_records(spec: LaneSpec) -> list[dict[str, Any]]:
    frame = load_lane(spec.orb_label, spec.orb_minutes, spec.rr_target, spec.instrument)
    if frame.empty:
        return []
    if spec.entry_model != "E2" or spec.confirm_bars != 1:
        raise RuntimeError(f"Unexpected live lane params outside current audit scope: {spec}")

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
        raise RuntimeError("No trades loaded for role audit")
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
    sharpe_d, sharpe_ann, business_days = compute_honest_sharpe(filtered_daily, start_date, end_date)
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


def correlation_rows(frame: pd.DataFrame, candidate_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate = frame[candidate_id]
    other_cols = [c for c in frame.columns if c != candidate_id]
    rows: list[dict[str, Any]] = []
    base_portfolio = frame[other_cols].sum(axis=1) if other_cols else pd.Series(0.0, index=frame.index)
    candidate_active = candidate != 0.0
    other_active = (frame[other_cols] != 0.0).sum(axis=1) if other_cols else pd.Series(0, index=frame.index)
    candidate_days = int(candidate_active.sum())
    for column in other_cols:
        lane = frame[column]
        overlap_mask = candidate_active & (lane != 0.0)
        rows.append(
            {
                "strategy_id": column,
                "corr_daily_r": safe_corr(candidate, lane),
                "overlap_days": int(overlap_mask.sum()),
                "overlap_pct_of_candidate_days": (
                    float(overlap_mask.sum() / candidate_days) if candidate_days else float("nan")
                ),
            }
        )
    portfolio_overlap_days = int((candidate_active & (other_active > 0)).sum())
    summary = {
        "corr_to_base_portfolio": safe_corr(candidate, base_portfolio),
        "candidate_trade_days": candidate_days,
        "days_with_any_live_overlap": portfolio_overlap_days,
        "pct_candidate_days_with_live_overlap": (
            float(portfolio_overlap_days / candidate_days) if candidate_days else float("nan")
        ),
        "avg_live_lanes_active_on_candidate_days": (
            float(other_active.loc[candidate_active].mean()) if candidate_days else float("nan")
        ),
    }
    return rows, summary


def determine_role_verdict(
    base_is: PortfolioSnapshot,
    plus_is: PortfolioSnapshot,
    candidate_is: PortfolioSnapshot,
    corr_summary_is: dict[str, Any],
    max_slots: int,
    live_lane_count: int,
) -> tuple[str, str]:
    free_slot = live_lane_count < max_slots
    sharpe_up = (
        base_is.sharpe_ann is not None
        and plus_is.sharpe_ann is not None
        and plus_is.sharpe_ann > base_is.sharpe_ann
    )
    annual_up = plus_is.annual_r > base_is.annual_r
    corr_to_book = corr_summary_is["corr_to_base_portfolio"]
    low_portfolio_corr = corr_to_book is not None and abs(corr_to_book) < 0.15
    if free_slot and sharpe_up and annual_up and low_portfolio_corr and candidate_is.annual_r > 0:
        return (
            "CONTINUE as allocator candidate",
            "Free slot exists, IS annual R and honest Sharpe improve when NYSE_CLOSE is added, "
            "and candidate/book daily correlation is low.",
        )
    if candidate_is.annual_r > 0:
        return (
            "CONTINUE as standalone monitor only",
            "Standalone NYSE_CLOSE remains positive, but allocator inclusion did not clear a clean "
            "risk-adjusted add test.",
        )
    return (
        "PARK",
        "Standalone NYSE_CLOSE does not retain positive contribution on the common IS window.",
    )


def fmt_num(value: float | None, digits: int = 3, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    spec = f"+.{digits}f" if signed else f".{digits}f"
    return format(float(value), spec)


def snapshot_row(snapshot: PortfolioSnapshot) -> str:
    return (
        f"| {snapshot.label} | {snapshot.start_date} | {snapshot.end_date} | "
        f"{snapshot.business_days} | {snapshot.total_trades} | {snapshot.trade_days} | "
        f"{fmt_num(snapshot.total_r, 1, signed=True)} | {fmt_num(snapshot.annual_r, 1, signed=True)} | "
        f"{fmt_num(snapshot.sharpe_ann, 3, signed=True)} | {fmt_num(snapshot.max_dd_r, 1)} | "
        f"{fmt_num(snapshot.worst_single_day_r, 1, signed=True)} | {fmt_num(snapshot.avg_trades_per_day, 3)} |"
    )


def render_doc(
    live_specs: list[LaneSpec],
    candidate: LaneSpec,
    base_is: PortfolioSnapshot,
    plus_is: PortfolioSnapshot,
    candidate_is: PortfolioSnapshot,
    base_oos: PortfolioSnapshot,
    plus_oos: PortfolioSnapshot,
    candidate_oos: PortfolioSnapshot,
    corr_rows_is: list[dict[str, Any]],
    corr_summary_is: dict[str, Any],
    verdict: str,
    reasoning: str,
    max_slots: int,
) -> str:
    corr_lines = [
        "| Live lane | Daily corr vs NYSE_CLOSE | Overlap days | Overlap % of NYSE_CLOSE days |",
        "|---|---:|---:|---:|",
    ]
    for row in corr_rows_is:
        corr_lines.append(
            f"| `{row['strategy_id']}` | {fmt_num(row['corr_daily_r'], 3, signed=True)} | "
            f"{row['overlap_days']} | {fmt_num(row['overlap_pct_of_candidate_days'] * 100 if not np.isnan(row['overlap_pct_of_candidate_days']) else float('nan'), 1)}% |"
        )

    delta_annual = plus_is.annual_r - base_is.annual_r
    delta_sharpe = (
        plus_is.sharpe_ann - base_is.sharpe_ann
        if plus_is.sharpe_ann is not None and base_is.sharpe_ann is not None
        else float("nan")
    )
    delta_dd = plus_is.max_dd_r - base_is.max_dd_r

    parts = [
        "# MNQ NYSE_CLOSE RR1.0 Role Audit",
        "",
        "Date: 2026-04-23",
        "",
        "## Scope",
        "",
        "Resolve the remaining honest question after the exact `ORB_G8` NYSE_CLOSE prereg was killed:",
        "does broad unfiltered `MNQ NYSE_CLOSE E2 RR1.0 CB1 O5` belong as an allocator add, a standalone monitor only, or nowhere?",
        "",
        "Truth used for this audit:",
        "",
        "- `gold.db::orb_outcomes`",
        "- `gold.db::daily_features`",
        "",
        "Comparison-only deployment context:",
        "",
        "- `docs/runtime/lane_allocation.json`",
        "- `validated_setups` for exact live-lane parameters",
        "- `trading_app/prop_profiles.py`",
        "",
        "## Live Book Context",
        "",
        f"- Profile: `{PROFILE_ID}`",
        f"- `max_slots={max_slots}`",
        f"- Current deployed MNQ lanes in allocation JSON: `{len(live_specs)}`",
        f"- Free slot exists before any replacement question: `{len(live_specs) < max_slots}`",
        f"- `NYSE_CLOSE` is currently excluded from `allowed_sessions` and from `build_raw_baseline_portfolio(... exclude_sessions={{\"NYSE_CLOSE\"}})`.",
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
            f"Candidate under test: `{candidate.strategy_id}` -> {candidate.orb_label} "
            f"O{candidate.orb_minutes} RR{candidate.rr_target} {candidate.filter_type}",
            "",
            "## Honest Portfolio Comparison",
            "",
            "Metrics use full business-day calendars on a common window so idle days dilute Sharpe honestly.",
            "",
            "| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            snapshot_row(base_is),
            snapshot_row(plus_is),
            snapshot_row(candidate_is),
            "",
            "IS deltas from adding NYSE_CLOSE to the live 6-lane book:",
            "",
            f"- Annualized R delta: `{fmt_num(delta_annual, 1, signed=True)}`",
            f"- Honest Sharpe delta: `{fmt_num(delta_sharpe, 3, signed=True)}`",
            f"- Max drawdown delta: `{fmt_num(delta_dd, 1, signed=True)}`",
            "",
            "## Candidate Diversification vs Live Book (IS)",
            "",
            *corr_lines,
            "",
            f"- Corr to aggregate live-book daily R: `{fmt_num(corr_summary_is['corr_to_base_portfolio'], 3, signed=True)}`",
            f"- NYSE_CLOSE trade days: `{corr_summary_is['candidate_trade_days']}`",
            f"- Candidate days overlapping any live lane: `{corr_summary_is['days_with_any_live_overlap']}` "
            f"(`{fmt_num(corr_summary_is['pct_candidate_days_with_live_overlap'] * 100 if not np.isnan(corr_summary_is['pct_candidate_days_with_live_overlap']) else float('nan'), 1)}%`)",
            f"- Avg live lanes active on NYSE_CLOSE days: `{fmt_num(corr_summary_is['avg_live_lanes_active_on_candidate_days'], 3)}`",
            "",
            "## 2026 OOS Monitor Only",
            "",
            "This section is monitor-only and NOT used as selection proof.",
            "",
            "| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            snapshot_row(base_oos),
            snapshot_row(plus_oos),
            snapshot_row(candidate_oos),
            "",
            "## Verdict",
            "",
            f"**Outcome:** `{verdict}`",
            "",
            reasoning,
            "",
            "## Interpretation",
            "",
            "- `ORB_G8` stays killed. This audit does not rescue a dead filter.",
            "- The remaining question is role. Because the current MNQ profile has a free slot, the honest allocator test is additive first, not replacement-first.",
            "- If additive inclusion helps the live-book math on the common IS window without meaningfully increasing co-movement, `NYSE_CLOSE` deserves continued allocator attention.",
            "- If additive inclusion fails, the branch should fall back to standalone-monitor status or park, not more filter shopping.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "./.venv-wsl/bin/python research/mnq_nyse_close_rr10_role_audit.py",
            "```",
            "",
            "No randomness. Read-only DB. No writes to `validated_setups` / `experimental_strategies` / `live_config` / `lane_allocation.json`.",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> int:
    live_specs = load_live_lane_specs()
    candidate = nyse_close_candidate()
    all_specs = live_specs + [candidate]
    trades_all = {spec.strategy_id: load_trade_records(spec) for spec in all_specs}
    start_all, end_all = common_window(trades_all)

    is_end = min(end_all, HOLDOUT_DATE - timedelta(days=1))
    if start_all > is_end:
        raise RuntimeError("No in-sample window available for role audit")

    oos_start = HOLDOUT_DATE
    has_oos = oos_start <= end_all

    base_trades = {spec.strategy_id: trades_all[spec.strategy_id] for spec in live_specs}
    plus_trades = {spec.strategy_id: trades_all[spec.strategy_id] for spec in all_specs}
    candidate_trades = {candidate.strategy_id: trades_all[candidate.strategy_id]}

    base_is = compute_snapshot("Live 6-lane book", base_trades, start_all, is_end)
    plus_is = compute_snapshot("Live book + NYSE_CLOSE", plus_trades, start_all, is_end)
    candidate_is = compute_snapshot("NYSE_CLOSE standalone", candidate_trades, start_all, is_end)

    if has_oos:
        base_oos = compute_snapshot("Live 6-lane book", base_trades, oos_start, end_all)
        plus_oos = compute_snapshot("Live book + NYSE_CLOSE", plus_trades, oos_start, end_all)
        candidate_oos = compute_snapshot("NYSE_CLOSE standalone", candidate_trades, oos_start, end_all)
    else:
        base_oos = compute_snapshot("Live 6-lane book", {}, oos_start, oos_start)
        plus_oos = compute_snapshot("Live book + NYSE_CLOSE", {}, oos_start, oos_start)
        candidate_oos = compute_snapshot("NYSE_CLOSE standalone", {}, oos_start, oos_start)

    frame_is = daily_return_frame(plus_trades, start_all, is_end)
    corr_rows_is, corr_summary_is = correlation_rows(frame_is, candidate.strategy_id)

    max_slots = ACCOUNT_PROFILES[PROFILE_ID].max_slots

    verdict, reasoning = determine_role_verdict(
        base_is=base_is,
        plus_is=plus_is,
        candidate_is=candidate_is,
        corr_summary_is=corr_summary_is,
        max_slots=max_slots,
        live_lane_count=len(live_specs),
    )
    RESULT_PATH.write_text(
        render_doc(
            live_specs=live_specs,
            candidate=candidate,
            base_is=base_is,
            plus_is=plus_is,
            candidate_is=candidate_is,
            base_oos=base_oos,
            plus_oos=plus_oos,
            candidate_oos=candidate_oos,
            corr_rows_is=corr_rows_is,
            corr_summary_is=corr_summary_is,
            verdict=verdict,
            reasoning=reasoning,
            max_slots=max_slots,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {RESULT_PATH}")
    print(f"Verdict: {verdict}")
    print(reasoning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
