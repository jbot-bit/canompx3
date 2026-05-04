#!/usr/bin/env python3
"""
Research-only CME FX futures ORB pilot.

Purpose:
- analyze the locked raw Databento pull for 6J / 6B / 6A
- compute research-only ORB cleanliness + fixed-baseline economics
- compare each candidate against the locked dead-FX and live-universe benchmarks
- write a fail-closed result pack without onboarding any new assets

This script is intentionally standalone. It does not touch:
- pipeline asset configs
- gold.db canonical bars_1m / daily_features / orb_outcomes for new assets
- validator / live_config / deployment surfaces

Binding pilot assumptions for the economics leg:
- ORB aperture: O5
- fixed baseline: E2 / CB1 / RR1.0
- descriptive companion metric: E1 / CB1 / RR1.0
- round-trip friction: locked $29.10 for 6J / 6B / 6A

The prereg under-specifies the economics implementation details. This script
locks them to the least-gameable baseline instead of allowing session-specific
"best of" selection.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import databento as db
import duckdb
import pandas as pd

from pipeline.asset_configs import ASSET_CONFIGS
from pipeline.build_daily_features import detect_break, detect_double_break
from pipeline.cost_model import CostSpec
from pipeline.dst import compute_trading_day_utc_range, orb_utc_window
from pipeline.ingest_dbn_mgc import choose_front_contract, compute_trading_days
from pipeline.paths import GOLD_DB_PATH
from trading_app.outcome_builder import compute_single_outcome

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "research" / "output"
RESULTS_DIR = PROJECT_ROOT / "docs" / "audit" / "results"

LOCKED_RR = 1.0
LOCKED_ORB_MINUTES = 5
LOCKED_CONFIRM_BARS = 1
LOCKED_TOTAL_FRICTION = 29.10

OUTRIGHT_MONTH_CODES = "FGHJKMNQUVXZ"

RAW_CONFIG: dict[str, dict[str, Any]] = {
    "6J": {
        "path": PROJECT_ROOT
        / "data"
        / "raw"
        / "databento"
        / "ohlcv-1m"
        / "6J"
        / "pilot_6J_2025-09-18_to_2026-04-16.ohlcv-1m.dbn.zst",
        "outright_pattern": re.compile(rf"^6J[{OUTRIGHT_MONTH_CODES}]\d{{1,2}}$"),
        "prefix_len": 2,
        "cost_spec": CostSpec(
            instrument="6J",
            point_value=12_500_000.0,
            commission_rt=4.10,
            spread_doubled=12.50,
            slippage=12.50,
            tick_size=0.0000005,
            min_ticks_floor=10,
        ),
    },
    "6B": {
        "path": PROJECT_ROOT
        / "data"
        / "raw"
        / "databento"
        / "ohlcv-1m"
        / "6B"
        / "pilot_6B_2025-09-18_to_2026-04-16.ohlcv-1m.dbn.zst",
        "outright_pattern": re.compile(rf"^6B[{OUTRIGHT_MONTH_CODES}]\d{{1,2}}$"),
        "prefix_len": 2,
        "cost_spec": CostSpec(
            instrument="6B",
            point_value=62_500.0,
            commission_rt=4.10,
            spread_doubled=12.50,
            slippage=12.50,
            tick_size=0.0001,
            min_ticks_floor=10,
        ),
    },
    "6A": {
        "path": PROJECT_ROOT
        / "data"
        / "raw"
        / "databento"
        / "ohlcv-1m"
        / "6A"
        / "pilot_6A_2025-09-18_to_2026-04-16.ohlcv-1m.dbn.zst",
        "outright_pattern": re.compile(rf"^6A[{OUTRIGHT_MONTH_CODES}]\d{{1,2}}$"),
        "prefix_len": 2,
        "cost_spec": CostSpec(
            instrument="6A",
            point_value=100_000.0,
            commission_rt=4.10,
            spread_doubled=12.50,
            slippage=12.50,
            tick_size=0.00005,
            min_ticks_floor=10,
        ),
    },
}

PILOT_CASES: list[dict[str, str]] = [
    {
        "asset": "6J",
        "session": "TOKYO_OPEN",
        "dead_fx_kind": "session",
        "dead_fx_symbol": "M6E",
        "dead_fx_session": "TOKYO_OPEN",
        "live_symbol": "MNQ",
        "live_session": "TOKYO_OPEN",
    },
    {
        "asset": "6B",
        "session": "LONDON_METALS",
        "dead_fx_kind": "family",
        "dead_fx_symbol": "M6E",
        "dead_fx_session": "",
        "live_symbol": "MNQ",
        "live_session": "EUROPE_FLOW",
    },
    {
        "asset": "6B",
        "session": "US_DATA_830",
        "dead_fx_kind": "family",
        "dead_fx_symbol": "M6E",
        "dead_fx_session": "",
        "live_symbol": "MNQ",
        "live_session": "COMEX_SETTLE",
    },
    {
        "asset": "6B",
        "session": "US_DATA_1000",
        "dead_fx_kind": "family",
        "dead_fx_symbol": "M6E",
        "dead_fx_session": "",
        "live_symbol": "MNQ",
        "live_session": "COMEX_SETTLE",
    },
    {
        "asset": "6A",
        "session": "LONDON_METALS",
        "dead_fx_kind": "family",
        "dead_fx_symbol": "M6E",
        "dead_fx_session": "",
        "live_symbol": "MNQ",
        "live_session": "EUROPE_FLOW",
    },
    {
        "asset": "6A",
        "session": "US_DATA_830",
        "dead_fx_kind": "family",
        "dead_fx_symbol": "M6E",
        "dead_fx_session": "",
        "live_symbol": "MNQ",
        "live_session": "COMEX_SETTLE",
    },
    {
        "asset": "6A",
        "session": "US_DATA_1000",
        "dead_fx_kind": "family",
        "dead_fx_symbol": "M6E",
        "dead_fx_session": "",
        "live_symbol": "MNQ",
        "live_session": "COMEX_SETTLE",
    },
]


@dataclass
class CoverageAudit:
    utc_dates: int
    brisbane_dates: int
    trading_days_bris_0900: int
    eligible_days: int
    first_trading_day: str | None
    last_trading_day: str | None
    truncated_start_day: str | None
    truncated_end_day: str | None
    sunday_reopen_days_excluded: int


@dataclass
class SessionMetrics:
    eligible_days: int
    e1_trades: int
    e2_trades: int
    double_break_rate: float | None
    fakeout_share: float | None
    continuation_e1: float | None
    continuation_e2: float | None
    cost_adjusted_expr: float | None
    median_orb_risk_usd: float | None
    friction_pct_of_median_risk: float | None
    monthly_stability_note: str


def _load_front_month_bars(symbol: str) -> pd.DataFrame:
    cfg = RAW_CONFIG[symbol]
    store = db.DBNStore.from_file(str(cfg["path"]))
    df = store.to_df().reset_index()
    df.rename(columns={"ts_event": "ts_utc"}, inplace=True)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["symbol"] = df["symbol"].astype(str)
    df = df.set_index("ts_utc").sort_index()

    trading_days = compute_trading_days(df)
    df["trading_day"] = trading_days.values

    per_day_frames: list[pd.DataFrame] = []
    outright_pattern = cfg["outright_pattern"]
    prefix_len = cfg["prefix_len"]
    for _trading_day, day_df in df.groupby("trading_day", sort=True):
        daily_volumes = day_df.groupby("symbol")["volume"].sum().to_dict()
        front = choose_front_contract(daily_volumes, outright_pattern=outright_pattern, prefix_len=prefix_len)
        if front is None:
            continue
        chosen = day_df[day_df["symbol"] == front].copy()
        if chosen.empty:
            continue
        chosen["front_symbol"] = front
        per_day_frames.append(chosen)

    if not per_day_frames:
        raise RuntimeError(f"{symbol}: no front-month bars selected from raw DBN")

    out = pd.concat(per_day_frames).sort_index()
    out = out.reset_index()
    return out


def _first_touch_direction(
    bars_df: pd.DataFrame,
    orb_high: float,
    orb_low: float,
    detection_window_start: datetime,
    detection_window_end: datetime,
) -> dict[str, Any] | None:
    window = bars_df[
        (bars_df["ts_utc"] >= pd.Timestamp(detection_window_start))
        & (bars_df["ts_utc"] < pd.Timestamp(detection_window_end))
    ].sort_values("ts_utc")
    if window.empty:
        return None

    for bar in window.itertuples():
        hit_high = float(bar.high) > orb_high
        hit_low = float(bar.low) < orb_low
        if not hit_high and not hit_low:
            continue
        if hit_high and hit_low:
            return {
                "touch_ts": bar.ts_utc.to_pydatetime(),
                "break_dir": None,
                "ambiguous": True,
            }
        return {
            "touch_ts": bar.ts_utc.to_pydatetime(),
            "break_dir": "long" if hit_high else "short",
            "ambiguous": False,
        }
    return None


def _orb_window_complete(bars_df: pd.DataFrame, orb_start: datetime, orb_end: datetime) -> bool:
    orb_bars = bars_df[
        (bars_df["ts_utc"] >= pd.Timestamp(orb_start)) & (bars_df["ts_utc"] < pd.Timestamp(orb_end))
    ].sort_values("ts_utc")
    if len(orb_bars) != LOCKED_ORB_MINUTES:
        return False
    expected = pd.date_range(start=pd.Timestamp(orb_start), periods=LOCKED_ORB_MINUTES, freq="1min", tz="UTC")
    actual = pd.DatetimeIndex(orb_bars["ts_utc"])
    return actual.equals(expected)


def _compute_candidate_metrics(asset: str, session: str) -> tuple[SessionMetrics, CoverageAudit, list[dict[str, Any]]]:
    bars = _load_front_month_bars(asset)
    bars = bars.sort_values("ts_utc")
    raw_utc_dates = int(bars["ts_utc"].dt.date.nunique())
    bris_dates = int(bars["ts_utc"].dt.tz_convert("Australia/Brisbane").dt.date.nunique())
    trading_days = sorted(
        {pd.Timestamp(td).date() if isinstance(td, pd.Timestamp) else td for td in bars["trading_day"]}
    )

    rows: list[dict[str, Any]] = []
    truncated_start_day = None
    truncated_end_day = None
    sunday_reopen_days_excluded = 0

    cost_spec: CostSpec = RAW_CONFIG[asset]["cost_spec"]

    for trading_day in trading_days:
        td_start, td_end = compute_trading_day_utc_range(trading_day)
        day_bars = bars[(bars["ts_utc"] >= pd.Timestamp(td_start)) & (bars["ts_utc"] < pd.Timestamp(td_end))].copy()
        if day_bars.empty:
            continue

        orb_start, orb_end = orb_utc_window(trading_day, session, LOCKED_ORB_MINUTES)
        if not _orb_window_complete(day_bars, orb_start, orb_end):
            day_min = day_bars["ts_utc"].min().tz_convert("Australia/Brisbane")
            day_max = day_bars["ts_utc"].max().tz_convert("Australia/Brisbane")
            if day_min.hour == 10 and day_min.minute == 0:
                truncated_start_day = truncated_start_day or trading_day.isoformat()
            elif day_max.hour == 9 and day_max.minute == 59:
                truncated_end_day = trading_day.isoformat()
            elif day_min.hour == 8 and day_max.hour == 8:
                sunday_reopen_days_excluded += 1
            continue

        orb_bars = day_bars[
            (day_bars["ts_utc"] >= pd.Timestamp(orb_start)) & (day_bars["ts_utc"] < pd.Timestamp(orb_end))
        ].sort_values("ts_utc")
        orb_high = float(orb_bars["high"].max())
        orb_low = float(orb_bars["low"].min())
        orb_size = float(orb_high - orb_low)

        break_info = detect_break(day_bars, trading_day, session, LOCKED_ORB_MINUTES, orb_high, orb_low)
        two_sided_breach = detect_double_break(day_bars, trading_day, session, LOCKED_ORB_MINUTES, orb_high, orb_low)
        touch_info = _first_touch_direction(day_bars, orb_high, orb_low, orb_end, td_end)

        e1 = None
        if break_info["break_dir"] is not None and break_info["break_ts"] is not None:
            e1 = compute_single_outcome(
                bars_df=day_bars,
                break_ts=break_info["break_ts"],
                orb_high=orb_high,
                orb_low=orb_low,
                break_dir=break_info["break_dir"],
                rr_target=LOCKED_RR,
                confirm_bars=LOCKED_CONFIRM_BARS,
                trading_day_end=td_end,
                cost_spec=cost_spec,
                entry_model="E1",
                orb_label=session,
                trading_day=trading_day,
                orb_minutes=LOCKED_ORB_MINUTES,
            )

        e2 = None
        if touch_info and not touch_info["ambiguous"] and touch_info["break_dir"] is not None:
            e2 = compute_single_outcome(
                bars_df=day_bars,
                break_ts=touch_info["touch_ts"],
                orb_high=orb_high,
                orb_low=orb_low,
                break_dir=touch_info["break_dir"],
                rr_target=LOCKED_RR,
                confirm_bars=LOCKED_CONFIRM_BARS,
                trading_day_end=td_end,
                cost_spec=cost_spec,
                entry_model="E2",
                orb_label=session,
                orb_end_utc=orb_end,
                trading_day=trading_day,
                orb_minutes=LOCKED_ORB_MINUTES,
            )

        rows.append(
            {
                "trading_day": trading_day,
                "front_symbol": str(day_bars["front_symbol"].iloc[0]),
                "orb_size": orb_size,
                "two_sided_breach": two_sided_breach,
                "break_dir_e1": break_info["break_dir"],
                "touch_dir_e2": touch_info["break_dir"] if touch_info else None,
                "touch_ambiguous": bool(touch_info["ambiguous"]) if touch_info else False,
                "e1_entry": bool(e1 and e1["entry_ts"] is not None),
                "e1_win": bool(e1 and e1["pnl_r"] is not None and e1["pnl_r"] > 0),
                "e1_pnl_r": e1["pnl_r"] if e1 else None,
                "e2_entry": bool(e2 and e2["entry_ts"] is not None),
                "e2_win": bool(e2 and e2["pnl_r"] is not None and e2["pnl_r"] > 0),
                "e2_loss": bool(e2 and e2["pnl_r"] is not None and e2["pnl_r"] < 0),
                "e2_pnl_r": e2["pnl_r"] if e2 else None,
                "e2_risk_dollars": e2["risk_dollars"] if e2 else None,
            }
        )

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        raise RuntimeError(f"{asset} {session}: no eligible pilot days after front-month + ORB completeness filters")

    e1_trades = int(detail_df["e1_entry"].sum())
    e2_trades = int(detail_df["e2_entry"].sum())

    def _safe_mean(mask: pd.Series) -> float | None:
        if len(mask) == 0:
            return None
        return float(mask.mean())

    double_break_rate = _safe_mean(detail_df["two_sided_breach"].fillna(False).astype(bool))
    continuation_e1 = float(detail_df.loc[detail_df["e1_entry"], "e1_win"].mean()) if e1_trades else None
    continuation_e2 = float(detail_df.loc[detail_df["e2_entry"], "e2_win"].mean()) if e2_trades else None
    fakeout_share = float(detail_df.loc[detail_df["e2_entry"], "e2_loss"].mean()) if e2_trades else None
    cost_adjusted_expr = float(detail_df.loc[detail_df["e2_entry"], "e2_pnl_r"].mean()) if e2_trades else None
    median_orb_risk_usd = float(detail_df.loc[detail_df["e2_entry"], "e2_risk_dollars"].median()) if e2_trades else None
    friction_pct = (
        LOCKED_TOTAL_FRICTION / median_orb_risk_usd if median_orb_risk_usd and median_orb_risk_usd > 0 else None
    )

    e2_month = detail_df.loc[detail_df["e2_entry"], ["trading_day", "e2_pnl_r"]].copy()
    if e2_month.empty:
        monthly_stability_note = "insufficient E2 trades for month split"
    else:
        e2_month["month"] = pd.to_datetime(e2_month["trading_day"]).dt.to_period("M").astype(str)
        monthly = e2_month.groupby("month").agg(n=("e2_pnl_r", "size"), expr=("e2_pnl_r", "mean")).reset_index()
        positive = int((monthly["expr"] > 0).sum())
        total = int(len(monthly))
        worst = monthly.sort_values("expr").iloc[0]
        monthly_stability_note = (
            f"{positive}/{total} months positive on E2 RR1.0; "
            f"worst {worst['month']}={worst['expr']:+.3f}R (n={int(worst['n'])})"
        )

    metrics = SessionMetrics(
        eligible_days=int(len(detail_df)),
        e1_trades=e1_trades,
        e2_trades=e2_trades,
        double_break_rate=double_break_rate,
        fakeout_share=fakeout_share,
        continuation_e1=continuation_e1,
        continuation_e2=continuation_e2,
        cost_adjusted_expr=cost_adjusted_expr,
        median_orb_risk_usd=median_orb_risk_usd,
        friction_pct_of_median_risk=friction_pct,
        monthly_stability_note=monthly_stability_note,
    )
    coverage = CoverageAudit(
        utc_dates=raw_utc_dates,
        brisbane_dates=bris_dates,
        trading_days_bris_0900=len(trading_days),
        eligible_days=int(len(detail_df)),
        first_trading_day=trading_days[0].isoformat() if trading_days else None,
        last_trading_day=trading_days[-1].isoformat() if trading_days else None,
        truncated_start_day=truncated_start_day,
        truncated_end_day=truncated_end_day,
        sunday_reopen_days_excluded=sunday_reopen_days_excluded,
    )
    return metrics, coverage, rows


def _query_session_metrics(con: duckdb.DuckDBPyConnection, symbol: str, session: str) -> SessionMetrics:
    size_col = f"orb_{session}_size"
    db_col = f"orb_{session}_double_break"
    sql = f"""
    WITH day_base AS (
      SELECT trading_day, {size_col} AS orb_size, {db_col} AS double_break
      FROM daily_features
      WHERE symbol = ? AND orb_minutes = ? AND {size_col} IS NOT NULL
    ),
    e1 AS (
      SELECT trading_day, pnl_r, risk_dollars, entry_ts
      FROM orb_outcomes
      WHERE symbol = ? AND orb_label = ? AND orb_minutes = ? AND rr_target = ? AND confirm_bars = ? AND entry_model = 'E1'
    ),
    e2 AS (
      SELECT trading_day, pnl_r, risk_dollars, entry_ts
      FROM orb_outcomes
      WHERE symbol = ? AND orb_label = ? AND orb_minutes = ? AND rr_target = ? AND confirm_bars = ? AND entry_model = 'E2'
    )
    SELECT
      COUNT(*) AS eligible_days,
      AVG(CASE WHEN day_base.double_break THEN 1.0 ELSE 0.0 END) AS double_break_rate,
      COUNT(*) FILTER (WHERE e1.entry_ts IS NOT NULL) AS e1_trades,
      COUNT(*) FILTER (WHERE e2.entry_ts IS NOT NULL) AS e2_trades,
      AVG(CASE WHEN e1.pnl_r > 0 THEN 1.0 ELSE 0.0 END) FILTER (WHERE e1.entry_ts IS NOT NULL) AS continuation_e1,
      AVG(CASE WHEN e2.pnl_r > 0 THEN 1.0 ELSE 0.0 END) FILTER (WHERE e2.entry_ts IS NOT NULL) AS continuation_e2,
      AVG(CASE WHEN e2.pnl_r < 0 THEN 1.0 ELSE 0.0 END) FILTER (WHERE e2.entry_ts IS NOT NULL) AS fakeout_share,
      AVG(e2.pnl_r) FILTER (WHERE e2.entry_ts IS NOT NULL) AS cost_adjusted_expr,
      MEDIAN(e2.risk_dollars) FILTER (WHERE e2.entry_ts IS NOT NULL) AS median_orb_risk_usd
    FROM day_base
    LEFT JOIN e1 USING (trading_day)
    LEFT JOIN e2 USING (trading_day)
    """
    row = con.execute(
        sql,
        [
            symbol,
            LOCKED_ORB_MINUTES,
            symbol,
            session,
            LOCKED_ORB_MINUTES,
            LOCKED_RR,
            LOCKED_CONFIRM_BARS,
            symbol,
            session,
            LOCKED_ORB_MINUTES,
            LOCKED_RR,
            LOCKED_CONFIRM_BARS,
        ],
    ).fetchone()
    if row is None:
        raise RuntimeError(f"no benchmark metrics for {symbol} {session}")

    median_risk = float(row[8]) if row[8] is not None else None
    friction_pct = LOCKED_TOTAL_FRICTION / median_risk if median_risk and median_risk > 0 else None
    return SessionMetrics(
        eligible_days=int(row[0]),
        double_break_rate=float(row[1]) if row[1] is not None else None,
        e1_trades=int(row[2]),
        e2_trades=int(row[3]),
        continuation_e1=float(row[4]) if row[4] is not None else None,
        continuation_e2=float(row[5]) if row[5] is not None else None,
        fakeout_share=float(row[6]) if row[6] is not None else None,
        cost_adjusted_expr=float(row[7]) if row[7] is not None else None,
        median_orb_risk_usd=median_risk,
        friction_pct_of_median_risk=friction_pct,
        monthly_stability_note="benchmark query path",
    )


def _weighted_mean(parts: list[tuple[float | None, int]]) -> float | None:
    valid = [(value, weight) for value, weight in parts if value is not None and weight > 0]
    if not valid:
        return None
    denom = sum(weight for _, weight in valid)
    if denom <= 0:
        return None
    return float(sum(value * weight for value, weight in valid) / denom)


def _query_family_mean_metrics(con: duckdb.DuckDBPyConnection, symbol: str) -> SessionMetrics:
    sessions = list(ASSET_CONFIGS[symbol]["enabled_sessions"])
    per_session = [_query_session_metrics(con, symbol, session) for session in sessions]

    eligible_days = sum(m.eligible_days for m in per_session)
    e1_trades = sum(m.e1_trades for m in per_session)
    e2_trades = sum(m.e2_trades for m in per_session)
    double_break_rate = _weighted_mean([(m.double_break_rate, m.eligible_days) for m in per_session])
    continuation_e1 = _weighted_mean([(m.continuation_e1, m.e1_trades) for m in per_session])
    continuation_e2 = _weighted_mean([(m.continuation_e2, m.e2_trades) for m in per_session])
    fakeout_share = _weighted_mean([(m.fakeout_share, m.e2_trades) for m in per_session])
    cost_adjusted_expr = _weighted_mean([(m.cost_adjusted_expr, m.e2_trades) for m in per_session])
    median_orb_risk_usd = _weighted_mean([(m.median_orb_risk_usd, m.e2_trades) for m in per_session])
    friction_pct = (
        LOCKED_TOTAL_FRICTION / median_orb_risk_usd if median_orb_risk_usd and median_orb_risk_usd > 0 else None
    )

    return SessionMetrics(
        eligible_days=eligible_days,
        e1_trades=e1_trades,
        e2_trades=e2_trades,
        double_break_rate=double_break_rate,
        fakeout_share=fakeout_share,
        continuation_e1=continuation_e1,
        continuation_e2=continuation_e2,
        cost_adjusted_expr=cost_adjusted_expr,
        median_orb_risk_usd=median_orb_risk_usd,
        friction_pct_of_median_risk=friction_pct,
        monthly_stability_note=f"family mean across {len(sessions)} M6E sessions",
    )


def _fmt_pct(x: float | None) -> str:
    return "NA" if x is None else f"{x * 100:.1f}%"


def _fmt_r(x: float | None) -> str:
    return "NA" if x is None else f"{x:+.3f}R"


def _fmt_usd(x: float | None) -> str:
    return "NA" if x is None else f"${x:,.2f}"


def _compare_direction(
    candidate: float | None, benchmark: float | None, *, higher_is_better: bool
) -> tuple[float | None, bool | None]:
    if candidate is None or benchmark is None:
        return None, None
    delta = candidate - benchmark
    better = delta > 0 if higher_is_better else delta < 0
    return delta, better


def _build_case_result(
    case: dict[str, str],
    candidate: SessionMetrics,
    coverage: CoverageAudit,
    dead_fx: SessionMetrics,
    live_proxy: SessionMetrics,
) -> dict[str, Any]:
    db_delta, _ = _compare_direction(candidate.double_break_rate, dead_fx.double_break_rate, higher_is_better=False)
    fakeout_delta, _ = _compare_direction(candidate.fakeout_share, dead_fx.fakeout_share, higher_is_better=False)
    cont_delta, _ = _compare_direction(candidate.continuation_e2, dead_fx.continuation_e2, higher_is_better=True)
    expr_delta_dead, _ = _compare_direction(
        candidate.cost_adjusted_expr, dead_fx.cost_adjusted_expr, higher_is_better=True
    )

    live_worse = 0
    for cand_val, live_val, higher_is_better in [
        (candidate.double_break_rate, live_proxy.double_break_rate, False),
        (candidate.fakeout_share, live_proxy.fakeout_share, False),
        (candidate.continuation_e2, live_proxy.continuation_e2, True),
        (candidate.cost_adjusted_expr, live_proxy.cost_adjusted_expr, True),
    ]:
        if cand_val is None or live_val is None:
            continue
        if higher_is_better:
            if cand_val < live_val:
                live_worse += 1
        else:
            if cand_val > live_val:
                live_worse += 1

    pass_cleanliness = (
        db_delta is not None
        and db_delta <= -0.10
        and fakeout_delta is not None
        and fakeout_delta <= -0.10
        and cont_delta is not None
        and cont_delta >= 0.05
    )
    pass_economics = (
        candidate.cost_adjusted_expr is not None
        and candidate.cost_adjusted_expr > 0.05
        and expr_delta_dead is not None
        and expr_delta_dead >= 0.10
        and candidate.friction_pct_of_median_risk is not None
        and candidate.friction_pct_of_median_risk < 0.35
    )
    pass_live_guardrail = live_worse < 3
    verdict = "GO" if pass_cleanliness and pass_economics and pass_live_guardrail else "NO_GO"

    return {
        "asset": case["asset"],
        "session": case["session"],
        "coverage_audit": asdict(coverage),
        "candidate": asdict(candidate),
        "dead_fx_label": (
            f"{case['dead_fx_symbol']} {case['dead_fx_session']}"
            if case["dead_fx_kind"] == "session"
            else f"{case['dead_fx_symbol']} broad ORB family mean"
        ),
        "dead_fx": asdict(dead_fx),
        "live_proxy_label": f"{case['live_symbol']} {case['live_session']}",
        "live_proxy": asdict(live_proxy),
        "delta_vs_dead_fx": {
            "double_break_rate": db_delta,
            "fakeout_share": fakeout_delta,
            "continuation_e2": cont_delta,
            "cost_adjusted_expr": expr_delta_dead,
        },
        "live_worse_count": live_worse,
        "pass_cleanliness": pass_cleanliness,
        "pass_economics": pass_economics,
        "pass_live_guardrail": pass_live_guardrail,
        "verdict": verdict,
    }


def _render_markdown(results: list[dict[str, Any]]) -> str:
    go_rows = [r for r in results if r["verdict"] == "GO"]
    stage_verdict = "GO" if go_rows else "NO_GO"

    lines = [
        "# CME FX Futures ORB Pilot",
        "",
        f"Stage verdict: **{stage_verdict}**",
        "",
        "Locked implementation:",
        f"- ORB aperture: `O{LOCKED_ORB_MINUTES}`",
        f"- Economics baseline: `E2 / CB{LOCKED_CONFIRM_BARS} / RR{LOCKED_RR:.1f}`",
        f"- Descriptive companion: `E1 / CB{LOCKED_CONFIRM_BARS} / RR{LOCKED_RR:.1f}`",
        f"- Locked round-trip friction: `${LOCKED_TOTAL_FRICTION:.2f}`",
        "",
        "Coverage audit:",
        "- Raw request metadata claimed a `180`-day pull window, but that is not the realized eligible-session sample.",
        "- Actual candidate coverage is measured from decoded raw bars after front-month filtering, Brisbane 09:00 trading-day assignment, and complete ORB-window enforcement.",
        "",
    ]

    for result in results:
        cand = result["candidate"]
        cov = result["coverage_audit"]
        dead = result["dead_fx"]
        live = result["live_proxy"]
        delta = result["delta_vs_dead_fx"]
        lines.extend(
            [
                f"## {result['asset']} {result['session']}",
                "",
                f"- Verdict: **{result['verdict']}**",
                f"- Coverage: raw UTC dates `{cov['utc_dates']}`, Brisbane trading days `{cov['trading_days_bris_0900']}`, eligible session days `{cov['eligible_days']}`",
                f"- Boundary losses: truncated start `{cov['truncated_start_day'] or 'none'}`, truncated end `{cov['truncated_end_day'] or 'none'}`, Sunday reopen pseudo-days excluded `{cov['sunday_reopen_days_excluded']}`",
                f"- Candidate metrics: double-break `{_fmt_pct(cand['double_break_rate'])}`, fakeout `{_fmt_pct(cand['fakeout_share'])}`, continuation E1 `{_fmt_pct(cand['continuation_e1'])}`, continuation E2 `{_fmt_pct(cand['continuation_e2'])}`, E2 ExpR `{_fmt_r(cand['cost_adjusted_expr'])}`, median risk `{_fmt_usd(cand['median_orb_risk_usd'])}`, friction/risk `{_fmt_pct(cand['friction_pct_of_median_risk'])}`",
                f"- Dead benchmark `{result['dead_fx_label']}`: double-break `{_fmt_pct(dead['double_break_rate'])}`, fakeout `{_fmt_pct(dead['fakeout_share'])}`, continuation E2 `{_fmt_pct(dead['continuation_e2'])}`, E2 ExpR `{_fmt_r(dead['cost_adjusted_expr'])}`",
                f"- Live benchmark `{result['live_proxy_label']}`: double-break `{_fmt_pct(live['double_break_rate'])}`, fakeout `{_fmt_pct(live['fakeout_share'])}`, continuation E2 `{_fmt_pct(live['continuation_e2'])}`, E2 ExpR `{_fmt_r(live['cost_adjusted_expr'])}`",
                f"- Delta vs dead FX: double-break `{_fmt_pct(delta['double_break_rate'])}`, fakeout `{_fmt_pct(delta['fakeout_share'])}`, continuation E2 `{_fmt_pct(delta['continuation_e2'])}`, E2 ExpR `{_fmt_r(delta['cost_adjusted_expr'])}`",
                f"- Locked gates: cleanliness `{result['pass_cleanliness']}`, economics `{result['pass_economics']}`, live guardrail `{result['pass_live_guardrail']}`",
                f"- Stability note: {cand['monthly_stability_note']}",
                "",
            ]
        )

    if go_rows:
        lines.extend(
            [
                "## Survived",
                "",
                "At least one asset-session cleared the locked pilot gates, so the pilot justifies a follow-up preregistration instead of immediate kill.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Failed Closed",
                "",
                "No asset-session met every locked gate. The correct action is to stop here and not rescue the pilot by extending windows, swapping assets, or widening the session list.",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    results: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for case in PILOT_CASES:
        candidate, coverage, case_rows = _compute_candidate_metrics(case["asset"], case["session"])
        if case["dead_fx_kind"] == "session":
            dead_fx = _query_session_metrics(con, case["dead_fx_symbol"], case["dead_fx_session"])
        else:
            dead_fx = _query_family_mean_metrics(con, case["dead_fx_symbol"])
        live_proxy = _query_session_metrics(con, case["live_symbol"], case["live_session"])
        result = _build_case_result(case, candidate, coverage, dead_fx, live_proxy)
        results.append(result)
        for row in case_rows:
            detail_rows.append({"asset": case["asset"], "session": case["session"], **row})

    artifact = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "locked_rr": LOCKED_RR,
        "locked_orb_minutes": LOCKED_ORB_MINUTES,
        "locked_confirm_bars": LOCKED_CONFIRM_BARS,
        "locked_total_friction": LOCKED_TOTAL_FRICTION,
        "results": results,
    }

    json_path = OUTPUT_DIR / "cme_fx_futures_orb_pilot.json"
    csv_path = OUTPUT_DIR / "cme_fx_futures_orb_pilot_detail.csv"
    md_path = RESULTS_DIR / "2026-04-17-cme-fx-futures-orb-pilot.md"

    json_path.write_text(json.dumps(artifact, indent=2, default=str) + "\n", encoding="utf-8")
    pd.DataFrame(detail_rows).to_csv(csv_path, index=False)
    md_path.write_text(_render_markdown(results), encoding="utf-8")

    stage_verdict = "GO" if any(r["verdict"] == "GO" for r in results) else "NO_GO"
    print(f"stage_verdict={stage_verdict}")
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    print(f"md={md_path}")


if __name__ == "__main__":
    main()
