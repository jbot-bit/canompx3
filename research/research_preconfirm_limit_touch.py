#!/usr/bin/env python3
"""
Research: Pre-confirm resting limit model (B)
============================================

Model B definition (user-specified):
- Resting limit order at ORB edge is valid on pure touch.
- No confirm-bar requirement.

This script evaluates a NEW research model (E_PRE_TOUCH) without changing
production entry models (E0/E1/E3). It compares E_PRE_TOUCH vs baseline E0/CB1
for the same RR target.

Assumptions:
- Direction uses existing daily_features break_dir per session/day.
- Entry price is ORB edge in the break direction:
    long  -> orb_high
    short -> orb_low
- Fill occurs at first bar in break-detection window where price touches edge
  (bar_low <= edge <= bar_high).

Outputs:
- research/output/preconfirm_limit_touch_summary.csv
- research/output/preconfirm_limit_touch_findings.md

Usage:
  python research/research_preconfirm_limit_touch.py
  python research/research_preconfirm_limit_touch.py --sessions 1000 1100 --rr-target 2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH
from pipeline.build_daily_features import (  # type: ignore
    _break_detection_window,
    compute_trading_day_utc_range,
)
from pipeline.cost_model import get_cost_spec
from trading_app.outcome_builder import _compute_outcomes_all_rr  # type: ignore

INSTRUMENTS = ["MGC", "MNQ", "MES"]
DEFAULT_SESSIONS = ["1000", "1100"]
DEFAULT_SIZE_GATES = {"MGC": 5.0, "MNQ": 4.0, "MES": 4.0}


def _load_daily_rows(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    session: str,
    use_size_gate: bool,
) -> pd.DataFrame:
    size_col = f"orb_{session}_size"
    gate_clause = ""
    if use_size_gate:
        gate_clause = f"AND {size_col} >= {DEFAULT_SIZE_GATES[symbol]}"

    q = f"""
        SELECT
            trading_day,
            symbol,
            orb_minutes,
            orb_{session}_high AS orb_high,
            orb_{session}_low  AS orb_low,
            orb_{session}_break_dir AS break_dir,
            orb_{session}_break_ts  AS break_ts,
            {size_col} AS orb_size
        FROM daily_features
        WHERE symbol = '{symbol}'
          AND orb_minutes = 5
          AND orb_{session}_break_dir IS NOT NULL
          AND orb_{session}_high IS NOT NULL
          AND orb_{session}_low IS NOT NULL
          {gate_clause}
        ORDER BY trading_day
    """
    return con.execute(q).fetchdf()


def _load_symbol_bars(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    first_day,
    last_day,
) -> pd.DataFrame:
    start_utc, _ = compute_trading_day_utc_range(first_day)
    _, end_utc = compute_trading_day_utc_range(last_day)

    bars = con.execute(
        """
        SELECT ts_utc, open, high, low, close, volume
        FROM bars_1m
        WHERE symbol = ?
          AND ts_utc >= ?::TIMESTAMPTZ
          AND ts_utc < ?::TIMESTAMPTZ
        ORDER BY ts_utc
        """,
        [symbol, start_utc.isoformat(), end_utc.isoformat()],
    ).fetchdf()

    if not bars.empty:
        bars["ts_utc"] = pd.to_datetime(bars["ts_utc"], utc=True)
    return bars


def _first_touch_entry(day_bars: pd.DataFrame, trading_day, session: str, break_dir: str, orb_high: float, orb_low: float):
    """Return first touch entry tuple: (entry_ts, entry_price, stop_price) or None."""
    w_start, w_end = _break_detection_window(trading_day, session, 5)
    w = day_bars[(day_bars["ts_utc"] >= w_start) & (day_bars["ts_utc"] < w_end)]
    if w.empty:
        return None

    if break_dir == "long":
        edge = orb_high
        stop = orb_low
    else:
        edge = orb_low
        stop = orb_high

    touches = w[(w["low"] <= edge) & (w["high"] >= edge)]
    if touches.empty:
        return None

    ts = touches.iloc[0]["ts_utc"].to_pydatetime()
    return ts, float(edge), float(stop)


def _evaluate_pre_touch_for_session(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    session: str,
    rr_target: float,
    use_size_gate: bool,
) -> pd.DataFrame:
    rows = _load_daily_rows(con, symbol, session, use_size_gate)
    if rows.empty:
        return pd.DataFrame()

    bars = _load_symbol_bars(con, symbol, rows.iloc[0]["trading_day"], rows.iloc[-1]["trading_day"])
    if bars.empty:
        return pd.DataFrame()

    ts_all = bars["ts_utc"].values
    cost_spec = get_cost_spec(symbol)

    out = []
    for r in rows.itertuples(index=False):
        td = r.trading_day
        td_start, td_end = compute_trading_day_utc_range(td)

        i0 = int(np.searchsorted(ts_all, pd.Timestamp(td_start).asm8, side="left"))
        i1 = int(np.searchsorted(ts_all, pd.Timestamp(td_end).asm8, side="left"))
        day_bars = bars.iloc[i0:i1]
        if day_bars.empty:
            continue

        entry = _first_touch_entry(
            day_bars=day_bars,
            trading_day=td,
            session=session,
            break_dir=r.break_dir,
            orb_high=float(r.orb_high),
            orb_low=float(r.orb_low),
        )
        if entry is None:
            continue

        entry_ts, entry_price, stop_price = entry
        signal = SimpleNamespace(
            triggered=True,
            entry_ts=entry_ts,
            entry_price=entry_price,
            stop_price=stop_price,
            confirm_bar_ts=entry_ts,
        )

        outcomes = _compute_outcomes_all_rr(
            bars_df=day_bars,
            signal=signal,
            orb_high=float(r.orb_high),
            orb_low=float(r.orb_low),
            break_dir=r.break_dir,
            rr_targets=[rr_target],
            trading_day_end=td_end,
            cost_spec=cost_spec,
            entry_model="E_PRE_TOUCH",
            orb_label=session,
            break_ts=r.break_ts,
        )
        o = outcomes[0]
        out.append(
            {
                "symbol": symbol,
                "session": session,
                "model": "E_PRE_TOUCH",
                "trading_day": td,
                "break_dir": r.break_dir,
                "orb_size": float(r.orb_size) if r.orb_size is not None else None,
                "rr_target": rr_target,
                "entry_ts": o.get("entry_ts"),
                "outcome": o.get("outcome"),
                "pnl_r": o.get("pnl_r"),
            }
        )

    return pd.DataFrame(out)


def _load_baseline_e0(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    session: str,
    rr_target: float,
    use_size_gate: bool,
) -> pd.DataFrame:
    size_col = f"orb_{session}_size"
    gate_clause = ""
    if use_size_gate:
        gate_clause = f"AND d.{size_col} >= {DEFAULT_SIZE_GATES[symbol]}"

    q = f"""
        SELECT
            o.symbol,
            o.orb_label AS session,
            'E0_CB1' AS model,
            o.trading_day,
            d.orb_{session}_break_dir AS break_dir,
            d.{size_col} AS orb_size,
            o.rr_target,
            o.entry_ts,
            o.outcome,
            o.pnl_r
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = '{symbol}'
          AND o.orb_label = '{session}'
          AND o.orb_minutes = 5
          AND o.entry_model = 'E0'
          AND o.confirm_bars = 1
          AND o.rr_target = {rr_target}
          AND o.pnl_r IS NOT NULL
          {gate_clause}
    """
    return con.execute(q).fetchdf()


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    use = df[df["pnl_r"].notna()].copy()
    if use.empty:
        return pd.DataFrame()

    grp = (
        use.groupby(["symbol", "session", "model"], as_index=False)
        .agg(
            n=("pnl_r", "count"),
            avg_r=("pnl_r", "mean"),
            total_r=("pnl_r", "sum"),
            wr=("pnl_r", lambda s: (s > 0).mean()),
        )
    )
    return grp.sort_values(["session", "symbol", "model"])


def _summarize_by_direction(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    use = df[df["pnl_r"].notna() & df["break_dir"].notna()].copy()
    if use.empty:
        return pd.DataFrame()

    grp = (
        use.groupby(["symbol", "session", "model", "break_dir"], as_index=False)
        .agg(
            n=("pnl_r", "count"),
            avg_r=("pnl_r", "mean"),
            total_r=("pnl_r", "sum"),
            wr=("pnl_r", lambda s: (s > 0).mean()),
        )
    )
    return grp.sort_values(["session", "symbol", "break_dir", "model"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Research pre-confirm resting limit touch model")
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS)
    parser.add_argument("--rr-target", type=float, default=2.0)
    parser.add_argument("--no-size-gate", action="store_true", help="Disable instrument size gates")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH
    use_size_gate = not args.no_size_gate

    con = duckdb.connect(str(db_path), read_only=True)

    all_rows = []
    for sym in INSTRUMENTS:
        for sess in args.sessions:
            pre = _evaluate_pre_touch_for_session(
                con=con,
                symbol=sym,
                session=sess,
                rr_target=args.rr_target,
                use_size_gate=use_size_gate,
            )
            base = _load_baseline_e0(
                con=con,
                symbol=sym,
                session=sess,
                rr_target=args.rr_target,
                use_size_gate=use_size_gate,
            )
            if not pre.empty:
                all_rows.append(pre)
            if not base.empty:
                all_rows.append(base)

    con.close()

    if not all_rows:
        print("No rows produced.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    summary = _summarize(combined)
    summary_dir = _summarize_by_direction(combined)

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "preconfirm_limit_touch_summary.csv"
    summary.to_csv(summary_path, index=False)

    summary_dir_path = out_dir / "preconfirm_limit_touch_direction_summary.csv"
    summary_dir.to_csv(summary_dir_path, index=False)

    pivot = summary.pivot_table(
        index=["symbol", "session"],
        columns="model",
        values="avg_r",
        aggfunc="first",
    ).reset_index()
    if "E_PRE_TOUCH" in pivot.columns and "E0_CB1" in pivot.columns:
        pivot["delta_pre_minus_e0"] = pivot["E_PRE_TOUCH"] - pivot["E0_CB1"]

    finding_lines = []
    finding_lines.append("# Pre-confirm Resting Limit Touch (Model B) — Findings")
    finding_lines.append("")
    finding_lines.append(f"- RR target: {args.rr_target}")
    finding_lines.append(f"- Sessions: {', '.join(args.sessions)}")
    finding_lines.append(f"- Size gates: {'ON' if use_size_gate else 'OFF'}")
    finding_lines.append("")
    finding_lines.append("## Summary (avgR)")

    for r in summary.itertuples(index=False):
        finding_lines.append(
            f"- {r.symbol} {r.session} {r.model}: N={r.n}, avgR={r.avg_r:+.4f}, WR={r.wr:.1%}, totR={r.total_r:+.1f}"
        )

    if "delta_pre_minus_e0" in pivot.columns:
        finding_lines.append("")
        finding_lines.append("## Delta vs E0_CB1")
        for r in pivot.itertuples(index=False):
            if hasattr(r, "delta_pre_minus_e0"):
                finding_lines.append(
                    f"- {r.symbol} {r.session}: Δ(pre - E0)={r.delta_pre_minus_e0:+.4f}"
                )

    if not summary_dir.empty:
        finding_lines.append("")
        finding_lines.append("## Directional summary (avgR)")
        for r in summary_dir.itertuples(index=False):
            finding_lines.append(
                f"- {r.symbol} {r.session} {r.break_dir} {r.model}: "
                f"N={r.n}, avgR={r.avg_r:+.4f}, WR={r.wr:.1%}"
            )

    findings_path = out_dir / "preconfirm_limit_touch_findings.md"
    findings_path.write_text("\n".join(finding_lines), encoding="utf-8")

    print(f"Saved: {summary_path}")
    print(f"Saved: {summary_dir_path}")
    print(f"Saved: {findings_path}")
    print("\nTop lines:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
