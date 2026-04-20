#!/usr/bin/env python3
"""Sweep reclaim standalone v2 — locked standalone reversal family.

Pre-reg:
  docs/audit/hypotheses/2026-04-21-sweep-reclaim-standalone-v2.yaml

Canonical sources only:
  - daily_features
  - bars_1m

No writes to validated_setups or live config.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.cost_model import get_cost_spec, to_r_multiple
from pipeline.dst import orb_utc_window
from research.lib import bh_fdr, connect_db, resolve_level_reference
from research.lib.level_interactions import classify_level_interaction
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

LOCKED_INSTRUMENTS = ["MES", "MNQ"]
LOCKED_SESSIONS = ["EUROPE_FLOW", "NYSE_OPEN"]
LOCKED_LEVELS = {
    "prev_day_high": "below",
    "prev_day_low": "above",
}
LOCKED_RR = 1.5
COMMON_START = date(2019, 5, 6)
HOLDOUT_START = HOLDOUT_SACRED_FROM
EVENT_WINDOW_MINUTES = 60
RECLAIM_LOOKAHEAD_BARS = 2
OUTPUT_DIR = Path("research/output")
OUTPUT_TRADES = OUTPUT_DIR / "sweep_reclaim_standalone_v2_trades.csv"
OUTPUT_CELLS = OUTPUT_DIR / "sweep_reclaim_standalone_v2_cells.csv"
OUTPUT_MD = Path("docs/audit/results/2026-04-21-sweep-reclaim-standalone-v2.md")


@dataclass(frozen=True)
class TradeObservation:
    instrument: str
    session: str
    level_name: str
    direction: str
    trading_day: date
    sample: str
    entry_ts: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    sweep_extreme: float
    interaction_bar_index: int
    reclaim_bar_index: int
    risk_points: float
    pnl_r: float
    outcome: str


@dataclass
class CellResult:
    instrument: str
    session: str
    level_name: str
    direction: str
    n_is: int
    expr_is: float
    win_rate_is: float
    t_is: float
    p_one_is: float
    q_bh: float
    h1_pass: bool
    wfe: float
    c6_pass: bool
    n_oos: int
    expr_oos: float
    ratio_oos_is: float
    c8_pass: bool
    era_min_expr: float
    era_n_eligible: int
    era_n_fail: int
    c9_pass: bool
    verdict: str


def reclaim_direction(reference_side: str) -> int:
    if reference_side == "below":
        return -1
    if reference_side == "above":
        return 1
    raise ValueError(f"Unsupported reference_side: {reference_side}")


def sweep_extreme_from_window(
    bars_window: pd.DataFrame,
    *,
    event_bar_index: int,
    reclaim_bar_index: int,
    direction: int,
) -> float:
    """Chronology-safe sweep extreme known by the reclaim close.

    Uses the max/min excursion between the swept close-through bar and the
    reclaim bar, inclusive. This is the narrowest conservative reading of
    "beyond sweep extreme" using only completed bars available at entry.
    """
    seq = bars_window.iloc[event_bar_index : reclaim_bar_index + 1]
    if direction == -1:
        return float(seq["high"].max())
    return float(seq["low"].min())


def stop_price_from_sweep_extreme(extreme: float, direction: int, tick_size: float) -> float:
    if direction == -1:
        return extreme + tick_size
    return extreme - tick_size


def target_price_from_rr(entry_price: float, stop_price: float, direction: int, rr: float) -> float:
    risk_points = abs(entry_price - stop_price)
    if direction == 1:
        return entry_price + rr * risk_points
    return entry_price - rr * risk_points


def one_tailed_positive_test(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 10:
        return float("nan"), float("nan")
    t_stat, _ = stats.ttest_1samp(values, 0.0)
    if np.isnan(t_stat):
        return float("nan"), float("nan")
    p_one = float(1.0 - stats.t.cdf(float(t_stat), len(values) - 1))
    return float(t_stat), p_one


def walk_forward_efficiency(is_df: pd.DataFrame, n_folds: int = 3) -> float:
    """C6: mean OOS-slice ExpR divided by full-IS ExpR."""
    if len(is_df) < n_folds * 30:
        return float("nan")
    ordered = is_df.sort_values("trading_day").reset_index(drop=True)
    is_mean = float(ordered["pnl_r"].mean())
    if is_mean <= 0 or np.isnan(is_mean):
        return float("nan")
    fold_edges = np.linspace(0, len(ordered), n_folds + 1, dtype=int)
    oos_means: list[float] = []
    for i in range(1, n_folds):
        oos_slice = ordered.iloc[fold_edges[i] : fold_edges[i + 1]]
        if len(oos_slice) < 20:
            continue
        oos_means.append(float(oos_slice["pnl_r"].mean()))
    if not oos_means:
        return float("nan")
    return float(np.mean(oos_means) / is_mean)


def era_stability(is_df: pd.DataFrame) -> tuple[float, int, int]:
    """C9: no calendar year with N>=20 and ExpR<-0.05."""
    if is_df.empty:
        return float("nan"), 0, 0
    tmp = is_df.copy()
    tmp["year"] = pd.to_datetime(tmp["trading_day"]).dt.year
    rows = tmp.groupby("year")["pnl_r"].agg(["count", "mean"]).reset_index()
    eligible = rows[rows["count"] >= 20]
    if eligible.empty:
        return float("nan"), 0, 0
    worst = float(eligible["mean"].min())
    n_fail = int((eligible["mean"] < -0.05).sum())
    return worst, int(len(eligible)), n_fail


def load_feature_rows(con) -> pd.DataFrame:
    instruments_sql = ", ".join(f"'{symbol}'" for symbol in LOCKED_INSTRUMENTS)
    sql = f"""
    SELECT trading_day, symbol, atr_20, prev_day_high, prev_day_low, prev_day_close
    FROM daily_features
    WHERE orb_minutes = 5
      AND symbol IN ({instruments_sql})
      AND trading_day >= ?
    ORDER BY symbol, trading_day
    """
    df = con.execute(sql, [COMMON_START]).fetchdf()
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    return df


def load_bars_for_day(con, symbol: str, trading_day: date) -> pd.DataFrame:
    from pipeline.dst import compute_trading_day_utc_range

    day_start, day_end = compute_trading_day_utc_range(trading_day)
    sql = """
    SELECT ts_utc, open, high, low, close
    FROM bars_1m
    WHERE symbol = ?
      AND ts_utc >= ?
      AND ts_utc < ?
    ORDER BY ts_utc
    """
    df = con.execute(sql, [symbol, day_start, day_end]).fetchdf()
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df


def session_window(bars_day: pd.DataFrame, trading_day: date, session: str) -> pd.DataFrame:
    start_utc, _ = orb_utc_window(trading_day, session, 1)
    end_utc = start_utc + timedelta(minutes=EVENT_WINDOW_MINUTES)
    return bars_day[
        (bars_day["ts_utc"] >= pd.Timestamp(start_utc))
        & (bars_day["ts_utc"] < pd.Timestamp(end_utc))
    ].reset_index(drop=True)


def resolve_trade_outcome(
    bars_day: pd.DataFrame,
    *,
    direction: int,
    entry_price: float,
    stop_price: float,
    target_price: float,
    start_idx: int,
    spec,
) -> tuple[str, float]:
    """Resolve stop/target from the bars after the entry bar; EOD close otherwise."""
    is_long = direction == 1
    for i in range(start_idx, len(bars_day)):
        bar = bars_day.iloc[i]
        stop_hit = bar["low"] <= stop_price if is_long else bar["high"] >= stop_price
        target_hit = bar["high"] >= target_price if is_long else bar["low"] <= target_price
        if stop_hit and target_hit:
            pnl_points = stop_price - entry_price if is_long else entry_price - stop_price
            return "ambiguous_loss", to_r_multiple(spec, entry_price, stop_price, pnl_points)
        if stop_hit:
            pnl_points = stop_price - entry_price if is_long else entry_price - stop_price
            return "loss", to_r_multiple(spec, entry_price, stop_price, pnl_points)
        if target_hit:
            pnl_points = target_price - entry_price if is_long else entry_price - target_price
            return "win", to_r_multiple(spec, entry_price, stop_price, pnl_points)

    last_close = float(bars_day.iloc[-1]["close"])
    pnl_points = (last_close - entry_price) if is_long else (entry_price - last_close)
    return "eod", to_r_multiple(spec, entry_price, stop_price, pnl_points)


def collect_trades() -> list[TradeObservation]:
    trades: list[TradeObservation] = []
    bar_cache: dict[tuple[str, date], pd.DataFrame] = {}

    with connect_db() as con:
        feature_rows = load_feature_rows(con)
        total_rows = len(feature_rows)

        for idx, row in enumerate(feature_rows.itertuples(index=False), start=1):
            if idx % 500 == 0:
                print(f"Progress: {idx}/{total_rows} feature rows", flush=True)

            symbol = row.symbol
            trading_day = row.trading_day
            sample = "IS" if trading_day < HOLDOUT_START else "OOS"
            spec = get_cost_spec(symbol)

            cache_key = (symbol, trading_day)
            if cache_key not in bar_cache:
                bar_cache[cache_key] = load_bars_for_day(con, symbol, trading_day)
            bars_day = bar_cache[cache_key]
            if bars_day.empty:
                continue

            feature_map = {
                "prev_day_high": row.prev_day_high,
                "prev_day_low": row.prev_day_low,
                "prev_day_close": row.prev_day_close,
            }

            for session in LOCKED_SESSIONS:
                bars_window = session_window(bars_day, trading_day, session)
                if len(bars_window) <= RECLAIM_LOOKAHEAD_BARS + 1:
                    continue

                for level_name, reference_side in LOCKED_LEVELS.items():
                    ref = resolve_level_reference(feature_map, level_name, target_session=session)
                    if ref.unavailable_reason is not None or ref.price is None:
                        continue

                    event = classify_level_interaction(
                        bars_window,
                        level_name=level_name,
                        level_price=ref.price,
                        reference_side=reference_side,
                        sweep_epsilon=0.0,
                        reclaim_lookahead_bars=RECLAIM_LOOKAHEAD_BARS,
                    )
                    if event.unavailable_reason is not None:
                        continue
                    if event.interaction_kind != "close_through" or not event.swept or not event.reclaimed:
                        continue
                    if event.bar_index is None or event.reclaim_bar_index is None:
                        continue

                    direction = reclaim_direction(reference_side)
                    entry_bar = bars_window.iloc[event.reclaim_bar_index]
                    entry_price = float(entry_bar["close"])
                    sweep_extreme = sweep_extreme_from_window(
                        bars_window,
                        event_bar_index=event.bar_index,
                        reclaim_bar_index=event.reclaim_bar_index,
                        direction=direction,
                    )
                    stop_price = stop_price_from_sweep_extreme(
                        sweep_extreme,
                        direction=direction,
                        tick_size=spec.tick_size,
                    )
                    risk_points = abs(entry_price - stop_price)
                    if risk_points <= 0:
                        continue
                    target_price = target_price_from_rr(entry_price, stop_price, direction, LOCKED_RR)

                    entry_ts = pd.Timestamp(entry_bar["ts_utc"])
                    after_entry = bars_day[bars_day["ts_utc"] > entry_ts].reset_index(drop=True)
                    if after_entry.empty:
                        continue
                    outcome, pnl_r = resolve_trade_outcome(
                        after_entry,
                        direction=direction,
                        entry_price=entry_price,
                        stop_price=stop_price,
                        target_price=target_price,
                        start_idx=0,
                        spec=spec,
                    )

                    trades.append(
                        TradeObservation(
                            instrument=symbol,
                            session=session,
                            level_name=level_name,
                            direction="long" if direction == 1 else "short",
                            trading_day=trading_day,
                            sample=sample,
                            entry_ts=entry_ts,
                            entry_price=entry_price,
                            stop_price=stop_price,
                            target_price=target_price,
                            sweep_extreme=sweep_extreme,
                            interaction_bar_index=int(event.bar_index),
                            reclaim_bar_index=int(event.reclaim_bar_index),
                            risk_points=risk_points,
                            pnl_r=float(pnl_r),
                            outcome=outcome,
                        )
                    )

    return trades


def summarize_trades(trades: list[TradeObservation]) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    if trades_df.empty:
        return trades_df, pd.DataFrame()

    cell_index = [
        (instrument, session, level_name)
        for instrument in LOCKED_INSTRUMENTS
        for session in LOCKED_SESSIONS
        for level_name in LOCKED_LEVELS
    ]

    summary_rows: list[dict] = []
    pvals: list[float] = []
    h1_candidates: list[bool] = []

    for instrument, session, level_name in cell_index:
        sub_is = trades_df[
            (trades_df["instrument"] == instrument)
            & (trades_df["session"] == session)
            & (trades_df["level_name"] == level_name)
            & (trades_df["sample"] == "IS")
        ].copy()
        sub_oos = trades_df[
            (trades_df["instrument"] == instrument)
            & (trades_df["session"] == session)
            & (trades_df["level_name"] == level_name)
            & (trades_df["sample"] == "OOS")
        ].copy()

        direction = "short" if level_name == "prev_day_high" else "long"
        is_vals = sub_is["pnl_r"].to_numpy(dtype=float)
        oos_vals = sub_oos["pnl_r"].to_numpy(dtype=float)

        n_is = len(is_vals)
        expr_is = float(np.mean(is_vals)) if n_is else float("nan")
        wr_is = float((is_vals > 0).mean()) if n_is else float("nan")
        t_is, p_one_is = one_tailed_positive_test(is_vals)

        h1_candidate = bool(n_is >= 100 and not np.isnan(expr_is) and expr_is > 0 and not np.isnan(p_one_is))
        pvals.append(float(p_one_is) if h1_candidate else 1.0)
        h1_candidates.append(h1_candidate)

        wfe = walk_forward_efficiency(sub_is)
        c6_pass = bool(not np.isnan(wfe) and wfe >= 0.50)

        n_oos = len(oos_vals)
        expr_oos = float(np.mean(oos_vals)) if n_oos else float("nan")
        if n_oos >= 20 and not np.isnan(expr_is) and expr_is != 0 and not np.isnan(expr_oos):
            ratio = float(expr_oos / expr_is)
            c8_pass = bool(expr_oos >= 0 and ratio >= 0.40)
        else:
            ratio = float("nan")
            c8_pass = False

        era_min_expr, era_n_eligible, era_n_fail = era_stability(sub_is)
        c9_pass = bool(era_n_eligible >= 1 and era_n_fail == 0)

        summary_rows.append(
            {
                "instrument": instrument,
                "session": session,
                "level_name": level_name,
                "direction": direction,
                "n_is": n_is,
                "expr_is": expr_is,
                "win_rate_is": wr_is,
                "t_is": t_is,
                "p_one_is": p_one_is,
                "wfe": wfe,
                "c6_pass": c6_pass,
                "n_oos": n_oos,
                "expr_oos": expr_oos,
                "ratio_oos_is": ratio,
                "c8_pass": c8_pass,
                "era_min_expr": era_min_expr,
                "era_n_eligible": era_n_eligible,
                "era_n_fail": era_n_fail,
                "c9_pass": c9_pass,
            }
        )

    bh_survivors = bh_fdr(pvals, q=0.05)
    for idx, row in enumerate(summary_rows):
        row["bh_survivor"] = idx in bh_survivors

    p_arr = np.array(pvals, dtype=float)
    order = np.argsort(p_arr)
    ranks = np.empty(len(p_arr), dtype=int)
    ranks[order] = np.arange(1, len(p_arr) + 1)
    q_vals = p_arr * len(p_arr) / ranks
    q_sorted = q_vals[order].copy()
    for i in range(len(q_sorted) - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q_final = np.empty_like(q_sorted)
    q_final[order] = q_sorted

    cell_results: list[CellResult] = []
    for idx, row in enumerate(summary_rows):
        h1_pass = bool(
            h1_candidates[idx]
            and row["bh_survivor"]
            and row["expr_is"] > 0
        )
        if h1_pass and row["c6_pass"] and row["c8_pass"] and row["c9_pass"]:
            verdict = "CANDIDATE_READY"
        elif h1_pass:
            verdict = "RESEARCH_SURVIVOR"
        else:
            verdict = "DEAD"
        cell_results.append(
            CellResult(
                instrument=row["instrument"],
                session=row["session"],
                level_name=row["level_name"],
                direction=row["direction"],
                n_is=row["n_is"],
                expr_is=row["expr_is"],
                win_rate_is=row["win_rate_is"],
                t_is=row["t_is"],
                p_one_is=row["p_one_is"],
                q_bh=float(q_final[idx]) if h1_candidates[idx] else float("nan"),
                h1_pass=h1_pass,
                wfe=row["wfe"],
                c6_pass=row["c6_pass"],
                n_oos=row["n_oos"],
                expr_oos=row["expr_oos"],
                ratio_oos_is=row["ratio_oos_is"],
                c8_pass=row["c8_pass"],
                era_min_expr=row["era_min_expr"],
                era_n_eligible=row["era_n_eligible"],
                era_n_fail=row["era_n_fail"],
                c9_pass=row["c9_pass"],
                verdict=verdict,
            )
        )

    cells_df = pd.DataFrame([c.__dict__ for c in cell_results])
    return trades_df, cells_df


def render_results(trades_df: pd.DataFrame, cells_df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Sweep Reclaim Standalone V2")
    lines.append("")
    lines.append("Pre-reg: `docs/audit/hypotheses/2026-04-21-sweep-reclaim-standalone-v2.yaml`.")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Instruments: {', '.join(LOCKED_INSTRUMENTS)}")
    lines.append(f"- Sessions: {', '.join(LOCKED_SESSIONS)}")
    lines.append("- Levels: prev_day_high / prev_day_low only")
    lines.append("- Entry: reclaim close")
    lines.append("- Stop: one tick beyond the maximum/minimum sweep excursion known by the reclaim close")
    lines.append(f"- Target: fixed {LOCKED_RR:.1f}R")
    lines.append(f"- Entry window: first {EVENT_WINDOW_MINUTES} minutes of the session")
    lines.append("- Outcome path: stop/target on bars after entry, otherwise end-of-day close")
    lines.append("- Selection uses pre-2026 only; 2026 is diagnostic OOS only")
    lines.append("")

    if trades_df.empty or cells_df.empty:
        lines.append("No trades collected under the locked scope.")
        return "\n".join(lines)

    total_trades = len(trades_df)
    total_is = int((trades_df["sample"] == "IS").sum())
    total_oos = int((trades_df["sample"] == "OOS").sum())
    survivors = cells_df[cells_df["verdict"] == "RESEARCH_SURVIVOR"]
    candidates = cells_df[cells_df["verdict"] == "CANDIDATE_READY"]
    dead = cells_df[cells_df["verdict"] == "DEAD"]

    lines.append("## Family Verdict")
    lines.append("")
    lines.append(f"- Locked family K: {len(cells_df)}")
    lines.append(f"- Trades collected: {total_trades} total ({total_is} IS, {total_oos} OOS)")
    lines.append(f"- CANDIDATE_READY: {len(candidates)}")
    lines.append(f"- RESEARCH_SURVIVOR: {len(survivors)}")
    lines.append(f"- DEAD: {len(dead)}")
    lines.append("")

    lines.append("## Per-Cell Table")
    lines.append("")
    lines.append("| Instrument | Session | Level | Dir | N_IS | ExpR_IS | WR_IS | t | p(1t) | q(BH) | WFE | N_OOS | ExpR_OOS | OOS/IS | Worst Era | Verdict |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in cells_df.sort_values(["verdict", "expr_is"], ascending=[True, False]).itertuples(index=False):
        def fmt(val: float, spec: str) -> str:
            return f"{val:{spec}}" if not np.isnan(val) else "-"
        lines.append(
            f"| {row.instrument} | {row.session} | {row.level_name} | {row.direction} | "
            f"{row.n_is} | {fmt(row.expr_is, '+.4f')} | {fmt(row.win_rate_is, '.1%')} | "
            f"{fmt(row.t_is, '+.3f')} | {fmt(row.p_one_is, '.4f')} | {fmt(row.q_bh, '.4f')} | "
            f"{fmt(row.wfe, '.3f')} | {row.n_oos} | {fmt(row.expr_oos, '+.4f')} | "
            f"{fmt(row.ratio_oos_is, '+.2f')} | {fmt(row.era_min_expr, '+.4f')} | **{row.verdict}** |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if len(candidates) > 0:
        lines.append("- At least one cell met H1 plus C6/C8/C9 and is promotion-eligible pending deployment-only criteria.")
    elif len(survivors) > 0:
        lines.append("- At least one cell passed H1 but failed a downstream gate. These are research survivors only, not deployable.")
    else:
        lines.append("- No cell survived H1 at the locked K=8 scope. This standalone family is dead under the current geometry.")

    lines.append("- C8 is only counted when N_OOS >= 20. Thin OOS cannot promote a cell.")
    lines.append("- The stop is explicitly the known sweep extreme by reclaim close, plus/minus one tick, to avoid discretionary stop placement.")
    lines.append("")
    lines.append("## Not Done")
    lines.append("")
    lines.append("- No write to validated_setups, edge_families, live_config, or account profiles.")
    lines.append("- No deployment recommendation beyond the verdict classes above.")
    lines.append("- No widening to overnight levels, first-retest entries, or next-liquidity targets.")
    return "\n".join(lines)


def main() -> int:
    trades = collect_trades()
    trades_df, cells_df = summarize_trades(trades)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(OUTPUT_TRADES, index=False)
    cells_df.to_csv(OUTPUT_CELLS, index=False)
    OUTPUT_MD.write_text(render_results(trades_df, cells_df), encoding="utf-8")

    if cells_df.empty:
        print("No trades collected.")
        return 0

    print(cells_df.sort_values(["verdict", "expr_is"], ascending=[True, False]).to_string(index=False))
    print(f"\nSaved: {OUTPUT_TRADES}")
    print(f"Saved: {OUTPUT_CELLS}")
    print(f"Saved: {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
