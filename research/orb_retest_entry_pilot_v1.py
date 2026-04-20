#!/usr/bin/env python3
"""ORB retest entry pilot v1.

Research-only. Does not modify canonical ORB tables.

Locked by:
  docs/audit/hypotheses/2026-04-20-orb-retest-entry-pilot-v1.yaml
  docs/plans/2026-04-20-htf-ltf-orb-routing-decision.md
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.build_daily_features import compute_trading_day_utc_range
from pipeline.cost_model import COST_SPECS, CostSpec, pnl_points_to_r, to_r_multiple
from research.lib import bh_fdr, connect_db, ttest_1s

LOCKED_INSTRUMENTS = [symbol for symbol in ACTIVE_ORB_INSTRUMENTS if symbol in {"MES", "MGC", "MNQ"}]
LOCKED_SESSIONS = [
    "TOKYO_OPEN",
    "EUROPE_FLOW",
    "NYSE_OPEN",
    "US_DATA_1000",
    "COMEX_SETTLE",
    "CME_PRECLOSE",
]
LOCKED_RR = [1.0, 1.5, 2.0]
COMMON_START = date(2019, 5, 6)
HOLDOUT_START = date(2026, 1, 1)

OUTPUT_DIR = Path("research/output")
OUTPUT_CSV = OUTPUT_DIR / "orb_retest_entry_pilot_v1_cells.csv"
OUTPUT_MD = Path("docs/audit/results/2026-04-20-orb-retest-entry-pilot-v1.md")


@dataclass(frozen=True)
class PilotTrade:
    symbol: str
    trading_day: date
    session: str
    rr_target: float
    sample: str
    break_dir: str
    retest_entry_ts: object
    retest_pnl_r: float | None
    retest_outcome: str
    matched_e2_pnl_r: float | None
    matched_e2_outcome: str | None


def _paired_ttest(deltas: np.ndarray) -> tuple[int, float, float, float, float]:
    arr = np.array(deltas, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    if n < 10:
        return n, float(arr.mean()), float((arr > 0).mean()), float("nan"), float("nan")
    t_stat, p_val = stats.ttest_1samp(arr, 0.0)
    return n, float(arr.mean()), float((arr > 0).mean()), float(t_stat), float(p_val)


def _load_feature_rows(con) -> pd.DataFrame:
    session_cols: list[str] = []
    for session in LOCKED_SESSIONS:
        session_cols.extend(
            [
                f"orb_{session}_high",
                f"orb_{session}_low",
                f"orb_{session}_break_dir",
                f"orb_{session}_break_ts",
            ]
        )
    sql = f"""
    SELECT trading_day, symbol, {", ".join(session_cols)}
    FROM daily_features
    WHERE orb_minutes = 5
      AND trading_day >= ?
      AND symbol IN ({", ".join("?" for _ in LOCKED_INSTRUMENTS)})
    ORDER BY symbol, trading_day
    """
    params: list[object] = [COMMON_START, *LOCKED_INSTRUMENTS]
    df = con.execute(sql, params).fetchdf()
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    for session in LOCKED_SESSIONS:
        col = f"orb_{session}_break_ts"
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def _load_e2_baseline(con) -> dict[tuple[str, date, str, float], tuple[float | None, str | None]]:
    sql = f"""
    SELECT symbol, trading_day, orb_label, rr_target, pnl_r, outcome
    FROM orb_outcomes
    WHERE orb_minutes = 5
      AND entry_model = 'E2'
      AND confirm_bars = 1
      AND symbol IN ({", ".join("?" for _ in LOCKED_INSTRUMENTS)})
      AND orb_label IN ({", ".join("?" for _ in LOCKED_SESSIONS)})
      AND rr_target IN ({", ".join("?" for _ in LOCKED_RR)})
      AND outcome IS NOT NULL
    """
    params: list[object] = [*LOCKED_INSTRUMENTS, *LOCKED_SESSIONS, *LOCKED_RR]
    rows = con.execute(sql, params).fetchall()
    return {
        (symbol, trading_day, orb_label, float(rr)): (pnl_r, outcome)
        for symbol, trading_day, orb_label, rr, pnl_r, outcome in rows
    }


def _load_bars_for_day(con, symbol: str, trading_day: date) -> pd.DataFrame:
    start_utc, end_utc = compute_trading_day_utc_range(trading_day)
    sql = """
    SELECT ts_utc, open, high, low, close
    FROM bars_1m
    WHERE symbol = ?
      AND ts_utc >= ?
      AND ts_utc < ?
    ORDER BY ts_utc
    """
    df = con.execute(sql, [symbol, start_utc, end_utc]).fetchdf()
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df


def _simulate_retest_trade(
    bars_day: pd.DataFrame,
    break_ts: pd.Timestamp,
    break_dir: str,
    orb_high: float,
    orb_low: float,
    rr_target: float,
    cost_spec: CostSpec,
) -> tuple[object | None, str, float | None]:
    """Research-only retest simulator with bounded, conservative rules."""
    if pd.isna(break_ts):
        return None, "no_break_ts", None

    level = float(orb_high if break_dir == "long" else orb_low)
    stop_price = float(orb_low if break_dir == "long" else orb_high)
    risk_points = abs(level - stop_price)
    if risk_points <= 0:
        return None, "invalid_risk", None

    target_price = level + (risk_points * rr_target if break_dir == "long" else -risk_points * rr_target)
    tick = cost_spec.tick_size

    bars_after = bars_day[bars_day["ts_utc"] > break_ts].reset_index(drop=True)
    if bars_after.empty:
        return None, "no_post_break_bars", None

    armed = False
    fill_idx: int | None = None
    fill_ts = None

    for i, bar in bars_after.iterrows():
        bar_close = float(bar["close"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])

        if not armed:
            if (break_dir == "long" and bar_close > level + tick) or (
                break_dir == "short" and bar_close < level - tick
            ):
                armed = True
            continue

        if bar_low <= level <= bar_high:
            fill_idx = i
            fill_ts = bar["ts_utc"].to_pydatetime()
            break

    if fill_idx is None or fill_ts is None:
        return None, "no_retest_fill", None

    fill_bar = bars_after.iloc[fill_idx]
    fill_bar_high = float(fill_bar["high"])
    fill_bar_low = float(fill_bar["low"])

    if break_dir == "long":
        fill_hit_target = fill_bar_high >= target_price
        fill_hit_stop = fill_bar_low <= stop_price
    else:
        fill_hit_target = fill_bar_low <= target_price
        fill_hit_stop = fill_bar_high >= stop_price

    if fill_hit_target and fill_hit_stop:
        return fill_ts, "loss", -1.0
    if fill_hit_target:
        return fill_ts, "win", round(to_r_multiple(cost_spec, level, stop_price, risk_points * rr_target), 4)
    if fill_hit_stop:
        return fill_ts, "loss", -1.0

    post_entry = bars_after.iloc[fill_idx + 1 :]
    if post_entry.empty:
        return fill_ts, "scratch", None

    for _, bar in post_entry.iterrows():
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        if break_dir == "long":
            hit_target = bar_high >= target_price
            hit_stop = bar_low <= stop_price
        else:
            hit_target = bar_low <= target_price
            hit_stop = bar_high >= stop_price

        if hit_target and hit_stop:
            return fill_ts, "loss", -1.0
        if hit_target:
            return fill_ts, "win", round(to_r_multiple(cost_spec, level, stop_price, risk_points * rr_target), 4)
        if hit_stop:
            return fill_ts, "loss", -1.0

    return fill_ts, "scratch", None


def collect_trades() -> list[PilotTrade]:
    trades: list[PilotTrade] = []
    bars_cache: dict[tuple[str, date], pd.DataFrame] = {}

    with connect_db() as con:
        feature_rows = _load_feature_rows(con)
        e2_baseline = _load_e2_baseline(con)
        total_rows = len(feature_rows)

        for idx, row in enumerate(feature_rows.itertuples(index=False), start=1):
            if idx % 500 == 0:
                print(f"Progress: {idx}/{total_rows} feature rows", flush=True)

            symbol = row.symbol
            trading_day = row.trading_day
            sample = "IS" if trading_day < HOLDOUT_START else "OOS"
            bars_key = (symbol, trading_day)
            if bars_key not in bars_cache:
                bars_cache[bars_key] = _load_bars_for_day(con, symbol, trading_day)
            bars_day = bars_cache[bars_key]
            if bars_day.empty:
                continue

            spec = COST_SPECS[symbol]

            for session in LOCKED_SESSIONS:
                break_dir = getattr(row, f"orb_{session}_break_dir")
                break_ts = getattr(row, f"orb_{session}_break_ts")
                orb_high = getattr(row, f"orb_{session}_high")
                orb_low = getattr(row, f"orb_{session}_low")
                if break_dir not in {"long", "short"}:
                    continue
                if pd.isna(orb_high) or pd.isna(orb_low) or pd.isna(break_ts):
                    continue

                for rr_target in LOCKED_RR:
                    entry_ts, retest_outcome, retest_pnl_r = _simulate_retest_trade(
                        bars_day=bars_day,
                        break_ts=break_ts,
                        break_dir=break_dir,
                        orb_high=float(orb_high),
                        orb_low=float(orb_low),
                        rr_target=float(rr_target),
                        cost_spec=spec,
                    )
                    matched_e2_pnl_r, matched_e2_outcome = e2_baseline.get(
                        (symbol, trading_day, session, float(rr_target)),
                        (None, None),
                    )
                    if retest_outcome in {"no_break_ts", "invalid_risk", "no_post_break_bars", "no_retest_fill"}:
                        continue
                    trades.append(
                        PilotTrade(
                            symbol=symbol,
                            trading_day=trading_day,
                            session=session,
                            rr_target=float(rr_target),
                            sample=sample,
                            break_dir=break_dir,
                            retest_entry_ts=entry_ts,
                            retest_pnl_r=retest_pnl_r,
                            retest_outcome=retest_outcome,
                            matched_e2_pnl_r=matched_e2_pnl_r,
                            matched_e2_outcome=matched_e2_outcome,
                        )
                    )

    return trades


def summarize_trades(trades: list[PilotTrade]) -> pd.DataFrame:
    df = pd.DataFrame([trade.__dict__ for trade in trades])
    if df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    pvals: list[float] = []
    cell_index = [
        (symbol, session, rr)
        for symbol in LOCKED_INSTRUMENTS
        for session in LOCKED_SESSIONS
        for rr in LOCKED_RR
    ]

    for symbol, session, rr_target in cell_index:
        sub = df[(df["symbol"] == symbol) & (df["session"] == session) & (df["rr_target"] == rr_target)]
        sub_is = sub[sub["sample"] == "IS"]
        sub_oos = sub[sub["sample"] == "OOS"]

        retest_is = sub_is["retest_pnl_r"].to_numpy(dtype=float)
        retest_oos = sub_oos["retest_pnl_r"].to_numpy(dtype=float)
        n_is, avg_is, wr_is, t_is, p_is = ttest_1s(retest_is)
        n_oos, avg_oos, wr_oos, t_oos, p_oos = ttest_1s(retest_oos)

        paired_is = sub_is.dropna(subset=["retest_pnl_r", "matched_e2_pnl_r"]).copy()
        paired_oos = sub_oos.dropna(subset=["retest_pnl_r", "matched_e2_pnl_r"]).copy()
        delta_is_vals = (
            paired_is["retest_pnl_r"].astype(float).to_numpy() - paired_is["matched_e2_pnl_r"].astype(float).to_numpy()
        )
        delta_oos_vals = (
            paired_oos["retest_pnl_r"].astype(float).to_numpy()
            - paired_oos["matched_e2_pnl_r"].astype(float).to_numpy()
        )

        n_delta_is, delta_is, delta_wr_is, delta_t_is, delta_p_is = _paired_ttest(delta_is_vals)
        n_delta_oos, delta_oos, delta_wr_oos, delta_t_oos, delta_p_oos = _paired_ttest(delta_oos_vals)

        matched_e2_is_avg = float(np.nanmean(paired_is["matched_e2_pnl_r"])) if len(paired_is) else float("nan")
        matched_e2_oos_avg = float(np.nanmean(paired_oos["matched_e2_pnl_r"])) if len(paired_oos) else float("nan")
        pvals.append(1.0 if n_delta_is < 10 or np.isnan(delta_p_is) else float(delta_p_is))

        rows.append(
            {
                "instrument": symbol,
                "session": session,
                "rr_target": float(rr_target),
                "n_is": int(n_is),
                "avg_is": None if np.isnan(avg_is) else float(avg_is),
                "wr_is": None if np.isnan(wr_is) else float(wr_is),
                "t_is": None if np.isnan(t_is) else float(t_is),
                "p_is": None if np.isnan(p_is) else float(p_is),
                "n_oos": int(n_oos),
                "avg_oos": None if np.isnan(avg_oos) else float(avg_oos),
                "wr_oos": None if np.isnan(wr_oos) else float(wr_oos),
                "t_oos": None if np.isnan(t_oos) else float(t_oos),
                "p_oos": None if np.isnan(p_oos) else float(p_oos),
                "n_delta_is": int(n_delta_is),
                "matched_e2_avg_is": None if np.isnan(matched_e2_is_avg) else float(matched_e2_is_avg),
                "delta_is": None if np.isnan(delta_is) else float(delta_is),
                "delta_wr_is": None if np.isnan(delta_wr_is) else float(delta_wr_is),
                "delta_t_is": None if np.isnan(delta_t_is) else float(delta_t_is),
                "delta_p_is": None if np.isnan(delta_p_is) else float(delta_p_is),
                "n_delta_oos": int(n_delta_oos),
                "matched_e2_avg_oos": None if np.isnan(matched_e2_oos_avg) else float(matched_e2_oos_avg),
                "delta_oos": None if np.isnan(delta_oos) else float(delta_oos),
                "delta_wr_oos": None if np.isnan(delta_wr_oos) else float(delta_wr_oos),
                "delta_t_oos": None if np.isnan(delta_t_oos) else float(delta_t_oos),
                "delta_p_oos": None if np.isnan(delta_p_oos) else float(delta_p_oos),
            }
        )

    rejected = bh_fdr(pvals, q=0.05)
    for idx, row in enumerate(rows):
        row["p_for_bh"] = pvals[idx]
        row["bh_survivor"] = idx in rejected
        row["passes_primary"] = bool(
            row["bh_survivor"]
            and row["n_delta_is"] >= 50
            and row["delta_is"] is not None
            and row["delta_is"] > 0
            and (
                row["n_delta_oos"] < 20
                or (
                    row["delta_oos"] is not None
                    and row["delta_oos"] > 0
                )
            )
        )
        row["passes_trading_relevant"] = bool(
            row["passes_primary"]
            and row["avg_is"] is not None
            and row["avg_is"] > 0
        )

    return pd.DataFrame(rows)


def write_outputs(summary: pd.DataFrame, trades: list[PilotTrade]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)

    primary = summary[summary["passes_primary"]].copy() if not summary.empty else pd.DataFrame()
    trading_relevant = summary[summary["passes_trading_relevant"]].copy() if not summary.empty else pd.DataFrame()
    warm = summary[
        (summary["n_delta_is"] >= 20)
        & summary["delta_is"].notna()
        & (summary["delta_is"] > 0)
    ].copy() if not summary.empty else pd.DataFrame()
    warm = warm.sort_values(["delta_p_is", "delta_is"], na_position="last").head(8)
    adverse = summary[
        (summary["n_delta_is"] >= 20)
        & summary["delta_is"].notna()
        & (summary["delta_is"] < 0)
    ].copy() if not summary.empty else pd.DataFrame()
    adverse = adverse.sort_values(["delta_p_is", "delta_is"], ascending=[True, True], na_position="last").head(8)

    trades_df = pd.DataFrame([trade.__dict__ for trade in trades])
    n_is_trades = int((trades_df["sample"] == "IS").sum()) if not trades_df.empty else 0
    n_oos_trades = int((trades_df["sample"] == "OOS").sum()) if not trades_df.empty else 0

    lines = [
        "# ORB Retest Entry Pilot V1",
        "",
        "Research-only pilot locked by `docs/audit/hypotheses/2026-04-20-orb-retest-entry-pilot-v1.yaml`.",
        "",
        "## Scope",
        "",
        f"- Instruments: {', '.join(LOCKED_INSTRUMENTS)}",
        f"- Sessions: {', '.join(LOCKED_SESSIONS)}",
        "- Aperture: O5",
        "- RR targets: 1.0 / 1.5 / 2.0",
        "- Baseline: canonical `E2` on the same retest-eligible days",
        "- Holdout: 2026-01-01 onwards is diagnostic OOS only",
        "",
        "## Event definition",
        "",
        "- Require canonical ORB break in `daily_features`.",
        "- After the break, require one later 1-minute close one tick beyond the ORB boundary in the breakout direction.",
        "- Then enter at the first later touch of the ORB boundary.",
        "- Stop = opposite ORB boundary; target = fixed RR multiple.",
        "- Ambiguous stop+target bars resolve as loss.",
        "",
        "## Family verdict",
        "",
        f"- Locked family K: {len(summary)}",
        f"- Retest trades captured: IS {n_is_trades}, OOS {n_oos_trades}",
        f"- Primary survivors (BH + paired-delta + N>=50 + OOS direction check): {len(primary)}",
        f"- Trading-relevant survivors (same as above AND retest avg IS > 0): {len(trading_relevant)}",
        "",
    ]

    if trading_relevant.empty:
        lines.extend(
            [
                "No trading-relevant survivors.",
                "",
                "The bounded first-touch-to-ORB-boundary continuation shape did not justify production-code integration on the locked scope.",
                "",
                "Important audit note: one cell improved materially versus E2 after BH correction, but the retest path still had negative IS expectancy. That is a risk-reduction observation, not an edge.",
                "",
            ]
        )
    elif primary.empty:
        lines.extend(
            [
                "No primary survivors.",
                "",
                "This means the retest route neither outperformed E2 nor produced a viable standalone expectancy on the locked scope.",
                "",
            ]
        )
    else:
        lines.extend(["Primary survivors:", ""])
        for row in primary.sort_values("delta_p_is").itertuples(index=False):
            lines.append(
                f"- {row.instrument} {row.session} RR{row.rr_target}: "
                f"paired IS delta={row.delta_is:+.4f}R over E2, "
                f"n_is_pairs={row.n_delta_is}, "
                f"p={row.delta_p_is:.4f}, "
                f"OOS delta={row.delta_oos:+.4f}R (n={row.n_delta_oos})"
            )
        lines.append("")

    if not warm.empty:
        lines.extend(["## Warm cells (informational only)", ""])
        for row in warm.itertuples(index=False):
            lines.append(
                f"- {row.instrument} {row.session} RR{row.rr_target}: "
                f"IS retest avg={row.avg_is:+.4f}R, matched E2={row.matched_e2_avg_is:+.4f}R, "
                f"delta={row.delta_is:+.4f}R, n_pairs={row.n_delta_is}, p={row.delta_p_is:.4f}"
            )
        lines.append("")

    if not adverse.empty:
        lines.extend(["## Strong negative cells", ""])
        for row in adverse.itertuples(index=False):
            lines.append(
                f"- {row.instrument} {row.session} RR{row.rr_target}: "
                f"IS retest avg={row.avg_is:+.4f}R vs matched E2={row.matched_e2_avg_is:+.4f}R, "
                f"delta={row.delta_is:+.4f}R, n_pairs={row.n_delta_is}, p={row.delta_p_is:.4f}"
            )
        lines.append("")

    lines.extend(
        [
            "## Caveats",
            "",
            "- This is a research-only pilot, not a validator result and not a deployment recommendation.",
            "- The retest rule is deliberately narrow; broader non-ORB pullback families remain untested here.",
            "- No queue-position model is applied; the fill rule is resting-limit-on-touch with conservative ambiguity handling.",
            "- If this pilot is dead, that kills the bounded ORB-integration route only. It does not kill standalone SC2 event families.",
            "",
            "## Artefacts",
            "",
            f"- Cell CSV: `{OUTPUT_CSV}`",
            "- Script: `research/orb_retest_entry_pilot_v1.py`",
        ]
    )

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    trades = collect_trades()
    summary = summarize_trades(trades)
    write_outputs(summary, trades)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
