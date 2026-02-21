#!/usr/bin/env python3
"""Dalton acceptance as filter ON/OFF over existing ORB outcomes.

Read-only analysis. No pipeline/live logic changes.

Filter definition (close_A_B):
- Open outside prior-day VA
- A and B 30m bracket closes inside prior-day VA

Then compare ORB outcome quality with filter OFF vs ON,
using only trades whose entry_ts occurs after B-close gate_ts
(to avoid lookahead).

Outputs:
- research/output/dalton_filter_uplift_summary.csv
- research/output/dalton_filter_uplift_notes.md
"""

from __future__ import annotations

from pathlib import Path
import sys
import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.build_daily_features import compute_trading_day_utc_range, _orb_utc_window
from research.archive.analyze_value_area import compute_volume_profile

DB_PATH = "gold.db"
SYMBOLS = ["MGC", "MES", "MNQ"]
ANCHORS = ["0900", "1000", "1100"]


def _slice_day(bars: pd.DataFrame, ts_all, td):
    s, e = compute_trading_day_utc_range(td)
    i0 = int(np.searchsorted(ts_all, pd.Timestamp(s).asm8, side="left"))
    i1 = int(np.searchsorted(ts_all, pd.Timestamp(e).asm8, side="left"))
    return bars.iloc[i0:i1]


def _close_in_va(bar: pd.Series, val: float, vah: float) -> bool:
    c = float(bar["close"])
    return val <= c <= vah


def build_filter_flags(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows = []

    for sym in SYMBOLS:
        tdays = [r[0] for r in con.execute(
            "SELECT DISTINCT trading_day FROM daily_features WHERE symbol=? AND orb_minutes=5 ORDER BY trading_day",
            [sym],
        ).fetchall()]
        if len(tdays) < 3:
            continue

        gs, _ = compute_trading_day_utc_range(tdays[0])
        _, ge = compute_trading_day_utc_range(tdays[-1])
        bars = con.execute(
            """
            SELECT ts_utc, open, high, low, close, volume
            FROM bars_1m
            WHERE symbol=?
              AND ts_utc>=?::TIMESTAMPTZ
              AND ts_utc<?::TIMESTAMPTZ
            ORDER BY ts_utc
            """,
            [sym, gs.isoformat(), ge.isoformat()],
        ).fetchdf()
        if bars.empty:
            continue

        bars["ts_utc"] = pd.to_datetime(bars["ts_utc"], utc=True)
        ts_all = bars["ts_utc"].values

        for anchor in ANCHORS:
            for i in range(1, len(tdays)):
                prev_td = tdays[i - 1]
                td = tdays[i]
                prev_bars = _slice_day(bars, ts_all, prev_td)
                day_bars = _slice_day(bars, ts_all, td)
                if prev_bars.empty or day_bars.empty:
                    continue

                prof = compute_volume_profile(prev_bars, bin_size=0.5)
                if prof is None:
                    continue

                val = float(prof["val"])
                vah = float(prof["vah"])

                start, _ = _orb_utc_window(td, anchor, 1)
                first = day_bars[(day_bars["ts_utc"] >= start) & (day_bars["ts_utc"] < start + pd.Timedelta(minutes=1))]
                if first.empty:
                    continue

                open_px = float(first.iloc[0]["open"])
                outside = int(not (val <= open_px <= vah))

                a_end = start + pd.Timedelta(minutes=30)
                b_end = start + pd.Timedelta(minutes=60)
                bars_a = day_bars[(day_bars["ts_utc"] >= start) & (day_bars["ts_utc"] < a_end)]
                bars_b = day_bars[(day_bars["ts_utc"] >= a_end) & (day_bars["ts_utc"] < b_end)]

                accepted = 0
                if outside and (not bars_a.empty) and (not bars_b.empty):
                    if _close_in_va(bars_a.iloc[-1], val, vah) and _close_in_va(bars_b.iloc[-1], val, vah):
                        accepted = 1

                rows.append(
                    {
                        "symbol": sym,
                        "trading_day": pd.Timestamp(td),
                        "anchor": anchor,
                        "dalton_outside": outside,
                        "dalton_accept": accepted,
                        "gate_ts": pd.Timestamp(b_end),
                    }
                )

    return pd.DataFrame(rows)


def max_dd(pnls: pd.Series) -> float:
    if pnls.empty:
        return 0.0
    c = pnls.cumsum()
    peak = c.cummax()
    dd = peak - c
    return float(dd.max())


def summarize(orb: pd.DataFrame) -> pd.DataFrame:
    out = []
    keys = ["symbol", "anchor", "entry_model", "confirm_bars", "rr_target"]

    for k, g in orb.groupby(keys):
        g = g.sort_values(["trading_day", "entry_ts"])

        base = g
        on = g[g["dalton_accept"] == 1]

        n_base = len(base)
        n_on = len(on)
        if n_base < 30 or n_on < 20:
            continue

        avg_base = float(base["pnl_r"].mean())
        avg_on = float(on["pnl_r"].mean())

        out.append(
            {
                "symbol": k[0],
                "anchor": k[1],
                "entry_model": k[2],
                "confirm_bars": int(k[3]),
                "rr_target": float(k[4]),
                "n_base": n_base,
                "n_on": n_on,
                "wr_base": float((base["pnl_r"] > 0).mean()),
                "wr_on": float((on["pnl_r"] > 0).mean()),
                "avg_r_base": avg_base,
                "avg_r_on": avg_on,
                "uplift_avg_r": avg_on - avg_base,
                "total_r_base": float(base["pnl_r"].sum()),
                "total_r_on": float(on["pnl_r"].sum()),
                "max_dd_base": max_dd(base["pnl_r"]),
                "max_dd_on": max_dd(on["pnl_r"]),
                "dd_delta": max_dd(on["pnl_r"]) - max_dd(base["pnl_r"]),
            }
        )

    if not out:
        return pd.DataFrame()

    return pd.DataFrame(out).sort_values(["uplift_avg_r"], ascending=False)


def main() -> int:
    con = duckdb.connect(DB_PATH, read_only=True)

    flags = build_filter_flags(con)
    if flags.empty:
        print("No Dalton flags built.")
        return 0

    # Pull orb outcomes and join day-level flags
    outcomes = con.execute(
        """
        SELECT symbol, trading_day, orb_label AS anchor, entry_model, confirm_bars,
               rr_target, pnl_r, entry_ts, orb_minutes
        FROM orb_outcomes
        WHERE orb_minutes = 5
          AND pnl_r IS NOT NULL
          AND entry_ts IS NOT NULL
          AND symbol IN ('MGC','MES','MNQ')
          AND orb_label IN ('0900','1000','1100')
        """
    ).fetchdf()
    con.close()

    outcomes["entry_ts"] = pd.to_datetime(outcomes["entry_ts"], utc=True)
    merged = outcomes.merge(flags, on=["symbol", "trading_day", "anchor"], how="inner")

    # no-lookahead guard: only trades after gate_ts can be filtered live
    merged = merged[merged["entry_ts"] >= merged["gate_ts"]].copy()

    summary = summarize(merged)

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "dalton_filter_uplift_summary.csv"
    md_path = out_dir / "dalton_filter_uplift_notes.md"

    if summary.empty:
        md_path.write_text("# Dalton Filter Uplift\n\nNo cells met sample thresholds (n_base>=30, n_on>=20).", encoding="utf-8")
        print("No summary rows after thresholds.")
        return 0

    summary.to_csv(csv_path, index=False)

    lines = ["# Dalton Filter ON/OFF Uplift", "", "No-lookahead: only trades with entry_ts >= A/B gate_ts included.", ""]

    top = summary.head(12)
    lines.append("## Top uplift cells")
    for r in top.itertuples(index=False):
        lines.append(
            f"- {r.symbol} {r.anchor} {r.entry_model} CB{r.confirm_bars} RR{r.rr_target}: "
            f"Nbase={r.n_base}, Non={r.n_on}, avgR {r.avg_r_base:+.4f}->{r.avg_r_on:+.4f} "
            f"(Δ={r.uplift_avg_r:+.4f}), DD {r.max_dd_base:.2f}->{r.max_dd_on:.2f}"
        )

    good = summary[(summary["uplift_avg_r"] > 0) & (summary["dd_delta"] <= 0)]
    lines.append("")
    lines.append(f"## Candidate keepers (uplift>0 and DD not worse): {len(good)}")
    for r in good.head(10).itertuples(index=False):
        lines.append(
            f"- {r.symbol} {r.anchor} {r.entry_model} CB{r.confirm_bars} RR{r.rr_target}: "
            f"ΔavgR={r.uplift_avg_r:+.4f}, ΔDD={r.dd_delta:+.2f}, N_on={r.n_on}"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print("\nTop rows:")
    print(summary.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
