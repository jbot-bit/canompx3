#!/usr/bin/env python3
"""A0 latency-buffer falsification.

Tests operational latency robustness WITHOUT changing historical fills:
- Base no-lookahead: leader_ts <= entry_ts
- Buffer-1m:        leader_ts <= entry_ts - 1 minute
- Buffer-2m:        leader_ts <= entry_ts - 2 minutes
- Buffer-5m:        leader_ts <= entry_ts - 5 minutes

Preset fixed to promoted A0 preset: base_plus_both
(follower fast<=15 and follower volume impulse >= train q60-style proxy using full sample q60 for quick check)
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "gold.db"


def stats(s: pd.Series, years: pd.Series) -> dict:
    if s.empty:
        return {"n": 0, "signals_per_year": np.nan, "avg_r": np.nan, "wr": np.nan}
    y = max(1, int(years.nunique()))
    return {
        "n": int(len(s)),
        "signals_per_year": float(len(s) / y),
        "avg_r": float(s.mean()),
        "wr": float((s > 0).mean()),
    }


def main() -> int:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    q = """
    SELECT o.trading_day,o.pnl_r,o.entry_ts,
           df_f.orb_US_EQUITY_OPEN_break_dir AS f_dir,
           df_f.orb_US_EQUITY_OPEN_break_delay_min AS f_delay,
           df_f.orb_US_EQUITY_OPEN_volume AS f_vol,
           df_f.orb_US_EQUITY_OPEN_break_bar_volume AS f_bvol,
           df_l.orb_US_EQUITY_OPEN_break_dir AS l_dir,
           df_l.orb_US_EQUITY_OPEN_break_ts  AS l_ts
    FROM orb_outcomes o
    JOIN daily_features df_f ON df_f.symbol=o.symbol AND df_f.trading_day=o.trading_day AND df_f.orb_minutes=o.orb_minutes
    JOIN daily_features df_l ON df_l.symbol='M6E' AND df_l.trading_day=o.trading_day AND df_l.orb_minutes=o.orb_minutes
    WHERE o.orb_minutes=5
      AND o.symbol='MES' AND o.orb_label='US_EQUITY_OPEN'
      AND o.entry_model='E0' AND o.confirm_bars=1 AND o.rr_target=3.0
      AND o.pnl_r IS NOT NULL AND o.entry_ts IS NOT NULL
    """
    df = con.execute(q).fetchdf()
    con.close()

    if df.empty:
        print("No rows.")
        return 0

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["l_ts"] = pd.to_datetime(df["l_ts"], utc=True)

    df["f_vol_imp"] = np.where((df["f_vol"].notna()) & (df["f_vol"] > 0), df["f_bvol"] / (df["f_vol"] / 5.0), np.nan)
    vol_q60 = df["f_vol_imp"].quantile(0.60)

    # Frozen A0 preset filters on follower
    follower_quality = (
        df["f_delay"].notna() & (df["f_delay"] <= 15)
        & df["f_vol_imp"].notna() & (df["f_vol_imp"] >= vol_q60)
    )

    base_dir = (
        df["f_dir"].isin(["long", "short"])
        & df["l_dir"].isin(["long", "short"])
        & (df["f_dir"] == df["l_dir"])
        & df["l_ts"].notna()
    )

    masks = {
        "no_buffer": base_dir & (df["l_ts"] <= df["entry_ts"]) & follower_quality,
        "buffer_1m": base_dir & (df["l_ts"] <= (df["entry_ts"] - pd.Timedelta(minutes=1))) & follower_quality,
        "buffer_2m": base_dir & (df["l_ts"] <= (df["entry_ts"] - pd.Timedelta(minutes=2))) & follower_quality,
        "buffer_5m": base_dir & (df["l_ts"] <= (df["entry_ts"] - pd.Timedelta(minutes=5))) & follower_quality,
    }

    rows = []
    baseline = df[masks["no_buffer"]]["pnl_r"]
    baseline_avg = float(baseline.mean()) if len(baseline) else np.nan

    for name, m in masks.items():
        on = df[m]["pnl_r"]
        st = stats(on, df.loc[m, "year"])
        rows.append(
            {
                "variant": name,
                "n": st["n"],
                "signals_per_year": st["signals_per_year"],
                "avg_r": st["avg_r"],
                "wr": st["wr"],
                "delta_vs_no_buffer": (st["avg_r"] - baseline_avg) if pd.notna(st["avg_r"]) and pd.notna(baseline_avg) else np.nan,
            }
        )

    out = pd.DataFrame(rows)

    out_dir = ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_csv = out_dir / "a0_latency_buffer_falsification.csv"
    p_md = out_dir / "a0_latency_buffer_falsification.md"

    out.to_csv(p_csv, index=False)

    lines = [
        "# A0 Latency-Buffer Falsification",
        "",
        "Interpretation: if small decision latency kills edge, live execution risk is high.",
        "",
    ]
    for r in out.itertuples(index=False):
        lines.append(
            f"- {r.variant}: N={r.n}, sig/yr={r.signals_per_year:.1f}, avgR={r.avg_r:+.4f}, Δvs_base={r.delta_vs_no_buffer:+.4f}"
        )

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_csv}")
    print(f"Saved: {p_md}")
    print(out.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
