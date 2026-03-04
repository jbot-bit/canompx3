#!/usr/bin/env python3
"""Surprise quick test: day-of-week overlay on A0.

A0 baseline condition:
- M6E_US_EQUITY_OPEN -> MES_US_EQUITY_OPEN
- E0 / CB1 / RR3.0
- same-dir + no-lookahead

Tests simple calendar overlays (fixed, no parameter search):
- exclude Wednesday
- exclude Wednesday+Thursday
- Monday+Tuesday only
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "gold.db"


def main() -> int:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    q = """
    SELECT o.trading_day,o.pnl_r,o.entry_ts,
           df_f.orb_US_EQUITY_OPEN_break_dir AS f_dir,
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

    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["l_ts"] = pd.to_datetime(df["l_ts"], utc=True)
    df["trading_day"] = pd.to_datetime(df["trading_day"])

    base = (
        df["f_dir"].isin(["long", "short"]) &
        df["l_dir"].isin(["long", "short"]) &
        (df["f_dir"] == df["l_dir"]) &
        df["l_ts"].notna() &
        (df["l_ts"] <= df["entry_ts"])
    )
    d = df[base].copy()
    d["dow"] = d["trading_day"].dt.dayofweek

    tests = {
        "base": pd.Series(True, index=d.index),
        "exclude_wed": d["dow"] != 2,
        "exclude_wed_thu": ~d["dow"].isin([2, 3]),
        "mon_tue_only": d["dow"].isin([0, 1]),
    }

    rows = []
    for name, m in tests.items():
        on = d.loc[m, "pnl_r"]
        off = d.loc[~m, "pnl_r"] if name != "base" else d.loc[~m, "pnl_r"]
        avg_on = float(on.mean()) if len(on) else float("nan")
        avg_off = float(off.mean()) if len(off) else float("nan")
        rows.append(
            {
                "variant": name,
                "n_on": int(len(on)),
                "signals_per_year": float(len(on) / max(1, d["trading_day"].dt.year.nunique())),
                "avg_on": avg_on,
                "avg_off": avg_off,
                "uplift_on_off": float(avg_on - avg_off) if len(off) else float("nan"),
            }
        )

    out = pd.DataFrame(rows)

    out_dir = ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_csv = out_dir / "surprise_calendar_dow_overlay.csv"
    p_md = out_dir / "surprise_calendar_dow_overlay.md"

    out.to_csv(p_csv, index=False)

    lines = ["# Surprise Calendar Overlay (A0)", ""]
    for r in out.itertuples(index=False):
        lines.append(
            f"- {r.variant}: N={r.n_on}, sig/yr={r.signals_per_year:.1f}, avg_on={r.avg_on:+.4f}, uplift={r.uplift_on_off:+.4f}"
        )
    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_csv}")
    print(f"Saved: {p_md}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
