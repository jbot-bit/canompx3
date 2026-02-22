#!/usr/bin/env python3
"""Wide composite non-lead-lag filters (breadth-first follow-up).

Focus: filters that may keep frequency >=150/year while improving avgR.
"""

from __future__ import annotations

import re
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = "gold.db"
SYMBOLS = ["M2K", "MES", "M6E", "MGC"]

MIN_BASE = 500
MIN_ON = 150
MIN_SIGNALS_PER_YEAR = 150
MIN_UPLIFT = 0.10
MIN_AVG_ON = 0.0
MIN_YEARS_TOTAL = 3
MIN_YEARS_POS_RATIO = 0.6


def safe_label(label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", label):
        raise ValueError(f"Unsafe label: {label}")
    return label


def yearly_uplift(df: pd.DataFrame, mask: pd.Series) -> tuple[int, int]:
    yp = 0
    yt = 0
    for _, g in df.groupby("year"):
        m = mask.loc[g.index]
        on = g.loc[m, "pnl_r"]
        off = g.loc[~m, "pnl_r"]
        if len(on) < 20 or len(off) < 20:
            continue
        yt += 1
        if on.mean() - off.mean() > 0:
            yp += 1
    return yp, yt


def oos_uplift(df: pd.DataFrame, mask: pd.Series) -> tuple[float, float]:
    counts = df.groupby("year").size().to_dict()
    if 2025 in counts and counts[2025] >= 200:
        test_year = 2025
    else:
        elig = [int(y) for y, n in counts.items() if n >= 200]
        if not elig:
            return np.nan, np.nan
        test_year = max(elig)

    tr = df[df["year"] < test_year]
    te = df[df["year"] == test_year]
    if tr.empty or te.empty:
        return np.nan, np.nan

    mtr = mask.loc[tr.index]
    mte = mask.loc[te.index]

    tr_on = tr.loc[mtr, "pnl_r"]
    tr_off = tr.loc[~mtr, "pnl_r"]
    te_on = te.loc[mte, "pnl_r"]
    te_off = te.loc[~mte, "pnl_r"]

    tr_up = float(tr_on.mean() - tr_off.mean()) if len(tr_on) >= 80 and len(tr_off) >= 80 else np.nan
    te_up = float(te_on.mean() - te_off.mean()) if len(te_on) >= 40 and len(te_off) >= 40 else np.nan
    return tr_up, te_up


def scan_symbol_session(con: duckdb.DuckDBPyConnection, symbol: str, session: str) -> pd.DataFrame:
    s = safe_label(session)
    q = f"""
    SELECT
      o.trading_day,
      o.entry_model,
      o.confirm_bars,
      o.rr_target,
      o.pnl_r,
      df.orb_{s}_break_dir AS break_dir,
      df.orb_{s}_break_delay_min AS break_delay,
      df.orb_{s}_size AS orb_size
    FROM orb_outcomes o
    JOIN daily_features df
      ON df.symbol=o.symbol
     AND df.trading_day=o.trading_day
     AND df.orb_minutes=o.orb_minutes
    WHERE o.orb_minutes=5
      AND o.symbol='{symbol}'
      AND o.orb_label='{session}'
      AND o.entry_model IN ('E0','E1')
      AND o.pnl_r IS NOT NULL
    """

    df = con.execute(q).fetchdf()
    if df.empty:
        return pd.DataFrame()

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year

    q40 = df["orb_size"].quantile(0.40)
    q60 = df["orb_size"].quantile(0.60)

    fast30 = df["break_delay"].notna() & (df["break_delay"] <= 30)
    size40 = df["orb_size"].notna() & (df["orb_size"] >= q40)
    size60 = df["orb_size"].notna() & (df["orb_size"] >= q60)

    filters = {
        "fast_le_30": fast30,
        "size_q40_plus": size40,
        "size_q60_plus": size60,
        "fast30_and_q40": fast30 & size40,
        "fast30_and_q60": fast30 & size60,
        "long_fast30": (df["break_dir"] == "long") & fast30,
        "short_fast30": (df["break_dir"] == "short") & fast30,
        "long_q60": (df["break_dir"] == "long") & size60,
        "short_q60": (df["break_dir"] == "short") & size60,
    }

    rows = []
    for (em, cb, rr), g in df.groupby(["entry_model", "confirm_bars", "rr_target"]):
        n_base = len(g)
        if n_base < MIN_BASE:
            continue
        years = max(1, g["year"].nunique())

        for fname, mask_all in filters.items():
            m = mask_all.loc[g.index]
            on = g.loc[m, "pnl_r"]
            off = g.loc[~m, "pnl_r"]
            if len(on) < MIN_ON or len(off) < MIN_ON:
                continue

            avg_on = float(on.mean())
            avg_off = float(off.mean())
            uplift = avg_on - avg_off
            yp, yt = yearly_uplift(g, m)
            tr_up, te_up = oos_uplift(g, m)

            rows.append(
                {
                    "symbol": symbol,
                    "session": session,
                    "entry_model": em,
                    "confirm_bars": int(cb),
                    "rr_target": float(rr),
                    "filter": fname,
                    "n_base": n_base,
                    "n_on": int(len(on)),
                    "signals_per_year": len(on) / years,
                    "avg_on": avg_on,
                    "avg_off": avg_off,
                    "uplift": uplift,
                    "wr_on": float((on > 0).mean()),
                    "wr_off": float((off > 0).mean()),
                    "years_pos": yp,
                    "years_total": yt,
                    "years_pos_ratio": (yp / yt) if yt else np.nan,
                    "train_uplift": tr_up,
                    "test_uplift": te_up,
                }
            )

    return pd.DataFrame(rows)


def main() -> int:
    con = duckdb.connect(DB_PATH, read_only=True)
    sess_df = con.execute(
        """
        SELECT symbol, orb_label, COUNT(*) AS n
        FROM orb_outcomes
        WHERE orb_minutes=5
          AND symbol IN ('M2K','MES','M6E','MGC')
        GROUP BY 1,2
        HAVING COUNT(*) >= 1000
        ORDER BY symbol, n DESC
        """
    ).fetchdf()

    parts = []
    for r in sess_df.itertuples(index=False):
        p = scan_symbol_session(con, r.symbol, r.orb_label)
        if not p.empty:
            parts.append(p)
    con.close()

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_all = out_dir / "wide_non_leadlag_composite_all.csv"
    p_short = out_dir / "wide_non_leadlag_composite_shortlist.csv"
    p_md = out_dir / "wide_non_leadlag_composite_notes.md"

    if not parts:
        p_md.write_text("# Composite scan\n\nNo rows.", encoding="utf-8")
        print("No rows")
        return 0

    all_df = pd.concat(parts, ignore_index=True)
    all_df = all_df.sort_values(["avg_on", "uplift"], ascending=False)
    all_df.to_csv(p_all, index=False)

    short = all_df[
        (all_df["signals_per_year"] >= MIN_SIGNALS_PER_YEAR)
        & (all_df["avg_on"] >= MIN_AVG_ON)
        & (all_df["uplift"] >= MIN_UPLIFT)
        & (all_df["years_total"] >= MIN_YEARS_TOTAL)
        & (all_df["years_pos_ratio"] >= MIN_YEARS_POS_RATIO)
        & (all_df["test_uplift"].fillna(-999) >= 0)
    ].copy().sort_values(["avg_on", "uplift"], ascending=False)
    short.to_csv(p_short, index=False)

    lines = [
        "# Wide Non-Lead-Lag Composite Scan",
        "",
        f"Rows total: {len(all_df)}",
        f"Shortlist: {len(short)}",
        "",
        "## Top shortlist",
    ]

    if short.empty:
        lines.append("- None met gates.")
    else:
        for r in short.head(20).itertuples(index=False):
            lines.append(
                f"- {r.symbol} {r.session} {r.entry_model}/CB{r.confirm_bars}/RR{r.rr_target} | {r.filter}: avg_on={r.avg_on:+.4f}, Δ={r.uplift:+.4f}, sig/yr={r.signals_per_year:.1f}, years+={r.years_pos}/{r.years_total}, testΔ={r.test_uplift:+.4f}"
            )

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_all}")
    print(f"Saved: {p_short}")
    print(f"Saved: {p_md}")
    print(short.head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
