#!/usr/bin/env python3
"""Wide non-lead-lag filter scan (breadth-first).

Goal:
- Find common-ground shinies without cross-asset dependency.
- Enforce high-frequency preference (~150+ usable signals/year).

Universe:
- Symbols: M2K, MES, M6E, MGC
- Entry models: E0, E1
- Filters: direction, break-speed, ORB-size, combined variants

Outputs:
- research/output/wide_non_leadlag_all.csv
- research/output/wide_non_leadlag_shortlist.csv
- research/output/wide_non_leadlag_notes.md
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
MODELS = ["E0", "E1"]

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


def _yearly_uplift(df: pd.DataFrame, mask: pd.Series) -> tuple[int, int]:
    yp = 0
    yt = 0
    for y, g in df.groupby("year"):
        m = mask.loc[g.index]
        on = g.loc[m, "pnl_r"]
        off = g.loc[~m, "pnl_r"]
        if len(on) < 20 or len(off) < 20:
            continue
        yt += 1
        if on.mean() - off.mean() > 0:
            yp += 1
    return yp, yt


def _oos_uplift(df: pd.DataFrame, mask: pd.Series) -> tuple[float, float, int, int]:
    # Holdout: prefer 2025 if present with enough rows; else latest year with >=200 rows
    counts = df.groupby("year").size().to_dict()
    if 2025 in counts and counts[2025] >= 200:
        test_year = 2025
    else:
        elig = [int(y) for y, n in counts.items() if n >= 200]
        if not elig:
            return np.nan, np.nan, 0, 0
        test_year = max(elig)

    tr = df[df["year"] < test_year]
    te = df[df["year"] == test_year]
    if tr.empty or te.empty:
        return np.nan, np.nan, 0, 0

    mtr = mask.loc[tr.index]
    mte = mask.loc[te.index]

    tr_on = tr.loc[mtr, "pnl_r"]
    tr_off = tr.loc[~mtr, "pnl_r"]
    te_on = te.loc[mte, "pnl_r"]
    te_off = te.loc[~mte, "pnl_r"]

    tr_up = float(tr_on.mean() - tr_off.mean()) if len(tr_on) >= 80 and len(tr_off) >= 80 else np.nan
    te_up = float(te_on.mean() - te_off.mean()) if len(te_on) >= 40 and len(te_off) >= 40 else np.nan
    return tr_up, te_up, int(len(te_on)), int(len(te_off))


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

    # Session-level size cutoffs
    q50 = df["orb_size"].quantile(0.50)
    q75 = df["orb_size"].quantile(0.75)

    filters = {
        "long_only": df["break_dir"] == "long",
        "short_only": df["break_dir"] == "short",
        "fast_le_15": df["break_delay"].notna() & (df["break_delay"] <= 15),
        "mid_15_60": df["break_delay"].notna() & (df["break_delay"] > 15) & (df["break_delay"] <= 60),
        "slow_gt_60": df["break_delay"].notna() & (df["break_delay"] > 60),
        "size_q50_plus": df["orb_size"].notna() & (df["orb_size"] >= q50),
        "size_q75_plus": df["orb_size"].notna() & (df["orb_size"] >= q75),
        "fast_and_q75": df["break_delay"].notna() & (df["break_delay"] <= 15) & df["orb_size"].notna() & (df["orb_size"] >= q75),
        "long_fast": (df["break_dir"] == "long") & df["break_delay"].notna() & (df["break_delay"] <= 15),
        "short_fast": (df["break_dir"] == "short") & df["break_delay"].notna() & (df["break_delay"] <= 15),
    }

    rows = []
    for (em, cb, rr), g in df.groupby(["entry_model", "confirm_bars", "rr_target"]):
        if em not in MODELS:
            continue
        n_base = len(g)
        if n_base < MIN_BASE:
            continue

        years_total_all = max(1, g["year"].nunique())

        for fname, fmask_all in filters.items():
            m = fmask_all.loc[g.index]
            on = g.loc[m, "pnl_r"]
            off = g.loc[~m, "pnl_r"]
            if len(on) < MIN_ON or len(off) < MIN_ON:
                continue

            avg_on = float(on.mean())
            avg_off = float(off.mean())
            uplift = avg_on - avg_off

            yp, yt = _yearly_uplift(g, m)
            tr_up, te_up, n_test_on, n_test_off = _oos_uplift(g, m)

            sig_per_year = len(on) / years_total_all

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
                    "signals_per_year": sig_per_year,
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
                    "n_test_on": n_test_on,
                    "n_test_off": n_test_off,
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

    p_all = out_dir / "wide_non_leadlag_all.csv"
    p_short = out_dir / "wide_non_leadlag_shortlist.csv"
    p_notes = out_dir / "wide_non_leadlag_notes.md"

    if not parts:
        p_notes.write_text("# Wide non-lead-lag scan\n\nNo rows met base thresholds.", encoding="utf-8")
        print("No rows.")
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
    ].copy()
    short = short.sort_values(["avg_on", "uplift", "signals_per_year"], ascending=False)
    short.to_csv(p_short, index=False)

    lines = [
        "# Wide Non-Lead-Lag Scan",
        "",
        "Breadth-first scan across symbols/sessions/models using single-asset filters.",
        f"Frequency gate: >= {MIN_SIGNALS_PER_YEAR} signals/year.",
        "",
        f"Total candidate rows: {len(all_df)}",
        f"Shortlist rows: {len(short)}",
        "",
        "## Top shortlist",
    ]

    if short.empty:
        lines.append("- None met hard gates this round.")
    else:
        for r in short.head(20).itertuples(index=False):
            lines.append(
                f"- {r.symbol} {r.session} {r.entry_model}/CB{r.confirm_bars}/RR{r.rr_target} | {r.filter}: "
                f"avg_on={r.avg_on:+.4f}, Δ={r.uplift:+.4f}, signals/yr={r.signals_per_year:.1f}, years+={r.years_pos}/{r.years_total}, testΔ={r.test_uplift:+.4f}"
            )

    p_notes.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_all}")
    print(f"Saved: {p_short}")
    print(f"Saved: {p_notes}")
    print("\nTop shortlist:")
    print(short.head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
