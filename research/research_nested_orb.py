"""
Nested ORB Research — does a 5-min break that ALSO clears the 15-min (or 30-min)
ORB range have higher expectancy than a 5-min-only break?

Hypothesis:
  - 5-min ORB breaks that simultaneously exceed the 15-min range are stronger
    momentum moves (two timeframes cleared at once)
  - 5-min-only breaks (price is still inside the 15-min range) are weaker /
    more likely to be noise

Categorisation per trade:
  - NESTED_30: entry_price clears the 30-min ORB boundary (strongest)
  - NESTED_15: entry_price clears the 15-min but NOT 30-min ORB boundary
  - SOLO_5:    entry_price is between 5-min and 15-min ORB edge (weakest)

For each category: N, avgR, win_rate, t-stat, p-value.
BH FDR correction applied across all reported cells.

Sessions tested: 0900, 1000, 1100, 1800, 0030, 2300 + US_EQUITY_OPEN, CME_CLOSE
Instruments: MGC, MES, MNQ (one at a time via --instrument flag)
"""

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.paths import GOLD_DB_PATH

# Sessions to test — map label to column prefix in daily_features
SESSION_LABELS = [
    "0900", "1000", "1100", "1800", "0030", "2300",
    "US_EQUITY_OPEN", "CME_CLOSE",
]

# Minimum N to report a cell
MIN_N = 30


def bh_correction(pvalues: list[float], q: float = 0.10) -> list[float]:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(pvalues)
    if n == 0:
        return []
    arr = np.array(pvalues)
    ranked = np.argsort(arr)
    adjusted = np.empty(n)
    for i, idx in enumerate(ranked):
        adjusted[idx] = arr[idx] * n / (i + 1)
    # Make monotone
    for i in range(n - 2, -1, -1):
        adjusted[ranked[i]] = min(adjusted[ranked[i]], adjusted[ranked[i + 1]])
    return adjusted.tolist()


def load_data(con: duckdb.DuckDBPyConnection, instrument: str) -> pd.DataFrame:
    """
    Pull 5-min orb_outcomes joined to 15-min and 30-min daily_features.
    Returns one row per trade with:
      entry_price, stop_price, pnl_r, orb_label, rr_target, confirm_bars, entry_model
      + 15-min and 30-min ORB high/low for each session column
    """
    df = con.execute(f"""
        SELECT
            o.trading_day,
            o.orb_label,
            o.rr_target,
            o.confirm_bars,
            o.entry_model,
            o.entry_price,
            o.stop_price,
            o.pnl_r,
            o.outcome,

            -- 15-min ORB bounds (all session columns — we'll pick the right one in Python)
            d15.orb_0900_high    AS h15_0900,    d15.orb_0900_low    AS l15_0900,
            d15.orb_1000_high    AS h15_1000,    d15.orb_1000_low    AS l15_1000,
            d15.orb_1100_high    AS h15_1100,    d15.orb_1100_low    AS l15_1100,
            d15.orb_1800_high    AS h15_1800,    d15.orb_1800_low    AS l15_1800,
            d15.orb_0030_high    AS h15_0030,    d15.orb_0030_low    AS l15_0030,
            d15.orb_2300_high    AS h15_2300,    d15.orb_2300_low    AS l15_2300,
            d15.orb_US_EQUITY_OPEN_high AS h15_US_EQUITY_OPEN,
            d15.orb_US_EQUITY_OPEN_low  AS l15_US_EQUITY_OPEN,
            d15.orb_CME_CLOSE_high AS h15_CME_CLOSE,
            d15.orb_CME_CLOSE_low  AS l15_CME_CLOSE,

            -- 30-min ORB bounds
            d30.orb_0900_high    AS h30_0900,    d30.orb_0900_low    AS l30_0900,
            d30.orb_1000_high    AS h30_1000,    d30.orb_1000_low    AS l30_1000,
            d30.orb_1100_high    AS h30_1100,    d30.orb_1100_low    AS l30_1100,
            d30.orb_1800_high    AS h30_1800,    d30.orb_1800_low    AS l30_1800,
            d30.orb_0030_high    AS h30_0030,    d30.orb_0030_low    AS l30_0030,
            d30.orb_2300_high    AS h30_2300,    d30.orb_2300_low    AS l30_2300,
            d30.orb_US_EQUITY_OPEN_high AS h30_US_EQUITY_OPEN,
            d30.orb_US_EQUITY_OPEN_low  AS l30_US_EQUITY_OPEN,
            d30.orb_CME_CLOSE_high AS h30_CME_CLOSE,
            d30.orb_CME_CLOSE_low  AS l30_CME_CLOSE

        FROM orb_outcomes o
        JOIN daily_features d15
            ON  d15.trading_day = o.trading_day
            AND d15.symbol      = o.symbol
            AND d15.orb_minutes = 15
        JOIN daily_features d30
            ON  d30.trading_day = o.trading_day
            AND d30.symbol      = o.symbol
            AND d30.orb_minutes = 30
        WHERE o.symbol      = '{instrument}'
          AND o.orb_minutes = 5
          AND o.outcome     IS NOT NULL
          AND o.pnl_r       IS NOT NULL
    """).df()
    return df


def classify_trade(row: pd.Series) -> str:
    """
    Determine SOLO_5 / NESTED_15 / NESTED_30 based on whether
    the entry price clears the 15-min or 30-min ORB boundary.
    """
    label = row["orb_label"]
    ep = row["entry_price"]
    sp = row["stop_price"]

    if pd.isna(ep) or pd.isna(sp):
        return "UNKNOWN"

    is_long = ep > sp

    h15 = row.get(f"h15_{label}")
    l15 = row.get(f"l15_{label}")
    h30 = row.get(f"h30_{label}")
    l30 = row.get(f"l30_{label}")

    if pd.isna(h15) or pd.isna(l15):
        return "NO_15MIN"

    if is_long:
        clears_15 = ep >= h15
        clears_30 = (not pd.isna(h30)) and ep >= h30
    else:
        clears_15 = ep <= l15
        clears_30 = (not pd.isna(l30)) and ep <= l30

    if clears_30:
        return "NESTED_30"
    elif clears_15:
        return "NESTED_15"
    else:
        return "SOLO_5"


def analyse(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (session, rr_target, confirm_bars, entry_model, nest_tier),
    compute N, avgR, win_rate, t-stat, p-value.
    Then add BH correction across all cells.
    """
    df["nest_tier"] = df.apply(classify_trade, axis=1)
    df = df[df["nest_tier"].isin(["SOLO_5", "NESTED_15", "NESTED_30"])]

    records = []
    groups = df.groupby(["orb_label", "rr_target", "confirm_bars", "entry_model", "nest_tier"])

    for (session, rr, cb, em, tier), grp in groups:
        r = grp["pnl_r"].dropna()
        n = len(r)
        if n < MIN_N:
            continue
        avg_r = r.mean()
        wr = (r > 0).mean()
        t_stat, p_val = stats.ttest_1samp(r, 0)
        records.append({
            "session": session,
            "rr": rr,
            "cb": cb,
            "em": em,
            "tier": tier,
            "n": n,
            "avg_r": round(avg_r, 4),
            "win_rate": round(wr, 3),
            "t_stat": round(t_stat, 3),
            "p_val": round(p_val, 4),
        })

    if not records:
        return pd.DataFrame()

    res = pd.DataFrame(records)
    res["p_bh"] = bh_correction(res["p_val"].tolist())
    res["p_bh"] = res["p_bh"].round(4)
    res["bh_sig"] = res["p_bh"] < 0.10
    res = res.sort_values(["session", "rr", "cb", "em", "tier"])
    return res


def print_summary(res: pd.DataFrame, instrument: str) -> None:
    if res.empty:
        print(f"\n{instrument}: No cells with N >= {MIN_N}")
        return

    print(f"\n{'='*80}")
    print(f"NESTED ORB RESULTS — {instrument}")
    print(f"{'='*80}")

    # Tier distribution
    tier_counts = res.groupby("tier")["n"].sum()
    print(f"\nTrade distribution by tier:\n{tier_counts.to_string()}")

    # BH survivors
    survivors = res[res["bh_sig"]]
    print(f"\nBH survivors (q=0.10): {len(survivors)} / {len(res)} cells")

    if not survivors.empty:
        print("\nSURVIVORS:")
        print(survivors[["session","rr","cb","em","tier","n","avg_r","win_rate","p_val","p_bh"]].to_string(index=False))

    # Per-session summary: avg_r by tier (averaged across rr/cb/em)
    print(f"\n{'-'*80}")
    print("Per-session avg_r by tier (averaged across all rr/cb/em combos):")
    pivot = res.groupby(["session", "tier"]).apply(
        lambda g: pd.Series({"avg_r": g["avg_r"].mean(), "n_cells": len(g), "total_n": g["n"].sum()}),
        include_groups=False,
    ).reset_index()
    print(pivot.to_string(index=False))


def save_output(res: pd.DataFrame, instrument: str) -> None:
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"nested_orb_{instrument}.csv"
    res.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Markdown summary
    md_path = out_dir / f"nested_orb_{instrument}.md"
    lines = [
        f"# Nested ORB Research — {instrument}",
        "",
        f"**BH survivors (q=0.10):** {len(res[res['bh_sig']])} / {len(res)} cells",
        "",
        "## Tier avg_r by session",
        "",
    ]
    if not res.empty:
        pivot = res.groupby(["session", "tier"])["avg_r"].mean().unstack(fill_value=float("nan"))
        lines.append(pivot.round(3).to_markdown())
        lines.append("")
        if not res[res["bh_sig"]].empty:
            lines.append("## BH Survivors")
            lines.append("")
            lines.append(res[res["bh_sig"]][["session","rr","cb","em","tier","n","avg_r","p_val","p_bh"]].to_markdown(index=False))
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Nested ORB research")
    parser.add_argument("--instrument", default="MGC", choices=["MGC", "MES", "MNQ"])
    parser.add_argument("--db-path", default=None)
    args = parser.parse_args()

    db_path = args.db_path or GOLD_DB_PATH
    print(f"DB: {db_path}")
    print(f"Instrument: {args.instrument}")

    con = duckdb.connect(str(db_path), read_only=True)
    df = load_data(con, args.instrument)
    print(f"Loaded {len(df):,} trades")

    # Quick tier distribution preview
    df["nest_tier"] = df.apply(classify_trade, axis=1)
    print("\nTier distribution (before N filter):")
    print(df["nest_tier"].value_counts().to_string())

    res = analyse(df)
    print_summary(res, args.instrument)
    if not res.empty:
        save_output(res, args.instrument)


if __name__ == "__main__":
    main()
