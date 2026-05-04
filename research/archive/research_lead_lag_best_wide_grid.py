#!/usr/bin/env python3
"""Best lead-lag pairs + wide strategy grid scan.

Workflow:
1) Load previously ranked lead-lag pairs (fast_lead_lag_extended_summary.csv)
2) Take top pairs by uplift (with minimum sample)
3) For each pair, scan wide follower strategy grid (entry_model/confirm_bars/rr_target)
4) Compute aggregate + yearly stability + quick OOS holdout

Outputs:
- research/output/lead_lag_best_wide_grid.csv
- research/output/lead_lag_best_wide_grid_shortlist.csv
- research/output/lead_lag_best_wide_grid.md
"""

from __future__ import annotations

import re
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = "gold.db"
PAIRS_SOURCE = PROJECT_ROOT / "research" / "output" / "fast_lead_lag_extended_summary.csv"

TOP_PAIRS = 14
MIN_PAIR_ON = 150
MIN_BASE = 500
MIN_ON = 80
MIN_OFF = 80
MIN_YEAR_ON = 25
MIN_YEAR_OFF = 25
MIN_UPLIFT = 0.12
MIN_ON_AVG = 0.0

SYMS = {"MES", "MNQ", "M2K", "M6E"}


def safe_label(label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", label):
        raise ValueError(f"Unsafe label: {label}")
    return label


def parse_pair(tag: str) -> tuple[str, str]:
    sym, sess = tag.split("_", 1)
    if sym not in SYMS:
        raise ValueError(f"Unexpected symbol in tag: {tag}")
    return sym, sess


def load_top_pairs() -> pd.DataFrame:
    if not PAIRS_SOURCE.exists():
        raise FileNotFoundError(f"Missing source file: {PAIRS_SOURCE}")

    s = pd.read_csv(PAIRS_SOURCE)
    s = s[(s["n_on"] >= MIN_PAIR_ON)].copy()
    s = s.sort_values(["uplift_on_vs_off", "avg_r_on"], ascending=False).head(TOP_PAIRS)
    return s[["leader", "follower", "n_on", "n_base", "uplift_on_vs_off"]]


def scan_pair(con: duckdb.DuckDBPyConnection, leader: str, follower: str) -> pd.DataFrame:
    lsym, lsess = parse_pair(leader)
    fsym, fsess = parse_pair(follower)

    l_dir = f"orb_{safe_label(lsess)}_break_dir"
    l_ts = f"orb_{safe_label(lsess)}_break_ts"
    f_dir = f"orb_{safe_label(fsess)}_break_dir"

    q = f"""
    WITH base AS (
      SELECT
        o.entry_model,
        o.confirm_bars,
        o.rr_target,
        EXTRACT(YEAR FROM o.trading_day) AS y,
        o.pnl_r,
        o.entry_ts,
        df_f.{f_dir} AS f_dir,
        df_l.{l_dir} AS l_dir,
        df_l.{l_ts}  AS l_ts
      FROM orb_outcomes o
      JOIN daily_features df_f
        ON df_f.symbol=o.symbol
       AND df_f.trading_day=o.trading_day
       AND df_f.orb_minutes=o.orb_minutes
      JOIN daily_features df_l
        ON df_l.symbol='{lsym}'
       AND df_l.trading_day=o.trading_day
       AND df_l.orb_minutes=o.orb_minutes
      WHERE o.orb_minutes=5
        AND o.symbol='{fsym}'
        AND o.orb_label='{fsess}'
        AND o.pnl_r IS NOT NULL
        AND o.entry_ts IS NOT NULL
    )
    SELECT
      entry_model,
      confirm_bars,
      rr_target,
      y,
      COUNT(*) AS n_base,
      SUM(CASE WHEN (l_dir IN ('long','short') AND f_dir IN ('long','short') AND l_ts IS NOT NULL AND l_ts<=entry_ts AND l_dir=f_dir) THEN 1 ELSE 0 END) AS n_on,
      SUM(CASE WHEN NOT (l_dir IN ('long','short') AND f_dir IN ('long','short') AND l_ts IS NOT NULL AND l_ts<=entry_ts AND l_dir=f_dir) THEN 1 ELSE 0 END) AS n_off,
      SUM(pnl_r) AS sum_base,
      SUM(CASE WHEN (l_dir IN ('long','short') AND f_dir IN ('long','short') AND l_ts IS NOT NULL AND l_ts<=entry_ts AND l_dir=f_dir) THEN pnl_r ELSE 0 END) AS sum_on,
      SUM(CASE WHEN NOT (l_dir IN ('long','short') AND f_dir IN ('long','short') AND l_ts IS NOT NULL AND l_ts<=entry_ts AND l_dir=f_dir) THEN pnl_r ELSE 0 END) AS sum_off
    FROM base
    GROUP BY 1,2,3,4
    """

    ydf = con.execute(q).fetchdf()
    if ydf.empty:
        return ydf

    ydf["leader"] = leader
    ydf["follower"] = follower
    ydf["symbol"] = fsym
    ydf["session"] = fsess
    return ydf


def aggregate_combo(ydf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    yearly_rows = []

    for (leader, follower, symbol, session, em, cb, rr), g in ydf.groupby(
        ["leader", "follower", "symbol", "session", "entry_model", "confirm_bars", "rr_target"]
    ):
        n_base = int(g["n_base"].sum())
        n_on = int(g["n_on"].sum())
        n_off = int(g["n_off"].sum())

        if n_base < MIN_BASE or n_on < MIN_ON or n_off < MIN_OFF:
            continue

        sum_base = float(g["sum_base"].sum())
        sum_on = float(g["sum_on"].sum())
        sum_off = float(g["sum_off"].sum())

        avg_base = sum_base / n_base
        avg_on = sum_on / n_on
        avg_off = sum_off / n_off
        uplift = avg_on - avg_off

        # Year stability
        yg = g.copy()
        yg = yg[(yg["n_on"] >= MIN_YEAR_ON) & (yg["n_off"] >= MIN_YEAR_OFF)]
        years = sorted(yg["y"].astype(int).tolist()) if not yg.empty else []
        years_pos = 0
        for r in yg.itertuples(index=False):
            a_on = float(r.sum_on) / float(r.n_on)
            a_off = float(r.sum_off) / float(r.n_off)
            up = a_on - a_off
            yearly_rows.append(
                {
                    "leader": leader,
                    "follower": follower,
                    "symbol": symbol,
                    "session": session,
                    "entry_model": em,
                    "confirm_bars": int(cb),
                    "rr_target": float(rr),
                    "year": int(r.y),
                    "n_on": int(r.n_on),
                    "n_off": int(r.n_off),
                    "avg_on": a_on,
                    "avg_off": a_off,
                    "uplift": up,
                }
            )
            if up > 0:
                years_pos += 1

        years_total = len(years)
        years_pos_ratio = (years_pos / years_total) if years_total else np.nan

        # Quick OOS: latest year with enough total rows for this combo
        g2 = g.copy()
        g2["n_tot_y"] = g2["n_base"]
        eligible_years = [int(r.y) for r in g2.itertuples(index=False) if int(r.n_base) >= 200]
        test_year = max(eligible_years) if eligible_years else None

        train_uplift = np.nan
        test_uplift = np.nan
        n_test_on = 0
        if test_year is not None:
            tr = g2[g2["y"] < test_year]
            te = g2[g2["y"] == test_year]
            tr_n_on, tr_n_off = int(tr["n_on"].sum()), int(tr["n_off"].sum())
            te_n_on, te_n_off = int(te["n_on"].sum()), int(te["n_off"].sum())
            n_test_on = te_n_on

            if tr_n_on >= MIN_ON and tr_n_off >= MIN_OFF:
                tr_avg_on = float(tr["sum_on"].sum()) / tr_n_on
                tr_avg_off = float(tr["sum_off"].sum()) / tr_n_off
                train_uplift = tr_avg_on - tr_avg_off
            if te_n_on >= max(20, MIN_ON // 3) and te_n_off >= max(20, MIN_OFF // 3):
                te_avg_on = float(te["sum_on"].sum()) / te_n_on
                te_avg_off = float(te["sum_off"].sum()) / te_n_off
                test_uplift = te_avg_on - te_avg_off

        rows.append(
            {
                "leader": leader,
                "follower": follower,
                "symbol": symbol,
                "session": session,
                "entry_model": em,
                "confirm_bars": int(cb),
                "rr_target": float(rr),
                "n_base": n_base,
                "n_on": n_on,
                "on_rate": n_on / n_base,
                "avg_r_on": avg_on,
                "avg_r_off": avg_off,
                "uplift": uplift,
                "years_pos": years_pos,
                "years_total": years_total,
                "years_pos_ratio": years_pos_ratio,
                "train_uplift": train_uplift,
                "test_uplift": test_uplift,
                "n_test_on": n_test_on,
            }
        )

    s = pd.DataFrame(rows)
    y = pd.DataFrame(yearly_rows)
    return s, y


def main() -> int:
    top_pairs = load_top_pairs()
    con = duckdb.connect(DB_PATH, read_only=True)

    parts = []
    yparts = []
    for r in top_pairs.itertuples(index=False):
        ydf = scan_pair(con, r.leader, r.follower)
        if ydf.empty:
            continue
        s, y = aggregate_combo(ydf)
        if not s.empty:
            parts.append(s)
        if not y.empty:
            yparts.append(y)

    con.close()

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_all = out_dir / "lead_lag_best_wide_grid.csv"
    p_short = out_dir / "lead_lag_best_wide_grid_shortlist.csv"
    p_md = out_dir / "lead_lag_best_wide_grid.md"

    if not parts:
        p_md.write_text("# Lead-lag best wide grid\n\nNo rows met thresholds.", encoding="utf-8")
        print("No rows met thresholds.")
        return 0

    all_df = pd.concat(parts, ignore_index=True)
    all_df = all_df.sort_values(["uplift", "avg_r_on"], ascending=False)
    all_df.to_csv(p_all, index=False)

    shortlist = all_df[
        (all_df["avg_r_on"] >= MIN_ON_AVG)
        & (all_df["uplift"] >= MIN_UPLIFT)
        & (all_df["years_total"] >= 2)
        & (all_df["years_pos_ratio"] >= 0.6)
        & (all_df["test_uplift"].fillna(-999) >= 0)
    ].copy()
    shortlist = shortlist.sort_values(["uplift", "test_uplift", "avg_r_on"], ascending=False)
    shortlist.to_csv(p_short, index=False)

    lines = [
        "# Lead-Lag Best Pairs Wide Grid",
        "",
        f"- Top pair source: {PAIRS_SOURCE.name}",
        f"- Pairs scanned: {len(top_pairs)}",
        f"- Combo thresholds: n_base>={MIN_BASE}, n_on/off>={MIN_ON}/{MIN_OFF}",
        f"- Promotion gates: avg_on>={MIN_ON_AVG:+.2f}, uplift>={MIN_UPLIFT:+.2f}, years_pos_ratio>=0.6, test_uplift>=0",
        "",
        f"Total combos passing base thresholds: {len(all_df)}",
        f"Shortlist (promotable): {len(shortlist)}",
        "",
        "## Top 15 all",
    ]

    for r in all_df.head(15).itertuples(index=False):
        lines.append(
            f"- {r.leader} -> {r.follower} | {r.entry_model}/CB{r.confirm_bars}/RR{r.rr_target}: "
            f"N_on={r.n_on}, avg_on={r.avg_r_on:+.4f}, avg_off={r.avg_r_off:+.4f}, Δ={r.uplift:+.4f}, "
            f"years+={r.years_pos}/{r.years_total}, testΔ={r.test_uplift:+.4f}"
        )

    lines.append("")
    lines.append("## Shortlist")
    if shortlist.empty:
        lines.append("- None met promotion gates.")
    else:
        for r in shortlist.head(15).itertuples(index=False):
            lines.append(
                f"- {r.leader} -> {r.follower} | {r.entry_model}/CB{r.confirm_bars}/RR{r.rr_target}: "
                f"avg_on={r.avg_r_on:+.4f}, Δ={r.uplift:+.4f}, years+={r.years_pos}/{r.years_total}, testΔ={r.test_uplift:+.4f}"
            )

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_all}")
    print(f"Saved: {p_short}")
    print(f"Saved: {p_md}")
    print("\nTop all:")
    print(all_df.head(20).to_string(index=False))
    print("\nShortlist:")
    print(shortlist.head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
