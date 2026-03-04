#!/usr/bin/env python3
"""Lead-lag old-style round 2: hunt big-gain challengers first, then frequency filter.

Restores the earlier high-gain discovery style:
- same-direction lead-lag
- no-lookahead (leader break ts <= follower entry ts)
- wide follower strategy grid

Outputs:
- research/output/lead_lag_oldstyle_round2_all.csv
- research/output/lead_lag_oldstyle_round2_topgain.csv
- research/output/lead_lag_oldstyle_round2_freq150.csv
- research/output/lead_lag_oldstyle_round2_notes.md
"""

from __future__ import annotations

import re
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = "gold.db"
PAIRS_SOURCE = PROJECT_ROOT / "research" / "output" / "fast_lead_lag_extended_summary.csv"

TOP_PAIRS = 30
MIN_PAIR_ON = 120
MIN_BASE = 300
MIN_ON = 80
MIN_OFF = 80
MIN_TEST_ON = 40
MIN_TEST_OFF = 40
MIN_TRAIN_ON = 80
MIN_TRAIN_OFF = 80


def safe_label(label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", label):
        raise ValueError(f"Unsafe label: {label}")
    return label


def parse_pair(tag: str) -> tuple[str, str]:
    sym, sess = tag.split("_", 1)
    return sym, sess


def is_valid_pair_tag(tag: str) -> bool:
    if not isinstance(tag, str):
        return False
    if "_" not in tag:
        return False
    sym, sess = tag.split("_", 1)
    if not sym or not sess:
        return False
    if not re.fullmatch(r"[A-Za-z0-9]+", sym):
        return False
    if not re.fullmatch(r"[A-Za-z0-9_]+", sess):
        return False
    return True


def load_top_pairs() -> pd.DataFrame:
    s = pd.read_csv(PAIRS_SOURCE)
    s = s.dropna(subset=["leader", "follower", "n_on", "uplift_on_vs_off", "avg_r_on"]).copy()
    s["leader"] = s["leader"].astype(str)
    s["follower"] = s["follower"].astype(str)
    s = s[s["leader"].map(is_valid_pair_tag) & s["follower"].map(is_valid_pair_tag)]
    s = s[(s["n_on"] >= MIN_PAIR_ON)].copy()
    s = s.drop_duplicates(subset=["leader", "follower"])

    # old-style: chase uplift + strong avg_on
    s = s.sort_values(["uplift_on_vs_off", "avg_r_on"], ascending=False).head(TOP_PAIRS)
    return s[["leader", "follower", "n_on", "n_base", "uplift_on_vs_off", "avg_r_on"]]


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
      SUM(CASE WHEN (l_dir IN ('long','short') AND f_dir IN ('long','short') AND l_ts IS NOT NULL AND l_ts<=entry_ts AND l_dir=f_dir) THEN pnl_r ELSE 0 END) AS sum_on,
      SUM(CASE WHEN NOT (l_dir IN ('long','short') AND f_dir IN ('long','short') AND l_ts IS NOT NULL AND l_ts<=entry_ts AND l_dir=f_dir) THEN pnl_r ELSE 0 END) AS sum_off,
      SUM(pnl_r) AS sum_base
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


def aggregate(ydf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (leader, follower, symbol, session, em, cb, rr), g in ydf.groupby(
        ["leader", "follower", "symbol", "session", "entry_model", "confirm_bars", "rr_target"]
    ):
        n_base = int(g["n_base"].sum())
        n_on = int(g["n_on"].sum())
        n_off = int(g["n_off"].sum())
        if n_base < MIN_BASE or n_on < MIN_ON or n_off < MIN_OFF:
            continue

        avg_on = float(g["sum_on"].sum()) / n_on
        avg_off = float(g["sum_off"].sum()) / n_off
        uplift = avg_on - avg_off

        years_pos = 0
        years_total = 0
        for r in g.itertuples(index=False):
            if int(r.n_on) < 40 or int(r.n_off) < 40:
                continue
            years_total += 1
            if (float(r.sum_on) / int(r.n_on)) - (float(r.sum_off) / int(r.n_off)) > 0:
                years_pos += 1

        # quick OOS 2025 + train uplift
        g2025 = g[g["y"] == 2025]
        gtrain = g[g["y"] < 2025]
        test_uplift = np.nan
        train_uplift = np.nan
        n_test_on = 0
        n_test_off = 0

        if not gtrain.empty:
            n_on_tr = int(gtrain["n_on"].sum())
            n_off_tr = int(gtrain["n_off"].sum())
            if n_on_tr >= MIN_TRAIN_ON and n_off_tr >= MIN_TRAIN_OFF:
                train_uplift = (float(gtrain["sum_on"].sum()) / n_on_tr) - (float(gtrain["sum_off"].sum()) / n_off_tr)

        if not g2025.empty:
            n_on_25 = int(g2025["n_on"].sum())
            n_off_25 = int(g2025["n_off"].sum())
            n_test_on = n_on_25
            n_test_off = n_off_25
            if n_on_25 >= MIN_TEST_ON and n_off_25 >= MIN_TEST_OFF:
                test_uplift = (float(g2025["sum_on"].sum()) / n_on_25) - (float(g2025["sum_off"].sum()) / n_off_25)

        # frequency normalization: use full historical years (prefer <=2025 to avoid partial current year distortion)
        years_set = set(int(v) for v in g["y"].dropna().tolist())
        full_years = sorted([y for y in years_set if y <= 2025])
        years_covered = len(full_years) if full_years else max(1, len(years_set))
        signals_per_year = n_on / years_covered

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
                "signals_per_year": signals_per_year,
                "avg_on": avg_on,
                "avg_off": avg_off,
                "uplift": uplift,
                "years_pos": years_pos,
                "years_total": years_total,
                "years_pos_ratio": (years_pos / years_total) if years_total else np.nan,
                "train_uplift": train_uplift,
                "test2025_uplift": test_uplift,
                "n_test_on": n_test_on,
                "n_test_off": n_test_off,
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    pairs = load_top_pairs()
    con = duckdb.connect(DB_PATH, read_only=True)

    parts = []
    for r in pairs.itertuples(index=False):
        ydf = scan_pair(con, r.leader, r.follower)
        if ydf.empty:
            continue
        a = aggregate(ydf)
        if not a.empty:
            parts.append(a)

    con.close()

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_all = out_dir / "lead_lag_oldstyle_round2_all.csv"
    p_top = out_dir / "lead_lag_oldstyle_round2_topgain.csv"
    p_f150 = out_dir / "lead_lag_oldstyle_round2_freq150.csv"
    p_md = out_dir / "lead_lag_oldstyle_round2_notes.md"

    if not parts:
        p_md.write_text("# Lead-lag oldstyle round2\n\nNo rows.", encoding="utf-8")
        print("No rows.")
        return 0

    all_df = pd.concat(parts, ignore_index=True)
    all_df = all_df.sort_values(["avg_on", "uplift"], ascending=False)
    all_df.to_csv(p_all, index=False)

    topgain = all_df[
        (all_df["avg_on"] > 0)
        & (all_df["uplift"] >= 0.18)
        & (all_df["years_total"] >= 3)
        & (all_df["years_pos_ratio"] >= 0.6)
        & (all_df["train_uplift"].fillna(-999) >= 0)
        & (all_df["test2025_uplift"].fillna(-999) >= 0)
        & (all_df["n_test_on"] >= MIN_TEST_ON)
        & (all_df["n_test_off"] >= MIN_TEST_OFF)
    ].copy().sort_values(["avg_on", "uplift"], ascending=False)
    topgain.to_csv(p_top, index=False)

    freq150 = topgain[topgain["signals_per_year"] >= 150].copy().sort_values(["avg_on", "uplift"], ascending=False)
    freq150.to_csv(p_f150, index=False)

    lines = [
        "# Lead-Lag Oldstyle Round2",
        "",
        f"Pairs scanned: {len(pairs)}",
        f"All combos: {len(all_df)}",
        f"Top-gain shortlist: {len(topgain)}",
        f"Top-gain + freq150: {len(freq150)}",
        "",
        "## Top gain",
    ]

    for r in topgain.head(20).itertuples(index=False):
        lines.append(
            f"- {r.leader}->{r.follower} {r.entry_model}/CB{r.confirm_bars}/RR{r.rr_target}: avg_on={r.avg_on:+.4f}, Δ={r.uplift:+.4f}, sig/yr={r.signals_per_year:.1f}, years+={r.years_pos}/{r.years_total}, trainΔ={r.train_uplift:+.4f}, test2025Δ={r.test2025_uplift:+.4f}, Ntest={int(r.n_test_on)}/{int(r.n_test_off)}"
        )

    lines.append("")
    lines.append("## Frequency 150+")
    if freq150.empty:
        lines.append("- None")
    else:
        for r in freq150.head(20).itertuples(index=False):
            lines.append(
                f"- {r.leader}->{r.follower} {r.entry_model}/CB{r.confirm_bars}/RR{r.rr_target}: avg_on={r.avg_on:+.4f}, Δ={r.uplift:+.4f}, sig/yr={r.signals_per_year:.1f}"
            )

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_all}")
    print(f"Saved: {p_top}")
    print(f"Saved: {p_f150}")
    print(f"Saved: {p_md}")
    print("\nTop gain:")
    print(topgain.head(25).to_string(index=False))
    print("\nFreq150:")
    print(freq150.head(25).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
