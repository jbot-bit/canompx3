#!/usr/bin/env python3
"""New angle: contrarian lead-lag scan (opposite-direction conditions).

Compares SAME vs OPP direction for selected high-liquidity leader->follower pairs.
No-lookahead enforced via leader_break_ts <= follower entry_ts.
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

TOP_PAIRS = 20
MIN_BASE = 300
MIN_ON = 80
MIN_OFF = 80


def safe_label(label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", label):
        raise ValueError(f"Unsafe label: {label}")
    return label


def valid_tag(tag: str) -> bool:
    if not isinstance(tag, str) or "_" not in tag:
        return False
    a, b = tag.split("_", 1)
    return bool(a) and bool(b) and (re.fullmatch(r"[A-Za-z0-9]+", a) is not None) and (re.fullmatch(r"[A-Za-z0-9_]+", b) is not None)


def parse_tag(tag: str) -> tuple[str, str]:
    return tag.split("_", 1)


def load_pairs() -> pd.DataFrame:
    s = pd.read_csv(PAIRS_SOURCE)
    s = s.dropna(subset=["leader", "follower", "n_base"]).copy()
    s["leader"] = s["leader"].astype(str)
    s["follower"] = s["follower"].astype(str)
    s = s[s["leader"].map(valid_tag) & s["follower"].map(valid_tag)]
    # choose dense pairs to avoid noise in this new angle
    s = s.sort_values(["n_base", "n_on"], ascending=False).drop_duplicates(["leader", "follower"]).head(TOP_PAIRS)
    return s[["leader", "follower", "n_base"]]


def scan_pair(con: duckdb.DuckDBPyConnection, leader: str, follower: str) -> pd.DataFrame:
    lsym, lsess = parse_tag(leader)
    fsym, fsess = parse_tag(follower)

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
      SUM(CASE WHEN (l_dir IN ('long','short') AND f_dir IN ('long','short') AND l_ts IS NOT NULL AND l_ts<=entry_ts AND l_dir=f_dir) THEN 1 ELSE 0 END) AS n_same,
      SUM(CASE WHEN (l_dir IN ('long','short') AND f_dir IN ('long','short') AND l_ts IS NOT NULL AND l_ts<=entry_ts AND l_dir<>f_dir) THEN 1 ELSE 0 END) AS n_opp,
      SUM(CASE WHEN (l_dir IN ('long','short') AND f_dir IN ('long','short') AND l_ts IS NOT NULL AND l_ts<=entry_ts AND l_dir=f_dir) THEN pnl_r ELSE 0 END) AS sum_same,
      SUM(CASE WHEN (l_dir IN ('long','short') AND f_dir IN ('long','short') AND l_ts IS NOT NULL AND l_ts<=entry_ts AND l_dir<>f_dir) THEN pnl_r ELSE 0 END) AS sum_opp,
      SUM(CASE WHEN (l_dir IN ('long','short') AND f_dir IN ('long','short') AND l_ts IS NOT NULL AND l_ts<=entry_ts) THEN pnl_r ELSE 0 END) AS sum_valid,
      SUM(CASE WHEN (l_dir IN ('long','short') AND f_dir IN ('long','short') AND l_ts IS NOT NULL AND l_ts<=entry_ts) THEN 1 ELSE 0 END) AS n_valid
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
    for (leader, follower, symbol, session, em, cb, rr), g in ydf.groupby([
        "leader", "follower", "symbol", "session", "entry_model", "confirm_bars", "rr_target"
    ]):
        n_base = int(g["n_base"].sum())
        n_same = int(g["n_same"].sum())
        n_opp = int(g["n_opp"].sum())
        n_valid = int(g["n_valid"].sum())
        if n_base < MIN_BASE or n_valid < (MIN_ON + MIN_OFF):
            continue
        if n_same < MIN_ON or n_opp < MIN_ON:
            continue

        avg_same = float(g["sum_same"].sum()) / n_same
        avg_opp = float(g["sum_opp"].sum()) / n_opp
        # compare each mode vs the alternative
        uplift_same_vs_opp = avg_same - avg_opp
        uplift_opp_vs_same = avg_opp - avg_same

        # OOS on 2025 (if enough)
        g25 = g[g["y"] == 2025]
        test_same = np.nan
        test_opp = np.nan
        n_test_same = 0
        n_test_opp = 0
        if not g25.empty:
            n_test_same = int(g25["n_same"].sum())
            n_test_opp = int(g25["n_opp"].sum())
            if n_test_same >= 40:
                test_same = float(g25["sum_same"].sum()) / n_test_same
            if n_test_opp >= 40:
                test_opp = float(g25["sum_opp"].sum()) / n_test_opp

        years_cov = max(1, int((g["n_base"] >= 200).sum()))
        rows.append({
            "leader": leader,
            "follower": follower,
            "symbol": symbol,
            "session": session,
            "entry_model": em,
            "confirm_bars": int(cb),
            "rr_target": float(rr),
            "n_base": n_base,
            "n_same": n_same,
            "n_opp": n_opp,
            "sigyr_same": n_same / years_cov,
            "sigyr_opp": n_opp / years_cov,
            "avg_same": avg_same,
            "avg_opp": avg_opp,
            "uplift_same_vs_opp": uplift_same_vs_opp,
            "uplift_opp_vs_same": uplift_opp_vs_same,
            "test2025_same": test_same,
            "test2025_opp": test_opp,
            "n_test_same": n_test_same,
            "n_test_opp": n_test_opp,
        })

    return pd.DataFrame(rows)


def main() -> int:
    pairs = load_pairs()
    con = duckdb.connect(DB_PATH, read_only=True)

    parts = []
    for r in pairs.itertuples(index=False):
        y = scan_pair(con, r.leader, r.follower)
        if y.empty:
            continue
        a = aggregate(y)
        if not a.empty:
            parts.append(a)
    con.close()

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_all = out_dir / "contrarian_leadlag_all.csv"
    p_top_same = out_dir / "contrarian_leadlag_top_same.csv"
    p_top_opp = out_dir / "contrarian_leadlag_top_opp.csv"
    p_md = out_dir / "contrarian_leadlag_notes.md"

    if not parts:
        p_md.write_text("# Contrarian lead-lag scan\n\nNo rows.", encoding="utf-8")
        print("No rows")
        return 0

    all_df = pd.concat(parts, ignore_index=True)
    all_df.to_csv(p_all, index=False)

    top_same = all_df[(all_df["avg_same"] > 0) & (all_df["uplift_same_vs_opp"] > 0.18) & (all_df["test2025_same"].fillna(-999) > 0)].copy()
    top_same = top_same.sort_values(["avg_same", "uplift_same_vs_opp"], ascending=False)
    top_same.to_csv(p_top_same, index=False)

    top_opp = all_df[(all_df["avg_opp"] > 0) & (all_df["uplift_opp_vs_same"] > 0.18) & (all_df["test2025_opp"].fillna(-999) > 0)].copy()
    top_opp = top_opp.sort_values(["avg_opp", "uplift_opp_vs_same"], ascending=False)
    top_opp.to_csv(p_top_opp, index=False)

    lines = [
        "# Contrarian Lead-Lag Scan",
        "",
        f"Pairs scanned: {len(pairs)}",
        f"All combos: {len(all_df)}",
        f"Top SAME: {len(top_same)}",
        f"Top OPP: {len(top_opp)}",
        "",
        "## Top SAME",
    ]
    for r in top_same.head(12).itertuples(index=False):
        lines.append(f"- {r.leader}->{r.follower} {r.entry_model}/CB{r.confirm_bars}/RR{r.rr_target}: avg_same={r.avg_same:+.4f}, Δsame-opp={r.uplift_same_vs_opp:+.4f}, sig/yr={r.sigyr_same:.1f}, test25={r.test2025_same:+.4f}")

    lines.append("")
    lines.append("## Top OPP")
    for r in top_opp.head(12).itertuples(index=False):
        lines.append(f"- {r.leader}->{r.follower} {r.entry_model}/CB{r.confirm_bars}/RR{r.rr_target}: avg_opp={r.avg_opp:+.4f}, Δopp-same={r.uplift_opp_vs_same:+.4f}, sig/yr={r.sigyr_opp:.1f}, test25={r.test2025_opp:+.4f}")

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_all}")
    print(f"Saved: {p_top_same}")
    print(f"Saved: {p_top_opp}")
    print(f"Saved: {p_md}")
    print("\nTop SAME:")
    print(top_same.head(20).to_string(index=False))
    print("\nTop OPP:")
    print(top_opp.head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
