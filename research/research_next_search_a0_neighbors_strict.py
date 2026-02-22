#!/usr/bin/env python3
"""Next search now: A0-neighbor strict scan with FDR.

Single pair family:
- Leader: M6E_US_EQUITY_OPEN
- Follower: MES_US_EQUITY_OPEN

Scans nearby strategy variants + fixed condition set.
Controls multiple tests with BH-FDR.
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "gold.db"

MODELS = ["E0", "E1", "E3"]
RRS = [2.0, 2.5, 3.0, 4.0]
CBS = [1, 2, 3, 4, 5]
N_PERM = 300
Q = 0.10


def bh_fdr(df: pd.DataFrame, p_col: str, q: float) -> pd.DataFrame:
    out = df.copy().sort_values(p_col).reset_index(drop=True)
    m = len(out)
    out["rank"] = np.arange(1, m + 1)
    out["bh_threshold"] = (out["rank"] / m) * q

    p = out[p_col].values.astype(float)
    adj = np.empty_like(p)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        val = p[i] * m / (i + 1)
        prev = min(prev, val)
        adj[i] = prev
    out["p_bh"] = np.minimum(adj, 1.0)

    pass_idx = out.index[out[p_col] <= out["bh_threshold"]]
    if len(pass_idx) == 0:
        out["fdr_pass"] = False
    else:
        k = pass_idx.max()
        out["fdr_pass"] = out.index <= k
    return out


def perm_p(mask: np.ndarray, years: np.ndarray, pnl: np.ndarray, obs: float, n_perm=300, seed=123) -> float:
    rng = np.random.default_rng(seed)
    ge = 0
    valid = 0
    uniq = np.unique(years)

    for _ in range(n_perm):
        sh = mask.copy()
        for y in uniq:
            idx = np.where(years == y)[0]
            sh[idx] = rng.permutation(sh[idx])
        n_on = int(sh.sum())
        n_off = len(sh) - n_on
        if n_on < 40 or n_off < 40:
            continue
        up = float(pnl[sh].mean() - pnl[~sh].mean())
        valid += 1
        if up >= obs:
            ge += 1

    return float((ge + 1) / (valid + 1)) if valid else np.nan


def main() -> int:
    con = duckdb.connect(str(DB_PATH), read_only=True)

    rows = []
    for em in MODELS:
        for cb in CBS:
            if em == "E3" and cb != 1:
                continue
            for rr in RRS:
                q = f"""
                SELECT o.trading_day,o.pnl_r,o.entry_ts,
                       df_f.orb_US_EQUITY_OPEN_break_dir AS f_dir,
                       df_f.orb_US_EQUITY_OPEN_break_delay_min AS f_delay,
                       df_f.orb_US_EQUITY_OPEN_break_bar_continues AS f_cont,
                       df_l.orb_US_EQUITY_OPEN_break_dir AS l_dir,
                       df_l.orb_US_EQUITY_OPEN_break_ts  AS l_ts
                FROM orb_outcomes o
                JOIN daily_features df_f ON df_f.symbol=o.symbol AND df_f.trading_day=o.trading_day AND df_f.orb_minutes=o.orb_minutes
                JOIN daily_features df_l ON df_l.symbol='M6E' AND df_l.trading_day=o.trading_day AND df_l.orb_minutes=o.orb_minutes
                WHERE o.orb_minutes=5
                  AND o.symbol='MES' AND o.orb_label='US_EQUITY_OPEN'
                  AND o.entry_model='{em}' AND o.confirm_bars={cb} AND o.rr_target={rr}
                  AND o.pnl_r IS NOT NULL AND o.entry_ts IS NOT NULL
                """
                df = con.execute(q).fetchdf()
                if df.empty:
                    continue

                df["trading_day"] = pd.to_datetime(df["trading_day"])
                df["year"] = df["trading_day"].dt.year
                df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
                df["l_ts"] = pd.to_datetime(df["l_ts"], utc=True)

                valid = (
                    df["f_dir"].isin(["long", "short"]) &
                    df["l_dir"].isin(["long", "short"]) &
                    df["l_ts"].notna() & (df["l_ts"] <= df["entry_ts"])
                )
                df = df[valid].copy()
                if len(df) < 160:
                    continue

                same = (df["l_dir"] == df["f_dir"])
                conds = {
                    "same_dir": same,
                    "same_short": same & (df["f_dir"] == "short"),
                    "same_fast15": same & df["f_delay"].notna() & (df["f_delay"] <= 15),
                    "same_short_fast15": same & (df["f_dir"] == "short") & df["f_delay"].notna() & (df["f_delay"] <= 15),
                    "same_cont": same & (df["f_cont"] == True),
                }

                years = df["year"].values
                pnl = df["pnl_r"].values

                for cname, cmask in conds.items():
                    m = cmask.values.astype(bool)
                    n_on = int(m.sum())
                    n_off = len(m) - n_on
                    if n_on < 40 or n_off < 40:
                        continue
                    avg_on = float(pnl[m].mean())
                    avg_off = float(pnl[~m].mean())
                    uplift = avg_on - avg_off
                    p = perm_p(m.copy(), years, pnl, uplift, n_perm=N_PERM, seed=123)

                    te = df[df["year"] == 2025]
                    test_up = np.nan
                    if not te.empty:
                        mt = cmask.loc[te.index].values.astype(bool)
                        if mt.sum() >= 30 and (len(mt) - mt.sum()) >= 30:
                            test_up = float(te.loc[mt, "pnl_r"].mean() - te.loc[~mt, "pnl_r"].mean())

                    rows.append({
                        "entry_model": em,
                        "confirm_bars": cb,
                        "rr_target": rr,
                        "condition": cname,
                        "n": len(df),
                        "n_on": n_on,
                        "signals_per_year": n_on / max(1, df["year"].nunique()),
                        "avg_on": avg_on,
                        "avg_off": avg_off,
                        "uplift": uplift,
                        "perm_p": p,
                        "test2025_uplift": test_up,
                    })

    con.close()

    out_dir = ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_all = out_dir / "next_search_a0_neighbors_all.csv"
    p_pass = out_dir / "next_search_a0_neighbors_passed.csv"
    p_md = out_dir / "next_search_a0_neighbors_report.md"

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        p_md.write_text("# Next search A0 neighbors\n\nNo rows.", encoding="utf-8")
        print("No rows")
        return 0

    tst = all_df.dropna(subset=["perm_p"]).copy()
    tst = bh_fdr(tst, "perm_p", Q)

    passed = tst[
        (tst["fdr_pass"] == True)
        & (tst["avg_on"] >= 0.20)
        & (tst["uplift"] >= 0.25)
        & (tst["test2025_uplift"].fillna(-999) > 0)
    ].copy().sort_values(["avg_on", "uplift"], ascending=False)

    all_df.to_csv(p_all, index=False)
    passed.to_csv(p_pass, index=False)

    lines = [
        "# Next Search: A0 Neighbors (strict + FDR)",
        "",
        f"Total tests: {len(all_df)}",
        f"FDR-valid tests: {len(tst)}",
        f"Passed strict A-tier: {len(passed)}",
        "",
    ]

    if passed.empty:
        lines.append("No new strict pass in this family.")
    else:
        lines.append("## Passed")
        for r in passed.head(20).itertuples(index=False):
            lines.append(
                f"- {r.entry_model}/CB{r.confirm_bars}/RR{r.rr_target} {r.condition}: avg_on={r.avg_on:+.4f}, uplift={r.uplift:+.4f}, sig/yr={r.signals_per_year:.1f}, p={r.perm_p:.4f}, p_bh={r.p_bh:.4f}, test2025Δ={r.test2025_uplift:+.4f}"
            )

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_all}")
    print(f"Saved: {p_pass}")
    print(f"Saved: {p_md}")
    print("\nPassed:")
    print(passed.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
