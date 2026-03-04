#!/usr/bin/env python3
"""Surprise angle: same-instrument session phase-shift edges with FDR control.

What is new here:
- Not cross-asset lead-lag.
- Focuses on intra-instrument handover logic (phase transitions).
- Uses BH-FDR to control multiple comparisons.

Outputs:
- research/output/surprise_phase_shift_fdr_all.csv
- research/output/surprise_phase_shift_fdr_passed.csv
- research/output/surprise_phase_shift_fdr_report.md
"""

from __future__ import annotations

from pathlib import Path
import re
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "gold.db"

SYMBOLS = ["MES", "MNQ", "M2K", "M6E"]
STRATEGY_SLICES = [
    ("E0", 1, 2.5),
    ("E1", 2, 2.5),
]
TRANSITIONS = [
    ("0900", "1000"),
    ("1000", "US_DATA_OPEN"),
    ("US_DATA_OPEN", "US_EQUITY_OPEN"),
    ("US_EQUITY_OPEN", "US_POST_EQUITY"),
    ("1800", "2300"),
    ("2300", "0030"),
]

MIN_ON = 80
MIN_OFF = 80
N_PERM = 200
FDR_Q = 0.10


def safe_label(lbl: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_]+", lbl) is None:
        raise ValueError(f"Unsafe label: {lbl}")
    return lbl


def bh_fdr(df: pd.DataFrame, p_col: str, q: float) -> pd.DataFrame:
    out = df.copy().sort_values(p_col).reset_index(drop=True)
    m = len(out)
    if m == 0:
        out["fdr_pass"] = False
        out["p_bh"] = np.nan
        return out

    out["rank"] = np.arange(1, m + 1)
    out["bh_threshold"] = (out["rank"] / m) * q

    # BH adjusted p-values (q-values)
    p = out[p_col].values.astype(float)
    adj = np.empty_like(p)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        val = p[i] * m / (i + 1)
        prev = min(prev, val)
        adj[i] = prev
    out["p_bh"] = np.minimum(adj, 1.0)

    # pass set
    pass_idx = out.index[out[p_col] <= out["bh_threshold"]]
    if len(pass_idx) == 0:
        out["fdr_pass"] = False
    else:
        k = pass_idx.max()
        out["fdr_pass"] = out.index <= k
    return out


def perm_pvalue(mask: np.ndarray, years: np.ndarray, pnl: np.ndarray, obs: float, n_perm: int = 200, seed: int = 123) -> float:
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
        if n_on < MIN_ON or n_off < MIN_OFF:
            continue

        up = float(pnl[sh].mean() - pnl[~sh].mean())
        valid += 1
        if up >= obs:
            ge += 1

    if valid == 0:
        return np.nan
    return float((ge + 1) / (valid + 1))


def main() -> int:
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # discover available sessions per symbol
    avail = con.execute(
        """
        SELECT symbol, orb_label, COUNT(*) n
        FROM orb_outcomes
        WHERE orb_minutes=5
          AND symbol IN ('MES','MNQ','M2K','M6E')
        GROUP BY 1,2
        HAVING COUNT(*) >= 500
        """
    ).fetchdf()
    available = {(r.symbol, r.orb_label) for r in avail.itertuples(index=False)}

    rows = []

    for sym in SYMBOLS:
        for leader_sess, follower_sess in TRANSITIONS:
            if (sym, leader_sess) not in available or (sym, follower_sess) not in available:
                continue

            ls = safe_label(leader_sess)
            fs = safe_label(follower_sess)

            for em, cb, rr in STRATEGY_SLICES:
                q = f"""
                SELECT
                  o.trading_day,
                  o.pnl_r,
                  o.entry_ts,
                  df_f.orb_{fs}_break_dir AS f_dir,
                  df_l.orb_{ls}_break_dir AS l_dir,
                  df_l.orb_{ls}_break_ts  AS l_ts,
                  df_l.orb_{ls}_break_delay_min AS l_delay,
                  df_l.orb_{ls}_break_bar_continues AS l_cont,
                  df_l.orb_{ls}_size AS l_size,
                  df_l.atr_20 AS l_atr
                FROM orb_outcomes o
                JOIN daily_features df_f
                  ON df_f.symbol=o.symbol
                 AND df_f.trading_day=o.trading_day
                 AND df_f.orb_minutes=o.orb_minutes
                JOIN daily_features df_l
                  ON df_l.symbol=o.symbol
                 AND df_l.trading_day=o.trading_day
                 AND df_l.orb_minutes=o.orb_minutes
                WHERE o.orb_minutes=5
                  AND o.symbol='{sym}'
                  AND o.orb_label='{follower_sess}'
                  AND o.entry_model='{em}'
                  AND o.confirm_bars={cb}
                  AND o.rr_target={rr}
                  AND o.pnl_r IS NOT NULL
                  AND o.entry_ts IS NOT NULL
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
                    df["l_ts"].notna() &
                    (df["l_ts"] <= df["entry_ts"])
                )
                df = df[valid].copy()
                if len(df) < (MIN_ON + MIN_OFF):
                    continue

                same = (df["l_dir"] == df["f_dir"])
                opp = ~same
                l_size_atr = np.where((df["l_atr"].notna()) & (df["l_atr"] > 0), df["l_size"] / df["l_atr"], np.nan)
                l_size_atr_s = pd.Series(l_size_atr, index=df.index)
                q70 = l_size_atr_s.quantile(0.70)

                conditions = {
                    "same": same,
                    "same_fast30": same & df["l_delay"].notna() & (df["l_delay"] <= 30),
                    "same_cont": same & (df["l_cont"] == True),
                    "opp": opp,
                    "opp_late_fail": opp & df["l_delay"].notna() & (df["l_delay"] >= 60) & (df["l_cont"] == False),
                    "opp_stretch": opp & l_size_atr_s.notna() & (l_size_atr_s >= q70) & df["l_delay"].notna() & (df["l_delay"] <= 30),
                }

                years = df["year"].values
                pnl = df["pnl_r"].values

                for cname, cmask in conditions.items():
                    if isinstance(cmask, pd.Series):
                        m = np.asarray(cmask.fillna(False).astype(bool).values, dtype=bool)
                    else:
                        m = np.asarray(cmask, dtype=bool)
                    n_on = int(m.sum())
                    n_off = len(m) - n_on
                    if n_on < MIN_ON or n_off < MIN_OFF:
                        continue

                    on = pnl[m]
                    off = pnl[~m]
                    avg_on = float(on.mean())
                    avg_off = float(off.mean())
                    uplift = avg_on - avg_off

                    p = perm_pvalue(m.copy(), years, pnl, uplift, n_perm=N_PERM, seed=123)

                    # quick OOS 2025
                    te = df[df["year"] == 2025]
                    test_uplift = np.nan
                    if not te.empty and isinstance(cmask, pd.Series):
                        mt = np.asarray(cmask.loc[te.index].fillna(False).astype(bool).values, dtype=bool)
                        if mt.sum() >= 40 and (len(mt) - mt.sum()) >= 40:
                            test_uplift = float(te.loc[mt, "pnl_r"].mean() - te.loc[~mt, "pnl_r"].mean())

                    years_cov = max(1, int(df["year"].nunique()))

                    rows.append(
                        {
                            "symbol": sym,
                            "leader_session": leader_sess,
                            "follower_session": follower_sess,
                            "entry_model": em,
                            "confirm_bars": cb,
                            "rr_target": rr,
                            "condition": cname,
                            "n": len(df),
                            "n_on": n_on,
                            "signals_per_year": n_on / years_cov,
                            "avg_on": avg_on,
                            "avg_off": avg_off,
                            "uplift": uplift,
                            "perm_p": p,
                            "test2025_uplift": test_uplift,
                        }
                    )

    con.close()

    out_dir = ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_all = out_dir / "surprise_phase_shift_fdr_all.csv"
    p_pass = out_dir / "surprise_phase_shift_fdr_passed.csv"
    p_md = out_dir / "surprise_phase_shift_fdr_report.md"

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        p_md.write_text("# Surprise phase-shift FDR\n\nNo tests met minimum sample requirements.", encoding="utf-8")
        print("No tests met minimum sample requirements.")
        return 0

    # Drop NaN p rows then FDR
    tst = all_df.dropna(subset=["perm_p"]).copy()
    tst = bh_fdr(tst, "perm_p", FDR_Q)

    # strict gain-first shortlist after FDR
    passed = tst[
        (tst["fdr_pass"] == True)
        & (tst["avg_on"] >= 0.20)
        & (tst["uplift"] >= 0.20)
        & (tst["test2025_uplift"].fillna(-999) > 0)
    ].copy().sort_values(["avg_on", "uplift"], ascending=False)

    all_df.to_csv(p_all, index=False)
    passed.to_csv(p_pass, index=False)

    lines = [
        "# Surprise Phase-Shift FDR Report",
        "",
        f"Total tests: {len(all_df)}",
        f"Tests with valid permutation p: {len(tst)}",
        f"FDR q: {FDR_Q}",
        f"FDR+gain passed: {len(passed)}",
        "",
        "## Top FDR+gain passes",
    ]

    if passed.empty:
        lines.append("- None in this run.")
    else:
        for r in passed.head(20).itertuples(index=False):
            lines.append(
                f"- {r.symbol} {r.leader_session}->{r.follower_session} {r.entry_model}/CB{r.confirm_bars}/RR{r.rr_target} {r.condition}: avg_on={r.avg_on:+.4f}, uplift={r.uplift:+.4f}, sig/yr={r.signals_per_year:.1f}, p={r.perm_p:.4f}, p_bh={r.p_bh:.4f}, test2025Δ={r.test2025_uplift:+.4f}"
            )

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_all}")
    print(f"Saved: {p_pass}")
    print(f"Saved: {p_md}")
    print("\nTop passed:")
    print(passed.head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
