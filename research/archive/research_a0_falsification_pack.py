#!/usr/bin/env python3
"""A0 falsification/placebo pack.

Tests whether A0 looks structural or like data-mined coincidence.
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "gold.db"


def perm_p(mask: np.ndarray, years: np.ndarray, pnl: np.ndarray, obs: float, n_perm=500, seed=123) -> float:
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
    if valid == 0:
        return np.nan
    return float((ge + 1) / (valid + 1))


def score(df: pd.DataFrame, mask: pd.Series) -> dict:
    on = df.loc[mask, "pnl_r"]
    off = df.loc[~mask, "pnl_r"]
    if len(on) < 20 or len(off) < 20:
        return {"n_on": len(on), "avg_on": np.nan, "avg_off": np.nan, "uplift": np.nan}
    return {
        "n_on": int(len(on)),
        "avg_on": float(on.mean()),
        "avg_off": float(off.mean()),
        "uplift": float(on.mean() - off.mean()),
    }


def main() -> int:
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # follower slice
    qf = """
    SELECT o.trading_day,o.pnl_r,o.entry_ts,
           df_f.orb_US_EQUITY_OPEN_break_dir AS f_dir
    FROM orb_outcomes o
    JOIN daily_features df_f ON df_f.symbol=o.symbol AND df_f.trading_day=o.trading_day AND df_f.orb_minutes=o.orb_minutes
    WHERE o.orb_minutes=5
      AND o.symbol='MES' AND o.orb_label='US_EQUITY_OPEN'
      AND o.entry_model='E0' AND o.confirm_bars=1 AND o.rr_target=3.0
      AND o.pnl_r IS NOT NULL AND o.entry_ts IS NOT NULL
    """
    f = con.execute(qf).fetchdf()

    ql = """
    SELECT trading_day,
           orb_US_EQUITY_OPEN_break_dir AS l_useo_dir,
           orb_US_EQUITY_OPEN_break_ts  AS l_useo_ts,
           orb_2300_break_dir AS l_2300_dir,
           orb_2300_break_ts  AS l_2300_ts,
           orb_0030_break_dir AS l_0030_dir,
           orb_0030_break_ts  AS l_0030_ts
    FROM daily_features
    WHERE symbol='M6E' AND orb_minutes=5
    """
    l = con.execute(ql).fetchdf()
    con.close()

    if f.empty or l.empty:
        print("No data.")
        return 0

    f["trading_day"] = pd.to_datetime(f["trading_day"])
    f["entry_ts"] = pd.to_datetime(f["entry_ts"], utc=True)
    f["year"] = f["trading_day"].dt.year

    l["trading_day"] = pd.to_datetime(l["trading_day"])
    for c in ["l_useo_ts", "l_2300_ts", "l_0030_ts"]:
        l[c] = pd.to_datetime(l[c], utc=True)

    df = f.merge(l, on="trading_day", how="left")

    valid_f = df["f_dir"].isin(["long", "short"])

    # Real A0
    m_real = (
        valid_f
        & df["l_useo_dir"].isin(["long", "short"])
        & (df["l_useo_dir"] == df["f_dir"])
        & df["l_useo_ts"].notna()
        & (df["l_useo_ts"] <= df["entry_ts"])
    )

    # Placebo 1: wrong leader session 2300
    m_wrong_2300 = (
        valid_f
        & df["l_2300_dir"].isin(["long", "short"])
        & (df["l_2300_dir"] == df["f_dir"])
        & df["l_2300_ts"].notna()
        & (df["l_2300_ts"] <= df["entry_ts"])
    )

    # Placebo 2: wrong leader session 0030
    m_wrong_0030 = (
        valid_f
        & df["l_0030_dir"].isin(["long", "short"])
        & (df["l_0030_dir"] == df["f_dir"])
        & df["l_0030_ts"].notna()
        & (df["l_0030_ts"] <= df["entry_ts"])
    )

    # Placebo 3: previous-day leader USEO (day-shift falsification)
    l_shift = l[["trading_day", "l_useo_dir", "l_useo_ts"]].sort_values("trading_day").copy()
    l_shift["trading_day"] = l_shift["trading_day"].shift(1)
    l_shift = l_shift.rename(columns={"l_useo_dir": "l_prev_dir", "l_useo_ts": "l_prev_ts"})
    d2 = df.merge(l_shift, on="trading_day", how="left")
    m_prevday = (
        valid_f
        & d2["l_prev_dir"].isin(["long", "short"])
        & (d2["l_prev_dir"] == d2["f_dir"])
        & d2["l_prev_ts"].notna()
        & (d2["l_prev_ts"] <= d2["entry_ts"])
    )

    # Placebo 4: impossible-causal (leader after entry)
    m_post = (
        valid_f
        & df["l_useo_dir"].isin(["long", "short"])
        & (df["l_useo_dir"] == df["f_dir"])
        & df["l_useo_ts"].notna()
        & (df["l_useo_ts"] > df["entry_ts"])
    )

    # Control: opposite dir same session
    m_opp = (
        valid_f
        & df["l_useo_dir"].isin(["long", "short"])
        & (df["l_useo_dir"] != df["f_dir"])
        & df["l_useo_ts"].notna()
        & (df["l_useo_ts"] <= df["entry_ts"])
    )

    tests = {
        "real_A0_same_session": m_real,
        "placebo_wrong_session_2300": m_wrong_2300,
        "placebo_wrong_session_0030": m_wrong_0030,
        "placebo_prevday_session": m_prevday,
        "placebo_leader_after_entry": m_post,
        "control_opposite_direction": m_opp,
    }

    rows = []
    for name, m in tests.items():
        sc = score(df if name != "placebo_prevday_session" else d2, m)
        rows.append({"test": name, **sc})

    out = pd.DataFrame(rows)

    # permutation p for real uplift
    real_row = out[out["test"] == "real_A0_same_session"].iloc[0]
    pval = np.nan
    if pd.notna(real_row["uplift"]):
        pval = perm_p(m_real.values.astype(bool), df["year"].values, df["pnl_r"].values, float(real_row["uplift"]), n_perm=N_PERM, seed=123)

    out_dir = ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_csv = out_dir / "a0_falsification_pack.csv"
    p_md = out_dir / "a0_falsification_pack.md"

    out.to_csv(p_csv, index=False)

    lines = [
        "# A0 Falsification Pack",
        "",
        "Goal: compare real signal against logically wrong/placebo constructions.",
        "",
    ]

    for r in out.itertuples(index=False):
        lines.append(f"- {r.test}: N_on={r.n_on}, avg_on={r.avg_on:+.4f}, avg_off={r.avg_off:+.4f}, uplift={r.uplift:+.4f}")

    lines.append("")
    lines.append(f"Permutation p-value for real_A0_same_session uplift: {pval:.6f}" if pd.notna(pval) else "Permutation p-value unavailable")

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_csv}")
    print(f"Saved: {p_md}")
    print(out.to_string(index=False))
    print("perm_p_real", pval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
