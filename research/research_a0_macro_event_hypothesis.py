#!/usr/bin/env python3
"""A0 macro-event hypothesis (strict, pre-registered).

Hypothesis family (fixed):
- A0 degrades on macro-event/shock days.
- Excluding event/shock regimes should improve A0 quality.

Data constraints:
- Explicit event tag available: is_nfp_day
- Shock proxy (pre-entry): rel_vol_US_DATA_OPEN (known before US_EQUITY_OPEN)

A0 base:
- M6E_US_EQUITY_OPEN -> MES_US_EQUITY_OPEN
- E0 / CB1 / RR3.0
- same-direction + no-lookahead
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "gold.db"
N_PERM = 500
SEED = 123


def perm_p(mask: np.ndarray, years: np.ndarray, pnl: np.ndarray, obs: float, n_perm: int, seed: int) -> float:
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


def main() -> int:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    q = """
    SELECT o.trading_day,o.pnl_r,o.entry_ts,
           df_f.orb_US_EQUITY_OPEN_break_dir AS f_dir,
           df_l.orb_US_EQUITY_OPEN_break_dir AS l_dir,
           df_l.orb_US_EQUITY_OPEN_break_ts  AS l_ts,
           df_f.is_nfp_day,
           df_f.rel_vol_US_DATA_OPEN
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
        print("No rows")
        return 0

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["l_ts"] = pd.to_datetime(df["l_ts"], utc=True)

    base = (
        df["f_dir"].isin(["long", "short"]) &
        df["l_dir"].isin(["long", "short"]) &
        (df["f_dir"] == df["l_dir"]) &
        df["l_ts"].notna() &
        (df["l_ts"] <= df["entry_ts"])
    )
    d = df[base].copy()
    if d.empty:
        print("No base rows")
        return 0

    shock_q90 = d["rel_vol_US_DATA_OPEN"].quantile(0.90)

    variants = {
        "base": pd.Series(True, index=d.index),
        "non_nfp": ~(d["is_nfp_day"] == True),
        "non_shock_q90": d["rel_vol_US_DATA_OPEN"].notna() & (d["rel_vol_US_DATA_OPEN"] <= shock_q90),
        "non_nfp_and_non_shock": (~(d["is_nfp_day"] == True)) & d["rel_vol_US_DATA_OPEN"].notna() & (d["rel_vol_US_DATA_OPEN"] <= shock_q90),
    }

    base_avg = float(d["pnl_r"].mean())
    years = d["year"].values
    pnl = d["pnl_r"].values

    rows = []
    for name, mser in variants.items():
        m = mser.values.astype(bool)
        on = d.loc[m, "pnl_r"]
        off = d.loc[~m, "pnl_r"]

        if name != "base" and (len(on) < 40 or len(off) < 40):
            continue

        avg_on = float(on.mean())
        avg_off = float(off.mean()) if len(off) else np.nan
        uplift = avg_on - avg_off if len(off) else np.nan
        delta_vs_base = avg_on - base_avg

        pval = np.nan
        if name != "base":
            pval = perm_p(m.copy(), years, pnl, uplift, n_perm=N_PERM, seed=SEED)

        # OOS delta vs base (2025)
        te = d[d["year"] == 2025]
        test_delta = np.nan
        if not te.empty:
            mt = mser.loc[te.index].values.astype(bool)
            te_on = te.loc[mt, "pnl_r"]
            if len(te_on) >= 20:
                test_delta = float(te_on.mean() - te["pnl_r"].mean())

        # verdict
        verdict = "BASELINE" if name == "base" else "KILL"
        if name != "base":
            if (avg_on >= 0.25 and delta_vs_base >= 0.08 and pd.notna(test_delta) and test_delta > 0 and pd.notna(pval) and pval <= 0.05 and len(on) >= 100):
                verdict = "PROMOTE"
            elif (avg_on > base_avg and pd.notna(test_delta) and test_delta > 0 and len(on) >= 60):
                verdict = "WATCH"

        rows.append(
            {
                "variant": name,
                "n_on": int(len(on)),
                "signals_per_year": float(len(on) / max(1, d["year"].nunique())),
                "avg_on": avg_on,
                "avg_off": avg_off,
                "uplift_on_off": uplift,
                "delta_vs_base": delta_vs_base,
                "test2025_delta_vs_base": test_delta,
                "perm_p": pval,
                "verdict": verdict,
            }
        )

    out = pd.DataFrame(rows).sort_values(["avg_on", "delta_vs_base"], ascending=False)

    out_dir = ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_csv = out_dir / "a0_macro_event_hypothesis.csv"
    p_md = out_dir / "a0_macro_event_hypothesis.md"

    out.to_csv(p_csv, index=False)

    lines = [
        "# A0 Macro-Event Hypothesis",
        "",
        "Hypothesis: A0 quality improves when excluding macro-event/shock regimes.",
        f"Base avgR: {base_avg:+.4f}",
        f"Shock threshold q90 rel_vol_US_DATA_OPEN: {shock_q90:.4f}",
        "",
    ]

    for r in out.itertuples(index=False):
        lines.append(
            f"- {r.variant} => {r.verdict}: N={r.n_on}, sig/yr={r.signals_per_year:.1f}, avg_on={r.avg_on:+.4f}, Δbase={r.delta_vs_base:+.4f}, test2025Δ={r.test2025_delta_vs_base:+.4f}, p={r.perm_p if pd.notna(r.perm_p) else 'NA'}"
        )

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_csv}")
    print(f"Saved: {p_md}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
