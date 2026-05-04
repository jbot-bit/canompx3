#!/usr/bin/env python
"""RR selection analysis: SharpeDD criterion + multi-account deployment sizing."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ENTRY_MODELS, SKIP_ENTRY_MODELS

active_models = [em for em in ENTRY_MODELS if em not in SKIP_ENTRY_MODELS]
placeholders = ", ".join(f"'{em}'" for em in active_models)

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
try:
    df = con.execute(f"""
    SELECT vs.instrument, vs.orb_label, vs.filter_type, vs.entry_model,
        vs.rr_target, vs.sample_size as N, vs.win_rate as WR, vs.expectancy_r as ExpR,
        vs.sharpe_ratio as Sharpe, vs.max_drawdown_r as MaxDD_R, vs.trades_per_year as TPY
    FROM validated_setups vs
    WHERE vs.entry_model IN ({placeholders}) AND vs.status = 'active'
    """).fetchdf()
finally:
    con.close()

df["AbsDD"] = df["MaxDD_R"].abs()

families = df.groupby(["instrument", "orb_label", "filter_type", "entry_model"])
results = []
for (inst, sess, filt, em), group in families:
    group = group.copy()
    if len(group) < 2:
        row = group.iloc[0]
        results.append(
            {
                "instrument": inst,
                "orb_label": sess,
                "filter_type": filt,
                "entry_model": em,
                "n_rr": 1,
                "rr_maxSharpe": row["rr_target"],
                "rr_maxExpR": row["rr_target"],
                "rr_sharpeDD": row["rr_target"],
                "rr_calmar": row["rr_target"],
                "dd_at_sharpeDD": row["AbsDD"],
                "sharpe_at_sharpeDD": row["Sharpe"],
                "expr_at_sharpeDD": row["ExpR"],
                "n_at_sharpeDD": row["N"],
                "tpy_at_sharpeDD": row["TPY"],
            }
        )
        continue

    best_sharpe_idx = group["Sharpe"].idxmax()
    best_expr_idx = group["ExpR"].idxmax()
    best_s = group.loc[best_sharpe_idx, "Sharpe"]
    best_n = group.loc[best_sharpe_idx, "N"]

    # Jobson-Korkie: find all RRs NOT significantly worse than best Sharpe
    rho = 0.7
    candidates = []
    for idx, row in group.iterrows():
        n_eff = min(row["N"], best_n)
        se_sq = (2.0 / n_eff) * (1 - rho) + (1.0 / (2 * n_eff)) * (
            best_s**2 + row["Sharpe"] ** 2 - 2 * best_s * row["Sharpe"] * rho**2
        )
        if se_sq <= 0:
            se_sq = 2.0 / n_eff
        z = (best_s - row["Sharpe"]) / np.sqrt(se_sq) if se_sq > 0 else 0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        if p > 0.05:
            candidates.append(idx)

    if candidates:
        sharpe_dd_idx = group.loc[candidates].sort_values("AbsDD").index[0]
    else:
        sharpe_dd_idx = best_sharpe_idx

    group["calmar"] = group["ExpR"] * group["TPY"] / group["AbsDD"].clip(lower=0.1)
    best_calmar_idx = group["calmar"].idxmax()

    sel = group.loc[sharpe_dd_idx]
    results.append(
        {
            "instrument": inst,
            "orb_label": sess,
            "filter_type": filt,
            "entry_model": em,
            "n_rr": len(group),
            "rr_maxSharpe": group.loc[best_sharpe_idx, "rr_target"],
            "rr_maxExpR": group.loc[best_expr_idx, "rr_target"],
            "rr_sharpeDD": sel["rr_target"],
            "rr_calmar": group.loc[best_calmar_idx, "rr_target"],
            "dd_at_sharpeDD": sel["AbsDD"],
            "sharpe_at_sharpeDD": sel["Sharpe"],
            "expr_at_sharpeDD": sel["ExpR"],
            "n_at_sharpeDD": sel["N"],
            "tpy_at_sharpeDD": sel["TPY"],
        }
    )

rdf = pd.DataFrame(results)
multi = rdf[rdf["n_rr"] > 1]
n = len(multi)

print("=== CRITERION COMPARISON (families with 2+ RR) ===")
print(f"Total: {n} families")
print()
for c1, c2 in [
    ("rr_maxSharpe", "rr_sharpeDD"),
    ("rr_maxExpR", "rr_sharpeDD"),
    ("rr_calmar", "rr_sharpeDD"),
]:
    agree = (multi[c1] == multi[c2]).sum()
    print(f"  {c1:15s} == {c2:15s}: {agree}/{n} ({agree / n:.0%})")

diff = (multi["rr_maxSharpe"] != multi["rr_sharpeDD"]).sum()
print(f"\nSharpeDD picks different RR from MaxSharpe: {diff}/{n} ({diff / n:.0%})")
diff2 = (multi["rr_maxExpR"] != multi["rr_sharpeDD"]).sum()
print(f"SharpeDD picks different RR from MaxExpR: {diff2}/{n} ({diff2 / n:.0%})")

print("\n=== RR DISTRIBUTION UNDER SharpeDD ===")
for rr in sorted(rdf["rr_sharpeDD"].unique()):
    c = (rdf["rr_sharpeDD"] == rr).sum()
    print(f"  RR{rr:<4g}: {c:3d} families ({c / len(rdf):.0%})")

print("\n=== DEPLOYMENT: DD ceiling filters portfolio ===")
for dd_ceil in [5, 10, 15, 20, 25, 30, 50, 999]:
    t = rdf[rdf["dd_at_sharpeDD"] <= dd_ceil]
    label = "unconstrained" if dd_ceil == 999 else f"MaxDD<={dd_ceil}R"
    if len(t) > 0:
        print(
            f"  {label:20s}: {len(t):3d}/{len(rdf)} families, "
            f"med Sharpe={t['sharpe_at_sharpeDD'].median():.3f}, "
            f"med DD={t['dd_at_sharpeDD'].median():.1f}R"
        )

print("\n=== ACCOUNT SIZING EXAMPLES ===")
for acct_dd, acct_name in [(2000, "2K prop"), (6000, "6K/150K prop"), (20000, "own capital")]:
    print(f"\n--- {acct_name} (MaxDD=${acct_dd:,}) ---")
    for dd_ceil in [15, 20, 25, 999]:
        t = rdf[rdf["dd_at_sharpeDD"] <= dd_ceil] if dd_ceil < 999 else rdf
        if len(t) == 0:
            continue
        rpt = acct_dd / t["dd_at_sharpeDD"]
        label = "unconstrained" if dd_ceil == 999 else f"DD<={dd_ceil}R"
        avg_dollar_per_trade = (rpt * t["expr_at_sharpeDD"]).mean()
        total_tpy = t["tpy_at_sharpeDD"].sum()
        print(
            f"  {label:15s}: {len(t):3d} families, "
            f"risk/trade=${rpt.median():.0f}-${rpt.max():.0f}, "
            f"avg ${avg_dollar_per_trade:.2f}/trade, "
            f"~{total_tpy:.0f} trades/yr"
        )

# Show how the SAME family looks across account types
print("\n=== SAME STRATEGIES, DIFFERENT SIZING ===")
print("(Top 10 families by SharpeDD score)")
rdf["score"] = rdf["sharpe_at_sharpeDD"] * rdf["n_at_sharpeDD"] ** 0.5
top = rdf.sort_values("score", ascending=False).head(10)
print(f"{'Inst':4s} {'Session':18s} {'Filter':15s} {'EM':3s} RR   DD     | $2K r/t  | $6K r/t  | $20K r/t")
for _, r in top.iterrows():
    dd = r["dd_at_sharpeDD"]
    print(
        f"{r.instrument:4s} {r.orb_label:18s} {r.filter_type:15s} {r.entry_model:3s} "
        f"RR{r.rr_sharpeDD:<4g} {dd:5.1f}R "
        f"| ${2000 / dd:6.0f}  | ${6000 / dd:6.0f}  | ${20000 / dd:6.0f}"
    )
