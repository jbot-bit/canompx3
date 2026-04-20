"""OVNRNG allocator-routing diagnostic.

PR #61's cross-session replication surfaced an allocator-style pattern:
Q5 ovn/atr (high overnight vol) is BEST on LONDON_METALS, US_DATA_1000,
NYSE_CLOSE and WORST on NYSE_OPEN. This tests whether a router rule
('trade top-K sessions given today's ovn/atr bin') beats uniform trading.

Method:
1. For each trading_day, compute global ovn/atr quintile from all-MNQ
   atr_20 and overnight_range (look-ahead clean at ≥17:00 Brisbane).
2. Build conditional ExpR(session | bin) 8x5 table on IS.
3. Simulate policies: UNIFORM (all 8 sessions), ROUTER_TOP_K using
   bin-conditional ranking, CONTROL_TOP_K using bin-agnostic ranking.
4. Compare per-trade ExpR, N, annualized Sharpe.

Canonical: orb_outcomes and daily_features (triple-joined on
trading_day, symbol, orb_minutes).
Read-only. No production code touched.
"""

from __future__ import annotations

import sys

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
HOLDOUT = pd.Timestamp("2026-01-01")

# Lookahead-clean sessions (≥17:00 Brisbane start).
SESSIONS = [
    "LONDON_METALS", "EUROPE_FLOW", "US_DATA_830", "NYSE_OPEN",
    "US_DATA_1000", "COMEX_SETTLE", "CME_PRECLOSE", "NYSE_CLOSE",
]


def load_all() -> pd.DataFrame:
    q = """
    SELECT o.trading_day, o.orb_label, o.pnl_r,
           d.overnight_range, d.atr_20
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = 'MNQ'
      AND o.orb_label IN ('LONDON_METALS','EUROPE_FLOW','US_DATA_830',
                          'NYSE_OPEN','US_DATA_1000','COMEX_SETTLE',
                          'CME_PRECLOSE','NYSE_CLOSE')
      AND o.orb_minutes = 5
      AND o.entry_model = 'E2'
      AND o.rr_target = 1.5
      AND o.confirm_bars = 1
      AND o.pnl_r IS NOT NULL
      AND o.trading_day < ?
      AND d.overnight_range IS NOT NULL
      AND d.atr_20 IS NOT NULL
      AND d.atr_20 > 0
    ORDER BY o.trading_day, o.orb_label
    """
    df = DB.execute(q, [str(HOLDOUT.date())]).fetchdf()
    df["ovn_atr"] = df["overnight_range"] / df["atr_20"]
    return df


def main() -> None:
    print("=" * 80)
    print("OVNRNG ALLOCATOR-ROUTING DIAGNOSTIC")
    print(f"ran {pd.Timestamp.now('UTC')}")
    print("MNQ E2 RR=1.5 CB=1 orb_minutes=5, IS only (trading_day < 2026-01-01)")
    print("=" * 80)

    df = load_all()
    print(f"\nUniverse: n={len(df)} trades across {df['trading_day'].nunique()} trading days")
    print(f"Sessions: {sorted(df['orb_label'].unique())}")

    # Global ovn/atr quintile per trading_day.
    # Each trading_day has one overnight_range and one atr_20 (instrument-wide),
    # so ovn/atr is the same for all sessions on that day.
    daily = df.groupby("trading_day").agg(
        ovn_atr=("ovn_atr", "first"),
    ).reset_index()
    daily["bin"] = pd.qcut(daily["ovn_atr"], 5, labels=["Q1","Q2","Q3","Q4","Q5"],
                            duplicates="drop")
    print(f"\novn/atr quintile boundaries on daily series (n={len(daily)}):")
    q_boundaries = daily["ovn_atr"].quantile([0.2, 0.4, 0.6, 0.8]).values
    print(f"  Q1≤{q_boundaries[0]:.3f}  Q2≤{q_boundaries[1]:.3f}  "
          f"Q3≤{q_boundaries[2]:.3f}  Q4≤{q_boundaries[3]:.3f}  Q5>{q_boundaries[3]:.3f}")

    df = df.merge(daily[["trading_day", "bin"]], on="trading_day")

    # ============================================================
    # Step 1 — Conditional ExpR(session | bin) table
    # ============================================================
    print("\n" + "-" * 80)
    print("STEP 1 — CONDITIONAL ExpR(session | ovn/atr bin)")
    print("-" * 80)
    pivot_n = df.pivot_table(index="orb_label", columns="bin", values="pnl_r",
                              aggfunc="count", observed=True)
    pivot_expr = df.pivot_table(index="orb_label", columns="bin", values="pnl_r",
                                 aggfunc="mean", observed=True)
    pivot_n = pivot_n.reindex(SESSIONS)
    pivot_expr = pivot_expr.reindex(SESSIONS)

    print(f"\n  {'Session':15s}  " + " ".join(f"{b:>7s}" for b in ["Q1","Q2","Q3","Q4","Q5"]))
    print(f"  {'(ExpR | N)':15s}")
    for s in SESSIONS:
        row = f"  {s:15s}  "
        for b in ["Q1","Q2","Q3","Q4","Q5"]:
            try:
                n = int(pivot_n.loc[s, b]) if not np.isnan(pivot_n.loc[s, b]) else 0
                e = pivot_expr.loc[s, b]
            except KeyError:
                n = 0; e = float("nan")
            row += f"  {e:+.3f}" if not np.isnan(e) else "       —"
            row += f"({n:>3d})"
        print(row)

    # Best session per bin
    print("\n  Best session per bin (argmax ExpR):")
    print(f"  {'bin':4s} {'session':15s} {'ExpR':>8s} {'N':>5s}")
    best_per_bin = {}
    for b in ["Q1","Q2","Q3","Q4","Q5"]:
        col = pivot_expr[b]
        best_s = col.idxmax()
        best_per_bin[b] = best_s
        print(f"  {b:4s} {best_s:15s} {col.max():>+8.3f} {int(pivot_n.loc[best_s, b]):>5d}")

    # Bin-agnostic ranking (for control policy)
    bin_agnostic_expr = df.groupby("orb_label", observed=True)["pnl_r"].mean().sort_values(ascending=False)
    print("\n  Bin-agnostic ranking (control):")
    for s, e in bin_agnostic_expr.items():
        n = (df["orb_label"] == s).sum()
        print(f"    {s:15s} ExpR={e:+.4f}  N={n}")

    # ============================================================
    # Step 2 — Policy simulation
    # ============================================================
    print("\n" + "-" * 80)
    print("STEP 2 — POLICY SIMULATION")
    print("-" * 80)

    def _policy_stats(pnl: pd.Series, label: str) -> dict:
        n = len(pnl)
        mean = pnl.mean() if n else float("nan")
        std = pnl.std(ddof=1) if n > 1 else float("nan")
        sr = mean / std if std and std > 0 else float("nan")
        sr_ann = sr * np.sqrt(252) if not np.isnan(sr) else float("nan")
        t_res = stats.ttest_1samp(pnl.values, 0.0) if n > 1 else None
        t = float(t_res[0]) if t_res is not None else float("nan")
        p = float(t_res[1]) if t_res is not None else float("nan")
        print(f"  {label:40s}  n={n:>5d}  ExpR={mean:+.4f}  "
              f"SR_ann={sr_ann:+.3f}  t={t:+.2f} p={p:.4f}")
        return {"label": label, "n": n, "mean": float(mean), "sr_ann": float(sr_ann),
                "t": t, "p": p}

    results = []
    results.append(_policy_stats(df["pnl_r"], "UNIFORM (all 8 sessions, every trade)"))

    # Router policies: per-day select top-K sessions by bin-conditional ExpR
    for k in [1, 2, 3]:
        selected = []
        for b in ["Q1","Q2","Q3","Q4","Q5"]:
            topk = pivot_expr[b].sort_values(ascending=False).head(k).index.tolist()
            for s in topk:
                selected.append((b, s))
        mask = pd.Series(False, index=df.index)
        for b, s in selected:
            mask |= (df["bin"] == b) & (df["orb_label"] == s)
        results.append(_policy_stats(df.loc[mask, "pnl_r"],
                                      f"ROUTER top-{k} (bin-conditional)"))

    # Control: top-K bin-agnostic (same K but fixed session set, no ovn/atr use)
    for k in [1, 2, 3]:
        topk_agn = bin_agnostic_expr.head(k).index.tolist()
        mask = df["orb_label"].isin(topk_agn)
        results.append(_policy_stats(df.loc[mask, "pnl_r"],
                                      f"CONTROL top-{k} (bin-agnostic — fixed set)"))

    # ============================================================
    # Step 3 — Incremental value of bin awareness
    # ============================================================
    print("\n" + "-" * 80)
    print("STEP 3 — BIN-AWARENESS INCREMENTAL VALUE")
    print("-" * 80)
    # For each K, compute (router bin-aware ExpR) - (control bin-agnostic ExpR)
    for k in [1, 2, 3]:
        rk = next(r for r in results if r["label"].startswith(f"ROUTER top-{k}"))
        ck = next(r for r in results if r["label"].startswith(f"CONTROL top-{k}"))
        delta_expr = rk["mean"] - ck["mean"]
        delta_sr = rk["sr_ann"] - ck["sr_ann"]
        print(f"  K={k}: router − control ΔExpR={delta_expr:+.4f}  ΔSR_ann={delta_sr:+.3f}  "
              f"(router n={rk['n']}, control n={ck['n']})")

    # ============================================================
    # Step 4 — Walk-forward honest validation (in-sample bias check)
    # ============================================================
    print("\n" + "-" * 80)
    print("STEP 4 — WALK-FORWARD: train on first half, apply to second half")
    print("-" * 80)
    df_sorted = df.sort_values("trading_day").reset_index(drop=True)
    mid = len(df_sorted) // 2
    mid_day = df_sorted.iloc[mid].trading_day
    train = df_sorted.iloc[:mid].copy()
    test = df_sorted.iloc[mid:].copy()
    print(f"  Train: {train['trading_day'].min()} → {train['trading_day'].max()} (n={len(train)})")
    print(f"  Test:  {test['trading_day'].min()} → {test['trading_day'].max()} (n={len(test)})")

    # Re-compute bins on train-only distribution (no peek)
    train_daily = train.groupby("trading_day").agg(ovn_atr=("ovn_atr","first")).reset_index()
    tr_bounds = train_daily["ovn_atr"].quantile([0.2, 0.4, 0.6, 0.8]).values
    def _bin_by_bounds(x: float) -> str:
        if x <= tr_bounds[0]: return "Q1"
        if x <= tr_bounds[1]: return "Q2"
        if x <= tr_bounds[2]: return "Q3"
        if x <= tr_bounds[3]: return "Q4"
        return "Q5"
    train["bin_tr"] = train["ovn_atr"].apply(_bin_by_bounds)
    test["bin_tr"] = test["ovn_atr"].apply(_bin_by_bounds)

    # Train the router: best session per bin on train-only
    tr_pivot = train.pivot_table(index="orb_label", columns="bin_tr", values="pnl_r",
                                  aggfunc="mean", observed=True)
    best_per_bin_tr = {b: tr_pivot[b].idxmax() for b in ["Q1","Q2","Q3","Q4","Q5"]
                       if b in tr_pivot.columns}
    tr_bin_agnostic = train.groupby("orb_label", observed=True)["pnl_r"].mean().sort_values(ascending=False)

    print(f"\n  Train-derived best session per bin (NO test peek):")
    for b, s in best_per_bin_tr.items():
        print(f"    {b}: {s} (train ExpR={tr_pivot.loc[s,b]:+.4f})")

    # Apply to test
    print(f"\n  Test-period policy simulation (out-of-train):")
    test_uniform = test["pnl_r"]
    test_router_mask = pd.Series(False, index=test.index)
    for b, s in best_per_bin_tr.items():
        test_router_mask |= (test["bin_tr"] == b) & (test["orb_label"] == s)
    test_router = test.loc[test_router_mask, "pnl_r"]
    test_control = test.loc[test["orb_label"] == tr_bin_agnostic.index[0], "pnl_r"]

    def _sr(pnl: pd.Series) -> tuple[float, float]:
        if len(pnl) < 2: return float("nan"), float("nan")
        m = pnl.mean(); s = pnl.std(ddof=1)
        sr = m / s if s > 0 else float("nan")
        return float(m), float(sr * np.sqrt(252)) if not np.isnan(sr) else float("nan")

    for label, pnl in [("UNIFORM (test)", test_uniform),
                       ("ROUTER top-1 (test, train-derived map)", test_router),
                       ("CONTROL top-1 (test, train-derived best-session)", test_control)]:
        m, sr = _sr(pnl)
        print(f"  {label:45s}  n={len(pnl):>5d}  ExpR={m:+.4f}  SR_ann={sr:+.3f}")

    # walk-forward summary
    m_r, sr_r = _sr(test_router)
    m_c, sr_c = _sr(test_control)
    delta_sr_wf = sr_r - sr_c
    print(f"\n  WALK-FORWARD ΔSR_ann (router − control) = {delta_sr_wf:+.3f}")
    if delta_sr_wf >= 0.30:
        wf_verdict = "ROUTER_HOLDS_OOS"
    elif delta_sr_wf >= 0.10:
        wf_verdict = "ROUTER_MARGINAL_OOS"
    elif delta_sr_wf >= -0.10:
        wf_verdict = "ROUTER_FLAT_OOS"
    else:
        wf_verdict = "ROUTER_FAILS_OOS"
    print(f"  Walk-forward verdict: {wf_verdict}")

    # ============================================================
    # Verdict
    # ============================================================
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    best_router = max((r for r in results if r["label"].startswith("ROUTER")),
                      key=lambda r: r["sr_ann"])
    uniform = next(r for r in results if r["label"].startswith("UNIFORM"))
    best_control = max((r for r in results if r["label"].startswith("CONTROL")),
                       key=lambda r: r["sr_ann"])
    print(f"  Best router: {best_router['label']}  SR_ann={best_router['sr_ann']:+.3f}")
    print(f"  Uniform    : SR_ann={uniform['sr_ann']:+.3f}")
    print(f"  Best control (bin-agnostic): {best_control['label']}  SR_ann={best_control['sr_ann']:+.3f}")
    print(f"\n  Router vs Uniform ΔSR_ann: {best_router['sr_ann']-uniform['sr_ann']:+.3f}")
    print(f"  Router vs best Control (isolates bin-awareness): {best_router['sr_ann']-best_control['sr_ann']:+.3f}")

    if best_router["sr_ann"] - best_control["sr_ann"] >= 0.30:
        v = "ROUTER_WORTHWHILE — bin awareness adds meaningful SR"
    elif best_router["sr_ann"] - best_control["sr_ann"] >= 0.10:
        v = "MARGINAL — bin awareness adds some SR but needs pre-reg verification"
    else:
        v = "NO_SIGNAL — bin awareness does not add SR beyond session concentration"
    print(f"\n  CLASSIFICATION: {v}")


if __name__ == "__main__":
    main()
