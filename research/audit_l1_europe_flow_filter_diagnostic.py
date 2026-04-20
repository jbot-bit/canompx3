"""L1 MNQ_EUROPE_FLOW ORB_G5 filter diagnostic.

Question: can an ATR-normalized replacement for ORB_G5 (absolute-points
threshold that has drifted from 57%→100% fire rate due to MNQ price
inflation) restore pre-2022 selectivity while retaining the historical
+0.021R IS Welch lift on L1?

Mechanism: ORB_G5 fires when orb_size > 5 points. MNQ ~3x price inflation
2019→2026 turned a 5-point filter from meaningful (57% fire 2019) into
pass-through (100% fire 2022+).

Candidate: orb_size / atr_20 ≥ THRESHOLD. ATR-20 scales with MNQ price,
so the ratio is stationary era-over-era.

Canonical truth only: orb_outcomes and daily_features (triple-joined).
Read-only. No production code touched.
"""

from __future__ import annotations

import sys

import duckdb
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal
from trading_app.eligibility.builder import parse_strategy_id

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
HOLDOUT = pd.Timestamp("2026-01-01")


def _welch(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a.values, b.values, equal_var=False)
    return float(t), float(p)


def main() -> None:
    spec = parse_strategy_id("MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5")
    print("=" * 80)
    print(f"L1 EUROPE_FLOW DIAGNOSTIC — orb_minutes={spec['orb_minutes']}  filter={spec['filter_type']}")
    print("=" * 80)

    q = """
    SELECT o.trading_day, o.pnl_r, o.entry_price,
           d.orb_EUROPE_FLOW_size,
           d.atr_20
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = ? AND o.orb_label = ? AND o.orb_minutes = ?
      AND o.entry_model = ? AND o.rr_target = ? AND o.confirm_bars = ?
      AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day
    """
    df = DB.execute(q, [
        spec["instrument"], spec["orb_label"], spec["orb_minutes"],
        spec["entry_model"], spec["rr_target"], spec["confirm_bars"],
    ]).fetchdf()
    df["year"] = pd.to_datetime(df["trading_day"]).dt.year
    df["orb_size"] = df["orb_EUROPE_FLOW_size"]  # alias for local use
    df["orb_size_over_atr"] = df["orb_size"] / df["atr_20"]
    print(f"\nL1 universe: n={len(df)}, {df.trading_day.min()} → {df.trading_day.max()}")

    # Compute legacy ORB_G5 signal
    sig_legacy = filter_signal(df, "ORB_G5", "EUROPE_FLOW")
    df["fire_legacy"] = sig_legacy

    # Step 1: per-year orb_size and orb/atr distribution
    print("\n" + "-" * 80)
    print("STEP 1 — ORB SIZE DISTRIBUTION BY YEAR (shows scale drift)")
    print("-" * 80)
    print(f"  {'Year':5s} {'N':>4s} {'orb μ':>7s} {'orb p50':>8s} {'orb/atr μ':>10s} "
          f"{'orb/atr p50':>12s} {'ORB_G5 fire':>12s}")
    for y, grp in df.groupby("year"):
        print(f"  {y:5d} {len(grp):>4d} {grp['orb_size'].mean():>7.1f} "
              f"{grp['orb_size'].median():>8.1f} "
              f"{grp['orb_size_over_atr'].mean():>10.3f} "
              f"{grp['orb_size_over_atr'].median():>12.3f} "
              f"{grp['fire_legacy'].mean()*100:>11.1f}%")

    # Step 2: candidate thresholds on orb_size_over_atr
    print("\n" + "-" * 80)
    print("STEP 2 — CANDIDATE ATR-NORMALIZED THRESHOLDS (per-year fire %)")
    print("-" * 80)
    candidates = [0.035, 0.045, 0.050, 0.055, 0.065, 0.080, 0.10]
    print(f"  {'Year':5s} {'N':>4s} {'legacy ORB_G5':>13s} " + " ".join(
        f"{'≥'+str(c):>6s}" for c in candidates))
    for y, grp in df.groupby("year"):
        fires = [grp.eval(f"orb_size_over_atr >= {c}").mean() * 100 for c in candidates]
        legacy = grp["fire_legacy"].mean() * 100
        print(f"  {y:5d} {len(grp):>4d} {legacy:>12.1f}% " +
              " ".join(f"{fire:>5.1f}%" for fire in fires))

    # Step 3: for each candidate, full-sample and per-era Welch fire-vs-non-fire
    print("\n" + "-" * 80)
    print("STEP 3 — IS WELCH FIRE-VS-NON-FIRE for each candidate (IS n={})".format(
        len(df[df.trading_day < HOLDOUT])))
    print("-" * 80)
    df_is = df[df.trading_day < HOLDOUT].copy()
    print(f"  {'threshold':12s} {'fire%':>7s} {'ExpR_fire':>10s} {'ExpR_nonf':>10s} "
          f"{'Δ':>7s} {'Welch t':>8s} {'p':>7s}")
    # baseline: ORB_G5
    fire_l = df_is[df_is["fire_legacy"] == 1]["pnl_r"]
    nonf_l = df_is[df_is["fire_legacy"] == 0]["pnl_r"]
    t, p = _welch(fire_l, nonf_l)
    print(f"  {'legacy G5':12s} {df_is['fire_legacy'].mean()*100:>6.1f}% "
          f"{fire_l.mean():>+10.4f} {nonf_l.mean():>+10.4f} "
          f"{fire_l.mean()-nonf_l.mean():>+7.4f} {t:>+8.2f} {p:>7.4f}")
    for c in candidates:
        col = df_is["orb_size_over_atr"] >= c
        fire = df_is[col]["pnl_r"]
        nonf = df_is[~col]["pnl_r"]
        t, p = _welch(fire, nonf)
        d = fire.mean() - nonf.mean()
        print(f"  {'≥'+str(c):12s} {col.mean()*100:>6.1f}% "
              f"{fire.mean():>+10.4f} {nonf.mean():>+10.4f} {d:>+7.4f} "
              f"{t:>+8.2f} {p:>7.4f}")

    # Step 4: 2026 OOS fire rate for the best candidate
    print("\n" + "-" * 80)
    print("STEP 4 — 2026 OOS FIRE RATE for each candidate")
    print("-" * 80)
    df_oos = df[df.trading_day >= HOLDOUT]
    if len(df_oos) > 0:
        print(f"  Universe n_oos={len(df_oos)}")
        print(f"  {'threshold':12s} {'fire%':>7s} {'ExpR_fire':>10s} {'ExpR_nonf':>10s} {'Δ':>7s}")
        fire_l = df_oos[df_oos["fire_legacy"] == 1]["pnl_r"]
        nonf_l = df_oos[df_oos["fire_legacy"] == 0]["pnl_r"]
        d = fire_l.mean() - nonf_l.mean() if len(nonf_l) else float("nan")
        print(f"  {'legacy G5':12s} {df_oos['fire_legacy'].mean()*100:>6.1f}% "
              f"{fire_l.mean():>+10.4f} {nonf_l.mean() if len(nonf_l) else float('nan'):>+10.4f} {d:>+7.4f}")
        for c in candidates:
            col = df_oos["orb_size_over_atr"] >= c
            fire = df_oos[col]["pnl_r"]
            nonf = df_oos[~col]["pnl_r"]
            d = fire.mean() - nonf.mean() if len(nonf) else float("nan")
            print(f"  {'≥'+str(c):12s} {col.mean()*100:>6.1f}% "
                  f"{fire.mean():>+10.4f} "
                  f"{nonf.mean() if len(nonf) else float('nan'):>+10.4f} {d:>+7.4f}")

    # Step 5: per-year Welch for the candidate best matching ~57% 2026 fire
    # (user will read this + pick)
    print("\n" + "-" * 80)
    print("STEP 5 — PER-YEAR WELCH for candidate ≥0.20 (illustrative)")
    print("-" * 80)
    c = 0.045
    print(f"  threshold = {c}")
    print(f"  {'Year':5s} {'N':>4s} {'fire%':>6s} {'ExpR_f':>8s} {'ExpR_n':>8s} {'Δ':>7s} "
          f"{'Welch t':>8s} {'p':>7s}")
    for y, grp in df.groupby("year"):
        col = grp["orb_size_over_atr"] >= c
        fire = grp[col]["pnl_r"]
        nonf = grp[~col]["pnl_r"]
        if len(fire) < 5 or len(nonf) < 5:
            continue
        t, p = _welch(fire, nonf)
        d = fire.mean() - nonf.mean()
        print(f"  {y:5d} {len(grp):>4d} {col.mean()*100:>5.1f}% "
              f"{fire.mean():>+8.4f} {nonf.mean():>+8.4f} {d:>+7.4f} "
              f"{t:>+8.2f} {p:>7.4f}")


if __name__ == "__main__":
    main()
