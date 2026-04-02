#!/usr/bin/env python3
"""Pre-session consolidation feature quintile scan.

Tests whether pre-session features predict ORB trade outcomes.
No direction assumption — data decides if quiet = bad OR quiet = good.

Features tested:
  1. prev_day_range / atr_20  (all 12 sessions, all instruments)
  2. overnight_range / atr_20  (8 clean sessions only — see LOOKAHEAD below)
  3. abs(gap_open_points) / atr_20  (all 12 sessions, all instruments)
  4. overnight_took_pdh  (binary, 8 clean sessions)
  5. overnight_took_pdl  (binary, 8 clean sessions)

LOOKAHEAD GUARD: overnight_* features use 09:00-17:00 Brisbane window.
Sessions starting inside that window are CONTAMINATED:
  CME_REOPEN, SINGAPORE_OPEN, TOKYO_OPEN, BRISBANE_1025
Clean sessions (ORB starts >= 17:00 Brisbane):
  EUROPE_FLOW, LONDON_METALS, US_DATA_830, NYSE_OPEN,
  US_DATA_1000, COMEX_SETTLE, CME_PRECLOSE, NYSE_CLOSE

Fixed outcome params: E2, CB1, RR1.0 (highest N, active entry model).
Win = pnl_r > 0. Scratches (NULL pnl_r) excluded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.paths import GOLD_DB_PATH

# --- Configuration ---
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR_TARGET = 1.0
ORB_MINUTES = 5
MIN_BIN_N = 20  # minimum trades per quintile bin to report

ALL_SESSIONS = [
    "CME_REOPEN",
    "SINGAPORE_OPEN",
    "TOKYO_OPEN",
    "BRISBANE_1025",
    "LONDON_METALS",
    "EUROPE_FLOW",
    "US_DATA_830",
    "NYSE_OPEN",
    "US_DATA_1000",
    "COMEX_SETTLE",
    "CME_PRECLOSE",
    "NYSE_CLOSE",
]

# Sessions where overnight_* features are CLEAN (no lookahead)
OVERNIGHT_CLEAN_SESSIONS = [
    "EUROPE_FLOW",
    "LONDON_METALS",
    "US_DATA_830",
    "NYSE_OPEN",
    "US_DATA_1000",
    "COMEX_SETTLE",
    "CME_PRECLOSE",
    "NYSE_CLOSE",
]

INSTRUMENTS = sorted(ACTIVE_ORB_INSTRUMENTS)


def load_data() -> pd.DataFrame:
    """Load joined daily_features + orb_outcomes."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    df = con.execute(
        """
        SELECT d.trading_day, d.symbol,
               d.prev_day_range, d.atr_20, d.overnight_range,
               d.gap_open_points, d.overnight_took_pdh, d.overnight_took_pdl,
               o.orb_label, o.pnl_r
        FROM daily_features d
        JOIN orb_outcomes o
          ON d.trading_day = o.trading_day
         AND d.symbol = o.symbol
         AND d.orb_minutes = o.orb_minutes
        WHERE d.orb_minutes = 5
          AND o.entry_model = $1
          AND o.confirm_bars = $2
          AND o.rr_target = $3
          AND o.pnl_r IS NOT NULL
          AND d.atr_20 IS NOT NULL
          AND d.atr_20 > 0
    """,
        [ENTRY_MODEL, CONFIRM_BARS, RR_TARGET],
    ).fetchdf()
    con.close()
    return df


def quintile_wr(group: pd.DataFrame, feature_col: str) -> dict | None:
    """Compute WR per quintile bin. Returns None if insufficient data."""
    valid = group[feature_col].dropna()
    if len(valid) < MIN_BIN_N * 5:
        return None

    try:
        group = group.copy()
        group["qbin"] = pd.qcut(group[feature_col], 5, labels=False, duplicates="drop")
    except ValueError:
        return None

    bins = sorted(group["qbin"].dropna().unique())
    if len(bins) < 4:  # need at least 4 bins for meaningful monotonicity
        return None

    results = []
    for b in bins:
        sub = group[group["qbin"] == b]
        n = len(sub)
        if n < MIN_BIN_N:
            return None
        wr = (sub["pnl_r"] > 0).mean()
        expr = sub["pnl_r"].mean()
        results.append({"bin": b, "n": n, "wr": wr, "expr": expr})

    wrs = [r["wr"] for r in results]
    spread = wrs[-1] - wrs[0]

    # Monotonicity: Spearman correlation of bin index vs WR
    bin_idx = list(range(len(wrs)))
    rho, rho_p = stats.spearmanr(bin_idx, wrs)

    return {
        "n_total": sum(r["n"] for r in results),
        "q1_wr": wrs[0],
        "q5_wr": wrs[-1],
        "spread": spread,
        "rho": rho,
        "rho_p": rho_p,
        "monotonic": abs(rho) > 0.8,
        "bins": results,
    }


def binary_wr(group: pd.DataFrame, feature_col: str) -> dict | None:
    """Compare WR for binary True vs False groups."""
    valid = group.dropna(subset=[feature_col])
    if len(valid) < MIN_BIN_N * 2:
        return None

    true_g = valid[valid[feature_col] == True]  # noqa: E712
    false_g = valid[valid[feature_col] == False]  # noqa: E712

    if len(true_g) < MIN_BIN_N or len(false_g) < MIN_BIN_N:
        return None

    wr_true = (true_g["pnl_r"] > 0).mean()
    wr_false = (false_g["pnl_r"] > 0).mean()

    # Fisher exact test on 2x2 contingency table
    a = int((true_g["pnl_r"] > 0).sum())
    b = int((true_g["pnl_r"] <= 0).sum())
    c = int((false_g["pnl_r"] > 0).sum())
    d_val = int((false_g["pnl_r"] <= 0).sum())
    _, fisher_p = stats.fisher_exact([[a, b], [c, d_val]])

    return {
        "n_true": len(true_g),
        "n_false": len(false_g),
        "wr_true": wr_true,
        "wr_false": wr_false,
        "spread": wr_true - wr_false,
        "fisher_p": fisher_p,
    }


def classify(spread: float, monotonic: bool) -> str:
    """Pre-registered verdict. No peeking."""
    abs_spread = abs(spread)
    if abs_spread < 0.03:
        return "NOISE"
    if abs_spread >= 0.05 and monotonic:
        return "SIGNAL"
    if abs_spread >= 0.05:
        return "PARTIAL"
    return "WEAK"


def main():
    print("Loading data...")
    df = load_data()
    print(
        f"Loaded {len(df):,} outcome rows across {df['symbol'].nunique()} instruments, "
        f"{df['orb_label'].nunique()} sessions"
    )
    print(f"Params: {ENTRY_MODEL} CB{CONFIRM_BARS} RR{RR_TARGET}")
    print()

    # Compute normalized features
    df["pdr_norm"] = df["prev_day_range"] / df["atr_20"]
    df["ovn_norm"] = df["overnight_range"] / df["atr_20"]
    df["gap_norm"] = df["gap_open_points"].abs() / df["atr_20"]

    # --- Continuous feature scans ---
    continuous_features = [
        ("pdr_norm", "prev_day_range/atr", ALL_SESSIONS),
        ("ovn_norm", "overnight_range/atr", OVERNIGHT_CLEAN_SESSIONS),
        ("gap_norm", "abs(gap)/atr", ALL_SESSIONS),
    ]

    all_results = []
    test_count = 0

    for col, label, sessions in continuous_features:
        print(f"{'=' * 80}")
        print(f"FEATURE: {label}")
        print(f"{'=' * 80}")
        print(
            f"{'Session':20s} {'Inst':4s} {'N':>6s} {'Q1 WR':>7s} {'Q5 WR':>7s} "
            f"{'Spread':>7s} {'Rho':>6s} {'Mono':>5s} {'Verdict':>12s}"
        )
        print("-" * 80)

        for sess in sessions:
            for inst in INSTRUMENTS:
                sub = df[(df["orb_label"] == sess) & (df["symbol"] == inst)]
                result = quintile_wr(sub, col)
                test_count += 1

                if result is None:
                    print(f"{sess:20s} {inst:4s} {'SKIP (low N)':>50s}")
                    continue

                verdict = classify(result["spread"], result["monotonic"])
                print(
                    f"{sess:20s} {inst:4s} {result['n_total']:6d} "
                    f"{result['q1_wr']:7.1%} {result['q5_wr']:7.1%} "
                    f"{result['spread']:+7.1%} {result['rho']:+6.2f} "
                    f"{'Y' if result['monotonic'] else 'N':>5s} {verdict:>12s}"
                )

                all_results.append(
                    {
                        "feature": label,
                        "session": sess,
                        "instrument": inst,
                        "type": "continuous",
                        **result,
                        "verdict": verdict,
                    }
                )
        print()

    # --- Binary feature scans ---
    binary_features = [
        ("overnight_took_pdh", "took_pdh", OVERNIGHT_CLEAN_SESSIONS),
        ("overnight_took_pdl", "took_pdl", OVERNIGHT_CLEAN_SESSIONS),
    ]

    for col, label, sessions in binary_features:
        print(f"{'=' * 80}")
        print(f"FEATURE: {label} (binary)")
        print(f"{'=' * 80}")
        print(
            f"{'Session':20s} {'Inst':4s} {'N_T':>5s} {'N_F':>5s} "
            f"{'WR_T':>7s} {'WR_F':>7s} {'Spread':>7s} {'Fisher p':>9s} {'Verdict':>12s}"
        )
        print("-" * 80)

        for sess in sessions:
            for inst in INSTRUMENTS:
                sub = df[(df["orb_label"] == sess) & (df["symbol"] == inst)]
                result = binary_wr(sub, col)
                test_count += 1

                if result is None:
                    print(f"{sess:20s} {inst:4s} {'SKIP (low N)':>55s}")
                    continue

                verdict = classify(result["spread"], monotonic=True)
                print(
                    f"{sess:20s} {inst:4s} {result['n_true']:5d} {result['n_false']:5d} "
                    f"{result['wr_true']:7.1%} {result['wr_false']:7.1%} "
                    f"{result['spread']:+7.1%} {result['fisher_p']:9.4f} {verdict:>12s}"
                )

                all_results.append(
                    {
                        "feature": label,
                        "session": sess,
                        "instrument": inst,
                        "type": "binary",
                        **result,
                        "verdict": verdict,
                    }
                )
        print()

    # --- Summary ---
    print(f"{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total tests: {test_count} (K for BH FDR = {test_count})")

    signals = [r for r in all_results if r["verdict"] == "SIGNAL"]
    partials = [r for r in all_results if r["verdict"] == "PARTIAL"]
    noise = [r for r in all_results if r["verdict"] == "NOISE"]
    weak = [r for r in all_results if r["verdict"] == "WEAK"]
    skipped = test_count - len(all_results)

    print(f"SIGNAL:  {len(signals)}")
    print(f"PARTIAL: {len(partials)}")
    print(f"WEAK:    {len(weak)}")
    print(f"NOISE:   {len(noise)}")
    print(f"SKIPPED: {skipped}")

    if signals:
        print()
        print("=== SIGNAL SURVIVORS (WR spread >= 5%, monotonic) ===")
        for s in sorted(signals, key=lambda x: abs(x["spread"]), reverse=True):
            direction = "HIGH_BETTER" if s["spread"] > 0 else "LOW_BETTER"
            print(f"  {s['feature']:25s} {s['session']:20s} {s['instrument']:4s} spread={s['spread']:+.1%} {direction}")

    if partials:
        print()
        print("=== PARTIAL (WR spread >= 5%, NOT monotonic) ===")
        for s in sorted(partials, key=lambda x: abs(x["spread"]), reverse=True):
            direction = "HIGH_BETTER" if s["spread"] > 0 else "LOW_BETTER"
            print(f"  {s['feature']:25s} {s['session']:20s} {s['instrument']:4s} spread={s['spread']:+.1%} {direction}")


if __name__ == "__main__":
    main()
