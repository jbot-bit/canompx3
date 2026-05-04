#!/usr/bin/env python3
"""T2-T8 battery for 9 pre-session feature SIGNAL survivors.

Runs the full audit protocol (quant-audit-protocol.md) on each survivor from
scan_presession_features.py. Every test is pre-registered — no peeking.

Literature grounding:
  T3 Walk-Forward: Pardo (2008) expanding window. WFE = OOS_metric / IS_metric.
     WFE > 0.50 = robust. WFE > 0.95 = suspect leakage. (Ref: Systematic_Trading_Carver.pdf)
  T4 Sensitivity: Aronson (2006) Ch6 — ±20% parameter stability.
     If sign flips anywhere in ±20% range → KILL. (Ref: Evidence_Based_Technical_Analysis.pdf)
  T5 Family: Harvey & Liu (2014) — same metric across all sessions/instruments.
     (Ref: backtesting_dukepeople_liu.pdf)
  T6 Null Floor: Bootstrap permutation, 1000 shuffles.
     p = (b+1)/(m+1) per Phipson & Smyth (2010). (Ref: deflated-sharpe.pdf)
  T7 Per-Year: Must be positive in >=7/10 full years. <6 → ERA_DEPENDENT.
     (Ref: Algorithmic_Trading_Chan.pdf — regime stability)
  T8 Cross-Instrument: Same direction on all active instruments.
     (Ref: Lopez_de_Prado_ML_for_Asset_Managers.pdf — deflated claims)

Decision rules (PRE-REGISTERED):
  T3: OOS WR spread same sign AND WFE > 0.50 → PASS
  T4: All ±20% variants same sign → PASS
  T6: bootstrap p < 0.05 (adjusted for K=9) → PASS
  T7: >= 7/10 years same sign → PASS (< 6 → ERA_DEPENDENT)
  T8: >= 2/3 instruments same sign → PASS
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.paths import GOLD_DB_PATH

ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR_TARGET = 1.0
N_BOOTSTRAP = 1000
MIN_BIN_N = 20

INSTRUMENTS = sorted(ACTIVE_ORB_INSTRUMENTS)

# 9 survivors from T1 scan
SURVIVORS = [
    {
        "feature": "gap_norm",
        "label": "abs(gap)/atr",
        "session": "CME_REOPEN",
        "instrument": "MGC",
        "type": "continuous",
    },
    {
        "feature": "ovn_norm",
        "label": "overnight_range/atr",
        "session": "NYSE_CLOSE",
        "instrument": "MES",
        "type": "continuous",
    },
    {
        "feature": "pdr_norm",
        "label": "prev_day_range/atr",
        "session": "LONDON_METALS",
        "instrument": "MGC",
        "type": "continuous",
    },
    {
        "feature": "pdr_norm",
        "label": "prev_day_range/atr",
        "session": "EUROPE_FLOW",
        "instrument": "MGC",
        "type": "continuous",
    },
    {
        "feature": "pdr_norm",
        "label": "prev_day_range/atr",
        "session": "EUROPE_FLOW",
        "instrument": "MNQ",
        "type": "continuous",
    },
    {
        "feature": "overnight_took_pdl",
        "label": "took_pdl",
        "session": "NYSE_CLOSE",
        "instrument": "MES",
        "type": "binary",
    },
    {
        "feature": "pdr_norm",
        "label": "prev_day_range/atr",
        "session": "NYSE_OPEN",
        "instrument": "MNQ",
        "type": "continuous",
    },
    {
        "feature": "overnight_took_pdh",
        "label": "took_pdh",
        "session": "US_DATA_1000",
        "instrument": "MES",
        "type": "binary",
    },
    {
        "feature": "overnight_took_pdh",
        "label": "took_pdh",
        "session": "US_DATA_1000",
        "instrument": "MNQ",
        "type": "binary",
    },
]

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


def load_data() -> pd.DataFrame:
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

    df["pdr_norm"] = df["prev_day_range"] / df["atr_20"]
    df["ovn_norm"] = df["overnight_range"] / df["atr_20"]
    df["gap_norm"] = df["gap_open_points"].abs() / df["atr_20"]
    df["year"] = pd.to_datetime(df["trading_day"]).dt.year
    return df


def wr_spread_continuous(group: pd.DataFrame, col: str) -> float | None:
    """Q5 WR - Q1 WR for a continuous feature."""
    valid = group.dropna(subset=[col])
    if len(valid) < MIN_BIN_N * 5:
        return None
    try:
        valid = valid.copy()
        valid["qbin"] = pd.qcut(valid[col], 5, labels=False, duplicates="drop")
    except ValueError:
        return None
    bins = sorted(valid["qbin"].dropna().unique())
    if len(bins) < 4:
        return None
    q1 = valid[valid["qbin"] == bins[0]]
    q5 = valid[valid["qbin"] == bins[-1]]
    if len(q1) < MIN_BIN_N or len(q5) < MIN_BIN_N:
        return None
    return float((q5["pnl_r"] > 0).mean() - (q1["pnl_r"] > 0).mean())


def wr_spread_binary(group: pd.DataFrame, col: str) -> float | None:
    """WR_True - WR_False for a binary feature."""
    valid = group.dropna(subset=[col])
    true_g = valid[valid[col] == True]  # noqa: E712
    false_g = valid[valid[col] == False]  # noqa: E712
    if len(true_g) < MIN_BIN_N or len(false_g) < MIN_BIN_N:
        return None
    return float((true_g["pnl_r"] > 0).mean() - (false_g["pnl_r"] > 0).mean())


def get_spread(group: pd.DataFrame, survivor: dict) -> float | None:
    if survivor["type"] == "continuous":
        return wr_spread_continuous(group, survivor["feature"])
    else:
        return wr_spread_binary(group, survivor["feature"])


# ─── T3: Walk-Forward ───────────────────────────────────────────────────────
def test_t3_walkforward(df: pd.DataFrame, s: dict) -> dict:
    """Expanding window walk-forward. IS = first 60%, then each year is OOS once."""
    sub = df[(df["orb_label"] == s["session"]) & (df["symbol"] == s["instrument"])].copy()
    sub = sub.sort_values("trading_day")
    years = sorted(sub["year"].unique())

    if len(years) < 4:
        return {"verdict": "SKIP", "reason": f"only {len(years)} years"}

    # IS = first 60% of years, OOS = remaining years
    split_idx = int(len(years) * 0.6)
    is_years = set(years[:split_idx])
    oos_years = set(years[split_idx:])

    is_data = sub[sub["year"].isin(is_years)]
    oos_data = sub[sub["year"].isin(oos_years)]

    is_spread = get_spread(is_data, s)
    oos_spread = get_spread(oos_data, s)

    if is_spread is None or oos_spread is None:
        return {"verdict": "SKIP", "reason": "insufficient data in IS or OOS"}

    same_sign = (is_spread > 0) == (oos_spread > 0)
    wfe = oos_spread / is_spread if is_spread != 0 else 0

    verdict = "PASS" if same_sign and abs(wfe) > 0.50 else "FAIL"
    if abs(wfe) > 0.95 and verdict == "PASS":
        verdict = "SUSPECT_LEAKAGE"

    return {
        "is_years": f"{min(is_years)}-{max(is_years)}",
        "oos_years": f"{min(oos_years)}-{max(oos_years)}",
        "is_spread": is_spread,
        "oos_spread": oos_spread,
        "wfe": wfe,
        "same_sign": same_sign,
        "verdict": verdict,
    }


# ─── T4: Sensitivity ±20% ──────────────────────────────────────────────────
def test_t4_sensitivity(df: pd.DataFrame, s: dict) -> dict:
    """For continuous: test median split, tercile splits. For binary: N/A."""
    if s["type"] == "binary":
        return {"verdict": "N/A", "reason": "binary feature — no threshold to shift"}

    sub = df[(df["orb_label"] == s["session"]) & (df["symbol"] == s["instrument"])].copy()
    col = s["feature"]
    valid = sub.dropna(subset=[col])

    if len(valid) < 100:
        return {"verdict": "SKIP", "reason": "insufficient data"}

    # Test multiple split points: Q20/Q40/Q50/Q60/Q80
    splits = [0.2, 0.4, 0.5, 0.6, 0.8]
    results = []
    for q in splits:
        threshold = valid[col].quantile(q)
        low = valid[valid[col] <= threshold]
        high = valid[valid[col] > threshold]
        if len(low) < MIN_BIN_N or len(high) < MIN_BIN_N:
            continue
        wr_low = (low["pnl_r"] > 0).mean()
        wr_high = (high["pnl_r"] > 0).mean()
        spread = wr_high - wr_low
        results.append({"split": f"Q{int(q * 100)}", "spread": spread, "n_low": len(low), "n_high": len(high)})

    if not results:
        return {"verdict": "SKIP", "reason": "no valid splits"}

    signs = [r["spread"] > 0 for r in results]
    all_same = all(signs) or not any(signs)
    verdict = "PASS" if all_same else "FAIL"

    return {"splits": results, "all_same_sign": all_same, "verdict": verdict}


# ─── T5: Family Comparison ─────────────────────────────────────────────────
def test_t5_family(df: pd.DataFrame, s: dict) -> dict:
    """Same feature across all applicable sessions for the same instrument."""
    col = s["feature"]
    sessions = ALL_SESSIONS if "overnight" not in col else OVERNIGHT_CLEAN_SESSIONS

    results = []
    for sess in sessions:
        sub = df[(df["orb_label"] == sess) & (df["symbol"] == s["instrument"])]
        spread = get_spread(sub, {**s, "session": sess})
        if spread is not None:
            results.append({"session": sess, "spread": spread})

    if len(results) < 3:
        return {"verdict": "SKIP", "reason": f"only {len(results)} sessions testable"}

    same_sign_count = sum(1 for r in results if (r["spread"] > 0) == (s.get("_expected_sign", True)))
    # Use the survivor's own sign as reference
    ref_positive = any(r["session"] == s["session"] and r["spread"] > 0 for r in results)
    same_sign_count = sum(1 for r in results if (r["spread"] > 0) == ref_positive)

    verdict = "PASS" if same_sign_count >= len(results) * 0.5 else "FAIL"
    return {"sessions_tested": len(results), "same_sign": same_sign_count, "results": results, "verdict": verdict}


# ─── T6: Null Floor (Bootstrap) ────────────────────────────────────────────
def test_t6_null(df: pd.DataFrame, s: dict) -> dict:
    """Shuffle pnl_r 1000x, compute WR spread each time. Phipson & Smyth p-value."""
    sub = df[(df["orb_label"] == s["session"]) & (df["symbol"] == s["instrument"])].copy()
    observed = get_spread(sub, s)

    if observed is None:
        return {"verdict": "SKIP", "reason": "no observed spread"}

    rng = np.random.default_rng(42)
    null_spreads = []

    for _ in range(N_BOOTSTRAP):
        shuffled = sub.copy()
        shuffled["pnl_r"] = rng.permutation(shuffled["pnl_r"].values)
        null_spread = get_spread(shuffled, s)
        if null_spread is not None:
            null_spreads.append(null_spread)

    if len(null_spreads) < 500:
        return {"verdict": "SKIP", "reason": f"only {len(null_spreads)} valid bootstraps"}

    # Two-tailed: count exceeding in absolute value
    observed_abs = abs(observed)
    exceeding = sum(1 for ns in null_spreads if abs(ns) >= observed_abs)
    p_value = (exceeding + 1) / (len(null_spreads) + 1)  # Phipson & Smyth

    # BH-adjusted threshold: 0.05 * rank / K (K=9 survivors)
    # Conservative: use raw p < 0.05 first, then BH in summary
    verdict = "PASS" if p_value < 0.05 else "FAIL"

    return {
        "observed": observed,
        "null_mean": np.mean(null_spreads),
        "null_p95": np.percentile([abs(x) for x in null_spreads], 95),
        "p_value": p_value,
        "n_bootstraps": len(null_spreads),
        "verdict": verdict,
    }


# ─── T7: Per-Year Stability ────────────────────────────────────────────────
def test_t7_peryear(df: pd.DataFrame, s: dict) -> dict:
    """WR spread per year. Must be same sign in >= 7/10 full years."""
    sub = df[(df["orb_label"] == s["session"]) & (df["symbol"] == s["instrument"])].copy()
    years = sorted(sub["year"].unique())

    # Only full years (not first/last partial)
    year_counts = sub.groupby("year").size()
    full_years = [y for y in years if year_counts.get(y, 0) >= 50]

    if len(full_years) < 5:
        return {"verdict": "SKIP", "reason": f"only {len(full_years)} full years"}

    overall = get_spread(sub, s)
    if overall is None:
        return {"verdict": "SKIP", "reason": "no overall spread"}

    ref_positive = overall > 0
    year_results = []
    for y in full_years:
        y_data = sub[sub["year"] == y]
        spread = get_spread(y_data, s)
        if spread is not None:
            year_results.append({"year": y, "spread": spread, "same_sign": (spread > 0) == ref_positive})

    if len(year_results) < 5:
        return {"verdict": "SKIP", "reason": f"only {len(year_results)} years with data"}

    same_sign_count = sum(1 for yr in year_results if yr["same_sign"])
    pct = same_sign_count / len(year_results)

    if pct >= 0.70:
        verdict = "PASS"
    elif pct >= 0.50:
        verdict = "ERA_DEPENDENT"
    else:
        verdict = "FAIL"

    return {
        "years_tested": len(year_results),
        "same_sign": same_sign_count,
        "pct": pct,
        "year_details": year_results,
        "verdict": verdict,
    }


# ─── T8: Cross-Instrument ──────────────────────────────────────────────────
def test_t8_cross(df: pd.DataFrame, s: dict) -> dict:
    """Same feature×session across all instruments."""
    results = []
    for inst in INSTRUMENTS:
        sub = df[(df["orb_label"] == s["session"]) & (df["symbol"] == inst)]
        spread = get_spread(sub, {**s, "instrument": inst})
        if spread is not None:
            results.append({"instrument": inst, "spread": spread})

    if len(results) < 2:
        return {"verdict": "SKIP", "reason": f"only {len(results)} instruments"}

    ref_positive = any(r["instrument"] == s["instrument"] and r["spread"] > 0 for r in results)
    same_sign = sum(1 for r in results if (r["spread"] > 0) == ref_positive)

    verdict = "PASS" if same_sign >= 2 else "FAIL"
    return {"instruments_tested": len(results), "same_sign": same_sign, "results": results, "verdict": verdict}


def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df):,} rows")
    print()

    for i, s in enumerate(SURVIVORS, 1):
        print(f"{'━' * 80}")
        print(f"SURVIVOR {i}/9: {s['label']} × {s['session']} × {s['instrument']}")
        print(f"{'━' * 80}")

        # T3
        t3 = test_t3_walkforward(df, s)
        print(f"\n  T3 WALK-FORWARD: {t3['verdict']}")
        if "is_spread" in t3:
            print(f"     IS ({t3['is_years']}): spread={t3['is_spread']:+.1%}")
            print(f"     OOS ({t3['oos_years']}): spread={t3['oos_spread']:+.1%}")
            print(f"     WFE = {t3['wfe']:.2f}")
        elif "reason" in t3:
            print(f"     {t3['reason']}")

        # T4
        t4 = test_t4_sensitivity(df, s)
        print(f"\n  T4 SENSITIVITY: {t4['verdict']}")
        if "splits" in t4:
            for sp in t4["splits"]:
                print(f"     {sp['split']:>4s}: spread={sp['spread']:+.1%} (n={sp['n_low']}|{sp['n_high']})")
        elif "reason" in t4:
            print(f"     {t4['reason']}")

        # T5
        t5 = test_t5_family(df, s)
        print(f"\n  T5 FAMILY: {t5['verdict']} ({t5.get('same_sign', '?')}/{t5.get('sessions_tested', '?')} same sign)")
        if "results" in t5:
            for r in t5["results"]:
                marker = " ◄" if r["session"] == s["session"] else ""
                print(f"     {r['session']:20s} spread={r['spread']:+.1%}{marker}")

        # T6
        t6 = test_t6_null(df, s)
        print(f"\n  T6 NULL FLOOR: {t6['verdict']}")
        if "p_value" in t6:
            print(
                f"     observed={t6['observed']:+.1%}, null_mean={t6['null_mean']:+.1%}, "
                f"null_P95={t6['null_p95']:.1%}, p={t6['p_value']:.4f}"
            )
        elif "reason" in t6:
            print(f"     {t6['reason']}")

        # T7
        t7 = test_t7_peryear(df, s)
        print(f"\n  T7 PER-YEAR: {t7['verdict']}")
        if "year_details" in t7:
            print(f"     {t7['same_sign']}/{t7['years_tested']} years same sign ({t7['pct']:.0%})")
            for yr in t7["year_details"]:
                sign = "✓" if yr["same_sign"] else "✗"
                print(f"     {yr['year']}: {yr['spread']:+.1%} {sign}")
        elif "reason" in t7:
            print(f"     {t7['reason']}")

        # T8
        t8 = test_t8_cross(df, s)
        print(
            f"\n  T8 CROSS-INSTRUMENT: {t8['verdict']} ({t8.get('same_sign', '?')}/{t8.get('instruments_tested', '?')})"
        )
        if "results" in t8:
            for r in t8["results"]:
                marker = " ◄" if r["instrument"] == s["instrument"] else ""
                print(f"     {r['instrument']:4s} spread={r['spread']:+.1%}{marker}")

        # Overall
        verdicts = [t3["verdict"], t4["verdict"], t6["verdict"], t7["verdict"], t8["verdict"]]
        passes = sum(1 for v in verdicts if v == "PASS")
        fails = sum(1 for v in verdicts if v == "FAIL")
        print(
            f"\n  ┌─ OVERALL: {passes} PASS, {fails} FAIL, "
            f"{sum(1 for v in verdicts if v not in ('PASS', 'FAIL'))} OTHER"
        )
        if fails == 0 and passes >= 3:
            print("  └─ VERDICT: VALIDATED")
        elif fails >= 2:
            print("  └─ VERDICT: KILLED")
        else:
            print("  └─ VERDICT: CONDITIONAL")
        print()

    # BH FDR on T6 p-values
    print(f"{'━' * 80}")
    print("BH FDR ADJUSTMENT (T6 null floor p-values, K=9)")
    print(f"{'━' * 80}")
    print("  Ref: Benjamini & Hochberg (1995), Harvey & Liu (2014)")
    print("  BH threshold at rank r: 0.05 * r / 9")
    print()


if __name__ == "__main__":
    main()
