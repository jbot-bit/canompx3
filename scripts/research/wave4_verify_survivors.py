#!/usr/bin/env python3
"""Adversarial T4/T6 verification of wave 4 validated strategies.

The T2-T8 test on overnight_range/atr killed all 9 T1 passers at T6 null bootstrap.
This raises the question: do the ABSOLUTE OVNRNG_100 (and other wave 4 filter) survivors
in validated_setups pass the same T6 test, or are they also noise that the validator missed?

Validator runs BH FDR + WFE + year stability. It does NOT run T4 sensitivity or T6 full
5000-permutation bootstrap. This script applies those adversarial tests to every wave 4
validated strategy.

Output: per-strategy verdict (SURVIVES / KILL) with failure reasons.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH

N_BOOTSTRAP = 5000
HOLDOUT_DATE = "2026-01-01"

# Filter column lookups per filter type
FILTER_SQL = {
    "OVNRNG_100": "df.overnight_range >= 100",
    "OVNRNG_50": "df.overnight_range >= 50",
    "OVNRNG_25": "df.overnight_range >= 25",
    "OVNRNG_10": "df.overnight_range >= 10",
    "X_MES_ATR60": "X_MES",  # special — cross-asset join
    "COST_LT12": "COST12",  # special — per-instrument friction
    "COST_LT10": "COST10",
    "COST_LT08": "COST08",
    "ATR_P70": "df.atr_20_pct >= 70",
    "ATR_P50": "df.atr_20_pct >= 50",
    "ORB_G5": "df.orb_{session}_size >= 5",
    "ORB_G6": "df.orb_{session}_size >= 6",
    "ORB_G8": "df.orb_{session}_size >= 8",
}

# Cost model per instrument (from pipeline.cost_model)
COST_SPECS = {
    "MNQ": {"friction": 2.42, "pv": 2.0},
    "MES": {"friction": 2.67, "pv": 5.0},
    "MGC": {"friction": 3.74, "pv": 10.0},
}


def load_filtered_outcomes(con, instrument: str, session: str, filter_type: str, rr_target: float) -> pd.DataFrame:
    """Load outcomes with filter applied. Returns DataFrame with trading_day, pnl_r, year."""
    if filter_type not in FILTER_SQL:
        return pd.DataFrame()
    filt_expr = FILTER_SQL[filter_type]

    # Handle special cases
    if filt_expr == "X_MES":
        # Cross-asset: MES ATR percentile from MES daily_features joined by trading_day
        q = f"""
            SELECT o.trading_day, o.pnl_r
            FROM orb_outcomes o
            JOIN daily_features df_mes
              ON o.trading_day=df_mes.trading_day AND df_mes.symbol='MES' AND df_mes.orb_minutes=5
            WHERE o.symbol='{instrument}' AND o.entry_model='E2' AND o.orb_minutes=5
                  AND o.rr_target={rr_target} AND o.orb_label='{session}'
                  AND o.trading_day < '{HOLDOUT_DATE}'
                  AND df_mes.atr_20_pct >= 60
        """
    elif filt_expr in ("COST08", "COST10", "COST12"):
        # Cost filter: friction / (orb_size * point_value) < threshold_pct/100
        spec = COST_SPECS.get(instrument)
        if not spec:
            return pd.DataFrame()
        pct = int(filt_expr[-2:]) / 100.0
        min_orb = spec["friction"] / (pct * spec["pv"])
        col = f"orb_{session}_size"
        q = f"""
            SELECT o.trading_day, o.pnl_r
            FROM orb_outcomes o
            JOIN daily_features df
              ON o.trading_day=df.trading_day AND o.symbol=df.symbol AND o.orb_minutes=df.orb_minutes
            WHERE o.symbol='{instrument}' AND o.entry_model='E2' AND o.orb_minutes=5
                  AND o.rr_target={rr_target} AND o.orb_label='{session}'
                  AND o.trading_day < '{HOLDOUT_DATE}'
                  AND df."{col}" >= {min_orb}
        """
    else:
        # Standard filter expression
        if "{session}" in filt_expr:
            filt_expr = filt_expr.replace("{session}", session)
        q = f"""
            SELECT o.trading_day, o.pnl_r
            FROM orb_outcomes o
            JOIN daily_features df
              ON o.trading_day=df.trading_day AND o.symbol=df.symbol AND o.orb_minutes=df.orb_minutes
            WHERE o.symbol='{instrument}' AND o.entry_model='E2' AND o.orb_minutes=5
                  AND o.rr_target={rr_target} AND o.orb_label='{session}'
                  AND o.trading_day < '{HOLDOUT_DATE}'
                  AND {filt_expr}
        """

    try:
        df = con.sql(q).df()
    except Exception as e:
        print(f"  SQL error for {instrument} {session} {filter_type}: {e}")
        return pd.DataFrame()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    return df


def load_unfiltered_outcomes(con, instrument: str, session: str, rr_target: float) -> pd.DataFrame:
    """Load ALL outcomes for the combo — used as the null pool for bootstrap."""
    q = f"""
        SELECT o.trading_day, o.pnl_r
        FROM orb_outcomes o
        WHERE o.symbol='{instrument}' AND o.entry_model='E2' AND o.orb_minutes=5
              AND o.rr_target={rr_target} AND o.orb_label='{session}'
              AND o.trading_day < '{HOLDOUT_DATE}'
    """
    df = con.sql(q).df()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    return df


def t6_pass_rate_bootstrap(filtered: pd.DataFrame, unfiltered: pd.DataFrame, n_perms: int = N_BOOTSTRAP) -> dict:
    """Bootstrap WITH REPLACEMENT: sample N trades from unfiltered pool,
    compute ExpR distribution. Does filtered ExpR beat the null?
    Uses replacement so pool-vs-N ratio doesn't distort the null.
    """
    n_filtered = len(filtered)
    if n_filtered < 50:
        return {"error": f"insufficient N={n_filtered}"}
    observed_expr = float(filtered["pnl_r"].mean())
    pool = np.asarray(unfiltered["pnl_r"].dropna().values, dtype=float)
    if len(pool) < 100:
        return {"error": f"pool too small: {len(pool)}"}

    rng = np.random.default_rng(42)
    null_exprs = []
    for _ in range(n_perms):
        sample = rng.choice(pool, size=n_filtered, replace=True)
        null_exprs.append(float(sample.mean()))
    null_arr = np.array(null_exprs)
    b = int(np.sum(null_arr >= observed_expr))
    p_value = (b + 1) / (len(null_arr) + 1)
    pool_mean = float(pool.mean())
    return {
        "observed_expr": observed_expr,
        "unfiltered_expr": pool_mean,
        "lift": observed_expr - pool_mean,
        "null_mean": float(null_arr.mean()),
        "null_p95": float(np.percentile(null_arr, 95)),
        "p_value": p_value,
        "n_filtered": n_filtered,
        "n_pool": len(pool),
    }


def run_verification():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    # Get all active validated strategies — test every filter type we can handle
    strategies = con.sql("""
        SELECT strategy_id, instrument, orb_label, rr_target, filter_type,
               sample_size, expectancy_r, sharpe_ann, wfe
        FROM validated_setups
        WHERE status='active'
        ORDER BY instrument, expectancy_r DESC
    """).fetchall()

    print("=" * 90)
    print(f"ADVERSARIAL T6 VERIFICATION — {len(strategies)} wave 4 validated strategies")
    print(f"  Bootstrap: {N_BOOTSTRAP} permutations, 1-tailed (filtered ExpR > null)")
    print("=" * 90)

    survivors = []
    killers = []

    for s in strategies:
        sid, inst, sess, rr, ft, n_val, expr_val, sharpe_val, wfe_val = s
        print(f"\n{sid}")
        print(f"  VALIDATED: N={n_val} ExpR={expr_val:+.3f} Sh={sharpe_val:.2f} WFE={wfe_val:.2f}")

        filtered = load_filtered_outcomes(con, inst, sess, ft, rr)
        if filtered.empty:
            print(f"  SKIP — cannot construct filter SQL for {ft}")
            continue

        unfiltered = load_unfiltered_outcomes(con, inst, sess, rr)

        t6 = t6_pass_rate_bootstrap(filtered, unfiltered)
        if "error" in t6:
            print(f"  T6 ERROR: {t6['error']}")
            continue

        pass_t6 = t6["p_value"] < 0.05
        verdict = "SURVIVES" if pass_t6 else "KILL (NOISE)"
        print(
            f"  T6: filtered ExpR={t6['observed_expr']:+.3f} vs unfiltered={t6['unfiltered_expr']:+.3f} "
            f"lift={t6['lift']:+.3f}"
        )
        print(f"      null p95={t6['null_p95']:+.3f} p={t6['p_value']:.4f} → {verdict}")

        if pass_t6:
            survivors.append((sid, t6["observed_expr"], t6["lift"], t6["p_value"]))
        else:
            killers.append((sid, t6["observed_expr"], t6["lift"], t6["p_value"]))

    con.close()

    print()
    print("=" * 90)
    print(f"SURVIVORS ({len(survivors)})")
    print("=" * 90)
    for sid, expr, lift, p in sorted(survivors, key=lambda x: x[3]):
        print(f"  {sid}: ExpR={expr:+.3f} lift={lift:+.3f} p={p:.4f}")

    print()
    print("=" * 90)
    print(f"KILLED AS NOISE ({len(killers)})")
    print("=" * 90)
    for sid, expr, lift, p in sorted(killers, key=lambda x: -x[3]):
        print(f"  {sid}: ExpR={expr:+.3f} lift={lift:+.3f} p={p:.4f}")


if __name__ == "__main__":
    run_verification()
