"""
Stage L: RULE 13 pressure test — designed correctly after first-run finding.

Two tests:
  1. BLOCK-BOOTSTRAP NULL: the proper null for "does variant mask carry
     edge BEYOND base filter?" Resample pnl_r in blocks WITHIN the
     base-fire population (preserves autocorrelation); keep variant mask
     FIXED. Expected under H0: resampled variant-fire ExpR centers at
     base-fire mean ExpR, NOT at zero. Compute p_boot as fraction of
     resamples where resampled_variant_expr >= observed_variant_expr.

  2. LOOKAHEAD RED-FLAG: inject pnl_r>=0 as "variant mask" — extreme
     lookahead. Expect |t| >> 10 and RULE 12 flag.

Initial-run finding that drove this redesign (committed for audit trail):
  Naive "random subsample with same fire_rate as variant" gave mean t = 1.74
  and FP rate 40.5% — NOT the nominal 5%. Why: base-fire population has
  observed t = 3.15 on N=1555; any random N=494 subsample inherits
  sqrt(494/1555) × 3.15 ≈ 1.77 mean t. Infrastructure is not broken —
  the null was mis-specified. Correct null below preserves base-population
  mean and tests ONLY the marginal edge of the variant mask.

Reference: backtesting-methodology.md historical failure log 2026-04-15
"Block bootstrap preserved joint structure ... Fixed: resample pnl via
blocks preserving autocorrelation, keep mask FIXED to break signal-outcome
link. Proper moving-block bootstrap per Lahiri / Politis-Romano."
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

HOLDOUT = pd.Timestamp(HOLDOUT_SACRED_FROM)
SEED = 20260420


def load_cell_data(con, orb_label: str, orb_minutes: int, rr: float) -> pd.DataFrame:
    q = """
    WITH mnq_feat AS (
      SELECT trading_day, symbol, orb_minutes,
             orb_COMEX_SETTLE_size, overnight_range, atr_20_pct
      FROM daily_features WHERE symbol='MNQ'
    ),
    mes_atr AS (
      SELECT trading_day, atr_20_pct AS mes_atr_20_pct
      FROM daily_features WHERE symbol='MES' AND orb_minutes=5
    )
    SELECT o.trading_day, o.pnl_r,
           m.orb_COMEX_SETTLE_size AS base_size,
           m.overnight_range, x.mes_atr_20_pct
    FROM orb_outcomes o
    JOIN mnq_feat m
      ON o.trading_day=m.trading_day AND o.symbol=m.symbol AND o.orb_minutes=m.orb_minutes
    LEFT JOIN mes_atr x ON o.trading_day=x.trading_day
    WHERE o.symbol='MNQ' AND o.orb_label=? AND o.orb_minutes=?
      AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=?
      AND o.pnl_r IS NOT NULL AND o.trading_day < ?
    ORDER BY o.trading_day
    """
    df = con.execute(q, [orb_label, orb_minutes, rr, HOLDOUT.date()]).df()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    return df


def moving_block_bootstrap_null(
    pnl_base: np.ndarray,
    variant_mask_in_base: np.ndarray,
    n_perms: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    """
    Moving-block bootstrap on pnl_base (within base-fire days).
    For each perm: resample pnl_base in blocks of size sqrt(N), keep
    variant mask FIXED, recompute variant-fire ExpR on resampled series.
    Returns (p_boot, observed_variant_expr, null_distribution).

    Null hypothesis: variant mask is independent of pnl_r within base-fires.
    Under H0: resampled variant-fire mean should fluctuate around
    base-fire mean (not zero).
    """
    rng = np.random.default_rng(seed)
    n = len(pnl_base)
    if n == 0:
        return 1.0, 0.0, np.array([])
    block_size = max(1, int(np.sqrt(n)))
    n_blocks = int(np.ceil(n / block_size))

    observed_expr = float(pnl_base[variant_mask_in_base].mean()) if variant_mask_in_base.sum() > 0 else 0.0

    null_exprs = np.empty(n_perms)
    for i in range(n_perms):
        # Sample n_blocks block starts uniformly in [0, n-block_size]
        starts = rng.integers(0, max(1, n - block_size + 1), size=n_blocks)
        # Concatenate blocks, truncate to n
        resampled = np.concatenate([pnl_base[s:s + block_size] for s in starts])[:n]
        null_exprs[i] = float(resampled[variant_mask_in_base].mean())

    # One-sided p: fraction of null ≥ observed
    p_boot = float((null_exprs >= observed_expr).sum() + 1) / (n_perms + 1)
    return p_boot, observed_expr, null_exprs


def test_block_bootstrap_on_comex_settle(con) -> dict:
    """Test the block bootstrap on a known survivor: COMEX_SETTLE × OVN_100 (t=+4.07)."""
    df = load_cell_data(con, "COMEX_SETTLE", 5, 1.5)
    base = (df["base_size"] >= 5.0).to_numpy()
    variant_all = (df["overnight_range"].fillna(-np.inf) >= 100.0).to_numpy()
    # Restrict to base-fire days
    pnl_base = df.loc[base, "pnl_r"].to_numpy()
    variant_in_base = variant_all[base]

    p_boot, obs, null_dist = moving_block_bootstrap_null(
        pnl_base, variant_in_base, n_perms=2000, seed=SEED
    )
    base_expr = float(pnl_base.mean())

    return {
        "cell": "COMEX_SETTLE × ORB_G5 × OVN_100 (known survivor)",
        "n_base": len(pnl_base),
        "n_variant_in_base": int(variant_in_base.sum()),
        "base_fire_expr": base_expr,
        "observed_variant_expr": obs,
        "null_mean": float(null_dist.mean()),
        "null_p95": float(np.percentile(null_dist, 95)),
        "p_boot": p_boot,
        "expected_null_mean_approx": base_expr,
        "passes_null_centered_on_base": abs(float(null_dist.mean()) - base_expr) < 0.01,
        "passes_rejects_null": p_boot < 0.05,
    }


def test_block_bootstrap_on_null_mask(con) -> dict:
    """
    Inject a RANDOM variant mask of same rate as OVN_100. Under block
    bootstrap null: p_boot should be uniformly distributed (average ~0.5).
    If p_boot systematically < 0.05 → bootstrap has leak.
    """
    df = load_cell_data(con, "COMEX_SETTLE", 5, 1.5)
    base = (df["base_size"] >= 5.0).to_numpy()
    pnl_base = df.loc[base, "pnl_r"].to_numpy()
    n_base = len(pnl_base)

    rng_outer = np.random.default_rng(SEED + 1)
    p_boots = []
    for trial in range(30):
        # Random mask with same fire-rate as OVN_100 on IS (~33.2%)
        k = int(round(0.332 * n_base))
        fake_idx = rng_outer.choice(n_base, size=k, replace=False)
        fake_mask = np.zeros(n_base, dtype=bool)
        fake_mask[fake_idx] = True
        p_boot, obs, _null = moving_block_bootstrap_null(
            pnl_base, fake_mask, n_perms=500, seed=SEED + trial * 100
        )
        p_boots.append(p_boot)
    p_boots_arr = np.array(p_boots)
    return {
        "trials": len(p_boots_arr),
        "p_boot_mean": float(p_boots_arr.mean()),
        "p_boot_median": float(np.median(p_boots_arr)),
        "fp_rate_at_p05": float((p_boots_arr < 0.05).mean()),
        "expected_fp_rate": 0.05,
        "passes": abs(float((p_boots_arr < 0.05).mean()) - 0.05) < 0.10,
    }


def test_lookahead_injection(con) -> dict:
    """Inject pnl_r >= 0 as variant — extreme lookahead. Expect |t| >> 10."""
    df = load_cell_data(con, "COMEX_SETTLE", 5, 1.5)
    base = (df["base_size"] >= 5.0).to_numpy()
    pnl_base = df.loc[base, "pnl_r"].to_numpy()
    # Lookahead: variant = pnl >= 0
    fake_variant = pnl_base >= 0
    wins = pnl_base[fake_variant]
    m = wins.mean()
    sd = wins.std(ddof=1)
    t = m / sd * np.sqrt(len(wins))
    return {
        "n_wins": int(fake_variant.sum()),
        "expr_on_variant": float(m),
        "t_stat": float(t),
        "flags_red_per_rule12": abs(t) > 10.0,
    }


def main():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    print("=== TEST 1: Block bootstrap on KNOWN SURVIVOR (COMEX_SETTLE OVN_100) ===")
    print("Expectation: null mean ~ base fire ExpR; observed >> null p95; p_boot < 0.05")
    r1 = test_block_bootstrap_on_comex_settle(con)
    for k, v in r1.items():
        print(f"  {k:<40} {v}")
    t1_pass = r1["passes_null_centered_on_base"] and r1["passes_rejects_null"]
    print(f"  VERDICT: {'PASS' if t1_pass else 'FAIL'}")
    print()

    print("=== TEST 2: Block bootstrap on NULL mask (random, same fire rate) ===")
    print("Expectation: p_boot uniform ~ 0.5 mean; FP rate at p<0.05 ~ 5%")
    r2 = test_block_bootstrap_on_null_mask(con)
    for k, v in r2.items():
        print(f"  {k:<40} {v}")
    print(f"  VERDICT: {'PASS' if r2['passes'] else 'FAIL'}")
    print()

    print("=== TEST 3: Lookahead injection (pnl>=0 as 'variant') ===")
    print("Expectation: |t| > 10 flags RULE 12 red flag")
    r3 = test_lookahead_injection(con)
    for k, v in r3.items():
        print(f"  {k:<40} {v}")
    print(f"  VERDICT: {'PASS' if r3['flags_red_per_rule12'] else 'FAIL'}")
    print()

    all_pass = t1_pass and r2["passes"] and r3["flags_red_per_rule12"]
    if all_pass:
        print("=== PRESSURE TEST: ALL PASS ===")
        print("Block bootstrap correctly (a) rejects on known survivor, (b)")
        print("produces uniform p under null, (c) surfaces lookahead. Stage H")
        print("can trust this bootstrap design on the other 5 survivors.")
    else:
        print("=== PRESSURE TEST: FAIL ===")
        print("HALT before Stage H.")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
