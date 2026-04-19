#!/usr/bin/env python3
"""Phase 2.9 A4 hardening — moving-block bootstrap on 2025 K_year BH survivors.

Confirmatory sub-audit on the 4 cells that passed Phase 2.9 K_year BH at
q=0.10 in year 2025. The skeptical-audit pass flagged these as dependence-
fragile (3 of 4 are MNQ COMEX_SETTLE different filters sharing bars). BHY
already corrected for test-statistic dependence across cells; this script
corrects at the PER-CELL level by resampling moving blocks of pnl_r to
build a non-parametric null that respects within-year serial correlation.

Methodology:
  Politis & Romano (1994) moving-block bootstrap with overlapping blocks of
  length L. For a given year's pnl_r series of length n, demean the series
  and resample ceil(n/L) blocks with replacement. Each replica is a null
  realization preserving the autocorrelation structure but with zero mean.
  p_bootstrap = (|rep_expr| >= |observed_expr|) proportion + continuity
  correction (Phipson & Smyth 2010).

Block length: L = max(2, round(n^(1/3))) per Hall-Horowitz-Jing (1995)
optimal MSE rate for block bootstrap.

Canonical delegations:
  - _window_stats via research.phase_2_9_comprehensive_multi_year_stratification
  - load_active_setups, compute_mode_a via research.mode_a_revalidation_active_setups
  - GOLD_DB_PATH, HOLDOUT_SACRED_FROM per usual

Usage:
  DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db \\
      uv run --frozen python research/phase_2_9_block_bootstrap_2025.py

Output: prints a comparison table (parametric year_t, year_p, bootstrap_p,
bootstrap_ci_95) for each of the 4 cells. Cites the commit SHA when run
from a clean working tree.
"""
from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402

# Target cells from phase_2_9_main.csv bh_year=True in 2025:
TARGETS = [
    {
        "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100",
        "instrument": "MNQ", "orb_label": "COMEX_SETTLE",
        "orb_minutes": 5, "rr_target": 1.0, "filter_type": "OVNRNG_100",
        "direction": "long",
    },
    {
        "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60",
        "instrument": "MNQ", "orb_label": "COMEX_SETTLE",
        "orb_minutes": 5, "rr_target": 1.0, "filter_type": "X_MES_ATR60",
        "direction": "long",
    },
    {
        "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
        "instrument": "MNQ", "orb_label": "COMEX_SETTLE",
        "orb_minutes": 5, "rr_target": 1.5, "filter_type": "OVNRNG_100",
        "direction": "long",
    },
    {
        "strategy_id": "MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30",
        "instrument": "MNQ", "orb_label": "SINGAPORE_OPEN",
        "orb_minutes": 30, "rr_target": 1.5, "filter_type": "ATR_P50",
        "direction": "long",
    },
]

N_REPLICAS: int = 10_000
SEED: int = 20260419  # lock for reproducibility


def fetch_year_pnl(con: duckdb.DuckDBPyConnection, spec: dict, year: int) -> np.ndarray:
    """Fetch filtered pnl_r series for the given cell in the given year.

    Uses canonical filter delegation via research.filter_utils.filter_signal
    -- same code path as _window_stats so results match phase_2_9_main.csv.
    """
    from research.filter_utils import filter_signal
    from trading_app.config import ALL_FILTERS, CrossAssetATRFilter

    sess = spec["orb_label"]
    sql = f"""
        SELECT o.trading_day, o.pnl_r, o.outcome, o.symbol, d.*
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ? AND o.orb_label = ? AND o.orb_minutes = ?
          AND o.entry_model = 'E2' AND o.confirm_bars = 1 AND o.rr_target = ?
          AND d.orb_{sess}_break_dir = ?
          AND o.pnl_r IS NOT NULL
          AND o.trading_day >= ? AND o.trading_day < ?
        ORDER BY o.trading_day
    """
    df = con.execute(sql, [
        spec["instrument"], sess, spec["orb_minutes"], spec["rr_target"],
        spec["direction"], date(year, 1, 1), date(year + 1, 1, 1),
    ]).df()
    if len(df) == 0:
        return np.array([])

    filter_type = spec["filter_type"]
    filt_obj = ALL_FILTERS.get(filter_type)
    if filt_obj is not None and isinstance(filt_obj, CrossAssetATRFilter):
        source = filt_obj.source_instrument
        if source != spec["instrument"]:
            src_rows = con.execute(
                "SELECT trading_day, atr_20_pct FROM daily_features "
                "WHERE symbol = ? AND orb_minutes = 5 AND atr_20_pct IS NOT NULL",
                [source],
            ).fetchall()
            src_map = {
                (td.date() if hasattr(td, "date") else td): float(pct)
                for td, pct in src_rows
            }
            df[f"cross_atr_{source}_pct"] = df["trading_day"].apply(
                lambda d: src_map.get(d.date() if hasattr(d, "date") else d)
            )
    fire = np.asarray(filter_signal(df, filter_type, sess)).astype(bool)
    return df[fire]["pnl_r"].astype(float).to_numpy()


def moving_block_bootstrap_p(
    pnl: np.ndarray, n_replicas: int, seed: int
) -> tuple[float, float, float, int]:
    """Politis-Romano overlapping MBB p-value under zero-mean null.

    Block length L = max(2, round(n^(1/3))) per Hall-Horowitz-Jing 1995.
    Demeans the series, forms all (n - L + 1) overlapping blocks, samples
    ceil(n/L) blocks with replacement, computes replica expr.

    Returns (observed_expr, bootstrap_p, block_length, n_obs).
    p_bootstrap = (n_exceed + 1) / (n_replicas + 1) per Phipson-Smyth 2010.
    """
    n = len(pnl)
    if n < 4:
        return float("nan"), float("nan"), 0, n
    L = max(2, int(round(n ** (1.0 / 3.0))))
    obs_expr = float(pnl.mean())
    demeaned = pnl - obs_expr
    # Overlapping blocks: start indices 0 .. n - L
    n_blocks_all = n - L + 1
    blocks = np.lib.stride_tricks.sliding_window_view(demeaned, L)
    assert blocks.shape == (n_blocks_all, L)
    # Per replica: sample ceil(n/L) blocks with replacement, concat, trim to n
    n_samp = int(math.ceil(n / L))
    rng = np.random.default_rng(seed)
    replicas = np.empty(n_replicas, dtype=np.float64)
    for i in range(n_replicas):
        idx = rng.integers(0, n_blocks_all, size=n_samp)
        rep = blocks[idx].reshape(-1)[:n]
        replicas[i] = rep.mean()
    # Two-sided p under zero-mean null
    n_exceed = int(np.sum(np.abs(replicas) >= abs(obs_expr)))
    p = (n_exceed + 1) / (n_replicas + 1)
    return obs_expr, float(p), L, n


def parametric_t(pnl: np.ndarray) -> tuple[float, float]:
    """One-sample t and two-sided Student-t p."""
    from scipy import stats
    n = len(pnl)
    if n < 2:
        return float("nan"), float("nan")
    mean = float(pnl.mean())
    sd = float(pnl.std(ddof=1))
    if sd == 0:
        return float("nan"), float("nan")
    t = mean / (sd / math.sqrt(n))
    p = float(2.0 * stats.t.sf(abs(t), df=n - 1))
    return t, p


def main() -> int:
    print("Phase 2.9 A4 -- moving-block bootstrap on 2025 K_year BH survivors")
    print(f"  GOLD_DB: {GOLD_DB_PATH}")
    print(f"  HOLDOUT_SACRED_FROM: {HOLDOUT_SACRED_FROM}")
    print(f"  n_replicas: {N_REPLICAS}  seed: {SEED}")
    print()
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        hdr = ("strategy_id", "n", "L", "obs_expr",
               "parametric_t", "parametric_p", "bootstrap_p",
               "consistent_with_parametric")
        print(f"{hdr[0]:50} {hdr[1]:>4} {hdr[2]:>3} {hdr[3]:>9} "
              f"{hdr[4]:>10} {hdr[5]:>11} {hdr[6]:>11} {hdr[7]:>10}")
        rows = []
        for spec in TARGETS:
            pnl = fetch_year_pnl(con, spec, 2025)
            if len(pnl) == 0:
                print(f"{spec['strategy_id']:50} EMPTY CELL")
                continue
            obs_expr, p_boot, L, n = moving_block_bootstrap_p(pnl, N_REPLICAS, SEED)
            t_par, p_par = parametric_t(pnl)
            # Consistent: same side of q=0.10 threshold
            consistent = (p_par < 0.10) == (p_boot < 0.10)
            rows.append({
                "strategy_id": spec["strategy_id"], "n": n, "L": L,
                "obs_expr": obs_expr, "parametric_t": t_par,
                "parametric_p": p_par, "bootstrap_p": p_boot,
                "consistent": consistent,
            })
            print(f"{spec['strategy_id']:50} {n:4d} {L:3d} "
                  f"{obs_expr:+9.4f} {t_par:+10.3f} {p_par:11.5f} "
                  f"{p_boot:11.5f} {'YES' if consistent else 'NO':>10}")
        print()
        # Summary
        n_consistent = sum(1 for r in rows if r["consistent"])
        print(f"Summary: {n_consistent}/{len(rows)} cells bootstrap-consistent with parametric t at q=0.10")
        if n_consistent == len(rows):
            print("VERDICT: Parametric t for 2025 survivors is NOT serial-correlation-inflated")
            print("         at the per-cell level. The dependence fragility lies at the")
            print("         cross-cell level (BHY-sensitive), not within-cell.")
        else:
            print("VERDICT: At least one 2025 survivor shows bootstrap disagreement with")
            print("         parametric t -- within-cell serial correlation is inflating the")
            print("         t-statistic. Flag as LEAKAGE_SUSPECT per quant-audit-protocol.")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
