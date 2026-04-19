#!/usr/bin/env python3
"""Phase 2.9 A5 hardening — White-style max-t family block bootstrap.

Addresses the third dependence direction that Phase 2.9 had not yet tested:
  - BH-1995: within-family FDR under independence / PRDS (done)
  - BHY-2001: within-family FDR under arbitrary dependency (done)
  - MBB per-cell: within-cell serial correlation (A4, done)
  - THIS FILE: family-level max-|t| null, preserving cross-cell dependence
    via aligned block resampling (White 2000 Bootstrap Reality Check adapted)

Target family: the 3 MNQ COMEX_SETTLE 2025 cells flagged as dependence-
fragile by BHY. They share underlying bars (same session + same instrument
+ same year) and only differ by filter. The skeptical-audit said they're
~1 observation triple-counted. This test gives a non-parametric p-value
for the family's strongest per-cell t-stat under a family-level null that
preserves the cross-cell correlation by resampling block indices JOINTLY
across cells.

### METHOD LABEL (be explicit, per no-bias discipline)

This is NOT Romano-Wolf (2007) FDP-StepM. FDP-StepM controls the False
Discovery Proportion via a step-down multiple testing procedure and
requires the Romano-Wolf algorithm. The canonical Romano-Wolf 2007 paper
is NOT present in resources/ and has not been locally extracted at
`docs/institutional/literature/`. Per `.claude/rules/institutional-rigor.md`
rule 7 ("Ground in local resources before training memory"), implementing
FDP-StepM from training memory would violate institutional-rigor.

What this IS: a White (2000) Bootstrap Reality Check style family-level
test. Chordia et al 2018 extract page 19 describes BRC: "estimate the
sampling distribution of the largest test statistic taking into account
the dependence structure of the individual test statistics, thereby
asymptotically controlling FWER." We adapt to finite-sample using a
moving-block bootstrap that shares block indices across cells of the
same (instrument, session, year) group, so cross-cell dependence on
shared bars is preserved.

Result: a family-level FWER p-value (NOT an FDR or FDP). Stricter than
BH; comparable to BHY in spirit but grounded in the observed dependence
rather than a worst-case c(M) multiplier.

### Target family for this run

The 3 MNQ COMEX_SETTLE 2025 BH_year survivors:
  - MNQ_COMEX_SETTLE_OVNRNG_100 RR1.0 × 2025
  - MNQ_COMEX_SETTLE_X_MES_ATR60 RR1.0 × 2025
  - MNQ_COMEX_SETTLE_OVNRNG_100 RR1.5 × 2025

Plus the SGP 2025 K_year survivor as an independent-family contrast:
  - MNQ_SINGAPORE_OPEN_ATR_P50_O30 RR1.5 × 2025

### Null

For each cell, demean pnl_r within the cell. Family null = max cell
is zero-mean, dependence structure of the observed bars is preserved.

### Bootstrap loop

For each of n_replicas:
  1. Pick a random block-start-index sequence ONCE for the family
     (n_blocks indices, each in [0, max_n - L])
  2. For each cell: reconstruct a demeaned replica by gluing its own
     bars at those block starts. Cells with fewer bars than L*n_blocks
     get trimmed to their own N.
  3. Compute each cell's bootstrap mean; compute max_|t|.

The max |t| distribution under this null is the family-FWER reference.
Observed max |t| across the 3 cells gives the family p-value.

Canonical delegations: same as phase_2_9_block_bootstrap_2025.py.

Usage:
  DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db \\
      uv run --frozen python research/phase_2_9_max_t_family_bootstrap.py
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

# COMEX_SETTLE 2025 family — BHY-fragile cluster
COMEX_FAMILY = [
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
]

# Independent contrast: SGP 2025 (different session, different instrument effect)
SGP_FAMILY = [
    {
        "strategy_id": "MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30",
        "instrument": "MNQ", "orb_label": "SINGAPORE_OPEN",
        "orb_minutes": 30, "rr_target": 1.5, "filter_type": "ATR_P50",
        "direction": "long",
    },
]

N_REPLICAS: int = 10_000
SEED: int = 20260420
TARGET_YEAR: int = 2025


def fetch_year_pnl(con: duckdb.DuckDBPyConnection, spec: dict, year: int) -> np.ndarray:
    """Same canonical-delegation fetch as the per-cell MBB script."""
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


def max_t_family_bootstrap(
    pnl_list: list[np.ndarray], n_replicas: int, seed: int
) -> tuple[list[float], list[float], float, float, int]:
    """White-style family BRC with aligned block bootstrap.

    For each cell, compute observed |t|. Under the family null (all cells
    zero-mean), generate n_replicas max-|t| replicas. p_family = proportion
    of replicas whose max-|t| >= observed max-|t|, continuity-corrected.

    Block length L = max(2, round(min_n^(1/3))) based on the smallest cell.
    Block indices are shared across cells per replica to preserve cross-cell
    correlation on shared bars (to the extent that bars overlap — here all
    3 COMEX_SETTLE cells are same session same instrument same year, so
    their bar-day alignment is nearly 1:1 on filter-fire days).

    Returns (obs_t_list, obs_p_list_parametric, obs_max_t, family_p, block_length).
    """
    from scipy import stats

    if any(len(p) < 4 for p in pnl_list):
        return [], [], float("nan"), float("nan"), 0
    min_n = min(len(p) for p in pnl_list)
    L = max(2, int(round(min_n ** (1.0 / 3.0))))

    obs_t = []
    obs_p_param = []
    for p in pnl_list:
        m = float(p.mean())
        sd = float(p.std(ddof=1))
        n = len(p)
        t = m / (sd / math.sqrt(n)) if sd > 0 else float("nan")
        p_par = float(2.0 * stats.t.sf(abs(t), df=n - 1)) if not math.isnan(t) else float("nan")
        obs_t.append(t)
        obs_p_param.append(p_par)

    obs_max_t = max(abs(t) for t in obs_t)

    # Pre-compute demeaned sliding-window blocks for each cell
    demeaned = [p - p.mean() for p in pnl_list]
    per_cell_blocks = [np.lib.stride_tricks.sliding_window_view(d, L) for d in demeaned]
    per_cell_nblocks = [b.shape[0] for b in per_cell_blocks]
    per_cell_n = [len(p) for p in pnl_list]

    # For aligned resampling: pick block-start indices in the reference (min-n)
    # cell's block space. Cells with more blocks use the same relative-position
    # index clipped to their own range.
    ref_nblocks = min(per_cell_nblocks)
    n_samp_per_cell = [int(math.ceil(n / L)) for n in per_cell_n]

    rng = np.random.default_rng(seed)
    max_t_replicas = np.empty(n_replicas, dtype=np.float64)

    for r in range(n_replicas):
        # Shared ref block-index sequence (length = max n_samp over cells)
        max_samp = max(n_samp_per_cell)
        ref_idx = rng.integers(0, ref_nblocks, size=max_samp)
        rep_t = []
        for i, (blocks, n, n_samp, demean_series) in enumerate(
            zip(per_cell_blocks, per_cell_n, n_samp_per_cell, demeaned, strict=True)
        ):
            nblocks_i = per_cell_nblocks[i]
            # Map ref_idx to this cell's block range
            idx_i = np.clip(ref_idx[:n_samp], 0, nblocks_i - 1)
            rep = blocks[idx_i].reshape(-1)[:n]
            rep_mean = rep.mean()
            sd = float(demean_series.std(ddof=1))
            if sd > 0:
                t_i = rep_mean / (sd / math.sqrt(n))
            else:
                t_i = 0.0
            rep_t.append(abs(t_i))
        max_t_replicas[r] = max(rep_t)

    n_exceed = int(np.sum(max_t_replicas >= obs_max_t))
    p_family = (n_exceed + 1) / (n_replicas + 1)
    return obs_t, obs_p_param, obs_max_t, float(p_family), L


def main() -> int:
    print("Phase 2.9 A5 -- White-style max-t family BRC (NOT Romano-Wolf FDP-StepM)")
    print(f"  GOLD_DB: {GOLD_DB_PATH}")
    print(f"  HOLDOUT_SACRED_FROM: {HOLDOUT_SACRED_FROM}")
    print(f"  n_replicas: {N_REPLICAS}  seed: {SEED}  year: {TARGET_YEAR}")
    print()
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        for family_name, family in [
            ("COMEX_SETTLE_2025 (3 cells, dependence-fragile)", COMEX_FAMILY),
            ("SINGAPORE_OPEN_2025 (1 cell, independent contrast)", SGP_FAMILY),
        ]:
            print(f"=== {family_name} ===")
            pnl_list = [fetch_year_pnl(con, spec, TARGET_YEAR) for spec in family]
            sizes = [len(p) for p in pnl_list]
            print(f"  sizes per cell: {sizes}")
            obs_t, obs_p_param, obs_max_t, p_family, L = max_t_family_bootstrap(
                pnl_list, N_REPLICAS, SEED
            )
            if math.isnan(p_family):
                print("  INSUFFICIENT DATA")
                print()
                continue
            print(f"  block length L: {L}")
            for spec, t, p_par in zip(family, obs_t, obs_p_param, strict=True):
                print(f"    {spec['strategy_id']:55} t={t:+7.3f} parametric_p={p_par:.5f}")
            print(f"  observed max |t| across family: {obs_max_t:.3f}")
            print(f"  family FWER p (White-BRC-style): {p_family:.5f}")
            if p_family < 0.05:
                print("  VERDICT: family max-|t| significant at 5% FWER")
            elif p_family < 0.10:
                print("  VERDICT: family max-|t| significant at 10% FWER")
            else:
                print("  VERDICT: family max-|t| NOT significant at 10% FWER")
            print()
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
