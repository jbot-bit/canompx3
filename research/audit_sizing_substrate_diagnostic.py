# research/audit_sizing_substrate_diagnostic.py
"""Sizing-substrate Stage-1 diagnostic.

Per docs/plans/2026-04-27-sizing-substrate-diagnostic-design.md v0.2 and the
locked pre-reg YAML at docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml.

Read-only over gold.db. Raises on any 2026 row.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def is_holdout_clean(df: pd.DataFrame, holdout: str) -> bool:
    """Return True iff no row's trading_day >= holdout. Raise RuntimeError otherwise."""
    holdout_ts = pd.Timestamp(holdout)
    if (pd.to_datetime(df["trading_day"]) >= holdout_ts).any():
        raise RuntimeError(f"holdout row detected (>= {holdout})")
    return True


def null_coverage_mark(f: pd.Series, threshold: float) -> tuple[str, float]:
    """Return ('OK', drop_frac) if drop_frac<=threshold, else ('INVALID', drop_frac)."""
    drop_frac = float(f.isna().mean())
    return ("OK" if drop_frac <= threshold else "INVALID", drop_frac)


def power_floor_mark(n: int, min_n: int) -> str:
    return "OK" if n >= min_n else "UNDERPOWERED"


def compute_quintile_lift(df: pd.DataFrame, feature_col: str, outcome_col: str) -> dict[str, Any]:
    """Bin into quintiles by feature_col, return per-quintile mean outcome + monotonic flag."""
    quintiles = pd.qcut(df[feature_col], q=5, labels=False, duplicates="drop")
    means = df.groupby(quintiles)[outcome_col].mean().sort_index().tolist()
    if len(means) < 5:
        return {"q_means": means, "monotonic": False, "q1_mean_r": float("nan"),
                "q5_mean_r": float("nan"), "q5_minus_q1": 0.0}
    inc = all(means[i] <= means[i + 1] for i in range(4))
    dec = all(means[i] >= means[i + 1] for i in range(4))
    return {"q_means": means, "monotonic": inc or dec,
            "q1_mean_r": means[0], "q5_mean_r": means[4],
            "q5_minus_q1": means[4] - means[0]}


def bootstrap_sized_vs_flat_ci(
    df: pd.DataFrame, feature_col: str, outcome_col: str,
    weights: tuple[float, float, float, float, float],
    predicted_sign: str, B: int, seed: int, alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap mean(sized) - mean(flat). Weights applied per-quintile in predicted-sign direction."""
    quintiles = pd.qcut(df[feature_col], q=5, labels=False, duplicates="drop")
    pnl = df[outcome_col].to_numpy()
    qarr = quintiles.to_numpy()
    w_pos = np.array(weights)
    w_neg = w_pos[::-1]
    w = w_pos if predicted_sign == "+" else w_neg
    weight_per_trade = w[qarr]  # mean is 1.0 by construction since qcut bins are equi-count
    sized_pnl = pnl * weight_per_trade
    delta_obs = sized_pnl.mean() - pnl.mean()

    rng = np.random.default_rng(seed)
    n = len(pnl)
    deltas = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        deltas[b] = sized_pnl[idx].mean() - pnl[idx].mean()
    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    return {"observed": float(delta_obs), "lo": lo, "hi": hi}


def apply_bh_fdr(pvals: list[float], q: float) -> list[bool]:
    """Benjamini-Hochberg FDR control. Returns list of pass/fail in original order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = q * (np.arange(1, m + 1)) / m
    passes_sorted = ranked <= thresholds
    if passes_sorted.any():
        max_k = int(np.max(np.where(passes_sorted)[0]))
        passes_sorted = np.arange(m) <= max_k
    out = np.zeros(m, dtype=bool)
    out[order] = passes_sorted
    return out.tolist()


def check_forecast_stability(
    df: pd.DataFrame, feature_col: str, window: int, max_rel_var: float,
) -> str:
    """STABLE if rolling SD relative-variation <= max_rel_var, else UNSTABLE."""
    rolling_sd = df[feature_col].rolling(window=window, min_periods=window // 2).std().dropna()
    if len(rolling_sd) < 2:
        return "STABLE"  # too short to flag instability
    sd_med = rolling_sd.median()
    if sd_med <= 0:
        return "STABLE"
    rel_var = (rolling_sd.max() - rolling_sd.min()) / sd_med
    return "STABLE" if rel_var <= max_rel_var else "UNSTABLE"


def sign_match_split_half(df: pd.DataFrame, feature_col: str, outcome_col: str) -> bool:
    """Split by median trading_day; require Spearman rho sign match in both halves."""
    df_sorted = df.sort_values("trading_day").reset_index(drop=True)
    median_day = df_sorted["trading_day"].iloc[len(df_sorted) // 2]
    h1 = df_sorted[df_sorted["trading_day"] < median_day]
    h2 = df_sorted[df_sorted["trading_day"] >= median_day]
    if len(h1) < 30 or len(h2) < 30:
        return False
    r1, _ = stats.spearmanr(h1[feature_col], h1[outcome_col])
    r2, _ = stats.spearmanr(h2[feature_col], h2[outcome_col])
    return bool(np.sign(r1) == np.sign(r2) and r1 != 0 and r2 != 0)


def classify_cell(cell: dict[str, Any]) -> str:
    """Return PASS / FAIL / INVALID per spec §5.3 + §5.4a."""
    if cell.get("null_status") == "INVALID":
        return "INVALID"
    if cell.get("power_status") == "UNDERPOWERED":
        return "FAIL"
    required = [
        abs(cell.get("rho", 0.0)) >= 0.10,
        cell.get("bh_fdr_pass") is True,
        cell.get("monotonic") is True,
        abs(cell.get("q5_minus_q1", 0.0)) >= 0.20,
        cell.get("sized_flat_delta_lo", 0.0) > 0,
        cell.get("split_half_rho_match") is True,
        cell.get("split_half_delta_match") is True,
        cell.get("predicted_sign") == cell.get("realized_sign"),
    ]
    return "PASS" if all(required) else "FAIL"
