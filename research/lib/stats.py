"""Statistical tests and metrics for research scripts.

Canonical implementations -- use these instead of inline copies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _scipy_stats():
    """Lazy import of scipy.stats — avoids import-time dependency."""
    from scipy import stats as _sp
    return _sp


def ttest_1s(arr, mu: float = 0.0) -> tuple[int, float, float, float, float]:
    """One-sample t-test. Returns (n, mean, win_rate, t_stat, p_value).

    NaN values are stripped. Returns NaN for t/p if n < 10.

    Extracted from research_compressed_spring.py:71 and research_avoid_crosscheck.py:68
    (identical implementations).
    """
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    if n < 10:
        return n, float(a.mean()), float((a > 0).mean()), float("nan"), float("nan")
    t, p = _scipy_stats().ttest_1samp(a, mu)
    return n, float(a.mean()), float((a > 0).mean()), float(t), float(p)


def bh_fdr(p_values: list | np.ndarray, q: float = 0.10) -> set[int]:
    """Benjamini-Hochberg FDR correction. Returns set of rejected hypothesis indices.

    Extracted from research_compressed_spring.py:90 and research_avoid_crosscheck.py:77
    (identical implementations).
    """
    n = len(p_values)
    if n == 0:
        return set()
    ranked = sorted(enumerate(p_values), key=lambda x: x[1])
    thresholds = [q * (k + 1) / n for k in range(n)]
    max_k = -1
    for k, (_, p) in enumerate(ranked):
        if p <= thresholds[k]:
            max_k = k
    if max_k < 0:
        return set()
    return {idx for idx, _ in ranked[: max_k + 1]}


def mannwhitney_2s(a, b) -> tuple[float, float]:
    """Two-sample Mann-Whitney U test. Returns (u_stat, p_value)."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan")
    u, p = _scipy_stats().mannwhitneyu(a, b, alternative="two-sided")
    return float(u), float(p)


def compute_metrics(pnls) -> dict | None:
    """Compute standard strategy metrics. Returns None if all NaN.

    Returns dict with keys: n, win_rate, avg_r, sharpe, max_dd, total_r.
    """
    a = np.array(pnls, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return None
    n = len(a)
    total_r = float(a.sum())
    avg_r = float(a.mean())
    win_rate = float((a > 0).mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    sharpe = avg_r / std if std > 0 else 0.0
    # Max drawdown in R-units
    cumulative = np.cumsum(a)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_dd = float(drawdown.max()) if len(drawdown) > 0 else 0.0
    return {
        "n": n,
        "win_rate": round(win_rate, 4),
        "avg_r": round(avg_r, 4),
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 4),
        "total_r": round(total_r, 4),
    }


def year_by_year(
    df: pd.DataFrame, date_col: str = "trading_day", value_col: str = "pnl_r"
) -> pd.DataFrame:
    """Per-year breakdown: n, mean, win_rate, p_value.

    Returns DataFrame with columns: year, n, mean, win_rate, p_value.
    """
    df = df.copy()
    df["_year"] = pd.to_datetime(df[date_col]).dt.year
    rows = []
    for year, group in df.groupby("_year"):
        vals = group[value_col].dropna().values
        n, mean, wr, _, p = ttest_1s(vals)
        rows.append({"year": int(year), "n": n, "mean": round(mean, 4),
                      "win_rate": round(wr, 4), "p_value": round(p, 6) if not np.isnan(p) else None})
    return pd.DataFrame(rows)


def expanding_stat(
    df: pd.DataFrame, col: str, min_periods: int = 20
) -> pd.Series:
    """Expanding-window mean (no lookahead). Returns Series aligned to df index."""
    return df[col].expanding(min_periods=min_periods).mean()
