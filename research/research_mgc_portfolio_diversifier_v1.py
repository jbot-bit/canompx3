"""H1 — MGC Portfolio-Diversifier Correlation + Sharpe Lift.

Pre-reg: docs/audit/hypotheses/2026-04-20-mgc-portfolio-diversifier.yaml

Question: does adding a hypothetical MGC return stream at 10% weight
to the 38-lane active book produce ΔSharpe ≥ 0.05 annualized,
conditional on max pairwise correlation < 0.50?

Theory: Markowitz (1952) mean-variance. Zero-alpha asset with
correlation ρ < 1 adds diversification benefit proportional to (1−ρ).

Pass: ΔSR ≥ 0.05 AND max_corr < 0.50
Kill: either threshold violated, or MGC destroys book Sharpe at any
weight in [5%, 20%].

No OOS peek. No parameter tuning. IS-structure test only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH

OUTPUT_DIR = PROJECT_ROOT / "research" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IS_START = "2022-06-13"  # MGC real-micro launch
IS_END = "2026-01-01"    # sacred holdout start

MGC_WEIGHT = 0.10
RNG = np.random.default_rng(20260420)


def connect():
    return duckdb.connect(str(GOLD_DB_PATH), read_only=True)


def load_active_lanes(con) -> pd.DataFrame:
    """38 active validated_setups with the dimensions needed to recompute daily pnl."""
    return con.execute(
        """
        SELECT strategy_id, instrument, orb_label, orb_minutes,
               entry_model, rr_target, confirm_bars, filter_type
        FROM validated_setups
        WHERE status='active'
        ORDER BY instrument, orb_label, orb_minutes, filter_type, strategy_id
        """
    ).df()


def load_lane_daily_returns(con, lane) -> pd.DataFrame:
    """Aggregate orb_outcomes pnl_r by trading_day for ONE lane within the IS window.

    We do NOT apply the filter here — we rely on ``validated_setups.expectancy_r``
    being the canonical stored value and approximate each lane's daily stream
    with unfiltered orb_outcomes matched on session + aperture + entry_model +
    rr_target + confirm_bars. Filter_type selection matters for per-trade
    inclusion, but for a portfolio-structure (correlation) test we need the
    WHOLE lane's available daily days, not just filter-firing days.

    CAVEAT documented in the result doc: this overestimates N per lane and
    includes days the deployed filter would have blocked. Acceptable for a
    CORRELATION measurement (which is about timing, not magnitude) but means
    absolute Sharpe figures must be read as ENVELOPES, not exact book-reality.
    """
    return con.execute(
        """
        SELECT trading_day::DATE AS d, SUM(pnl_r) AS pnl_r
        FROM orb_outcomes
        WHERE symbol = ?
          AND orb_label = ?
          AND orb_minutes = ?
          AND entry_model = ?
          AND rr_target = ?
          AND confirm_bars = ?
          AND trading_day >= ?::DATE
          AND trading_day <  ?::DATE
          AND pnl_r IS NOT NULL
        GROUP BY trading_day::DATE
        ORDER BY 1
        """,
        [
            lane["instrument"],
            lane["orb_label"],
            int(lane["orb_minutes"]),
            lane["entry_model"],
            float(lane["rr_target"]),
            int(lane["confirm_bars"]),
            IS_START,
            IS_END,
        ],
    ).df()


def load_mgc_daily_stream(con, filter_label: str) -> pd.DataFrame:
    """MGC daily ExpR, session-rotated (all sessions × 5-min aperture, E2 CB1, RR1.0).

    Two variants:
    - ``raw``: no filter, every ORB break trade.
    - ``ovnrng``: OVNRNG_100 pre-session price-safe filter.

    Aggregates pnl_r across sessions on the same trading_day so the series
    is a single "daily MGC return" per day, one-contract-equivalent per
    signal. The sum rather than mean because different sessions produce
    different numbers of fires; summing matches a portfolio-of-sessions
    construction. Trading days with no fires get ZERO (not NaN) to respect
    the daily calendar the book trades on.
    """
    if filter_label == "raw":
        q = """
            SELECT trading_day::DATE AS d,
                   SUM(pnl_r) AS pnl_r,
                   COUNT(*) AS n_fires
            FROM orb_outcomes
            WHERE symbol='MGC'
              AND orb_minutes = 5
              AND entry_model = 'E2'
              AND confirm_bars = 1
              AND rr_target = 1.0
              AND trading_day >= ?::DATE
              AND trading_day <  ?::DATE
              AND pnl_r IS NOT NULL
            GROUP BY trading_day::DATE
            ORDER BY 1
        """
        params = [IS_START, IS_END]
    elif filter_label == "ovnrng_100":
        # OVNRNG_100 = overnight_range >= 100 pts. Canonical OvernightRangeAbsFilter.
        q = """
            SELECT o.trading_day::DATE AS d,
                   SUM(o.pnl_r) AS pnl_r,
                   COUNT(*) AS n_fires
            FROM orb_outcomes o
            JOIN daily_features d
              ON o.trading_day = d.trading_day
             AND o.symbol = d.symbol
             AND o.orb_minutes = d.orb_minutes
            WHERE o.symbol='MGC'
              AND o.orb_minutes = 5
              AND o.entry_model = 'E2'
              AND o.confirm_bars = 1
              AND o.rr_target = 1.0
              AND o.trading_day >= ?::DATE
              AND o.trading_day <  ?::DATE
              AND o.pnl_r IS NOT NULL
              AND d.overnight_range >= 100.0
            GROUP BY o.trading_day::DATE
            ORDER BY 1
        """
        params = [IS_START, IS_END]
    else:
        raise ValueError(f"unknown filter_label: {filter_label}")
    return con.execute(q, params).df()


def build_calendar(con) -> pd.DataFrame:
    """Distinct trading_days in IS window across all three instruments."""
    return con.execute(
        """
        SELECT DISTINCT trading_day::DATE AS d
        FROM orb_outcomes
        WHERE symbol IN ('MGC','MNQ','MES')
          AND trading_day >= ?::DATE
          AND trading_day <  ?::DATE
        ORDER BY 1
        """,
        [IS_START, IS_END],
    ).df()


def align_to_calendar(daily_df: pd.DataFrame, cal: pd.DataFrame) -> np.ndarray:
    """Return a 1-D numpy array aligned to `cal.d`, zero on missing days."""
    if daily_df.empty:
        return np.zeros(len(cal), dtype=float)
    merged = cal.merge(daily_df, on="d", how="left")
    return merged["pnl_r"].fillna(0.0).to_numpy()


def ann_sharpe(returns: np.ndarray, trading_days_per_year: int = 252) -> float:
    """Annualized Sharpe, zero if insufficient signal."""
    if returns.std(ddof=1) == 0 or len(returns) < 2:
        return 0.0
    return (returns.mean() / returns.std(ddof=1)) * np.sqrt(trading_days_per_year)


def block_bootstrap_ci(
    returns: np.ndarray,
    stat_fn,
    block: int = 5,
    n_boot: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Moving-block bootstrap CI for a scalar statistic."""
    n = len(returns)
    if n < block:
        return (float("nan"), float("nan"))
    n_blocks = int(np.ceil(n / block))
    stats = np.empty(n_boot, dtype=float)
    idx_base = np.arange(block)
    starts_max = n - block + 1
    for b in range(n_boot):
        starts = RNG.integers(0, starts_max, size=n_blocks)
        idx = (starts[:, None] + idx_base[None, :]).ravel()[:n]
        stats[b] = stat_fn(returns[idx])
    lo = np.nanquantile(stats, (1 - ci) / 2)
    hi = np.nanquantile(stats, 1 - (1 - ci) / 2)
    return float(lo), float(hi)


def per_year_delta_sharpe(
    book: np.ndarray,
    mgc: np.ndarray,
    dates: pd.Series,
    weight: float,
) -> dict:
    """ΔSR per calendar year with zero-forward-look."""
    years = dates.dt.year.to_numpy()
    out = {}
    for y in sorted(set(years)):
        mask = years == y
        if mask.sum() < 30:
            continue
        sr_book = ann_sharpe(book[mask])
        combo = (1 - weight) * book[mask] + weight * mgc[mask]
        sr_combo = ann_sharpe(combo)
        out[int(y)] = {"sr_book": sr_book, "sr_combo": sr_combo, "delta": sr_combo - sr_book}
    return out


def main() -> None:
    con = connect()
    try:
        cal = build_calendar(con)
        print(f"Trading-day calendar in IS ({IS_START} → {IS_END}): {len(cal)} days")

        lanes = load_active_lanes(con)
        print(f"Active lanes: {len(lanes)}")

        lane_series = {}
        coverage_report = []
        for _, lane in lanes.iterrows():
            df = load_lane_daily_returns(con, lane)
            arr = align_to_calendar(df, cal)
            lane_series[lane["strategy_id"]] = arr
            coverage_report.append(
                {
                    "strategy_id": lane["strategy_id"],
                    "instrument": lane["instrument"],
                    "session": lane["orb_label"],
                    "orb_min": int(lane["orb_minutes"]),
                    "rr": float(lane["rr_target"]),
                    "filter": lane["filter_type"],
                    "trading_days_with_data": int((arr != 0).sum()),
                    "mean_daily_r": float(arr.mean()),
                    "std_daily_r": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                }
            )

        # Book = equal-weighted sum of 38 lane streams / 38
        book_matrix = np.column_stack([lane_series[s] for s in lane_series])
        book = book_matrix.mean(axis=1)
        print(
            f"Book daily stream: mean={book.mean():+.4f}R "
            f"std={book.std(ddof=1):.4f} sharpe={ann_sharpe(book):+.2f}"
        )

        # MGC streams
        results = {}
        for filter_label in ("raw", "ovnrng_100"):
            mgc_df = load_mgc_daily_stream(con, filter_label)
            if mgc_df.empty:
                print(f"  MGC {filter_label}: EMPTY — insufficient data")
                results[filter_label] = {"empty": True}
                continue

            mgc = align_to_calendar(mgc_df, cal)
            mgc_active_days = int((mgc != 0).sum())
            print(
                f"  MGC {filter_label}: active days={mgc_active_days}, "
                f"mean={mgc.mean():+.4f}R std={mgc.std(ddof=1):.4f} "
                f"sharpe={ann_sharpe(mgc):+.2f}"
            )

            # Pairwise correlations MGC vs each lane
            pairwise = {}
            for strategy_id, arr in lane_series.items():
                if arr.std(ddof=1) == 0 or mgc.std(ddof=1) == 0:
                    pairwise[strategy_id] = float("nan")
                    continue
                pairwise[strategy_id] = float(np.corrcoef(arr, mgc)[0, 1])
            corr_vals = np.array([v for v in pairwise.values() if not np.isnan(v)])
            max_corr = float(np.nanmax(np.abs(corr_vals)))
            mean_corr = float(np.nanmean(corr_vals))
            median_corr = float(np.nanmedian(corr_vals))
            print(
                f"  MGC {filter_label} vs 38 lanes: "
                f"max|corr|={max_corr:.3f} mean={mean_corr:+.3f} median={median_corr:+.3f}"
            )

            # Sharpe lift at weight sweep
            def sr_combo(returns_mgc, returns_book, w):
                combo = (1 - w) * returns_book + w * returns_mgc
                return ann_sharpe(combo)

            sr_book = ann_sharpe(book)
            weight_sweep = {}
            for w in (0.05, 0.10, 0.15, 0.20):
                sr_w = sr_combo(mgc, book, w)
                weight_sweep[w] = {"sr_combo": sr_w, "delta": sr_w - sr_book}

            # Bootstrap CI at the pre-registered 10% weight
            def delta_at_10(returns):
                half = len(returns) // 2
                b = returns[:half]
                m = returns[half:]
                if len(b) != len(m):
                    minn = min(len(b), len(m))
                    b = b[:minn]; m = m[:minn]
                return ann_sharpe((1 - MGC_WEIGHT) * b + MGC_WEIGHT * m) - ann_sharpe(b)

            # Better bootstrap: stack (book, mgc) pairs and block-resample
            pairs = np.column_stack([book, mgc])
            def delta_paired(pairs_arr):
                b = pairs_arr[:, 0]
                m = pairs_arr[:, 1]
                sr_combo_b = ann_sharpe((1 - MGC_WEIGHT) * b + MGC_WEIGHT * m)
                sr_b = ann_sharpe(b)
                return sr_combo_b - sr_b

            ci_lo, ci_hi = block_bootstrap_ci(pairs, delta_paired, block=5, n_boot=1000)

            # Per-year breakdown
            per_year = per_year_delta_sharpe(book, mgc, cal["d"], MGC_WEIGHT)

            results[filter_label] = {
                "empty": False,
                "mgc_stream_stats": {
                    "active_days": mgc_active_days,
                    "mean_r": float(mgc.mean()),
                    "std_r": float(mgc.std(ddof=1)),
                    "sharpe": ann_sharpe(mgc),
                },
                "correlations": {
                    "max_abs": max_corr,
                    "mean": mean_corr,
                    "median": median_corr,
                    "per_lane": pairwise,
                },
                "sharpe_lift": {
                    "sr_book": sr_book,
                    "weight_sweep": {f"{int(k * 100)}%": v for k, v in weight_sweep.items()},
                    "bootstrap_ci_at_10pct": {"lo": ci_lo, "hi": ci_hi},
                    "per_year": per_year,
                },
            }

        # Pre-reg kill criteria evaluation
        verdict = {}
        for fl, r in results.items():
            if r.get("empty"):
                verdict[fl] = "INSUFFICIENT_DATA"
                continue
            max_corr = r["correlations"]["max_abs"]
            delta_at_10 = r["sharpe_lift"]["weight_sweep"]["10%"]["delta"]
            any_destructive = any(
                v["delta"] < -0.05 for v in r["sharpe_lift"]["weight_sweep"].values()
            )
            if any_destructive:
                verdict[fl] = "KILL_C3_destructive"
            elif max_corr >= 0.50:
                verdict[fl] = "KILL_C1_correlation_too_high"
            elif delta_at_10 < 0.05:
                verdict[fl] = "KILL_C2_insufficient_sharpe_lift"
            else:
                verdict[fl] = "PASS_provisional"

        summary = {
            "hypothesis_id": "mgc-portfolio-diversifier",
            "run_date": "2026-04-20",
            "is_window": [IS_START, IS_END],
            "n_trading_days": len(cal),
            "n_active_lanes": len(lanes),
            "book_baseline_sharpe": ann_sharpe(book),
            "verdicts": verdict,
            "filter_variants": list(results.keys()),
            "kill_criteria": {
                "C1": "max|corr| < 0.50",
                "C2": "ΔSR_at_10%_weight ≥ 0.05",
                "C3": "no destructive (<-0.05) at any weight 5-20%",
            },
        }

        out_json = OUTPUT_DIR / "mgc_portfolio_diversifier_v1.json"
        out_json.write_text(
            json.dumps({"summary": summary, "results": results, "coverage": coverage_report}, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nWrote {out_json}")

        print("\nVERDICTS:")
        for fl, v in verdict.items():
            print(f"  {fl}: {v}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
