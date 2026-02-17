"""
Strategy reverse-engineering: match a trade log's flip behavior to known strategy families.

Methodology:
- Extract flip timestamps + directions from trade log
- Compute candidate strategy signals on 5m bars
- Score by behavioral similarity (flip matching), NOT PnL
- Tolerance: +/-1 bar (5 minutes)

Families:
  A: Price MA crossover (fast/slow)
  B: RSI vs RSI-MA crossover
  C: Trend slope + pullback
  D: Chop-avoidance wrapper (ATR filter on top of A/B/C)
"""

from pathlib import Path
from datetime import timedelta
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trade_log(path: str | Path) -> pd.DataFrame:
    """Parse DT trade log CSV into flip DataFrame.

    Returns DataFrame with columns: trade_num, ts, direction, price
    """
    df = pd.read_csv(path)
    entries = df[df["Type"].str.startswith("Entry")].copy()
    entries["direction"] = entries["Type"].str.replace("Entry ", "")
    entries["ts"] = pd.to_datetime(entries["Date and time"])
    entries["trade_num"] = entries["Trade #"]
    entries["price"] = entries["Price USD"]
    return entries[["trade_num", "ts", "direction", "price"]].reset_index(drop=True)

def load_bars_5m(start: str, end: str, db_path: Path | None = None) -> pd.DataFrame:
    """Load 5m bars from DB for the given date range."""
    import duckdb
    path = db_path or GOLD_DB_PATH
    con = duckdb.connect(str(path), read_only=True)
    try:
        df = con.execute("""
            SELECT ts_utc AS ts, open, high, low, close, volume
            FROM bars_5m
            WHERE symbol = 'MGC'
              AND ts_utc >= ?
              AND ts_utc <= ?
            ORDER BY ts_utc
        """, [start, end]).fetchdf()
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
        return df
    finally:
        con.close()

# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

def compute_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    """Add all candidate indicators to bars DataFrame."""
    df = bars.copy()
    c = df["close"]

    # Moving averages
    for period in [5, 8, 10, 13, 20, 30, 50, 100]:
        df[f"ma_{period}"] = c.rolling(period).mean()

    # RSI (Wilder's)
    for period in [7, 9, 14]:
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # RSI MAs
    for rsi_len in [7, 9, 14]:
        for ma_len in [5, 9, 14]:
            df[f"rsi_{rsi_len}_ma_{ma_len}"] = df[f"rsi_{rsi_len}"].rolling(ma_len).mean()

    # ATR (for chop detection)
    h, lo, prev_c = df["high"], df["low"], c.shift(1)
    tr = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pctile"] = df["atr_14"].rolling(100).rank(pct=True)

    # MA slopes (5-bar delta)
    for period in [20, 50]:
        df[f"ma_{period}_slope"] = df[f"ma_{period}"].diff(5)

    return df

# ---------------------------------------------------------------------------
# Candidate strategy families
# ---------------------------------------------------------------------------

def family_a_signals(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """Family A: MA crossover. Returns direction series ('long'/'short')."""
    direction = pd.Series("short", index=df.index)
    direction[df[f"ma_{fast}"] > df[f"ma_{slow}"]] = "long"
    return direction

def family_b_signals(df: pd.DataFrame, rsi_len: int, ma_len: int) -> pd.Series:
    """Family B: RSI vs RSI-MA crossover."""
    col_rsi = f"rsi_{rsi_len}"
    col_ma = f"rsi_{rsi_len}_ma_{ma_len}"
    direction = pd.Series("short", index=df.index)
    direction[df[col_rsi] > df[col_ma]] = "long"
    return direction

def family_c_signals(
    df: pd.DataFrame, trend_ma: int, rsi_len: int, rsi_long_thresh: float, rsi_short_thresh: float,
) -> pd.Series:
    """Family C: Trend slope + pullback. Long when trend up + RSI pullback recovers."""
    slope_col = f"ma_{trend_ma}_slope"
    rsi_col = f"rsi_{rsi_len}"
    direction = pd.Series("short", index=df.index)
    # Long when slope > 0 AND RSI above long threshold
    long_mask = (df[slope_col] > 0) & (df[rsi_col] > rsi_long_thresh)
    # Short when slope < 0 AND RSI below short threshold
    short_mask = (df[slope_col] < 0) & (df[rsi_col] < rsi_short_thresh)
    direction[long_mask] = "long"
    direction[short_mask] = "short"
    # Ambiguous bars: carry forward previous direction
    ambiguous = ~long_mask & ~short_mask
    direction[ambiguous] = np.nan
    direction = direction.ffill().fillna("short")
    return direction

def extract_flips(direction: pd.Series, timestamps: pd.Series) -> pd.DataFrame:
    """Extract flip points from a direction series.

    Returns DataFrame with columns: ts, direction (the NEW direction after flip)
    """
    changed = direction != direction.shift(1)
    # Skip the first bar (no prior to compare)
    changed.iloc[0] = False
    flips = pd.DataFrame({
        "ts": timestamps[changed].values,
        "direction": direction[changed].values,
    })
    return flips.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class MatchScore:
    family: str
    params: dict
    flip_hit_rate: float       # % of real flips matched within tolerance
    false_flip_rate: float     # candidate flips with no matching real flip
    direction_accuracy: float  # correct direction when matched
    mean_timing_error: float   # mean bars early/late
    total_candidate_flips: int
    total_real_flips: int
    matched_count: int
    composite: float           # weighted score

def score_candidate(
    real_flips: pd.DataFrame,
    candidate_flips: pd.DataFrame,
    tolerance_bars: int = 1,
    bar_duration_minutes: int = 5,
) -> dict:
    """Score a candidate strategy against real flips.

    Args:
        real_flips: DataFrame with ts, direction
        candidate_flips: DataFrame with ts, direction
        tolerance_bars: +/- N bars for matching
        bar_duration_minutes: bar size in minutes

    Returns dict with scoring metrics.
    """
    tolerance = timedelta(minutes=tolerance_bars * bar_duration_minutes)
    n_real = len(real_flips)
    n_cand = len(candidate_flips)

    if n_real == 0 or n_cand == 0:
        return {
            "flip_hit_rate": 0.0,
            "false_flip_rate": 1.0,
            "direction_accuracy": 0.0,
            "mean_timing_error": float("inf"),
            "matched_count": 0,
            "total_real_flips": n_real,
            "total_candidate_flips": n_cand,
            "composite": 0.0,
        }

    real_ts = real_flips["ts"].values.astype("datetime64[ns]")
    real_dirs = real_flips["direction"].values
    cand_ts = candidate_flips["ts"].values.astype("datetime64[ns]")
    cand_dirs = candidate_flips["direction"].values

    tol_ns = int(tolerance.total_seconds() * 1e9)

    # For each real flip, find closest candidate flip within tolerance
    matched_real = 0
    matched_direction_correct = 0
    timing_errors = []
    cand_matched_indices = set()

    for i in range(n_real):
        diffs = np.abs(cand_ts.astype(np.int64) - real_ts[i].astype(np.int64))
        min_idx = np.argmin(diffs)
        min_diff = diffs[min_idx]

        if min_diff <= tol_ns:
            matched_real += 1
            cand_matched_indices.add(min_idx)
            timing_errors.append(min_diff / 1e9 / 60 / bar_duration_minutes)  # in bars
            if cand_dirs[min_idx] == real_dirs[i]:
                matched_direction_correct += 1

    flip_hit_rate = matched_real / n_real if n_real > 0 else 0.0
    false_flips = n_cand - len(cand_matched_indices)
    false_flip_rate = false_flips / n_cand if n_cand > 0 else 0.0
    direction_accuracy = matched_direction_correct / matched_real if matched_real > 0 else 0.0
    mean_timing = np.mean(timing_errors) if timing_errors else float("inf")

    # Composite: weighted sum (higher = better)
    # Hit rate most important, then direction, penalize false flips
    composite = (
        0.40 * flip_hit_rate
        + 0.30 * direction_accuracy
        + 0.20 * (1.0 - false_flip_rate)
        + 0.10 * max(0, 1.0 - mean_timing)  # closer to 0 timing error = better
    )

    return {
        "flip_hit_rate": round(flip_hit_rate, 4),
        "false_flip_rate": round(false_flip_rate, 4),
        "direction_accuracy": round(direction_accuracy, 4),
        "mean_timing_error": round(mean_timing, 2),
        "matched_count": matched_real,
        "total_real_flips": n_real,
        "total_candidate_flips": n_cand,
        "composite": round(composite, 4),
    }

# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid_search(
    bars: pd.DataFrame,
    real_flips: pd.DataFrame,
    tolerance_bars: int = 1,
) -> list[MatchScore]:
    """Run all candidate families against real flips. Returns sorted results."""
    df = compute_indicators(bars)
    ts = df["ts"]
    results = []

    # Family A: MA crossover
    fast_periods = [5, 8, 10, 13, 20]
    slow_periods = [30, 50, 100]
    for fast in fast_periods:
        for slow in slow_periods:
            if fast >= slow:
                continue
            direction = family_a_signals(df, fast, slow)
            cand_flips = extract_flips(direction, ts)
            score = score_candidate(real_flips, cand_flips, tolerance_bars)
            results.append(MatchScore(
                family="A_MA_Cross",
                params={"fast": fast, "slow": slow},
                **score,
            ))

    # Family B: RSI vs RSI-MA
    for rsi_len in [7, 9, 14]:
        for ma_len in [5, 9, 14]:
            direction = family_b_signals(df, rsi_len, ma_len)
            cand_flips = extract_flips(direction, ts)
            score = score_candidate(real_flips, cand_flips, tolerance_bars)
            results.append(MatchScore(
                family="B_RSI_Cross",
                params={"rsi": rsi_len, "ma": ma_len},
                **score,
            ))

    # Family C: Trend slope + pullback
    for trend_ma in [20, 50]:
        for rsi_len in [7, 9, 14]:
            for long_thresh in [40, 45, 50]:
                short_thresh = 100 - long_thresh  # symmetric
                direction = family_c_signals(df, trend_ma, rsi_len, long_thresh, short_thresh)
                cand_flips = extract_flips(direction, ts)
                score = score_candidate(real_flips, cand_flips, tolerance_bars)
                results.append(MatchScore(
                    family="C_Trend_PB",
                    params={"trend_ma": trend_ma, "rsi": rsi_len, "long_th": long_thresh},
                    **score,
                ))

    # Family D: Chop wrapper on top A/B winners (applied after initial sort)
    # We'll do this as a post-filter on top-scoring A/B candidates

    # Sort by composite score descending
    results.sort(key=lambda x: x.composite, reverse=True)
    return results

def apply_chop_filter(
    bars_with_indicators: pd.DataFrame,
    base_direction: pd.Series,
    atr_pctile_min: float = 0.3,
) -> pd.Series:
    """Family D: suppress flips during low-vol regimes.

    When ATR percentile < threshold, carry forward the previous direction
    (don't flip in chop).
    """
    filtered = base_direction.copy()
    low_vol = bars_with_indicators["atr_pctile"] < atr_pctile_min
    filtered[low_vol] = np.nan
    filtered = filtered.ffill().fillna("short")
    return filtered

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    trade_log_path = Path(__file__).resolve().parent.parent.parent / "DT_V2_COMEX_MINI_MGC1!_2026-02-10.csv"

    print("Loading trade log...")
    real_flips = load_trade_log(trade_log_path)
    print(f"  {len(real_flips)} flips: {real_flips['ts'].min()} to {real_flips['ts'].max()}")

    # Pad date range by 100 bars (~500min) for indicator warmup
    start = (real_flips["ts"].min() - timedelta(days=3)).strftime("%Y-%m-%d")
    end = (real_flips["ts"].max() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
    print(f"Loading 5m bars from DB: {start} to {end}...")
    bars = load_bars_5m(start, end)
    print(f"  {len(bars)} bars loaded")

    if bars.empty:
        print("ERROR: No bars found in DB for this date range.")
        return

    print("\nRunning grid search (tolerance: +/-1 bar = +/-5min)...")
    results = run_grid_search(bars, real_flips, tolerance_bars=1)
    print(f"  {len(results)} candidates scored\n")

    # Top 15
    print("=" * 90)
    print(f"{'Rank':<5} {'Family':<15} {'Params':<35} {'Hit%':<7} {'DirAcc':<7} {'FalseF':<7} {'TimErr':<7} {'Score':<7}")
    print("=" * 90)
    for i, r in enumerate(results[:15], 1):
        params_str = str(r.params)
        print(
            f"{i:<5} {r.family:<15} {params_str:<35} "
            f"{r.flip_hit_rate:<7.1%} {r.direction_accuracy:<7.1%} "
            f"{r.false_flip_rate:<7.1%} {r.mean_timing_error:<7.2f} "
            f"{r.composite:<7.4f}"
        )

    # Detail on top 3
    print("\n" + "=" * 90)
    print("TOP 3 DETAIL")
    print("=" * 90)
    for i, r in enumerate(results[:3], 1):
        print(f"\n#{i}: {r.family} {r.params}")
        print(f"  Flips matched: {r.matched_count}/{r.total_real_flips} real flips")
        print(f"  Candidate produced: {r.total_candidate_flips} flips total")
        print(f"  False flips: {r.total_candidate_flips - r.matched_count}")
        print(f"  Direction correct: {r.direction_accuracy:.1%}")
        print(f"  Mean timing error: {r.mean_timing_error:.2f} bars")
        print(f"  Composite score: {r.composite:.4f}")

    # Family D: Apply chop filter to top 3 base strategies
    print("\n" + "=" * 90)
    print("FAMILY D: CHOP FILTER ON TOP 3")
    print("=" * 90)
    df = compute_indicators(bars)
    ts = df["ts"]
    for i, r in enumerate(results[:3], 1):
        if r.family == "A_MA_Cross":
            base_dir = family_a_signals(df, r.params["fast"], r.params["slow"])
        elif r.family == "B_RSI_Cross":
            base_dir = family_b_signals(df, r.params["rsi"], r.params["ma"])
        else:
            continue

        for atr_min in [0.2, 0.3, 0.4]:
            filtered_dir = apply_chop_filter(df, base_dir, atr_pctile_min=atr_min)
            cand_flips = extract_flips(filtered_dir, ts)
            score = score_candidate(real_flips, cand_flips, tolerance_bars=1)
            print(
                f"  #{i} + ATR>{atr_min:.0%}: "
                f"Hit={score['flip_hit_rate']:.1%} Dir={score['direction_accuracy']:.1%} "
                f"False={score['false_flip_rate']:.1%} Score={score['composite']:.4f} "
                f"(flips: {score['total_candidate_flips']})"
            )

if __name__ == "__main__":
    main()
