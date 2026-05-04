"""Bull-day short avoidance — deployed-lane verification (Pathway B, K=1).

Confirms or refutes the 2026-04-04 bull_short_avoidance signal on the
ONE deployed lane that motivated it: MNQ NYSE_OPEN E2 RR1.0 CB1 COST_LT12.

Memory claim (2026-04-04):
    Shorts after bull prior days underperform shorts after bear prior days.
    Delta ~+0.072R, p=0.0007, 14/17 positive years, NYSE_OPEN drives 60% of effect.

This script tests the claim on the SPECIFIC deployed lane under:
  - Canonical filter delegation (COST_LT12 via research.filter_utils)
  - Mode A IS (trading_day < 2026-01-01)
  - Mode A OOS (2026-01-01 .. latest)
  - Moving-block bootstrap null (pnl resampled in blocks, label mask fixed)

Theory (Pathway B, K=1):
    Prior-day direction serves as an overnight-sentiment / mean-reversion proxy.
    After a strong up-day, overnight dip-buying flow resists intraday shorts
    even when the break confirms. The effect is specifically asymmetric
    (bear-day longs are fine — not a symmetric "prior-day momentum"
    artefact). The effect is strongest at NYSE_OPEN where cash-session
    participation is highest.

@research-source: docs/audit/hypotheses/2026-04-20-bull-short-deployed-lane.md (to be written if confirmed)
@data-source: orb_outcomes JOIN daily_features (orb_minutes=5), canonical COST_LT12 via filter_signal
"""

from __future__ import annotations

import sys
import warnings
from datetime import date

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from research.filter_utils import filter_signal  # noqa: E402
from research.oos_power import (  # noqa: E402
    format_power_report,
    oos_ttest_power,
    power_verdict,
)
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402

INSTRUMENT = "MNQ"
ORB_LABEL = "NYSE_OPEN"
ORB_MINUTES = 5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR_TARGET = 1.0
FILTER_KEY = "COST_LT12"
BLOCK_LEN = 5  # trading days per block (conservative for daily-scale autocorr)
N_BOOT = 5000
RNG_SEED = 20260420


def load_lane_df(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Triple-join orb_outcomes + daily_features for the deployed lane, return filtered df."""
    # Raw join — MUST NOT pre-alias orb_* columns; filter_signal needs them raw.
    q = f"""
    SELECT o.trading_day, o.symbol, o.orb_minutes, o.orb_label,
           o.entry_price, o.stop_price, o.pnl_r,
           d.prev_day_direction,
           d.*
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = '{INSTRUMENT}'
      AND o.orb_label = '{ORB_LABEL}'
      AND o.orb_minutes = {ORB_MINUTES}
      AND o.entry_model = '{ENTRY_MODEL}'
      AND o.confirm_bars = {CONFIRM_BARS}
      AND o.rr_target = {RR_TARGET}
      AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day
    """
    df = con.execute(q).fetchdf()
    # Dedup any inadvertent column collisions from d.*
    df = df.loc[:, ~df.columns.duplicated()]
    # Canonical filter delegation — COST_LT12 on NYSE_OPEN
    fire = filter_signal(df, FILTER_KEY, ORB_LABEL)
    df = df.loc[fire == 1].copy()
    # Derive direction from entry/stop geometry (canonical long/short rule)
    df["direction"] = np.where(df["entry_price"] > df["stop_price"], "long", "short")
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    df["year"] = pd.to_datetime(df["trading_day"]).dt.year
    return df


def summarize_shorts(shorts: pd.DataFrame, label: str) -> dict:
    """Compute bear vs bull split stats + Welch-t."""
    bear = shorts.loc[shorts["prev_day_direction"] == "bear", "pnl_r"].to_numpy()
    bull = shorts.loc[shorts["prev_day_direction"] == "bull", "pnl_r"].to_numpy()
    out: dict = {"label": label, "N_bear": len(bear), "N_bull": len(bull)}
    if len(bear) == 0 or len(bull) == 0:
        out.update(dict(mean_bear=np.nan, mean_bull=np.nan, wr_bear=np.nan, wr_bull=np.nan,
                        delta=np.nan, t=np.nan, p=np.nan, wr_spread=np.nan))
        return out
    out["mean_bear"] = float(bear.mean())
    out["mean_bull"] = float(bull.mean())
    out["wr_bear"] = float((bear > 0).mean())
    out["wr_bull"] = float((bull > 0).mean())
    out["delta"] = out["mean_bear"] - out["mean_bull"]
    out["wr_spread"] = out["wr_bear"] - out["wr_bull"]
    if len(bear) >= 2 and len(bull) >= 2:
        res = stats.ttest_ind(bear, bull, equal_var=False)
        out["t"] = float(np.asarray(res[0]))  # type: ignore[index]
        out["p"] = float(np.asarray(res[1]))  # type: ignore[index]
    else:
        out["t"] = np.nan
        out["p"] = np.nan
    return out


def moving_block_bootstrap_p(shorts_is: pd.DataFrame, n_boot: int, block_len: int, seed: int) -> tuple[float, float, np.ndarray]:
    """Null: resample pnl_r in blocks (preserves autocorr); keep prev_day_direction FIXED.

    Returns (p, observed_delta, null_deltas).
    """
    # Sort by trading_day for proper block structure
    s = shorts_is.sort_values("trading_day").reset_index(drop=True).copy()
    labels = s["prev_day_direction"].to_numpy()
    pnl = s["pnl_r"].to_numpy()
    mask_bear = labels == "bear"
    mask_bull = labels == "bull"
    observed_delta = pnl[mask_bear].mean() - pnl[mask_bull].mean()

    n = len(pnl)
    n_blocks = int(np.ceil(n / block_len))
    rng = np.random.default_rng(seed)
    null_deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        # Draw random block start indices with replacement
        starts = rng.integers(0, max(1, n - block_len + 1), size=n_blocks)
        idx_parts = [np.arange(s, s + block_len) for s in starts]
        idx = np.concatenate(idx_parts)[:n]
        pnl_boot = pnl[idx]  # resampled pnl
        # Labels stay FIXED — breaks signal-outcome link per 2026-04-15 lesson
        null_deltas[i] = pnl_boot[mask_bear].mean() - pnl_boot[mask_bull].mean()
    # Two-tailed
    p = ((np.abs(null_deltas) >= np.abs(observed_delta)).sum() + 1) / (n_boot + 1)
    return float(p), float(observed_delta), null_deltas


def main() -> None:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    print("=" * 90)
    print("BULL-DAY SHORT AVOIDANCE — DEPLOYED LANE VERIFY (Pathway B, K=1)")
    print("=" * 90)
    print(f"Lane: {INSTRUMENT} {ORB_LABEL} {ENTRY_MODEL} "
          f"RR{RR_TARGET} CB{CONFIRM_BARS} {FILTER_KEY}")
    print(f"Holdout cutoff (Mode A sacred): {HOLDOUT_SACRED_FROM}")
    print(f"Block len = {BLOCK_LEN}, n_boot = {N_BOOT}, seed = {RNG_SEED}")
    print()

    df = load_lane_df(con)
    print(f"Filter-passed trades on lane: N = {len(df)}")
    print(f"  date range: {df['trading_day'].min()} .. {df['trading_day'].max()}")
    dir_counts = df["direction"].value_counts().to_dict()
    print(f"  direction split: {dir_counts}")
    pd_counts = df["prev_day_direction"].value_counts(dropna=False).to_dict()
    print(f"  prev_day_direction split: {pd_counts}")
    print()

    shorts = df.loc[df["direction"] == "short"].copy()
    print(f"Shorts on lane: N = {len(shorts)}")
    print()

    # --- IS / OOS partition (Mode A) ---
    holdout = HOLDOUT_SACRED_FROM
    if isinstance(holdout, str):
        holdout = date.fromisoformat(holdout)
    shorts_is = shorts.loc[shorts["trading_day"] < holdout].copy()
    shorts_oos = shorts.loc[shorts["trading_day"] >= holdout].copy()
    print(f"IS (trading_day < {holdout}): N = {len(shorts_is)}")
    print(f"OOS (>= {holdout}): N = {len(shorts_oos)}")
    print()

    # --- IS / OOS split summaries ---
    is_stats = summarize_shorts(shorts_is, "IS")
    oos_stats = summarize_shorts(shorts_oos, "OOS")

    def fmt(s):
        return (f"  {s['label']}: bear N={s['N_bear']:4d} mean={s['mean_bear']:+.4f} WR={s['wr_bear']:.3f} | "
                f"bull N={s['N_bull']:4d} mean={s['mean_bull']:+.4f} WR={s['wr_bull']:.3f} | "
                f"delta={s['delta']:+.4f} wr_spread={s['wr_spread']:+.3f} "
                f"t={s['t']:.2f} p_welch={s['p']:.4f}")
    print("--- IS / OOS split ---")
    print(fmt(is_stats))
    print(fmt(oos_stats))

    # dir_match per C8 (IS / OOS same sign) — RULE 3.3: gated by power
    dir_match = (np.sign(is_stats["delta"]) == np.sign(oos_stats["delta"])) if (
        not np.isnan(is_stats["delta"]) and not np.isnan(oos_stats["delta"])
    ) else False
    print(f"  dir_match (sign IS == sign OOS): {dir_match}")

    # --- RULE 3.3: OOS power floor — compute before treating dir_match as a kill ---
    is_pooled_std = float(np.sqrt(
        ((is_stats["N_bear"] - 1) * shorts_is.loc[shorts_is["prev_day_direction"] == "bear", "pnl_r"].var(ddof=1)
         + (is_stats["N_bull"] - 1) * shorts_is.loc[shorts_is["prev_day_direction"] == "bull", "pnl_r"].var(ddof=1))
        / max(1, (is_stats["N_bear"] + is_stats["N_bull"] - 2))
    ))
    pwr = oos_ttest_power(
        is_delta=is_stats["delta"],
        is_pooled_std=is_pooled_std,
        n_oos_a=max(2, oos_stats["N_bear"]),
        n_oos_b=max(2, oos_stats["N_bull"]),
        alpha=0.05,
    )
    print(format_power_report(pwr, label="RULE 3.3 OOS power floor"))
    tier = power_verdict(pwr["power"])
    if tier != "CAN_REFUTE":
        print(f"  >>> dir_match is {tier} — NOT a hard kill criterion at this N.")
    print()

    # --- Block bootstrap null on IS (proper: pnl resampled, labels fixed) ---
    print(f"--- Block bootstrap null ({N_BOOT} draws, block_len={BLOCK_LEN}, labels FIXED) ---")
    p_boot, obs_delta, nulls = moving_block_bootstrap_p(
        shorts_is, n_boot=N_BOOT, block_len=BLOCK_LEN, seed=RNG_SEED
    )
    print(f"  Observed delta (IS): {obs_delta:+.4f}")
    print(f"  Null P025: {np.quantile(nulls, 0.025):+.4f}")
    print(f"  Null P975: {np.quantile(nulls, 0.975):+.4f}")
    print(f"  Null mean: {nulls.mean():+.4f}  std: {nulls.std():.4f}")
    print(f"  Two-tailed block-bootstrap p: {p_boot:.4f}")
    print()

    # --- Year-by-year (IS only; OOS too short for per-year) ---
    print("--- Year-by-year (IS) ---")
    pos_years = 0
    neg_years = 0
    years = sorted(shorts_is["year"].unique())
    for y in years:
        yr = shorts_is.loc[shorts_is["year"] == y]
        bear_y = yr.loc[yr["prev_day_direction"] == "bear", "pnl_r"]
        bull_y = yr.loc[yr["prev_day_direction"] == "bull", "pnl_r"]
        if len(bear_y) < 5 or len(bull_y) < 5:
            print(f"  {y}: bear N={len(bear_y):3d} bull N={len(bull_y):3d}  (skip, too few)")
            continue
        delta_y = bear_y.mean() - bull_y.mean()
        pos_years += (delta_y > 0)
        neg_years += (delta_y <= 0)
        print(f"  {y}: bear N={len(bear_y):3d} mean={bear_y.mean():+.4f} | "
              f"bull N={len(bull_y):3d} mean={bull_y.mean():+.4f} | delta={delta_y:+.4f}")
    total_yr = pos_years + neg_years
    if total_yr > 0:
        print(f"\n  bear > bull years: {pos_years}/{total_yr}")
    print()

    # --- Dollar impact at half-size ---
    print("--- Dollar impact (IS, half-size on bull-day shorts) ---")
    bear_is = shorts_is.loc[shorts_is["prev_day_direction"] == "bear", "pnl_r"]
    bull_is = shorts_is.loc[shorts_is["prev_day_direction"] == "bull", "pnl_r"]
    print(f"  Bear-day shorts IS: N={len(bear_is)}, total R = {bear_is.sum():+.2f}, mean = {bear_is.mean():+.4f}")
    print(f"  Bull-day shorts IS: N={len(bull_is)}, total R = {bull_is.sum():+.2f}, mean = {bull_is.mean():+.4f}")
    if bull_is.mean() > 0:
        print(f"  [Bull-day shorts are still profitable — half-size, NOT skip]")
    elif bull_is.mean() < 0:
        print(f"  [Bull-day shorts LOSE on this lane — skip may be viable]")
    if len(shorts_oos) > 0:
        bear_oos = shorts_oos.loc[shorts_oos["prev_day_direction"] == "bear", "pnl_r"]
        bull_oos = shorts_oos.loc[shorts_oos["prev_day_direction"] == "bull", "pnl_r"]
        print(f"  Bear-day shorts OOS: N={len(bear_oos)}, total R = {bear_oos.sum():+.2f}, mean = {bear_oos.mean() if len(bear_oos) else np.nan:+.4f}")
        print(f"  Bull-day shorts OOS: N={len(bull_oos)}, total R = {bull_oos.sum():+.2f}, mean = {bull_oos.mean() if len(bull_oos) else np.nan:+.4f}")
    print()

    # --- Decision summary ---
    print("=" * 90)
    print("DECISION INPUTS")
    print("=" * 90)
    print(f"  IS Welch p:           {is_stats['p']:.4f}")
    print(f"  IS block-boot p:      {p_boot:.4f}")
    print(f"  IS WR spread:         {is_stats['wr_spread']:+.3f}  (>0.03 = WR signal)")
    print(f"  dir_match IS/OOS:     {dir_match}")
    print(f"  OOS power tier:       {tier}")
    if total_yr > 0:
        print(f"  bear>bull years:      {pos_years}/{total_yr}")
    print()
    print("Criteria (RULE 3.3-aware — power-gated dir_match):")
    print("  CONFIRMED  = block-boot p < 0.01 AND WR spread > 0.03 AND bear-year-share >= 0.70")
    print("               AND (OOS tier == CAN_REFUTE implies dir_match TRUE)")
    print("  CONDITIONAL= IS passes but OOS tier != CAN_REFUTE -> IS real, OOS unverified")
    print("  BORDERLINE = 0.01 <= block-boot p < 0.05  OR  WR spread 0.02-0.03  OR  bear-year-share 0.55-0.70")
    print("  REJECTED   = block-boot p >= 0.05  OR  WR spread <= 0")
    print("               OR  (OOS tier == CAN_REFUTE AND dir_match FALSE)")
    print()
    con.close()


if __name__ == "__main__":
    main()
