"""Bull-day short avoidance — pooled-deployed-lane OOS test (Pathway B, K=1).

Follow-up to `bull_short_avoidance_deployed_lane_verify.py` (2026-04-20).
The lane-specific verify found IS was valid on `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`
(p_boot=0.018, 7/7 years, WR spread +7.8%) but lane-specific OOS had only 7.9%
power to detect the IS effect — a RULE 3.3 STATISTICALLY_USELESS tier. Corrected
verdict was CONDITIONAL — UNVERIFIED, not REJECTED.

This script runs the highest-EV next test: pool 2026 Q1 shorts across ALL 6
DEPLOY lanes (`docs/runtime/lane_allocation.json` rebalance 2026-04-18) under
the same bull/bear prior-day partition. Same hypothesis, pooled scope →
OOS per-group N climbs from ~20 to hopefully ≥100, which is where RULE 3.3
flips to CAN_REFUTE.

Hypothesis (Pathway B, K=1, theory citation mandatory):
    At the deployed-portfolio level, bear-prior-day shorts outperform
    bull-prior-day shorts by a meaningful R-delta. This is the same prior-
    day-sentiment / bull-exhaustion mechanism tested on the single lane;
    the pooled scope is purely for N inflation, NOT for multiplicity
    broadening (K stays 1, because the hypothesis is the same).

Data discipline:
    - Canonical layers only: `orb_outcomes`, `daily_features`, `bars_1m`
    - Canonical filter delegation via `research.filter_utils.filter_signal`
      for EACH lane's individual filter
    - Lane roster from `docs/runtime/lane_allocation.json` (live truth)
    - Mode A holdout per `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`
    - Per-lane shorts are KEPT SEPARATE by trading_day × orb_label —
      same day with TOKYO_OPEN and NYSE_OPEN shorts stays two rows (real
      portfolio exposure), not deduplicated

RULE 3.3 compliance: explicit OOS power report next to dir_match verdict.
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

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

LANE_ALLOCATION_PATH = Path("docs/runtime/lane_allocation.json")
BLOCK_LEN = 5
N_BOOT = 5000
RNG_SEED = 20260420


def parse_strategy_id(sid: str) -> dict:
    """Decompose a strategy_id like `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`.

    Some IDs have trailing `_O15` to denote orb_minutes=15 (e.g.
    `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`). We detect and split.
    """
    parts = sid.split("_")
    # Suffix O15 or O30?
    orb_minutes = 5
    if parts[-1] in ("O15", "O30"):
        orb_minutes = int(parts[-1][1:])
        parts = parts[:-1]
    filter_key = parts[-1]
    # But filter_key may itself be 2+ tokens (e.g. COST_LT12 → ['COST','LT12'] originally)
    # Strategy_id join is '_', so COST_LT12 ends up as tokens ['COST','LT12'].
    # Reconstruct by walking back from end while parts don't match the rr/cb/em format.
    # Easier: known tokens for entry_model (E2/E3), confirm_bars (CB1/CB2), rr_target (RR...).
    # So walk left until we hit RR.
    idx_rr = next(i for i, p in enumerate(parts) if p.startswith("RR"))
    # Everything after CB... (CB is idx_rr+1) to end is filter_key pieces
    cb_idx = idx_rr + 1
    filter_tokens = parts[cb_idx + 1 :]
    filter_key = "_".join(filter_tokens)
    rr_target = float(parts[idx_rr].replace("RR", ""))
    confirm_bars = int(parts[cb_idx].replace("CB", ""))
    entry_model = parts[idx_rr - 1]  # e.g. E2
    instrument = parts[0]
    orb_label = "_".join(parts[1 : idx_rr - 1])
    return dict(
        instrument=instrument,
        orb_label=orb_label,
        orb_minutes=orb_minutes,
        entry_model=entry_model,
        confirm_bars=confirm_bars,
        rr_target=rr_target,
        filter_key=filter_key,
    )


def load_deploy_lanes() -> list[dict]:
    """Return parsed lane specs for every DEPLOY lane in the live allocator."""
    with open(LANE_ALLOCATION_PATH) as f:
        alloc = json.load(f)
    specs = []
    for lane in alloc["lanes"]:
        if lane.get("status") != "DEPLOY":
            continue
        spec = parse_strategy_id(lane["strategy_id"])
        spec["strategy_id"] = lane["strategy_id"]
        specs.append(spec)
    return specs


def load_lane_shorts(con: duckdb.DuckDBPyConnection, spec: dict) -> pd.DataFrame:
    """Triple-join orb_outcomes + daily_features for one lane, apply canonical
    filter, return only shorts with prev_day_direction populated."""
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
    WHERE o.symbol = '{spec["instrument"]}'
      AND o.orb_label = '{spec["orb_label"]}'
      AND o.orb_minutes = {spec["orb_minutes"]}
      AND o.entry_model = '{spec["entry_model"]}'
      AND o.confirm_bars = {spec["confirm_bars"]}
      AND o.rr_target = {spec["rr_target"]}
      AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day
    """
    df = con.execute(q).fetchdf()
    df = df.loc[:, ~df.columns.duplicated()]
    if df.empty:
        return df
    fire = filter_signal(df, spec["filter_key"], spec["orb_label"])
    df = df.loc[fire == 1].copy()
    df["direction"] = np.where(df["entry_price"] > df["stop_price"], "long", "short")
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    df["year"] = pd.to_datetime(df["trading_day"]).dt.year
    df["lane_id"] = spec["strategy_id"]
    shorts = df.loc[
        (df["direction"] == "short") & df["prev_day_direction"].isin(["bear", "bull"])
    ].copy()
    return shorts


def split_group_stats(df: pd.DataFrame, group_col: str, group_a: str, group_b: str, value_col: str = "pnl_r") -> dict[str, float]:
    a = df.loc[df[group_col] == group_a, value_col].to_numpy()
    b = df.loc[df[group_col] == group_b, value_col].to_numpy()
    out: dict[str, float] = {"N_a": float(len(a)), "N_b": float(len(b))}
    if len(a) == 0 or len(b) == 0:
        out.update(dict(mean_a=np.nan, mean_b=np.nan, wr_a=np.nan, wr_b=np.nan,
                        delta=np.nan, wr_spread=np.nan, t=np.nan, p=np.nan, pooled_std=np.nan))
        return out
    out["mean_a"] = float(a.mean())
    out["mean_b"] = float(b.mean())
    out["wr_a"] = float((a > 0).mean())
    out["wr_b"] = float((b > 0).mean())
    out["delta"] = out["mean_a"] - out["mean_b"]
    out["wr_spread"] = out["wr_a"] - out["wr_b"]
    if len(a) >= 2 and len(b) >= 2:
        res = stats.ttest_ind(a, b, equal_var=False)
        out["t"] = float(np.asarray(res[0]))  # type: ignore[index]
        out["p"] = float(np.asarray(res[1]))  # type: ignore[index]
        out["pooled_std"] = float(np.sqrt(
            ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
        ))
    else:
        out["t"] = np.nan
        out["p"] = np.nan
        out["pooled_std"] = np.nan
    return out


def moving_block_bootstrap_p(
    shorts_is: pd.DataFrame, n_boot: int, block_len: int, seed: int
) -> tuple[float, float, np.ndarray]:
    """Null: block-resample pnl, keep prev_day_direction labels FIXED."""
    s = shorts_is.sort_values("trading_day").reset_index(drop=True).copy()
    labels = s["prev_day_direction"].to_numpy()
    pnl = s["pnl_r"].to_numpy()
    mask_bear = labels == "bear"
    mask_bull = labels == "bull"
    observed = pnl[mask_bear].mean() - pnl[mask_bull].mean()
    n = len(pnl)
    n_blocks = int(np.ceil(n / block_len))
    rng = np.random.default_rng(seed)
    null_deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        starts = rng.integers(0, max(1, n - block_len + 1), size=n_blocks)
        idx = np.concatenate([np.arange(s0, s0 + block_len) for s0 in starts])[:n]
        pnl_boot = pnl[idx]
        null_deltas[i] = pnl_boot[mask_bear].mean() - pnl_boot[mask_bull].mean()
    p = ((np.abs(null_deltas) >= np.abs(observed)).sum() + 1) / (n_boot + 1)
    return float(p), float(observed), null_deltas


def main() -> None:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    holdout = HOLDOUT_SACRED_FROM
    if isinstance(holdout, str):
        holdout = date.fromisoformat(holdout)

    print("=" * 90)
    print("BULL-DAY SHORT AVOIDANCE - POOLED DEPLOYED LANES OOS (Pathway B, K=1)")
    print("=" * 90)
    print(f"Allocator snapshot: {LANE_ALLOCATION_PATH}")
    print(f"Mode A holdout: {holdout}")
    print(f"Block len = {BLOCK_LEN}, n_boot = {N_BOOT}, seed = {RNG_SEED}")
    print()

    specs = load_deploy_lanes()
    print(f"Deployed lanes: {len(specs)}")
    for s in specs:
        print(f"  - {s['strategy_id']}  (filter={s['filter_key']}, orb_minutes={s['orb_minutes']})")
    print()

    frames = []
    for s in specs:
        lane_shorts = load_lane_shorts(con, s)
        print(f"  {s['strategy_id']}: shorts N = {len(lane_shorts)}")
        frames.append(lane_shorts)
    pooled = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"\nPooled shorts (all 6 DEPLOY lanes): N = {len(pooled)}")
    if pooled.empty:
        print("No pooled data. Abort.")
        con.close()
        return
    print(f"  date range: {pooled['trading_day'].min()} .. {pooled['trading_day'].max()}")
    print(f"  prev_day_direction: {pooled['prev_day_direction'].value_counts().to_dict()}")
    print()

    is_shorts = pooled.loc[pooled["trading_day"] < holdout].copy()
    oos_shorts = pooled.loc[pooled["trading_day"] >= holdout].copy()
    print(f"IS  (trading_day < {holdout}): N = {len(is_shorts)}")
    print(f"OOS (>= {holdout}):             N = {len(oos_shorts)}")
    print()

    is_stats = split_group_stats(is_shorts, "prev_day_direction", "bear", "bull")
    oos_stats = split_group_stats(oos_shorts, "prev_day_direction", "bear", "bull")

    def fmt(label, s):
        return (f"  {label}: bear N={int(s['N_a']):4d} mean={s['mean_a']:+.4f} WR={s['wr_a']:.3f} | "
                f"bull N={int(s['N_b']):4d} mean={s['mean_b']:+.4f} WR={s['wr_b']:.3f} | "
                f"delta={s['delta']:+.4f} wr_spread={s['wr_spread']:+.3f} t={s['t']:.2f} p_welch={s['p']:.4f}")

    print("--- IS / OOS split (pooled across all 6 DEPLOY lanes) ---")
    print(fmt("IS ", is_stats))
    print(fmt("OOS", oos_stats))
    dir_match = (
        np.sign(is_stats["delta"]) == np.sign(oos_stats["delta"])
        if not (np.isnan(is_stats["delta"]) or np.isnan(oos_stats["delta"]))
        else False
    )
    print(f"  dir_match (IS sign == OOS sign): {dir_match}")
    print()

    # --- RULE 3.3 power floor ---
    print("--- RULE 3.3 OOS power floor ---")
    pwr = oos_ttest_power(
        is_delta=is_stats["delta"],
        is_pooled_std=is_stats["pooled_std"],
        n_oos_a=max(2, oos_stats["N_a"]),
        n_oos_b=max(2, oos_stats["N_b"]),
        alpha=0.05,
    )
    print(format_power_report(pwr))
    tier = power_verdict(pwr["power"])
    if tier != "CAN_REFUTE":
        print(f"  >>> OOS tier {tier}: dir_match is NOT a hard kill at this N.")
    print()

    # --- Block bootstrap null on pooled IS ---
    print(f"--- Block bootstrap null on pooled IS ({N_BOOT} draws, block_len={BLOCK_LEN}, labels FIXED) ---")
    p_boot, obs_delta, nulls = moving_block_bootstrap_p(
        is_shorts, n_boot=N_BOOT, block_len=BLOCK_LEN, seed=RNG_SEED
    )
    print(f"  Observed delta (IS pooled): {obs_delta:+.4f}")
    print(f"  Null P025 / P975:           {np.quantile(nulls, 0.025):+.4f} / {np.quantile(nulls, 0.975):+.4f}")
    print(f"  Two-tailed block-boot p:    {p_boot:.4f}")
    print()

    # --- Per-lane breakdown (is the effect universal or concentrated?) ---
    print("--- Per-lane breakdown (IS) ---")
    for sid in sorted(is_shorts["lane_id"].unique()):
        sub = is_shorts.loc[is_shorts["lane_id"] == sid]
        st = split_group_stats(sub, "prev_day_direction", "bear", "bull")
        print(f"  {sid}: bear N={int(st['N_a']):4d} m={st['mean_a']:+.3f} | "
              f"bull N={int(st['N_b']):4d} m={st['mean_b']:+.3f} | delta={st['delta']:+.4f} p={st['p']:.3f}")
    print()
    print("--- Per-lane breakdown (OOS) ---")
    for sid in sorted(oos_shorts["lane_id"].unique()):
        sub = oos_shorts.loc[oos_shorts["lane_id"] == sid]
        st = split_group_stats(sub, "prev_day_direction", "bear", "bull")
        print(f"  {sid}: bear N={int(st['N_a']):3d} m={st['mean_a']:+.3f} | "
              f"bull N={int(st['N_b']):3d} m={st['mean_b']:+.3f} | delta={st['delta']:+.4f}")
    print()

    # --- Year-by-year (pooled IS) ---
    print("--- Year-by-year (pooled IS) ---")
    pos_yr = neg_yr = 0
    for y in sorted(is_shorts["year"].unique()):
        yr = is_shorts.loc[is_shorts["year"] == y]
        st = split_group_stats(yr, "prev_day_direction", "bear", "bull")
        if np.isnan(st["delta"]):
            continue
        if st["delta"] > 0:
            pos_yr += 1
        else:
            neg_yr += 1
        print(f"  {y}: bear N={int(st['N_a']):4d} m={st['mean_a']:+.4f} | "
              f"bull N={int(st['N_b']):4d} m={st['mean_b']:+.4f} | delta={st['delta']:+.4f}")
    print(f"\n  bear > bull years: {pos_yr}/{pos_yr + neg_yr}")
    print()

    # --- Decision inputs ---
    print("=" * 90)
    print("DECISION INPUTS (RULE 3.3 aware)")
    print("=" * 90)
    print(f"  IS pooled delta:        {is_stats['delta']:+.4f} R")
    print(f"  IS pooled Welch p:      {is_stats['p']:.4f}")
    print(f"  IS block-boot p:        {p_boot:.4f}")
    print(f"  IS WR spread:           {is_stats['wr_spread']:+.3f}")
    print(f"  bear>bull IS years:     {pos_yr}/{pos_yr + neg_yr}")
    print(f"  OOS delta:              {oos_stats['delta']:+.4f} R")
    print(f"  OOS Welch p:            {oos_stats['p']:.4f}")
    print(f"  dir_match IS/OOS:       {dir_match}")
    print(f"  OOS power tier:         {tier}")
    print()
    print("Criteria (RULE 3.3-aware, pre-committed before seeing numbers):")
    print("  CONFIRMED    = block-boot p < 0.01 AND WR spread > 0.03 AND bear-year-share >= 0.70")
    print("                 AND (OOS tier == CAN_REFUTE implies dir_match TRUE)")
    print("  CONDITIONAL  = IS passes but OOS tier != CAN_REFUTE   (IS real, OOS unverified)")
    print("  BORDERLINE   = 0.01 <= block-boot p < 0.05            (IS signal weak)")
    print("  REJECTED     = block-boot p >= 0.05 OR WR spread <= 0 ")
    print("                 OR (OOS tier == CAN_REFUTE AND dir_match FALSE)")

    con.close()


if __name__ == "__main__":
    main()
