#!/usr/bin/env python3
"""Phase 2.7 caveats verification — no metadata trusted.

Validates two of the three caveats raised in Phase 2.7 result doc:

  Caveat (a): "2024 was empirically high-vol" framing
             → verified by computing ATR_20_pct + GARCH_vol_pct medians
               PER YEAR 2019-2025 across MNQ/MES/MGC universes, to test
               whether 2024 is a genuine outlier vs other known high-vol
               years (2018 pre-data, 2020 COVID, 2022 rate-hike, etc).

  Caveat (c): "GOLD pool correlation gate"
             → verified by calling canonical
               `trading_app.lane_correlation.check_candidate_correlation`
               on each of the 5 GOLD lanes against:
                 (1) each other (pairwise intra-GOLD)
                 (2) the currently-deployed profile (if populated)

Caveat (b) (allocator portfolio sim) is deferred — requires a distinct
simulator harness and is larger-scope than verification.

Literature-resource grounding:
  - Chan 2008 Ch 7 § volatility regime is most tractable classification
    (resources/Quantitative_Trading_Chan_2008.pdf pp 119-126; extract at
    docs/institutional/literature/chan_2008_ch7_regime_switching.md)
  - Canonical correlation gate per trading_app/lane_correlation.py
    (RHO_REJECT_THRESHOLD=0.70, SUBSET_REJECT_THRESHOLD=0.80)

Canonical delegations:
  - GOLD_DB_PATH from pipeline.paths
  - HOLDOUT_SACRED_FROM from trading_app.holdout_policy
  - ACTIVE_ORB_INSTRUMENTS from pipeline.asset_configs
  - check_candidate_correlation from trading_app.lane_correlation
  - get_profile_lane_definitions from trading_app.prop_profiles
    (read via check_candidate_correlation internally)
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402
from trading_app.lane_correlation import check_candidate_correlation  # noqa: E402
from trading_app.prop_profiles import ACCOUNT_PROFILES  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "research" / "output"


GOLD_LANES: list[dict] = [
    {"strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100",
     "instrument": "MNQ", "orb_label": "COMEX_SETTLE", "orb_minutes": 5,
     "entry_model": "E2", "confirm_bars": 1, "rr_target": 1.0,
     "filter_type": "OVNRNG_100"},
    {"strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60",
     "instrument": "MNQ", "orb_label": "COMEX_SETTLE", "orb_minutes": 5,
     "entry_model": "E2", "confirm_bars": 1, "rr_target": 1.0,
     "filter_type": "X_MES_ATR60"},
    {"strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60",
     "instrument": "MNQ", "orb_label": "COMEX_SETTLE", "orb_minutes": 5,
     "entry_model": "E2", "confirm_bars": 1, "rr_target": 1.5,
     "filter_type": "X_MES_ATR60"},
    {"strategy_id": "MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30",
     "instrument": "MNQ", "orb_label": "SINGAPORE_OPEN", "orb_minutes": 5,
     "entry_model": "E2", "confirm_bars": 1, "rr_target": 1.5,
     "filter_type": "ATR_P50_O30"},
    {"strategy_id": "MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15",
     "instrument": "MNQ", "orb_label": "US_DATA_1000", "orb_minutes": 5,
     "entry_model": "E2", "confirm_bars": 1, "rr_target": 1.5,
     "filter_type": "VWAP_MID_ALIGNED_O15"},
]


def verify_caveat_a_vol_regime(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per-year ATR_20_pct + GARCH_vol_pct medians across instruments."""
    rows = []
    for inst in ACTIVE_ORB_INSTRUMENTS:
        df = con.execute(
            """
            SELECT
                CAST(EXTRACT(YEAR FROM trading_day) AS INT) AS year,
                atr_20_pct,
                garch_forecast_vol_pct
            FROM daily_features
            WHERE symbol = ? AND orb_minutes = 5
              AND trading_day < ?
            """,
            [inst, HOLDOUT_SACRED_FROM],
        ).df()
        for yr, sub in df.groupby("year"):
            rows.append({
                "instrument": inst,
                "year": int(yr),
                "n_days": len(sub),
                "atr20_pct_median": (
                    float(sub["atr_20_pct"].dropna().median())
                    if len(sub["atr_20_pct"].dropna()) else None
                ),
                "garch_vol_pct_median": (
                    float(sub["garch_forecast_vol_pct"].dropna().median())
                    if len(sub["garch_forecast_vol_pct"].dropna()) else None
                ),
            })
    return pd.DataFrame(rows).sort_values(["instrument", "year"])


def verify_caveat_c_correlation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise GOLD correlation + each-vs-deployed-profile correlation."""
    # Pairwise intra-GOLD: candidate against each other as a mock "deployed"
    pairwise_rows = []
    for i, cand in enumerate(GOLD_LANES):
        for j, other in enumerate(GOLD_LANES):
            if i >= j:
                continue
            # Build a mock profile with only the "other" lane, run gate
            report = check_candidate_correlation(
                candidate_lane=cand,
                profile_id=None,  # will fail profile lookup — use direct approach
                con=None,
            ) if False else None  # placeholder — direct approach below

    # Direct approach: load daily pnl per lane, compute Pearson on shared days
    import duckdb as _duckdb
    from trading_app.lane_correlation import _load_lane_daily_pnl, _pearson
    con = _duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        pnl_by_lane = {
            lane["strategy_id"]: _load_lane_daily_pnl(con, lane)
            for lane in GOLD_LANES
        }
    finally:
        con.close()

    for i, cand in enumerate(GOLD_LANES):
        for j, other in enumerate(GOLD_LANES):
            if i >= j:
                continue
            cand_pnl = pnl_by_lane[cand["strategy_id"]]
            other_pnl = pnl_by_lane[other["strategy_id"]]
            shared = sorted(set(cand_pnl) & set(other_pnl))
            n_shared = len(shared)
            smaller = min(len(cand_pnl), len(other_pnl))
            subset_cov = n_shared / smaller if smaller > 0 else 0.0
            if n_shared >= 5:
                xs = [cand_pnl[d] for d in shared]
                ys = [other_pnl[d] for d in shared]
                rho = _pearson(xs, ys)
            else:
                rho = 0.0
            same_session = cand["orb_label"] == other["orb_label"]
            pairwise_rows.append({
                "lane_a": cand["strategy_id"],
                "lane_b": other["strategy_id"],
                "shared_days": n_shared,
                "subset_coverage": subset_cov,
                "rho": rho,
                "same_session": same_session,
                "rho_reject": rho > 0.70,
                "subset_reject_if_same_session": same_session and subset_cov > 0.80,
            })

    # GOLD vs deployed profile (if any profile defines MNQ lanes)
    deployed_rows = []
    for profile_id in ACCOUNT_PROFILES.keys():
        for cand in GOLD_LANES:
            try:
                report = check_candidate_correlation(
                    candidate_lane=cand,
                    profile_id=profile_id,
                )
            except Exception as e:  # noqa: BLE001
                deployed_rows.append({
                    "profile": profile_id,
                    "candidate": cand["strategy_id"],
                    "error": f"{type(e).__name__}: {e}",
                })
                continue
            deployed_rows.append({
                "profile": profile_id,
                "candidate": cand["strategy_id"],
                "gate_pass": report.gate_pass,
                "worst_rho": round(report.worst_rho, 3),
                "worst_subset": round(report.worst_subset, 3),
                "n_deployed_lanes": len(report.pairs),
                "reject_reasons": "; ".join(report.reject_reasons),
            })

    return pd.DataFrame(pairwise_rows), pd.DataFrame(deployed_rows)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Caveat (a)
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True, config={"access_mode": "READ_ONLY"})
    try:
        vol_df = verify_caveat_a_vol_regime(con)
    finally:
        con.close()

    vol_csv = OUTPUT_DIR / "phase_2_7_caveat_a_vol_by_year.csv"
    vol_df.to_csv(vol_csv, index=False)

    print("=" * 70)
    print("CAVEAT (a) VERIFICATION — vol regime by year (Mode A, canonical data)")
    print("=" * 70)
    print()

    pivot_atr = vol_df.pivot(index="year", columns="instrument", values="atr20_pct_median")
    print("ATR_20_pct MEDIAN by year × instrument:")
    print(pivot_atr.round(2).to_string())
    print()
    pivot_garch = vol_df.pivot(index="year", columns="instrument", values="garch_vol_pct_median")
    print("GARCH_vol_pct MEDIAN by year × instrument:")
    print(pivot_garch.round(2).to_string())
    print()

    # Rank: is 2024 the year with highest median ATR_20_pct?
    print("Year ranking by ATR_20_pct (MNQ, descending):")
    mnq_rank = (
        vol_df[vol_df["instrument"] == "MNQ"]
        [["year", "atr20_pct_median", "n_days"]]
        .sort_values("atr20_pct_median", ascending=False)
    )
    print(mnq_rank.to_string(index=False))
    print()

    # Caveat (c)
    pairwise_df, deployed_df = verify_caveat_c_correlation()
    pw_csv = OUTPUT_DIR / "phase_2_7_caveat_c_gold_pairwise.csv"
    dep_csv = OUTPUT_DIR / "phase_2_7_caveat_c_gold_vs_deployed.csv"
    pairwise_df.to_csv(pw_csv, index=False)
    deployed_df.to_csv(dep_csv, index=False)

    print("=" * 70)
    print("CAVEAT (c) VERIFICATION — GOLD pool correlation gate")
    print("=" * 70)
    print()
    print("Pairwise GOLD (rho > 0.70 rejects parallel deploy):")
    print(pairwise_df[[
        "lane_a", "lane_b", "shared_days", "subset_coverage",
        "rho", "same_session", "rho_reject", "subset_reject_if_same_session"
    ]].to_string(index=False))
    print()

    any_reject = pairwise_df["rho_reject"].any() or pairwise_df["subset_reject_if_same_session"].any()
    print(f"Intra-GOLD any rejection: {any_reject}")
    print()

    if len(deployed_df):
        print("GOLD vs deployed profiles (canonical check_candidate_correlation):")
        print(deployed_df.to_string(index=False))
        print()

    print(f"Written:")
    print(f"  {vol_csv}")
    print(f"  {pw_csv}")
    print(f"  {dep_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
