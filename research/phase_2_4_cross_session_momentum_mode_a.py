#!/usr/bin/env python3
"""Phase 2.4 — cross-session momentum SGP→EUROPE_FLOW portfolio re-eval (Mode A).

Scope is set by `docs/plans/2026-04-19-max-ev-extraction-campaign-plan.md` § 2.4:
re-score the A/B/C/D portfolio-construction trade-off from
`docs/audit/deploy_readiness/2026-04-15-sgp-momentum-deploy-readiness.md` § 6
using Mode A (`trading_day < HOLDOUT_SACRED_FROM`) IS ExpR on both sides.

Not new discovery. Pre-reg is the prior file
`docs/audit/hypotheses/2026-04-13-cross-session-sgp-europe-flow.yaml`.

Canonical delegations:
  - `compute_mode_a` for per-lane Mode A ExpR/Sharpe/year_break
  - `filter_signal` via `compute_mode_a` (it calls canonical ALL_FILTERS)
  - `HOLDOUT_SACRED_FROM` from `trading_app.holdout_policy`
  - `GOLD_DB_PATH` from `pipeline.paths`

Outputs:
  - research/output/phase_2_4_cross_session_momentum_mode_a_lanes.csv
  - research/output/phase_2_4_cross_session_momentum_mode_a_options.csv
  - stdout summary table
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from research.mode_a_revalidation_active_setups import compute_mode_a  # noqa: E402
from research.filter_utils import filter_signal  # noqa: E402
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "research" / "output"

RR_TARGETS: tuple[float, ...] = (1.0, 1.5, 2.0)

# Locked trade surface for this Phase 2.4 portfolio re-eval.
INSTRUMENT = "MNQ"
SESSION = "EUROPE_FLOW"
ORB_MINUTES = 5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1

# Execution direction for this lane is LONG (per validated_setups rows for
# both ORB_G5 and the SGP hypothesis pre-reg).
EXECUTION_SPEC = '{"direction": "long"}'


@dataclass
class LaneModeA:
    strategy_id: str
    filter_type: str
    rr_target: float
    n: int = 0
    expr: float | None = None
    sharpe_ann: float | None = None
    wr: float | None = None
    sd: float | None = None
    year_break: dict[int, dict[str, Any]] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        years_positive = sum(1 for y in self.year_break.values() if y.get("positive"))
        years_total = len(self.year_break)
        return {
            "strategy_id": self.strategy_id,
            "filter_type": self.filter_type,
            "rr_target": self.rr_target,
            "mode_a_n": self.n,
            "mode_a_expr": self.expr,
            "mode_a_sharpe_ann": self.sharpe_ann,
            "mode_a_wr": self.wr,
            "mode_a_sd": self.sd,
            "years_positive": years_positive,
            "years_total": years_total,
        }


def spec_for(filter_type: str, rr: float) -> dict[str, Any]:
    return {
        "strategy_id": f"{INSTRUMENT}_{SESSION}_{ENTRY_MODEL}_RR{rr}_CB{CONFIRM_BARS}_{filter_type}",
        "instrument": INSTRUMENT,
        "orb_label": SESSION,
        "orb_minutes": ORB_MINUTES,
        "entry_model": ENTRY_MODEL,
        "confirm_bars": CONFIRM_BARS,
        "rr_target": rr,
        "filter_type": filter_type,
        "execution_spec": EXECUTION_SPEC,
    }


def compute_lane(con: duckdb.DuckDBPyConnection, filter_type: str, rr: float) -> LaneModeA:
    spec = spec_for(filter_type, rr)
    n, expr, sharpe_ann, wr, year_break, sd = compute_mode_a(con, spec)
    return LaneModeA(
        strategy_id=spec["strategy_id"],
        filter_type=filter_type,
        rr_target=rr,
        n=n,
        expr=expr,
        sharpe_ann=sharpe_ann,
        wr=wr,
        sd=sd,
        year_break=year_break,
    )


def load_lane_fires(
    con: duckdb.DuckDBPyConnection, rr: float
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (base_df, orb_g5_mask, sgp_mask, pnl_g5, pnl_sgp) on Mode A
    EUROPE_FLOW long break-day universe. Pnl series are NaN where filter != fire.
    """
    sql = """
        SELECT o.trading_day, o.pnl_r, o.outcome, o.symbol,
               d.*
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.entry_model = ?
          AND o.confirm_bars = ?
          AND o.rr_target = ?
          AND d.orb_EUROPE_FLOW_break_dir = 'long'
          AND o.pnl_r IS NOT NULL
          AND o.trading_day < ?
        ORDER BY o.trading_day
    """
    df = con.execute(
        sql,
        [INSTRUMENT, SESSION, ORB_MINUTES, ENTRY_MODEL, CONFIRM_BARS, rr, HOLDOUT_SACRED_FROM],
    ).df()
    if len(df) == 0:
        empty = pd.Series(dtype=bool)
        return df, empty, empty, pd.Series(dtype=float), pd.Series(dtype=float)

    g5_fire = np.asarray(filter_signal(df, "ORB_G5", SESSION)).astype(bool)
    sgp_fire = np.asarray(filter_signal(df, "CROSS_SGP_MOMENTUM", SESSION)).astype(bool)
    pnl = df["pnl_r"].astype(float).to_numpy()
    pnl_g5 = np.where(g5_fire, pnl, np.nan)
    pnl_sgp = np.where(sgp_fire, pnl, np.nan)

    return (
        df,
        pd.Series(g5_fire, index=df.index, name="g5"),
        pd.Series(sgp_fire, index=df.index, name="sgp"),
        pd.Series(pnl_g5, index=df.index, name="pnl_g5"),
        pd.Series(pnl_sgp, index=df.index, name="pnl_sgp"),
    )


def overlap_stats(
    g5: pd.Series, sgp: pd.Series, pnl_g5: pd.Series, pnl_sgp: pd.Series
) -> dict[str, Any]:
    """Return fire-set overlap + correlation stats on same-day break universe.

    Reports BOTH correlation definitions:
      - ``rho_mask`` — Pearson on 0/1 fire indicators across all break days
        (set-membership independence; what "do these lanes fire together?" asks).
      - ``rho_canonical`` — Pearson on Mode A shared-day pnl_r, matching
        ``trading_app.lane_correlation.check_candidate_correlation`` (line 111).
        This is the deployment gate definition: rho > 0.70 blocks parallel deploy.
    """
    both = g5 & sgp
    g5_only = g5 & ~sgp
    sgp_only = sgp & ~g5
    neither = ~g5 & ~sgp

    # Fire-mask correlation (set-membership independence)
    g5_mask_i = g5.astype(int).to_numpy()
    sgp_mask_i = sgp.astype(int).to_numpy()
    if g5_mask_i.std() > 0 and sgp_mask_i.std() > 0:
        rho_mask = float(np.corrcoef(g5_mask_i, sgp_mask_i)[0, 1])
    else:
        rho_mask = float("nan")

    # Canonical deployment-gate rho: Pearson on pnl_r over shared-fire days,
    # mirroring lane_correlation._pearson against shared = set(cand) & set(dep).
    shared_mask = both.to_numpy()
    if int(shared_mask.sum()) >= 5:
        xs = pnl_g5.to_numpy()[shared_mask]
        ys = pnl_sgp.to_numpy()[shared_mask]
        if np.nanstd(xs) > 0 and np.nanstd(ys) > 0:
            rho_canonical = float(np.corrcoef(xs, ys)[0, 1])
        else:
            rho_canonical = float("nan")
    else:
        rho_canonical = float("nan")

    # Composite AND mask (Option C candidate)
    composite_pnl = np.where(both.to_numpy(), pnl_g5.to_numpy(), np.nan)

    def _mean(x: np.ndarray) -> float | None:
        vals = x[~np.isnan(x)]
        return float(vals.mean()) if len(vals) else None

    return {
        "n_break_days": int(len(g5)),
        "n_both": int(both.sum()),
        "n_g5_only": int(g5_only.sum()),
        "n_sgp_only": int(sgp_only.sum()),
        "n_neither": int(neither.sum()),
        "rho_mask": rho_mask,
        "rho_canonical": rho_canonical,
        "composite_n": int((~np.isnan(composite_pnl)).sum()),
        "composite_expr": _mean(composite_pnl),
    }


def score_options(
    g5_by_rr: dict[float, LaneModeA],
    sgp_by_rr: dict[float, LaneModeA],
    overlap_by_rr: dict[float, dict[str, Any]],
) -> pd.DataFrame:
    """Score portfolio options A/B/C/D per RR under Mode A."""
    rows: list[dict[str, Any]] = []
    for rr in RR_TARGETS:
        g = g5_by_rr[rr]
        s = sgp_by_rr[rr]
        ov = overlap_by_rr[rr]

        # Annualise assuming ~7 IS years (Mode A window is 2019-05-06 → 2025-12-31)
        ann_years = 6.65  # per 2026-04-13 pre-reg data_horizon_years_clean
        g_r_per_yr = (g.n / ann_years) * (g.expr or 0.0) if g.n else None
        s_r_per_yr = (s.n / ann_years) * (s.expr or 0.0) if s.n else None
        c_r_per_yr = (ov["composite_n"] / ann_years) * (ov["composite_expr"] or 0.0) if ov["composite_n"] else None

        rows.append({
            "rr_target": rr,
            # Option A — keep deployed L1 (ORB_G5)
            "A_strategy": g.strategy_id,
            "A_N": g.n, "A_ExpR": g.expr, "A_Sharpe_ann": g.sharpe_ann, "A_WR": g.wr,
            "A_R_per_yr": g_r_per_yr,
            # Option B — swap to SGP
            "B_strategy": s.strategy_id,
            "B_N": s.n, "B_ExpR": s.expr, "B_Sharpe_ann": s.sharpe_ann, "B_WR": s.wr,
            "B_R_per_yr": s_r_per_yr,
            # Option C — composite ORB_G5 AND SGP_TAKE (intersection)
            "C_N": ov["composite_n"], "C_ExpR": ov["composite_expr"],
            "C_R_per_yr": c_r_per_yr,
            # Option D — parallel deploy
            # Canonical gate is Pearson on shared-fire pnl_r, rho > 0.70 → blocked.
            # Fire-mask rho is a separate set-membership independence measure.
            "D_rho_canonical": ov["rho_canonical"],
            "D_rho_mask": ov["rho_mask"],
            "D_gate_blocked": (
                bool(ov["rho_canonical"] >= 0.70)
                if ov["rho_canonical"] is not None and not np.isnan(ov["rho_canonical"])
                else None
            ),
            # Delta B vs A — per-trade quality gap
            "B_minus_A_ExpR": (s.expr - g.expr) if (s.expr is not None and g.expr is not None) else None,
            "B_minus_A_R_per_yr": ((s_r_per_yr or 0.0) - (g_r_per_yr or 0.0)) if (s_r_per_yr is not None and g_r_per_yr is not None) else None,
            # Overlap breakdown
            "overlap_n_both": ov["n_both"],
            "overlap_n_g5_only": ov["n_g5_only"],
            "overlap_n_sgp_only": ov["n_sgp_only"],
            "overlap_n_neither": ov["n_neither"],
            "overlap_n_break_days": ov["n_break_days"],
        })
    return pd.DataFrame(rows)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True, config={"access_mode": "READ_ONLY"})
    try:
        g5_by_rr: dict[float, LaneModeA] = {}
        sgp_by_rr: dict[float, LaneModeA] = {}
        overlap_by_rr: dict[float, dict[str, Any]] = {}

        for rr in RR_TARGETS:
            g5_by_rr[rr] = compute_lane(con, "ORB_G5", rr)
            sgp_by_rr[rr] = compute_lane(con, "CROSS_SGP_MOMENTUM", rr)
            _, g5, sgp, pnl_g5, pnl_sgp = load_lane_fires(con, rr)
            overlap_by_rr[rr] = overlap_stats(g5, sgp, pnl_g5, pnl_sgp)
    finally:
        con.close()

    # Lane CSV
    lane_rows: list[dict[str, Any]] = []
    for rr in RR_TARGETS:
        lane_rows.append(g5_by_rr[rr].as_row())
        lane_rows.append(sgp_by_rr[rr].as_row())
    lanes_df = pd.DataFrame(lane_rows)
    lanes_path = OUTPUT_DIR / "phase_2_4_cross_session_momentum_mode_a_lanes.csv"
    lanes_df.to_csv(lanes_path, index=False)

    # Option-scoring CSV
    options_df = score_options(g5_by_rr, sgp_by_rr, overlap_by_rr)
    options_path = OUTPUT_DIR / "phase_2_4_cross_session_momentum_mode_a_options.csv"
    options_df.to_csv(options_path, index=False)

    # Stdout summary
    print("PHASE 2.4 — CROSS-SESSION MOMENTUM MODE A RE-EVAL")
    print(f"Mode A cutoff: trading_day < {HOLDOUT_SACRED_FROM}")
    print(f"Surface: {INSTRUMENT} {SESSION} O{ORB_MINUTES} {ENTRY_MODEL} CB{CONFIRM_BARS} long")
    print()
    print("Per-lane Mode A:")
    print(lanes_df.to_string(index=False, float_format=lambda v: f"{v:+.4f}" if isinstance(v, float) else str(v)))
    print()
    print("Option scoring A/B/C/D:")
    print(options_df.to_string(index=False, float_format=lambda v: f"{v:+.4f}" if isinstance(v, float) else str(v)))
    print()
    try:
        print(f"Written: {lanes_path.relative_to(PROJECT_ROOT)}")
        print(f"Written: {options_path.relative_to(PROJECT_ROOT)}")
    except ValueError:
        print(f"Written: {lanes_path}")
        print(f"Written: {options_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
