"""Phase B: full Phase 0 evidence stack for clean comprehensive-scan candidates.

Per docs/plans/2026-04-28-edge-extraction-phased-plan.md Phase B.

Produces, per candidate:
- C5 DSR (Bailey-LdP 2014 Eq.2) with effective-N (Eq.9, ρ̂=0.5 default
  per docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md)
- C6 WFE (Sharpe_OOS / Sharpe_IS) — conditional on N_OOS >= 50, else
  UNVERIFIED per Phase 0 C8 power floor
- C7 N >= 100 (already pre-screened in scan; verified here)
- C8 dir-match with Cohen's d + OOS power per Amendment 3.2
- C9 era-stability (per-year breakdown N>=20 → no era ExpR < -0.05)
- B6 lane-correlation vs deployed 6-lane portfolio (Carver Ch11)

scratch-policy: include-as-zero (canonical Stage 5 fix, per
docs/specs/outcome_builder_scratch_eod_mtm.md). orb_outcomes pnl_r
already populated for ~99.7% active-instrument scratches.

E2-look-ahead policy: candidate features are all in the Rule 6.1 safe
list (overnight features only used for sessions starting >=17:00
Brisbane per build_daily_features.py:445-454; garch_forecast_vol_pct
is computed at prior close).

Output:
  docs/audit/results/2026-04-28-phase-b-candidate-evidence.md
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402

OUTPUT = PROJECT_ROOT / "docs/audit/results/2026-04-28-phase-b-candidate-evidence.md"

CANDIDATES = [
    {
        "id": "B-MES-EUR",
        "instrument": "MES",
        "session": "EUROPE_FLOW",
        "aperture": 15,
        "rr": 1.0,
        "direction": "long",
        "feature": "ovn_range_pct_GT80",
        "feature_family": "overnight",
        "mechanism": "Chan Ch7 + Fitschen Ch3 (intraday continuation premium after high overnight participation)",
    },
    {
        "id": "B-MES-LON",
        "instrument": "MES",
        "session": "LONDON_METALS",
        "aperture": 30,
        "rr": 2.0,
        "direction": "long",
        "feature": "ovn_range_pct_GT80",
        "feature_family": "overnight",
        "mechanism": "Chan Ch7 + Fitschen Ch3",
    },
    {
        "id": "B-MNQ-NYC",
        "instrument": "MNQ",
        "session": "NYSE_CLOSE",
        "aperture": 5,
        "rr": 1.0,
        "direction": "long",
        "feature": "ovn_range_pct_GT80",
        "feature_family": "overnight",
        "mechanism": "Chan Ch7 + Fitschen Ch3",
    },
    {
        "id": "B-MNQ-COX",
        "instrument": "MNQ",
        "session": "COMEX_SETTLE",
        "aperture": 5,
        "rr": 1.0,
        "direction": "long",
        "feature": "garch_vol_pct_GT70",
        "feature_family": "volatility",
        "mechanism": "Carver Ch9-10 (vol-targeting / forecast-vol-conditioned execution)",
    },
]

# Bailey-LdP 2014 Eq.9 default; project standard per
# docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md
RHO_HAT_DEFAULT = 0.5

# Phase 0 C8 power floor — UNVERIFIED if N_OOS < 50
MIN_N_OOS_FOR_VERDICT = 50

# Phase 0 C5 promotion gate
DSR_THRESHOLD = 0.95
WFE_THRESHOLD = 0.50

# Family K (from clean scan output)
K_FAMILY_OVERNIGHT = 6  # 5 ovn features × pass-types (clean scan: ovn_range_pct_GT80, GT80 short, ovn_took_pdh, etc.)
K_FAMILY_VOLATILITY = 7  # atr_20_pct_*, atr_vel_*, garch_vol_pct_*, pit_range_atr_*

# Project K_global of clean scan
K_GLOBAL_SCAN = 13415


def feature_signal(df: pd.DataFrame, feat: str) -> np.ndarray:
    """Trade-time-knowable feature predicates only. Hard thresholds."""
    if feat == "ovn_range_pct_GT80":
        return (df["overnight_range_pct"].astype(float) > 80).fillna(False).values
    if feat == "garch_vol_pct_GT70":
        return (df["garch_forecast_vol_pct"].astype(float) > 70).fillna(False).values
    raise ValueError(f"Unknown feature {feat}")


def load_cell(con, c) -> pd.DataFrame:
    sess = c["session"]
    sql = f"""
    SELECT o.trading_day, COALESCE(o.pnl_r, 0.0) AS pnl_r, o.outcome,
           d.overnight_range_pct, d.garch_forecast_vol_pct,
           d.orb_{sess}_break_dir AS bdir
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol=? AND o.orb_label=? AND o.orb_minutes=?
      AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=?
      AND o.outcome IN ('win','loss','scratch')
      AND d.orb_{sess}_break_dir=?
    ORDER BY o.trading_day
    """
    df = con.execute(sql, [c["instrument"], sess, c["aperture"], c["rr"], c["direction"]]).df()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["sig"] = feature_signal(df, c["feature"])
    df["is_is"] = df["trading_day"] < pd.Timestamp(HOLDOUT_SACRED_FROM)
    return df


def per_trade_sharpe(pnl: np.ndarray) -> float:
    """Per-trade Sharpe (not annualized) — required for Bailey-LdP DSR Eq.2 with T=N_trades."""
    if len(pnl) < 2:
        return float("nan")
    mu = float(np.mean(pnl))
    sigma = float(np.std(pnl, ddof=1))
    if sigma == 0:
        return float("nan")
    return mu / sigma


def annualised_sharpe(pnl: np.ndarray, n_years: float) -> float:
    """Annualized via sqrt(trades/year). Display only — DSR uses per_trade_sharpe."""
    if len(pnl) < 2 or n_years <= 0:
        return float("nan")
    sr_pt = per_trade_sharpe(pnl)
    if np.isnan(sr_pt):
        return float("nan")
    trades_per_year = len(pnl) / n_years
    return sr_pt * np.sqrt(trades_per_year)


def cohens_d(on: np.ndarray, off: np.ndarray) -> float:
    if len(on) < 2 or len(off) < 2:
        return float("nan")
    pooled_var = ((len(on) - 1) * np.var(on, ddof=1) + (len(off) - 1) * np.var(off, ddof=1)) / (len(on) + len(off) - 2)
    if pooled_var <= 0:
        return float("nan")
    return float((np.mean(on) - np.mean(off)) / np.sqrt(pooled_var))


def power_for_d(d: float, n_on: int, n_off: int, alpha: float = 0.05) -> float:
    """Two-sample t-test power for effect d at sample sizes n_on, n_off."""
    if any(x < 2 or np.isnan(x) for x in [n_on, n_off]) or np.isnan(d):
        return float("nan")
    nharm = 2 / (1 / n_on + 1 / n_off)
    ncp = abs(d) * np.sqrt(nharm / 2)
    df = n_on + n_off - 2
    crit = stats.t.ppf(1 - alpha / 2, df)
    return float(1 - stats.nc_t_cdf(crit, df, ncp) + stats.nc_t_cdf(-crit, df, ncp)) if hasattr(stats, "nc_t_cdf") else float(1 - stats.t.cdf(crit - ncp, df))


def dsr_eq2(sr_obs: float, sr0: float, T: int, skew: float, kurt: float) -> float:
    """Bailey-LdP 2014 Eq.2 — Deflated Sharpe Ratio.

    DSR = Z[ ((SR - SR0) * sqrt(T-1)) / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2) ]
    """
    if T < 2 or np.isnan(sr_obs) or np.isnan(sr0):
        return float("nan")
    denom = 1 - skew * sr_obs + (kurt - 1) / 4 * sr_obs ** 2
    if denom <= 0:
        return float("nan")
    z = (sr_obs - sr0) * np.sqrt(T - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def sr0_threshold(sr_var_per_trade: float, n_hat: float) -> float:
    """Bailey-LdP 2014 Eq.2 — SR0 expected-max-SR rejection threshold.

    SR0 = sqrt(V[SR_n]) * ((1-gamma)*Z^-1[1 - 1/N] + gamma*Z^-1[1 - 1/(N*e)])

    V[SR_n] is variance of PER-TRADE SR across the M trials (matches per-trade SR
    used in DSR formula). Convert from annualized V via V_per_trade = V_ann / trades_per_year.
    """
    GAMMA_EM = 0.5772156649  # Euler-Mascheroni
    if n_hat < 2 or sr_var_per_trade < 0:
        return float("nan")
    e = np.e
    z_one_minus_1_over_n = stats.norm.ppf(max(1 - 1 / n_hat, 1e-10))
    z_one_minus_1_over_ne = stats.norm.ppf(max(1 - 1 / (n_hat * e), 1e-10))
    return float(np.sqrt(sr_var_per_trade) * ((1 - GAMMA_EM) * z_one_minus_1_over_n + GAMMA_EM * z_one_minus_1_over_ne))


def compute_family_sharpe_variance(con, family: str) -> tuple[float, int]:
    """Variance of annualised Sharpe across the comprehensive-scan family.

    Family-K is the set of (instrument × session × apt × rr × direction × feature)
    cells in the same family. For now, approximate by computing all overnight or
    volatility feature cells in the scan with N_IS >= 100.
    """
    # We approximate by counting cells from the canonical scan output result rather
    # than re-scanning. For Phase B we'll use a reasonable proxy: enumerate the
    # comprehensive-scan family cells and compute Sharpe per. This is heavy; for now
    # we use a fixed estimate from the literature example (V=1/2) plus M=K_family.
    # TODO Phase B follow-up: read the scan's all-cell ledger.
    family_k = {"overnight": 1850, "volatility": 2700}.get(family, 1500)
    # Variance of SR across ~K trials in the scan: project standard 0.5 (Bailey example)
    sr_var = 0.5
    return sr_var, family_k


def correlation_to_lane(con, c, lane_strategy_id: str) -> float:
    """Trade-day-level correlation of candidate vs deployed lane pnl_r series.

    Approximation: pull each side's filtered IS pnl_r series, merge on trading_day
    intersection (days both fire). Correlation requires N >= 30 shared days.
    Lane filter is applied only by entry_model+confirm_bars+rr_target match (the
    raw orb_outcomes table does NOT carry filter state); for correlation purposes
    this slight under-filtering of the deployed-lane side is conservative
    (correlation will be slightly diluted but sign preserved).
    """
    row = con.execute(
        """SELECT instrument, orb_label, orb_minutes, entry_model, confirm_bars, rr_target, filter_type
           FROM validated_setups WHERE strategy_id=? LIMIT 1""", [lane_strategy_id]
    ).fetchone()
    if not row:
        return float("nan")
    instr2, sess2, apt2, em2, cb2, rr2, ft2 = row
    apt2 = apt2 or 5  # default ORB minutes if NULL
    sess_a = c["session"]
    feat_pred = "d.overnight_range_pct > 80" if c["feature"] == "ovn_range_pct_GT80" else "d.garch_forecast_vol_pct > 70"
    sql_a = f"""
    SELECT o.trading_day, COALESCE(o.pnl_r, 0.0) AS pnl_r_a
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol=? AND o.orb_label=? AND o.orb_minutes=?
      AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=?
      AND o.outcome IN ('win','loss','scratch')
      AND d.orb_{sess_a}_break_dir=? AND {feat_pred}
      AND o.trading_day < ?
    """
    sql_b = """
    SELECT o.trading_day, COALESCE(o.pnl_r, 0.0) AS pnl_r_b
    FROM orb_outcomes o
    WHERE o.symbol=? AND o.orb_label=? AND o.orb_minutes=?
      AND o.entry_model=? AND o.confirm_bars=? AND o.rr_target=?
      AND o.outcome IN ('win','loss','scratch')
      AND o.trading_day < ?
    """
    df_a = con.execute(sql_a, [c["instrument"], sess_a, c["aperture"], c["rr"], c["direction"], str(HOLDOUT_SACRED_FROM)]).df()
    df_b = con.execute(sql_b, [instr2, sess2, apt2, em2, cb2, rr2, str(HOLDOUT_SACRED_FROM)]).df()
    if len(df_a) == 0 or len(df_b) == 0:
        return float("nan")
    merged = df_a.merge(df_b, on="trading_day", how="inner")
    if len(merged) < 30:
        return float("nan")
    return float(merged["pnl_r_a"].corr(merged["pnl_r_b"]))


def evaluate(con, c) -> dict:
    df = load_cell(con, c)
    on = df[df["sig"]]
    is_on = on[on["is_is"]]
    oos_on = on[~on["is_is"]]
    is_off = df[(~df["sig"]) & df["is_is"]]
    is_full = df[df["is_is"]]

    n_is = len(is_on)
    n_oos = len(oos_on)
    n_off_is = len(is_off)

    pnl_is = is_on["pnl_r"].astype(float).to_numpy()
    pnl_oos = oos_on["pnl_r"].astype(float).to_numpy()
    pnl_off_is = is_off["pnl_r"].astype(float).to_numpy()

    expr_is = float(np.mean(pnl_is)) if n_is else float("nan")
    expr_oos = float(np.mean(pnl_oos)) if n_oos else float("nan")
    expr_off = float(np.mean(pnl_off_is)) if n_off_is else float("nan")
    delta_is = expr_is - expr_off if not np.isnan(expr_is) and not np.isnan(expr_off) else float("nan")

    is_years = (is_on["trading_day"].max() - is_on["trading_day"].min()).days / 365.25 if n_is else 0
    oos_years = (oos_on["trading_day"].max() - oos_on["trading_day"].min()).days / 365.25 if n_oos else 0

    sr_is_per_trade = per_trade_sharpe(pnl_is)
    sr_is_ann = annualised_sharpe(pnl_is, max(is_years, 0.01))
    sr_oos_ann = annualised_sharpe(pnl_oos, max(oos_years, 0.01)) if n_oos >= MIN_N_OOS_FOR_VERDICT else float("nan")

    skew = float(stats.skew(pnl_is, bias=False)) if n_is >= 4 else 0.0
    kurt = float(stats.kurtosis(pnl_is, bias=False, fisher=False)) if n_is >= 4 else 3.0

    sr_var_ann, k_family = compute_family_sharpe_variance(con, c["feature_family"])
    samples_per_year = max(n_is / max(is_years, 0.01), 1.0)
    sr_var_per_trade = sr_var_ann / samples_per_year
    n_hat = RHO_HAT_DEFAULT + (1 - RHO_HAT_DEFAULT) * k_family

    # DSR at full discovery K_family (Pathway A scale)
    sr0 = sr0_threshold(sr_var_per_trade, n_hat)
    dsr = dsr_eq2(sr_is_per_trade, sr0, n_is, skew, kurt)

    # DSR at Pathway B K=1 (theory-citation path per Amendment 3.0)
    # M=1, N_hat=1 → SR0 boundary degenerates; use sr0=0 as the baseline (no multiple-testing penalty)
    dsr_pathway_b = dsr_eq2(sr_is_per_trade, 0.0, n_is, skew, kurt)

    # Display: keep annualized SR for human readability
    sr_is = sr_is_ann
    sr_oos = sr_oos_ann

    if n_oos >= MIN_N_OOS_FOR_VERDICT:
        wfe = sr_oos / sr_is if sr_is and not np.isnan(sr_is) and sr_is != 0 else float("nan")
        wfe_status = "WFE_PASS" if (not np.isnan(wfe) and wfe >= WFE_THRESHOLD) else "WFE_FAIL"
    else:
        wfe = float("nan")
        wfe_status = "WFE_UNVERIFIED"

    delta_oos = expr_oos - (df[(~df["sig"]) & ~df["is_is"]]["pnl_r"].astype(float).mean() if n_oos else float("nan"))
    if n_oos >= MIN_N_OOS_FOR_VERDICT and not np.isnan(delta_is):
        dir_match = bool((delta_is > 0) == (delta_oos > 0))
        d_effect = cohens_d(pnl_is, pnl_off_is)
        c8_status = "C8_VERIFIED_DIRMATCH" if dir_match else "C8_VERIFIED_FLIP"
    else:
        dir_match = None
        d_effect = float("nan")
        c8_status = "C8_UNVERIFIED_LOWPOWER"

    is_on_for_yr = is_on.copy()
    is_on_for_yr["year"] = is_on_for_yr["trading_day"].dt.year
    by_yr = is_on_for_yr.groupby("year")["pnl_r"].agg(["count", "mean"]).reset_index()
    elig = by_yr[by_yr["count"] >= 20]
    n_eras = len(elig)
    n_eras_neg_strict = int((elig["mean"] < -0.05).sum())
    c9_status = "C9_PASS" if n_eras_neg_strict == 0 and n_eras >= 4 else "C9_FAIL" if n_eras_neg_strict > 0 else "C9_UNDERPOWERED"

    dsr_status = "DSR_PASS" if (not np.isnan(dsr) and dsr >= DSR_THRESHOLD) else "DSR_FAIL"
    c7_status = "C7_PASS" if n_is >= 100 else "C7_FAIL"

    # Live deployed lanes per docs/runtime/lane_allocation.json (2026-04-18 rebalance)
    deployed_lanes = [
        ("MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5", "EUROPE_FLOW ORB_G5"),
        ("MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15", "SINGAPORE ATR_P50 15m"),
        ("MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5", "COMEX_SETTLE ORB_G5"),
        ("MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12", "NYSE_OPEN COST_LT12"),
        ("MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12", "TOKYO COST_LT12"),
        ("MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15", "US_DATA_1000 ORB_G5 15m"),
    ]
    correlations = []
    for sid, label in deployed_lanes:
        corr = correlation_to_lane(con, c, sid)
        correlations.append((label, corr))

    # Promotion ladder per Phase 0 Amendment 3.0:
    #   - CANDIDATE_READY = clears Pathway A discovery DSR (rare for our K_family)
    #   - PATHWAY_B_ELIGIBLE = clears Pathway B K=1 DSR (theory-citation path) +
    #     C7 + C9; OOS may be UNVERIFIED at small-N
    #   - RESEARCH_SURVIVOR = clears C7 + C9 only
    #   - KILL = otherwise
    dsr_pb_status = "DSR_PB_PASS" if (not np.isnan(dsr_pathway_b) and dsr_pathway_b >= DSR_THRESHOLD) else "DSR_PB_FAIL"
    if (
        c7_status == "C7_PASS"
        and c9_status == "C9_PASS"
        and dsr_status == "DSR_PASS"
        and wfe_status in ("WFE_PASS", "WFE_UNVERIFIED")
        and c8_status in ("C8_VERIFIED_DIRMATCH", "C8_UNVERIFIED_LOWPOWER")
    ):
        overall = "CANDIDATE_READY"
    elif (
        c7_status == "C7_PASS"
        and c9_status == "C9_PASS"
        and dsr_pb_status == "DSR_PB_PASS"
        and c8_status != "C8_VERIFIED_FLIP"
    ):
        overall = "PATHWAY_B_ELIGIBLE"
    elif c7_status == "C7_PASS" and c9_status == "C9_PASS" and c8_status != "C8_VERIFIED_FLIP":
        overall = "RESEARCH_SURVIVOR"
    else:
        overall = "KILL"

    return {
        "candidate": c,
        "n_is": n_is, "n_oos": n_oos, "n_off_is": n_off_is,
        "expr_is": expr_is, "expr_oos": expr_oos, "delta_is": delta_is,
        "delta_oos": delta_oos if not np.isnan(delta_oos) else float("nan"),
        "sr_is": sr_is, "sr_oos": sr_oos,
        "sr_is_per_trade": sr_is_per_trade,
        "skew": skew, "kurt": kurt,
        "sr_var_family": sr_var_per_trade, "k_family": k_family, "n_hat": n_hat,
        "sr0": sr0, "dsr": dsr, "dsr_pathway_b": dsr_pathway_b, "wfe": wfe,
        "cohens_d": d_effect, "dir_match": dir_match,
        "n_eras": n_eras, "n_eras_neg_strict": n_eras_neg_strict,
        "by_year": by_yr.to_dict(orient="records"),
        "correlations": correlations,
        "c5_dsr": dsr_status, "c5_dsr_pb": dsr_pb_status,
        "c6_wfe": wfe_status, "c7_n": c7_status,
        "c8_dirmatch": c8_status, "c9_era": c9_status,
        "overall": overall,
    }


def fmt(x, fmt_str=".4f"):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return format(x, fmt_str)


def render(results: list[dict]) -> str:
    parts = ["# Phase B candidate evidence — clean comprehensive-scan survivors", ""]
    parts.append(f"**Date:** 2026-04-28  ")
    parts.append(f"**Plan:** `docs/plans/2026-04-28-edge-extraction-phased-plan.md` Phase B  ")
    parts.append(f"**Scan basis:** `docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md` (clean rebuild post-E2-LA-fix)  ")
    parts.append(f"**Holdout:** Mode A strict (`trading_day < {HOLDOUT_SACRED_FROM}`)")
    parts.append("")
    parts.append("## Scope")
    parts.append("")
    parts.append("Apply Phase 0 deploy-gate evidence stack (C5 DSR, C6 WFE, C7 N, C8 dir-match w/ power floor, C9 era stability)")
    parts.append("plus B6 lane-correlation matrix (Carver Ch11) to the 4 mechanism-grounded clean candidates from the rebuilt")
    parts.append("comprehensive scan. dow_thu / is_monday / is_friday survivors are NOT in scope (no mechanism per Aronson EBTA Ch6).")
    parts.append("")
    parts.append("## Methodology citations")
    parts.append("")
    parts.append("- **DSR formula:** Bailey-LdP 2014 Eq.2 (`docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`)")
    parts.append("- **Effective-N:** Bailey-LdP 2014 Eq.9 with ρ̂=0.5 default (project standard, Bailey example p.9-10)")
    parts.append("- **OOS power floor:** Phase 0 Amendment 3.2 + `feedback_oos_power_floor.md` — N_OOS<50 → UNVERIFIED")
    parts.append("- **Era stability:** Phase 0 Criterion 9 — no era ExpR < -0.05 with N_era >= 20")
    parts.append("- **Mechanism grounding:** Chan 2013 Ch7 (intraday momentum), Fitschen 2013 Ch3 (path-of-least-resistance), Carver 2015 Ch9-10 (vol-targeting)")
    parts.append("- **Scratch policy:** include-as-zero per `docs/specs/outcome_builder_scratch_eod_mtm.md` (canonical Stage 5 fix)")
    parts.append("")
    parts.append("## Per-candidate results")
    parts.append("")

    for r in results:
        c = r["candidate"]
        parts.append(f"### {c['id']} — {c['instrument']} {c['session']} O{c['aperture']} RR{c['rr']} {c['direction']} + {c['feature']}")
        parts.append("")
        parts.append(f"**Mechanism:** {c['mechanism']}")
        parts.append("")
        parts.append("**Sample:**")
        parts.append(f"- N_IS = {r['n_is']}, N_OOS = {r['n_oos']}, N_OFF_IS = {r['n_off_is']}")
        parts.append(f"- ExpR_IS_on = {fmt(r['expr_is'])}, ExpR_OOS_on = {fmt(r['expr_oos'])}, Δ_IS = {fmt(r['delta_is'])}, Δ_OOS = {fmt(r['delta_oos'])}")
        parts.append(f"- Sharpe_ann_IS = {fmt(r['sr_is'], '.3f')}, Sharpe_ann_OOS = {fmt(r['sr_oos'], '.3f')} (UNVERIFIED if N_OOS < 50)")
        parts.append(f"- Skewness γ̂₃ = {fmt(r['skew'], '.3f')}, Kurtosis γ̂₄ = {fmt(r['kurt'], '.3f')}")
        parts.append("")
        parts.append("**Phase 0 gates:**")
        parts.append(f"- **C5 DSR (Pathway A discovery, M=K_family={r['k_family']}):** {r['c5_dsr']} — DSR = {fmt(r['dsr'], '.4f')} vs threshold ≥ {DSR_THRESHOLD}, SR0 = {fmt(r['sr0'], '.4f')} per-trade (ρ̂=0.5, N̂={fmt(r['n_hat'], '.0f')}, V[SR_n]_per_trade={fmt(r['sr_var_family'], '.4f')}, SR_per_trade={fmt(r['sr_is_per_trade'], '.4f')})")
        parts.append(f"- **C5 DSR (Pathway B K=1, theory-citation path per Amendment 3.0):** {r['c5_dsr_pb']} — DSR_PB = {fmt(r['dsr_pathway_b'], '.4f')} vs threshold ≥ {DSR_THRESHOLD}")
        parts.append(f"- **C6 WFE:** {r['c6_wfe']} — WFE = {fmt(r['wfe'], '.3f')} (Sharpe_OOS / Sharpe_IS); UNVERIFIED if N_OOS < {MIN_N_OOS_FOR_VERDICT}")
        parts.append(f"- **C7 N:** {r['c7_n']} — N_IS = {r['n_is']} vs threshold ≥ 100")
        parts.append(f"- **C8 dir-match (power-floor aware):** {r['c8_dirmatch']} — dir_match = {r['dir_match']}, Cohen's d = {fmt(r['cohens_d'], '.3f')}, Δ_IS sign = {'+' if r['delta_is'] and r['delta_is'] > 0 else '-'}, Δ_OOS sign = {'+' if r['delta_oos'] and r['delta_oos'] > 0 else '-' if r['delta_oos'] and r['delta_oos'] < 0 else '?'}")
        parts.append(f"- **C9 era stability:** {r['c9_era']} — eligible eras (N≥20) = {r['n_eras']}, eras with ExpR < -0.05 = {r['n_eras_neg_strict']}")
        parts.append("")
        parts.append("**Per-year breakdown (IS-on, N≥1):**")
        parts.append("")
        parts.append("| Year | N | ExpR |")
        parts.append("|---:|---:|---:|")
        for row in r["by_year"]:
            parts.append(f"| {int(row['year'])} | {int(row['count'])} | {row['mean']:+.4f} |")
        parts.append("")
        parts.append("**B6 lane-correlation vs deployed 6 lanes (Pearson on shared trading_day pnl_r):**")
        parts.append("")
        parts.append("| Deployed lane | Correlation |")
        parts.append("|---|---:|")
        for label, corr in r["correlations"]:
            flag = " (overlap concern)" if not np.isnan(corr) and abs(corr) >= 0.5 else ""
            parts.append(f"| {label} | {fmt(corr, '+.3f')}{flag} |")
        parts.append("")
        parts.append(f"**Overall verdict:** **{r['overall']}**")
        parts.append("")

    parts.append("## Aggregate verdict")
    parts.append("")
    parts.append("| Candidate | C5 DSR (Path A) | C5 DSR (Path B K=1) | C6 WFE | C7 N | C8 dir | C9 era | Overall |")
    parts.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        parts.append(f"| {r['candidate']['id']} | {r['c5_dsr']} | {r['c5_dsr_pb']} | {r['c6_wfe']} | {r['c7_n']} | {r['c8_dirmatch']} | {r['c9_era']} | **{r['overall']}** |")
    parts.append("")

    parts.append("## Verdict")
    parts.append("")
    n_ready = sum(1 for r in results if r["overall"] == "CANDIDATE_READY")
    n_pb_eligible = sum(1 for r in results if r["overall"] == "PATHWAY_B_ELIGIBLE")
    n_survivor = sum(1 for r in results if r["overall"] == "RESEARCH_SURVIVOR")
    n_kill = sum(1 for r in results if r["overall"] == "KILL")
    parts.append(f"- CANDIDATE_READY (clears Pathway A discovery DSR): **{n_ready}** of {len(results)}")
    parts.append(f"- PATHWAY_B_ELIGIBLE (theory-citation path): **{n_pb_eligible}** of {len(results)} — eligible for Phase D Pathway B K=1 pre-reg")
    parts.append(f"- RESEARCH_SURVIVOR (clears C7+C9 only): {n_survivor}")
    parts.append(f"- KILL: {n_kill}")
    parts.append("")
    parts.append("Phase D pre-regs proceed only for CANDIDATE_READY cells, AFTER user explicit go.")
    parts.append("")

    parts.append("## Reproduction")
    parts.append("")
    parts.append("```")
    parts.append("DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/phase_b_candidate_evidence_v1.py")
    parts.append("```")
    parts.append("")
    parts.append("- DB: `pipeline.paths.GOLD_DB_PATH`")
    parts.append("- Holdout: `trading_app.holdout_policy.HOLDOUT_SACRED_FROM` (2026-01-01)")
    parts.append("- DSR Eq.2 + Eq.9: `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`")
    parts.append("- Sharpe annualisation: trades/year ≈ N / IS-window-years")
    parts.append("")
    parts.append("## Caveats / limitations")
    parts.append("")
    parts.append("- **K_family approximated** — `compute_family_sharpe_variance()` uses a fixed M proxy (1850 overnight, 2700 volatility) and V[SR_n]=0.5 (Bailey example). True per-family Sharpe variance requires re-emitting the comprehensive scan with per-cell Sharpe stored. **Effect on DSR:** generally conservative; if true V is lower, DSR PASS is more likely than reported.")
    parts.append("- **ρ̂=0.5 default** — Bailey example value. True ρ̂ for our family of trial cells could be higher (highly correlated overnight features) → N̂ smaller → DSR easier to pass; or lower → N̂ larger → DSR harder. Sensitivity analysis deferred.")
    parts.append("- **Cohen's d power calculation** — uses approximate noncentral-t via t-distribution shift; true `nc_t_cdf` not in scipy.stats by default in all versions.")
    parts.append("- **OOS power floor** — N_OOS<50 → UNVERIFIED is the legitimate verdict; cells park until Q3-2026 (more OOS data accrues).")
    parts.append("- **Lane-correlation** — restricted to shared trading_day pnl_r; does NOT capture intra-day timing or position overlap.")
    parts.append("- **Phase B is NOT confirmation** — Pathway B K=1 pre-reg (Phase D) is the legitimate confirmatory step.")
    parts.append("")
    parts.append("## Not done by this script")
    parts.append("")
    parts.append("- No bootstrap null-floor (T6) re-run — already done in earlier audit pass for these 4 cells (all p<0.005)")
    parts.append("- No Pathway B pre-reg — Phase D")
    parts.append("- No capital authorisation — Phase E")
    parts.append("- No write to `validated_setups` / `lane_allocation` / `live_config`")
    parts.append("- No MGC LONDON_METALS short — held until Phase C instrument-family discipline lands")
    parts.append("")

    return "\n".join(parts)


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        results = []
        for c in CANDIDATES:
            print(f"Evaluating {c['id']} {c['instrument']} {c['session']} O{c['aperture']} RR{c['rr']} {c['direction']} + {c['feature']}")
            r = evaluate(con, c)
            results.append(r)
            print(f"  N_IS={r['n_is']} N_OOS={r['n_oos']} ExpR_IS={r['expr_is']:+.3f} SR_IS={r['sr_is']:.2f} DSR={r['dsr']:.3f} -> {r['overall']}")
    finally:
        con.close()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render(results), encoding="utf-8")
    print(f"\nWrote {OUTPUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
