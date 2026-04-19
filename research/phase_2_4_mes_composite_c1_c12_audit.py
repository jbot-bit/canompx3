#!/usr/bin/env python3
"""MES EUROPE_FLOW ORB_G5 AND CROSS_SGP_MOMENTUM composite — C1-C12 audit.

Pre-reg: docs/audit/hypotheses/2026-04-19-mes-europe-flow-g5-sgp-composite-v1.yaml
Origin: Phase 2.4 third-pass adversarial reframe (commit 01ec8ecd).

K=1, Pathway B, individual. Confirmatory T8-cross-instrument, not new discovery.

Literature grounding (per pre-reg § theory_citation):
  - Chan 2013 Ch 7 (chan_2013_ch7_intraday_momentum.md) — stop-cascade
    mechanism on equity-index futures at session open (FSTX APR 13% Sharpe 1.4)
  - Fitschen 2013 Ch 3 (fitschen_2013_path_of_least_resistance.md) — intraday
    trend-follow on equity indices
  - Chordia et al 2018 — t>=3.00 with-theory threshold
  - Harvey-Liu 2015 Exhibit 4 — N>=100 deployable
  - Bailey et al 2013 — MinBTL (trivially satisfied at K=1)
  - LdP 2020 — WFE >= 0.50

Outputs:
  - stdout summary + verdict
  - research/output/phase_2_4_mes_composite_audit.csv (numeric criteria table)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.data_era import micro_launch_day  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from research.filter_utils import filter_signal  # noqa: E402
from research.mode_a_revalidation_active_setups import (  # noqa: E402
    C4_T_WITH_THEORY,
    C7_MIN_N,
    C9_ERA_THRESHOLD,
    C9_MIN_N_PER_ERA,
)
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "research" / "output"

# Locked surface (matches pre-reg baseline_mode_a_computed block)
INSTRUMENT = "MES"
SESSION = "EUROPE_FLOW"
ORB_MINUTES = 5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR_TARGET = 1.5
DIRECTION = "long"

# C6 Walk-Forward Efficiency threshold — no shared canonical source exists yet
# across trading_app/ or research/ for this constant; mirrored pattern from
# research/htf_path_a_prev_week_v1_scan.py:76 which inlines the same value.
# @research-source docs/institutional/pre_registered_criteria.md § Criterion 6
# @research-source docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md
# @revalidated-for 2026-04-19 (matches pre_registered_criteria.md v2.7 + Amendment 3.0)
C6_WFE_THRESHOLD: float = 0.50

# T0 tautology threshold is a methodology constant from backtesting-methodology.md § RULE 7
# Shared deviation from the quant-audit-protocol threshold of 0.70 is intentional — this
# audit tests a COMPOSITE that is a logical subset of both constituent filters (rho up to
# ~0.9 is expected by construction). The 0.90 threshold flags TRUE repackaging only.
# @research-source .claude/rules/backtesting-methodology.md § RULE 7
T0_TAUT_THRESHOLD: float = 0.90


def fetch_universe(con, instrument: str, direction: str, rr: float) -> pd.DataFrame:
    """Fetch full (IS+OOS) break universe with daily_features joined (triple-join)."""
    sql = """
        SELECT o.trading_day, o.pnl_r, o.outcome, d.*
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
          AND d.orb_EUROPE_FLOW_break_dir = ?
          AND o.pnl_r IS NOT NULL
        ORDER BY o.trading_day
    """
    return con.execute(
        sql,
        [instrument, SESSION, ORB_MINUTES, ENTRY_MODEL, CONFIRM_BARS, rr, direction],
    ).df()


def composite_fire(df: pd.DataFrame) -> np.ndarray:
    g5 = np.asarray(filter_signal(df, "ORB_G5", SESSION)).astype(bool)
    sgp = np.asarray(filter_signal(df, "CROSS_SGP_MOMENTUM", SESSION)).astype(bool)
    return g5 & sgp


def t_stat(expr: float, sd: float, n: int) -> float:
    if n < 2 or sd is None or sd == 0:
        return float("nan")
    return expr / (sd / math.sqrt(n))


def walk_forward_efficiency(df_is: pd.DataFrame, fire_is: np.ndarray) -> tuple[float, list[dict]]:
    """Expanding-window walk-forward: each full IS year is OOS exactly once.
    WFE = OOS_Sharpe / IS_Sharpe (LdP 2020).
    """
    df_is = df_is.copy()
    df_is["year"] = pd.to_datetime(df_is["trading_day"]).dt.year
    years = sorted(df_is["year"].unique())
    if len(years) < 3:
        return float("nan"), []

    pnl = df_is["pnl_r"].to_numpy()
    folds: list[dict] = []
    for y in years[1:]:  # need at least 1 year IS before first OOS
        train_mask = df_is["year"].values < y
        test_mask = df_is["year"].values == y

        train_fire = train_mask & fire_is
        test_fire = test_mask & fire_is

        train_pnl = pnl[train_fire]
        test_pnl = pnl[test_fire]

        if len(train_pnl) < 10 or len(test_pnl) < 5:
            continue

        train_sh = (train_pnl.mean() / train_pnl.std(ddof=1)) if train_pnl.std(ddof=1) > 0 else 0.0
        test_sh = (test_pnl.mean() / test_pnl.std(ddof=1)) if test_pnl.std(ddof=1) > 0 else 0.0
        folds.append({
            "test_year": int(y),
            "train_n": len(train_pnl),
            "train_expr": float(train_pnl.mean()),
            "train_sharpe": float(train_sh),
            "test_n": len(test_pnl),
            "test_expr": float(test_pnl.mean()),
            "test_sharpe": float(test_sh),
        })

    if not folds:
        return float("nan"), folds
    mean_train_sh = np.mean([f["train_sharpe"] for f in folds])
    mean_test_sh = np.mean([f["test_sharpe"] for f in folds])
    wfe = (mean_test_sh / mean_train_sh) if mean_train_sh > 0 else float("nan")
    return float(wfe), folds


def year_break(df: pd.DataFrame, fire: np.ndarray) -> list[dict]:
    df = df.copy()
    df["year"] = pd.to_datetime(df["trading_day"]).dt.year
    pnl = df["pnl_r"].to_numpy()
    out: list[dict] = []
    for y in sorted(df["year"].unique()):
        m = (df["year"].values == y) & fire
        if not m.any():
            out.append({"year": int(y), "n": 0, "expr": None})
            continue
        yp = pnl[m]
        out.append({"year": int(y), "n": int(m.sum()), "expr": float(yp.mean())})
    return out


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True, config={"access_mode": "READ_ONLY"})

    try:
        # === Full universe (IS + OOS) ===
        df_all = fetch_universe(con, INSTRUMENT, DIRECTION, RR_TARGET)
        df_all["td"] = pd.to_datetime(df_all["trading_day"]).dt.date
        is_mask = df_all["td"].values < HOLDOUT_SACRED_FROM
        oos_mask = df_all["td"].values >= HOLDOUT_SACRED_FROM

        df_is = df_all[is_mask].reset_index(drop=True)
        df_oos = df_all[oos_mask].reset_index(drop=True)

        fire_is = composite_fire(df_is)
        fire_oos = composite_fire(df_oos) if len(df_oos) else np.array([], dtype=bool)

        # IS composite
        pnl_is = df_is["pnl_r"].to_numpy()
        comp_is = pnl_is[fire_is]
        n_is = len(comp_is)
        expr_is = float(comp_is.mean()) if n_is else None
        sd_is = float(comp_is.std(ddof=1)) if n_is > 1 else None
        t_is = t_stat(expr_is, sd_is, n_is) if expr_is is not None else float("nan")
        # C4 requires win-rate on the filtered subset, not the full break universe.
        wr_is_on = (
            float((df_is["outcome"][fire_is].astype(str) == "win").mean())
            if n_is
            else None
        )

        # OOS composite
        pnl_oos = df_oos["pnl_r"].to_numpy() if len(df_oos) else np.array([])
        comp_oos = pnl_oos[fire_oos] if len(pnl_oos) else np.array([])
        n_oos = len(comp_oos)
        expr_oos = float(comp_oos.mean()) if n_oos else None

        # p-value — two-sided t-test on IS composite vs null (ExpR=0)
        if n_is > 1 and sd_is and sd_is > 0:
            from scipy import stats
            _t_unused, p_raw = stats.ttest_1samp(comp_is, 0.0)
            # ttest_1samp's t-statistic matches our manual t_is up to float tolerance;
            # we report t_is (derived from ExpR/sd/sqrt(N)) for audit transparency.
            p_val = float(p_raw)
        else:
            p_val = float("nan")

        # C1 — pre-reg file exists
        prereg_path = PROJECT_ROOT / "docs/audit/hypotheses/2026-04-19-mes-europe-flow-g5-sgp-composite-v1.yaml"
        c1 = prereg_path.exists()

        # C2 MinBTL K=1 — trivially passes (2·ln(1)/E[max]² = 0)
        c2 = True

        # C3 BH-FDR K=1 — single hypothesis, so p<0.05 suffices
        c3 = bool(p_val < 0.05) if not math.isnan(p_val) else False

        # C4 Chordia t
        c4 = bool(t_is >= C4_T_WITH_THEORY) if not math.isnan(t_is) else False

        # C6 WFE (walk-forward on IS composite)
        wfe, folds = walk_forward_efficiency(df_is, fire_is)
        c6 = bool(wfe >= C6_WFE_THRESHOLD) if not math.isnan(wfe) else False

        # C7 sample size
        c7 = bool(n_is >= C7_MIN_N)

        # C8 OOS: positive-sign consistency AND ExpR not negative.
        # Per .claude/rules/backtesting-methodology.md RULE 3.2:
        #   N_oos < 5         → UNEVALUABLE (can't compute reliably)
        #   5 <= N_oos < 30   → DIRECTIONAL (evidence is directional-only, low power)
        #   N_oos >= 30       → CONFIRMATORY (full-power OOS check)
        # Pass/fail logic unchanged — the criterion is still met if the sign
        # matches and OOS is positive. Power tier disambiguates how strong the
        # pass is. A DIRECTIONAL pass is honest evidence, not CONFIRMATORY.
        if n_oos < 5:
            c8_power_tier = "UNEVALUABLE"
            c8_direction_match = None
            c8_positive = None
            c8 = False
        else:
            c8_power_tier = "DIRECTIONAL" if n_oos < 30 else "CONFIRMATORY"
            c8_direction_match = bool(
                expr_is is not None and expr_oos is not None
                and ((expr_is > 0 and expr_oos > 0) or (expr_is < 0 and expr_oos < 0))
            )
            c8_positive = bool(expr_oos > 0) if expr_oos is not None else False
            c8 = c8_direction_match and c8_positive

        # C9 era stability
        yb = year_break(df_is, fire_is)
        c9_fail_years = [y for y in yb if y["n"] >= C9_MIN_N_PER_ERA and y["expr"] is not None and y["expr"] < C9_ERA_THRESHOLD]
        c9 = len(c9_fail_years) == 0

        # C10 MICRO-only (first composite trade >= instrument's MICRO-launch date).
        # Delegates to canonical pipeline.data_era.micro_launch_day — never inline
        # the launch date.
        micro_launch = micro_launch_day(INSTRUMENT)
        if n_is:
            first_trade_day = df_is.loc[fire_is, "trading_day"].min()
            first_date = pd.Timestamp(first_trade_day).date()
            c10 = bool(first_date >= micro_launch)
        else:
            first_date = None
            c10 = False

        # T0 tautology — composite should be a PROPER subset, not a repackage
        if n_is:
            g5_is = np.asarray(filter_signal(df_is, "ORB_G5", SESSION)).astype(bool)
            sgp_is = np.asarray(filter_signal(df_is, "CROSS_SGP_MOMENTUM", SESSION)).astype(bool)
            if g5_is.std() > 0 and fire_is.std() > 0:
                rho_comp_g5 = float(np.corrcoef(g5_is.astype(int), fire_is.astype(int))[0, 1])
            else:
                rho_comp_g5 = float("nan")
            if sgp_is.std() > 0 and fire_is.std() > 0:
                rho_comp_sgp = float(np.corrcoef(sgp_is.astype(int), fire_is.astype(int))[0, 1])
            else:
                rho_comp_sgp = float("nan")
            t0_g5_pass = (not math.isnan(rho_comp_g5)) and abs(rho_comp_g5) <= T0_TAUT_THRESHOLD
            t0_sgp_pass = (not math.isnan(rho_comp_sgp)) and abs(rho_comp_sgp) <= T0_TAUT_THRESHOLD
        else:
            rho_comp_g5 = rho_comp_sgp = float("nan")
            t0_g5_pass = t0_sgp_pass = False

        # T5 direction symmetry — short composite should NOT exceed long composite
        df_short = fetch_universe(con, INSTRUMENT, "short", RR_TARGET)
        df_short = df_short[df_short["trading_day"].apply(lambda d: pd.Timestamp(d).date() < HOLDOUT_SACRED_FROM)]
        if len(df_short):
            fire_short = composite_fire(df_short)
            short_pnl = df_short["pnl_r"].to_numpy()[fire_short]
            n_short = len(short_pnl)
            expr_short = float(short_pnl.mean()) if n_short else None
        else:
            n_short = 0
            expr_short = None
        t5_pass = (expr_short is None) or (expr_is is None) or (expr_short <= expr_is)

        # T8 MNQ direction consistency (from prior Phase 2.4 audit)
        df_mnq = fetch_universe(con, "MNQ", DIRECTION, RR_TARGET)
        df_mnq = df_mnq[df_mnq["trading_day"].apply(lambda d: pd.Timestamp(d).date() < HOLDOUT_SACRED_FROM)]
        fire_mnq = composite_fire(df_mnq)
        mnq_pnl = df_mnq["pnl_r"].to_numpy()[fire_mnq] if len(df_mnq) else np.array([])
        expr_mnq = float(mnq_pnl.mean()) if len(mnq_pnl) else None
        t8_pass = (expr_mnq is not None and expr_is is not None and ((expr_mnq > 0) == (expr_is > 0)))

        # ----- Output -----
        criteria_rows = [
            {"criterion": "C1_pre_reg_exists", "pass": c1, "value": str(prereg_path.exists())},
            {"criterion": "C2_minBTL", "pass": c2, "value": "K=1 trivial"},
            {"criterion": "C3_BH_FDR", "pass": c3, "value": f"p={p_val:.5f}"},
            {"criterion": "C4_chordia_t", "pass": c4, "value": f"t={t_is:.3f} (threshold {C4_T_WITH_THEORY})"},
            {"criterion": "C6_WFE", "pass": c6, "value": f"wfe={wfe:.3f} (threshold {C6_WFE_THRESHOLD})"},
            {"criterion": "C7_N_deployable", "pass": c7, "value": f"N={n_is} (threshold {C7_MIN_N})"},
            {"criterion": "C8_OOS_sign_consistent", "pass": c8,
             "value": f"N_oos={n_oos} expr_oos={expr_oos if expr_oos is None else f'{expr_oos:+.4f}'} power_tier={c8_power_tier}"},
            {"criterion": "C9_era_stability", "pass": c9, "value": f"fail_years={[(y['year'], y['n'], round(y['expr'], 4) if y['expr'] else None) for y in c9_fail_years]}"},
            {"criterion": "C10_micro_only", "pass": c10, "value": f"first_trade={first_date}"},
            {"criterion": "T0_tautology_g5", "pass": t0_g5_pass, "value": f"rho={rho_comp_g5:.3f}"},
            {"criterion": "T0_tautology_sgp", "pass": t0_sgp_pass, "value": f"rho={rho_comp_sgp:.3f}"},
            {"criterion": "T5_direction_asymmetry", "pass": t5_pass, "value": f"long_expr={expr_is:+.4f}, short_expr={expr_short if expr_short is None else f'{expr_short:+.4f}'}, short_N={n_short}"},
            {"criterion": "T8_cross_instrument_sign", "pass": t8_pass, "value": f"MES_expr={expr_is:+.4f}, MNQ_expr={expr_mnq:+.4f}" if expr_mnq is not None else "MNQ missing"},
        ]

        result_df = pd.DataFrame(criteria_rows)
        csv_path = OUTPUT_DIR / "phase_2_4_mes_composite_audit.csv"
        result_df.to_csv(csv_path, index=False)

        verdict_parts: list[str] = []
        pass_all = True
        for row in criteria_rows:
            mark = "PASS" if row["pass"] else "FAIL"
            if not row["pass"]:
                pass_all = False
            verdict_parts.append(f"  {row['criterion']:<30} {mark}  {row['value']}")

        verdict = "PASS_FULL_AUDIT" if pass_all else "FAIL_ONE_OR_MORE"

        print("MES EUROPE_FLOW ORB_G5 AND CROSS_SGP_MOMENTUM — C1-C12 AUDIT")
        print(f"Pre-reg: {prereg_path.relative_to(PROJECT_ROOT)}")
        print(f"Mode A cutoff: trading_day < {HOLDOUT_SACRED_FROM}")
        print(f"Surface: {INSTRUMENT} {SESSION} O{ORB_MINUTES} {ENTRY_MODEL} CB{CONFIRM_BARS} RR{RR_TARGET} {DIRECTION}")
        print()
        print(f"IS composite: N={n_is}, ExpR={expr_is:+.4f}" if expr_is is not None else "IS composite: empty")
        print(f"  sd={sd_is:.4f}, t={t_is:+.3f}, p={p_val:.5f}")
        print(f"  WR(composite on IS) = {wr_is_on:.4f}" if wr_is_on is not None else "")
        print(f"OOS composite: N={n_oos}, ExpR={expr_oos if expr_oos is None else f'{expr_oos:+.4f}'}")
        print(f"WFE folds: {len(folds)}, mean_train_Sh={np.mean([f['train_sharpe'] for f in folds]):.3f}, mean_test_Sh={np.mean([f['test_sharpe'] for f in folds]):.3f}, WFE={wfe:.3f}" if folds else "WFE: insufficient folds")
        print()
        print("Year breakdown:")
        for y in yb:
            mark = "**" if (y["n"] >= C9_MIN_N_PER_ERA and y["expr"] is not None and y["expr"] < C9_ERA_THRESHOLD) else "  "
            if y["expr"] is not None:
                print(f"  {mark} {y['year']}  N={y['n']:>3}  ExpR={y['expr']:+.4f}")
            else:
                print(f"  {mark} {y['year']}  N={y['n']:>3}  ExpR=—")
        print()
        print("Criteria:")
        for line in verdict_parts:
            print(line)
        print()
        print(f"VERDICT: {verdict}")
        try:
            print(f"Written: {csv_path.relative_to(PROJECT_ROOT)}")
        except ValueError:
            print(f"Written: {csv_path}")
    finally:
        con.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
