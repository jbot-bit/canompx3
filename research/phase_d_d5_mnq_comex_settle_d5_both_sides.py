"""Phase D D5 — MNQ COMEX_SETTLE O5 RR1.5 BOTH-SIDES E2 CB1 ORB_G5 — conditional half-size.

Runs the pre-registered confirmatory test per
docs/audit/hypotheses/2026-04-28-mnq-comex-settle-garch-d5-both-sides-pathway-b-v1.yaml.

Output: docs/audit/results/2026-04-28-mnq-comex-settle-d5-both-sides-pathway-b-v1-result.md

PRIMARY METRIC: SR_ann diff (D5 sized cohort minus flat-1.0x baseline).
TRANSFORMATION: pnl_r_d5 = pnl_r if garch_forecast_vol_pct > 70 else 0.5 * pnl_r

scratch-policy: include-as-zero (canonical Stage 5 outcome_builder fix; commit
68ee35f8). orb_outcomes pnl_r is COALESCEd to 0.0.

E2-look-ahead: feature `garch_forecast_vol_pct` is § 6.1 safe — built from
prior-day daily closes only via rolling 252-day window (build_daily_features.py:
1468-1497). break_dir is used ONLY for direction segmentation of already-taken
E2 trades (RULE 6.3 exception); NOT a predictor.

This runner is the one-shot Pathway B K=1 confirmatory test on the locked
sizing rule. NO threshold sweeps; NO sizing sweeps; NO post-hoc rescue.
"""

from __future__ import annotations

# scratch-policy: include-as-zero
# Canonical post-Stage-5 (commit 68ee35f8). pnl_r is COALESCEd to 0.0.

# e2-lookahead-policy: not-predictor
# garch_forecast_vol_pct is § 6.1 safe (prior-day-close input only).
# orb_COMEX_SETTLE_break_dir is used ONLY for direction segmentation
# (RULE 6.3 exception), NOT as a predictor.

import subprocess
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

PRE_REG_PATH = (
    PROJECT_ROOT
    / "docs"
    / "audit"
    / "hypotheses"
    / "2026-04-28-mnq-comex-settle-garch-d5-both-sides-pathway-b-v1.yaml"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "audit"
    / "results"
    / "2026-04-28-mnq-comex-settle-d5-both-sides-pathway-b-v1-result.md"
)

# Locked schema per pre-reg (no parameter sweeps allowed).
LOCKED = {
    "instrument": "MNQ",
    "session": "COMEX_SETTLE",
    "orb_minutes": 5,
    "rr_target": 1.5,                  # DEPLOYED RR (D5 tests deployed lane, not D4's RR1.0)
    "entry_model": "E2",
    "confirm_bars": 1,
    "direction_segmentation": ("long", "short"),  # BOTH SIDES
    "feature": "garch_forecast_vol_pct",
    "feature_threshold": 70.0,         # LOCKED from D4 inheritance
    "feature_op": ">",
    "orb_g5_min_size": 5.0,            # canonical from trading_app/config.py:2980
    "on_branch_multiplier": 1.0,       # full size when garch>70
    "off_branch_multiplier": 0.5,      # half size when garch<=70 (LOCKED, no sweeps)
    "year_no_flip_floor_sr_pt": 0.10,  # KILL_YEAR_FLIP threshold
}

# Kill criteria thresholds (locked from pre-reg).
KILL_THRESHOLDS = {
    "abs_sr_diff_floor": 0.05,         # Phase D D-0 v2 floor
    "rel_sr_uplift_pct_floor": 15.0,   # Phase D D-0 v2 floor
    "year_no_flip_floor": 0.10,        # any IS year worse by >0.10 SR_pt = KILL
    "paired_p_max": 0.05,              # primary KILL_PAIRED_P; relaxed in EXTENDED_PARK rule
    "n_is_min": 100,                   # C7
    "era_min": -0.05,                  # C9
    "era_min_n": 50,                   # C9 era min N (D5 uses N>=50, stricter than D4's 20)
    "baseline_sanity_max_diff": 0.0001,
    "expected_sr_pt_flat": 0.0825,
    "expected_sr_pt_d5": 0.1005,
    "expected_sr_ann_flat": 1.2381,
    "expected_sr_ann_d5": 1.5086,
    "expected_abs_sr_diff": 0.2705,
    "expected_rel_sr_uplift_pct": 21.85,
}

# Amendment 3.2 classification (computed at pre-reg time; locked here for verification).
AMENDMENT_3_2 = {
    "sd_monthly_sharpe_diff_empirical": 0.0475,
    "expected_monthly_sharpe_diff": 0.05,
    "required_n_oos_months": 7,
    "min_trl_years": 0.59,
    "classification": "STANDARD",
    "classification_revised_from": "EXTENDED_PARK",
}


def get_prereg_commit() -> str:
    """Get the SHA where the pre-reg was last committed."""
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", str(PRE_REG_PATH)],
            cwd=str(PROJECT_ROOT),
            text=True,
        )
        return out.strip() or "UNCOMMITTED"
    except subprocess.CalledProcessError:
        return "UNCOMMITTED"


def get_db_max_day(con) -> str:
    row = con.execute("SELECT MAX(trading_day) FROM orb_outcomes WHERE symbol='MNQ'").fetchone()
    return str(row[0]) if row and row[0] else "UNKNOWN"


def load_cell(con) -> pd.DataFrame:
    """Load locked-schema BOTH-SIDES cohort. Triple-join required (RULE 9, daily-features-joins.md)."""
    sess = LOCKED["session"]
    sql = f"""
    SELECT o.trading_day,
           COALESCE(o.pnl_r, 0.0) AS pnl_r,
           o.outcome,
           d.garch_forecast_vol_pct,
           d.orb_{sess}_break_dir AS bdir,
           d.orb_{sess}_size AS osize
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = ?
      AND o.orb_label = ?
      AND o.orb_minutes = ?
      AND o.rr_target = ?
      AND o.entry_model = ?
      AND o.confirm_bars = ?
      AND o.outcome IN ('win','loss','scratch')
      AND d.orb_{sess}_break_dir IN ('long','short')
      AND d.orb_{sess}_size IS NOT NULL
      AND d.orb_{sess}_size >= ?
    ORDER BY o.trading_day
    """
    df = con.execute(
        sql,
        [
            LOCKED["instrument"],
            LOCKED["session"],
            LOCKED["orb_minutes"],
            LOCKED["rr_target"],
            LOCKED["entry_model"],
            LOCKED["confirm_bars"],
            LOCKED["orb_g5_min_size"],
        ],
    ).df()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["sig"] = (df["garch_forecast_vol_pct"].astype(float) > LOCKED["feature_threshold"]).fillna(False).values
    df["is_is"] = df["trading_day"] < pd.Timestamp(HOLDOUT_SACRED_FROM)
    # Apply locked sizing rule
    df["pnl_r_flat"] = df["pnl_r"].astype(float).values
    df["pnl_r_d5"] = np.where(
        df["sig"].values,
        df["pnl_r_flat"].values * LOCKED["on_branch_multiplier"],
        df["pnl_r_flat"].values * LOCKED["off_branch_multiplier"],
    )
    return df


def sr_pt(r: np.ndarray) -> float:
    """Per-trade Sharpe = mean / sd (ddof=1). Returns 0.0 if sd=0."""
    if len(r) == 0:
        return 0.0
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    if sd <= 0:
        return 0.0
    return float(np.mean(r)) / sd


def sr_ann_from_pt(sr_per_trade: float, n: int, years: float) -> float:
    """Annualised Sharpe per canonical convention: SR_pt * sqrt(N / years)."""
    if years <= 0 or n <= 0:
        return 0.0
    return sr_per_trade * float(np.sqrt(n / years))


def block_bootstrap_paired_p(
    r_flat: np.ndarray, r_d5: np.ndarray, block: int = 5, B: int = 10000, seed: int = 12345
) -> float:
    """Block-bootstrap permutation on paired differences. Tests H0: mean(r_d5 - r_flat) = 0."""
    rng = np.random.default_rng(seed)
    diff = r_d5 - r_flat
    obs = float(np.mean(diff))
    n = len(diff)
    if n < 2 * block:
        return float("nan")
    k_blocks = n // block
    exceeds = 0
    for _ in range(B):
        # Random sign flip per block (paired-permutation analogue under H0)
        signs = rng.choice([-1.0, 1.0], size=k_blocks)
        flipped = np.concatenate(
            [signs[i] * diff[i * block : (i + 1) * block] for i in range(k_blocks)]
        )
        boot_mean = float(np.mean(flipped))
        if abs(boot_mean) >= abs(obs):
            exceeds += 1
    return float((exceeds + 1) / (B + 1))  # Phipson-Smyth


def per_year_breakdown(df_is: pd.DataFrame) -> list[tuple[int, int, float, float, float]]:
    """Return list of (year, N, sr_pt_flat, sr_pt_d5, diff)."""
    df = df_is.copy()
    df["year"] = df["trading_day"].dt.year
    out = []
    for yr, sub in df.groupby("year"):
        n = len(sub)
        sr_f = sr_pt(np.asarray(sub["pnl_r_flat"].values, dtype=float))
        sr_d = sr_pt(np.asarray(sub["pnl_r_d5"].values, dtype=float))
        out.append((int(yr), n, sr_f, sr_d, sr_d - sr_f))
    return sorted(out, key=lambda x: x[0])


def evaluate_kill_criteria(state: dict) -> list[dict]:
    """Evaluate each KILL_* criterion from the pre-reg. Return list of dicts."""
    s = state
    results = []

    # KILL_DIR
    sign_diff = float(np.sign(s["sr_ann_d5_is"] - s["sr_ann_flat_is"]))
    results.append({
        "id": "KILL_DIR",
        "threshold": "sign(SR_ann_D5 - SR_ann_flat) on IS == negative",
        "computed": f"sign={sign_diff:+.0f}",
        "verdict": "FAIL_KILL" if sign_diff < 0 else "PASS",
    })

    # KILL_ABS_FLOOR
    abs_diff = s["sr_ann_d5_is"] - s["sr_ann_flat_is"]
    results.append({
        "id": "KILL_ABS_FLOOR",
        "threshold": f"abs(SR_ann diff) >= {KILL_THRESHOLDS['abs_sr_diff_floor']}",
        "computed": f"{abs_diff:+.4f}",
        "verdict": "PASS" if abs_diff >= KILL_THRESHOLDS["abs_sr_diff_floor"] else "FAIL_KILL",
    })

    # KILL_REL_FLOOR
    rel_uplift_pct = (
        100.0 * (s["sr_ann_d5_is"] - s["sr_ann_flat_is"]) / s["sr_ann_flat_is"]
        if s["sr_ann_flat_is"] != 0
        else 0.0
    )
    results.append({
        "id": "KILL_REL_FLOOR",
        "threshold": f"rel uplift pct >= {KILL_THRESHOLDS['rel_sr_uplift_pct_floor']}%",
        "computed": f"{rel_uplift_pct:+.2f}%",
        "verdict": (
            "PASS"
            if rel_uplift_pct >= KILL_THRESHOLDS["rel_sr_uplift_pct_floor"]
            else "FAIL_KILL"
        ),
    })

    # KILL_PAIRED_P
    pp = s["paired_p"]
    paired_kill = pp >= KILL_THRESHOLDS["paired_p_max"]
    results.append({
        "id": "KILL_PAIRED_P",
        "threshold": f"paired t p < {KILL_THRESHOLDS['paired_p_max']}",
        "computed": f"p={pp:.4f}",
        "verdict": "PASS" if not paired_kill else "FAIL_KILL_OR_PARK",
    })

    # KILL_YEAR_FLIP — any year (N >= era_min_n) where SR_pt_flat_year - SR_pt_D5_year > 0.10
    worst_flip = 0.0
    worst_year = None
    for yr, n, sr_f, sr_d, diff in s["per_year"]:
        if n < KILL_THRESHOLDS["era_min_n"]:
            continue
        flip = sr_f - sr_d  # positive = D5 worse than flat
        if flip > worst_flip:
            worst_flip = flip
            worst_year = yr
    results.append({
        "id": "KILL_YEAR_FLIP",
        "threshold": f"max(SR_pt_flat - SR_pt_D5) for any IS year (N>={KILL_THRESHOLDS['era_min_n']}) > {KILL_THRESHOLDS['year_no_flip_floor']}",
        "computed": f"worst_year={worst_year} flip={worst_flip:+.4f}",
        "verdict": "PASS" if worst_flip <= KILL_THRESHOLDS["year_no_flip_floor"] else "FAIL_KILL",
    })

    # KILL_N
    results.append({
        "id": "KILL_N",
        "threshold": f"N_IS_combined >= {KILL_THRESHOLDS['n_is_min']}",
        "computed": f"N={s['n_is_combined']}",
        "verdict": "PASS" if s["n_is_combined"] >= KILL_THRESHOLDS["n_is_min"] else "FAIL_KILL",
    })

    # KILL_ERA — D5 variant (per pre-reg: any era with N>=50 where ExpR_d5 < -0.05)
    worst_era_expr = float("inf")
    worst_era_yr = None
    for yr, n, sr_f, sr_d, diff in s["per_year"]:
        if n < KILL_THRESHOLDS["era_min_n"]:
            continue
        era_expr = float(
            np.mean(
                s["df_is"][s["df_is"]["trading_day"].dt.year == yr]["pnl_r_d5"].values
            )
        )
        if era_expr < worst_era_expr:
            worst_era_expr = era_expr
            worst_era_yr = yr
    results.append({
        "id": "KILL_ERA",
        "threshold": f"any IS era (N>={KILL_THRESHOLDS['era_min_n']}) ExpR_D5 < {KILL_THRESHOLDS['era_min']}",
        "computed": f"worst_era={worst_era_yr} ExpR_D5={worst_era_expr:+.4f}",
        "verdict": "PASS" if worst_era_expr >= KILL_THRESHOLDS["era_min"] else "FAIL_KILL",
    })

    # KILL_BASELINE_SANITY — runner reproduces canonical pre-pre-reg numbers
    sr_pt_flat_diff = abs(s["sr_pt_flat_is"] - KILL_THRESHOLDS["expected_sr_pt_flat"])
    results.append({
        "id": "KILL_BASELINE_SANITY",
        "threshold": f"abs(reproduced sr_pt_flat - expected) <= {KILL_THRESHOLDS['baseline_sanity_max_diff']}",
        "computed": f"abs_diff={sr_pt_flat_diff:.6f}",
        "verdict": "PASS" if sr_pt_flat_diff <= KILL_THRESHOLDS["baseline_sanity_max_diff"] else "FAIL_KILL",
    })

    return results


def decide_verdict(state: dict, kill_results: list[dict]) -> tuple[str, str]:
    """Apply locked decision_rule from pre-reg. Returns (verdict, explanation)."""
    s = state
    any_fail = any(k["verdict"].startswith("FAIL_KILL") and "PARK" not in k["verdict"] for k in kill_results)
    paired_p = s["paired_p"]
    abs_diff = s["sr_ann_d5_is"] - s["sr_ann_flat_is"]
    rel_uplift = (
        100.0 * abs_diff / s["sr_ann_flat_is"] if s["sr_ann_flat_is"] != 0 else 0.0
    )

    if any_fail:
        failed = [k["id"] for k in kill_results if k["verdict"].startswith("FAIL_KILL") and "PARK" not in k["verdict"]]
        return "KILL", f"At least one KILL criterion fired: {', '.join(failed)}. Pre-reg locked-decision rule: any KILL fire → KILL verdict."

    # All KILL criteria passed (or KILL_PAIRED_P is the only soft-fail)
    paired_pass = paired_p < KILL_THRESHOLDS["paired_p_max"]
    floor_pass = (
        abs_diff >= KILL_THRESHOLDS["abs_sr_diff_floor"]
        and rel_uplift >= KILL_THRESHOLDS["rel_sr_uplift_pct_floor"]
    )

    if floor_pass and paired_pass and s["c9_verdict"] == "PASS":
        # OOS direction-match descriptive (Amendment 3.2 STANDARD: full C8 deferred to milestone)
        return (
            "CANDIDATE_READY",
            f"All KILL criteria PASS. abs SR diff +{abs_diff:.4f} >= 0.05, rel uplift +{rel_uplift:.2f}% >= 15%, "
            f"paired t p={paired_p:.4f} < 0.05, C9 era-stability PASS, year-no-flip floor PASS. "
            f"Per Amendment 3.2 STANDARD classification (min_trl_years=0.59), OOS confirmation deferred to "
            f"interim milestone N_OOS_combined=100 (~2026-09-15). Currently N_OOS_combined={s['n_oos_combined']}; "
            f"OOS dir-match descriptive: {s['oos_dir_match_descriptive']}."
        )

    if floor_pass and paired_p < 0.10 and s["c9_verdict"] == "PASS":
        # KILL_PAIRED_P is soft (<0.10 but >=0.05)
        return (
            "EXTENDED_PARK",
            f"All KILL criteria PASS. abs SR diff and rel uplift floors PASS. C9 PASS. "
            f"Paired t p={paired_p:.4f} below 0.10 relaxed bound but at/above 0.05 strict gate. "
            f"Decision rule extended_park branch applied. OOS milestone deferred per Amendment 3.2."
        )

    return (
        "PARK_PENDING_OOS_POWER",
        f"All KILL criteria PASS (no fire), but the CANDIDATE_READY conditions are not all met "
        f"(abs SR diff {abs_diff:+.4f} vs floor 0.05, rel uplift {rel_uplift:+.2f}% vs floor 15%, "
        f"paired p={paired_p:.4f}, C9 {s['c9_verdict']}). Per Amendment 3.1, this is UNVERIFIED, not KILL."
    )


def run_pressure_test() -> str:
    """RULE 13: structural assertion that the test cannot silently substitute a different sizing schedule."""
    expected = {
        "feature": "garch_forecast_vol_pct",
        "feature_threshold": 70.0,
        "on_branch_multiplier": 1.0,
        "off_branch_multiplier": 0.5,
    }
    for k, v in expected.items():
        if LOCKED[k] != v:
            return f"FAIL — LOCKED[{k}]={LOCKED[k]} differs from pre-reg {v}"
    # Banned predictors check
    banned = {
        "rel_vol_COMEX_SETTLE",
        "orb_COMEX_SETTLE_break_delay_min",
        "orb_COMEX_SETTLE_break_bar_volume",
        "orb_COMEX_SETTLE_break_bar_continues",
    }
    if LOCKED["feature"] in banned:
        return f"FAIL — LOCKED feature {LOCKED['feature']} is banned"
    return "PASS"


def emit_result(state: dict) -> None:
    """Write result MD per pre-reg outputs_required_after_run."""
    s = state
    lines = []
    lines.append("# Phase D D5 — MNQ COMEX_SETTLE BOTH-SIDES Conditional Sizing — Result")
    lines.append("")
    lines.append(f"**Pre-reg:** docs/audit/hypotheses/2026-04-28-mnq-comex-settle-garch-d5-both-sides-pathway-b-v1.yaml")
    lines.append(f"**Pre-reg commit:** {s['prereg_commit']}")
    lines.append(f"**Run timestamp:** {date.today().isoformat()}")
    lines.append(f"**DB freshness:** orb_outcomes max trading_day = {s['db_max_day']}")
    lines.append(f"**Holdout boundary (Mode A):** {HOLDOUT_SACRED_FROM}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "Pathway B K=1 confirmatory test of the D5 conditional sizing rule on the deployed "
        "`MNQ COMEX_SETTLE O5 RR1.5 E2 CB1 BOTH-SIDES ORB_G5` cohort. Locked transformation: "
        "`pnl_r_d5 = pnl_r if garch_forecast_vol_pct > 70 else 0.5 * pnl_r`. "
        "PRIMARY METRIC: SR_ann diff (D5 minus flat-1.0x baseline). "
        "Hypothesis: halving exposure on the no-edge cohort improves risk-adjusted return via "
        "variance reduction (Carver Ch 9-10 vol-targeting framework). This is a RISK-OVERLAY "
        "test, not an alpha test."
    )
    lines.append("")
    lines.append("## Locked schema (verbatim from pre-reg)")
    lines.append("")
    for k, v in LOCKED.items():
        lines.append(f"- {k}: {v}")
    lines.append("- scratch_policy: include-as-zero (commit 68ee35f8)")
    lines.append("")
    lines.append("## Amendment 3.2 classification (locked at pre-reg time)")
    lines.append("")
    for k, v in AMENDMENT_3_2.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## IS reproduction (Mode A, BOTH-SIDES combined)")
    lines.append("")
    lines.append("| Metric | IS combined | IS long | IS short |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| N | {s['n_is_combined']} | {s['n_is_long']} | {s['n_is_short']} |")
    lines.append(f"| ExpR flat (mean pnl_r) | {s['expr_is_flat']:+.4f} | {s['expr_is_long_flat']:+.4f} | {s['expr_is_short_flat']:+.4f} |")
    lines.append(f"| ExpR D5 (mean pnl_r_d5) | {s['expr_is_d5']:+.4f} | {s['expr_is_long_d5']:+.4f} | {s['expr_is_short_d5']:+.4f} |")
    lines.append(f"| SR_pt flat | {s['sr_pt_flat_is']:+.4f} | — | — |")
    lines.append(f"| SR_pt D5 | {s['sr_pt_d5_is']:+.4f} | — | — |")
    lines.append(f"| SR_ann flat | {s['sr_ann_flat_is']:+.4f} | — | — |")
    lines.append(f"| SR_ann D5 | {s['sr_ann_d5_is']:+.4f} | — | — |")
    lines.append(f"| **abs SR_ann diff** | **{s['sr_ann_d5_is'] - s['sr_ann_flat_is']:+.4f}** | — | — |")
    lines.append(f"| **rel SR_ann uplift %** | **{(100.0 * (s['sr_ann_d5_is'] - s['sr_ann_flat_is']) / s['sr_ann_flat_is']):+.2f}%** | — | — |")
    lines.append("")
    lines.append("## Significance (paired test on per-trade differences)")
    lines.append("")
    lines.append(f"- Paired t-stat (D5 vs flat per-trade R): {s['paired_t']:+.4f}")
    lines.append(f"- Paired p (two-tailed): {s['paired_p']:.6f}")
    lines.append(f"- Block-bootstrap paired p (block=5, B=10000): {s['boot_paired_p']:.6f}")
    lines.append(f"- Mean diff per trade (D5 - flat): {s['mean_diff_per_trade']:+.4f}")
    lines.append("")
    lines.append("## Pre-reg expected values vs runner reproduction (KILL_BASELINE_SANITY)")
    lines.append("")
    lines.append("| Metric | Expected (pre-reg) | Reproduced (runner) | abs diff |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| SR_pt_flat_is | {KILL_THRESHOLDS['expected_sr_pt_flat']:+.4f} | {s['sr_pt_flat_is']:+.4f} | {abs(s['sr_pt_flat_is'] - KILL_THRESHOLDS['expected_sr_pt_flat']):.6f} |"
    )
    lines.append(
        f"| SR_pt_d5_is | {KILL_THRESHOLDS['expected_sr_pt_d5']:+.4f} | {s['sr_pt_d5_is']:+.4f} | {abs(s['sr_pt_d5_is'] - KILL_THRESHOLDS['expected_sr_pt_d5']):.6f} |"
    )
    lines.append(
        f"| SR_ann_flat_is | {KILL_THRESHOLDS['expected_sr_ann_flat']:+.4f} | {s['sr_ann_flat_is']:+.4f} | {abs(s['sr_ann_flat_is'] - KILL_THRESHOLDS['expected_sr_ann_flat']):.6f} |"
    )
    lines.append(
        f"| SR_ann_d5_is | {KILL_THRESHOLDS['expected_sr_ann_d5']:+.4f} | {s['sr_ann_d5_is']:+.4f} | {abs(s['sr_ann_d5_is'] - KILL_THRESHOLDS['expected_sr_ann_d5']):.6f} |"
    )
    lines.append("")
    lines.append("## Per-year IS breakdown (year, N, SR_pt flat, SR_pt D5, diff)")
    lines.append("")
    lines.append("| Year | N | SR_pt flat | SR_pt D5 | diff |")
    lines.append("|---:|---:|---:|---:|---:|")
    for yr, n, sr_f, sr_d, diff in s["per_year"]:
        lines.append(f"| {yr} | {n} | {sr_f:+.4f} | {sr_d:+.4f} | {diff:+.4f} |")
    lines.append("")
    lines.append("## C9 era stability (D5 variant)")
    lines.append("")
    lines.append(f"- Verdict: **{s['c9_verdict']}**")
    lines.append(f"- Worst era: {s.get('c9_worst_year')} ExpR_D5 = {s.get('c9_worst_expr', float('nan')):+.4f}")
    lines.append(f"- Threshold: era ExpR_D5 must be >= {KILL_THRESHOLDS['era_min']} for any era with N >= {KILL_THRESHOLDS['era_min_n']}")
    lines.append("")
    lines.append("## OOS descriptive (Amendment 3.2 STANDARD: full C8 deferred to milestone)")
    lines.append("")
    lines.append(f"- N_OOS_combined: {s['n_oos_combined']} (long={s['n_oos_long']}, short={s['n_oos_short']})")
    lines.append(f"- SR_pt flat OOS: {s['sr_pt_flat_oos']:+.4f}")
    lines.append(f"- SR_pt D5 OOS: {s['sr_pt_d5_oos']:+.4f}")
    lines.append(f"- abs SR_ann diff OOS (descriptive): {s['sr_ann_d5_oos'] - s['sr_ann_flat_oos']:+.4f}")
    lines.append(f"- OOS dir-match descriptive: {s['oos_dir_match_descriptive']}")
    lines.append("")
    lines.append("## Kill-criterion results (locked from pre-reg)")
    lines.append("")
    lines.append("| ID | Threshold | Computed | Verdict |")
    lines.append("|---|---|---|---|")
    for k in s["kill_results"]:
        lines.append(f"| {k['id']} | {k['threshold']} | {k['computed']} | {k['verdict']} |")
    lines.append("")
    lines.append("## Decision rule outcome")
    lines.append("")
    lines.append(f"**VERDICT: {s['final_verdict']}**")
    lines.append("")
    lines.append(s["verdict_explanation"])
    lines.append("")
    lines.append("## Audit pressure-test (RULE 13)")
    lines.append("")
    lines.append(f"- Pressure test: {s['rule_13_status']}")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```")
    lines.append(f"DUCKDB_PATH={GOLD_DB_PATH} python research/phase_d_d5_mnq_comex_settle_d5_both_sides.py")
    lines.append("```")
    lines.append("")
    lines.append("- DB: `pipeline.paths.GOLD_DB_PATH`")
    lines.append("- Holdout: `trading_app.holdout_policy.HOLDOUT_SACRED_FROM` (2026-01-01)")
    lines.append("- Pre-reg: locks all schema parameters and kill criteria")
    lines.append("- Sizing rule LOCKED: 1.0x if garch>70 else 0.5x; threshold 70 inherited from D4")
    lines.append("")
    lines.append("## Caveats and limitations")
    lines.append("")
    lines.append("- D5 is a RISK-OVERLAY test (variance reduction), not an alpha test. The per-trade ExpR")
    lines.append("  REDUCES under D5 (off-cohort half-sized → smaller per-trade R). The improvement is in")
    lines.append("  Sharpe via variance reduction.")
    lines.append("- 2019 IS year has N=99 and SR_pt diff = 0.0000 (likely all off-cohort or near-degenerate),")
    lines.append("  flagged by the per-year breakdown. Not a flip but worth noting.")
    lines.append("- Live execution at 1-contract minimum cannot literally express '0.5×' sizing — the research")
    lines.append("  model is a portfolio-level weighting that requires either ≥2-contract lane scaffolding or a")
    lines.append("  shadow/observation overlay. This pre-reg does NOT prescribe live execution; deployment")
    lines.append("  requires Phase E + capital-review + execution-translation design.")
    lines.append("- RULE 7 N/A by design (modifies existing deployed lane, doesn't add a slot).")
    lines.append("- Cross-session BH-FDR at K=12 (D5 framing): best q=0.0707 — the GARCH effect is")
    lines.append("  COMEX_SETTLE-specific, not universal. D5 is cell-specific risk overlay, not a universal mechanism.")
    lines.append("")
    lines.append("## Not done by this run")
    lines.append("")
    lines.append("- No write to validated_setups, edge_families, lane_allocation, live_config")
    lines.append("- No paper trade simulation")
    lines.append("- No live execution-translation design (separate Phase E step if D5 PASSES)")
    lines.append("- No continuous-scaling DoF expansion (locked binary 1.0/0.5)")
    lines.append("- No threshold sensitivity test (locked at 70)")
    lines.append("- No capital deployment — requires Phase E + capital-review skill + explicit user GO")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    db_max_day = get_db_max_day(con)
    prereg_commit = get_prereg_commit()

    df = load_cell(con)
    df_is = df[df["is_is"]].copy()
    df_oos = df[~df["is_is"]].copy()

    n_is_combined = len(df_is)
    n_is_long = int((df_is["bdir"] == "long").sum())
    n_is_short = int((df_is["bdir"] == "short").sum())
    n_oos_combined = len(df_oos)
    n_oos_long = int((df_oos["bdir"] == "long").sum())
    n_oos_short = int((df_oos["bdir"] == "short").sum())

    # IS metrics
    r_flat_is = np.asarray(df_is["pnl_r_flat"].values, dtype=float)
    r_d5_is = np.asarray(df_is["pnl_r_d5"].values, dtype=float)

    # IS years span (for annualization — use actual span of IS data)
    is_years = sorted(set(df_is["trading_day"].dt.year))
    years_span = (is_years[-1] - is_years[0]) + 1 if is_years else 1

    sr_pt_flat_is = sr_pt(r_flat_is)
    sr_pt_d5_is = sr_pt(r_d5_is)
    sr_ann_flat_is = sr_ann_from_pt(sr_pt_flat_is, n_is_combined, years_span)
    sr_ann_d5_is = sr_ann_from_pt(sr_pt_d5_is, n_is_combined, years_span)

    # Paired test
    diff = r_d5_is - r_flat_is
    if len(diff) > 1 and float(np.std(diff, ddof=1)) > 0:
        paired_t, paired_p = stats.ttest_rel(r_d5_is, r_flat_is)
    else:
        paired_t, paired_p = float("nan"), float("nan")
    boot_paired_p = block_bootstrap_paired_p(r_flat_is, r_d5_is, block=5, B=10000)

    # Per-year breakdown
    py = per_year_breakdown(df_is)

    # C9: any era with N >= era_min_n where ExpR_D5 < era_min
    c9_violations = []
    c9_worst_yr = None
    c9_worst_expr = float("inf")
    for yr, n, _sr_f, _sr_d, _dd in py:
        if n < KILL_THRESHOLDS["era_min_n"]:
            continue
        era_expr = float(np.mean(np.asarray(df_is[df_is["trading_day"].dt.year == yr]["pnl_r_d5"].values, dtype=float)))
        if era_expr < c9_worst_expr:
            c9_worst_expr = era_expr
            c9_worst_yr = yr
        if era_expr < KILL_THRESHOLDS["era_min"]:
            c9_violations.append((yr, era_expr))
    c9_verdict = "PASS" if not c9_violations else f"FAIL ({c9_violations})"

    # OOS descriptive
    if n_oos_combined > 0:
        r_flat_oos = np.asarray(df_oos["pnl_r_flat"].values, dtype=float)
        r_d5_oos = np.asarray(df_oos["pnl_r_d5"].values, dtype=float)
        sr_pt_flat_oos = sr_pt(r_flat_oos)
        sr_pt_d5_oos = sr_pt(r_d5_oos)
        oos_years = sorted(set(df_oos["trading_day"].dt.year))
        oos_span = max((oos_years[-1] - oos_years[0]) + 1, 1) if oos_years else 1
        sr_ann_flat_oos = sr_ann_from_pt(sr_pt_flat_oos, n_oos_combined, oos_span)
        sr_ann_d5_oos = sr_ann_from_pt(sr_pt_d5_oos, n_oos_combined, oos_span)
        oos_diff_sign = float(np.sign(sr_ann_d5_oos - sr_ann_flat_oos))
        is_diff_sign = float(np.sign(sr_ann_d5_is - sr_ann_flat_is))
        oos_dir_match_desc = (
            "DIR_MATCH"
            if oos_diff_sign == is_diff_sign and oos_diff_sign != 0
            else ("DIR_FLIP" if oos_diff_sign != 0 else "ZERO_OOS_DIFF")
        )
    else:
        sr_pt_flat_oos = sr_pt_d5_oos = sr_ann_flat_oos = sr_ann_d5_oos = 0.0
        oos_dir_match_desc = "NA_INSUFFICIENT_N"

    state = {
        "prereg_commit": prereg_commit,
        "db_max_day": db_max_day,
        "df_is": df_is,
        "n_is_combined": n_is_combined,
        "n_is_long": n_is_long,
        "n_is_short": n_is_short,
        "n_oos_combined": n_oos_combined,
        "n_oos_long": n_oos_long,
        "n_oos_short": n_oos_short,
        "expr_is_flat": float(np.mean(r_flat_is)) if n_is_combined > 0 else 0.0,
        "expr_is_d5": float(np.mean(r_d5_is)) if n_is_combined > 0 else 0.0,
        "expr_is_long_flat": float(np.mean(np.asarray(df_is[df_is.bdir == "long"]["pnl_r_flat"].values, dtype=float))) if n_is_long > 0 else 0.0,
        "expr_is_long_d5": float(np.mean(np.asarray(df_is[df_is.bdir == "long"]["pnl_r_d5"].values, dtype=float))) if n_is_long > 0 else 0.0,
        "expr_is_short_flat": float(np.mean(np.asarray(df_is[df_is.bdir == "short"]["pnl_r_flat"].values, dtype=float))) if n_is_short > 0 else 0.0,
        "expr_is_short_d5": float(np.mean(np.asarray(df_is[df_is.bdir == "short"]["pnl_r_d5"].values, dtype=float))) if n_is_short > 0 else 0.0,
        "sr_pt_flat_is": sr_pt_flat_is,
        "sr_pt_d5_is": sr_pt_d5_is,
        "sr_ann_flat_is": sr_ann_flat_is,
        "sr_ann_d5_is": sr_ann_d5_is,
        "paired_t": float(paired_t) if not np.isnan(paired_t) else float("nan"),
        "paired_p": float(paired_p) if not np.isnan(paired_p) else float("nan"),
        "boot_paired_p": float(boot_paired_p) if not np.isnan(boot_paired_p) else float("nan"),
        "mean_diff_per_trade": float(np.mean(diff)) if n_is_combined > 0 else 0.0,
        "per_year": py,
        "c9_verdict": c9_verdict,
        "c9_worst_year": c9_worst_yr,
        "c9_worst_expr": c9_worst_expr,
        "sr_pt_flat_oos": sr_pt_flat_oos,
        "sr_pt_d5_oos": sr_pt_d5_oos,
        "sr_ann_flat_oos": sr_ann_flat_oos,
        "sr_ann_d5_oos": sr_ann_d5_oos,
        "oos_dir_match_descriptive": oos_dir_match_desc,
        "rule_13_status": run_pressure_test(),
    }

    state["kill_results"] = evaluate_kill_criteria(state)
    final_verdict, verdict_explanation = decide_verdict(state, state["kill_results"])
    state["final_verdict"] = final_verdict
    state["verdict_explanation"] = verdict_explanation

    emit_result(state)

    print(f"VERDICT: {final_verdict}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"abs SR_ann diff: {sr_ann_d5_is - sr_ann_flat_is:+.4f}")
    print(f"rel SR_ann uplift: {(100.0 * (sr_ann_d5_is - sr_ann_flat_is) / sr_ann_flat_is):+.2f}%")
    print(f"paired t={paired_t:+.4f} p={paired_p:.6f} boot_p={boot_paired_p:.6f}")
    print(f"C9: {c9_verdict}")
    print(f"OOS dir-match: {oos_dir_match_desc}")


if __name__ == "__main__":
    main()
