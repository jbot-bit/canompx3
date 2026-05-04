"""Phase D D1 — MES EUROPE_FLOW O15 RR1.0 long + ovn_range_pct > 80 — Pathway B K=1.

Runs the pre-registered confirmatory test per
docs/audit/hypotheses/2026-04-28-mes-europe-flow-ovn-range-pathway-b-v1.yaml.

Output: docs/audit/results/2026-04-28-mes-europe-flow-pathway-b-v1-result.md

scratch-policy: include-as-zero (canonical Stage 5 outcome_builder fix; commit
68ee35f8). orb_outcomes pnl_r is COALESCEd to 0.0 for the residual <0.5%
of scratch-EOD-no-MTM rows that legitimately remain NULL per the Stage 4
spec (docs/specs/outcome_builder_scratch_eod_mtm.md:368-372).

E2-look-ahead: feature `overnight_range_pct` is § 6.1 safe (RULE 1.2 valid
domain — overnight_* features valid for ORB sessions starting >= 17:00
Brisbane; EUROPE_FLOW starts 18:00). break_dir is used ONLY for direction
segmentation of already-taken E2 trades (RULE 6.3 exception); it is NOT
used as a predictor.

Reproduces independent SQL verified 2026-04-28: delta_IS=0.2459 exact match.

This runner is the one-shot Pathway B K=1 confirmatory test. NO threshold
sweeps; NO post-hoc rescue. The pre-reg is committed first; the runner
emits the verdict per locked decision_rule.
"""

from __future__ import annotations

# scratch-policy: include-as-zero
# Canonical post-Stage-5 (commit 68ee35f8). pnl_r is COALESCEd to 0.0 only for
# the residual ≤0.5% of rows that legitimately keep NULL pnl_r per the Stage 4
# spec (no later bar for EOD MTM). Verified live 2026-04-28: MES 99.65%,
# MGC 99.82%, MNQ 99.72% scratch-pnl populated. Drift check #N+1
# `check_orb_outcomes_scratch_pnl` enforces the ≥99% floor.

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

# Reuse Phase B helpers — same canonical math, no re-implementation.
sys.path.insert(0, str(PROJECT_ROOT / "research"))
from phase_b_candidate_evidence_v1 import (  # noqa: E402
    annualised_sharpe,
    cohens_d,
    dsr_eq2,
    per_trade_sharpe,
    power_for_d,
    sr0_threshold,
)

PRE_REG_PATH = (
    PROJECT_ROOT
    / "docs"
    / "audit"
    / "hypotheses"
    / "2026-04-28-mes-europe-flow-ovn-range-pathway-b-v1.yaml"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "audit"
    / "results"
    / "2026-04-28-mes-europe-flow-pathway-b-v1-result.md"
)

# Locked schema per pre-reg (no parameter sweeps allowed).
LOCKED = {
    "instrument": "MES",
    "session": "EUROPE_FLOW",
    "orb_minutes": 15,
    "rr_target": 1.0,
    "entry_model": "E2",
    "confirm_bars": 1,
    "direction": "long",
    "feature": "overnight_range_pct",
    "feature_threshold": 80.0,
    "feature_op": ">",  # strict greater-than
}

# Kill criteria thresholds (locked; do not edit without amending pre-reg).
KILL_THRESHOLDS = {
    "raw_p_max": 0.05,
    "abs_t_min": 3.0,  # Chordia with-theory floor
    "n_is_min": 100,  # C7
    "era_min": -0.05,
    "sharpe_ann_min": 0.0,  # Amendment 3.0 condition 2b
    "baseline_sanity_max_diff": 0.001,
    "expected_delta_is": 0.2459,
}

DSR_THRESHOLD = 0.95  # C5
WFE_THRESHOLD = 0.50  # C6 (only applied when N_OOS_on >= MIN_N_OOS_FOR_C6)
MIN_N_OOS_FOR_C6 = 50  # C8 power floor (Amendment 3.2)
OOS_POWER_FLOOR = 0.50  # power floor for OOS dir-match
ERA_MIN_N = 20  # C9 era stability minimum-N per era

# Pathway A discovery K (PROVENANCE_ONLY — for documentation, not promotion gate).
K_FAMILY_OVERNIGHT_PROVENANCE = 1850


def load_cell(con):
    """Load locked-schema cell. Returns DataFrame with pnl_r, sig, is_is."""
    sess = LOCKED["session"]
    sql = f"""
    SELECT o.trading_day,
           COALESCE(o.pnl_r, 0.0) AS pnl_r,
           o.outcome,
           d.overnight_range_pct
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
      AND d.orb_{sess}_break_dir = ?
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
            LOCKED["direction"],
        ],
    ).df()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["sig"] = (df["overnight_range_pct"].astype(float) > LOCKED["feature_threshold"]).fillna(False).values
    df["is_is"] = df["trading_day"] < pd.Timestamp(HOLDOUT_SACRED_FROM)
    return df


def block_bootstrap_p(on: np.ndarray, off: np.ndarray, block: int = 5, B: int = 10000, seed: int = 12345) -> float:
    """Block-bootstrap permutation test on shuffled signal labels."""
    rng = np.random.default_rng(seed)
    obs = float(np.mean(on) - np.mean(off))
    pooled = np.concatenate([on, off])
    n_on = len(on)
    n_total = len(pooled)
    if n_total < 2 * block:
        return float("nan")
    k_blocks = n_total // block
    exceeds = 0
    for _ in range(B):
        # Shuffle by blocks (preserves short-range autocorrelation)
        block_order = rng.permutation(k_blocks)
        shuffled = np.concatenate([pooled[i * block : (i + 1) * block] for i in block_order])
        if len(shuffled) < n_on:
            continue
        boot_on = shuffled[:n_on]
        boot_off = shuffled[n_on:]
        boot_diff = float(np.mean(boot_on) - np.mean(boot_off))
        if abs(boot_diff) >= abs(obs):
            exceeds += 1
    return float((exceeds + 1) / (B + 1))  # Phipson-Smyth


def era_stability(df_is_on: pd.DataFrame) -> tuple[str, list[tuple[int, int, float]]]:
    """Per-year breakdown; C9 PASS if no era with N >= ERA_MIN_N has ExpR < ERA_MIN."""
    if df_is_on.empty:
        return "C9_NO_DATA", []
    yrs = df_is_on["trading_day"].dt.year
    df = df_is_on.assign(year=yrs)
    rows = df.groupby("year")["pnl_r"].agg(["count", "mean"]).reset_index()
    breakdown = [(int(r["year"]), int(r["count"]), float(r["mean"])) for _, r in rows.iterrows()]
    eligible = [b for b in breakdown if b[1] >= ERA_MIN_N]
    if not eligible:
        return "C9_NO_ELIGIBLE_ERAS", breakdown
    fails = [b for b in eligible if b[2] < KILL_THRESHOLDS["era_min"]]
    return ("C9_FAIL" if fails else "C9_PASS"), breakdown


def emit_result(state: dict) -> None:
    """Write result MD to docs/audit/results/."""
    s = state
    lines = []
    lines.append("# Phase D D1 — MES EUROPE_FLOW Pathway B K=1 — Result")
    lines.append("")
    lines.append("**Pre-reg:** docs/audit/hypotheses/2026-04-28-mes-europe-flow-ovn-range-pathway-b-v1.yaml")
    lines.append(f"**Pre-reg commit:** {s['prereg_commit']}")
    lines.append(f"**Run timestamp:** {date.today().isoformat()}")
    lines.append(f"**DB freshness:** orb_outcomes max trading_day = {s['db_max_day']}")
    lines.append(f"**Holdout boundary (Mode A):** {HOLDOUT_SACRED_FROM}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "Pathway B K=1 confirmatory test of B-MES-EUR (the highest-EV "
        "PATHWAY_B_ELIGIBLE candidate from the 2026-04-28 Phase B per-candidate "
        "evidence pass). The question this run answers: under the locked schema, "
        "do all KILL criteria pass and do the C5/C6/C7/C8/C9 gates support "
        "promotion to CANDIDATE_READY, or does the OOS power floor force a "
        "PARK verdict, or does any KILL criterion fire?"
    )
    lines.append("")
    lines.append("## Locked schema (verbatim from pre-reg)")
    lines.append("")
    for k, v in LOCKED.items():
        lines.append(f"- {k}: {v}")
    lines.append(f"- scratch_policy: include-as-zero (commit 68ee35f8)")
    lines.append("")
    lines.append("## IS / OOS reproduction (Mode A)")
    lines.append("")
    lines.append("| Metric | IS | OOS |")
    lines.append("|---|---:|---:|")
    lines.append(f"| N total | {s['n_is']} | {s['n_oos']} |")
    lines.append(f"| N on (signal=true) | {s['n_is_on']} | {s['n_oos_on']} |")
    lines.append(f"| N off | {s['n_is_off']} | {s['n_oos_off']} |")
    lines.append(f"| ExpR on | {s['expr_is_on']:+.4f} | {s['expr_oos_on']:+.4f} |")
    lines.append(f"| ExpR off | {s['expr_is_off']:+.4f} | {s['expr_oos_off']:+.4f} |")
    lines.append(f"| Δ (on − off) | {s['delta_is']:+.4f} | {s['delta_oos']:+.4f} |")
    lines.append(f"| Sharpe per-trade | {s['sr_pt_is']:+.4f} | {s.get('sr_pt_oos', float('nan')):+.4f} |")
    lines.append(f"| Sharpe annualised | {s['sr_ann_is']:+.4f} | — (UNVERIFIED if N_OOS<{MIN_N_OOS_FOR_C6}) |")
    lines.append(f"| Skewness γ̂₃ | {s['skew_is']:+.4f} | — |")
    lines.append(f"| Kurtosis γ̂₄ (excess) | {s['kurt_is']:+.4f} | — |")
    lines.append("")
    lines.append("## Significance (IS, K=1 — see pre-reg testing_discipline)")
    lines.append("")
    lines.append(f"- Welch t-stat: {s['welch_t']:+.4f}")
    lines.append(f"- Welch p (two-tailed): {s['welch_p']:.6f}")
    lines.append(f"- Block-bootstrap p (block=5, B=10000): {s['boot_p']:.6f}")
    lines.append(f"- Cohen's d (IS effect): {s['cohens_d']:+.4f}")
    lines.append(f"- OOS power for d at N_OOS_on={s['n_oos_on']}: {s['oos_power']:.4f}")
    lines.append("")
    lines.append("## Phase 0 gates (post-Amendment 3.0/3.2)")
    lines.append("")
    lines.append("| Gate | Threshold | Computed | Verdict |")
    lines.append("|---|---|---|---|")
    lines.append(f"| C5 DSR (Pathway A K_family={K_FAMILY_OVERNIGHT_PROVENANCE}) [provenance] | ≥ {DSR_THRESHOLD} | {s['dsr_pa']:.4f} | {s['dsr_pa_verdict']} |")
    lines.append(f"| C5 DSR (Pathway B K=1) [primary gate] | ≥ {DSR_THRESHOLD} | {s['dsr_pb']:.4f} | {s['dsr_pb_verdict']} |")
    lines.append(f"| C6 WFE | ≥ {WFE_THRESHOLD} (if N_OOS≥{MIN_N_OOS_FOR_C6}) | {s['wfe_str']} | {s['c6_verdict']} |")
    lines.append(f"| C7 N_IS_on | ≥ {KILL_THRESHOLDS['n_is_min']} | {s['n_is_on']} | {s['c7_verdict']} |")
    lines.append(f"| C8 dir-match (power-floor aware) | dir_match AND power≥{OOS_POWER_FLOOR} | dir={s['dir_match']}, power={s['oos_power']:.3f} | {s['c8_verdict']} |")
    lines.append(f"| C9 era stability (N_era≥{ERA_MIN_N}, no era < {KILL_THRESHOLDS['era_min']}) | PASS | {s['c9_verdict']} | {s['c9_verdict']} |")
    lines.append("")
    lines.append("## Per-year IS breakdown")
    lines.append("")
    lines.append("| Year | N | ExpR |")
    lines.append("|---:|---:|---:|")
    for yr, n, mu in s["era_breakdown"]:
        lines.append(f"| {yr} | {n} | {mu:+.4f} |")
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
    lines.append(f"DUCKDB_PATH={GOLD_DB_PATH} python research/phase_d_d1_mes_europe_flow_pathway_b.py")
    lines.append("```")
    lines.append("")
    lines.append("- DB: `pipeline.paths.GOLD_DB_PATH`")
    lines.append("- Holdout: `trading_app.holdout_policy.HOLDOUT_SACRED_FROM` (2026-01-01)")
    lines.append("- Pre-reg: locks all schema parameters and kill criteria")
    lines.append("- Helpers: imported from `research/phase_b_candidate_evidence_v1.py` (canonical)")
    lines.append("")
    lines.append("## Caveats / limitations")
    lines.append("")
    lines.append("- N_OOS_on is small (3-month sacred holdout). C8 power floor invokes UNVERIFIED, not KILL.")
    lines.append("- This is one Pathway B test, K=1 by design. The decision rule's PARK case is the legitimate")
    lines.append("  verdict when OOS is underpowered; UNVERIFIED is not failure per Amendment 3.2.")
    lines.append("- DSR Pathway A K_family is approximated at 1850 (overnight family proxy from Phase B). Sensitivity")
    lines.append("  not exhaustive — true K may differ; the Pathway B K=1 path is what governs promotion here.")
    lines.append("")
    lines.append("## Not done by this run")
    lines.append("")
    lines.append("- No write to validated_setups, edge_families, lane_allocation, live_config")
    lines.append("- No paper trade simulation")
    lines.append("- No CPCV — deferred to future amendment if N_OOS power floor remains active in Q3-2026")
    lines.append("- No capital deployment — that requires Phase E + capital-review skill + explicit user GO")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_pressure_test(con) -> str:
    """RULE 13: inject a known-banned predictor and assert the runner refuses to use it.

    We don't have a free-form feature plug-in here — locking is by design. So
    we assert that LOCKED["feature"] is in the safe list and the threshold is
    a documented non-percentile cut. This is the runner's pressure test:
    structural assertion that the test cannot silently swap in a banned feature.
    """
    safe_features = {
        "overnight_range_pct",
        "overnight_range",
        "atr_20",
        "atr_20_pct",
        "garch_forecast_vol_pct",
        "rel_vol_session_norm",  # session-relative volume (not break-bar)
    }
    banned_predictors = {"pnl_r", "outcome", "mae_r", "mfe_r", "double_break"}
    if LOCKED["feature"] in banned_predictors:
        return f"FAIL — locked feature {LOCKED['feature']} is in banned-predictor set"
    if LOCKED["feature"] not in safe_features:
        return f"FAIL — locked feature {LOCKED['feature']} not in declared safe-feature set"
    return "PASS"


def get_prereg_commit() -> str:
    """Read git for the most recent commit touching the pre-reg yaml."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(PRE_REG_PATH.relative_to(PROJECT_ROOT))],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(PROJECT_ROOT),
        )
        return out.stdout.strip() or "PRE_REG_NOT_YET_COMMITTED"
    except (subprocess.TimeoutExpired, OSError):
        return "PRE_REG_COMMIT_UNAVAILABLE"


def get_db_max_day(con) -> str:
    row = con.execute("SELECT MAX(trading_day) FROM orb_outcomes WHERE symbol='MES'").fetchone()
    return str(row[0]) if row and row[0] else "UNKNOWN"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    print("=== Phase D D1 — MES EUROPE_FLOW Pathway B K=1 ===")
    print(f"DB: {GOLD_DB_PATH}")
    if not GOLD_DB_PATH.exists():
        print("ERROR: gold.db not found — set DUCKDB_PATH or run from repo root")
        sys.exit(1)

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    # Pressure test (RULE 13)
    rule_13_status = run_pressure_test(con)
    print(f"Pressure test (RULE 13): {rule_13_status}")
    if not rule_13_status.startswith("PASS"):
        print("Aborting — pressure test failed")
        sys.exit(2)

    df = load_cell(con)
    is_df = df[df["is_is"]]
    oos_df = df[~df["is_is"]]
    is_on = is_df[is_df["sig"]]
    is_off = is_df[~is_df["sig"]]
    oos_on = oos_df[oos_df["sig"]]
    oos_off = oos_df[~oos_df["sig"]]

    n_is = len(is_df)
    n_is_on = len(is_on)
    n_is_off = len(is_off)
    n_oos = len(oos_df)
    n_oos_on = len(oos_on)
    n_oos_off = len(oos_off)

    pnl_is_on = is_on["pnl_r"].astype(float).to_numpy()
    pnl_is_off = is_off["pnl_r"].astype(float).to_numpy()
    pnl_oos_on = oos_on["pnl_r"].astype(float).to_numpy()
    pnl_oos_off = oos_off["pnl_r"].astype(float).to_numpy()

    expr_is_on = float(np.mean(pnl_is_on)) if n_is_on else float("nan")
    expr_is_off = float(np.mean(pnl_is_off)) if n_is_off else float("nan")
    expr_oos_on = float(np.mean(pnl_oos_on)) if n_oos_on else float("nan")
    expr_oos_off = float(np.mean(pnl_oos_off)) if n_oos_off else float("nan")
    delta_is = expr_is_on - expr_is_off
    delta_oos = expr_oos_on - expr_oos_off

    # IS effect significance
    if n_is_on >= 2 and n_is_off >= 2:
        welch = stats.ttest_ind(pnl_is_on, pnl_is_off, equal_var=False)
        welch_t = float(welch.statistic)
        welch_p = float(welch.pvalue)
    else:
        welch_t = float("nan")
        welch_p = float("nan")

    print("Computing block-bootstrap p (B=10000)...")
    boot_p = block_bootstrap_p(pnl_is_on, pnl_is_off, block=5, B=10000)

    # Sharpe (per-trade, annualised)
    sr_pt_is = per_trade_sharpe(pnl_is_on)
    is_years = (
        (is_on["trading_day"].max() - is_on["trading_day"].min()).days / 365.25 if n_is_on >= 2 else 1.0
    )
    sr_ann_is = annualised_sharpe(pnl_is_on, max(is_years, 0.5))

    sr_pt_oos = per_trade_sharpe(pnl_oos_on) if n_oos_on >= 2 else float("nan")

    # Skewness / kurtosis (excess)
    skew_is = float(stats.skew(pnl_is_on)) if n_is_on >= 3 else float("nan")
    kurt_is = float(stats.kurtosis(pnl_is_on, fisher=True)) if n_is_on >= 4 else float("nan")

    # DSR Pathway A (provenance) and Pathway B (primary gate)
    sr_var_per_trade_pa = 0.5 / max(is_years, 0.5)  # convert annualised V=0.5 to per-trade
    sr0_pa = sr0_threshold(sr_var_per_trade_pa, K_FAMILY_OVERNIGHT_PROVENANCE)
    sr_var_per_trade_pb = 0.5 / max(is_years, 0.5)
    sr0_pb = sr0_threshold(sr_var_per_trade_pb, 1.0)  # K=1 → degenerate; force minimum-floor SR0=0
    if not np.isfinite(sr0_pb) or sr0_pb < 0:
        sr0_pb = 0.0
    dsr_pa = dsr_eq2(sr_pt_is, sr0_pa, n_is_on, skew_is, kurt_is + 3 if np.isfinite(kurt_is) else 3)
    dsr_pb = dsr_eq2(sr_pt_is, sr0_pb, n_is_on, skew_is, kurt_is + 3 if np.isfinite(kurt_is) else 3)

    # C8 power
    cd_is = cohens_d(pnl_is_on, pnl_is_off)
    oos_power = power_for_d(cd_is, n_oos_on, n_oos_off) if n_oos_on >= 2 and n_oos_off >= 2 else float("nan")
    if np.isnan(oos_power):
        oos_power = 0.0  # treat unknown as below floor

    # C8 dir-match (only if powered)
    if n_oos_on >= MIN_N_OOS_FOR_C6 and oos_power >= OOS_POWER_FLOOR:
        dir_match = (np.sign(delta_is) == np.sign(delta_oos)) if not np.isnan(delta_oos) else False
        c8_verdict = "C8_PASS" if dir_match else "C8_FAIL"
    else:
        dir_match = (np.sign(delta_is) == np.sign(delta_oos)) if not np.isnan(delta_oos) else False
        c8_verdict = "C8_GATE_INACTIVE_LOWPOWER"

    # C6 WFE
    if n_oos_on >= MIN_N_OOS_FOR_C6:
        sr_ann_oos = annualised_sharpe(
            pnl_oos_on,
            max((oos_on["trading_day"].max() - oos_on["trading_day"].min()).days / 365.25, 0.1),
        )
        wfe = sr_ann_oos / sr_ann_is if abs(sr_ann_is) > 1e-9 else float("nan")
        wfe_str = f"{wfe:+.4f}"
        c6_verdict = "C6_PASS" if (np.isfinite(wfe) and wfe >= WFE_THRESHOLD) else "C6_FAIL"
    else:
        wfe_str = "GATE_INACTIVE_LOWPOWER"
        c6_verdict = "C6_GATE_INACTIVE_LOWPOWER"

    # C7
    c7_verdict = "C7_PASS" if n_is_on >= KILL_THRESHOLDS["n_is_min"] else "C7_FAIL"

    # C9
    c9_verdict, era_breakdown = era_stability(is_on)

    # Kill-criterion table
    kill_results = []
    kill_results.append(
        {
            "id": "KILL_RAWP",
            "threshold": f">= {KILL_THRESHOLDS['raw_p_max']}",
            "computed": f"welch_p = {welch_p:.6f}",
            "verdict": "PASS" if welch_p < KILL_THRESHOLDS["raw_p_max"] else "FAIL",
        }
    )
    kill_results.append(
        {
            "id": "KILL_DIR",
            "threshold": "negative on IS",
            "computed": f"sign(delta_IS) = {'+' if delta_is > 0 else '-'}",
            "verdict": "PASS" if delta_is > 0 else "FAIL",
        }
    )
    kill_results.append(
        {
            "id": "KILL_T",
            "threshold": f"< {KILL_THRESHOLDS['abs_t_min']}",
            "computed": f"|t| = {abs(welch_t):.4f}",
            "verdict": "PASS" if abs(welch_t) >= KILL_THRESHOLDS["abs_t_min"] else "FAIL",
        }
    )
    kill_results.append(
        {
            "id": "KILL_N",
            "threshold": f"< {KILL_THRESHOLDS['n_is_min']}",
            "computed": f"N_IS_on = {n_is_on}",
            "verdict": "PASS" if n_is_on >= KILL_THRESHOLDS["n_is_min"] else "FAIL",
        }
    )
    era_min_observed = min((b[2] for b in era_breakdown if b[1] >= ERA_MIN_N), default=float("inf"))
    kill_results.append(
        {
            "id": "KILL_ERA",
            "threshold": f"< {KILL_THRESHOLDS['era_min']} (any era N>={ERA_MIN_N})",
            "computed": f"min_era_ExpR(eligible) = {era_min_observed:+.4f}",
            "verdict": "PASS" if era_min_observed >= KILL_THRESHOLDS["era_min"] else "FAIL",
        }
    )
    kill_results.append(
        {
            "id": "KILL_SHARPE",
            "threshold": f"<= {KILL_THRESHOLDS['sharpe_ann_min']}",
            "computed": f"sharpe_ann_IS = {sr_ann_is:+.4f}",
            "verdict": "PASS" if sr_ann_is > KILL_THRESHOLDS["sharpe_ann_min"] else "FAIL",
        }
    )
    sanity_diff = abs(delta_is - KILL_THRESHOLDS["expected_delta_is"])
    kill_results.append(
        {
            "id": "KILL_BASELINE_SANITY",
            "threshold": f"> {KILL_THRESHOLDS['baseline_sanity_max_diff']}",
            "computed": f"|delta_IS - expected| = {sanity_diff:.6f}",
            "verdict": "PASS" if sanity_diff <= KILL_THRESHOLDS["baseline_sanity_max_diff"] else "FAIL",
        }
    )

    any_kill = any(k["verdict"] == "FAIL" for k in kill_results)

    # Final decision
    dsr_pa_verdict = "DSR_PA_FAIL" if (np.isnan(dsr_pa) or dsr_pa < DSR_THRESHOLD) else "DSR_PA_PASS"
    dsr_pb_verdict = "DSR_PB_FAIL" if (np.isnan(dsr_pb) or dsr_pb < DSR_THRESHOLD) else "DSR_PB_PASS"

    if any_kill:
        final_verdict = "KILL"
        verdict_explanation = (
            "At least one locked KILL criterion fired. The hypothesis is killed under "
            "the pre-registered decision rule. No post-hoc rescue per Amendment 3.0 + "
            "research-truth-protocol.md. Strategy is logged as Pathway B FAIL with the "
            "pre-reg SHA stamp; no promotion to validated_setups."
        )
    elif (
        dsr_pb_verdict == "DSR_PB_PASS"
        and c7_verdict == "C7_PASS"
        and c9_verdict == "C9_PASS"
        and sr_ann_is > 0
    ):
        if c6_verdict == "C6_PASS" and c8_verdict == "C8_PASS":
            final_verdict = "CANDIDATE_READY"
            verdict_explanation = (
                "All non-conditional gates PASS, OOS C6 and C8 are powered and "
                "PASS. Cell is CANDIDATE_READY pending Phase E capital-review."
            )
        elif c6_verdict == "C6_GATE_INACTIVE_LOWPOWER" or c8_verdict == "C8_GATE_INACTIVE_LOWPOWER":
            final_verdict = "PARK_PENDING_OOS_POWER"
            verdict_explanation = (
                f"All non-conditional gates PASS (C5 DSR_PB={dsr_pb:.4f}, C7 N={n_is_on}, "
                f"C9 era stable, Sharpe_ann_IS={sr_ann_is:+.4f}). C6/C8 are "
                f"GATE_INACTIVE_LOWPOWER because N_OOS_on={n_oos_on} < {MIN_N_OOS_FOR_C6} "
                f"power floor (Amendment 3.2). UNVERIFIED ≠ KILL — cell parks until "
                f"N_OOS_on accrues to ≥{MIN_N_OOS_FOR_C6} (estimate Q3-2026 at current "
                f"trade rate). Pre-reg locked; no re-tuning permitted on accrual."
            )
        else:
            final_verdict = "KILL"
            verdict_explanation = (
                f"C6 = {c6_verdict}, C8 = {c8_verdict}. Both must be PASS or "
                f"GATE_INACTIVE_LOWPOWER for promotion or park. KILL otherwise."
            )
    else:
        final_verdict = "KILL"
        verdict_explanation = (
            f"DSR_PB={dsr_pb:.4f} ({dsr_pb_verdict}), C7={c7_verdict}, C9={c9_verdict}, "
            f"Sharpe_ann_IS={sr_ann_is:+.4f}. One or more non-conditional gates failed. "
            "KILL per pre-reg decision_rule."
        )

    state = {
        "prereg_commit": get_prereg_commit(),
        "db_max_day": get_db_max_day(con),
        "n_is": n_is,
        "n_is_on": n_is_on,
        "n_is_off": n_is_off,
        "n_oos": n_oos,
        "n_oos_on": n_oos_on,
        "n_oos_off": n_oos_off,
        "expr_is_on": expr_is_on,
        "expr_is_off": expr_is_off,
        "expr_oos_on": expr_oos_on,
        "expr_oos_off": expr_oos_off,
        "delta_is": delta_is,
        "delta_oos": delta_oos,
        "sr_pt_is": sr_pt_is,
        "sr_pt_oos": sr_pt_oos,
        "sr_ann_is": sr_ann_is,
        "skew_is": skew_is,
        "kurt_is": kurt_is,
        "welch_t": welch_t,
        "welch_p": welch_p,
        "boot_p": boot_p,
        "cohens_d": cd_is,
        "oos_power": oos_power,
        "dsr_pa": dsr_pa,
        "dsr_pa_verdict": dsr_pa_verdict,
        "dsr_pb": dsr_pb,
        "dsr_pb_verdict": dsr_pb_verdict,
        "wfe_str": wfe_str,
        "c6_verdict": c6_verdict,
        "c7_verdict": c7_verdict,
        "c8_verdict": c8_verdict,
        "c9_verdict": c9_verdict,
        "dir_match": dir_match,
        "era_breakdown": era_breakdown,
        "kill_results": kill_results,
        "final_verdict": final_verdict,
        "verdict_explanation": verdict_explanation,
        "rule_13_status": rule_13_status,
    }

    print(f"VERDICT: {final_verdict}")
    print(verdict_explanation)
    emit_result(state)
    print(f"Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
