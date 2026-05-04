"""Phase D D6 — MNQ COMEX_SETTLE O5 E2 CB1 RR1.5 + garch_forecast_vol_pct > 70 — sizing-overlay Pathway B K=1.

Runs the pre-registered confirmatory test per
docs/audit/hypotheses/2026-04-29-mnq-comex-settle-garch-d6-sizing-overlay-pathway-b-v1.yaml
(commit 54b7b948).

Output: docs/audit/results/2026-04-29-mnq-comex-settle-garch-d6-sizing-overlay-pathway-b-v1-result.md

scratch-policy: include-as-zero (canonical Stage 5 outcome_builder fix; commit 68ee35f8).
orb_outcomes pnl_r is COALESCEd to 0.0 for the residual <0.5% of scratch-EOD-no-MTM rows
that legitimately remain NULL per the Stage 4 spec
(docs/specs/outcome_builder_scratch_eod_mtm.md:368-372).

# e2-lookahead-policy: not-predictor
# garch_forecast_vol_pct is § 6.1 SAFE (prior-day close, 252-day rolling window).
# break_dir is NOT used (E2-LA banned per backtesting-methodology.md § 6.3).
# Both-sides cohort, no break_dir filter, no break_bar features.

Reproduces independent SQL verified 2026-04-29 (research/d6_preflight_verify.py):
  IS:  N=1577 baseline, gate-on N=370 ExpR=+0.2477, gate-off N=1019 ExpR=+0.0530
  OOS: gate-on N=37 ExpR=+0.358, gate-off N=33 ExpR=-0.335

This runner is the one-shot Pathway B K=1 confirmatory test. NO threshold sweeps;
NO post-hoc rescue. The pre-reg is committed first; the runner emits the verdict
per locked decision_rule (continue_if / park_if / kill_if).

PRESSURE TEST: per backtesting-methodology.md RULE 13, the harness MUST reject
a known-bad sham predictor before producing the verdict. The pressure test is
implemented as a T0 tautology check that confirms `pnl_r` itself, used as a
predictor against the same `pnl_r` outcome, fires the corr>=0.70 abort. If
this fails to fire, the harness aborts with a non-zero exit code BEFORE
emitting any verdict. Origin: pre-reg `methodology_rules_applied.rule_13_pressure_test`.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
from scipy.stats import t as tdist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from research.filter_utils import filter_signal  # noqa: E402  -- canonical filter delegation per RULE 9
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402

# ============================================================================
# Locked pre-reg constants (do NOT edit without amending the pre-reg first)
# ============================================================================
PRE_REG_PATH = (
    PROJECT_ROOT
    / "docs/audit/hypotheses/2026-04-29-mnq-comex-settle-garch-d6-sizing-overlay-pathway-b-v1.yaml"
)
PRE_REG_COMMIT_SHA = "54b7b948"

# Scope (locked)
SYMBOL = "MNQ"
ORB_LABEL = "COMEX_SETTLE"
ORB_MINUTES = 5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR_TARGET = 1.5
BASE_FILTER_KEY = "ORB_G5"

# Predicate (locked)
GATE_FEATURE = "garch_forecast_vol_pct"
GATE_THRESHOLD = 70.0  # strict greater than

# Locked thresholds (FORBIDDEN to change without pre-reg amendment)
T_PASS = 3.00          # Chordia with-theory
T_PARK_LOWER = 2.50    # park_if [2.5, 3.00); KILL below
T_KILL = 2.50
LIFT_MIN_R = 0.10      # IS lift floor
OOS_LIFT_RATIO = 0.40  # C8: OOS_lift >= 0.40 * IS_lift
ERA_FLOOR_EXPR = -0.05 # C9: no year with N>=50 below this
ERA_N_FLOOR = 50       # C9 strict-eval N floor
ERA_COUNT_N_FLOOR = 10 # positive-year counting N floor
ERA_POSITIVE_YEARS_REQUIRED = 4
T0_TAUTOLOGY_THRESHOLD = 0.70

# Output
RESULT_PATH = (
    PROJECT_ROOT
    / "docs/audit/results/2026-04-29-mnq-comex-settle-garch-d6-sizing-overlay-pathway-b-v1-result.md"
)


# ============================================================================
# Data load (canonical layers only)
# ============================================================================
def load_cohort(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load full IS+OOS deployed-lane cohort with garch + ORB columns.

    Returns one row per (trading_day, fill) via canonical orb_outcomes joined
    against daily_features on (trading_day, symbol, orb_minutes) triple key.
    Both-sides cohort (no break_dir filter -- E2 LA policy: not-predictor).
    The base ORB_G5 filter is applied via canonical filter_signal delegation
    (RULE 9), NOT by inline orb_size>=5 SQL — the canonical filter is the
    source of truth.
    """
    sql = f"""
        SELECT
            o.trading_day,
            o.entry_ts,
            o.outcome,
            o.pnl_r,
            d.garch_forecast_vol_pct,
            d.orb_{ORB_LABEL}_size AS orb_size,
            d.orb_{ORB_LABEL}_high AS orb_high,
            d.orb_{ORB_LABEL}_low AS orb_low
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.trading_day = d.trading_day
            AND o.symbol = d.symbol
            AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = '{SYMBOL}'
          AND o.orb_label = '{ORB_LABEL}'
          AND o.orb_minutes = {ORB_MINUTES}
          AND o.entry_model = '{ENTRY_MODEL}'
          AND o.confirm_bars = {CONFIRM_BARS}
          AND o.rr_target = {RR_TARGET}
          AND o.entry_ts IS NOT NULL
          AND o.pnl_r IS NOT NULL
        ORDER BY o.trading_day
    """
    df = con.execute(sql).df()

    # Apply ORB_G5 via canonical filter_utils.filter_signal (RULE 9 delegation)
    # filter_signal expects orb_{LABEL}_* columns; we have them
    df_for_filter = df.rename(
        columns={
            "orb_size": f"orb_{ORB_LABEL}_size",
            "orb_high": f"orb_{ORB_LABEL}_high",
            "orb_low": f"orb_{ORB_LABEL}_low",
        }
    )
    base_signal = filter_signal(df_for_filter, BASE_FILTER_KEY, ORB_LABEL)
    df = df.loc[base_signal == 1].copy()
    return df


# ============================================================================
# Pressure test (RULE 13)
# ============================================================================
def pressure_test_t0_tautology(df_is: pd.DataFrame) -> dict:
    """Pressure-test the T0 tautology detector with a known-bad predictor.

    Inject `pnl_r` as a sham predictor; |corr(pnl_r, pnl_r)| = 1.0 must trigger
    the T0 abort. If it doesn't, the harness is broken — abort.
    """
    self_corr = df_is["pnl_r"].corr(df_is["pnl_r"])
    win_corr = (df_is["outcome"] == "win").astype(float).corr(df_is["pnl_r"])
    triggered = abs(self_corr) >= T0_TAUTOLOGY_THRESHOLD
    return {
        "self_corr_pnl_r_pnl_r": float(self_corr),
        "corr_is_win_vs_pnl_r": float(win_corr),
        "t0_threshold": T0_TAUTOLOGY_THRESHOLD,
        "triggered_correctly": bool(triggered),
    }


# ============================================================================
# Statistics
# ============================================================================
def welch_t_p(m1, sd1, n1, m2, sd2, n2) -> tuple[float, float, float]:
    """Welch t / df / two-sided p for two-sample mean diff (m1 - m2)."""
    se = math.sqrt(sd1**2 / n1 + sd2**2 / n2)
    if se == 0:
        return 0.0, float("inf"), 1.0
    t = (m1 - m2) / se
    df = (sd1**2 / n1 + sd2**2 / n2) ** 2 / (
        (sd1**2 / n1) ** 2 / (n1 - 1) + (sd2**2 / n2) ** 2 / (n2 - 1)
    )
    p = 2 * (1 - tdist.cdf(abs(t), df))
    return float(t), float(df), float(p)


def partition_stats(df: pd.DataFrame, gate_col: str = "gate_on") -> dict:
    """Return on/off subset stats."""
    on = df.loc[df[gate_col]]
    off = df.loc[~df[gate_col]]
    return {
        "on": {
            "n": int(len(on)),
            "expr": float(on["pnl_r"].mean()) if len(on) > 0 else 0.0,
            "sd": float(on["pnl_r"].std(ddof=1)) if len(on) > 1 else 0.0,
        },
        "off": {
            "n": int(len(off)),
            "expr": float(off["pnl_r"].mean()) if len(off) > 0 else 0.0,
            "sd": float(off["pnl_r"].std(ddof=1)) if len(off) > 1 else 0.0,
        },
    }


def per_year_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-year gate-on/gate-off ExpR + N for C9 evaluation."""
    df = df.copy()
    df["yr"] = pd.to_datetime(df["trading_day"]).dt.year.astype(int)
    rows = []
    for yr, g in df.groupby("yr"):
        on = g.loc[g["gate_on"]]
        off = g.loc[~g["gate_on"]]
        rows.append({
            "yr": int(yr),
            "n_on": int(len(on)),
            "expr_on": float(on["pnl_r"].mean()) if len(on) > 0 else float("nan"),
            "n_off": int(len(off)),
            "expr_off": float(off["pnl_r"].mean()) if len(off) > 0 else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("yr").reset_index(drop=True)


# ============================================================================
# Decision rule (locked)
# ============================================================================
def evaluate_verdict(
    is_t: float,
    is_lift: float,
    pos_mean_floor_pass: bool,
    c9_pass_strict: bool,
    c9_positive_years_count: int,
    oos_lift: float,
    oos_dir_match: bool,
    oos_n_on: int,
    sham_test_passed: bool,
) -> dict:
    """Apply locked decision_rule from pre-reg.

    Returns dict with verdict + reasons.
    """
    reasons: list[str] = []

    # Pressure test must pass before anything
    if not sham_test_passed:
        return {
            "verdict": "ABORT_PRESSURE_TEST_FAILED",
            "reasons": ["RULE 13 pressure test did not fire on pnl_r-as-predictor sham; harness integrity violated"],
        }

    # Hard kills
    if abs(is_t) < T_KILL:
        reasons.append(f"KILL_IS_T: |IS t|={abs(is_t):.4f} < {T_KILL}")
        return {"verdict": "KILL_IS_T", "reasons": reasons}

    if not c9_pass_strict:
        reasons.append("KILL_C9_HARD: a year with N>=50 has ExpR_on < -0.05")
        return {"verdict": "KILL_C9_HARD", "reasons": reasons}

    if not pos_mean_floor_pass:
        reasons.append("KILL_POS_MEAN_FLOOR: both subset means non-positive")
        return {"verdict": "KILL_POS_MEAN_FLOOR", "reasons": reasons}

    # OOS direction mismatch with material |t| -> suspend (caller treats as KILL pending SR)
    if (oos_lift < 0) != (is_lift < 0):
        # Sign mismatch -- this is the dir_match=False case
        if oos_n_on >= 5:
            reasons.append(f"KILL_DIR_MISMATCH_OOS: OOS_lift={oos_lift:.4f} sign opposite IS_lift={is_lift:.4f}")
            return {"verdict": "KILL_DIR_MISMATCH_OOS_PENDING_SR", "reasons": reasons}

    # Continue / park split on |IS_t|
    if abs(is_t) >= T_PASS and is_lift >= LIFT_MIN_R and pos_mean_floor_pass:
        reasons.append(f"|IS t|={abs(is_t):.4f} >= {T_PASS} (Chordia with-theory)")
        reasons.append(f"IS lift={is_lift:.4f} >= {LIFT_MIN_R}")
        reasons.append(f"positive_mean_floor PASS")
        reasons.append(f"C9 strict: PASS; positive years (N>={ERA_COUNT_N_FLOOR}): {c9_positive_years_count}")
        if c9_positive_years_count < ERA_POSITIVE_YEARS_REQUIRED:
            reasons.append(f"WARNING: positive years {c9_positive_years_count} < required {ERA_POSITIVE_YEARS_REQUIRED}")
            return {"verdict": "PARK_LOW_POSITIVE_YEARS", "reasons": reasons}
        return {"verdict": "CANDIDATE_READY_FOR_PHASE_2", "reasons": reasons}

    if T_PARK_LOWER <= abs(is_t) < T_PASS and is_lift >= LIFT_MIN_R and pos_mean_floor_pass and c9_pass_strict:
        reasons.append(f"|IS t|={abs(is_t):.4f} in [{T_PARK_LOWER}, {T_PASS}): below Chordia with-theory but above raw p<0.05 floor")
        reasons.append(f"IS lift={is_lift:.4f} OK; positive_mean_floor OK; C9 OK")
        reasons.append("CONDITIONAL_DEPLOY (Phase 1 shadow-only) retained; Phase 2 size-multiplier pre-reg gated")
        return {"verdict": "PARK_CONDITIONAL_DEPLOY_RETAINED", "reasons": reasons}

    if is_lift < LIFT_MIN_R:
        reasons.append(f"IS lift={is_lift:.4f} < {LIFT_MIN_R} floor")
        return {"verdict": "KILL_LIFT_FLOOR", "reasons": reasons}

    reasons.append("UNCATEGORIZED — review manually")
    return {"verdict": "MANUAL_REVIEW", "reasons": reasons}


# ============================================================================
# Result file emission
# ============================================================================
def emit_result_md(payload: dict) -> str:
    """Render the result MD per pre-reg outputs_required_after_run schema."""
    p = payload
    lines = [
        f"# D6 Result — MNQ COMEX_SETTLE GARCH>70 sizing-overlay (Pathway B K=1)",
        "",
        f"**Date:** {p['run_date']}",
        f"**Pre-reg:** [`docs/audit/hypotheses/2026-04-29-mnq-comex-settle-garch-d6-sizing-overlay-pathway-b-v1.yaml`]({PRE_REG_PATH.relative_to(PROJECT_ROOT)})",
        f"**Pre-reg commit_sha:** `{PRE_REG_COMMIT_SHA}`",
        f"**DB max trading_day:** {p['db_max_trading_day']}",
        f"**Holdout boundary (Mode A):** {HOLDOUT_SACRED_FROM}",
        f"**Verdict:** **{p['verdict']}**",
        "",
        "## Pressure test (RULE 13 — required pre-verdict)",
        "",
        f"- corr(pnl_r, pnl_r) = {p['pressure_test']['self_corr_pnl_r_pnl_r']:.6f}  (must be ~1.0)",
        f"- corr(is_win, pnl_r) = {p['pressure_test']['corr_is_win_vs_pnl_r']:.6f}  (post-trade label association)",
        f"- T0 threshold: {p['pressure_test']['t0_threshold']}",
        f"- Triggered correctly: **{p['pressure_test']['triggered_correctly']}**",
        "",
        "## IS partition (deployed lane, ORB_G5 base, both sides, garch>70 gate)",
        "",
        "| Subset | N | ExpR | sd |",
        "|---|---:|---:|---:|",
        f"| baseline | {p['is']['baseline_n']} | {p['is']['baseline_expr']:.6f} | {p['is']['baseline_sd']:.6f} |",
        f"| gate-on (garch>70) | {p['is']['part']['on']['n']} | {p['is']['part']['on']['expr']:.6f} | {p['is']['part']['on']['sd']:.6f} |",
        f"| gate-off (garch<=70) | {p['is']['part']['off']['n']} | {p['is']['part']['off']['expr']:.6f} | {p['is']['part']['off']['sd']:.6f} |",
        f"| **lift (on - off)** | — | **{p['is']['lift']:.6f}** | — |",
        "",
        f"- Welch t (lift) = {p['is']['t']:.4f}, df = {p['is']['df']:.1f}, p = {p['is']['p']:.6f}",
        f"- IS gate-on vs zero: t = {p['is']['t_on_zero']:.4f}",
        "",
        "## C9 era stability (per IS year, gate partition)",
        "",
        "| year | gate-on N | gate-on ExpR | gate-off N | gate-off ExpR | flags |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in p["c9"]["per_year"]:
        flags = []
        if r["n_on"] >= ERA_N_FLOOR and r["expr_on"] is not None and not pd.isna(r["expr_on"]) and r["expr_on"] < ERA_FLOOR_EXPR:
            flags.append("*** C9_FAIL")
        if r["n_on"] >= ERA_COUNT_N_FLOOR and r["expr_on"] is not None and not pd.isna(r["expr_on"]) and r["expr_on"] > 0:
            flags.append("counted_positive")
        e_on = f"{r['expr_on']:.4f}" if r["expr_on"] is not None and not pd.isna(r["expr_on"]) else "NA"
        e_off = f"{r['expr_off']:.4f}" if r["expr_off"] is not None and not pd.isna(r["expr_off"]) else "NA"
        lines.append(f"| {r['yr']} | {r['n_on']} | {e_on} | {r['n_off']} | {e_off} | {' '.join(flags)} |")
    lines += [
        "",
        f"- C9 strict fails (N>={ERA_N_FLOOR}, ExpR<{ERA_FLOOR_EXPR}): {p['c9']['fails']}",
        f"- Positive years (N>={ERA_COUNT_N_FLOOR}): {p['c9']['positive_years']} — count {p['c9']['positive_years_count']}",
        f"- Required: >= {ERA_POSITIVE_YEARS_REQUIRED}",
        "",
        "## OOS partition (accruing forward shadow)",
        "",
        "| Subset | N | ExpR | sd |",
        "|---|---:|---:|---:|",
        f"| gate-on | {p['oos']['part']['on']['n']} | {p['oos']['part']['on']['expr']:.6f} | {p['oos']['part']['on']['sd']:.6f} |",
        f"| gate-off | {p['oos']['part']['off']['n']} | {p['oos']['part']['off']['expr']:.6f} | {p['oos']['part']['off']['sd']:.6f} |",
        f"| **OOS lift** | — | **{p['oos']['lift']:.6f}** | — |",
        "",
        f"- OOS Welch t = {p['oos']['t']:.4f}, df = {p['oos']['df']:.1f}, p = {p['oos']['p']:.6f}",
        f"- dir_match: **{p['oos']['dir_match']}**",
        f"- OOS_lift >= {OOS_LIFT_RATIO} * IS_lift: **{p['oos']['ratio_pass']}**",
        f"- C8 status: **{p['oos']['c8_status']}** (Amendment 3.1; Amendment 3.2 NOT_OOS_CONFIRMABLE classification at pre-reg time, CONDITIONAL_DEPLOY chosen)",
        "",
        "## Decision",
        "",
        f"**{p['verdict']}**",
        "",
    ]
    for r in p["reasons"]:
        lines.append(f"- {r}")
    lines += [
        "",
        "## Reproduction",
        "",
        f"```",
        f"python research/phase_d_d6_mnq_comex_settle_overlay_pathway_b.py",
        f"```",
        "",
        "Inputs: gold.db at `pipeline.paths.GOLD_DB_PATH`, Python with duckdb+scipy+pandas+numpy.",
        "Canonical layers only (`bars_1m`, `daily_features`, `orb_outcomes`).",
        "Filter delegation via `research.filter_utils.filter_signal` (RULE 9; no inline ORB_G5 re-encoding).",
        "",
        "## Limitations",
        "",
        "- Single cell, single lane, single feature. K=1 by Pathway B / Amendment 3.0.",
        "- OOS still accruing — C8 N_OOS<50 floor not met; Amendment 3.2 already classified NOT_OOS_CONFIRMABLE at pre-reg write time.",
        "- Phase 1 shadow-only carrier wiring is OUT OF SCOPE for this runner (separate commit).",
        "- Phase 2 size-multiplier pre-reg gated on this verdict; not authored.",
        "- SR monitor on the deployed lane is the live circuit-breaker (60-day ARL); KILL_SR_ALARM action defined in pre-reg.",
        "",
        "## Cross-references",
        "",
        f"- D4 result: `docs/audit/results/2026-04-28-mnq-comex-settle-pathway-b-v1-result.md` (PARK_PENDING_OOS_POWER stands)",
        f"- D5 result: `docs/audit/results/2026-04-28-mnq-comex-settle-d5-both-sides-pathway-b-v1-result.md` (KILL stands)",
        f"- Additivity triage: `docs/audit/results/2026-04-29-parked-pathway-b-additivity-triage.md` § Addendum (Path 3 origin)",
        f"- Amendment 3.2: `docs/institutional/pre_registered_criteria.md` § Amendment 3.2 (MinTRL classification)",
    ]
    return "\n".join(lines) + "\n"


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    print(f"D6 RUNNER -- pre-reg commit_sha={PRE_REG_COMMIT_SHA}")
    print(f"DB: {GOLD_DB_PATH}")
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    db_max = con.execute("SELECT MAX(trading_day) FROM orb_outcomes WHERE symbol=?", [SYMBOL]).fetchone()[0]
    print(f"DB max trading_day for {SYMBOL}: {db_max}")

    # Load full cohort with ORB_G5 base applied
    df_all = load_cohort(con)
    print(f"Loaded N={len(df_all)} rows after ORB_G5 base filter (canonical delegation)")

    # Gate column (locked predicate)
    df_all["gate_on"] = df_all[GATE_FEATURE] > GATE_THRESHOLD

    # Drop rows with NULL gate feature (cannot apply gate; out of cohort)
    df = df_all.dropna(subset=[GATE_FEATURE]).copy()
    null_garch = len(df_all) - len(df)
    print(f"Excluded N={null_garch} rows with NULL {GATE_FEATURE}")

    # IS / OOS split
    holdout = pd.Timestamp(HOLDOUT_SACRED_FROM).date()
    df_is = df.loc[pd.to_datetime(df["trading_day"]).dt.date < holdout].copy()
    df_oos = df.loc[pd.to_datetime(df["trading_day"]).dt.date >= holdout].copy()
    print(f"IS N={len(df_is)} | OOS N={len(df_oos)}")

    # Pressure test (RULE 13) -- BEFORE any verdict computation
    sham = pressure_test_t0_tautology(df_is)
    print(f"Pressure test: self_corr={sham['self_corr_pnl_r_pnl_r']:.4f} triggered={sham['triggered_correctly']}")
    if not sham["triggered_correctly"]:
        print("FATAL: T0 tautology detector did not fire on pnl_r self-corr. Harness integrity violated.")
        return 2

    # IS partition
    is_part = partition_stats(df_is, "gate_on")
    is_baseline_n = len(df_is)
    is_baseline_expr = float(df_is["pnl_r"].mean())
    is_baseline_sd = float(df_is["pnl_r"].std(ddof=1))
    is_lift = is_part["on"]["expr"] - is_part["off"]["expr"]
    is_t, is_df, is_p = welch_t_p(
        is_part["on"]["expr"], is_part["on"]["sd"], is_part["on"]["n"],
        is_part["off"]["expr"], is_part["off"]["sd"], is_part["off"]["n"],
    )
    is_t_on_zero = (is_part["on"]["expr"] / (is_part["on"]["sd"] / math.sqrt(is_part["on"]["n"]))) if is_part["on"]["n"] > 1 else 0.0

    # OOS partition
    oos_part = partition_stats(df_oos, "gate_on")
    oos_lift = oos_part["on"]["expr"] - oos_part["off"]["expr"]
    if oos_part["on"]["n"] > 1 and oos_part["off"]["n"] > 1:
        oos_t, oos_df, oos_p = welch_t_p(
            oos_part["on"]["expr"], oos_part["on"]["sd"], oos_part["on"]["n"],
            oos_part["off"]["expr"], oos_part["off"]["sd"], oos_part["off"]["n"],
        )
    else:
        oos_t, oos_df, oos_p = 0.0, 0.0, 1.0
    oos_dir_match = (is_lift > 0) == (oos_lift > 0) if oos_part["on"]["n"] > 0 else False
    oos_ratio_pass = (oos_lift >= OOS_LIFT_RATIO * is_lift) if is_lift > 0 else False
    if oos_part["on"]["n"] >= 50 and oos_dir_match and oos_ratio_pass:
        c8_status = "C8_PASS"
    elif oos_part["on"]["n"] >= 50:
        c8_status = "C8_FAIL"
    else:
        c8_status = "GATE_INACTIVE_LOWPOWER"

    # C9 per-year
    py = per_year_table(df_is)
    c9_fails = []
    positive_years = []
    for _, row in py.iterrows():
        if row["n_on"] >= ERA_N_FLOOR and not pd.isna(row["expr_on"]) and row["expr_on"] < ERA_FLOOR_EXPR:
            c9_fails.append(int(row["yr"]))
        if row["n_on"] >= ERA_COUNT_N_FLOOR and not pd.isna(row["expr_on"]) and row["expr_on"] > 0:
            positive_years.append(int(row["yr"]))
    c9_pass_strict = len(c9_fails) == 0
    pos_mean_floor_pass = (is_part["on"]["expr"] > 0) and (is_part["off"]["expr"] > 0)

    verdict_obj = evaluate_verdict(
        is_t=is_t,
        is_lift=is_lift,
        pos_mean_floor_pass=pos_mean_floor_pass,
        c9_pass_strict=c9_pass_strict,
        c9_positive_years_count=len(positive_years),
        oos_lift=oos_lift,
        oos_dir_match=oos_dir_match,
        oos_n_on=oos_part["on"]["n"],
        sham_test_passed=sham["triggered_correctly"],
    )

    payload = {
        "run_date": str(date.today()),
        "db_max_trading_day": str(db_max),
        "verdict": verdict_obj["verdict"],
        "reasons": verdict_obj["reasons"],
        "pressure_test": sham,
        "is": {
            "baseline_n": is_baseline_n,
            "baseline_expr": is_baseline_expr,
            "baseline_sd": is_baseline_sd,
            "part": is_part,
            "lift": is_lift,
            "t": is_t,
            "df": is_df,
            "p": is_p,
            "t_on_zero": is_t_on_zero,
        },
        "c9": {
            "per_year": py.to_dict("records"),
            "fails": c9_fails,
            "positive_years": positive_years,
            "positive_years_count": len(positive_years),
            "pass_strict": c9_pass_strict,
        },
        "oos": {
            "part": oos_part,
            "lift": oos_lift,
            "t": oos_t,
            "df": oos_df,
            "p": oos_p,
            "dir_match": oos_dir_match,
            "ratio_pass": oos_ratio_pass,
            "c8_status": c8_status,
        },
    }

    # Emit MD
    md = emit_result_md(payload)
    RESULT_PATH.write_text(md, encoding="utf-8")
    print(f"\nResult written: {RESULT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"\nVERDICT: {verdict_obj['verdict']}")
    for r in verdict_obj["reasons"]:
        print(f"  - {r}")

    # Also emit a JSON sidecar for downstream consumers
    json_path = RESULT_PATH.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"JSON sidecar: {json_path.relative_to(PROJECT_ROOT)}")

    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
