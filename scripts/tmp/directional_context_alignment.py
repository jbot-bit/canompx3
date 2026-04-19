#!/usr/bin/env python3
"""Directional Context Alignment — Phase A Screening.

Hypothesis: docs/audit/hypotheses/2026-04-13-directional-context-alignment.yaml
Holdout: Mode A, 2026-01-01 sacred.

Tests 3 mechanisms (M3 overnight momentum, M4 cross-session state, M5 pre-1000
expansion) across 6 hypothesis groups = 18 trials total.

Read-only. No pipeline changes. Re-runnable.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402

HOLDOUT_DATE = "2026-01-01"
HYPOTHESIS_FILE = PROJECT_ROOT / "docs" / "audit" / "hypotheses" / "2026-04-13-directional-context-alignment.yaml"
PHASE_A_K = 18  # total Phase A trials for BH FDR

# Verify hypothesis file SHA
HYP_SHA = hashlib.sha256(HYPOTHESIS_FILE.read_bytes()).hexdigest()
print(f"Hypothesis file SHA: {HYP_SHA[:16]}")
print(f"Holdout: {HOLDOUT_DATE} (Mode A sacred)")
print(f"Phase A trials: {PHASE_A_K}")
print()


def welch_t_test(aligned: np.ndarray, misaligned: np.ndarray) -> tuple[float, float]:
    """Welch's t-test for unequal variances. Returns (t_stat, p_value)."""
    if len(aligned) < 5 or len(misaligned) < 5:
        return 0.0, 1.0
    a = np.asarray(aligned, dtype=np.float64)
    b = np.asarray(misaligned, dtype=np.float64)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return 0.0, 1.0
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def bh_fdr(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR correction. Returns list of booleans."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed, 1):
        threshold = alpha * rank / n
        if p <= threshold:
            significant[orig_idx] = True
        else:
            break
    return significant


def year_by_year(con: duckdb.DuckDBPyConnection, query_template: str, params: dict, label: str) -> None:
    """Print year-by-year breakdown for a signal."""
    yearly = con.sql(f"""
        WITH base AS ({query_template})
        SELECT EXTRACT(YEAR FROM trading_day) as yr,
               COUNT(*) as n,
               SUM(CASE WHEN aligned THEN 1 ELSE 0 END) as n_al,
               ROUND(AVG(CASE WHEN aligned THEN pnl_r END), 4) as al_r,
               ROUND(AVG(CASE WHEN NOT aligned THEN pnl_r END), 4) as mis_r
        FROM base
        GROUP BY yr ORDER BY yr
    """).fetchall()
    print(f"    Year-by-year ({label}):")
    for row in yearly:
        al_r = row[3] if row[3] is not None else 0
        mis_r = row[4] if row[4] is not None else 0
        lift = al_r - mis_r
        print(
            f"      {int(row[0])}: N={row[1]:>4} aligned={row[2]:>4} "
            f"al_R={al_r:>+7.4f} mis_R={mis_r:>+7.4f} lift={lift:>+7.3f}"
        )


def run_m3_overnight_momentum(
    con: duckdb.DuckDBPyConnection, session: str, orb_min: int, rr: float, instrument: str, period: str
) -> dict:
    """M3: Overnight directional momentum.

    Aligned = (long break + took_pdh only) OR (short break + took_pdl only).
    """
    date_filter = f"o.trading_day < '{HOLDOUT_DATE}'" if period == "IS" else f"o.trading_day >= '{HOLDOUT_DATE}'"
    bd_col = f"orb_{session}_break_dir"

    query = f"""
        SELECT o.trading_day, o.pnl_r,
               d.{bd_col} as break_dir,
               d.overnight_took_pdh as tph,
               d.overnight_took_pdl as tpl,
               d.overnight_range_pct as ovnrng,
               CASE WHEN (d.{bd_col} = 'long' AND d.overnight_took_pdh = true
                          AND d.overnight_took_pdl = false)
                    OR (d.{bd_col} = 'short' AND d.overnight_took_pdl = true
                        AND d.overnight_took_pdh = false)
                    THEN true ELSE false END as aligned
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day AND o.symbol = d.symbol
          AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = '{instrument}' AND o.orb_label = '{session}'
          AND o.orb_minutes = {orb_min} AND o.entry_model = 'E2'
          AND o.confirm_bars = 1 AND o.rr_target = {rr}
          AND {date_filter}
          AND d.{bd_col} IN ('long', 'short')
    """
    rows = con.sql(query).fetchall()
    if not rows:
        return {"n": 0}

    aligned_r = np.array([float(r[1]) for r in rows if r[6] and r[1] is not None], dtype=np.float64)
    misaligned_r = np.array([float(r[1]) for r in rows if not r[6] and r[1] is not None], dtype=np.float64)

    # Confound: avg OVNRNG for aligned vs misaligned
    al_ovnrng = np.array([float(r[5]) for r in rows if r[6] and r[5] is not None], dtype=np.float64)
    mis_ovnrng = np.array([float(r[5]) for r in rows if not r[6] and r[5] is not None], dtype=np.float64)

    t, p = welch_t_test(aligned_r, misaligned_r)

    return {
        "n": len(rows),
        "n_aligned": len(aligned_r),
        "n_misaligned": len(misaligned_r),
        "aligned_avg_r": float(np.mean(aligned_r)) if len(aligned_r) else None,
        "misaligned_avg_r": float(np.mean(misaligned_r)) if len(misaligned_r) else None,
        "t_stat": t,
        "p_value": p,
        "al_ovnrng_avg": float(np.mean(al_ovnrng)) if len(al_ovnrng) else None,
        "mis_ovnrng_avg": float(np.mean(mis_ovnrng)) if len(mis_ovnrng) else None,
        "query": query,
    }


def run_m4_cross_session_state(
    con: duckdb.DuckDBPyConnection,
    prior_session: str,
    current_session: str,
    orb_min: int,
    rr: float,
    instrument: str,
    period: str,
) -> dict:
    """M4: Cross-session 4-state machine.

    States: PRIOR_WIN_ALIGN, PRIOR_WIN_OPPOSED, PRIOR_LOSS_ALIGN, PRIOR_LOSS_OPPOSED
    Prior win = prior session pnl_r > 0 (same day).
    Aligned = prior and current break in same direction.
    """
    date_filter = f"o.trading_day < '{HOLDOUT_DATE}'" if period == "IS" else f"o.trading_day >= '{HOLDOUT_DATE}'"
    prior_dir = f"orb_{prior_session}_break_dir"
    prior_outcome = f"orb_{prior_session}_outcome"
    curr_dir = f"orb_{current_session}_break_dir"

    query = f"""
        SELECT o.trading_day, o.pnl_r,
               d.{prior_dir} as prior_dir,
               d.{prior_outcome} as prior_outcome,
               d.{curr_dir} as curr_dir,
               CASE WHEN d.{prior_dir} = d.{curr_dir} THEN true ELSE false END as same_dir,
               CASE WHEN d.{prior_outcome} IN ('win', 'target') THEN 'win' ELSE 'loss' END as prior_result,
               -- 4-state
               CASE
                 WHEN d.{prior_outcome} IN ('win','target') AND d.{prior_dir} = d.{curr_dir}
                   THEN 'PRIOR_WIN_ALIGN'
                 WHEN d.{prior_outcome} IN ('win','target') AND d.{prior_dir} != d.{curr_dir}
                   THEN 'PRIOR_WIN_OPPOSED'
                 WHEN d.{prior_outcome} NOT IN ('win','target') AND d.{prior_dir} = d.{curr_dir}
                   THEN 'PRIOR_LOSS_ALIGN'
                 ELSE 'PRIOR_LOSS_OPPOSED'
               END as state
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day AND o.symbol = d.symbol
          AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = '{instrument}' AND o.orb_label = '{current_session}'
          AND o.orb_minutes = {orb_min} AND o.entry_model = 'E2'
          AND o.confirm_bars = 1 AND o.rr_target = {rr}
          AND {date_filter}
          AND d.{prior_dir} IN ('long', 'short')
          AND d.{curr_dir} IN ('long', 'short')
          AND d.{prior_outcome} IS NOT NULL
    """
    rows = con.sql(query).fetchall()
    if not rows:
        return {"n": 0}

    # Binary: same-dir vs opposite
    same_r = np.array([float(r[1]) for r in rows if r[5] and r[1] is not None], dtype=np.float64)
    opp_r = np.array([float(r[1]) for r in rows if not r[5] and r[1] is not None], dtype=np.float64)
    t_binary, p_binary = welch_t_test(same_r, opp_r)

    # 4-state breakdown
    states = {}
    for state_name in ["PRIOR_WIN_ALIGN", "PRIOR_WIN_OPPOSED", "PRIOR_LOSS_ALIGN", "PRIOR_LOSS_OPPOSED"]:
        vals = np.array([float(r[1]) for r in rows if r[7] == state_name and r[1] is not None], dtype=np.float64)
        states[state_name] = {
            "n": len(vals),
            "avg_r": float(np.mean(vals)) if len(vals) else None,
        }

    # Best actionable split: TAKE states vs VETO states
    # Codex finding: take = WIN_ALIGN + LOSS_OPPOSED, veto = WIN_OPPOSED + LOSS_ALIGN
    take_r = np.array(
        [float(r[1]) for r in rows if r[7] in ("PRIOR_WIN_ALIGN", "PRIOR_LOSS_OPPOSED") and r[1] is not None],
        dtype=np.float64,
    )
    veto_r = np.array(
        [float(r[1]) for r in rows if r[7] in ("PRIOR_WIN_OPPOSED", "PRIOR_LOSS_ALIGN") and r[1] is not None],
        dtype=np.float64,
    )
    t_4state, p_4state = welch_t_test(take_r, veto_r)

    return {
        "n": len(rows),
        "binary": {
            "n_same": len(same_r),
            "n_opp": len(opp_r),
            "same_avg_r": float(np.mean(same_r)) if len(same_r) else None,
            "opp_avg_r": float(np.mean(opp_r)) if len(opp_r) else None,
            "t": t_binary,
            "p": p_binary,
        },
        "states": states,
        "take_veto": {
            "n_take": len(take_r),
            "n_veto": len(veto_r),
            "take_avg_r": float(np.mean(take_r)) if len(take_r) else None,
            "veto_avg_r": float(np.mean(veto_r)) if len(veto_r) else None,
            "t": t_4state,
            "p": p_4state,
        },
        "query": query,
    }


def run_m5_pre1000_expansion(
    con: duckdb.DuckDBPyConnection, session: str, orb_min: int, rr: float, instrument: str, period: str
) -> dict:
    """M5: Pre-1000 range expansion.

    Aligned = (long break + took_pdh_before_1000) OR (short break + took_pdl_before_1000).
    """
    date_filter = f"o.trading_day < '{HOLDOUT_DATE}'" if period == "IS" else f"o.trading_day >= '{HOLDOUT_DATE}'"
    bd_col = f"orb_{session}_break_dir"

    query = f"""
        SELECT o.trading_day, o.pnl_r,
               d.{bd_col} as break_dir,
               d.took_pdh_before_1000 as tph,
               d.took_pdl_before_1000 as tpl,
               CASE WHEN (d.{bd_col} = 'long' AND d.took_pdh_before_1000 = true)
                    OR (d.{bd_col} = 'short' AND d.took_pdl_before_1000 = true)
                    THEN true ELSE false END as aligned
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day AND o.symbol = d.symbol
          AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = '{instrument}' AND o.orb_label = '{session}'
          AND o.orb_minutes = {orb_min} AND o.entry_model = 'E2'
          AND o.confirm_bars = 1 AND o.rr_target = {rr}
          AND {date_filter}
          AND d.{bd_col} IN ('long', 'short')
          AND d.took_pdh_before_1000 IS NOT NULL
    """
    rows = con.sql(query).fetchall()
    if not rows:
        return {"n": 0}

    aligned_r = np.array([float(r[1]) for r in rows if r[5] and r[1] is not None], dtype=np.float64)
    misaligned_r = np.array([float(r[1]) for r in rows if not r[5] and r[1] is not None], dtype=np.float64)
    t, p = welch_t_test(aligned_r, misaligned_r)

    return {
        "n": len(rows),
        "n_aligned": len(aligned_r),
        "n_misaligned": len(misaligned_r),
        "aligned_avg_r": float(np.mean(aligned_r)) if len(aligned_r) else None,
        "misaligned_avg_r": float(np.mean(misaligned_r)) if len(misaligned_r) else None,
        "t_stat": t,
        "p_value": p,
    }


def main() -> None:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    all_p_values: list[tuple[str, float]] = []  # (label, p) for BH FDR

    # ================================================================
    # MECHANISM M3: Overnight Directional Momentum
    # ================================================================
    print("=" * 70)
    print("MECHANISM M3: Overnight Directional Momentum")
    print("Signal: took_pdh_only + long = aligned; took_pdl_only + short = aligned")
    print("=" * 70)

    for hyp_id, session in [(1, "SINGAPORE_OPEN"), (2, "TOKYO_OPEN")]:
        print(f"\n--- Hypothesis {hyp_id}: M3 on MNQ {session} O5 ---")
        for rr in [1.0, 1.5, 2.0]:
            label = f"M3_{session}_RR{rr}"

            is_result = run_m3_overnight_momentum(con, session, 5, rr, "MNQ", "IS")
            oos_result = run_m3_overnight_momentum(con, session, 5, rr, "MNQ", "OOS")

            if is_result["n"] < 30:
                print(f"  RR{rr}: SKIP (IS N={is_result['n']} < 30)")
                all_p_values.append((label, 1.0))
                continue

            al_r = is_result["aligned_avg_r"] or 0
            mis_r = is_result["misaligned_avg_r"] or 0
            lift = al_r - mis_r
            oos_al = oos_result.get("aligned_avg_r") or 0
            oos_mis = oos_result.get("misaligned_avg_r") or 0
            oos_lift = oos_al - oos_mis

            all_p_values.append((label, is_result["p_value"]))

            print(
                f"  RR{rr}: IS N={is_result['n']} "
                f"al={is_result['n_aligned']}({al_r:+.4f}) "
                f"mis={is_result['n_misaligned']}({mis_r:+.4f}) "
                f"lift={lift:+.3f} t={is_result['t_stat']:.2f} p={is_result['p_value']:.6f}"
            )
            print(
                f"         OOS N={oos_result.get('n', 0)} "
                f"al={oos_result.get('n_aligned', 0)}({oos_al:+.4f}) "
                f"mis={oos_result.get('n_misaligned', 0)}({oos_mis:+.4f}) "
                f"lift={oos_lift:+.3f}"
            )

            # Confound check
            if is_result.get("al_ovnrng_avg"):
                print(
                    f"         Confound: aligned avg_ovnrng={is_result['al_ovnrng_avg']:.1f}% "
                    f"misaligned={is_result['mis_ovnrng_avg']:.1f}%"
                )

    # ================================================================
    # MECHANISM M4: Cross-Session State Machine
    # ================================================================
    print("\n" + "=" * 70)
    print("MECHANISM M4: Cross-Session State Machine (4-state)")
    print("States: PRIOR_WIN_ALIGN, PRIOR_WIN_OPPOSED, PRIOR_LOSS_ALIGN, PRIOR_LOSS_OPPOSED")
    print("Take = WIN_ALIGN + LOSS_OPPOSED; Veto = WIN_OPPOSED + LOSS_ALIGN")
    print("=" * 70)

    for hyp_id, (prior, current) in [(3, ("NYSE_OPEN", "US_DATA_1000")), (4, ("COMEX_SETTLE", "CME_PRECLOSE"))]:
        print(f"\n--- Hypothesis {hyp_id}: M4 {prior}>{current} MNQ O5 ---")
        for rr in [1.0, 1.5, 2.0]:
            label = f"M4_{prior}_{current}_RR{rr}"

            is_result = run_m4_cross_session_state(con, prior, current, 5, rr, "MNQ", "IS")
            oos_result = run_m4_cross_session_state(con, prior, current, 5, rr, "MNQ", "OOS")

            if is_result["n"] < 30:
                print(f"  RR{rr}: SKIP (IS N={is_result['n']} < 30)")
                all_p_values.append((label, 1.0))
                continue

            # Use the 4-state take/veto p-value (more informative than binary)
            tv = is_result["take_veto"]
            tv_oos = oos_result.get("take_veto", {})
            all_p_values.append((label, tv["p"]))

            take_r = tv["take_avg_r"] or 0
            veto_r = tv["veto_avg_r"] or 0
            lift = take_r - veto_r

            print(
                f"  RR{rr}: IS N={is_result['n']} "
                f"take={tv['n_take']}({take_r:+.4f}) "
                f"veto={tv['n_veto']}({veto_r:+.4f}) "
                f"lift={lift:+.3f} t={tv['t']:.2f} p={tv['p']:.6f}"
            )

            # Binary comparison too
            b = is_result["binary"]
            print(
                f"         Binary: same={b['n_same']}({(b['same_avg_r'] or 0):+.4f}) "
                f"opp={b['n_opp']}({(b['opp_avg_r'] or 0):+.4f}) "
                f"t={b['t']:.2f} p={b['p']:.6f}"
            )

            # OOS
            if oos_result.get("n", 0) > 0:
                tv_o = oos_result.get("take_veto", {})
                b_o = oos_result.get("binary", {})
                print(
                    f"         OOS N={oos_result['n']} "
                    f"take={tv_o.get('n_take', 0)}({(tv_o.get('take_avg_r') or 0):+.4f}) "
                    f"veto={tv_o.get('n_veto', 0)}({(tv_o.get('veto_avg_r') or 0):+.4f}) "
                    f"lift={(tv_o.get('take_avg_r') or 0) - (tv_o.get('veto_avg_r') or 0):+.3f}"
                )

            # 4-state breakdown
            print("         States:")
            for state_name in ["PRIOR_WIN_ALIGN", "PRIOR_WIN_OPPOSED", "PRIOR_LOSS_ALIGN", "PRIOR_LOSS_OPPOSED"]:
                is_s = is_result["states"].get(state_name, {})
                oos_s = oos_result.get("states", {}).get(state_name, {})
                print(
                    f"           {state_name}: IS N={is_s.get('n', 0)} "
                    f"R={is_s.get('avg_r') or 0:+.4f} | "
                    f"OOS N={oos_s.get('n', 0)} R={oos_s.get('avg_r') or 0:+.4f}"
                )

    # ================================================================
    # MECHANISM M5: Pre-1000 Range Expansion
    # ================================================================
    print("\n" + "=" * 70)
    print("MECHANISM M5: Pre-1000 Range Expansion (exploratory)")
    print("Signal: took_pdh_before_1000 + long = aligned; took_pdl + short = aligned")
    print("WARNING: 13.8% pass rate — expect thin aligned N")
    print("=" * 70)

    for hyp_id, session in [(5, "NYSE_OPEN"), (6, "US_DATA_1000")]:
        print(f"\n--- Hypothesis {hyp_id}: M5 on MNQ {session} O5 ---")
        for rr in [1.0, 1.5, 2.0]:
            label = f"M5_{session}_RR{rr}"

            is_result = run_m5_pre1000_expansion(con, session, 5, rr, "MNQ", "IS")
            oos_result = run_m5_pre1000_expansion(con, session, 5, rr, "MNQ", "OOS")

            if is_result["n"] < 30:
                print(f"  RR{rr}: SKIP (IS N={is_result['n']} < 30)")
                all_p_values.append((label, 1.0))
                continue

            if is_result["n_aligned"] < 30:
                print(f"  RR{rr}: UNDERPOWERED (aligned N={is_result['n_aligned']} < 30)")
                all_p_values.append((label, 1.0))
                continue

            al_r = is_result["aligned_avg_r"] or 0
            mis_r = is_result["misaligned_avg_r"] or 0
            lift = al_r - mis_r

            all_p_values.append((label, is_result["p_value"]))

            print(
                f"  RR{rr}: IS N={is_result['n']} "
                f"al={is_result['n_aligned']}({al_r:+.4f}) "
                f"mis={is_result['n_misaligned']}({mis_r:+.4f}) "
                f"lift={lift:+.3f} t={is_result['t_stat']:.2f} p={is_result['p_value']:.6f}"
            )
            oos_al = oos_result.get("aligned_avg_r") or 0
            oos_mis = oos_result.get("misaligned_avg_r") or 0
            print(
                f"         OOS N={oos_result.get('n', 0)} "
                f"al={oos_result.get('n_aligned', 0)}({oos_al:+.4f}) "
                f"mis={oos_result.get('n_misaligned', 0)}({oos_mis:+.4f}) "
                f"lift={oos_al - oos_mis:+.3f}"
            )

    # ================================================================
    # PHASE A.5: MES Cross-Instrument Robustness (for Phase A survivors)
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE A.5: MES Cross-Instrument Quick Check")
    print("(Not counted as separate trials — robustness check only)")
    print("=" * 70)

    # Run same best-case combos on MES
    mes_checks = [
        ("M3", "SINGAPORE_OPEN", None, 5, 1.5),
        ("M3", "TOKYO_OPEN", None, 5, 1.5),
        ("M4", "US_DATA_1000", "NYSE_OPEN", 5, 1.5),
        ("M4", "CME_PRECLOSE", "COMEX_SETTLE", 5, 1.5),
    ]
    for mech, sess, prior, om, rr in mes_checks:
        if mech == "M3":
            is_r = run_m3_overnight_momentum(con, sess, om, rr, "MES", "IS")
            oos_r = run_m3_overnight_momentum(con, sess, om, rr, "MES", "OOS")
            if is_r["n"] > 30:
                al = is_r["aligned_avg_r"] or 0
                mis = is_r["misaligned_avg_r"] or 0
                oos_al = oos_r.get("aligned_avg_r") or 0
                oos_mis = oos_r.get("misaligned_avg_r") or 0
                print(
                    f"  {mech} MES {sess}: IS lift={al - mis:+.3f} (N={is_r['n']}) "
                    f"OOS lift={oos_al - oos_mis:+.3f} (N={oos_r.get('n', 0)})"
                )
        elif mech == "M4":
            is_r = run_m4_cross_session_state(con, prior, sess, om, rr, "MES", "IS")
            oos_r = run_m4_cross_session_state(con, prior, sess, om, rr, "MES", "OOS")
            if is_r["n"] > 30:
                tv = is_r["take_veto"]
                tv_o = oos_r.get("take_veto", {})
                take_r = tv["take_avg_r"] or 0
                veto_r = tv["veto_avg_r"] or 0
                oos_take = tv_o.get("take_avg_r") or 0
                oos_veto = tv_o.get("veto_avg_r") or 0
                print(
                    f"  {mech} MES {prior}>{sess}: IS lift={take_r - veto_r:+.3f} (N={is_r['n']}) "
                    f"OOS lift={oos_take - oos_veto:+.3f} (N={oos_r.get('n', 0)})"
                )

    # ================================================================
    # BH FDR at Phase A K=18
    # ================================================================
    print("\n" + "=" * 70)
    print(f"BH FDR CORRECTION (K={PHASE_A_K})")
    print("=" * 70)

    p_vals = [p for _, p in all_p_values]
    labels = [l for l, _ in all_p_values]
    sig = bh_fdr(p_vals)

    survivors = []
    print(f"\n{'Label':>40} {'p-value':>10} {'BH sig':>7}")
    print("-" * 60)
    for i, (label, p) in enumerate(all_p_values):
        status = "YES ***" if sig[i] else "no"
        print(f"{label:>40} {p:>10.6f} {status:>7}")
        if sig[i]:
            survivors.append(label)

    print(f"\nBH FDR survivors: {len(survivors)} / {len(all_p_values)}")
    for s in survivors:
        print(f"  {s}")

    # ================================================================
    # VERDICT
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE A VERDICT")
    print("=" * 70)

    if not survivors:
        print("ZERO BH FDR survivors. ALL mechanisms DEAD at Phase A.")
        print("No Phase B expansion warranted.")
    else:
        surviving_mechs = set(s.split("_")[0] for s in survivors)
        print(f"Surviving mechanisms: {surviving_mechs}")
        print("Phase B expansion warranted for surviving mechanisms.")
        print("Next: pre-register Phase B trials for survivors only.")

    con.close()


if __name__ == "__main__":
    main()
