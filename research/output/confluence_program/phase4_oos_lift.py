"""Phase 4: OOS Lift Validation for Stacked Confluence Filters.

Tests whether the in-sample stacking lift (Phase 3b) survives out-of-sample.

Methodology:
  - 8 COMPLEMENTARY O5 pairs from Phase 3b (N_both >= 100 IS) + 1 negative control
  - IS = trading_day < 2024-01-01, OOS = 2024-01-01 to 2025-12-31
  - IS champion pre-committed (fix max-bias: choose better filter in IS, compare OOS)
  - Bootstrap 95% CI of OOS lift (Efron percentile, 1000 resamples)
  - 5-gate decision: lift>0, CI excludes 0, ratio>=1.5x, N>=30, WR improves

Literature grounding:
  - Harvey & Liu (2014): selection bias when discovery window = backtest window
  - Carver (Systematic Trading): 1.5x hurdle for stacking over sample size penalty
  - Efron & Tibshirani (1993): bootstrap percentile CI for interaction effects
  - Aronson Ch 6: each combination = new hypothesis under multiple testing

Decision rules (PRE-COMMITTED before running):
  ALL five must pass for DEPLOY:
    1. Lift_OOS > 0
    2. Lift_OOS 95% CI excludes 0
    3. ExpR_both / ExpR_champion >= 1.5 in OOS
    4. N_both_OOS >= 30
    5. WR_both_OOS > WR_champion_OOS (signal, not arithmetic)
  Any failure = CLOSE the pair.

@research-source research/output/confluence_program/phase3b_redundancy.py
@entry-models E2
@revalidated-for E2 event-based (2026-03-30)
"""

import sys

sys.path.insert(0, r"C:\Users\joshd\canompx3")

import duckdb
import numpy as np
import pandas as pd
from pipeline.paths import GOLD_DB_PATH

OUT = r"C:\Users\joshd\canompx3\research\output\confluence_program"
IS_CUTOFF = "2024-01-01"
OOS_CUTOFF = "2026-01-01"
N_BOOT = 1000
CARVER_HURDLE = 1.5
RNG = np.random.default_rng(42)

# fmt: off
# 8 COMPLEMENTARY O5 pairs from Phase 3b conditional_lift.csv (N_both >= 100)
# + 1 negative control (MGC NYSE_OPEN O30 — the only REDUNDANT pair)
PAIRS = [
    # (symbol, session, orb_minutes, feat_a, thresh_a, dir_a, feat_b, thresh_b, dir_b, label)
    ("MNQ", "CME_PRECLOSE", 5, "rel_vol",       1.31,   "high", "orb_volume",     3791.4,  "high", "CME_PRE rv+ov"),
    ("MNQ", "CME_PRECLOSE", 5, "rel_vol",       1.31,   "high", "overnight_range", 23.5,   "high", "CME_PRE rv+ovn"),
    ("MNQ", "CME_PRECLOSE", 5, "atr_20_pct",   22.22,   "high", "rel_vol",         1.31,   "high", "CME_PRE atr+rv"),
    ("MNQ", "CME_PRECLOSE", 5, "rel_vol",       1.31,   "high", "orb_volume",     3791.4,  "high", "CME_PRE ov+rv"),  # same as #1, skip below
    ("MNQ", "NYSE_CLOSE",   5, "rel_vol",       2.46,   "high", "orb_size_norm",   0.10,   "high", "NYSE_CL rv+os"),
    ("MNQ", "NYSE_CLOSE",   5, "orb_volume",  4812.6,   "high", "orb_size_norm",   0.10,   "high", "NYSE_CL ov+os"),
    ("MNQ", "NYSE_CLOSE",   5, "orb_volume",  4812.6,   "high", "rel_vol",         2.46,   "high", "NYSE_CL ov+rv"),
    ("MNQ", "NYSE_CLOSE",   5, "orb_volume",  4812.6,   "high", "atr_vel_ratio",   1.05,   "high", "NYSE_CL ov+atvr"),
    ("MES", "CME_PRECLOSE", 5, "rel_vol",       1.73,   "high", "orb_size_norm",   0.12,   "high", "MES_PRE rv+os"),
    # Negative control — REDUNDANT pair from Phase 3b (lift = -0.032 IS)
    ("MGC", "NYSE_OPEN",   30, "rel_vol",       2.30,   "high", "orb_size_norm",   0.32,   "high", "NEG_CTRL mgc"),
]
# fmt: on

# Remove duplicate (pair #4 = pair #1)
PAIRS.pop(3)


def get_feat_expr(feat: str, session: str) -> str:
    """Map feature name to SQL expression against daily_features (d.)."""
    if feat == "orb_volume":
        return f'd."orb_{session}_volume"'
    elif feat == "orb_size_norm":
        return f'd."orb_{session}_size" / NULLIF(d.atr_20, 0)'
    elif feat == "rel_vol":
        return f'd."rel_vol_{session}"'
    elif feat == "overnight_range":
        return "d.overnight_range"
    elif feat == "atr_20_pct":
        return "d.atr_20_pct"
    elif feat == "atr_vel_ratio":
        return "d.atr_vel_ratio"
    raise ValueError(f"Unknown feature: {feat}")


def compute_stats(pnl: np.ndarray) -> dict:
    """Compute ExpR, WR, N from pnl array."""
    if len(pnl) == 0:
        return {"n": 0, "expr": np.nan, "wr": np.nan}
    return {
        "n": len(pnl),
        "expr": float(np.mean(pnl)),
        "wr": float(np.mean(pnl > 0) * 100),
    }


def bootstrap_lift_ci(
    pnl_both: np.ndarray, pnl_champion: np.ndarray, n_boot: int = N_BOOT
) -> tuple[float, float]:
    """Bootstrap 95% CI of lift = mean(both) - mean(champion).

    Uses Efron percentile method. Resamples trade indices within each group
    independently (paired by calendar day is NOT assumed — conservative).
    """
    if len(pnl_both) < 10 or len(pnl_champion) < 10:
        return (np.nan, np.nan)

    lifts = np.empty(n_boot)
    for b in range(n_boot):
        idx_b = RNG.integers(0, len(pnl_both), size=len(pnl_both))
        idx_c = RNG.integers(0, len(pnl_champion), size=len(pnl_champion))
        lifts[b] = np.mean(pnl_both[idx_b]) - np.mean(pnl_champion[idx_c])

    return (float(np.percentile(lifts, 2.5)), float(np.percentile(lifts, 97.5)))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default=None, help="Override DB path")
    args = parser.parse_args()

    db_path = args.db if args.db else str(GOLD_DB_PATH)
    con = duckdb.connect(db_path, read_only=True)
    results = []

    print("=" * 80)
    print("PHASE 4: OOS LIFT VALIDATION FOR STACKED CONFLUENCE FILTERS")
    print("=" * 80)
    print(f"IS: trading_day < {IS_CUTOFF}")
    print(f"OOS: {IS_CUTOFF} <= trading_day < {OOS_CUTOFF}")
    print(f"Bootstrap: {N_BOOT} resamples, seed=42")
    print(f"Carver hurdle: {CARVER_HURDLE}x")
    print(f"Pairs: {len(PAIRS)} ({len(PAIRS) - 1} candidates + 1 negative control)")
    print()

    for sym, sess, om, fa, ta, da, fb, tb, db, label in PAIRS:
        expr_a = get_feat_expr(fa, sess)
        expr_b = get_feat_expr(fb, sess)

        cond_a = f"({expr_a}) >= {ta}" if da == "high" else f"({expr_a}) <= {ta}"
        cond_b = f"({expr_b}) >= {tb}" if db == "high" else f"({expr_b}) <= {tb}"

        # Fetch all trades with feature values
        df = con.sql(f"""
            SELECT
                d.trading_day,
                o.pnl_r,
                CASE WHEN {cond_a} THEN 1 ELSE 0 END as pass_a,
                CASE WHEN {cond_b} THEN 1 ELSE 0 END as pass_b,
                CASE WHEN d.trading_day < '{IS_CUTOFF}' THEN 'IS' ELSE 'OOS' END as window
            FROM orb_outcomes o
            JOIN daily_features d ON o.trading_day = d.trading_day
                AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
            WHERE o.symbol = '{sym}' AND o.orb_minutes = {om}
                AND o.orb_label = '{sess}' AND o.entry_model = 'E2'
                AND o.rr_target = 1.0 AND o.pnl_r IS NOT NULL
                AND ({expr_a}) IS NOT NULL AND ({expr_b}) IS NOT NULL
                AND d.trading_day < '{OOS_CUTOFF}'
        """).fetchdf()

        if len(df) == 0:
            print(f"  {label}: NO DATA — skipping")
            continue

        # Split IS / OOS
        is_df = df[df["window"] == "IS"]
        oos_df = df[df["window"] == "OOS"]

        # IS stats
        is_a = compute_stats(is_df[is_df["pass_a"] == 1]["pnl_r"].values)
        is_b = compute_stats(is_df[is_df["pass_b"] == 1]["pnl_r"].values)
        is_both = compute_stats(
            is_df[(is_df["pass_a"] == 1) & (is_df["pass_b"] == 1)]["pnl_r"].values
        )

        # IS champion pre-commitment (fix max bias)
        if is_a["expr"] >= is_b["expr"]:
            champion_label = fa
            champion_is = "A"
        else:
            champion_label = fb
            champion_is = "B"

        # OOS stats
        oos_a = compute_stats(oos_df[oos_df["pass_a"] == 1]["pnl_r"].values)
        oos_b = compute_stats(oos_df[oos_df["pass_b"] == 1]["pnl_r"].values)
        oos_both = compute_stats(
            oos_df[(oos_df["pass_a"] == 1) & (oos_df["pass_b"] == 1)]["pnl_r"].values
        )

        # OOS champion = same filter chosen in IS (pre-committed)
        oos_champ = oos_a if champion_is == "A" else oos_b

        # Lift
        lift_is = is_both["expr"] - max(is_a["expr"], is_b["expr"])
        lift_oos = oos_both["expr"] - oos_champ["expr"] if not np.isnan(oos_both["expr"]) else np.nan

        # Bootstrap CI of OOS lift
        pnl_both_oos = oos_df[(oos_df["pass_a"] == 1) & (oos_df["pass_b"] == 1)]["pnl_r"].values
        pnl_champ_oos = (
            oos_df[oos_df["pass_a"] == 1]["pnl_r"].values
            if champion_is == "A"
            else oos_df[oos_df["pass_b"] == 1]["pnl_r"].values
        )
        ci_lo, ci_hi = bootstrap_lift_ci(pnl_both_oos, pnl_champ_oos)

        # Ratio
        ratio = (
            oos_both["expr"] / oos_champ["expr"]
            if oos_champ["expr"] > 0 and not np.isnan(oos_both["expr"])
            else np.nan
        )

        # 5-gate decision
        g1_lift_pos = not np.isnan(lift_oos) and lift_oos > 0
        g2_ci_excl_zero = not np.isnan(ci_lo) and ci_lo > 0
        g3_carver = not np.isnan(ratio) and ratio >= CARVER_HURDLE
        g4_n_min = oos_both["n"] >= 30
        g5_wr = (
            not np.isnan(oos_both["wr"])
            and not np.isnan(oos_champ["wr"])
            and oos_both["wr"] > oos_champ["wr"]
        )

        gates_passed = sum([g1_lift_pos, g2_ci_excl_zero, g3_carver, g4_n_min, g5_wr])
        if gates_passed == 5:
            verdict = "DEPLOY"
        elif g1_lift_pos and g4_n_min:
            verdict = "MARGINAL"
        else:
            verdict = "DEAD"

        is_neg_ctrl = label.startswith("NEG_CTRL")

        row = {
            "label": label,
            "symbol": sym,
            "session": sess,
            "orb_minutes": om,
            "feat_a": fa,
            "feat_b": fb,
            "champion_is": champion_label,
            "n_total_is": len(is_df),
            "n_both_is": is_both["n"],
            "expr_a_is": round(is_a["expr"], 5),
            "expr_b_is": round(is_b["expr"], 5),
            "expr_both_is": round(is_both["expr"], 5),
            "lift_is": round(lift_is, 5),
            "n_total_oos": len(oos_df),
            "n_both_oos": oos_both["n"],
            "expr_champ_oos": round(oos_champ["expr"], 5) if not np.isnan(oos_champ["expr"]) else None,
            "wr_champ_oos": round(oos_champ["wr"], 2) if not np.isnan(oos_champ["wr"]) else None,
            "expr_both_oos": round(oos_both["expr"], 5) if not np.isnan(oos_both["expr"]) else None,
            "wr_both_oos": round(oos_both["wr"], 2) if not np.isnan(oos_both["wr"]) else None,
            "lift_oos": round(lift_oos, 5) if not np.isnan(lift_oos) else None,
            "ci_lo": round(ci_lo, 5) if not np.isnan(ci_lo) else None,
            "ci_hi": round(ci_hi, 5) if not np.isnan(ci_hi) else None,
            "ratio": round(ratio, 3) if not np.isnan(ratio) else None,
            "g1_lift_pos": g1_lift_pos,
            "g2_ci_excl_zero": g2_ci_excl_zero,
            "g3_carver_1.5x": g3_carver,
            "g4_n_min_30": g4_n_min,
            "g5_wr_improves": g5_wr,
            "gates_passed": gates_passed,
            "verdict": verdict,
            "negative_control": is_neg_ctrl,
        }
        results.append(row)

        # Print
        print(f"--- {label} ({sym} {sess} O{om}) ---")
        print(f"  Champion (IS): {champion_label}")
        print(f"  IS:  A={is_a['expr']:+.4f} B={is_b['expr']:+.4f} "
              f"Both={is_both['expr']:+.4f} Lift={lift_is:+.4f} N_both={is_both['n']}")
        oos_lift_str = f"{lift_oos:+.4f}" if not np.isnan(lift_oos) else "N/A"
        print(f"  OOS: Champ={oos_champ['expr']:+.4f} "
              f"Both={oos_both['expr']:+.4f} Lift={oos_lift_str} N_both={oos_both['n']}")
        ci_str = f"[{ci_lo:+.4f}, {ci_hi:+.4f}]" if not np.isnan(ci_lo) else "N/A"
        print(f"  CI95: {ci_str}  Ratio: {ratio:.3f}x" if not np.isnan(ratio) else f"  CI95: {ci_str}  Ratio: N/A")
        print(f"  WR: champ={oos_champ['wr']:.1f}% both={oos_both['wr']:.1f}%"
              if not np.isnan(oos_both['wr']) else "  WR: insufficient data")
        gate_str = " ".join([
            f"G1:{'PASS' if g1_lift_pos else 'FAIL'}",
            f"G2:{'PASS' if g2_ci_excl_zero else 'FAIL'}",
            f"G3:{'PASS' if g3_carver else 'FAIL'}",
            f"G4:{'PASS' if g4_n_min else 'FAIL'}",
            f"G5:{'PASS' if g5_wr else 'FAIL'}",
        ])
        ctrl_tag = " [NEGATIVE CONTROL]" if is_neg_ctrl else ""
        print(f"  Gates: {gate_str} -> {verdict}{ctrl_tag}")
        print()

    # Save CSV
    rdf = pd.DataFrame(results)
    rdf.to_csv(f"{OUT}/phase4_oos_lift.csv", index=False)
    print(f"Results saved to {OUT}/phase4_oos_lift.csv")

    # Summary
    print()
    print("=" * 80)
    print("PHASE 4 SUMMARY")
    print("=" * 80)
    candidates = rdf[~rdf["negative_control"]]
    n_deploy = len(candidates[candidates["verdict"] == "DEPLOY"])
    n_marginal = len(candidates[candidates["verdict"] == "MARGINAL"])
    n_dead = len(candidates[candidates["verdict"] == "DEAD"])
    print(f"  Candidates: {len(candidates)}")
    print(f"  DEPLOY: {n_deploy}")
    print(f"  MARGINAL: {n_marginal}")
    print(f"  DEAD: {n_dead}")

    neg_ctrl = rdf[rdf["negative_control"]]
    if len(neg_ctrl) > 0:
        nc = neg_ctrl.iloc[0]
        ctrl_ok = nc["verdict"] == "DEAD"
        print(f"  Negative control: {nc['verdict']} ({'CORRECT' if ctrl_ok else 'WARNING — test may be broken'})")

    if n_deploy == 0:
        print()
        print("  VERDICT: Stacking is DEAD OOS. Individual filters are sufficient.")
        print("  ACTION: Add NO-GO to STRATEGY_BLUEPRINT.md §5.")
        print("  REASON: In-sample lift is selection bias + regression to mean.")
        print("          Features are independent (Phase 3b rho < 0.40) but independence")
        print("          does not imply stacking benefit (Carver, Systematic Trading).")
    elif n_deploy >= 1:
        print()
        print(f"  VERDICT: {n_deploy} pair(s) survive all 5 gates.")
        print("  ACTION: Register as CompositeFilter in config.py.")
        deploy_rows = candidates[candidates["verdict"] == "DEPLOY"]
        for _, dr in deploy_rows.iterrows():
            print(f"    {dr['label']}: lift={dr['lift_oos']:+.4f} ratio={dr['ratio']:.2f}x")

    con.close()


if __name__ == "__main__":
    main()
