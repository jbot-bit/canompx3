#!/usr/bin/env python3
"""
Validate composite 1800 E3 strategy: 15m AGREE filter + split targets.

Standalone validation script -- reads from gold.db, produces honest report.
Does NOT modify any tables or existing code.

Usage:
    python trading_app/validate_1800_composite.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(line_buffering=True)

import duckdb
from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec, stress_test_costs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ORB_LABEL = "1800"
ENTRY_MODEL = "E3"
CONFIRM_BARS = 4
MIN_ORB_SIZE = 6.0  # G6+ filter
EXCLUDE_YEAR = 2021  # structurally different regime
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "1800_composite_validation.txt"

# ---------------------------------------------------------------------------
# Step 1: Data Assembly
# ---------------------------------------------------------------------------
def load_composite_data(db_path: Path) -> list[dict]:
    """Load and merge RR1.0 + RR2.0 outcomes with 5m/15m break directions."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        # Load orb_outcomes for 1800 E3 CB4, RR1.0 and RR2.0
        outcomes = con.execute("""
            SELECT trading_day, rr_target, outcome, pnl_r,
                   entry_price, stop_price, target_price
            FROM orb_outcomes
            WHERE orb_label = ? AND entry_model = ? AND confirm_bars = ?
              AND rr_target IN (1.0, 2.0)
              AND orb_minutes = 5
              AND outcome IN ('win', 'loss')
            ORDER BY trading_day, rr_target
        """, [ORB_LABEL, ENTRY_MODEL, CONFIRM_BARS]).fetchdf()

        # Load 5m daily_features for G6+ filter and break direction
        df5 = con.execute("""
            SELECT trading_day,
                   orb_1800_size AS orb_size,
                   orb_1800_break_dir AS dir_5m
            FROM daily_features
            WHERE symbol = 'MGC' AND orb_minutes = 5
              AND orb_1800_break_dir IS NOT NULL
        """).fetchdf()

        # Load 15m daily_features for break direction
        df15 = con.execute("""
            SELECT trading_day,
                   orb_1800_break_dir AS dir_15m
            FROM daily_features
            WHERE symbol = 'MGC' AND orb_minutes = 15
              AND orb_1800_break_dir IS NOT NULL
        """).fetchdf()
    finally:
        con.close()

    # Pivot outcomes: one row per trading_day with RR1.0 and RR2.0 columns
    rr1 = outcomes[outcomes["rr_target"] == 1.0].rename(
        columns={"outcome": "outcome_rr1", "pnl_r": "pnl_r_rr1"}
    )[["trading_day", "outcome_rr1", "pnl_r_rr1", "entry_price", "stop_price"]]

    rr2 = outcomes[outcomes["rr_target"] == 2.0].rename(
        columns={"outcome": "outcome_rr2", "pnl_r": "pnl_r_rr2"}
    )[["trading_day", "outcome_rr2", "pnl_r_rr2"]]

    # Merge RR1 and RR2 on trading_day (must have both)
    merged = rr1.merge(rr2, on="trading_day", how="inner")

    # Join 5m features
    merged = merged.merge(df5, on="trading_day", how="inner")

    # Join 15m features
    merged = merged.merge(df15, on="trading_day", how="inner")

    # Apply G6+ filter
    merged = merged[merged["orb_size"] >= MIN_ORB_SIZE].copy()

    # Exclude 2021
    merged["year"] = merged["trading_day"].apply(
        lambda d: d.year if hasattr(d, "year") else int(str(d)[:4])
    )
    merged = merged[merged["year"] != EXCLUDE_YEAR].copy()

    # Compute agree flag
    merged["agree"] = merged["dir_5m"] == merged["dir_15m"]

    # Combined PnL (2 contracts, each risking 1R)
    merged["pnl_r_combined"] = merged["pnl_r_rr1"] + merged["pnl_r_rr2"]

    return merged.to_dict("records")

# ---------------------------------------------------------------------------
# Step 2: Metrics Computation
# ---------------------------------------------------------------------------
def compute_composite_metrics(rows: list[dict], label: str = "") -> dict:
    """Compute metrics for a set of trade rows."""
    if not rows:
        return {"label": label, "N": 0}

    n = len(rows)
    pnl_combined = [r["pnl_r_combined"] for r in rows]
    wins_rr1 = sum(1 for r in rows if r["outcome_rr1"] == "win")
    wins_rr2 = sum(1 for r in rows if r["outcome_rr2"] == "win")

    total_r = sum(pnl_combined)
    exp_r = total_r / n  # per trade (risking 2R per trade)
    exp_r_per_r = exp_r / 2.0  # per R risked

    # Sharpe
    mean_r = total_r / n
    if n > 1:
        var = sum((p - mean_r) ** 2 for p in pnl_combined) / (n - 1)
        std_r = var ** 0.5
        sharpe = mean_r / std_r if std_r > 0 else None
    else:
        sharpe = None

    # Max drawdown
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in rows:
        cum += r["pnl_r_combined"]
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    # Profit factor
    gross_wins = sum(p for p in pnl_combined if p > 0)
    gross_losses = abs(sum(p for p in pnl_combined if p < 0))
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Yearly breakdown
    yearly = {}
    for r in rows:
        y = str(r["year"])
        if y not in yearly:
            yearly[y] = {"trades": 0, "wins_rr1": 0, "wins_rr2": 0, "total_r": 0.0}
        yearly[y]["trades"] += 1
        if r["outcome_rr1"] == "win":
            yearly[y]["wins_rr1"] += 1
        if r["outcome_rr2"] == "win":
            yearly[y]["wins_rr2"] += 1
        yearly[y]["total_r"] += r["pnl_r_combined"]

    return {
        "label": label,
        "N": n,
        "wr_rr1": wins_rr1 / n,
        "wr_rr2": wins_rr2 / n,
        "total_r": total_r,
        "exp_r": exp_r,
        "exp_r_per_r": exp_r_per_r,
        "sharpe": sharpe,
        "max_dd_r": max_dd,
        "profit_factor": pf,
        "yearly": yearly,
    }

# ---------------------------------------------------------------------------
# Step 3: Stress Testing
# ---------------------------------------------------------------------------
def stress_test_pnl(rows: list[dict], multiplier: float) -> list[dict]:
    """Re-compute PnL at stressed friction levels."""
    spec_base = get_cost_spec("MGC")
    spec_stress = stress_test_costs(spec_base, multiplier)

    stressed = []
    for r in rows:
        entry = r["entry_price"]
        stop = r["stop_price"]
        if entry is None or stop is None:
            continue

        risk_pts = abs(entry - stop)
        if risk_pts == 0:
            continue

        # Re-compute pnl_r for each RR target under stressed costs
        new_row = dict(r)
        for rr_key, outcome_key in [("pnl_r_rr1", "outcome_rr1"),
                                     ("pnl_r_rr2", "outcome_rr2")]:
            if r[outcome_key] == "win":
                # Winner: pnl_points = target distance from entry
                rr_val = 1.0 if "rr1" in rr_key else 2.0
                pnl_pts = risk_pts * rr_val
            else:
                # Loser: pnl_points = -risk_pts
                pnl_pts = -risk_pts

            # R-multiple with stressed costs
            risk_dollars = risk_pts * spec_stress.point_value + spec_stress.total_friction
            pnl_dollars = pnl_pts * spec_stress.point_value - spec_stress.total_friction
            new_row[rr_key] = pnl_dollars / risk_dollars

        new_row["pnl_r_combined"] = new_row["pnl_r_rr1"] + new_row["pnl_r_rr2"]
        stressed.append(new_row)

    return stressed

def find_breakeven_multiplier(rows: list[dict]) -> float:
    """Binary search for friction multiplier where ExpR = 0."""
    lo, hi = 1.0, 10.0
    for _ in range(30):
        mid = (lo + hi) / 2
        stressed = stress_test_pnl(rows, mid)
        if not stressed:
            return lo
        total_r = sum(r["pnl_r_combined"] for r in stressed)
        exp_r = total_r / len(stressed)
        if exp_r > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

# ---------------------------------------------------------------------------
# Step 4: Walk-Forward OOS
# ---------------------------------------------------------------------------
def walk_forward_expanding(rows: list[dict]) -> list[dict]:
    """Expanding window: train on earlier years, test on next."""
    folds = [
        {"train": [2022, 2023], "test": 2024},
        {"train": [2022, 2023, 2024], "test": 2025},
        {"train": [2022, 2023, 2024, 2025], "test": 2026},
    ]
    results = []
    for fold in folds:
        train = [r for r in rows if r["year"] in fold["train"]]
        test = [r for r in rows if r["year"] == fold["test"]]
        m_train = compute_composite_metrics(train, f"IS {fold['train']}")
        m_test = compute_composite_metrics(test, f"OOS {fold['test']}")
        results.append({"fold": fold, "is": m_train, "oos": m_test})
    return results

def leave_one_year_out(rows: list[dict]) -> list[dict]:
    """Leave-one-year-out cross-validation (2022-2025)."""
    results = []
    for leave_out in [2022, 2023, 2024, 2025]:
        train = [r for r in rows if r["year"] != leave_out and r["year"] != EXCLUDE_YEAR]
        test = [r for r in rows if r["year"] == leave_out]
        m_train = compute_composite_metrics(train, f"IS (excl {leave_out})")
        m_test = compute_composite_metrics(test, f"OOS {leave_out}")
        results.append({"leave_out": leave_out, "is": m_train, "oos": m_test})
    return results

# ---------------------------------------------------------------------------
# Step 5: Regime Analysis
# ---------------------------------------------------------------------------
def regime_analysis(rows: list[dict]) -> dict:
    """Year-by-year and ORB size bucket analysis."""
    # Year breakdown
    by_year = {}
    for r in rows:
        y = r["year"]
        by_year.setdefault(y, []).append(r)

    # ORB size buckets
    buckets = {"6-8pt": [], "8-10pt": [], "10+pt": []}
    for r in rows:
        s = r["orb_size"]
        if s >= 10:
            buckets["10+pt"].append(r)
        elif s >= 8:
            buckets["8-10pt"].append(r)
        else:
            buckets["6-8pt"].append(r)

    # % of PnL from 2025-2026
    total_r = sum(r["pnl_r_combined"] for r in rows)
    recent_r = sum(r["pnl_r_combined"] for r in rows if r["year"] in (2025, 2026))
    pct_recent = (recent_r / total_r * 100) if total_r != 0 else 0

    return {
        "by_year": {y: compute_composite_metrics(rs, str(y)) for y, rs in sorted(by_year.items())},
        "by_bucket": {k: compute_composite_metrics(v, k) for k, v in buckets.items()},
        "pct_pnl_2025_2026": pct_recent,
        "total_r": total_r,
        "recent_r": recent_r,
    }

# ---------------------------------------------------------------------------
# Step 6: Direction Split
# ---------------------------------------------------------------------------
def direction_split(rows: list[dict]) -> dict:
    """Split metrics by long vs short."""
    longs = [r for r in rows if r["dir_5m"] == "long"]
    shorts = [r for r in rows if r["dir_5m"] == "short"]
    m_long = compute_composite_metrics(longs, "LONG")
    m_short = compute_composite_metrics(shorts, "SHORT")

    total_r = sum(r["pnl_r_combined"] for r in rows)
    long_r = sum(r["pnl_r_combined"] for r in longs)
    pct_long = (long_r / total_r * 100) if total_r != 0 else 0

    return {"long": m_long, "short": m_short, "pct_long": pct_long}

# ---------------------------------------------------------------------------
# Step 7: Comparison Matrix (4 variants)
# ---------------------------------------------------------------------------
def build_comparison(all_rows: list[dict]) -> dict:
    """Compare 4 strategy variants."""
    # A: Baseline (no agree filter, single RR2.0 contract)
    # Use pnl_r_rr2 only, all rows
    a_rows = [dict(r, pnl_r_combined=r["pnl_r_rr2"]) for r in all_rows]
    m_a = compute_composite_metrics(a_rows, "A: Baseline RR2.0")

    # B: + 15m AGREE only (single RR2.0 contract)
    agree = [r for r in all_rows if r["agree"]]
    b_rows = [dict(r, pnl_r_combined=r["pnl_r_rr2"]) for r in agree]
    m_b = compute_composite_metrics(b_rows, "B: +AGREE RR2.0")

    # C: Split targets only (no agree filter)
    m_c = compute_composite_metrics(all_rows, "C: Split RR1+2")

    # D: Full composite (agree + split)
    m_d = compute_composite_metrics(agree, "D: AGREE+Split")

    return {"A": m_a, "B": m_b, "C": m_c, "D": m_d}

# ---------------------------------------------------------------------------
# Step 8: Report
# ---------------------------------------------------------------------------
def fmt_pct(v, decimals=1):
    return f"{v*100:.{decimals}f}%" if v is not None else "N/A"

def fmt_r(v, decimals=2):
    return f"{v:+.{decimals}f}R" if v is not None else "N/A"

def fmt_f(v, decimals=2):
    return f"{v:.{decimals}f}" if v is not None else "N/A"

def generate_report(
    all_rows, agree_rows, metrics, stress_results, wf_results, loo_results,
    regime, dirs, comparison, be_mult
):
    lines = []

    def w(line=""):
        lines.append(line)

    w("=" * 70)
    w("  1800 COMPOSITE STRATEGY VALIDATION")
    w("  Mandate: Honesty over outcome")
    w("=" * 70)
    w()

    # --- Honesty flags ---
    flags = []
    if metrics["N"] < 50:
        flags.append(f"[!] SMALL SAMPLE: N={metrics['N']} (< 50)")
    if regime["pct_pnl_2025_2026"] > 60:
        flags.append(f"[!] REGIME CONCENTRATION: {regime['pct_pnl_2025_2026']:.0f}% of PnL from 2025-2026")
    # Check OOS degradation from walk-forward
    for wf in wf_results:
        is_exp = wf["is"].get("exp_r")
        oos_exp = wf["oos"].get("exp_r")
        if is_exp and oos_exp and is_exp > 0 and oos_exp < is_exp * 0.5:
            flags.append(f"[!] OOS DEGRADATION: {wf['fold']['test']} OOS ExpR < 50% of IS")
    if abs(dirs["pct_long"]) > 80 or abs(dirs["pct_long"]) < 20:
        dominant = "LONG" if dirs["pct_long"] > 80 else "SHORT"
        flags.append(f"[!] DIRECTION SKEW: {dominant} carries {max(dirs['pct_long'], 100-dirs['pct_long']):.0f}% of edge")
    # Pre-2024 viability
    pre2024_trades = sum(
        1 for r in agree_rows if r["year"] < 2024
    )
    pre2024_years = len(set(r["year"] for r in agree_rows if r["year"] < 2024))
    if pre2024_years > 0 and pre2024_trades / pre2024_years < 5:
        flags.append(f"[!] PRE-2024 VIABILITY: {pre2024_trades/pre2024_years:.1f} trades/year before 2024")
    if be_mult < 1.3:
        flags.append(f"[!] FRAGILE EDGE: breakeven at {be_mult:.2f}x friction (< 1.3x)")

    # Summary
    w("--- SUMMARY ---")
    if not flags:
        w("  No honesty flags triggered. Proceed with caution (all strategies deserve skepticism).")
    else:
        w(f"  {len(flags)} HONESTY FLAG(S):")
        for f in flags:
            w(f"    {f}")
    w()

    go = metrics["N"] >= 30 and be_mult >= 1.3 and metrics["exp_r_per_r"] > 0
    if go and len(flags) <= 2:
        w("  VERDICT: CONDITIONAL GO (REGIME strategy -- not standalone)")
    elif go:
        w("  VERDICT: CAUTION -- edge exists but multiple flags raised")
    else:
        w("  VERDICT: NO-GO -- insufficient evidence for live trading")
    w()

    # --- Raw Metrics ---
    w("--- RAW METRICS (AGREE trades, G6+ filter, excl 2021) ---")
    w(f"  Trades (N):        {metrics['N']}")
    w(f"  WR at RR1.0:       {fmt_pct(metrics['wr_rr1'])}")
    w(f"  WR at RR2.0:       {fmt_pct(metrics['wr_rr2'])}")
    w(f"  Total R (2-lot):   {fmt_r(metrics['total_r'])}")
    w(f"  ExpR per trade:    {fmt_r(metrics['exp_r'])} (risking 2R)")
    w(f"  ExpR per R risked: {fmt_r(metrics['exp_r_per_r'])}")
    w(f"  Sharpe:            {fmt_f(metrics['sharpe'])}")
    w(f"  Max Drawdown:      {fmt_r(metrics['max_dd_r'])}")
    w(f"  Profit Factor:     {fmt_f(metrics['profit_factor'])}")
    w(f"  Classification:    {'CORE' if metrics['N'] >= 100 else 'REGIME' if metrics['N'] >= 30 else 'INVALID'}")
    w()

    # --- Stress Test ---
    w("--- STRESS TEST ---")
    for mult, label in [(1.0, "Base"), (1.5, "1.5x"), (2.0, "2.0x")]:
        sr = stress_results[mult]
        w(f"  {label:5s} friction: N={sr['N']}, ExpR/trade={fmt_r(sr['exp_r'])}, "
          f"ExpR/R={fmt_r(sr['exp_r_per_r'])}, Sharpe={fmt_f(sr['sharpe'])}")
    w(f"  Breakeven friction multiplier: {be_mult:.2f}x")
    w()

    # --- Walk-Forward ---
    w("--- WALK-FORWARD (Expanding Window) ---")
    for wf in wf_results:
        fold = wf["fold"]
        is_m = wf["is"]
        oos_m = wf["oos"]
        w(f"  Train {fold['train']} -> Test {fold['test']}:")
        w(f"    IS:  N={is_m['N']}, ExpR={fmt_r(is_m.get('exp_r'))}, Sharpe={fmt_f(is_m.get('sharpe'))}")
        w(f"    OOS: N={oos_m['N']}, ExpR={fmt_r(oos_m.get('exp_r'))}, Sharpe={fmt_f(oos_m.get('sharpe'))}")
    w()

    w("--- LEAVE-ONE-YEAR-OUT ---")
    for loo in loo_results:
        is_m = loo["is"]
        oos_m = loo["oos"]
        w(f"  Leave out {loo['leave_out']}:")
        w(f"    IS:  N={is_m['N']}, ExpR={fmt_r(is_m.get('exp_r'))}")
        w(f"    OOS: N={oos_m['N']}, ExpR={fmt_r(oos_m.get('exp_r'))}")
    w()

    # --- Regime ---
    w("--- REGIME ANALYSIS ---")
    w(f"  PnL from 2025-2026: {regime['pct_pnl_2025_2026']:.1f}% ({fmt_r(regime['recent_r'])} of {fmt_r(regime['total_r'])})")
    w()
    w("  Year-by-Year:")
    for y, m in regime["by_year"].items():
        w(f"    {y}: N={m['N']}, WR1={fmt_pct(m.get('wr_rr1'))}, WR2={fmt_pct(m.get('wr_rr2'))}, "
          f"Total={fmt_r(m.get('total_r'))}, ExpR={fmt_r(m.get('exp_r'))}")
    w()
    w("  ORB Size Buckets:")
    for bucket, m in regime["by_bucket"].items():
        w(f"    {bucket:8s}: N={m['N']}, ExpR={fmt_r(m.get('exp_r'))}, WR2={fmt_pct(m.get('wr_rr2'))}")
    w()

    # --- Direction ---
    w("--- DIRECTION SPLIT ---")
    w(f"  LONG edge share: {dirs['pct_long']:.1f}%")
    for d_label in ["long", "short"]:
        m = dirs[d_label]
        w(f"  {d_label.upper():6s}: N={m['N']}, WR1={fmt_pct(m.get('wr_rr1'))}, "
          f"WR2={fmt_pct(m.get('wr_rr2'))}, Total={fmt_r(m.get('total_r'))}, "
          f"ExpR={fmt_r(m.get('exp_r'))}")
    w()

    # --- Comparison ---
    w("--- COMPARISON MATRIX ---")
    w(f"  {'Variant':<25s} {'N':>5s} {'ExpR/trade':>12s} {'ExpR/R':>10s} {'Sharpe':>8s} {'MaxDD':>8s} {'PF':>8s}")
    w(f"  {'-'*25} {'-'*5} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for key in ["A", "B", "C", "D"]:
        m = comparison[key]
        w(f"  {m['label']:<25s} {m['N']:>5d} {fmt_r(m.get('exp_r')):>12s} "
          f"{fmt_r(m.get('exp_r_per_r')):>10s} {fmt_f(m.get('sharpe')):>8s} "
          f"{fmt_r(m.get('max_dd_r')):>8s} {fmt_f(m.get('profit_factor')):>8s}")
    w()

    # --- Honest Assessment ---
    w("--- HONEST ASSESSMENT ---")
    w()
    w("  STRENGTHS:")
    if metrics["exp_r_per_r"] and metrics["exp_r_per_r"] > 0.3:
        w(f"    + High per-R expectancy ({fmt_r(metrics['exp_r_per_r'])})")
    if metrics.get("wr_rr1") and metrics["wr_rr1"] > 0.7:
        w(f"    + Cash register leg (RR1.0) has {fmt_pct(metrics['wr_rr1'])} WR")
    if be_mult > 2.0:
        w(f"    + Survives {be_mult:.1f}x friction stress test")
    if metrics.get("profit_factor") and metrics["profit_factor"] > 2.0:
        w(f"    + Profit factor {fmt_f(metrics['profit_factor'])}")
    w()
    w("  WEAKNESSES:")
    if metrics["N"] < 50:
        w(f"    - Only {metrics['N']} trades -- REGIME classification, not CORE")
    if regime["pct_pnl_2025_2026"] > 50:
        w(f"    - {regime['pct_pnl_2025_2026']:.0f}% of PnL concentrated in 2025-2026")
    if pre2024_years > 0 and pre2024_trades / pre2024_years < 5:
        w(f"    - Only {pre2024_trades/pre2024_years:.1f} trades/year before 2024 -- thin history")
    w()
    w("  RISKS:")
    w("    - 15m AGREE is a new filter concept not in the validated grid")
    w("    - Split targets double contract count (execution complexity)")
    w("    - ORB regime shift could eliminate G6+ opportunities")
    w()
    w("  UNKNOWNS:")
    w("    - Fill-bar exit granularity (R1) not yet applied to orb_outcomes")
    w("    - 2016-2020 outcomes not built -- would extend backtest history")
    w("    - Volume filter interaction not tested")
    w()
    w("=" * 70)
    w("  END OF VALIDATION REPORT")
    w("=" * 70)

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data from gold.db...")
    all_rows = load_composite_data(GOLD_DB_PATH)
    print(f"  Loaded {len(all_rows)} rows (G6+, excl 2021, both directions)")

    agree_rows = [r for r in all_rows if r["agree"]]
    print(f"  AGREE rows (5m == 15m direction): {len(agree_rows)}")
    print()

    if not agree_rows:
        print("ERROR: No AGREE trades found. Cannot validate.")
        sys.exit(1)

    # Step 2: Metrics
    metrics = compute_composite_metrics(agree_rows, "Composite AGREE")

    # Step 3: Stress test
    stress_results = {}
    for mult in [1.0, 1.5, 2.0]:
        stressed = stress_test_pnl(agree_rows, mult)
        stress_results[mult] = compute_composite_metrics(stressed, f"{mult}x")
    be_mult = find_breakeven_multiplier(agree_rows)

    # Step 4: Walk-forward
    wf_results = walk_forward_expanding(agree_rows)
    loo_results = leave_one_year_out(agree_rows)

    # Step 5: Regime
    regime = regime_analysis(agree_rows)

    # Step 6: Direction
    dirs = direction_split(agree_rows)

    # Step 7: Comparison
    comparison = build_comparison(all_rows)

    # Step 8: Report
    report = generate_report(
        all_rows, agree_rows, metrics, stress_results,
        wf_results, loo_results, regime, dirs, comparison, be_mult
    )

    print(report)

    # Save to artifacts
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {ARTIFACT_PATH}")

if __name__ == "__main__":
    main()
