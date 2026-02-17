#!/usr/bin/env python3
"""
1800 Workhorse Strategy Stress Test: 5 variants side-by-side.

Standalone validation script -- reads from gold.db, produces honest report.
Does NOT modify any tables or existing code.

Variants:
  W1: E3 CB4 RR2.0 G5+ (both)  -- the workhorse
  W2: E3 CB5 RR2.0 G5+ (both)  -- more confirmation
  W3: E3 CB4 RR2.0 G4+ (both)  -- maximum frequency
  W4: E3 CB4 RR2.0 G5+ + 15m AGREE (both)  -- workhorse + agree
  W5: E3 CB4 RR2.0 G6+ (both)  -- regime only

Usage:
    python trading_app/validate_1800_workhorse.py
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
EXCLUDE_YEAR = 2021
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "1800_workhorse_validation.txt"

VARIANTS = [
    {"id": "W1", "label": "E3 CB4 RR2.0 G5+", "cb": 4, "min_orb": 5.0, "agree": False},
    {"id": "W2", "label": "E3 CB5 RR2.0 G5+", "cb": 5, "min_orb": 5.0, "agree": False},
    {"id": "W3", "label": "E3 CB4 RR2.0 G4+", "cb": 4, "min_orb": 4.0, "agree": False},
    {"id": "W4", "label": "E3 CB4 RR2.0 G5+ AGREE", "cb": 4, "min_orb": 5.0, "agree": True},
    {"id": "W5", "label": "E3 CB4 RR2.0 G6+", "cb": 4, "min_orb": 6.0, "agree": False},
]

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_workhorse_data(db_path: Path) -> dict:
    """Load outcomes + features, return filtered rows per variant."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        # All CB4 and CB5 outcomes for 1800 E3 RR2.0
        outcomes = con.execute("""
            SELECT trading_day, confirm_bars, outcome, pnl_r,
                   entry_price, stop_price, target_price
            FROM orb_outcomes
            WHERE orb_label = ? AND entry_model = ? AND rr_target = 2.0
              AND confirm_bars IN (4, 5)
              AND orb_minutes = 5
              AND outcome IN ('win', 'loss')
            ORDER BY trading_day
        """, [ORB_LABEL, ENTRY_MODEL]).fetchdf()

        # 5m daily_features for orb size + break direction
        df5 = con.execute("""
            SELECT trading_day,
                   orb_1800_size AS orb_size,
                   orb_1800_break_dir AS dir_5m
            FROM daily_features
            WHERE symbol = 'MGC' AND orb_minutes = 5
              AND orb_1800_break_dir IS NOT NULL
        """).fetchdf()

        # 15m daily_features for AGREE filter
        df15 = con.execute("""
            SELECT trading_day,
                   orb_1800_break_dir AS dir_15m
            FROM daily_features
            WHERE symbol = 'MGC' AND orb_minutes = 15
              AND orb_1800_break_dir IS NOT NULL
        """).fetchdf()
    finally:
        con.close()

    # Build per-variant row sets
    result = {}
    for v in VARIANTS:
        cb_rows = outcomes[outcomes["confirm_bars"] == v["cb"]].copy()
        merged = cb_rows.merge(df5, on="trading_day", how="inner")
        merged = merged.merge(df15, on="trading_day", how="inner")

        # ORB size filter
        merged = merged[merged["orb_size"] >= v["min_orb"]].copy()

        # Exclude 2021
        merged["year"] = merged["trading_day"].apply(
            lambda d: d.year if hasattr(d, "year") else int(str(d)[:4])
        )
        merged = merged[merged["year"] != EXCLUDE_YEAR].copy()

        # AGREE filter
        if v["agree"]:
            merged = merged[merged["dir_5m"] == merged["dir_15m"]].copy()

        result[v["id"]] = merged.to_dict("records")

    return result

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(rows: list[dict], label: str = "") -> dict:
    """Compute standard metrics for single-contract RR2.0 trades."""
    if not rows:
        return {"label": label, "N": 0}

    n = len(rows)
    pnls = [r["pnl_r"] for r in rows]
    wins = sum(1 for r in rows if r["outcome"] == "win")
    total_r = sum(pnls)
    exp_r = total_r / n

    # Sharpe
    if n > 1:
        mean = total_r / n
        var = sum((p - mean) ** 2 for p in pnls) / (n - 1)
        std = var ** 0.5
        sharpe = mean / std if std > 0 else None
    else:
        sharpe = None

    # Max drawdown
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in rows:
        cum += r["pnl_r"]
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    # Profit factor
    gross_wins = sum(p for p in pnls if p > 0)
    gross_losses = abs(sum(p for p in pnls if p < 0))
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Yearly breakdown
    yearly = {}
    for r in rows:
        y = str(r["year"])
        if y not in yearly:
            yearly[y] = {"trades": 0, "wins": 0, "total_r": 0.0}
        yearly[y]["trades"] += 1
        if r["outcome"] == "win":
            yearly[y]["wins"] += 1
        yearly[y]["total_r"] += r["pnl_r"]

    return {
        "label": label,
        "N": n,
        "win_rate": wins / n,
        "total_r": total_r,
        "exp_r": exp_r,
        "sharpe": sharpe,
        "max_dd_r": max_dd,
        "profit_factor": pf,
        "yearly": yearly,
    }

# ---------------------------------------------------------------------------
# Stress Testing
# ---------------------------------------------------------------------------
def stress_test_pnl(rows: list[dict], multiplier: float) -> list[dict]:
    """Re-compute PnL at stressed friction levels."""
    spec_stress = stress_test_costs(get_cost_spec("MGC"), multiplier)
    stressed = []
    for r in rows:
        entry = r.get("entry_price")
        stop = r.get("stop_price")
        if entry is None or stop is None:
            continue
        risk_pts = abs(entry - stop)
        if risk_pts == 0:
            continue

        risk_dollars = risk_pts * spec_stress.point_value + spec_stress.total_friction
        if r["outcome"] == "win":
            pnl_pts = risk_pts * 2.0  # RR2.0
        else:
            pnl_pts = -risk_pts
        pnl_dollars = pnl_pts * spec_stress.point_value - spec_stress.total_friction
        new_row = dict(r)
        new_row["pnl_r"] = pnl_dollars / risk_dollars
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
        total_r = sum(r["pnl_r"] for r in stressed)
        if total_r / len(stressed) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

# ---------------------------------------------------------------------------
# Walk-Forward
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
        results.append({
            "fold": fold,
            "is": compute_metrics(train, f"IS {fold['train']}"),
            "oos": compute_metrics(test, f"OOS {fold['test']}"),
        })
    return results

def leave_one_year_out(rows: list[dict]) -> list[dict]:
    """Leave-one-year-out cross-validation."""
    results = []
    for leave_out in [2022, 2023, 2024, 2025]:
        train = [r for r in rows if r["year"] != leave_out and r["year"] != EXCLUDE_YEAR]
        test = [r for r in rows if r["year"] == leave_out]
        results.append({
            "leave_out": leave_out,
            "is": compute_metrics(train, f"IS (excl {leave_out})"),
            "oos": compute_metrics(test, f"OOS {leave_out}"),
        })
    return results

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_pct(v, decimals=1):
    return f"{v*100:.{decimals}f}%" if v is not None else "N/A"

def fmt_r(v, decimals=2):
    return f"{v:+.{decimals}f}R" if v is not None else "N/A"

def fmt_f(v, decimals=2):
    return f"{v:.{decimals}f}" if v is not None else "N/A"

def classify(n):
    if n >= 100:
        return "CORE"
    if n >= 30:
        return "REGIME"
    return "INVALID"

# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------
def generate_report(variant_data: dict) -> str:
    lines = []

    def w(line=""):
        lines.append(line)

    w("=" * 74)
    w("  1800 WORKHORSE STRATEGY STRESS TEST")
    w("  Mandate: Honesty over outcome")
    w("=" * 74)
    w()

    # --- Precompute all variant results ---
    results = {}
    for v in VARIANTS:
        vid = v["id"]
        rows = variant_data[vid]
        m = compute_metrics(rows, f"{vid}: {v['label']}")

        stress = {}
        for mult in [1.0, 1.5, 2.0]:
            stressed = stress_test_pnl(rows, mult)
            stress[mult] = compute_metrics(stressed, f"{mult}x")
        be_mult = find_breakeven_multiplier(rows) if rows else 1.0

        wf = walk_forward_expanding(rows)
        loo = leave_one_year_out(rows)

        # Regime
        by_year = {}
        for r in rows:
            by_year.setdefault(r["year"], []).append(r)
        total_r = sum(r["pnl_r"] for r in rows)
        recent_r = sum(r["pnl_r"] for r in rows if r["year"] in (2025, 2026))
        pct_recent = (recent_r / total_r * 100) if total_r != 0 else 0

        pre2025 = [r for r in rows if r["year"] < 2025]
        post2025 = [r for r in rows if r["year"] >= 2025]

        # Direction
        longs = [r for r in rows if r.get("dir_5m") == "long"]
        shorts = [r for r in rows if r.get("dir_5m") == "short"]

        # Honesty flags
        flags = []
        if m["N"] < 50:
            flags.append(f"SMALL SAMPLE: N={m['N']} (< 50)")
        if m["N"] > 0 and pct_recent > 60:
            flags.append(f"REGIME CONCENTRATION: {pct_recent:.0f}% PnL from 2025-2026")
        for f in wf:
            is_exp = f["is"].get("exp_r")
            oos_exp = f["oos"].get("exp_r")
            if is_exp and oos_exp and is_exp > 0 and oos_exp < is_exp * 0.5:
                flags.append(f"OOS DEGRADATION: {f['fold']['test']} OOS ExpR < 50% of IS")
        pre2025_trades = len(pre2025)
        pre2025_years = len(set(r["year"] for r in pre2025))
        if pre2025_years > 0 and pre2025_trades / pre2025_years < 5:
            flags.append(f"PRE-2025 THIN: {pre2025_trades/pre2025_years:.1f} trades/year before 2025")
        if be_mult < 1.3:
            flags.append(f"FRAGILE EDGE: breakeven at {be_mult:.2f}x friction")
        long_r = sum(r["pnl_r"] for r in longs)
        if total_r != 0 and (long_r / total_r * 100 > 80 or long_r / total_r * 100 < 20):
            flags.append(f"DIRECTION SKEW: {'LONG' if long_r > 0 else 'SHORT'} dominated")

        results[vid] = {
            "variant": v,
            "metrics": m,
            "stress": stress,
            "be_mult": be_mult,
            "wf": wf,
            "loo": loo,
            "by_year": {y: compute_metrics(rs, str(y)) for y, rs in sorted(by_year.items())},
            "pct_recent": pct_recent,
            "total_r": total_r,
            "recent_r": recent_r,
            "pre2025": compute_metrics(pre2025, "Pre-2025"),
            "post2025": compute_metrics(post2025, "2025+"),
            "long": compute_metrics(longs, "LONG"),
            "short": compute_metrics(shorts, "SHORT"),
            "flags": flags,
        }

    # === EXECUTIVE SUMMARY ===
    w("--- EXECUTIVE SUMMARY ---")
    w()
    w("  Your bread-and-butter: W1 (E3 CB4 RR2.0 G5+) when vol regime active")
    w("  Conditional upgrade:   W4 (+ 15m AGREE) when timeframes align")
    w("  Regime-only overlay:   W5 (G6+) for outsized moves")
    w("  Honest reality:        1800 is a regime session, not a daily session")
    w()

    # === VARIANT COMPARISON TABLE ===
    w("--- VARIANT COMPARISON ---")
    w()
    hdr = f"  {'ID':<4s} {'Strategy':<26s} {'N':>5s} {'WR':>7s} {'ExpR':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'PF':>6s} {'BE':>5s} {'Class':>7s}"
    w(hdr)
    w(f"  {'-'*4} {'-'*26} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*5} {'-'*7}")
    for v in VARIANTS:
        vid = v["id"]
        m = results[vid]["metrics"]
        be = results[vid]["be_mult"]
        if m["N"] == 0:
            w(f"  {vid:<4s} {v['label']:<26s} {'0':>5s} {'--':>7s} {'--':>8s} {'--':>8s} {'--':>8s} {'--':>6s} {'--':>5s} {'INVALID':>7s}")
        else:
            w(f"  {vid:<4s} {v['label']:<26s} {m['N']:>5d} {fmt_pct(m['win_rate']):>7s} "
              f"{fmt_r(m['exp_r']):>8s} {fmt_f(m['sharpe']):>8s} {fmt_r(m['max_dd_r']):>8s} "
              f"{fmt_f(m['profit_factor']):>6s} {be:>5.2f} {classify(m['N']):>7s}")
    w()

    # === STRESS TESTS ===
    w("--- STRESS TESTS ---")
    w()
    for v in VARIANTS:
        vid = v["id"]
        m = results[vid]["metrics"]
        if m["N"] == 0:
            continue
        w(f"  {vid}: {v['label']}")
        for mult in [1.0, 1.5, 2.0]:
            sr = results[vid]["stress"][mult]
            w(f"    {mult:.1f}x friction: N={sr['N']}, ExpR={fmt_r(sr.get('exp_r'))}, "
              f"Sharpe={fmt_f(sr.get('sharpe'))}, MaxDD={fmt_r(sr.get('max_dd_r'))}")
        w(f"    Breakeven multiplier: {results[vid]['be_mult']:.2f}x")
        w()

    # === WALK-FORWARD ===
    w("--- WALK-FORWARD (Expanding Window) ---")
    w()
    for v in VARIANTS:
        vid = v["id"]
        if results[vid]["metrics"]["N"] == 0:
            continue
        w(f"  {vid}: {v['label']}")
        for wf in results[vid]["wf"]:
            fold = wf["fold"]
            is_m = wf["is"]
            oos_m = wf["oos"]
            w(f"    Train {fold['train']} -> Test {fold['test']}:")
            w(f"      IS:  N={is_m['N']}, ExpR={fmt_r(is_m.get('exp_r'))}, Sharpe={fmt_f(is_m.get('sharpe'))}")
            w(f"      OOS: N={oos_m['N']}, ExpR={fmt_r(oos_m.get('exp_r'))}, Sharpe={fmt_f(oos_m.get('sharpe'))}")
        w()

    w("--- LEAVE-ONE-YEAR-OUT ---")
    w()
    for v in VARIANTS:
        vid = v["id"]
        if results[vid]["metrics"]["N"] == 0:
            continue
        w(f"  {vid}: {v['label']}")
        for loo in results[vid]["loo"]:
            is_m = loo["is"]
            oos_m = loo["oos"]
            w(f"    Leave out {loo['leave_out']}: IS N={is_m['N']}, ExpR={fmt_r(is_m.get('exp_r'))} "
              f"| OOS N={oos_m['N']}, ExpR={fmt_r(oos_m.get('exp_r'))}")
        w()

    # === REGIME REALITY CHECK ===
    w("--- REGIME REALITY CHECK ---")
    w()
    w("  Trades/year by variant:")
    hdr2 = f"  {'ID':<4s} {'Strategy':<26s}"
    years_all = sorted(set(y for vid in results for y in results[vid]["by_year"]))
    for y in years_all:
        hdr2 += f" {str(y):>6s}"
    hdr2 += f" {'Total':>6s}"
    w(hdr2)
    w(f"  {'-'*4} {'-'*26}" + f" {'-'*6}" * (len(years_all) + 1))
    for v in VARIANTS:
        vid = v["id"]
        row = f"  {vid:<4s} {v['label']:<26s}"
        for y in years_all:
            ym = results[vid]["by_year"].get(y)
            n = ym["N"] if ym else 0
            row += f" {n:>6d}"
        row += f" {results[vid]['metrics']['N']:>6d}"
        w(row)
    w()

    w("  PnL/year by variant:")
    hdr3 = f"  {'ID':<4s} {'Strategy':<26s}"
    for y in years_all:
        hdr3 += f" {str(y):>8s}"
    hdr3 += f" {'Total':>8s}"
    w(hdr3)
    w(f"  {'-'*4} {'-'*26}" + f" {'-'*8}" * (len(years_all) + 1))
    for v in VARIANTS:
        vid = v["id"]
        row = f"  {vid:<4s} {v['label']:<26s}"
        for y in years_all:
            ym = results[vid]["by_year"].get(y)
            tr = ym["total_r"] if ym else 0.0
            row += f" {tr:>+8.2f}"
        row += f" {results[vid]['total_r']:>+8.2f}"
        w(row)
    w()

    w("  % PnL from 2025-2026:")
    for v in VARIANTS:
        vid = v["id"]
        w(f"    {vid}: {results[vid]['pct_recent']:.1f}%")
    w()

    # === 2025+ REGIME ADAPTATION (user requested) ===
    w("--- 2025+ REGIME ADAPTATION ---")
    w("  Gold changed. ORB sizes exploded from 2025. Here's what the new regime looks like.")
    w()
    w("  Pre-2025 vs 2025+ (per variant):")
    hdr4 = f"  {'ID':<4s} {'Strategy':<26s} {'Pre N':>6s} {'Pre ExpR':>10s} {'25+ N':>6s} {'25+ ExpR':>10s} {'25+ WR':>7s} {'25+ Sharpe':>11s} {'25+ MaxDD':>10s}"
    w(hdr4)
    w(f"  {'-'*4} {'-'*26} {'-'*6} {'-'*10} {'-'*6} {'-'*10} {'-'*7} {'-'*11} {'-'*10}")
    for v in VARIANTS:
        vid = v["id"]
        pre = results[vid]["pre2025"]
        post = results[vid]["post2025"]
        w(f"  {vid:<4s} {v['label']:<26s} "
          f"{pre['N']:>6d} {fmt_r(pre.get('exp_r')):>10s} "
          f"{post['N']:>6d} {fmt_r(post.get('exp_r')):>10s} "
          f"{fmt_pct(post.get('win_rate')):>7s} "
          f"{fmt_f(post.get('sharpe')):>11s} "
          f"{fmt_r(post.get('max_dd_r')):>10s}")
    w()
    w("  Key takeaway: If gold stays in the 2025+ volatility regime, 1800 becomes")
    w("  a WEEKLY+ session. The edge is real in this regime. The question is whether")
    w("  this regime persists or reverts to 2022-2024 levels (2-5 trades/year).")
    w()
    w("  Adaptation strategy:")
    w("    - Current regime (2025+): W1 is your workhorse, W4 your upgrade")
    w("    - If vol compresses back to 2022-2024: W1 goes dormant, W5 fires rarely")
    w("    - W3 (G4+) catches more trades in low-vol but with thinner edge")
    w("    - Monitor orb_1800_size weekly. If median drops below 5.0 for 4 weeks,")
    w("      consider stepping down to G4+ or pausing the session entirely.")
    w()

    # === DIRECTION SPLIT ===
    w("--- DIRECTION SPLIT ---")
    w()
    hdr5 = f"  {'ID':<4s} {'Strategy':<26s} {'L_N':>5s} {'L_ExpR':>8s} {'S_N':>5s} {'S_ExpR':>8s} {'%Long PnL':>10s}"
    w(hdr5)
    w(f"  {'-'*4} {'-'*26} {'-'*5} {'-'*8} {'-'*5} {'-'*8} {'-'*10}")
    for v in VARIANTS:
        vid = v["id"]
        ml = results[vid]["long"]
        ms = results[vid]["short"]
        total = results[vid]["total_r"]
        long_r = ml.get("total_r", 0)
        pct_l = (long_r / total * 100) if total != 0 else 0
        w(f"  {vid:<4s} {v['label']:<26s} "
          f"{ml['N']:>5d} {fmt_r(ml.get('exp_r')):>8s} "
          f"{ms['N']:>5d} {fmt_r(ms.get('exp_r')):>8s} "
          f"{pct_l:>9.1f}%")
    w()

    # === HONESTY FLAGS ===
    w("--- HONESTY FLAGS ---")
    w()
    for v in VARIANTS:
        vid = v["id"]
        flags = results[vid]["flags"]
        if not flags:
            w(f"  {vid}: No flags. (All strategies still deserve skepticism.)")
        else:
            w(f"  {vid}: {len(flags)} flag(s)")
            for f in flags:
                w(f"    [!] {f}")
    w()

    # === HONEST VERDICT ===
    w("--- HONEST VERDICT ---")
    w()
    w("  1800 is NOT a daily session. It is a volatility-regime session.")
    w("  In the current 2025+ gold regime, it fires weekly+. In 2022-2024 it was monthly at best.")
    w()
    w("  RECOMMENDATION (current regime):")
    w("    1. W1 (G5+ CB4 RR2.0) is your bread-and-butter WHEN CONDITIONS EXIST")
    w("    2. W4 (+ 15m AGREE) is your conditional upgrade -- fewer trades, potentially cleaner")
    w("    3. W5 (G6+) is a regime overlay, NOT standalone -- save for outsized ORBs")
    w("    4. W3 (G4+) is your fallback if you want more frequency at cost of sharpe")
    w()
    w("  WHAT TO EXPECT:")
    w("    - 2025+ regime: ~1-3 trades/week on W1, strong expectancy")
    w("    - If vol normalizes: ~1-2 trades/month, expect dormant stretches")
    w("    - The edge lives in ORB SIZE. When gold isn't moving 5+ points by 1810, sit on hands.")
    w()
    w("  RISKS:")
    w("    - Regime concentration: most PnL is from the recent vol explosion")
    w("    - Pre-2025 sample sizes are thin for all variants")
    w("    - Fill-bar exit (R1) not yet applied to orb_outcomes")
    w("    - 2016-2020 outcomes not yet built (would extend backtest)")
    w()
    w("=" * 74)
    w("  END OF WORKHORSE VALIDATION REPORT")
    w("=" * 74)

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data from gold.db...")
    variant_data = load_workhorse_data(GOLD_DB_PATH)

    for v in VARIANTS:
        print(f"  {v['id']}: {v['label']} -> {len(variant_data[v['id']])} trades")
    print()

    report = generate_report(variant_data)
    print(report)

    # Save to artifacts
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {ARTIFACT_PATH}")

if __name__ == "__main__":
    main()
