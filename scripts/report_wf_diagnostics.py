"""
Walk-forward diagnostic report from JSONL results.

Reads data/walkforward_results.jsonl and shows per-strategy WF window
breakdown: total/valid/positive/sparse windows, aggregate OOS stats, and
pass/fail status with rejection reason.

Usage:
    python scripts/report_wf_diagnostics.py --instrument MGC
    python scripts/report_wf_diagnostics.py --instrument MGC --failed-only --detail
    python scripts/report_wf_diagnostics.py --instrument MGC --passed-only
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

JSONL_PATH = PROJECT_ROOT / "data" / "walkforward_results.jsonl"


def load_results(jsonl_path, instrument=None):
    """Load JSONL, dedup by strategy_id (latest timestamp wins)."""
    if not jsonl_path.exists():
        return {}

    latest = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec["strategy_id"]
            ts = rec.get("timestamp", "")
            if instrument and rec.get("instrument") != instrument:
                continue
            if sid not in latest or ts > latest[sid].get("timestamp", ""):
                latest[sid] = rec
    return latest


def classify_windows(rec):
    """Count sparse and negative windows from window list."""
    min_trades = rec["params"].get("min_trades_per_window", 15)
    sparse = 0
    negative = 0
    for w in rec.get("windows", []):
        if w["test_n"] < min_trades:
            sparse += 1
        elif not w["test_pass"]:
            negative += 1
    return sparse, negative


def print_summary_table(results, detail=False):
    """Print formatted diagnostic table."""
    if not results:
        print("No results found.")
        return

    recs = sorted(
        results.values(),
        key=lambda r: (-r["n_valid_windows"], -r.get("agg_oos_exp_r", 0)),
    )

    min_win = recs[0]["params"].get("min_valid_windows", 3) if recs else 3
    min_trades = recs[0]["params"].get("min_trades_per_window", 15) if recs else 15

    # Header
    print(f"{'Strategy':<45} {'Tot':>3} {'Val':>3} {'+ve':>3} {'Spr':>3} "
          f"{'OOS_N':>5} {'OOS_ExpR':>8} {'Status':>6}  Reason")
    print("-" * 120)

    pass_count = 0
    fail_count = 0
    fail_sparse = 0
    fail_negative = 0
    fail_other = 0

    for rec in recs:
        sid = rec["strategy_id"]
        total = rec["n_total_windows"]
        valid = rec["n_valid_windows"]
        positive = rec["n_positive_windows"]
        sparse, _ = classify_windows(rec)
        oos_n = rec["total_oos_trades"]
        oos_expr = rec["agg_oos_exp_r"]
        passed = rec["passed"]
        reason = rec.get("rejection_reason") or ""

        status = "PASS" if passed else "FAIL"
        if passed:
            pass_count += 1
        else:
            fail_count += 1
            if "Insufficient valid" in reason:
                fail_sparse += 1
            elif "negative" in reason.lower() or "Negative" in reason:
                fail_negative += 1
            else:
                fail_other += 1

        expr_str = f"{oos_expr:+.4f}" if oos_n > 0 else "   N/A"

        print(f"{sid:<45} {total:>3} {valid:>3} {positive:>3} {sparse:>3} "
              f"{oos_n:>5} {expr_str:>8} {status:>6}  {reason}")

        if detail and rec.get("windows"):
            for w in rec["windows"]:
                wstart = w["window_start"]
                wend = w["window_end"]
                n = w["test_n"]
                expr = w["test_exp_r"]
                wr = w["test_wr"]
                sharpe = w["test_sharpe"]
                if n < min_trades:
                    label = f"SPARSE (< {min_trades})"
                elif w["test_pass"]:
                    label = "VALID"
                else:
                    label = "NEGATIVE"
                expr_s = f"ExpR={expr:+.4f}" if expr is not None else "ExpR=N/A"
                wr_s = f"WR={wr:.0%}" if wr is not None else "WR=N/A"
                sh_s = f"Sharpe={sharpe:.2f}" if sharpe is not None else "Sharpe=N/A"
                print(f"    {wstart}..{wend}:  N={n:<4} {expr_s}  {wr_s}  {sh_s}  {label}")
            print()

    # Footer
    print("-" * 120)
    total_strats = pass_count + fail_count
    print(f"Total: {total_strats} strategies | "
          f"{pass_count} PASSED | {fail_count} FAILED "
          f"(sparse={fail_sparse}, negative={fail_negative}, other={fail_other})")
    print(f"WF params: min_valid_windows={min_win}, "
          f"min_trades_per_window={min_trades}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Walk-forward diagnostic report from JSONL results"
    )
    parser.add_argument("--instrument", default="MGC", help="Instrument (default: MGC)")
    parser.add_argument("--failed-only", action="store_true", help="Show only failed strategies")
    parser.add_argument("--passed-only", action="store_true", help="Show only passed strategies")
    parser.add_argument("--detail", action="store_true", help="Show per-window breakdown")
    parser.add_argument("--jsonl", default=None, help="Path to JSONL file (default: data/walkforward_results.jsonl)")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl) if args.jsonl else JSONL_PATH
    results = load_results(jsonl_path, instrument=args.instrument)

    if args.failed_only:
        results = {k: v for k, v in results.items() if not v["passed"]}
    elif args.passed_only:
        results = {k: v for k, v in results.items() if v["passed"]}

    inst = args.instrument
    min_win = next(iter(results.values()))["params"].get("min_valid_windows", 3) if results else "?"
    print(f"=== WF Diagnostic Report ({inst}, wf_min_windows={min_win}) ===")
    print()

    print_summary_table(results, detail=args.detail)


if __name__ == "__main__":
    main()
