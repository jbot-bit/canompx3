#!/usr/bin/env python3
"""
Null Envelope Analysis — White's Reality Check (2000).

Analyzes null test databases from multiple seeds to compute a stable noise
ceiling for the NOISE_EXPR_FLOOR gate in strategy_validator.py.

Modes:
  --analyze DIR     Scan a directory for null_test.db files and compute envelope
  --analyze-glob P  Glob pattern for null test DB files
  --update-config   After analysis, update NOISE_EXPR_FLOOR in trading_app/config.py

Usage:
    # Analyze existing seed DBs in temp directories
    python scripts/tools/null_envelope.py --analyze C:/Users/joshd/AppData/Local/Temp --update-config

    # Analyze specific DB files
    python scripts/tools/null_envelope.py --analyze-glob "C:/tmp/null_test_seed*/null_test.db"

    # Run 10 seeds then analyze (long — hours)
    python scripts/tools/null_envelope.py --run-seeds 10 --update-config
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def find_null_dbs(search_dir: str) -> list[Path]:
    """Find all null_test.db files under search_dir matching our naming pattern."""
    results = []
    for root, _dirs, files in os.walk(search_dir):
        if "null_test.db" in files:
            p = Path(root) / "null_test.db"
            # Verify it's actually a null test DB (has validated_setups table)
            try:
                with duckdb.connect(str(p), read_only=True) as con:
                    con.execute("SELECT 1 FROM validated_setups LIMIT 1")
                results.append(p)
            except Exception:
                continue
    return sorted(results)


def extract_seed_from_path(db_path: Path) -> int | None:
    """Try to extract seed number from directory name like null_test_seed42_xxx."""
    m = re.search(r"seed(\d+)", str(db_path))
    return int(m.group(1)) if m else None


def analyze_db(db_path: Path) -> dict:
    """Extract noise survivor stats from one null test database."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        total = con.execute("SELECT COUNT(*) FROM validated_setups").fetchone()[0]
        if total == 0:
            return {"path": str(db_path), "total": 0, "survivors": []}

        rows = con.execute(
            """SELECT entry_model, expectancy_r, sharpe_ann, sample_size,
                      strategy_id, instrument, orb_label, filter_type, rr_target
               FROM validated_setups
               ORDER BY expectancy_r DESC"""
        ).fetchall()

        survivors = []
        for r in rows:
            survivors.append(
                {
                    "entry_model": r[0],
                    "expectancy_r": r[1],
                    "sharpe_ann": r[2],
                    "sample_size": r[3],
                    "strategy_id": r[4],
                    "instrument": r[5],
                    "orb_label": r[6],
                    "filter_type": r[7],
                    "rr_target": r[8],
                }
            )

        return {
            "path": str(db_path),
            "seed": extract_seed_from_path(db_path),
            "total": total,
            "survivors": survivors,
        }
    finally:
        con.close()


def compute_envelope(all_results: list[dict]) -> dict:
    """Compute the noise envelope across all seeds."""
    all_survivors = []
    for r in all_results:
        all_survivors.extend(r["survivors"])

    if not all_survivors:
        print("  No survivors found across any seed — pipeline rejects all noise!")
        return {"E1": {"floor": 0.05}, "E2": {"floor": 0.05}}

    envelope = {}
    for em in ["E1", "E2"]:
        subs = [s for s in all_survivors if s["entry_model"] == em]
        if not subs:
            envelope[em] = {
                "count": 0,
                "max_expr": 0.0,
                "p95_expr": 0.0,
                "p90_expr": 0.0,
                "median_expr": 0.0,
                "max_sharpe": 0.0,
                "p95_sharpe": 0.0,
                "floor": 0.05,  # conservative default
            }
            continue

        exprs = np.array([s["expectancy_r"] for s in subs])
        sharpes = np.array([s["sharpe_ann"] for s in subs])

        # Floor = P95 of pooled null survivor ExpR, rounded up to nearest 0.01.
        # Matches production logic: strategy_validator uses NOISE_FLOOR_BY_INSTRUMENT
        # (set from P95) to compute noise_risk flag.
        raw_max = float(np.max(exprs))
        p95 = float(np.percentile(exprs, 95))
        floor = np.ceil(p95 * 100) / 100  # round up to 0.01

        envelope[em] = {
            "count": len(subs),
            "max_expr": raw_max,
            "p95_expr": p95,
            "p90_expr": float(np.percentile(exprs, 90)),
            "median_expr": float(np.median(exprs)),
            "max_sharpe": float(np.max(sharpes)),
            "p95_sharpe": float(np.percentile(sharpes, 95)),
            "floor": floor,
        }

    return envelope


def update_config_file(envelope: dict, n_seeds: int) -> None:
    """Update NOISE_EXPR_FLOOR in trading_app/config.py with new values."""
    config_path = PROJECT_ROOT / "trading_app" / "config.py"
    content = config_path.read_text()

    e1_floor = envelope["E1"]["floor"]
    e2_floor = envelope["E2"]["floor"]
    e1_max = envelope["E1"].get("max_expr", 0)
    e2_max = envelope["E2"].get("max_expr", 0)
    e1_count = envelope["E1"].get("count", 0)
    e2_count = envelope["E2"].get("count", 0)

    # Find and replace the NOISE_EXPR_FLOOR block
    pattern = re.compile(
        r"(# ── Noise ExpR floor per entry model.*?)"
        r"(NOISE_EXPR_FLOOR: dict\[str, float\] = \{.*?\})",
        re.DOTALL,
    )

    new_comment = (
        f"# ── Noise ExpR floor per entry model (adversarial validation 2026-03-18) ──\n"
        f"# Null test ({n_seeds} seeds) ran the full pipeline on random-walk bars.\n"
        f"# Noise survivors set a floor: any production strategy below this is\n"
        f"# indistinguishable from chance.\n"
        f"#\n"
        f"# E2 (stop-market) has structural near-breakeven on noise (-0.004R avg),\n"
        f"# so noise easily produces E2 strategies with apparent edge.\n"
        f"# E1 (limit entry) costs real money immediately (-0.118R avg on noise),\n"
        f"# so noise rarely produces E1 survivors.\n"
        f"#\n"
        f"# Floors are P95 of pooled noise survivor ExpR, rounded UP (ceil to 0.01):\n"
        f"#   E2: {e2_count} noise survivors, P95 ExpR → floor {e2_floor} (max={e2_max:.4f})\n"
        f"#   E1: {e1_count} noise survivors, P95 ExpR → floor {e1_floor} (max={e1_max:.4f})\n"
        f"#\n"
        f"# @research-source null_test_{n_seeds}_seeds (White's Reality Check 2026-03-18)\n"
        f"# @entry-models E1, E2\n"
        f"# @revalidated-for E1/E2 event-based sessions (2026-03-18)\n"
    )

    new_dict = f'NOISE_EXPR_FLOOR: dict[str, float] = {{\n    "E1": {e1_floor},\n    "E2": {e2_floor},\n}}'

    new_block = new_comment + new_dict
    new_content = pattern.sub(new_block, content)

    if new_content == content:
        print("  WARNING: Could not find NOISE_EXPR_FLOOR block to update!")
        print(f"  Recommended values: E1={e1_floor}, E2={e2_floor}")
        return

    config_path.write_text(new_content)
    print(f"  Updated config.py: E1={e1_floor}, E2={e2_floor}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Null envelope analysis (White's Reality Check)")
    parser.add_argument("--analyze", type=str, help="Directory to scan for null_test.db files")
    parser.add_argument("--analyze-glob", type=str, help="Glob pattern for null test DB files")
    parser.add_argument("--run-seeds", type=int, help="Run N seeds of null test first (SLOW)")
    parser.add_argument("--start-seed", type=int, default=0, help="First seed (default: 0)")
    parser.add_argument("--apertures", type=int, nargs="+", default=[5, 15, 30], help="Apertures to test")
    parser.add_argument("--update-config", action="store_true", help="Update NOISE_EXPR_FLOOR in config.py")
    args = parser.parse_args()

    db_paths: list[Path] = []

    # Run seeds if requested
    if args.run_seeds:
        print(f"Running {args.run_seeds} seeds of null test (this takes hours)...")
        aperture_args = []
        for a in args.apertures:
            aperture_args.extend(["--apertures", str(a)])

        result = subprocess.run(
            [
                sys.executable,
                "scripts/tests/test_synthetic_null.py",
                "--seeds",
                str(args.run_seeds),
                "--start-seed",
                str(args.start_seed),
                "--keep-db",
                *aperture_args,
            ],
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            print(f"WARNING: Null test exited with code {result.returncode}")
            print("Some seeds may have survivors — that's expected. Proceeding with analysis.")

    # Find DBs
    if args.analyze:
        db_paths = find_null_dbs(args.analyze)
    elif args.analyze_glob:
        db_paths = [Path(p) for p in sorted(glob.glob(args.analyze_glob))]
    else:
        # Default: scan permanent seed dir first, then temp dir
        permanent_dir = PROJECT_ROOT / "scripts" / "tests" / "null_seeds"
        db_paths = find_null_dbs(str(permanent_dir)) if permanent_dir.exists() else []
        if not db_paths:
            tmp = os.environ.get("TEMP", os.environ.get("TMPDIR", "/tmp"))
            db_paths = find_null_dbs(tmp)

    if not db_paths:
        print("No null test databases found!")
        return 1

    print(f"\nFound {len(db_paths)} null test database(s):")
    for p in db_paths:
        seed = extract_seed_from_path(p)
        seed_str = f" (seed {seed})" if seed is not None else ""
        print(f"  {p}{seed_str}")

    # Analyze each
    print(f"\n{'=' * 60}")
    print("ANALYZING NULL TEST RESULTS")
    print(f"{'=' * 60}")

    all_results = []
    for p in db_paths:
        r = analyze_db(p)
        all_results.append(r)
        seed = r.get("seed", "?")
        print(f"\n  Seed {seed}: {r['total']} survivors")
        if r["total"] > 0:
            # Show top 3 by ExpR
            for s in r["survivors"][:3]:
                print(f"    {s['strategy_id']}: ExpR={s['expectancy_r']:.4f}, Sharpe={s['sharpe_ann']:.4f}")

    # Compute envelope
    print(f"\n{'=' * 60}")
    print(f"NOISE ENVELOPE ({len(all_results)} seeds)")
    print(f"{'=' * 60}")

    envelope = compute_envelope(all_results)
    n_seeds = len(all_results)

    for em in ["E1", "E2"]:
        e = envelope[em]
        print(f"\n  {em}:")
        print(f"    Survivors: {e.get('count', 0)}")
        if e.get("count", 0) > 0:
            print(f"    Max ExpR:    {e['max_expr']:.4f}")
            print(f"    P95 ExpR:    {e['p95_expr']:.4f}")
            print(f"    P90 ExpR:    {e['p90_expr']:.4f}")
            print(f"    Median ExpR: {e['median_expr']:.4f}")
            print(f"    Max Sharpe:  {e['max_sharpe']:.4f}")
            print(f"    P95 Sharpe:  {e['p95_sharpe']:.4f}")
        print(f"    FLOOR:     {e['floor']}")

    # Update config if requested
    if args.update_config:
        print(f"\n{'=' * 60}")
        print("UPDATING CONFIG")
        print(f"{'=' * 60}")
        update_config_file(envelope, n_seeds)

    return 0


if __name__ == "__main__":
    sys.exit(main())
