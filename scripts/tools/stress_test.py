"""
Stress-test top edges from the Full Rundown.
Tries to BREAK your best-looking strategies with honesty tests.

Usage:
    python scripts/tools/stress_test.py
    python scripts/tools/stress_test.py --db C:/db/gold.db

Tests run:
  1. Size filter survival   - does the edge exist WITHOUT the ORB size filter?
  2. Year-over-year stability - does it exist every year, or just one hot stretch?
  3. Parameter sensitivity   - change CB and RR by +/-1 / +/-0.5. Does the edge collapse?
  4. Long vs short breakdown - is the edge one-directional only?
  5. Scratch rate check      - are most trades dying before hitting any target?
  6. Honest verdict          - plain-English summary of what survived
"""

import sys
import os
import duckdb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB = PROJECT_ROOT / "gold.db"


def get_db_path():
    for i, arg in enumerate(sys.argv):
        if arg == "--db" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
    env = os.environ.get("DUCKDB_PATH")
    if env:
        return Path(env)
    return DEFAULT_DB


def print_header(title):
    print()
    print(f"  {'=' * 64}")
    print(f"  {title}")
    print(f"  {'=' * 64}")
    print()


def print_table(headers, rows, col_widths=None):
    if not rows:
        print("  No data.")
        return
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(str(h))
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(str(row[i] if row[i] is not None else "")))
            col_widths.append(min(max_w + 2, 22))
    header_str = "  "
    for h, w in zip(headers, col_widths):
        header_str += str(h).ljust(w)
    print(header_str)
    print("  " + "-" * sum(col_widths))
    for row in rows:
        row_str = "  "
        for val, w in zip(row, col_widths):
            row_str += str(val if val is not None else "-").ljust(w)
        print(row_str)


def discover_top_edges(con):
    """Find the top edges per instrument -- N>=50, highest avg R."""
    edges = []
    symbols = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM orb_outcomes ORDER BY symbol"
    ).fetchall()]

    for sym in symbols:
        rows = con.execute("""
            SELECT orb_label, entry_model, confirm_bars, rr_target,
                   COUNT(*) as n,
                   ROUND(AVG(pnl_r), 4) as avg_r,
                   ROUND(AVG(CASE WHEN outcome='win' THEN 100.0 ELSE 0.0 END), 1) as wr
            FROM orb_outcomes
            WHERE symbol = ? AND outcome IN ('win', 'loss')
            GROUP BY orb_label, entry_model, confirm_bars, rr_target
            HAVING COUNT(*) >= 50
            ORDER BY AVG(pnl_r) DESC
            LIMIT 5
        """, [sym]).fetchall()

        for r in rows:
            if r[5] > 0.01:  # only edges with avg R > 0.01
                edges.append({
                    "symbol": sym,
                    "session": r[0],
                    "entry_model": r[1],
                    "cb": r[2],
                    "rr": r[3],
                    "n": r[4],
                    "avg_r": r[5],
                    "wr": r[6],
                })
    return edges


def test_1_no_size_filter(con, edge):
    """Does the edge survive WITHOUT any ORB size filter?

    The full rundown uses raw orb_outcomes which already has no size filter.
    But we compare: what happens on SMALL ORB days vs LARGE ORB days?
    If the edge only exists on large ORB days, it's the size filter doing the work,
    not the parameter combo.
    """
    sym = edge["symbol"]
    sess = edge["session"]
    em = edge["entry_model"]
    cb = edge["cb"]
    rr = edge["rr"]

    # Check if daily_features has the size column for this session
    try:
        size_col = f"orb_{sess}_size"
        # Split into small (<5pt) and large (>=5pt) ORBs
        rows = con.execute(f"""
            SELECT
                CASE WHEN d.{size_col} >= 5.0 THEN 'LARGE (>=5pt)'
                     ELSE 'SMALL (<5pt)' END as orb_bucket,
                COUNT(*) as n,
                ROUND(AVG(CASE WHEN o.outcome='win' THEN 100.0 ELSE 0.0 END), 1) as wr,
                ROUND(AVG(o.pnl_r), 4) as avg_r,
                ROUND(SUM(o.pnl_r), 2) as total_r
            FROM orb_outcomes o
            JOIN daily_features d ON o.trading_day = d.trading_day AND o.symbol = d.symbol
            WHERE o.symbol = ? AND o.orb_label = ? AND o.entry_model = ?
              AND o.confirm_bars = ? AND o.rr_target = ?
              AND o.outcome IN ('win', 'loss')
              AND d.{size_col} IS NOT NULL
            GROUP BY orb_bucket
            ORDER BY orb_bucket
        """, [sym, sess, em, cb, rr]).fetchall()
    except Exception:
        return {"pass": None, "reason": f"Could not find size column for session {sess}"}

    if not rows:
        return {"pass": None, "reason": "No data"}

    result = {r[0]: {"n": r[1], "wr": r[2], "avg_r": r[3], "total_r": r[4]} for r in rows}

    small = result.get("SMALL (<5pt)", {"n": 0, "avg_r": 0, "total_r": 0})
    large = result.get("LARGE (>=5pt)", {"n": 0, "avg_r": 0, "total_r": 0})

    print_table(
        ["ORB Size", "N", "Win %", "Avg R", "Total R"],
        rows
    )

    if large.get("n", 0) == 0:
        return {"pass": None, "reason": "No large-ORB trades found"}

    if small.get("n", 0) < 10:
        return {"pass": True, "reason": f"Only {small.get('n', 0)} small-ORB trades -- edge is mostly large-ORB by nature"}

    # If small ORBs also profitable, the edge isn't just a size filter artifact
    if small.get("avg_r", 0) > 0:
        return {"pass": True, "reason": f"Edge survives on small ORBs too (avg R = {small['avg_r']:+.4f}). Not just a size filter artifact."}
    else:
        # If ONLY large ORBs work, the edge IS the size filter
        return {"pass": False, "reason": f"Small ORBs LOSE (avg R = {small['avg_r']:+.4f}). Edge only works on large ORBs -- it's the size filter doing the work, not the params."}


def test_2_year_over_year(con, edge):
    """Does the edge exist every year, or just one hot period?"""
    sym = edge["symbol"]
    sess = edge["session"]
    em = edge["entry_model"]
    cb = edge["cb"]
    rr = edge["rr"]

    rows = con.execute("""
        SELECT YEAR(trading_day) as yr,
               COUNT(*) as n,
               ROUND(AVG(CASE WHEN outcome='win' THEN 100.0 ELSE 0.0 END), 1) as wr,
               ROUND(AVG(pnl_r), 4) as avg_r,
               ROUND(SUM(pnl_r), 2) as total_r
        FROM orb_outcomes
        WHERE symbol = ? AND orb_label = ? AND entry_model = ?
          AND confirm_bars = ? AND rr_target = ?
          AND outcome IN ('win', 'loss')
        GROUP BY YEAR(trading_day)
        ORDER BY yr
    """, [sym, sess, em, cb, rr]).fetchall()

    print_table(["Year", "N", "Win %", "Avg R", "Total R"], rows)

    if len(rows) < 2:
        return {"pass": False, "reason": f"Only {len(rows)} year(s) of data -- cannot assess stability. INSUFFICIENT DATA."}

    positive_years = sum(1 for r in rows if r[3] > 0)
    negative_years = sum(1 for r in rows if r[3] < -0.01)
    total_years = len(rows)

    # Check if any single year dominates the total R
    total_r = sum(r[4] for r in rows)
    if total_r > 0:
        max_year_r = max(r[4] for r in rows)
        dominance = max_year_r / total_r if total_r != 0 else 0
    else:
        dominance = 0

    if negative_years == 0 and total_years >= 3:
        return {"pass": True, "reason": f"Positive in ALL {total_years} years. Stable."}
    elif negative_years == 0 and total_years == 2:
        return {"pass": True, "reason": f"Positive in both years, but only 2 years -- needs more history to confirm."}
    elif positive_years >= total_years * 0.7:
        bad = [f"{r[0]}({r[3]:+.4f})" for r in rows if r[3] < -0.01]
        return {"pass": True, "reason": f"Positive in {positive_years}/{total_years} years. Losing: {', '.join(bad) or 'minor'}. Acceptable."}
    elif dominance > 0.7 and total_years >= 3:
        best_yr = max(rows, key=lambda r: r[4])
        return {"pass": False, "reason": f"One year ({best_yr[0]}) accounts for {dominance*100:.0f}% of all returns. Regime-dependent, not structural."}
    else:
        bad = [f"{r[0]}({r[3]:+.4f})" for r in rows if r[3] < -0.01]
        return {"pass": False, "reason": f"Only positive in {positive_years}/{total_years} years. Losing: {', '.join(bad)}. Unstable edge."}


def test_3_parameter_sensitivity(con, edge):
    """Change CB by +/-1 and RR by +/-0.5. Does the edge survive?"""
    sym = edge["symbol"]
    sess = edge["session"]
    em = edge["entry_model"]
    cb = edge["cb"]
    rr = edge["rr"]

    # Build neighbor grid: CB +/- 1, RR +/- 0.5
    cb_vals = sorted(set([max(1, cb - 1), cb, cb + 1]))
    rr_vals = sorted(set([max(0.5, rr - 0.5), rr, rr + 0.5]))

    results = []
    for test_cb in cb_vals:
        for test_rr in rr_vals:
            row = con.execute("""
                SELECT COUNT(*) as n,
                       ROUND(AVG(CASE WHEN outcome='win' THEN 100.0 ELSE 0.0 END), 1) as wr,
                       ROUND(AVG(pnl_r), 4) as avg_r
                FROM orb_outcomes
                WHERE symbol = ? AND orb_label = ? AND entry_model = ?
                  AND confirm_bars = ? AND rr_target = ?
                  AND outcome IN ('win', 'loss')
            """, [sym, sess, em, test_cb, test_rr]).fetchone()

            is_orig = (test_cb == cb and abs(test_rr - rr) < 0.01)
            marker = " <-- ORIGINAL" if is_orig else ""
            n = row[0] if row else 0
            wr = row[1] if row and row[0] > 0 else 0
            avg_r = row[2] if row and row[0] > 0 else 0

            results.append((
                f"CB{test_cb}/RR{test_rr:.1f}",
                n,
                f"{wr}%",
                f"{avg_r:+.4f}" if n > 0 else "-",
                marker
            ))

    print_table(["Params", "N", "Win %", "Avg R", ""], results)

    # Assess: how many neighbors are also positive?
    neighbor_count = 0
    positive_neighbors = 0
    original_r = edge["avg_r"]

    for test_cb in cb_vals:
        for test_rr in rr_vals:
            if test_cb == cb and abs(test_rr - rr) < 0.01:
                continue
            # Only count neighbors that actually HAVE data (N >= 20)
            row = con.execute("""
                SELECT COUNT(*) as n, AVG(pnl_r) as avg_r
                FROM orb_outcomes
                WHERE symbol = ? AND orb_label = ? AND entry_model = ?
                  AND confirm_bars = ? AND rr_target = ?
                  AND outcome IN ('win', 'loss')
            """, [sym, sess, em, test_cb, test_rr]).fetchone()
            if row and row[0] and row[0] >= 20:
                neighbor_count += 1
                if row[1] and row[1] > 0:
                    positive_neighbors += 1

    if neighbor_count == 0:
        return {"pass": None, "reason": "No neighbors with data (N>=20) to test. Grid may not have these CB/RR combos."}

    pct = positive_neighbors / neighbor_count * 100
    if positive_neighbors == neighbor_count:
        return {"pass": True, "reason": f"ALL {neighbor_count} neighbors with data are profitable. Robust."}
    elif pct >= 60:
        return {"pass": True, "reason": f"{positive_neighbors}/{neighbor_count} neighbors profitable ({pct:.0f}%). Edge is reasonably robust."}
    elif pct >= 40:
        return {"pass": False, "reason": f"Only {positive_neighbors}/{neighbor_count} neighbors profitable ({pct:.0f}%). Edge is parameter-sensitive -- could be curve-fitted."}
    else:
        return {"pass": False, "reason": f"Only {positive_neighbors}/{neighbor_count} neighbors profitable ({pct:.0f}%). Edge COLLAPSES with small param changes. Likely curve-fitted."}


def test_4_long_vs_short(con, edge):
    """Is the edge one-directional only?"""
    sym = edge["symbol"]
    sess = edge["session"]
    em = edge["entry_model"]
    cb = edge["cb"]
    rr = edge["rr"]

    dir_col = f"orb_{sess}_break_dir"
    try:
        rows = con.execute(f"""
            SELECT
                UPPER(d.{dir_col}) as direction,
                COUNT(*) as n,
                ROUND(AVG(CASE WHEN o.outcome='win' THEN 100.0 ELSE 0.0 END), 1) as wr,
                ROUND(AVG(o.pnl_r), 4) as avg_r,
                ROUND(SUM(o.pnl_r), 2) as total_r
            FROM orb_outcomes o
            JOIN daily_features d ON o.trading_day = d.trading_day AND o.symbol = d.symbol
            WHERE o.symbol = ? AND o.orb_label = ? AND o.entry_model = ?
              AND o.confirm_bars = ? AND o.rr_target = ?
              AND o.outcome IN ('win', 'loss')
              AND d.{dir_col} IS NOT NULL
            GROUP BY direction
            ORDER BY direction
        """, [sym, sess, em, cb, rr]).fetchall()
    except Exception:
        return {"pass": None, "reason": f"Could not find direction column for session {sess}"}

    if not rows:
        return {"pass": None, "reason": "No data"}

    print_table(["Direction", "N", "Win %", "Avg R", "Total R"], rows)

    result = {r[0]: {"n": r[1], "avg_r": r[3]} for r in rows}
    long_r = result.get("LONG", {}).get("avg_r", 0)
    short_r = result.get("SHORT", {}).get("avg_r", 0)
    long_n = result.get("LONG", {}).get("n", 0)
    short_n = result.get("SHORT", {}).get("n", 0)

    both_positive = long_r > 0 and short_r > 0
    only_long = long_r > 0 and short_r <= 0
    only_short = short_r > 0 and long_r <= 0
    neither = long_r <= 0 and short_r <= 0

    if both_positive:
        return {"pass": True, "reason": f"Both directions profitable (L: {long_r:+.4f}, S: {short_r:+.4f}). Bidirectional edge."}
    elif only_long and long_n >= 30:
        return {"pass": True, "reason": f"LONG-ONLY edge (L: {long_r:+.4f} N={long_n}, S: {short_r:+.4f} N={short_n}). Valid -- many sessions have directional bias."}
    elif only_short and short_n >= 30:
        return {"pass": True, "reason": f"SHORT-ONLY edge (S: {short_r:+.4f} N={short_n}, L: {long_r:+.4f} N={long_n}). Valid -- may indicate mean-reversion session."}
    elif neither:
        return {"pass": False, "reason": f"NEITHER direction profitable individually (L: {long_r:+.4f}, S: {short_r:+.4f}). Aggregate avg R may be misleading."}
    else:
        winner = "LONG" if long_r > short_r else "SHORT"
        loser_n = short_n if winner == "LONG" else long_n
        if loser_n < 15:
            return {"pass": True, "reason": f"One direction dominant ({winner}), but losing side has only N={loser_n} -- insufficient to judge."}
        return {"pass": False, "reason": f"One-directional only ({winner}). Other side is a loser. Edge halves your trade count."}


def test_5_scratch_rate(con, edge):
    """Are most trades dying before hitting target?"""
    sym = edge["symbol"]
    sess = edge["session"]
    em = edge["entry_model"]
    cb = edge["cb"]
    rr = edge["rr"]

    rows = con.execute("""
        SELECT outcome,
               COUNT(*) as n,
               ROUND(AVG(pnl_r), 4) as avg_r
        FROM orb_outcomes
        WHERE symbol = ? AND orb_label = ? AND entry_model = ?
          AND confirm_bars = ? AND rr_target = ?
        GROUP BY outcome
        ORDER BY n DESC
    """, [sym, sess, em, cb, rr]).fetchall()

    total = sum(r[1] for r in rows)
    if total == 0:
        return {"pass": None, "reason": "No trades"}

    print_table(["Outcome", "N", "Avg R"], rows)
    print(f"  Total outcomes: {total}")

    outcome_dict = {r[0]: r[1] for r in rows}
    scratch = outcome_dict.get("scratch", 0) + outcome_dict.get("no_break", 0) + outcome_dict.get("expired", 0)
    resolved = outcome_dict.get("win", 0) + outcome_dict.get("loss", 0)

    scratch_pct = scratch / total * 100 if total > 0 else 0

    if resolved == 0:
        return {"pass": False, "reason": "ZERO resolved trades (all scratches/no-breaks). This is not a tradeable edge."}

    if scratch_pct > 60:
        return {"pass": False, "reason": f"{scratch_pct:.0f}% of trades never resolve (scratch/no-break/expired). "
                f"Only {resolved} out of {total} actually hit target or stop. Extremely inefficient."}
    elif scratch_pct > 40:
        return {"pass": True, "reason": f"{scratch_pct:.0f}% unresolved trades. Moderate -- common for higher RR targets. "
                f"{resolved} trades actually resolved."}
    else:
        return {"pass": True, "reason": f"Only {scratch_pct:.0f}% unresolved. {resolved}/{total} trades hit target or stop. Good resolution rate."}


def render_verdict(edge, results):
    """Plain-English verdict."""
    tests = list(results.items())
    passed = sum(1 for _, r in tests if r["pass"] is True)
    failed = sum(1 for _, r in tests if r["pass"] is False)
    unknown = sum(1 for _, r in tests if r["pass"] is None)
    total = len(tests)

    label = f"{edge['symbol']} {edge['session']}/{edge['entry_model']}/CB{edge['cb']}/RR{edge['rr']}"

    print()
    print(f"  +{'=' * 62}+")
    print(f"  | VERDICT: {label:<52}|")
    print(f"  | Avg R: {edge['avg_r']:+.4f}   WR: {edge['wr']}%   N: {edge['n']:<22}|")
    print(f"  +{'-' * 62}+")

    for name, r in tests:
        if r["pass"] is True:
            icon = "PASS"
        elif r["pass"] is False:
            icon = "FAIL"
        else:
            icon = " ?? "
        # Wrap reason to fit
        reason = r["reason"][:56]
        print(f"  | [{icon}] {name:<18} {reason:<37}|")

    print(f"  +{'-' * 62}+")

    if failed == 0 and passed >= 3:
        verdict = "SURVIVED SCRUTINY"
        detail = f"Passed {passed}/{total} tests. This edge looks real."
    elif failed == 0 and passed < 3:
        verdict = "INSUFFICIENT DATA"
        detail = f"Only {passed} tests conclusive. Need more data."
    elif failed == 1 and passed >= 3:
        verdict = "MOSTLY SURVIVED"
        detail = f"Passed {passed}/{total}, failed 1. Worth monitoring."
    elif failed <= 2 and passed >= 2:
        verdict = "MIXED -- CAUTION"
        detail = f"Passed {passed}, failed {failed}. Needs more investigation."
    else:
        verdict = "DID NOT SURVIVE"
        detail = f"Failed {failed}/{total} tests. Likely curve-fitted or regime-dependent."

    print(f"  | {verdict:<62}|")
    print(f"  | {detail:<62}|")
    print(f"  +{'=' * 62}+")
    print()

    return verdict


# ============================================================
# MAIN
# ============================================================

def main():
    db_path = get_db_path()
    if not db_path.exists():
        print(f"\n  Database not found: {db_path}")
        sys.exit(1)

    con = duckdb.connect(str(db_path), read_only=True)

    try:
        print()
        print("  +=========================================================+")
        print("  |              EDGE STRESS TEST                            |")
        print("  |  Trying to honestly break your best strategies           |")
        print("  +=========================================================+")
        print()
        print(f"  DB: {db_path}")
        print()
        print("  Finding top edges (N>=50, avg R > 0.01)...")
        print()

        edges = discover_top_edges(con)

        if not edges:
            print("  No edges found with N>=50 and avg R > 0.01.")
            print("  You may need to rebuild outcomes first.")
            return

        print(f"  Found {len(edges)} candidate edge(s) to stress-test:\n")
        for i, e in enumerate(edges, 1):
            print(f"    {i}. {e['symbol']} {e['session']}/{e['entry_model']}/CB{e['cb']}/RR{e['rr']}"
                  f"  avg R = {e['avg_r']:+.4f}  WR = {e['wr']}%  N = {e['n']}")
        print()
        print("  " + "=" * 64)

        all_verdicts = []

        for i, edge in enumerate(edges, 1):
            label = f"{edge['symbol']} {edge['session']}/{edge['entry_model']}/CB{edge['cb']}/RR{edge['rr']}"
            print()
            print(f"  +===========================================================+")
            print(f"  |  STRESS TESTING [{i}/{len(edges)}]: {label:<40}|")
            print(f"  +===========================================================+")

            results = {}

            # Test 1: Size filter
            print(f"\n  -- TEST 1: ORB Size Filter Survival --")
            results["Size Filter"] = test_1_no_size_filter(con, edge)
            print(f"  -> {results['Size Filter']['reason']}")

            # Test 2: Year-over-year
            print(f"\n  -- TEST 2: Year-Over-Year Stability --")
            results["YoY Stability"] = test_2_year_over_year(con, edge)
            print(f"  -> {results['YoY Stability']['reason']}")

            # Test 3: Parameter sensitivity
            print(f"\n  -- TEST 3: Parameter Sensitivity (CB+/-1, RR+/-0.5) --")
            results["Param Sensitivity"] = test_3_parameter_sensitivity(con, edge)
            print(f"  -> {results['Param Sensitivity']['reason']}")

            # Test 4: Long vs short
            print(f"\n  -- TEST 4: Long vs Short Breakdown --")
            results["Long/Short"] = test_4_long_vs_short(con, edge)
            print(f"  -> {results['Long/Short']['reason']}")

            # Test 5: Scratch rate
            print(f"\n  -- TEST 5: Scratch Rate Check --")
            results["Scratch Rate"] = test_5_scratch_rate(con, edge)
            print(f"  -> {results['Scratch Rate']['reason']}")

            # Verdict
            verdict = render_verdict(edge, results)
            all_verdicts.append((label, verdict))

        # -- FINAL SUMMARY --
        print()
        print("  +=========================================================+")
        print("  |                  FINAL SUMMARY                           |")
        print("  +=========================================================+")
        print()

        survived = [v for v in all_verdicts if v[1] == "SURVIVED SCRUTINY"]
        mostly = [v for v in all_verdicts if v[1] == "MOSTLY SURVIVED"]
        mixed = [v for v in all_verdicts if v[1] == "MIXED -- CAUTION"]
        failed = [v for v in all_verdicts if v[1] == "DID NOT SURVIVE"]
        insufficient = [v for v in all_verdicts if v[1] == "INSUFFICIENT DATA"]

        if survived:
            print("  SURVIVED SCRUTINY (5/5):")
            for label, verdict in survived:
                print(f"    [Y] {label}")

        if mostly:
            print("  MOSTLY SURVIVED (4/5):")
            for label, verdict in mostly:
                print(f"    [~] {label}")

        if mixed:
            print("  MIXED -- CAUTION (3/5 or 2+2):")
            for label, verdict in mixed:
                print(f"    [?] {label}")

        if failed:
            print("  DID NOT SURVIVE:")
            for label, verdict in failed:
                print(f"    [X] {label}")

        if insufficient:
            print("  INSUFFICIENT DATA:")
            for label, verdict in insufficient:
                print(f"    [-] {label}")

        print()
        print(f"  Total edges tested: {len(all_verdicts)}")
        print(f"  Survived: {len(survived)}  |  Mostly: {len(mostly)}  |  Mixed: {len(mixed)}  |  Failed: {len(failed)}  |  Insufficient: {len(insufficient)}")
        print()

        # Honest caveat per RESEARCH_RULES.md
        print("  CAVEATS:")
        print("  " + "-" * 58)
        print("  - This tests raw orb_outcomes (NO size filter applied).")
        print("    Validated strategies with G5+/G6+ filters may differ.")
        print("  - Year-over-year test limited by available outcome data.")
        print("  - Parameter sensitivity uses +/-1 CB / +/-0.5 RR grid only.")
        print("  - All results are IN-SAMPLE unless walk-forward validated.")
        print("  - Per RESEARCH_RULES.md: in-sample positive != confirmed edge.")
        print()

    finally:
        con.close()


if __name__ == "__main__":
    main()
