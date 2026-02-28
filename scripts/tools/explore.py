"""
Interactive data explorer for the ORB pipeline.
No SQL knowledge required. Just run it and pick from the menu.

Usage:
    python scripts/tools/explore.py
    python scripts/tools/explore.py --db C:/db/gold.db
"""

import sys
import duckdb
from pathlib import Path
from datetime import datetime

# --- Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB = PROJECT_ROOT / "gold.db"

def get_db_path():
    """Get DB path from args or default."""
    import os
    for i, arg in enumerate(sys.argv):
        if arg == "--db" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
    env = os.environ.get("DUCKDB_PATH")
    if env:
        return Path(env)
    return DEFAULT_DB


def clear_screen():
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def pause():
    print()
    input("  Press Enter to go back to menu...")


def print_header(title):
    print()
    print(f"  {'=' * 60}")
    print(f"  {title}")
    print(f"  {'=' * 60}")
    print()


def print_table(headers, rows, col_widths=None):
    """Print a simple formatted table."""
    if not rows:
        print("  No data found.")
        return

    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(str(h))
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(str(row[i] if row[i] is not None else "")))
            col_widths.append(min(max_w + 2, 20))

    # Header
    header_str = "  "
    for h, w in zip(headers, col_widths):
        header_str += str(h).ljust(w)
    print(header_str)
    print("  " + "-" * sum(col_widths))

    # Rows
    for row in rows:
        row_str = "  "
        for val, w in zip(row, col_widths):
            if val is None:
                val = "-"
            row_str += str(val).ljust(w)
        print(row_str)


def pick_instrument(con):
    """Let user pick an instrument from what's in the DB."""
    result = con.execute(
        "SELECT DISTINCT symbol FROM orb_outcomes ORDER BY symbol"
    ).fetchall()
    symbols = [r[0] for r in result]

    if not symbols:
        # Fall back to bars_1m
        result = con.execute(
            "SELECT DISTINCT symbol FROM bars_1m ORDER BY symbol"
        ).fetchall()
        symbols = [r[0] for r in result]

    if not symbols:
        print("  No data found in database.")
        return None

    if len(symbols) == 1:
        return symbols[0]

    print()
    print("  Pick instrument:")
    for i, s in enumerate(symbols, 1):
        print(f"    {i}. {s}")
    print()

    while True:
        choice = input("  Enter number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(symbols):
            return symbols[int(choice) - 1]
        print("  Invalid choice, try again.")


def pick_session(con, instrument):
    """Let user pick a session from what's in the DB."""
    result = con.execute(
        "SELECT DISTINCT orb_label FROM orb_outcomes WHERE symbol = ? ORDER BY orb_label",
        [instrument]
    ).fetchall()
    sessions = [r[0] for r in result]

    if not sessions:
        print("  No sessions found.")
        return None

    print()
    print("  Pick session:")
    print(f"    0. ALL sessions")
    for i, s in enumerate(sessions, 1):
        print(f"    {i}. {s}")
    print()

    while True:
        choice = input("  Enter number: ").strip()
        if choice == "0":
            return None  # means ALL
        if choice.isdigit() and 1 <= int(choice) <= len(sessions):
            return sessions[int(choice) - 1]
        print("  Invalid choice, try again.")


# ============================================================
# MENU OPTIONS
# ============================================================

def overview(con):
    """What data do I have?"""
    print_header("DATA OVERVIEW")

    # Bars
    print("  1-MINUTE BARS:")
    rows = con.execute("""
        SELECT symbol,
               FORMAT('{:,}', COUNT(*)) as bars,
               MIN(ts_utc)::DATE as first_date,
               MAX(ts_utc)::DATE as last_date
        FROM bars_1m GROUP BY symbol ORDER BY symbol
    """).fetchall()
    print_table(["Symbol", "Bars", "First", "Last"], rows)

    print()

    # Outcomes
    print("  OUTCOMES:")
    rows = con.execute("""
        SELECT symbol,
               FORMAT('{:,}', COUNT(*)) as outcomes,
               MIN(trading_day) as first_day,
               MAX(trading_day) as last_day,
               FORMAT('{:,}', COUNT(DISTINCT trading_day)) as unique_days
        FROM orb_outcomes GROUP BY symbol ORDER BY symbol
    """).fetchall()
    print_table(["Symbol", "Outcomes", "First", "Last", "Days"], rows)

    print()

    # Strategies
    print("  VALIDATED STRATEGIES:")
    try:
        rows = con.execute("""
            SELECT instrument, COUNT(*) as strategies,
                   SUM(CASE WHEN classification = 'CORE' THEN 1 ELSE 0 END) as core,
                   SUM(CASE WHEN classification = 'REGIME' THEN 1 ELSE 0 END) as regime
            FROM validated_setups GROUP BY instrument ORDER BY instrument
        """).fetchall()
        print_table(["Symbol", "Total", "CORE", "REGIME"], rows)
    except Exception:
        print("  No validated strategies yet.")

    pause()


def win_rate_by_session(con):
    """Win rate breakdown by session."""
    print_header("WIN RATE BY SESSION")
    instrument = pick_instrument(con)
    if not instrument:
        return

    rows = con.execute("""
        SELECT orb_label as session,
               COUNT(*) as trades,
               ROUND(AVG(CASE WHEN outcome='win' THEN 100.0 ELSE 0.0 END), 1) as win_rate,
               ROUND(AVG(pnl_r), 4) as avg_pnl_r,
               ROUND(SUM(pnl_r), 1) as total_r
        FROM orb_outcomes
        WHERE symbol = ? AND outcome IN ('win', 'loss')
        GROUP BY orb_label
        ORDER BY orb_label
    """, [instrument]).fetchall()

    print(f"  Instrument: {instrument}")
    print(f"  (All RR targets and confirm bars combined)")
    print()
    print_table(["Session", "Trades", "Win %", "Avg R", "Total R"], rows)
    pause()


def best_rr_target(con):
    """Which RR target performs best?"""
    print_header("PERFORMANCE BY RR TARGET")
    instrument = pick_instrument(con)
    if not instrument:
        return

    session = pick_session(con, instrument)

    where = "symbol = ? AND outcome IN ('win', 'loss')"
    params = [instrument]
    if session:
        where += " AND orb_label = ?"
        params.append(session)

    rows = con.execute(f"""
        SELECT rr_target as RR,
               COUNT(*) as trades,
               ROUND(AVG(CASE WHEN outcome='win' THEN 100.0 ELSE 0.0 END), 1) as win_rate,
               ROUND(AVG(pnl_r), 4) as avg_pnl_r,
               ROUND(SUM(pnl_r), 1) as total_r
        FROM orb_outcomes
        WHERE {where}
        GROUP BY rr_target
        ORDER BY rr_target
    """, params).fetchall()

    label = f"{instrument} / {session or 'ALL sessions'}"
    print(f"  {label}")
    print()
    print_table(["RR", "Trades", "Win %", "Avg R", "Total R"], rows)
    pause()


def entry_model_comparison(con):
    """E1 vs E3 comparison."""
    print_header("ENTRY MODEL COMPARISON (E1 vs E3)")
    instrument = pick_instrument(con)
    if not instrument:
        return

    session = pick_session(con, instrument)

    where = "symbol = ? AND outcome IN ('win', 'loss')"
    params = [instrument]
    if session:
        where += " AND orb_label = ?"
        params.append(session)

    rows = con.execute(f"""
        SELECT entry_model as model,
               COUNT(*) as trades,
               ROUND(AVG(CASE WHEN outcome='win' THEN 100.0 ELSE 0.0 END), 1) as win_rate,
               ROUND(AVG(pnl_r), 4) as avg_pnl_r,
               ROUND(SUM(pnl_r), 1) as total_r
        FROM orb_outcomes
        WHERE {where}
        GROUP BY entry_model
        ORDER BY entry_model
    """, params).fetchall()

    label = f"{instrument} / {session or 'ALL sessions'}"
    print(f"  {label}")
    print()
    print_table(["Model", "Trades", "Win %", "Avg R", "Total R"], rows)
    pause()


def confirm_bars_comparison(con):
    """Which confirm bars setting works best?"""
    print_header("CONFIRM BARS COMPARISON")
    instrument = pick_instrument(con)
    if not instrument:
        return

    session = pick_session(con, instrument)

    where = "symbol = ? AND outcome IN ('win', 'loss')"
    params = [instrument]
    if session:
        where += " AND orb_label = ?"
        params.append(session)

    rows = con.execute(f"""
        SELECT confirm_bars as CB,
               COUNT(*) as trades,
               ROUND(AVG(CASE WHEN outcome='win' THEN 100.0 ELSE 0.0 END), 1) as win_rate,
               ROUND(AVG(pnl_r), 4) as avg_pnl_r,
               ROUND(SUM(pnl_r), 1) as total_r
        FROM orb_outcomes
        WHERE {where}
        GROUP BY confirm_bars
        ORDER BY confirm_bars
    """, params).fetchall()

    label = f"{instrument} / {session or 'ALL sessions'}"
    print(f"  {label}")
    print()
    print_table(["CB", "Trades", "Win %", "Avg R", "Total R"], rows)
    pause()


def yearly_breakdown(con):
    """Performance by year — see if edge is stable or regime-dependent."""
    print_header("YEARLY BREAKDOWN")
    instrument = pick_instrument(con)
    if not instrument:
        return

    session = pick_session(con, instrument)

    where = "symbol = ? AND outcome IN ('win', 'loss')"
    params = [instrument]
    if session:
        where += " AND orb_label = ?"
        params.append(session)

    rows = con.execute(f"""
        SELECT YEAR(trading_day) as year,
               COUNT(*) as trades,
               ROUND(AVG(CASE WHEN outcome='win' THEN 100.0 ELSE 0.0 END), 1) as win_rate,
               ROUND(AVG(pnl_r), 4) as avg_pnl_r,
               ROUND(SUM(pnl_r), 1) as total_r
        FROM orb_outcomes
        WHERE {where}
        GROUP BY YEAR(trading_day)
        ORDER BY YEAR(trading_day)
    """, params).fetchall()

    label = f"{instrument} / {session or 'ALL sessions'}"
    print(f"  {label}")
    print()
    print_table(["Year", "Trades", "Win %", "Avg R", "Total R"], rows)
    pause()


def long_vs_short(con):
    """Long vs short performance."""
    print_header("LONG vs SHORT")
    instrument = pick_instrument(con)
    if not instrument:
        return

    session = pick_session(con, instrument)

    where = "o.symbol = ? AND o.outcome IN ('win', 'loss')"
    params = [instrument]
    if session:
        where += " AND o.orb_label = ?"
        params.append(session)

    # Join with daily_features to get break_dir
    rows = con.execute(f"""
        SELECT
            CASE
                WHEN d.orb_{session or '0900'}_break_dir = 'long' THEN 'LONG'
                WHEN d.orb_{session or '0900'}_break_dir = 'short' THEN 'SHORT'
                ELSE 'UNKNOWN'
            END as direction,
            COUNT(*) as trades,
            ROUND(AVG(CASE WHEN o.outcome='win' THEN 100.0 ELSE 0.0 END), 1) as win_rate,
            ROUND(AVG(o.pnl_r), 4) as avg_pnl_r
        FROM orb_outcomes o
        JOIN daily_features d ON o.trading_day = d.trading_day AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
        WHERE {where}
        GROUP BY direction
        ORDER BY direction
    """, params).fetchall()

    label = f"{instrument} / {session or '0900'}"
    print(f"  {label}")
    print()
    print_table(["Direction", "Trades", "Win %", "Avg R"], rows)
    pause()


def outcome_distribution(con):
    """Win / Loss / Scratch / Early Exit breakdown."""
    print_header("OUTCOME DISTRIBUTION")
    instrument = pick_instrument(con)
    if not instrument:
        return

    session = pick_session(con, instrument)

    where = "symbol = ?"
    params = [instrument]
    if session:
        where += " AND orb_label = ?"
        params.append(session)

    total = con.execute(f"""
        SELECT COUNT(*) FROM orb_outcomes WHERE {where}
    """, params).fetchone()[0]

    rows = con.execute(f"""
        SELECT outcome,
               COUNT(*) as count,
               ROUND(COUNT(*) * 100.0 / {total}, 1) as pct,
               ROUND(AVG(pnl_r), 4) as avg_pnl_r
        FROM orb_outcomes
        WHERE {where}
        GROUP BY outcome
        ORDER BY count DESC
    """, params).fetchall()

    label = f"{instrument} / {session or 'ALL sessions'}"
    print(f"  {label} ({total:,} total outcomes)")
    print()
    print_table(["Outcome", "Count", "% of Total", "Avg R"], rows)
    pause()


def best_combos(con):
    """Top 10 best performing parameter combinations."""
    print_header("TOP 10 PARAMETER COMBOS (by Avg R)")
    instrument = pick_instrument(con)
    if not instrument:
        return

    session = pick_session(con, instrument)

    where = "symbol = ? AND outcome IN ('win', 'loss')"
    params = [instrument]
    if session:
        where += " AND orb_label = ?"
        params.append(session)

    rows = con.execute(f"""
        SELECT orb_label as sess,
               entry_model as EM,
               confirm_bars as CB,
               rr_target as RR,
               COUNT(*) as N,
               ROUND(AVG(CASE WHEN outcome='win' THEN 100.0 ELSE 0.0 END), 1) as wr,
               ROUND(AVG(pnl_r), 4) as avg_r
        FROM orb_outcomes
        WHERE {where}
        GROUP BY orb_label, entry_model, confirm_bars, rr_target
        HAVING COUNT(*) >= 30
        ORDER BY AVG(pnl_r) DESC
        LIMIT 10
    """, params).fetchall()

    label = f"{instrument} / {session or 'ALL sessions'} (min 30 trades)"
    print(f"  {label}")
    print()
    print_table(["Session", "EM", "CB", "RR", "N", "Win %", "Avg R"], rows)
    pause()


def worst_combos(con):
    """Bottom 10 worst performing parameter combinations."""
    print_header("BOTTOM 10 PARAMETER COMBOS (by Avg R)")
    instrument = pick_instrument(con)
    if not instrument:
        return

    session = pick_session(con, instrument)

    where = "symbol = ? AND outcome IN ('win', 'loss')"
    params = [instrument]
    if session:
        where += " AND orb_label = ?"
        params.append(session)

    rows = con.execute(f"""
        SELECT orb_label as sess,
               entry_model as EM,
               confirm_bars as CB,
               rr_target as RR,
               COUNT(*) as N,
               ROUND(AVG(CASE WHEN outcome='win' THEN 100.0 ELSE 0.0 END), 1) as wr,
               ROUND(AVG(pnl_r), 4) as avg_r
        FROM orb_outcomes
        WHERE {where}
        GROUP BY orb_label, entry_model, confirm_bars, rr_target
        HAVING COUNT(*) >= 30
        ORDER BY AVG(pnl_r) ASC
        LIMIT 10
    """, params).fetchall()

    label = f"{instrument} / {session or 'ALL sessions'} (min 30 trades)"
    print(f"  {label}")
    print()
    print_table(["Session", "EM", "CB", "RR", "N", "Win %", "Avg R"], rows)
    pause()


def full_rundown(con):
    """The big one — runs everything, gives you a plain-English report with recommendations."""
    print_header("FULL RUNDOWN")
    print("  Scanning your entire database. Sit tight...\n")

    issues = []
    good = []
    actions = []

    # ── 1. DATA COVERAGE ──
    print("  [1/7] Checking data coverage...")
    bars = con.execute("""
        SELECT symbol, COUNT(*) as n,
               MIN(ts_utc)::DATE as first, MAX(ts_utc)::DATE as last,
               COUNT(DISTINCT ts_utc::DATE) as days
        FROM bars_1m GROUP BY symbol ORDER BY symbol
    """).fetchall()
    outcomes = con.execute("""
        SELECT symbol, COUNT(*) as n,
               MIN(trading_day) as first, MAX(trading_day) as last,
               COUNT(DISTINCT trading_day) as days
        FROM orb_outcomes GROUP BY symbol ORDER BY symbol
    """).fetchall()

    bars_dict = {r[0]: {"n": r[1], "first": r[2], "last": r[3], "days": r[4]} for r in bars}
    out_dict = {r[0]: {"n": r[1], "first": r[2], "last": r[3], "days": r[4]} for r in outcomes}

    for sym, bd in bars_dict.items():
        od = out_dict.get(sym)
        if not od:
            issues.append(f"  !! {sym}: Has {bd['days']} days of bars but ZERO outcomes. Needs outcome rebuild.")
            actions.append(f"Run outcome builder for {sym}")
        elif od["days"] < bd["days"] * 0.5:
            gap_pct = round((1 - od["days"] / bd["days"]) * 100)
            issues.append(f"  !! {sym}: Outcomes cover only {od['days']}/{bd['days']} bar days ({gap_pct}% gap). Rebuild needed.")
            actions.append(f"Rebuild outcomes for {sym} (missing {gap_pct}% of available data)")
        else:
            good.append(f"  OK {sym}: {od['days']} days of outcomes, {bd['days']} days of bars")

    for sym in out_dict:
        if sym not in bars_dict:
            issues.append(f"  ?? {sym}: Has outcomes but no bars — possible orphaned data")

    # ── 2. WIN RATE BY SESSION ──
    print("  [2/7] Analyzing win rates by session...")
    for sym in out_dict:
        sess_rows = con.execute("""
            SELECT orb_label,
                   COUNT(*) as n,
                   ROUND(AVG(CASE WHEN outcome='win' THEN 100.0 ELSE 0.0 END), 1) as wr,
                   ROUND(AVG(pnl_r), 4) as avg_r
            FROM orb_outcomes
            WHERE symbol = ? AND outcome IN ('win', 'loss')
            GROUP BY orb_label ORDER BY orb_label
        """, [sym]).fetchall()

        best_sess = max(sess_rows, key=lambda r: r[3]) if sess_rows else None
        worst_sess = min(sess_rows, key=lambda r: r[3]) if sess_rows else None

        if best_sess and best_sess[3] > 0:
            good.append(f"  OK {sym} best session: {best_sess[0]} (avg R = {best_sess[3]:+.4f}, WR = {best_sess[2]}%, N = {best_sess[1]})")
        if worst_sess and worst_sess[3] < -0.05:
            issues.append(f"  !! {sym} worst session: {worst_sess[0]} (avg R = {worst_sess[3]:+.4f}, WR = {worst_sess[2]}%, N = {worst_sess[1]}) — consistently losing money")

    # ── 3. BEST RR TARGETS ──
    print("  [3/7] Finding best RR targets...")
    for sym in out_dict:
        rr_rows = con.execute("""
            SELECT rr_target,
                   ROUND(AVG(pnl_r), 4) as avg_r,
                   COUNT(*) as n
            FROM orb_outcomes
            WHERE symbol = ? AND outcome IN ('win', 'loss')
            GROUP BY rr_target ORDER BY rr_target
        """, [sym]).fetchall()

        best_rr = max(rr_rows, key=lambda r: r[1]) if rr_rows else None
        worst_rr = min(rr_rows, key=lambda r: r[1]) if rr_rows else None

        if best_rr:
            good.append(f"  OK {sym} best RR: {best_rr[0]} (avg R = {best_rr[1]:+.4f})")
        if worst_rr and worst_rr[1] < -0.05:
            issues.append(f"  !! {sym} RR {worst_rr[0]} is a consistent loser (avg R = {worst_rr[1]:+.4f})")

    # ── 4. E1 vs E3 ──
    print("  [4/7] Comparing entry models...")
    for sym in out_dict:
        em_rows = con.execute("""
            SELECT entry_model,
                   ROUND(AVG(pnl_r), 4) as avg_r,
                   COUNT(*) as n
            FROM orb_outcomes
            WHERE symbol = ? AND outcome IN ('win', 'loss')
            GROUP BY entry_model ORDER BY entry_model
        """, [sym]).fetchall()

        if len(em_rows) >= 2:
            e1 = next((r for r in em_rows if r[0] == "E1"), None)
            e3 = next((r for r in em_rows if r[0] == "E3"), None)
            if e1 and e3:
                if e1[1] > e3[1] + 0.01:
                    good.append(f"  OK {sym}: E1 outperforms E3 ({e1[1]:+.4f} vs {e3[1]:+.4f})")
                elif e3[1] > e1[1] + 0.01:
                    good.append(f"  OK {sym}: E3 outperforms E1 ({e3[1]:+.4f} vs {e1[1]:+.4f})")
                else:
                    good.append(f"  OK {sym}: E1 and E3 roughly equal ({e1[1]:+.4f} vs {e3[1]:+.4f})")

    # ── 5. YEAR-OVER-YEAR STABILITY ──
    print("  [5/7] Checking year-over-year stability...")
    for sym in out_dict:
        yr_rows = con.execute("""
            SELECT YEAR(trading_day) as yr,
                   ROUND(AVG(pnl_r), 4) as avg_r,
                   COUNT(*) as n
            FROM orb_outcomes
            WHERE symbol = ? AND outcome IN ('win', 'loss')
            GROUP BY YEAR(trading_day) ORDER BY yr
        """, [sym]).fetchall()

        if len(yr_rows) >= 2:
            rs = [r[1] for r in yr_rows]
            if max(rs) > 0 and min(rs) < -0.02:
                bad_yrs = [f"{r[0]}({r[1]:+.4f})" for r in yr_rows if r[1] < -0.02]
                issues.append(f"  !! {sym}: Unstable year-over-year. Losing years: {', '.join(bad_yrs)}")
            elif all(r > -0.01 for r in rs):
                good.append(f"  OK {sym}: Stable across all years ({len(yr_rows)} years of data)")
        elif len(yr_rows) == 1:
            issues.append(f"  !! {sym}: Only 1 year of outcome data — can't assess stability. Need more data.")
            actions.append(f"Rebuild {sym} outcomes to cover full bar history")

    # ── 6. OUTCOME MIX ──
    print("  [6/7] Checking outcome distribution...")
    for sym in out_dict:
        mix = con.execute("""
            SELECT outcome, COUNT(*) as n
            FROM orb_outcomes WHERE symbol = ?
            GROUP BY outcome ORDER BY n DESC
        """, [sym]).fetchall()
        mix_dict = {r[0]: r[1] for r in mix}
        total = sum(mix_dict.values())
        scratch_pct = round(mix_dict.get("scratch", 0) / total * 100, 1) if total > 0 else 0
        if scratch_pct > 40:
            issues.append(f"  !! {sym}: {scratch_pct}% of outcomes are scratches (no target/stop hit). Lots of dead trades.")
        elif scratch_pct > 25:
            good.append(f"  OK {sym}: {scratch_pct}% scratches (moderate — normal for wider RR targets)")
        else:
            good.append(f"  OK {sym}: Only {scratch_pct}% scratches (good resolution rate)")

    # ── 7. TOP COMBOS CHECK ──
    print("  [7/7] Finding strongest edges...")
    for sym in out_dict:
        top = con.execute("""
            SELECT orb_label, entry_model, confirm_bars, rr_target,
                   COUNT(*) as n,
                   ROUND(AVG(pnl_r), 4) as avg_r
            FROM orb_outcomes
            WHERE symbol = ? AND outcome IN ('win', 'loss')
            GROUP BY orb_label, entry_model, confirm_bars, rr_target
            HAVING COUNT(*) >= 50
            ORDER BY AVG(pnl_r) DESC
            LIMIT 3
        """, [sym]).fetchall()

        if top and top[0][5] > 0.02:
            for t in top:
                good.append(f"  ** {sym} EDGE: {t[0]}/E{t[1][-1]}/CB{t[2]}/RR{t[3]} — avg R = {t[5]:+.4f}, N = {t[4]}")
        elif top:
            issues.append(f"  !! {sym}: No parameter combo with avg R > 0.02 and N >= 50. Edge may be thin.")

    # ── HEARTBEAT CHECK ──
    db_path = get_db_path()
    heartbeat = db_path.parent / "outcome_builder.heartbeat"
    if heartbeat.exists():
        content = heartbeat.read_text().strip()
        try:
            hb_time = datetime.fromisoformat(content.split(" | ")[0])
            age_min = (datetime.now() - hb_time).total_seconds() / 60
            if age_min < 20:
                good.append(f"  OK Outcome builder is running (last heartbeat {age_min:.0f} min ago)")
            else:
                issues.append(f"  !! Outcome builder heartbeat is {age_min:.0f} min old — process may be dead")
                actions.append("Check if outcome builder is still running, restart if dead")
        except Exception:
            pass

    # ══════════════════════════════════════════
    # PRINT THE REPORT
    # ══════════════════════════════════════════
    clear_screen()
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║            FULL RUNDOWN REPORT               ║")
    print("  ╚══════════════════════════════════════════════╝")

    print()
    print("  WHAT'S WORKING:")
    print("  " + "-" * 50)
    for g in good:
        print(g)

    if issues:
        print()
        print("  ISSUES FOUND:")
        print("  " + "-" * 50)
        for i in issues:
            print(i)

    if actions:
        print()
        print("  WHAT TO DO NEXT:")
        print("  " + "-" * 50)
        for idx, a in enumerate(actions, 1):
            print(f"  {idx}. {a}")

    if not issues and not actions:
        print()
        print("  Everything looks good. No issues found.")

    # Quick stats
    print()
    print("  QUICK STATS:")
    print("  " + "-" * 50)
    for sym in sorted(set(list(bars_dict.keys()) + list(out_dict.keys()))):
        bd = bars_dict.get(sym, {})
        od = out_dict.get(sym, {})
        print(f"  {sym}: {bd.get('days', 0)} bar days, {od.get('days', 0)} outcome days, "
              f"{od.get('first', '?')} to {od.get('last', '?')}")

    pause()


def custom_query(con):
    """Run your own SQL (for when you learn some)."""
    print_header("CUSTOM SQL QUERY")
    print("  Type your SQL query (or 'back' to return):")
    print("  Hint: SHOW TABLES  to see all tables")
    print("  Hint: DESCRIBE table_name  to see columns")
    print()

    while True:
        query = input("  SQL> ").strip()
        if query.lower() in ('back', 'exit', 'quit', 'q'):
            return
        if not query:
            continue
        try:
            result = con.execute(query)
            if result.description:
                headers = [d[0] for d in result.description]
                rows = result.fetchall()
                print()
                print_table(headers, rows[:50])
                if len(rows) > 50:
                    print(f"\n  ... showing 50 of {len(rows)} rows")
            else:
                print("  Done.")
        except Exception as e:
            print(f"  Error: {e}")
        print()


def rebuild_status(con):
    """Check outcome rebuild progress."""
    print_header("OUTCOME REBUILD STATUS")

    # What outcomes exist
    print("  CURRENT OUTCOMES:")
    rows = con.execute("""
        SELECT symbol,
               FORMAT('{:,}', COUNT(*)) as outcomes,
               MIN(trading_day) as first_day,
               MAX(trading_day) as last_day,
               FORMAT('{:,}', COUNT(DISTINCT trading_day)) as days
        FROM orb_outcomes GROUP BY symbol ORDER BY symbol
    """).fetchall()
    print_table(["Symbol", "Outcomes", "First Day", "Last Day", "Days"], rows)

    print()

    # What bars exist (potential coverage)
    print("  AVAILABLE BARS (potential coverage):")
    rows = con.execute("""
        SELECT symbol,
               MIN(ts_utc)::DATE as first_bar,
               MAX(ts_utc)::DATE as last_bar,
               FORMAT('{:,}', COUNT(DISTINCT ts_utc::DATE)) as bar_days
        FROM bars_1m GROUP BY symbol ORDER BY symbol
    """).fetchall()
    print_table(["Symbol", "First Bar", "Last Bar", "Bar Days"], rows)

    print()

    # Gap analysis
    print("  GAP ANALYSIS (bars exist but no outcomes):")
    rows = con.execute("""
        SELECT b.symbol,
               MIN(b.day) as gap_start,
               MAX(b.day) as gap_end,
               COUNT(DISTINCT b.day) as missing_days
        FROM (SELECT DISTINCT symbol, ts_utc::DATE as day FROM bars_1m) b
        LEFT JOIN (SELECT DISTINCT symbol, trading_day FROM orb_outcomes) o
            ON b.symbol = o.symbol AND b.day = o.trading_day
        WHERE o.trading_day IS NULL
        GROUP BY b.symbol
        ORDER BY b.symbol
    """).fetchall()
    print_table(["Symbol", "Gap Start", "Gap End", "Missing Days"], rows)

    # Heartbeat
    print()
    db_path = get_db_path()
    heartbeat = db_path.parent / "outcome_builder.heartbeat"
    if heartbeat.exists():
        content = heartbeat.read_text().strip()
        print(f"  HEARTBEAT: {content}")
    else:
        print("  HEARTBEAT: No heartbeat file found (builder not running or using different DB)")

    pause()


# ============================================================
# MAIN MENU
# ============================================================

MENU = [
    ("FULL RUNDOWN — just tell me everything", full_rundown),
    ("Data overview — what do I have?", overview),
    ("Win rate by session", win_rate_by_session),
    ("Best RR target", best_rr_target),
    ("Entry model comparison (E1 vs E3)", entry_model_comparison),
    ("Confirm bars comparison", confirm_bars_comparison),
    ("Yearly breakdown — is the edge stable?", yearly_breakdown),
    ("Long vs Short performance", long_vs_short),
    ("Outcome distribution (win/loss/scratch)", outcome_distribution),
    ("Top 10 best parameter combos", best_combos),
    ("Bottom 10 worst parameter combos", worst_combos),
    ("Outcome rebuild status", rebuild_status),
    ("Custom SQL query (advanced)", custom_query),
]


def main():
    db_path = get_db_path()
    if not db_path.exists():
        print(f"\n  Database not found: {db_path}")
        print("  Run: python pipeline/init_db.py")
        sys.exit(1)

    con = duckdb.connect(str(db_path), read_only=True)

    try:
        while True:
            clear_screen()
            print()
            print("  ╔══════════════════════════════════════════════╗")
            print("  ║         ORB PIPELINE DATA EXPLORER          ║")
            print("  ╠══════════════════════════════════════════════╣")
            print(f"  ║  DB: {str(db_path)[-40:].ljust(40)} ║")
            print("  ╚══════════════════════════════════════════════╝")
            print()

            for i, (label, _) in enumerate(MENU, 1):
                print(f"    {i:2d}. {label}")
            print()
            print(f"     0. Exit")
            print()

            choice = input("  Pick a number: ").strip()

            if choice == "0":
                print("\n  Later. 👋\n")
                break
            if choice.isdigit() and 1 <= int(choice) <= len(MENU):
                try:
                    MENU[int(choice) - 1][1](con)
                except Exception as e:
                    print(f"\n  Error: {e}")
                    pause()
            else:
                print("  Invalid choice.")
                import time
                time.sleep(0.5)
    finally:
        con.close()


if __name__ == "__main__":
    main()
