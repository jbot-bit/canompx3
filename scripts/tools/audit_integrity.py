#!/usr/bin/env python3
"""Full data integrity audit — checks everything is honest and consistent.

Each check returns list[str] violations. Exit code 0 = all passed, 1 = failures.
Human-readable output is preserved for interactive use.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import duckdb
from pipeline.asset_configs import ASSET_CONFIGS
from pipeline.paths import GOLD_DB_PATH

ACTIVE_INSTRUMENTS = ['MGC', 'MNQ', 'MES', 'M2K']


def _connect():
    return duckdb.connect(str(GOLD_DB_PATH), read_only=True)


def check_outcome_coverage(con) -> list[str]:
    """1. Outcome coverage vs enabled_sessions."""
    violations = []
    for inst in ACTIVE_INSTRUMENTS:
        cfg = ASSET_CONFIGS[inst]
        enabled = set(cfg['enabled_sessions'])
        r = con.execute(
            "SELECT DISTINCT orb_label FROM orb_outcomes WHERE symbol = ?",
            [cfg['symbol']],
        ).fetchall()
        in_db = set(row[0] for row in r)
        missing = enabled - in_db
        extra = in_db - enabled
        if missing:
            violations.append(f"  {inst}: missing sessions in outcomes: {sorted(missing)}")
        if extra:
            violations.append(f"  {inst}: extra sessions in outcomes: {sorted(extra)}")
    return violations


def check_validated_session_integrity(con) -> list[str]:
    """2. Validated_setups sessions must be in enabled_sessions."""
    violations = []
    r = con.execute("""
        SELECT instrument, orb_label, COUNT(*)
        FROM validated_setups WHERE status='active'
        GROUP BY instrument, orb_label ORDER BY instrument, orb_label
    """).fetchall()
    for inst, sess, n in r:
        cfg = ASSET_CONFIGS.get(inst)
        if cfg and sess not in cfg.get('enabled_sessions', []):
            violations.append(
                f"  {inst} x {sess} ({n} strategies) NOT in enabled_sessions"
            )
    return violations


def check_edge_family_integrity(con) -> list[str]:
    """3. No orphan validated strategies (NULL family_hash)."""
    violations = []
    r_orphan = con.execute("""
        SELECT COUNT(*) FROM validated_setups v
        WHERE v.status = 'active' AND v.family_hash IS NULL
    """).fetchone()[0]
    if r_orphan > 0:
        violations.append(f"  {r_orphan} validated strategies with NULL family_hash")
    return violations


def check_e0_contamination(con) -> list[str]:
    """4. No E0 rows in any trading table."""
    violations = []
    for table in ['orb_outcomes', 'experimental_strategies', 'validated_setups']:
        r = con.execute(
            f"SELECT COUNT(*) FROM {table} WHERE entry_model = 'E0'"
        ).fetchone()[0]
        if r > 0:
            violations.append(f"  {table}: {r} E0 rows (should be 0)")
    return violations


def check_old_session_names(con) -> list[str]:
    """5. No old fixed-clock session names in DB."""
    violations = []
    for table in ['orb_outcomes', 'experimental_strategies', 'validated_setups']:
        r = con.execute(f"""
            SELECT DISTINCT orb_label FROM {table}
            WHERE orb_label IN ('0900','1000','1100','1800','2300','0030')
        """).fetchall()
        if r:
            violations.append(
                f"  {table}: old session names: {[x[0] for x in r]}"
            )
    return violations


def check_e0_cb2_contamination(con) -> list[str]:
    """6. No E0 + CB>1 rows."""
    violations = []
    for table in ['orb_outcomes', 'experimental_strategies', 'validated_setups']:
        r = con.execute(
            f"SELECT COUNT(*) FROM {table} WHERE entry_model = 'E0' AND confirm_bars > 1"
        ).fetchone()[0]
        if r > 0:
            violations.append(f"  {table}: {r} E0+CB>1 rows (should be 0)")
    return violations


def check_daily_features_multiplicity(con) -> list[str]:
    """7. daily_features row counts (informational — no violations)."""
    return []  # Informational only


def check_outcome_counts(con) -> list[str]:
    """8. Outcome counts by aperture (informational — no violations)."""
    return []  # Informational only


def check_validated_by_orb_minutes(con) -> list[str]:
    """9. Validated strategies by orb_minutes (informational — no violations)."""
    return []  # Informational only


def check_experimental_by_orb_minutes(con) -> list[str]:
    """10. Experimental strategies by orb_minutes (informational — no violations)."""
    return []  # Informational only


def check_dead_instrument_contamination(con) -> list[str]:
    """11. No dead instruments in validated_setups."""
    violations = []
    r = con.execute("""
        SELECT instrument, COUNT(*) FROM validated_setups
        WHERE status = 'active'
        AND instrument NOT IN ('MGC','MNQ','MES','M2K')
        GROUP BY instrument
    """).fetchall()
    for inst, n in r:
        violations.append(f"  {inst}: {n} active validated strategies (dead instrument)")
    return violations


def check_duplicate_strategy_ids(con) -> list[str]:
    """12. No duplicate strategy_ids in validated_setups."""
    violations = []
    r = con.execute("""
        SELECT strategy_id, COUNT(*) as n FROM validated_setups
        GROUP BY strategy_id HAVING COUNT(*) > 1
    """).fetchall()
    if r:
        violations.append(f"  {len(r)} duplicate strategy_ids in validated_setups")
    return violations


def check_outcome_date_ranges(con) -> list[str]:
    """13. Outcome date ranges (informational — no violations)."""
    return []  # Informational only


def check_fdr_breakdown(con) -> list[str]:
    """14. FDR breakdown (informational — no violations)."""
    return []  # Informational only


def check_win_rate_sanity(con) -> list[str]:
    """15. Win rates in sane range [20%-85%].

    Lower bound is 20% (not 30%) because high-RR ORB strategies with
    strict filters legitimately have win rates in the 22-30% range.
    """
    violations = []
    r = con.execute("""
        SELECT strategy_id, win_rate FROM validated_setups
        WHERE status='active' AND (win_rate > 0.85 OR win_rate < 0.20)
    """).fetchall()
    if r:
        violations.append(f"  {len(r)} strategies with extreme win rates (outside 20-85%)")
    return violations


def check_negative_expectancy(con) -> list[str]:
    """16. No active strategies with expectancy_r <= 0."""
    violations = []
    r = con.execute("""
        SELECT COUNT(*) FROM validated_setups
        WHERE status='active' AND expectancy_r <= 0
    """).fetchone()[0]
    if r > 0:
        violations.append(f"  {r} active strategies with expectancy_r <= 0")
    return violations


def check_table_row_counts(con) -> list[str]:
    """17. Table row counts (informational — no violations)."""
    return []  # Informational only


# ── Ordered check registry ────────────────────────────────────────────
CHECKS = [
    ("1. Outcome coverage vs enabled_sessions", check_outcome_coverage),
    ("2. Validated_setups session integrity", check_validated_session_integrity),
    ("3. Edge family integrity (orphan check)", check_edge_family_integrity),
    ("4. E0 contamination", check_e0_contamination),
    ("5. Old session name check", check_old_session_names),
    ("6. E0 CB2+ contamination", check_e0_cb2_contamination),
    ("7. Daily_features row counts", check_daily_features_multiplicity),
    ("8. Outcomes by orb_minutes", check_outcome_counts),
    ("9. Validated by orb_minutes", check_validated_by_orb_minutes),
    ("10. Experimental by orb_minutes", check_experimental_by_orb_minutes),
    ("11. Dead instrument check", check_dead_instrument_contamination),
    ("12. Duplicate strategy IDs", check_duplicate_strategy_ids),
    ("13. Outcome date ranges", check_outcome_date_ranges),
    ("14. FDR breakdown", check_fdr_breakdown),
    ("15. Win rate sanity", check_win_rate_sanity),
    ("16. Negative expectancy check", check_negative_expectancy),
    ("17. Table row counts", check_table_row_counts),
]


def _print_informational(con):
    """Print informational stats that don't produce violations."""
    print('\n--- 7. DAILY_FEATURES ROW COUNTS ---')
    for row in con.execute("""
        SELECT symbol, orb_minutes, COUNT(*) as n FROM daily_features
        WHERE symbol IN ('MGC','MNQ','MES','M2K')
        GROUP BY symbol, orb_minutes ORDER BY symbol, orb_minutes
    """).fetchall():
        print(f'  {row[0]} orb_{row[1]}m: {row[2]:,} rows')

    print('\n--- 8. OUTCOMES BY ORB_MINUTES ---')
    for row in con.execute("""
        SELECT symbol, orb_minutes, COUNT(*) FROM orb_outcomes
        WHERE symbol IN ('MGC','MNQ','MES','M2K')
        GROUP BY symbol, orb_minutes ORDER BY symbol, orb_minutes
    """).fetchall():
        print(f'  {row[0]} {row[1]}m: {row[2]:,}')

    print('\n--- 9. VALIDATED STRATEGIES BY ORB_MINUTES ---')
    for row in con.execute("""
        SELECT orb_minutes, instrument, COUNT(*) FROM validated_setups
        WHERE status='active'
        GROUP BY orb_minutes, instrument ORDER BY orb_minutes, instrument
    """).fetchall():
        print(f'  {row[0]}m {row[1]}: {row[2]}')
    for row in con.execute("""
        SELECT orb_minutes, COUNT(*) FROM validated_setups
        WHERE status='active' GROUP BY orb_minutes ORDER BY orb_minutes
    """).fetchall():
        print(f'    TOTAL {row[0]}m: {row[1]}')

    print('\n--- 10. EXPERIMENTAL STRATEGIES BY ORB_MINUTES ---')
    for row in con.execute("""
        SELECT orb_minutes, instrument, COUNT(*) FROM experimental_strategies
        WHERE instrument IN ('MGC','MNQ','MES','M2K')
        GROUP BY orb_minutes, instrument ORDER BY orb_minutes, instrument
    """).fetchall():
        print(f'  {row[0]}m {row[1]}: {row[2]:,}')

    print('\n--- 13. OUTCOME DATE RANGES ---')
    for row in con.execute("""
        SELECT symbol, MIN(trading_day), MAX(trading_day), COUNT(DISTINCT trading_day)
        FROM orb_outcomes WHERE symbol IN ('MGC','MNQ','MES','M2K')
        GROUP BY symbol ORDER BY symbol
    """).fetchall():
        print(f'  {row[0]}: {row[1]} to {row[2]} ({row[3]:,} days)')

    print('\n--- 14. FDR BREAKDOWN ---')
    total_fdr = 0
    for row in con.execute("""
        SELECT instrument,
               SUM(CASE WHEN fdr_significant THEN 1 ELSE 0 END) as fdr_yes,
               SUM(CASE WHEN NOT fdr_significant THEN 1 ELSE 0 END) as fdr_no,
               COUNT(*) as total
        FROM validated_setups WHERE status='active'
        GROUP BY instrument ORDER BY instrument
    """).fetchall():
        print(f'  {row[0]}: {row[1]} FDR-sig / {row[2]} not / {row[3]} total')
        total_fdr += row[1]
    print(f'  TOTAL FDR significant: {total_fdr}')

    print('\n--- 17. TABLE ROW COUNTS ---')
    for table in ['bars_1m', 'bars_5m', 'daily_features', 'orb_outcomes',
                  'experimental_strategies', 'validated_setups', 'edge_families']:
        r = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f'  {table}: {r:,}')


def main():
    print('=' * 70)
    print('INTEGRITY AUDIT — EVERYTHING MUST BE HONEST')
    print('=' * 70)

    con = _connect()
    all_violations = []

    try:
        for label, check_fn in CHECKS:
            print(f'\n--- {label} ---')
            v = check_fn(con)
            if v:
                print('  FAILED:')
                for line in v:
                    print(line)
                all_violations.extend(v)
            else:
                print('  OK')

        # Print informational stats (no violations, just data)
        _print_informational(con)
    finally:
        con.close()

    print('\n' + '=' * 70)
    if all_violations:
        print(f'INTEGRITY AUDIT FAILED: {len(all_violations)} violation(s)')
        print('=' * 70)
        sys.exit(1)
    else:
        print('INTEGRITY AUDIT PASSED: all 17 checks clean')
        print('=' * 70)
        sys.exit(0)


if __name__ == "__main__":
    main()
