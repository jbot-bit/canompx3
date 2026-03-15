#!/usr/bin/env python3
"""
Post-rebuild assertion suite — catches bad rebuilds before they go unnoticed.

Runs 6 data integrity assertions after a rebuild chain completes:
  A1: Row count decrease (WARNING)
  A2: Date continuity gaps in bars_1m (FAIL)
  A3: Cross-table FK: orb_outcomes → daily_features (FAIL)
  A4: Strategy count drop > threshold (WARNING)
  A5: Outcome coverage per session × aperture (FAIL)
  A6: Schema alignment: daily_features column count (FAIL)

Severity:
  FAIL    — operator must investigate. Orchestrator marks rebuild as WARNING.
  WARNING — logged, rebuild continues.

Usage:
    python scripts/tools/assert_rebuild.py                     # all instruments
    python scripts/tools/assert_rebuild.py --instrument MGC    # one instrument
    python scripts/tools/assert_rebuild.py --json              # machine-readable output
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.audit_log import get_previous_counts
from pipeline.build_daily_features import VALID_ORB_MINUTES
from pipeline.dst import SESSION_CATALOG
from pipeline.init_db import ORB_LABELS
from pipeline.paths import GOLD_DB_PATH

# Number of per-session ORB columns in daily_features DDL
_ORB_COLUMNS_PER_SESSION = 14
# Static columns in daily_features (everything before the ORB block)
# Includes: PK(3), bar_count(1), session stats(6), RSI(1), daily OHLC(4),
# gap(1), ATR(1), ATR velocity(2), compression tiers(6), DST flags(2),
# calendar flags(3), DOW flags(3), prior day(6), overnight/pre-1000(5),
# liquidity sweep(4), day_type(1), GARCH(2) = 51
_STATIC_COLUMN_COUNT = 51
EXPECTED_DAILY_FEATURES_COLUMNS = _STATIC_COLUMN_COUNT + (len(ORB_LABELS) * _ORB_COLUMNS_PER_SESSION)

# Strategy count drop threshold — WARNING if active count < this fraction of previous
STRATEGY_DROP_THRESHOLD = 0.70

APERTURES = VALID_ORB_MINUTES  # canonical source — never hardcode


# ---------------------------------------------------------------------------
# Assertion results
# ---------------------------------------------------------------------------


class AssertionResult:
    """One assertion result."""

    def __init__(self, assertion_id: str, severity: str, passed: bool, message: str):
        self.assertion_id = assertion_id
        self.severity = severity  # "FAIL" or "WARNING"
        self.passed = passed
        self.message = message

    def to_dict(self) -> dict:
        return {
            "id": self.assertion_id,
            "severity": self.severity,
            "passed": self.passed,
            "message": self.message,
        }

    def __repr__(self) -> str:
        status = "PASS" if self.passed else self.severity
        return f"[{self.assertion_id}] {status}: {self.message}"


# ---------------------------------------------------------------------------
# Individual assertions
# ---------------------------------------------------------------------------


def assert_row_count_no_decrease(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
) -> list[AssertionResult]:
    """A1: Check that row counts didn't decrease for key tables."""
    results = []
    tables = ["bars_1m", "bars_5m", "daily_features", "orb_outcomes"]

    for table in tables:
        prev = get_previous_counts(con, instrument, table)
        if prev is None:
            continue  # No prior log — nothing to compare

        try:
            col = "instrument" if table in ("experimental_strategies", "validated_setups") else "symbol"
            row = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {col} = $1",
                [instrument],
            ).fetchone()
            current = row[0] if row else 0
        except duckdb.CatalogException:
            continue  # Table doesn't exist yet

        if current < prev:
            results.append(
                AssertionResult(
                    "A1",
                    "WARNING",
                    False,
                    f"{instrument}/{table}: rows decreased {prev} → {current}",
                )
            )

    if not results:
        results.append(AssertionResult("A1", "WARNING", True, f"{instrument}: no row count decreases"))

    return results


def assert_date_continuity(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    max_gap_days: int = 3,
) -> AssertionResult:
    """A2: Check for gaps > max_gap_days calendar days in bars_1m trading days."""
    try:
        rows = con.execute(
            """
            SELECT DISTINCT ts_utc::DATE AS d
            FROM bars_1m
            WHERE symbol = $1
            ORDER BY d
            """,
            [instrument],
        ).fetchall()
    except duckdb.CatalogException:
        return AssertionResult("A2", "FAIL", False, f"{instrument}: bars_1m table not found")

    if len(rows) < 2:
        return AssertionResult("A2", "FAIL", True, f"{instrument}: <2 dates in bars_1m, skipping gap check")

    dates = [r[0] for r in rows]
    gaps = []
    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i - 1]).days
        if gap > max_gap_days:
            gaps.append((dates[i - 1], dates[i], gap))

    if gaps:
        worst = max(gaps, key=lambda g: g[2])
        return AssertionResult(
            "A2",
            "FAIL",
            False,
            f"{instrument}: {len(gaps)} gap(s) > {max_gap_days}d in bars_1m. "
            f"Worst: {worst[0]} → {worst[1]} ({worst[2]}d)",
        )

    return AssertionResult("A2", "FAIL", True, f"{instrument}: bars_1m date continuity OK")


def assert_cross_table_fk(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
) -> AssertionResult:
    """A3: Check orb_outcomes rows have matching daily_features."""
    try:
        row = con.execute(
            """
            SELECT COUNT(*)
            FROM orb_outcomes o
            LEFT JOIN daily_features d
              ON o.symbol = d.symbol
              AND o.trading_day = d.trading_day
              AND o.orb_minutes = d.orb_minutes
            WHERE o.symbol = $1 AND d.symbol IS NULL
            """,
            [instrument],
        ).fetchone()
        orphan_count = row[0] if row else 0
    except duckdb.CatalogException:
        return AssertionResult("A3", "FAIL", False, f"{instrument}: orb_outcomes or daily_features missing")

    if orphan_count > 0:
        return AssertionResult(
            "A3",
            "FAIL",
            False,
            f"{instrument}: {orphan_count} orb_outcomes rows without matching daily_features",
        )

    return AssertionResult("A3", "FAIL", True, f"{instrument}: orb_outcomes FK integrity OK")


def assert_strategy_count_stable(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    threshold: float = STRATEGY_DROP_THRESHOLD,
) -> AssertionResult:
    """A4: Check validated_setups count didn't drop below threshold of previous.

    Compares total rows (not just active) to match what the audit log records
    via get_previous_counts — which stores total rows_after, not filtered counts.
    """
    prev = get_previous_counts(con, instrument, "validated_setups")

    try:
        row = con.execute(
            "SELECT COUNT(*) FROM validated_setups WHERE instrument = $1",
            [instrument],
        ).fetchone()
        current = row[0] if row else 0
    except duckdb.CatalogException:
        return AssertionResult("A4", "WARNING", True, f"{instrument}: validated_setups not found, skipping")

    if prev is None or prev == 0:
        return AssertionResult("A4", "WARNING", True, f"{instrument}: no prior strategy count, skipping")

    ratio = current / prev
    if ratio < threshold:
        return AssertionResult(
            "A4",
            "WARNING",
            False,
            f"{instrument}: strategies dropped {prev} → {current} ({ratio:.0%} of previous, threshold {threshold:.0%})",
        )

    return AssertionResult(
        "A4",
        "WARNING",
        True,
        f"{instrument}: strategy count stable ({current}, was {prev})",
    )


def assert_outcome_coverage(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
) -> AssertionResult:
    """A5: Check every enabled session has outcomes for all apertures (5/15/30)."""
    enabled_sessions = list(SESSION_CATALOG.keys())
    missing = []

    for session in enabled_sessions:
        for aperture in APERTURES:
            try:
                row = con.execute(
                    """
                    SELECT COUNT(*) FROM orb_outcomes
                    WHERE symbol = $1 AND orb_label = $2 AND orb_minutes = $3
                    """,
                    [instrument, session, aperture],
                ).fetchone()
                count = row[0] if row else 0
            except duckdb.CatalogException:
                count = 0

            if count == 0:
                missing.append(f"{session}/O{aperture}")

    if missing:
        sample = missing[:5]
        extra = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        return AssertionResult(
            "A5",
            "FAIL",
            False,
            f"{instrument}: {len(missing)} session×aperture combo(s) missing outcomes: {', '.join(sample)}{extra}",
        )

    return AssertionResult(
        "A5",
        "FAIL",
        True,
        f"{instrument}: all {len(enabled_sessions)}×{len(APERTURES)} session×aperture combos have outcomes",
    )


def assert_schema_alignment(
    con: duckdb.DuckDBPyConnection,
) -> AssertionResult:
    """A6: Check daily_features column count matches expected from init_db.py."""
    try:
        cols = con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'daily_features'"
        ).fetchall()
        actual = len(cols)
    except Exception:
        return AssertionResult("A6", "FAIL", False, "daily_features table not found or not queryable")

    if actual < EXPECTED_DAILY_FEATURES_COLUMNS:
        return AssertionResult(
            "A6",
            "FAIL",
            False,
            f"daily_features has {actual} columns, expected >= {EXPECTED_DAILY_FEATURES_COLUMNS} "
            f"({len(ORB_LABELS)} sessions × {_ORB_COLUMNS_PER_SESSION} + {_STATIC_COLUMN_COUNT} static)",
        )

    return AssertionResult(
        "A6",
        "FAIL",
        True,
        f"daily_features schema aligned ({actual} columns, min {EXPECTED_DAILY_FEATURES_COLUMNS})",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_assertions(
    con: duckdb.DuckDBPyConnection,
    instrument: str | None = None,
) -> list[AssertionResult]:
    """Run all assertions. Returns list of results.

    If instrument is None, runs for all active instruments (A1-A5 per instrument,
    A6 once globally).
    """
    instruments = [instrument] if instrument else list(ACTIVE_ORB_INSTRUMENTS)
    results: list[AssertionResult] = []

    for inst in instruments:
        results.extend(assert_row_count_no_decrease(con, inst))
        results.append(assert_date_continuity(con, inst))
        results.append(assert_cross_table_fk(con, inst))
        results.append(assert_strategy_count_stable(con, inst))
        results.append(assert_outcome_coverage(con, inst))

    # A6 is global (not per-instrument)
    results.append(assert_schema_alignment(con))

    return results


def has_failures(results: list[AssertionResult]) -> bool:
    """Return True if any FAIL-severity assertion did not pass."""
    return any(not r.passed and r.severity == "FAIL" for r in results)


def has_warnings(results: list[AssertionResult]) -> bool:
    """Return True if any WARNING-severity assertion did not pass."""
    return any(not r.passed and r.severity == "WARNING" for r in results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Post-rebuild data integrity assertions")
    parser.add_argument("--instrument", type=str, default=None, help="Instrument (default: all active)")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Machine-readable JSON output")
    parser.add_argument("--db-path", type=str, default=None, help="Database path (default: GOLD_DB_PATH)")

    args = parser.parse_args()
    db_path = args.db_path or str(GOLD_DB_PATH)

    if args.instrument and args.instrument not in ACTIVE_ORB_INSTRUMENTS:
        parser.error(f"Unknown instrument '{args.instrument}'. Valid: {', '.join(sorted(ACTIVE_ORB_INSTRUMENTS))}")

    con = duckdb.connect(db_path, read_only=True)
    try:
        results = run_assertions(con, instrument=args.instrument)
    finally:
        con.close()

    if args.json_output:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print("=" * 60)
        print("POST-REBUILD ASSERTIONS")
        print("=" * 60)
        for r in results:
            print(f"  {r}")
        print()

        failures = [r for r in results if not r.passed and r.severity == "FAIL"]
        warnings = [r for r in results if not r.passed and r.severity == "WARNING"]
        passed = [r for r in results if r.passed]

        print(f"  {len(passed)} passed, {len(warnings)} warnings, {len(failures)} failures")

        if failures:
            print("\n  FAILURES (require investigation):")
            for f in failures:
                print(f"    {f}")

    sys.exit(1 if has_failures(results) else 0)


if __name__ == "__main__":
    main()
