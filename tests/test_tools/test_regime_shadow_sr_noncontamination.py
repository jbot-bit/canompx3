"""
G4 — sr_monitor non-contamination by construction.

sr_monitor.prepare_monitor_inputs reads `paper_trades WHERE strategy_id = ?`
with NO execution_source filter (trading_app/sr_monitor.py:174-182). A naive
reading says shadow rows could contaminate live SR calibration. They cannot, by
construction:

  - Deploy lanes are CORE tier (sample_size >= 100).
  - The REGIME shadow universe is REGIME tier (sample_size 30-99).
  - classify_strategy makes the tiers DISJOINT, so no strategy_id can carry both
    shadow and live/backfill rows. SR is queried per strategy_id, so a shadow
    strategy_id's rows never enter a deploy lane's SR stream.

These tests defend that disjointness (not assume it). If a future change ever
intends to point sr_monitor at REGIME shadow lanes, it MUST opt in by
execution_source — never implicitly — and these tests will fail first.
"""

from __future__ import annotations

import time

import duckdb
import pytest

from pipeline.paths import GOLD_DB_PATH
from trading_app.config import CORE_MIN_SAMPLES, classify_strategy

# DuckDB has NO connection lock-timeout option (verified against the official
# duckdb-web docs, connect/concurrency.md): a read-only open blocks/raises while
# another process holds the single read-write lock. Under the pre-commit hook the
# canonical gold.db is frequently locked by a live bot / CanonMPX_DailyRefresh /
# peer session, so an unguarded connect here hangs until the hook's test timeout
# (observed 2026-06-03). This non-hermetic monitoring test must therefore poll for
# the lock to free and SKIP (environment condition, not a code defect) rather than
# hang — the canonical wait-on pattern from .claude/rules/condition-based-waiting.md
# ("DuckDB write-lock released → a read_only connect succeeds (try/except, retry)").
_LOCK_WAIT_S = 6.0
_LOCK_POLL_S = 0.5


def _connect_ro_or_skip(reason_label: str) -> duckdb.DuckDBPyConnection:
    """Open a read-only connection to the canonical DB, polling briefly for a
    held write-lock to free. SKIP (never hang, never fail) if still locked —
    lock contention is an environment state, not a regression in this test."""
    deadline = time.monotonic() + _LOCK_WAIT_S
    last_exc: Exception | None = None
    while True:
        try:
            return duckdb.connect(str(GOLD_DB_PATH), read_only=True)
        except (duckdb.IOException, duckdb.Error) as exc:  # held write-lock, etc.
            last_exc = exc
            if time.monotonic() > deadline:
                pytest.skip(
                    f"canonical gold.db locked by a concurrent writer "
                    f"({reason_label}); skipping non-hermetic live-DB check: {exc}"
                )
            time.sleep(_LOCK_POLL_S)


def test_classifier_makes_regime_and_core_disjoint():
    """The structural invariant the whole defense rests on: a sample_size is
    never classified as both REGIME and CORE."""
    for n in (30, 60, 99, 100, 150, 500):
        tier = classify_strategy(n)
        is_regime = 30 <= n <= 99
        is_core = n >= CORE_MIN_SAMPLES
        assert not (is_regime and is_core), "REGIME and CORE bands must not overlap"
        if is_regime:
            assert tier == "REGIME"
        if is_core:
            assert tier == "CORE"


@pytest.mark.slow
def test_shadow_universe_is_regime_only_disjoint_from_core_deploy():
    """Every shadow-universe strategy_id classifies REGIME; therefore none can be
    a CORE-tier deploy lane. Run against the live DB (read-only); skip if the
    canonical DB is unavailable in this environment.

    Marked `slow` (kept out of the pre-commit fast gate, runs in full CI): it
    calls build_universe() which runs compute_fitness for every active REGIME lane
    (~60 live queries, ~2min) and depends on the canonical gold.db being readable
    — under the lock-churning pre-commit window a concurrent writer can block a
    mid-build query and time out the gate. The cheap structural invariant
    (test_classifier_makes_regime_and_core_disjoint) stays in the fast gate."""
    if not GOLD_DB_PATH.exists():
        pytest.skip("canonical gold.db not present in this environment")

    # Gate on lock availability FIRST: build_universe() opens its own read-only
    # connection internally, so confirm the DB is openable (or skip) before
    # calling it — otherwise that internal connect would hang on a held lock.
    _connect_ro_or_skip("pre-build lock probe").close()

    from scripts.tools.regime_shadow_universe import build_universe

    lanes = build_universe()
    if not lanes:
        pytest.skip("no active REGIME lanes in this DB snapshot")

    shadow_ids = {x.strategy_id for x in lanes}

    # Every shadow lane must classify REGIME (the disjointness premise).
    for x in lanes:
        assert classify_strategy(x.sample_size) == "REGIME", (
            f"{x.strategy_id} sample_size={x.sample_size} is not REGIME — would break the disjointness defense"
        )

    # No shadow id may coincide with a CORE-tier active validated setup (the
    # population deploy lanes are drawn from).
    con = _connect_ro_or_skip("core-id lookup")
    try:
        core_ids = {
            r[0]
            for r in con.execute(
                "SELECT strategy_id FROM validated_setups WHERE status = 'active' AND sample_size >= ?",
                [CORE_MIN_SAMPLES],
            ).fetchall()
        }
    finally:
        con.close()

    overlap = shadow_ids & core_ids
    assert not overlap, f"shadow/CORE-deploy tier overlap (contamination risk): {overlap}"
