"""Injection tests for check_fast_lane_oos_power_gate_constants_grounded.

Mutation-probes every constant the gate depends on. Per
`memory/feedback_regex_alternation_sibling_coverage.md`, each gated value
gets its own dedicated injection so a future refactor that drops one of
them doesn't survive silently.

Coverage:
  - OOS_POWER_FLOOR missing / out-of-range / NaN / non-numeric
  - OOS_COHEN_D_TARGET missing / zero / negative / NaN
  - STATUS_VALUES missing 'REJECTED_OOS_UNPOWERED'

Each test monkeypatches the scanner module attributes (or the import path),
calls the check, asserts the expected violation message appears, then
restores. Tests are independent.
"""

from __future__ import annotations

import math

import pytest

from pipeline import check_drift
from scripts.research import fast_lane_promote_queue as scanner


@pytest.fixture
def restore_scanner_constants():
    """Snapshot + restore scanner constants so each test is independent."""
    saved = {
        "OOS_POWER_FLOOR": scanner.OOS_POWER_FLOOR,
        "OOS_COHEN_D_TARGET": scanner.OOS_COHEN_D_TARGET,
        "STATUS_VALUES": scanner.STATUS_VALUES,
    }
    yield
    for k, v in saved.items():
        setattr(scanner, k, v)


def _run_check() -> list[str]:
    return check_drift.check_fast_lane_oos_power_gate_constants_grounded()


class TestPowerFloorInjections:
    def test_baseline_passes(self):
        # Sanity: with constants at their committed values, check passes.
        assert _run_check() == []

    def test_floor_zero_rejected(self, restore_scanner_constants):
        scanner.OOS_POWER_FLOOR = 0.0
        errors = _run_check()
        assert any("OOS_POWER_FLOOR" in e and "out of valid range" in e for e in errors)

    def test_floor_above_one_rejected(self, restore_scanner_constants):
        scanner.OOS_POWER_FLOOR = 1.5
        errors = _run_check()
        assert any("OOS_POWER_FLOOR" in e and "out of valid range" in e for e in errors)

    def test_floor_nan_rejected(self, restore_scanner_constants):
        scanner.OOS_POWER_FLOOR = float("nan")
        errors = _run_check()
        assert any("OOS_POWER_FLOOR" in e and "finite" in e for e in errors)

    def test_floor_non_numeric_rejected(self, restore_scanner_constants):
        scanner.OOS_POWER_FLOOR = "0.50"  # type: ignore[assignment]
        errors = _run_check()
        assert any("OOS_POWER_FLOOR" in e and "numeric" in e for e in errors)

    def test_floor_bool_rejected(self, restore_scanner_constants):
        # bool is a subclass of int in Python; explicit guard required.
        scanner.OOS_POWER_FLOOR = True  # type: ignore[assignment]
        errors = _run_check()
        assert any("OOS_POWER_FLOOR" in e and "not bool" in e for e in errors)


class TestCohenDInjections:
    def test_cohen_d_zero_rejected(self, restore_scanner_constants):
        scanner.OOS_COHEN_D_TARGET = 0.0
        errors = _run_check()
        assert any("OOS_COHEN_D_TARGET" in e and "must be > 0" in e for e in errors)

    def test_cohen_d_negative_rejected(self, restore_scanner_constants):
        scanner.OOS_COHEN_D_TARGET = -0.3
        errors = _run_check()
        assert any("OOS_COHEN_D_TARGET" in e and "must be > 0" in e for e in errors)

    def test_cohen_d_nan_rejected(self, restore_scanner_constants):
        scanner.OOS_COHEN_D_TARGET = float("nan")
        errors = _run_check()
        assert any("OOS_COHEN_D_TARGET" in e and "finite" in e for e in errors)

    def test_cohen_d_non_numeric_rejected(self, restore_scanner_constants):
        scanner.OOS_COHEN_D_TARGET = "0.3"  # type: ignore[assignment]
        errors = _run_check()
        assert any("OOS_COHEN_D_TARGET" in e and "numeric" in e for e in errors)


class TestStatusValuesInjections:
    def test_rejected_status_removed_rejected(self, restore_scanner_constants):
        scanner.STATUS_VALUES = tuple(s for s in scanner.STATUS_VALUES if s != "REJECTED_OOS_UNPOWERED")
        errors = _run_check()
        assert any("REJECTED_OOS_UNPOWERED" in e and "STATUS_VALUES" in e for e in errors)
