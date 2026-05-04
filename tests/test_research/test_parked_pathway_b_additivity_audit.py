"""Scope-lock tests for the 2026-04-29 D2/D4 additivity audit runner.

The runner has a dated filename and therefore is not a normal Python
import target. These tests pin the candidate scope, the canonical
filter resolution, and the profile_id so a future edit cannot silently
drift the audit's input set.

No DB access. No filter behavior tested here — that lives in
tests/test_trading_app/test_config.py::TestStrictGtVariants.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "research" / "2026-04-29-parked-pathway-b-additivity-audit.py"


@pytest.fixture(scope="module")
def runner_mod():
    """Load the dated runner script as a module via importlib."""
    assert RUNNER_PATH.exists(), f"runner missing: {RUNNER_PATH}"
    spec = importlib.util.spec_from_file_location("_audit_runner_2026_04_29", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_candidate_set_is_d2_and_d4_only(runner_mod):
    labels = [c["label"] for c in runner_mod.CANDIDATES]
    assert labels == ["D2_B-MES-EUR", "D4_B-MNQ-COX"], (
        "Audit scope is locked to D2 + D4. Adding more candidates without "
        "extending this test risks silent scope expansion."
    )


def test_profile_id_locked(runner_mod):
    assert runner_mod.PROFILE_ID == "topstep_50k_mnq_auto"


def test_d2_candidate_dict_locked(runner_mod):
    d2 = next(c for c in runner_mod.CANDIDATES if c["label"] == "D2_B-MES-EUR")
    assert d2["instrument"] == "MES"
    assert d2["orb_label"] == "EUROPE_FLOW"
    assert d2["orb_minutes"] == 15
    assert d2["entry_model"] == "E2"
    assert d2["confirm_bars"] == 1
    assert d2["rr_target"] == 1.0
    assert d2["filter_type"] == "OVNRNG_PCT_GT80"


def test_d4_candidate_dict_locked(runner_mod):
    d4 = next(c for c in runner_mod.CANDIDATES if c["label"] == "D4_B-MNQ-COX")
    assert d4["instrument"] == "MNQ"
    assert d4["orb_label"] == "COMEX_SETTLE"
    assert d4["orb_minutes"] == 5
    assert d4["entry_model"] == "E2"
    assert d4["confirm_bars"] == 1
    assert d4["rr_target"] == 1.0
    assert d4["filter_type"] == "GARCH_VOL_PCT_GT70"


def test_candidate_filter_types_resolve_in_all_filters(runner_mod):
    """Each candidate's filter_type must be a canonical ALL_FILTERS key."""
    from trading_app.config import ALL_FILTERS

    for c in runner_mod.CANDIDATES:
        assert c["filter_type"] in ALL_FILTERS, (
            f"{c['label']} references unknown filter_type {c['filter_type']!r} — "
            f"audit would fail-closed at _load_strategy_outcomes."
        )


def test_candidate_filters_are_hypothesis_scoped(runner_mod):
    """Both audit filter_types must NOT be in BASE_GRID_FILTERS.

    The audit only works because OVNRNG_PCT_GT80 / GARCH_VOL_PCT_GT70 are
    candidate-only canonical registrations (per 2026-04-29 audit cycle).
    If a future edit promotes them into the legacy grid, this test fails
    so the broader-scan implication is reviewed before landing.
    """
    from trading_app.config import BASE_GRID_FILTERS

    for c in runner_mod.CANDIDATES:
        assert c["filter_type"] not in BASE_GRID_FILTERS, (
            f"{c['filter_type']} leaked into BASE_GRID_FILTERS — review whether "
            f"the audit's hypothesis-scoped premise still holds."
        )


def test_candidate_pre_reg_paths_exist(runner_mod):
    """Each candidate must point at a real pre-reg yaml on disk."""
    for c in runner_mod.CANDIDATES:
        p = REPO_ROOT / c["pre_reg"]
        assert p.exists(), f"{c['label']}: pre_reg path missing: {p}"
