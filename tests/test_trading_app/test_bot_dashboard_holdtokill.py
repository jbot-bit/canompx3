"""Tests for the kill-switch open-position truth claim.

The pre-2026-05-14 cockpit HoldToKill UI counted open positions by reading
``position_qty`` / ``open_position`` fields that DO NOT EXIST in the
canonical ``bot_state`` payload — see
``trading_app.live.bot_state.build_state_snapshot`` (status field values are
``WAITING`` / ``ARMED`` / ``IN_TRADE`` / ``FLAT``). That bug made the
operator-confirmation modal unreachable, turning kill into fail-OPEN on the
confirmation path.

This module gates the Python truth claim exercised by the HTML mirror.
"""

from __future__ import annotations

import pytest

from trading_app.live.bot_dashboard import _count_open_positions_from_state


def test_missing_state_returns_none() -> None:
    """No state dict at all → unknown → caller MUST show modal."""
    assert _count_open_positions_from_state(None) is None


def test_present_false_returns_none() -> None:
    """Stale snapshot (``present=False``) → unknown → caller MUST show modal."""
    assert _count_open_positions_from_state({"present": False, "state": {}}) is None


def test_present_but_no_inner_state_returns_none() -> None:
    """Malformed payload (``state`` missing, None, or non-dict) → unknown.

    Three distinct upstream failure modes — ALL must return None to keep
    operator confirmation fail-CLOSED. The ``state: None`` branch is the
    one a future refactor is most likely to coalesce with "empty lanes"
    and silently downgrade to a 0-count direct kill.
    """
    assert _count_open_positions_from_state({"present": True}) is None
    assert _count_open_positions_from_state({"present": True, "state": None}) is None
    assert _count_open_positions_from_state({"present": True, "state": "broken"}) is None


def test_no_lanes_returns_zero() -> None:
    """State present, no lanes dict → zero positions, kill may proceed."""
    payload = {"present": True, "state": {}}
    assert _count_open_positions_from_state(payload) == 0


def test_all_waiting_returns_zero() -> None:
    """State present, all lanes WAITING → zero positions."""
    payload = {
        "present": True,
        "state": {
            "lanes": {
                "MNQ_NYSE_OPEN_E2_RR1_5_CB1_O5": {"status": "WAITING"},
                "MNQ_COMEX_SETTLE_E2_RR1_5_CB1_O5": {"status": "WAITING"},
            },
        },
    }
    assert _count_open_positions_from_state(payload) == 0


def test_one_in_trade_returns_one() -> None:
    """Single lane IN_TRADE → 1, modal must show."""
    payload = {
        "present": True,
        "state": {
            "lanes": {
                "MNQ_NYSE_OPEN_E2_RR1_5_CB1_O5": {"status": "IN_TRADE"},
                "MNQ_COMEX_SETTLE_E2_RR1_5_CB1_O5": {"status": "WAITING"},
            },
        },
    }
    assert _count_open_positions_from_state(payload) == 1


def test_two_in_trade_returns_two() -> None:
    """Two lanes IN_TRADE → 2."""
    payload = {
        "present": True,
        "state": {
            "lanes": {
                "MNQ_a": {"status": "IN_TRADE"},
                "MNQ_b": {"status": "IN_TRADE"},
                "MNQ_c": {"status": "FLAT"},
            },
        },
    }
    assert _count_open_positions_from_state(payload) == 2


def test_armed_status_not_counted_as_in_trade() -> None:
    """ARMED is pre-entry — not an open position. FLAT is post-exit — not open."""
    payload = {
        "present": True,
        "state": {
            "lanes": {
                "a": {"status": "ARMED"},
                "b": {"status": "FLAT"},
            },
        },
    }
    assert _count_open_positions_from_state(payload) == 0


@pytest.mark.parametrize("bad", [[], "string", 42, 3.14])
def test_non_dict_state_returns_none(bad: object) -> None:
    """Defensive type check on the outer state argument."""
    assert _count_open_positions_from_state(bad) is None  # type: ignore[arg-type]


def test_legacy_fields_no_longer_consulted() -> None:
    """Pre-fix code read ``position_qty`` / ``open_position`` from a WAITING lane.

    With the canonical fix in place, presence of these legacy fields must NOT
    inflate the count — only ``status == "IN_TRADE"`` matters.
    """
    payload = {
        "present": True,
        "state": {
            "lanes": {
                "ghost": {"status": "WAITING", "position_qty": 3, "open_position": True},
            },
        },
    }
    assert _count_open_positions_from_state(payload) == 0
