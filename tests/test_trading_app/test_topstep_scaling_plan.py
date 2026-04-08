"""Tests for trading_app.topstep_scaling_plan (F-1 enforcer module).

@canonical-source docs/research-input/topstep/images/xfa_scaling_chart.png
@canonical-source docs/research-input/topstep/topstep_scaling_plan_article.md
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pytest

from trading_app.topstep_scaling_plan import (
    SCALING_PLAN_LADDER,
    lots_for_position,
    max_lots_for_xfa,
    micros_to_mini_equivalent,
    total_open_lots,
)


# ─── Ladder integrity ───────────────────────────────────────────────────


class TestLadderIntegrity:
    """Verify the canonical ladder values match the official PNG."""

    def test_ladder_has_three_tiers(self):
        assert set(SCALING_PLAN_LADDER.keys()) == {50_000, 100_000, 150_000}

    def test_ladder_50k_canonical(self):
        ladder = SCALING_PLAN_LADDER[50_000]
        assert ladder == [(0.0, 2), (1500.0, 3), (2000.0, 5)]

    def test_ladder_100k_canonical(self):
        ladder = SCALING_PLAN_LADDER[100_000]
        assert ladder == [(0.0, 3), (1500.0, 4), (2000.0, 5), (3000.0, 10)]

    def test_ladder_150k_canonical(self):
        ladder = SCALING_PLAN_LADDER[150_000]
        assert ladder == [(0.0, 3), (1500.0, 4), (2000.0, 5), (3000.0, 10), (4500.0, 15)]


# ─── max_lots_for_xfa ───────────────────────────────────────────────────


class TestMaxLotsForXfa50K:
    def test_day_one_zero_balance(self):
        """Fresh 50K XFA: balance=$0 → 2 lots."""
        assert max_lots_for_xfa(50_000, 0) == 2

    def test_below_first_threshold(self):
        assert max_lots_for_xfa(50_000, 1_499.99) == 2

    def test_at_first_threshold(self):
        assert max_lots_for_xfa(50_000, 1_500.00) == 3

    def test_between_first_and_second(self):
        assert max_lots_for_xfa(50_000, 1_999.99) == 3

    def test_at_second_threshold(self):
        assert max_lots_for_xfa(50_000, 2_000.00) == 5

    def test_above_top_tier(self):
        assert max_lots_for_xfa(50_000, 100_000.00) == 5  # cap is 5 for 50K


class TestMaxLotsForXfa100K:
    def test_day_one(self):
        assert max_lots_for_xfa(100_000, 0) == 3

    def test_at_1500(self):
        assert max_lots_for_xfa(100_000, 1_500) == 4

    def test_at_2000(self):
        assert max_lots_for_xfa(100_000, 2_000) == 5

    def test_at_3000(self):
        assert max_lots_for_xfa(100_000, 3_000) == 10

    def test_above_top_tier(self):
        assert max_lots_for_xfa(100_000, 50_000) == 10


class TestMaxLotsForXfa150K:
    def test_day_one(self):
        assert max_lots_for_xfa(150_000, 0) == 3

    def test_at_1500(self):
        assert max_lots_for_xfa(150_000, 1_500) == 4

    def test_at_2000(self):
        assert max_lots_for_xfa(150_000, 2_000) == 5

    def test_at_3000(self):
        assert max_lots_for_xfa(150_000, 3_000) == 10

    def test_at_4500(self):
        assert max_lots_for_xfa(150_000, 4_500) == 15

    def test_above_top_tier(self):
        assert max_lots_for_xfa(150_000, 100_000) == 15


class TestMaxLotsForXfaErrors:
    def test_unknown_account_size_raises(self):
        with pytest.raises(KeyError):
            max_lots_for_xfa(75_000, 0)

    def test_negative_balance_raises(self):
        with pytest.raises(ValueError):
            max_lots_for_xfa(50_000, -100)


# ─── micros_to_mini_equivalent ──────────────────────────────────────────


class TestMicrosToMini:
    def test_zero_micros(self):
        assert micros_to_mini_equivalent(0) == 0

    def test_one_micro_is_one_mini_equiv(self):
        """The smallest non-zero exposure counts as 1 mini-equivalent (ceiling)."""
        assert micros_to_mini_equivalent(1) == 1

    def test_nine_micros_still_one(self):
        assert micros_to_mini_equivalent(9) == 1

    def test_ten_micros_is_one_mini(self):
        assert micros_to_mini_equivalent(10) == 1

    def test_eleven_micros_is_two_minis(self):
        """Partial second mini still counts toward limit (ceiling)."""
        assert micros_to_mini_equivalent(11) == 2

    def test_twenty_micros_is_two(self):
        assert micros_to_mini_equivalent(20) == 2

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            micros_to_mini_equivalent(-1)


# ─── lots_for_position ──────────────────────────────────────────────────


class TestLotsForPosition:
    def test_micro_mnq_uses_10_to_1_ratio(self):
        assert lots_for_position("MNQ", 1) == 1
        assert lots_for_position("MNQ", 10) == 1
        assert lots_for_position("MNQ", 11) == 2
        assert lots_for_position("MNQ", 50) == 5

    def test_micro_mes_uses_10_to_1_ratio(self):
        assert lots_for_position("MES", 10) == 1

    def test_micro_mgc_uses_10_to_1_ratio(self):
        assert lots_for_position("MGC", 10) == 1

    def test_full_mini_nq_is_one_to_one(self):
        assert lots_for_position("NQ", 1) == 1
        assert lots_for_position("NQ", 5) == 5

    def test_full_mini_es_is_one_to_one(self):
        assert lots_for_position("ES", 5) == 5

    def test_full_mini_gc_is_one_to_one(self):
        assert lots_for_position("GC", 3) == 3


# ─── total_open_lots ────────────────────────────────────────────────────


class _State(Enum):
    ENTERED = "ENTERED"
    ARMED = "ARMED"
    EXITED = "EXITED"


@dataclass
class _Strategy:
    instrument: str


@dataclass
class _Trade:
    strategy: _Strategy
    contracts: int
    state: _State = _State.ENTERED


class TestTotalOpenLots:
    def test_no_trades_returns_zero(self):
        assert total_open_lots([]) == 0

    def test_one_micro_trade(self):
        active = [_Trade(_Strategy("MNQ"), 5)]
        assert total_open_lots(active) == 1  # 5 micros = 1 mini-equiv (ceiling)

    def test_one_mini_trade(self):
        active = [_Trade(_Strategy("NQ"), 3)]
        assert total_open_lots(active) == 3

    def test_mix_of_micros_and_minis(self):
        active = [
            _Trade(_Strategy("MNQ"), 10),  # 1 mini-equiv
            _Trade(_Strategy("MES"), 20),  # 2 mini-equiv
            _Trade(_Strategy("ES"), 1),    # 1 mini
        ]
        assert total_open_lots(active) == 4

    def test_armed_does_not_count(self):
        armed = _Trade(_Strategy("MNQ"), 10)
        armed.state = _State.ARMED
        assert total_open_lots([armed]) == 0

    def test_exited_does_not_count(self):
        exited = _Trade(_Strategy("MNQ"), 10)
        exited.state = _State.EXITED
        assert total_open_lots([exited]) == 0

    def test_filter_by_instrument(self):
        active = [
            _Trade(_Strategy("MNQ"), 10),
            _Trade(_Strategy("MES"), 10),
        ]
        assert total_open_lots(active, instrument="MNQ") == 1
        assert total_open_lots(active, instrument="MES") == 1
        assert total_open_lots(active, instrument="MGC") == 0
        assert total_open_lots(active) == 2  # both


# ─── Day-1 violation scenario (the audit's core finding) ───────────────


class TestDayOneViolationScenario:
    """The exact scenario the audit (F-1) flagged as a BLOCKER."""

    def test_5_simultaneous_lanes_breach_50k_day1(self):
        """topstep_50k_mnq_auto has 5 lanes × 1 micro each.
        Day 1: balance=$0 → max 2 lots → 5 simultaneous = 5 lots > 2 = violation.

        Each lane is 1 micro = 1 mini-equivalent (ceiling). 5 lanes = 5 mini-equivalents.
        """
        active = [_Trade(_Strategy("MNQ"), 1) for _ in range(5)]
        total = total_open_lots(active)
        day_max = max_lots_for_xfa(50_000, 0)
        assert total == 5
        assert day_max == 2
        assert total > day_max  # confirmed: this WOULD violate the Scaling Plan

    def test_2_simultaneous_lanes_within_50k_day1(self):
        """2 simultaneous 1-micro positions = 2 mini-equivalents = at-cap, OK."""
        active = [_Trade(_Strategy("MNQ"), 1) for _ in range(2)]
        total = total_open_lots(active)
        day_max = max_lots_for_xfa(50_000, 0)
        assert total == 2
        assert day_max == 2
        assert total <= day_max
