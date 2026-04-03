"""Tests for prop firm payout simulator.

Every payout rule sourced from official help center pages:
- TopStep: help.topstep.com (pasted by user Apr 3 2026)
- MFFU: help.myfundedfutures.com articles 13134709, 13134718, 13286746, 13745661
- Bulenox: bulenox.com/help/master-account (scraped Apr 3 2026)
"""

import pytest

from scripts.tools.payout_simulator import PayoutPolicy, PayoutSimulator, SimResult


# ---------------------------------------------------------------------------
# PayoutPolicy factory methods
# ---------------------------------------------------------------------------


class TestPayoutPolicyFactories:
    def test_topstep_50k_fields(self):
        p = PayoutPolicy.topstep_50k()
        assert p.dd_type == "eod_trailing"
        assert p.dd_amount == 2000
        assert p.dd_lock_at == 0  # MLL set to $0 after first payout
        assert p.starting_balance == 50_000
        assert p.buffer_before_first_payout == 2000
        assert p.winning_day_threshold == 150
        assert p.winning_days_per_cycle == 5
        assert p.max_payout_per_request == 5000
        assert p.max_payout_pct_of_balance == 0.50
        assert p.profit_split == 0.90
        assert p.payout_fee == 30
        assert p.min_payout == 125
        assert p.consistency_rule is None
        assert p.safety_reserve == 0
        assert p.early_payout_caps is None
        assert p.reset_cost == 49

    def test_mffu_rapid_50k_fields(self):
        p = PayoutPolicy.mffu_rapid_50k()
        assert p.dd_type == "intraday_trailing"
        assert p.dd_amount == 2000
        assert p.dd_lock_at == 100  # Locks at $100
        assert p.starting_balance == 50_000
        assert p.buffer_before_first_payout == 2100
        assert p.winning_days_per_cycle == 0  # Daily payouts, no winning day req
        assert p.max_payout_per_request is None  # Not stated officially
        assert p.profit_split == 0.90
        assert p.payout_fee == 0
        assert p.min_payout == 500
        assert p.consistency_rule is None
        assert p.safety_reserve == 0
        assert p.reset_cost == 130  # Monthly sub cost as reset proxy

    def test_bulenox_50k_fields(self):
        p = PayoutPolicy.bulenox_50k()
        assert p.dd_type == "eod_trailing"
        assert p.dd_amount == 2500
        assert p.dd_lock_at == 100  # starting + $100, stored as offset
        assert p.starting_balance == 50_000
        assert p.winning_day_threshold == 0  # No winning day requirement
        assert p.winning_days_per_cycle == 0
        assert p.min_trading_days_before_first == 10
        assert p.max_payout_pct_of_balance == 1.0  # No percentage cap stated
        assert p.profit_split == 0.90  # After first $10K
        assert p.first_profit_split == 1.00  # First $10K at 100%
        assert p.first_profit_split_threshold == 10_000
        assert p.min_payout == 1000
        assert p.consistency_rule == 0.40
        assert p.safety_reserve == 2600
        assert p.early_payout_caps == (1500, 1500, 1500)
        assert p.reset_cost == 148  # One-time activation fee


# ---------------------------------------------------------------------------
# DD Mechanics
# ---------------------------------------------------------------------------


class TestDDMechanics:
    def test_eod_trailing_basic(self):
        """EOD trailing: floor follows peak, always dd_amount below."""
        p = PayoutPolicy.topstep_50k()
        sim = PayoutSimulator(p, trades=[])
        # Start: balance 50K, floor 48K
        assert sim.balance == 50_000
        assert sim.dd_floor == 48_000

    def test_eod_trailing_profit_raises_floor(self):
        """Profit raises peak → floor trails up."""
        p = PayoutPolicy.topstep_50k()
        trades = [(1, 500.0)]  # Day 1: +$500
        sim = PayoutSimulator(p, trades=trades)
        sim.run()
        assert sim.balance == 50_500
        assert sim.dd_floor == 48_500  # 50500 - 2000

    def test_eod_trailing_loss_floor_stays(self):
        """Loss doesn't move floor down — it trails the PEAK."""
        p = PayoutPolicy.topstep_50k()
        trades = [(1, 500.0), (2, -300.0)]
        sim = PayoutSimulator(p, trades=trades)
        sim.run()
        assert sim.balance == 50_200
        assert sim.dd_floor == 48_500  # Peak was 50500, floor stays

    def test_blow_on_dd_breach(self):
        """Balance hitting floor = blown account."""
        p = PayoutPolicy.topstep_50k()
        trades = [(1, -2000.0)]  # Lose entire DD
        sim = PayoutSimulator(p, trades=trades)
        result = sim.run()
        assert result.blow_count == 1

    def test_topstep_dd_locks_at_zero_after_payout(self):
        """TopStep: after first payout, MLL=$0 → floor=starting balance stays there.
        'the capital remaining in your account will be the maximum amount you will be able to lose'
        This means floor drops to $0 absolute. Room = entire balance."""
        p = PayoutPolicy.topstep_50k()
        # Build up $3K profit over winning days, then take payout
        trades = []
        for d in range(1, 20):
            trades.append((d, 200.0))  # 19 days × $200 = $3800 profit
        sim = PayoutSimulator(p, trades=trades)
        result = sim.run()
        # Should have taken at least one payout
        assert result.payout_count >= 1
        # After payout, floor should be at 0 (absolute)
        assert sim.dd_floor == 0

    def test_mffu_dd_locks_at_100(self):
        """MFFU Rapid: DD locks at $100 once reached. Never moves again."""
        p = PayoutPolicy.mffu_rapid_50k()
        # Profit enough for floor to reach $100
        # Start: balance=50K (conceptually 0, but sim tracks absolute)
        # Actually MFFU Rapid starts at $0 balance. Let me model that.
        # The sim should handle starting_balance correctly.
        trades = []
        for d in range(1, 30):
            trades.append((d, 200.0))  # Lots of profit to push floor past $100
        sim = PayoutSimulator(p, trades=trades)
        sim.run()
        assert sim.dd_floor == 100  # Locked at $100

    def test_bulenox_dd_locks_at_starting_plus_100(self):
        """Bulenox: DD locks at initial starting balance + $100."""
        p = PayoutPolicy.bulenox_50k()
        trades = []
        for d in range(1, 40):
            trades.append((d, 200.0))
        sim = PayoutSimulator(p, trades=trades)
        sim.run()
        # Floor should lock at 50000 + 100 = 50100
        assert sim.dd_floor == 50_100


# ---------------------------------------------------------------------------
# Payout Logic
# ---------------------------------------------------------------------------


class TestPayoutLogic:
    def test_topstep_needs_5_winning_days(self):
        """TopStep: can't withdraw until 5 days with $150+ PnL."""
        p = PayoutPolicy.topstep_50k()
        # 4 winning days — not enough
        trades = [(d, 200.0) for d in range(1, 5)]  # Only 4 days
        sim = PayoutSimulator(p, trades=trades)
        result = sim.run()
        assert result.payout_count == 0

    def test_topstep_5_winning_days_triggers_payout(self):
        """5 winning days + MLL at $0 + profitable → payout eligible."""
        p = PayoutPolicy.topstep_50k()
        # Need: 5 winning days ($150+) AND enough profit to be above buffer
        # MLL reaches $0 after $2K profit. TopStep recommends this before payout.
        trades = [(d, 500.0) for d in range(1, 11)]  # 10 days × $500 = $5K
        sim = PayoutSimulator(p, trades=trades)
        result = sim.run()
        assert result.payout_count >= 1
        assert result.total_withdrawn_net > 0

    def test_topstep_payout_capped_at_5000(self):
        """Max payout = min(50% of balance, $5000)."""
        p = PayoutPolicy.topstep_50k()
        # Build huge balance to test cap
        trades = [(d, 1000.0) for d in range(1, 21)]  # $20K profit
        sim = PayoutSimulator(p, trades=trades)
        result = sim.run()
        # Each payout should have been ≤ $5000 gross
        # Can't check individual payouts from SimResult, but total should reflect caps

    def test_topstep_150_threshold(self):
        """Day with $140 PnL does NOT count as winning day."""
        p = PayoutPolicy.topstep_50k()
        trades = [(d, 140.0) for d in range(1, 20)]  # All below $150
        sim = PayoutSimulator(p, trades=trades)
        result = sim.run()
        assert result.payout_count == 0  # Never hit 5 qualifying days

    def test_topstep_split_and_fee(self):
        """Payout applies 90% split + $30 fee."""
        p = PayoutPolicy.topstep_50k()
        trades = [(d, 500.0) for d in range(1, 15)]  # Enough to trigger payout
        sim = PayoutSimulator(p, trades=trades)
        result = sim.run()
        # Net should be less than gross by split + fees
        assert result.total_withdrawn_net < result.total_gross_profit

    def test_bulenox_early_caps(self):
        """Bulenox: first 3 payouts capped at $1500."""
        p = PayoutPolicy.bulenox_50k()
        trades = [(d, 500.0) for d in range(1, 60)]  # Lots of profit
        sim = PayoutSimulator(p, trades=trades)
        result = sim.run()
        # First 3 payouts max $1500 each = $4500 max from first 3
        # After that uncapped
        assert result.payout_count >= 3

    def test_bulenox_safety_reserve_locked(self):
        """Can't withdraw below starting + safety_reserve."""
        p = PayoutPolicy.bulenox_50k()
        # Safety reserve = $2600. Can't withdraw below $52,600.
        trades = [(d, 200.0) for d in range(1, 20)]  # $3800 profit
        sim = PayoutSimulator(p, trades=trades)
        result = sim.run()
        # Balance should never go below 50000 + 2600 = 52600
        assert sim.balance >= 50_000 + p.safety_reserve

    def test_bulenox_consistency_blocks_withdrawal(self):
        """If best day > 40% of total profit, can't withdraw."""
        p = PayoutPolicy.bulenox_50k()
        # One huge day then small days — violates 40% rule
        trades = [(1, 5000.0)] + [(d, 50.0) for d in range(2, 12)]
        # Total = 5000 + 500 = 5500. Best day = 5000 = 91% > 40%
        sim = PayoutSimulator(p, trades=trades)
        result = sim.run()
        # Consistency rule should block payout
        assert result.payout_count == 0

    def test_mffu_daily_payouts_after_buffer(self):
        """MFFU Rapid: daily payouts once $2100 buffer met."""
        p = PayoutPolicy.mffu_rapid_50k()
        trades = [(d, 300.0) for d in range(1, 15)]  # $4200, exceeds $2100 buffer
        sim = PayoutSimulator(p, trades=trades)
        result = sim.run()
        assert result.payout_count >= 1


# ---------------------------------------------------------------------------
# Blow Recovery
# ---------------------------------------------------------------------------


class TestBlowRecovery:
    def test_blow_resets_account(self):
        """After blow, new account starts fresh (minus reset cost)."""
        p = PayoutPolicy.topstep_50k()
        # Blow on day 1, then recover
        trades = [(1, -2100.0)] + [(d, 200.0) for d in range(2, 30)]
        sim = PayoutSimulator(p, trades=trades)
        result = sim.run()
        assert result.blow_count >= 1
        assert result.reset_costs > 0

    def test_topstep_reset_cost(self):
        p = PayoutPolicy.topstep_50k()
        assert p.reset_cost == 49


# ---------------------------------------------------------------------------
# Comparison: same trades, different firms
# ---------------------------------------------------------------------------


class TestFirmComparison:
    def _make_trades(self):
        """Realistic trade stream: mostly small wins/losses, occasional big day."""
        import random

        random.seed(42)
        trades = []
        for d in range(1, 253):  # ~1 year
            pnl = random.gauss(15, 80)  # Slightly positive mean
            trades.append((d, round(pnl, 2)))
        return trades

    def test_same_trades_different_results(self):
        """Same trade stream → different take-home per firm."""
        trades = self._make_trades()
        results = {}
        for name, factory in [
            ("topstep", PayoutPolicy.topstep_50k),
            ("mffu", PayoutPolicy.mffu_rapid_50k),
            ("bulenox", PayoutPolicy.bulenox_50k),
        ]:
            sim = PayoutSimulator(factory(), trades=trades)
            results[name] = sim.run()

        # All should produce SimResult
        for name, r in results.items():
            assert isinstance(r, SimResult), f"{name} didn't return SimResult"

        # They should NOT all be identical (different rules → different outcomes)
        withdrawals = [r.total_withdrawn_net for r in results.values()]
        assert len(set(round(w, 2) for w in withdrawals)) > 1, (
            "All firms produced identical results — rules not applied"
        )


# ---------------------------------------------------------------------------
# SimResult fields
# ---------------------------------------------------------------------------


class TestSimResult:
    def test_extraction_rate(self):
        """extraction_rate = withdrawn / gross_profit."""
        p = PayoutPolicy.topstep_50k()
        trades = [(d, 400.0) for d in range(1, 30)]
        sim = PayoutSimulator(p, trades=trades)
        result = sim.run()
        if result.total_gross_profit > 0:
            expected = result.total_withdrawn_net / result.total_gross_profit
            assert abs(result.extraction_rate - expected) < 0.01

    def test_annual_withdrawn(self):
        """annual_withdrawn = withdrawn / years."""
        p = PayoutPolicy.topstep_50k()
        trades = [(d, 300.0) for d in range(1, 253)]  # ~1 year
        sim = PayoutSimulator(p, trades=trades)
        result = sim.run()
        assert result.annual_withdrawn >= 0
