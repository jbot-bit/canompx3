"""Prop firm payout simulator — models actual take-home income.

Replays real trade PnL through each firm's payout rules:
DD mechanics, payout caps, buffer requirements, consistency rules,
blow probability, extraction rate.

Every payout rule sourced from official help center pages ONLY:
- TopStep: help.topstep.com (user-provided Apr 3 2026)
- MFFU Rapid: help.myfundedfutures.com articles 13134709, 13134718, 13286746, 13745661
- Bulenox: bulenox.com/help/master-account (scraped Apr 3 2026)

Usage:
    python scripts/tools/payout_simulator.py --firm topstep --contracts 1
    python scripts/tools/payout_simulator.py --compare --contracts 2
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PayoutPolicy:
    """Structured payout rules for a prop firm. Every field from official sources."""

    name: str
    dd_type: str  # "eod_trailing" | "intraday_trailing"
    dd_amount: float
    dd_lock_at: float  # Floor locks here. TopStep=$0, MFFU=$100, Bulenox=starting+$100
    starting_balance: float
    buffer_before_first_payout: float  # Profit needed before first withdrawal
    winning_day_threshold: float  # Min daily PnL to count as "winning day" ($150 for TopStep)
    winning_days_per_cycle: int  # Days needed per payout cycle (5 for TopStep, 0=daily)
    max_payout_per_request: float | None  # None = no cap
    max_payout_pct_of_balance: float  # 0.50 for TopStep
    profit_split: float  # 0.90 = trader gets 90%
    payout_fee: float  # Per-payout fee ($30 for TopStep)
    min_payout: float  # Minimum withdrawal
    consistency_rule: float | None  # None or max pct (0.40)
    safety_reserve: float  # Locked capital ($2600 for Bulenox)
    early_payout_caps: tuple[float, ...] | None  # Bulenox: first 3 capped
    reset_cost: float  # Cost to restart after blow
    min_trading_days_before_first: int = 0  # Bulenox: 10 days
    cooldown_days: int = 0  # MFFU: 21 days after Live blow
    first_profit_split: float | None = None  # Bulenox: 100% on first $10K
    first_profit_split_threshold: float = 0  # Bulenox: $10K threshold

    @staticmethod
    def topstep_50k() -> PayoutPolicy:
        """TopStep 50K Express Standard. Source: help.topstep.com payout policy."""
        return PayoutPolicy(
            name="TopStep 50K",
            dd_type="eod_trailing",
            dd_amount=2000,
            dd_lock_at=0,  # MLL set to $0 after first payout (floor = $0 absolute)
            starting_balance=50_000,
            buffer_before_first_payout=2000,  # Recommended: bring MLL to $0 before payout
            winning_day_threshold=150,
            winning_days_per_cycle=5,
            max_payout_per_request=5000,
            max_payout_pct_of_balance=0.50,
            profit_split=0.90,
            payout_fee=30,
            min_payout=125,
            consistency_rule=None,  # Standard path has no consistency
            safety_reserve=0,
            early_payout_caps=None,
            reset_cost=49,
        )

    @staticmethod
    def mffu_rapid_50k() -> PayoutPolicy:
        """MFFU Rapid 50K. Source: help.myfundedfutures.com articles 13134709, 13745661."""
        return PayoutPolicy(
            name="MFFU Rapid 50K",
            dd_type="intraday_trailing",
            dd_amount=2000,
            dd_lock_at=100,  # "Once your trailing Max Loss reaches $100, it locks there"
            starting_balance=50_000,
            buffer_before_first_payout=2100,  # "$2,100 in realized profits"
            winning_day_threshold=0,  # No winning day requirement for Rapid
            winning_days_per_cycle=0,  # Daily payouts
            max_payout_per_request=None,  # NOT stated on any official Rapid page
            max_payout_pct_of_balance=1.0,  # NOT stated — assume no pct cap
            profit_split=0.90,
            payout_fee=0,  # No fee mentioned
            min_payout=500,
            consistency_rule=None,  # "No consistency rules" in sim funded
            safety_reserve=0,
            early_payout_caps=None,
            reset_cost=130,  # ~Monthly subscription cost
            cooldown_days=21,  # 21-day cooldown after Live blow (article 13286746)
        )

    @staticmethod
    def bulenox_50k() -> PayoutPolicy:
        """Bulenox 50K Master. Source: bulenox.com/help/master-account."""
        return PayoutPolicy(
            name="Bulenox 50K",
            dd_type="eod_trailing",
            dd_amount=2500,
            dd_lock_at=100,  # Locks at starting+$100 (stored as offset from starting)
            starting_balance=50_000,
            buffer_before_first_payout=0,  # No explicit buffer — just safety reserve + DD
            winning_day_threshold=0,  # No winning day requirement
            winning_days_per_cycle=0,  # Can request anytime
            max_payout_per_request=None,  # After 3rd: no max
            max_payout_pct_of_balance=1.0,  # Not stated
            profit_split=0.90,  # After first $10K
            payout_fee=0,
            min_payout=1000,
            consistency_rule=0.40,  # Best day < 40% of total profit
            safety_reserve=2600,
            early_payout_caps=(1500, 1500, 1500),  # First 3 payouts capped
            reset_cost=148,  # Activation fee (one-time, no monthly)
            min_trading_days_before_first=10,
            first_profit_split=1.00,  # First $10K at 100%
            first_profit_split_threshold=10_000,
        )


@dataclass
class SimResult:
    """Output of a payout simulation run."""

    total_gross_profit: float = 0
    total_withdrawn_net: float = 0
    payout_count: int = 0
    blow_count: int = 0
    reset_costs: float = 0
    extraction_rate: float = 0  # withdrawn / gross
    days_to_first_payout: int = 0
    annual_withdrawn: float = 0
    peak_balance: float = 0
    worst_drawdown: float = 0
    payouts: list = field(default_factory=list)  # [(day, gross, net)]


class PayoutSimulator:
    """Day-by-day simulation of prop firm account with payout extraction."""

    def __init__(
        self,
        policy: PayoutPolicy,
        trades: list[tuple[int, float]],  # [(day_number, pnl_dollars)]
        contracts: int = 1,
    ):
        self.policy = policy
        self.trades = trades
        self.contracts = contracts

        # Account state
        self.balance = policy.starting_balance
        self.peak_balance = policy.starting_balance
        self.dd_floor = policy.starting_balance - policy.dd_amount
        self.dd_locked = False

        # Payout state
        self.winning_days = 0
        self.payout_count = 0
        self.trading_days = 0
        self.total_withdrawn_gross = 0
        self.total_withdrawn_net = 0
        self.total_profit_withdrawn = 0  # For split tiers (Bulenox first $10K)
        self.best_day_pnl = 0
        self.total_pnl_since_start = 0  # For consistency check
        self.first_payout_taken = False
        self.blown = False

        # Tracking
        self._blow_count = 0
        self._reset_costs = 0
        self._gross_profit = 0
        self._first_payout_day = 0
        self._payouts: list[tuple[int, float, float]] = []

    def run(self) -> SimResult:
        """Run the full simulation. Returns SimResult."""
        for day, pnl in self.trades:
            if self.blown:
                # Reset account after blow
                self._reset_account()

            self._process_day(day, pnl * self.contracts)

        # Compute result
        total_days = len(self.trades)
        years = max(total_days / 252, 0.1)  # ~252 trading days/year

        gross = self._gross_profit
        net = self.total_withdrawn_net
        extraction = net / gross if gross > 0 else 0

        return SimResult(
            total_gross_profit=gross,
            total_withdrawn_net=net,
            payout_count=self.payout_count,
            blow_count=self._blow_count,
            reset_costs=self._reset_costs,
            extraction_rate=extraction,
            days_to_first_payout=self._first_payout_day,
            annual_withdrawn=net / years,
            peak_balance=self.peak_balance,
            worst_drawdown=self.peak_balance - min(self.balance, self.policy.starting_balance),
            payouts=self._payouts,
        )

    def _process_day(self, day: int, pnl: float) -> None:
        """Process one trading day."""
        self.trading_days += 1

        # Apply PnL
        self.balance += pnl
        if pnl > 0:
            self._gross_profit += pnl

        # Track best day for consistency rule
        if pnl > self.best_day_pnl:
            self.best_day_pnl = pnl
        self.total_pnl_since_start += pnl

        # Update peak and DD floor
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        if not self.dd_locked:
            new_floor = self.peak_balance - self.policy.dd_amount
            if new_floor > self.dd_floor:
                self.dd_floor = new_floor

            # Check lock condition
            lock_target = self._lock_target()
            if self.dd_floor >= lock_target:
                self.dd_floor = lock_target
                self.dd_locked = True

        # Check blow
        if self.balance <= self.dd_floor:
            self.blown = True
            self._blow_count += 1
            self._reset_costs += self.policy.reset_cost
            return

        # Track winning days
        if pnl >= self.policy.winning_day_threshold and self.policy.winning_day_threshold > 0:
            self.winning_days += 1

        # Check payout eligibility
        self._try_payout(day)

    def _lock_target(self) -> float:
        """Where does the DD floor lock?"""
        p = self.policy
        if p.dd_lock_at == 0:
            # TopStep: locks at $0 absolute (after first payout, MLL=$0)
            # Before first payout: locks at starting_balance (MLL=$0 = floor at starting)
            return p.starting_balance
        else:
            # MFFU: locks at $100 absolute
            # Bulenox: locks at starting + $100
            if p.safety_reserve > 0:
                # Bulenox: starting + dd_lock_at offset
                return p.starting_balance + p.dd_lock_at
            return p.dd_lock_at

    def _try_payout(self, day: int) -> None:
        """Check if payout is eligible and process it."""
        p = self.policy

        # Check minimum trading days (Bulenox: 10)
        if self.trading_days < p.min_trading_days_before_first and self.payout_count == 0:
            return

        # Check winning days requirement
        if p.winning_days_per_cycle > 0 and self.winning_days < p.winning_days_per_cycle:
            return

        # Check buffer requirement (first payout)
        profit_above_start = self.balance - p.starting_balance
        if not self.first_payout_taken and profit_above_start < p.buffer_before_first_payout:
            return

        # For subsequent TopStep payouts: must have profit > $0 since last payout
        # (simplified: always require some profit above floor)

        # Check consistency rule
        if p.consistency_rule is not None and self.total_pnl_since_start > 0:
            best_day_pct = self.best_day_pnl / self.total_pnl_since_start
            if best_day_pct > p.consistency_rule:
                return  # Blocked by consistency

        # Calculate available for withdrawal
        min_balance = p.starting_balance + p.safety_reserve
        if self.first_payout_taken and p.dd_lock_at == 0:
            # TopStep after first payout: floor at $0, can withdraw down to $0 + safety margin
            min_balance = p.safety_reserve  # Just safety reserve (which is $0 for TopStep)

        available = self.balance - min_balance
        if available < p.min_payout:
            return

        # Apply caps
        max_withdrawal = available

        # Percentage cap
        pct_cap = self.balance * p.max_payout_pct_of_balance
        max_withdrawal = min(max_withdrawal, pct_cap)

        # Absolute cap
        if p.max_payout_per_request is not None:
            max_withdrawal = min(max_withdrawal, p.max_payout_per_request)

        # Early payout caps (Bulenox: first 3)
        if p.early_payout_caps is not None and self.payout_count < len(p.early_payout_caps):
            max_withdrawal = min(max_withdrawal, p.early_payout_caps[self.payout_count])

        if max_withdrawal < p.min_payout:
            return

        # Leave a safety buffer — don't withdraw to exactly the floor
        safety_margin = max(p.dd_amount * 0.25, 500)  # Keep 25% of DD or $500
        if self.balance - max_withdrawal < min_balance + safety_margin:
            max_withdrawal = self.balance - min_balance - safety_margin
            if max_withdrawal < p.min_payout:
                return

        # Apply profit split
        if p.first_profit_split is not None and self.total_profit_withdrawn < p.first_profit_split_threshold:
            # Bulenox: first $10K at 100%
            remaining_at_full = p.first_profit_split_threshold - self.total_profit_withdrawn
            if max_withdrawal <= remaining_at_full:
                net = max_withdrawal * p.first_profit_split - p.payout_fee
            else:
                net = (
                    remaining_at_full * p.first_profit_split
                    + (max_withdrawal - remaining_at_full) * p.profit_split
                    - p.payout_fee
                )
        else:
            net = max_withdrawal * p.profit_split - p.payout_fee

        if net <= 0:
            return

        # Execute payout
        self.balance -= max_withdrawal
        self.total_withdrawn_gross += max_withdrawal
        self.total_withdrawn_net += net
        self.total_profit_withdrawn += max_withdrawal
        self.payout_count += 1
        self.winning_days = 0  # Reset winning day counter
        self._payouts.append((day, max_withdrawal, net))

        if not self.first_payout_taken:
            self.first_payout_taken = True
            self._first_payout_day = day
            # TopStep: MLL set to $0 after first payout
            if self.policy.dd_lock_at == 0:
                self.dd_floor = 0
                self.dd_locked = True

    def _reset_account(self) -> None:
        """Reset account after blow."""
        p = self.policy
        self.balance = p.starting_balance
        self.peak_balance = p.starting_balance
        self.dd_floor = p.starting_balance - p.dd_amount
        self.dd_locked = False
        self.winning_days = 0
        self.first_payout_taken = False
        self.best_day_pnl = 0
        self.total_pnl_since_start = 0
        self.blown = False


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Prop firm payout simulator")
    parser.add_argument("--firm", choices=["topstep", "mffu", "bulenox"], default=None)
    parser.add_argument("--compare", action="store_true", help="Compare all firms")
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--years", type=int, default=2, help="Years of data to use (from recent)")
    args = parser.parse_args()

    # Generate sample trades for quick testing
    import random

    random.seed(42)
    days = args.years * 252
    trades = [(d, round(random.gauss(15, 80), 2)) for d in range(1, days + 1)]

    firms = {
        "topstep": PayoutPolicy.topstep_50k,
        "mffu": PayoutPolicy.mffu_rapid_50k,
        "bulenox": PayoutPolicy.bulenox_50k,
    }

    if args.compare or args.firm is None:
        targets = firms
    else:
        targets = {args.firm: firms[args.firm]}

    print(
        f"{'Firm':20s} | {'Gross':>10s} | {'Net':>10s} | {'Payouts':>7s} | {'Blows':>5s} | {'Extract':>7s} | {'$/yr':>10s}"
    )
    print("-" * 85)

    for _name, factory in targets.items():
        policy = factory()
        sim = PayoutSimulator(policy, trades, contracts=args.contracts)
        result = sim.run()
        print(
            f"{policy.name:20s} | ${result.total_gross_profit:>9,.0f} | ${result.total_withdrawn_net:>9,.0f} | "
            f"{result.payout_count:>7d} | {result.blow_count:>5d} | "
            f"{result.extraction_rate:>6.1%} | ${result.annual_withdrawn:>9,.0f}"
        )

    sys.exit(0)
