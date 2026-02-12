"""
Risk management for live/replay trading.

Enforces position limits, daily loss circuit breakers, and per-ORB
concentration limits. Designed to work with ExecutionEngine.

Usage:
    limits = RiskLimits(max_daily_loss_r=-5.0, max_concurrent_positions=3)
    rm = RiskManager(limits)
    allowed, reason = rm.can_enter(trade, active_trades, daily_pnl_r)
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class RiskLimits:
    """Immutable risk parameters for a trading session."""
    max_daily_loss_r: float = -5.0          # Circuit breaker threshold (R)
    max_concurrent_positions: int = 3       # Max open positions at once
    max_per_orb_positions: int = 1          # Max positions from same ORB
    max_daily_trades: int = 15              # Max entries per day
    drawdown_warning_r: float = -3.0        # Log warning threshold (R)
    corr_threshold_for_reduction: float = 0.5 # Correlation above this triggers size reduction
    min_correlation_factor: float = 0.3     # Min assumed correlation if not in lookup, used in effective exposure calculation


class RiskManager:
    """
    Enforces risk limits for the execution engine.

    Risk checks are ordered fail-fast:
    1. Circuit breaker (daily loss limit)
    2. Max concurrent positions
    3. Max per ORB positions
    4. Max daily trades
    5. Drawdown warning (allows entry, logs warning)
    """

    def __init__(self, limits: RiskLimits,
                 corr_lookup: dict[tuple[str, str], float] | None = None):
        self.limits = limits
        self._corr_lookup = corr_lookup or {}
        self.daily_pnl_r: float = 0.0
        self.daily_trade_count: int = 0
        self.trading_day: date | None = None
        self._halted: bool = False
        self._warnings: list[str] = []

    def daily_reset(self, trading_day: date) -> None:
        """Reset all daily counters for a new trading day."""
        self.daily_pnl_r = 0.0
        self.daily_trade_count = 0
        self.trading_day = trading_day
        self._halted = False
        self._warnings = []

    def can_enter(
        self,
        strategy_id: str,
        orb_label: str,
        active_trades: list,
        daily_pnl_r: float,
        market_state=None,
    ) -> tuple[bool, str, float]: # Added float to return type

        suggested_contract_factor = 1.0

        # Check 1: Circuit breaker
        if self._halted or daily_pnl_r <= self.limits.max_daily_loss_r:
            self._halted = True
            return False, f"circuit_breaker: daily PnL {daily_pnl_r:.2f}R <= {self.limits.max_daily_loss_r}R", 0.0

        # Check 2: Max concurrent positions (correlation-weighted if available)
        entered = [t for t in active_trades if hasattr(t, 'state') and t.state.value == "ENTERED"]
        if self._corr_lookup and entered:
            # Calculate effective exposure from correlations
            effective_exposure = 0.0
            for t in entered:
                corr_val = self._corr_lookup.get(
                    (strategy_id, t.strategy_id),
                    self._corr_lookup.get((t.strategy_id, strategy_id), None)
                )
                if corr_val is None:
                    # If correlation is not found, assume a default (e.g., limits.default_correlation_assumption)
                    # For now, let's use a conservative default if not found
                    corr_val = 0.5 # Placeholder, will be replaced with configurable default
                
                # Use max of corr_val and limits.min_correlation_factor to avoid over-discounting strong negative correlations
                effective_exposure += max(corr_val, self.limits.min_correlation_factor) 
            
            # Add 1.0 for the current strategy (if allowed, it will take 1 position)
            # This sum should not exceed max_concurrent_positions
            
            # If the effective exposure plus the current strategy's "weight" exceeds the limit, reject
            if (effective_exposure + self.limits.min_correlation_factor) > self.limits.max_concurrent_positions:
                return False, f"corr_concurrent: effective exposure {effective_exposure + self.limits.min_correlation_factor:.1f} >= {self.limits.max_concurrent_positions}", 0.0

            # If effective exposure is high but within limits, suggest a reduced contract factor
            # This is a heuristic that can be refined. For now, a simple linear reduction
            if effective_exposure > self.limits.corr_threshold_for_reduction * self.limits.max_concurrent_positions:
                reduction_factor = 1.0 - (effective_exposure / self.limits.max_concurrent_positions) * 0.5 # Example reduction
                suggested_contract_factor = max(0.1, reduction_factor) # Ensure not too small

        else:
            if len(entered) >= self.limits.max_concurrent_positions:
                return False, f"max_concurrent: {len(entered)} >= {self.limits.max_concurrent_positions}", 0.0

        # Check 3: Max per ORB
        orb_count = sum(
            1 for t in active_trades
            if hasattr(t, 'orb_label') and t.orb_label == orb_label
            and hasattr(t, 'state') and t.state.value == "ENTERED"
        )
        if orb_count >= self.limits.max_per_orb_positions:
            return False, f"max_per_orb: {orb_count} positions on {orb_label}", 0.0

        # Check 4: Max daily trades
        if self.daily_trade_count >= self.limits.max_daily_trades:
            return False, f"max_daily_trades: {self.daily_trade_count} >= {self.limits.max_daily_trades}", 0.0

        # Check 5: Drawdown warning (allow, but record warning)
        if daily_pnl_r <= self.limits.drawdown_warning_r:
            self._warnings.append(
                f"drawdown_warning: daily PnL {daily_pnl_r:.2f}R <= {self.limits.drawdown_warning_r}R"
            )

        # Check 6: Chop awareness (warn only, does not block)
        if market_state is not None and hasattr(market_state, 'signals'):
            if market_state.signals.chop_detected:
                self._warnings.append(
                    f"chop_warning: chop detected for {strategy_id} on {orb_label}"
                )

        return True, "", suggested_contract_factor



    def on_trade_entry(self) -> None:
        """Record a new trade entry."""
        self.daily_trade_count += 1

    def on_trade_exit(self, pnl_r: float) -> None:
        """Update daily PnL after a trade exits."""
        self.daily_pnl_r += pnl_r
        if self.daily_pnl_r <= self.limits.max_daily_loss_r:
            self._halted = True

    def is_halted(self) -> bool:
        """True if circuit breaker has been triggered."""
        return self._halted

    @property
    def warnings(self) -> list[str]:
        """Return any warnings generated during the session."""
        return list(self._warnings)

    def get_status(self) -> dict:
        """Return current risk state."""
        return {
            "trading_day": self.trading_day,
            "daily_pnl_r": round(self.daily_pnl_r, 4),
            "daily_trade_count": self.daily_trade_count,
            "halted": self._halted,
            "warnings": len(self._warnings),
        }
