"""
Risk management for live/replay trading.

Enforces position limits, daily loss circuit breakers, and per-ORB
concentration limits. Designed to work with ExecutionEngine.

Usage:
    limits = RiskLimits(max_daily_loss_r=-5.0, max_concurrent_positions=3)
    rm = RiskManager(limits)
    allowed, reason = rm.can_enter(trade, active_trades, daily_pnl_r)
"""

import logging
from dataclasses import dataclass, replace
from datetime import date

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskLimits:
    """Immutable risk parameters for a trading session."""

    max_daily_loss_r: float = -5.0  # Circuit breaker threshold (R)
    max_concurrent_positions: int = 3  # Max open positions at once
    max_per_orb_positions: int = 1  # Max positions from same ORB (aperture-specific when orb_minutes provided)
    max_per_session_positions: int = 2  # Max positions from same session across ALL apertures
    max_daily_trades: int = 15  # Max entries per day
    drawdown_warning_r: float = -3.0  # Log warning threshold (R)
    corr_threshold_for_reduction: float = 0.5  # Correlation above this triggers size reduction
    min_correlation_factor: float = (
        0.3  # Min assumed correlation if not in lookup, used in effective exposure calculation
    )
    max_equity_drawdown_r: float | None = None  # Multi-day drawdown limit (R from peak). None = disabled.

    # F-1 TopStep Scaling Plan enforcement.
    # @canonical-source docs/research-input/topstep/topstep_scaling_plan_article.md
    # @canonical-image docs/research-input/topstep/images/xfa_scaling_chart.png
    # Set to 50_000 / 100_000 / 150_000 only when the bot is connected to a
    # TopStep Express Funded Account. Default None disables the check (e.g.
    # for non-TopStep deployments or Trading Combine practice accounts).
    topstep_xfa_account_size: int | None = None


class RiskManager:
    """
    Enforces risk limits for the execution engine.

    Risk checks are ordered fail-fast:
    1. Circuit breaker (daily loss limit)
    2. Hedging guard (F-2)
    3. TopStep Scaling Plan (F-1) — only when topstep_xfa_account_size is set
    4. Max concurrent positions
    5. Max per ORB positions
    6. Max daily trades
    7. Drawdown warning (allows entry, logs warning)
    """

    def __init__(self, limits: RiskLimits, corr_lookup: dict[tuple[str, str], float] | None = None):
        self.limits = limits
        self._corr_lookup = corr_lookup or {}
        self.daily_pnl_r: float = 0.0
        self.daily_trade_count: int = 0
        self.trading_day: date | None = None
        self._halted: bool = False
        self._warnings: list[str] = []
        # Multi-day equity tracking (persists across daily_reset)
        self.cumulative_pnl_r: float = 0.0
        self.equity_high_water_r: float = 0.0
        self._equity_halted: bool = False
        # F-1 TopStep Scaling Plan: EOD balance for the active XFA. Set by the
        # orchestrator at session start (and after each EOD rollover) by reading
        # broker equity from the HWM tracker. None means "not yet known" — the
        # check fails-closed when balance is required.
        self._topstep_xfa_eod_balance: float | None = None

    def set_topstep_xfa_eod_balance(self, balance: float) -> None:
        """Set the latest end-of-day XFA balance for Scaling Plan enforcement.

        @canonical-source docs/research-input/topstep/topstep_scaling_plan_article.md
        @verbatim "Your maximum number of contracts allowed to trade under the
                   scaling plan does not increase throughout the trading day."

        Called by the orchestrator at session start (and after each EOD
        rollover). The check uses this value, NOT the live intraday equity,
        because the canonical rule prohibits intraday scaling-up.
        """
        self._topstep_xfa_eod_balance = balance

    def disable_f1(self, reason: str) -> None:
        """Disable F-1 TopStep XFA Scaling Plan enforcement at runtime.

        Used when the broker reports a non-XFA account (e.g. Trading Combine)
        despite the profile config claiming XFA. Replaces the immutable
        RiskLimits with a new frozen instance where topstep_xfa_account_size
        is None, so can_enter() skips the F-1 check.

        Idempotent. No-op if F-1 is already disabled.
        """
        if self.limits.topstep_xfa_account_size is None:
            return
        log.warning(
            "F-1 TopStep XFA Scaling Plan DISABLED by broker-reality check: %s",
            reason,
        )
        self.limits = replace(self.limits, topstep_xfa_account_size=None)
        self._topstep_xfa_eod_balance = None

    def daily_reset(self, trading_day: date) -> None:
        """Reset daily counters for a new trading day.

        Multi-day equity tracking (cumulative_pnl_r, equity_high_water_r,
        _equity_halted) persists across resets. Use equity_reset() to
        start a fresh simulation.
        """
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
        orb_minutes: int | None = None,
        instrument: str | None = None,
        direction: str | None = None,
    ) -> tuple[bool, str, float]:  # Added float to return type
        suggested_contract_factor = 1.0

        # Check 0: Multi-day equity drawdown
        if self._equity_halted:
            drawdown = self.cumulative_pnl_r - self.equity_high_water_r
            return (
                False,
                f"equity_drawdown: {drawdown:.2f}R from peak (limit {self.limits.max_equity_drawdown_r}R)",
                0.0,
            )

        # Check 1: Circuit breaker
        if self._halted or daily_pnl_r <= self.limits.max_daily_loss_r:
            self._halted = True
            return False, f"circuit_breaker: daily PnL {daily_pnl_r:.2f}R <= {self.limits.max_daily_loss_r}R", 0.0

        # Check 1b: F-2 same-instrument opposite-direction guard.
        # @canonical-source docs/research-input/topstep/topstep_cross_account_hedging.md
        # @verbatim "Cross-account hedging occurs when you hold opposite positions
        #            across multiple accounts at the same time. This means you're
        #            simultaneously long and short the same instrument (or highly
        #            correlated/fungible instruments)."
        # @verbatim "Yes! You can trade the same instrument across multiple accounts.
        #            What's prohibited is holding opposite positions simultaneously."
        # @audit-finding F-2 BLOCKER — refuses entries that would create an opposing
        # position on the same instrument within the same account. CopyOrderRouter
        # mirrors trades across copies, so an intra-account hedge would also be a
        # cross-account hedge → 3rd offense = PERMANENT account closure.
        #
        # The check is opt-in (instrument and direction default None for backward
        # compat). When both are provided, the function scans active_trades for any
        # ENTERED trade on the SAME instrument with OPPOSITE direction.
        if instrument is not None and direction is not None:
            opposite_direction = "short" if direction.lower() == "long" else "long"
            for t in active_trades:
                if not hasattr(t, "state") or t.state.value != "ENTERED":
                    continue
                # Read instrument from trade.strategy.instrument (the canonical path)
                # or fall back to a top-level trade.instrument attribute if present.
                t_instrument = None
                if hasattr(t, "strategy") and hasattr(t.strategy, "instrument"):
                    t_instrument = t.strategy.instrument
                elif hasattr(t, "instrument"):
                    t_instrument = t.instrument
                if t_instrument != instrument:
                    continue
                t_direction = getattr(t, "direction", None)
                if t_direction is None:
                    continue
                if t_direction.lower() == opposite_direction:
                    return (
                        False,
                        (
                            f"hedging_guard: cannot enter {direction.upper()} {instrument} — "
                            f"existing {t_direction.upper()} position on same instrument "
                            f"({t.strategy_id}). Cross-account hedging is prohibited "
                            f"(F-2 / TopStep CME Rule 534)."
                        ),
                        0.0,
                    )

        # Check 1c: F-1 TopStep XFA Scaling Plan enforcement.
        # @canonical-source docs/research-input/topstep/topstep_scaling_plan_article.md
        # @canonical-image docs/research-input/topstep/images/xfa_scaling_chart.png
        # @verbatim "Your maximum number of contracts allowed to trade under the
        #            scaling plan does not increase throughout the trading day. If
        #            your earnings meet or exceed the required amount to scale up,
        #            you still need to wait until the following session to trade
        #            the next Scaling Plan level."
        # @verbatim "Errors in the Scaling Plan corrected in less than 10 seconds
        #            will be ignored. If traders leave on too many contracts for
        #            10 seconds or more, even if only by a few seconds, their
        #            account may be reviewed."
        # @audit-finding F-1 BLOCKER — refuses entries that would push the bot's
        # net mini-equivalent exposure above the day's Scaling Plan tier.
        #
        # Only fires when the bot is connected to a TopStep XFA (i.e., when
        # limits.topstep_xfa_account_size is set). The orchestrator must call
        # set_topstep_xfa_eod_balance() at session start with the broker's
        # reported balance — otherwise the check fails-closed (no balance
        # known → no entries allowed).
        if self.limits.topstep_xfa_account_size is not None and instrument is not None:
            from trading_app.topstep_scaling_plan import (
                max_lots_for_xfa,
                project_total_open_lots,
            )

            if self._topstep_xfa_eod_balance is None:
                return (
                    False,
                    (
                        "topstep_scaling_plan: EOD XFA balance unknown — refusing entry. "
                        "Orchestrator must call set_topstep_xfa_eod_balance() at session start. "
                        "(F-1 fail-closed)"
                    ),
                    0.0,
                )

            try:
                day_max = max_lots_for_xfa(
                    self.limits.topstep_xfa_account_size, self._topstep_xfa_eod_balance
                )
            except (KeyError, ValueError) as e:
                return (
                    False,
                    f"topstep_scaling_plan: ladder lookup failed: {e} (F-1 fail-closed)",
                    0.0,
                )

            # Project total exposure assuming the new entry lands with 1 contract.
            # project_total_open_lots aggregates contracts per instrument BEFORE
            # applying the micro-to-mini ceiling, matching the canonical rule
            # "2 lots = 20 micros = any combination summing to 2 mini-equivalents".
            # See docs/audit/2026-04-11-criterion-11-f1-false-alarm.md for the
            # false-alarm audit that this fix closes.
            #
            # Note: the execution engine sizes the actual position AFTER this
            # check. If a larger size would push exposure above day_max, the
            # engine's pre-submit guard must re-check with the real contract
            # count. For the common 1-contract-per-lane case this projection
            # is exact.
            projected = project_total_open_lots(
                active_trades, instrument, new_contracts=1
            )

            if projected > day_max:
                return (
                    False,
                    (
                        f"topstep_scaling_plan: projected {projected} mini-equiv lots > "
                        f"day_max {day_max} for {self.limits.topstep_xfa_account_size//1000}K XFA "
                        f"at EOD balance ${self._topstep_xfa_eod_balance:,.2f}. "
                        f"(F-1 — TopStep Scaling Plan ladder)"
                    ),
                    0.0,
                )

        # Check 2: Max concurrent positions (correlation-weighted if available)
        entered = [t for t in active_trades if hasattr(t, "state") and t.state.value == "ENTERED"]
        if self._corr_lookup and entered:
            # Calculate effective exposure from correlations
            effective_exposure = 0.0
            for t in entered:
                corr_val = self._corr_lookup.get(
                    (strategy_id, t.strategy_id), self._corr_lookup.get((t.strategy_id, strategy_id), None)
                )
                if corr_val is None:
                    corr_val = self.limits.min_correlation_factor

                # Use max of corr_val and limits.min_correlation_factor to avoid over-discounting strong negative correlations
                effective_exposure += max(corr_val, self.limits.min_correlation_factor)

            # Add min_correlation_factor for the new trade (conservative: assumes at least
            # this much correlation with existing positions, even if uncorrelated).
            # If effective exposure exceeds max_concurrent_positions, reject.
            if (effective_exposure + self.limits.min_correlation_factor) > self.limits.max_concurrent_positions:
                return (
                    False,
                    f"corr_concurrent: effective exposure {effective_exposure + self.limits.min_correlation_factor:.1f} >= {self.limits.max_concurrent_positions}",
                    0.0,
                )

            # If effective exposure is high but within limits, suggest a reduced contract factor
            # This is a heuristic that can be refined. For now, a simple linear reduction
            if effective_exposure > self.limits.corr_threshold_for_reduction * self.limits.max_concurrent_positions:
                reduction_factor = (
                    1.0 - (effective_exposure / self.limits.max_concurrent_positions) * 0.5
                )  # Example reduction
                suggested_contract_factor = max(0.1, reduction_factor)  # Ensure not too small

        else:
            if len(entered) >= self.limits.max_concurrent_positions:
                return False, f"max_concurrent: {len(entered)} >= {self.limits.max_concurrent_positions}", 0.0

        # Check 3: Max per ORB (aperture-specific when orb_minutes provided)
        # When orb_minutes is given, O5 and O30 on the same session are different ORBs.
        orb_count = sum(
            1
            for t in active_trades
            if hasattr(t, "orb_label")
            and t.orb_label == orb_label
            and (orb_minutes is None or not hasattr(t, "orb_minutes") or t.orb_minutes == orb_minutes)
            and hasattr(t, "state")
            and t.state.value == "ENTERED"
        )
        if orb_count >= self.limits.max_per_orb_positions:
            return False, f"max_per_orb: {orb_count} positions on {orb_label}", 0.0

        # Check 3b: Max per session across ALL apertures
        session_count = sum(
            1
            for t in active_trades
            if hasattr(t, "orb_label")
            and t.orb_label == orb_label
            and hasattr(t, "state")
            and t.state.value == "ENTERED"
        )
        if session_count >= self.limits.max_per_session_positions:
            return (
                False,
                f"max_per_session: {session_count} on {orb_label}",
                0.0,
            )

        # Check 4: Max daily trades
        if self.daily_trade_count >= self.limits.max_daily_trades:
            return False, f"max_daily_trades: {self.daily_trade_count} >= {self.limits.max_daily_trades}", 0.0

        # Check 5: Drawdown warning (allow, but record warning)
        if daily_pnl_r <= self.limits.drawdown_warning_r:
            msg = f"drawdown_warning: daily PnL {daily_pnl_r:.2f}R <= {self.limits.drawdown_warning_r}R"
            self._warnings.append(msg)
            log.warning("RiskManager [%s]: %s", strategy_id, msg)

        # Check 6: Chop awareness (warn only, does not block)
        if market_state is not None and hasattr(market_state, "signals"):
            if market_state.signals.chop_detected:
                msg = f"chop_warning: chop detected for {strategy_id} on {orb_label}"
                self._warnings.append(msg)
                log.warning("RiskManager: %s", msg)

        # Check 7: Concurrent same-session different-aperture sizing
        # When an O5 trade is active and O30 enters (or vice versa), suggest half-size
        # to limit correlated same-direction exposure.
        if orb_minutes is not None:
            same_session_diff_aperture = sum(
                1
                for t in active_trades
                if hasattr(t, "orb_label")
                and t.orb_label == orb_label
                and hasattr(t, "orb_minutes")
                and t.orb_minutes != orb_minutes
                and hasattr(t, "state")
                and t.state.value == "ENTERED"
            )
            if same_session_diff_aperture > 0:
                suggested_contract_factor = min(suggested_contract_factor, 0.5)

        return True, "", suggested_contract_factor

    def on_trade_entry(self) -> None:
        """Record a new trade entry."""
        self.daily_trade_count += 1

    def on_trade_exit(self, pnl_r: float) -> None:
        """Update daily PnL after a trade exits."""
        self.daily_pnl_r += pnl_r
        if self.daily_pnl_r <= self.limits.max_daily_loss_r:
            self._halted = True
        # Multi-day equity tracking
        self.cumulative_pnl_r += pnl_r
        if self.cumulative_pnl_r > self.equity_high_water_r:
            self.equity_high_water_r = self.cumulative_pnl_r
        if self.limits.max_equity_drawdown_r is not None and not self._equity_halted:
            drawdown = self.cumulative_pnl_r - self.equity_high_water_r
            if drawdown <= self.limits.max_equity_drawdown_r:
                self._equity_halted = True

    def equity_reset(self) -> None:
        """Reset multi-day equity tracking. Call at start of a new simulation."""
        self.cumulative_pnl_r = 0.0
        self.equity_high_water_r = 0.0
        self._equity_halted = False

    def is_halted(self) -> bool:
        """True if daily circuit breaker or equity drawdown limit has been triggered."""
        return self._halted or self._equity_halted

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
            "cumulative_pnl_r": round(self.cumulative_pnl_r, 4),
            "equity_high_water_r": round(self.equity_high_water_r, 4),
            "equity_drawdown_r": round(self.cumulative_pnl_r - self.equity_high_water_r, 4),
            "equity_halted": self._equity_halted,
        }
