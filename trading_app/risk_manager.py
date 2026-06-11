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
    # True dollar-denominated daily-loss circuit breaker. None = disabled (R cap
    # only). When set (positive dollars), the breaker halts when cumulative
    # realized daily P&L <= -max_daily_loss_dollars, independent of R. Per-account
    # semantic: the engine accrues ONE account's realized dollars; CopyOrderRouter
    # mirrors to shadows, so each account protects its own broker MLL.
    # @canonical-source docs/specs/daily_loss_dollar_cap.md
    # Sized from real-2026 risk distribution + Carver Table 20 (≤25% of MLL).
    max_daily_loss_dollars: float | None = None
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
        # Cumulative realized daily P&L in dollars (for the dollar circuit
        # breaker). Only accrues when on_trade_exit receives a non-None
        # pnl_dollars; unknown-dollar exits contribute nothing (fail-safe: the
        # R cap still governs). Reset each daily_reset.
        #
        # PRIMARY-ACCOUNT scalar. This is the single-account belt and stays the
        # public API every existing reader uses (session_orchestrator crash
        # recovery, get_status, tests). The per-account map below mirrors it for
        # the primary and adds independent belts for shadow accounts when a
        # contract map is configured (Stage 2 — copies>1). Unconfigured = inert:
        # the dict carries only the primary and behaves byte-identically.
        self.daily_pnl_dollars: float = 0.0
        # Stage 2 — per-account (account-keyed) MODELED daily-loss belts.
        # Configured by the orchestrator at session start via
        # configure_accounts({account_id: contracts}). Each account accrues its
        # own MODELED realized dollars (primary pnl_dollars scaled by its
        # contract ratio) and halts independently. Empty = single-account path
        # (the scalar above governs; these maps stay unused). The primary's id,
        # captured at configure time, is the contract-ratio denominator and the
        # default account for can_enter / on_trade_exit when account_id is None.
        self._account_contracts: dict[int, int] = {}
        self._account_pnl_dollars: dict[int, float] = {}
        self._account_halted: dict[int, bool] = {}
        self._primary_account_id: int | None = None
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

    def configure_accounts(
        self,
        account_contracts: dict[int, int],
        primary_account_id: int,
        restored_pnl_dollars: dict[int, float] | None = None,
    ) -> None:
        """Stage 2 — enable per-account MODELED daily-loss belts.

        `account_contracts` maps every live account_id (primary + shadows) to the
        contract count it trades. `primary_account_id` is the account the engine's
        single position represents — its contract count is the ratio denominator,
        and it is the default account for can_enter / on_trade_exit.

        Each account then accrues its OWN modeled realized dollars:
        `pnl_dollars × (account_contracts / primary_contracts)`, and halts
        INDEPENDENTLY at -max_daily_loss_dollars. When all accounts trade the
        primary's contract count (today's 1:1 CopyOrderRouter mirror), every
        account charges the same dollars and halts together — correct for
        copies=1/mirror. It diverges the moment Stage 3 gives accounts different
        contract counts.

        Single-account / mirror semantics are byte-identical to the pre-Stage-2
        scalar path: configuring {primary: c} alone makes `_account_pnl_dollars`
        track `daily_pnl_dollars` exactly.

        `restored_pnl_dollars` (crash recovery) seeds each account's belt with its
        OWN persisted modeled dollars so a same-day restart re-derives each
        account's halt — not just the primary's. When None/empty, only the
        primary belt is seeded from the scalar `daily_pnl_dollars` (its persisted
        value), and shadows start the day flat.

        Fail-closed: the primary MUST appear in the map (it is the ratio basis);
        a missing or non-positive primary contract count is a configuration error,
        not a silently-tolerated state.
        """
        if primary_account_id not in account_contracts:
            raise ValueError(
                f"configure_accounts: primary_account_id {primary_account_id} not in "
                f"account_contracts {sorted(account_contracts)} — cannot scale per-account dollars."
            )
        primary_contracts = account_contracts[primary_account_id]
        if primary_contracts <= 0:
            raise ValueError(
                f"configure_accounts: primary contracts must be > 0 (got {primary_contracts}) — "
                f"it is the per-account modeled-dollar ratio denominator."
            )
        self._account_contracts = dict(account_contracts)
        self._primary_account_id = primary_account_id
        # Seed belts for the configured day.
        self._account_pnl_dollars = {aid: 0.0 for aid in account_contracts}
        if restored_pnl_dollars:
            # Crash recovery: restore each account's OWN persisted modeled dollars.
            # Ignore any restored account not in the current roster (roster change
            # across restart) — fail-safe: an unknown account's loss cannot bind a
            # belt that no longer exists.
            for aid, dollars in restored_pnl_dollars.items():
                if aid in self._account_pnl_dollars:
                    self._account_pnl_dollars[aid] = dollars
        elif self.daily_pnl_dollars != 0.0:
            # Same-day restart with NO per-account restore (pre-Stage-2 state file
            # that persisted only the primary scalar). Seed EVERY belt from the
            # scalar, not just the primary. Under today's 1:1 CopyOrderRouter
            # mirror every shadow's modeled loss equals the primary's, so this is
            # the TRUE modeled value — and it is the safe direction regardless:
            # a capital guard must fail toward over-halt (false-BLOCK), never
            # leave a shadow that breached its cap unhalted (false-PASS).
            # @audit-finding 2026-06-11 evidence-auditor CONDITIONAL — shadow
            # flat-start false-PASS window on pre-Stage-2 crash restart.
            for aid in self._account_pnl_dollars:
                self._account_pnl_dollars[aid] = self.daily_pnl_dollars
        else:
            # Fresh day, no restore: the primary belt agrees with the scalar
            # (0.0); shadows start flat. (daily_pnl_dollars == 0.0 here.)
            self._account_pnl_dollars[primary_account_id] = self.daily_pnl_dollars
        # Re-derive each account's halt latch from its seeded dollars.
        self._account_halted = {aid: False for aid in account_contracts}
        if self.limits.max_daily_loss_dollars is not None:
            for aid, dollars in self._account_pnl_dollars.items():
                if dollars <= -self.limits.max_daily_loss_dollars:
                    self._account_halted[aid] = True

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
        self.daily_pnl_dollars = 0.0
        # Per-account belts reset with the scalar. Keep the configured roster
        # (contracts + account ids persist across days, like the contract map);
        # only the daily accrual + halt latch reset.
        self._account_pnl_dollars = {aid: 0.0 for aid in self._account_contracts}
        self._account_halted = {aid: False for aid in self._account_contracts}
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
        account_id: int | None = None,
    ) -> tuple[bool, str, float]:  # Added float to return type
        suggested_contract_factor = 1.0

        # Stage 2 — resolve the account whose belt this entry is checked against.
        # No configured accounts → single-account path (the scalar belt governs,
        # account-keyed branch never taken). Configured but account_id None →
        # default to the primary (the engine's single position represents it).
        belt_account_id: int | None = None
        if self._account_contracts:
            belt_account_id = account_id if account_id is not None else self._primary_account_id

        # Check 0: Multi-day equity drawdown
        if self._equity_halted:
            drawdown = self.cumulative_pnl_r - self.equity_high_water_r
            return (
                False,
                f"equity_drawdown: {drawdown:.2f}R from peak (limit {self.limits.max_equity_drawdown_r}R)",
                0.0,
            )

        # Check 1: Circuit breaker (R cap OR dollar cap — either binds).
        # When halted, report whichever cap is actually breached so the operator
        # sees the real cause (the dollar breaker can trip via on_trade_exit
        # before can_enter is next called, setting _halted with no R breach).
        #
        # Stage 2 — when accounts are configured, the dollar cap is checked
        # against the ENTERING account's own modeled belt (belt_pnl_dollars /
        # belt_halted), so one account can halt while another keeps trading. The
        # primary's belt mirrors the scalar exactly, so the single-account /
        # mirror path is unchanged. The R cap stays portfolio-wide (the engine's
        # daily_pnl_r is one shared number — per-account R would need per-account
        # fills, which is Stage 3).
        if belt_account_id is not None:
            belt_pnl_dollars = self._account_pnl_dollars.get(belt_account_id, 0.0)
            belt_halted = self._account_halted.get(belt_account_id, False)
        else:
            belt_pnl_dollars = self.daily_pnl_dollars
            belt_halted = self._halted
        dollar_breached = (
            self.limits.max_daily_loss_dollars is not None and belt_pnl_dollars <= -self.limits.max_daily_loss_dollars
        )
        if belt_halted or daily_pnl_r <= self.limits.max_daily_loss_r or dollar_breached:
            # Latch the breached belt. The portfolio-wide self._halted latches on
            # the R cap (shared) or — for the single-account path — the scalar
            # dollar breach. The per-account latch records which account is out.
            if belt_account_id is not None:
                if dollar_breached or belt_halted:
                    self._account_halted[belt_account_id] = True
                if daily_pnl_r <= self.limits.max_daily_loss_r:
                    self._halted = True  # R cap is portfolio-wide
            else:
                self._halted = True
            if dollar_breached and daily_pnl_r > self.limits.max_daily_loss_r:
                acct_note = f" (account {belt_account_id})" if belt_account_id is not None else ""
                return (
                    False,
                    f"circuit_breaker: daily PnL ${belt_pnl_dollars:.0f} "
                    f"<= -${self.limits.max_daily_loss_dollars:.0f}{acct_note}",
                    0.0,
                )
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
                day_max = max_lots_for_xfa(self.limits.topstep_xfa_account_size, self._topstep_xfa_eod_balance)
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
            projected = project_total_open_lots(active_trades, instrument, new_contracts=1)

            if projected > day_max:
                return (
                    False,
                    (
                        f"topstep_scaling_plan: projected {projected} mini-equiv lots > "
                        f"day_max {day_max} for {self.limits.topstep_xfa_account_size // 1000}K XFA "
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

    def on_trade_exit(self, pnl_r: float, pnl_dollars: float | None = None, account_id: int | None = None) -> None:
        """Update daily PnL after a trade exits.

        pnl_dollars is the realized dollar P&L for the PRIMARY account's
        contracts (computed by the engine from actual_r × risk_points ×
        point_value × contracts). When None (risk_points unknown upstream), the
        dollar accrual is skipped — fail-safe: the R cap still governs, and the
        dollar breaker simply does not see this trade rather than guessing a
        value.

        Stage 2 — when accounts are configured (configure_accounts), each
        account is charged its OWN MODELED dollars: the primary's pnl_dollars
        scaled by `account_contracts / primary_contracts`, and each account's
        belt latches independently. `account_id` is reserved for the Stage-3 path
        where a single account's real fill is charged directly; with no map it is
        ignored (single-account scalar path). The scalar daily_pnl_dollars (and
        self._halted) continue to track the PRIMARY so every existing reader —
        crash recovery, get_status — is unchanged.
        """
        self.daily_pnl_r += pnl_r
        if self.daily_pnl_r <= self.limits.max_daily_loss_r:
            self._halted = True
        # Dollar circuit breaker accrual (primary-account realized dollars).
        if pnl_dollars is not None:
            self.daily_pnl_dollars += pnl_dollars
            if (
                self.limits.max_daily_loss_dollars is not None
                and self.daily_pnl_dollars <= -self.limits.max_daily_loss_dollars
            ):
                self._halted = True
            # Stage 2 — fan MODELED dollars to every configured account belt.
            self._accrue_account_dollars(pnl_dollars)
        # Multi-day equity tracking
        self.cumulative_pnl_r += pnl_r
        if self.cumulative_pnl_r > self.equity_high_water_r:
            self.equity_high_water_r = self.cumulative_pnl_r
        if self.limits.max_equity_drawdown_r is not None and not self._equity_halted:
            drawdown = self.cumulative_pnl_r - self.equity_high_water_r
            if drawdown <= self.limits.max_equity_drawdown_r:
                self._equity_halted = True

    def _accrue_account_dollars(self, primary_pnl_dollars: float) -> None:
        """Charge each configured account its MODELED share of a primary exit.

        No-op when no account map is configured (single-account path). Otherwise
        each account is charged `primary_pnl_dollars × (its contracts / primary
        contracts)` and its belt latches independently at -max_daily_loss_dollars.

        Fail-closed: the primary contract count was validated > 0 in
        configure_accounts, so the ratio is always well-defined here. A
        configured account missing from the contract map cannot happen (the maps
        are built together), so there is no silent skip.
        """
        if not self._account_contracts or self._primary_account_id is None:
            return
        primary_contracts = self._account_contracts[self._primary_account_id]
        for aid, contracts in self._account_contracts.items():
            modeled = primary_pnl_dollars * (contracts / primary_contracts)
            self._account_pnl_dollars[aid] = self._account_pnl_dollars.get(aid, 0.0) + modeled
            if (
                self.limits.max_daily_loss_dollars is not None
                and self._account_pnl_dollars[aid] <= -self.limits.max_daily_loss_dollars
            ):
                self._account_halted[aid] = True

    def account_pnl_dollars(self, account_id: int) -> float:
        """Modeled realized daily dollars for one account (0.0 if unconfigured)."""
        return self._account_pnl_dollars.get(account_id, 0.0)

    def export_account_pnl_dollars(self) -> dict[int, float]:
        """Snapshot of every configured account's modeled daily dollars.

        For crash-recovery persistence (session_safety_state). Empty dict when no
        accounts are configured (single-account / signal-only) — the scalar
        daily_pnl_dollars is the only belt to persist there.
        """
        return dict(self._account_pnl_dollars)

    def is_account_halted(self, account_id: int) -> bool:
        """True if the given account's per-account dollar belt has tripped.

        Falls back to the portfolio-wide halt when no account map is configured
        (single-account path) so callers get a consistent answer either way.
        """
        if not self._account_contracts:
            return self._halted or self._equity_halted
        return self._account_halted.get(account_id, False) or self._equity_halted

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
        status = {
            "trading_day": self.trading_day,
            "daily_pnl_r": round(self.daily_pnl_r, 4),
            "daily_pnl_dollars": round(self.daily_pnl_dollars, 2),
            "daily_trade_count": self.daily_trade_count,
            "halted": self._halted,
            "warnings": len(self._warnings),
            "cumulative_pnl_r": round(self.cumulative_pnl_r, 4),
            "equity_high_water_r": round(self.equity_high_water_r, 4),
            "equity_drawdown_r": round(self.cumulative_pnl_r - self.equity_high_water_r, 4),
            "equity_halted": self._equity_halted,
        }
        # Stage 2 — surface per-account belts only when configured (copies>1),
        # so the single-account status payload is unchanged.
        if self._account_contracts:
            status["accounts"] = {
                str(aid): {
                    "pnl_dollars": round(self._account_pnl_dollars.get(aid, 0.0), 2),
                    "halted": self._account_halted.get(aid, False),
                    "contracts": contracts,
                }
                for aid, contracts in self._account_contracts.items()
            }
            status["primary_account_id"] = self._primary_account_id
        return status
