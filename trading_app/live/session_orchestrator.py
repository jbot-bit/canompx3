"""
Live trading session orchestrator.
DataFeed → BarAggregator → ExecutionEngine → OrderRouter → PerformanceMonitor.

VERIFIED API NOTES:
- Portfolio must be injected via __init__(portfolio=...) from prop_profiles.ACCOUNT_PROFILES
- engine.on_bar(bar_dict) — bar_dict must have 'ts_utc' key, not 'ts_event'
- engine.on_trading_day_start(date) — call before first bar of day
- engine.on_trading_day_end() -> list[TradeEvent] — closes open positions at EOD
- TradeEvent.event_type: "ENTRY" or "EXIT" (not TradeState enum)
- TradeEvent.price: entry fill on ENTRY, exit fill on EXIT
- entry_model: look up from self._strategy_map[event.strategy_id].entry_model
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from pipeline.calendar_filters import day_of_week, is_friday, is_nfp_day, is_opex_day
from pipeline.cost_model import CostSpec, get_cost_spec
from pipeline.daily_backfill import run_backfill_for_instrument
from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH, LIVE_JOURNAL_DB_PATH
from trading_app.execution_engine import ExecutionEngine
from trading_app.live.bar_aggregator import Bar
from trading_app.live.bar_persister import BarPersister
from trading_app.live.broker_factory import create_broker_components, get_broker_name
from trading_app.live.live_market_state import LiveORBBuilder
from trading_app.live.performance_monitor import PerformanceMonitor, TradeRecord
from trading_app.live.position_tracker import PositionState, PositionTracker
from trading_app.live.trade_journal import TradeJournal, generate_trade_id
from trading_app.portfolio import Portfolio, PortfolioStrategy
from trading_app.risk_manager import RiskLimits, RiskManager

log = logging.getLogger(__name__)


def _is_trading_combine_account(account_meta: dict | None) -> tuple[bool, str]:
    """Detect a TopStep Trading Combine (evaluation) account from broker metadata.

    Returns (is_tc, reason). When True, F-1 Scaling Plan must be disabled —
    the XFA scaling ladder does not apply to TC accounts.

    Observed TopStep TC name pattern: '50KTC-V2-451890-20372221'. The 'TC'
    marker is the distinguishing signal. XFA accounts are expected to use
    different naming (e.g. 'XFA' or 'EFA' markers); those fall through as
    not-TC and F-1 stays governed by the profile config.

    Returns (False, '') when metadata is None — treated as "unknown", profile
    config is trusted. Caller may log a caution but should not fail-closed
    unconditionally because the broker API may be transiently unavailable.
    """
    if account_meta is None:
        return False, ""
    name = account_meta.get("name") or ""
    # Match "TC" as a whole-token marker (e.g. "50KTC-V2-...") to avoid
    # false positives on substrings that happen to contain the letters.
    upper = name.upper()
    if "TC-" in upper or upper.endswith("TC") or "-TC" in upper:
        return True, f"Trading Combine marker in account name: {name!r}"
    return False, ""


def _resolve_topstep_xfa_account_size(prof) -> int | None:
    """Return the TopStep XFA account size for F-1 Scaling Plan enforcement.

    F-1 enforces the TopStep Express Funded Account scaling ladder. It applies
    ONLY to TopStep XFA accounts (not LFA, not Trading Combine, not other firms).
    Returns None (F-1 disabled) for any non-XFA profile. Raises RuntimeError
    fail-closed for a TopStep XFA whose account_size is not a known tier —
    the alternative would be a KeyError mid-session on the first entry attempt.
    """
    if prof is None:
        return None
    if getattr(prof, "firm", None) != "topstep":
        return None
    if not getattr(prof, "is_express_funded", False):
        return None
    from trading_app.topstep_scaling_plan import SCALING_PLAN_LADDER

    account_size = prof.account_size
    if account_size not in SCALING_PLAN_LADDER:
        raise RuntimeError(
            f"FAIL-CLOSED: TopStep XFA profile has unknown account_size={account_size}. "
            f"Valid XFA sizes: {sorted(SCALING_PLAN_LADDER.keys())}"
        )
    return account_size


def _apply_broker_reality_check(
    *,
    positions,
    order_router,
    risk_mgr,
    initial_equity: float,
    logger=None,
) -> str:
    """F-1 broker-reality check + session-start EOD balance seed.

    Call this ONLY when F-1 is active
    (risk_mgr.limits.topstep_xfa_account_size is not None). Queries broker
    account metadata via positions.query_account_metadata(account_id):

      - If metadata reveals a Trading Combine account, disable F-1 via
        risk_mgr.disable_f1(reason). The XFA scaling ladder does not apply
        to TC accounts.
      - Otherwise (XFA confirmed OR broker metadata unavailable), seed the
        session's EOD balance via risk_mgr.set_topstep_xfa_eod_balance(equity).
        Missing metadata is treated as "trust the profile config" — caller
        is responsible for having verified F-1 applies before invoking.

    Returns a status code for logging/testing:
      - "tc"                 Trading Combine detected, F-1 disabled
      - "xfa"                Non-TC metadata, EOD balance set
      - "xfa_missing_meta"   Broker returned None, EOD balance set on trust

    This helper is extracted from the HWM-init block so it can be tested as
    a unit without having to construct a SessionOrchestrator. See
    TestF1OrchestratorRolloverWiring in test_session_orchestrator.py.
    """
    account_id = order_router.account_id if order_router else 0
    account_meta = positions.query_account_metadata(account_id)
    is_tc, tc_reason = _is_trading_combine_account(account_meta)
    if is_tc:
        risk_mgr.disable_f1(tc_reason)
        return "tc"
    risk_mgr.set_topstep_xfa_eod_balance(initial_equity)
    if logger is not None:
        logger.info(
            "F-1 XFA EOD balance set at session start: $%.2f",
            initial_equity,
        )
    return "xfa" if account_meta is not None else "xfa_missing_meta"


@dataclass
class SessionStats:
    """Observability counters — tracks success/failure for every silent-failure component."""

    notifications_sent: int = 0
    notifications_failed: int = 0
    brackets_submitted: int = 0
    brackets_failed: int = 0
    bracket_cancels_ok: int = 0
    bracket_cancels_failed: int = 0
    fill_polls_run: int = 0
    fill_polls_confirmed: int = 0
    fill_polls_failed: int = 0
    reconnect_attempts: int = 0
    bars_received: int = 0
    engine_errors: int = 0
    events_processed: int = 0
    orb_cap_skips: int = 0


class SessionOrchestrator:
    # JSONL file for UI signal display — written by this process, read by Streamlit
    SIGNALS_FILE = Path(__file__).parent.parent.parent / "live_signals.jsonl"

    def __init__(
        self,
        instrument: str,
        broker: str | None = None,
        demo: bool = True,
        account_id: int = 0,
        signal_only: bool = False,
        force_orphans: bool = False,
        portfolio: Portfolio | None = None,
        shadow_account_ids: list[int] | None = None,
    ):
        self.instrument = instrument
        self.demo = demo
        self.signal_only = signal_only
        # Trading day = 09:00 Brisbane → 09:00 next day Brisbane.
        # If started before 09:00, we're still in yesterday's trading day.
        bris_now = datetime.now(ZoneInfo("Australia/Brisbane"))
        if bris_now.hour < 9:
            self.trading_day = (bris_now - timedelta(days=1)).date()
        else:
            self.trading_day = bris_now.date()

        # Create broker components via factory
        self._broker_name = broker or get_broker_name()
        components = create_broker_components(self._broker_name, demo=demo)
        self.auth = components["auth"]
        self._feed_class = components["feed_class"]
        contracts_cls = components["contracts_class"]
        self._positions_cls = components["positions_class"]

        # Portfolio MUST be injected — build_live_portfolio is DEPRECATED (resolves to 0 strategies)
        if portfolio is None:
            raise RuntimeError(
                f"No portfolio injected for {instrument}. "
                "Pass a portfolio from prop_profiles.ACCOUNT_PROFILES via select_for_profile(). "
                "build_live_portfolio() is DEPRECATED and resolves to 0 strategies."
            )
        self.portfolio = portfolio
        log.info("Using injected portfolio: %d strategies", len(portfolio.strategies))

        if not self.portfolio.strategies:
            raise RuntimeError(f"No active strategies for {instrument}")

        # Log active apertures per session (multi-aperture support)
        apertures: dict[str, set[int]] = {}
        for s in self.portfolio.strategies:
            apertures.setdefault(s.orb_label, set()).add(s.orb_minutes)
        for label, mins in sorted(apertures.items()):
            if len(mins) > 1:
                log.info("Multi-aperture: %s → %s", label, sorted(mins))

        # Strategy lookup map for resolving entry_model from strategy_id on TradeEvents
        self._strategy_map: dict[str, PortfolioStrategy] = {s.strategy_id: s for s in self.portfolio.strategies}

        # ORB cap map: (orb_label, instrument) -> max risk in points.
        # Values are compared against event.risk_points (stop distance, NOT raw ORB size).
        self._orb_caps: dict[tuple[str, str], float] = {}
        _is_profile = portfolio is not None and portfolio.strategies and portfolio.strategies[0].source == "profile"
        profile_id = None
        if portfolio is not None and portfolio.name.startswith("profile_"):
            profile_id = portfolio.name.removeprefix("profile_")
        try:
            from trading_app.prop_profiles import get_lane_registry

            for (label, instrument), info in get_lane_registry(profile_id=profile_id).items():
                cap = info.get("max_orb_size_pts")
                if cap is not None:
                    self._orb_caps[(label, instrument)] = cap
                    log.info("ORB cap loaded: %s/%s max=%.1f pts risk", label, instrument, cap)
        except Exception:
            if _is_profile:
                raise  # Fail-closed: prop accounts MUST have working cap loading
            log.warning("Could not load ORB caps from lane registry — caps DISABLED")

        # Per-trade max risk in dollars (account-level cap, None = no limit)
        self._max_risk_per_trade: float | None = None
        if portfolio is not None and portfolio.name.startswith("profile_"):
            try:
                from trading_app.prop_profiles import ACCOUNT_PROFILES

                pid = portfolio.name.removeprefix("profile_")
                prof = ACCOUNT_PROFILES.get(pid)
                if prof is not None and prof.max_risk_per_trade is not None:
                    self._max_risk_per_trade = prof.max_risk_per_trade
                    log.info("Max risk per trade: $%.0f", self._max_risk_per_trade)
            except Exception:
                raise  # Fail-closed: profile accounts MUST have working risk cap loading

        # Regime gate: load paused strategies from allocator output.
        # Strategies PAUSED by the allocator (session regime COLD) are blocked
        # at entry time. Fail-closed for profile accounts, fail-open for paper/signal.
        self._regime_paused: set[str] = set()
        try:
            _alloc_path = Path(__file__).resolve().parents[2] / "docs" / "runtime" / "lane_allocation.json"
            if _alloc_path.exists():
                import json as _json

                _alloc_data = _json.loads(_alloc_path.read_text())
                self._regime_paused = {e["strategy_id"] for e in _alloc_data.get("paused", [])}
                if self._regime_paused:
                    log.warning(
                        "REGIME GATE: %d strategies PAUSED — entries will be blocked",
                        len(self._regime_paused),
                    )
            else:
                log.info("No lane_allocation.json — regime gate disabled")
        except Exception:
            if _is_profile:
                raise  # Fail-closed: prop accounts MUST have working regime gate
            log.warning("Failed to load lane_allocation.json — regime gate disabled (fail-open)")

        # Execution stack
        self.cost_spec: CostSpec = get_cost_spec(instrument)
        cost = self.cost_spec
        # Compute max equity drawdown in R from profile if available.
        # matched_prof is captured here and reused downstream for F-1 TopStep XFA
        # scaling plan wiring into RiskLimits, avoiding a duplicate profile lookup.
        max_equity_dd_r = None
        matched_prof = None
        if portfolio is not None and portfolio.strategies:
            first = portfolio.strategies[0]
            if first.source == "profile":
                # Compute avg risk from ALL strategies with valid risk data (not just first)
                strats_with_risk = [
                    s for s in portfolio.strategies if s.median_risk_dollars and s.median_risk_dollars > 0
                ]
                if not strats_with_risk and not signal_only:
                    raise RuntimeError(
                        "FAIL-CLOSED: Profile portfolio has no strategies with median_risk_dollars. "
                        "Cannot compute max DD protection. Fix validated_setups data."
                    )
                try:
                    from trading_app.prop_profiles import ACCOUNT_PROFILES, get_account_tier

                    # Find the profile that generated this portfolio
                    for pid, prof in ACCOUNT_PROFILES.items():
                        if portfolio.name == f"profile_{pid}":
                            matched_prof = prof
                            tier = get_account_tier(prof.firm, prof.account_size)
                            # strats_with_risk was filtered on `s.median_risk_dollars and
                            # s.median_risk_dollars > 0` above — all values guaranteed non-None
                            # and > 0. `or 0.0` narrows the type for pyright.
                            avg_risk = sum(s.median_risk_dollars or 0.0 for s in strats_with_risk) / max(
                                1, len(strats_with_risk)
                            )
                            if avg_risk > 0:
                                max_equity_dd_r = -abs(tier.max_dd / avg_risk)
                                log.info(
                                    "Max DD tracking ENABLED: $%.0f / $%.1f avg_risk = %.1fR",
                                    tier.max_dd,
                                    avg_risk,
                                    max_equity_dd_r,
                                )
                            break
                    else:
                        log.warning(
                            "Profile-sourced portfolio '%s' matched no ACCOUNT_PROFILES entry — "
                            "max DD tracking DISABLED (fail-open)",
                            portfolio.name,
                        )
                except Exception as e:
                    if not signal_only:
                        raise RuntimeError(
                            f"FAIL-CLOSED: Cannot compute max_equity_drawdown_r from profile: {e}. "
                            f"Refusing to trade without DD protection on prop account."
                        ) from e
                    log.warning("Failed to compute max_equity_drawdown_r from profile: %s", e)

        # F-1 TopStep XFA Scaling Plan: only set when matched profile is a
        # TopStep Express Funded Account. Returns None (F-1 disabled) for
        # every other profile. Raises fail-closed for unknown XFA tiers.
        topstep_xfa_account_size = _resolve_topstep_xfa_account_size(matched_prof)

        risk_limits = RiskLimits(
            max_daily_loss_r=-abs(self.portfolio.max_daily_loss_r),
            max_concurrent_positions=self.portfolio.max_concurrent_positions,
            max_equity_drawdown_r=max_equity_dd_r,
            topstep_xfa_account_size=topstep_xfa_account_size,
        )
        self.risk_mgr = RiskManager(risk_limits)
        if topstep_xfa_account_size is not None:
            log.info(
                "F-1 TopStep XFA Scaling Plan ACTIVE: account_size=$%d",
                topstep_xfa_account_size,
            )
        if max_equity_dd_r is not None:
            log.info("Risk limits: daily_loss=%.1fR, max_DD=%.1fR", risk_limits.max_daily_loss_r, max_equity_dd_r)
        else:
            log.info("Risk limits: daily_loss=%.1fR, max_DD=DISABLED", risk_limits.max_daily_loss_r)
        # ML subsystem removed 2026-04-11 (ML V3 sprint Stage 4). V1/V2/V3
        # all DEAD per docs/audit/hypotheses/2026-04-11-ml-v3-pooled-confluence-postmortem.md.
        # Blueprint NO-GO registry contains the permanent entry. The Layer 1
        # raw baseline (p<1e-9, +272R 2025) does not need ML assistance.
        from trading_app.config import E2_ORDER_TIMEOUT

        role_resolver = None
        if profile_id is not None:
            try:
                from trading_app.conditional_overlays import RoleResolver

                role_resolver = RoleResolver(profile_id, today=self.trading_day)
                log.info("Conditional overlay RoleResolver enabled for profile=%s", profile_id)
            except Exception as exc:
                log.warning("Conditional overlay RoleResolver unavailable: %s", exc)

        self.engine = ExecutionEngine(
            portfolio=self.portfolio,
            cost_spec=cost,
            risk_manager=self.risk_mgr,
            live_session_costs=True,
            e2_order_timeout=E2_ORDER_TIMEOUT,
            role_resolver=role_resolver,
        )

        # NOTE: Crash recovery moved after _safety_state init (line ~387).
        # _safety_state is created later in __init__ because it depends on
        # self.portfolio.name and instrument which are set above.

        # Contract resolution (needed even in signal-only for front-month lookup)
        contracts = contracts_cls(auth=self.auth, demo=demo)

        # Order routing only needed when placing real/demo orders
        if signal_only:
            self.order_router = None
            self.positions = None
            log.info("Signal-only mode: order router skipped")
        else:
            if account_id == 0:
                account_id = contracts.resolve_account_id()
            router_cls = components["router_class"]
            primary_router = router_cls(
                account_id=account_id,
                auth=self.auth,
                demo=demo,
                tick_size=self.cost_spec.tick_size,
            )

            # Multi-account copy trading: wrap primary + shadows in CopyOrderRouter
            if shadow_account_ids:
                from trading_app.live.copy_order_router import CopyOrderRouter

                shadow_routers = [
                    router_cls(account_id=sid, auth=self.auth, demo=demo, tick_size=self.cost_spec.tick_size)
                    for sid in shadow_account_ids
                ]
                self.order_router = CopyOrderRouter(primary_router, shadow_routers)
                log.info(
                    "Copy trading: primary=%d, shadows=%s (%d total accounts)",
                    account_id,
                    shadow_account_ids,
                    1 + len(shadow_account_ids),
                )
            else:
                self.order_router = primary_router

            self.positions = self._positions_cls(auth=self.auth)
            self._notifications_broken = False

            # Position reconciliation on startup (M2.5 P0: crash recovery)
            try:
                orphans = self.positions.query_open(account_id)
                if orphans:
                    log.critical("ORPHANED POSITIONS DETECTED on session start: %s", orphans)
                    self._notify(f"ORPHAN DETECTED: {orphans}")
                    if not force_orphans:
                        raise RuntimeError(
                            f"Refusing to start: {len(orphans)} orphaned position(s) detected. "
                            f"Close them manually or pass --force-orphans to acknowledge the risk."
                        )
                    log.warning("--force-orphans: continuing with %d orphaned position(s)", len(orphans))
            except NotImplementedError:
                log.warning(
                    "ORPHAN DETECTION DISABLED — %s broker adapter does not implement query_open(). "
                    "You must manually verify no orphaned positions exist before trading.",
                    self._broker_name,
                )
            except RuntimeError:
                raise  # re-raise our own orphan-blocking error
            except Exception as e:
                log.error(
                    "Position query failed on startup: %s — ORPHAN DETECTION FAILED. Cannot verify broker state.",
                    e,
                )
                self._notify(f"ORPHAN CHECK FAILED: {e} — verify no open positions")
                if not force_orphans:
                    raise RuntimeError(
                        f"Orphan detection failed ({e}). Cannot verify broker state. "
                        f"Close positions manually or pass --force-orphans to proceed at your own risk."
                    ) from e
                log.warning("--force-orphans: proceeding despite orphan check failure")

            # Clean up orphaned bracket orders from previous crashes.
            # These are AutoBracket-tagged orders that survived a prior session.
            # If not cancelled, they will fire and open unwanted positions.
            try:
                contract_sym = contracts.resolve_front_month(instrument)
                cancelled = self.order_router.cancel_bracket_orders(contract_sym)
                if cancelled > 0:
                    log.warning(
                        "STARTUP CLEANUP: cancelled %d orphaned bracket orders on %s",
                        cancelled,
                        contract_sym,
                    )
                    self._notify(f"STARTUP: Cancelled {cancelled} orphaned bracket orders")
            except Exception as e:
                log.warning("Bracket orphan cleanup failed on startup: %s", e)

        # Live infrastructure
        self.orb_builder = LiveORBBuilder(instrument, self.trading_day)
        # PerformanceMonitor takes list[PortfolioStrategy] (has strategy_id + expectancy_r)
        self.monitor = PerformanceMonitor(self.portfolio.strategies)

        # Persistent trade journal — survives process crashes (separate DB to avoid contention)
        journal_mode = "signal" if signal_only else ("demo" if demo else "live")
        journal_path = LIVE_JOURNAL_DB_PATH
        self.journal = TradeJournal(journal_path, mode=journal_mode)
        if not self.journal.is_healthy:
            if journal_mode == "live":
                raise RuntimeError(
                    f"TradeJournal failed to open {journal_path} — refusing to start live session "
                    "without trade persistence. Fix the journal path or use --demo."
                )
            log.warning("TradeJournal unhealthy — trades will NOT be persisted (mode=%s)", journal_mode)

        # Position lifecycle tracker — replaces ad-hoc _entry_prices dict
        self._positions = PositionTracker()
        # R2-C5: initialize to session start time so watchdog always has a baseline.
        # Without this, if feed dies before any bar arrives, watchdog skips forever
        # because it checks `if _last_bar_at is None: continue`.
        self._last_bar_at: datetime | None = datetime.now(UTC)
        # Crash-recoverable safety state (persisted to data/state/ on every mutation)
        from trading_app.live.session_safety_state import SessionSafetyState

        self._safety_state = SessionSafetyState(self.portfolio.name, instrument)
        self._kill_switch_fired = self._safety_state.kill_switch_fired
        self._notifications_broken = False  # set by self-test
        self._bar_count = 0  # total bars received this session
        self._bar_persister = BarPersister(instrument, db_path=str(GOLD_DB_PATH))
        self._close_time_forced = self._safety_state.close_time_forced

        # Crash recovery: restore daily P&L so RiskManager re-derives halt state.
        # Only restore if trading_day matches (daily PnL resets each day).
        _saved_day = self._safety_state.trading_day
        if _saved_day == str(self.trading_day) and self._safety_state.daily_pnl_r != 0.0:
            self.engine.daily_pnl_r = self._safety_state.daily_pnl_r
            log.critical(
                "CRASH RECOVERY: restored daily_pnl_r=%.2fR from %s",
                self._safety_state.daily_pnl_r,
                _saved_day,
            )

        # DD PROTECTION — TWO LAYERS
        # Layer 1: RiskManager — intraday R-units, resets each session
        #   Stops a single bad session from burning too much capital.
        #   Tracks cumulative_pnl_r and equity_high_water_r in memory.
        #   Resets on daily_reset(). Lost on process restart.
        # Layer 2: AccountHWMTracker — cross-session dollars, permanent HWM
        #   Enforces prop firm EOD trailing DD rule. Breach = account terminated.
        #   Persists to data/state/account_hwm_{id}.json. Never resets automatically.
        #   Polls broker equity every ~10 bars. Triggers kill switch on breach.
        # BOTH layers must clear before any order is submitted.
        # Layer 1 resets daily. Layer 2 never resets automatically.
        self._hwm_tracker = None
        if not signal_only and portfolio is not None and portfolio.strategies:
            first = portfolio.strategies[0]
            if first.source == "profile":
                try:
                    from trading_app.account_hwm_tracker import AccountHWMTracker
                    from trading_app.prop_profiles import ACCOUNT_PROFILES, get_account_tier, get_firm_spec

                    for pid, prof in ACCOUNT_PROFILES.items():
                        if portfolio.name == f"profile_{pid}":
                            tier = get_account_tier(prof.firm, prof.account_size)
                            firm_spec = get_firm_spec(prof.firm)
                            acct_id = str(account_id) if account_id else pid
                            # Freeze EOD trailing accounts at the level where MLL locks
                            # at $0 forever, plus a $100 safety buffer.
                            #
                            # @canonical-source docs/research-input/topstep/topstep_mll_article.md
                            # @verbatim "For a $50,000 Express Funded Account, your Maximum
                            #            Loss Limit starts at -$2,000 and trails upward as your
                            #            balance grows. Once your balance reaches $2,000, the
                            #            Maximum Loss Limit stays at $0."
                            # @audit-finding F-5 (MED — formula now differentiates XFA vs TC)
                            #
                            # XFA accounts start at $0 broker equity. The peak that locks the
                            # MLL at $0 is `max_dd` (e.g. $2,000 for 50K). Add a $100 buffer.
                            #
                            # TC accounts start at `account_size` broker equity. The peak that
                            # locks the MLL at $0 is `account_size + max_dd`. Add a $100 buffer.
                            freeze = None
                            if firm_spec.dd_type == "eod_trailing":
                                if prof.is_express_funded:
                                    freeze = tier.max_dd + 100  # XFA starts at $0
                                else:
                                    freeze = prof.account_size + tier.max_dd + 100  # TC
                            self._hwm_tracker = AccountHWMTracker(
                                account_id=acct_id,
                                firm=prof.firm,
                                dd_limit_dollars=float(tier.max_dd),
                                dd_type=firm_spec.dd_type,
                                freeze_at_balance=freeze,
                            )
                            # Query initial equity from broker
                            if self.positions is not None:
                                initial_equity = self.positions.query_equity(
                                    self.order_router.account_id if self.order_router else 0
                                )
                                if initial_equity is not None:
                                    self._hwm_tracker.update_equity(initial_equity)
                                    self._hwm_tracker.record_session_start(initial_equity)
                                    halted, reason = self._hwm_tracker.check_halt()
                                    if halted:
                                        raise RuntimeError(
                                            f"FAIL-CLOSED: Account HWM DD limit breached — {reason}. "
                                            f"Refusing to start session."
                                        )
                                    log.info("HWM tracker: %s", reason)
                                    # F-1 TopStep XFA Scaling Plan: feed session-start EOD balance
                                    # to the risk manager so max_lots_for_xfa can cap today's contract
                                    # count. Guarded on topstep_xfa_account_size being set (F-1 active).
                                    # Broker-reality check: if the connected account is a Trading
                                    # Combine (not XFA), F-1 does not apply — disable it based on
                                    # broker metadata. See _apply_broker_reality_check above.
                                    if self.risk_mgr.limits.topstep_xfa_account_size is not None:
                                        _apply_broker_reality_check(
                                            positions=self.positions,
                                            order_router=self.order_router,
                                            risk_mgr=self.risk_mgr,
                                            initial_equity=initial_equity,
                                            logger=log,
                                        )
                                else:
                                    log.warning(
                                        "HWM tracker: broker equity unavailable at startup — will init on first poll"
                                    )
                            break
                except ImportError as e:
                    raise RuntimeError(
                        "HWM tracker: account_hwm_tracker module not available — "
                        "cannot trade prop account without DD tracking"
                    ) from e
                except RuntimeError:
                    raise  # Re-raise halt and other fail-closed errors — NEVER swallow these
                except Exception as e:
                    raise RuntimeError(
                        f"HWM tracker init failed: {e} — cannot trade prop account without DD tracking"
                    ) from e

        # Prop firm close time (ET) — used for post-market buffer and force-flatten
        self._close_hour_et: int | None = None
        self._close_min_et: int | None = None
        self._profile_id_for_lane_ctl: str | None = None
        if portfolio is not None and portfolio.strategies and portfolio.strategies[0].source == "profile":
            try:
                from trading_app.prop_profiles import ACCOUNT_PROFILES, get_firm_spec

                for pid, prof in ACCOUNT_PROFILES.items():
                    if portfolio.name == f"profile_{pid}":
                        self._profile_id_for_lane_ctl = pid
                        firm = get_firm_spec(prof.firm)
                        if firm.close_time_et and firm.close_time_et != "none":
                            parts = firm.close_time_et.split(":")
                            self._close_hour_et = int(parts[0])
                            self._close_min_et = int(parts[1])
                            log.info(
                                "Firm close time: %s ET (%02d:%02d)",
                                firm.close_time_et,
                                self._close_hour_et,
                                self._close_min_et,
                            )
                        break
            except Exception as e:
                log.warning("Failed to load firm close time: %s", e)
        self._stats = SessionStats()  # observability counters
        self._poller_active = False  # set True once fill poller runs a cycle
        self._consecutive_engine_errors = 0  # circuit breaker for engine crashes
        # Restore blocked strategies from crash-recovery state (if any)
        self._blocked_strategies: set[str] = set(self._safety_state.blocked_strategies.keys())
        self._blocked_strategy_reasons: dict[str, str] = dict(self._safety_state.blocked_strategies)

        # Circuit breaker: blocks order submission after 5 consecutive broker failures
        from trading_app.live.circuit_breaker import CircuitBreaker

        self._circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

        # Resolve front-month contract symbol (needed even in signal-only for logging)
        self.contract_symbol = contracts.resolve_front_month(instrument)
        self._account_name = self.portfolio.name  # For dashboard display
        log.info(
            "Session ready: %s → %s (%s)",
            instrument,
            self.contract_symbol,
            "SIGNAL-ONLY" if signal_only else ("DEMO" if demo else "LIVE"),
        )

        # Write session-start marker to signals file
        self._write_signal_record(
            {
                "type": "SESSION_START",
                "instrument": instrument,
                "contract": self.contract_symbol,
                "mode": "signal_only" if signal_only else ("demo" if demo else "live"),
            }
        )

        # Build partial daily_features_row from what's available pre-session.
        # Without this, fail-closed filters (VOL_RV12_N20, DOW, calendar) silently
        # reject ALL trades because their required columns are missing.
        # Build per-orb_minutes daily_features rows so filters evaluate
        # against the correct aperture's ORB columns.
        unique_om = {s.orb_minutes for s in self.engine.portfolio.strategies}
        df_rows = {}
        for om in sorted(unique_om):
            df_rows[om] = self._build_daily_features_row(self.trading_day, instrument, orb_minutes=om)

        # Signal engine that a new trading day is starting
        self.engine.on_trading_day_start(self.trading_day, daily_features_rows=df_rows)

        # Crash recovery: seed engine with strategies that already traded today.
        # Prevents duplicate entries after restart mid-session.
        # Fail-closed: if journal is broken in live/demo mode, refuse to start.
        if not self.journal.is_healthy and not signal_only:
            raise RuntimeError(
                "FAIL-CLOSED: Trade journal is not healthy — cannot perform crash recovery. "
                "Duplicate entries could occur. Fix journal or use --signal-only."
            )
        already_traded = self.journal.get_strategy_ids_for_day(self.trading_day)
        for sid in already_traded:
            self.engine.mark_strategy_traded(sid)
        if already_traded:
            log.warning(
                "RESTART RECOVERY: %d strategies already traded today — will NOT re-enter: %s",
                len(already_traded),
                sorted(already_traded),
            )

        # Restore position tracker from incomplete journal trades (crash recovery).
        # If the bot crashed with open positions, the journal has entry records but no exits.
        # Restoring these lets the orchestrator properly track and exit broker-held positions.
        # Check both today and yesterday to catch cross-midnight restarts.
        if not signal_only:
            incomplete = self.journal.incomplete_trades(trading_day=self.trading_day)
            prev_day = self.trading_day - timedelta(days=1)
            prev_incomplete = self.journal.incomplete_trades(trading_day=prev_day)
            if prev_incomplete:
                log.warning(
                    "Found %d incomplete trades from previous day %s — including in crash recovery",
                    len(prev_incomplete),
                    prev_day,
                )
                incomplete.extend(prev_incomplete)
            for trade in incomplete:
                sid = trade.get("strategy_id", "")
                direction = trade.get("direction", "")
                fill = trade.get("fill_entry") or trade.get("engine_entry")
                validated_fill = self._validate_fill_price(float(fill) if fill else None, f"RESTORE {sid}")
                if sid and direction and validated_fill and self._positions.get(sid) is None:
                    record = self._positions.on_entry_sent(sid, direction, validated_fill)
                    if record:
                        self._positions.on_entry_filled(sid, validated_fill)
                        # validated_fill is the same price used by on_entry_filled;
                        # it's already a validated float at this point.
                        log.warning(
                            "POSITION RESTORED from journal: %s %s @ %.2f",
                            sid,
                            direction,
                            validated_fill,
                        )
            if incomplete:
                log.warning(
                    "CRASH RECOVERY: restored %d incomplete positions from journal",
                    len(incomplete),
                )

        self._load_paused_lane_blocks()

        mode = "SIGNAL" if signal_only else ("DEMO" if demo else "LIVE")
        self._notify(f"Session started: {instrument} ({mode})")

    def _fire_kill_switch(self) -> None:
        """Trigger emergency flatten. Persisted to survive crashes."""
        self._kill_switch_fired = True
        self._safety_state.kill_switch_fired = True
        self._safety_state.save()

    def _force_close_time(self) -> None:
        """Trigger EOD force-flatten. Persisted to survive crashes."""
        self._close_time_forced = True
        self._safety_state.close_time_forced = True
        self._safety_state.save()

    def _block_strategy(self, strategy_id: str, reason: str, *, persist: bool = True) -> None:
        """Add a runtime block with explicit reason.

        Args:
            strategy_id: Strategy to block.
            reason: Human-readable reason for the block.
            persist: If True (default), write to SessionSafetyState for
                crash recovery. Use False for blocks that are re-derivable
                at the next session start (e.g. lifecycle-sourced pauses
                and SR-ALARM reviews read from read_lifecycle_state).
                Persisting those would cause stale blocks to survive after
                the underlying review changes (fixed 2026-04-14).
        """
        self._blocked_strategies.add(strategy_id)
        self._blocked_strategy_reasons[strategy_id] = reason
        if persist:
            self._safety_state.blocked_strategies[strategy_id] = reason
            self._safety_state.save()

    def _load_paused_lane_blocks(self) -> None:
        """Load operational lifecycle blocks into the runtime block set.

        Lifecycle blocks (pause_strategy_id, SR-ALARM with no WATCH review,
        Criterion 11 regime fails) are fully re-derived from the canonical
        registries every session start. They MUST NOT be persisted to the
        safety-state file — see `_block_strategy(persist=False)` docstring.
        """
        if not self._profile_id_for_lane_ctl:
            return
        try:
            from trading_app.lifecycle_state import read_lifecycle_state

            lifecycle = read_lifecycle_state(profile_id=self._profile_id_for_lane_ctl, today=self.trading_day)
            blocked_ids = lifecycle["blocked_strategy_ids"]
            blocked_reasons = lifecycle["blocked_reason_by_strategy"]
            for strategy_id in blocked_ids:
                reason = blocked_reasons.get(strategy_id, "Paused pending manual review")
                self._block_strategy(strategy_id, reason, persist=False)
            if blocked_ids:
                log.warning("Loaded %d lifecycle lane blocks", len(blocked_ids))
        except Exception as e:
            log.warning("Failed to load lifecycle lane blocks: %s", e)

    @staticmethod
    def _build_daily_features_row(trading_day: date, instrument: str, orb_minutes: int = 5) -> dict:
        """Build a daily_features_row from DB + calendar for live execution.

        Without this, fail-closed filters (VOL_RV12_N20, DOW, break speed, calendar)
        silently reject ALL trades because their required columns are None.

        Populates:
          - Calendar (exact for today): is_nfp_day, is_opex_day, is_friday, day_of_week
          - From most recent DB row (yesterday's proxy): atr_20, atr_vel_regime,
            compression tiers, rel_vol_*, break_delay_min, etc.
          - Computed: median_atr_20 (rolling 252-day median for vol-scaling)

        The rel_vol_* and break_delay_min values are yesterday's — imperfect but
        vastly better than None (which silently kills every VOL/FAST strategy).
        orb_{label}_size is set by ExecutionEngine from live ORB, overriding the DB value.
        """
        row: dict = {}

        # Calendar flags — exact for today
        row["is_nfp_day"] = is_nfp_day(trading_day)
        row["is_opex_day"] = is_opex_day(trading_day)
        row["is_friday"] = is_friday(trading_day)
        row["day_of_week"] = day_of_week(trading_day)

        # Load ALL columns from the most recent daily_features row.
        # This gives filters yesterday's values as proxies for today.
        try:
            import duckdb

            with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
                configure_connection(con)
                df = con.execute(
                    """
                    SELECT * FROM daily_features
                    WHERE symbol = ? AND orb_minutes = ?
                      AND trading_day = (
                          SELECT MAX(trading_day) FROM daily_features
                          WHERE symbol = ? AND orb_minutes = ?
                      )
                """,
                    [instrument, orb_minutes, instrument, orb_minutes],
                ).fetchdf()
                if not df.empty:
                    latest = df.iloc[0].to_dict()
                    # Merge all DB columns (ATR, compression, rel_vol, break speed, etc.)
                    # Calendar flags from above override any DB values (exact for today).
                    for k, v in latest.items():
                        if k not in row:  # don't overwrite today's calendar flags
                            row[k] = v
                    # Staleness warning — daily_features should be from yesterday or today
                    latest_day = latest.get("trading_day")
                    if latest_day is not None:
                        gap = (trading_day - (latest_day.date() if hasattr(latest_day, "date") else latest_day)).days
                        if gap > 5:
                            raise RuntimeError(
                                f"FAIL-CLOSED: daily_features is {gap} days stale (latest: {latest_day}). "
                                f"Filters would silently reject all trades. Run: "
                                f"python pipeline/build_daily_features.py --instrument {instrument}"
                            )
                        elif gap > 3:
                            log.warning("daily_features data is %d days stale (latest: %s)", gap, latest_day)

                # median_atr_20 is NOT in daily_features — it's a rolling median computed
                # by paper_trader. Compute it here for live vol-scaling.
                median_result = con.execute(
                    """
                    SELECT MEDIAN(atr_20) FROM daily_features
                    WHERE symbol = ? AND orb_minutes = 5 AND atr_20 IS NOT NULL
                      AND trading_day < ? AND trading_day >= ? - INTERVAL '504 DAY'
                """,
                    [instrument, trading_day, trading_day],
                ).fetchone()
                if median_result and median_result[0] is not None:
                    row["median_atr_20"] = float(median_result[0])

                # Cross-asset ATR: load source instruments' latest atr_20_pct for
                # CrossAssetATRFilter. Derive sources dynamically from ALL_FILTERS
                # so new cross-asset filters don't require code changes here.
                from trading_app.config import ALL_FILTERS, CrossAssetATRFilter

                _cross_sources = {
                    f.source_instrument for f in ALL_FILTERS.values() if isinstance(f, CrossAssetATRFilter)
                }
                for source in _cross_sources:
                    if source == instrument:
                        continue
                    src_result = con.execute(
                        """SELECT atr_20_pct FROM daily_features
                           WHERE symbol = ? AND orb_minutes = 5 AND atr_20_pct IS NOT NULL
                           ORDER BY trading_day DESC LIMIT 1""",
                        [source],
                    ).fetchone()
                    if src_result and src_result[0] is not None:
                        row[f"cross_atr_{source}_pct"] = float(src_result[0])
        except RuntimeError:
            raise  # re-raise our own fail-closed errors
        except Exception as e:
            raise RuntimeError(
                f"FAIL-CLOSED: Cannot load daily_features for {instrument}: {e}. "
                f"Filters would silently reject all trades. Fix the database or run: "
                f"python pipeline/build_daily_features.py --instrument {instrument}"
            ) from e

        log.info(
            "Daily features row: atr_20=%s, atr_vel=%s, nfp=%s, opex=%s, dow=%s",
            row.get("atr_20"),
            row.get("atr_vel_regime"),
            row.get("is_nfp_day"),
            row.get("is_opex_day"),
            row.get("day_of_week"),
        )
        return row

    def _on_feed_stale(self, gap_seconds: float, stale_count: int) -> None:
        """Called by BrokerFeed when data feed goes stale or dies. Sends alert.

        stale_count == -1 means feed exhausted all reconnect attempts (permanently dead).
        R2-C5: when feed is permanently dead and positions are open, schedule
        emergency flatten. The watchdog may never fire if _last_bar_at is None
        (no bars received before death), so this is the only flatten trigger.
        """
        if stale_count == -1:
            msg = f"FEED DEAD: all reconnect attempts exhausted for {self.instrument}"
            log.critical(msg)
            self._notify(msg)
            # R2-C5: flatten if positions exist — watchdog alone is insufficient
            if self._positions.active_positions() and not self._kill_switch_fired:
                self._fire_kill_switch()
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._emergency_flatten())
                except RuntimeError:
                    # No running event loop (post_session context) — cannot schedule
                    log.critical("FEED DEAD: cannot schedule flatten (no event loop) — MANUAL CLOSE REQUIRED")
        else:
            msg = f"FEED STALE: {gap_seconds:.0f}s no data (check {stale_count})"
            log.critical(msg)
            self._notify(msg)

    def _validate_fill_price(self, fill_price: float | None, context: str) -> float | None:
        """Validate fill price from broker. Returns price if valid, None if rejected."""
        if fill_price is None:
            return None

        import math

        # Basic sanity: must be positive, finite number
        if not isinstance(fill_price, int | float):
            log.critical("BAD FILL (%s): not numeric: %r", context, fill_price)
            self._notify(f"BAD FILL ({context}): price is not numeric: {fill_price}")
            return None

        if math.isnan(fill_price) or math.isinf(fill_price):
            log.critical("BAD FILL (%s): NaN/inf: %r", context, fill_price)
            self._notify(f"BAD FILL ({context}): price is NaN/inf")
            return None

        if fill_price <= 0:
            log.critical("BAD FILL (%s): non-positive: %s", context, fill_price)
            self._notify(f"BAD FILL ({context}): price <= 0: {fill_price}")
            return None

        # Range check: within 10% of last bar close (if available)
        if hasattr(self, "orb_builder") and self.orb_builder is not None:
            last_close = getattr(self.orb_builder, "last_close", None)
            if isinstance(last_close, int | float) and last_close > 0:
                deviation = abs(fill_price - last_close) / last_close
                if deviation > 0.10:
                    log.critical(
                        "BAD FILL (%s): price %.4f deviates %.1f%% from last close %.4f",
                        context,
                        fill_price,
                        deviation * 100,
                        last_close,
                    )
                    self._notify(
                        f"BAD FILL ({context}): {fill_price} deviates {deviation:.1%} from market {last_close}"
                    )
                    return None

        return float(fill_price)

    def _minutes_to_close_et(self) -> float | None:
        """Minutes until firm close time in ET. None if no close time set."""
        if self._close_hour_et is None:
            return None
        try:
            et_now = datetime.now(ZoneInfo("America/New_York"))
            close_today = et_now.replace(
                hour=self._close_hour_et,
                minute=self._close_min_et or 0,
                second=0,
                microsecond=0,
            )
            diff = (close_today - et_now).total_seconds() / 60.0
            return diff
        except Exception:
            return None

    def _publish_state(self) -> None:
        """Write bot state to JSON for dashboard consumption. Never raises."""
        try:
            from trading_app.live.bot_state import build_state_snapshot, write_state

            mode = "SIGNAL" if self.signal_only else ("DEMO" if self.demo else "LIVE")
            account_id = self.order_router.account_id if self.order_router else 0
            snapshot = build_state_snapshot(
                mode=mode,
                instrument=self.instrument,
                contract=self.contract_symbol,
                trading_day=self.trading_day,
                account_id=account_id,
                account_name=getattr(self, "_account_name", ""),
                daily_pnl_r=self.engine.daily_pnl_r,
                daily_loss_limit_r=self.risk_mgr.limits.max_daily_loss_r,
                max_equity_dd_r=self.risk_mgr.limits.max_equity_drawdown_r,
                bars_received=self._stats.bars_received,
                strategies=self.portfolio.strategies,
                active_trades=self.engine.active_trades,
                completed_trades=self.engine.completed_trades,
            )
            # Add copy trading info if CopyOrderRouter is active
            from trading_app.live.copy_order_router import CopyOrderRouter

            if isinstance(self.order_router, CopyOrderRouter):
                snapshot["copy_accounts"] = self.order_router.all_account_ids
                snapshot["shadow_count"] = self.order_router.shadow_count
            write_state(snapshot)
        except Exception:
            pass  # Dashboard state is best-effort — never kill the trading loop

    def _notify(self, message: str) -> None:
        """Send Telegram notification. Never raises — notifications must not kill the trading loop.

        If notifications were flagged broken by self-test, skips Telegram and logs
        to STDOUT so the session log still captures every event.
        """
        from trading_app.live.alert_engine import record_operator_alert

        mode = "SIGNAL" if self.signal_only else ("DEMO" if self.demo else "LIVE")
        record_operator_alert(
            message=message,
            instrument=self.instrument,
            profile=getattr(self, "_account_name", None) or getattr(self.portfolio, "name", None),
            mode=mode,
            source="session_orchestrator",
            trading_day=str(getattr(self, "trading_day", "")) or None,
        )
        if self._notifications_broken:
            print(f"[NOTIFY-FALLBACK] {self.instrument}: {message}")
            log.warning("Notification (fallback): %s", message)
            self._stats.notifications_failed += 1
            return
        try:
            from trading_app.live.notifications import notify

            notify(self.instrument, message)
            self._stats.notifications_sent += 1
        except Exception as e:
            self._stats.notifications_failed += 1
            log.error("Notification failed (will not retry): %s", e)
            if self._stats.notifications_failed == 1:
                print(f"!!! NOTIFICATION FAILURE: {e} — check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID !!!")

    def _verify_notifications(self) -> bool:
        """Send test notification via the same path as _notify(). Returns False if broken."""
        try:
            from trading_app.live.notifications import notify

            notify(self.instrument, "SELF-TEST: notifications working")
            log.info("Notification self-test passed")
            return True
        except Exception as e:
            log.critical("NOTIFICATION SELF-TEST FAILED: %s", e)
            print(f"!!! NOTIFICATIONS ARE BROKEN: {e} !!!")
            return False

    def _verify_brackets(self) -> bool:
        """Verify bracket order support works, not just claimed."""
        if self.order_router is None:
            return True  # signal-only
        if not self.order_router.supports_native_brackets():
            log.info("Broker does not support brackets — no crash protection")
            return True
        try:
            spec = self.order_router.build_bracket_spec(
                direction="long",
                symbol="TEST",
                entry_price=100.0,
                stop_price=99.0,
                target_price=102.0,
                qty=1,
            )
            if spec is None:
                log.warning("build_bracket_spec returned None despite supports_native_brackets=True")
                return False
            log.info("Bracket spec self-test passed")
            return True
        except Exception as e:
            log.critical("BRACKET SELF-TEST FAILED: %s", e)
            return False

    def _verify_fill_poller(self) -> bool:
        """Verify broker supports order status queries for fill polling.

        Returns True if the endpoint exists (even if it returns an error for
        order ID 0 — that's expected). Only returns False for NotImplementedError,
        which means the broker genuinely doesn't support polling.
        """
        if self.order_router is None or self.signal_only:
            return True
        try:
            self.order_router.query_order_status(0)
        except NotImplementedError:
            log.warning("Broker does not support query_order_status — fill poller will be inactive")
            return False
        except Exception as e:  # noqa: BLE001 — 404/auth errors mean endpoint exists
            log.info("Fill poller endpoint exists (non-fatal error: %s)", e)
        return True

    def run_self_tests(self) -> dict[str, bool]:
        """Run all component self-tests. Returns {component: passed}."""
        results = {}
        results["notifications"] = self._verify_notifications()
        results["brackets"] = self._verify_brackets()
        results["fill_poller"] = self._verify_fill_poller()

        print("\n  SELF-TEST RESULTS:")
        for component, passed in results.items():
            status = "PASS" if passed else "FAIL"
            marker = "  " if passed else "!!"
            print(f"  {marker} {component:20s} {status}")
        print()

        self._notifications_broken = not results["notifications"]
        return results

    def _write_signal_record(self, extra: dict) -> None:
        """Append a signal record to the JSONL file read by the Live Monitor UI.

        Thread safety: safe under single-threaded asyncio (all orchestrators share
        one event loop). If this ever moves to multi-process, this append needs a lock.
        """
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "instrument": self.instrument,
            **extra,
        }
        try:
            with open(self.SIGNALS_FILE, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as e:
            log.warning("Could not write signal record: %s", e)

    async def _check_trading_day_rollover(self, bar_ts_utc) -> None:
        """Detect 9:00 AM Brisbane boundary crossing and roll to new trading day.

        Without this, a session running past 9 AM Brisbane would continue using
        yesterday's ORB windows, calendar flags, daily P&L, and risk limits.

        Uses _handle_event() for each EOD close so broker orders are submitted
        (not just engine-side closes).
        """
        _bris = ZoneInfo("Australia/Brisbane")
        bris_time = bar_ts_utc.astimezone(_bris)
        if bris_time.hour < 9:
            bar_trading_day = (bris_time - timedelta(days=1)).date()
        else:
            bar_trading_day = bris_time.date()

        if bar_trading_day == self.trading_day:
            return

        log.info("Trading day rollover: %s -> %s", self.trading_day, bar_trading_day)

        # Close previous day's open positions via _handle_event (submits broker orders)
        eod_events = self.engine.on_trading_day_end()
        for event in eod_events:
            try:
                await self._handle_event(event)
            except Exception as e:
                msg = f"ROLLOVER CLOSE FAILED: {event.strategy_id} ({event.event_type}) — position may remain open: {e}"
                log.error(msg)
                self._notify(msg)

        # Orphan check: if any positions survived the close loop, they are stuck
        # at the broker. Engine reset (below) will forget them — log and notify.
        rollover_orphans = self._positions.active_positions()
        if rollover_orphans:
            orphan_ids = [(r.strategy_id, r.state.value) for r in rollover_orphans]
            msg = f"ROLLOVER ORPHANS: {orphan_ids} — positions open at broker, engine will not track. MANUAL CLOSE REQUIRED."
            log.critical(msg)
            self._notify(msg)
            # CONTAINMENT: block new entries for orphaned strategies until manual resolution.
            # Prevents doubling up: engine would re-enter the same strategy on the new day
            # while the old position is still open at the broker.
            for r in rollover_orphans:
                self._block_strategy(r.strategy_id, "Orphaned broker position — manual resolution required")
            log.critical("ENTRY BLOCKED for %s until manual orphan resolution", list(self._blocked_strategies))

        # Start new trading day
        self.trading_day = bar_trading_day
        unique_om = {s.orb_minutes for s in self.engine.portfolio.strategies}
        df_rows = {}
        for om in sorted(unique_om):
            df_rows[om] = self._build_daily_features_row(self.trading_day, self.instrument, orb_minutes=om)
        self.engine.on_trading_day_start(self.trading_day, daily_features_rows=df_rows)

        # R2-C7: Re-seed engine dedup for orphaned strategies so they cannot re-arm.
        # on_trading_day_start() cleared active_trades, so without this the engine
        # would re-enter an orphaned strategy on the new day → 2x exposure at broker.
        # _blocked_strategies already blocks at orchestrator level, but this prevents
        # the engine from even generating the ENTRY event.
        for r in rollover_orphans:
            self.engine.mark_strategy_traded(r.strategy_id)

        # Re-seed journal dedup for the new day (same logic as init).
        # Without this, strategies that traded earlier today (e.g. from a crashed
        # prior process) would not be blocked after rollover clears completed_trades.
        already_traded = self.journal.get_strategy_ids_for_day(self.trading_day)
        for sid in already_traded:
            self.engine.mark_strategy_traded(sid)
        if already_traded:
            log.warning(
                "ROLLOVER DEDUP: %d strategies already traded on %s — will NOT re-enter: %s",
                len(already_traded),
                self.trading_day,
                sorted(already_traded),
            )

        self.orb_builder = LiveORBBuilder(self.instrument, self.trading_day)
        self.monitor.reset_daily()
        self.risk_mgr.daily_reset(self.trading_day)
        self._consecutive_engine_errors = 0
        self._close_time_forced = False  # Reset so next day's close-time flatten can fire

        # F-1 TopStep XFA Scaling Plan: refresh EOD balance from broker equity
        # so today's contract cap reflects yesterday's session close.
        #
        # Canonical rule (topstep_scaling_plan_article.md line 47):
        # "buying power will increase or decrease based on your end-of-day P&L"
        #
        # ProjectX query_equity returns realized balance (cash). When the bot is
        # FLAT at rollover, realized balance == EOD balance — correct for F-1.
        # When orphaned positions remain (close loop failed above), realized
        # balance may UNDER-represent true equity if orphans have unrealized
        # losses, which would make F-1's cap LOOSER than it should be. That is
        # not a safe direction. Fail-closed: skip the refresh and keep the last
        # known good EOD balance. Orphaned strategies are already blocked for
        # entries via _blocked_strategies, so skipping is safe.
        if self.risk_mgr.limits.topstep_xfa_account_size is not None and self.positions is not None:
            active_at_rollover = self._positions.active_positions()
            if active_at_rollover:
                log.warning(
                    "F-1 XFA EOD balance NOT refreshed at rollover: %d active positions "
                    "(orphans from close loop) — realized balance may be stale; keeping "
                    "last known good EOD balance. Blocked strategies: %s",
                    len(active_at_rollover),
                    sorted(self._blocked_strategies),
                )
            else:
                try:
                    eod_equity = self.positions.query_equity(self.order_router.account_id if self.order_router else 0)
                    if eod_equity is not None:
                        self.risk_mgr.set_topstep_xfa_eod_balance(eod_equity)
                        log.info("F-1 XFA EOD balance refreshed at rollover: $%.2f", eod_equity)
                    else:
                        log.warning("F-1 XFA EOD balance NOT refreshed at rollover (broker equity unavailable)")
                except Exception as e:
                    log.warning("F-1 XFA EOD balance refresh failed: %s", e)

        log.info("New trading day started: %s", self.trading_day)

    async def _on_bar(self, bar: Bar) -> None:
        """Called for each completed 1-minute bar from DataFeed."""
        # Bar heartbeat monitoring
        now = datetime.now(UTC)
        if self._last_bar_at is not None:
            gap = (now - self._last_bar_at).total_seconds()
            if gap > 180:  # 3 minutes without a bar
                log.critical("BAR HEARTBEAT: %.0fs since last bar — feed may be dead", gap)
                self._notify(f"BAR HEARTBEAT: {gap:.0f}s since last bar — feed may be dead")
                stale = self._positions.stale_positions(timeout_seconds=300)
                if stale:
                    log.critical("STALE ORDERS: %s", [(s.strategy_id, s.state.value) for s in stale])
                    self._notify(f"STALE ORDERS: {[(s.strategy_id, s.state.value) for s in stale]}")
                    # R2-C6: re-attempt close for stuck PENDING_EXIT positions.
                    # Without this, a REST outage with live WebSocket leaves positions
                    # stuck indefinitely — kill switch only fires on feed silence.
                    for sr in stale:
                        if (
                            sr.state == PositionState.PENDING_EXIT
                            and not self.signal_only
                            and self.order_router is not None
                        ):
                            await self._retry_stuck_exit(sr)
        self._last_bar_at = now
        self._bar_count += 1
        self._stats.bars_received += 1

        # Persist bar for Databento-free daily pipeline
        self._bar_persister.append(bar)

        # Force-flatten within 5 minutes of firm close time (prevents positions at cutoff)
        if not self._close_time_forced and self._close_hour_et is not None:
            mins_to_close = self._minutes_to_close_et()
            if mins_to_close is not None and 0 < mins_to_close <= 5.0:
                active = self._positions.active_positions()
                if active:
                    self._force_close_time()
                    msg = f"CLOSE TIME FLATTEN: {len(active)} position(s) being closed ({mins_to_close:.0f}min before cutoff)"
                    log.critical(msg)
                    self._notify(msg)
                    await self._emergency_flatten()

        # Periodic bar heartbeat log (every 10 bars ≈ 10 minutes)
        if self._bar_count % 10 == 0:
            active = len(self._positions.active_positions())
            n_orbs = sum(1 for o in self.engine.orbs.values() if o.complete)
            log.info(
                "BAR HEARTBEAT: %d bars, %d ORBs complete, %d active positions, %d trades",
                self._bar_count,
                n_orbs,
                active,
                self.monitor.trade_count,
            )
            # F-2b: Cross-account divergence proactive check (every 10 bars).
            # @canonical-source docs/research-input/topstep/topstep_cross_account_hedging.md
            # If CopyOrderRouter has detected any shadow operation failure since
            # the last reset, halt the session before any further trades. The
            # next submit() would raise ShadowDivergenceError anyway, but this
            # proactive check halts BEFORE the next signal fires (smaller window
            # of asymmetric position state across copies).
            if self.order_router is not None and self.order_router.is_degraded():
                diverged = self.order_router.degraded_accounts()
                msg = (
                    f"F-2b SHADOW DIVERGENCE HALT: {len(diverged)} shadow account(s) "
                    f"out of sync with primary: {diverged}. Triggering kill switch. "
                    f"Manual reconciliation required before resuming."
                )
                log.critical(msg)
                self._notify(msg)
                self._fire_kill_switch()
                await self._emergency_flatten()

            # HWM equity poll (every 10 bars ≈ 10 minutes)
            if self._hwm_tracker is not None and self.positions is not None and self.order_router is not None:
                try:
                    equity = self.positions.query_equity(self.order_router.account_id)
                    self._hwm_tracker.update_equity(equity)
                    halted, reason = self._hwm_tracker.check_halt()
                    if halted:
                        log.critical("HWM DD HALT: %s — triggering kill switch", reason)
                        self._notify(f"ACCOUNT DD LIMIT: {reason}")
                        self._fire_kill_switch()
                        await self._emergency_flatten()
                    elif "WARN" in reason:
                        log.warning("HWM: %s", reason)
                except Exception as e:
                    log.warning("HWM equity poll failed: %s", e)

        # Kill switch fired = we already emergency-flattened at the broker.
        # Do NOT process further bars — engine doesn't know positions are closed,
        # so it would generate duplicate EXIT orders for already-flattened positions.
        if self._kill_switch_fired:
            return

        # Check if we've crossed the 9:00 AM Brisbane boundary
        try:
            await self._check_trading_day_rollover(bar.ts_utc)
        except Exception as e:
            log.critical("Trading day rollover failed: %s — skipping rollover, feed continues", e)
            self._notify(f"ROLLOVER ERROR: {e}")
            # Continue processing bars with stale daily features rather than
            # killing the feed loop. Existing positions still need management.

        self.orb_builder.on_bar(bar)

        # Update order router's last known market price for price collar validation
        if self.order_router is not None and hasattr(self.order_router, "update_market_price"):
            self.order_router.update_market_price(bar.close)

        # bar.as_dict() returns {ts_utc, open, high, low, close, volume}
        # — exactly what ExecutionEngine.on_bar() expects
        if self._consecutive_engine_errors >= 5:
            return  # engine paused — bars dropped until rollover resets counter
        try:
            events = self.engine.on_bar(bar.as_dict())
            self._consecutive_engine_errors = 0
        except Exception as e:
            log.critical("Engine error processing bar: %s — bar dropped, feed continues", e)
            self._notify(f"ENGINE ERROR: {e}")
            self._stats.engine_errors += 1
            self._consecutive_engine_errors += 1
            if self._consecutive_engine_errors == 5:
                msg = f"ENGINE CIRCUIT BREAKER: {self._consecutive_engine_errors} consecutive errors — engine paused until next trading day"
                log.critical(msg)
                self._notify(msg)
                # R2-H5: flatten open positions BEFORE pausing engine.
                # Without brackets (Tradovate) or with failed bracket submission,
                # positions would be completely unmanaged until rollover (potentially hours).
                if self._positions.active_positions():
                    log.critical(
                        "ENGINE CIRCUIT BREAKER: flattening %d open positions before pause",
                        len(self._positions.active_positions()),
                    )
                    await self._emergency_flatten()
            return

        for event in events:
            await self._handle_event(event)

        # R2-M1: publish state AFTER engine processing and event handling.
        # Previously ran before engine.on_bar(), adding 2-50ms disk write latency
        # to the signal detection critical path. Dashboard staleness of 1 bar is acceptable.
        self._publish_state()

    def _compute_actual_r(self, entry_price: float, exit_price: float, direction: str, risk_pts: float) -> float:
        """Compute cost-adjusted R-multiple from entry/exit prices."""
        direction_sign = 1.0 if direction == "long" else -1.0
        gross_pts = direction_sign * (exit_price - entry_price)
        # Subtract transaction costs (spread + slippage + commission) in points
        net_pts = gross_pts - self.cost_spec.friction_in_points
        return net_pts / risk_pts if risk_pts > 0 else 0.0

    def _record_exit(
        self,
        event,
        entry_price: float,
        exit_fill_price: float | None = None,
        entry_slippage: float | None = None,
        journal_trade_id: str | None = None,
        order_id_exit: str | int | None = None,
    ) -> None:
        """Record a completed trade (EXIT or SCRATCH) in the performance monitor.

        Uses broker fill prices when available for more accurate P&L.
        Falls back to engine prices (event.price) when fills are unknown.
        """
        strategy = self._strategy_map[event.strategy_id]
        exit_price = exit_fill_price if exit_fill_price is not None else event.price

        # Use engine's authoritative pnl_r (session-adjusted costs) when available;
        # fall back to local computation only if not present.
        if event.pnl_r is not None:
            actual_r = event.pnl_r
        else:
            risk_pts = event.risk_points or strategy.median_risk_points
            if not risk_pts:
                log.error(
                    "No risk_points for %s — actual_r set to 0.0 (cannot compute without risk)",
                    event.strategy_id,
                )
                actual_r = 0.0
            else:
                actual_r = self._compute_actual_r(entry_price, exit_price, event.direction, risk_pts)

        # Compute total slippage (entry + exit) in points
        slippage_pts = 0.0
        if entry_slippage is not None:
            slippage_pts += entry_slippage
        if exit_fill_price is not None:
            slippage_pts += exit_fill_price - event.price

        record = TradeRecord(
            strategy_id=event.strategy_id,
            trading_day=self.trading_day,
            direction=event.direction,
            entry_price=entry_price,
            exit_price=exit_price,
            actual_r=actual_r,
            expected_r=strategy.expectancy_r,
            slippage_pts=slippage_pts,
        )
        alert = self.monitor.record_trade(record)
        if alert:
            log.warning(alert)
            self._notify(alert)

        # Persist exit to trade journal (fail-open)
        if journal_trade_id:
            # Compute dollar P&L for prop firm accounting
            risk_pts = event.risk_points or strategy.median_risk_points or 0.0
            pnl_dollars = actual_r * risk_pts * self.cost_spec.point_value * event.contracts if risk_pts else None
            self.journal.record_exit(
                trade_id=journal_trade_id,
                engine_exit=event.price,
                fill_exit=exit_fill_price,
                actual_r=actual_r,
                expected_r=strategy.expectancy_r,
                slippage_pts=slippage_pts,
                pnl_dollars=pnl_dollars,
                exit_reason=event.event_type.lower(),
                order_id_exit=order_id_exit,
                cusum_alarm=alert is not None,
            )

        # Persist daily P&L for crash recovery (daily loss circuit breaker)
        self._safety_state.daily_pnl_r = self.engine.daily_pnl_r
        self._safety_state.trading_day = str(self.trading_day)
        self._safety_state.save()

    async def _submit_bracket(self, event, strategy, entry_price: float) -> None:
        """Submit broker-side stop/target bracket after entry fill. Never raises."""
        if self.order_router is None or not self.order_router.supports_native_brackets():
            return
        try:
            risk_pts = event.risk_points or strategy.median_risk_points
            if not risk_pts:
                log.error(
                    "No risk_points for %s — skipping bracket (would place wrong stop/target)",
                    event.strategy_id,
                )
                return
            mult = getattr(strategy, "stop_multiplier", 1.0) or 1.0
            stop_dist = risk_pts * mult
            sign = 1 if event.direction == "long" else -1
            stop_price = entry_price - sign * stop_dist
            target_price = entry_price + sign * stop_dist * strategy.rr_target

            bracket = self.order_router.build_bracket_spec(
                direction=event.direction,
                symbol=self.contract_symbol,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                qty=event.contracts,
            )
            if bracket is None:
                log.warning("Bracket spec returned None for %s — NO CRASH PROTECTION", event.strategy_id)
                return
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.order_router.submit, bracket)
            # Store bracket order IDs on the position record
            record = self._positions.get(event.strategy_id)
            if record is not None:
                bracket_ids = result.get("order_ids", []) if isinstance(result, dict) else []
                if not bracket_ids:
                    oid = result.get("order_id") if isinstance(result, dict) else getattr(result, "order_id", None)
                    if oid:
                        bracket_ids = [oid]
                record.bracket_order_ids = bracket_ids
            log.info("Bracket submitted for %s: stop=%.2f target=%.2f", event.strategy_id, stop_price, target_price)
            self._stats.brackets_submitted += 1
        except Exception as e:
            self._stats.brackets_failed += 1
            log.warning("Bracket submit failed for %s (position still managed by engine): %s", event.strategy_id, e)

    async def _cancel_brackets(self, strategy_id: str) -> None:
        """Cancel bracket orders before submitting exit. Never raises.

        R2-H4: Strategy-scoped — only cancels the specific bracket order IDs
        stored on this strategy's PositionRecord. Does NOT sweep contract-wide,
        which would nuke other strategies' stop/target protection.

        Contract-wide orphan cleanup is handled separately at startup (line ~254).
        """
        if self.order_router is None:
            return
        loop = asyncio.get_running_loop()

        record = self._positions.get(strategy_id)
        if record is None or not record.bracket_order_ids:
            log.warning(
                "No bracket order IDs stored for %s — skipping bracket cancel "
                "(position may have engine-only stop/target management)",
                strategy_id,
            )
            return

        for oid in record.bracket_order_ids:
            try:
                await loop.run_in_executor(None, self.order_router.cancel, oid)
                self._stats.bracket_cancels_ok += 1
            except Exception as e:
                self._stats.bracket_cancels_failed += 1
                log.warning("Bracket cancel failed for %s order %s (may have filled): %s", strategy_id, oid, e)

        # Brief pause to let cancellations settle at broker before exit order
        await asyncio.sleep(0.5)

    EXIT_RETRY_MAX = 3
    EXIT_RETRY_BACKOFF = 1.0  # seconds, linear: 1s, 2s, 3s

    async def _submit_exit_with_retry(self, spec, strategy_id: str):
        """Submit an exit order with retry. ENTRY orders are NOT retried (miss = acceptable).

        Returns the broker result dict/dataclass on success.
        Raises on final failure after all retries exhausted.
        """
        assert self.order_router is not None, (
            "_submit_exit_with_retry called but order_router is None "
            "(signal_only=True). Live-only code path reached in signal mode."
        )
        loop = asyncio.get_running_loop()
        for attempt in range(self.EXIT_RETRY_MAX):
            try:
                result = await loop.run_in_executor(None, self.order_router.submit, spec)
                self._circuit_breaker.record_success()
                return result
            except Exception as e:
                # "already flat" is not a retryable error — position was closed by bracket
                err_msg = str(e).lower()
                if "no position" in err_msg or "already flat" in err_msg:
                    raise  # let caller handle bracket-closed case
                if attempt < self.EXIT_RETRY_MAX - 1:
                    wait = self.EXIT_RETRY_BACKOFF * (attempt + 1)
                    log.warning(
                        "Exit retry %d/%d for %s: %s (next attempt in %.0fs)",
                        attempt + 1,
                        self.EXIT_RETRY_MAX,
                        strategy_id,
                        e,
                        wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    self._circuit_breaker.record_failure()
                    msg = (
                        f"EXIT FAILED after {self.EXIT_RETRY_MAX} retries for "
                        f"{strategy_id}: {e} — MANUAL CLOSE REQUIRED"
                    )
                    log.critical(msg)
                    self._notify(msg)
                    raise

    async def _retry_stuck_exit(self, record) -> None:
        """R2-C6: Re-attempt close for a PENDING_EXIT position stuck > 300s.

        Called from _on_bar stale detection. Refreshes auth, then retries the exit.
        On success: removes from tracker. On failure: blocks new entries for strategy
        and sends CRITICAL alert with actionable details.
        """
        assert self.order_router is not None, (
            "_retry_stuck_exit called but order_router is None "
            "(signal_only=True). Live-only code path reached in signal mode."
        )
        sid = record.strategy_id
        log.critical("STUCK EXIT RECOVERY: re-attempting close for %s", sid)
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.auth.refresh_if_needed)
        except Exception as e:
            log.warning("Auth refresh failed before stuck exit retry for %s: %s", sid, e)

        if record.direction is None:
            msg = f"STUCK EXIT: {sid} has no direction — MANUAL CLOSE REQUIRED"
            log.critical(msg)
            self._notify(msg)
            self._block_strategy(sid, "Stuck exit without direction — manual close required")
            return

        exit_spec = self.order_router.build_exit_spec(
            direction=record.direction,
            symbol=self.contract_symbol,
            qty=record.contracts,
        )
        try:
            result = await self._submit_exit_with_retry(exit_spec, sid)
            order_id = result.get("order_id") if isinstance(result, dict) else getattr(result, "order_id", None)
            self._positions.on_exit_filled(sid)
            msg = f"STUCK EXIT RECOVERED: {sid} closed → orderId={order_id}"
            log.critical(msg)
            self._notify(msg)
        except Exception as e:
            msg = (
                f"STUCK EXIT RETRY FAILED: {sid} still PENDING_EXIT after recovery attempt. "
                f"Direction={record.direction}, contracts={record.contracts}. "
                f"Error: {e}. MANUAL CLOSE REQUIRED — new entries BLOCKED for this strategy."
            )
            log.critical(msg)
            self._notify(msg)
            self._block_strategy(sid, "Stuck exit retry failed — manual close required")
            # Reset stale timer to throttle retries — without this, _retry_stuck_exit
            # fires on every bar gap (>180s), generating repeated exit attempts and
            # notification spam. With this, next retry is at least 300s away.
            record.state_changed_at = datetime.now(UTC)

    async def _handle_event(self, event) -> None:
        """
        Handle a TradeEvent from ExecutionEngine.
        TradeEvent fields: event_type, strategy_id, timestamp, price, direction, contracts, reason
        event.price = entry fill on ENTRY events, exit fill on EXIT events
        There is NO entry_model, pnl_r, or expectancy_r on TradeEvent.

        HTTP order submission runs in a thread-pool executor so the async event loop
        (and the Tradovate heartbeat task) are never blocked.
        """
        self._stats.events_processed += 1
        strategy = self._strategy_map.get(event.strategy_id)
        if strategy is None:
            log.warning("Unknown strategy_id: %s", event.strategy_id)
            return

        if event.event_type == "ENTRY":
            # Orphan containment gate (applies to both signal-only and live)
            if event.strategy_id in self._blocked_strategies:
                reason = self._blocked_strategy_reasons.get(event.strategy_id, "Blocked pending manual review.")
                if "orphan" in reason.lower() or "manual close" in reason.lower():
                    msg = (
                        f"ENTRY BLOCKED — {event.strategy_id} has orphaned position at broker. "
                        "Resolve orphan before new entries."
                    )
                    record_type = "ENTRY_BLOCKED_ORPHAN"
                else:
                    msg = f"ENTRY BLOCKED — {event.strategy_id} is paused. {reason}"
                    record_type = "ENTRY_BLOCKED_PAUSED"
                log.critical(msg)
                self._notify(msg)
                self._write_signal_record({"type": record_type, "strategy_id": event.strategy_id, "reason": reason})
                return

            # ORB cap check — risk management gate (prevents oversized trades).
            # event.risk_points = stop distance in points (abs(entry - stop)), set
            # by ExecutionEngine. With 0.75x stops, risk_points ~ 0.75 * ORB range.
            # Cap at 150 pts risk = $300 max loss per trade on MNQ ($2/pt).
            orb_cap = self._orb_caps.get((strategy.orb_label, strategy.instrument))
            if orb_cap is not None and event.risk_points is not None:
                if event.risk_points >= orb_cap:
                    log.info(
                        "ORB_CAP_SKIP: %s/%s risk=%.1f pts >= cap=%.1f pts. Trade skipped.",
                        strategy.orb_label,
                        strategy.instrument,
                        event.risk_points,
                        orb_cap,
                    )
                    self._stats.orb_cap_skips += 1
                    self._write_signal_record(
                        {
                            "type": "ORB_CAP_SKIP",
                            "strategy_id": event.strategy_id,
                            "risk_pts": event.risk_points,
                            "cap_pts": orb_cap,
                        }
                    )
                    return

            # Per-trade max risk guard — dollar-based account-level cap.
            # Converts risk_points to dollars via cost_spec.point_value.
            if self._max_risk_per_trade is not None and event.risk_points is not None:
                risk_dollars = event.risk_points * self.cost_spec.point_value * event.contracts
                if risk_dollars > self._max_risk_per_trade:
                    log.warning(
                        "MAX_RISK_SKIP: %s risk=$%.0f > cap=$%.0f (%.1f pts × $%.2f × %d ct). Trade skipped.",
                        event.strategy_id,
                        risk_dollars,
                        self._max_risk_per_trade,
                        event.risk_points,
                        self.cost_spec.point_value,
                        event.contracts,
                    )
                    self._write_signal_record(
                        {
                            "type": "MAX_RISK_SKIP",
                            "strategy_id": event.strategy_id,
                            "risk_dollars": round(risk_dollars, 2),
                            "cap_dollars": self._max_risk_per_trade,
                        }
                    )
                    return

            # HWM DD halt gate — fail-closed on prop accounts.
            # Checked on every entry (not just every 10 bars) to close the gap
            # between periodic equity polls and actual order submission.
            if self._hwm_tracker is not None:
                halted, reason = self._hwm_tracker.check_halt()
                if halted:
                    log.critical(
                        "ENTRY BLOCKED — HWM DD halt: %s (strategy=%s)",
                        reason,
                        event.strategy_id,
                    )
                    self._notify(f"ENTRY BLOCKED (DD HALT): {event.strategy_id} — {reason}")
                    self._write_signal_record({"type": "ENTRY_BLOCKED_DD_HALT", "strategy_id": event.strategy_id})
                    return

            # Regime gate — block entries for strategies paused by allocator.
            # Session regime COLD = negative 6mo trailing ExpR. Loaded from
            # lane_allocation.json at init. Fail-open if file missing.
            if event.strategy_id in self._regime_paused:
                log.info(
                    "REGIME_PAUSED: %s — session COLD per allocator. Entry blocked.",
                    event.strategy_id,
                )
                self._write_signal_record({"type": "REGIME_PAUSED", "strategy_id": event.strategy_id})
                return

            if self.signal_only:
                record = self._positions.on_signal_entry(
                    event.strategy_id, event.price, event.direction, contracts=event.contracts
                )
                if record is None:
                    log.warning("Duplicate entry REJECTED for %s (signal-only)", event.strategy_id)
                    return
                log.info(
                    "⚡ SIGNAL [%s]: %s %s @ %.2f  ← trade this manually on Tradovate/TradingView",
                    event.strategy_id,
                    event.direction.upper(),
                    self.contract_symbol,
                    event.price,
                )
                # Persist signal entry to journal (fail-open)
                trade_id = generate_trade_id()
                record.journal_trade_id = trade_id
                self.journal.record_entry(
                    trade_id=trade_id,
                    trading_day=self.trading_day,
                    instrument=self.instrument,
                    strategy_id=event.strategy_id,
                    direction=event.direction,
                    entry_model=strategy.entry_model,
                    engine_entry=event.price,
                    contracts=event.contracts,
                )

                self._write_signal_record(
                    {
                        "type": "SIGNAL_ENTRY",
                        "strategy_id": event.strategy_id,
                        "contract": self.contract_symbol,
                        "direction": event.direction.upper(),
                        "price": event.price,
                        "contracts": event.contracts,
                    }
                )
                return

            # Past the signal_only early return: live mode is active.
            # order_router is non-None whenever signal_only=False (see __init__ L266-297).
            assert self.order_router is not None, (
                "_handle_event ENTRY: order_router is None but signal_only=False — broken invariant from __init__."
            )

            if not self._circuit_breaker.should_allow_request():
                log.critical("CIRCUIT BREAKER OPEN — skipping ENTRY for %s", event.strategy_id)
                self._notify(f"CIRCUIT BREAKER OPEN — skipping ENTRY for {event.strategy_id}")
                self._write_signal_record({"type": "CIRCUIT_BREAKER", "strategy_id": event.strategy_id})
                return

            # Post-market buffer: block entries within 10 minutes of firm close
            mins_to_close = self._minutes_to_close_et()
            if mins_to_close is not None and mins_to_close <= 10.0:
                msg = f"POST-MARKET BUFFER: skipping ENTRY for {event.strategy_id} ({mins_to_close:.0f}min to close)"
                log.warning(msg)
                self._notify(msg)
                return

            # Check position tracker BEFORE broker submit — reject duplicates
            # before they become orphaned broker orders
            pre_record = self._positions.on_entry_sent(
                event.strategy_id, event.direction, event.price, contracts=event.contracts
            )
            if pre_record is None:
                log.warning("Duplicate entry REJECTED for %s — not submitting to broker", event.strategy_id)
                return

            spec = self.order_router.build_order_spec(
                direction=event.direction,
                entry_model=strategy.entry_model,
                entry_price=event.price,
                symbol=self.contract_symbol,
                qty=event.contracts,
            )

            # Merge bracket into entry for atomic submission (native brackets only)
            _bracket_merged = False
            if self.order_router.supports_native_brackets():
                risk_pts = event.risk_points or strategy.median_risk_points
                if risk_pts:
                    mult = getattr(strategy, "stop_multiplier", 1.0) or 1.0
                    stop_dist = risk_pts * mult
                    sign = 1 if event.direction == "long" else -1
                    stop_price = event.price - sign * stop_dist
                    target_price = event.price + sign * stop_dist * strategy.rr_target
                    bracket = self.order_router.build_bracket_spec(
                        direction=event.direction,
                        symbol=self.contract_symbol,
                        entry_price=event.price,
                        stop_price=stop_price,
                        target_price=target_price,
                        qty=event.contracts,
                    )
                    if bracket:
                        spec = self.order_router.merge_bracket_into_entry(spec, bracket)
                        _bracket_merged = True
                        log.info(
                            "Bracket merged into entry for %s: risk=%.2f rr=%.1f",
                            event.strategy_id,
                            stop_dist,
                            strategy.rr_target,
                        )
                    else:
                        log.critical(
                            "NAKED POSITION RISK: build_bracket_spec returned None for %s",
                            event.strategy_id,
                        )
                else:
                    log.critical(
                        "NAKED POSITION RISK: no risk_points for %s — cannot build bracket "
                        "(risk_points=%s, median_risk_points=%s)",
                        event.strategy_id,
                        event.risk_points,
                        strategy.median_risk_points,
                    )

            # SAFETY GATE: refuse entry without bracket protection in live/demo mode
            if not _bracket_merged and not self.signal_only:
                if self.order_router.supports_native_brackets():
                    msg = (
                        f"ENTRY BLOCKED: {event.strategy_id} — no bracket protection. "
                        f"Position would be NAKED (no stop loss). Refusing entry."
                    )
                    log.critical(msg)
                    self._notify(msg)
                    self._positions.pop(event.strategy_id)
                    return

            loop = asyncio.get_running_loop()
            try:
                result = await loop.run_in_executor(None, self.order_router.submit, spec)
                self._circuit_breaker.record_success()
            except Exception as e:
                self._circuit_breaker.record_failure()
                log.error("ENTRY order failed for %s: %s", event.strategy_id, e)
                # Rollback position tracker — order never reached broker.
                # Safe: single event loop + GIL means no concurrent modification.
                self._positions.pop(event.strategy_id)
                return
            order_id = result.get("order_id") if isinstance(result, dict) else getattr(result, "order_id", None)
            raw_fill = result.get("fill_price") if isinstance(result, dict) else getattr(result, "fill_price", None)
            fill_price = self._validate_fill_price(raw_fill, f"ENTRY {event.strategy_id}")

            # Update tracker with broker order_id
            pre_record.entry_order_id = order_id
            if fill_price is not None:
                self._positions.on_entry_filled(event.strategy_id, fill_price)
                slippage = fill_price - event.price
                log.info(
                    "ENTRY FILL: %s %s engine=%.2f fill=%.2f slip=%+.4f pts → orderId=%s",
                    event.strategy_id,
                    event.direction,
                    event.price,
                    fill_price,
                    slippage,
                    order_id,
                )
            else:
                log.info(
                    "ENTRY order: %s %s @ %.2f → orderId=%s (fill pending)",
                    event.strategy_id,
                    event.direction,
                    event.price,
                    order_id,
                )

            # Persist entry to trade journal (fail-open)
            trade_id = generate_trade_id()
            pre_record.journal_trade_id = trade_id
            self.journal.record_entry(
                trade_id=trade_id,
                trading_day=self.trading_day,
                instrument=self.instrument,
                strategy_id=event.strategy_id,
                direction=event.direction,
                entry_model=strategy.entry_model,
                engine_entry=event.price,
                fill_entry=fill_price,
                broker=self._broker_name,
                order_id_entry=order_id,
                contracts=event.contracts,
            )

            self._write_signal_record(
                {
                    "type": "ORDER_ENTRY",
                    "strategy_id": event.strategy_id,
                    "contract": self.contract_symbol,
                    "direction": event.direction.upper(),
                    "price": event.price,
                    "fill_price": fill_price,
                    "contracts": event.contracts,
                    "order_id": order_id,
                }
            )

            # Bracket handling: native brackets were merged into entry above;
            # non-native brackets get submitted separately post-fill.
            if _bracket_merged:
                record = self._positions.get(event.strategy_id)
                if record is not None and order_id and self.order_router.has_queryable_bracket_legs():
                    # Verify bracket legs were actually created at the broker.
                    # ProjectX creates bracket legs as separate orders with IDs entry_id+1 (SL)
                    # and entry_id+2 (TP), tagged with 'AutoBracket'.
                    try:
                        sl_id, tp_id = self.order_router.verify_bracket_legs(order_id, self.contract_symbol)
                        if sl_id and tp_id:
                            record.bracket_order_ids = [sl_id, tp_id]
                            log.info(
                                "BRACKET VERIFIED: %s SL=%d TP=%d",
                                event.strategy_id,
                                sl_id,
                                tp_id,
                            )
                        else:
                            missing = []
                            if not sl_id:
                                missing.append("SL")
                            if not tp_id:
                                missing.append("TP")
                            msg = (
                                f"BRACKET LEGS MISSING for {event.strategy_id}: "
                                f"{', '.join(missing)} not found after entry fill. "
                                f"POSITION MAY BE UNPROTECTED."
                            )
                            log.critical(msg)
                            self._notify(msg)
                            # Store what we have — partial protection is better than none
                            record.bracket_order_ids = [x for x in [sl_id, tp_id] if x]
                    except Exception as e:
                        log.error("Bracket verification failed for %s: %s", event.strategy_id, e)
                        if order_id is not None:
                            # Fallback: assume bracket IDs are sequential (best guess)
                            record.bracket_order_ids = [order_id + 1, order_id + 2]
                        else:
                            log.critical(
                                "BRACKET FALLBACK IMPOSSIBLE: order_id is None for %s — "
                                "cannot derive bracket leg IDs. Position may be UNPROTECTED.",
                                event.strategy_id,
                            )
                            self._notify(
                                f"NAKED POSITION RISK: bracket verification failed and "
                                f"order_id is None for {event.strategy_id}"
                            )
                self._stats.brackets_submitted += 1
            else:
                actual_entry = fill_price if fill_price is not None else event.price
                await self._submit_bracket(event, strategy, actual_entry)

        elif event.event_type in ("EXIT", "SCRATCH"):
            entry_price = self._positions.best_entry_price(event.strategy_id, event.price)
            if self._positions.get(event.strategy_id) is None:
                log.warning("EXIT for %s with no prior ENTRY — using engine exit price as fallback", event.strategy_id)

            if self.signal_only:
                log.info(
                    "⚡ EXIT SIGNAL [%s]: close %s @ %.2f  ← close manually",
                    event.strategy_id,
                    event.direction.upper(),
                    event.price,
                )
                pos_rec = self._positions.get(event.strategy_id)
                jtid = pos_rec.journal_trade_id if pos_rec else None
                self._record_exit(
                    event,
                    entry_price,
                    entry_slippage=pos_rec.entry_slippage if pos_rec else None,
                    journal_trade_id=jtid,
                )
                self._positions.pop(event.strategy_id)
                self._write_signal_record(
                    {
                        "type": "SIGNAL_EXIT",
                        "strategy_id": event.strategy_id,
                        "contract": self.contract_symbol,
                        "direction": event.direction.upper(),
                        "price": event.price,
                    }
                )
                return

            # Past the signal_only early return: live mode is active.
            assert self.order_router is not None, (
                "_handle_event EXIT: order_router is None but signal_only=False — broken invariant from __init__."
            )

            # Cancel any broker-side bracket orders before submitting exit
            await self._cancel_brackets(event.strategy_id)

            # Submit close order and capture exit fill
            # NOTE: exits NEVER blocked by circuit breaker — can't leave positions open
            if not self._circuit_breaker.should_allow_request():
                msg = f"CIRCUIT BREAKER OPEN — EXIT for {event.strategy_id} submitted anyway"
                log.critical(msg)
                self._notify(msg)

            exit_pos_rec = self._positions.get(event.strategy_id)
            exit_jtid = exit_pos_rec.journal_trade_id if exit_pos_rec else None
            self._positions.on_exit_sent(event.strategy_id)
            exit_spec = self.order_router.build_exit_spec(
                direction=event.direction,
                symbol=self.contract_symbol,
                qty=event.contracts,
            )
            try:
                result = await self._submit_exit_with_retry(exit_spec, event.strategy_id)
            except Exception as e:
                # Handle "already flat" from bracket filling between cancel and exit
                err_msg = str(e).lower()
                if "no position" in err_msg or "already flat" in err_msg:
                    log.info("Position already closed by bracket for %s — skipping exit order", event.strategy_id)
                    closed_rec = self._positions.on_exit_filled(event.strategy_id)
                    self._record_exit(
                        event,
                        entry_price,
                        entry_slippage=closed_rec.entry_slippage if closed_rec else None,
                        journal_trade_id=exit_jtid,
                    )
                    return
                # All retries exhausted — position remains open at broker, stuck in PENDING_EXIT.
                # Recovery paths: (1) stale_positions detects after 300s,
                # (2) kill switch flattens on feed death, (3) manual close.
                # Note: if this strategy later becomes a rollover orphan, the orphan
                # containment gate (_blocked_strategies) prevents new entries until
                # manual resolution — so "new entry overwrites PENDING_EXIT" is NOT
                # a recovery path for orphaned strategies.
                #
                # _submit_exit_with_retry already notified at the retry-exhaustion level.
                # This second notify is intentional — belt-and-suspenders for overnight failures.
                stuck_msg = (
                    f"EXIT FAILED — {event.strategy_id} stuck in PENDING_EXIT. "
                    "Position open at broker. Kill switch or manual close required."
                )
                log.critical(stuck_msg)
                self._notify(stuck_msg)
                self._write_signal_record({"type": "EXIT_FAILED", "strategy_id": event.strategy_id, "error": str(e)})
                return
            order_id = result.get("order_id") if isinstance(result, dict) else getattr(result, "order_id", None)
            raw_exit_fill = (
                result.get("fill_price") if isinstance(result, dict) else getattr(result, "fill_price", None)
            )
            exit_fill = self._validate_fill_price(raw_exit_fill, f"EXIT {event.strategy_id}")

            if exit_fill is not None:
                exit_slip = exit_fill - event.price
                log.info(
                    "EXIT FILL: %s %s engine=%.2f fill=%.2f slip=%+.4f pts → orderId=%s",
                    event.strategy_id,
                    event.direction,
                    event.price,
                    exit_fill,
                    exit_slip,
                    order_id,
                )
            else:
                log.info(
                    "%s close order: %s %s @ %.2f → orderId=%s",
                    event.event_type,
                    event.strategy_id,
                    event.direction,
                    event.price,
                    order_id,
                )

            closed_rec = self._positions.on_exit_filled(event.strategy_id, fill_price=exit_fill)
            self._record_exit(
                event,
                entry_price,
                exit_fill_price=exit_fill,
                entry_slippage=closed_rec.entry_slippage if closed_rec else None,
                journal_trade_id=exit_jtid,
                order_id_exit=order_id,
            )
            self._write_signal_record(
                {
                    "type": f"ORDER_{event.event_type}",
                    "strategy_id": event.strategy_id,
                    "contract": self.contract_symbol,
                    "direction": event.direction.upper(),
                    "price": event.price,
                    "fill_price": exit_fill,
                }
            )

        elif event.event_type == "REJECT":
            log.warning("REJECT: %s — %s", event.strategy_id, event.reason)
            self._notify(f"REJECTED: {event.strategy_id} — {event.reason}")
            self._write_signal_record(
                {
                    "type": "REJECT",
                    "strategy_id": event.strategy_id,
                    "reason": getattr(event, "reason", ""),
                }
            )

    # Kill switch: emergency flatten if feed dies with open positions.
    # 5 minutes of silence = assume feed is dead. This is the last line of defense.
    KILL_SWITCH_TIMEOUT = 300.0  # seconds without a bar before emergency flatten
    KILL_SWITCH_CHECK_INTERVAL = 30.0  # how often the watchdog checks

    async def _emergency_flatten(self) -> None:
        """Nuclear option: market-close every open position immediately.

        Runs when the feed is dead and we're blind with open exposure.
        Retries aggressively — the goal is to get flat no matter what.
        """
        active = self._positions.active_positions()
        if not active:
            return

        msg = f"KILL SWITCH: Feed dead >{self.KILL_SWITCH_TIMEOUT:.0f}s with {len(active)} open position(s). Emergency flatten ALL."
        log.critical(msg)
        self._notify(msg)
        self._write_signal_record(
            {
                "type": "KILL_SWITCH",
                "reason": "feed_dead",
                "positions": [r.strategy_id for r in active],
            }
        )

        if self.signal_only or self.order_router is None:
            msg = f"MANUAL CLOSE REQUIRED: Signal-only mode — flatten {[r.strategy_id for r in active]} NOW"
            log.critical(msg)
            self._notify(msg)
            return

        loop = asyncio.get_running_loop()
        for record in active:
            if record.direction is None:
                msg = (
                    f"KILL SWITCH: {record.strategy_id} has no direction — "
                    f"CANNOT FLATTEN SAFELY. MANUAL CLOSE REQUIRED."
                )
                log.critical(msg)
                self._notify(msg)
                continue

            # R2-C4: Cancel bracket legs BEFORE exit to prevent orphaned orders
            # opening unwanted positions after the flatten closes us out.
            if record.bracket_order_ids and self.order_router is not None:
                for oid in record.bracket_order_ids:
                    try:
                        await loop.run_in_executor(None, self.order_router.cancel, oid)
                    except Exception as e:
                        log.warning(
                            "KILL SWITCH: bracket cancel failed for %s order %s (proceeding): %s",
                            record.strategy_id,
                            oid,
                            e,
                        )

            direction = record.direction
            for attempt in range(3):
                try:
                    # Sync call intentional: emergency flatten is time-critical.
                    # Blocking the event loop for ~100ms is acceptable when the
                    # bot is in KILL_SWITCH state — no other processing matters.
                    self.auth.refresh_if_needed()
                    exit_spec = self.order_router.build_exit_spec(
                        direction=direction,
                        symbol=self.contract_symbol,
                        qty=record.contracts,
                    )
                    result = await loop.run_in_executor(None, self.order_router.submit, exit_spec)
                    order_id = result.get("order_id") if isinstance(result, dict) else getattr(result, "order_id", None)
                    msg = f"KILL SWITCH FLATTEN: {record.strategy_id} {direction} → orderId={order_id} (attempt {attempt + 1})"
                    log.critical(msg)
                    self._notify(msg)
                    self._positions.on_exit_filled(record.strategy_id)
                    # Persist kill switch exit to journal (fail-open)
                    if record.journal_trade_id:
                        self.journal.record_exit(
                            trade_id=record.journal_trade_id,
                            exit_reason="kill_switch",
                            order_id_exit=order_id,
                        )
                    break
                except Exception as e:
                    msg = f"KILL SWITCH FLATTEN FAILED: {record.strategy_id} attempt {attempt + 1}/3 — {e}"
                    log.critical(msg)
                    self._notify(msg)
                    await asyncio.sleep(2**attempt)
            else:
                msg = f"MANUAL CLOSE REQUIRED: Failed to flatten {record.strategy_id} after 3 attempts"
                log.critical(msg)
                self._notify(msg)

    async def _watchdog(self) -> None:
        """Independent watchdog task — fires kill switch if feed goes silent.

        Runs on its own asyncio schedule, not dependent on bar arrival.
        This is the fail-safe that protects against the feed dying silently.
        The watchdog MUST NOT die — it wraps everything in try/except so that
        a transient error doesn't kill the last line of defense.
        """
        while True:
            try:
                await asyncio.sleep(self.KILL_SWITCH_CHECK_INTERVAL)

                if self._kill_switch_fired:
                    continue  # already fired, don't spam

                if self._last_bar_at is None:
                    continue  # haven't received any bars yet

                gap = (datetime.now(UTC) - self._last_bar_at).total_seconds()
                if gap > self.KILL_SWITCH_TIMEOUT and self._positions.active_positions():
                    self._fire_kill_switch()
                    await self._emergency_flatten()
            except asyncio.CancelledError:
                raise  # normal shutdown
            except Exception as e:
                log.error("Watchdog error (will retry): %s", e)

    HEARTBEAT_INTERVAL = 1800.0  # 30 minutes between heartbeat notifications

    async def _heartbeat_notifier(self) -> None:
        """Send periodic alive notification. Absence of heartbeat = notifications broken."""
        # Emit immediate heartbeat so user knows session is alive without waiting 30 min
        try:
            n_strategies = len(self.portfolio.strategies)
            mode = "SIGNAL" if self.signal_only else ("DEMO" if self.demo else "LIVE")
            self._notify(f"Heartbeat: session alive, {n_strategies} strategies loaded ({mode})")
        except Exception as e:
            log.error("Initial heartbeat failed: %s", e)
        while True:
            try:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                n_trades = self.monitor.trade_count
                active = len(self._positions.active_positions())
                poller_status = "ON" if self._poller_active else "OFF"
                self._notify(
                    f"Heartbeat: {self._bar_count} bars, {n_trades} trades, {active} active, poller={poller_status}"
                )
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.error("Heartbeat error (will retry): %s", e)

    FILL_POLL_INTERVAL = 5.0  # seconds between fill status checks

    async def _fill_poller(self) -> None:
        """Poll PENDING_ENTRY orders for fill confirmation every 5s.

        E2 stop-market orders rest until price reaches stop level. Without polling,
        PositionTracker stays PENDING_ENTRY indefinitely. This background task
        detects fills and cancellations.
        """
        assert self.order_router is not None, (
            "_fill_poller started but order_router is None (signal_only=True). "
            "Live-only background task reached in signal mode."
        )
        while True:
            try:
                await asyncio.sleep(self.FILL_POLL_INTERVAL)
                self._poller_active = True
                pending_count = 0
                for record in self._positions.active_positions():
                    if record.state != PositionState.PENDING_ENTRY:
                        continue
                    if record.entry_order_id is None:
                        continue
                    pending_count += 1
                    try:
                        loop = asyncio.get_running_loop()
                        status = await loop.run_in_executor(
                            None, self.order_router.query_order_status, record.entry_order_id
                        )
                        # Re-check state after await — another coroutine may have handled this
                        current = self._positions.get(record.strategy_id)
                        if current is None or current.state != PositionState.PENDING_ENTRY:
                            continue
                        self._stats.fill_polls_run += 1
                        if status["status"] == "Filled":
                            raw_poll_fill = status.get("fill_price")
                            fill_price = self._validate_fill_price(raw_poll_fill, f"POLL {record.strategy_id}")
                            if fill_price is not None:
                                self._positions.on_entry_filled(record.strategy_id, fill_price)
                                # Update journal with confirmed fill price
                                if record.journal_trade_id:
                                    self.journal.update_entry_fill(
                                        trade_id=record.journal_trade_id,
                                        fill_entry=fill_price,
                                    )
                            log.info("Fill confirmed for %s: %s", record.strategy_id, status)
                            self._stats.fill_polls_confirmed += 1
                        elif status["status"] in ("Cancelled", "Rejected"):
                            self._positions.pop(record.strategy_id)
                            # R2-C3: notify engine to remove ghost trade from active_trades.
                            # Without this, the engine keeps emitting EXIT events for a
                            # position that was never filled at the broker.
                            self.engine.cancel_trade(record.strategy_id)
                            log.warning("Order %s for %s: %s", status["status"], record.strategy_id, status)
                            self._notify(f"Order {status['status']}: {record.strategy_id} — entry cancelled by broker")
                    except NotImplementedError:
                        log.debug("Fill poller: %s broker does not support order polling", self._broker_name)
                    except Exception as e:
                        self._stats.fill_polls_failed += 1
                        log.warning("Fill poll failed for %s: %s", record.strategy_id, e)
                if pending_count > 0:
                    log.info("Fill poller: checked %d pending orders", pending_count)
            except asyncio.CancelledError:
                return
            except Exception:
                log.exception("Fill poller iteration error — continuing")  # poller must never crash

    # Orchestrator-level reconnect: covers the case where the feed exhausts its
    # internal reconnects (20 attempts) and run() returns cleanly.
    ORCHESTRATOR_MAX_RECONNECTS = 5
    ORCHESTRATOR_BACKOFF_INITIAL = 30.0
    ORCHESTRATOR_BACKOFF_MAX = 300.0

    async def run(self) -> None:
        # Re-check trading day at run start — catches restart across day boundary
        bris_now = datetime.now(ZoneInfo("Australia/Brisbane"))
        if bris_now.hour < 9:
            actual_day = (bris_now - timedelta(days=1)).date()
        else:
            actual_day = bris_now.date()
        if actual_day != self.trading_day:
            log.warning(
                "Trading day corrected on run start: %s -> %s (init was stale)",
                self.trading_day,
                actual_day,
            )
            self.trading_day = actual_day

        # Market calendar checks — FAIL-LOUD on holidays, adjust on early close
        self._flatten_on_start = False
        try:
            from pipeline.market_calendar import (
                effective_close_et,
                is_cme_holiday,
                is_early_close,
                is_market_open_at,
            )

            utc_now = datetime.now(ZoneInfo("UTC"))
            us_date = datetime.now(ZoneInfo("America/New_York")).date()

            if is_cme_holiday(us_date):
                # Date says holiday — but verify market isn't actually open.
                # Sunday 6 PM ET = Sunday date, but CME opens for Monday's session.
                # Use is_market_open_at for ground truth.
                market_open_now = is_market_open_at(utc_now)
                market_opens_soon = is_market_open_at(utc_now + timedelta(hours=2))
                if not market_open_now and not market_opens_soon:
                    msg = (
                        f"CME HOLIDAY ({us_date}) — ALL SESSIONS BLOCKED. "
                        "Refusing to trade. Check cmegroup.com/holiday-calendar."
                    )
                    log.critical(msg)
                    self._notify(msg)
                    raise RuntimeError(msg)
                log.info(
                    "US date %s is non-session but market is open or opens soon (overnight session) — proceeding",
                    us_date,
                )

            if is_early_close(us_date):
                log.warning(
                    "EARLY CLOSE DAY (%s) — exchange closes at 12:00 PM CT / 1:00 PM ET",
                    us_date,
                )
                self._notify(f"Early close day ({us_date}). Afternoon sessions will not fire.")
                if self._close_hour_et is not None:
                    from datetime import time as dt_time

                    firm_close = dt_time(self._close_hour_et, self._close_min_et or 0)
                    eff = effective_close_et(us_date, firm_close_et=firm_close)
                    if eff is not None and eff < firm_close:
                        self._close_hour_et = eff.hour
                        self._close_min_et = eff.minute
                        log.warning(
                            "Force-flatten adjusted: %02d:%02d ET (was %s)",
                            eff.hour,
                            eff.minute,
                            firm_close,
                        )
                        mins = self._minutes_to_close_et()
                        if mins is not None and mins < 0:
                            log.critical(
                                "Adjusted close already passed (%d min ago) — "
                                "will emergency-flatten any open positions",
                                abs(int(mins)),
                            )
                            self._flatten_on_start = True
        except ImportError:
            log.critical("market_calendar not available — CANNOT verify holiday status")
            self._notify("WARNING: market_calendar import failed — holiday check SKIPPED")
        except RuntimeError:
            raise  # Re-raise the holiday block

        # Run component self-tests before accepting any bars
        results = self.run_self_tests()

        # Fail-closed: broken notifications in live trading = flying blind
        if not self.signal_only and not results.get("notifications", False):
            msg = "ABORTING: notifications broken — cannot trade without alerts"
            log.critical(msg)
            print(f"!!! {msg} !!!")
            raise RuntimeError(msg)

        backoff = self.ORCHESTRATOR_BACKOFF_INITIAL
        watchdog = asyncio.create_task(self._watchdog())
        heartbeat = asyncio.create_task(self._heartbeat_notifier())
        poller = None
        if not self.signal_only:
            poller = asyncio.create_task(self._fill_poller())
        try:
            for attempt in range(self.ORCHESTRATOR_MAX_RECONNECTS + 1):
                if self._kill_switch_fired:
                    msg = "Kill switch fired — not reconnecting"
                    log.critical(msg)
                    self._notify(msg)
                    return

                # Feedless brokers (e.g. Tradovate, webhook entry) must not reach
                # the feed-based reconnect loop. Assertion fires only if invariant
                # is broken — otherwise narrows type for pyright.
                assert self._feed_class is not None, (
                    f"SessionOrchestrator.run() reached feed loop for broker "
                    f"'{self._broker_name}' which has no market data feed. "
                    "Feedless brokers must use a different entry path (e.g. webhook_server)."
                )
                feed = self._feed_class(
                    self.auth,
                    on_bar=self._on_bar,
                    on_stale=self._on_feed_stale,
                    demo=self.demo,
                )
                log.info(
                    "Starting feed (attempt %d/%d): %s (broker: %s)",
                    attempt + 1,
                    self.ORCHESTRATOR_MAX_RECONNECTS + 1,
                    self.contract_symbol,
                    self._broker_name,
                )
                try:
                    self._last_bar_at = None  # reset heartbeat to avoid false watchdog trigger
                    await feed.run(self.contract_symbol)

                    # Distinguish stop-file exit from feed exhaustion
                    if feed.was_stopped:
                        log.info("Feed stopped by user — clean exit")
                        return
                    log.warning("Feed exited without stop request — reconnecting")

                except Exception as e:
                    log.critical("Feed crashed: %s", e)
                    self._notify(f"Feed crashed: {e}")

                if attempt < self.ORCHESTRATOR_MAX_RECONNECTS:
                    try:
                        await asyncio.get_running_loop().run_in_executor(None, self.auth.refresh_if_needed)
                    except Exception as exc:
                        log.warning("Auth refresh failed before reconnect: %s", exc)

                    # Re-resolve front-month contract (handles contract roll mid-session)
                    try:
                        from trading_app.live.projectx.contract_resolver import ProjectXContracts

                        _contracts = ProjectXContracts(auth=self.auth)
                        _contracts._contract_cache.clear()  # Force fresh resolution
                        new_symbol = _contracts.resolve_front_month(self.instrument)
                        if new_symbol != self.contract_symbol:
                            log.warning(
                                "CONTRACT ROLLED: %s -> %s (updating feed subscription)",
                                self.contract_symbol,
                                new_symbol,
                            )
                            self._notify(f"CONTRACT ROLLED: {self.contract_symbol} -> {new_symbol}")
                            self.contract_symbol = new_symbol
                        else:
                            log.info("Contract unchanged on reconnect: %s", self.contract_symbol)
                    except Exception as exc:
                        log.warning(
                            "Contract re-resolution failed on reconnect: %s (keeping %s)", exc, self.contract_symbol
                        )

                    self._stats.reconnect_attempts += 1
                    self._notify(f"Reconnecting in {backoff:.0f}s (attempt {attempt + 2})")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self.ORCHESTRATOR_BACKOFF_MAX)

            msg = f"Exhausted {self.ORCHESTRATOR_MAX_RECONNECTS} orchestrator reconnects — session dead"
            log.critical(msg)
            self._notify(f"CRITICAL: {msg}")
        finally:
            watchdog.cancel()
            heartbeat.cancel()
            if poller:
                poller.cancel()

    def post_session(self) -> None:
        """EOD: close open positions, log summary, run incremental backfill.

        Called from a synchronous finally block after asyncio.run() completes,
        so we start a fresh event loop for the async close operations.
        Each event is wrapped individually so one failure doesn't abort the rest
        (CRIT-4: preventing open positions from being abandoned on error).
        """
        # If kill switch already fired, positions are already closed at the broker.
        # Skip EOD close to avoid duplicate close orders.
        if self._kill_switch_fired:
            log.info("Kill switch was activated — skipping EOD close (positions already flattened)")
            eod_events = []
        else:
            eod_events = self.engine.on_trading_day_end()

        async def _close_all() -> None:
            for event in eod_events:
                try:
                    await self._handle_event(event)
                except Exception as e:
                    log.error(
                        "EOD close failed for %s (%s) — position may remain open: %s",
                        event.strategy_id,
                        event.event_type,
                        e,
                    )

        if eod_events and not self.signal_only:
            # Pre-refresh auth token before close loop — if network is back,
            # this ensures we have a valid token. If still down, we log CRITICAL.
            auth_ok = False
            for attempt in range(3):
                try:
                    self.auth.get_token()
                    auth_ok = True
                    break
                except Exception as e:
                    msg = f"Auth refresh attempt {attempt + 1}/3 failed before EOD close: {e}"
                    log.critical(msg)
                    self._notify(msg)
                    # BLOCKING SLEEP — acceptable here because:
                    # 1. post_session() runs AFTER asyncio.run() exits (no event loop)
                    # 2. Total worst-case block: 1+2+4 = 7 seconds across 3 retries
                    # 3. No positions can enter during post_session() — only exits
                    # 4. Risk window: market can move 7s during auth retry, but
                    #    bracket orders at the broker still protect the position
                    import time

                    time.sleep(min(2**attempt, 4))  # Cap at 4s to limit risk window
            if not auth_ok:
                msg = f"MANUAL CLOSE REQUIRED: Auth failed after 3 attempts. {len(eod_events)} position(s) may remain open"
                log.critical(msg)
                self._notify(msg)
            asyncio.run(_close_all())
        elif eod_events:
            asyncio.run(_close_all())

        summary = self.monitor.daily_summary()
        log.info("EOD summary: %s", summary)
        log.info("SESSION STATS: %s", self._stats)
        n_notifs_ok = self._stats.notifications_sent
        n_notifs_total = n_notifs_ok + self._stats.notifications_failed
        n_brackets_ok = self._stats.brackets_submitted
        n_brackets_total = n_brackets_ok + self._stats.brackets_failed
        self._notify(
            f"EOD: {summary.get('n_trades', 0)} trades, {summary.get('total_r', 0):.2f}R | "
            f"{self._stats.bars_received} bars, {n_notifs_ok}/{n_notifs_total} notifs OK, "
            f"{n_brackets_ok}/{n_brackets_total} brackets OK, "
            f"{self._stats.fill_polls_confirmed} fills confirmed, "
            f"{self._stats.reconnect_attempts} reconnects"
        )

        # Record session end in HWM tracker
        if self._hwm_tracker is not None and self.positions is not None and self.order_router is not None:
            try:
                end_equity = self.positions.query_equity(self.order_router.account_id)
                if end_equity is not None:
                    self._hwm_tracker.update_equity(end_equity)
                    self._hwm_tracker.record_session_end(end_equity)
                    _, reason = self._hwm_tracker.check_halt()
                    log.info("HWM session close: %s", reason)
            except Exception as e:
                log.warning("HWM session-end recording failed: %s", e)

        # EOD position reconciliation (M2.5 P0)
        if self.positions and not self.signal_only:
            context = " (post-kill-switch)" if self._kill_switch_fired else ""
            try:
                account_id = self.order_router.account_id if self.order_router else 0
                remaining = self.positions.query_open(account_id)
                if remaining:
                    msg = f"EOD RECONCILIATION{context}: {len(remaining)} positions still open after session end"
                    log.critical(msg)
                    self._notify(msg)
                    if self._kill_switch_fired:
                        msg = f"Kill switch flatten may have failed — MANUAL CLOSE REQUIRED for: {[r.get('contract_id', '?') for r in remaining]}"
                        log.critical(msg)
                        self._notify(msg)
                else:
                    log.info("EOD reconciliation%s: all positions flat", context)
            except NotImplementedError:
                log.warning("EOD reconciliation skipped — broker does not support position queries")
            except Exception as e:
                log.error("EOD position reconciliation failed: %s — cannot confirm positions are flat", e)
                self._notify(f"EOD RECON FAILED: {e} — manually verify all positions are flat")

        # Close trade journal — flushes any pending writes
        self.journal.close()

        # Persist captured bars to bars_1m (Databento-free daily pipeline)
        n_persisted = self._bar_persister.flush_to_db()
        if n_persisted > 0:
            log.info("Bar persister: %d bars written to bars_1m", n_persisted)

        # Clear crash-recovery state on clean session end.
        # If blocked strategies or kill switch fired, leave state for next startup.
        if not self._blocked_strategies and not self._kill_switch_fired:
            self._safety_state.clear()
        else:
            log.warning(
                "Safety state NOT cleared — kill_switch=%s, blocked=%d. "
                "State preserved for crash recovery on next startup.",
                self._kill_switch_fired,
                len(self._blocked_strategies),
            )

        try:
            run_backfill_for_instrument(self.instrument)
        except Exception as e:
            log.error("EOD backfill failed: %s", e)
