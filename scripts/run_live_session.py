"""
ORB Live Session — entry point.

THREE MODES (pick exactly one):

  --signal-only   Shows trade signals. YOU place orders manually on your platform.
                  No account connection needed. Safe to run any time.

  --demo          Auto-places orders on your broker's DEMO account (paper trading).
                  Requires broker credentials in .env. No real money.

  --live          Auto-places orders with REAL MONEY. Requires CONFIRM + broker creds.

INSTRUMENT SELECTION:

  --instrument MGC       Single instrument
  --all                  All active instruments (from ACTIVE_ORB_INSTRUMENTS)

Examples:
    python scripts/run_live_session.py --all --signal-only
    python scripts/run_live_session.py --instrument MGC --signal-only
    python scripts/run_live_session.py --instrument MGC --demo
    python scripts/run_live_session.py --instrument MGC --live
"""

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

from trading_app.live.instance_lock import acquire_instance_lock, release_instance_lock
from trading_app.live.session_orchestrator import SessionOrchestrator


def _run_lightweight_component_self_tests(
    *,
    instrument: str,
    components: dict[str, Any] | None = None,
) -> dict[str, bool]:
    """Probe notifications + broker bracket/fill-poller endpoints.

    Mirrors the production verifiers in `SessionOrchestrator._verify_brackets`
    and `SessionOrchestrator._verify_fill_poller` so preflight reports the
    same PASS/FAIL the live session would compute. `components` is the dict
    returned by `create_broker_components` (populated by `_check_auth` and
    threaded through `PreflightContext`). When `components` is None
    (auth failed upstream), broker probes are recorded as False so the
    summary line surfaces the gap instead of stubbing True.
    """

    results: dict[str, bool] = {}

    try:
        from trading_app.live.notifications import notify

        if notify(instrument, "SELF-TEST: notifications working"):
            results["notifications"] = True
        else:
            log.critical("NOTIFICATION SELF-TEST FAILED: notify() returned False")
            print("!!! NOTIFICATIONS ARE BROKEN (send_telegram raised; see log) !!!")
            results["notifications"] = False
    except Exception as e:
        log.critical("NOTIFICATION SELF-TEST FAILED: %s", e)
        print(f"!!! NOTIFICATIONS ARE BROKEN: {e} !!!")
        results["notifications"] = False

    results["brackets"] = _probe_brackets(components)
    results["fill_poller"] = _probe_fill_poller(components)
    return results


def _probe_brackets(components: dict[str, Any] | None) -> bool:
    """Probe broker bracket support without placing an order.

    Mirrors `SessionOrchestrator._verify_brackets` (session_orchestrator.py).
    Returns True when the broker advertises bracket support AND
    `build_bracket_spec` produces a non-None spec; False when the broker
    advertises support but the builder returns None or raises. Returns True
    in the "broker has no bracket support" case (matches the production
    verifier's signal-only / unsupported branch).
    """
    if components is None:
        return False
    try:
        router_cls = components["router_class"]
        auth = components["auth"]
        # account_id=0 is the canonical preflight sentinel — same value
        # _verify_fill_poller passes to query_order_status. No order is placed.
        router = router_cls(account_id=0, auth=auth)
        if not router.supports_native_brackets():
            log.info("Broker does not support native brackets — preflight reports PASS (no bracket required)")
            return True
        spec = router.build_bracket_spec(
            direction="long",
            symbol="TEST",
            entry_price=100.0,
            stop_price=99.0,
            target_price=102.0,
            qty=1,
        )
        if spec is None:
            log.warning("BRACKET PROBE FAILED: build_bracket_spec returned None despite supports_native_brackets=True")
            return False
        log.info("Bracket probe PASS")
        return True
    except Exception as e:  # noqa: BLE001 — router construction failure = probe failure
        log.critical("BRACKET PROBE FAILED: %s", e)
        return False


def _probe_fill_poller(components: dict[str, Any] | None) -> bool:
    """Probe broker fill-poller endpoint without consuming an order id.

    Mirrors `SessionOrchestrator._verify_fill_poller`: the only failure
    that flips the verdict is `NotImplementedError` (broker genuinely cannot
    poll). Any other exception means the endpoint exists and returned an
    expected error for the sentinel order_id=0 (typically 404/auth/validation).
    """
    if components is None:
        return False
    try:
        router_cls = components["router_class"]
        auth = components["auth"]
        router = router_cls(account_id=0, auth=auth)
        try:
            router.query_order_status(0)
        except NotImplementedError:
            log.warning("Broker does not support query_order_status — fill poller will be inactive")
            return False
        except Exception as e:  # noqa: BLE001 — endpoint-exists confirmation, not a real failure
            log.info("Fill poller endpoint exists (sentinel call returned non-fatal: %s)", e)
        log.info("Fill poller probe PASS")
        return True
    except Exception as e:
        log.critical("FILL POLLER PROBE FAILED (router construction): %s", e)
        return False


@dataclass
class PreflightContext:
    """Mutable state shared across preflight checks.

    `components` is set by check_auth and consumed by check_contracts +
    check_notifications. `components_all_pass` is set by check_notifications
    and read by the final summary print.
    """

    instrument: str
    broker_name: str
    demo: bool
    portfolio: Any  # Portfolio | None — typed loosely to avoid runtime import cycle
    components: dict[str, Any] | None = None
    components_all_pass: bool = True
    # Copy-trading dry-run inputs (A.6.5 — preflight gap fix 2026-05-16)
    profile_id: str | None = None
    requested_account_id: int | None = None
    signal_only: bool = False


@dataclass
class CheckResult:
    """Return shape for every preflight check."""

    passed: bool
    message: str  # printed inline after "[i/N] <name>... "


# Type alias for a check function. Each check reads/mutates ctx and returns
# a CheckResult. The runner enforces ordering and counts via len(checks).
CheckFn = Callable[[PreflightContext], CheckResult]


def _check_auth(ctx: PreflightContext) -> CheckResult:
    """Auth check"""
    from trading_app.live.broker_factory import create_broker_components

    try:
        ctx.components = create_broker_components(ctx.broker_name, demo=ctx.demo)
        token = ctx.components["auth"].get_token()
        return CheckResult(True, f"OK (token: {token[:8]}...)")
    except Exception as e:
        ctx.components = None
        return CheckResult(False, f"FAILED: {e}")


def _check_portfolio(ctx: PreflightContext) -> CheckResult:
    """Portfolio check"""
    try:
        if ctx.portfolio is None:
            raise RuntimeError(
                f"No portfolio injected for {ctx.instrument}. "
                "Pass --profile or --raw-baseline to build a portfolio from "
                "prop_profiles.ACCOUNT_PROFILES. build_live_portfolio() is DEPRECATED."
            )
        pf = ctx.portfolio
        # Detail lines are printed AFTER the result, mirroring the original
        # behaviour where the summary header lands on the inline line and the
        # per-strategy / NOTE lines stream below.
        head = f"OK ({len(pf.strategies)} strategies)"
        details = [
            f"    {s.strategy_id} | {s.orb_label} {s.entry_model} "
            f"RR{s.rr_target} O{s.orb_minutes} | WR={s.win_rate:.0%} "
            f"ExpR={s.expectancy_r:.3f} N={s.sample_size}"
            for s in pf.strategies
        ]
        details.append(f"    NOTE: Using injected portfolio ({len(pf.strategies)} strategies)")
        return CheckResult(True, "\n".join([head, *details]))
    except Exception as e:
        return CheckResult(False, f"FAILED: {e}")


def _check_daily_features(ctx: PreflightContext) -> CheckResult:
    """Daily features freshness"""
    # R2-M5: use Brisbane trading day, not date.today() (Windows system time).
    # For late-night sessions (NYSE_OPEN at 00:30 Brisbane), date.today() gives
    # yesterday's date, validating stale data and producing a false-positive pass.
    try:
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        bris_now = datetime.now(ZoneInfo("Australia/Brisbane"))
        if bris_now.hour < 9:
            preflight_trading_day = (bris_now - timedelta(days=1)).date()
        else:
            preflight_trading_day = bris_now.date()
        # Preflight check is deliberately at O5 reference aperture — health probe
        # for atr_20/atr_vel_regime population, not a per-lane scoring read.
        row = SessionOrchestrator._build_daily_features_row(preflight_trading_day, ctx.instrument, orb_minutes=5)
        atr = row.get("atr_20")
        vel = row.get("atr_vel_regime")
        if atr is None or vel is None:
            missing = []
            if atr is None:
                missing.append("atr_20")
            if vel is None:
                missing.append("atr_vel_regime")
            # Check if portfolio has filters that depend on atr_20 or atr_vel_regime
            # atr_20: PDR_* (PrevDayRangeNorm), GAP_* (GapNorm), X_*_ATR* (CrossAssetATR), ATR70/80 (OwnATRPct)
            # atr_vel_regime: ATR_VEL (ATRVelocity)
            atr_prefixes = ("PDR_", "GAP_", "X_MES_ATR", "X_MGC_ATR", "ATR70_VOL", "ATR_P")
            vel_prefixes = ("ATR_VEL",)
            needs_atr = ctx.portfolio is not None and (
                (atr is None and any(s.filter_type.startswith(atr_prefixes) for s in ctx.portfolio.strategies))
                or (vel is None and any(s.filter_type.startswith(vel_prefixes) for s in ctx.portfolio.strategies))
            )
            if needs_atr:
                return CheckResult(
                    False,
                    f"FAILED: {', '.join(missing)} = None and portfolio has "
                    f"ATR-dependent filters — run pipeline/build_daily_features.py",
                )
            return CheckResult(
                True,
                f"WARN: {', '.join(missing)} = None — run pipeline/build_daily_features.py",
            )
        return CheckResult(True, f"OK (atr_20={atr}, atr_vel={vel})")
    except Exception as e:
        return CheckResult(False, f"FAILED: {e}")


def _check_contracts(ctx: PreflightContext) -> CheckResult:
    """Contract resolution"""
    if ctx.components is None:
        return CheckResult(False, "SKIPPED (auth failed)")
    try:
        contracts_cls = ctx.components["contracts_class"]
        contracts = contracts_cls(auth=ctx.components["auth"], demo=ctx.demo)
        front = contracts.resolve_front_month(ctx.instrument)
        return CheckResult(True, f"OK ({front})")
    except Exception as e:
        return CheckResult(False, f"FAILED: {e}")


def _check_notifications(ctx: PreflightContext) -> CheckResult:
    """Notifications + broker bracket/fill-poller probes.

    Threads `ctx.components` (set by `_check_auth`) into the self-test helper
    so the bracket and fill-poller probes exercise the real router class
    rather than rubber-stamping True. Failed probes surface explicitly in
    the inline message (e.g. `OK · brackets:FAIL · fill_poller:FAIL`) so an
    operator scanning the preflight tail sees the gap before clicking Start.
    """
    try:
        if ctx.components is None:
            raise RuntimeError("auth failed")
        test_results = _run_lightweight_component_self_tests(
            instrument=ctx.instrument,
            components=ctx.components,
        )
        ctx.components_all_pass = all(test_results.values())
        # Build a deterministic per-component status suffix so the summary line
        # always lists every probe (PASS or FAIL), not just the failures.
        status_suffix = " · ".join(f"{name}:{'PASS' if ok else 'FAIL'}" for name, ok in test_results.items())
        if ctx.components_all_pass:
            return CheckResult(True, f"OK ({status_suffix})")
        # Probes are surfaced but non-blocking at preflight time; the live
        # SessionOrchestrator re-runs `run_self_tests` at startup and that path
        # is the authoritative gate.
        return CheckResult(True, f"WARNINGS ({status_suffix})")
    except Exception as e:
        return CheckResult(False, f"FAILED: {e}")


def _check_trade_journal(ctx: PreflightContext) -> CheckResult:
    """Trade journal health"""
    # session_orchestrator only enforces journal health when mode == "live", so a
    # broken journal stays invisible until session start. Surface it in preflight
    # so operators see the failure before committing to a session launch.
    try:
        from pipeline.paths import LIVE_JOURNAL_DB_PATH
        from trading_app.live.trade_journal import TradeJournal

        journal = TradeJournal(LIVE_JOURNAL_DB_PATH, mode="preflight")
        if journal.is_healthy:
            return CheckResult(True, f"OK ({LIVE_JOURNAL_DB_PATH.name})")
        return CheckResult(False, f"FAILED: TradeJournal could not open {LIVE_JOURNAL_DB_PATH}")
    except Exception as e:
        return CheckResult(False, f"FAILED: {e}")


def _check_copy_trading_accounts(ctx: PreflightContext) -> CheckResult:
    """Copy-trading account resolution (dry run).

    Closes A.6.5 preflight gap (see fix-account-id-sentinel-mismatch.md).
    Mirrors the live-start branch at run_live_session.py:672-693 but never
    constructs a router or places an order — broker account list is fetched
    read-only and the selection helper is invoked dry-run.
    """
    if ctx.profile_id is None:
        return CheckResult(True, "SKIPPED (no profile — raw-baseline path)")
    try:
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        prof = ACCOUNT_PROFILES.get(ctx.profile_id)
        if prof is None:
            return CheckResult(False, f"FAILED: profile {ctx.profile_id!r} not in ACCOUNT_PROFILES")
        if prof.copies <= 1:
            return CheckResult(True, f"SKIPPED (profile.copies={prof.copies})")
        if ctx.signal_only:
            return CheckResult(True, "SKIPPED (signal-only — no account resolution needed)")
        if ctx.components is None:
            # auth_check already failed; we cannot verify copy-trading without
            # broker contracts. Report FAILED (not SKIPPED) so the operator sees
            # the unverified state for a profile that requires copy trading.
            return CheckResult(False, "FAILED: auth failed (cannot verify copy-trading)")
        contracts_cls = ctx.components["contracts_class"]
        contracts = contracts_cls(auth=ctx.components["auth"], demo=ctx.demo)
        all_accounts = contracts.resolve_all_account_ids()
        # Dry-run the exact selection helper that runs at live-start (line 688).
        _select_primary_and_shadow_accounts(
            all_accounts=all_accounts,
            n_copies=prof.copies,
            requested_account_id=ctx.requested_account_id,
        )
        n = len(all_accounts)
        return CheckResult(True, f"OK (copies={prof.copies}, {n} accounts discovered)")
    except Exception as e:
        return CheckResult(False, f"FAILED: {e}")


# Ordered list of checks. State coupling: _check_auth populates
# ctx.components (consumed by _check_contracts, _check_notifications, and
# _check_copy_trading_accounts); _check_notifications sets
# ctx.components_all_pass (read by the summary branch in _run_preflight).
# Reordering breaks the contract — see stage doc
# preflight-checks-total-hardcode.md § risk register.
PREFLIGHT_CHECKS: list[CheckFn] = [
    _check_auth,
    _check_portfolio,
    _check_daily_features,
    _check_contracts,
    _check_notifications,
    _check_trade_journal,
    _check_copy_trading_accounts,
]


def _should_launch_dashboard() -> bool:
    """Return False when this session was launched by the dashboard itself."""
    return os.environ.get("CANOMPX3_DASHBOARD_ORIGIN") != "1"


def _run_preflight(
    instrument: str,
    broker: str | None,
    demo: bool,
    portfolio=None,
    profile_id: str | None = None,
    requested_account_id: int | None = None,
    signal_only: bool = False,
) -> bool:
    """Pre-flight validation. Returns True if all checks pass.

    Counts derive from len(PREFLIGHT_CHECKS) — adding or removing a check
    auto-updates the [i/N] header and the final summary. Closes
    debt-ledger entry `preflight-checks-total-hardcode`.

    `profile_id` and `requested_account_id` (added 2026-05-16, A.6.5) feed the
    copy-trading dry-run check. Both default to None so non-profile callers
    (raw-baseline) and existing fixtures continue to work unchanged.
    """
    from trading_app.live.broker_factory import get_broker_name

    ctx = PreflightContext(
        instrument=instrument,
        broker_name=broker or get_broker_name(),
        demo=demo,
        portfolio=portfolio,
        profile_id=profile_id,
        requested_account_id=requested_account_id,
        signal_only=signal_only,
    )

    checks_total = len(PREFLIGHT_CHECKS)
    checks_passed = 0

    # First check is auth — header includes broker name. Other checks use
    # the function's own __doc__ (first-line title). The original output
    # parenthesised the instrument on the portfolio line; keep that form.
    titles = {
        _check_auth: f"Auth check ({ctx.broker_name})",
        _check_portfolio: f"Portfolio check ({ctx.instrument})",
    }

    print()  # leading blank line, mirroring the original "\n[1/N]..." header
    for i, check in enumerate(PREFLIGHT_CHECKS, 1):
        title = titles.get(check) or (check.__doc__ or check.__name__).strip().splitlines()[0]
        print(f"[{i}/{checks_total}] {title}...", end=" ", flush=True)
        result = check(ctx)
        print(result.message)
        if result.passed:
            checks_passed += 1

    print(f"\nPreflight: {checks_passed}/{checks_total} passed")
    if checks_passed == checks_total:
        if not ctx.components_all_pass:
            print("All checks passed, but component warnings present. Review above.\n")
        else:
            print("All clear — ready to trade.\n")
    else:
        print("FIX FAILURES before starting a live session.\n")
    return checks_passed == checks_total


def _select_primary_and_shadow_accounts(
    *,
    all_accounts: list[tuple[int, str]],
    n_copies: int,
    requested_account_id: int | None,
) -> tuple[int, list[int] | None]:
    """Pick the primary account and shadow set for copy trading.

    Bug-fix 2026-04-25: previously this sliced all_accounts[:n_copies] FIRST
    then checked `requested_account_id in account_ids`. If the user-specified
    account was past the slice horizon (e.g. profile.copies=2 and the user
    wants the 3rd-listed XFA), the check silently failed and the code routed
    to all_accounts[0] — the WRONG account. Now: validate the account exists,
    then move it to the front so the slice always includes it. Hard-fail if
    the user's choice doesn't exist at the broker.

    Args:
        all_accounts: list of (account_id, account_name) tuples from the broker.
        n_copies: total accounts to use (primary + shadows).
        requested_account_id: if set, this MUST be one of all_accounts and will
            be the primary; raises RuntimeError otherwise.

    Returns:
        (primary_id, shadow_account_ids) where shadow_account_ids is a list of
        zero-or-more shadow account IDs (None if no shadows).
    """
    if requested_account_id == 0:
        requested_account_id = None
    all_account_ids = [aid for aid, _name in all_accounts]
    if requested_account_id is not None:
        if requested_account_id not in all_account_ids:
            raise RuntimeError(
                f"--account-id {requested_account_id} is not in the broker's discovered "
                f"accounts {all_account_ids}. Verify the account ID is correct and "
                f"the account is active and visible at the broker."
            )
        # Move user's account to the front so it's always inside the n_copies slice.
        all_account_ids.remove(requested_account_id)
        all_account_ids.insert(0, requested_account_id)

    account_ids = all_account_ids[:n_copies]
    if requested_account_id is not None:
        # Already at index 0 by construction above.
        account_ids.remove(requested_account_id)
        primary_id = requested_account_id
    else:
        primary_id = account_ids[0]
        account_ids = account_ids[1:]

    shadow_account_ids = account_ids if account_ids else None
    return primary_id, shadow_account_ids


def _print_mode_banner(mode: str, instrument: str) -> None:
    lines = {
        "signal": [
            "╔══════════════════════════════════════════╗",
            "║   MODE: SIGNAL ONLY — no orders placed   ║",
            "║   Watch for ⚡ SIGNAL lines in the log   ║",
            "║   Trade manually on your platform        ║",
            "╚══════════════════════════════════════════╝",
        ],
        "demo": [
            "╔══════════════════════════════════════════╗",
            "║   MODE: DEMO — paper orders only         ║",
            "║   Connected to broker DEMO account        ║",
            "║   No real money at risk                  ║",
            "╚══════════════════════════════════════════╝",
        ],
        "live": [
            "╔══════════════════════════════════════════╗",
            "║  ⚠  MODE: LIVE — REAL MONEY ORDERS  ⚠   ║",
            "║   Connected to your LIVE account         ║",
            "║   Orders execute with real capital       ║",
            "╚══════════════════════════════════════════╝",
        ],
    }
    print()
    for line in lines[mode]:
        log.info(line)
    log.info("   Instrument: %s", instrument)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ORB live session",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Instrument selection: --instrument XYZ or --all
    inst_group = parser.add_mutually_exclusive_group(required=False)
    inst_group.add_argument("--instrument", help="e.g. MGC, MNQ, MES, M2K")
    inst_group.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Run all active instruments (MGC, MNQ, MES, M2K) concurrently",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--signal-only",
        action="store_true",
        default=False,
        help="Show signals only — no orders placed. Trade manually. (SAFEST)",
    )
    mode_group.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Auto-place orders on broker DEMO account (paper trading)",
    )
    mode_group.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Auto-place REAL MONEY orders — requires typing CONFIRM",
    )

    parser.add_argument(
        "--account-id",
        type=int,
        default=None,
        help="Numeric account ID (default: auto-discover from API)",
    )
    parser.add_argument(
        "--broker",
        default=None,
        help="Broker: 'projectx' or 'tradovate' (default: from BROKER env var or 'projectx')",
    )
    parser.add_argument(
        "--force-orphans",
        action="store_true",
        default=False,
        help="Continue even if orphaned broker positions are detected on startup",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        default=False,
        help="Run pre-flight checks (auth, portfolio, daily_features, contract) then exit — no trading",
    )
    parser.add_argument(
        "--auto-confirm",
        action="store_true",
        default=False,
        help="Skip interactive CONFIRM prompt for --live. Safety gate is the dashboard UI "
        "(must type LIVE to unlock). CLI users: you accept full responsibility.",
    )
    parser.add_argument(
        "--raw-baseline",
        action="store_true",
        default=False,
        help="Use raw baseline portfolio from orb_outcomes (no validated_setups needed)",
    )
    parser.add_argument("--rr-target", type=float, default=1.0, help="RR target for raw baseline (default 1.0)")
    parser.add_argument(
        "--exclude-sessions",
        type=str,
        default="NYSE_CLOSE",
        help="Comma-separated sessions to exclude from raw baseline (default NYSE_CLOSE)",
    )
    parser.add_argument(
        "--stop-multiplier",
        type=float,
        default=1.0,
        help="Stop multiplier (1.0=standard, 0.75=tight prop stop)",
    )
    parser.add_argument("--entry-model", type=str, default="E2", help="Entry model for raw baseline (default E2)")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Load portfolio from prop_profiles.py account profile (e.g. 'apex_50k_manual'). "
        "Uses exact validated strategies from the profile's daily_lanes.",
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=0,
        help="Copy trades to N accounts (0=auto from profile.copies, 1=single account). "
        "Discovers all account IDs from broker API. Primary gets full tracking, "
        "shadows get best-effort order replication.",
    )
    args = parser.parse_args()

    # Build custom portfolio if requested (shared by preflight and session)
    raw_portfolio = None
    if args.profile:
        if args.raw_baseline:
            print("--profile and --raw-baseline are mutually exclusive.")
            sys.exit(1)
        from trading_app.prop_profiles import ACCOUNT_PROFILES, effective_daily_lanes

        profile = ACCOUNT_PROFILES[args.profile]
        profile_instruments = sorted({lane.instrument for lane in effective_daily_lanes(profile)})

        if len(profile_instruments) > 1:
            # Multi-instrument profile → route to MultiInstrumentRunner
            # (build_profile_portfolio can't handle mixed instruments in one call)
            args._multi_instrument_profile = True
            args._profile_instruments = profile_instruments
            log.info(
                "Profile '%s' has %d instruments: %s → will use MultiInstrumentRunner",
                args.profile,
                len(profile_instruments),
                profile_instruments,
            )
        else:
            args._multi_instrument_profile = False
            from trading_app.portfolio import build_profile_portfolio

            raw_portfolio = build_profile_portfolio(profile_id=args.profile)
            if not args.all and args.instrument is None:
                args.instrument = raw_portfolio.instrument
            elif args.instrument and args.instrument != raw_portfolio.instrument:
                print(
                    f"WARNING: --instrument {args.instrument} conflicts with profile instrument "
                    f"{raw_portfolio.instrument}. Using profile instrument."
                )
                args.instrument = raw_portfolio.instrument
            log.info(
                "Profile '%s': %d strategies loaded for %s",
                args.profile,
                len(raw_portfolio.strategies),
                raw_portfolio.instrument,
            )
    elif args.raw_baseline:
        if args.all:
            print("--raw-baseline + --all not supported. Use --instrument X.")
            sys.exit(1)
        from trading_app.portfolio import build_raw_baseline_portfolio

        exclude = {s.strip() for s in args.exclude_sessions.split(",") if s.strip()}
        raw_portfolio = build_raw_baseline_portfolio(
            instrument=args.instrument,
            rr_target=args.rr_target,
            entry_model=args.entry_model,
            exclude_sessions=exclude,
            stop_multiplier=args.stop_multiplier,
        )
        log.info("Raw baseline: %d strategies loaded", len(raw_portfolio.strategies))

    # Validate instrument is set (required unless --profile inferred it or multi-instrument)
    _is_multi_profile = getattr(args, "_multi_instrument_profile", False)
    if not args.instrument and not args.all and not _is_multi_profile:
        print("ERROR: --instrument or --all is required (unless --profile is used).")
        sys.exit(1)

    if args.preflight:
        if args.all:
            print("Preflight with --all not supported. Use --instrument X.")
            sys.exit(1)
        demo = not args.live
        if _is_multi_profile:
            # Multi-instrument profile: run preflight per instrument
            from trading_app.portfolio import build_profile_portfolio

            all_ok = True
            for inst in args._profile_instruments:
                print(f"\n{'=' * 50}")
                print(f"Preflight: {inst}")
                print(f"{'=' * 50}")
                inst_portfolio = build_profile_portfolio(profile_id=args.profile, instrument=inst)
                ok = _run_preflight(
                    inst,
                    args.broker,
                    demo,
                    portfolio=inst_portfolio,
                    profile_id=args.profile,
                    requested_account_id=args.account_id,
                    signal_only=args.signal_only,
                )
                if not ok:
                    all_ok = False
            sys.exit(0 if all_ok else 1)
        else:
            ok = _run_preflight(
                args.instrument,
                args.broker,
                demo,
                portfolio=raw_portfolio,
                profile_id=args.profile,
                requested_account_id=args.account_id,
                signal_only=args.signal_only,
            )
            sys.exit(0 if ok else 1)

    # Default to signal-only if no mode specified (safest default)
    if not args.signal_only and not args.demo and not args.live:
        log.info("No mode specified — defaulting to --signal-only (safest)")
        args.signal_only = True

    # Determine execution mode
    if args.signal_only:
        signal_only = True
        demo = True  # still needs auth for bar feed, but no orders placed
    elif args.demo:
        if args.all:
            print("--all + --demo not supported. Use --instrument X for demo trading.")
            sys.exit(1)
        signal_only = False
        demo = True
    else:  # --live
        if args.all:
            print("--all + --live not supported. Use --instrument X for live trading.")
            sys.exit(1)
        if not args.auto_confirm:
            confirm = input("\n⚠  LIVE MODE — real money orders will be placed.\n   Type CONFIRM to proceed: ").strip()
            if confirm != "CONFIRM":
                print("Aborted.")
                sys.exit(0)
        else:
            log.warning("LIVE MODE — auto-confirmed (dashboard launch)")
        signal_only = False
        demo = False

    # Stop-file path — cleaned up after session ends (feeds no longer delete it)
    _stop_file = Path(__file__).parent.parent / "live_session.stop"

    # Multi-instrument mode: --all OR multi-instrument --profile
    if args.all or _is_multi_profile:
        from trading_app.live.multi_runner import MultiInstrumentRunner

        if _is_multi_profile:
            _insts = args._profile_instruments
            for inst in _insts:
                acquire_instance_lock(inst)
            _names = ", ".join(_insts)
            _print_mode_banner("signal" if signal_only else "live", f"PROFILE {args.profile} ({_names})")
            runner = MultiInstrumentRunner(
                instruments=_insts,
                broker=args.broker,
                demo=demo,
                signal_only=signal_only,
                account_id=args.account_id,
                force_orphans=args.force_orphans,
                profile_id=args.profile,
            )
        else:
            from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

            for inst in ACTIVE_ORB_INSTRUMENTS:
                acquire_instance_lock(inst)
            _all_names = ", ".join(ACTIVE_ORB_INSTRUMENTS)
            _print_mode_banner("signal", f"ALL ({_all_names})")
            runner = MultiInstrumentRunner(
                instruments=None,  # uses ACTIVE_ORB_INSTRUMENTS
                broker=args.broker,
                demo=demo,
                signal_only=signal_only,
                account_id=args.account_id,
                force_orphans=args.force_orphans,
            )
        try:
            asyncio.run(runner.run())
        finally:
            runner.post_session()
            release_instance_lock()
        return

    # Single-instrument mode (existing path)
    acquire_instance_lock(args.instrument)
    _print_mode_banner("signal" if signal_only else ("demo" if demo else "live"), args.instrument)

    # Launch dashboard as background subprocess for CLI starts only. Dashboard
    # button starts set CANOMPX3_DASHBOARD_ORIGIN=1 so the bot does not open a
    # duplicate browser/server behind itself.
    _dashboard_proc = None
    if _should_launch_dashboard():
        try:
            from trading_app.live.bot_dashboard import launch_dashboard_background

            _dashboard_proc = launch_dashboard_background()
        except Exception as e:
            log.warning("Dashboard launch failed (non-fatal): %s", e)

    # Multi-account copy trading: discover shadow accounts
    shadow_account_ids = None
    n_copies = args.copies
    if n_copies == 0 and args.profile:
        # Auto from profile.copies (0 = use profile value)
        from trading_app.prop_profiles import get_profile

        prof = get_profile(args.profile)
        n_copies = prof.copies

    if n_copies > 1 and not signal_only:
        from trading_app.live.broker_factory import create_broker_components, get_broker_name

        broker_name = args.broker or get_broker_name()
        components = create_broker_components(broker_name, demo=demo)
        contracts = components["contracts_class"](auth=components["auth"], demo=demo)
        all_accounts = contracts.resolve_all_account_ids()
        log.info("Account discovery: %d accounts found, copies=%d", len(all_accounts), n_copies)

        if len(all_accounts) < n_copies:
            log.warning(
                "Requested %d copies but only %d accounts found — using all available",
                n_copies,
                len(all_accounts),
            )

        primary_id, shadow_account_ids = _select_primary_and_shadow_accounts(
            all_accounts=all_accounts,
            n_copies=n_copies,
            requested_account_id=args.account_id,
        )
        args.account_id = primary_id
        log.info(
            "Copy trading: primary=%d, shadows=%s",
            primary_id,
            shadow_account_ids or "none",
        )

    session = SessionOrchestrator(
        instrument=args.instrument,
        broker=args.broker,
        demo=demo,
        account_id=args.account_id,
        signal_only=signal_only,
        force_orphans=args.force_orphans,
        portfolio=raw_portfolio,
        shadow_account_ids=shadow_account_ids,
    )

    try:
        asyncio.run(session.run())
    finally:
        session.post_session()
        release_instance_lock()
        _stop_file.unlink(missing_ok=True)
        # Clear bot state so dashboard shows STOPPED
        try:
            from trading_app.live.bot_state import clear_state

            clear_state()
        except Exception:
            pass
        # Terminate dashboard subprocess (prevent orphan on Windows)
        if _dashboard_proc is not None:
            try:
                _dashboard_proc.terminate()
                _dashboard_proc.wait(timeout=5)
                log.info("Dashboard subprocess terminated")
            except Exception:
                try:
                    _dashboard_proc.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
