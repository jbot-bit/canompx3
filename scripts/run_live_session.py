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
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

from trading_app.live.instance_lock import acquire_instance_lock, release_instance_lock
from trading_app.live.session_orchestrator import SessionOrchestrator


def _run_preflight(instrument: str, broker: str | None, demo: bool, portfolio=None) -> bool:
    """Pre-flight validation. Returns True if all checks pass."""

    from trading_app.live.broker_factory import create_broker_components, get_broker_name

    checks_passed = 0
    checks_total = 5  # NOTE: must match number of check blocks (1-5) below — update if adding/removing checks

    # 1. Auth check
    broker_name = broker or get_broker_name()
    print(f"\n[1/{checks_total}] Auth check ({broker_name})...", end=" ", flush=True)
    try:
        components = create_broker_components(broker_name, demo=demo)
        token = components["auth"].get_token()
        print(f"OK (token: {token[:8]}...)")
        checks_passed += 1
    except Exception as e:
        print(f"FAILED: {e}")
        components = None

    # 2. Portfolio check
    print(f"[2/{checks_total}] Portfolio check ({instrument})...", end=" ", flush=True)
    try:
        if portfolio is not None:
            pf = portfolio
            notes = [f"Using injected portfolio ({len(pf.strategies)} strategies)"]
        else:
            raise RuntimeError(
                f"No portfolio injected for {instrument}. "
                "Pass --profile or --raw-baseline to build a portfolio from "
                "prop_profiles.ACCOUNT_PROFILES. build_live_portfolio() is DEPRECATED."
            )
        print(f"OK ({len(pf.strategies)} strategies)")
        for s in pf.strategies:
            print(
                f"    {s.strategy_id} | {s.orb_label} {s.entry_model} "
                f"RR{s.rr_target} O{s.orb_minutes} | WR={s.win_rate:.0%} "
                f"ExpR={s.expectancy_r:.3f} N={s.sample_size}"
            )
        for note in notes:
            print(f"    NOTE: {note}")
        checks_passed += 1
    except Exception as e:
        print(f"FAILED: {e}")

    # 3. Daily features freshness
    # R2-M5: use Brisbane trading day, not date.today() (Windows system time).
    # For late-night sessions (NYSE_OPEN at 00:30 Brisbane), date.today() gives
    # yesterday's date, validating stale data and producing a false-positive pass.
    print(f"[3/{checks_total}] Daily features freshness...", end=" ", flush=True)
    try:
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        bris_now = datetime.now(ZoneInfo("Australia/Brisbane"))
        if bris_now.hour < 9:
            preflight_trading_day = (bris_now - timedelta(days=1)).date()
        else:
            preflight_trading_day = bris_now.date()
        row = SessionOrchestrator._build_daily_features_row(preflight_trading_day, instrument)
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
            needs_atr = portfolio is not None and (
                (atr is None and any(s.filter_type.startswith(atr_prefixes) for s in portfolio.strategies))
                or (vel is None and any(s.filter_type.startswith(vel_prefixes) for s in portfolio.strategies))
            )
            if needs_atr:
                print(
                    f"FAILED: {', '.join(missing)} = None and portfolio has "
                    f"ATR-dependent filters — run pipeline/build_daily_features.py"
                )
            else:
                print(f"WARN: {', '.join(missing)} = None — run pipeline/build_daily_features.py")
                checks_passed += 1
        else:
            print(f"OK (atr_20={atr}, atr_vel={vel})")
            checks_passed += 1
    except Exception as e:
        print(f"FAILED: {e}")

    # 4. Contract resolution
    print(f"[4/{checks_total}] Contract resolution...", end=" ", flush=True)
    if components is not None:
        try:
            contracts_cls = components["contracts_class"]
            contracts = contracts_cls(auth=components["auth"], demo=demo)
            front = contracts.resolve_front_month(instrument)
            print(f"OK ({front})")
            checks_passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
    else:
        print("SKIPPED (auth failed)")

    # 5. Component self-tests (notifications, brackets, fill poller)
    all_pass = True  # default if check 5 fails entirely
    print(f"[5/{checks_total}] Component self-tests...", end=" ", flush=True)
    orch = None
    try:
        orch = SessionOrchestrator(
            instrument=instrument,
            broker=broker_name,
            demo=demo,
            signal_only=True,  # safe: no orders, just test components
            portfolio=portfolio,
        )
        test_results = orch.run_self_tests()
        all_pass = all(test_results.values())
        if all_pass:
            print("OK (all components verified)")
            checks_passed += 1
        else:
            failed = [k for k, v in test_results.items() if not v]
            print(f"WARNINGS: {', '.join(failed)}")
            # Don't fail preflight for component warnings — they're informational
            checks_passed += 1
    except Exception as e:
        print(f"FAILED: {e}")
    finally:
        # CRITICAL: close journal DB connection to release Windows file lock.
        # Without this, live_journal.db stays locked after preflight exits,
        # blocking the actual trading session from opening it.
        if orch is not None:
            try:
                orch.journal.close()
            except Exception:
                pass

    print(f"\nPreflight: {checks_passed}/{checks_total} passed")
    if checks_passed == checks_total:
        if not all_pass:
            print("All checks passed, but component warnings present. Review above.\n")
        else:
            print("All clear — ready to trade.\n")
    else:
        print("FIX FAILURES before starting a live session.\n")
    return checks_passed == checks_total


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
        default=0,
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
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        from trading_app.prop_profiles import effective_daily_lanes

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
                print(f"\n{'='*50}")
                print(f"Preflight: {inst}")
                print(f"{'='*50}")
                inst_portfolio = build_profile_portfolio(profile_id=args.profile, instrument=inst)
                ok = _run_preflight(inst, args.broker, demo, portfolio=inst_portfolio)
                if not ok:
                    all_ok = False
            sys.exit(0 if all_ok else 1)
        else:
            ok = _run_preflight(args.instrument, args.broker, demo, portfolio=raw_portfolio)
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

    # Launch dashboard as background subprocess (non-fatal if it fails)
    _dashboard_proc = None
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

        account_ids = [aid for aid, _name in all_accounts[:n_copies]]
        if args.account_id and args.account_id in account_ids:
            # User-specified account is primary
            account_ids.remove(args.account_id)
            primary_id = args.account_id
        else:
            primary_id = account_ids[0]
            account_ids = account_ids[1:]

        shadow_account_ids = account_ids if account_ids else None
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
