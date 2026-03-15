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

from trading_app.live.session_orchestrator import SessionOrchestrator


def _run_preflight(instrument: str, broker: str | None, demo: bool) -> bool:
    """Pre-flight validation. Returns True if all checks pass."""
    from datetime import date

    from pipeline.paths import GOLD_DB_PATH
    from trading_app.live.broker_factory import create_broker_components, get_broker_name
    from trading_app.live_config import build_live_portfolio

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
        portfolio, notes = build_live_portfolio(db_path=GOLD_DB_PATH, instrument=instrument)
        print(f"OK ({len(portfolio.strategies)} strategies)")
        for s in portfolio.strategies:
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
    print(f"[3/{checks_total}] Daily features freshness...", end=" ", flush=True)
    try:
        row = SessionOrchestrator._build_daily_features_row(date.today(), instrument)
        atr = row.get("atr_20")
        vel = row.get("atr_vel_regime")
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
    try:
        orch = SessionOrchestrator(
            instrument=instrument,
            broker=broker_name,
            demo=demo,
            signal_only=True,  # safe: no orders, just test components
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
    inst_group = parser.add_mutually_exclusive_group(required=True)
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
    args = parser.parse_args()

    if args.preflight:
        if args.all:
            print("Preflight with --all not supported. Use --instrument X.")
            sys.exit(1)
        demo = not args.live
        ok = _run_preflight(args.instrument, args.broker, demo)
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
        confirm = input("\n⚠  LIVE MODE — real money orders will be placed.\n   Type CONFIRM to proceed: ").strip()
        if confirm != "CONFIRM":
            print("Aborted.")
            sys.exit(0)
        signal_only = False
        demo = False

    # Stop-file path — cleaned up after session ends (feeds no longer delete it)
    _stop_file = Path(__file__).parent.parent / "live_session.stop"

    # Multi-instrument mode (signal-only only — demo/live blocked above)
    if args.all:
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
        from trading_app.live.multi_runner import MultiInstrumentRunner

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
        return

    # Single-instrument mode (existing path)
    _print_mode_banner("signal" if signal_only else ("demo" if demo else "live"), args.instrument)

    session = SessionOrchestrator(
        instrument=args.instrument,
        broker=args.broker,
        demo=demo,
        account_id=args.account_id,
        signal_only=signal_only,
        force_orphans=args.force_orphans,
    )

    try:
        asyncio.run(session.run())
    finally:
        session.post_session()
        _stop_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
