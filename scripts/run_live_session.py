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

# `subprocess` is re-exported (not used directly here): the preflight tests patch
# `rls.subprocess.run`, and the moved gates in preflight.py observe it because
# `rls.subprocess` and `preflight.subprocess` are the SAME shared module object.
import argparse
import asyncio
import io
import logging
import os
import subprocess  # noqa: F401 — test-seam re-export (see note above)
import sys
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode the CONFIRM prompt's
# warning sign / em-dash and crashes input() before the operator can confirm.
# Canonical guard mirrors research/allocator_scarcity_surface_audit.py.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tools import sweep_orphan_rings
from trading_app.live.instance_lock import acquire_instance_lock, release_instance_lock

# Preflight engine — single source of truth (App Overhaul Stage 1).
# The 15-gate engine moved to trading_app/live/preflight.py so every surface
# (this CLI, the dashboard) reads ONE in-process verdict. Re-imported back into
# this namespace so the CLI run-and-print wrapper (_run_preflight below) and the
# test suite's `monkeypatch.setattr(rls, "PREFLIGHT_CHECKS", ...)` / `rls.CheckResult`
# / `rls._check_*` / `rls._pid_is_alive` / `rls._select_primary_and_shadow_accounts`
# swaps keep resolving against this module exactly as before the extraction.
# main() also calls _select_primary_and_shadow_accounts at live-start (the
# canonical account-selection helper); _probe_brackets / _probe_fill_poller are
# re-imported so the source-grep test pinning them present here still passes and
# the run_self_tests monkeypatch surface is preserved.
from trading_app.live.preflight import (  # noqa: F401 — re-exported for namespace parity + tests
    PREFLIGHT_CHECKS,
    CheckFn,
    CheckResult,
    GateResult,
    PreflightContext,
    PreflightReport,
    _check_account_binding,
    _check_auth,
    _check_contracts,
    _check_copy_trading_accounts,
    _check_daily_features,
    _check_live_readiness_report,
    _check_notifications,
    _check_portfolio,
    _check_project_pulse_for_live,
    _check_repo_drift_for_live,
    _check_shadow_copy_loss_protection,
    _check_sr_state,
    _check_survival_report,
    _check_telemetry_maturity,
    _check_trade_journal,
    _passing_check_is_advisory,
    _pid_is_alive,
    _probe_brackets,
    _probe_fill_poller,
    _run_lightweight_component_self_tests,
    _select_primary_and_shadow_accounts,
    run_preflight,
)
from trading_app.live.session_orchestrator import SessionOrchestrator


def _silence_pysignalr_negotiation_log() -> None:
    """Drop pysignalr's full-negotiation-URL INFO log.

    pysignalr.transport (`websocket.py:320`) logs the SignalR negotiation
    URL at INFO via `_logger.info('Performing negotiation, URL: ...')`.
    The URL carries the `?access_token=<JWT>` query parameter; the JWT
    decodes to user identity + role + expiry. Local-only blast (logs land
    in $TEMP or logs/live/) but credential-leak class — any log share or
    screenshot would leak the JWT.

    INFO -> WARNING removes the negotiation-URL line plus the "State
    change" INFO trio (line 225) and the "Sending/Awaiting/Completed
    handshake" lines. The orchestrator's own INFO ("Connected to ProjectX
    Market Hub", "Subscribed to quotes: CON.F.US.MNQ.M26") still surfaces
    feed connectivity. Reconnect failures still log at WARNING via
    `_logger.warning('Connection closed: ...')` (websocket.py:211).
    """
    logging.getLogger("pysignalr.transport").setLevel(logging.WARNING)


def _sweep_startup_orphan_rings() -> None:
    try:
        deleted = sweep_orphan_rings.sweep(sweep_orphan_rings.DEFAULT_RING_DIR)
    except Exception:
        log.warning("Startup ring sweep failed; continuing without deleting orphan rings", exc_info=True)
        return
    if deleted:
        log.info("Startup ring sweep removed %d orphan ring file(s)", len(deleted))


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
    requested_copies: int = 0,
    strict_zero_warn: bool = False,
) -> bool:
    """Pre-flight validation. Returns True if all checks pass.

    Counts derive from len(PREFLIGHT_CHECKS) — adding or removing a check
    auto-updates the [i/N] header and the final summary. Closes
    debt-ledger entry `preflight-checks-total-hardcode`.

    `profile_id` and `requested_account_id` (added 2026-05-16, A.6.5) feed the
    copy-trading dry-run check. Both default to None so non-profile callers
    (raw-baseline) and existing fixtures continue to work unchanged.

    `strict_zero_warn` (added 2026-06-08): for the `--preflight` exit path used
    by real-money launchers (START_BOT.bat live), treat a passing-but-advisory
    check (WARN/SKIPPED, classified via the canonical `_normalize_check_status`)
    as blocking — returning False so the launcher refuses to arm. This ports the
    dashboard arm-guard's mode=live "warn also blocks" rule to the direct-launch
    path. It applies ONLY to real-money runs: demo and signal-only place no live
    orders, so advisory checks stay non-blocking there (no false block).
    """
    # Delegate to the canonical in-process engine (trading_app/live/preflight.py)
    # for the verdict, then render it to stdout in the exact [i/N] header format
    # the dashboard's subprocess-fallback regex and the source-grep tests expect.
    # `PREFLIGHT_CHECKS` is read from THIS module's namespace (re-imported above)
    # so the test suite's `monkeypatch.setattr(rls, "PREFLIGHT_CHECKS", ...)`
    # injection still drives both the count and the gates that run.
    report = run_preflight(
        instrument,
        broker,
        demo,
        portfolio=portfolio,
        profile_id=profile_id,
        requested_account_id=requested_account_id,
        signal_only=signal_only,
        requested_copies=requested_copies,
        strict_zero_warn=strict_zero_warn,
        checks=PREFLIGHT_CHECKS,
    )

    # checks_total derives dynamically from the gate list (closes debt-ledger
    # entry `preflight-checks-total-hardcode`); it equals report.checks_total by
    # construction since run_preflight() was passed this same PREFLIGHT_CHECKS.
    checks_total = len(PREFLIGHT_CHECKS)
    print()  # leading blank line, mirroring the original "\n[1/N]..." header
    for i, gate in enumerate(report.gates, 1):
        print(f"[{i}/{checks_total}] {gate.title}...", end=" ", flush=True)
        print(gate.message)

    print(f"\nPreflight: {report.checks_passed}/{checks_total} passed")
    if report.checks_warned:
        print(f"Preflight warnings: {report.checks_warned}")
    if report.all_passed:
        if report.strict_block:
            print("STRICT-ZERO-WARN: blocked — passing preflight still has WARN/SKIPPED checks.\n")
        elif not report.components_all_pass:
            print("All checks passed, but component warnings present. Review above.\n")
        else:
            print("All clear — ready to trade.\n")
    else:
        print("FIX FAILURES before starting a live session.\n")
    return report.launch_ok


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
    _silence_pysignalr_negotiation_log()
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
        "--strict-zero-warn",
        action="store_true",
        default=False,
        help="With --preflight on a real-money (--live) run, exit non-zero if any passing check "
        "is advisory (WARN/SKIPPED). Ports the dashboard's mode=live warn-blocking arm guard to "
        "the direct-launch path (START_BOT.bat live). No effect on demo/signal-only preflight.",
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
                    requested_copies=args.copies,
                    strict_zero_warn=args.strict_zero_warn,
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
                requested_copies=args.copies,
                strict_zero_warn=args.strict_zero_warn,
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

    # MANDATORY pre-launch capital gate (hard, no skip flag).
    #
    # Every order-routing launch (--demo / --live, CLI *or* dashboard) MUST pass
    # the canonical preflight chain before any side effect. Previously this chain
    # ran ONLY under `if args.preflight:` (an exit-after-report path), so a direct
    # `--live --auto-confirm` — the exact form the dashboard launches
    # (bot_dashboard.py builds `--live --auto-confirm` with NO --preflight; its
    # own preflight subprocess is an advisory pre-click UI check, not a gate on
    # the launch process) — reached SessionOrchestrator with no C11 survival,
    # C12 SR, live-readiness, project-pulse, repo-drift, or telemetry gate. The
    # orchestrator's startup lifecycle read is fail-open and never refuses launch.
    #
    # signal-only is exempt: it places no orders and the _check_* functions
    # self-SKIP on signal_only (they accumulate evidence). Delegates to the same
    # canonical _run_preflight used by the --preflight block — no re-encoding.
    if not signal_only:
        if _is_multi_profile:
            from trading_app.portfolio import build_profile_portfolio

            _gate_ok = True
            for _inst in args._profile_instruments:
                _inst_portfolio = build_profile_portfolio(profile_id=args.profile, instrument=_inst)
                if not _run_preflight(
                    _inst,
                    args.broker,
                    demo,
                    portfolio=_inst_portfolio,
                    profile_id=args.profile,
                    requested_account_id=args.account_id,
                    signal_only=signal_only,
                    requested_copies=args.copies,
                ):
                    _gate_ok = False
            if not _gate_ok:
                print(
                    "ERROR: pre-launch preflight FAILED — order-routing launch blocked. "
                    "Fix the failing check(s) above, then relaunch. (No skip flag by design.)"
                )
                sys.exit(1)
        else:
            if not _run_preflight(
                args.instrument,
                args.broker,
                demo,
                portfolio=raw_portfolio,
                profile_id=args.profile,
                requested_account_id=args.account_id,
                signal_only=signal_only,
                requested_copies=args.copies,
            ):
                print(
                    "ERROR: pre-launch preflight FAILED — order-routing launch blocked. "
                    "Fix the failing check(s) above, then relaunch. (No skip flag by design.)"
                )
                sys.exit(1)

    # Publish the canonical planned-launch surface BEFORE orchestrator init so
    # the dashboard can render the unambiguous "Next launch: …" banner. The
    # orchestrator's own state (bot_state.json) will supersede this once
    # running. Only write when --profile resolves the launch (CLI ad-hoc runs
    # without a profile cannot be schema-verified against ACCOUNT_PROFILES).
    if args.profile:
        try:
            from trading_app.live.planned_launch import write_planned_launch

            _mode_label = "SIGNAL" if signal_only else ("DEMO" if demo else "LIVE")
            write_planned_launch(
                profile_id=args.profile,
                mode=_mode_label,
                source="CLI",
                copies=args.copies or None,
                instruments=[args.instrument] if args.instrument else None,
                broker_accounts_count=args.copies or None,
            )
        except Exception as _e:  # noqa: BLE001 — never block the launch on a UI surface
            log.warning("planned_launch write failed: %s", _e)

    # Stop-file path — cleaned up after session ends (feeds no longer delete it)
    _stop_file = Path(__file__).parent.parent / "live_session.stop"

    _sweep_startup_orphan_rings()

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
            # Clear the planned-launch artifact so the dashboard does not
            # display a stale "Next launch: …" banner after the session ends.
            # Operator must re-run START_BOT.bat / CLI to re-publish intent.
            try:
                from trading_app.live.planned_launch import clear_planned_launch

                clear_planned_launch()
            except Exception:
                pass
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

    if not demo and not signal_only and n_copies > 1:
        print(
            "ERROR: live multi-copy launch blocked by SHADOW-MLL. "
            "Use --copies 1 for the first live pilot or implement per-shadow loss belts."
        )
        sys.exit(1)

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

    # Clean-teardown signal handlers (F1). Route OS stop signals into the same
    # `finally` teardown below so the dashboard child + instance lock + journal
    # are cleaned up even when the operator does NOT use Ctrl+C. Without this,
    # SIGTERM (`taskkill`) and Windows Ctrl+Break skip the finally → orphaned
    # dashboard subprocess holding port 8080 ("stray command" on next launch).
    #
    # Grounded in CPython docs: asyncio's loop.add_signal_handler is Unix-only
    # (Proactor loop raises NotImplementedError on Windows), so we use the
    # synchronous signal.signal handler, which is portable. The handler does the
    # MINIMAL safe thing — raise KeyboardInterrupt — which propagates out of
    # asyncio.run() and into the `finally`, reusing the proven teardown path.
    #
    # IMPORTANT — this is teardown, NOT flatten. ProjectX brackets are
    # server-side (docs/reference/PROJECTX_API_REFERENCE.md:171-176): an open
    # position keeps its stop/target at the broker after this process exits. We
    # deliberately do NOT close positions on a stop signal; we only release
    # our-side state. The Stage-3 restart guard is the backstop against a
    # hard-kill (taskkill /F) that no in-process handler can intercept.
    def _request_teardown(signum, _frame):  # pragma: no cover - exercised via signal
        log.warning("Received signal %s — initiating clean teardown (position left on broker bracket)", signum)
        raise KeyboardInterrupt

    import signal as _signal

    _teardown_signals = [_signal.SIGINT, _signal.SIGTERM]
    if hasattr(_signal, "SIGBREAK"):  # Windows Ctrl+Break
        _teardown_signals.append(_signal.SIGBREAK)
    for _sig in _teardown_signals:
        try:
            _signal.signal(_sig, _request_teardown)
        except (ValueError, OSError) as e:
            # ValueError: not main thread; OSError: signal unsupported on platform.
            log.warning("Could not register handler for signal %s (non-fatal): %s", _sig, e)

    try:
        asyncio.run(session.run())
    except KeyboardInterrupt:
        # Clean teardown path. The signal handlers above (_request_teardown)
        # turn SIGINT (Ctrl+C) / SIGTERM (dashboard STOP -> proc.terminate())
        # into KeyboardInterrupt; it propagates out of asyncio.run() here. We
        # swallow it and log a single clean line instead of letting Python
        # print a multi-frame traceback to the console — a stop is a normal,
        # operator-initiated event, not a crash. The `finally:` below still
        # runs (instance lock release, stop-file unlink, bot_state clear,
        # dashboard-child terminate), so teardown is unchanged. NOTE: this is
        # teardown only — open positions stay on their server-side broker
        # bracket (see _request_teardown). Fail-closed: even if this except
        # were somehow skipped, the finally still cleans up.
        log.info("Session stopped (clean teardown — open positions remain on broker bracket)")
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
        # Clear the planned-launch artifact too — same reason. Without this, the
        # banner shows "Next launch: <mode>" indefinitely after a session ends
        # which could mislead the operator about what the next click will do.
        try:
            from trading_app.live.planned_launch import clear_planned_launch

            clear_planned_launch()
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
            # Symmetric cleanup: clear the dashboard PID file so the next launch
            # has nothing stale to reap. (On an unclean [X]-close this finally is
            # skipped and the file persists — that is exactly when the next
            # launch's _reap_orphan_dashboard needs it, so this clear is correct
            # only on the clean path.)
            try:
                from trading_app.live.bot_dashboard import _DASHBOARD_PID_FILE

                _DASHBOARD_PID_FILE.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
