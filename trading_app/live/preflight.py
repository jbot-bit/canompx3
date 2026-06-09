"""Single source of preflight truth — the canonical live-launch gate engine.

App Overhaul Stage 1 (cohesion seam). This module houses the 15-gate preflight
engine that was previously inline in ``scripts/run_live_session.py``. The engine
is now ONE in-process source of truth, replacing the divergent copies that made
the app "feel disjointed":

  - ``scripts/run_live_session.py`` re-imports these names and keeps only the
    run-and-print CLI wrapper (``_run_preflight``) + ``main()``.

Stage 1b (DEFERRED — not yet wired): ``trading_app/live/bot_dashboard.py`` is
intended to call :func:`run_preflight` in-process via an adapter. **Today it
still shells out** to ``python -m scripts.run_live_session --preflight`` and
regex-parses stdout (``_run_preflight_subprocess`` / ``_parse_preflight_output``
in bot_dashboard.py). That subprocess path is the legacy seam this module's
Stage 1b will retire; until then the launcher's ``_run_preflight`` renders the
report to stdout in the exact ``[i/N] <title>... <message>`` format the dashboard
regex tolerates, so the two paths stay in agreement.

The engine is **behavior-preserving**: gate verdicts, ordering, fail-closed
semantics, and the ``strict_zero_warn`` rule are byte-for-byte what they were in
the launcher. Only the residence changed.

Two consumption shapes:

  - :func:`run_preflight` returns a structured :class:`PreflightReport`
    (per-gate :class:`GateResult` + overall + ``strict_block``). No printing,
    no ``sys.exit`` — pure computation. This is the new shared verdict.
  - The launcher's ``_run_preflight`` renders that report to stdout in the exact
    ``[i/N] <title>... <message>`` format the dashboard regex still tolerates
    (so the subprocess fallback and the source-grep tests stay green).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trading_app.live.session_orchestrator import SessionOrchestrator

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    components: Any = None  # BrokerComponents (TypedDict) | None — Any avoids import cycle
    components_all_pass: bool = True
    # Copy-trading dry-run inputs (A.6.5 — preflight gap fix 2026-05-16)
    profile_id: str | None = None
    requested_account_id: int | None = None
    signal_only: bool = False
    requested_copies: int = 0


@dataclass
class CheckResult:
    """Return shape for every preflight check."""

    passed: bool
    message: str  # printed inline after "[i/N] <name>... "


def _passing_check_is_advisory(message: str) -> bool:
    """True when a *passing* preflight check normalizes to a warning ("warn").

    Reuses the CANONICAL dashboard mapping (`_normalize_check_status`) rather
    than re-encoding the OK/WARN/SKIPPED contract. That keeps the START_BOT
    direct-live strict-zero-warn gate in exact lockstep with the dashboard's
    arm guard (bot_dashboard.action_start) and with the fail-closed drift check
    `check_preflight_status_token_parity`. The status token is the first word of
    the message (the same token `_parse_preflight_output` extracts), e.g.
    "SKIPPED (...)" -> "SKIPPED", "WARN: ..." -> "WARN".

    Import is local + fail-closed: if the dashboard mapping cannot be loaded, we
    treat the check as advisory (return True) so strict-zero-warn errs toward
    BLOCKING a real-money launch, never toward silently passing one.
    """
    token = message.strip().split(maxsplit=1)[0] if message.strip() else ""
    if not token:
        return False
    try:
        from trading_app.live.bot_dashboard import _normalize_check_status
    except Exception:  # noqa: BLE001 — fail-closed: unknown -> advisory -> blocks
        return True
    return _normalize_check_status(token) == "warn"


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


def _check_survival_report(ctx: PreflightContext) -> CheckResult:
    """Criterion 11 survival evidence"""
    if ctx.signal_only:
        return CheckResult(True, "SKIPPED (signal-only accumulates evidence; no capital at risk)")
    if ctx.profile_id is None:
        # FAIL-CLOSED: a --demo/--live session here routes orders
        # (signal_only is False), but C11 lifecycle state is profile-keyed —
        # with no profile there is no survival evidence to validate. Skipping
        # would let an order-routing session pass with zero evidence. Launch
        # with --profile so the capital gate has something to check. (Closes
        # baf99cfe H2 audit defect; sanctioned launches always pass --profile.)
        return CheckResult(
            False,
            "FAILED: order-routing session has no --profile; Criterion 11 "
            "survival evidence is profile-keyed and cannot be validated. "
            "Launch with --profile or use --signal-only.",
        )
    try:
        from trading_app.lifecycle_state import read_lifecycle_state

        lifecycle = read_lifecycle_state(profile_id=ctx.profile_id)
        criterion11 = lifecycle.get("criterion11", {})
        if criterion11.get("gate_ok"):
            op = criterion11.get("operational_pass_probability")
            op_text = f"{float(op):.1%}" if isinstance(op, int | float) else "unknown"
            return CheckResult(True, f"OK (operational pass {op_text})")
        msg = criterion11.get("gate_msg") or criterion11.get("reason") or "Criterion 11 survival state invalid"
        return CheckResult(False, f"FAILED: {msg}")
    except Exception as e:
        return CheckResult(False, f"FAILED: {e}")


def _check_sr_state(ctx: PreflightContext) -> CheckResult:
    """Criterion 12 SR state"""
    if ctx.signal_only:
        return CheckResult(True, "SKIPPED (signal-only accumulates SR stream; no capital at risk)")
    if ctx.profile_id is None:
        # FAIL-CLOSED: see _check_survival_report — an order-routing session
        # with no --profile has no profile-keyed C12 SR state to validate, so
        # the gate must block rather than silently pass. (baf99cfe H2 defect.)
        return CheckResult(
            False,
            "FAILED: order-routing session has no --profile; Criterion 12 SR "
            "state is profile-keyed and cannot be validated. Launch with "
            "--profile or use --signal-only.",
        )
    try:
        from trading_app.lifecycle_state import read_lifecycle_state

        lifecycle = read_lifecycle_state(profile_id=ctx.profile_id)
        criterion12 = lifecycle.get("criterion12", {})
        if not criterion12.get("available"):
            return CheckResult(False, "FAILED: Criterion 12 SR state missing; refresh control state")
        if not criterion12.get("valid"):
            reason = criterion12.get("reason") or "invalid SR state"
            return CheckResult(False, f"FAILED: Criterion 12 SR state {reason}; refresh control state")
        counts = criterion12.get("counts", {})
        alarm_count = counts.get("ALARM", 0) if isinstance(counts, dict) else 0
        no_data_count = counts.get("NO_DATA", 0) if isinstance(counts, dict) else 0
        return CheckResult(True, f"OK (alarm={alarm_count}, no_data={no_data_count})")
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
        # Order-routing launches (--demo / --live) FAIL-CLOSED on any probe
        # failure: a degraded bracket builder or fill-poller means entries could
        # land without crash protection or with untracked fills. The
        # SessionOrchestrator records these probes as broker_status="degraded"
        # but does NOT abort on bracket/fill-poller failure (only on broken
        # notifications — run_self_tests sets _notifications_broken alone), so
        # preflight is the authoritative launch gate for these. Now that the
        # mandatory pre-launch gate makes _run_preflight binding for every
        # order-routing launch (CLI + dashboard), a non-blocking WARNINGS here
        # would let real orders route on a known-degraded broker path.
        if not ctx.signal_only:
            return CheckResult(False, f"FAILED ({status_suffix})")
        # Signal-only places no orders; probes are advisory and surfaced only.
        return CheckResult(True, f"WARNINGS ({status_suffix})")
    except Exception as e:
        return CheckResult(False, f"FAILED: {e}")


def _pid_is_alive(pid: int) -> bool | None:
    """Best-effort liveness for a journal-lock holder PID.

    Returns True (alive), False (provably dead), or None (cannot determine).
    Self-contained on purpose — this mirrors the canonical
    ``scripts/tools/worktree_guard.py::_pid_is_alive`` logic but is NOT imported
    from it, so the live launcher does not couple to that module's lease
    internals / ``filelock`` dependency for a one-shot diagnostic probe.

    Windows is the load-bearing case: ``os.kill(pid, 0)`` does not reliably
    raise for a dead PID there, so probe via ``OpenProcess`` + exit code
    (STILL_ACTIVE == 259 means alive). We only have a bare PID from DuckDB's
    error text (no creation time), so PID reuse cannot be ruled out — "alive"
    means "a process with this PID exists", not "definitely the lock holder".
    """
    if not isinstance(pid, int) or pid <= 0:
        return None
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            # Rationale: fixed Win32 API constants (winnt.h), not tunable values.
            # PROCESS_QUERY_LIMITED_INFORMATION (0x1000) is the minimum access right
            # OpenProcess needs to call GetExitCodeProcess — narrower than
            # PROCESS_QUERY_INFORMATION so the probe works for processes owned by
            # other users without elevation. STILL_ACTIVE (259) is the sentinel
            # GetExitCodeProcess returns while a process is running; any other code
            # means it has exited. Both are OS-defined; changing them breaks the
            # liveness probe.
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False  # cannot open → not found / gone
            try:
                code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return None
                return code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    except OSError:
        return None


def _check_trade_journal(ctx: PreflightContext) -> CheckResult:
    """Trade journal health"""
    # session_orchestrator only enforces journal health when mode == "live", so a
    # broken journal stays invisible until session start. Surface it in preflight
    # so operators see the failure before committing to a session launch.
    try:
        from pipeline.paths import LIVE_JOURNAL_DB_PATH
        from trading_app.live.trade_journal import TradeJournal, TradeJournalLockedError

        journal = TradeJournal(LIVE_JOURNAL_DB_PATH, mode="preflight")
        if journal.is_healthy:
            return CheckResult(True, f"OK ({LIVE_JOURNAL_DB_PATH.name})")
        err = journal.last_error
        if isinstance(err, TradeJournalLockedError):
            # The "lock" is a DuckDB writer lock held by a LIVE process, not a
            # lockfile — there is nothing to "clear" if the holder is gone
            # (DuckDB releases on holder exit). So diagnose the holder PID's
            # liveness and tell the operator which action actually applies.
            holder = err.holder_pid
            if holder is None:
                return CheckResult(
                    False,
                    "LOCKED by another process (PID unknown). "
                    "Run: scripts/tools/stop_live.ps1 -NoPrompt to clear, then retry.",
                )
            alive = _pid_is_alive(holder)
            if alive is True:
                return CheckResult(
                    False,
                    f"LOCKED by live PID {holder}. Run: scripts/tools/stop_live.ps1 -NoPrompt to stop it, then retry.",
                )
            if alive is False:
                return CheckResult(
                    False,
                    f"LOCKED by stale/dead PID {holder} (holder gone — DuckDB should release). "
                    "Retry; if it persists, run: scripts/tools/stop_live.ps1 -NoPrompt.",
                )
            return CheckResult(
                False,
                f"LOCKED by PID {holder} (liveness unknown). "
                "Run: scripts/tools/stop_live.ps1 -NoPrompt to clear, then retry.",
            )
        return CheckResult(False, f"FAILED: TradeJournal could not open {LIVE_JOURNAL_DB_PATH}")
    except Exception as e:
        return CheckResult(False, f"FAILED: {e}")


def _check_repo_drift_for_live(ctx: PreflightContext) -> CheckResult:
    """Repo drift gate for live mode"""
    if ctx.signal_only:
        return CheckResult(True, "SKIPPED (signal-only)")
    if ctx.demo:
        return CheckResult(True, "SKIPPED (demo)")

    root = PROJECT_ROOT
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult(False, f"FAILED: cannot verify git state ({exc})")
    if result.returncode != 0:
        detail = result.stderr.strip() or str(result.returncode)
        return CheckResult(False, f"FAILED: git status failed ({detail})")

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    branch = lines[0] if lines else "## unknown"
    changes = lines[1:]
    if "behind" in branch.lower():
        return CheckResult(False, f"FAILED: repo behind origin ({branch})")
    if "ahead" in branch.lower():
        return CheckResult(False, f"FAILED: repo ahead of origin ({branch}); push or isolate before live launch")

    # Exclude files that are legitimately-always-dirty in the real multi-terminal
    # + live-journal operating environment, so the gate fails ONLY on genuine
    # code/config drift (the thing that actually risks capital):
    #   - live_journal.db  : the running session writes it every tick — never clean
    #   - HANDOFF.md       : a peer terminal's session-log surface, churns constantly
    #   - untracked ('??')  : new docs/scratch are not committed code drift
    # A change touching trading_app/, scripts/, pipeline/, trading config, or any
    # tracked source file STILL fails closed — that is the capital-protecting intent.
    # Exact repo-root paths (NOT suffixes): both ignored files live at the repo
    # root, so porcelain emits them as exactly "HANDOFF.md" / "live_journal.db".
    # A suffix match (`endswith`) was over-broad — any tracked CODE file ending
    # in those names (e.g. trading_app/foo_HANDOFF.md, data/test_live_journal.db)
    # would be waved past this live-arming capital gate. Exact-match is strictly
    # tighter and closes that hole.
    _DRIFT_IGNORE_PATHS = frozenset({"live_journal.db", "HANDOFF.md"})
    material_changes = []
    for line in changes:
        status_code = line[:2]
        if status_code.strip() == "??":
            continue  # untracked: not committed-code drift
        # Porcelain rename entries are "R  old -> new" — gate on the current
        # (renamed-to) path, not the raw "old -> new" string. split(' -> ')[-1]
        # yields `new` for renames and is a no-op for ordinary entries.
        path = line[3:].strip().split(" -> ")[-1].strip().strip('"')
        if path in _DRIFT_IGNORE_PATHS:
            continue  # always-dirty operational file (root HANDOFF.md / live_journal.db)
        material_changes.append(line)

    if material_changes:
        return CheckResult(
            False,
            f"FAILED: repo has uncommitted CODE/config drift ({len(material_changes)} path(s): "
            f"{', '.join(c[3:].strip() for c in material_changes[:5])}); commit or isolate before live launch",
        )
    ignored = len(changes) - len(material_changes)
    note = f"; {ignored} always-dirty/untracked path(s) ignored" if ignored else ""
    return CheckResult(True, f"OK ({branch}{note})")


def _check_telemetry_maturity(ctx: PreflightContext) -> CheckResult:
    """Telemetry maturity (signal-log distinct trading_days vs floor).

    Reads the canonical signal log files at repo root via
    trading_app.live.telemetry_maturity.evaluate_telemetry_maturity. The
    gate is orthogonal to the copy-trading gate (which checks broker-side
    account topology); this one counts distinct trading_days of bot
    uptime per instrument.

    Verdict matrix (see telemetry_maturity.py module docstring "DOCTRINE
    NOTE" for why this is advisory rather than hard-gated):

    - signal_only: OK (this is the path that accumulates the count).
    - demo: WARN (no real capital at risk).
    - live + Express-Funded prop profile (is_express_funded=True): OK —
      telemetry gate WAIVED. Operator decision 2026-06-01: the funded-account
      wrapper (Topstep XFA / Tradeify / Bulenox) insulates real personal
      capital, so the 30-day signal-maturity floor must NEVER block a funded
      live launch. See docs/governance/decisions/ + .claude/rules/
      telemetry-maturity-waiver.md.
    - live + real-capital broker (profile is_express_funded=False, unknown
      profile, or no profile): FAIL — conservative default. Real personal
      capital still requires the maturity floor; the waiver above is
      Express-Funded-only and does NOT relax this path.

    --all (no single instrument fixed): ctx.instrument may be a single
    symbol or a sentinel like "ALL"; evaluated against MNQ as the primary
    canonical instrument when ctx.instrument is unrecognized.
    """
    try:
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
        from trading_app.live.session_orchestrator import SessionOrchestrator
        from trading_app.live.telemetry_maturity import (
            VERDICT_MATURE,
            evaluate_telemetry_maturity,
        )

        signals_dir = SessionOrchestrator.SIGNALS_DIR
        target_instrument = ctx.instrument if ctx.instrument in ACTIVE_ORB_INSTRUMENTS else "MNQ"
        report = evaluate_telemetry_maturity(signals_dir, instrument=target_instrument, profile_id=ctx.profile_id)

        n = report.n_unique_trading_days
        floor = report.min_required
        if report.verdict == VERDICT_MATURE:
            return CheckResult(True, f"OK ({n}/{floor} distinct {target_instrument} trading_days; gate clear)")

        # Below floor — branch on capital-at-risk classification.
        if ctx.signal_only:
            return CheckResult(
                True,
                f"OK (signal-only: {n}/{floor} distinct {target_instrument} trading_days; auto-clears at {floor})",
            )

        # Resolve real-capital classification. Default conservative: anything
        # that isn't explicitly identifiable as demo OR Express-Funded prop is
        # treated as real-capital live and stays FAIL.
        is_real_capital_live = False
        if not ctx.demo:
            if ctx.profile_id is None:
                is_real_capital_live = True  # raw-baseline live = treat as real capital
            else:
                from trading_app.prop_profiles import ACCOUNT_PROFILES

                prof = ACCOUNT_PROFILES.get(ctx.profile_id)
                if prof is None:
                    is_real_capital_live = True  # unknown profile = treat as real capital
                elif not prof.is_express_funded:
                    is_real_capital_live = True  # real-capital live broker

        if is_real_capital_live:
            return CheckResult(
                False,
                f"FAILED: UNVERIFIED_INSUFFICIENT_TELEMETRY ({n}/{floor} distinct "
                f"{target_instrument} trading_days; run --signal-only until {floor})",
            )

        # Express-Funded live (Topstep XFA et al.): telemetry gate WAIVED per
        # operator decision 2026-06-01 — the funded-account wrapper insulates
        # real personal capital, so the maturity floor must never block a
        # funded live launch. Demo stays advisory WARN (no capital, but not a
        # deliberate waiver). See .claude/rules/telemetry-maturity-waiver.md.
        if ctx.demo:
            return CheckResult(
                True,
                f"WARN: telemetry below floor ({n}/{floor} distinct {target_instrument} "
                f"trading_days, demo — advisory; see telemetry_maturity.py module docstring)",
            )
        return CheckResult(
            True,
            f"OK (telemetry gate waived for Express-Funded profile={ctx.profile_id}; "
            f"{n}/{floor} distinct {target_instrument} trading_days — funded wrapper "
            f"insulates real capital, operator decision 2026-06-01)",
        )
    except Exception as e:
        return CheckResult(False, f"FAILED: {e}")


def _check_live_readiness_report(ctx: PreflightContext) -> CheckResult:
    """Strict live-readiness report"""
    if ctx.signal_only:
        return CheckResult(True, "SKIPPED (signal-only)")
    if ctx.profile_id is None:
        if ctx.demo:
            return CheckResult(True, "SKIPPED (no profile)")
        return CheckResult(False, "FAILED: live launch requires --profile for strict readiness")

    try:
        from scripts.tools.live_readiness_report import build_live_readiness_report, launch_blocking_strict_warnings

        report = build_live_readiness_report(profile_id=ctx.profile_id, effective_copies=ctx.requested_copies)
        strict = report.get("strict_zero_warn") or {}
        blockers = list(strict.get("blockers") or [])
        blocking_warnings = launch_blocking_strict_warnings(strict)
        if strict.get("green") is True:
            if not ctx.demo and blocking_warnings:
                msg = "; ".join(str(warning) for warning in blocking_warnings[:3])
                if len(blocking_warnings) > 3:
                    msg += f"; +{len(blocking_warnings) - 3} more"
                return CheckResult(False, f"FAILED: live readiness has blocking strict warnings ({msg})")
            return CheckResult(True, "OK (strict_zero_warn green)")
        msg = "; ".join(str(blocker) for blocker in blockers[:3])
        if len(blockers) > 3:
            msg += f"; +{len(blockers) - 3} more"
        if ctx.demo:
            return CheckResult(True, f"WARN: live readiness not green for demo ({msg})")
        return CheckResult(False, f"FAILED: live readiness not green ({msg})")
    except Exception as e:
        if ctx.demo:
            return CheckResult(True, f"WARN: live readiness report unavailable in demo ({e})")
        return CheckResult(False, f"FAILED: live readiness report unavailable ({e})")


def _check_project_pulse_for_live(ctx: PreflightContext) -> CheckResult:
    """Project pulse launch blocker for live mode.

    Demo and signal-only are evidence-accrual paths. Real-money launch must
    fail closed when the operator pulse still reports broken/high live-control
    findings or an explicit capital recommendation block.
    """
    if ctx.signal_only:
        return CheckResult(True, "SKIPPED (signal-only)")
    if ctx.demo:
        return CheckResult(True, "SKIPPED (demo)")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/tools/project_pulse.py",
                "--fast",
                "--format",
                "json",
                "--no-cache",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult(False, f"FAILED: project_pulse unavailable ({exc})")

    stdout = result.stdout.strip()
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        detail = (result.stderr or stdout or str(result.returncode)).strip().splitlines()[-1:]
        suffix = f": {detail[0]}" if detail else ""
        return CheckResult(False, f"FAILED: project_pulse JSON parse failed ({exc}){suffix}")

    capital_recommendation = str(payload.get("capital_recommendation") or "")
    if "blocked" in capital_recommendation.lower():
        return CheckResult(False, f"FAILED: {capital_recommendation}")

    blocking_items = [
        item
        for item in payload.get("items", [])
        if isinstance(item, dict)
        and item.get("category") == "broken"
        and str(item.get("severity") or "").lower() in {"critical", "high"}
    ]
    if blocking_items:
        summaries = "; ".join(str(item.get("summary") or "unknown") for item in blocking_items[:3])
        if len(blocking_items) > 3:
            summaries += f"; +{len(blocking_items) - 3} more"
        return CheckResult(False, f"FAILED: project_pulse blocker(s): {summaries}")

    if result.returncode != 0:
        return CheckResult(False, f"FAILED: project_pulse exited {result.returncode}")

    return CheckResult(True, "OK (project_pulse no live blockers)")


def _check_account_binding(ctx: PreflightContext) -> CheckResult:
    """Live order-routing must bind to an explicit broker account.

    Capital review A (2026-06-06): ContractResolver.resolve_account_id() returns
    ``accounts[0]`` from /api/Account/search. A profile-backed LIVE session with
    multiple active broker accounts and no explicit ``--account-id`` would route
    orders to whichever account the broker lists first — possibly a different
    account (e.g. 100K XFA or an eval combine) than the one the Criterion 11
    survival proof was computed for. That silently voids the profile-bound
    capital guard.

    Rule (fail-closed):
      * signal-only / no-profile        → SKIPPED (no order routing).
      * exactly one active account       → OK (accounts[0] is unambiguous).
      * explicit --account-id present    → OK iff it exists at the broker
                                           (delegated to the same selection
                                           helper used at live-start).
      * LIVE + >1 account + no binding   → FAILED (would default to accounts[0]).

    Demo is lenient (paper account, no real capital): a missing binding is a
    WARN-equivalent PASS so demo dry-runs are not blocked.
    """
    if ctx.signal_only:
        return CheckResult(True, "SKIPPED (signal-only — no order routing)")
    if ctx.profile_id is None:
        return CheckResult(True, "SKIPPED (no profile — raw-baseline path)")
    if ctx.components is None:
        return CheckResult(False, "FAILED: auth failed (cannot verify account binding)")
    try:
        contracts_cls = ctx.components["contracts_class"]
        contracts = contracts_cls(auth=ctx.components["auth"], demo=ctx.demo)
        all_accounts = contracts.resolve_all_account_ids()
        n = len(all_accounts)
        if n == 0:
            return CheckResult(False, "FAILED: no active broker accounts discovered")

        requested = ctx.requested_account_id
        if requested is not None and requested != 0:
            # Reuse the exact live-start validation: hard-fails if unknown.
            _select_primary_and_shadow_accounts(
                all_accounts=all_accounts,
                n_copies=1,
                requested_account_id=requested,
            )
            return CheckResult(True, f"OK (bound to --account-id {requested})")

        if n == 1:
            only_id, only_name = all_accounts[0]
            return CheckResult(True, f"OK (single broker account {only_name} id={only_id})")

        # >1 account and no explicit binding.
        if ctx.demo:
            return CheckResult(
                True,
                f"WARN: {n} accounts and no --account-id; demo routes to first. Live requires --account-id.",
            )
        ids = [aid for aid, _ in all_accounts]
        return CheckResult(
            False,
            f"FAILED: live profile-backed routing with {n} broker accounts {ids} "
            "and no --account-id would default to accounts[0] (possibly the WRONG "
            "account). Pass --account-id to bind the profile to its broker account.",
        )
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
        n_copies = ctx.requested_copies if ctx.requested_copies > 0 else prof.copies
        if n_copies <= 1:
            # Single-account pilot: copy-trading is genuinely not applicable, so
            # this is a clean PASS — not a SKIP. The dashboard maps SKIPPED -> warn
            # (_normalize_check_status), and the live-launch guard blocks on any
            # warn under strict-zero-warn parity (bot_dashboard.py action_start),
            # which would falsely block a single-account funded live launch. Emit
            # OK so a copies=1 pilot is launchable via the dashboard CTA.
            return CheckResult(True, f"OK (copies={n_copies}, single-account — copy-trading N/A)")
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
            n_copies=n_copies,
            requested_account_id=ctx.requested_account_id,
        )
        n = len(all_accounts)
        return CheckResult(True, f"OK (copies={n_copies}, {n} accounts discovered)")
    except Exception as e:
        return CheckResult(False, f"FAILED: {e}")


def _check_shadow_copy_loss_protection(ctx: PreflightContext) -> CheckResult:
    """Shadow-copy loss protection"""
    if ctx.signal_only:
        return CheckResult(True, "SKIPPED (signal-only)")
    if ctx.profile_id is None:
        return CheckResult(True, "SKIPPED (no profile)")

    try:
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        prof = ACCOUNT_PROFILES.get(ctx.profile_id)
        if prof is None:
            return CheckResult(False, f"FAILED: profile {ctx.profile_id!r} not in ACCOUNT_PROFILES")
        n_copies = ctx.requested_copies if ctx.requested_copies > 0 else prof.copies
        if n_copies <= 1:
            return CheckResult(True, f"OK (single-account pilot, copies={n_copies})")
        if ctx.demo:
            return CheckResult(True, f"WARN: multi-copy demo only (copies={n_copies}); live requires copies=1")
        return CheckResult(
            False,
            f"FAILED: SHADOW-MLL blocker - live copies={n_copies} but software daily-loss/HWM "
            "protection is primary-account only. Use --copies 1 or implement per-shadow loss belts.",
        )
    except Exception as e:
        return CheckResult(False, f"FAILED: {e}")


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


# Ordered list of checks. State coupling: _check_auth populates
# ctx.components (consumed by _check_contracts, _check_notifications, and
# _check_copy_trading_accounts); _check_notifications sets
# ctx.components_all_pass (read by the summary branch in _run_preflight).
# Reordering breaks the contract — see stage doc
# preflight-checks-total-hardcode.md § risk register.
PREFLIGHT_CHECKS: list[CheckFn] = [
    _check_auth,
    _check_portfolio,
    _check_survival_report,
    _check_sr_state,
    _check_daily_features,
    _check_contracts,
    _check_notifications,
    _check_trade_journal,
    _check_repo_drift_for_live,
    _check_telemetry_maturity,
    _check_live_readiness_report,
    _check_project_pulse_for_live,
    _check_account_binding,
    _check_copy_trading_accounts,
    _check_shadow_copy_loss_protection,
]


def _gate_title(check: CheckFn, ctx: PreflightContext) -> str:
    """Resolve the printed header text for a gate.

    Auth + portfolio carry a parenthesised broker / instrument suffix in the
    printed header (preserved exactly from the legacy launcher). All other
    checks derive their title from the function's __doc__ first line.
    """
    if check is _check_auth:
        return f"Auth check ({ctx.broker_name})"
    if check is _check_portfolio:
        return f"Portfolio check ({ctx.instrument})"
    return (check.__doc__ or check.__name__).strip().splitlines()[0]


@dataclass
class GateResult:
    """Structured per-gate outcome — the report-shaped twin of CheckResult.

    `identifier` is the check function name (e.g. "_check_auth"); `title` is the
    printed header text; `status` is the canonical token bucket
    (pass / warn / fail) resolved from passed + advisory; `message` is the full
    inline message; `advisory` is True iff a *passing* gate normalizes to "warn"
    (drives strict_block).
    """

    identifier: str
    title: str
    passed: bool
    advisory: bool
    status: str  # "pass" | "warn" | "fail"
    message: str


@dataclass
class PreflightReport:
    """Structured result of one preflight run — the single source of truth.

    Every surface (CLI print path, dashboard arm-guard) reads THIS instead of
    re-deriving a verdict from parsed stdout. `strict_block` is computed
    identically to the legacy launcher: strict_zero_warn requested AND not demo
    AND not signal-only AND at least one passing-advisory gate.

    `launch_ok` is the canonical go/no-go: all gates passed AND not strict_block.
    """

    gates: list[GateResult] = field(default_factory=list)
    checks_total: int = 0
    checks_passed: int = 0
    checks_warned: int = 0
    components_all_pass: bool = True
    strict_block: bool = False

    @property
    def all_passed(self) -> bool:
        return self.checks_total > 0 and self.checks_passed == self.checks_total

    @property
    def launch_ok(self) -> bool:
        return self.all_passed and not self.strict_block

    @property
    def overall(self) -> str:
        """pass / warn / fail — parity with _parse_preflight_output's overall."""
        if any(g.status == "fail" for g in self.gates):
            return "fail"
        if self.strict_block or any(g.status == "warn" for g in self.gates):
            return "warn"
        return "pass"


def _gate_status_token(passed: bool, advisory: bool) -> str:
    """Resolve a gate's canonical bucket (pass/warn/fail).

    A failing gate is always "fail". A passing-advisory gate is "warn". A
    passing non-advisory gate is "pass". This is the report-level twin of the
    inline-token classification the dashboard performs on parsed stdout, so the
    in-process path and the subprocess fallback agree gate-for-gate.
    """
    if not passed:
        return "fail"
    return "warn" if advisory else "pass"


def run_preflight(
    instrument: str,
    broker: str | None,
    demo: bool,
    portfolio: Any = None,
    profile_id: str | None = None,
    requested_account_id: int | None = None,
    signal_only: bool = False,
    requested_copies: int = 0,
    strict_zero_warn: bool = False,
    checks: list[CheckFn] | None = None,
) -> PreflightReport:
    """Run the canonical 15-gate preflight and return a structured report.

    This is the SINGLE shared engine. It runs every gate in canonical order,
    classifies each outcome, and computes `strict_block` exactly as the legacy
    launcher did. It does NOT print and does NOT exit — callers decide how to
    render or act on the report:

      - `scripts/run_live_session.py::_run_preflight` renders it to stdout and
        returns `report.launch_ok` (preserving the CLI exit semantics).
      - `bot_dashboard.py` consumes the report in-process via a fail-closed
        adapter, replacing the third subprocess + stdout regex-parse.

    `strict_zero_warn` semantics are unchanged: it only bites on real-money runs
    (not demo, not signal-only); demo/signal-only place no live orders so
    advisory gates stay non-blocking there.

    `checks` lets a caller pass its OWN gate list (the launcher passes its
    re-imported `PREFLIGHT_CHECKS` so the test suite's
    `monkeypatch.setattr(rls, "PREFLIGHT_CHECKS", ...)` injection drives both the
    count and the gates that run). Defaults to this module's canonical
    `PREFLIGHT_CHECKS`.
    """
    from trading_app.live.broker_factory import get_broker_name

    active_checks = checks if checks is not None else PREFLIGHT_CHECKS

    ctx = PreflightContext(
        instrument=instrument,
        broker_name=broker or get_broker_name(),
        demo=demo,
        portfolio=portfolio,
        profile_id=profile_id,
        requested_account_id=requested_account_id,
        signal_only=signal_only,
        requested_copies=requested_copies,
    )

    report = PreflightReport(checks_total=len(active_checks))
    for check in active_checks:
        title = _gate_title(check, ctx)
        result = check(ctx)
        advisory = result.passed and _passing_check_is_advisory(result.message)
        report.gates.append(
            GateResult(
                identifier=check.__name__,
                title=title,
                passed=result.passed,
                advisory=advisory,
                status=_gate_status_token(result.passed, advisory),
                message=result.message,
            )
        )
        if result.passed:
            report.checks_passed += 1
            if advisory:
                report.checks_warned += 1

    report.components_all_pass = ctx.components_all_pass
    # strict-zero-warn only bites on real-money runs. demo/signal-only place no
    # live orders, so an advisory check stays non-blocking there (no false block
    # of a paper/signal launch).
    report.strict_block = bool(strict_zero_warn) and not demo and not ctx.signal_only and report.checks_warned > 0
    return report
