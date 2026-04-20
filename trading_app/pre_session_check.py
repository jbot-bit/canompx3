"""Pre-session system health check — HARD GATE for manual Phase 1 trading.

No check passing = no trade that session. This is not advisory.

Usage:
    python -m trading_app.pre_session_check --session NYSE_CLOSE
    python -m trading_app.pre_session_check --session SINGAPORE_OPEN
    python -m trading_app.pre_session_check --session COMEX_SETTLE
    python -m trading_app.pre_session_check --session NYSE_OPEN
    python -m trading_app.pre_session_check --session TOKYO_OPEN  (MGC shadow)
"""

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH, LIVE_JOURNAL_DB_PATH
from trading_app.lifecycle_state import read_lifecycle_state

STATE_DIR = Path(__file__).resolve().parents[1] / "data" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

HALT_FILE = STATE_DIR / "halt_trading.json"


def check_equity_cooldown(portfolio_name: str, instrument: str) -> tuple[bool, str]:
    """Check if a mandatory cooling period is active (self-funded equity halt).

    After an equity halt (max DD breached), a 24h cooldown is enforced.
    This prevents emotional re-entry after a major loss event.
    """
    from datetime import UTC, datetime

    from trading_app.live.session_safety_state import SessionSafetyState

    state = SessionSafetyState(portfolio_name, instrument)
    if not state.cooldown_until:
        return True, "No cooldown active"
    try:
        cooldown_end = datetime.fromisoformat(state.cooldown_until)
        if datetime.now(UTC) >= cooldown_end:
            # Cooldown expired — clear it
            state.cooldown_until = ""
            state.save()
            return True, "Cooldown expired (auto-cleared)"
        remaining = cooldown_end - datetime.now(UTC)
        hours = remaining.total_seconds() / 3600
        return False, f"EQUITY COOLDOWN: {hours:.1f}h remaining (until {state.cooldown_until})"
    except (ValueError, TypeError):
        return False, "BLOCKED: cooldown_until is set but unparseable — manual inspection needed"


def check_manual_halt() -> tuple[bool, str]:
    """Check if the user has manually halted trading for today."""
    if not HALT_FILE.exists():
        return True, "No manual halt active"
    try:
        data = json.loads(HALT_FILE.read_text())
        if not data.get("active", False):
            return True, "Manual halt: inactive"
        expires = data.get("expires")
        if expires and date.fromisoformat(expires) < date.today():
            return True, "Manual halt: expired (auto-resumed)"
        reason = data.get("reason", "no reason given")
        return False, f"MANUAL HALT: {reason}"
    except (json.JSONDecodeError, OSError):
        return False, "BLOCKED: halt file exists but unreadable — cannot verify halt status"


def write_halt(reason: str) -> None:
    """Write a manual halt file. Expires tomorrow by default."""
    from datetime import timedelta

    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    data = {
        "active": True,
        "reason": reason,
        "created": datetime.now(UTC).isoformat(),
        "expires": tomorrow,
    }
    HALT_FILE.write_text(json.dumps(data, indent=2))
    print(f"HALT ACTIVE until {tomorrow}: {reason}")


def clear_halt() -> None:
    """Remove the manual halt file."""
    if HALT_FILE.exists():
        HALT_FILE.unlink()
        print("Manual halt CLEARED. Trading resumed.")
    else:
        print("No halt was active.")


def check_data_freshness(con, instrument: str) -> tuple[bool, str]:
    """Check gold.db has recent data for this instrument."""
    row = con.execute("SELECT MAX(ts_utc) FROM bars_1m WHERE symbol = ?", [instrument]).fetchone()
    if not row or not row[0]:
        return False, f"No bars_1m data for {instrument}"
    latest = row[0]
    now_utc = datetime.now(UTC)
    if hasattr(latest, "timestamp"):
        gap_hours = (now_utc.timestamp() - latest.timestamp()) / 3600
    else:
        gap_hours = 999
    if gap_hours > 48:
        return False, f"Data stale: latest bar {latest} ({gap_hours:.0f}h ago)"
    return True, f"Latest bar: {latest} ({gap_hours:.1f}h ago)"


def check_paper_trades_accessible(con) -> tuple[bool, str]:
    """Check paper_trades table is accessible."""
    try:
        n = con.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
        return True, f"paper_trades: {n} rows"
    except Exception as e:
        return False, f"paper_trades inaccessible: {e}"


def check_dd_circuit_breaker() -> tuple[bool, str]:
    """Check DD status from AccountHWMTracker state files.

    Replaces the previous dd_circuit_breaker.json ghost check (file was never
    written by any code path). Now reads the authoritative HWM tracker state
    which IS written by session_orchestrator on every equity poll.
    """
    # Primary: check HWM tracker state files (authoritative source)
    hwm_files = list(STATE_DIR.glob("account_hwm_*.json"))
    hwm_files = [f for f in hwm_files if "CORRUPT" not in f.name]
    if hwm_files:
        for f in hwm_files:
            try:
                text = f.read_text()
                if not text.strip():
                    return False, f"BLOCKED: HWM file unreadable ({f.name} is empty)"
                data = json.loads(text)
                if data.get("halt_triggered"):
                    return (
                        False,
                        f"DD HALT ACTIVE: account {data.get('account_id', '?')} — DD ${data.get('dd_used_dollars', 0):.0f} >= limit ${data.get('dd_limit_dollars', 0):.0f}",
                    )
            except (json.JSONDecodeError, ValueError):
                return False, f"BLOCKED: HWM file unreadable ({f.name} is corrupt)"
            except OSError:
                return False, f"BLOCKED: HWM file unreadable ({f.name})"
        return True, "DD circuit breaker: clear (HWM tracker)"
    return True, "No DD tracker state (first session — will init from broker)"


def check_daily_equity(profile_id: str | None = None) -> tuple[bool, str]:
    """Check today's equity tracking state."""
    eq_file = STATE_DIR / f"equity_{date.today()}.json"
    if not eq_file.exists():
        # Create it
        data = {"date": str(date.today()), "starting_equity": None, "current_dd": 0.0}
        eq_file.write_text(json.dumps(data, indent=2))
        return True, "Equity file created for today. Record starting equity before first trade."
    data = json.loads(eq_file.read_text())
    dd = data.get("current_dd", 0.0)

    # DLL from canonical ACCOUNT_TIERS (not hardcoded)
    from trading_app.prop_profiles import ACCOUNT_TIERS, get_profile, resolve_profile_id

    dll = 1000.0  # fallback
    try:
        resolved_profile_id = resolve_profile_id(profile_id)
        prof = get_profile(resolved_profile_id)
        tier = ACCOUNT_TIERS.get((prof.firm, prof.account_size))
        if tier and tier.daily_loss_limit:
            dll = tier.daily_loss_limit
    except Exception as exc:
        print(
            f"WARNING: could not load DLL for profile {profile_id!r} — using fallback ${dll:,.0f} ({exc})",
            file=sys.stderr,
        )

    if dd <= -dll:
        return False, f"DAILY DD LIMIT BREACHED: ${dd:.0f} (limit -${dll:,.0f})"
    if dd <= -dll * 0.8:
        return True, f"WARNING: daily DD at ${dd:.0f} (halt at -${dll:,.0f})"
    return True, f"Daily DD: ${dd:.0f}"


def check_slippage_pilot_progress(con) -> str:
    """Check progress toward MNQ slippage pilot (30 live trades needed)."""
    try:
        row = con.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE execution_source = 'live' AND slippage_ticks IS NOT NULL"
        ).fetchone()
        n = row[0] if row else 0
        return f"MNQ slippage pilot: {n}/30 live trades with slippage recorded"
    except (OSError, RuntimeError) as e:
        return f"MNQ slippage pilot: cannot query ({e})"


def check_live_attribution_health(
    lanes: list[dict],
    *,
    db_path: Path = GOLD_DB_PATH,
    journal_path: Path = LIVE_JOURNAL_DB_PATH,
) -> tuple[bool, str]:
    """Warn when current lane attribution still has no real evidence.

    This is advisory, not a hard gate. It makes the Phase 2 evidence gap
    visible in the canonical pre-session operator surface.
    """
    strategy_ids = [lane["strategy_id"] for lane in lanes]
    if not strategy_ids:
        return True, "Live attribution: no lanes resolved for this session"

    placeholders = ",".join("?" * len(strategy_ids))

    live_rows = 0
    try:
        with duckdb.connect(str(db_path), read_only=True) as con:
            configure_connection(con)
            tables = {
                row[0]
                for row in con.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
                ).fetchall()
            }
            if "paper_trades" in tables:
                cols = {row[1] for row in con.execute("PRAGMA table_info('paper_trades')").fetchall()}
                exec_source_expr = "execution_source" if "execution_source" in cols else "'unknown'"
                row = con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM paper_trades
                    WHERE strategy_id IN ({placeholders})
                      AND {exec_source_expr} IN ('live', 'shadow')
                    """,
                    strategy_ids,
                ).fetchone()
                live_rows = int(row[0]) if row else 0
    except Exception as e:
        return True, f"WARN: live attribution check could not read paper_trades ({e})"

    event_rows = 0
    try:
        if journal_path.exists():
            with duckdb.connect(str(journal_path), read_only=True) as con:
                configure_connection(con)
                tables = {
                    row[0]
                    for row in con.execute(
                        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
                    ).fetchall()
                }
                if "live_signal_events" in tables:
                    row = con.execute(
                        f"SELECT COUNT(*) FROM live_signal_events WHERE strategy_id IN ({placeholders})",
                        strategy_ids,
                    ).fetchone()
                    event_rows = int(row[0]) if row else 0
    except Exception as e:
        return True, f"WARN: live attribution check could not read live_signal_events ({e})"

    if live_rows > 0:
        return True, f"Live attribution: {live_rows} completed live/shadow rows, {event_rows} event rows"
    if event_rows > 0:
        return True, f"WARN: live attribution has {event_rows} event rows but 0 completed live/shadow rows yet"
    lane_labels = ", ".join(sorted(lane["orb_label"] for lane in lanes))
    return True, f"WARN: no live attribution evidence yet for current lane(s): {lane_labels}"


def check_hwm_tracker() -> tuple[bool, str]:
    """Check account HWM DD tracker status."""
    from pathlib import Path as _P

    hwm_dir = _P(__file__).resolve().parents[1] / "data" / "state"
    hwm_files = list(hwm_dir.glob("account_hwm_*.json"))
    if not hwm_files:
        return True, "No HWM tracker active (first session — will init from broker)"

    results = []
    any_halt = False
    for f in hwm_files:
        if "CORRUPT" in f.name:
            continue
        try:
            data = json.loads(f.read_text())
            acct = data.get("account_id", "?")
            hwm = data.get("hwm_dollars", 0)
            used = data.get("dd_used_dollars", 0)
            limit = data.get("dd_limit_dollars", 0)
            pct = data.get("dd_pct_used", 0)
            halted = data.get("halt_triggered", False)
            hwm_date = (data.get("hwm_timestamp") or "")[:10]
            remaining = limit - used

            if halted:
                any_halt = True
                results.append(f"HALT {acct}: DD ${used:.0f} >= limit ${limit:.0f}")
            elif pct >= 0.75:
                results.append(
                    f"WARN {acct}: DD ${used:.0f}/{limit:.0f} ({pct:.0%}) — "
                    f"${remaining:.0f} remaining. HWM ${hwm:.0f} on {hwm_date}"
                )
            else:
                results.append(f"OK {acct}: DD ${used:.0f}/{limit:.0f} ({pct:.0%}) — ${remaining:.0f} remaining")
        except Exception as e:
            any_halt = True  # fail-closed: can't verify DD state → block
            results.append(f"BLOCKED {f.name}: {e}")

    msg = " | ".join(results)
    return (not any_halt), f"DD TRACKER: {msg}"


def check_topstep_xfa_aggregate_cap() -> tuple[bool, str]:
    """Enforce TopStep's 5-XFA simultaneous-active cap.

    @canonical-source docs/research-input/topstep/topstep_xfa_parameters.txt  (article 8284215, scraped 2026-04-08)
    @verbatim "You can have up to 5 active Express Funded Accounts at the same time."
    @audit-finding F-6 (MEDIUM — startup gate to prevent future activation accidents)

    Sums `copies` across every TopStep AccountProfile with active=True. Fails fast
    if the total exceeds 5. Returns OK with a status string otherwise.
    """
    try:
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        active_topstep = [p for p in ACCOUNT_PROFILES.values() if p.active and p.firm == "topstep"]
        total_copies = sum(p.copies for p in active_topstep)
        cap = 5

        if total_copies > cap:
            offenders = ", ".join(f"{p.profile_id}(x{p.copies})" for p in active_topstep)
            return False, (
                f"BLOCKED: TopStep 5-XFA cap breached — {total_copies} active copies across "
                f"{len(active_topstep)} profile(s): {offenders}. Disable profiles until total ≤ {cap}."
            )

        if total_copies == cap:
            offenders = ", ".join(f"{p.profile_id}(x{p.copies})" for p in active_topstep)
            return True, f"WARNING: at TopStep 5-XFA cap ({total_copies}/{cap}) — {offenders}"

        return True, f"TopStep XFA aggregate: {total_copies}/{cap} active copies"
    except Exception as e:  # noqa: BLE001 — startup gate must surface anything
        return False, f"BLOCKED: TopStep XFA cap check failed: {e}"


def check_consistency_rule(profile_id: str | None = None) -> tuple[bool, str]:
    """Check prop firm consistency rule status."""
    try:
        from trading_app.consistency_tracker import check_profile_consistency
        from trading_app.prop_profiles import get_profile, resolve_profile_id

        resolved_profile_id = resolve_profile_id(profile_id)
        profile = get_profile(resolved_profile_id)
        result = check_profile_consistency(resolved_profile_id, instrument=None)
        if result is None:
            return True, f"Consistency: no active rule for {profile.profile_id}"
        status_ok = result.status != "BREACH"
        return status_ok, (
            f"Consistency {profile.profile_id} ({result.limit_pct:.0%} rule): best day ${result.best_day_pnl:.0f} on {result.best_day_date} "
            f"= {result.windfall_pct:.1f}% of ${result.total_profit:.0f} total — {result.status}"
        )
    except Exception as e:
        return False, f"BLOCKED: Consistency check failed: {e}"


def _resolve_session_lane(session: str, profile_id: str | None = None) -> tuple[str, dict]:
    """Resolve one session lane for the requested profile, failing closed on ambiguity."""
    from trading_app.prop_profiles import get_profile_lane_definitions, resolve_profile_id

    resolved_profile_id = resolve_profile_id(profile_id)
    lanes = get_profile_lane_definitions(resolved_profile_id)
    matches = [lane for lane in lanes if lane["orb_label"] == session]
    if not matches:
        valid = ", ".join(sorted({lane["orb_label"] for lane in lanes}))
        raise ValueError(f"Unknown session '{session}' for profile '{resolved_profile_id}'. Valid: {valid}")
    if len(matches) > 1:
        options = ", ".join(lane["strategy_id"] for lane in matches)
        raise ValueError(
            f"Session '{session}' has multiple lanes in profile '{resolved_profile_id}'. "
            f"Use a strategy-specific tool. Options: {options}"
        )
    return resolved_profile_id, matches[0]


def _resolve_session_lanes(session: str, profile_id: str | None = None) -> tuple[str, list[dict]]:
    """Resolve all matching session lanes for the requested profile."""
    from trading_app.prop_profiles import get_profile_lane_definitions, resolve_profile_id

    resolved_profile_id = resolve_profile_id(profile_id)
    lanes = get_profile_lane_definitions(resolved_profile_id)
    matches = [lane for lane in lanes if lane["orb_label"] == session]
    if not matches:
        valid = ", ".join(sorted({lane["orb_label"] for lane in lanes}))
        raise ValueError(f"Unknown session '{session}' for profile '{resolved_profile_id}'. Valid: {valid}")
    return resolved_profile_id, matches


def check_forward_monitor_freshness() -> tuple[bool, str]:
    """Check if forward_monitor has been run recently."""
    fm_file = Path(__file__).resolve().parents[1] / "data" / "forward_monitoring" / "latest.json"
    if not fm_file.exists():
        return True, "WARN: forward_monitor never run. Run: python -m scripts.tools.forward_monitor"
    try:
        data = json.loads(fm_file.read_text())
        ts = data.get("timestamp", "")
        return True, f"Forward monitor last run: {ts}"
    except (json.JSONDecodeError, OSError):
        return True, "WARN: forward_monitor output unreadable"


def check_account_survival(profile_id: str | None = None) -> tuple[bool, str]:
    """Fail-closed Criterion 11 gate based on the shared lifecycle reader."""
    try:
        lifecycle = read_lifecycle_state(profile_id=profile_id)
        return _account_survival_from_lifecycle(lifecycle)
    except Exception as e:
        return False, f"BLOCKED: Criterion 11 gate failed: {e}"


def check_lane_lifecycle(strategy_id: str, profile_id: str | None = None) -> tuple[bool, str]:
    """Check lane-level operational lifecycle status from the shared reader."""
    try:
        lifecycle = read_lifecycle_state(profile_id=profile_id)
    except Exception as e:
        return False, f"BLOCKED: lifecycle state unavailable ({e})"

    return _lane_lifecycle_from_lifecycle(lifecycle, strategy_id)


def _account_survival_from_lifecycle(lifecycle: dict) -> tuple[bool, str]:
    """Interpret Criterion 11 from a shared lifecycle snapshot."""
    c11 = lifecycle["criterion11"]
    return bool(c11["gate_ok"]), str(c11["gate_msg"])


def _lane_lifecycle_from_lifecycle(lifecycle: dict, strategy_id: str) -> tuple[bool, str]:
    """Interpret lane-level lifecycle status from a shared snapshot."""
    state = lifecycle["strategy_states"].get(strategy_id, {})
    c12 = lifecycle["criterion12"]
    if state.get("blocked"):
        reason = state.get("block_reason") or "Blocked pending manual review"
        return False, f"BLOCKED: {strategy_id} — {reason}"

    if c12.get("valid"):
        sr_status = state.get("sr_status")
        if sr_status == "ALARM" and state.get("sr_review_outcome") == "watch":
            return True, f"Criterion 12 SR reviewed WATCH for {strategy_id}"
        if sr_status == "CONTINUE":
            return True, f"Criterion 12 SR clear for {strategy_id}"
        if sr_status == "NO_DATA":
            return True, f"Criterion 12 SR has no data yet for {strategy_id}"
        if sr_status:
            return True, f"Criterion 12 SR status for {strategy_id}: {sr_status}"

    reason = c12.get("reason")
    if c12.get("available") and not c12.get("valid"):
        return True, f"WARN: Criterion 12 SR state stale/mismatched ({reason})"
    return True, "Criterion 12 SR state unavailable"


def check_allocation_staleness_gate() -> tuple[bool, str]:
    """Check if lane allocation is stale (>35d warn, >60d block).

    Uses check_allocation_staleness from lane_allocator.py.
    Missing file = BLOCK (fail-closed).
    """
    from trading_app.lane_allocator import check_allocation_staleness

    status, days_old = check_allocation_staleness()
    if status == "BLOCK":
        if days_old == -1:
            return False, "BLOCKED: No lane_allocation.json. Run: python scripts/tools/rebalance_lanes.py"
        return False, f"BLOCKED: allocation {days_old}d old (>60d). Run: python scripts/tools/rebalance_lanes.py"
    if status == "WARNING":
        return True, f"WARN: allocation {days_old}d old (>35d). Rebalance soon."
    return True, f"Allocation {days_old}d old — fresh"


def check_lane_mismatch(session: str, lane: dict) -> tuple[bool, str]:
    """Warn if deployed lane differs from allocator recommendation.

    Reads lane_allocation.json and compares the deployed strategy_id
    for this session+instrument to the allocator's recommended strategy_id.
    Non-blocking (warning only) — user decides whether to follow recommendation.
    """
    alloc_path = Path(__file__).resolve().parents[1] / "docs" / "runtime" / "lane_allocation.json"
    if not alloc_path.exists():
        return True, "No allocation file — cannot compare"

    try:
        import json as _json

        data = _json.loads(alloc_path.read_text())
        # Key by (instrument, orb_label) to handle multi-instrument same-session
        recommended = {
            (entry["instrument"], entry["orb_label"]): entry["strategy_id"] for entry in data.get("lanes", [])
        }
        paused_ids = {entry["strategy_id"] for entry in data.get("paused", [])}
    except (KeyError, _json.JSONDecodeError):
        return True, "Cannot parse allocation file"

    deployed_sid = lane.get("strategy_id", "")
    instrument = lane.get("instrument", "")
    rec_sid = recommended.get((instrument, session))

    if rec_sid is None:
        # Check if this session is actively paused
        if deployed_sid in paused_ids:
            return True, f"WARN: {deployed_sid} is PAUSED by allocator"
        return True, f"WARN: {session} not in allocator recommendation"
    if deployed_sid == rec_sid:
        return True, f"Matches recommendation: {rec_sid}"
    return True, f"MISMATCH: deployed={deployed_sid}, recommended={rec_sid}"


def check_signal_exists(con, session: str, lane: dict, today: date) -> tuple[bool, str]:
    """Check if today has outcome data for this session (signal fired)."""
    instrument = lane["instrument"]
    orb_min = lane["orb_minutes"]
    entry_model = lane["entry_model"]
    row = con.execute(
        """SELECT COUNT(*) FROM orb_outcomes
           WHERE symbol = ? AND orb_label = ? AND orb_minutes = ?
             AND entry_model = ? AND trading_day = ?""",
        [instrument, session, orb_min, entry_model, today],
    ).fetchone()
    n = row[0] if row else 0
    if n > 0:
        return True, f"Signal exists for {session} today ({n} outcomes)"
    return True, f"No signal yet for {session} today (may not have fired yet)"


def run_checks(session: str, profile_id: str | None = None) -> bool:
    """Run all checks for a session. Returns True if GO.

    Checks run in order: manual halt → data freshness → account state →
    signal → DD budget → lane-specific.
    """
    try:
        resolved_profile_id, lanes = _resolve_session_lanes(session, profile_id)
    except ValueError as e:
        print(f"ERROR: {e}")
        return False
    today = date.today()
    results = []
    unique_instruments = sorted({lane["instrument"] for lane in lanes})

    # Market calendar (check FIRST — holidays override everything)
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from pipeline.market_calendar import is_cme_holiday, is_early_close

        us_date = datetime.now(ZoneInfo("America/New_York")).date()
        if is_cme_holiday(us_date):
            print(f"BLOCKED: CME HOLIDAY ({us_date}) — all sessions closed.")
            return False
        if is_early_close(us_date):
            print(f"WARNING: Early close day ({us_date}). Exchange closes 12:00 PM CT / 1:00 PM ET.")
            print("         Afternoon sessions (COMEX_SETTLE, CME_PRECLOSE, NYSE_CLOSE, CME_REOPEN) may not fire.")
    except ImportError:
        print("WARNING: market_calendar not available — holiday/early-close check SKIPPED")
        print("         Install exchange-calendars: uv sync")

    # Manual halt (check first — overrides everything)
    ok, msg = check_manual_halt()
    results.append(("Manual halt", ok, msg))

    # TopStep 5-XFA aggregate cap (F-6 — startup gate, prevents future activation accidents)
    ok, msg = check_topstep_xfa_aggregate_cap()
    results.append(("TopStep 5-XFA cap", ok, msg))

    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        configure_connection(con)

        # Data freshness
        for instrument in unique_instruments:
            ok, msg = check_data_freshness(con, instrument)
            results.append((f"Data freshness ({instrument})", ok, msg))

        ok, msg = check_paper_trades_accessible(con)
        results.append(("Paper trades table", ok, msg))

        ok, msg = check_forward_monitor_freshness()
        results.append(("Forward monitor", ok, msg))

        lifecycle = read_lifecycle_state(profile_id=resolved_profile_id)

        ok, msg = _account_survival_from_lifecycle(lifecycle)
        results.append(("Criterion 11 survival", ok, msg))

        for lane in lanes:
            ok, msg = _lane_lifecycle_from_lifecycle(lifecycle, lane["strategy_id"])
            results.append((f"Lane lifecycle ({lane['instrument']})", ok, msg))

        # Allocation staleness
        ok, msg = check_allocation_staleness_gate()
        results.append(("Allocation staleness", ok, msg))

        # Lane mismatch: deployed vs allocator recommendation
        for lane in lanes:
            ok, msg = check_lane_mismatch(session, lane)
            results.append((f"Lane vs recommendation ({lane['instrument']})", ok, msg))

        # Account state
        ok, msg = check_hwm_tracker()
        results.append(("HWM DD tracker", ok, msg))

        ok, msg = check_dd_circuit_breaker()
        results.append(("DD circuit breaker (intraday)", ok, msg))

        ok, msg = check_daily_equity(resolved_profile_id)
        results.append(("Daily equity", ok, msg))

        ok, msg = check_consistency_rule(resolved_profile_id)
        results.append(("Consistency rule", ok, msg))

        # Signal
        for lane in lanes:
            ok, msg = check_signal_exists(con, session, lane, today)
            results.append((f"Signal check ({lane['instrument']})", ok, msg))

        # Slippage pilot progress
        slip_msg = check_slippage_pilot_progress(con)
        results.append(("Slippage pilot", True, slip_msg))

    ok, msg = check_live_attribution_health(lanes)
    results.append(("Live attribution", ok, msg))

    # DD budget check (from daily_lanes)
    try:
        from trading_app.prop_portfolio import check_daily_lanes_dd_budget, resolve_daily_lanes
        from trading_app.prop_profiles import effective_daily_lanes, get_profile

        profile = get_profile(resolved_profile_id)
        if effective_daily_lanes(profile):
            resolved = resolve_daily_lanes(profile, db_path=GOLD_DB_PATH, trading_day=today)
            _, total_dd, dd_limit, over = check_daily_lanes_dd_budget(profile, resolved)
            if over:
                pct = total_dd / dd_limit * 100
                results.append(
                    (
                        "DD budget",
                        True,  # Warning, not blocking — user chose these lanes
                        f"⚠ OVER-COMMITTED: ${total_dd:,.0f} / ${dd_limit:,.0f} ({pct:.0f}%). "
                        f"Intraday DD halt is your safety net.",
                    )
                )
            else:
                results.append(("DD budget", True, f"${total_dd:,.0f} / ${dd_limit:,.0f} — within budget"))
    except Exception as e:
        results.append(("DD budget", True, f"Cannot check: {e}"))

    # Lane 2 enforcement
    for lane in lanes:
        if lane.get("is_half_size"):
            results.append((f"Lane 2 sizing ({lane['instrument']})", True, "0.5x MANDATORY — 1 micro lot max"))

    # Shadow-only check
    for lane in lanes:
        if lane.get("shadow_only"):
            results.append(
                (f"Shadow mode ({lane['instrument']})", True, "MGC TOKYO_OPEN: shadow-trade ONLY, no real capital")
            )

    # Print results
    all_pass = True
    print(f"\n{'=' * 70}")
    print(f"PRE-SESSION CHECK: {resolved_profile_id} | {session} | {today} | {', '.join(unique_instruments)}")
    print(f"Lanes: {len(lanes)}")
    for lane in lanes:
        orb_cap = lane.get("max_orb_size_pts")
        cap_str = f"{orb_cap:.0f} pts" if orb_cap else "NONE"
        print(f"Strategy: {lane['strategy_id']}")
        print(
            f"  {lane['instrument']} | Filter: {lane['filter_type']} | RR: {lane['rr_target']} | "
            f"ORB: O{lane['orb_minutes']} | ORB Cap: {cap_str}"
        )
    print(f"{'=' * 70}")

    for name, ok, msg in results:
        if not ok:
            all_pass = False
        indicator = "[+]" if ok else "[X]"
        print(f"  {indicator} {name}: {msg}")

    print(f"{'=' * 70}")
    gate = "GO" if all_pass else "NO-GO"
    print(f"  GATE STATUS: {gate}")
    if not all_pass:
        print("  ACTION: Fix FAIL items above before trading this session.")
    print(f"{'=' * 70}")

    # Lane 2 confirmation
    if any(lane.get("is_half_size") for lane in lanes) and all_pass:
        print("\n  ** Lane 2 (SINGAPORE_OPEN RR4.0) is 0.5x sizing. **")
        print("  ** You MUST enter only 1 micro lot. **")

    # Log to file
    log_data = {
        "date": str(today),
        "session": session,
        "profile_id": resolved_profile_id,
        "lane_count": len(lanes),
        "strategies": [lane["strategy_id"] for lane in lanes],
        "gate": gate,
        "checks": [{"name": n, "pass": o, "detail": m} for n, o, m in results],
        "timestamp": datetime.now(UTC).isoformat(),
    }
    log_file = STATE_DIR / f"session_checklist_{today}_{session}.json"
    log_file.write_text(json.dumps(log_data, indent=2))

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Pre-session health check (hard gate)")
    parser.add_argument("--profile", help="Execution profile id. Required if multiple active profiles exist.")
    parser.add_argument("--session", help="Session to check")
    parser.add_argument("--halt", metavar="REASON", help="Halt all trading until tomorrow")
    parser.add_argument("--resume", action="store_true", help="Clear manual halt")
    args = parser.parse_args()

    if args.halt:
        write_halt(args.halt)
        sys.exit(0)
    if args.resume:
        clear_halt()
        sys.exit(0)
    if not args.session:
        parser.error("--session is required (or use --halt/--resume)")

    ok = run_checks(args.session, profile_id=args.profile)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
