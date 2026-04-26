"""
Persistent dollar-based High Water Mark tracker for prop firm trailing drawdown.

Supported prop firms use EOD trailing drawdown from the account's
ALL-TIME high water mark in dollars. The existing RiskManager tracks R-units within
a single process lifetime — it resets on restart and doesn't know about dollar equity.

This module:
- Persists HWM across sessions in JSON state files
- Tracks DD used vs DD limit in dollars
- Halts trading when DD >= limit (fail-closed)
- Integrates with session_orchestrator, pre_session_check, weekly_review
"""

import json
import logging
import math
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

_DEFAULT_STATE_DIR = Path(__file__).resolve().parent.parent / "data" / "state"
_BRISBANE = ZoneInfo("Australia/Brisbane")


def _is_finite_equity(value: float) -> bool:
    """Return True iff value is a real number (not NaN, not +/-Inf).

    Stage 2 audit-gate SG4 fix — broker contract is `float | None` but
    technically a NaN/Inf is a valid `float`. Treating NaN as a real
    equity reading would silently bypass every halt comparison (NaN >=
    limit is False, NaN < threshold is False — kill switch never fires).
    Tracker treats non-finite as a poll failure (route through the
    consecutive-failure halt path).
    """
    return isinstance(value, int | float) and math.isfinite(value)


def state_file_age_days(path: Path) -> float | None:
    """Compute age in days from persisted last_equity_timestamp.

    Pure function. No side effects, no logging. Returns None ONLY when:
      - file does not exist, OR
      - file unreadable (OSError on read).

    Resolution order for the age value (Stage 2 audit-gate SG1 fix —
    closes fail-closed gate bypass for files with null/missing/unparseable
    timestamps):
      1. Parse JSON; read `last_equity_timestamp`. If valid, use it.
      2. If timestamp missing/null/unparseable BUT file readable, fall back
         to file mtime. Persisted by `_save_state` at every write, so mtime
         tracks last-attempted-persist — operationally a strictly fresher
         lower bound than last_equity_timestamp.
      3. Only return None if BOTH paths fail (file unreadable OR mtime
         unavailable).

    Single source of truth for state-file age. Used by:
      - _load_state (B2/B3 stale-state gate, this module)
      - check_topstep_inactivity_window (Stage 4 pre-session check, future)

    HWM persistence integrity hardening design v3 § 5 sub-bullet
    "Shared age computation" — never reimplement either consumer.
    """
    if not path.exists():
        return None
    # Try the in-file timestamp first (most precise).
    try:
        raw = json.loads(path.read_text())
        ts = raw.get("last_equity_timestamp") if isinstance(raw, dict) else None
        if ts:
            last = datetime.fromisoformat(ts)
            if last.tzinfo is None:
                last = last.replace(tzinfo=UTC)
            delta = datetime.now(UTC) - last
            return delta.total_seconds() / 86400.0
    except (json.JSONDecodeError, ValueError, TypeError):
        # JSON corrupt OR timestamp unparseable — fall through to mtime
        # fallback so a corrupt file with a recent mtime does not bypass
        # the stale gate (closes audit-gate SG1).
        pass
    except OSError:
        return None
    # Fallback: file mtime. Catches null/missing/unparseable timestamp
    # without bypassing the fail-closed gate.
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    delta_seconds = datetime.now(UTC).timestamp() - mtime
    return delta_seconds / 86400.0


@dataclass
class SessionLogEntry:
    date: str
    start_equity: float
    end_equity: float
    peak_intraday: float
    session_dd: float


@dataclass
class HWMState:
    account_id: str
    firm: str
    hwm_dollars: float
    hwm_timestamp: str
    last_equity: float
    last_equity_timestamp: str
    dd_used_dollars: float
    dd_limit_dollars: float
    dd_pct_used: float
    halt_triggered: bool
    halt_timestamp: str | None
    warning_level: str = "CLEAR"  # "CLEAR" | "WARNING_50" | "WARNING_75" | "HALT"
    session_log: list[dict] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return not self.halt_triggered

    @property
    def dd_remaining_dollars(self) -> float:
        return max(0.0, self.dd_limit_dollars - self.dd_used_dollars)


# UNGROUNDED — operational default.
# Rationale: ~5 trading days at 6 sessions/day. Bounds session-log retention
# so persisted JSON does not grow unboundedly. No literature citation.
_MAX_SESSION_LOG = 30

# UNGROUNDED — operational default.
# Rationale: ~30 min visibility window at 10-bar polling cadence. Long enough
# to absorb a transient broker outage; short enough to halt before DD ages out
# of view. No literature citation.
_MAX_CONSECUTIVE_POLL_FAILURES = 3

# UNGROUNDED operational heuristic. Figure borrowed by analogy from TopStep
# inactivity window (topstep_xfa_parameters.txt:349-351), NOT derived.
# Rationale: bot-state freshness, not broker-account activity. Boundary
# direction: >= 30 days raises (file age, not match window). False-positive
# resolution: operator archives or deletes the state file.
_STATE_STALENESS_FAIL_DAYS = 30

# UNGROUNDED — operational suppression floor.
# Rationale: 24h floor blocks restart-cycle noise. Under 24h silent;
# 24h–30 days warn + notify; >= 30 days raises via _STATE_STALENESS_FAIL_DAYS.
_STATE_STALENESS_WARN_DAYS = 1


class AccountHWMTracker:
    """Persistent dollar-based HWM tracker for prop firm trailing DD.

    Supports two trailing modes via dd_type:
    - "eod_trailing" (default): HWM only advances at session close via record_session_end().
      Intraday update_equity() tracks equity for halt detection but does NOT ratchet HWM.
      This matches the supported EOD trailing mechanics used in the active project.
    - "intraday_trailing": HWM advances on every update_equity() call (legacy behavior).

    Supports freeze_at_balance: when HWM reaches this level, it locks permanently.
    Example: freeze_at_balance = starting_balance + max_dd + 100.

    @canonical-source: docs/research-input/topstep/topstep_xfa_parameters.txt:289 —
        "If you break a rule, your Express Funded Account will be liquidated immediately"
    @canonical-source: docs/research-input/topstep/topstep_mll_article.md —
        EOD trailing mechanic verbatim
    @verbatim: see source files for exact wording; this class implements the
        operator-side enforcement of those rules in software.
    @audit-finding: HWM persistence integrity audit 2026-04-25 (UNSUPPORTED-1) —
        the eod_trailing description was previously informal ("matches the supported
        EOD trailing mechanics used in the active project"). Closed by Stage 2 of
        docs/plans/2026-04-25-hwm-persistence-integrity-hardening-design.md (v3 § 5).
    """

    def __init__(
        self,
        account_id: str,
        firm: str,
        dd_limit_dollars: float,
        state_dir: Path | None = None,
        *,
        dd_type: str = "eod_trailing",
        freeze_at_balance: float | None = None,
        daily_loss_limit: float | None = None,
        weekly_loss_limit: float | None = None,
        notify_callback: Callable[[str], None] | None = None,
    ):
        if dd_limit_dollars <= 0:
            raise ValueError(f"dd_limit_dollars must be positive, got {dd_limit_dollars}")
        if dd_type not in ("eod_trailing", "intraday_trailing", "none"):
            raise ValueError(f"dd_type must be eod_trailing/intraday_trailing/none, got {dd_type}")
        if daily_loss_limit is not None and daily_loss_limit <= 0:
            raise ValueError(f"daily_loss_limit must be positive, got {daily_loss_limit}")
        if weekly_loss_limit is not None and weekly_loss_limit <= 0:
            raise ValueError(f"weekly_loss_limit must be positive, got {weekly_loss_limit}")

        self._account_id = str(account_id)
        self._firm = firm
        self._dd_limit = dd_limit_dollars
        self._dd_type = dd_type
        self._freeze_at = freeze_at_balance
        self._hwm_frozen = False
        self._daily_loss_limit = daily_loss_limit
        self._weekly_loss_limit = weekly_loss_limit
        self._notify_callback = notify_callback
        self._state_dir = state_dir or _DEFAULT_STATE_DIR
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._state_dir / f"account_hwm_{self._account_id}.json"

        self._hwm: float = 0.0
        self._hwm_ts: str = ""
        self._last_equity: float = 0.0
        self._last_equity_ts: str = ""
        self._halt: bool = False
        self._halt_ts: str | None = None
        self._halt_reason: str = ""
        self._session_log: list[dict] = []
        self._session_start_equity: float | None = None
        self._session_peak: float = 0.0
        self._consecutive_poll_failures: int = 0

        # Period loss tracking (non-ratcheting, resets at boundaries)
        self._daily_start_equity: float | None = None
        self._daily_start_date: str | None = None  # YYYY-MM-DD Brisbane trading day
        self._weekly_start_equity: float | None = None
        self._weekly_start_date: str | None = None  # Monday YYYY-MM-DD Brisbane

        self._load_state()

    def _state_path(self) -> Path:
        return self._state_file

    def _safe_notify(self, message: str) -> None:
        """Bounded dispatch of an operator-visible message via notify_callback.

        No-op when callback is None. Wraps callback exceptions so a notify
        failure NEVER breaks tracker construction or update_equity flow.
        Per design v3 § 5 AC 14 (synchronous-Telegram startup-latency hazard
        accepted; callback exception MUST NOT propagate).
        """
        if self._notify_callback is None:
            return
        try:
            self._notify_callback(message)
        except Exception as exc:
            log.error("HWM notify_callback dispatch failed: %s — continuing", exc)

    def _load_state(self) -> None:
        if not self._state_file.exists():
            log.info("HWM tracker: no state file for account %s — will init on first equity update", self._account_id)
            return

        # Stale-state gate (B2/B3) — fires BEFORE the JSON load attempt so
        # a 30-day-old state file with valid JSON still raises. Uses the
        # shared state_file_age_days() helper (single source of truth shared
        # with Stage 4's pre-session inactivity check).
        age_days = state_file_age_days(self._state_file)
        if age_days is not None:
            if age_days >= _STATE_STALENESS_FAIL_DAYS:
                msg = (
                    f"STALE_STATE_FAIL: HWM state file age {age_days:.1f} days "
                    f">= {_STATE_STALENESS_FAIL_DAYS} day fail-closed threshold "
                    f"({self._state_file}). Operator must archive (rename to "
                    f"{self._state_file.name}.STALE_<YYYYMMDD>.json) or delete "
                    f"the file before resuming. The 30-day figure is an operational "
                    f"heuristic borrowed by analogy from TopStep's account-trading-"
                    f"inactivity rule, NOT derived from it; see "
                    f"_STATE_STALENESS_FAIL_DAYS comment."
                )
                raise RuntimeError(msg)
            if age_days >= _STATE_STALENESS_WARN_DAYS:
                log.warning(
                    "HWM state file is %.1f days old (account %s) — approaching %d-day fail-closed threshold",
                    age_days,
                    self._account_id,
                    _STATE_STALENESS_FAIL_DAYS,
                )
                self._safe_notify(
                    f"HWM STATE STALE: account {self._account_id} state file "
                    f"is {age_days:.1f} days old (fail-closed at "
                    f"{_STATE_STALENESS_FAIL_DAYS} days)"
                )

        try:
            data = json.loads(self._state_file.read_text())
            self._hwm = float(data.get("hwm_dollars", 0.0))
            self._hwm_ts = data.get("hwm_timestamp", "")
            self._last_equity = float(data.get("last_equity", 0.0))
            self._last_equity_ts = data.get("last_equity_timestamp", "")
            self._halt = bool(data.get("halt_triggered", False))
            self._halt_ts = data.get("halt_timestamp")
            self._halt_reason = data.get("halt_reason", "")
            self._consecutive_poll_failures = int(data.get("consecutive_poll_failures", 0))
            self._hwm_frozen = bool(data.get("hwm_frozen", False))
            self._daily_start_equity = data.get("daily_start_equity")
            self._daily_start_date = data.get("daily_start_date")
            self._weekly_start_equity = data.get("weekly_start_equity")
            self._weekly_start_date = data.get("weekly_start_date")
            self._session_log = data.get("session_log", [])[-_MAX_SESSION_LOG:]
            log.info(
                "HWM tracker loaded: account=%s firm=%s HWM=$%.2f last=$%.2f halt=%s",
                self._account_id,
                self._firm,
                self._hwm,
                self._last_equity,
                self._halt,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            log.error(
                "HWM state file corrupt for account %s: %s — saving backup and reinitialising", self._account_id, e
            )
            corrupt_path = (
                self._state_dir
                / f"account_hwm_{self._account_id}_CORRUPT_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.json"
            )
            shutil.copy2(self._state_file, corrupt_path)
            log.error("Corrupt state backed up to %s", corrupt_path)
            self._safe_notify(
                f"HWM STATE CORRUPT: account {self._account_id} state file "
                f"unreadable ({type(e).__name__}: {e}); backed up to "
                f"{corrupt_path.name}; reinitialising from broker on next poll"
            )
            # Reset to fresh state — will init on first equity update
            self._hwm = 0.0
            self._hwm_ts = ""

    def _advance_hwm(self, equity: float, timestamp: str) -> None:
        """Advance HWM if equity exceeds it. Respects freeze logic."""
        if self._hwm_frozen:
            return
        if equity <= self._hwm:
            return
        old_hwm = self._hwm
        self._hwm = equity
        self._hwm_ts = timestamp
        log.info("NEW_HWM: $%.2f -> $%.2f (+$%.2f)", old_hwm, equity, equity - old_hwm)
        # Check freeze condition
        if self._freeze_at is not None and self._hwm >= self._freeze_at:
            self._hwm_frozen = True
            log.info(
                "HWM FROZEN at $%.2f (freeze_at=$%.2f). Threshold locked at $%.2f permanently.",
                self._hwm,
                self._freeze_at,
                self._hwm - self._dd_limit,
            )

    @staticmethod
    def _brisbane_trading_day(utc_iso: str | None = None) -> str:
        """Compute the Brisbane trading day (YYYY-MM-DD).

        Trading day boundary is 09:00 Brisbane. Times before 09:00
        belong to the PREVIOUS calendar day's trading session.
        """
        if utc_iso:
            dt = datetime.fromisoformat(utc_iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            bris = dt.astimezone(_BRISBANE)
        else:
            bris = datetime.now(_BRISBANE)
        if bris.hour < 9:
            bris = bris - timedelta(days=1)
        return bris.strftime("%Y-%m-%d")

    @staticmethod
    def _brisbane_week_monday(utc_iso: str | None = None) -> str:
        """Compute the Monday date for the current Brisbane trading week.

        Week boundary is Monday 09:00 Brisbane. Before that = previous week.
        """
        if utc_iso:
            dt = datetime.fromisoformat(utc_iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            bris = dt.astimezone(_BRISBANE)
        else:
            bris = datetime.now(_BRISBANE)
        if bris.hour < 9:
            bris = bris - timedelta(days=1)
        # Monday = weekday 0
        monday = bris - timedelta(days=bris.weekday())
        return monday.strftime("%Y-%m-%d")

    def _check_period_resets(self, equity: float) -> None:
        """Reset daily/weekly counters if period boundary crossed."""
        now_iso = datetime.now(UTC).isoformat()
        today = self._brisbane_trading_day(now_iso)
        this_monday = self._brisbane_week_monday(now_iso)

        # Daily reset
        if self._daily_loss_limit is not None:
            if self._daily_start_date is None or self._daily_start_date != today:
                old = self._daily_start_date
                self._daily_start_equity = equity
                self._daily_start_date = today
                # Clear daily halt if it was set
                if self._halt and self._halt_reason == "DAILY_LOSS":
                    self._halt = False
                    self._halt_ts = None
                    self._halt_reason = ""
                    log.info("Daily loss limit RESET: new trading day %s (was %s)", today, old)

        # Weekly reset
        if self._weekly_loss_limit is not None:
            if self._weekly_start_date is None or self._weekly_start_date != this_monday:
                old = self._weekly_start_date
                self._weekly_start_equity = equity
                self._weekly_start_date = this_monday
                # Clear weekly halt if it was set
                if self._halt and self._halt_reason == "WEEKLY_LOSS":
                    self._halt = False
                    self._halt_ts = None
                    self._halt_reason = ""
                    log.info("Weekly loss limit RESET: new week %s (was %s)", this_monday, old)

    def _save_state(self) -> None:
        data = {
            "account_id": self._account_id,
            "firm": self._firm,
            "hwm_dollars": round(self._hwm, 2),
            "hwm_timestamp": self._hwm_ts,
            "last_equity": round(self._last_equity, 2),
            "last_equity_timestamp": self._last_equity_ts,
            "dd_used_dollars": round(self._dd_used(), 2),
            "dd_limit_dollars": self._dd_limit,
            "dd_pct_used": round(self._dd_pct(), 4),
            "halt_triggered": self._halt,
            "halt_timestamp": self._halt_ts,
            "halt_reason": self._halt_reason,
            "consecutive_poll_failures": self._consecutive_poll_failures,
            "hwm_frozen": self._hwm_frozen,
            "dd_type": self._dd_type,
            "daily_start_equity": self._daily_start_equity,
            "daily_start_date": self._daily_start_date,
            "weekly_start_equity": self._weekly_start_equity,
            "weekly_start_date": self._weekly_start_date,
            "session_log": self._session_log[-_MAX_SESSION_LOG:],
        }
        tmp = self._state_file.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(self._state_file)
        except OSError as exc:
            # B6: persist-IO failure dispatch. Narrow OSError catch (NOT
            # broad Exception) so JSON-encoding bugs still surface as
            # programming errors. Notify-then-reraise preserves existing
            # caller semantics (callers see the OSError); the dispatch is
            # the only new behavior. Distinct from the broker-poll failure
            # path: this does NOT increment _consecutive_poll_failures.
            log.error(
                "HWM state persist failed for account %s: %s — re-raising",
                self._account_id,
                exc,
            )
            self._safe_notify(
                f"STATE_PERSIST_FAIL: HWM state file write failed for account "
                f"{self._account_id} ({self._state_file}): {exc!r}"
            )
            raise

    def _dd_used(self) -> float:
        if self._hwm <= 0:
            return 0.0
        return max(0.0, self._hwm - self._last_equity)

    def _dd_pct(self) -> float:
        if self._dd_limit <= 0:
            return 0.0
        return self._dd_used() / self._dd_limit

    def update_equity(self, current_equity: float | None) -> HWMState:
        """Update with current broker equity. Returns current state."""
        # SG4 fix (Stage 2 audit-gate): treat NaN/Inf equity as a poll
        # failure. Without this, NaN propagates into _last_equity, then
        # _dd_used()/_dd_pct() return NaN, and every halt comparison
        # (>=, >) silently evaluates False — the kill switch never fires
        # on a degraded broker that returns NaN. Route through the
        # consecutive-failure path so the 3-strike halt mechanism
        # actually engages.
        if current_equity is not None and not _is_finite_equity(current_equity):
            log.error(
                "HWM: equity poll returned non-finite value %r — treating as poll failure",
                current_equity,
            )
            current_equity = None
        if current_equity is None:
            self._consecutive_poll_failures += 1
            log.warning(
                "HWM: equity poll returned None (failure %d/%d)",
                self._consecutive_poll_failures,
                _MAX_CONSECUTIVE_POLL_FAILURES,
            )
            if self._consecutive_poll_failures >= _MAX_CONSECUTIVE_POLL_FAILURES:
                self._halt = True
                self._halt_reason = "POLL_FAILURE"
                self._halt_ts = datetime.now(UTC).isoformat()
                log.critical(
                    "HWM HALT: %d consecutive equity poll failures — halting for safety",
                    self._consecutive_poll_failures,
                )
            self._save_state()
            return self._build_state()

        prior_failures = self._consecutive_poll_failures
        self._consecutive_poll_failures = 0
        if prior_failures > 0:
            # B5: poll-failure recovery dispatch. Steady-state successes
            # (prior_failures == 0) stay silent — mutation guard against spam.
            # SG3 fix (Stage 2 audit-gate): if halt fired during the failure
            # streak (POLL_FAILURE), append a halt-still-active qualifier so
            # the operator does not read "RECOVERY" as "safe to resume". The
            # poll did recover; the halt persists until reset_halt() is called.
            halt_qualifier = ""
            if self._halt and self._halt_reason == "POLL_FAILURE":
                halt_qualifier = (
                    " — account REMAINS HALTED (POLL_FAILURE). Manual reset_halt() required before trading resumes."
                )
            log.info(
                "HWM: equity poll RECOVERED after %d consecutive failure(s)%s",
                prior_failures,
                " (halt still active)" if halt_qualifier else "",
            )
            self._safe_notify(
                f"HWM POLL RECOVERY: equity poll succeeded after "
                f"{prior_failures} consecutive failure(s) (account "
                f"{self._account_id}){halt_qualifier}"
            )
        now = datetime.now(UTC).isoformat()
        self._last_equity = current_equity
        self._last_equity_ts = now

        # First equity reading — initialise HWM from broker
        if self._hwm <= 0:
            self._hwm = current_equity
            self._hwm_ts = now
            log.info("HWM initialised from broker: $%.2f", current_equity)

        # Check and reset daily/weekly periods before halt checks
        self._check_period_resets(current_equity)

        # Advance HWM based on dd_type:
        # - eod_trailing: HWM only advances at session close (record_session_end)
        # - intraday_trailing / none: HWM advances on every equity update
        if self._dd_type != "eod_trailing":
            self._advance_hwm(current_equity, now)

        # Track intraday peak for session log
        if current_equity > self._session_peak:
            self._session_peak = current_equity

        # Check halt conditions (trailing DD, then daily, then weekly)
        if not self._halt:
            if self._dd_used() >= self._dd_limit:
                self._halt = True
                self._halt_ts = now
                self._halt_reason = "DD_TRAILING"
                log.critical(
                    "HWM HALT TRIGGERED: DD $%.2f >= limit $%.2f (HWM=$%.2f, equity=$%.2f)",
                    self._dd_used(),
                    self._dd_limit,
                    self._hwm,
                    current_equity,
                )
            elif self._daily_loss_limit is not None and self._daily_start_equity is not None:
                daily_loss = self._daily_start_equity - current_equity
                if daily_loss >= self._daily_loss_limit:
                    self._halt = True
                    self._halt_ts = now
                    self._halt_reason = "DAILY_LOSS"
                    log.critical(
                        "DAILY LOSS HALT: loss $%.2f >= limit $%.2f (start=$%.2f, equity=$%.2f)",
                        daily_loss,
                        self._daily_loss_limit,
                        self._daily_start_equity,
                        current_equity,
                    )
            elif self._weekly_loss_limit is not None and self._weekly_start_equity is not None:
                weekly_loss = self._weekly_start_equity - current_equity
                if weekly_loss >= self._weekly_loss_limit:
                    self._halt = True
                    self._halt_ts = now
                    self._halt_reason = "WEEKLY_LOSS"
                    log.critical(
                        "WEEKLY LOSS HALT: loss $%.2f >= limit $%.2f (start=$%.2f, equity=$%.2f)",
                        weekly_loss,
                        self._weekly_loss_limit,
                        self._weekly_start_equity,
                        current_equity,
                    )

        self._save_state()
        return self._build_state()

    def check_halt(self) -> tuple[bool, str]:
        """Check if trading should be halted. Returns (should_halt, reason).

        Halt reasons: DD_TRAILING, DAILY_LOSS, WEEKLY_LOSS, POLL_FAILURE, or empty.
        """
        if self._halt:
            if self._halt_reason == "POLL_FAILURE":
                return True, (
                    f"POLL_FAILURE: {self._consecutive_poll_failures} consecutive equity poll failures "
                    f"— halted for safety (last equity=${self._last_equity:.2f})"
                )
            if self._halt_reason == "DAILY_LOSS":
                daily_loss = (self._daily_start_equity or 0) - self._last_equity
                return True, (
                    f"DAILY_LOSS: loss ${daily_loss:.2f} >= limit ${self._daily_loss_limit:.2f} "
                    f"(day started ${self._daily_start_equity:.2f}, equity=${self._last_equity:.2f})"
                )
            if self._halt_reason == "WEEKLY_LOSS":
                weekly_loss = (self._weekly_start_equity or 0) - self._last_equity
                return True, (
                    f"WEEKLY_LOSS: loss ${weekly_loss:.2f} >= limit ${self._weekly_loss_limit:.2f} "
                    f"(week started ${self._weekly_start_equity:.2f}, equity=${self._last_equity:.2f})"
                )
            # DD_TRAILING or legacy halt (no reason stored)
            return True, (
                f"DD_TRAILING: DD ${self._dd_used():.2f} >= limit ${self._dd_limit:.2f} "
                f"(HWM=${self._hwm:.2f} on {self._hwm_ts[:10]}, equity=${self._last_equity:.2f})"
            )

        pct = self._dd_pct()
        remaining = self._dd_limit - self._dd_used()
        if pct >= 0.75:
            return False, (
                f"HWM_WARNING_75: DD ${self._dd_used():.2f} = {pct:.0%} of ${self._dd_limit:.2f} limit "
                f"(${remaining:.2f} remaining)"
            )

        if pct >= 0.50:
            return False, (
                f"HWM_WARNING_50: DD ${self._dd_used():.2f} = {pct:.0%} of ${self._dd_limit:.2f} limit "
                f"(${remaining:.2f} remaining)"
            )

        return False, (f"HWM_OK: DD ${self._dd_used():.2f} = {pct:.0%} of ${self._dd_limit:.2f} limit")

    def record_session_start(self, equity: float) -> None:
        """Called at session open. Does NOT reset HWM."""
        self._session_start_equity = equity
        self._session_peak = equity
        log.info("HWM session start: equity=$%.2f, HWM=$%.2f, DD=$%.2f", equity, self._hwm, self._dd_used())

    def record_session_end(self, equity: float) -> None:
        """Called at session close. Appends to rolling session log.

        For eod_trailing firms, this is where HWM advances — NOT during
        intraday update_equity() calls. This matches the firm mechanic:
        threshold recalculated from highest EOD close, not intraday peaks.
        """
        self._last_equity = equity
        # EOD commit: advance HWM now (suppressed during intraday polls)
        if self._dd_type == "eod_trailing":
            self._advance_hwm(equity, datetime.now(UTC).isoformat())
        entry = {
            "date": datetime.now(UTC).isoformat(),
            "start_equity": round(self._session_start_equity or equity, 2),
            "end_equity": round(equity, 2),
            "peak_intraday": round(self._session_peak, 2),
            "session_dd": round((self._session_start_equity or equity) - equity, 2),
        }
        self._session_log.append(entry)
        self._session_log = self._session_log[-_MAX_SESSION_LOG:]
        self._save_state()
        log.info(
            "HWM session end: start=$%.2f end=$%.2f peak=$%.2f dd=$%.2f",
            entry["start_equity"],
            equity,
            self._session_peak,
            entry["session_dd"],
        )

    def reset_halt(self, operator_note: str = "") -> None:
        """Manual halt reset. Requires explicit call — never automatic."""
        old_halt = self._halt
        self._halt = False
        self._halt_ts = None
        self._halt_reason = ""
        self._consecutive_poll_failures = 0
        self._save_state()
        log.warning(
            "HWM halt MANUALLY RESET: was_halted=%s, equity=$%.2f, HWM=$%.2f, note='%s'",
            old_halt,
            self._last_equity,
            self._hwm,
            operator_note,
        )

    def get_status_summary(self) -> dict:
        """For weekly_review and pre_session_check integration."""
        summary = {
            "account_id": self._account_id,
            "firm": self._firm,
            "hwm_dollars": round(self._hwm, 2),
            "hwm_date": self._hwm_ts[:10] if self._hwm_ts else "never",
            "last_equity": round(self._last_equity, 2),
            "dd_used_dollars": round(self._dd_used(), 2),
            "dd_limit_dollars": self._dd_limit,
            "dd_pct_used": round(self._dd_pct() * 100, 1),
            "dd_remaining_dollars": round(self._dd_limit - self._dd_used(), 2),
            "hwm_frozen": self._hwm_frozen,
            "dd_type": self._dd_type,
            "halt_triggered": self._halt,
            "halt_reason": self._halt_reason,
            "halt_timestamp": self._halt_ts,
            "sessions_tracked": len(self._session_log),
        }
        if self._daily_loss_limit is not None:
            daily_loss = (self._daily_start_equity or 0) - self._last_equity if self._daily_start_equity else 0
            summary["daily_loss_limit"] = self._daily_loss_limit
            summary["daily_loss_current"] = round(max(0, daily_loss), 2)
            summary["daily_start_date"] = self._daily_start_date
        if self._weekly_loss_limit is not None:
            weekly_loss = (self._weekly_start_equity or 0) - self._last_equity if self._weekly_start_equity else 0
            summary["weekly_loss_limit"] = self._weekly_loss_limit
            summary["weekly_loss_current"] = round(max(0, weekly_loss), 2)
            summary["weekly_start_date"] = self._weekly_start_date
        return summary

    def _build_state(self) -> HWMState:
        pct = self._dd_pct()
        if self._halt:
            wl = "HALT"
        elif pct >= 0.75:
            wl = "WARNING_75"
        elif pct >= 0.50:
            wl = "WARNING_50"
        else:
            wl = "CLEAR"

        return HWMState(
            account_id=self._account_id,
            firm=self._firm,
            hwm_dollars=round(self._hwm, 2),
            hwm_timestamp=self._hwm_ts,
            last_equity=round(self._last_equity, 2),
            last_equity_timestamp=self._last_equity_ts,
            dd_used_dollars=round(self._dd_used(), 2),
            dd_limit_dollars=self._dd_limit,
            dd_pct_used=round(self._dd_pct(), 4),
            halt_triggered=self._halt,
            halt_timestamp=self._halt_ts,
            warning_level=wl,
            session_log=self._session_log[-_MAX_SESSION_LOG:],
        )
