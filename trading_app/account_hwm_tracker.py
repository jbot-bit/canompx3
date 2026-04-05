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
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

_DEFAULT_STATE_DIR = Path(__file__).resolve().parent.parent / "data" / "state"
_BRISBANE = ZoneInfo("Australia/Brisbane")


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


_MAX_SESSION_LOG = 30
_MAX_CONSECUTIVE_POLL_FAILURES = 3


class AccountHWMTracker:
    """Persistent dollar-based HWM tracker for prop firm trailing DD.

    Supports two trailing modes via dd_type:
    - "eod_trailing" (default): HWM only advances at session close via record_session_end().
      Intraday update_equity() tracks equity for halt detection but does NOT ratchet HWM.
      This matches the supported EOD trailing mechanics used in the active project.
    - "intraday_trailing": HWM advances on every update_equity() call (legacy behavior).

    Supports freeze_at_balance: when HWM reaches this level, it locks permanently.
    Example: freeze_at_balance = starting_balance + max_dd + 100.
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

    def _load_state(self) -> None:
        if not self._state_file.exists():
            log.info("HWM tracker: no state file for account %s — will init on first equity update", self._account_id)
            return

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
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self._state_file)

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

        self._consecutive_poll_failures = 0
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
