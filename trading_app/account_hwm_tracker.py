"""
Persistent dollar-based High Water Mark tracker for prop firm trailing drawdown.

Prop firms (Apex, TopStep, Tradeify) use EOD trailing drawdown from the account's
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
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_STATE_DIR = Path(__file__).resolve().parent.parent / "data" / "state"


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


_WARN_THRESHOLD = 0.75
_MAX_SESSION_LOG = 30
_MAX_CONSECUTIVE_POLL_FAILURES = 3


class AccountHWMTracker:
    """Persistent dollar-based HWM tracker for prop firm trailing DD."""

    def __init__(
        self,
        account_id: str,
        firm: str,
        dd_limit_dollars: float,
        state_dir: Path | None = None,
    ):
        if dd_limit_dollars <= 0:
            raise ValueError(f"dd_limit_dollars must be positive, got {dd_limit_dollars}")

        self._account_id = str(account_id)
        self._firm = firm
        self._dd_limit = dd_limit_dollars
        self._state_dir = state_dir or _DEFAULT_STATE_DIR
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._state_dir / f"account_hwm_{self._account_id}.json"

        self._hwm: float = 0.0
        self._hwm_ts: str = ""
        self._last_equity: float = 0.0
        self._last_equity_ts: str = ""
        self._halt: bool = False
        self._halt_ts: str | None = None
        self._session_log: list[dict] = []
        self._session_start_equity: float | None = None
        self._session_peak: float = 0.0
        self._consecutive_poll_failures: int = 0

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

        # New high water mark
        if current_equity > self._hwm:
            old_hwm = self._hwm
            self._hwm = current_equity
            self._hwm_ts = now
            log.info("NEW_HWM: $%.2f -> $%.2f (+$%.2f)", old_hwm, current_equity, current_equity - old_hwm)

        # Track intraday peak for session log
        if current_equity > self._session_peak:
            self._session_peak = current_equity

        # Check halt condition
        if not self._halt and self._dd_used() >= self._dd_limit:
            self._halt = True
            self._halt_ts = now
            log.critical(
                "HWM HALT TRIGGERED: DD $%.2f >= limit $%.2f (HWM=$%.2f, equity=$%.2f)",
                self._dd_used(),
                self._dd_limit,
                self._hwm,
                current_equity,
            )

        self._save_state()
        return self._build_state()

    def check_halt(self) -> tuple[bool, str]:
        """Check if trading should be halted. Returns (should_halt, reason)."""
        if self._halt:
            return True, (
                f"HWM_HALT: DD ${self._dd_used():.2f} >= limit ${self._dd_limit:.2f} "
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
        """Called at session close. Appends to rolling session log."""
        self._last_equity = equity
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
        return {
            "account_id": self._account_id,
            "firm": self._firm,
            "hwm_dollars": round(self._hwm, 2),
            "hwm_date": self._hwm_ts[:10] if self._hwm_ts else "never",
            "last_equity": round(self._last_equity, 2),
            "dd_used_dollars": round(self._dd_used(), 2),
            "dd_limit_dollars": self._dd_limit,
            "dd_pct_used": round(self._dd_pct() * 100, 1),
            "dd_remaining_dollars": round(self._dd_limit - self._dd_used(), 2),
            "halt_triggered": self._halt,
            "halt_timestamp": self._halt_ts,
            "sessions_tracked": len(self._session_log),
        }

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
