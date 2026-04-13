"""Crash-recoverable persistence for critical session safety flags.

Flags persisted:
  - kill_switch_fired: emergency flatten triggered (orphaned positions)
  - close_time_forced: EOD force-flatten initiated
  - blocked_strategies: strategy IDs blocked due to orphan/stuck-exit containment
  - shadow_failures: CopyOrderRouter divergence state (cross-account hedging)
  - daily_pnl_r: intraday P&L in R-units (for daily loss circuit breaker recovery)

Pattern: save-on-mutate (atomic write via tmp+replace), load-on-init.
Follows the same convention as AccountHWMTracker in data/state/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_STATE_DIR = Path(__file__).resolve().parents[2] / "data" / "state"


class SessionSafetyState:
    """Persist critical safety flags so they survive process crashes."""

    def __init__(self, portfolio_name: str, instrument: str) -> None:
        self._portfolio_name = portfolio_name
        self._instrument = instrument
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        self._state_file = _STATE_DIR / f"session_safety_{portfolio_name}_{instrument}.json"

        # Defaults — overridden by _load_state if file exists
        self.kill_switch_fired: bool = False
        self.close_time_forced: bool = False
        self.blocked_strategies: dict[str, str] = {}  # strategy_id → reason
        self.shadow_failures: dict[str, str] = {}  # account_id → failure description
        self.daily_pnl_r: float = 0.0  # intraday P&L for daily loss circuit breaker
        self.trading_day: str = ""  # ISO date — daily_pnl_r only valid if matches current day
        self.cooldown_until: str = ""  # ISO datetime — mandatory pause after equity halt (self-funded)

        self._load_state()

    def _load_state(self) -> None:
        if not self._state_file.exists():
            return
        try:
            data = json.loads(self._state_file.read_text())
            self.kill_switch_fired = bool(data.get("kill_switch_fired", False))
            self.close_time_forced = bool(data.get("close_time_forced", False))
            self.blocked_strategies = dict(data.get("blocked_strategies", {}))
            self.shadow_failures = dict(data.get("shadow_failures", {}))
            self.daily_pnl_r = float(data.get("daily_pnl_r", 0.0))
            self.trading_day = str(data.get("trading_day", ""))
            self.cooldown_until = str(data.get("cooldown_until", ""))
            if self.kill_switch_fired or self.blocked_strategies or self.shadow_failures:
                log.critical(
                    "CRASH RECOVERY: loaded safety state from %s — "
                    "kill_switch=%s, blocked=%d strategies, shadow_failures=%d",
                    self._state_file.name,
                    self.kill_switch_fired,
                    len(self.blocked_strategies),
                    len(self.shadow_failures),
                )
        except Exception:
            log.exception("Failed to load safety state from %s — treating as clean start", self._state_file)

    def save(self) -> None:
        """Atomic write of current state to disk."""
        data = {
            "portfolio": self._portfolio_name,
            "instrument": self._instrument,
            "kill_switch_fired": self.kill_switch_fired,
            "close_time_forced": self.close_time_forced,
            "blocked_strategies": self.blocked_strategies,
            "shadow_failures": self.shadow_failures,
            "daily_pnl_r": round(self.daily_pnl_r, 4),
            "trading_day": self.trading_day,
            "cooldown_until": self.cooldown_until,
        }
        tmp = self._state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self._state_file)

    def clear(self) -> None:
        """Remove state file on clean session end."""
        self._state_file.unlink(missing_ok=True)
        tmp = self._state_file.with_suffix(".tmp")
        tmp.unlink(missing_ok=True)
