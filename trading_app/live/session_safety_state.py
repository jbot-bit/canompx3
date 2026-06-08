"""Crash-recoverable persistence for critical session safety flags.

Flags persisted:
  - kill_switch_fired: emergency flatten triggered (orphaned positions)
  - close_time_forced: EOD force-flatten initiated
  - blocked_strategies: strategy IDs blocked due to orphan/stuck-exit containment
      NOTE: Only NON-derivable blocks persist here. Lifecycle-sourced blocks
      (pause_strategy_id, SR-ALARM reviews) are re-derived at each session
      start by `_load_paused_lane_blocks` in session_orchestrator and should
      be added via `_block_strategy(..., persist=False)`. Persisting those
      would cause stale blocks to survive after the underlying review
      changes (bug fixed 2026-04-14).
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
        self.daily_pnl_dollars: float = 0.0  # intraday realized $ for dollar daily-loss breaker
        self.trading_day: str = ""  # ISO date — daily_pnl_r/dollars only valid if matches current day
        self.cooldown_until: str = ""  # ISO datetime — mandatory pause after equity halt (self-funded)
        # R3: last time a feed connection was stable for >= ORCHESTRATOR_STABLE_RUN_SECS.
        # Used by run() to reset the reconnect counter so 24h operation isn't halted by
        # startup-era flaps. Empty string = no stable connection recorded yet.
        self.last_connected_at: str = ""  # ISO datetime UTC

        self._load_state()

    # Legacy persisted-block reason patterns that are now re-derived from
    # lifecycle state every session and must NOT survive across restarts.
    # Added 2026-04-14 as a one-time migration for files written under the
    # pre-fix persist-everything model.
    _LEGACY_LIFECYCLE_BLOCK_PATTERNS = (
        "Criterion 12 SR ALARM",
        "Criterion 11 regime",
        "Paused pending manual review",
    )

    def _load_state(self) -> None:
        if not self._state_file.exists():
            return
        try:
            data = json.loads(self._state_file.read_text())
            self.kill_switch_fired = bool(data.get("kill_switch_fired", False))
            self.close_time_forced = bool(data.get("close_time_forced", False))

            raw_blocks = dict(data.get("blocked_strategies", {}))
            # Migrate: drop lifecycle-sourced blocks that are now re-derived
            # at session start. Only real crash-recovery blocks (orphan
            # positions, stuck exits) should persist here.
            legacy = {
                sid: reason
                for sid, reason in raw_blocks.items()
                if any(pat in str(reason) for pat in self._LEGACY_LIFECYCLE_BLOCK_PATTERNS)
            }
            self.blocked_strategies = {sid: reason for sid, reason in raw_blocks.items() if sid not in legacy}
            if legacy:
                log.info(
                    "Dropped %d legacy lifecycle-sourced block(s) from persisted state "
                    "(now re-derived from lifecycle registry at startup): %s",
                    len(legacy),
                    sorted(legacy.keys()),
                )

            self.shadow_failures = dict(data.get("shadow_failures", {}))
            self.daily_pnl_r = float(data.get("daily_pnl_r", 0.0))
            self.daily_pnl_dollars = float(data.get("daily_pnl_dollars", 0.0))
            self.trading_day = str(data.get("trading_day", ""))
            self.cooldown_until = str(data.get("cooldown_until", ""))
            self.last_connected_at = str(data.get("last_connected_at", ""))

            # Persist the cleaned state so future loads skip the migration.
            if legacy:
                self.save()

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
            "daily_pnl_dollars": round(self.daily_pnl_dollars, 2),
            "trading_day": self.trading_day,
            "cooldown_until": self.cooldown_until,
            "last_connected_at": self.last_connected_at,
        }
        tmp = self._state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self._state_file)

    def expire_stale_kill_switch(self, today: str) -> bool:
        """Auto-expire a kill switch left over from a prior, now-closed day.

        A kill switch means "we left a position in a dangerous state." That
        concern is real only for the SAME trading day: once markets close and
        reopen, EOD flatten has resolved any open position, so a prior-day kill
        switch has no bearing on today. This mirrors the daily-P&L day-gate in
        session_orchestrator.py (daily_pnl is only restored when trading_day
        matches today) — the kill switch was the one halt-state that was
        restored unconditionally, silently halting the next day's session
        (n=1 repro 2026-06-09: a 2026-06-08 kill switch blocked the 06-09 launch).

        Fail-closed: if `trading_day` is empty/unknown we CANNOT prove the flag
        is stale, so we preserve it. A same-day kill switch (crash-restart into
        a still-live position) is also preserved — that is the C1-race
        protection and must not be cleared.

        Returns True iff a stale kill switch was expired (and persisted).
        """
        if not self.kill_switch_fired:
            return False
        if not self.trading_day or self.trading_day == today:
            return False  # same-day or unknown — preserve (fail-closed)
        log.critical(
            "STALE KILL SWITCH EXPIRED: kill_switch fired on %s, today is %s — "
            "auto-clearing (prior-day flag; EOD flatten resolved any open position). "
            "Cleared %d persisted block(s) for the same reason.",
            self.trading_day,
            today,
            len(self.blocked_strategies),
        )
        self.kill_switch_fired = False
        # Clear blocked_strategies too: the only blocks persisted here are
        # position-state concerns (orphan / stuck-exit, persist=True in
        # session_orchestrator._block_strategy) — resolved by the same EOD
        # flatten that makes the kill switch stale. Lifecycle blocks are never
        # persisted (persist=False) and are re-derived at startup regardless.
        self.blocked_strategies = {}
        # shadow_failures is INTENTIONALLY PRESERVED across the day boundary.
        # It is cross-account CopyOrderRouter divergence state, not position
        # state — a day rollover does not prove the divergence is resolved, so
        # fail-closed and keep it (it does not gate order submission; it is
        # surfaced for operator attention). Do not clear it here.
        self.save()
        return True

    def clear(self) -> None:
        """Remove state file on clean session end."""
        self._state_file.unlink(missing_ok=True)
        tmp = self._state_file.with_suffix(".tmp")
        tmp.unlink(missing_ok=True)


def _preflight_report(portfolio_name: str, instrument: str, today: str) -> int:
    """Read-only launcher preflight: report persisted safety state vs `today`.

    Does NOT mutate state — the orchestrator owns expiry via
    expire_stale_kill_switch at startup. This only tells the operator what will
    happen so a halt is never a silent mystery. Returns:
      0  clean (no kill switch, will trade)
      0  stale kill switch (orchestrator will AUTO-EXPIRE — informational)
      2  SAME-DAY kill switch (will HALT today — operator action needed)
    Fail-open: any error prints a note and returns 0 (never blocks a launch).
    """
    try:
        state = SessionSafetyState(portfolio_name, instrument)
    except Exception as exc:
        # Fail-open: a diagnostic preflight must NEVER block a launch. But the
        # exception is RECORDED (not swallowed) per institutional-rigor §6 —
        # documented fail-open + logged cause.
        log.exception("safety-preflight could not read safety state — continuing (fail-open)")
        print(f"  [safety-preflight] could not read safety state ({exc}) — continuing (fail-open)")
        return 0
    if not state.kill_switch_fired:
        print("  [safety-preflight] OK — no kill switch, clean start")
        return 0
    if state.trading_day and state.trading_day != today:
        print(
            f"  [safety-preflight] STALE kill switch from {state.trading_day} "
            f"(today {today}) — will AUTO-EXPIRE on launch. No action needed."
        )
        return 0
    print(
        f"  [safety-preflight] !! SAME-DAY kill switch (fired {state.trading_day or 'unknown'}) "
        f"is ACTIVE and will HALT trading today.\n"
        f"     This usually means a real halt earlier today (daily-loss / orphan / DD breach).\n"
        f"     If you are SURE it is safe, clear it from the dashboard or delete:\n"
        f"     {state._state_file}"
    )
    return 2


def _canonical_trading_day_today() -> str:
    """Current trading day (ISO) using the canonical Brisbane 09:00 boundary —
    identical rule to SessionOrchestrator.__init__ so the launcher preflight
    and the orchestrator agree on "today" (no re-encoded drift)."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    bris_now = datetime.now(ZoneInfo("Australia/Brisbane"))
    day = (bris_now - timedelta(days=1)).date() if bris_now.hour < 9 else bris_now.date()
    return str(day)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Session safety state launcher preflight")
    ap.add_argument("--profile", required=True, help="profile_id (e.g. topstep_50k_mnq_auto)")
    ap.add_argument("--instrument", required=True)
    ap.add_argument("--today", default=None, help="ISO date; defaults to canonical Brisbane trading day")
    _args = ap.parse_args()
    _today = _args.today or _canonical_trading_day_today()
    # SessionSafetyState is keyed on portfolio.name, which build_profile_portfolio
    # sets to f"profile_{profile_id}" (portfolio.py:953). Apply the same prefix so
    # the preflight reads the SAME file the orchestrator wrote — accept either the
    # raw profile_id or an already-prefixed name (idempotent).
    _portfolio_name = _args.profile if _args.profile.startswith("profile_") else f"profile_{_args.profile}"
    raise SystemExit(_preflight_report(_portfolio_name, _args.instrument, _today))
