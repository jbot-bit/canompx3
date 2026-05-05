"""Enhanced circuit breaker for broker API resilience with exponential backoff.

Features:
- Consecutive failure tracking with configurable threshold
- Exponential backoff for recovery timeout
- Graceful degradation with probe requests
- State change notifications via callbacks
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger(__name__)


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    initial_recovery_timeout: float = 30.0  # seconds
    max_recovery_timeout: float = 300.0  # 5 minutes max
    backoff_factor: float = 2.0

    consecutive_failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _current_timeout: float = field(default=0.0, init=False)
    _state_change_callbacks: list[Callable[[str, str], None]] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._current_timeout = self.initial_recovery_timeout

    @property
    def is_open(self) -> bool:
        return self.consecutive_failures >= self.failure_threshold

    @property
    def state(self) -> str:
        if self.is_open:
            return "open"
        elif self.consecutive_failures > 0:
            return "half-open"
        return "closed"

    def add_state_change_callback(self, callback: Callable[[str, str], None]) -> None:
        """Add callback for state changes: callback(old_state, new_state)"""
        self._state_change_callbacks.append(callback)

    def _notify_state_change(self, old_state: str, new_state: str) -> None:
        for callback in self._state_change_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                log.error("Circuit breaker state change callback failed: %s", e)

    def should_allow_request(self) -> bool:
        if not self.is_open:
            return True
        if self._opened_at is None:
            return False
        elapsed = time.monotonic() - self._opened_at
        if elapsed >= self._current_timeout:
            log.info("Circuit breaker: allowing probe request after %.1fs (timeout: %.1fs)", 
                    elapsed, self._current_timeout)
            return True
        return False

    def record_success(self) -> None:
        old_state = self.state
        if self.consecutive_failures > 0:
            log.info("Circuit breaker: reset after %d failures", self.consecutive_failures)
        self.consecutive_failures = 0
        self._opened_at = None
        self._current_timeout = self.initial_recovery_timeout
        if old_state != self.state:
            self._notify_state_change(old_state, self.state)

    def record_failure(self) -> None:
        old_state = self.state
        self.consecutive_failures += 1
        
        if self.is_open and self._opened_at is None:
            # First time opening - set initial timeout
            self._opened_at = time.monotonic()
            self._current_timeout = min(
                self._current_timeout * self.backoff_factor, 
                self.max_recovery_timeout
            )
            log.warning(
                "Circuit breaker OPEN after %d consecutive failures. Blocking requests for %.1fs.",
                self.consecutive_failures,
                self._current_timeout,
            )
            if old_state != self.state:
                self._notify_state_change(old_state, self.state)
        elif self.is_open and self._opened_at is not None:
            # Subsequent failure while open - extend timeout with backoff
            self._current_timeout = min(
                self._current_timeout * self.backoff_factor,
                self.max_recovery_timeout
            )
            self._opened_at = time.monotonic()  # Reset timer
            log.warning(
                "Circuit breaker: probe failed (%d consecutive). Extending block to %.1fs.",
                self.consecutive_failures,
                self._current_timeout,
            )
