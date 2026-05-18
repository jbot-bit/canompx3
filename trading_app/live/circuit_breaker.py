"""Circuit breaker for broker API resilience.

Opens after N consecutive failures, blocks requests for M seconds,
then allows one probe request. Resets on success.
"""

import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0  # seconds

    consecutive_failures: int = field(default=0, init=False)
    last_error_class: str | None = field(default=None, init=False)
    _opened_at: float | None = field(default=None, init=False)

    @property
    def is_open(self) -> bool:
        return self.consecutive_failures >= self.failure_threshold

    def should_allow_request(self) -> bool:
        if not self.is_open:
            return True
        if self._opened_at is None:
            return False
        elapsed = time.monotonic() - self._opened_at
        if elapsed >= self.recovery_timeout:
            log.info("Circuit breaker: allowing probe request after %.1fs", elapsed)
            return True
        return False

    def record_success(self) -> None:
        if self.consecutive_failures > 0:
            log.info("Circuit breaker: reset after %d failures", self.consecutive_failures)
        self.consecutive_failures = 0
        self.last_error_class = None
        self._opened_at = None

    def record_failure(self, error_class: str | None = None) -> None:
        """Record one failure attempt.

        `error_class` is the BrokerHTTPClient classification letter (A/B/C/D/E/F/G
        or "perm"/"X") when called by the HTTP client failure_hook (Stage 4 wiring).
        Existing orchestrator-level callers (exit-submit, entry-submit retry
        exhaustion) pass no arg and preserve their original semantics —
        ``last_error_class`` only updates when a non-None class is passed,
        so an orchestrator-level failure does not overwrite the most recent
        HTTP class that produced it.
        """
        self.consecutive_failures += 1
        if error_class is not None:
            self.last_error_class = error_class
        if self.is_open and self._opened_at is None:
            self._opened_at = time.monotonic()
            log.warning(
                "Circuit breaker OPEN after %d consecutive failures (last_class=%s). "
                "Blocking requests for %.0fs.",
                self.consecutive_failures,
                self.last_error_class,
                self.recovery_timeout,
            )
        elif self.is_open and self._opened_at is not None:
            # Failed probe — restart the recovery timer so we don't flood
            self._opened_at = time.monotonic()
            log.warning(
                "Circuit breaker: probe failed (%d consecutive, last_class=%s). "
                "Re-blocking for %.0fs.",
                self.consecutive_failures,
                self.last_error_class,
                self.recovery_timeout,
            )
