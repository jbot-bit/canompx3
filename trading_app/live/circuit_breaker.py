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
        self._opened_at = None

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.is_open and self._opened_at is None:
            self._opened_at = time.monotonic()
            log.warning(
                "Circuit breaker OPEN after %d consecutive failures. Blocking requests for %.0fs.",
                self.consecutive_failures,
                self.recovery_timeout,
            )
