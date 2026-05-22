"""Tests for broker API circuit breaker."""

import time

from trading_app.live.circuit_breaker import CircuitBreaker


def test_circuit_opens_after_consecutive_failures():
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
    for _ in range(3):
        breaker.record_failure()
    assert breaker.is_open is True


def test_circuit_closed_below_threshold():
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_open is False


def test_circuit_resets_on_success():
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    assert breaker.is_open is False
    assert breaker.consecutive_failures == 0


def test_circuit_allows_probe_after_timeout():
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_open is True
    assert breaker.should_allow_request() is False  # not yet

    time.sleep(0.15)
    assert breaker.should_allow_request() is True  # probe allowed


def test_circuit_stays_open_before_timeout():
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.should_allow_request() is False


# ---------------------------------------------------------------------------
# F30: Failed probe must restart recovery timer (not stay permeable forever)
# ---------------------------------------------------------------------------


def test_failed_probe_resets_recovery_timer():
    """After recovery_timeout, a failed probe must re-block (not flood)."""
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_open is True

    # Wait for recovery timeout — probe allowed
    time.sleep(0.15)
    assert breaker.should_allow_request() is True

    # Probe fails — timer must reset, blocking again
    breaker.record_failure()
    assert breaker.should_allow_request() is False  # re-blocked

    # Wait again — second probe allowed
    time.sleep(0.15)
    assert breaker.should_allow_request() is True


def test_successful_probe_resets_breaker():
    """Successful probe after timeout fully resets the circuit breaker."""
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_open is True

    time.sleep(0.15)
    assert breaker.should_allow_request() is True

    # Probe succeeds — full reset
    breaker.record_success()
    assert breaker.is_open is False
    assert breaker.consecutive_failures == 0


# ---------------------------------------------------------------------------
# Stage 4: record_failure(error_class) back-compat + observability
# ---------------------------------------------------------------------------


def test_record_failure_accepts_no_arg_back_compat():
    """Existing orchestrator-level callers pass no arg — must still work."""
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
    # The two existing call sites in session_orchestrator (exit-submit and
    # entry-submit retry exhaustion at :2111 and :2428) call this with no arg.
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.consecutive_failures == 2
    assert breaker.is_open is False
    # last_error_class remains None when no class is passed
    assert breaker.last_error_class is None


def test_record_failure_tracks_last_error_class():
    """When error_class is provided (HTTP-client wiring), it surfaces on the dataclass."""
    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=1.0)
    breaker.record_failure(error_class="B")  # read timeout
    assert breaker.last_error_class == "B"
    assert breaker.consecutive_failures == 1

    breaker.record_failure(error_class="D")  # 5xx
    assert breaker.last_error_class == "D"
    assert breaker.consecutive_failures == 2


def test_orchestrator_failure_does_not_overwrite_http_class():
    """Orchestrator-layer record_failure() must not clobber the HTTP class.

    Scenario: a 5xx storm at the HTTP layer drove the breaker partway open
    (last_error_class="D"). Then the orchestrator's exit-retry exhausts and
    calls record_failure() with no arg. The dashboard operator still needs
    to see "D" to know what caused the failure, not "None".
    """
    breaker = CircuitBreaker(failure_threshold=10, recovery_timeout=1.0)
    breaker.record_failure(error_class="D")
    breaker.record_failure(error_class="D")
    assert breaker.last_error_class == "D"

    breaker.record_failure()  # orchestrator-layer, no arg
    # The most-recent classified failure ("D") is preserved.
    assert breaker.last_error_class == "D"
    assert breaker.consecutive_failures == 3


def test_record_success_clears_last_error_class():
    """Successful read clears the last_error_class so the dashboard shows clean state."""
    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=1.0)
    breaker.record_failure(error_class="E")  # 429
    assert breaker.last_error_class == "E"

    breaker.record_success()
    assert breaker.last_error_class is None
    assert breaker.consecutive_failures == 0
