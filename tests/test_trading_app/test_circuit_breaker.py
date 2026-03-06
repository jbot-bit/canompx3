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
