"""Tests for scripts.tools.generate_trade_sheet."""

from pathlib import Path

from scripts.tools.generate_trade_sheet import FitnessCheckResult, _check_fitness, _fitness_badge


def test_check_fitness_caches_success(monkeypatch):
    calls = {"count": 0}

    class DummyFitness:
        fitness_status = "FIT"

    def fake_compute_fitness(strategy_id, db_path):
        calls["count"] += 1
        return DummyFitness()

    monkeypatch.setattr("scripts.tools.generate_trade_sheet.compute_fitness", fake_compute_fitness)

    cache = {}
    first = _check_fitness("SID_1", Path("gold.db"), cache)
    second = _check_fitness("SID_1", Path("gold.db"), cache)

    assert first == FitnessCheckResult(status="FIT", error=None)
    assert second == first
    assert calls["count"] == 1


def test_check_fitness_returns_unknown_with_error(monkeypatch):
    def fake_compute_fitness(strategy_id, db_path):
        raise RuntimeError("boom")

    monkeypatch.setattr("scripts.tools.generate_trade_sheet.compute_fitness", fake_compute_fitness)

    result = _check_fitness("SID_2", Path("gold.db"), {})

    assert result.status == "UNKNOWN"
    assert result.error == "RuntimeError: boom"


def test_fitness_badge_unknown_is_not_decay():
    badge = _fitness_badge("UNKNOWN")

    assert "badge-unknown" in badge
    assert "badge-decay" not in badge
