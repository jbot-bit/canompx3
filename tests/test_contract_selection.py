"""
Tests for pipeline.ingest_dbn_mgc contract selection functions.

Tests choose_front_contract() and parse_expiry() — pure functions, no DB needed.
"""

import pytest
import re
from pipeline.ingest_dbn_mgc import choose_front_contract, parse_expiry, GC_OUTRIGHT_PATTERN


class TestParseExpiry:
    """Tests for contract expiry parsing."""

    def test_gcg5(self):
        year, month = parse_expiry("GCG5", prefix_len=2)
        assert year == 2005
        assert month == 2  # G = February

    def test_gcz25(self):
        year, month = parse_expiry("GCZ25", prefix_len=2)
        assert year == 2025
        assert month == 12  # Z = December

    def test_gcm4(self):
        year, month = parse_expiry("GCM4", prefix_len=2)
        assert year == 2004
        assert month == 6  # M = June

    def test_gcf26(self):
        year, month = parse_expiry("GCF26", prefix_len=2)
        assert year == 2026
        assert month == 1  # F = January

    def test_nq_prefix_len_2(self):
        year, month = parse_expiry("NQH5", prefix_len=2)
        assert year == 2005
        assert month == 3  # H = March

    def test_invalid_month_code_raises(self):
        with pytest.raises((ValueError, IndexError)):
            parse_expiry("GCA5", prefix_len=2)  # A is not a valid month code

    def test_two_digit_year_above_50(self):
        year, month = parse_expiry("GCG75", prefix_len=2)
        assert year == 1975
        assert month == 2


class TestChooseFrontContract:
    """Tests for deterministic front-month selection."""

    def test_single_contract(self):
        volumes = {"GCM4": 1000}
        result = choose_front_contract(volumes, outright_pattern=GC_OUTRIGHT_PATTERN, prefix_len=2)
        assert result == "GCM4"

    def test_highest_volume_wins(self):
        volumes = {"GCM4": 500, "GCZ4": 1000, "GCG5": 200}
        result = choose_front_contract(volumes, outright_pattern=GC_OUTRIGHT_PATTERN, prefix_len=2)
        assert result == "GCZ4"

    def test_tie_by_expiry(self):
        # Equal volume, GCM4 expires earlier (June) than GCZ4 (December)
        volumes = {"GCZ4": 500, "GCM4": 500}
        result = choose_front_contract(volumes, outright_pattern=GC_OUTRIGHT_PATTERN, prefix_len=2)
        assert result == "GCM4"

    def test_tie_by_lexicographic(self):
        # If expiry parse fails, fall back to lexicographic
        pattern = re.compile(r'^TEST\w+$')
        volumes = {"TESTB": 500, "TESTA": 500}
        result = choose_front_contract(volumes, outright_pattern=pattern, prefix_len=4)
        assert result == "TESTA"

    def test_no_outrights_returns_none(self):
        # Spreads don't match GC outright pattern
        volumes = {"GCH1-GCJ1": 1000}
        result = choose_front_contract(volumes, outright_pattern=GC_OUTRIGHT_PATTERN, prefix_len=2)
        assert result is None

    def test_empty_volumes_returns_none(self):
        result = choose_front_contract({}, outright_pattern=GC_OUTRIGHT_PATTERN, prefix_len=2)
        assert result is None

    def test_spreads_filtered_out(self):
        volumes = {
            "GCM4": 100,
            "GCM4-GCZ4": 5000,  # Spread, should be filtered
        }
        result = choose_front_contract(volumes, outright_pattern=GC_OUTRIGHT_PATTERN, prefix_len=2)
        assert result == "GCM4"

    def test_log_func_called_on_tie(self):
        logs = []
        volumes = {"GCM4": 500, "GCZ4": 500}
        choose_front_contract(volumes, outright_pattern=GC_OUTRIGHT_PATTERN, prefix_len=2, log_func=lambda msg: logs.append(msg))
        assert len(logs) > 0
        assert "TIE" in logs[0]
