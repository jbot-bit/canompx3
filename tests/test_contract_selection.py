"""
Tests for pipeline.ingest_dbn_mgc contract selection functions.

Tests choose_front_contract() and parse_expiry() — pure functions, no DB needed.
"""

import pytest
import re
from pipeline.ingest_dbn_mgc import choose_front_contract, parse_expiry, MGC_OUTRIGHT_PATTERN


class TestParseExpiry:
    """Tests for contract expiry parsing."""

    def test_mgcg5(self):
        year, month = parse_expiry("MGCG5", prefix_len=3)
        assert year == 2005
        assert month == 2  # G = February

    def test_mgcz25(self):
        year, month = parse_expiry("MGCZ25", prefix_len=3)
        assert year == 2025
        assert month == 12  # Z = December

    def test_mgcm4(self):
        year, month = parse_expiry("MGCM4", prefix_len=3)
        assert year == 2004
        assert month == 6  # M = June

    def test_mgcf26(self):
        year, month = parse_expiry("MGCF26", prefix_len=3)
        assert year == 2026
        assert month == 1  # F = January

    def test_nq_prefix_len_2(self):
        year, month = parse_expiry("NQH5", prefix_len=2)
        assert year == 2005
        assert month == 3  # H = March

    def test_invalid_month_code_raises(self):
        with pytest.raises((ValueError, IndexError)):
            parse_expiry("MGCA5", prefix_len=3)  # A is not a valid month code

    def test_two_digit_year_above_50(self):
        year, month = parse_expiry("MGCG75", prefix_len=3)
        assert year == 1975
        assert month == 2


class TestChooseFrontContract:
    """Tests for deterministic front-month selection."""

    def test_single_contract(self):
        volumes = {"MGCM4": 1000}
        result = choose_front_contract(volumes)
        assert result == "MGCM4"

    def test_highest_volume_wins(self):
        volumes = {"MGCM4": 500, "MGCZ4": 1000, "MGCG5": 200}
        result = choose_front_contract(volumes)
        assert result == "MGCZ4"

    def test_tie_by_expiry(self):
        # Equal volume, MGCM4 expires earlier (June) than MGCZ4 (December)
        volumes = {"MGCZ4": 500, "MGCM4": 500}
        result = choose_front_contract(volumes)
        assert result == "MGCM4"

    def test_tie_by_lexicographic(self):
        # If expiry parse fails, fall back to lexicographic
        # Both have same volume, use pattern that includes non-parseable symbols
        pattern = re.compile(r'^TEST\w+$')
        volumes = {"TESTB": 500, "TESTA": 500}
        result = choose_front_contract(volumes, outright_pattern=pattern, prefix_len=4)
        assert result == "TESTA"

    def test_no_outrights_returns_none(self):
        # Spreads don't match MGC outright pattern
        volumes = {"MGC-MGCM4": 1000}
        result = choose_front_contract(volumes)
        assert result is None

    def test_empty_volumes_returns_none(self):
        result = choose_front_contract({})
        assert result is None

    def test_spreads_filtered_out(self):
        volumes = {
            "MGCM4": 100,
            "MGC-MGCM4-MGCZ4": 5000,  # Spread, should be filtered
        }
        result = choose_front_contract(volumes)
        assert result == "MGCM4"

    def test_log_func_called_on_tie(self):
        logs = []
        volumes = {"MGCM4": 500, "MGCZ4": 500}
        choose_front_contract(volumes, log_func=lambda msg: logs.append(msg))
        assert len(logs) > 0
        assert "TIE" in logs[0]
