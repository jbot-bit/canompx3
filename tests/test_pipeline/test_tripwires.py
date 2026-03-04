"""
Tripwire tests: stable invariants that detect data corruption or config desync.

These catch regressions that would otherwise go unnoticed until a live trade fails.
"""

from datetime import date

import pytest

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, ASSET_CONFIGS
from pipeline.cost_model import COST_SPECS, CostSpec, get_cost_spec
from pipeline.dst import SESSION_CATALOG, DYNAMIC_ORB_RESOLVERS


# ── Tripwire 1: Cost model internal consistency ────────────────────────


class TestCostModelConsistency:
    """Verify cost model arithmetic is internally consistent."""

    def test_total_friction_equals_sum_of_parts(self):
        """total_friction = commission + spread + slippage (no silent rounding)."""
        for inst, spec in COST_SPECS.items():
            expected = spec.commission_rt + spec.spread_doubled + spec.slippage
            assert spec.total_friction == pytest.approx(expected), (
                f"{inst}: total_friction={spec.total_friction} != "
                f"{spec.commission_rt}+{spec.spread_doubled}+{spec.slippage}={expected}"
            )

    def test_friction_in_points_positive(self):
        """Friction in points must be positive for all instruments."""
        for inst, spec in COST_SPECS.items():
            assert spec.friction_in_points > 0, f"{inst}: friction_in_points <= 0"

    def test_every_active_instrument_has_cost_spec(self):
        """Every active ORB instrument must have a cost spec."""
        for inst in ACTIVE_ORB_INSTRUMENTS:
            spec = get_cost_spec(inst)
            assert isinstance(spec, CostSpec), f"{inst}: get_cost_spec returned {type(spec)}"

    def test_cost_spec_instrument_field_matches_key(self):
        """CostSpec.instrument must match its dict key."""
        for inst, spec in COST_SPECS.items():
            assert spec.instrument == inst, (
                f"Key={inst} but spec.instrument={spec.instrument}"
            )

    def test_unknown_instrument_raises(self):
        """get_cost_spec must raise for unknown instruments (fail-closed)."""
        with pytest.raises(ValueError):
            get_cost_spec("FAKE_INSTRUMENT")


# ── Tripwire 2: DST transition day correctness (2026) ─────────────────


class TestDstTransitions2026:
    """Pin 2026 DST transition dates for US and UK.

    If these fail, session resolvers are producing wrong times
    on the most dangerous days of the year.
    """

    def test_us_spring_forward_2026(self):
        """US DST starts March 8, 2026 (second Sunday of March)."""
        from pipeline.dst import is_us_dst
        assert is_us_dst(date(2026, 3, 7)) is False   # Saturday before
        assert is_us_dst(date(2026, 3, 8)) is True     # Spring forward
        assert is_us_dst(date(2026, 3, 9)) is True     # Monday after

    def test_us_fall_back_2026(self):
        """US DST ends November 1, 2026 (first Sunday of November)."""
        from pipeline.dst import is_us_dst
        assert is_us_dst(date(2026, 10, 31)) is True   # Saturday before
        assert is_us_dst(date(2026, 11, 1)) is False    # Fall back
        assert is_us_dst(date(2026, 11, 2)) is False    # Monday after

    def test_uk_spring_forward_2026(self):
        """UK BST starts March 29, 2026 (last Sunday of March)."""
        from pipeline.dst import is_uk_dst
        assert is_uk_dst(date(2026, 3, 28)) is False   # Saturday before
        assert is_uk_dst(date(2026, 3, 29)) is True     # Spring forward
        assert is_uk_dst(date(2026, 3, 30)) is True     # Monday after

    def test_uk_fall_back_2026(self):
        """UK BST ends October 25, 2026 (last Sunday of October)."""
        from pipeline.dst import is_uk_dst
        assert is_uk_dst(date(2026, 10, 24)) is True   # Saturday before
        assert is_uk_dst(date(2026, 10, 25)) is False    # Fall back
        assert is_uk_dst(date(2026, 10, 26)) is False    # Monday after

    def test_all_resolvers_handle_us_transition_day(self):
        """Every DST-affected resolver must return valid times on US spring-forward."""
        transition = date(2026, 3, 8)
        for label, resolver in DYNAMIC_ORB_RESOLVERS.items():
            h, m = resolver(transition)
            assert 0 <= h <= 23 and 0 <= m <= 59, (
                f"{label} returned ({h}, {m}) on US DST transition"
            )


# ── Tripwire 3: Asset config ↔ Session catalog sync ───────────────────


class TestAssetSessionSync:
    """Verify asset configs and session catalog are in sync."""

    def test_enabled_sessions_exist_in_catalog(self):
        """Every enabled_session in ASSET_CONFIGS must be in SESSION_CATALOG."""
        for inst, cfg in ASSET_CONFIGS.items():
            for session in cfg.get("enabled_sessions", []):
                assert session in SESSION_CATALOG, (
                    f"{inst}: enabled_session '{session}' not in SESSION_CATALOG"
                )

    def test_active_instruments_have_enabled_sessions(self):
        """Every active instrument must have at least 1 enabled session."""
        for inst in ACTIVE_ORB_INSTRUMENTS:
            sessions = ASSET_CONFIGS[inst].get("enabled_sessions", [])
            assert len(sessions) > 0, f"{inst}: no enabled_sessions"

    def test_active_instruments_have_symbol(self):
        """Every active instrument must have a 'symbol' field."""
        for inst in ACTIVE_ORB_INSTRUMENTS:
            assert "symbol" in ASSET_CONFIGS[inst], f"{inst}: missing 'symbol'"
