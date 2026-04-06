"""Tests for trading_app.eligibility.decomposition — the atom registry.

These tests prove:
- NO_FILTER decomposes to zero atoms (always ELIGIBLE)
- Single filters decompose to one atom (ORB_G5, PIT_MIN, COST_LT10, etc.)
- Composites decompose to multiple atoms (ORB_G5_FAST5_CONT → 3 atoms)
- Unknown filters return an explicit UNKNOWN placeholder atom, never silent
- Feature column resolution handles session templating correctly
"""

from __future__ import annotations

from trading_app.eligibility.decomposition import (
    AtomSpec,
    atr_velocity_atom_template,
    calendar_atom_template,
    decompose,
)
from trading_app.eligibility.types import (
    ConditionCategory,
    ConfidenceTier,
    ResolvesAt,
)


class TestNoFilter:
    """NO_FILTER produces zero atoms — the strategy is always eligible by
    filter definition (but overlays still apply)."""

    def test_returns_empty_tuple(self):
        assert decompose("NO_FILTER") == ()


class TestOrbSize:
    def test_orb_g5_single_atom(self):
        atoms = decompose("ORB_G5")
        assert len(atoms) == 1
        atom = atoms[0]
        assert "ORB size >= 5 pts" in atom.name
        assert atom.category == ConditionCategory.INTRA_SESSION
        assert atom.resolves_at == ResolvesAt.ORB_FORMATION
        assert atom.threshold == 5.0
        assert atom.comparator == ">="

    def test_orb_g6_threshold(self):
        atoms = decompose("ORB_G6")
        assert len(atoms) == 1
        assert atoms[0].threshold == 6.0

    def test_orb_g4_l12_band_filter(self):
        atoms = decompose("ORB_G4_L12")
        # Should produce two atoms: >= 4 and < 12
        assert len(atoms) == 2
        assert atoms[0].threshold == 4.0
        assert atoms[0].comparator == ">="
        assert atoms[1].threshold == 12.0
        assert atoms[1].comparator == "<"


class TestCostRatio:
    def test_cost_lt10(self):
        atoms = decompose("COST_LT10")
        assert len(atoms) == 1
        assert "cost ratio < 10%" in atoms[0].name
        assert atoms[0].threshold == 10.0

    def test_cost_lt12(self):
        atoms = decompose("COST_LT12")
        assert atoms[0].threshold == 12.0


class TestOvernightRange:
    def test_ovnrng_100(self):
        atoms = decompose("OVNRNG_100")
        assert len(atoms) == 1
        assert atoms[0].threshold == 100.0
        assert atoms[0].category == ConditionCategory.PRE_SESSION
        assert atoms[0].feature_column == "overnight_range"


class TestPitMin:
    def test_pit_min_is_validated_for_all_three_instruments(self):
        atoms = decompose("PIT_MIN")
        assert len(atoms) == 1
        atom = atoms[0]
        assert atom.threshold == 0.10
        assert atom.confidence_tier == ConfidenceTier.PROVEN
        validated = dict.fromkeys(atom.validated_for)  # (inst, session) keys
        assert ("MGC", "CME_REOPEN") in validated
        assert ("MNQ", "CME_REOPEN") in validated
        assert ("MES", "CME_REOPEN") in validated


class TestPDR:
    def test_pdr_r080_ratio_parsing(self):
        atoms = decompose("PDR_R080")
        assert len(atoms) == 1
        assert atoms[0].threshold == 0.80

    def test_pdr_r125_ratio_parsing(self):
        atoms = decompose("PDR_R125")
        assert atoms[0].threshold == 1.25


class TestGap:
    def test_gap_r005_ratio_parsing(self):
        atoms = decompose("GAP_R005")
        assert len(atoms) == 1
        assert atoms[0].threshold == 0.005

    def test_gap_r015_ratio_parsing(self):
        atoms = decompose("GAP_R015")
        assert atoms[0].threshold == 0.015


class TestCrossAssetATR:
    def test_x_mes_atr70(self):
        atoms = decompose("X_MES_ATR70")
        assert len(atoms) == 1
        assert atoms[0].feature_column == "cross_atr_MES_pct"
        assert atoms[0].threshold == 70.0

    def test_x_mes_atr60(self):
        atoms = decompose("X_MES_ATR60")
        assert atoms[0].threshold == 60.0


class TestFastFilter:
    def test_fast5_in_composite(self):
        atoms = decompose("ORB_G5_FAST5")
        names = [a.name for a in atoms]
        assert any("break delay <= 5 min" in n for n in names)

    def test_fast5_validated_for_is_mnq_specific(self):
        atoms = decompose("FAST5")
        assert len(atoms) == 1
        validated = atoms[0].validated_for
        # MNQ should be in the list; MGC EUROPE_FLOW should not
        mnq_sessions = [s for (i, s) in validated if i == "MNQ"]
        assert len(mnq_sessions) > 0
        assert ("MGC", "EUROPE_FLOW") not in validated


class TestDirection:
    def test_dir_long(self):
        atoms = decompose("DIR_LONG")
        assert len(atoms) == 1
        assert atoms[0].category == ConditionCategory.DIRECTIONAL
        assert atoms[0].threshold == "long"

    def test_dir_short(self):
        atoms = decompose("DIR_SHORT")
        assert atoms[0].threshold == "short"


class TestComposites:
    """Composite filters decompose into multiple atomic conditions."""

    def test_orb_g5_fast5_cont_produces_three_atoms(self):
        atoms = decompose("ORB_G5_FAST5_CONT")
        # Expect: ORB size, break delay, break bar continues
        assert len(atoms) >= 3
        names = [a.name for a in atoms]
        assert any("ORB size" in n for n in names)
        assert any("break delay" in n for n in names)
        assert any("break bar" in n for n in names)

    def test_cost_lt10_fast5_produces_two_atoms(self):
        atoms = decompose("COST_LT10_FAST5")
        assert len(atoms) >= 2
        names = [a.name for a in atoms]
        assert any("cost ratio" in n for n in names)
        assert any("break delay" in n for n in names)

    def test_ovnrng_25_fast5_produces_two_atoms(self):
        atoms = decompose("OVNRNG_25_FAST5")
        assert len(atoms) >= 2
        names = [a.name for a in atoms]
        assert any("overnight range" in n for n in names)
        assert any("break delay" in n for n in names)


class TestDOW:
    def test_nomon_atom(self):
        atoms = decompose("ORB_G5_NOMON")
        names = [a.name for a in atoms]
        assert any("!= Monday" in n for n in names)


class TestUnknownFilter:
    """Unknown filters produce an explicit UNKNOWN placeholder — never silent."""

    def test_unknown_returns_placeholder(self):
        atoms = decompose("TOTALLY_UNKNOWN_FILTER_XYZ")
        assert len(atoms) == 1
        assert "UNKNOWN" in atoms[0].name
        assert atoms[0].confidence_tier == ConfidenceTier.UNKNOWN

    def test_empty_string_returns_placeholder(self):
        atoms = decompose("")
        assert len(atoms) == 1
        assert "UNKNOWN" in atoms[0].name


class TestFeatureColumnResolution:
    def test_session_templated_column_resolves(self):
        atoms = decompose("ORB_G5")
        col = atoms[0].resolve_feature_column("CME_REOPEN")
        assert col == "orb_CME_REOPEN_size"

    def test_non_templated_column_passes_through(self):
        atoms = decompose("PIT_MIN")
        col = atoms[0].resolve_feature_column("CME_REOPEN")
        assert col == "pit_range_atr"  # not templated


class TestAtomSpecImmutability:
    """AtomSpec is a frozen dataclass."""

    def test_atomspec_is_frozen(self):
        import pytest

        atom = AtomSpec(
            name="test",
            category=ConditionCategory.PRE_SESSION,
            resolves_at=ResolvesAt.STARTUP,
            threshold=0.10,
            comparator=">=",
            source_filter="TEST",
        )
        with pytest.raises((AttributeError, TypeError)):
            atom.name = "changed"  # type: ignore[misc]


class TestOverlayTemplates:
    def test_calendar_atom_template(self):
        atom = calendar_atom_template()
        assert atom.category == ConditionCategory.OVERLAY
        assert atom.source_filter == "calendar"

    def test_atr_velocity_atom_template(self):
        atom = atr_velocity_atom_template()
        assert atom.category == ConditionCategory.OVERLAY
        assert atom.confidence_tier == ConfidenceTier.PROVEN
        validated = atom.validated_for
        assert ("MGC", "CME_REOPEN") in validated
        assert ("MGC", "TOKYO_OPEN") in validated
        # MNQ / MES must NOT be in validated — per research, no signal
        assert ("MNQ", "CME_REOPEN") not in validated
