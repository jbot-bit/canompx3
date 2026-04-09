"""Tests for trading_app/hypothesis_loader.py — Phase 4 Stages 4.0 + 4.1."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from pathlib import Path

import pytest
import yaml

from trading_app.hypothesis_loader import (
    HypothesisLoaderError,
    HypothesisScope,
    ScopePredicate,
    check_mode_a_consistency,
    compute_file_sha,
    enforce_minbtl_bound,
    extract_scope_predicate,
    find_hypothesis_file_by_sha,
    hypothesis_dir,
    load_hypothesis_by_sha,
    load_hypothesis_metadata,
)


def _write_minimal_hypothesis(path: Path, total_trials: int = 60, with_theory: bool = True) -> None:
    """Helper: write a minimal valid hypothesis YAML to ``path``."""
    body: dict = {
        "metadata": {
            "name": "test_hypothesis",
            "date_locked": "2026-04-08",
            "holdout_date": "2026-01-01",
            "total_expected_trials": total_trials,
        },
        "hypotheses": [
            {
                "id": 1,
                "name": "synthetic",
                "filter": {"type": "ORB_G5", "column": "orb_size", "thresholds": [5]},
                "scope": {"sessions": ["NYSE_OPEN"]},
            }
        ],
    }
    if with_theory:
        body["hypotheses"][0]["theory_citation"] = (
            "docs/institutional/literature/synthetic_test.md"
        )
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")


class TestComputeFileSha:
    """SHA is deterministic and content-derived."""

    def test_sha_for_known_content(self, tmp_path):
        path = tmp_path / "x.yaml"
        path.write_text("hello world", encoding="utf-8")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert compute_file_sha(path) == expected

    def test_sha_changes_when_content_changes(self, tmp_path):
        path = tmp_path / "x.yaml"
        path.write_text("a", encoding="utf-8")
        sha_a = compute_file_sha(path)
        path.write_text("b", encoding="utf-8")
        sha_b = compute_file_sha(path)
        assert sha_a != sha_b

    def test_sha_deterministic_across_two_reads(self, tmp_path):
        path = tmp_path / "x.yaml"
        path.write_text("stable content", encoding="utf-8")
        assert compute_file_sha(path) == compute_file_sha(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(HypothesisLoaderError, match="not found"):
            compute_file_sha(tmp_path / "nope.yaml")


class TestLoadHypothesisMetadata:
    """Schema validation and metadata extraction."""

    def test_minimal_valid_file_loads(self, tmp_path):
        path = tmp_path / "h.yaml"
        _write_minimal_hypothesis(path)
        meta = load_hypothesis_metadata(path)
        assert meta["name"] == "test_hypothesis"
        assert meta["total_expected_trials"] == 60
        assert meta["holdout_date"] == date(2026, 1, 1)
        assert meta["has_theory"] is True
        assert meta["sha"] == compute_file_sha(path)
        assert meta["path"] == str(path)

    def test_theory_absent_yields_has_theory_false(self, tmp_path):
        path = tmp_path / "h.yaml"
        _write_minimal_hypothesis(path, with_theory=False)
        meta = load_hypothesis_metadata(path)
        assert meta["has_theory"] is False

    def test_missing_metadata_raises(self, tmp_path):
        path = tmp_path / "h.yaml"
        path.write_text("hypotheses: [a]", encoding="utf-8")
        with pytest.raises(HypothesisLoaderError, match="missing required top-level keys"):
            load_hypothesis_metadata(path)

    def test_missing_hypotheses_list_raises(self, tmp_path):
        path = tmp_path / "h.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "metadata": {
                        "name": "x",
                        "date_locked": "2026-04-08",
                        "holdout_date": "2026-01-01",
                        "total_expected_trials": 10,
                    }
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(HypothesisLoaderError, match="missing required top-level keys"):
            load_hypothesis_metadata(path)

    def test_missing_total_expected_trials_raises(self, tmp_path):
        path = tmp_path / "h.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "metadata": {
                        "name": "x",
                        "date_locked": "2026-04-08",
                        "holdout_date": "2026-01-01",
                    },
                    "hypotheses": [{"id": 1}],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(HypothesisLoaderError, match="missing required metadata keys"):
            load_hypothesis_metadata(path)

    def test_zero_trials_raises(self, tmp_path):
        path = tmp_path / "h.yaml"
        _write_minimal_hypothesis(path, total_trials=0)
        with pytest.raises(HypothesisLoaderError, match="positive int"):
            load_hypothesis_metadata(path)

    def test_negative_trials_raises(self, tmp_path):
        path = tmp_path / "h.yaml"
        _write_minimal_hypothesis(path, total_trials=-5)
        with pytest.raises(HypothesisLoaderError, match="positive int"):
            load_hypothesis_metadata(path)

    def test_invalid_holdout_date_raises(self, tmp_path):
        path = tmp_path / "h.yaml"
        body = {
            "metadata": {
                "name": "x",
                "date_locked": "2026-04-08",
                "holdout_date": "not-a-date",
                "total_expected_trials": 10,
            },
            "hypotheses": [{"id": 1}],
        }
        path.write_text(yaml.safe_dump(body), encoding="utf-8")
        with pytest.raises(HypothesisLoaderError, match="holdout_date"):
            load_hypothesis_metadata(path)

    def test_holdout_date_as_yaml_native_date(self, tmp_path):
        # YAML's safe_load returns datetime.date for bare 2026-01-01 (no quotes).
        path = tmp_path / "h.yaml"
        path.write_text(
            "metadata:\n"
            "  name: x\n"
            "  date_locked: 2026-04-08\n"
            "  holdout_date: 2026-01-01\n"
            "  total_expected_trials: 10\n"
            "hypotheses:\n"
            "  - id: 1\n",
            encoding="utf-8",
        )
        meta = load_hypothesis_metadata(path)
        assert meta["holdout_date"] == date(2026, 1, 1)

    def test_empty_hypotheses_list_raises(self, tmp_path):
        path = tmp_path / "h.yaml"
        body = {
            "metadata": {
                "name": "x",
                "date_locked": "2026-04-08",
                "holdout_date": "2026-01-01",
                "total_expected_trials": 10,
            },
            "hypotheses": [],
        }
        path.write_text(yaml.safe_dump(body), encoding="utf-8")
        with pytest.raises(HypothesisLoaderError, match="non-empty list"):
            load_hypothesis_metadata(path)

    def test_invalid_yaml_raises(self, tmp_path):
        path = tmp_path / "h.yaml"
        path.write_text("metadata: {invalid: [", encoding="utf-8")
        with pytest.raises(HypothesisLoaderError, match="not valid YAML"):
            load_hypothesis_metadata(path)


class TestFindHypothesisFileBySha:
    """SHA → file lookup against a custom search directory."""

    def test_finds_matching_file(self, tmp_path):
        path = tmp_path / "h.yaml"
        _write_minimal_hypothesis(path)
        sha = compute_file_sha(path)
        found = find_hypothesis_file_by_sha(sha, search_dir=tmp_path)
        assert found == path

    def test_returns_none_for_unknown_sha(self, tmp_path):
        path = tmp_path / "h.yaml"
        _write_minimal_hypothesis(path)
        unknown_sha = "0" * 64
        found = find_hypothesis_file_by_sha(unknown_sha, search_dir=tmp_path)
        assert found is None

    def test_returns_none_for_missing_directory(self, tmp_path):
        nonexistent = tmp_path / "nope"
        found = find_hypothesis_file_by_sha("abc" * 21 + "a", search_dir=nonexistent)
        assert found is None

    def test_case_insensitive_sha_match(self, tmp_path):
        path = tmp_path / "h.yaml"
        _write_minimal_hypothesis(path)
        sha = compute_file_sha(path).upper()
        found = find_hypothesis_file_by_sha(sha, search_dir=tmp_path)
        assert found == path

    def test_picks_correct_file_among_multiple(self, tmp_path):
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        _write_minimal_hypothesis(a, total_trials=50)
        _write_minimal_hypothesis(b, total_trials=80)
        sha_b = compute_file_sha(b)
        found = find_hypothesis_file_by_sha(sha_b, search_dir=tmp_path)
        assert found == b


class TestLoadHypothesisBySha:
    """The convenience composition: find by SHA, then parse metadata."""

    def test_returns_metadata_when_found(self, tmp_path):
        path = tmp_path / "h.yaml"
        _write_minimal_hypothesis(path, total_trials=42)
        sha = compute_file_sha(path)
        meta = load_hypothesis_by_sha(sha, search_dir=tmp_path)
        assert meta is not None
        assert meta["total_expected_trials"] == 42

    def test_returns_none_when_not_found(self, tmp_path):
        meta = load_hypothesis_by_sha("0" * 64, search_dir=tmp_path)
        assert meta is None


class TestHypothesisDir:
    """The canonical directory anchor."""

    def test_returns_repo_relative_path(self):
        d = hypothesis_dir()
        # Must point at <repo>/docs/audit/hypotheses
        assert d.name == "hypotheses"
        assert d.parent.name == "audit"
        assert d.parent.parent.name == "docs"


# ---------------------------------------------------------------------------
# Phase 4 Stage 4.1 additions — tests for ScopePredicate, enforce_minbtl_bound,
# check_mode_a_consistency, and extract_scope_predicate.
# ---------------------------------------------------------------------------


def _make_scoped_hypothesis(
    *,
    instruments: list[str] | None = None,
    sessions: list[str] | None = None,
    rr_targets: list[float] | None = None,
    entry_models: list[str] | None = None,
    confirm_bars: list[int] | None = None,
    stop_multipliers: list[float] | None = None,
    filter_type: str = "OVNRNG",
    expected_trials: int = 10,
    hypothesis_id: int = 1,
) -> dict:
    """Build a well-formed hypothesis dict with a full scope block.

    Used by Stage 4.1 tests — the existing ``_write_minimal_hypothesis``
    helper above only sets ``scope.sessions`` and is insufficient for
    ``extract_scope_predicate`` which requires all six scope dimensions.
    """
    return {
        "id": hypothesis_id,
        "name": f"test_hypothesis_{hypothesis_id}",
        "theory_citation": "docs/institutional/literature/synthetic.md",
        "economic_basis": "synthetic test fixture",
        "filter": {
            "type": filter_type,
            "column": "synthetic",
            "thresholds": [1, 2, 3],
        },
        "scope": {
            "instruments": instruments or ["MNQ"],
            "sessions": sessions or ["NYSE_OPEN"],
            "rr_targets": rr_targets or [1.0, 1.5, 2.0],
            "entry_models": entry_models or ["E2"],
            "confirm_bars": confirm_bars or [1],
            "stop_multipliers": stop_multipliers or [1.0],
        },
        "expected_trial_count": expected_trials,
        "kill_criteria": ["placeholder"],
    }


class TestEnforceMinbtlBound:
    """Criterion 2 MinBTL canonical enforcement — locked 300/2000 bounds."""

    def test_clean_mode_200_trials_passes(self):
        verdict, reason = enforce_minbtl_bound({"total_expected_trials": 200})
        assert verdict is None
        assert reason is None

    def test_clean_mode_exactly_300_passes(self):
        """300 is the inclusive upper bound for clean mode."""
        verdict, _ = enforce_minbtl_bound({"total_expected_trials": 300})
        assert verdict is None

    def test_clean_mode_301_rejects(self):
        verdict, reason = enforce_minbtl_bound({"total_expected_trials": 301})
        assert verdict == "REJECTED"
        assert reason is not None
        assert "criterion_2" in reason
        assert "300" in reason
        assert "clean" in reason.lower()

    def test_proxy_mode_without_opt_in_rejects(self):
        """Proxy mode requested without metadata.data_source_mode='proxy' → reject."""
        verdict, reason = enforce_minbtl_bound(
            {"total_expected_trials": 500},
            on_proxy_data=True,
        )
        assert verdict == "REJECTED"
        assert reason is not None
        assert "data_source_mode" in reason

    def test_proxy_mode_without_disclosure_rejects(self):
        verdict, reason = enforce_minbtl_bound(
            {
                "total_expected_trials": 500,
                "metadata": {"data_source_mode": "proxy"},
            },
            on_proxy_data=True,
        )
        assert verdict == "REJECTED"
        assert reason is not None
        assert "disclosure" in reason.lower()

    def test_proxy_mode_with_full_opt_in_500_passes(self):
        meta = {
            "total_expected_trials": 500,
            "metadata": {
                "data_source_mode": "proxy",
                "data_source_disclosure": "NQ parent used pre-2024-02-05",
            },
        }
        verdict, _ = enforce_minbtl_bound(meta, on_proxy_data=True)
        assert verdict is None

    def test_proxy_mode_exactly_2000_passes(self):
        """2000 is the inclusive upper bound for proxy mode."""
        meta = {
            "total_expected_trials": 2000,
            "metadata": {
                "data_source_mode": "proxy",
                "data_source_disclosure": "NQ parent pre-2024-02-05",
            },
        }
        verdict, _ = enforce_minbtl_bound(meta, on_proxy_data=True)
        assert verdict is None

    def test_proxy_mode_2001_rejects(self):
        meta = {
            "total_expected_trials": 2001,
            "metadata": {
                "data_source_mode": "proxy",
                "data_source_disclosure": "NQ parent pre-2024-02-05",
            },
        }
        verdict, reason = enforce_minbtl_bound(meta, on_proxy_data=True)
        assert verdict == "REJECTED"
        assert reason is not None
        assert "2000" in reason

    def test_missing_total_trials_raises(self):
        with pytest.raises(HypothesisLoaderError, match="total_expected_trials"):
            enforce_minbtl_bound({})

    def test_zero_total_trials_raises(self):
        with pytest.raises(HypothesisLoaderError, match="total_expected_trials"):
            enforce_minbtl_bound({"total_expected_trials": 0})

    def test_bool_total_trials_raises(self):
        """Regression guard for the bool-is-subclass-of-int trap.

        ``isinstance(True, int)`` is True in Python, so a YAML file with
        ``total_expected_trials: true`` would silently be treated as 1 trial
        without the explicit bool exclusion. Phase A review finding S-1.
        """
        with pytest.raises(HypothesisLoaderError, match="total_expected_trials"):
            enforce_minbtl_bound({"total_expected_trials": True})
        with pytest.raises(HypothesisLoaderError, match="total_expected_trials"):
            enforce_minbtl_bound({"total_expected_trials": False})


class TestCheckModeAConsistency:
    """Amendment 2.7 sacred boundary enforcement."""

    def test_exactly_sacred_boundary_passes(self):
        """2026-01-01 is the inclusive sacred-from date."""
        check_mode_a_consistency({"holdout_date": date(2026, 1, 1)})  # no raise

    def test_earlier_date_passes(self):
        check_mode_a_consistency({"holdout_date": date(2025, 6, 1)})

    def test_post_sacred_date_rejects(self):
        with pytest.raises(HypothesisLoaderError, match="Amendment 2.7"):
            check_mode_a_consistency({"holdout_date": date(2026, 4, 1)})

    def test_datetime_input_normalized(self):
        """datetime input should be normalized to date for comparison."""
        check_mode_a_consistency({"holdout_date": datetime(2026, 1, 1, 12, 0, 0)})

    def test_missing_holdout_date_raises(self):
        with pytest.raises(HypothesisLoaderError, match="holdout_date"):
            check_mode_a_consistency({})

    def test_wrong_type_raises(self):
        with pytest.raises(HypothesisLoaderError, match="date or datetime"):
            check_mode_a_consistency({"holdout_date": "2026-01-01"})


class TestScopePredicate:
    """Per-hypothesis bundling — prevents cross-pollination across hypotheses."""

    def _simple_meta(self) -> dict:
        """Two hypotheses — one MNQ/OVNRNG/EUROPE_FLOW, one MGC/ORB_G/CME_REOPEN."""
        return {
            "hypotheses": [
                _make_scoped_hypothesis(
                    hypothesis_id=1,
                    filter_type="OVNRNG",
                    instruments=["MNQ"],
                    sessions=["EUROPE_FLOW"],
                    rr_targets=[1.0, 1.5, 2.0],
                    expected_trials=12,
                ),
                _make_scoped_hypothesis(
                    hypothesis_id=2,
                    filter_type="ORB_G",
                    instruments=["MGC"],
                    sessions=["CME_REOPEN"],
                    rr_targets=[2.0, 2.5],
                    expected_trials=8,
                ),
            ]
        }

    def test_extract_mnq_returns_one_hypothesis(self):
        pred = extract_scope_predicate(self._simple_meta(), instrument="MNQ")
        assert len(pred.hypotheses) == 1
        assert pred.instrument == "MNQ"
        assert pred.total_declared_trials == 12

    def test_extract_mgc_returns_other_hypothesis(self):
        pred = extract_scope_predicate(self._simple_meta(), instrument="MGC")
        assert len(pred.hypotheses) == 1
        assert pred.total_declared_trials == 8

    def test_unknown_instrument_fails_loud(self):
        with pytest.raises(HypothesisLoaderError, match="MES"):
            extract_scope_predicate(self._simple_meta(), instrument="MES")

    def test_no_cross_pollination_across_hypotheses(self):
        """OVNRNG + CME_REOPEN must be REJECTED even though each half
        exists separately in the two hypotheses. This is the critical
        bundling invariant from the audit."""
        meta = {
            "hypotheses": [
                _make_scoped_hypothesis(
                    hypothesis_id=1,
                    filter_type="OVNRNG",
                    instruments=["MNQ"],
                    sessions=["EUROPE_FLOW"],
                ),
                _make_scoped_hypothesis(
                    hypothesis_id=2,
                    filter_type="ORB_G",
                    instruments=["MNQ"],
                    sessions=["CME_REOPEN"],
                ),
            ]
        }
        pred = extract_scope_predicate(meta, instrument="MNQ")
        assert len(pred.hypotheses) == 2
        # Valid: OVNRNG + EUROPE_FLOW
        assert pred.accepts(
            orb_label="EUROPE_FLOW",
            filter_type="OVNRNG",
            entry_model="E2",
            rr_target=1.5,
            confirm_bars=1,
            stop_multiplier=1.0,
        )
        # Valid: ORB_G + CME_REOPEN
        assert pred.accepts(
            orb_label="CME_REOPEN",
            filter_type="ORB_G",
            entry_model="E2",
            rr_target=1.5,
            confirm_bars=1,
            stop_multiplier=1.0,
        )
        # INVALID: OVNRNG + CME_REOPEN — cross-pollination
        assert not pred.accepts(
            orb_label="CME_REOPEN",
            filter_type="OVNRNG",
            entry_model="E2",
            rr_target=1.5,
            confirm_bars=1,
            stop_multiplier=1.0,
        )
        # INVALID: ORB_G + EUROPE_FLOW — the other cross-pollination
        assert not pred.accepts(
            orb_label="EUROPE_FLOW",
            filter_type="ORB_G",
            entry_model="E2",
            rr_target=1.5,
            confirm_bars=1,
            stop_multiplier=1.0,
        )

    def test_accepts_requires_all_six_dimensions(self):
        """Changing any one dimension from valid to invalid must reject."""
        meta = {
            "hypotheses": [
                _make_scoped_hypothesis(
                    filter_type="OVNRNG",
                    instruments=["MNQ"],
                    sessions=["EUROPE_FLOW"],
                    rr_targets=[2.0],
                    entry_models=["E2"],
                    confirm_bars=[1],
                    stop_multipliers=[1.0],
                )
            ]
        }
        pred = extract_scope_predicate(meta, instrument="MNQ")
        # Baseline: all 6 dimensions valid
        assert pred.accepts(
            orb_label="EUROPE_FLOW",
            filter_type="OVNRNG",
            entry_model="E2",
            rr_target=2.0,
            confirm_bars=1,
            stop_multiplier=1.0,
        )
        # Each of the 6 dimensions can individually invalidate. Kwargs are
        # inlined per call to keep pyright's type inference narrow (dict
        # unpacking widens to Union[str, float] which breaks accepts()'s
        # strict per-parameter typing).
        assert not pred.accepts(
            orb_label="NYSE_OPEN",
            filter_type="OVNRNG",
            entry_model="E2",
            rr_target=2.0,
            confirm_bars=1,
            stop_multiplier=1.0,
        )
        assert not pred.accepts(
            orb_label="EUROPE_FLOW",
            filter_type="ORB_G",
            entry_model="E2",
            rr_target=2.0,
            confirm_bars=1,
            stop_multiplier=1.0,
        )
        assert not pred.accepts(
            orb_label="EUROPE_FLOW",
            filter_type="OVNRNG",
            entry_model="E1",
            rr_target=2.0,
            confirm_bars=1,
            stop_multiplier=1.0,
        )
        assert not pred.accepts(
            orb_label="EUROPE_FLOW",
            filter_type="OVNRNG",
            entry_model="E2",
            rr_target=1.5,
            confirm_bars=1,
            stop_multiplier=1.0,
        )
        assert not pred.accepts(
            orb_label="EUROPE_FLOW",
            filter_type="OVNRNG",
            entry_model="E2",
            rr_target=2.0,
            confirm_bars=2,
            stop_multiplier=1.0,
        )
        assert not pred.accepts(
            orb_label="EUROPE_FLOW",
            filter_type="OVNRNG",
            entry_model="E2",
            rr_target=2.0,
            confirm_bars=1,
            stop_multiplier=0.75,
        )

    def test_allowed_sessions_union_across_hypotheses(self):
        meta = {
            "hypotheses": [
                _make_scoped_hypothesis(
                    hypothesis_id=1,
                    filter_type="OVNRNG",
                    instruments=["MNQ"],
                    sessions=["EUROPE_FLOW", "NYSE_OPEN"],
                ),
                _make_scoped_hypothesis(
                    hypothesis_id=2,
                    filter_type="ORB_G",
                    instruments=["MNQ"],
                    sessions=["CME_REOPEN"],
                ),
            ]
        }
        pred = extract_scope_predicate(meta, instrument="MNQ")
        assert pred.allowed_sessions() == frozenset({"EUROPE_FLOW", "NYSE_OPEN", "CME_REOPEN"})

    def test_total_declared_trials_sums(self):
        meta = {
            "hypotheses": [
                _make_scoped_hypothesis(hypothesis_id=1, instruments=["MNQ"], expected_trials=30),
                _make_scoped_hypothesis(hypothesis_id=2, instruments=["MNQ"], expected_trials=45, filter_type="ORB_G"),
            ]
        }
        pred = extract_scope_predicate(meta, instrument="MNQ")
        assert pred.total_declared_trials == 75

    def test_predicate_is_hashable_and_frozen(self):
        """Frozen dataclass instances are hashable — safe for sharing."""
        meta = {"hypotheses": [_make_scoped_hypothesis(instruments=["MNQ"])]}
        pred = extract_scope_predicate(meta, instrument="MNQ")
        # Hashable
        hash(pred)
        # Frozen — cannot mutate. The attribute name is computed at runtime
        # to prevent the type checker from statically resolving it against
        # the frozen dataclass's read-only slots. The runtime FrozenInstanceError
        # (subclass of AttributeError) is what we're asserting.
        attr_name = "".join(["instr", "ument"])
        with pytest.raises((AttributeError, TypeError)):
            setattr(pred, attr_name, "MES")

    def test_malformed_scope_missing_instruments_raises(self):
        meta = {
            "hypotheses": [
                {
                    "id": 1,
                    "filter": {"type": "X", "column": "y", "thresholds": [1]},
                    "scope": {
                        "sessions": ["NYSE_OPEN"],
                        "rr_targets": [1.0],
                        "entry_models": ["E2"],
                        "confirm_bars": [1],
                        "stop_multipliers": [1.0],
                    },
                    "expected_trial_count": 5,
                }
            ]
        }
        with pytest.raises(HypothesisLoaderError, match="instruments"):
            extract_scope_predicate(meta, instrument="MNQ")

    def test_malformed_scope_missing_filter_type_raises(self):
        meta = {
            "hypotheses": [
                {
                    "id": 1,
                    "filter": {},
                    "scope": {
                        "instruments": ["MNQ"],
                        "sessions": ["NYSE_OPEN"],
                        "rr_targets": [1.0],
                        "entry_models": ["E2"],
                        "confirm_bars": [1],
                        "stop_multipliers": [1.0],
                    },
                    "expected_trial_count": 5,
                }
            ]
        }
        with pytest.raises(HypothesisLoaderError, match="filter.type"):
            extract_scope_predicate(meta, instrument="MNQ")

    def test_empty_hypotheses_list_raises(self):
        with pytest.raises(HypothesisLoaderError, match="no hypotheses"):
            extract_scope_predicate({"hypotheses": []}, instrument="MNQ")


class TestTestingMode:
    """Amendment 3.0: testing_mode field exposed at top level."""

    def test_individual_mode_surfaced(self, tmp_path):
        """testing_mode: individual appears in the returned dict at top level."""
        p = tmp_path / "hyp.yaml"
        body = {
            "metadata": {
                "name": "test_individual",
                "date_locked": "2026-04-09",
                "holdout_date": "2026-01-01",
                "total_expected_trials": 1,
                "testing_mode": "individual",
            },
            "hypotheses": [
                {
                    "id": 1,
                    "name": "single_test",
                    "theory_citation": "Crabel 1990",
                    "filter": {"type": "ORB_G5", "column": "orb_size"},
                    "scope": {"sessions": ["NYSE_OPEN"]},
                }
            ],
        }
        p.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
        meta = load_hypothesis_metadata(p)
        assert meta["testing_mode"] == "individual"

    def test_family_mode_default(self, tmp_path):
        """When testing_mode absent, defaults to 'family'."""
        p = tmp_path / "hyp.yaml"
        _write_minimal_hypothesis(p, total_trials=1)
        meta = load_hypothesis_metadata(p)
        assert meta["testing_mode"] == "family"

    def test_explicit_family_mode(self, tmp_path):
        """testing_mode: family explicitly set."""
        p = tmp_path / "hyp.yaml"
        body = {
            "metadata": {
                "name": "test_family",
                "date_locked": "2026-04-09",
                "holdout_date": "2026-01-01",
                "total_expected_trials": 5,
                "testing_mode": "family",
            },
            "hypotheses": [
                {
                    "id": 1,
                    "name": "h1",
                    "filter": {"type": "ORB_G5", "column": "orb_size"},
                    "scope": {"sessions": ["NYSE_OPEN"]},
                }
            ],
        }
        p.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
        meta = load_hypothesis_metadata(p)
        assert meta["testing_mode"] == "family"

    def test_individual_requires_theory(self, tmp_path):
        """testing_mode: individual without theory_citation raises."""
        p = tmp_path / "hyp.yaml"
        body = {
            "metadata": {
                "name": "test_no_theory",
                "date_locked": "2026-04-09",
                "holdout_date": "2026-01-01",
                "total_expected_trials": 1,
                "testing_mode": "individual",
            },
            "hypotheses": [
                {
                    "id": 1,
                    "name": "no_theory_h",
                    "filter": {"type": "ORB_G5", "column": "orb_size"},
                    "scope": {"sessions": ["NYSE_OPEN"]},
                }
            ],
        }
        p.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
        with pytest.raises(HypothesisLoaderError, match="theory_citation"):
            load_hypothesis_metadata(p)
