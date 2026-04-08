"""Tests for trading_app/hypothesis_loader.py — Phase 4 Stage 4.0 read-side."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest
import yaml

from trading_app.hypothesis_loader import (
    HypothesisLoaderError,
    compute_file_sha,
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
