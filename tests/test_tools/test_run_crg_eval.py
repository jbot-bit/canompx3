"""Tests for scripts/tools/run_crg_eval.py.

Covers:
- Schema validation: required keys present
- Schema validation: missing test_commits sha rejected
- Schema validation: missing search_queries query/expected rejected
- Dry-run path: exits 0 without running benchmarks
- _summarize: token_efficiency halt math
- _summarize: handles empty results gracefully
- HALT_RATIO_THRESHOLD constant matches v2 plan (1.111 = ~10% savings)
- Live config file (configs/canompx3-crg-eval.yaml) parses cleanly
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "tools" / "run_crg_eval.py"
LIVE_CONFIG = PROJECT_ROOT / "configs" / "canompx3-crg-eval.yaml"


def _load_script() -> ModuleType:
    """Load the script as a module (it has a CLI entry point).

    Robust against the fact that /scripts/tools/ is not on sys.path.
    """
    spec = importlib.util.spec_from_file_location("run_crg_eval", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_crg_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def script() -> ModuleType:
    return _load_script()


class TestLoadConfig:
    def test_live_config_parses(self, script: ModuleType) -> None:
        """The committed configs/canompx3-crg-eval.yaml must be valid."""
        cfg = script._load_config(LIVE_CONFIG)
        assert cfg["name"] == "canompx3"
        assert len(cfg["test_commits"]) >= 1
        assert len(cfg["search_queries"]) >= 1

    def test_missing_required_key_rejected(self, script: ModuleType, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            yaml.safe_dump({"name": "canompx3", "test_commits": []}),  # no search_queries
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as exc:
            script._load_config(bad)
        assert "search_queries" in str(exc.value)

    def test_missing_sha_in_test_commit_rejected(self, script: ModuleType, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            yaml.safe_dump(
                {
                    "name": "x",
                    "test_commits": [{"description": "no sha"}],
                    "search_queries": [],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as exc:
            script._load_config(bad)
        assert "sha" in str(exc.value)

    def test_missing_search_query_fields_rejected(self, script: ModuleType, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            yaml.safe_dump(
                {
                    "name": "x",
                    "test_commits": [],
                    "search_queries": [{"query": "no expected"}],  # missing expected
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as exc:
            script._load_config(bad)
        assert "expected" in str(exc.value)

    def test_nonexistent_path_rejected(self, script: ModuleType, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc:
            script._load_config(tmp_path / "does_not_exist.yaml")
        assert "not found" in str(exc.value).lower()


class TestSummarize:
    def test_halt_ratio_threshold_constant(self, script: ModuleType) -> None:
        """v2 plan halt threshold: <10% savings ⟺ ratio < 1.111."""
        assert abs(script.HALT_RATIO_THRESHOLD - 1.111) < 0.001

    def test_summarize_pass_above_threshold(self, script: ModuleType) -> None:
        results = {
            "token_efficiency": [
                {"naive_to_graph_ratio": 5.0},
                {"naive_to_graph_ratio": 10.0},
                {"naive_to_graph_ratio": 2.0},
            ],
            "search_quality": [{"reciprocal_rank": 1.0}, {"reciprocal_rank": 0.5}],
            "impact_accuracy": [
                {"precision": 0.8, "recall": 0.9},
                {"precision": 0.6, "recall": 0.7},
            ],
        }
        summary = script._summarize(results)
        assert summary["token_efficiency"]["halt_pr4a"] is False
        assert summary["token_efficiency"]["median_naive_to_graph_ratio"] == 5.0
        assert summary["search_quality"]["n_queries"] == 2
        assert summary["search_quality"]["mean_reciprocal_rank"] == 0.75
        assert summary["impact_accuracy"]["avg_precision"] == 0.7
        assert summary["impact_accuracy"]["avg_recall"] == 0.8

    def test_summarize_halt_below_threshold(self, script: ModuleType) -> None:
        results = {
            "token_efficiency": [
                {"naive_to_graph_ratio": 1.05},
                {"naive_to_graph_ratio": 1.0},
                {"naive_to_graph_ratio": 1.08},
            ],
            "search_quality": [],
            "impact_accuracy": [],
        }
        summary = script._summarize(results)
        assert summary["token_efficiency"]["halt_pr4a"] is True

    def test_summarize_empty_results(self, script: ModuleType) -> None:
        """All benchmarks failed → no halt (no signal), no division-by-zero."""
        results: dict = {
            "token_efficiency": [],
            "search_quality": [],
            "impact_accuracy": [],
        }
        summary = script._summarize(results)
        assert summary["token_efficiency"]["halt_pr4a"] is False
        assert summary["search_quality"]["mean_reciprocal_rank"] == 0.0
        assert summary["impact_accuracy"]["avg_precision"] == 0.0


class TestDryRun:
    def test_dry_run_exits_zero(self, script: ModuleType) -> None:
        """--dry-run validates the live config and exits 0."""
        rc = script.main(["--dry-run", "--config", str(LIVE_CONFIG)])
        assert rc == 0

    def test_dry_run_rejects_bad_config(self, script: ModuleType, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("name: x\n", encoding="utf-8")  # missing required keys
        with pytest.raises(SystemExit):
            script.main(["--dry-run", "--config", str(bad)])
