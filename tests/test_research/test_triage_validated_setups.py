"""Tests for the triage_validated_setups script (Improvement 3 / Stage C).

Covers:
- Score components (pooled_t_headroom, n_adequacy, oos_power_readiness,
  era_stability, non_artifact) and the total_score aggregator
- ``collect_seen_strategy_ids`` filtering (only result MDs count)
- ``build_draft_yaml`` shape contract (required scope fields + provenance)
- ``_slugify_strategy_id`` produces filename-safe slugs matching the existing
  fast-lane MD slug convention
- ``write_draft`` round-trip + overwrite protection
- Check #165 catches: drafts with triage_provenance but no
  source_validated_setup_strategy_id; drafts with non-mapping
  triage_provenance; clean drafts (no marker) pass; hand-authored drafts
  WITH source_validated_setup_strategy_id pass
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from scripts.research import triage_validated_setups as triage

# ---------- Fixtures ----------


def _make_row(**overrides) -> dict:
    base = {
        "strategy_id": "MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12_O15",
        "instrument": "MNQ",
        "orb_label": "TOKYO_OPEN",
        "orb_minutes": 15,
        "entry_model": "E2",
        "confirm_bars": 1,
        "rr_target": 1.5,
        "filter_type": "COST_LT12",
        "sample_size": 220,
        "expectancy_r": 0.18,
        "sharpe_ratio": 0.30,
        "years_tested": 6,
        "all_years_positive": True,
    }
    base.update(overrides)
    return base


# ---------- Score components ----------


class TestPooledTHeadroom:
    def test_zero_when_below_threshold(self):
        # sharpe=0.1, n=100 -> t=1.0 < 2.5
        assert triage.compute_pooled_t_headroom(0.1, 100) == 0.0

    def test_zero_at_zero_n(self):
        assert triage.compute_pooled_t_headroom(0.3, 0) == 0.0

    def test_positive_when_above_threshold(self):
        # sharpe=0.4, n=100 -> t=4.0 > 2.5 -> headroom = 1.5/4.0 = 0.375
        h = triage.compute_pooled_t_headroom(0.4, 100)
        assert 0.37 < h < 0.38

    def test_zero_on_nan(self):
        assert triage.compute_pooled_t_headroom(float("nan"), 100) == 0.0


class TestNAdequacy:
    def test_zero_when_n_zero(self):
        assert triage.compute_n_adequacy(0) == 0.0

    def test_linear_below_target(self):
        assert triage.compute_n_adequacy(100) == pytest.approx(0.5, abs=1e-6)

    def test_capped_at_one(self):
        assert triage.compute_n_adequacy(triage.N_ADEQUACY_TARGET * 5) == 1.0


class TestOOSPowerReadiness:
    def test_zero_below_floor(self):
        score, raw = triage.compute_oos_power_readiness(0.3, 200, 10)
        assert score == 0.0
        assert raw == 0.0

    def test_returns_bounded_power_above_floor(self):
        score, raw = triage.compute_oos_power_readiness(0.4, 200, 100)
        assert 0.0 <= score <= 1.0
        # Sharpe 0.4 with N=100 OOS should yield substantial power
        assert score > 0.5

    def test_zero_on_nan_sharpe(self):
        score, _ = triage.compute_oos_power_readiness(float("nan"), 200, 100)
        assert score == 0.0


class TestEraStability:
    def test_one_when_all_positive_and_enough_years(self):
        assert triage.compute_era_stability(6, True) == 1.0

    def test_zero_when_too_few_years(self):
        assert triage.compute_era_stability(3, True) == 0.0

    def test_zero_when_not_all_positive(self):
        assert triage.compute_era_stability(7, False) == 0.0


class TestNonArtifact:
    def test_one_for_unflagged_filter(self):
        assert triage.compute_non_artifact("COST_LT12") == 1.0


class TestScoreCandidate:
    def test_total_score_aggregator(self):
        c = triage.score_candidate(_make_row(), oos_n=80)
        # Weights sum to 1, no component above 1 — total bounded [0, 1].
        assert 0.0 <= c.total_score <= 1.0
        # With sharpe=0.30 + n=220, t=4.45 (above 2.5) + n_adq=1 + era=1 + non_art=1,
        # we expect a decent total above the SCORE_SKIP_FLOOR.
        assert c.total_score > 0.5

    def test_weights_sum_to_one(self):
        assert sum(triage.SCORE_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-9)


# ---------- collect_seen_strategy_ids ----------


class TestCollectSeenStrategyIDs:
    def test_empty_dir_returns_empty_set(self, tmp_path: Path):
        d = tmp_path / "results"
        d.mkdir()
        assert triage.collect_seen_strategy_ids(d) == set()

    def test_extracts_strategy_id_from_title(self, tmp_path: Path):
        d = tmp_path / "results"
        d.mkdir()
        (d / "2026-05-18-foo-fast-lane-v1.md").write_text(
            "# Chordia strict unlock audit — MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12_O15\n\nbody\n",
            encoding="utf-8",
        )
        seen = triage.collect_seen_strategy_ids(d)
        assert "MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12_O15" in seen

    def test_ignores_files_without_title(self, tmp_path: Path):
        d = tmp_path / "results"
        d.mkdir()
        (d / "2026-05-18-unrelated.md").write_text("# Unrelated audit\n\nbody\n", encoding="utf-8")
        assert triage.collect_seen_strategy_ids(d) == set()


# ---------- build_draft_yaml shape ----------


class TestBuildDraftYAML:
    def test_carries_required_fields(self):
        c = triage.score_candidate(_make_row(), oos_n=80)
        text = triage.build_draft_yaml(c, today=date(2026, 5, 19))
        parsed = yaml.safe_load(text)
        assert parsed["metadata"]["template_version"] == "fast_lane_v5.1"
        assert parsed["metadata"]["n_trials"] == 1
        assert parsed["triage_provenance"]["source_validated_setup_strategy_id"] == c.strategy_id
        assert parsed["scope"]["instrument"] == "MNQ"
        assert parsed["scope"]["session"] == "TOKYO_OPEN"
        assert parsed["scope"]["orb_minutes"] == 15
        assert parsed["holdout"]["holdout_date"] == "2026-01-01"
        assert parsed["data_policy"]["scratch_policy"] == "realized-eod"

    def test_score_components_round_trip(self):
        c = triage.score_candidate(_make_row(), oos_n=80)
        text = triage.build_draft_yaml(c, today=date(2026, 5, 19))
        parsed = yaml.safe_load(text)
        comps = parsed["triage_provenance"]["score_components"]
        for key in triage.SCORE_WEIGHTS:
            assert key in comps


# ---------- _slugify_strategy_id ----------


class TestSlugify:
    def test_collapses_dots_in_rr(self):
        slug = triage._slugify_strategy_id("MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12_O15")
        assert "1.5" not in slug
        assert "rr15" in slug

    def test_lowercase_and_hyphenated(self):
        slug = triage._slugify_strategy_id("MES_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_O30")
        assert slug == "mes-cme-preclose-e2-rr10-cb1-atr-p30-o30"


# ---------- write_draft round-trip ----------


class TestWriteDraft:
    def test_writes_and_reads_back(self, tmp_path: Path):
        c = triage.score_candidate(_make_row(), oos_n=80)
        path = triage.write_draft(c, drafts_dir=tmp_path, today=date(2026, 5, 19))
        assert path.exists()
        assert path.name.endswith(".draft.yaml")
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert parsed["triage_provenance"]["source_validated_setup_strategy_id"] == c.strategy_id

    def test_refuses_overwrite_by_default(self, tmp_path: Path):
        c = triage.score_candidate(_make_row(), oos_n=80)
        triage.write_draft(c, drafts_dir=tmp_path, today=date(2026, 5, 19))
        with pytest.raises(FileExistsError):
            triage.write_draft(c, drafts_dir=tmp_path, today=date(2026, 5, 19))

    def test_overwrite_flag(self, tmp_path: Path):
        c = triage.score_candidate(_make_row(), oos_n=80)
        triage.write_draft(c, drafts_dir=tmp_path, today=date(2026, 5, 19))
        # Should not raise
        triage.write_draft(c, drafts_dir=tmp_path, today=date(2026, 5, 19), overwrite=True)


# ---------- Check #165 drift-check injection probes ----------


class TestCheck165Drift:
    def test_clean_state_no_drafts_passes(self, tmp_path: Path):
        from pipeline.check_drift import check_triage_provenance_completeness

        d = tmp_path / "drafts"
        # Directory does not exist -- fresh-tree state.
        violations = check_triage_provenance_completeness(drafts_dir=d)
        assert violations == []

    def test_empty_drafts_dir_passes(self, tmp_path: Path):
        from pipeline.check_drift import check_triage_provenance_completeness

        d = tmp_path / "drafts"
        d.mkdir()
        violations = check_triage_provenance_completeness(drafts_dir=d)
        assert violations == []

    def test_draft_without_provenance_block_passes(self, tmp_path: Path):
        from pipeline.check_drift import check_triage_provenance_completeness

        d = tmp_path / "drafts"
        d.mkdir()
        (d / "hand-authored.yaml").write_text("metadata:\n  name: hand_authored\n", encoding="utf-8")
        violations = check_triage_provenance_completeness(drafts_dir=d)
        assert violations == []

    def test_triage_draft_with_valid_provenance_passes(self, tmp_path: Path):
        from pipeline.check_drift import check_triage_provenance_completeness

        d = tmp_path / "drafts"
        d.mkdir()
        c = triage.score_candidate(_make_row(), oos_n=80)
        triage.write_draft(c, drafts_dir=d, today=date(2026, 5, 19))
        violations = check_triage_provenance_completeness(drafts_dir=d)
        assert violations == []

    def test_triage_draft_missing_strategy_id_is_caught(self, tmp_path: Path):
        # INJECTION PROBE: declare a triage_provenance block without
        # source_validated_setup_strategy_id, assert check catches it.
        from pipeline.check_drift import check_triage_provenance_completeness

        d = tmp_path / "drafts"
        d.mkdir()
        (d / "broken.draft.yaml").write_text(
            "metadata:\n  name: broken\ntriage_provenance:\n"
            "  source: 'scripts/research/triage_validated_setups.py'\n"
            "  rank_score: 0.7\n",
            encoding="utf-8",
        )
        violations = check_triage_provenance_completeness(drafts_dir=d)
        assert any("broken.draft.yaml" in v and "source_validated_setup_strategy_id" in v for v in violations)

    def test_triage_draft_with_empty_strategy_id_is_caught(self, tmp_path: Path):
        from pipeline.check_drift import check_triage_provenance_completeness

        d = tmp_path / "drafts"
        d.mkdir()
        (d / "empty.draft.yaml").write_text(
            "triage_provenance:\n  source_validated_setup_strategy_id: ''\n",
            encoding="utf-8",
        )
        violations = check_triage_provenance_completeness(drafts_dir=d)
        assert any("empty.draft.yaml" in v for v in violations)

    def test_non_mapping_provenance_block_is_caught(self, tmp_path: Path):
        from pipeline.check_drift import check_triage_provenance_completeness

        d = tmp_path / "drafts"
        d.mkdir()
        (d / "weird.draft.yaml").write_text("triage_provenance: 'not a mapping'\n", encoding="utf-8")
        violations = check_triage_provenance_completeness(drafts_dir=d)
        assert any("weird.draft.yaml" in v and "mapping" in v for v in violations)
