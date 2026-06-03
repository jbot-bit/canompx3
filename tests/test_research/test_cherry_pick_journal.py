"""Tests for the cherry-pick journal writer + enricher (Improvement 1).

Covers:
- ``oos_power_tier_for`` categorical derivation
- ``build_journal_entry`` shape + rounding
- ``append_journal_entry`` idempotency + iter monotonicity
- Enricher parsing of heavyweight result MDs (title, verdict, IS-row t)
- Enricher mutation rules (skip already-resolved, preserve hand-edited
  lesson_label, derive lesson from verdict + power tier)
- Check #164 catches: missing journal, malformed entries, non-monotonic iter,
  unknown power tier, escalated queue rows without a journal entry
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from scripts.research import cherry_pick_journal_enricher as enricher
from scripts.research import cherry_pick_ranker as cpr

# ---------- Fixtures ----------


def _make_candidate(
    *,
    strategy_id: str = "MNQ_TEST_E1_RR1.0_CB1_TEST_FILTER",
    pooled_t: float = 4.5,
    pooled_n: int = 220,
    pooled_expr: float = 0.18,
    pooling_artifact: bool = False,
    oos_n: int = 40,
    oos_expr: float = 0.10,
    dir_match: bool = True,
    deflation_headroom: float = 0.158,
    n_adequacy: float = 1.0,
    oos_power_readiness: float = 0.65,
    non_artifact: float = 1.0,
    era_stability_proxy: float = 0.0,
    skip_recommended: bool = False,
    result_md: str = "docs/audit/results/2026-05-19-mnq-test-fast-lane-v1.md",
) -> cpr.RankedCandidate:
    breakdown = cpr.ScoreBreakdown(
        deflation_headroom=deflation_headroom,
        n_adequacy=n_adequacy,
        oos_power_readiness=oos_power_readiness,
        dir_match=1.0 if dir_match else 0.0,
        non_artifact=non_artifact,
        era_stability_proxy=era_stability_proxy,
    )
    return cpr.RankedCandidate(
        strategy_id=strategy_id,
        direction="long",
        pooled_t=pooled_t,
        pooled_n=pooled_n,
        pooled_expr=pooled_expr,
        pooling_artifact=pooling_artifact,
        oos_n=oos_n,
        oos_expr=oos_expr,
        dir_match=dir_match,
        score=breakdown,
        skip_recommended=skip_recommended,
        result_md=result_md,
        structural_hash="0123456789abcdef",
        k_lineage={"K_lane": 1, "K_family": 1, "K_global": 1},
        n_hat=pooled_n,
    )


# ---------- oos_power_tier_for ----------


class TestOOSPowerTierFor:
    def test_no_oos_when_n_zero_and_nan(self):
        c = _make_candidate(oos_n=0, oos_expr=float("nan"), oos_power_readiness=0.0)
        assert cpr.oos_power_tier_for(c) == "NA_NO_OOS"

    def test_below_n_floor(self):
        c = _make_candidate(oos_n=14, oos_power_readiness=0.0)
        assert cpr.oos_power_tier_for(c) == "NA_N_BELOW_FLOOR"

    def test_statistically_useless(self):
        c = _make_candidate(oos_n=50, oos_power_readiness=0.30)
        assert cpr.oos_power_tier_for(c) == "STATISTICALLY_USELESS"

    def test_directional_only(self):
        c = _make_candidate(oos_n=50, oos_power_readiness=0.65)
        assert cpr.oos_power_tier_for(c) == "DIRECTIONAL_ONLY"

    def test_can_refute(self):
        c = _make_candidate(oos_n=200, oos_power_readiness=0.90)
        assert cpr.oos_power_tier_for(c) == "CAN_REFUTE"

    def test_boundary_below_floor(self):
        # OOS_N_FLOOR = 30; N=29 is below
        c = _make_candidate(oos_n=cpr.OOS_N_FLOOR - 1, oos_power_readiness=0.99)
        assert cpr.oos_power_tier_for(c) == "NA_N_BELOW_FLOOR"

    def test_boundary_at_floor(self):
        # N=30 passes the floor; power tier dominates from there
        c = _make_candidate(oos_n=cpr.OOS_N_FLOOR, oos_power_readiness=0.49)
        assert cpr.oos_power_tier_for(c) == "STATISTICALLY_USELESS"


# ---------- build_journal_entry / append_journal_entry ----------


class TestJournalAppend:
    def test_build_entry_shape(self):
        c = _make_candidate()
        e = cpr.build_journal_entry(c, iter_num=7, today=date(2026, 5, 20))
        assert e["iter"] == 7
        assert e["date"] == "2026-05-20"
        assert e["strategy_id"] == c.strategy_id
        assert e["pooled_t"] == 4.5
        assert e["pooled_n"] == 220
        assert e["oos_n"] == 40
        assert e["oos_power_tier"] == "DIRECTIONAL_ONLY"
        assert e["heavyweight_verdict"] is None
        assert e["t_observed_post_clustered_se"] is None
        assert e["lesson_label"] is None
        assert set(e["components"]) == {
            "deflation_headroom",
            "n_adequacy",
            "oos_power_readiness",
            "dir_match",
            "non_artifact",
            "era_stability_proxy",
        }

    def test_pooled_t_nan_serializes_as_null(self):
        c = _make_candidate(pooled_t=float("nan"))
        e = cpr.build_journal_entry(c, iter_num=1, today=date(2026, 5, 20))
        assert e["pooled_t"] is None

    def test_append_creates_file_when_missing(self, tmp_path: Path):
        j = tmp_path / "journal.yaml"
        c = _make_candidate()
        entry = cpr.append_journal_entry(j, c, today=date(2026, 5, 20))
        assert j.exists()
        payload = yaml.safe_load(j.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert len(payload["entries"]) == 1
        assert payload["entries"][0]["iter"] == 1
        assert entry["iter"] == 1

    def test_append_is_idempotent_per_strategy_per_day(self, tmp_path: Path):
        j = tmp_path / "journal.yaml"
        c = _make_candidate(strategy_id="MNQ_SAMEDAY")
        e1 = cpr.append_journal_entry(j, c, today=date(2026, 5, 20))
        e2 = cpr.append_journal_entry(j, c, today=date(2026, 5, 20))
        payload = yaml.safe_load(j.read_text(encoding="utf-8"))
        assert len(payload["entries"]) == 1
        assert e1["iter"] == e2["iter"] == 1

    def test_append_increments_iter_across_strategies(self, tmp_path: Path):
        j = tmp_path / "journal.yaml"
        cpr.append_journal_entry(j, _make_candidate(strategy_id="A"), today=date(2026, 5, 20))
        cpr.append_journal_entry(j, _make_candidate(strategy_id="B"), today=date(2026, 5, 20))
        cpr.append_journal_entry(j, _make_candidate(strategy_id="C"), today=date(2026, 5, 21))
        payload = yaml.safe_load(j.read_text(encoding="utf-8"))
        iters = [e["iter"] for e in payload["entries"]]
        assert iters == [1, 2, 3]

    def test_load_malformed_raises(self, tmp_path: Path):
        j = tmp_path / "journal.yaml"
        j.write_text("- not a mapping\n", encoding="utf-8")
        with pytest.raises(ValueError):
            cpr._load_journal(j)


# ---------- Enricher: result MD parser ----------


_FAKE_HEAVYWEIGHT_MD = """\
# Chordia strict unlock audit — MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12

**Prereq file:** `docs/audit/hypotheses/2026-05-02-mnq-comex-costlt12-chordia-unlock-v1.yaml`

## Verdict

**MEASURED verdict:** `PASS_CHORDIA`

IS clears strict threshold 3.79.

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 1494 | 1252 | 83.80% | 5 | 0 | 0.1109 | 0.0930 | 0.1214 | 4.294 | 0.00002 |
| OOS | 72 | 70 | 97.22% | 1 | 0 | 0.0795 | 0.0773 | 0.0839 | 0.702 | 0.48272 |
"""


class TestEnricherParser:
    def test_parses_pass_chordia(self, tmp_path: Path):
        md = tmp_path / "2026-05-02-mnq-comex-chordia-unlock-v1.md"
        md.write_text(_FAKE_HEAVYWEIGHT_MD, encoding="utf-8")
        result = enricher.parse_heavyweight_result(md)
        assert result is not None
        assert result.strategy_id == "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12"
        assert result.verdict == "PASS_CHORDIA"
        assert result.t_clustered == pytest.approx(4.294, abs=1e-6)

    def test_returns_none_when_no_title(self, tmp_path: Path):
        md = tmp_path / "2026-05-02-stray-doc-chordia-unlock-v1.md"
        md.write_text("# Random unrelated doc\n", encoding="utf-8")
        assert enricher.parse_heavyweight_result(md) is None

    def test_returns_none_when_no_verdict(self, tmp_path: Path):
        md = tmp_path / "2026-05-02-strategy-chordia-unlock-v1.md"
        md.write_text(
            "# Chordia strict unlock audit — MNQ_TEST\n\nNo verdict here.\n",
            encoding="utf-8",
        )
        assert enricher.parse_heavyweight_result(md) is None

    def test_filename_pattern_filters_non_heavyweight(self, tmp_path: Path):
        # Sanity: collect_heavyweight_outcomes ignores files that don't match.
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        good = results_dir / "2026-05-02-foo-chordia-unlock-v1.md"
        good.write_text(_FAKE_HEAVYWEIGHT_MD, encoding="utf-8")
        bad = results_dir / "2026-05-02-foo-fast-lane-v1.md"
        bad.write_text(_FAKE_HEAVYWEIGHT_MD, encoding="utf-8")
        outcomes = enricher.collect_heavyweight_outcomes(results_dir)
        # Only the unlock file should land.
        assert len(outcomes) == 1
        sid = next(iter(outcomes))
        assert outcomes[sid].result_md_path.endswith("chordia-unlock-v1.md")


# ---------- Enricher: mutation rules ----------


class TestEnricherMutation:
    def _seed_journal(self, tmp_path: Path) -> Path:
        j = tmp_path / "journal.yaml"
        payload = {
            "schema_version": 1,
            "entries": [
                {
                    "iter": 1,
                    "date": "2026-05-19",
                    "strategy_id": "MNQ_TEST_LANE",
                    "rank_score": 0.50,
                    "components": {
                        "deflation_headroom": 0.0,
                        "n_adequacy": 1.0,
                        "oos_power_readiness": 0.0,
                        "dir_match": 0.0,
                        "non_artifact": 1.0,
                        "era_stability_proxy": 0.0,
                    },
                    "pooled_t": 3.06,
                    "pooled_n": 200,
                    "oos_n": 14,
                    "oos_power_tier": "NA_N_BELOW_FLOOR",
                    "bridge_draft_path": None,
                    "heavyweight_verdict": None,
                    "t_observed_post_clustered_se": None,
                    "lesson_label": None,
                },
            ],
        }
        j.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return j

    def test_fills_pass_chordia_verdict(self, tmp_path: Path):
        j = self._seed_journal(tmp_path)
        journal = yaml.safe_load(j.read_text(encoding="utf-8"))
        outcomes = {
            "MNQ_TEST_LANE": enricher.HeavyweightOutcome(
                strategy_id="MNQ_TEST_LANE",
                verdict="PASS_CHORDIA",
                t_clustered=4.29,
                result_md_path="docs/audit/results/2026-05-02-mnq-test-chordia-unlock-v1.md",
            )
        }
        mutated = enricher.enrich_entries(journal, outcomes)
        assert len(mutated) == 1
        e = journal["entries"][0]
        assert e["heavyweight_verdict"] == "PASS_CHORDIA"
        assert e["t_observed_post_clustered_se"] == 4.29
        assert e["lesson_label"] == "T_HELD_AFTER_CLUSTERED_SE"

    def test_preserves_hand_edited_lesson(self, tmp_path: Path):
        j = self._seed_journal(tmp_path)
        journal = yaml.safe_load(j.read_text(encoding="utf-8"))
        journal["entries"][0]["lesson_label"] = "HAND_EDITED_REASON"
        outcomes = {
            "MNQ_TEST_LANE": enricher.HeavyweightOutcome(
                strategy_id="MNQ_TEST_LANE",
                verdict="PASS_CHORDIA",
                t_clustered=4.29,
                result_md_path="x",
            )
        }
        enricher.enrich_entries(journal, outcomes)
        assert journal["entries"][0]["lesson_label"] == "HAND_EDITED_REASON"

    def test_skips_already_resolved(self, tmp_path: Path):
        j = self._seed_journal(tmp_path)
        journal = yaml.safe_load(j.read_text(encoding="utf-8"))
        journal["entries"][0]["heavyweight_verdict"] = "PASS_CHORDIA"
        outcomes = {
            "MNQ_TEST_LANE": enricher.HeavyweightOutcome(
                strategy_id="MNQ_TEST_LANE",
                verdict="FAIL_STRICT",  # Would conflict if not skipped
                t_clustered=2.0,
                result_md_path="x",
            )
        }
        mutated = enricher.enrich_entries(journal, outcomes)
        assert mutated == []
        assert journal["entries"][0]["heavyweight_verdict"] == "PASS_CHORDIA"

    def test_park_with_underpowered_oos_gets_unverified_lesson(self, tmp_path: Path):
        j = self._seed_journal(tmp_path)
        journal = yaml.safe_load(j.read_text(encoding="utf-8"))
        # Entry has oos_power_tier=NA_N_BELOW_FLOOR; PARK should reflect that.
        outcomes = {
            "MNQ_TEST_LANE": enricher.HeavyweightOutcome(
                strategy_id="MNQ_TEST_LANE",
                verdict="PARK",
                t_clustered=2.0,
                result_md_path="x",
            )
        }
        enricher.enrich_entries(journal, outcomes)
        assert journal["entries"][0]["lesson_label"] == "PARK_BUT_OOS_UNDERPOWERED"


# ---------- Check #164: drift-check injection probes ----------


class TestCheck164Drift:
    def _seed_valid_files(self, tmp_path: Path) -> tuple[Path, Path]:
        j = tmp_path / "journal.yaml"
        q = tmp_path / "promote_queue.yaml"
        j.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "entries": [
                        {
                            "iter": 1,
                            "date": "2026-05-19",
                            "strategy_id": "MNQ_TEST_LANE",
                            "rank_score": 0.5,
                            "components": {
                                "deflation_headroom": 0.0,
                                "n_adequacy": 1.0,
                                "oos_power_readiness": 0.0,
                                "dir_match": 0.0,
                                "non_artifact": 1.0,
                                "era_stability_proxy": 0.0,
                            },
                            "pooled_t": 3.06,
                            "pooled_n": 200,
                            "oos_n": 14,
                            "oos_power_tier": "NA_N_BELOW_FLOOR",
                            "bridge_draft_path": None,
                            "heavyweight_verdict": None,
                            "t_observed_post_clustered_se": None,
                            "lesson_label": None,
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        q.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "entries": [
                        {
                            "strategy_id": "MNQ_TEST_LANE",
                            "status": "QUEUED",
                            "heavyweight_prereg": None,
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return j, q

    def test_clean_state_passes(self, tmp_path: Path):
        from pipeline.check_drift import check_cherry_pick_journal_integrity

        j, q = self._seed_valid_files(tmp_path)
        violations = check_cherry_pick_journal_integrity(journal_path=j, queue_path=q)
        assert violations == []

    def test_missing_journal_fails_closed(self, tmp_path: Path):
        from pipeline.check_drift import check_cherry_pick_journal_integrity

        j = tmp_path / "nonexistent.yaml"
        q = tmp_path / "queue.yaml"
        q.write_text(yaml.safe_dump({"entries": []}), encoding="utf-8")
        violations = check_cherry_pick_journal_integrity(journal_path=j, queue_path=q)
        assert any("journal missing" in v for v in violations)

    def test_unknown_power_tier_is_caught(self, tmp_path: Path):
        # INJECTION PROBE: replace tier with bogus value, assert check catches it.
        from pipeline.check_drift import check_cherry_pick_journal_integrity

        j, q = self._seed_valid_files(tmp_path)
        payload = yaml.safe_load(j.read_text(encoding="utf-8"))
        payload["entries"][0]["oos_power_tier"] = "BOGUS_TIER"
        j.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        violations = check_cherry_pick_journal_integrity(journal_path=j, queue_path=q)
        assert any("oos_power_tier" in v and "BOGUS_TIER" in v for v in violations)

    def test_non_monotonic_iter_is_caught(self, tmp_path: Path):
        # INJECTION PROBE: skip iter 2, assert check catches the gap.
        from pipeline.check_drift import check_cherry_pick_journal_integrity

        j, q = self._seed_valid_files(tmp_path)
        payload = yaml.safe_load(j.read_text(encoding="utf-8"))
        second = dict(payload["entries"][0])
        second["iter"] = 3  # Should be 2
        second["strategy_id"] = "MNQ_NEXT_LANE"
        second["date"] = "2026-05-20"
        payload["entries"].append(second)
        j.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        violations = check_cherry_pick_journal_integrity(journal_path=j, queue_path=q)
        assert any("iter=3" in v and "expected 2" in v for v in violations)

    def test_escalated_queue_without_journal_entry_is_caught(self, tmp_path: Path):
        # INJECTION PROBE: queue row with heavyweight_prereg set but no journal -> violation.
        from pipeline.check_drift import check_cherry_pick_journal_integrity

        j, q = self._seed_valid_files(tmp_path)
        queue_payload = yaml.safe_load(q.read_text(encoding="utf-8"))
        queue_payload["entries"].append(
            {
                "strategy_id": "MNQ_ORPHAN_ESCALATION",
                "status": "ESCALATED",
                "heavyweight_prereg": "docs/audit/hypotheses/2026-05-19-orphan.yaml",
            }
        )
        q.write_text(yaml.safe_dump(queue_payload, sort_keys=False), encoding="utf-8")
        violations = check_cherry_pick_journal_integrity(journal_path=j, queue_path=q)
        assert any("MNQ_ORPHAN_ESCALATION" in v for v in violations)

    def test_revoked_queue_with_heavyweight_prereg_is_exempt(self, tmp_path: Path):
        from pipeline.check_drift import check_cherry_pick_journal_integrity

        j, q = self._seed_valid_files(tmp_path)
        queue_payload = yaml.safe_load(q.read_text(encoding="utf-8"))
        queue_payload["entries"].append(
            {
                "strategy_id": "MNQ_REVOKED_AFTER_BRIDGE",
                "status": "REVOKED",
                "heavyweight_prereg": "docs/audit/hypotheses/2026-05-19-revoked.yaml",
            }
        )
        q.write_text(yaml.safe_dump(queue_payload, sort_keys=False), encoding="utf-8")
        violations = check_cherry_pick_journal_integrity(journal_path=j, queue_path=q)
        assert violations == []

    def test_missing_required_field_is_caught(self, tmp_path: Path):
        # INJECTION PROBE: drop the components field, assert check catches it.
        from pipeline.check_drift import check_cherry_pick_journal_integrity

        j, q = self._seed_valid_files(tmp_path)
        payload = yaml.safe_load(j.read_text(encoding="utf-8"))
        del payload["entries"][0]["components"]
        j.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        violations = check_cherry_pick_journal_integrity(journal_path=j, queue_path=q)
        assert any("missing required field 'components'" in v for v in violations)
