"""Tests for external strategy intake validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.tools import external_strategy_intake_check as intake


def _write_record(tmp_path: Path, record: dict) -> Path:
    path = tmp_path / "intake.yaml"
    path.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
    return path


def _base_record(**overrides: object) -> dict:
    record: dict = {
        "source": {
            "title": "Outside idea",
            "url_or_path": "https://example.com/idea",
            "source_type": "video",
            "reviewed_date": "2026-05-11",
            "authority_level": "low",
        },
        "mechanism_family": "Failed breakout / reclaim after a liquidity sweep.",
        "packaging_removed": ["marketing performance claim", "example ticker anchoring"],
        "repo_coverage": "adjacent",
        "best_role": "filter",
        "baseline_to_beat": "Plain ORB direction and prior-day-level interaction.",
        "decision": "DOC_ONLY",
        "bias_risks": ["lookahead", "cherry_pick", "multiplicity"],
        "negative_evidence": ["Fails versus the plain baseline after costs."],
        "golden_nuggets": ["Require trial count before optimizer claims are trusted."],
        "next_action": "Keep as process note only; no prereg yet.",
        "evidence_refs": ["docs/audit/results/2026-05-11-tradingview-ai-backtesting-engine-teardown.md"],
    }
    record.update(overrides)
    return record


def _errors_for(tmp_path: Path, record: dict) -> list[str]:
    return intake.validate_file(_write_record(tmp_path, record))


def test_valid_bin_record_passes(tmp_path: Path) -> None:
    record = _base_record(decision="BIN", next_action="Close; no follow-up.")
    assert _errors_for(tmp_path, record) == []


def test_valid_doc_only_record_passes(tmp_path: Path) -> None:
    assert _errors_for(tmp_path, _base_record()) == []


def test_valid_prereg_candidate_passes(tmp_path: Path) -> None:
    record = _base_record(
        decision="PREREG_CANDIDATE",
        trial_budget={
            "max_trials": 24,
            "mode": "clean",
            "source_trial_count_disclosed": 12,
        },
        kill_criteria=[
            "Kill if net expectancy <= 0.20R after costs.",
            "Kill if OOS trade count < 80.",
        ],
        oos_policy="Single locked holdout; no iterative OOS feedback.",
        optimization_space={
            "parameters": [
                {"name": "reclaim_bars", "values": [1, 2, 3]},
                {"name": "sweep_distance_r", "values": [0.25, 0.5]},
            ],
            "constraints": ["reclaim_bars <= 3"],
        },
        stability_surface_required=True,
    )

    assert _errors_for(tmp_path, record) == []


def test_missing_source_metadata_fails(tmp_path: Path) -> None:
    record = _base_record(source={"title": "Too thin"})
    errors = _errors_for(tmp_path, record)
    assert any("source.url_or_path" in error for error in errors)


def test_missing_baseline_fails(tmp_path: Path) -> None:
    record = _base_record(baseline_to_beat="")
    errors = _errors_for(tmp_path, record)
    assert any("baseline_to_beat" in error for error in errors)


def test_missing_evidence_refs_fails(tmp_path: Path) -> None:
    record = _base_record(evidence_refs=[])
    errors = _errors_for(tmp_path, record)
    assert any("evidence_refs must be a non-empty list" in error for error in errors)


def test_list_fields_must_be_lists_not_strings(tmp_path: Path) -> None:
    record = _base_record(bias_risks="lookahead, multiplicity")
    errors = _errors_for(tmp_path, record)
    assert any("bias_risks must be a non-empty list" in error for error in errors)


def test_unknown_repo_coverage_fails(tmp_path: Path) -> None:
    record = _base_record(repo_coverage="sounds_new")
    errors = _errors_for(tmp_path, record)
    assert any("repo_coverage must be one of" in error for error in errors)


def test_prereg_candidate_requires_trial_budget(tmp_path: Path) -> None:
    record = _base_record(decision="PREREG_CANDIDATE")
    errors = _errors_for(tmp_path, record)
    assert any("trial_budget.max_trials" in error for error in errors)


def test_prereg_candidate_rejects_vague_kill_criteria(tmp_path: Path) -> None:
    record = _base_record(
        decision="PREREG_CANDIDATE",
        trial_budget={"max_trials": 12, "mode": "clean"},
        kill_criteria=["Kill if it looks weak."],
        oos_policy="Single locked holdout; no iterative OOS feedback.",
    )
    errors = _errors_for(tmp_path, record)
    assert any("numeric threshold kill_criteria" in error for error in errors)


def test_prereg_candidate_rejects_number_without_threshold_kill_criteria(tmp_path: Path) -> None:
    record = _base_record(
        decision="PREREG_CANDIDATE",
        trial_budget={"max_trials": 12, "mode": "clean"},
        kill_criteria=["Kill during the 2026 review if it feels weak."],
        oos_policy="Single locked holdout; no iterative OOS feedback.",
    )
    errors = _errors_for(tmp_path, record)
    assert any("numeric threshold kill_criteria" in error for error in errors)


def test_prereg_candidate_requires_clean_or_proxy_budget_mode(tmp_path: Path) -> None:
    record = _base_record(
        decision="PREREG_CANDIDATE",
        trial_budget={"max_trials": 12, "mode": "not_applicable"},
        kill_criteria=["Kill if net expectancy <= 0.10R."],
        oos_policy="Single locked holdout; no iterative OOS feedback.",
    )
    errors = _errors_for(tmp_path, record)
    assert any("trial_budget.mode" in error for error in errors)


def test_prereg_candidate_cannot_be_visualization_or_execution_only(tmp_path: Path) -> None:
    record = _base_record(
        decision="PREREG_CANDIDATE",
        best_role="visualization_aid",
        trial_budget={"max_trials": 12, "mode": "clean"},
        kill_criteria=["Kill if net expectancy <= 0.10R."],
        oos_policy="Single locked holdout; no iterative OOS feedback.",
    )
    errors = _errors_for(tmp_path, record)
    assert any("PREREG_CANDIDATE best_role" in error for error in errors)


def test_iterative_oos_wording_fails(tmp_path: Path) -> None:
    record = _base_record(
        decision="PREREG_CANDIDATE",
        trial_budget={"max_trials": 12, "mode": "clean"},
        kill_criteria=["Kill if net expectancy <= 0.10R."],
        oos_policy="Use OOS, then tune again if needed.",
    )
    errors = _errors_for(tmp_path, record)
    assert any("iterative OOS" in error for error in errors)


def test_holdout_selection_wording_fails(tmp_path: Path) -> None:
    record = _base_record(
        decision="PREREG_CANDIDATE",
        trial_budget={"max_trials": 12, "mode": "clean"},
        kill_criteria=["Kill if net expectancy <= 0.10R."],
        oos_policy="Use holdout to select the best variant.",
    )
    errors = _errors_for(tmp_path, record)
    assert any("iterative OOS" in error for error in errors)


def test_optimizer_claim_requires_source_trial_count(tmp_path: Path) -> None:
    record = _base_record(
        source_claims={"optimizer_variants": 6713},
        trial_budget={"max_trials": 12, "mode": "clean"},
    )
    errors = _errors_for(tmp_path, record)
    assert any("source_trial_count_disclosed" in error for error in errors)


def test_multi_param_space_requires_constraints_and_stability_surface(tmp_path: Path) -> None:
    record = _base_record(
        decision="PREREG_CANDIDATE",
        trial_budget={"max_trials": 12, "mode": "clean"},
        kill_criteria=["Kill if net expectancy <= 0.10R."],
        oos_policy="Single locked holdout; no iterative OOS feedback.",
        optimization_space={
            "parameters": [
                {"name": "fast", "values": [10, 20]},
                {"name": "slow", "values": [50, 100]},
            ],
        },
    )
    errors = _errors_for(tmp_path, record)
    assert any("optimization_space.constraints" in error for error in errors)
    assert any("stability_surface_required" in error for error in errors)


def test_pine_import_requires_pine_risk_flags(tmp_path: Path) -> None:
    record = _base_record(source={**_base_record()["source"], "source_type": "pine_script"})
    errors = _errors_for(tmp_path, record)
    assert any("pine_risk_flags" in error for error in errors)


def test_handoff_and_memory_are_not_valid_evidence(tmp_path: Path) -> None:
    record = _base_record(evidence_refs=["HANDOFF.md", "memory/2026-05-11.md"])
    errors = _errors_for(tmp_path, record)
    assert any("HANDOFF.md" in error for error in errors)
    assert any("memory/" in error for error in errors)


def test_screenshots_are_not_valid_evidence(tmp_path: Path) -> None:
    record = _base_record(evidence_refs=["screenshots/backtest.png"])
    errors = _errors_for(tmp_path, record)
    assert any("screenshots" in error for error in errors)
