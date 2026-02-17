"""Tests for the full pipeline orchestrator step registry."""
import sys
from pathlib import Path

import pytest

def test_step_registry_is_ordered():
    """Full pipeline steps must be in correct dependency order."""
    from pipeline.run_full_pipeline import FULL_PIPELINE_STEPS

    step_names = [s[0] for s in FULL_PIPELINE_STEPS]
    expected_order = [
        "ingest", "build_5m", "build_features", "audit",
        "build_outcomes", "discover", "validate",
    ]
    assert step_names == expected_order

def test_step_functions_are_callable():
    """Every step in the registry must be a callable."""
    from pipeline.run_full_pipeline import FULL_PIPELINE_STEPS

    for name, desc, func in FULL_PIPELINE_STEPS:
        assert callable(func), f"Step {name} is not callable"

def test_dry_run_does_not_execute(capsys):
    """--dry-run should print plan without executing."""
    from pipeline.run_full_pipeline import print_dry_run, FULL_PIPELINE_STEPS

    print_dry_run(FULL_PIPELINE_STEPS, "MGC")
    captured = capsys.readouterr()
    assert "build_outcomes" in captured.out
    assert "discover" in captured.out
    assert "validate" in captured.out

def test_skip_to_works():
    """--skip-to should skip steps before the named step."""
    from pipeline.run_full_pipeline import get_steps_from, FULL_PIPELINE_STEPS

    steps = get_steps_from(FULL_PIPELINE_STEPS, "build_outcomes")
    names = [s[0] for s in steps]
    assert names == ["build_outcomes", "discover", "validate"]
    assert "ingest" not in names

def test_skip_to_invalid_raises():
    """--skip-to with invalid step name should raise ValueError."""
    from pipeline.run_full_pipeline import get_steps_from, FULL_PIPELINE_STEPS

    with pytest.raises(ValueError, match="Unknown step"):
        get_steps_from(FULL_PIPELINE_STEPS, "nonexistent_step")

def test_skip_to_first_step_returns_all():
    """--skip-to with first step should return all steps."""
    from pipeline.run_full_pipeline import get_steps_from, FULL_PIPELINE_STEPS

    steps = get_steps_from(FULL_PIPELINE_STEPS, "ingest")
    assert len(steps) == len(FULL_PIPELINE_STEPS)
