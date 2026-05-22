"""Tests for the derived Fast Lane trial index."""

from __future__ import annotations

from scripts.research.fast_lane_trial_index import build_trial_index


def _row(
    trial_id: str,
    structural_hash: str,
    *,
    run_id: str | None = None,
    instrument: str = "MNQ",
    orb_label: str = "US_DATA_1000",
    orb_minutes: int = 30,
) -> dict:
    return {
        "run_id": run_id or f"runner-{trial_id}",
        "trial_id": trial_id,
        "structural_hash": structural_hash,
        "k_lineage": {
            "instrument": instrument,
            "orb_label": orb_label,
            "orb_minutes": orb_minutes,
        },
    }


def test_trial_index_filters_corrected_rows_before_counting() -> None:
    rows = [
        _row("trial-a", "aaaaaaaaaaaaaaaa", run_id="scanner-old-1"),
        _row("trial-b", "aaaaaaaaaaaaaaaa", run_id="runner-real-1"),
    ]
    corrections = [
        {
            "action": "exclude_from_v2_k_counts",
            "selector": {"run_id_prefix": "scanner-"},
        }
    ]

    index = build_trial_index(rows, corrections)

    assert index["total_v2_trials"] == 1
    assert index["by_structural_hash"]["aaaaaaaaaaaaaaaa"]["K_structural"] == 1
    assert index["by_structural_hash"]["aaaaaaaaaaaaaaaa"]["trial_ids"] == ["trial-b"]


def test_trial_index_counts_lane_scope_deterministically() -> None:
    rows = [
        _row("trial-b", "bbbbbbbbbbbbbbbb", instrument="MES", orb_label="CME_PRECLOSE", orb_minutes=30),
        _row("trial-a", "aaaaaaaaaaaaaaaa", instrument="MNQ", orb_label="US_DATA_1000", orb_minutes=30),
        _row("trial-c", "cccccccccccccccc", instrument="MNQ", orb_label="US_DATA_1000", orb_minutes=30),
    ]

    index = build_trial_index(rows, corrections=[])

    assert list(index["by_lane"]) == ["MES|CME_PRECLOSE|30", "MNQ|US_DATA_1000|30"]
    assert index["by_lane"]["MNQ|US_DATA_1000|30"]["K_lane"] == 2
    assert index["by_lane"]["MNQ|US_DATA_1000|30"]["trial_ids"] == ["trial-a", "trial-c"]
