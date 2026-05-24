"""Bridge pre-flight refusal tests (Stage 2A.3).

Four refusal triggers, one test each. Each test:

  1. Builds a synthetic FastLaneSource (result_md path + scope dict).
  2. Seeds promote_queue.yaml / graveyard_digest.yaml such that exactly
     ONE refusal rule fires.
  3. Calls ``build_heavyweight_prereg`` and asserts ``BridgeRefused``.
  4. Asserts ``drafts/<slug>.draft.yaml`` does NOT exist after the call
     (file-state before + after).

The OOS-power gate stays in the scanner; the bridge refuses anything other
than QUEUED, so the bridge does not re-run the OOS power calculation here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.research.fast_lane_structural_hash import compute_structural_hash
from scripts.research.fast_lane_to_heavyweight_bridge import (
    BridgeRefused,
    FastLaneSource,
    _bridge_preflight_refuse,
    draft_path_for,
)


def _bridge_refused_type() -> type[BridgeRefused]:
    """Return the current module exception class after any importlib.reload."""
    return _bridge_preflight_refuse.__globals__["BridgeRefused"]


def _make_source(
    *,
    strategy_id: str = "MGC_LONDON_METALS_E1_RR1.0_CB2_ATR_P50",
    instrument: str = "MGC",
    session: str = "LONDON_METALS",
    orb_minutes: int = 30,
    entry_model: str = "E1",
    confirm_bars: int = 2,
    rr_target: float = 1.0,
    direction: str = "long",
    filter_type: str = "ATR_P50",
) -> FastLaneSource:
    return FastLaneSource(
        result_md_rel=f"docs/audit/results/synthetic-{strategy_id.lower()}.md",
        source_yaml_rel=(f"docs/audit/hypotheses/synthetic-{strategy_id.lower()}.yaml"),
        scope={
            "instrument": instrument,
            "strategy_id": strategy_id,
            "session": session,
            "orb_minutes": orb_minutes,
            "entry_model": entry_model,
            "confirm_bars": confirm_bars,
            "rr_target": rr_target,
            "direction": direction,
            "filter_type": filter_type,
        },
    )


def _scope_to_hash_inputs(scope: dict[str, Any]) -> dict[str, Any]:
    raw_dir = scope.get("direction", "BOTH")
    if isinstance(raw_dir, str) and raw_dir.strip().lower() == "pooled":
        raw_dir = "BOTH"
    return {
        "instrument": scope["instrument"],
        "orb_label": scope["session"],
        "orb_minutes": int(scope["orb_minutes"]),
        "rr_target": float(scope["rr_target"]),
        "entry_model": scope["entry_model"],
        "confirm_bars": int(scope["confirm_bars"]),
        "filter_type": scope.get("filter_type", ""),
        "direction": raw_dir,
        "filter_threshold": scope.get("filter_threshold", ""),
    }


def _make_promote_queue(
    tmp_path: Path,
    *,
    strategy_id: str,
    structural_hash: str,
    k_lane: int,
    k_declared: int = 1,
    status: str = "QUEUED",
    include_k_lineage: bool = True,
) -> Path:
    """Emit a synthetic promote_queue.yaml the bridge pre-flight can read."""
    entry: dict[str, Any] = {
        "strategy_id": strategy_id,
        "status": status,
        "structural_hash": structural_hash,
        "n_hat": 50,
    }
    if include_k_lineage:
        entry["k_lineage"] = {
            "K_lane": k_lane,
            "K_family": k_lane,
            "K_global": k_lane,
            "K_declared_in_prereg": k_declared,
            "K_effective_minBTL": 0.0,
            "bh_fdr_passes": {
                "K_family": True,
                "K_lane": True,
                "K_global": True,
            },
            "correlation_haircut_N_hat": 50,
            "rho_hat_assumed": 0.5,
        }
    payload = {
        "schema_version": 1,
        "warning": (
            "DERIVED STATE - do not hand-edit. Rebuilt from result MDs + "
            "revocation sidecars + heavyweight preregs + action-queue.yaml "
            "on every scanner run. Drift check #157 reconstructs and diffs."
        ),
        "entries": [entry],
    }
    qp = tmp_path / "promote_queue.yaml"
    qp.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return qp


def _make_digest(tmp_path: Path, entries: list[dict[str, Any]]) -> Path:
    dp = tmp_path / "fast_lane_graveyard_digest.yaml"
    dp.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "do_not_hand_edit": True,
                "entries": entries,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return dp


@pytest.fixture
def isolate_drafts(tmp_path, monkeypatch):
    """Redirect DRAFTS_DIR at module level so a leak would be visible
    under tmp_path/drafts, not the real repo drafts/."""
    import scripts.research.fast_lane_to_heavyweight_bridge as bridge_mod

    monkeypatch.setattr(bridge_mod, "DRAFTS_DIR", tmp_path / "drafts", raising=True)
    monkeypatch.setattr(
        bridge_mod,
        "HYPOTHESES_DIR",
        tmp_path / "hypotheses",
        raising=True,
    )
    (tmp_path / "drafts").mkdir(parents=True, exist_ok=True)
    yield


# ---- 1. banned entry_model (E0 / E3) -------------------------------------


def test_bridge_refuses_banned_entry_model(tmp_path, isolate_drafts):
    """E3 entry refuses; no draft on disk."""
    source = _make_source(
        strategy_id="MGC_LONDON_METALS_E3_RR1.0_CB1_ATR_P50",
        entry_model="E3",
        confirm_bars=1,
    )
    expected_draft = tmp_path / "drafts" / draft_path_for(source.scope["strategy_id"], "2026-05-20").name

    assert not expected_draft.exists()

    with pytest.raises(_bridge_refused_type(), match="graveyard-banned"):
        _bridge_preflight_refuse(
            source,
            queue_path=tmp_path / "nonexistent_queue.yaml",
            digest_path=tmp_path / "nonexistent_digest.yaml",
        )

    assert not expected_draft.exists()


# ---- 2. E2 + canonical lookahead filter ----------------------------------


def test_bridge_refuses_e2_lookahead(tmp_path, isolate_drafts):
    """E2 entry with a canonical excluded filter prefix refuses."""
    from trading_app.config import ALL_FILTERS, E2_EXCLUDED_FILTER_PREFIXES

    filter_type = next(f for f in sorted(ALL_FILTERS) if any(f.startswith(p) for p in E2_EXCLUDED_FILTER_PREFIXES))
    source = _make_source(
        strategy_id=f"MGC_LONDON_METALS_E2_RR1.0_CB1_{filter_type}",
        entry_model="E2",
        confirm_bars=1,
        filter_type=filter_type,
    )
    expected_draft = tmp_path / "drafts" / draft_path_for(source.scope["strategy_id"], "2026-05-20").name

    assert not expected_draft.exists()

    with pytest.raises(_bridge_refused_type(), match="E2_EXCLUDED_FILTER_PREFIXES"):
        _bridge_preflight_refuse(
            source,
            queue_path=tmp_path / "nonexistent_queue.yaml",
            digest_path=tmp_path / "nonexistent_digest.yaml",
        )

    assert not expected_draft.exists()


# ---- 3. graveyard lane-level hash match ----------------------------------


def test_bridge_refuses_graveyard_lane(tmp_path, isolate_drafts):
    """A graveyard lane-level structural_hash match refuses."""
    source = _make_source()
    structural_hash = compute_structural_hash(_scope_to_hash_inputs(source.scope))

    digest_path = _make_digest(
        tmp_path,
        [
            {
                "source_path": "chatgpt_bundle/06_RD_GRAVEYARD.md",
                "title": "Synthetic graveyard lane match for tests",
                "status": "DEAD",
                "hash_kind": "lane",
                "structural_hash": structural_hash,
                "lane_inputs": {},
            }
        ],
    )
    queue_path = _make_promote_queue(
        tmp_path,
        strategy_id=source.scope["strategy_id"],
        structural_hash=structural_hash,
        k_lane=0,
    )
    expected_draft = tmp_path / "drafts" / draft_path_for(source.scope["strategy_id"], "2026-05-20").name

    assert not expected_draft.exists()

    with pytest.raises(_bridge_refused_type(), match="graveyard"):
        _bridge_preflight_refuse(source, queue_path=queue_path, digest_path=digest_path)

    assert not expected_draft.exists()


# ---- 4. bridge status + provenance + K-overrun gates ----------------------


def test_bridge_refuses_non_queued_promote_status(tmp_path, isolate_drafts):
    """Only QUEUED promote rows can produce drafts."""
    source = _make_source()
    structural_hash = compute_structural_hash(_scope_to_hash_inputs(source.scope))

    queue_path = _make_promote_queue(
        tmp_path,
        strategy_id=source.scope["strategy_id"],
        structural_hash=structural_hash,
        k_lane=0,
        status="REJECTED_OOS_UNPOWERED",
    )
    expected_draft = tmp_path / "drafts" / draft_path_for(source.scope["strategy_id"], "2026-05-20").name

    assert not expected_draft.exists()

    with pytest.raises(_bridge_refused_type(), match="status='REJECTED_OOS_UNPOWERED'.*expected QUEUED"):
        _bridge_preflight_refuse(
            source,
            queue_path=queue_path,
            digest_path=tmp_path / "empty_digest.yaml",
        )

    assert not expected_draft.exists()


def test_bridge_refuses_missing_v2_k_lineage(tmp_path, isolate_drafts):
    """Bridge must not author drafts from queue rows missing V2 provenance."""
    source = _make_source()
    structural_hash = compute_structural_hash(_scope_to_hash_inputs(source.scope))
    queue_path = _make_promote_queue(
        tmp_path,
        strategy_id=source.scope["strategy_id"],
        structural_hash=structural_hash,
        k_lane=0,
        include_k_lineage=False,
    )

    with pytest.raises(_bridge_refused_type(), match="missing k_lineage"):
        _bridge_preflight_refuse(
            source,
            queue_path=queue_path,
            digest_path=tmp_path / "empty_digest.yaml",
        )


def test_bridge_allows_k_lane_within_declared_budget(tmp_path, isolate_drafts):
    """V2 semantics: K_lane=2 is allowed when prereg declared K=3."""
    source = _make_source()
    structural_hash = compute_structural_hash(_scope_to_hash_inputs(source.scope))
    queue_path = _make_promote_queue(
        tmp_path,
        strategy_id=source.scope["strategy_id"],
        structural_hash=structural_hash,
        k_lane=2,
        k_declared=3,
    )

    _bridge_preflight_refuse(
        source,
        queue_path=queue_path,
        digest_path=tmp_path / "empty_digest.yaml",
    )


def test_bridge_refuses_k_lane_over_declared_budget(tmp_path, isolate_drafts):
    """V2 semantics: K overrun is K_lane > K_declared_in_prereg."""
    source = _make_source()
    structural_hash = compute_structural_hash(_scope_to_hash_inputs(source.scope))
    queue_path = _make_promote_queue(
        tmp_path,
        strategy_id=source.scope["strategy_id"],
        structural_hash=structural_hash,
        k_lane=4,
        k_declared=3,
    )

    with pytest.raises(_bridge_refused_type(), match="K_lane=4.*K_declared_in_prereg=3"):
        _bridge_preflight_refuse(
            source,
            queue_path=queue_path,
            digest_path=tmp_path / "empty_digest.yaml",
        )
