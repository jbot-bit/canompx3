"""Tests for the `risk_cap_pts` honest risk-cap override (C11 cap remediation).

Prereg:  docs/audit/hypotheses/2026-06-05-c11-cap-x080-remediation-v1.yaml
Result:  docs/audit/results/2026-06-05-c11-cap-x080-remediation-v1.md

The override is a manual policy field stored SEPARATELY from the empirical
`p90_orb_pts`. Three invariants are locked here:

1. Loader precedence: ``load_allocation_lanes`` / ``get_profile_lane_definitions``
   prefer ``risk_cap_pts`` over ``p90_orb_pts`` when present, and a present-but-zero
   cap is NOT treated as absent (the falsy-``or`` trap).
2. Rebalancer durability: ``save_allocation`` re-injects ``risk_cap_pts`` across a
   from-scratch regen (else C11 silently re-fails on the next monthly rebalance).
3. Gate↔live parity: BOTH the survival gate and the live engine read the same
   capped value through ``get_profile_lane_definitions``.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import trading_app.lane_allocator as lane_allocator
from trading_app.lane_allocator import LaneScore, save_allocation
from trading_app.prop_profiles import (
    get_profile_lane_definitions,
    load_allocation_lanes,
)

# ── Loader precedence ──────────────────────────────────────────────


def _write_profile_json(path: Path, lanes: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "rebalance_date": "2026-05-30",
                "trailing_window_months": 12,
                "profile_id": "topstep_50k_mnq_auto",
                "lanes": lanes,
                "paused": [],
                "stale": [],
                "displaced": [],
                "all_scores_count": len(lanes),
            },
            indent=2,
        )
    )


def _deploy_lane(**over) -> dict:
    base = {
        "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
        "instrument": "MNQ",
        "orb_label": "COMEX_SETTLE",
        "orb_minutes": 5,
        "rr_target": 1.5,
        "filter_type": "OVNRNG_100",
        "status": "DEPLOY",
        "p90_orb_pts": 49.8,
    }
    base.update(over)
    return base


def test_loader_prefers_risk_cap_over_p90(tmp_path):
    p = tmp_path / "topstep_50k_mnq_auto.json"
    _write_profile_json(p, [_deploy_lane(risk_cap_pts=37.35)])
    specs = load_allocation_lanes("topstep_50k_mnq_auto", allocation_path=str(p))
    assert len(specs) == 1
    # Cap (37.35) wins over p90 (49.8).
    assert specs[0].max_orb_size_pts == 37.35


def test_loader_falls_back_to_p90_when_no_cap(tmp_path):
    p = tmp_path / "topstep_50k_mnq_auto.json"
    _write_profile_json(p, [_deploy_lane()])  # no risk_cap_pts
    specs = load_allocation_lanes("topstep_50k_mnq_auto", allocation_path=str(p))
    assert specs[0].max_orb_size_pts == 49.8


def test_loader_zero_cap_is_not_treated_as_absent(tmp_path):
    """A present-but-zero cap must NOT fall through to p90 (the falsy-``or`` trap).

    The drift honesty check forbids a zero cap from ever existing, but the loader
    must still distinguish *present-and-zero* from *absent* so a misconfiguration
    surfaces as a 0-size cap (which the drift gate then blocks) rather than
    silently reverting to the uncapped p90.
    """
    p = tmp_path / "topstep_50k_mnq_auto.json"
    _write_profile_json(p, [_deploy_lane(risk_cap_pts=0)])
    specs = load_allocation_lanes("topstep_50k_mnq_auto", allocation_path=str(p))
    assert specs[0].max_orb_size_pts == 0  # NOT 49.8


# ── Rebalancer durability (save_allocation preserves the override) ──


def _make_score(**over) -> LaneScore:
    base = dict(
        strategy_id="MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
        instrument="MNQ",
        orb_label="COMEX_SETTLE",
        orb_minutes=5,
        rr_target=1.5,
        filter_type="OVNRNG_100",
        confirm_bars=1,
        stop_multiplier=1.0,
        trailing_expr=0.23,
        trailing_n=161,
        trailing_months=12,
        annual_r_estimate=34.3,
        trailing_wr=0.52,
        session_regime_expr=0.08,
        months_negative=0,
        months_positive_since_last_neg_streak=0,
        status="DEPLOY",
        status_reason="test",
        chordia_verdict="PASS_PROTOCOL_A",
        chordia_audit_age_days=0,
        c8_oos_status="PASSED",
    )
    base.update(over)
    return LaneScore(**base)


def test_save_allocation_preserves_risk_cap_across_regen(tmp_path):
    """The silent-re-fail trap: a from-scratch regen must carry risk_cap_pts forward."""
    out = tmp_path / "alloc.json"
    score = _make_score()
    orb_stats = {("MNQ", "COMEX_SETTLE", 5): (29.1, 49.8)}

    # 1st write — no prior file, so no cap yet (uncapped baseline).
    save_allocation(
        [score], [score], date(2026, 5, 30), "topstep_50k_mnq_auto", output_path=str(out), orb_size_stats=orb_stats
    )
    data = json.loads(out.read_text())
    assert "risk_cap_pts" not in data["lanes"][0]

    # Operator hand-adds the honest cap.
    data["lanes"][0]["risk_cap_pts"] = 37.35
    out.write_text(json.dumps(data, indent=2))

    # 2nd write — REGEN from scratch. Without preservation this strips the cap.
    save_allocation(
        [score], [score], date(2026, 5, 30), "topstep_50k_mnq_auto", output_path=str(out), orb_size_stats=orb_stats
    )
    regen = json.loads(out.read_text())
    assert regen["lanes"][0]["risk_cap_pts"] == 37.35  # survived
    assert regen["lanes"][0]["p90_orb_pts"] == 49.8  # p90 truth untouched


def test_save_allocation_does_not_fabricate_cap(tmp_path):
    """Preservation must never invent a cap for a lane that never had one."""
    out = tmp_path / "alloc.json"
    score = _make_score()
    orb_stats = {("MNQ", "COMEX_SETTLE", 5): (29.1, 49.8)}
    save_allocation(
        [score], [score], date(2026, 5, 30), "topstep_50k_mnq_auto", output_path=str(out), orb_size_stats=orb_stats
    )
    save_allocation(
        [score], [score], date(2026, 5, 30), "topstep_50k_mnq_auto", output_path=str(out), orb_size_stats=orb_stats
    )
    data = json.loads(out.read_text())
    assert "risk_cap_pts" not in data["lanes"][0]


def test_read_existing_risk_caps_failsoft_on_missing(tmp_path):
    """Missing prior file → empty map, never a crash (fresh first-write state)."""
    missing = tmp_path / "nope.json"
    caps = lane_allocator._read_existing_risk_caps("topstep_50k_mnq_auto", missing, explicit_path=True)
    assert caps == {}


def test_read_existing_risk_caps_failsoft_on_corrupt(tmp_path):
    """Corrupt prior file → empty map, never a crash."""
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json")
    caps = lane_allocator._read_existing_risk_caps("topstep_50k_mnq_auto", bad, explicit_path=True)
    assert caps == {}


# ── Gate↔live parity (canonical profile, no fixture) ───────────────


def test_canonical_profile_loader_returns_capped_values():
    """The real topstep_50k_mnq_auto must resolve to the capped values that
    clear C11 (37.35 / 107.4 / 33.15), proving the JSON edit reached the loader
    that both the gate and the live engine consume.
    """
    defs = {d["strategy_id"]: d["max_orb_size_pts"] for d in get_profile_lane_definitions("topstep_50k_mnq_auto")}
    assert defs["MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100"] == 37.35
    assert defs["MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15"] == 107.4
    assert defs["MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08"] == 33.15
