"""Companion tests for FAST_LANE v5.1 runner-branch automation.

Tests ``_fast_lane_verdict_v5_1`` pure-function gate semantics against the
gate table specified in
``docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml`` § screen + § outcomes,
and tests the result-MD block renderer ``_fast_lane_block_lines``.

These tests do NOT touch the heavyweight ``_verdict`` function — that path
is covered by ``test_chordia_strict_unlock_v1_emissions.py`` and the runner's
existing real-data invocations. Heavyweight + FAST_LANE coexist by design;
neither overrides the other.
"""

from __future__ import annotations

import pytest

from research import chordia_strict_unlock_v1 as runner

# ----------------------------- helpers ----------------------------------------


def _is_result(
    *,
    t: float = 3.5,
    n_fired: int = 200,
    expr: float = 0.15,
    fire_rate: float = 0.20,
    long_expr: float = 0.14,
    short_expr: float = 0.16,
) -> dict[str, object]:
    """Build a minimal IS result dict mirroring runner._evaluate_split output."""
    return {
        "sample": "IS",
        "n_universe": 1000,
        "n_fired": n_fired,
        "scratch_n": 30,
        "null_non_scratch_n": 0,
        "fire_rate": fire_rate,
        "expr": expr,
        "policy_ev": expr * fire_rate,
        "std_r": 1.0,
        "sharpe": expr,
        "t": t,
        "p_two_sided": 0.001,
        "long_n": 100,
        "long_t": 2.5,
        "long_expr": long_expr,
        "short_n": 100,
        "short_t": 2.5,
        "short_expr": short_expr,
    }


def _boundary(proof: bool = True) -> dict[str, object]:
    return {
        "max_IS_trading_day": "2025-12-31",
        "min_OOS_trading_day": "2026-01-02",
        "holdout_boundary_value": "2026-01-01",
        "holdout_boundary_proof": proof,
    }


# ----------------------------- gate 1: holdout -------------------------------


def test_holdout_proof_false_forces_needs_more() -> None:
    """Template § outcomes line 130-131: ANY holdout problem forces NEEDS-MORE."""
    v, reason, rows = runner._fast_lane_verdict_v5_1(_is_result(), _boundary(proof=False), "long")
    assert v == "NEEDS-MORE"
    assert "Holdout boundary not proven" in reason
    # All subsequent gates must record "not evaluated" so the operator audit trail is honest.
    assert rows[0]["pass"] is False
    for r in rows[1:]:
        # Bypass row for non-pooled lanes is the one exception — it always records "bypass".
        assert r["pass"] in ("not evaluated", "bypass")


def test_holdout_proof_true_passes_gate_1() -> None:
    v, _, rows = runner._fast_lane_verdict_v5_1(_is_result(), _boundary(proof=True), "long")
    assert v == "PROMOTE"
    assert rows[0]["pass"] is True


# ----------------------------- gate 2: fire-rate -----------------------------


@pytest.mark.parametrize("fire", [0.04, 0.96, 0.999, 0.001])
def test_fire_rate_outside_band_kills(fire: float) -> None:
    """Fire-rate outside [0.05, 0.95] => KILL on gate 2 before t-stat is consulted."""
    v, reason, rows = runner._fast_lane_verdict_v5_1(_is_result(fire_rate=fire), _boundary(), "long")
    assert v == "KILL"
    assert "degenerate filter" in reason
    assert rows[1]["pass"] is False


def test_fire_rate_inside_band_passes_gate_2() -> None:
    v, _, rows = runner._fast_lane_verdict_v5_1(_is_result(fire_rate=0.50), _boundary(), "long")
    assert v == "PROMOTE"
    assert rows[1]["pass"] is True


# ----------------------------- gate 3: ExpR ----------------------------------


@pytest.mark.parametrize("expr", [0.0, -0.01, -0.5])
def test_expr_non_positive_kills(expr: float) -> None:
    v, _, rows = runner._fast_lane_verdict_v5_1(_is_result(expr=expr), _boundary(), "long")
    assert v == "KILL"
    assert rows[2]["pass"] is False


def test_expr_nan_kills() -> None:
    v, _, rows = runner._fast_lane_verdict_v5_1(_is_result(expr=float("nan")), _boundary(), "long")
    assert v == "KILL"
    assert rows[2]["pass"] is False


# ----------------------------- gate 4: N -------------------------------------


@pytest.mark.parametrize("n", [0, 1, 49])
def test_n_below_50_kills(n: int) -> None:
    v, _, rows = runner._fast_lane_verdict_v5_1(_is_result(n_fired=n), _boundary(), "long")
    assert v == "KILL"
    assert rows[3]["pass"] is False


def test_n_at_floor_passes_gate_4() -> None:
    v, _, rows = runner._fast_lane_verdict_v5_1(_is_result(n_fired=50), _boundary(), "long")
    assert v == "PROMOTE"
    assert rows[3]["pass"] is True


# ----------------------------- gate 5: per-direction sign-check --------------


def test_pooled_sign_mismatch_needs_more() -> None:
    """Template § per_direction_sign_check rule_when_runner_emits_both_fields."""
    v, reason, rows = runner._fast_lane_verdict_v5_1(
        _is_result(long_expr=0.20, short_expr=-0.05),
        _boundary(),
        "pooled",
    )
    assert v == "NEEDS-MORE"
    assert "per-direction sign-match" in reason
    assert rows[4]["pass"] is False


def test_pooled_missing_long_needs_more() -> None:
    """Template § per_direction_sign_check rule_when_runner_does_not_emit_fields."""
    v, reason, _ = runner._fast_lane_verdict_v5_1(
        _is_result(long_expr=float("nan"), short_expr=0.10),
        _boundary(),
        "pooled",
    )
    assert v == "NEEDS-MORE"
    assert "per-direction sign-match" in reason


def test_pooled_missing_short_needs_more() -> None:
    v, _, _ = runner._fast_lane_verdict_v5_1(
        _is_result(long_expr=0.10, short_expr=float("nan")),
        _boundary(),
        "pooled",
    )
    assert v == "NEEDS-MORE"


def test_pooled_same_sign_passes() -> None:
    v, _, rows = runner._fast_lane_verdict_v5_1(
        _is_result(long_expr=0.15, short_expr=0.10),
        _boundary(),
        "pooled",
    )
    assert v == "PROMOTE"
    assert rows[4]["pass"] is True


def test_long_only_direction_bypasses_sign_check() -> None:
    """Template line 128: single-direction lanes bypass per-direction sign-check."""
    v, _, rows = runner._fast_lane_verdict_v5_1(
        _is_result(long_expr=0.15, short_expr=float("nan")),
        _boundary(),
        "long",
    )
    assert v == "PROMOTE"
    assert rows[4]["pass"] == "bypass"


def test_short_only_direction_bypasses_sign_check() -> None:
    v, _, rows = runner._fast_lane_verdict_v5_1(
        _is_result(long_expr=float("nan"), short_expr=0.15),
        _boundary(),
        "short",
    )
    assert v == "PROMOTE"
    assert rows[4]["pass"] == "bypass"


# ----------------------------- gate 6: t-band --------------------------------


@pytest.mark.parametrize("t", [3.0, 3.5, 10.0])
def test_t_above_promote_threshold_promotes(t: float) -> None:
    v, _, _ = runner._fast_lane_verdict_v5_1(_is_result(t=t), _boundary(), "long")
    assert v == "PROMOTE"


@pytest.mark.parametrize("t", [2.5, 2.75, 2.999])
def test_t_in_needs_more_band(t: float) -> None:
    v, reason, _ = runner._fast_lane_verdict_v5_1(_is_result(t=t), _boundary(), "long")
    assert v == "NEEDS-MORE"
    assert "NEEDS-MORE band" in reason


@pytest.mark.parametrize("t", [0.0, 1.0, 2.4999, -1.0])
def test_t_below_needs_more_kills(t: float) -> None:
    v, _, _ = runner._fast_lane_verdict_v5_1(_is_result(t=t), _boundary(), "long")
    assert v == "KILL"


def test_t_nan_kills() -> None:
    v, _, _ = runner._fast_lane_verdict_v5_1(_is_result(t=float("nan")), _boundary(), "long")
    assert v == "KILL"


# ----------------------------- precedence ------------------------------------


def test_holdout_failure_takes_precedence_over_kill_gates() -> None:
    """Holdout proof failure forces NEEDS-MORE even when subsequent gates would KILL."""
    bad_everywhere = _is_result(t=0.5, n_fired=10, expr=-1.0, fire_rate=0.99)
    v, _, _ = runner._fast_lane_verdict_v5_1(bad_everywhere, _boundary(proof=False), "long")
    assert v == "NEEDS-MORE"


def test_fire_rate_kill_takes_precedence_over_t_kill() -> None:
    """Fire-rate gate (gate 2) decides before t-stat band (gate 6)."""
    v, reason, _ = runner._fast_lane_verdict_v5_1(
        _is_result(fire_rate=0.97, t=0.1),
        _boundary(),
        "long",
    )
    assert v == "KILL"
    assert "degenerate filter" in reason


# ----------------------------- gate-rows structure ---------------------------


def test_gate_rows_has_six_entries_for_pooled() -> None:
    _, _, rows = runner._fast_lane_verdict_v5_1(_is_result(), _boundary(), "pooled")
    assert len(rows) == 6
    names = [r["name"] for r in rows]
    assert names == [
        "Holdout boundary proof",
        "Fire-rate band",
        "ExpR_IS strict positive",
        "N_IS_on triage min",
        "Per-direction sign-check (pooled)",
        "t-stat band",
    ]


def test_gate_rows_has_six_entries_for_long_only() -> None:
    """Single-direction lanes still get all 6 rows; gate 5 records 'bypass'."""
    _, _, rows = runner._fast_lane_verdict_v5_1(_is_result(), _boundary(), "long")
    assert len(rows) == 6
    assert rows[4]["pass"] == "bypass"


# ----------------------------- result-MD block renderer -----------------------


def test_block_lines_contain_load_bearing_sentinel() -> None:
    """The drift check greps for this exact heading — do not rename."""
    _, _, rows = runner._fast_lane_verdict_v5_1(_is_result(), _boundary(), "long")
    lines = runner._fast_lane_block_lines("PROMOTE", "test reason", rows, "long")
    block = "\n".join(lines)
    assert "## FAST_LANE v5.1 verdict (automated)" in block
    assert "**FAST_LANE verdict:** `PROMOTE`" in block


def test_block_lines_emit_fire_rate_diagnostic_note_when_gate_2_fails() -> None:
    """Per stage design § 7: surface 'degenerate filter, not t-stat failure' explicitly."""
    _, _, rows = runner._fast_lane_verdict_v5_1(_is_result(fire_rate=0.97), _boundary(), "long")
    lines = runner._fast_lane_block_lines("KILL", "fire-rate kill", rows, "long")
    block = "\n".join(lines)
    assert "Diagnostic note" in block
    assert "degenerate filter" in block


def test_block_lines_omit_fire_rate_diagnostic_when_gate_2_passes() -> None:
    """Diagnostic only fires when gate 2 is the deciding gate."""
    _, _, rows = runner._fast_lane_verdict_v5_1(_is_result(), _boundary(), "long")
    lines = runner._fast_lane_block_lines("PROMOTE", "all good", rows, "long")
    block = "\n".join(lines)
    assert "Diagnostic note" not in block


def test_block_lines_carry_promote_authorize_stanza() -> None:
    """Doctrine-preserving prose from template lines 191-194 must be in the block."""
    _, _, rows = runner._fast_lane_verdict_v5_1(_is_result(), _boundary(), "long")
    lines = runner._fast_lane_block_lines("PROMOTE", "ok", rows, "long")
    block = "\n".join(lines)
    assert "What PROMOTE authorizes" in block
    assert "What PROMOTE does NOT authorize" in block
    assert "Capital allocation" in block
    assert "paper-trade + SR-monitor" in block


# ----------------------------- end-to-end golden ------------------------------


def test_today_nyse_close_replay_would_kill_on_fire_rate() -> None:
    """Today's NYSE_CLOSE heavyweight run: t=1.882, fire=0.9644.

    Stage design § 7 self-check: under FAST_LANE v5.1 framing this cell KILLs
    on gate 2 (fire-rate 0.9644 > 0.95) BEFORE the t-stat band is consulted.
    This is the diagnostic the heavyweight verdict (FAIL_STRICT_CHORDIA at
    t < 3.79) misses — degenerate filter, not weak edge.
    """
    is_dict = _is_result(t=1.882, n_fired=1246, expr=0.0382, fire_rate=0.9644)
    result = runner._fast_lane_verdict_v5_1(is_dict, _boundary(), "pooled")
    v, _, rows = result
    assert v == "KILL"
    assert "Fire-rate" in rows[1]["name"]
    assert rows[1]["pass"] is False
    # t-band row records "not evaluated" because gate 2 already decided.
    assert rows[5]["pass"] == "not evaluated"


def test_existing_v5_1_instance_replay_would_promote() -> None:
    """The 2026-05-18-mnq-usdata1000 PROMOTE replay (operator wrote PROMOTE).

    Numbers from docs/audit/results/2026-05-18-mnq-usdata1000-...-fast-lane-v1.md
    summary.csv IS row: t=3.064, ExpR=0.171, N=226, fire=0.147, direction=long.
    """
    is_dict = _is_result(t=3.064, n_fired=226, expr=0.171, fire_rate=0.147)
    v, _, _ = runner._fast_lane_verdict_v5_1(is_dict, _boundary(), "long")
    assert v == "PROMOTE"


def test_t_just_below_3_lands_in_needs_more_not_kill() -> None:
    """Boundary test: t=2.999 must produce NEEDS-MORE (not KILL or PROMOTE)."""
    v, _, _ = runner._fast_lane_verdict_v5_1(_is_result(t=2.999), _boundary(), "long")
    assert v == "NEEDS-MORE"


def test_t_at_promote_threshold_is_inclusive() -> None:
    """Boundary test: t=3.0 exact must produce PROMOTE."""
    v, _, _ = runner._fast_lane_verdict_v5_1(_is_result(t=3.0), _boundary(), "long")
    assert v == "PROMOTE"
