"""Tests for scripts/research/fast_lane_promote_queue.py.

Synthetic v5.1 result MD fixtures via tmp_path. No DB, no live data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.research import fast_lane_promote_queue as flpq


# Canonical v5.1 result MD template the runner emits. We slot the per-cell
# numbers into the {placeholders}; everything else is the v5.1 contract.
_TEMPLATE = """# Chordia strict unlock audit — {sid}

**Prereq file:** `docs/audit/hypotheses/x.yaml`

## Scope

Single-lane K=1 confirmatory replay.

## Verdict

**MEASURED verdict:** `FAIL_STRICT_CHORDIA`

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | {nu} | {nf} | {fp}% | 0 | 0 | {expr} | 0.0 | 0.3 | {t} | 0.001 |
| OOS | 100 | 10 | 10.00% | 0 | 0 | 0.1 | 0.01 | 0.1 | 0.5 | 0.5 |

## Directional breakdown

| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |
|---|---:|---:|---:|---:|---:|---:|
| IS | {ln} | {lex} | {lt} | {sn} | {sex} | {st} |
| OOS | 0 | nan | nan | 0 | nan | nan |

## FAST_LANE v5.1 verdict (automated)

**FAST_LANE verdict:** `{verdict}`

_Scope direction at screen: `'{direction}'`. _
"""


def _write_md(
    dirpath: Path,
    *,
    sid: str,
    verdict: str = "PROMOTE",
    direction: str = "pooled",
    nu: int = 1500,
    nf: int = 200,
    fp: float = 13.33,
    expr: float = 0.18,
    t: float = 3.20,
    ln: int = 100,
    lex: float = 0.18,
    lt: float = 3.10,
    sn: int = 100,
    sex: float = 0.18,
    st: float = 3.10,
    filename: str | None = None,
) -> Path:
    text = _TEMPLATE.format(
        sid=sid,
        verdict=verdict,
        direction=direction,
        nu=nu,
        nf=nf,
        fp=fp,
        expr=expr,
        t=t,
        ln=ln,
        lex=lex,
        lt=lt,
        sn=sn,
        sex=sex,
        st=st,
    )
    fname = filename or f"2026-05-18-{sid.lower()}-fast-lane-v1.md"
    path = dirpath / fname
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def empty_hypotheses(tmp_path):
    d = tmp_path / "hypotheses"
    d.mkdir()
    return d


@pytest.fixture
def empty_action_queue(tmp_path):
    # Point at a file that doesn't exist - find_park_entry returns None.
    return tmp_path / "action-queue.yaml"


def _scan(results_dir, hypotheses_dir, action_queue, *, oos_window_days=10_000):
    """Scan helper for synthetic test fixtures.

    ``oos_window_days=10_000`` (default) is deliberately large so the
    pre-flight OOS-power gate (RULE 3.3) does not auto-REJECT synthetic
    PROMOTE rows whose fire-rate would yield insufficient OOS power under
    a realistic short window. Tests that exercise the gate itself live in
    ``tests/test_research/test_fast_lane_oos_power_gate.py`` and pass
    realistic windows directly. Tests in THIS module focus on
    parse/classify/cache behavior orthogonal to the OOS-power gate; the
    large window keeps those orthogonal behaviors deterministic.
    """
    return flpq.scan(
        results_dir,
        hypotheses_dir=hypotheses_dir,
        action_queue=action_queue,
        oos_window_days=oos_window_days,
        oos_window_error=None,
    )


# ---------------------------------------------------------------- tests --


def test_scanner_parses_promote_result(tmp_path, empty_hypotheses, empty_action_queue):
    """Synthetic v5.1 PROMOTE MD parses into a complete PromoteEntry."""
    d = tmp_path / "results"
    d.mkdir()
    _write_md(d, sid="MNQ_X_E1_RR1.0_CB1_FOO_O5")

    entries = _scan(d, empty_hypotheses, empty_action_queue)
    assert len(entries) == 1
    e = entries[0]
    assert e.strategy_id == "MNQ_X_E1_RR1.0_CB1_FOO_O5"
    assert e.pooled_n == 200
    assert e.long_n == 100
    assert e.short_n == 100
    assert e.pooled_t == pytest.approx(3.20, abs=1e-6)
    assert e.long_fire == pytest.approx(100 / 1500, abs=1e-6)


def test_scanner_flags_pooling_artifact(tmp_path, empty_hypotheses, empty_action_queue):
    """Lane #2's actual numbers: pooled t=3.30, long t=2.12 N=47, short t=2.53 N=50.
    Both per-direction sub-stats KILL standalone (fire-rate < 5% + N<50/t<2.5)."""
    d = tmp_path / "results"
    d.mkdir()
    _write_md(
        d,
        sid="MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_16K",
        direction="pooled",
        nu=1494,
        nf=97,
        fp=6.49,
        expr=0.4676,
        t=3.300,
        ln=47,
        lex=0.4427,
        lt=2.120,
        sn=50,
        sex=0.4909,
        st=2.525,
    )

    entries = _scan(d, empty_hypotheses, empty_action_queue)
    assert len(entries) == 1
    e = entries[0]
    assert e.pooling_artifact is True
    assert e.long_side_verdict == "KILL_AS_STANDALONE"
    assert e.short_side_verdict == "KILL_AS_STANDALONE"
    assert e.status == "ERROR"
    assert e.error_reason is not None
    assert "pooling artifact" in e.error_reason


def test_scanner_passes_clean_single_direction(tmp_path, empty_hypotheses, empty_action_queue):
    """Lane #1's actual numbers: PROMOTE long t=3.065, N=221, fire 14.68%.
    Filter is directional (PD_CLEAR_LONG) so short_n=5 is near-zero leak."""
    d = tmp_path / "results"
    d.mkdir()
    _write_md(
        d,
        sid="MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30",
        direction="long",
        nu=1539,
        nf=226,
        fp=14.68,
        expr=0.1708,
        t=3.064,
        ln=221,
        lex=0.1747,
        lt=3.065,
        sn=5,
        sex=0.0,
        st=float("nan"),
    )

    entries = _scan(d, empty_hypotheses, empty_action_queue)
    assert len(entries) == 1
    e = entries[0]
    assert e.pooling_artifact is False
    assert e.status == "QUEUED"
    assert e.long_side_verdict == "PROMOTE_AS_STANDALONE"
    # short t is NaN (5 trades, no meaningful t-stat). Side verdict is N_A
    # because t isn't computable. Pooling-artifact rule doesn't apply to
    # single-direction lanes regardless.
    assert e.short_side_verdict == "N_A"


def test_scanner_detects_orphan_promote(tmp_path, empty_hypotheses, empty_action_queue):
    """A clean QUEUED PROMOTE with no cache file on disk = first run, no diff
    yet — but the entry IS visible in the report. Drift integration is
    separately tested in test_drift_check_fails_*."""
    d = tmp_path / "results"
    d.mkdir()
    _write_md(d, sid="MNQ_ORPHAN_E1_RR1.0_CB1_FOO_O5")

    entries = _scan(d, empty_hypotheses, empty_action_queue)
    assert len(entries) == 1
    assert entries[0].status == "QUEUED"
    # Diff against missing cache - should be first-run message.
    diff = flpq.diff_against_cache(entries, tmp_path / "does_not_exist.yaml")
    assert any("first run" in line for line in diff)


def test_scanner_passes_when_revocation_sidecar_exists(
    tmp_path, empty_hypotheses, empty_action_queue
):
    """PROMOTE MD + adjacent .revocation.md sidecar => REVOKED.
    Even when the per-direction gate ALSO flags pooling-artifact (lane #2's
    real case), the sidecar wins - REVOKED, not ERROR."""
    d = tmp_path / "results"
    d.mkdir()
    md_path = _write_md(
        d,
        sid="MNQ_REVOKED_E2_RR2.0_CB1_FOO_O5",
        direction="pooled",
        nu=1500,
        nf=100,
        fp=6.67,
        ln=47,
        lt=2.12,
        sn=50,
        st=2.53,
    )
    # Sidecar file next to the result MD.
    md_path.with_name(md_path.stem + ".revocation.md").write_text(
        "# Revocation note\n\nPooled is sample-doubling artifact.\n", encoding="utf-8"
    )

    entries = _scan(d, empty_hypotheses, empty_action_queue)
    assert len(entries) == 1
    e = entries[0]
    assert e.status == "REVOKED"
    assert e.revocation_sidecar is not None
    assert "revocation.md" in e.revocation_sidecar


def test_scanner_passes_when_heavyweight_prereg_referenced(tmp_path, empty_action_queue):
    """PROMOTE + heavyweight prereg (non-v5.1 template) matching strategy_id => ESCALATED."""
    d = tmp_path / "results"
    d.mkdir()
    _write_md(d, sid="MNQ_ESCALATED_E1_RR1.0_CB1_FOO_O5")

    h = d.parent / "hypotheses"
    h.mkdir()
    (h / "heavy-prereg.yaml").write_text(
        yaml.safe_dump(
            {
                "metadata": {"template_version": "chordia_strict_v1"},
                "scope": {"strategy_id": "MNQ_ESCALATED_E1_RR1.0_CB1_FOO_O5"},
            }
        ),
        encoding="utf-8",
    )

    entries = _scan(d, h, empty_action_queue)
    assert len(entries) == 1
    assert entries[0].status == "ESCALATED"
    hp = entries[0].heavyweight_prereg
    assert hp is not None and "heavy-prereg.yaml" in hp


def test_scanner_ignores_self_referencing_v5_1_prereg_as_heavyweight(
    tmp_path, empty_action_queue
):
    """The fast_lane_v5.1 prereg that birthed the PROMOTE MD must NOT count
    as a heavyweight prereg. Otherwise every PROMOTE auto-flips ESCALATED."""
    d = tmp_path / "results"
    d.mkdir()
    _write_md(d, sid="MNQ_QUEUED_E1_RR1.0_CB1_FOO_O5")

    h = d.parent / "hypotheses"
    h.mkdir()
    (h / "v5-1-prereg.yaml").write_text(
        yaml.safe_dump(
            {
                "metadata": {"template_version": "fast_lane_v5.1"},
                "scope": {"strategy_id": "MNQ_QUEUED_E1_RR1.0_CB1_FOO_O5"},
            }
        ),
        encoding="utf-8",
    )

    entries = _scan(d, h, empty_action_queue)
    assert len(entries) == 1
    assert entries[0].status == "QUEUED"
    assert entries[0].heavyweight_prereg is None


def test_scanner_passes_when_park_entry_present(tmp_path, empty_hypotheses):
    """PROMOTE + action-queue.yaml park entry => PARKED."""
    d = tmp_path / "results"
    d.mkdir()
    _write_md(d, sid="MNQ_PARKED_E1_RR1.0_CB1_FOO_O5")

    aq = tmp_path / "action-queue.yaml"
    aq.write_text(
        yaml.safe_dump(
            {
                "entries": [
                    {"strategy_id": "MNQ_PARKED_E1_RR1.0_CB1_FOO_O5", "status": "park"}
                ]
            }
        ),
        encoding="utf-8",
    )

    entries = _scan(d, empty_hypotheses, aq)
    assert len(entries) == 1
    assert entries[0].status == "PARKED"
    assert entries[0].park_entry is not None


def test_scanner_emits_error_on_missing_directional_breakdown(
    tmp_path, empty_hypotheses, empty_action_queue
):
    """PROMOTE MD without `## Directional breakdown` => ERROR (no UNKNOWN)."""
    d = tmp_path / "results"
    d.mkdir()
    text = """# Chordia strict unlock audit — MNQ_BROKEN_E1_RR1.0_CB1_FOO_O5

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 1500 | 200 | 13.33% | 0 | 0 | 0.18 | 0.0 | 0.3 | 3.20 | 0.001 |

## FAST_LANE v5.1 verdict (automated)

**FAST_LANE verdict:** `PROMOTE`

_Scope direction at screen: `'pooled'`. _
"""
    (d / "2026-05-18-broken-fast-lane-v1.md").write_text(text, encoding="utf-8")

    entries = _scan(d, empty_hypotheses, empty_action_queue)
    assert len(entries) == 1
    e = entries[0]
    assert e.status == "ERROR"
    assert e.error_reason is not None
    assert "Directional breakdown" in e.error_reason


def test_scanner_skips_non_promote_results(tmp_path, empty_hypotheses, empty_action_queue):
    """KILL and NEEDS-MORE result MDs are not in the PROMOTE queue at all."""
    d = tmp_path / "results"
    d.mkdir()
    _write_md(d, sid="MNQ_KILL_E1_RR1.0_CB1_FOO_O5", verdict="KILL",
              filename="2026-05-18-kill-fast-lane-v1.md")
    _write_md(d, sid="MNQ_NM_E1_RR1.0_CB1_FOO_O5", verdict="NEEDS-MORE",
              filename="2026-05-18-nm-fast-lane-v1.md")
    _write_md(d, sid="MNQ_PROMOTE_E1_RR1.0_CB1_FOO_O5",
              filename="2026-05-18-promote-fast-lane-v1.md")

    entries = _scan(d, empty_hypotheses, empty_action_queue)
    assert len(entries) == 1
    assert entries[0].strategy_id == "MNQ_PROMOTE_E1_RR1.0_CB1_FOO_O5"


def test_revoked_status_wins_over_escalated(tmp_path, empty_action_queue):
    """REVOKED must NOT silently flip back to ESCALATED if a heavyweight
    prereg is later authored for the same strategy_id. Revocation is final."""
    d = tmp_path / "results"
    d.mkdir()
    md_path = _write_md(d, sid="MNQ_REVOKED_THEN_HEAVY_E1_RR1.0_CB1_FOO_O5")
    md_path.with_name(md_path.stem + ".revocation.md").write_text(
        "# revoked\n", encoding="utf-8"
    )

    h = d.parent / "hypotheses"
    h.mkdir()
    (h / "heavy.yaml").write_text(
        yaml.safe_dump(
            {
                "metadata": {"template_version": "chordia_strict_v1"},
                "scope": {"strategy_id": "MNQ_REVOKED_THEN_HEAVY_E1_RR1.0_CB1_FOO_O5"},
            }
        ),
        encoding="utf-8",
    )

    entries = _scan(d, h, empty_action_queue)
    assert entries[0].status == "REVOKED"


def test_status_enum_no_unknown(tmp_path, empty_hypotheses, empty_action_queue):
    """Every entry's status must be one of the 5 canonical values."""
    d = tmp_path / "results"
    d.mkdir()
    _write_md(d, sid="A")
    _write_md(d, sid="B", verdict="KILL", filename="b-fast-lane.md")
    _write_md(
        d,
        sid="C",
        direction="pooled",
        nu=1500,
        ln=47,
        lt=2.12,
        sn=50,
        st=2.53,
        filename="c-fast-lane.md",
    )

    entries = _scan(d, empty_hypotheses, empty_action_queue)
    for e in entries:
        assert e.status in flpq.STATUS_VALUES, f"unknown status: {e.status!r}"


def test_diff_against_cache_reports_status_change(tmp_path, empty_hypotheses, empty_action_queue):
    """Cache shows QUEUED; rescan with sidecar present yields CHANGED line."""
    d = tmp_path / "results"
    d.mkdir()
    md_path = _write_md(d, sid="MNQ_CHANGES_E1_RR1.0_CB1_FOO_O5")
    cache = tmp_path / "cache.yaml"

    entries = _scan(d, empty_hypotheses, empty_action_queue)
    cache.write_text(flpq.serialize_queue(entries), encoding="utf-8")
    assert any("up to date" in line for line in flpq.diff_against_cache(entries, cache))

    # Now add a sidecar -> REVOKED.
    md_path.with_name(md_path.stem + ".revocation.md").write_text("x", encoding="utf-8")
    entries2 = _scan(d, empty_hypotheses, empty_action_queue)
    diff = flpq.diff_against_cache(entries2, cache)
    assert any("CHANGED" in line and "QUEUED -> REVOKED" in line for line in diff)


def test_cli_dry_run_default(tmp_path, empty_hypotheses, empty_action_queue, capsys):
    """Default CLI invocation is dry-run; no file is written."""
    d = tmp_path / "results"
    d.mkdir()
    _write_md(d, sid="MNQ_CLI_E1_RR1.0_CB1_FOO_O5")
    cache = tmp_path / "cache.yaml"

    rc = flpq.main(
        [
            "--results-dir", str(d),
            "--hypotheses-dir", str(empty_hypotheses),
            "--action-queue", str(empty_action_queue),
            "--cache-path", str(cache),
            # Large OOS window so the pre-flight OOS-power gate (RULE 3.3)
            # does not auto-REJECT this synthetic PROMOTE fixture.
            "--oos-window-days", "10000",
        ]
    )
    assert rc == 0
    assert not cache.exists(), "dry-run must NOT write the cache"
    out = capsys.readouterr().out
    assert "PROMOTE queue" in out
    assert "QUEUED (1)" in out


def test_cli_write_creates_cache(tmp_path, empty_hypotheses, empty_action_queue):
    """--write refreshes the cache file."""
    d = tmp_path / "results"
    d.mkdir()
    _write_md(d, sid="MNQ_WRITE_E1_RR1.0_CB1_FOO_O5")
    cache = tmp_path / "cache.yaml"

    rc = flpq.main(
        [
            "--write",
            "--results-dir", str(d),
            "--hypotheses-dir", str(empty_hypotheses),
            "--action-queue", str(empty_action_queue),
            "--cache-path", str(cache),
            "--oos-window-days", "10000",
        ]
    )
    assert rc == 0
    assert cache.exists()
    data = yaml.safe_load(cache.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert len(data["entries"]) == 1
    assert data["entries"][0]["strategy_id"] == "MNQ_WRITE_E1_RR1.0_CB1_FOO_O5"


def test_cli_write_and_dry_run_mutually_exclusive(
    tmp_path, empty_hypotheses, empty_action_queue
):
    with pytest.raises(SystemExit):
        flpq.main(
            [
                "--write",
                "--dry-run",
                "--results-dir", str(tmp_path),
                "--hypotheses-dir", str(empty_hypotheses),
                "--action-queue", str(empty_action_queue),
            ]
        )


def test_cli_exit_code_2_on_error_entry(tmp_path, empty_hypotheses, empty_action_queue):
    """Any ERROR entry yields exit code 2 (drift-check integration signal)."""
    d = tmp_path / "results"
    d.mkdir()
    _write_md(
        d,
        sid="MNQ_POOLING_E2_RR2.0_CB1_ORB_VOL_16K",
        direction="pooled",
        nu=1494,
        nf=97,
        fp=6.49,
        ln=47,
        lt=2.120,
        sn=50,
        st=2.525,
    )

    rc = flpq.main(
        [
            "--results-dir", str(d),
            "--hypotheses-dir", str(empty_hypotheses),
            "--action-queue", str(empty_action_queue),
            "--cache-path", str(tmp_path / "x.yaml"),
            "--oos-window-days", "10000",
        ]
    )
    assert rc == 2
