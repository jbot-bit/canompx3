"""Tests for the evidence-pack generator (PR1).

Per stage acceptance criteria 1-15 in
``docs/runtime/stages/evidence-pack-generator-stage1.md``.

The tests are hermetic: they never read the live ``gold.db`` and never
require the contamination registry to exist on disk. They use ``tmp_path``
project roots and synthetic prereg/result fixtures.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

# Make scripts/tools importable as a package root for these tests, so the
# CLI entrypoint can resolve ``evidence_pack`` regardless of the cwd from
# which pytest is invoked.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS_DIR = _REPO_ROOT / "scripts" / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

# Imports under test
import build_evidence_pack as cli  # noqa: E402
from evidence_pack import PACK_VERSION  # noqa: E402
from evidence_pack.gates import (  # noqa: E402
    evaluate_all,
    gate_chordia_t,
    gate_dsr,
    gate_pre_registered,
)
from evidence_pack.manifest import (  # noqa: E402
    Contamination,
    GateResult,
    Manifest,
    to_json_bytes,
)
from evidence_pack.renderers import (  # noqa: E402
    render_decision_card_md,
    render_gate_table_json,
    render_report_md,
)

# ─────────────────────────── Fixture builders ────────────────────────────


def _write_prereg(
    project_root: Path,
    *,
    slug: str = "synthetic",
    holdout_date: str | None = "2026-01-01",
    commit_sha: str = "abc1234",
    k_global: int = 12,
    total_trials: int = 12,
) -> Path:
    hyp_dir = project_root / "docs" / "audit" / "hypotheses"
    hyp_dir.mkdir(parents=True, exist_ok=True)
    path = hyp_dir / f"2026-04-28-{slug}-prereg.yaml"
    parts = [f"# Synthetic prereg — {slug}", "metadata:", f"  name: {slug!r}"]
    if holdout_date is not None:
        parts.append(f"  holdout_date: {holdout_date!r}")
    parts.append(f"  commit_sha: {commit_sha!r}")
    parts.append(f"  k_global: {k_global}")
    parts.append(f"  total_expected_trials: {total_trials}")
    parts.append("  testing_mode: 'individual'")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path


def _write_result_md(
    project_root: Path,
    *,
    slug: str = "synthetic",
    git_head: str = "deadbeef1234",
    pooled_finding: bool = False,
    derived_truth: bool = False,
    pooled_misuse_test_flag: bool = False,
) -> Path:
    res_dir = project_root / "docs" / "audit" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    path = res_dir / f"2026-04-28-{slug}.md"
    front = ["---"]
    if pooled_finding:
        front.append("pooled_finding: true")
        front.append("flip_rate_pct: 67.0")
        front.append("heterogeneity_ack: true")
    if pooled_misuse_test_flag:
        front.append("_pooled_misuse_test_flag: true")
    front.append("---")
    body = [
        f"# Result — {slug}",
        "",
        f"- Git HEAD: `{git_head}`",
        "- DB schema fingerprint: `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`",
        "- Bootstrap seed: 42",
        "- K = 12 declared",
    ]
    if derived_truth:
        body.append("- Truth source: validated_setups")
    path.write_text("\n".join(front + body) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A clean tmp project root, with cli.PROJECT_ROOT redirected here."""

    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    # Also kill DB lookup paths so tests do not stat the real gold.db
    monkeypatch.setattr(cli, "db_fingerprint", lambda _p: None)
    monkeypatch.setattr(cli, "repo_git_sha", lambda _p: "0" * 40)
    return tmp_path


# ─────────────────────────── Test cases ──────────────────────────────────


# Acceptance #14 sanity: imports succeed.
def test_module_loads_with_pack_version() -> None:
    assert PACK_VERSION == "0.1.0"
    importlib.reload(cli)


# Acceptance #1: determinism.
def test_manifest_serialization_deterministic(project_root: Path) -> None:
    prereg = _write_prereg(project_root)
    _write_result_md(project_root)
    # Same explicit out-name component → manifest run_iso8601 stays equal,
    # which is the only field that would otherwise differ.
    out_a = project_root / "stable-run-iso"
    m_a = cli.build_manifest(
        slug="synthetic",
        prereg_path=prereg,
        project_root=project_root,
        git_sha="0" * 40,
        fingerprint="ff" * 16,
        db_path_str=str(project_root / "gold.db"),
        allow_fingerprint_drift=False,
        out_dir=out_a,
    )
    m_b = cli.build_manifest(
        slug="synthetic",
        prereg_path=prereg,
        project_root=project_root,
        git_sha="0" * 40,
        fingerprint="ff" * 16,
        db_path_str=str(project_root / "gold.db"),
        allow_fingerprint_drift=False,
        out_dir=out_a,  # same out_dir as m_a → same run_iso
    )
    assert to_json_bytes(m_a) == to_json_bytes(m_b)


# Acceptance #2: hypothesis missing.
def test_fail_closed_hypothesis_missing(project_root: Path) -> None:
    manifest = cli.build_manifest(
        slug="ghost",
        prereg_path=None,
        project_root=project_root,
        git_sha=None,
        fingerprint=None,
        db_path_str=str(project_root / "gold.db"),
        allow_fingerprint_drift=False,
        out_dir=project_root / "ghost-run",
    )
    assert manifest.verdict == "INCOMPLETE_EVIDENCE"
    assert any("pre-reg" in r for r in manifest.verdict_reasons)


# Acceptance #3: holdout missing.
def test_fail_closed_holdout_not_declared(project_root: Path) -> None:
    prereg = _write_prereg(project_root, holdout_date=None)
    manifest = cli.build_manifest(
        slug="holdout-missing",
        prereg_path=prereg,
        project_root=project_root,
        git_sha="0" * 40,
        fingerprint="ff" * 16,
        db_path_str=str(project_root / "gold.db"),
        allow_fingerprint_drift=False,
        out_dir=project_root / "holdout-run",
    )
    assert manifest.verdict == "INCOMPLETE_EVIDENCE"
    assert any("Holdout" in r for r in manifest.verdict_reasons)


# Acceptance #4: derived layers as truth.
def test_fail_closed_derived_layers_as_truth(project_root: Path) -> None:
    prereg = _write_prereg(project_root)
    _write_result_md(project_root, derived_truth=True)
    manifest = cli.build_manifest(
        slug="synthetic",
        prereg_path=prereg,
        project_root=project_root,
        git_sha="0" * 40,
        fingerprint="ff" * 16,
        db_path_str=str(project_root / "gold.db"),
        allow_fingerprint_drift=False,
        out_dir=project_root / "derived-run",
    )
    assert manifest.verdict == "INCOMPLETE_EVIDENCE"
    assert any("derived layer" in r for r in manifest.verdict_reasons)


# Acceptance #5: fingerprint drift.
def test_fail_closed_fingerprint_drift(project_root: Path) -> None:
    prereg = _write_prereg(project_root)
    _write_result_md(project_root)
    # Plant a prior pack with a different fingerprint at the same git_sha.
    pack_dir = project_root / cli.PACK_ROOT / "synthetic" / "prior"
    pack_dir.mkdir(parents=True)
    (pack_dir / "manifest.json").write_text(
        json.dumps(
            {
                "git_sha": "0" * 40,
                "db_fingerprint": "old-fingerprint",
            }
        ),
        encoding="utf-8",
    )
    manifest = cli.build_manifest(
        slug="synthetic",
        prereg_path=prereg,
        project_root=project_root,
        git_sha="0" * 40,
        fingerprint="new-fingerprint",
        db_path_str=str(project_root / "gold.db"),
        allow_fingerprint_drift=False,
        out_dir=project_root / "drift-run",
    )
    assert manifest.verdict == "INCOMPLETE_EVIDENCE"
    assert any("fingerprint drift" in r for r in manifest.verdict_reasons)

    # And: --allow-fingerprint-drift escapes the gate.
    manifest_ok = cli.build_manifest(
        slug="synthetic",
        prereg_path=prereg,
        project_root=project_root,
        git_sha="0" * 40,
        fingerprint="new-fingerprint",
        db_path_str=str(project_root / "gold.db"),
        allow_fingerprint_drift=True,
        out_dir=project_root / "drift-run-allow",
    )
    assert manifest_ok.verdict == "CONDITIONAL"


# Acceptance #6: pooled-finding gate.
def test_pooled_finding_lane_misuse(project_root: Path) -> None:
    prereg = _write_prereg(project_root)
    _write_result_md(project_root, pooled_finding=True, pooled_misuse_test_flag=True)
    manifest = cli.build_manifest(
        slug="synthetic",
        prereg_path=prereg,
        project_root=project_root,
        git_sha="0" * 40,
        fingerprint="ff" * 16,
        db_path_str=str(project_root / "gold.db"),
        allow_fingerprint_drift=False,
        out_dir=project_root / "pooled-run",
    )
    assert manifest.verdict == "LANE_NOT_SUPPORTED_BY_POOLED"


# Acceptance #7: DSR cross-check policy.
def test_dsr_is_cross_check_only() -> None:
    g_missing = gate_dsr(None)
    g_present = gate_dsr(0.5)
    g_high = gate_dsr(0.99)
    assert g_missing.status == "UNCOMPUTED"
    assert g_present.status == "CROSS_CHECK_ONLY"
    assert g_high.status == "CROSS_CHECK_ONLY"


# Acceptance #8: Chordia severity benchmark.
def test_chordia_is_severity_benchmark() -> None:
    g = gate_chordia_t(4.0)
    assert g.status == "CROSS_CHECK_ONLY"


# Acceptance #9: Contamination registry — three paths.
def test_contamination_registry_present_clean(project_root: Path) -> None:
    # Plant a registry whose tainted list does NOT contain the result commit.
    reg = project_root / "docs" / "audit" / "results" / "2026-04-28-e2-lookahead-contamination-registry.md"
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text(
        "# E2 contamination registry\n\nTainted commits: cafef00d\n",
        encoding="utf-8",
    )
    cont = cli.evaluate_contamination(project_root, "deadbee")
    assert cont.registry_status == "PRESENT"
    assert cont.status == "CLEAN"
    assert cont.hits == ()


def test_contamination_registry_present_tainted(project_root: Path) -> None:
    reg = project_root / "docs" / "audit" / "results" / "2026-04-28-e2-lookahead-contamination-registry.md"
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text(
        "# E2 contamination registry\n\nTainted commits: deadbee, cafef00d\n",
        encoding="utf-8",
    )
    cont = cli.evaluate_contamination(project_root, "deadbeef1234")
    assert cont.registry_status == "PRESENT"
    assert cont.status == "TAINTED"
    assert "deadbee" in cont.hits


def test_contamination_registry_absent(project_root: Path) -> None:
    # No registry written.
    cont = cli.evaluate_contamination(project_root, "deadbee")
    assert cont.registry_status == "MISSING"
    assert cont.status == "UNCOMPUTED"
    # Decision card on this manifest must carry an amber banner.
    prereg = _write_prereg(project_root)
    _write_result_md(project_root)
    manifest = cli.build_manifest(
        slug="synthetic",
        prereg_path=prereg,
        project_root=project_root,
        git_sha="0" * 40,
        fingerprint="ff" * 16,
        db_path_str=str(project_root / "gold.db"),
        allow_fingerprint_drift=False,
        out_dir=project_root / "no-registry",
    )
    card = render_decision_card_md(manifest)
    assert "CONTAMINATION REGISTRY MISSING" in card
    # Verdict not downgraded by missing registry.
    assert manifest.verdict in {"CONDITIONAL", "PASS"}


# Acceptance #10: real-fixture snapshot — sized down to a structural test
# rather than a byte-exact golden file (the live prereg is on disk and may
# evolve; full byte-exact coverage is for a follow-up PR per design § 13).
def test_real_fixture_runs(tmp_path: Path) -> None:
    real_root = _REPO_ROOT
    real_prereg = real_root / "docs" / "audit" / "hypotheses" / ("2026-04-27-sizing-substrate-prereg.yaml")
    if not real_prereg.exists():
        pytest.skip("Real fixture prereg not present in this checkout.")
    out_dir = tmp_path / "real-run"
    manifest = cli.build_manifest(
        slug="2026-04-27-sizing-substrate-prereg",
        prereg_path=real_prereg,
        project_root=real_root,
        git_sha="0" * 40,
        fingerprint="ff" * 16,
        db_path_str=str(real_root / "gold.db"),
        allow_fingerprint_drift=False,
        out_dir=out_dir,
    )
    # Hypothesis block populated from real frontmatter
    assert manifest.hypothesis["commit_sha"] != "(missing)"
    # Card renders without exception
    card = render_decision_card_md(manifest)
    assert "Evidence Pack" in card
    # Report renders without exception
    rep = render_report_md(manifest)
    assert "Queries" in rep


# Acceptance #11: slug resolver — three input modes (in PR1, prereg fully
# implemented; the other two route by slug).
def test_resolver_modes(project_root: Path) -> None:
    prereg = _write_prereg(project_root, slug="resolver-test")
    path, slug = cli.resolve_prereg(str(prereg.relative_to(project_root)), project_root)
    assert path == prereg
    assert slug.endswith("resolver-test-prereg")
    # Slug-only globbing
    path2, _ = cli.resolve_prereg("resolver-test", project_root)
    assert path2 == prereg
    # Total miss
    path3, _ = cli.resolve_prereg("does-not-exist", project_root)
    assert path3 is None


# Acceptance #12: provenance siblings — gate-level. Every GateResult exposes
# a populated source. (Per-leaf provenance for the broader manifest is a
# follow-up; in PR1 we assert the gate-record contract here.)
def test_every_gate_has_source() -> None:
    inputs: dict = {
        "prereg_path": "docs/audit/hypotheses/x.yaml",
        "prereg_sha": "abc",
        "k_global": 12,
        "p_value": 0.001,
        "wfe": 0.6,
        "n_trades": 200,
        "oos_status": "PASSED",
        "min_era_expr": -0.04,
        "first_trade_day": "2023-01-01",
        "survival_pct": 0.91,
        "sr_status": "CONTINUE",
        "total_trials": 12,
    }
    gates = evaluate_all(inputs)
    assert len(gates) == 12
    for g in gates:
        assert isinstance(g, GateResult)
        assert g.source, f"gate {g.name} missing source"
        # No silent PASS for DSR or Chordia.
        if g.name in {"C5_dsr", "C4_chordia_severity"}:
            assert g.status in {"CROSS_CHECK_ONLY", "UNCOMPUTED"}


# Manifest invariant: tables_used must be a subset of canonical layers.
def test_manifest_rejects_non_canonical_tables(project_root: Path) -> None:
    with pytest.raises(ValueError, match="non-canonical"):
        Manifest(
            pack_version="0.1.0",
            slug="x",
            run_iso8601="2026-04-28T00-00-00Z",
            git_sha="0" * 40,
            db_fingerprint="ff" * 16,
            db_path=str(project_root / "gold.db"),
            holdout_date="2026-01-01",
            hypothesis={},
            result={},
            validated_setups=None,
            tables_used=("validated_setups",),  # disallowed
            is_oos_split={},
            k_framings={},
            gates=tuple(),
            queries=tuple(),
            contamination=Contamination(
                registry_paths=(),
                registry_status="MISSING",
                hits=(),
                status="UNCOMPUTED",
                expected_glob="x",
            ),
            verdict="CONDITIONAL",
            verdict_reasons=("test",),
        )


# Manifest invariant: registry MISSING must imply contamination UNCOMPUTED.
def test_contamination_missing_implies_uncomputed() -> None:
    with pytest.raises(ValueError, match="invariant"):
        Contamination(
            registry_paths=(),
            registry_status="MISSING",
            hits=(),
            status="CLEAN",
            expected_glob="x",
        )


# Renderer: gate table JSON has exactly len(manifest.gates) rows.
def test_gate_table_json_round_trip(project_root: Path) -> None:
    prereg = _write_prereg(project_root)
    _write_result_md(project_root)
    manifest = cli.build_manifest(
        slug="synthetic",
        prereg_path=prereg,
        project_root=project_root,
        git_sha="0" * 40,
        fingerprint="ff" * 16,
        db_path_str=str(project_root / "gold.db"),
        allow_fingerprint_drift=False,
        out_dir=project_root / "gate-run",
    )
    rows = json.loads(render_gate_table_json(manifest).decode("utf-8"))
    assert len(rows) == len(manifest.gates) == 12
    assert {r["name"] for r in rows} >= {
        "C1_pre_registered",
        "C5_dsr",
        "C8_oos_2026",
    }


# Smoke: end-to-end main() writes 4 files.
def test_main_writes_four_files(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prereg = _write_prereg(project_root, slug="e2e")
    _write_result_md(project_root, slug="e2e")
    out = project_root / "out"
    rc = cli.main(
        [
            "--prereg",
            str(prereg.relative_to(project_root)),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert (out / "manifest.json").is_file()
    assert (out / "decision_card.md").is_file()
    assert (out / "gate_table.json").is_file()
    assert (out / "report.md").is_file()


# C1 helper: prereg gate is PASS only with both path and sha.
def test_c1_pre_registered_helper() -> None:
    assert gate_pre_registered(None, None).status == "FAIL"
    assert gate_pre_registered("p.yaml", None).status == "UNCOMPUTED"
    assert gate_pre_registered("p.yaml", "abc").status == "PASS"
