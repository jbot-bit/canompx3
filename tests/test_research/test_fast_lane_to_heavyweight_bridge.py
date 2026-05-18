"""Unit tests for the fast-lane -> heavyweight Chordia bridge generator.

Covers source loading, draft YAML schema, scope inheritance, theory_citation
absence (field-presence trap defense), and end-to-end CLI smoke.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.research import fast_lane_to_heavyweight_bridge as bridge


def _write_fast_lane_pair(
    tmp_path: Path,
    *,
    strategy_id: str = "MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30",
    stem: str = "2026-05-18-mnq-usdata1000-e1-rr10-pd-clear-long-o30-fast-lane-v1",
) -> tuple[Path, Path, Path]:
    """Create a tmp_path/(results|hypotheses) pair and a matched MD+YAML."""
    results = tmp_path / "results"
    hypotheses = tmp_path / "hypotheses"
    results.mkdir()
    hypotheses.mkdir()
    md = results / f"{stem}.md"
    yml = hypotheses / f"{stem}.yaml"
    md.write_text(
        "# Chordia strict unlock audit ~ " + strategy_id + "\n\n"
        "**FAST_LANE verdict:** `PROMOTE`\n\n",
        encoding="utf-8",
    )
    payload = {
        "metadata": {
            "theory_grant": False,
            "template_version": "fast_lane_v5.1",
            "date_locked": "2026-05-18T00:00:00+10:00",
        },
        "scope": {
            "instrument": "MNQ",
            "strategy_id": strategy_id,
            "session": "US_DATA_1000",
            "orb_minutes": 30,
            "entry_model": "E1",
            "confirm_bars": 2,
            "rr_target": 1.0,
            "direction": "long",
            "filter_type": "PD_CLEAR_LONG",
        },
    }
    yml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return md, yml, hypotheses


class TestLoadFastLaneSource:
    def test_loads_matched_pair(self, tmp_path):
        md, yml, hyp = _write_fast_lane_pair(tmp_path)
        src = bridge.load_fast_lane_source(md, hypotheses_dir=hyp)
        assert src is not None
        assert src.scope["strategy_id"] == (
            "MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30"
        )
        # Paths must be repo-relative-or-absolute, never raw Windows backslash
        assert "\\" not in src.result_md_rel
        assert "\\" not in src.source_yaml_rel

    def test_returns_none_when_md_missing(self, tmp_path):
        _, _, hyp = _write_fast_lane_pair(tmp_path)
        src = bridge.load_fast_lane_source(
            tmp_path / "missing.md", hypotheses_dir=hyp
        )
        assert src is None

    def test_returns_none_when_yaml_missing(self, tmp_path):
        md, yml, hyp = _write_fast_lane_pair(tmp_path)
        yml.unlink()
        assert bridge.load_fast_lane_source(md, hypotheses_dir=hyp) is None

    def test_returns_none_when_scope_lacks_strategy_id(self, tmp_path):
        md, yml, hyp = _write_fast_lane_pair(tmp_path)
        payload = yaml.safe_load(yml.read_text(encoding="utf-8"))
        del payload["scope"]["strategy_id"]
        yml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        assert bridge.load_fast_lane_source(md, hypotheses_dir=hyp) is None


class TestBuildHeavyweightPrereg:
    def test_theory_grant_is_false(self, tmp_path):
        md, _, hyp = _write_fast_lane_pair(tmp_path)
        src = bridge.load_fast_lane_source(md, hypotheses_dir=hyp)
        assert src is not None
        prereg = bridge.build_heavyweight_prereg(src, today="2026-05-19")
        assert prereg["metadata"]["theory_grant"] is False

    def test_theory_citation_is_NOT_a_yaml_key(self, tmp_path):
        """Field-presence trap defense: must not write theory_citation as a YAML key.

        Prose mentions ("adding theory_citation field" in the operator
        checklist) are fine -- the loader's field-presence check inspects
        the YAML key path, not free-text in adjacent values.
        """
        md, _, hyp = _write_fast_lane_pair(tmp_path)
        src = bridge.load_fast_lane_source(md, hypotheses_dir=hyp)
        assert src is not None
        prereg = bridge.build_heavyweight_prereg(src, today="2026-05-19")
        # Top-level metadata must not carry the key
        assert "theory_citation" not in prereg["metadata"], (
            "bridge must NOT emit theory_citation key -- field presence "
            "trips loader (see feedback_chordia_theory_citation_field_"
            "presence_trap.md)"
        )
        # If hypotheses block is present, no hypothesis may carry theory_citation
        for hyp_block in prereg.get("hypotheses", []):
            assert "theory_citation" not in hyp_block

    def test_serialized_yaml_has_no_theory_citation_key_line(self, tmp_path):
        """Serialized YAML must not have `theory_citation:` as a key line.

        Match the YAML-key shape, not the substring. Prose mentions are
        permissible (and necessary -- the operator checklist names the
        field they'd add for the upgrade path).
        """
        import re

        md, _, hyp = _write_fast_lane_pair(tmp_path)
        src = bridge.load_fast_lane_source(md, hypotheses_dir=hyp)
        assert src is not None
        prereg = bridge.build_heavyweight_prereg(src, today="2026-05-19")
        text = yaml.safe_dump(prereg, sort_keys=False)
        key_line = re.search(r"^\s*theory_citation:", text, re.MULTILINE)
        assert key_line is None, (
            "found `theory_citation:` YAML key in serialized output -- "
            "field-presence trap"
        )

    def test_is_triage_screen_is_false(self, tmp_path):
        md, _, hyp = _write_fast_lane_pair(tmp_path)
        src = bridge.load_fast_lane_source(md, hypotheses_dir=hyp)
        assert src is not None
        prereg = bridge.build_heavyweight_prereg(src, today="2026-05-19")
        assert prereg["metadata"]["is_triage_screen"] is False

    def test_template_version_is_heavyweight(self, tmp_path):
        md, _, hyp = _write_fast_lane_pair(tmp_path)
        src = bridge.load_fast_lane_source(md, hypotheses_dir=hyp)
        assert src is not None
        prereg = bridge.build_heavyweight_prereg(src, today="2026-05-19")
        assert prereg["metadata"]["template_version"] == "chordia_strict_v1"

    def test_scope_inherited_from_source(self, tmp_path):
        md, _, hyp = _write_fast_lane_pair(tmp_path)
        src = bridge.load_fast_lane_source(md, hypotheses_dir=hyp)
        assert src is not None
        prereg = bridge.build_heavyweight_prereg(src, today="2026-05-19")
        scope = prereg["scope"]
        assert scope["instrument"] == "MNQ"
        assert scope["session"] == "US_DATA_1000"
        assert scope["orb_minutes"] == 30
        assert scope["entry_model"] == "E1"
        assert scope["filter_type"] == "PD_CLEAR_LONG"

    def test_execution_gate_starts_closed(self, tmp_path):
        md, _, hyp = _write_fast_lane_pair(tmp_path)
        src = bridge.load_fast_lane_source(md, hypotheses_dir=hyp)
        assert src is not None
        prereg = bridge.build_heavyweight_prereg(src, today="2026-05-19")
        assert prereg["execution_gate"]["allowed_now"] is False, (
            "operator must explicitly flip after literature/power review"
        )

    def test_methodology_rules_match_canonical_constant(self, tmp_path):
        """methodology_rules_applied keys must be exactly METHODOLOGY_RULES_APPLIED."""
        md, _, hyp = _write_fast_lane_pair(tmp_path)
        src = bridge.load_fast_lane_source(md, hypotheses_dir=hyp)
        assert src is not None
        prereg = bridge.build_heavyweight_prereg(src, today="2026-05-19")
        keys = set(prereg["methodology_rules_applied"].keys())
        assert keys == set(bridge.METHODOLOGY_RULES_APPLIED)

    def test_upstream_provenance_role_is_PROVENANCE_ONLY(self, tmp_path):
        md, _, hyp = _write_fast_lane_pair(tmp_path)
        src = bridge.load_fast_lane_source(md, hypotheses_dir=hyp)
        assert src is not None
        prereg = bridge.build_heavyweight_prereg(src, today="2026-05-19")
        assert (
            prereg["upstream_discovery_provenance"]["role"] == "PROVENANCE_ONLY"
        )

    def test_chordia_threshold_basis_cites_criterion_4(self, tmp_path):
        md, _, hyp = _write_fast_lane_pair(tmp_path)
        src = bridge.load_fast_lane_source(md, hypotheses_dir=hyp)
        assert src is not None
        prereg = bridge.build_heavyweight_prereg(src, today="2026-05-19")
        basis = prereg["primary_schema"]["chordia_threshold_basis"]
        assert "Criterion 4" in basis
        assert "3.79" in basis

    def test_filter_grounding_status_is_unsupported(self, tmp_path):
        md, _, hyp = _write_fast_lane_pair(tmp_path)
        src = bridge.load_fast_lane_source(md, hypotheses_dir=hyp)
        assert src is not None
        prereg = bridge.build_heavyweight_prereg(src, today="2026-05-19")
        assert (
            prereg["grounding"]["filter_grounding_status"]["verdict"]
            == "UNSUPPORTED"
        )


class TestDraftPath:
    def test_uses_drafts_dir(self):
        p = bridge.draft_path_for("MNQ_FOO_BAR", "2026-05-19")
        assert p.parent == bridge.DRAFTS_DIR

    def test_includes_date(self):
        p = bridge.draft_path_for("MNQ_FOO", "2026-05-19")
        assert "2026-05-19" in p.name

    def test_ends_with_draft_yaml(self):
        p = bridge.draft_path_for("MNQ_FOO", "2026-05-19")
        assert p.name.endswith(".draft.yaml")

    def test_kebab_case_slug(self):
        p = bridge.draft_path_for("MNQ_US_DATA_1000_E1", "2026-05-19")
        assert "mnq-us-data-1000-e1" in p.name


class TestWriteDraft:
    def test_writes_yaml_file(self, tmp_path):
        prereg = {"metadata": {"theory_grant": False}, "scope": {"strategy_id": "X"}}
        out = tmp_path / "drafts" / "test.draft.yaml"
        bridge.write_draft(prereg, out)
        assert out.exists()
        loaded = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert loaded == prereg

    def test_creates_parent_dir(self, tmp_path):
        out = tmp_path / "new_subdir" / "test.draft.yaml"
        bridge.write_draft({"a": 1}, out)
        assert out.parent.exists()


class TestMain:
    def test_dry_run_emits_yaml(self, tmp_path, capsys, monkeypatch):
        import re

        md, _, hyp = _write_fast_lane_pair(tmp_path)
        monkeypatch.setattr(bridge, "HYPOTHESES_DIR", hyp)
        rc = bridge.main([str(md), "--today", "2026-05-19", "--dry-run"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "theory_grant: false" in out
        # No `theory_citation:` YAML key in the emitted output. Prose
        # mentions in adjacent values are permissible.
        assert re.search(r"^\s*theory_citation:", out, re.MULTILINE) is None

    def test_writes_to_drafts_dir(self, tmp_path, monkeypatch):
        md, _, hyp = _write_fast_lane_pair(tmp_path)
        drafts = tmp_path / "drafts"
        monkeypatch.setattr(bridge, "HYPOTHESES_DIR", hyp)
        monkeypatch.setattr(bridge, "DRAFTS_DIR", drafts)
        rc = bridge.main([str(md), "--today", "2026-05-19"])
        assert rc == 0
        # One draft file created
        files = list(drafts.glob("*.draft.yaml"))
        assert len(files) == 1
        loaded = yaml.safe_load(files[0].read_text(encoding="utf-8"))
        assert loaded["metadata"]["theory_grant"] is False
        assert "theory_citation" not in loaded["metadata"]

    def test_exits_nonzero_on_missing_source(self, tmp_path, capsys):
        rc = bridge.main([str(tmp_path / "missing.md"), "--dry-run"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "ERROR" in err
