from __future__ import annotations

from pathlib import Path

from pipeline import check_drift


def _patch_dirs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(check_drift, "PIPELINE_DIR", tmp_path / "pipeline")
    monkeypatch.setattr(check_drift, "TRADING_APP_DIR", tmp_path / "trading_app")
    monkeypatch.setattr(check_drift, "SCRIPTS_DIR", tmp_path / "scripts")
    monkeypatch.setattr(check_drift, "RESEARCH_DIR", tmp_path / "research")


def _mkfile(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _mk_generated_docs(root: Path) -> None:
    for rel in (
        "docs/governance/system_authority_map.md",
        "docs/context/README.md",
        "docs/context/source-catalog.md",
        "docs/context/task-routes.md",
        "docs/context/institutional-contracts.md",
    ):
        _mkfile(
            root / rel,
            "# Generated\n\nGenerated from `fixture`. Do not edit by hand.\n",
        )


class TestContextRoutingGovernance:
    def test_context_generated_docs_catches_missing_files(self, tmp_path: Path, monkeypatch) -> None:
        _patch_dirs(monkeypatch, tmp_path)

        violations = check_drift.check_context_generated_docs()

        assert violations
        joined = "\n".join(violations).replace("\\", "/")
        assert "docs/context/README.md missing" in joined

    def test_agents_context_router_ref_catches_missing_reference(self, tmp_path: Path, monkeypatch) -> None:
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "AGENTS.md", "# AGENTS\n")

        violations = check_drift.check_agents_mentions_context_resolver()

        assert violations
        assert "context_resolver.py" in "\n".join(violations)

    def test_startup_docs_context_router_ref_catches_missing_reference(self, tmp_path: Path, monkeypatch) -> None:
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "CLAUDE.md", "# CLAUDE\n")
        _mkfile(tmp_path / "CODEX.md", "# CODEX\n")

        violations = check_drift.check_startup_docs_reference_context_router()

        assert violations
        joined = "\n".join(violations)
        assert "CLAUDE.md" in joined
        assert "CODEX.md" in joined

    def test_context_view_contracts_catch_invalid_payload(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "scripts.tools.context_views.build_view",
            lambda view, root, db_path: {
                "sections": {
                    "canonical_state": {"handoff": {"summary": "wrong"}},
                    "live_operational_state": {},
                    "non_authoritative_context": {},
                },
                "section_sources": {
                    "canonical_state": ["x"],
                    "live_operational_state": ["y"],
                    "non_authoritative_context": ["z"],
                },
            },
        )

        violations = check_drift.check_context_view_contracts()

        assert violations
        assert "handoff context leaked into canonical_state" in "\n".join(violations)


class TestDocHygieneContracts:
    def test_catches_placeholder_prereg_stamp(self, tmp_path: Path, monkeypatch) -> None:
        _patch_dirs(monkeypatch, tmp_path)
        _mk_generated_docs(tmp_path)
        _mkfile(
            tmp_path / "docs/audit/hypotheses/test.yaml",
            "# Committing commit hash: TO_BE_STAMPED\nmetadata: {}\n",
        )

        violations = check_drift.check_doc_hygiene_contracts()

        assert violations
        assert "placeholder provenance marker" in "\n".join(violations)

    def test_design_only_must_not_have_entrypoint(self, tmp_path: Path, monkeypatch) -> None:
        _patch_dirs(monkeypatch, tmp_path)
        _mk_generated_docs(tmp_path)
        _mkfile(
            tmp_path / "docs/audit/hypotheses/gate0.yaml",
            """
metadata:
  name: gate0
execution:
  mode: design_only
  entrypoint: research/run_gate0.py
""",
        )

        violations = check_drift.check_doc_hygiene_contracts()

        assert violations
        assert "execution.mode=design_only" in "\n".join(violations)

    def test_executable_prereg_requires_existing_runner(self, tmp_path: Path, monkeypatch) -> None:
        _patch_dirs(monkeypatch, tmp_path)
        _mk_generated_docs(tmp_path)
        _mkfile(
            tmp_path / "docs/audit/hypotheses/run.yaml",
            """
metadata:
  name: run
execution:
  mode: execute
  entrypoint: research/missing_runner.py
""",
        )

        violations = check_drift.check_doc_hygiene_contracts()

        assert violations
        assert "references missing path" in "\n".join(violations)

    def test_generated_docs_need_source_and_do_not_edit_markers(self, tmp_path: Path, monkeypatch) -> None:
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "docs/governance/system_authority_map.md", "# Missing markers\n")

        violations = check_drift.check_doc_hygiene_contracts()

        joined = "\n".join(violations)
        assert "missing generated-source marker" in joined
        assert "missing do-not-edit marker" in joined

    def test_clean_design_only_and_generated_docs_pass(self, tmp_path: Path, monkeypatch) -> None:
        _patch_dirs(monkeypatch, tmp_path)
        _mk_generated_docs(tmp_path)
        _mkfile(
            tmp_path / "docs/audit/hypotheses/gate0.yaml",
            """
metadata:
  name: gate0
execution:
  mode: design_only
  entrypoint: null
""",
        )

        violations = check_drift.check_doc_hygiene_contracts()

        assert violations == []
