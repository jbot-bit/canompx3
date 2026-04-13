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


class TestContextRoutingGovernance:
    def test_context_generated_docs_catches_missing_files(self, tmp_path: Path, monkeypatch) -> None:
        _patch_dirs(monkeypatch, tmp_path)

        violations = check_drift.check_context_generated_docs()

        assert violations
        assert "docs/context/README.md missing" in "\n".join(violations)

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
