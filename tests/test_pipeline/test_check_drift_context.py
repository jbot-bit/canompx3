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


def _mk_handoff_compact_fixtures(
    root: Path,
    *,
    handoff_lines: int,
    next_steps: list[str] | None = None,
    queue_items: list[dict] | None = None,
    ledger: str = "# Decision Ledger\n- entry 1\n",
    queue_present: bool = True,
) -> None:
    """Scaffold the three files check_handoff_compact_policy reads."""
    body = ["# HANDOFF"]
    if next_steps:
        body.append("## Next Steps — Active")
        for i, step in enumerate(next_steps, 1):
            body.append(f"{i}. {step}")
    while len(body) < handoff_lines:
        body.append("")
    _mkfile(root / "HANDOFF.md", "\n".join(body) + "\n")

    _mkfile(root / "docs/runtime/decision-ledger.md", ledger)

    if queue_present:
        import yaml

        queue_body = {
            "schema_version": 1,
            "items": queue_items or [{"id": "seed_item", "title": "Seed item", "next_action": "seed"}],
        }
        _mkfile(root / "docs/runtime/action-queue.yaml", yaml.safe_dump(queue_body))


class TestHandoffCompactPolicy:
    def test_passes_when_compact_and_ledger_and_queue_present(self, tmp_path: Path, monkeypatch) -> None:
        _patch_dirs(monkeypatch, tmp_path)
        _mk_handoff_compact_fixtures(
            tmp_path,
            handoff_lines=10,
            next_steps=["Seed item — do the seed"],
            queue_items=[{"id": "seed_item", "title": "Seed item", "next_action": "do the seed"}],
        )

        assert check_drift.check_handoff_compact_policy() == []

    def test_flags_bloated_handoff(self, tmp_path: Path, monkeypatch) -> None:
        _patch_dirs(monkeypatch, tmp_path)
        _mk_handoff_compact_fixtures(
            tmp_path,
            handoff_lines=250,
            queue_items=[{"id": "seed", "title": "Seed", "next_action": "."}],
        )

        violations = check_drift.check_handoff_compact_policy()
        joined = "\n".join(violations)
        assert "250 lines" in joined
        assert ">100" in joined

    def test_flags_missing_ledger(self, tmp_path: Path, monkeypatch) -> None:
        _patch_dirs(monkeypatch, tmp_path)
        _mk_handoff_compact_fixtures(tmp_path, handoff_lines=10, ledger="")

        violations = check_drift.check_handoff_compact_policy()

        assert any("decision-ledger.md missing or empty" in v for v in violations)

    def test_flags_missing_queue(self, tmp_path: Path, monkeypatch) -> None:
        _patch_dirs(monkeypatch, tmp_path)
        _mk_handoff_compact_fixtures(tmp_path, handoff_lines=10, queue_present=False)

        violations = check_drift.check_handoff_compact_policy()

        assert any("action-queue.yaml missing" in v for v in violations)

    def test_flags_orphan_next_step(self, tmp_path: Path, monkeypatch) -> None:
        _patch_dirs(monkeypatch, tmp_path)
        _mk_handoff_compact_fixtures(
            tmp_path,
            handoff_lines=10,
            next_steps=["Zebra quantum thruster refactor — invent something new"],
            queue_items=[{"id": "seed", "title": "Seed", "next_action": "do nothing"}],
        )

        violations = check_drift.check_handoff_compact_policy()

        assert any("zebra quantum thruster" in v.lower() for v in violations)

    def test_handles_punctuation_variants(self, tmp_path: Path, monkeypatch) -> None:
        # Queue title uses hyphens ("Cross-asset earlier-session"); HANDOFF tokenization
        # drops them. Normalization must make the two match.
        _patch_dirs(monkeypatch, tmp_path)
        _mk_handoff_compact_fixtures(
            tmp_path,
            handoff_lines=10,
            next_steps=["Cross-asset earlier-session chronology spec — write it"],
            queue_items=[
                {
                    "id": "cross_asset",
                    "title": "Cross-asset earlier-session chronology spec",
                    "next_action": "write it",
                }
            ],
        )

        assert check_drift.check_handoff_compact_policy() == []
