from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.tools import bootstrap_health_proof


def test_bootstrap_health_proof_flags_missing_expected_interpreter(tmp_path: Path, monkeypatch) -> None:
    missing_python = tmp_path / ".venv-wsl" / "bin" / "python"
    monkeypatch.setattr(bootstrap_health_proof, "_expected_interpreter", lambda _root, _context: missing_python)
    monkeypatch.setattr(
        bootstrap_health_proof, "_git_state", lambda _root: {"branch": "main", "head": "abc", "dirty_files": []}
    )
    monkeypatch.setattr(
        bootstrap_health_proof, "_database_state", lambda _path: {"path": str(tmp_path / "gold.db"), "exists": True}
    )
    monkeypatch.setattr(bootstrap_health_proof, "_active_mutating_claims", lambda _root: [])

    proof = bootstrap_health_proof.build_bootstrap_health_proof(
        root=tmp_path,
        context="codex-wsl",
        db_path=tmp_path / "gold.db",
        pulse_payload={"counts": {"broken": 0}, "items": []},
    )

    assert proof["expected_interpreter"] == str(missing_python)
    assert proof["expected_interpreter_exists"] is False
    assert any(blocker["code"] == "missing_expected_interpreter" for blocker in proof["blockers"])
    assert proof["next_command"] == f"uv sync --frozen --python {missing_python}"


def test_bootstrap_health_proof_flags_missing_db_dirty_tree_and_claim_collision(tmp_path: Path, monkeypatch) -> None:
    expected_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    expected_python.parent.mkdir(parents=True)
    expected_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(bootstrap_health_proof, "_expected_interpreter", lambda _root, _context: expected_python)
    monkeypatch.setattr(
        bootstrap_health_proof,
        "_git_state",
        lambda _root: {"branch": "feature", "head": "abc", "dirty_files": [" M HANDOFF.md"]},
    )
    monkeypatch.setattr(
        bootstrap_health_proof,
        "_active_mutating_claims",
        lambda _root: [
            SimpleNamespace(
                tool="claude",
                branch="feature",
                head_sha="abc",
                pid=1234,
                mode="mutating",
                root=str(tmp_path),
                runtime="windows",
                fresh=True,
            )
        ],
    )

    proof = bootstrap_health_proof.build_bootstrap_health_proof(
        root=tmp_path,
        context="claude-windows",
        db_path=tmp_path / "missing-gold.db",
        pulse_payload={
            "counts": {"broken": 1},
            "items": [
                {"category": "broken", "summary": "startup failure", "action": "python scripts/tools/project_pulse.py"}
            ],
        },
    )

    codes = {blocker["code"] for blocker in proof["blockers"]}
    assert {"missing_canonical_db", "dirty_tree", "mutating_claim_collision", "pulse_broken"}.issubset(codes)
    assert proof["mutating_session_claims"]["active"] is True
    assert proof["pulse"]["broken_count"] == 1
    assert proof["pulse"]["startup_blockers"][0]["summary"] == "startup failure"


def test_bootstrap_health_proof_writes_json_and_markdown(tmp_path: Path, monkeypatch) -> None:
    expected_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    expected_python.parent.mkdir(parents=True)
    expected_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(bootstrap_health_proof, "_expected_interpreter", lambda _root, _context: expected_python)
    monkeypatch.setattr(
        bootstrap_health_proof, "_git_state", lambda _root: {"branch": "main", "head": "abc", "dirty_files": []}
    )
    monkeypatch.setattr(bootstrap_health_proof, "_active_mutating_claims", lambda _root: [])

    db_path = tmp_path / "gold.db"
    db_path.write_text("db", encoding="utf-8")
    proof = bootstrap_health_proof.build_bootstrap_health_proof(
        root=tmp_path,
        context="claude-windows",
        db_path=db_path,
        pulse_payload={"counts": {"broken": 0}, "items": []},
    )
    paths = bootstrap_health_proof.write_bootstrap_health_artifacts(proof, tmp_path / "out")

    assert paths["json"].name == "bootstrap_health_proof.json"
    assert paths["markdown"].name == "bootstrap_health_proof.md"
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["schema_version"] == 1
    assert "Bootstrap Health Proof" in paths["markdown"].read_text(encoding="utf-8")
