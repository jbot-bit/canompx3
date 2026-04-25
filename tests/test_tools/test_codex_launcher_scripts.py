from pathlib import Path


def test_wsl_launcher_scripts_call_mount_guard() -> None:
    root = Path(__file__).resolve().parents[2]
    project = (root / "scripts" / "infra" / "codex-project.sh").read_text(encoding="utf-8")
    search = (root / "scripts" / "infra" / "codex-project-search.sh").read_text(encoding="utf-8")
    review = (root / "scripts" / "infra" / "codex-review.sh").read_text(encoding="utf-8")
    worktree = (root / "scripts" / "infra" / "codex-worktree.sh").read_text(encoding="utf-8")
    sync_guard = (root / "scripts" / "infra" / "codex-wsl-sync.sh").read_text(encoding="utf-8")

    assert 'wsl_mount_guard.py" --root "$ROOT"' in project
    assert 'wsl_mount_guard.py" --root "$ROOT"' in search
    assert "task_route_packet.py" in project
    assert "task_route_packet.py" in search
    assert 'wsl_mount_guard.py" --root "$ROOT"' in review
    assert 'python3 "$ROOT/scripts/tools/wsl_mount_guard.py" --root "$ROOT"' in worktree
    assert "task_route_packet.py" in worktree
    assert '--related-root "$SOURCE_ROOT"' in sync_guard
    assert "--claim codex" in sync_guard
    assert "--mode mutating" in sync_guard
