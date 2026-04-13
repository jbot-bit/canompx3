from pathlib import Path, PurePosixPath

from scripts.tools import wsl_mount_guard


def test_parse_proc_mounts_splits_entries() -> None:
    entries = wsl_mount_guard.parse_proc_mounts(
        "C:\\\\ /mnt/c 9p rw,nosuid,nodev,noatime 0 0\n"
        "/dev/sdc / ext4 rw,relatime 0 0\n"
    )

    assert len(entries) == 2
    assert entries[0].mount_point == "/mnt/c"
    assert entries[0].fs_type == "9p"
    assert "rw" in entries[0].options


def test_find_mount_entry_prefers_deepest_match() -> None:
    entries = [
        wsl_mount_guard.MountEntry("drvfs", "/mnt/c", "9p", ("rw",)),
        wsl_mount_guard.MountEntry("drvfs", "/mnt/c/Users", "9p", ("rw",)),
    ]

    entry = wsl_mount_guard.find_mount_entry(PurePosixPath("/mnt/c/Users/joshd/canompx3"), entries)

    assert entry is not None
    assert entry.mount_point == "/mnt/c/Users"


def test_collect_mount_issues_reports_read_only_and_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    git_dir = tmp_path / "repo.git"
    root.mkdir()
    git_dir.mkdir()
    mount_text = "\n".join(
        [
            "C:\\\\ /mnt/c 9p ro,metadata 0 0",
            "C:\\\\ /mnt/c 9p rw,metadata 0 0",
        ]
    )

    issues = wsl_mount_guard.collect_mount_issues(
        Path("/mnt/c/Users/joshd/canompx3"),
        mount_text=mount_text,
        git_dir=git_dir,
    )

    assert any("read-only" in issue for issue in issues)
    assert any("multiple active entries" in issue for issue in issues)


def test_collect_mount_issues_reports_write_probe_failure(monkeypatch: object, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    git_dir = tmp_path / "repo.git"
    root.mkdir()
    git_dir.mkdir()

    def fake_probe(path: Path) -> str | None:
        if path == root:
            return "permission denied"
        return None

    monkeypatch.setattr(wsl_mount_guard, "probe_write_access", fake_probe)

    issues = wsl_mount_guard.collect_mount_issues(root, mount_text="", git_dir=git_dir)

    assert issues == ["repo root write probe failed: permission denied"]


def test_format_failure_mentions_linux_filesystem(tmp_path: Path) -> None:
    message = wsl_mount_guard.format_failure(tmp_path, ["mount /mnt/c is read-only"])

    assert "wsl --shutdown" in message
    assert "~/canompx3" in message
    assert "Linux filesystem" in message
