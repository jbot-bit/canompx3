"""Tests for scripts/tools/sync_pinecone.py — sync orchestrator."""

import json
import tempfile
from pathlib import Path

import pytest

from scripts.tools.sync_pinecone import (
    load_manifest,
    load_state,
    save_state,
    hash_file,
    collect_files,
    detect_changes,
    check_basename_conflicts,
    MANIFEST_PATH,
    STATE_PATH,
)


# ---------------------------------------------------------------------------
# 1. Manifest loading
# ---------------------------------------------------------------------------


def test_load_manifest():
    """Verify manifest loads and has expected structure."""
    manifest = load_manifest()

    assert "version" in manifest
    assert "assistant_name" in manifest
    assert manifest["assistant_name"] == "orb-research"
    assert "content_tiers" in manifest

    tiers = manifest["content_tiers"]
    for tier_name in ("static", "living", "memory", "research_output", "generated"):
        assert tier_name in tiers, f"Missing tier: {tier_name}"


# ---------------------------------------------------------------------------
# 2. State persistence
# ---------------------------------------------------------------------------


def test_state_persistence(tmp_path, monkeypatch):
    """Verify state saves and loads with hashes."""
    state_file = tmp_path / ".pinecone_sync_state.json"

    # Monkeypatch STATE_PATH to use temp file
    import scripts.tools.sync_pinecone as mod

    monkeypatch.setattr(mod, "STATE_PATH", state_file)

    # First load — no file exists
    state = load_state()
    assert state == {}

    # Save state
    test_state = {
        "last_sync": "2026-03-02T10:00:00Z",
        "assistant_name": "orb-research",
        "hashes": {
            "TRADING_RULES.md": "abc123",
            "trading_app/config.py": "def456",
        },
        "file_count": 2,
    }
    save_state(test_state)

    # Reload
    loaded = load_state()
    assert loaded == test_state
    assert loaded["hashes"]["TRADING_RULES.md"] == "abc123"
    assert loaded["hashes"]["trading_app/config.py"] == "def456"


# ---------------------------------------------------------------------------
# 3. File hashing
# ---------------------------------------------------------------------------


def test_file_hash_deterministic(tmp_path):
    """Verify SHA256 is deterministic — same content = same hash."""
    f1 = tmp_path / "file1.txt"
    f2 = tmp_path / "file2.txt"
    f3 = tmp_path / "file3.txt"

    content = "Hello, Pinecone!\nLine 2\n"
    f1.write_text(content, encoding="utf-8")
    f2.write_text(content, encoding="utf-8")
    f3.write_text("Different content\n", encoding="utf-8")

    h1 = hash_file(f1)
    h2 = hash_file(f2)
    h3 = hash_file(f3)

    # Same content = same hash
    assert h1 == h2

    # Different content = different hash
    assert h1 != h3

    # Hash is a 64-char hex string (SHA256)
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


# ---------------------------------------------------------------------------
# 4. Change detection
# ---------------------------------------------------------------------------


def test_change_detection(tmp_path):
    """Verify changed files are detected, unchanged are not."""
    # Create test files
    file_a = tmp_path / "file_a.md"
    file_b = tmp_path / "file_b.md"
    file_c = tmp_path / "file_c.md"

    file_a.write_text("Content A", encoding="utf-8")
    file_b.write_text("Content B", encoding="utf-8")
    file_c.write_text("Content C", encoding="utf-8")

    collected = {
        "static": [(file_a, "file_a.md"), (file_b, "file_b.md")],
        "living": [(file_c, "file_c.md")],
    }

    # First run — no previous hashes, everything is "changed"
    changed, hashes = detect_changes(collected, {})
    assert "static" in changed
    assert "living" in changed
    assert len(changed["static"]) == 2
    assert len(changed["living"]) == 1

    # Second run — nothing changed
    changed2, hashes2 = detect_changes(collected, hashes)
    assert len(changed2) == 0
    assert hashes == hashes2

    # Third run — modify one file
    file_b.write_text("Content B modified", encoding="utf-8")
    changed3, hashes3 = detect_changes(collected, hashes)
    assert "static" in changed3
    assert len(changed3["static"]) == 1
    assert changed3["static"][0][1] == "file_b.md"
    assert "living" not in changed3

    # Force flag — everything re-uploaded
    changed4, _ = detect_changes(collected, hashes3, force=True)
    total = sum(len(v) for v in changed4.values())
    assert total == 3


# ---------------------------------------------------------------------------
# 5. Basename conflict detection
# ---------------------------------------------------------------------------


def test_basename_conflict_detection(tmp_path):
    """Verify basename conflicts are detected across tiers."""
    # Two files with same basename in different directories
    dir_a = tmp_path / "dir_a"
    dir_b = tmp_path / "dir_b"
    dir_a.mkdir()
    dir_b.mkdir()

    (dir_a / "README.md").write_text("A", encoding="utf-8")
    (dir_b / "README.md").write_text("B", encoding="utf-8")
    (dir_a / "unique.md").write_text("C", encoding="utf-8")

    collected = {
        "static": [(dir_a / "README.md", "dir_a/README.md")],
        "living": [
            (dir_b / "README.md", "dir_b/README.md"),
            (dir_a / "unique.md", "dir_a/unique.md"),
        ],
    }

    conflicts = check_basename_conflicts(collected)
    assert len(conflicts) == 1
    assert "README.md" in conflicts[0]

    # No conflicts
    collected_clean = {
        "static": [(dir_a / "unique.md", "dir_a/unique.md")],
        "living": [(dir_b / "README.md", "dir_b/README.md")],
    }
    assert check_basename_conflicts(collected_clean) == []


# ---------------------------------------------------------------------------
# 6. File collection (live manifest)
# ---------------------------------------------------------------------------


def test_collect_files_from_manifest():
    """Verify collect_files returns files for all tiers from live manifest."""
    manifest = load_manifest()
    collected = collect_files(manifest)

    # All 5 tiers should be present
    assert set(collected.keys()) == {"static", "living", "memory", "research_output", "generated"}

    # Static files should exist (we know TRADING_RULES.md is one)
    assert len(collected["static"]) > 0
    static_keys = [key for _, key in collected["static"]]
    assert "TRADING_RULES.md" in static_keys

    # Living files should exist
    assert len(collected["living"]) > 0
    living_keys = [key for _, key in collected["living"]]
    assert "trading_app/config.py" in living_keys

    # Memory files should exist (MEMORY.md at minimum)
    assert len(collected["memory"]) > 0
    memory_keys = [key for _, key in collected["memory"]]
    assert "memory/MEMORY.md" in memory_keys

    # All paths should be absolute and exist
    for tier, files in collected.items():
        for abs_path, rel_key in files:
            assert abs_path.is_absolute(), f"Not absolute: {abs_path}"
            assert abs_path.exists(), f"Does not exist: {abs_path} (key={rel_key})"
