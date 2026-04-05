"""Tests for scripts/tools/sync_pinecone.py — sync orchestrator."""

import json
import tempfile
from pathlib import Path

import pytest

from scripts.tools.sync_pinecone import (
    MANIFEST_PATH,
    STATE_PATH,
    TOOLS_DIR,
    _classify_research_file,
    bundle_research_output,
    check_basename_conflicts,
    collect_files,
    detect_changes,
    ensure_utf8,
    hash_file,
    load_manifest,
    load_state,
    prepare_living_as_md,
    save_state,
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

    # All 6 tiers should be present
    assert set(collected.keys()) == {"static", "living", "memory", "research_output", "generated", "coaching"}

    # Static files should exist (we know TRADING_RULES.md is one)
    assert len(collected["static"]) > 0
    static_keys = [key for _, key in collected["static"]]
    assert "TRADING_RULES.md" in static_keys

    # Living files should exist (converted to .md for Pinecone upload)
    assert len(collected["living"]) > 0
    living_keys = [key for _, key in collected["living"]]
    assert "scripts/tools/_living_config.md" in living_keys

    # Memory files should exist (MEMORY.md at minimum)
    assert len(collected["memory"]) > 0
    memory_keys = [key for _, key in collected["memory"]]
    assert "memory/MEMORY.md" in memory_keys

    # All paths should be absolute and exist
    for _tier, files in collected.items():
        for abs_path, rel_key in files:
            assert abs_path.is_absolute(), f"Not absolute: {abs_path}"
            assert abs_path.exists(), f"Does not exist: {abs_path} (key={rel_key})"

    # Research output should be bundled (< 15 bundles, not 80+ individual files)
    assert len(collected["research_output"]) < 15
    research_keys = [key for _, key in collected["research_output"]]
    assert any("_bundle_" in k for k in research_keys)

    # Total file count should be under 250 (Pinecone practical limit for sync performance)
    total = sum(len(files) for files in collected.values())
    assert total < 250, f"Total files ({total}) exceeds Pinecone 250-file limit"


# ---------------------------------------------------------------------------
# 7. Research file classification
# ---------------------------------------------------------------------------


def test_classify_research_file():
    """Verify research files are classified into correct topic bundles."""
    assert _classify_research_file("a0_asymmetry_hypothesis.md") == "a0_analysis"
    assert _classify_research_file("dalton_80_rule_notes.md") == "dalton_research"
    assert _classify_research_file("lead_lag_best_wide_grid.md") == "leadlag_research"
    assert _classify_research_file("fast_lead_lag_notes.md") == "leadlag_research"
    assert _classify_research_file("fast_proxy_filter_notes.md") == "leadlag_research"
    assert _classify_research_file("forward_gate_status_latest.md") == "forward_gate"
    assert _classify_research_file("session_correlation_findings.md") == "session_research"
    assert _classify_research_file("shinies_shortlist.md") == "shinies_research"
    assert _classify_research_file("wide_regime_quality_notes.md") == "wide_regime"
    assert _classify_research_file("dst_analysis_summary.md") == "dst_research"
    assert _classify_research_file("DST_ANALYSIS_INDEX.md") == "dst_research"
    # Anything not matching a known prefix goes to misc
    assert _classify_research_file("compressed_orb_findings.md") == "misc_research"
    assert _classify_research_file("winner_speed_findings.md") == "misc_research"


# ---------------------------------------------------------------------------
# 8. Research output bundling
# ---------------------------------------------------------------------------


def test_bundle_research_output(tmp_path, monkeypatch):
    """Verify research files are bundled into topic markdown files."""
    import scripts.tools.sync_pinecone as mod

    monkeypatch.setattr(mod, "TOOLS_DIR", tmp_path)

    # Create fake research files
    files = [
        (tmp_path / "a0_test1.md", "research/output/a0_test1.md"),
        (tmp_path / "a0_test2.md", "research/output/a0_test2.md"),
        (tmp_path / "dalton_test.md", "research/output/dalton_test.md"),
        (tmp_path / "random_file.md", "research/output/random_file.md"),
    ]
    for path, _ in files:
        path.write_text(f"Content of {path.name}", encoding="utf-8")

    result = bundle_research_output(files)

    # Should produce 3 bundles: a0_analysis, dalton_research, misc_research
    assert len(result) == 3
    bundle_names = [p.name for p, _ in result]
    assert "_bundle_a0_analysis.md" in bundle_names
    assert "_bundle_dalton_research.md" in bundle_names
    assert "_bundle_misc_research.md" in bundle_names

    # Verify bundle content
    for abs_path, _rel_key in result:
        assert abs_path.exists()
        content = abs_path.read_text(encoding="utf-8")
        assert "# Research Bundle:" in content
        assert "Files included:" in content


# ---------------------------------------------------------------------------
# 9. Living file .py → .md conversion
# ---------------------------------------------------------------------------


def test_prepare_living_as_md(tmp_path, monkeypatch):
    """Verify .py files are converted to .md with fenced code blocks."""
    import scripts.tools.sync_pinecone as mod

    monkeypatch.setattr(mod, "TOOLS_DIR", tmp_path)

    py_file = tmp_path / "config.py"
    py_file.write_text("INSTRUMENTS = ['MGC', 'MNQ']\n", encoding="utf-8")
    md_file = tmp_path / "notes.md"
    md_file.write_text("# Notes\nSome text\n", encoding="utf-8")

    files = [
        (py_file, "trading_app/config.py"),
        (md_file, "docs/notes.md"),
    ]

    result = prepare_living_as_md(files)
    assert len(result) == 2

    # .py file should be converted
    converted_path, converted_key = result[0]
    assert converted_path.name == "_living_config.md"
    assert converted_key == "scripts/tools/_living_config.md"
    content = converted_path.read_text(encoding="utf-8")
    assert "# config.py (Source of Truth)" in content
    assert "```python" in content
    assert "INSTRUMENTS = ['MGC', 'MNQ']" in content

    # .md file should pass through unchanged
    passthrough_path, passthrough_key = result[1]
    assert passthrough_path == md_file
    assert passthrough_key == "docs/notes.md"


# ---------------------------------------------------------------------------
# 10. UTF-8 sanitization
# ---------------------------------------------------------------------------


def test_ensure_utf8_valid_file(tmp_path, monkeypatch):
    """Valid UTF-8 files are returned as-is."""
    import scripts.tools.sync_pinecone as mod

    monkeypatch.setattr(mod, "TOOLS_DIR", tmp_path)

    valid = tmp_path / "valid.md"
    valid.write_text("Hello world\n", encoding="utf-8")

    result = ensure_utf8(valid)
    assert result == valid  # Same path, no copy needed


def test_ensure_utf8_invalid_file(tmp_path, monkeypatch):
    """Invalid UTF-8 files get a clean copy with replacement chars."""
    import scripts.tools.sync_pinecone as mod

    monkeypatch.setattr(mod, "TOOLS_DIR", tmp_path)

    bad = tmp_path / "bad.md"
    bad.write_bytes(b"Hello \xff\xfe world\n")

    result = ensure_utf8(bad)
    assert result != bad
    assert result.name == "_clean_bad.md"
    # Clean file should be valid UTF-8
    content = result.read_text(encoding="utf-8")
    assert "Hello" in content
    assert "world" in content
