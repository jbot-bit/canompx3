"""Tests for scripts/tools/pinecone_manifest.json structure and file existence."""

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "scripts" / "tools" / "pinecone_manifest.json"


@pytest.fixture
def manifest():
    """Load the pinecone manifest."""
    with open(MANIFEST_PATH, "r") as f:
        return json.load(f)


def test_manifest_structure(manifest):
    """Verify required top-level keys and tier structure exist."""
    # Top-level keys
    assert "version" in manifest
    assert "assistant_name" in manifest
    assert "content_tiers" in manifest
    assert "excluded" in manifest

    tiers = manifest["content_tiers"]

    # Required tier names
    for tier_name in ("static", "research_output", "living", "memory", "generated"):
        assert tier_name in tiers, f"Missing tier: {tier_name}"
        assert "description" in tiers[tier_name], f"Missing description in {tier_name}"

    # Static and living must have explicit file lists
    assert isinstance(tiers["static"]["files"], list)
    assert len(tiers["static"]["files"]) > 0
    assert isinstance(tiers["living"]["files"], list)
    assert len(tiers["living"]["files"]) > 0

    # Research output must have glob patterns and auto_discover
    assert "glob_patterns" in tiers["research_output"]
    assert tiers["research_output"]["auto_discover"] is True

    # Memory must have base_path and glob_pattern
    assert "base_path" in tiers["memory"]
    assert "glob_pattern" in tiers["memory"]
    assert tiers["memory"]["auto_discover"] is True

    # Generated must have output_dir and files
    assert "output_dir" in tiers["generated"]
    assert isinstance(tiers["generated"]["files"], list)
    assert len(tiers["generated"]["files"]) > 0

    # Excluded must have patterns
    assert "patterns" in manifest["excluded"]
    assert isinstance(manifest["excluded"]["patterns"], list)
    assert len(manifest["excluded"]["patterns"]) > 0


def test_manifest_static_files_exist(manifest):
    """Verify all files in static.files exist on disk."""
    missing = []
    for rel_path in manifest["content_tiers"]["static"]["files"]:
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            missing.append(rel_path)
    assert missing == [], f"Static files missing from disk: {missing}"


def test_manifest_living_files_exist(manifest):
    """Verify all files in living.files exist on disk."""
    missing = []
    for rel_path in manifest["content_tiers"]["living"]["files"]:
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            missing.append(rel_path)
    assert missing == [], f"Living files missing from disk: {missing}"
