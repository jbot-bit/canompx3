"""Tests for Pinecone coaching tier sync."""

import json
from pathlib import Path

import pytest


class TestManifestHasCoachingTier:
    def test_coaching_tier_exists_in_manifest(self):
        manifest_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "tools" / "pinecone_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tiers = manifest["content_tiers"]
        assert "coaching" in tiers, "coaching tier missing from pinecone_manifest.json"
        assert "files" in tiers["coaching"]


class TestCoachingTierCollection:
    def test_collects_profile_and_digests(self, tmp_path):
        # Create mock coaching files
        profile = tmp_path / "trader_profile.json"
        profile.write_text(json.dumps({"version": 1}))
        digests = tmp_path / "coaching_digests.jsonl"
        digests.write_text(json.dumps({"date": "2026-03-06"}) + "\n")

        from scripts.tools.sync_pinecone import collect_coaching_files

        files = collect_coaching_files(data_dir=tmp_path)
        paths = [str(p) for p, _ in files]
        assert any("trader_profile" in p for p in paths)
        assert any("coaching_digests" in p for p in paths)
