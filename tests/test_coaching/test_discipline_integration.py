"""Tests for discipline coach integration with AI coach."""

import json
from pathlib import Path

import pytest

from ui.discipline_data import load_coaching_note


class TestLoadCoachingNote:
    def test_returns_latest_note(self, tmp_path):
        digests_path = tmp_path / "digests.jsonl"
        digests_path.write_text(
            json.dumps({"date": "2026-03-05", "coaching_note": "Old note"})
            + "\n"
            + json.dumps({"date": "2026-03-06", "coaching_note": "Latest note"})
            + "\n"
        )
        note = load_coaching_note(digests_path=digests_path)
        assert note == "Latest note"

    def test_returns_none_if_no_digests(self, tmp_path):
        note = load_coaching_note(digests_path=tmp_path / "nonexistent.jsonl")
        assert note is None

    def test_returns_none_if_no_coaching_note_field(self, tmp_path):
        digests_path = tmp_path / "digests.jsonl"
        digests_path.write_text(json.dumps({"date": "2026-03-06", "summary": "no note"}) + "\n")
        note = load_coaching_note(digests_path=digests_path)
        assert note is None
