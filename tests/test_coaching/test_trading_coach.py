"""Tests for interactive trading coach."""

import json

from scripts.tools.trading_coach import build_chat_system_prompt, load_recent_digests


class TestBuildChatSystemPrompt:
    def test_includes_profile(self):
        profile = {"version": 1, "strengths": [{"trait": "patience"}]}
        prompt = build_chat_system_prompt(profile, [])
        assert "patience" in prompt
        assert "performance coach" in prompt.lower()

    def test_includes_recent_digests(self):
        profile = {"version": 1}
        digests = [{"date": "2026-03-06", "coaching_note": "Great discipline today"}]
        prompt = build_chat_system_prompt(profile, digests)
        assert "Great discipline today" in prompt


class TestLoadRecentDigests:
    def test_loads_last_n(self, tmp_path):
        path = tmp_path / "digests.jsonl"
        lines = [json.dumps({"date": f"2026-03-0{i}", "summary": f"day {i}"}) for i in range(1, 8)]
        path.write_text("\n".join(lines) + "\n")
        recent = load_recent_digests(n=3, path=path)
        assert len(recent) == 3
        assert recent[-1]["date"] == "2026-03-07"

    def test_returns_empty_if_missing(self, tmp_path):
        recent = load_recent_digests(n=5, path=tmp_path / "nonexistent.jsonl")
        assert recent == []
