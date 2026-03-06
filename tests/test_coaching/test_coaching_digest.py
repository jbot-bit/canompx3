"""Tests for coaching digest engine."""

import json

from scripts.tools.coaching_digest import (
    apply_profile_patch,
    build_digest_prompt,
    load_trader_profile,
    parse_digest_response,
    save_trader_profile,
)


class TestBuildDigestPrompt:
    def test_includes_profile_and_trades(self):
        profile = {"version": 1, "strengths": [], "growth_edges": []}
        trades = [{"trade_id": "t1", "pnl_dollar": 100}]
        prompt = build_digest_prompt(profile, trades)
        assert "version" in prompt
        assert "t1" in prompt
        assert "pnl_dollar" in prompt


class TestParseDigestResponse:
    def test_parses_valid_json(self):
        raw = json.dumps(
            {
                "digest": {
                    "summary": "Good session",
                    "trade_grades": [{"trade_id": "t1", "grade": "A", "reason": "Clean entry"}],
                    "patterns_observed": ["patience"],
                    "coaching_note": "Keep it up.",
                    "metrics": {
                        "trades": 1,
                        "win_rate": 1.0,
                        "gross_pnl": 100,
                        "fees": 2,
                        "net_pnl": 98,
                    },
                },
                "profile_patch": {
                    "strengths": [{"trait": "patience", "confidence": 0.6, "evidence_count": 1}],
                },
            }
        )
        digest, patch = parse_digest_response(raw)
        assert digest["summary"] == "Good session"
        assert len(patch["strengths"]) == 1

    def test_handles_markdown_fenced_json(self):
        raw = (
            "```json\n"
            + json.dumps(
                {
                    "digest": {
                        "summary": "x",
                        "trade_grades": [],
                        "patterns_observed": [],
                        "coaching_note": "y",
                        "metrics": {},
                    },
                    "profile_patch": {},
                }
            )
            + "\n```"
        )
        digest, patch = parse_digest_response(raw)
        assert digest["summary"] == "x"


class TestApplyProfilePatch:
    def test_adds_new_strength(self):
        profile = {"version": 1, "strengths": [], "growth_edges": []}
        patch = {"strengths": [{"trait": "patience", "confidence": 0.6, "evidence_count": 1}]}
        apply_profile_patch(profile, patch)
        assert len(profile["strengths"]) == 1
        assert profile["version"] == 2

    def test_updates_existing_strength(self):
        profile = {
            "version": 3,
            "strengths": [{"trait": "patience", "confidence": 0.5, "evidence_count": 2}],
            "growth_edges": [],
        }
        patch = {"strengths": [{"trait": "patience", "confidence": 0.7, "evidence_count": 3}]}
        apply_profile_patch(profile, patch)
        assert profile["strengths"][0]["confidence"] == 0.7
        assert profile["version"] == 4

    def test_does_not_increment_version_on_empty_patch(self):
        profile = {"version": 5, "strengths": [], "growth_edges": []}
        apply_profile_patch(profile, {})
        assert profile["version"] == 5

    def test_patches_inchworm_dict(self):
        profile = {
            "version": 1,
            "inchworm": {"c_game_patterns": ["revenge_spiral"], "b_game_patterns": [], "a_game_indicators": []},
        }
        patch = {
            "inchworm": {
                "c_game_patterns": ["tilt_escalation"],
                "a_game_indicators": ["calm_sizing"],
            }
        }
        apply_profile_patch(profile, patch)
        assert "revenge_spiral" in profile["inchworm"]["c_game_patterns"]
        assert "tilt_escalation" in profile["inchworm"]["c_game_patterns"]
        assert profile["inchworm"]["a_game_indicators"] == ["calm_sizing"]
        assert profile["version"] == 2

    def test_patches_emotional_profile_dict(self):
        profile = {
            "version": 1,
            "emotional_profile": {"primary_emotion": None, "tilt_indicators": ["fast_reentry"]},
        }
        patch = {
            "emotional_profile": {
                "primary_emotion": "tilt",
                "tilt_indicators": ["size_increase"],
            }
        }
        apply_profile_patch(profile, patch)
        assert profile["emotional_profile"]["primary_emotion"] == "tilt"
        assert "fast_reentry" in profile["emotional_profile"]["tilt_indicators"]
        assert "size_increase" in profile["emotional_profile"]["tilt_indicators"]

    def test_inchworm_deduplicates_list_values(self):
        profile = {
            "version": 1,
            "inchworm": {"c_game_patterns": ["revenge_spiral"]},
        }
        patch = {"inchworm": {"c_game_patterns": ["revenge_spiral", "new_pattern"]}}
        apply_profile_patch(profile, patch)
        assert profile["inchworm"]["c_game_patterns"].count("revenge_spiral") == 1
        assert "new_pattern" in profile["inchworm"]["c_game_patterns"]


class TestProfilePersistence:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "profile.json"
        profile = {"version": 1, "strengths": [{"trait": "x"}], "growth_edges": []}
        save_trader_profile(profile, path=path)
        loaded = load_trader_profile(path=path)
        assert loaded["strengths"][0]["trait"] == "x"
