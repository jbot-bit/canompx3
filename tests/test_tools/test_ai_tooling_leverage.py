from __future__ import annotations

from scripts.tools import ai_tooling_leverage


def test_official_deprecation_becomes_blocking_change() -> None:
    report = ai_tooling_leverage.build_ai_tooling_leverage(
        external_items=[
            {
                "source_type": "official",
                "vendor": "openai",
                "source_url": "https://developers.openai.com/api/docs/assistants/migration",
                "published_at": "2026-05-31",
                "claim": "Assistants API will shut down; migrate to Responses API",
                "local_repo_touchpoints": ["scripts/tools"],
                "role": "migration_blocker",
            }
        ]
    )

    card = report["cards"][0]
    assert card["impact_type"] == "breaking_change"
    assert card["disposition"] == "BLOCKING_CHANGE"
    assert card["evidence_class"] == "MEASURED"
    assert len(card["alternative_framings_checked"]) >= 2
    assert card["disconfirming_check"]
    assert card["risks_if_ignored"]


def test_community_tooling_claim_cannot_adopt_now() -> None:
    report = ai_tooling_leverage.build_ai_tooling_leverage(
        external_items=[
            {
                "source_type": "community",
                "vendor": "openai",
                "source_url": "https://reddit.example/r/codex",
                "claim": "New Codex model is cheaper and better for all audits",
                "local_repo_touchpoints": ["scripts/tools"],
                "role": "audit_improver",
            }
        ]
    )

    card = report["cards"][0]
    assert card["disposition"] == "WATCH"
    assert card["evidence_class"] == "UNSUPPORTED"
    assert "official" in card["acceptance_check"].lower()


def test_same_item_different_role_changes_disposition() -> None:
    cost_card = ai_tooling_leverage.classify_tooling_item(
        {
            "source_type": "official",
            "vendor": "openai",
            "source_url": "https://developers.openai.com/api/reference/resources/admin/subresources/organization/subresources/usage",
            "claim": "Usage and Costs API exposes token and spend data",
            "local_repo_touchpoints": ["scripts/tools/daily_project_radar.py"],
            "role": "cost_reducer",
        }
    )
    irrelevant_card = ai_tooling_leverage.classify_tooling_item(
        {
            "source_type": "official",
            "vendor": "openai",
            "source_url": "https://developers.openai.com/api/reference/resources/admin/subresources/organization/subresources/usage",
            "claim": "Usage and Costs API exposes token and spend data",
            "local_repo_touchpoints": [],
            "role": "not_relevant",
        }
    )

    assert cost_card["disposition"] == "EVALUATE"
    assert cost_card["impact_type"] == "cost_reduction"
    assert irrelevant_card["disposition"] == "IGNORE"


def test_negative_control_is_ignored_not_promoted() -> None:
    report = ai_tooling_leverage.build_ai_tooling_leverage(
        external_items=[
            {
                "source_type": "official",
                "vendor": "openai",
                "source_url": "https://developers.openai.com/api/docs/changelog",
                "claim": "New image style preset for consumer image generation",
                "local_repo_touchpoints": [],
                "role": "not_relevant",
            }
        ]
    )

    assert report["cards"] == []
    assert report["negative_controls"][0]["disposition"] == "IGNORE"
    assert report["lane_audit"]["false_negative_sample"]


def test_missing_high_impact_source_blocks_strong_disposition() -> None:
    report = ai_tooling_leverage.build_ai_tooling_leverage(
        external_items=[
            {
                "source_type": "official",
                "vendor": "anthropic",
                "source_url": "https://platform.claude.com/docs/en/release-notes/overview",
                "claim": "Tool search can reduce token usage in large tool catalogs",
                "local_repo_touchpoints": [".codex", ".claude"],
                "role": "automation_unlock",
            }
        ],
        skipped_sources=[
            {
                "source": "OpenAI deprecations",
                "reason": "network unavailable",
                "impact": "could hide migration blocker",
                "downgrades_verdict": True,
            }
        ],
    )

    assert report["cards"][0]["disposition"] == "WATCH"
    assert report["silence_ledger"][0]["downgrades_verdict"] is True
    assert report["lane_audit"]["confidence_limiter"] != "none"
