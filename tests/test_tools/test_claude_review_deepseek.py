"""Tests for the OpenCode/DeepSeek commit gate."""

from scripts.tools import claude_review_deepseek as gate


def test_allows_docs_prompts_and_tooling() -> None:
    assert (
        gate.blocked_files(
            [
                "docs/workflows/opencode.md",
                "docs/prompts/opencode-third-pov.md",
                "scripts/tools/claude_review_deepseek.py",
                "tests/test_tools/test_claude_review_deepseek.py",
            ]
        )
        == []
    )


def test_blocks_live_research_and_data_truth_surfaces() -> None:
    blocked = gate.blocked_files(
        [
            "trading_app/prop_profiles.py",
            "trading_app/live/session_orchestrator.py",
            "pipeline/build_daily_features.py",
            "research/new_scan.py",
            "docs/audit/results/2026-05-30-result.md",
            "gold.db",
        ]
    )

    assert [path for path, _reason in blocked] == [
        "trading_app/prop_profiles.py",
        "trading_app/live/session_orchestrator.py",
        "pipeline/build_daily_features.py",
        "research/new_scan.py",
        "docs/audit/results/2026-05-30-result.md",
        "gold.db",
    ]


def test_normalizes_windows_paths_before_classification() -> None:
    assert gate.blocked_files([r"trading_app\live_config.py"]) == [
        ("trading_app/live_config.py", "protected canonical surface")
    ]
