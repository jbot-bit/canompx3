"""End-to-end smoke test: fills -> trades -> digest (mocked Claude)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.tools.coaching_digest import (
    DIGESTS_PATH,
    generate_digest,
    load_trader_profile,
    save_trader_profile,
)
from scripts.tools.fetch_broker_fills import save_fills
from scripts.tools.trade_matcher import match_fills_to_trades, save_trades


@pytest.fixture
def coaching_data_dir(tmp_path):
    """Set up a temporary coaching data directory."""
    profile = {
        "version": 1,
        "last_updated": "",
        "strengths": [],
        "growth_edges": [],
        "behavioral_patterns": [],
        "goals": [],
        "session_tendencies": {},
        "emotional_profile": {"tilt_indicators": [], "calm_indicators": []},
        "account_summary": {},
    }
    profile_path = tmp_path / "trader_profile.json"
    save_trader_profile(profile, path=profile_path)
    return tmp_path, profile_path


def test_full_pipeline_mocked(coaching_data_dir):
    """Fills -> trades -> digest with mocked Claude API."""
    tmp_path, profile_path = coaching_data_dir

    # 1. Create fills
    fills = [
        {
            "fill_id": "topstepx-1",
            "broker": "topstepx",
            "account_id": 123,
            "account_name": "test-acct",
            "instrument": "MNQ",
            "timestamp": "2026-03-06T13:00:00+00:00",
            "side": "BUY",
            "size": 2,
            "price": 24800.0,
            "pnl": 0,
            "fees": 1.0,
        },
        {
            "fill_id": "topstepx-2",
            "broker": "topstepx",
            "account_id": 123,
            "account_name": "test-acct",
            "instrument": "MNQ",
            "timestamp": "2026-03-06T13:02:00+00:00",
            "side": "SELL",
            "size": 2,
            "price": 24810.0,
            "pnl": 20.0,
            "fees": 1.0,
        },
    ]
    fills_path = tmp_path / "fills.jsonl"
    save_fills(fills, path=fills_path)

    # 2. Match trades
    trades = match_fills_to_trades(fills)
    assert len(trades) == 1
    assert trades[0]["direction"] == "LONG"
    assert trades[0]["pnl_dollar"] == 20.0

    # 3. Generate digest (mock Claude API)
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(
            text=json.dumps(
                {
                    "digest": {
                        "summary": "1 MNQ trade, net +$20",
                        "trade_grades": [{"trade_id": trades[0]["trade_id"], "grade": "B", "reason": "Clean entry"}],
                        "patterns_observed": ["patience"],
                        "coaching_note": "Good discipline on this trade.",
                        "metrics": {"trades": 1, "win_rate": 1.0, "gross_pnl": 20, "fees": 2, "net_pnl": 18},
                    },
                    "profile_patch": {
                        "strengths": [{"trait": "patience", "confidence": 0.6, "evidence_count": 1}],
                    },
                }
            )
        )
    ]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    mock_anthropic_module = MagicMock()
    mock_anthropic_module.Anthropic.return_value = mock_client

    with patch.dict(sys.modules, {"anthropic": mock_anthropic_module}):
        digest = generate_digest(trades, profile_path=profile_path)

    assert digest is not None
    assert digest["summary"] == "1 MNQ trade, net +$20"

    # 4. Verify profile was updated
    updated_profile = load_trader_profile(path=profile_path)
    assert updated_profile["version"] == 2
    assert len(updated_profile["strengths"]) == 1
