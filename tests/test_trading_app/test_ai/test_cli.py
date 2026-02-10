"""Tests for trading_app.ai.cli."""

import sys
from unittest.mock import patch

import pytest


class TestCLIArgParsing:
    """Test argument parsing without API calls."""

    def test_query_required(self):
        """CLI requires a query argument."""
        with patch("sys.argv", ["cli"]):
            with pytest.raises(SystemExit):
                from trading_app.ai.cli import main
                main()

    def test_missing_api_key_exits(self):
        """CLI exits with error if no API key."""
        with patch("sys.argv", ["cli", "--db", "/nonexistent/path/no.db", "test question"]), \
             patch.dict("os.environ", {}, clear=True), \
             patch("trading_app.ai.cli._load_env"):
            with pytest.raises(SystemExit) as exc_info:
                from trading_app.ai.cli import main
                main()
            assert exc_info.value.code == 1

    def test_missing_db_exits(self):
        """CLI exits with error if DB not found."""
        with patch("sys.argv", ["cli", "--db", "/nonexistent/db", "question"]), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
            with pytest.raises(SystemExit) as exc_info:
                from trading_app.ai.cli import main
                main()
            assert exc_info.value.code == 1
