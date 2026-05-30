"""Tests for trading_app.mcp_server (no DB required)."""

import json
from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd
import pytest

from trading_app.mcp_server import (
    _ALLOWED_PARAMS,
    MAX_MCP_ROWS,
    _generate_warnings,
    _get_ai_research_packet,
    _get_db_access_policy,
    _get_db_freshness,
    _get_db_health,
    _get_db_snapshot_manifest,
    _get_strategy_fitness,
    _list_available_queries,
    _query_trading_db,
)


class TestListAvailableQueries:
    def test_returns_all_templates(self):
        result = _list_available_queries()
        assert len(result) == 18
        names = {t["template"] for t in result}
        assert "strategy_lookup" in names
        assert "table_counts" in names

    def test_each_has_description(self):
        for t in _list_available_queries():
            assert t["description"], f"Missing description for {t['template']}"


class TestQueryTradingDb:
    def test_invalid_template_returns_error(self):
        result = _query_trading_db(template="nonexistent")
        assert "error" in result
        assert "nonexistent" in result["error"]

    @patch("trading_app.mcp_server.SQLAdapter")
    def test_valid_query_returns_data(self, mock_cls):
        mock_adapter = MagicMock()
        mock_adapter.execute.return_value = pd.DataFrame({"orb_label": ["CME_REOPEN"], "expectancy_r": [0.15]})
        mock_cls.return_value = mock_adapter

        result = _query_trading_db(template="validated_summary")
        assert result["row_count"] == 1
        assert result["columns"] == ["orb_label", "expectancy_r"]
        assert len(result["rows"]) == 1

    @patch("trading_app.mcp_server.SQLAdapter")
    def test_adapter_exception_returns_error(self, mock_cls):
        mock_adapter = MagicMock()
        mock_adapter.execute.side_effect = ValueError("bad param")
        mock_cls.return_value = mock_adapter

        result = _query_trading_db(template="strategy_lookup", orb_label="CME_REOPEN")
        assert "error" in result
        assert "bad param" in result["error"]

    @patch("trading_app.mcp_server.SQLAdapter")
    def test_params_forwarded(self, mock_cls):
        mock_adapter = MagicMock()
        mock_adapter.execute.return_value = pd.DataFrame()
        mock_cls.return_value = mock_adapter

        _query_trading_db(
            template="strategy_lookup",
            orb_label="LONDON_METALS",
            entry_model="E3",
            filter_type="ORB_G6",
            min_sample_size=50,
            limit=10,
        )

        call_args = mock_adapter.execute.call_args[0][0]
        assert call_args.template.value == "strategy_lookup"
        assert call_args.parameters["orb_label"] == "LONDON_METALS"
        assert call_args.parameters["entry_model"] == "E3"
        assert call_args.parameters["filter_type"] == "ORB_G6"
        assert call_args.parameters["min_sample_size"] == 50
        assert call_args.parameters["limit"] == 10

    @patch("trading_app.mcp_server.SQLAdapter")
    def test_new_params_forwarded(self, mock_cls):
        """rr_target and confirm_bars are forwarded to SQLAdapter."""
        mock_adapter = MagicMock()
        mock_adapter.execute.return_value = pd.DataFrame()
        mock_cls.return_value = mock_adapter

        _query_trading_db(
            template="outcomes_stats",
            orb_label="CME_REOPEN",
            rr_target=2.0,
            confirm_bars=1,
        )

        call_args = mock_adapter.execute.call_args[0][0]
        assert call_args.template.value == "outcomes_stats"
        assert call_args.parameters["rr_target"] == 2.0
        assert call_args.parameters["confirm_bars"] == 1


class TestGuardrails:
    """Guardrail enforcement tests."""

    def test_raw_sql_rejected(self):
        """No raw SQL -- only enum templates accepted."""
        result = _query_trading_db(template="SELECT * FROM bars_1m")
        assert "error" in result

    def test_sql_injection_template_rejected(self):
        result = _query_trading_db(template="strategy_lookup; DROP TABLE --")
        assert "error" in result

    @patch("trading_app.mcp_server.SQLAdapter")
    def test_limit_capped_at_max_mcp_rows(self, mock_cls):
        """Limit > MAX_MCP_ROWS gets clamped server-side."""
        mock_adapter = MagicMock()
        mock_adapter.execute.return_value = pd.DataFrame()
        mock_cls.return_value = mock_adapter

        _query_trading_db(template="validated_summary", limit=999999)

        call_args = mock_adapter.execute.call_args[0][0]
        assert call_args.parameters["limit"] <= MAX_MCP_ROWS

    @patch("trading_app.mcp_server.SQLAdapter")
    def test_result_truncated_beyond_cap(self, mock_cls):
        """Even if adapter returns too many rows, MCP truncates."""
        mock_adapter = MagicMock()
        # Return more rows than the cap
        mock_adapter.execute.return_value = pd.DataFrame({"x": range(MAX_MCP_ROWS + 100)})
        mock_cls.return_value = mock_adapter

        result = _query_trading_db(template="validated_summary")
        assert result["row_count"] <= MAX_MCP_ROWS

    def test_allowed_params_whitelist(self):
        """Only known parameter keys are in the allowlist."""
        assert {
            "orb_label",
            "entry_model",
            "filter_type",
            "min_sample_size",
            "limit",
            "instrument",
            "rr_target",
            "confirm_bars",
        } == _ALLOWED_PARAMS


class TestGetStrategyFitness:
    def test_invalid_instrument_returns_error(self):
        """Dead/unknown instrument returns error, not silent empty result."""
        result = _get_strategy_fitness(instrument="M2K")
        assert "error" in result
        assert "M2K" in result["error"]

    def test_invalid_instrument_with_typo(self):
        result = _get_strategy_fitness(instrument="FAKE")
        assert "error" in result


class TestGenerateWarnings:
    def test_no_filter_warning(self):
        df = pd.DataFrame({"filter_type": ["NO_FILTER"], "sample_size": [200]})
        w = _generate_warnings(df)
        assert any("NO_FILTER" in x for x in w)

    def test_small_sample_warning(self):
        df = pd.DataFrame({"sample_size": [10, 50, 200]})
        w = _generate_warnings(df)
        assert any("INVALID" in x for x in w)

    def test_regime_sample_warning(self):
        df = pd.DataFrame({"sample_size": [50]})
        w = _generate_warnings(df)
        assert any("REGIME" in x for x in w)

    def test_empty_df_no_warnings(self):
        assert _generate_warnings(pd.DataFrame()) == []

    def test_none_no_warnings(self):
        assert _generate_warnings(None) == []


class TestAiResearchPacket:
    @patch("trading_app.ai.research_packet.build_research_packet")
    def test_get_ai_research_packet_returns_packet(self, mock_build):
        mock_build.return_value = {"packet_kind": "ai_research_packet", "task": {"text": "Plan repo research"}}
        result = _get_ai_research_packet(task="Plan repo research", profile="deepseek_planning")
        assert result["packet_kind"] == "ai_research_packet"
        assert result["task"]["text"] == "Plan repo research"

    @patch("trading_app.ai.research_packet.build_research_packet")
    def test_get_ai_research_packet_returns_error_payload(self, mock_build):
        mock_build.side_effect = ValueError("missing required env")
        result = _get_ai_research_packet(task="Plan repo research", profile="deepseek_planning")
        assert result["error"] == "missing required env"
        assert result["profile"] == "deepseek_planning"


def _make_health_db(path):
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE daily_features (trading_day DATE, symbol VARCHAR)")
    con.execute("INSERT INTO daily_features VALUES ('2026-05-29', 'MNQ')")
    con.execute("CREATE TABLE validated_setups (strategy_id VARCHAR, instrument VARCHAR)")
    con.execute("INSERT INTO validated_setups VALUES ('S1', 'MNQ')")
    con.close()


class TestDbOperationalTools:
    def test_db_health_reports_read_only_open_and_horizon(self, tmp_path):
        db_path = tmp_path / "gold.db"
        _make_health_db(db_path)

        result = _get_db_health(db_path=db_path)

        assert result["status"] == "OK"
        assert result["db_path"] == str(db_path)
        assert result["exists"] is True
        assert result["read_only_open_ok"] is True
        assert result["access"]["write_enabled"] is False
        assert result["horizon"]["daily_features"]["max_trading_day"] == "2026-05-29"

    def test_db_health_fails_closed_when_missing(self, tmp_path):
        result = _get_db_health(db_path=tmp_path / "missing.db")

        assert result["status"] == "MISSING"
        assert result["exists"] is False
        assert result["read_only_open_ok"] is False
        assert "missing" in result["open_error"].lower()

    def test_db_freshness_reports_missing_tables_without_silent_success(self, tmp_path):
        db_path = tmp_path / "gold.db"
        con = duckdb.connect(str(db_path))
        con.execute("CREATE TABLE daily_features (trading_day DATE, symbol VARCHAR)")
        con.execute("INSERT INTO daily_features VALUES ('2026-05-29', 'MNQ')")
        con.close()

        result = _get_db_freshness(db_path=db_path)

        assert result["status"] == "OK"
        assert result["tables"]["daily_features"]["exists"] is True
        assert result["tables"]["daily_features"]["row_count"] == 1
        assert result["tables"]["orb_outcomes"]["exists"] is False

    def test_db_access_policy_is_local_read_only_and_write_disabled(self):
        policy = _get_db_access_policy()

        assert policy["default_transport"] == "stdio"
        assert policy["http_enabled"] is False
        assert policy["write_enabled"] is False
        assert policy["raw_sql_writes_enabled"] is False
        assert policy["github_live_db_access"] == "forbidden"

    def test_snapshot_manifest_lists_only_valid_approved_manifests(self, tmp_path):
        root = tmp_path / "snapshots"
        good = root / "snap-a"
        good.mkdir(parents=True)
        (good / "manifest.json").write_text(
            json.dumps(
                {
                    "manifest_version": 1,
                    "snapshot_id": "snap-a",
                    "generated_at_utc": "2026-05-30T00:00:00+00:00",
                    "source_db": {"path": "C:/repo/gold.db", "mtime_utc": "2026-05-29T00:00:00+00:00"},
                    "tables": {"daily_features": {"row_count": 1}},
                    "horizon": {"daily_features": {"max_trading_day": "2026-05-29"}},
                }
            ),
            encoding="utf-8",
        )
        bad = root / "snap-b"
        bad.mkdir()
        (bad / "manifest.json").write_text(json.dumps({"snapshot_id": "snap-b"}), encoding="utf-8")

        result = _get_db_snapshot_manifest(snapshot_root=root)

        assert result["status"] == "OK_WITH_ERRORS"
        assert [snap["snapshot_id"] for snap in result["snapshots"]] == ["snap-a"]
        assert result["errors"][0]["snapshot_id"] == "snap-b"
