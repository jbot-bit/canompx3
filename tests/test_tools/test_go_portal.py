"""Tests for scripts/tools/go_portal.py.

Coverage:
  - render_portal returns HTML + JSON payload with 10 panels.
  - Panel-level error isolation: one failing panel does not break the others.
  - Empty-state messaging when underlying data sources are missing.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from scripts.tools import go_portal
from trading_app.strategy_fitness import FitnessReport, FitnessScore


@pytest.fixture
def patched_freshness_drift(monkeypatch):
    """Stub all slow / DB-bound helpers so unit tests are fast and offline."""
    import scripts.tools.strategy_lab_mcp_server as strategy_lab

    monkeypatch.setattr(
        go_portal,
        "_query_data_freshness",
        lambda: {"MNQ": {"max_trading_day": "2026-05-19", "stale_days": 0, "is_stale": False}},
    )
    monkeypatch.setattr(
        go_portal,
        "_drift_status",
        lambda: {"status": "PASS", "detail": "all checks passed"},
    )
    monkeypatch.setattr(strategy_lab, "_list_validated_rows", lambda instrument: [])
    monkeypatch.setattr(go_portal, "_fitness_status_map", lambda instruments: {})
    monkeypatch.setattr(go_portal, "_git_sha", lambda: "deadbeef")
    # Stub every panel — the render-orchestrator behavior is the unit under test here.
    monkeypatch.setattr(go_portal, "panel_deployed_lanes", lambda *a, **kw: "<p>stub-1</p>")
    monkeypatch.setattr(go_portal, "panel_promotable", lambda *a, **kw: "<p>stub-2</p>")
    monkeypatch.setattr(go_portal, "panel_promote_queue", lambda *a, **kw: ("<p>stub-3</p>", []))
    monkeypatch.setattr(go_portal, "panel_oos_rejections", lambda *a, **kw: "<p>stub-4</p>")
    monkeypatch.setattr(go_portal, "panel_cherry_pick_top5", lambda: "<p>stub-5</p>")
    monkeypatch.setattr(go_portal, "panel_drafts", lambda: "<p>stub-6</p>")
    monkeypatch.setattr(go_portal, "panel_journal_pending", lambda: "<p>stub-7</p>")
    monkeypatch.setattr(go_portal, "panel_next_24h", lambda *a, **kw: "<p>stub-8</p>")
    monkeypatch.setattr(go_portal, "panel_holdout", lambda *a, **kw: "<p>stub-9</p>")
    monkeypatch.setattr(go_portal, "panel_freshness_drift", lambda *a, **kw: "<p>stub-10</p>")


def test_render_portal_emits_all_panels(patched_freshness_drift):
    html, payload = go_portal.render_portal()
    # 10 panels declared in the payload.
    assert payload["panels"] == {str(i): True for i in range(1, 11)}
    # Banner choice respects any_stale flag.
    assert payload["any_stale"] is False
    assert "banner-ok" in html
    # Every panel has a stable anchor target.
    for i in range(1, 11):
        assert f"id='p{i}'" in html
    # Footer carries timestamp + sha.
    assert "deadbeef" in html


def test_render_portal_stale_banner_when_db_stale(monkeypatch):
    import scripts.tools.strategy_lab_mcp_server as strategy_lab

    monkeypatch.setattr(
        go_portal,
        "_query_data_freshness",
        lambda: {"MNQ": {"max_trading_day": "2026-05-01", "stale_days": 18, "is_stale": True}},
    )
    monkeypatch.setattr(go_portal, "_drift_status", lambda: {"status": "PASS", "detail": ""})
    monkeypatch.setattr(strategy_lab, "_list_validated_rows", lambda instrument: [])
    monkeypatch.setattr(go_portal, "_fitness_status_map", lambda instruments: {})
    monkeypatch.setattr(go_portal, "_git_sha", lambda: "abc")
    monkeypatch.setattr(go_portal, "panel_deployed_lanes", lambda *a, **kw: "<p>stub-1</p>")
    monkeypatch.setattr(go_portal, "panel_promotable", lambda *a, **kw: "<p>stub-2</p>")
    monkeypatch.setattr(go_portal, "panel_promote_queue", lambda *a, **kw: ("<p>stub-3</p>", []))
    monkeypatch.setattr(go_portal, "panel_oos_rejections", lambda *a, **kw: "<p>stub-4</p>")
    monkeypatch.setattr(go_portal, "panel_cherry_pick_top5", lambda: "<p>stub-5</p>")
    monkeypatch.setattr(go_portal, "panel_drafts", lambda: "<p>stub-6</p>")
    monkeypatch.setattr(go_portal, "panel_journal_pending", lambda: "<p>stub-7</p>")
    monkeypatch.setattr(go_portal, "panel_next_24h", lambda *a, **kw: "<p>stub-8</p>")
    monkeypatch.setattr(go_portal, "panel_holdout", lambda *a, **kw: "<p>stub-9</p>")
    monkeypatch.setattr(go_portal, "panel_freshness_drift", lambda *a, **kw: "<p>stub-10</p>")
    html, payload = go_portal.render_portal()
    assert payload["any_stale"] is True
    assert "banner-stale" in html
    assert "STALE DATA" in html


def test_panel_error_isolation(monkeypatch, patched_freshness_drift):
    """If panel 1 raises, panels 2-10 must still render and panel 1 emits an error block."""

    def boom():
        raise RuntimeError("synthetic failure for test")

    # Patch panel_deployed_lanes to raise; everything else proceeds normally.
    monkeypatch.setattr(go_portal, "panel_deployed_lanes", lambda *a, **kw: boom())
    # Stub other DB / file-system bound helpers so the test is hermetic.
    monkeypatch.setattr(go_portal, "panel_promotable", lambda *a, **kw: "<p>stub-2</p>")
    monkeypatch.setattr(go_portal, "panel_promote_queue", lambda *a, **kw: ("<p>stub-3</p>", []))
    monkeypatch.setattr(go_portal, "panel_oos_rejections", lambda *a, **kw: "<p>stub-4</p>")
    monkeypatch.setattr(go_portal, "panel_cherry_pick_top5", lambda: "<p>stub-5</p>")
    monkeypatch.setattr(go_portal, "panel_drafts", lambda: "<p>stub-6</p>")
    monkeypatch.setattr(go_portal, "panel_journal_pending", lambda: "<p>stub-7</p>")
    monkeypatch.setattr(go_portal, "panel_next_24h", lambda *a, **kw: "<p>stub-8</p>")
    monkeypatch.setattr(go_portal, "panel_holdout", lambda *a, **kw: "<p>stub-9</p>")
    monkeypatch.setattr(go_portal, "panel_freshness_drift", lambda *a, **kw: "<p>stub-10</p>")

    html, _ = go_portal.render_portal()
    # Panel 1 error block rendered, others still present.
    assert "Panel 1 unavailable" in html
    for n in range(2, 11):
        assert f"stub-{n}" in html


def test_cherry_pick_empty_state(tmp_path, monkeypatch):
    """If no cherry_pick_ranking_*.csv exists, panel 5 emits empty-state, not a crash."""
    monkeypatch.setattr(go_portal, "RUNTIME_DIR", tmp_path)
    body = go_portal.panel_cherry_pick_top5()
    assert "No cherry_pick_ranking" in body


def test_drafts_empty_state(tmp_path, monkeypatch):
    """Missing drafts directory -> empty-state, no crash."""
    monkeypatch.setattr(go_portal, "DRAFTS_DIR", tmp_path / "does_not_exist")
    body = go_portal.panel_drafts()
    assert "No drafts directory" in body


def test_drafts_rejected_sidecar(tmp_path, monkeypatch):
    """Drafts paired with .rejected.txt show REJECTED in the sidecar column."""
    monkeypatch.setattr(go_portal, "DRAFTS_DIR", tmp_path)
    draft = tmp_path / "2026-05-19-foo.draft.yaml"
    draft.write_text(
        "metadata:\n  theory_grant: false\n  total_expected_trials: 1\n  purpose: 'test purpose string'\n",
        encoding="utf-8",
    )
    rejection = tmp_path / "2026-05-19-foo.rejected.txt"
    rejection.write_text("rejected\n", encoding="utf-8")
    body = go_portal.panel_drafts()
    assert "REJECTED" in body
    assert "test purpose string" in body


def test_journal_pending_no_entries(tmp_path, monkeypatch):
    journal = tmp_path / "j.yaml"
    journal.write_text(
        "schema_version: 1\nentries:\n  - iter: 1\n    strategy_id: X\n    heavyweight_verdict: PASS_CHORDIA\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(go_portal, "JOURNAL_PATH", journal)
    body = go_portal.panel_journal_pending()
    assert "All journal entries have resolved verdicts" in body


def test_journal_pending_shows_null_and_deferred(tmp_path, monkeypatch):
    journal = tmp_path / "j.yaml"
    journal.write_text(
        "schema_version: 1\nentries:\n"
        "  - iter: 1\n    strategy_id: X_NULL\n"
        "    heavyweight_verdict: null\n    oos_power_tier: NA_NO_OOS\n"
        "  - iter: 2\n    strategy_id: Y_DEFERRED\n"
        "    heavyweight_verdict: DEFERRED_NOT_RUN\n    oos_power_tier: STATISTICALLY_USELESS\n"
        "  - iter: 3\n    strategy_id: Z_PASS\n"
        "    heavyweight_verdict: PASS_CHORDIA\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(go_portal, "JOURNAL_PATH", journal)
    body = go_portal.panel_journal_pending()
    assert "X_NULL" in body
    assert "Y_DEFERRED" in body
    assert "Z_PASS" not in body


def test_main_writes_html(tmp_path, monkeypatch, patched_freshness_drift, capsys):
    out_path = tmp_path / "portal.html"
    rc = go_portal.main(["--no-open", "--out", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    body = out_path.read_text(encoding="utf-8")
    assert "OKAY GO portal" in body
    # main() prints the output path so the operator can find it even if browser-open fails.
    captured = capsys.readouterr()
    assert str(out_path) in captured.out


def test_main_json_mode(monkeypatch, patched_freshness_drift, capsys):
    rc = go_portal.main(["--json"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["panels"] == {str(i): True for i in range(1, 11)}
    assert payload["git_sha"] == "deadbeef"


def test_render_portal_shares_bulk_fitness_between_panels(monkeypatch):
    validated_rows = [
        {"strategy_id": "MNQ_A", "instrument": "MNQ"},
        {"strategy_id": "ES_B", "instrument": "ES"},
    ]
    shared_map = {"MNQ_A": "FIT", "ES_B": "WATCH"}
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        go_portal,
        "_query_data_freshness",
        lambda: {"MNQ": {"max_trading_day": "2026-05-19", "stale_days": 0, "is_stale": False}},
    )
    monkeypatch.setattr(go_portal, "_drift_status", lambda: {"status": "PASS", "detail": "all checks passed"})
    monkeypatch.setattr(go_portal, "_git_sha", lambda: "deadbeef")
    monkeypatch.setattr(
        go_portal, "_fitness_status_map", lambda instruments: shared_map if instruments == {"MNQ", "ES"} else {}
    )

    import scripts.tools.strategy_lab_mcp_server as strategy_lab

    monkeypatch.setattr(strategy_lab, "_list_validated_rows", lambda instrument: validated_rows)

    def _panel1(profile_filter, instrument_filter, include_paused=False, fitness_map=None):
        seen["panel1_map"] = fitness_map
        return "<p>stub-1</p>"

    def _panel2(instrument_filter, *, validated_rows=None, fitness_map=None):
        seen["panel2_rows"] = validated_rows
        seen["panel2_map"] = fitness_map
        return "<p>stub-2</p>"

    monkeypatch.setattr(go_portal, "panel_deployed_lanes", _panel1)
    monkeypatch.setattr(go_portal, "panel_promotable", _panel2)
    monkeypatch.setattr(go_portal, "panel_promote_queue", lambda *a, **kw: ("<p>stub-3</p>", []))
    monkeypatch.setattr(go_portal, "panel_oos_rejections", lambda *a, **kw: "<p>stub-4</p>")
    monkeypatch.setattr(go_portal, "panel_cherry_pick_top5", lambda: "<p>stub-5</p>")
    monkeypatch.setattr(go_portal, "panel_drafts", lambda: "<p>stub-6</p>")
    monkeypatch.setattr(go_portal, "panel_journal_pending", lambda: "<p>stub-7</p>")
    monkeypatch.setattr(go_portal, "panel_next_24h", lambda *a, **kw: "<p>stub-8</p>")
    monkeypatch.setattr(go_portal, "panel_holdout", lambda *a, **kw: "<p>stub-9</p>")
    monkeypatch.setattr(go_portal, "panel_freshness_drift", lambda *a, **kw: "<p>stub-10</p>")

    go_portal.render_portal()

    assert seen["panel1_map"] is shared_map
    assert seen["panel2_map"] is shared_map
    assert seen["panel2_rows"] is validated_rows


def _score(strategy_id: str, fitness_status: str) -> FitnessScore:
    return FitnessScore(
        strategy_id=strategy_id,
        full_period_exp_r=0.1,
        full_period_sharpe=1.0,
        full_period_sample=100,
        rolling_exp_r=0.1,
        rolling_sharpe=1.0,
        rolling_win_rate=0.55,
        rolling_sample=25,
        rolling_window_months=18,
        recent_sharpe_30=0.8,
        recent_sharpe_60=0.9,
        sharpe_delta_30=-0.2,
        sharpe_delta_60=-0.1,
        fitness_status=fitness_status,
        fitness_notes=f"{fitness_status} notes",
    )


def test_panel_deployed_lanes_hides_paused_by_default_and_uses_bulk_fitness(monkeypatch):
    import scripts.tools.strategy_lab_mcp_server as strategy_lab
    import trading_app.strategy_fitness as strategy_fitness

    go_portal._fitness_status_map_for_instrument.cache_clear()

    doc = {"lanes": [{"strategy_id": "MNQ_ACTIVE", "session_id": "NYSE_OPEN", "entry_model": "E2"}], "paused": []}
    idx = {
        "MNQ_ACTIVE": {
            "strategy_id": "MNQ_ACTIVE",
            "_allocation_state": "active",
            "session_id": "NYSE_OPEN",
            "entry_model": "E2",
        },
        "MNQ_PAUSED": {
            "strategy_id": "MNQ_PAUSED",
            "_allocation_state": "paused",
            "session_id": "COMEX_SETTLE",
            "entry_model": "E1",
        },
    }

    monkeypatch.setattr(strategy_lab, "_load_allocation_doc", lambda *a, **kw: doc)
    monkeypatch.setattr(strategy_lab, "_allocation_index", lambda *a, **kw: idx)
    monkeypatch.setattr(
        go_portal, "_selected_fitness_status_map", lambda strategy_ids: {"MNQ_ACTIVE": "FIT", "MNQ_PAUSED": "WATCH"}
    )

    def _boom(*args, **kwargs):
        raise AssertionError("per-strategy compute_fitness should not be used")

    monkeypatch.setattr(strategy_fitness, "compute_fitness", _boom)

    body = go_portal.panel_deployed_lanes(None, None)

    assert "MNQ_ACTIVE" in body
    assert "MNQ_PAUSED" not in body
    assert "1 paused hidden" in body


def test_panel_deployed_lanes_include_paused_opt_in(monkeypatch):
    import scripts.tools.strategy_lab_mcp_server as strategy_lab
    import trading_app.strategy_fitness as strategy_fitness

    go_portal._fitness_status_map_for_instrument.cache_clear()

    doc = {"lanes": [{"strategy_id": "MNQ_ACTIVE"}], "paused": [{"strategy_id": "MNQ_PAUSED"}]}
    idx = {
        "MNQ_ACTIVE": {
            "strategy_id": "MNQ_ACTIVE",
            "_allocation_state": "active",
            "session_id": "NYSE_OPEN",
            "entry_model": "E2",
        },
        "MNQ_PAUSED": {
            "strategy_id": "MNQ_PAUSED",
            "_allocation_state": "paused",
            "session_id": "COMEX_SETTLE",
            "entry_model": "E1",
        },
    }

    monkeypatch.setattr(strategy_lab, "_load_allocation_doc", lambda *a, **kw: doc)
    monkeypatch.setattr(strategy_lab, "_allocation_index", lambda *a, **kw: idx)
    monkeypatch.setattr(
        go_portal, "_selected_fitness_status_map", lambda strategy_ids: {"MNQ_ACTIVE": "FIT", "MNQ_PAUSED": "WATCH"}
    )

    body = go_portal.panel_deployed_lanes(None, None, include_paused=True)

    assert "MNQ_ACTIVE" in body
    assert "MNQ_PAUSED" in body
    assert "paused hidden" not in body


def test_panel_promotable_uses_bulk_fitness_not_per_row(monkeypatch):
    import scripts.tools.strategy_lab_mcp_server as strategy_lab
    import trading_app.strategy_fitness as strategy_fitness

    go_portal._fitness_status_map_for_instrument.cache_clear()

    validated = [
        {
            "strategy_id": "MNQ_FIT",
            "instrument": "MNQ",
            "orb_label": "NYSE_OPEN",
            "entry_model": "E2",
            "expectancy_r": 0.1234,
            "sample_size": 99,
        },
        {
            "strategy_id": "MNQ_WATCH",
            "instrument": "MNQ",
            "orb_label": "COMEX_SETTLE",
            "entry_model": "E1",
            "expectancy_r": 0.1111,
            "sample_size": 88,
        },
    ]

    monkeypatch.setattr(strategy_lab, "_list_validated_rows", lambda instrument: validated)
    monkeypatch.setattr(strategy_lab, "_load_allocation_doc", lambda *a, **kw: {"lanes": [], "paused": []})
    monkeypatch.setattr(strategy_lab, "_allocation_index", lambda *a, **kw: {})
    monkeypatch.setattr(
        strategy_fitness,
        "compute_portfolio_fitness",
        lambda **kwargs: FitnessReport(
            as_of_date=date(2026, 6, 1),
            scores=[_score("MNQ_FIT", "FIT"), _score("MNQ_WATCH", "WATCH")],
            summary={"fit": 1, "watch": 1, "decay": 0, "stale": 0},
        ),
    )

    def _boom(*args, **kwargs):
        raise AssertionError("per-strategy compute_fitness should not be used")

    monkeypatch.setattr(strategy_fitness, "compute_fitness", _boom)

    body = go_portal.panel_promotable(None)

    assert "MNQ_FIT" in body
    assert "MNQ_WATCH" not in body


def test_panel_deployed_lanes_uses_supplied_shared_fitness_map(monkeypatch):
    import scripts.tools.strategy_lab_mcp_server as strategy_lab

    doc = {"lanes": [{"strategy_id": "MNQ_ACTIVE"}], "paused": []}
    idx = {
        "MNQ_ACTIVE": {
            "strategy_id": "MNQ_ACTIVE",
            "_allocation_state": "active",
            "session_id": "NYSE_OPEN",
            "entry_model": "E2",
        }
    }

    monkeypatch.setattr(strategy_lab, "_load_allocation_doc", lambda *a, **kw: doc)
    monkeypatch.setattr(strategy_lab, "_allocation_index", lambda *a, **kw: idx)
    monkeypatch.setattr(
        go_portal,
        "_selected_fitness_status_map",
        lambda strategy_ids: (_ for _ in ()).throw(AssertionError("shared fitness map should bypass selected reload")),
    )

    body = go_portal.panel_deployed_lanes(None, None, fitness_map={"MNQ_ACTIVE": "FIT"})

    assert "MNQ_ACTIVE" in body
    assert "FIT" in body
