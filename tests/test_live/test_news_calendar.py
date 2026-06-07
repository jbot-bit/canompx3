"""Real pytest port of the news_calendar awareness module's self-checks.

Ported from the handoff script-style suites (test_news_calendar / _extra /
_payload / _review) into idiomatic ``test_*`` functions. Every assertion is
network-isolated: ``fetch_calendar`` is only ever exercised with an injected
``loader=`` so no real HTTP is attempted, and cache/fired ledgers write to
``tmp_path``. Session timing is proven against the *canonical* ``pipeline.dst``
resolvers (not re-encoded here), including a DST winter!=summer assertion.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime, timedelta

import pytest

from pipeline.dst import orb_utc_window
from trading_app.live import news_calendar as nc

# --- Shared fixtures of feed-shaped raw events (faireconomy JSON shape) --------

EVENTS = [
    {
        "title": "ISM Manufacturing PMI",
        "country": "USD",
        "date": "2026-06-01T10:00:00-04:00",
        "impact": "High",
        "forecast": "53.3",
        "previous": "52.7",
    },
    {
        "title": "ADP Non-Farm Employment",
        "country": "USD",
        "date": "2026-06-03T08:15:00-04:00",
        "impact": "High",
        "forecast": "118K",
        "previous": "109K",
    },
    {
        "title": "ISM Services PMI",
        "country": "USD",
        "date": "2026-06-03T10:00:00-04:00",
        "impact": "High",
        "forecast": "53.7",
        "previous": "53.6",
    },
    {
        "title": "Non-Farm Employment Change",
        "country": "USD",
        "date": "2026-06-05T08:30:00-04:00",
        "impact": "High",
        "forecast": "85K",
        "previous": "115K",
        "actual": "52K",
    },
    {
        "title": "Average Hourly Earnings m/m",
        "country": "USD",
        "date": "2026-06-05T08:30:00-04:00",
        "impact": "High",
        "forecast": "0.3%",
        "previous": "0.2%",
    },
    {
        "title": "BOE Gov Bailey Speaks",
        "country": "GBP",
        "date": "2026-06-04T11:40:00-04:00",
        "impact": "High",
        "forecast": "",
        "previous": "",
    },
    {
        "title": "JOLTS Job Openings",
        "country": "USD",
        "date": "2026-06-02T10:00:00-04:00",
        "impact": "Medium",
        "forecast": "6.87M",
        "previous": "6.87M",
    },
    {
        "title": "Crude Oil Inventories",
        "country": "USD",
        "date": "2026-06-03T10:30:00-04:00",
        "impact": "Low",
        "forecast": "-2.9M",
        "previous": "-3.3M",
    },
    {"title": "Malformed Date Event", "country": "USD", "date": "not-a-date", "impact": "High", "forecast": "1"},
]

NOW_PRE = datetime(2026, 5, 30, tzinfo=UTC)


@pytest.fixture
def relevant():
    """The 5 USD-High events surviving the anti-spam filter, keyed by title."""
    rel = nc.relevant_events(EVENTS, now_utc=NOW_PRE)
    return rel, {e["title"]: e for e in rel}


# --- [1] anti-spam filter -----------------------------------------------------


def test_filter_keeps_only_usd_high(relevant):
    rel, _ = relevant
    assert len(rel) == 5


def test_filter_drops_non_usd_high(relevant):
    _, by = relevant
    assert "BOE Gov Bailey Speaks" not in by  # GBP High dropped


def test_filter_drops_usd_medium(relevant):
    _, by = relevant
    assert "JOLTS Job Openings" not in by


def test_filter_drops_usd_low(relevant):
    _, by = relevant
    assert "Crude Oil Inventories" not in by


def test_filter_drops_malformed_date_without_crashing(relevant):
    _, by = relevant
    assert "Malformed Date Event" not in by


# --- [2] canonical session mapping --------------------------------------------


def test_nfp_maps_in_us_data_830(relevant):
    _, by = relevant
    nfp = by["Non-Farm Employment Change"]
    assert nfp["session"] == "US_DATA_830"
    assert nfp["session_status"] == "in"


def test_nfp_brisbane_2230_summer(relevant):
    _, by = relevant
    assert by["Non-Farm Employment Change"]["when_bris"].strftime("%H:%M") == "22:30"


def test_ism_manufacturing_maps_us_data_1000(relevant):
    _, by = relevant
    assert by["ISM Manufacturing PMI"]["session"] == "US_DATA_1000"


def test_adp_0815_is_near_us_data_830(relevant):
    _, by = relevant
    adp = by["ADP Non-Farm Employment"]
    assert adp["session_status"] == "near"
    assert adp["session"] == "US_DATA_830"


# --- [3] effect / surprise ----------------------------------------------------


def test_nfp_effect_released(relevant):
    _, by = relevant
    assert by["Non-Farm Employment Change"]["effect"]["released"] is True


def test_nfp_surprise_pct(relevant):
    _, by = relevant
    # actual 52K vs forecast 85K -> (52-85)/85 * 100 = -38.8%
    assert by["Non-Farm Employment Change"]["effect"]["surprise_pct"] == -38.8


def test_ahe_not_released(relevant):
    _, by = relevant
    assert by["Average Hourly Earnings m/m"]["effect"]["released"] is False


# --- [4] parse_num edge cases -------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", None),
        (None, None),
        ("2.54|3.9", None),  # ambiguous pipe
        ("<0.1%", None),  # inequality
        (">5", None),
        ("~3", None),  # approx
        ("6.87M", 6_870_000),
        ("-23.1B", -23.1e9),
        ("0.3%", 0.3),
        ("1.2T", 1.2e12),
        ("118K", 118_000),
        ("85K", 85_000),
    ],
)
def test_parse_num(raw, expected):
    assert nc.parse_num(raw) == expected


# --- [5] fire-once alerts -----------------------------------------------------


def test_due_alert_fires_once(relevant):
    rel, by = relevant
    nfp_dt = by["Non-Farm Employment Change"]["when_utc"]
    now = nfp_dt - timedelta(minutes=10)
    send1, fired = nc.due_alerts(rel, now_utc=now, fired=set())
    assert any("Non-Farm" in e["title"] for e in send1)
    send2, _ = nc.due_alerts(rel, now_utc=now, fired=fired)
    assert len(send2) == 0  # no repeat — fire-once


def test_due_alerts_lead_zero_fires():
    nfp = [
        {
            "title": "NFP",
            "country": "USD",
            "date": "2026-06-05T08:30:00-04:00",
            "impact": "High",
            "forecast": "85K",
            "actual": "52K",
        }
    ]
    rel = nc.relevant_events(nfp, now_utc=NOW_PRE)
    w = rel[0]["when_utc"]
    sent, _ = nc.due_alerts(rel, now_utc=w, fired=set())
    assert len(sent) == 1


def test_due_alerts_passed_event_no_fire():
    nfp = [
        {
            "title": "NFP",
            "country": "USD",
            "date": "2026-06-05T08:30:00-04:00",
            "impact": "High",
            "forecast": "85K",
            "actual": "52K",
        }
    ]
    rel = nc.relevant_events(nfp, now_utc=NOW_PRE)
    w = rel[0]["when_utc"]
    sent, _ = nc.due_alerts(rel, now_utc=w + timedelta(minutes=1), fired=set())
    assert len(sent) == 0


def test_due_alerts_outside_15m_window_no_fire():
    nfp = [
        {
            "title": "NFP",
            "country": "USD",
            "date": "2026-06-05T08:30:00-04:00",
            "impact": "High",
            "forecast": "85K",
            "actual": "52K",
        }
    ]
    rel = nc.relevant_events(nfp, now_utc=NOW_PRE)
    w = rel[0]["when_utc"]
    sent, _ = nc.due_alerts(rel, now_utc=w - timedelta(minutes=16), fired=set())
    assert len(sent) == 0


# --- [6] DST-flip + NYSE midnight-crossing (canonical trading-day) ------------

EDGE = [
    {"title": "CPI winter", "country": "USD", "date": "2026-01-13T08:30:00-05:00", "impact": "High", "forecast": "0.3"},
    {"title": "NFP summer", "country": "USD", "date": "2026-06-05T08:30:00-04:00", "impact": "High", "forecast": "85K"},
    {
        "title": "NYSE cash open",
        "country": "USD",
        "date": "2026-06-05T09:30:00-04:00",
        "impact": "High",
        "forecast": "1",
    },
]


@pytest.fixture
def edge_by():
    return {e["title"]: e for e in nc.relevant_events(EDGE, now_utc=datetime(2026, 1, 1, tzinfo=UTC))}


def test_winter_0830et_maps_us_data_830(edge_by):
    cpi = edge_by["CPI winter"]
    assert cpi["session"] == "US_DATA_830"
    assert cpi["session_status"] == "in"


def test_summer_0830et_maps_us_data_830(edge_by):
    nfp = edge_by["NFP summer"]
    assert nfp["session"] == "US_DATA_830"
    assert nfp["session_status"] == "in"


def test_dst_shift_proven_winter_bris_differs_from_summer(edge_by):
    """Both events are 08:30 ET, but the canonical resolver yields different
    Brisbane wall-times across the DST boundary — proves we are NOT applying a
    fixed offset (the contamination class pipeline.dst guards against)."""
    w = edge_by["CPI winter"]["when_bris"].strftime("%H:%M")
    s = edge_by["NFP summer"]["when_bris"].strftime("%H:%M")
    assert w != s


def test_nyse_0930et_maps_nyse_open(edge_by):
    assert edge_by["NYSE cash open"]["session"] == "NYSE_OPEN"


# --- half-open window boundary (event exactly at window end is NOT 'in') ------


def test_window_start_is_in():
    s, _ = orb_utc_window(date(2026, 6, 5), "US_DATA_830", 15)
    st, lbl, _ = nc.session_for(s, ["US_DATA_830"])
    assert st == "in"


def test_window_end_is_not_in():
    _, e = orb_utc_window(date(2026, 6, 5), "US_DATA_830", 15)
    st, lbl, _ = nc.session_for(e, ["US_DATA_830"])
    assert not (st == "in" and lbl == "US_DATA_830")


def test_window_inside_is_in():
    s, _ = orb_utc_window(date(2026, 6, 5), "US_DATA_830", 15)
    st, _, _ = nc.session_for(s + timedelta(minutes=1), ["US_DATA_830"])
    assert st == "in"


def test_comex_settle_maps():
    cs, _ = orb_utc_window(date(2026, 6, 5), "COMEX_SETTLE", 15)
    st, lbl, _ = nc.session_for(cs)
    assert st == "in"
    assert lbl == "COMEX_SETTLE"


# --- instruments_for ----------------------------------------------------------


def test_instruments_for_usd():
    assert nc.instruments_for({"country": "USD"}) == ("MNQ", "MES", "MGC")


def test_instruments_for_unknown_currency_empty():
    assert nc.instruments_for({"country": "EUR"}) == ()


# --- fetch_calendar: cache / TTL / fail-open / fallback (network-isolated) ----


def test_cold_fetch_writes_cache(tmp_path):
    cache = str(tmp_path / "state" / "news_cache.json")
    now0 = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)
    calls = []

    def good(url):
        calls.append(url)
        return [{"k": 1}]

    assert nc.fetch_calendar(cache_path=cache, loader=good, now=now0) == [{"k": 1}]
    assert len(calls) == 1


def test_warm_cache_within_ttl_skips_loader(tmp_path):
    cache = str(tmp_path / "state" / "news_cache.json")
    now0 = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)
    nc.fetch_calendar(cache_path=cache, loader=lambda u: [{"k": 1}], now=now0)

    def boom(url):
        raise RuntimeError("loader must not be called within TTL")

    got = nc.fetch_calendar(cache_path=cache, loader=boom, now=now0 + timedelta(minutes=10))
    assert got == [{"k": 1}]


def test_expired_cache_refetches(tmp_path):
    cache = str(tmp_path / "state" / "news_cache.json")
    now0 = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)
    nc.fetch_calendar(cache_path=cache, ttl_s=60, loader=lambda u: [{"k": 1}], now=now0)
    got = nc.fetch_calendar(cache_path=cache, ttl_s=60, loader=lambda u: [{"k": 2}], now=now0 + timedelta(hours=2))
    assert got == [{"k": 2}]


def test_expired_and_loader_down_fails_open_to_stale(tmp_path):
    cache = str(tmp_path / "state" / "news_cache.json")
    now0 = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)
    nc.fetch_calendar(cache_path=cache, ttl_s=60, loader=lambda u: [{"k": 2}], now=now0)

    def boom(url):
        raise RuntimeError("net down")

    got = nc.fetch_calendar(cache_path=cache, ttl_s=60, loader=boom, now=now0 + timedelta(hours=4))
    assert got == [{"k": 2}]  # serves last-good cache


def test_no_cache_and_loader_down_falls_back_to_majors(tmp_path):
    cache = str(tmp_path / "missing_cache.json")
    now0 = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)

    def boom(url):
        raise RuntimeError("net down")

    fb = nc.fetch_calendar(cache_path=cache, loader=boom, now=now0)
    assert len(fb) >= 1
    assert fb[0]["impact"] == "High"
    assert fb[0]["country"] == "USD"


# --- fallback_majors: deterministic first-Friday NFP --------------------------


def test_fallback_majors_first_friday_nfp():
    fm = nc.fallback_majors(datetime(2026, 6, 1, tzinfo=UTC), months=1)
    d = datetime.fromisoformat(fm[0]["date"])
    assert d.month == 6
    assert d.day == 5  # first Friday of June 2026
    assert d.weekday() == 4  # Friday
    assert d.hour == 8
    assert d.minute == 30


def test_fallback_majors_spans_requested_months():
    fm = nc.fallback_majors(datetime(2026, 6, 1, tzinfo=UTC), months=2)
    assert len(fm) == 2
    assert all(ev["impact"] == "High" and ev["country"] == "USD" for ev in fm)


def test_fallback_majors_carry_no_fabricated_numbers():
    """Integrity guard: the offline placeholder must NOT invent forecast/actual
    numbers — only a real recurring date with empty value fields."""
    fm = nc.fallback_majors(datetime(2026, 6, 1, tzinfo=UTC), months=2)
    for ev in fm:
        assert ev["forecast"] == ""
        assert ev["previous"] == ""
        assert "actual" not in ev


# --- provenance: fetch_calendar(with_source=True) tells the TRUE source -------


def test_source_live_on_successful_fetch(tmp_path):
    cache = str(tmp_path / "c.json")
    now0 = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)
    events, source = nc.fetch_calendar(cache_path=cache, loader=lambda u: [{"k": 1}], now=now0, with_source=True)
    assert source == "live"
    assert events == [{"k": 1}]


def test_source_cache_within_ttl(tmp_path):
    cache = str(tmp_path / "c.json")
    now0 = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)
    nc.fetch_calendar(cache_path=cache, loader=lambda u: [{"k": 1}], now=now0)
    _, source = nc.fetch_calendar(
        cache_path=cache, loader=lambda u: [{"k": 9}], now=now0 + timedelta(minutes=5), with_source=True
    )
    assert source == "cache"


def test_source_stale_cache_when_feed_down(tmp_path):
    cache = str(tmp_path / "c.json")
    now0 = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)
    nc.fetch_calendar(cache_path=cache, ttl_s=60, loader=lambda u: [{"k": 2}], now=now0)

    def boom(url):
        raise RuntimeError("net down")

    events, source = nc.fetch_calendar(
        cache_path=cache, ttl_s=60, loader=boom, now=now0 + timedelta(hours=4), with_source=True
    )
    assert source == "stale-cache"
    assert events == [{"k": 2}]  # serves last-good, but labelled stale


def test_source_fallback_when_feed_down_and_no_cache(tmp_path):
    cache = str(tmp_path / "missing.json")
    now0 = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)

    def boom(url):
        raise RuntimeError("net down")

    events, source = nc.fetch_calendar(cache_path=cache, loader=boom, now=now0, with_source=True)
    assert source == "fallback"
    assert len(events) >= 1


def test_payload_honest_source_passthrough():
    """news_payload must stamp the TRUE source it is given, never a hardcoded
    'faireconomy' — so a fallback payload cannot masquerade as live."""
    pl = nc.news_payload(PAYLOAD_EVENTS, now_utc=NOW_PRE, source="fallback")
    assert pl["source"] == "fallback"


# --- fired ledger: persist + prune + round-trip -------------------------------


def test_save_fired_keeps_recent(tmp_path):
    fp = str(tmp_path / "fired.json")
    fired = {"NFP|2026-06-05T12:30:00+00:00", "OLD|2026-05-01T00:00:00+00:00"}
    kept = nc.save_fired(fp, fired, now=datetime(2026, 6, 6, tzinfo=UTC), max_age_days=2)
    assert any("NFP" in k for k in kept)


def test_save_fired_prunes_stale(tmp_path):
    fp = str(tmp_path / "fired.json")
    fired = {"NFP|2026-06-05T12:30:00+00:00", "OLD|2026-05-01T00:00:00+00:00"}
    kept = nc.save_fired(fp, fired, now=datetime(2026, 6, 6, tzinfo=UTC), max_age_days=2)
    assert not any("OLD" in k for k in kept)


def test_fired_ledger_round_trips_from_disk(tmp_path):
    fp = str(tmp_path / "fired.json")
    fired = {"NFP|2026-06-05T12:30:00+00:00", "OLD|2026-05-01T00:00:00+00:00"}
    kept = nc.save_fired(fp, fired, now=datetime(2026, 6, 6, tzinfo=UTC), max_age_days=2)
    assert nc.load_fired(fp) == kept


def test_load_fired_missing_file_is_empty(tmp_path):
    assert nc.load_fired(str(tmp_path / "nope.json")) == set()


def test_save_fired_keeps_unparseable_key(tmp_path):
    fp = str(tmp_path / "fired.json")
    kept = nc.save_fired(fp, {"weird-no-pipe-no-iso"}, now=datetime(2026, 6, 6, tzinfo=UTC))
    assert "weird-no-pipe-no-iso" in kept  # kept rather than silently dropped


# --- signal: plain-English signs / tones --------------------------------------


def _signals_by_title(events, now=NOW_PRE):
    return {e["title"]: e["signal"] for e in nc.relevant_events(events, now_utc=now)}


def test_signal_released_big_surprise_is_hot():
    ev = [
        {
            "title": "Non-Farm Employment Change",
            "country": "USD",
            "date": "2026-06-05T08:30:00-04:00",
            "impact": "High",
            "forecast": "85K",
            "actual": "52K",
        }
    ]
    sig = _signals_by_title(ev)["Non-Farm Employment Change"]
    assert sig["sign"] == "HOT"
    assert "vs forecast" in sig["text"]


def test_signal_in_session_is_hot_and_volatile():
    ev = [
        {
            "title": "ISM Manufacturing PMI",
            "country": "USD",
            "date": "2026-06-01T10:00:00-04:00",
            "impact": "High",
            "forecast": "53.3",
        }
    ]
    sig = _signals_by_title(ev)["ISM Manufacturing PMI"]
    assert sig["sign"] == "HOT"
    assert "volatile" in sig["text"]


def test_signal_near_session_is_heads_up():
    ev = [
        {
            "title": "ADP Non-Farm Employment",
            "country": "USD",
            "date": "2026-06-03T08:15:00-04:00",
            "impact": "High",
            "forecast": "118K",
        }
    ]
    sig = _signals_by_title(ev)["ADP Non-Farm Employment"]
    assert sig["sign"] == "HEADS-UP"


def test_signal_outside_session_is_watch():
    out = [
        {"title": "Outside", "country": "USD", "date": "2026-06-03T14:00:00-04:00", "impact": "High", "forecast": "1"}
    ]
    sig = nc.relevant_events(out, now_utc=NOW_PRE)[0]["signal"]
    assert sig["sign"] == "WATCH"
    assert "outside" in sig["text"]


# --- session_only filter (panel-vs-alert contract) ----------------------------


def test_session_only_false_keeps_outside_event():
    out = [
        {
            "title": "Outside Event",
            "country": "USD",
            "date": "2026-06-03T14:00:00-04:00",
            "impact": "High",
            "forecast": "1",
        }
    ]
    kept = nc.relevant_events(out, now_utc=NOW_PRE)
    assert len(kept) == 1
    assert kept[0]["session"] is None


def test_session_only_true_drops_outside_event():
    out = [
        {
            "title": "Outside Event",
            "country": "USD",
            "date": "2026-06-03T14:00:00-04:00",
            "impact": "High",
            "forecast": "1",
        }
    ]
    kept = nc.relevant_events(out, now_utc=NOW_PRE, session_only=True)
    assert len(kept) == 0


# --- news_payload: the exact shape /api/news returns & the panel consumes ------

PAYLOAD_EVENTS = [
    {
        "title": "Non-Farm Employment Change",
        "country": "USD",
        "date": "2026-06-05T08:30:00-04:00",
        "impact": "High",
        "forecast": "85K",
        "actual": "52K",
    },
    {
        "title": "ADP Non-Farm Employment",
        "country": "USD",
        "date": "2026-06-03T08:15:00-04:00",
        "impact": "High",
        "forecast": "118K",
    },
    {"title": "JOLTS", "country": "USD", "date": "2026-06-02T10:00:00-04:00", "impact": "Medium", "forecast": "1"},
]


@pytest.fixture
def payload():
    return nc.news_payload(PAYLOAD_EVENTS, now_utc=NOW_PRE)


def test_payload_top_level_keys(payload):
    assert set(payload) == {"events", "fetched_at", "source"}


def test_payload_filters_medium(payload):
    assert len(payload["events"]) == 2  # JOLTS (Medium) dropped


def test_payload_source(payload):
    assert payload["source"] == "faireconomy"


def test_payload_fetched_at_is_hhmm(payload):
    assert re.fullmatch(r"\d{2}:\d{2}", payload["fetched_at"])


def test_payload_every_event_carries_signal(payload):
    for e in payload["events"]:
        sig = e.get("signal", {})
        assert {"sign", "tone", "text"} <= set(sig)
        assert sig["sign"] in ("HOT", "HEADS-UP", "WATCH")
        assert len(sig["text"]) > 10


def test_payload_is_json_serialisable_no_datetime_leak(payload):
    """relevant_events leaves when_utc/when_bris as datetime; news_payload must
    isoformat them so FastAPI's default JSON encoder never raises."""
    s = json.dumps(payload)
    assert '"signal"' in s
    assert '"when_bris"' in s


def test_payload_datetimes_are_iso_strings(payload):
    for e in payload["events"]:
        assert isinstance(e["when_utc"], str)
        assert isinstance(e["when_bris"], str)
        # round-trips back to an aware datetime
        assert datetime.fromisoformat(e["when_utc"]).tzinfo is not None


def test_relevant_events_rows_carry_signal():
    rows = nc.relevant_events(PAYLOAD_EVENTS, now_utc=NOW_PRE)
    assert "signal" in rows[0]


# --- ordering guarantee -------------------------------------------------------


def test_relevant_events_sorted_by_time(relevant):
    rel, _ = relevant
    times = [e["when_utc"] for e in rel]
    assert times == sorted(times)


# --- /api/news endpoint + canonical-alert wiring (FastAPI, network-isolated) ---
#
# These exercise the real dashboard route handler end-to-end (including the
# asyncio.to_thread offload) with the feed loader monkeypatched — no real HTTP.
# Cache/fired/alert ledgers are redirected to tmp_path so no runtime state is
# touched and the live-arm path is never reached.

from pipeline.dst import (  # noqa: E402
    compute_trading_day_from_timestamp,
    orb_utc_window,
)


@pytest.fixture
def dash(tmp_path, monkeypatch):
    """Dashboard app + news_calendar with ledgers redirected to tmp_path."""
    from fastapi.testclient import TestClient

    from trading_app.live import alert_engine as ae
    from trading_app.live import bot_dashboard as bd

    monkeypatch.setattr(bd, "NEWS_CACHE_PATH", tmp_path / "news_cache.json")
    monkeypatch.setattr(bd, "NEWS_FIRED_PATH", tmp_path / "news_fired.json")
    monkeypatch.setattr(ae, "ALERTS_PATH", tmp_path / "operator_alerts.jsonl")
    return bd, ae, nc, TestClient(bd.app)


def test_api_news_happy_path_valid_payload(dash, monkeypatch):
    bd, ae, ncmod, client = dash
    monkeypatch.setattr(ncmod, "_http_get_json", lambda url, timeout=5: list(PAYLOAD_EVENTS))
    r = client.get("/api/news")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"events", "fetched_at", "source"}
    assert body["source"] == "live"  # successful fetch -> honest 'live' provenance
    assert len(body["events"]) == 2  # Medium JOLTS filtered out


def test_api_news_feed_down_no_cache_fails_open_not_500(dash, monkeypatch):
    bd, ae, ncmod, client = dash

    def boom(url, timeout=5):
        raise RuntimeError("net down")

    monkeypatch.setattr(ncmod, "_http_get_json", boom)
    r = client.get("/api/news")
    assert r.status_code == 200  # fallback majors, never a 500
    body = r.json()
    assert set(body) == {"events", "fetched_at", "source"}
    # honest provenance: feed-down + no cache must report 'fallback', NOT 'live'
    assert body["source"] == "fallback"


def test_api_news_never_leaks_datetime(dash, monkeypatch):
    bd, ae, ncmod, client = dash
    monkeypatch.setattr(ncmod, "_http_get_json", lambda url, timeout=5: list(PAYLOAD_EVENTS))
    body = client.get("/api/news").json()
    for e in body["events"]:
        assert isinstance(e["when_utc"], str)
        assert isinstance(e["when_bris"], str)


def _in_session_due_feed():
    """A USD-High event anchored INSIDE a real US_DATA_830 window via the
    canonical resolver, so relevant_events maps it session_status='in'."""
    td = compute_trading_day_from_timestamp(datetime(2026, 6, 5, 12, 30, tzinfo=UTC))
    start, _ = orb_utc_window(td, "US_DATA_830", 15)
    ev_utc = start + timedelta(minutes=2)
    # feed dates are tz-aware ISO; the module re-converts to UTC, so a UTC ISO
    # string is equivalent input — avoids needing a fixed ET offset here.
    iso = ev_utc.isoformat()
    return [{"title": "NFP", "country": "USD", "date": iso, "impact": "High", "forecast": "85K"}], ev_utc


def test_news_heads_up_fires_once_via_canonical_alert(dash):
    bd, ae, ncmod, _client = dash
    feed, ev_utc = _in_session_due_feed()
    events = ncmod.relevant_events(feed, session_only=True)
    assert events and events[0]["session_status"] == "in"

    now10 = ev_utc - timedelta(minutes=10)
    fired = ncmod.load_fired(str(bd.NEWS_FIRED_PATH))
    due, fired = ncmod.due_alerts(events, now_utc=now10, fired=fired)
    assert len(due) == 1
    for ev in due:
        hm = ev["when_bris"].strftime("%H:%M")
        ae.record_operator_alert(
            message=f"NEWS HEADS-UP: {ev['title']} @ {ev['session']} {hm} Bris",
            source="news",
        )
    ncmod.save_fired(str(bd.NEWS_FIRED_PATH), fired)

    alerts = ae.read_operator_alerts(limit=10)
    news = [a for a in alerts if a.get("category") == "news_event"]
    assert len(news) == 1
    assert news[0]["level"] == "warning"
    assert news[0]["source"] == "news"
    assert news[0]["message"].startswith("NEWS HEADS-UP:")

    # fire-once: replaying with the persisted ledger yields no new due alert
    fired2 = ncmod.load_fired(str(bd.NEWS_FIRED_PATH))
    due2, _ = ncmod.due_alerts(events, now_utc=now10, fired=fired2)
    assert len(due2) == 0


def test_news_heads_up_classifies_warning():
    from trading_app.live.alert_engine import classify_operator_alert

    level, category = classify_operator_alert("NEWS HEADS-UP: ISM Manufacturing PMI @ US_DATA_1000 00:00 Bris")
    assert level == "warning"
    assert category == "news_event"
