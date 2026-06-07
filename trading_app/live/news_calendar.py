"""High-impact economic-news awareness for the ORB dashboard (display + alert only).

Source: faireconomy weekly JSON mirror of the Forex Factory calendar
(no API key, no scraping). AWARENESS-ONLY: never touches entry/filter/capital.
Session timing AND trading-day attribution come from canonical pipeline.dst
helpers (orb_utc_window, compute_trading_day_from_timestamp) — never re-encoded.

Portable across Python 3.10 (sandbox) and 3.11+ (repo): uses timezone.utc.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from pipeline.dst import (
    SESSION_CATALOG,
    compute_trading_day_from_timestamp,
    orb_utc_window,
)

BRISBANE = ZoneInfo("Australia/Brisbane")
log = logging.getLogger(__name__)

CCY_INSTRUMENTS = {"USD": ("MNQ", "MES", "MGC")}
US_SESSION_LABELS = (
    "US_DATA_830", "NYSE_PREOPEN", "NYSE_OPEN", "US_DATA_1000",
    "COMEX_SETTLE", "NYSE_CLOSE",
)
FEED_THISWEEK = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
# Rationale: an event is flagged "near" a session if it lands within 60 min
# before the ORB window opens. One hour is the standard pre-event positioning
# window — it spans the typical 30-60 min build-up of volatility/liquidity
# ahead of a high-impact print, while staying short enough that an event an
# hour out is genuinely relevant to the upcoming session (not next-session
# noise). Awareness-only threshold; never gates entry/sizing.
NEAR_WINDOW_MIN = 60
# Rationale: fire the operator heads-up alert at most 15 min before the event.
# 15 min is the conventional "final prep" lead — long enough to read the alert
# and check exposure before the print, short enough to avoid premature alerts
# that the operator forgets by event time. Awareness-only; never gates capital.
PRE_ALERT_MIN = 15


def parse_num(raw):
    """Parse forecast/actual. None for blank/ambiguous (x|y, ranges, <, >)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or any(c in s for c in ("|", "<", ">", "~")):
        return None
    neg = s.startswith("-")
    s = s.lstrip("+-").replace(",", "").replace("%", "")
    mult = 1.0
    if s[-1:].upper() in ("K", "M", "B", "T"):
        mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[s[-1].upper()]
        s = s[:-1]
    try:
        val = float(s) * mult
    except ValueError:
        return None
    return -val if neg else val


def is_relevant(ev):
    return ev.get("impact") == "High" and ev.get("country") in CCY_INSTRUMENTS


def instruments_for(ev):
    return CCY_INSTRUMENTS.get(ev.get("country", ""), ())


def session_for(dt_utc, labels=US_SESSION_LABELS, orb_minutes=15):
    """Attribute an event to an ORB session via canonical resolvers.
    Trading day from compute_trading_day_from_timestamp (handles 09:00-Bris
    boundary + NYSE midnight-crossing). Returns (status, label, minutes)."""
    td = compute_trading_day_from_timestamp(dt_utc)
    nearest = None
    for lbl in labels:
        if lbl not in SESSION_CATALOG:
            continue
        try:
            start, end = orb_utc_window(td, lbl, orb_minutes)
        except (KeyError, ValueError):
            log.warning("orb_utc_window unresolved for %s on %s; skipping", lbl, td, exc_info=True)
            continue
        if start <= dt_utc < end:
            return ("in", lbl, 0)
        gap_min = (start - dt_utc).total_seconds() / 60
        if 0 < gap_min <= NEAR_WINDOW_MIN and (nearest is None or gap_min < nearest[2]):
            nearest = ("near", lbl, int(gap_min))
    return nearest if nearest else (None, None, None)


def effect(ev):
    out = {"severity": ev.get("impact"), "surprise_pct": None, "released": False}
    actual, forecast = parse_num(ev.get("actual")), parse_num(ev.get("forecast"))
    if actual is not None and forecast not in (None, 0):
        out["released"] = True
        out["surprise_pct"] = round((actual - forecast) / abs(forecast) * 100, 1)
    return out


def relevant_events(raw_events, now_utc=None, labels=US_SESSION_LABELS, orb_minutes=15, session_only=False):
    now_utc = now_utc or datetime.now(UTC)
    out = []
    for ev in raw_events:
        if not is_relevant(ev):
            continue
        try:
            dt = datetime.fromisoformat(ev["date"]).astimezone(UTC)
        except (KeyError, ValueError):
            continue
        status, label, mins = session_for(dt, labels, orb_minutes)
        if session_only and label is None:
            continue
        row = {
            "title": ev.get("title", "?"),
            "when_utc": dt,
            "when_bris": dt.astimezone(BRISBANE),
            "instruments": instruments_for(ev),
            "session": label,
            "session_status": status,
            "session_gap_min": mins,
            "effect": effect(ev),
            "upcoming": dt > now_utc,
        }
        row["signal"] = signal(row)
        out.append(row)
    out.sort(key=lambda e: e["when_utc"])
    return out


def due_alerts(events, now_utc=None, fired=None):
    """Fire-once pre-event heads-up. Caller persists `fired` across restarts."""
    now_utc = now_utc or datetime.now(UTC)
    fired = set(fired or ())
    send = []
    for e in events:
        if e["session"] is None:
            continue
        key = e["title"] + "|" + e["when_utc"].isoformat()
        lead = (e["when_utc"] - now_utc).total_seconds() / 60
        if 0 <= lead <= PRE_ALERT_MIN and key not in fired:
            send.append(e)
            fired.add(key)
    return send, fired


# ===== Fetch / cache / fallback / fired-ledger (network-isolated for testing) =====

def _read_cache(path):
    import json
    import os
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return {"ts": datetime.fromisoformat(d["ts"]), "events": d["events"]}
    except Exception:
        return None


def _write_cache(path, events, now):
    import json
    import os
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"ts": now.isoformat(), "events": events}, fh)
    os.replace(tmp, path)


def _http_get_json(url, timeout=5):
    import json
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "orb-dashboard/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (trusted host)
        return json.loads(r.read().decode("utf-8"))


def fallback_majors(now=None, months=2):
    """Deterministic recurring US High events for when the feed is unavailable.
    NFP = first Friday 08:30 America/New_York. Returns feed-shaped rows."""
    import calendar as _cal
    from datetime import datetime as _dt
    now = now or datetime.now(UTC)
    ny = ZoneInfo("America/New_York")
    out, y, m = [], now.year, now.month
    for _ in range(months):
        fridays = [d for d in _cal.Calendar().itermonthdates(y, m)
                   if d.month == m and d.weekday() == 4]
        f = fridays[0]
        dt = _dt(f.year, f.month, f.day, 8, 30, tzinfo=ny)
        out.append({"title": "Non-Farm Employment Change", "country": "USD",
                    "date": dt.isoformat(), "impact": "High", "forecast": "", "previous": ""})
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def fetch_calendar(*, cache_path, ttl_s=3600, loader=None, now=None, with_source=False):
    """Raw events with TTL cache, fail-open to stale cache, else fallback majors.
    `loader` is injectable so the network call can be tested in isolation.

    When `with_source=True`, returns (events, source) where source is one of
    'live' | 'cache' | 'stale-cache' | 'fallback' — so callers can tell the
    operator the TRUE provenance and never imply live data when it is offline
    fallback or stale. Default (False) returns just events for back-compat."""
    loader = loader or _http_get_json
    now = now or datetime.now(UTC)
    cached = _read_cache(cache_path)
    if cached and (now - cached["ts"]).total_seconds() < ttl_s:
        return (cached["events"], "cache") if with_source else cached["events"]
    try:
        events = loader(FEED_THISWEEK)
        _write_cache(cache_path, events, now)
        return (events, "live") if with_source else events
    except Exception:
        log.warning("news feed fetch failed; falling back to %s",
                    "stale cache" if cached else "deterministic majors", exc_info=True)
        if cached:
            # fail-open to last-good — but mark it stale so the UI says so.
            return (cached["events"], "stale-cache") if with_source else cached["events"]
        # last resort: deterministic recurring majors (NFP first-Friday); these
        # carry empty forecast/previous — no fabricated numbers — and are clearly
        # labelled 'fallback' so they are never mistaken for live feed data.
        return (fallback_majors(now), "fallback") if with_source else fallback_majors(now)


def load_fired(path):
    import json
    import os
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return set(json.load(fh))
    except Exception:
        return set()


def save_fired(path, fired, now=None, max_age_days=2):
    """Persist the fire-once ledger, pruning keys whose event time is >2 days old."""
    import json
    import os
    now = now or datetime.now(UTC)
    keep = set()
    for k in fired:
        iso = k.rsplit("|", 1)[-1]
        try:
            ts = datetime.fromisoformat(iso)
            if (now - ts).total_seconds() <= max_age_days * 86400:
                keep.add(k)
        except ValueError:
            keep.add(k)  # unparseable -> keep rather than silently drop
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(sorted(keep), fh)
    os.replace(tmp, path)
    return keep


# ===== Plain-English "action sign" (awareness only — no size/skip instructions) =====

def signal(e):
    """Turn a relevant_events row into an easy-ingest sign + statement.
    sign: HOT | HEADS-UP | WATCH ; tone: red | amber | cyan.
    Describes EXPECTED volatility only — never a position/size/skip instruction."""
    ins = "/".join(e.get("instruments") or ()) or "MNQ/MES/MGC"
    name = e.get("title", "?")
    hm = e["when_bris"].strftime("%H:%M") if e.get("when_bris") else "??:??"
    eff = e.get("effect") or {}
    st, sess = e.get("session_status"), e.get("session")
    if eff.get("released") and eff.get("surprise_pct") is not None:
        sp = eff["surprise_pct"]
        big = abs(sp) >= 10
        magnitude = "outsized" if big else "modest"
        return {"sign": "HOT" if big else "WATCH", "tone": "red" if big else "cyan",
                "text": f"{name} came in {sp:+.0f}% vs forecast — {magnitude} {ins} move likely."}
    if st == "in":
        return {"sign": "HOT", "tone": "red",
                "text": f"{name} lands inside {sess} ({hm} Bris) — expect a volatile, choppy {ins} open."}
    if st == "near":
        gap = e.get("session_gap_min") or 0
        return {"sign": "HEADS-UP", "tone": "amber",
                "text": f"{name} ~{gap}m before {sess} ({hm} Bris) — early {ins} volatility possible."}
    return {"sign": "WATCH", "tone": "cyan",
            "text": f"{name} at {hm} Bris — high-impact but outside your sessions."}


def news_payload(raw_events, now_utc=None, source="faireconomy", **kw):
    """Exact JSON shape the dashboard /api/news route should return and the
    panel consumes: {events:[...with signal, ISO datetimes...], fetched_at, source}.
    Endpoint becomes a one-liner around this — no per-event mapping to forget.

    `source` is the TRUE data provenance (pass the second element of
    fetch_calendar(..., with_source=True)) so the panel never implies live feed
    data when it is actually 'cache' / 'stale-cache' / 'fallback'."""
    now_utc = now_utc or datetime.now(UTC)
    evs = relevant_events(raw_events, now_utc=now_utc, **kw)
    out = []
    for e in evs:
        d = dict(e)
        d["when_utc"] = e["when_utc"].isoformat()
        d["when_bris"] = e["when_bris"].isoformat()
        out.append(d)
    return {"events": out,
            "fetched_at": now_utc.astimezone(BRISBANE).strftime("%H:%M"),
            "source": source}
