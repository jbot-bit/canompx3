"""OKAY GO portal — read-only HTML aggregator.

Single entry point that surfaces the research -> validation -> ready-to-trade
pipeline at a glance. Ten panels, all sourced from canonical surfaces, no
discovery, no writes outside the rendered HTML.

Doctrine notes:
  * `validated_setups` is a DERIVED layer banned for DISCOVERY by
    `.claude/rules/research-truth-protocol.md` Layer Classification. This
    portal reads it for DISPLAY only (panel 2 - promotable candidates) via
    the canonical helper `_list_validated_rows` from
    `scripts.tools.strategy_lab_mcp_server`. No filter logic is re-encoded
    here; the helper already applies `deployable_validated_relation`.
  * SESSION_CATALOG, COST_SPECS, GOLD_DB_PATH, HOLDOUT_SACRED_FROM, and
    `effective_daily_lanes()` are queried live per the volatile-data rule
    (CLAUDE.md). Nothing is cached.
  * `compute_fitness` is date-agnostic (reads current `daily_features`
    regime row) so this portal has no `--date` flag - everything renders
    as-of-now.
  * Per-panel try/except wraps each render so one bad data source can't
    take the whole dashboard down (`institutional-rigor.md` Section 6:
    no silent failures - errors render explicitly).
  * Open in browser by default per CLAUDE.md "open in browser by default"
    convention; `--no-open` is opt-out.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import traceback
import webbrowser
from datetime import date, datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNTIME_DIR = PROJECT_ROOT / "docs" / "runtime"
DRAFTS_DIR = PROJECT_ROOT / "docs" / "audit" / "hypotheses" / "drafts"
JOURNAL_PATH = RUNTIME_DIR / "cherry_pick_journal.yaml"
STALENESS_TRADING_DAYS = 2  # red banner threshold


# ---------------------------------------------------------------------------
# data freshness + drift (panel 10 + top banner)
# ---------------------------------------------------------------------------


def _query_data_freshness() -> dict[str, Any]:
    import duckdb

    from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
    from pipeline.paths import GOLD_DB_PATH

    today = date.today()
    rows: dict[str, dict[str, Any]] = {}
    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        for inst in sorted(ACTIVE_ORB_INSTRUMENTS):
            r = con.execute(
                "SELECT MAX(trading_day) FROM orb_outcomes WHERE symbol = ?",
                [inst],
            ).fetchone()
            max_td = r[0] if r else None
            stale_days = (today - max_td).days if isinstance(max_td, date) else None
            rows[inst] = {
                "max_trading_day": max_td.isoformat() if isinstance(max_td, date) else None,
                "stale_days": stale_days,
                "is_stale": (stale_days is not None and stale_days > STALENESS_TRADING_DAYS),
            }
    return rows


def _drift_status() -> dict[str, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "pipeline/check_drift.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=240,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"status": "UNKNOWN", "detail": f"check_drift.py did not complete: {exc}"}
    status = "PASS" if proc.returncode == 0 else "FAIL"
    tail = (proc.stdout or proc.stderr or "").splitlines()[-3:]
    return {"status": status, "detail": " | ".join(tail)[:400]}


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------


def panel_deployed_lanes(profile_filter: str | None, instrument_filter: str | None) -> str:
    from scripts.tools.strategy_lab_mcp_server import _allocation_index, _load_allocation_doc
    from trading_app.strategy_fitness import compute_fitness

    doc = _load_allocation_doc()
    if doc is None:
        return _empty("No lane_allocation.json found - run rebalance or check docs/runtime/.")
    idx = _allocation_index(doc)
    if not idx:
        return _empty("Allocation index is empty (no lanes or paused entries).")

    rows: list[tuple[str, str, str, str, str]] = []
    for sid, entry in sorted(idx.items()):
        if instrument_filter and not sid.startswith(instrument_filter + "_"):
            continue
        if profile_filter:
            profiles = entry.get("profile_ids") or entry.get("profiles") or []
            if profile_filter not in profiles:
                continue
        state = entry.get("_allocation_state", "?")
        try:
            score = compute_fitness(sid)
            fitness = score.fitness_status
        except Exception as exc:
            fitness = f"ERR({type(exc).__name__})"
        rows.append(
            (
                sid,
                state.upper(),
                fitness,
                str(entry.get("session_id") or entry.get("orb_label") or "-"),
                str(entry.get("entry_model") or "-"),
            )
        )

    if not rows:
        return _empty("No deployed lanes match the current filters.")
    header = ["strategy_id", "state", "fitness", "session", "entry_model"]
    return _table(header, rows)


def panel_promotable(instrument_filter: str | None) -> str:
    from scripts.tools.strategy_lab_mcp_server import (
        _allocation_index,
        _list_validated_rows,
        _load_allocation_doc,
    )
    from trading_app.strategy_fitness import compute_fitness

    validated = _list_validated_rows(instrument_filter)
    if not validated:
        return _empty("No validated rows for the active instrument filter.")
    doc = _load_allocation_doc()
    allocated_ids = set(_allocation_index(doc).keys()) if doc else set()

    rows: list[tuple[str, ...]] = []
    for row in validated:
        sid = row.get("strategy_id")
        if not sid or sid in allocated_ids:
            continue
        try:
            score = compute_fitness(sid)
            status = score.fitness_status
        except Exception as exc:
            status = f"ERR({type(exc).__name__})"
        if status != "FIT":
            continue
        rows.append(
            (
                str(sid),
                str(row.get("instrument", "-")),
                str(row.get("orb_label", "-")),
                str(row.get("entry_model", "-")),
                f"{row.get('expectancy_r', 0.0):.3f}",
                str(row.get("sample_size", "-")),
            )
        )
    if not rows:
        return _empty("No FIT-but-unallocated candidates right now.")
    header = ["strategy_id", "instrument", "session", "entry_model", "ExpR (IS)", "N"]
    return _table(header, rows[:50])


def panel_promote_queue(instrument_filter: str | None) -> tuple[str, list[Any]]:
    from scripts.research.fast_lane_promote_queue import scan

    entries = scan(oos_window_days=None)
    if instrument_filter:
        entries = [e for e in entries if e.strategy_id.startswith(instrument_filter + "_")]
    if not entries:
        return _empty("Promote queue is empty (no fast-lane result MDs match)."), []

    grouped: dict[str, list[Any]] = {}
    for e in entries:
        grouped.setdefault(e.status, []).append(e)

    parts: list[str] = []
    for status in sorted(grouped.keys()):
        bucket = grouped[status]
        rows = [
            (
                e.strategy_id,
                e.direction,
                f"{e.pooled_t:.2f}",
                f"{e.pooled_expr:.3f}",
                str(e.pooled_n),
                e.error_reason or "-",
            )
            for e in bucket
        ]
        parts.append(f"<h4>{escape(status)} ({len(bucket)})</h4>")
        parts.append(
            _table(
                ["strategy_id", "direction", "pooled_t", "pooled_ExpR", "N", "reason"],
                rows,
            )
        )
    return "\n".join(parts), entries


def panel_oos_rejections(promote_entries: list[Any]) -> str:
    rejected = [e for e in promote_entries if e.status == "REJECTED_OOS_UNPOWERED"]
    if not rejected:
        return _empty(
            "No PROMOTE results rejected by the OOS-power gate currently. "
            "(Pre-flight gate at fast_lane_promote_queue.scan; see commit 8ff55b98.)"
        )
    rows = [(e.strategy_id, e.direction, f"{e.pooled_t:.2f}", str(e.pooled_n), e.error_reason or "-") for e in rejected]
    return _table(["strategy_id", "direction", "pooled_t", "N", "reason"], rows)


def panel_cherry_pick_top5() -> str:
    csvs = sorted(RUNTIME_DIR.glob("cherry_pick_ranking_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        return _empty("No cherry_pick_ranking_*.csv available - run /cherry-pick to generate.")
    latest = csvs[0]
    rows: list[tuple[str, ...]] = []
    try:
        with latest.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for i, r in enumerate(reader):
                if i >= 5:
                    break
                rows.append(
                    (
                        r.get("rank", "-"),
                        r.get("strategy_id", "-"),
                        r.get("direction", "-"),
                        r.get("score", "-"),
                        r.get("oos_n", "-"),
                        r.get("skip_recommended", "-"),
                    )
                )
    except OSError as exc:
        return _empty(f"Could not read {latest.name}: {exc}")
    if not rows:
        return _empty(f"{latest.name} has no rows.")
    header = ["rank", "strategy_id", "direction", "score", "OOS_N", "skip?"]
    body = _table(header, rows)
    return f"<p class='meta'>source: {escape(str(latest.relative_to(PROJECT_ROOT)))}</p>{body}"


def panel_drafts() -> str:
    if not DRAFTS_DIR.exists():
        return _empty("No drafts directory yet.")
    drafts = sorted(DRAFTS_DIR.glob("*.draft.yaml"))
    if not drafts:
        return _empty("No .draft.yaml files awaiting human review.")
    rows: list[tuple[str, ...]] = []
    for d in drafts:
        rejection = d.with_name(d.stem.replace(".draft", "") + ".rejected.txt")
        sidecar = (
            "REJECTED"
            if rejection.exists()
            else ("READY" if d.with_name(d.stem.replace(".draft", "") + ".grounded.yaml").exists() else "PENDING")
        )
        purpose = _peek_yaml_field(d, ["metadata", "purpose"]) or "-"
        theory = _peek_yaml_field(d, ["metadata", "theory_grant"]) or "-"
        trials = _peek_yaml_field(d, ["metadata", "total_expected_trials"]) or "-"
        rows.append((d.name, sidecar, str(theory), str(trials), _truncate(str(purpose), 80)))
    header = ["draft", "sidecar", "theory_grant", "trials", "purpose"]
    return _table(header, rows)


def panel_journal_pending() -> str:
    import yaml

    if not JOURNAL_PATH.exists():
        return _empty("No cherry_pick_journal.yaml.")
    try:
        doc = yaml.safe_load(JOURNAL_PATH.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return _empty(f"Journal unreadable: {exc}")
    entries = doc.get("entries") or []
    pending = [e for e in entries if e.get("heavyweight_verdict") in (None, "DEFERRED_NOT_RUN")]
    if not pending:
        return _empty("All journal entries have resolved verdicts.")
    rows = [
        (
            str(e.get("iter", "-")),
            str(e.get("strategy_id", "-")),
            str(e.get("oos_power_tier", "-")),
            str(e.get("heavyweight_verdict") or "NULL"),
            str(e.get("lesson_label") or "-"),
        )
        for e in pending
    ]
    return _table(["iter", "strategy_id", "OOS_power", "verdict", "lesson"], rows)


def panel_next_24h(instrument_filter: str | None) -> str:
    from datetime import timedelta

    from pipeline.dst import SESSION_CATALOG, orb_utc_window
    from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
    from scripts.tools.strategy_lab_mcp_server import _allocation_index, _load_allocation_doc
    from zoneinfo import ZoneInfo

    today = date.today()
    days = [today, today + timedelta(days=1)]
    doc = _load_allocation_doc()
    idx = _allocation_index(doc) if doc else {}

    deployed_sessions: dict[tuple[str, str], str] = {}
    for sid, entry in idx.items():
        inst = entry.get("instrument") or (sid.split("_", 1)[0] if "_" in sid else "")
        sess = entry.get("session_id") or entry.get("orb_label") or ""
        if inst and sess:
            deployed_sessions[(inst, sess)] = entry.get("_allocation_state", "active")

    brisbane = ZoneInfo("Australia/Brisbane")
    instruments = sorted(ACTIVE_ORB_INSTRUMENTS)
    if instrument_filter:
        instruments = [i for i in instruments if i == instrument_filter]

    rows: list[tuple[str, ...]] = []
    now_utc = datetime.now(timezone.utc)
    horizon = now_utc + timedelta(hours=24)
    for trading_day in days:
        for label in SESSION_CATALOG.keys():
            try:
                start_utc, _end_utc = orb_utc_window(trading_day, label, 5)
            except ValueError:
                continue
            if not (now_utc <= start_utc <= horizon):
                continue
            start_bne = start_utc.astimezone(brisbane)
            for inst in instruments:
                state = deployed_sessions.get((inst, label), "-")
                rows.append(
                    (
                        inst,
                        label,
                        start_bne.strftime("%a %H:%M"),
                        start_utc.strftime("%Y-%m-%d %H:%M"),
                        state.upper(),
                    )
                )
    if not rows:
        return _empty("No sessions fire in the next 24 hours under the current filter.")
    rows.sort(key=lambda r: r[3])
    return _table(["instrument", "session", "Brisbane", "UTC", "deployed?"], rows)


def panel_holdout(freshness: dict[str, Any]) -> str:
    from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

    today = date.today()
    days_since_holdout = (today - HOLDOUT_SACRED_FROM).days
    oos_days = []
    for inst, info in freshness.items():
        td = info.get("max_trading_day")
        if td:
            try:
                td_d = date.fromisoformat(td)
                oos_days.append((inst, (td_d - HOLDOUT_SACRED_FROM).days))
            except ValueError:
                continue
    rows = [(inst, f"{n} days") for inst, n in sorted(oos_days)]
    if not rows:
        rows = [("(no instrument data)", "-")]
    body = _table(["instrument", "OOS days accumulated"], rows)
    return (
        f"<p class='meta'>HOLDOUT_SACRED_FROM = {HOLDOUT_SACRED_FROM.isoformat()} "
        f"({days_since_holdout} days since holdout start)</p>{body}"
    )


def panel_freshness_drift(freshness: dict[str, Any], drift: dict[str, str]) -> str:
    rows = [
        (
            inst,
            info.get("max_trading_day") or "-",
            str(info.get("stale_days") if info.get("stale_days") is not None else "-"),
            "STALE" if info.get("is_stale") else "OK",
        )
        for inst, info in sorted(freshness.items())
    ]
    table = _table(["instrument", "max_trading_day", "days_stale", "status"], rows)
    drift_class = "ok" if drift.get("status") == "PASS" else "stale"
    drift_html = (
        f"<p class='meta'>check_drift: "
        f"<span class='badge {drift_class}'>{escape(drift.get('status', '?'))}</span> "
        f"<span class='small'>{escape(drift.get('detail', ''))}</span></p>"
    )
    return drift_html + table


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _peek_yaml_field(path: Path, keys: list[str]) -> Any:
    import yaml

    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    cur: Any = doc
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "..."


def _empty(msg: str) -> str:
    return f"<p class='empty'>{escape(msg)}</p>"


def _table(header: list[str], rows: list[tuple[str, ...]]) -> str:
    if not rows:
        return _empty("(no rows)")
    thead = "".join(f"<th>{escape(h)}</th>" for h in header)
    body_rows = "".join("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{body_rows}</tbody></table>"


def _render_panel(num: int, title: str, fn) -> str:
    try:
        body = fn()
    except Exception:  # per-panel error isolation (institutional-rigor.md § 6)
        tb = traceback.format_exc().splitlines()[-3:]
        body = f"<div class='panel-error'>Panel {num} unavailable: <pre>{escape(' | '.join(tb))}</pre></div>"
    return (
        f"<details open id='p{num}'><summary><a href='#p{num}'>"
        f"{num}. {escape(title)}</a></summary><div class='panel-body'>{body}</div></details>"
    )


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "?"
    except (subprocess.TimeoutExpired, OSError):
        return "?"


# ---------------------------------------------------------------------------
# CSS + HTML scaffold
# ---------------------------------------------------------------------------


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 1em; color: #111; }
h1 { margin: 0 0 0.2em 0; }
.subtitle { color: #555; margin-bottom: 1em; }
.banner-stale { background: #c0392b; color: white; padding: 0.6em 1em; margin: 0 0 1em 0; border-radius: 4px; font-weight: bold; }
.banner-ok { background: #27ae60; color: white; padding: 0.4em 1em; margin: 0 0 1em 0; border-radius: 4px; }
details { background: #f7f7f7; border: 1px solid #ddd; border-radius: 4px; padding: 0.4em 0.8em; margin: 0.4em 0; }
details summary { cursor: pointer; font-weight: 600; font-size: 1.05em; }
details summary a { color: #111; text-decoration: none; }
.panel-body { padding: 0.4em 0 0 0; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.9em; margin-top: 0.4em; }
th, td { padding: 0.3em 0.6em; border-bottom: 1px solid #eee; text-align: left; }
th { background: #ececec; font-weight: 600; }
tr:hover td { background: #fafafa; }
.empty { color: #777; font-style: italic; margin: 0.4em 0; }
.panel-error { color: #c0392b; background: #fdecea; padding: 0.6em; border-radius: 4px; }
.panel-error pre { white-space: pre-wrap; font-size: 0.85em; margin: 0.4em 0 0 0; }
.meta { color: #555; font-size: 0.85em; margin: 0.2em 0; }
.small { font-family: monospace; font-size: 0.8em; }
.badge { display: inline-block; padding: 0.1em 0.5em; border-radius: 3px; font-weight: 600; font-size: 0.85em; }
.badge.ok { background: #27ae60; color: white; }
.badge.stale { background: #c0392b; color: white; }
.toc { background: #eef; padding: 0.6em; border-radius: 4px; margin-bottom: 1em; }
.toc a { display: inline-block; margin-right: 1em; }
footer { margin-top: 2em; color: #777; font-size: 0.8em; }
@media (min-width: 1000px) {
  .panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 0.6em; }
}
"""


def render_portal(
    profile_filter: str | None = None,
    instrument_filter: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Render the portal. Returns (html_string, json_payload)."""
    freshness = _query_data_freshness()
    any_stale = any(info.get("is_stale") for info in freshness.values())
    drift = _drift_status()

    # Panel 3 also feeds panel 4 (rejection sub-view) so render it once.
    promote_html: str = ""
    promote_entries: list[Any] = []

    def _panel3():
        nonlocal promote_html, promote_entries
        promote_html, promote_entries = panel_promote_queue(instrument_filter)
        return promote_html

    panels_html = [
        _render_panel(1, "Deployed lanes + fitness", lambda: panel_deployed_lanes(profile_filter, instrument_filter)),
        _render_panel(2, "Promotable candidates (FIT, unallocated)", lambda: panel_promotable(instrument_filter)),
        _render_panel(3, "PROMOTE queue (live state)", _panel3),
        _render_panel(4, "OOS-power rejections", lambda: panel_oos_rejections(promote_entries)),
        _render_panel(5, "Cherry-pick top-5 (latest ranking)", panel_cherry_pick_top5),
        _render_panel(6, "Drafts awaiting human review", panel_drafts),
        _render_panel(7, "Cherry-pick journal: pending verdicts", panel_journal_pending),
        _render_panel(8, "Next 24h sessions", lambda: panel_next_24h(instrument_filter)),
        _render_panel(9, "Holdout window status", lambda: panel_holdout(freshness)),
        _render_panel(10, "Data freshness + drift status", lambda: panel_freshness_drift(freshness, drift)),
    ]

    banner = (
        "<div class='banner-stale'>STALE DATA: at least one instrument is "
        f"&gt; {STALENESS_TRADING_DAYS} trading days behind today.</div>"
        if any_stale
        else "<div class='banner-ok'>Data freshness OK.</div>"
    )
    toc = "<div class='toc'>" + " ".join(f"<a href='#p{i}'>{i}</a>" for i in range(1, 11)) + "</div>"

    rendered_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sha = _git_sha()
    filter_note = []
    if profile_filter:
        filter_note.append(f"profile={profile_filter}")
    if instrument_filter:
        filter_note.append(f"instrument={instrument_filter}")
    filter_html = f"<p class='subtitle'>Filters: {escape(', '.join(filter_note))}</p>" if filter_note else ""

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>OKAY GO portal</title>
<style>{CSS}</style></head><body>
<h1>OKAY GO portal</h1>
<p class='subtitle'>Read-only aggregation of research -> validation -> ready-to-trade state.</p>
{filter_html}
{banner}
{toc}
<div class='panels'>
{"".join(panels_html)}
</div>
<footer>Rendered {rendered_ts} UTC | git {sha} | source: scripts/tools/go_portal.py</footer>
</body></html>
"""

    payload = {
        "rendered_at": rendered_ts,
        "git_sha": sha,
        "freshness": freshness,
        "drift": drift,
        "any_stale": any_stale,
        "filters": {"profile": profile_filter, "instrument": instrument_filter},
        "panels": {str(i): True for i in range(1, 11)},
    }
    return html, payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OKAY GO portal - read-only HTML aggregator.")
    ap.add_argument("--no-open", action="store_true", help="Write HTML but do not open browser.")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable summary to stdout instead of HTML.")
    ap.add_argument("--instrument", default=None, help="Filter to one instrument (e.g. MNQ, MES, MGC).")
    ap.add_argument("--profile", default=None, help="Filter to one prop profile id.")
    ap.add_argument("--out", default=None, help="Override output HTML path.")
    args = ap.parse_args(argv)

    html, payload = render_portal(profile_filter=args.profile, instrument_filter=args.instrument)

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0

    out_path = Path(args.out) if args.out else RUNTIME_DIR / f"go_portal_{date.today().isoformat()}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(str(out_path))
    if not args.no_open:
        try:
            webbrowser.open(out_path.as_uri())
        except Exception:
            # Fall through - we already printed the path so operator can open it manually.
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
