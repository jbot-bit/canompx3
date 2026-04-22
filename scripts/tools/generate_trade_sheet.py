#!/usr/bin/env python3
"""
Daily Trade Sheet Generator V3 — Unified Regime-Aware Timeline.

Three data sources merged into one timeline:
  1. DEPLOYED — lanes from active prop_profiles
  2. OPPORTUNITIES — validated CORE strategies (N >= 100) passing all gates
  3. MANUAL — all positive strategies including REGIME tier (N >= 30)

Regime awareness:
  - ATR percentile banner per session (HIGH/NORMAL/LOW)
  - Filter status per row: ACTIVE (passes today), VERIFY (check at session), INACTIVE (dimmed)
  - Frequency column (~X/yr or ~X/mo)

Usage:
    python scripts/tools/generate_trade_sheet.py
    python scripts/tools/generate_trade_sheet.py --deployed-only
    python scripts/tools/generate_trade_sheet.py --date 2026-03-04
    python scripts/tools/generate_trade_sheet.py --no-open
"""

import argparse
import json
import re as _re_module
import sys
import webbrowser
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.cost_model import get_cost_spec
from pipeline.dst import SESSION_CATALOG
from pipeline.paths import GOLD_DB_PATH
from trading_app.eligibility.builder import (
    VALIDATION_FRESHNESS_DAYS,
    build_eligibility_report,
    parse_strategy_id,
)
from trading_app.prop_profiles import ACCOUNT_PROFILES, effective_daily_lanes
from trading_app.strategy_fitness import compute_fitness
from trading_app.validated_shelf import deployable_validated_relation

# Dollar gate: expected $/trade must be >= this multiplier * RT friction.
# Was LIVE_MIN_EXPECTANCY_DOLLARS_MULT in live_config.py (1.3).
_DOLLAR_GATE_MULT = 1.3


@dataclass(frozen=True)
class FitnessCheckResult:
    """Resolved fitness status for a strategy, including lookup failures."""

    status: str
    error: str | None = None


# ── Filter → plain English ────────────────────────────────────────────


def _filter_description(filter_type: str) -> str:
    """Convert filter_type to plain English for the trade sheet."""
    # Exact matches first
    exact = {
        "NO_FILTER": "Any ORB size",
        "DIR_LONG": "LONG ONLY",
        "DIR_SHORT": "SHORT ONLY",
        "ATR70_VOL": "ATR > 70th pct",
        "ATR_P30": "ATR > 30th pct",
        "ATR_P50": "ATR > 50th pct",
    }
    if filter_type in exact:
        return exact[filter_type]

    ft = filter_type

    # Composite filters: parse components
    parts = []

    # Volume/RV filters: VOL_RV{threshold}_N{window}
    rv_match = _re_module.match(r"VOL_RV(\d+)_N(\d+)", ft)
    if rv_match:
        thresh = int(rv_match.group(1)) / 10
        return f"Vol >= {thresh:.1f}x median"

    # Cost filters: COST_LT{cents}
    cost_match = _re_module.match(r"COST_LT(\d+)", ft)
    if cost_match:
        cents = int(cost_match.group(1))
        return f"Cost < {cents}% of ORB"

    # Cross-asset filters: X_MES_ATR{pct}
    xmatch = _re_module.match(r"X_(\w+)_ATR(\d+)", ft)
    if xmatch:
        ref = xmatch.group(1)
        pct = xmatch.group(2)
        return f"{ref} ATR > {pct}th pct"

    # Overnight range filters
    ovn_match = _re_module.match(r"OVNRNG_(\d+)", ft)
    if ovn_match:
        pct = ovn_match.group(1)
        return f"Overnight range > {pct}th pct"

    # ORB volume filters
    orbvol_match = _re_module.match(r"ORB_VOL_(\d+)K?", ft)
    if orbvol_match:
        vol = orbvol_match.group(1)
        return f"ORB volume > {vol}K"

    # ORB size component
    for g in ["G2", "G3", "G4", "G5", "G6", "G8"]:
        if f"ORB_{g}" in ft or ft.startswith(g):
            pts = g[1:]
            parts.append(f"ORB >= {pts} pts")
            break

    # Break quality composites
    if "CONT" in ft:
        parts.append("continuation only")
    if "FAST5" in ft:
        parts.append("break within 5 min")
    if "FAST10" in ft:
        parts.append("break within 10 min")

    # DOW composites
    if "NOMON" in ft:
        parts.append("skip Monday")
    if "NOFRI" in ft:
        parts.append("skip Friday")
    if "NOTUE" in ft:
        parts.append("skip Tuesday")

    if parts:
        return " + ".join(parts)

    return filter_type  # fallback: show raw


def _direction_rule(filter_type: str) -> str:
    """Determine direction constraint from filter_type."""
    if "DIR_LONG" in filter_type:
        return "LONG ONLY"
    if "DIR_SHORT" in filter_type:
        return "SHORT ONLY"
    if "CONT" in filter_type:
        return "CONT"
    return "ANY"


def _passes_dollar_gate(row: dict, instrument: str) -> tuple[bool, float | None]:
    """Check if expected $/trade >= _DOLLAR_GATE_MULT * RT cost.

    Returns (passes, exp_dollars). Fail-closed: returns (False, None) when
    median_risk_points is missing or cost spec is unavailable.
    """
    median_risk_pts = row.get("median_risk_points")
    exp_r = row.get("expectancy_r")
    if median_risk_pts is None or exp_r is None:
        return False, None
    try:
        spec = get_cost_spec(instrument)
    except Exception as exc:
        print(f"  WARNING: cost spec lookup failed for {instrument}: {exc}", flush=True)
        return False, None
    exp_d = exp_r * median_risk_pts * spec.point_value
    min_dollars = _DOLLAR_GATE_MULT * spec.total_friction
    return exp_d >= min_dollars, exp_d


def _check_fitness(
    strategy_id: str,
    db_path: Path,
    cache: dict[str, FitnessCheckResult] | None = None,
) -> FitnessCheckResult:
    """Quick fitness check. Returns cached status and surfaces lookup failures."""
    if cache is not None and strategy_id in cache:
        return cache[strategy_id]

    try:
        f = compute_fitness(strategy_id, db_path=db_path)
        result = FitnessCheckResult(status=f.fitness_status)
    except Exception as exc:
        result = FitnessCheckResult(status="UNKNOWN", error=f"{type(exc).__name__}: {exc}")

    if cache is not None:
        cache[strategy_id] = result
    return result


# ── Regime context & filter classification ───────────────────────────


def _get_regime_context(db_path: Path) -> dict[str, dict]:
    """Query latest daily_features per instrument for regime indicators.

    Returns dict keyed by instrument symbol with atr_pct, overnight_range_pct,
    as_of date, and cross_atrs for cross-asset filter checking.
    """
    instruments = list(ACTIVE_ORB_INSTRUMENTS)
    result: dict[str, dict] = {}

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        for instr in instruments:
            row = con.execute(
                """
                SELECT trading_day, atr_20_pct, overnight_range_pct
                FROM daily_features
                WHERE symbol = ?
                  AND atr_20_pct IS NOT NULL
                ORDER BY trading_day DESC
                LIMIT 1
                """,
                [instr],
            ).fetchone()

            if row is None:
                result[instr] = {
                    "atr_pct": None,
                    "overnight_range_pct": None,
                    "as_of": None,
                    "cross_atrs": {},
                }
            else:
                result[instr] = {
                    "atr_pct": row[1],
                    "overnight_range_pct": row[2],
                    "as_of": row[0].date() if hasattr(row[0], "date") else row[0],
                    "cross_atrs": {},
                }
    finally:
        con.close()

    # Build cross_atrs: for each instrument, map the OTHER instruments' ATR pct
    for instr in instruments:
        result[instr]["cross_atrs"] = {
            other: result[other]["atr_pct"]
            for other in instruments
            if other != instr and result[other]["atr_pct"] is not None
        }

    return result


def _prefetch_feature_rows(
    trades: list[dict],
    db_path: Path,
) -> dict[tuple[str, int], dict | None]:
    """Pre-fetch latest-available daily_features row per unique (instrument, aperture).

    Strategy: pull the LATEST row per pair (regardless of trading_day) so the
    pre-session brief has actual data to evaluate against, even when today's
    daily_features row has not yet been built. The canonical eligibility
    builder tags freshness (FRESH / PRIOR_DAY / STALE / NO_DATA) from the row's
    trading_day vs the requested trading_day, so the UI can surface stale data
    without crashing.

    Matches the pattern used by _get_regime_context: latest row per instrument
    for regime indicators. Here we extend it to per-aperture because
    daily_features has 3 rows per (trading_day, symbol) — one per orb_minutes
    value. The canonical builder needs the row matching the strategy's aperture.

    Returns a dict keyed by (instrument, aperture) → row dict or None.
    Fail-closed: if the DB connection itself raises, propagate.
    """
    pairs: set[tuple[str, int]] = set()
    for t in trades:
        instrument = t["instrument"]
        aperture = int(t.get("aperture", 5) or 5)
        pairs.add((instrument, aperture))

    result: dict[tuple[str, int], dict | None] = {}
    if not pairs:
        return result

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        for instrument, aperture in pairs:
            row = con.execute(
                """
                SELECT *
                FROM daily_features
                WHERE symbol = ?
                  AND orb_minutes = ?
                ORDER BY trading_day DESC
                LIMIT 1
                """,
                [instrument, aperture],
            ).fetchone()
            if row is None:
                result[(instrument, aperture)] = None
            else:
                cols = [desc[0] for desc in con.description]
                # strict=True matches the canonical _fetch_feature_row in
                # trading_app.eligibility.builder (gate 1 review parity fix)
                result[(instrument, aperture)] = dict(zip(cols, row, strict=True))
    finally:
        con.close()

    return result


def _enrich_trades_with_eligibility(
    trades: list[dict],
    trading_day: date,
    feature_rows: dict[tuple[str, int], dict | None],
) -> None:
    """Attach canonical eligibility fields to each trade dict.

    Delegates filter semantics to build_eligibility_report from
    trading_app.eligibility.builder. NO re-encoded filter logic.

    Attaches six keys on success:
      elig_overall     — OverallStatus string value (ELIGIBLE/INELIGIBLE/
                         DATA_MISSING/NEEDS_LIVE_DATA)
      elig_blocking    — tuple of condition names currently blocking the lane
      elig_pending     — tuple of condition names awaiting intra-session resolution
      elig_stale       — bool: any condition with STALE_VALIDATION status
      elig_size_mult   — float: product of all passing conditions' size multipliers
      elig_freshness   — FreshnessStatus string value (FRESH/PRIOR_DAY/STALE/NO_DATA)

    On exception (unknown filter_type, broken describe() contract, etc.):
    attaches the same six keys with overall="UNKNOWN" and defaults, plus a
    seventh key elig_error carrying the exception text. Prints a WARNING to
    stdout (matches _check_fitness pattern) so broken strategies are visible
    without aborting the whole sheet.
    """
    for t in trades:
        sid = t["strategy_id"]
        instrument = t["instrument"]
        aperture = int(t.get("aperture", 5) or 5)
        feature_row = feature_rows.get((instrument, aperture))

        try:
            report = build_eligibility_report(
                strategy_id=sid,
                trading_day=trading_day,
                feature_row=feature_row,
            )
            t["elig_overall"] = report.overall_status.value
            t["elig_blocking"] = tuple(c.name for c in report.blocking_conditions)
            t["elig_pending"] = tuple(c.name for c in report.pending_conditions)
            t["elig_stale"] = any(c.status.value == "STALE_VALIDATION" for c in report.conditions)
            t["elig_size_mult"] = float(report.effective_size_multiplier)
            t["elig_freshness"] = report.freshness_status.value
        except Exception as exc:  # noqa: BLE001 — adapter boundary, surface as visible UNKNOWN
            err_msg = f"{type(exc).__name__}: {exc}"
            print(f"  WARNING: eligibility build failed for {sid}: {err_msg}", flush=True)
            t["elig_overall"] = "UNKNOWN"
            t["elig_blocking"] = ()
            t["elig_pending"] = ()
            t["elig_stale"] = False
            t["elig_size_mult"] = 1.0
            t["elig_freshness"] = ""
            t["elig_error"] = err_msg


def _load_trailing_stats() -> dict[str, dict]:
    """Load trailing stats from lane_allocation.json if it exists.

    Returns dict keyed by strategy_id with trailing_expr, trailing_wr, trailing_n.
    Fail-open: returns empty dict if file missing or unreadable.
    """
    alloc_path = PROJECT_ROOT / "docs" / "runtime" / "lane_allocation.json"
    if not alloc_path.exists():
        return {}
    try:
        data = json.loads(alloc_path.read_text())
        result = {}
        for lane in data.get("lanes", []):
            sid = lane.get("strategy_id")
            if sid and lane.get("trailing_expr") is not None:
                result[sid] = {
                    "trailing_expr": lane["trailing_expr"],
                    "trailing_wr": lane.get("trailing_wr"),
                    "trailing_n": lane.get("trailing_n"),
                    "trailing_window": data.get("trailing_window_months", 12),
                    "session_regime": lane.get("session_regime"),
                    "alloc_status": lane.get("status"),
                }
        return result
    except (json.JSONDecodeError, OSError, KeyError):
        return {}


def _enrich_trades_with_regime(
    all_trades: list[dict],
    regime_ctx: dict[str, dict],
    db_path: Path,
) -> None:
    """Add filter_status, trades_per_year, trailing stats, and regime_atr_pct."""
    # Batch-fetch years_tested for all strategy_ids
    sids = [t["strategy_id"] for t in all_trades]
    years_map: dict[str, float] = {}
    tpy_map: dict[str, float] = {}

    if sids:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            rows = con.execute(
                """
                SELECT strategy_id, years_tested, trades_per_year
                FROM validated_setups
                WHERE strategy_id IN (SELECT UNNEST(?::VARCHAR[]))
                """,
                [sids],
            ).fetchall()
            for sid, yt, tpy in rows:
                if yt is not None:
                    years_map[sid] = float(yt)
                if tpy is not None:
                    tpy_map[sid] = float(tpy)
        finally:
            con.close()

    # Load trailing stats from lane allocator (12mo window)
    trailing = _load_trailing_stats()

    for t in all_trades:
        # Trades per year — prefer DB trades_per_year, fall back to sample/years
        sid = t["strategy_id"]
        if sid in tpy_map:
            t["trades_per_year"] = tpy_map[sid]
        elif sid in years_map and years_map[sid] > 0:
            t["trades_per_year"] = t.get("sample_size", 0) / years_map[sid]
        else:
            t["trades_per_year"] = None

        # Regime ATR percentile for this instrument
        ctx = regime_ctx.get(t["instrument"])
        t["regime_atr_pct"] = ctx.get("atr_pct") if ctx else None

        # Trailing stats overlay — use 12mo trailing when available
        ts = trailing.get(sid)
        if ts:
            t["validated_wr"] = t["win_rate"]
            t["validated_expr"] = t["exp_r"]
            t["win_rate"] = ts["trailing_wr"]
            t["exp_r"] = ts["trailing_expr"]
            t["stats_source"] = f"{ts['trailing_window']}mo"
            t["trailing_n"] = ts["trailing_n"]
        else:
            t["stats_source"] = "all"


# ── Session time resolution ───────────────────────────────────────────


def _resolve_session_times(trading_day: date) -> dict[str, tuple[int, int]]:
    """Resolve all session start times in Brisbane for a given date."""
    times = {}
    for label, entry in SESSION_CATALOG.items():
        if entry["type"] == "dynamic":
            resolver = entry["resolver"]
            h, m = resolver(trading_day)
            times[label] = (h, m)
    return times


def _format_time(h: int, m: int) -> str:
    """Format (hour, minute) as HH:MM with AM/PM."""
    period = "AM" if h < 12 else "PM"
    display_h = h % 12
    if display_h == 0:
        display_h = 12
    return f"{display_h}:{m:02d} {period}"


def _sort_key(h: int, m: int) -> int:
    """Sort sessions by Brisbane time, starting from 8 AM (trading day start)."""
    # Shift so 8 AM = 0, wrapping midnight sessions to after evening
    shifted = (h * 60 + m - 8 * 60) % (24 * 60)
    return shifted


# ── Data collection ───────────────────────────────────────────────────


def _fetch_exact_lane_variant(con: duckdb.DuckDBPyConnection, strategy_id: str) -> dict | None:
    """Load one exact strategy row from validated first, experimental second.

    This keeps exact profile lanes honest during the research -> runtime
    bridge: validated rows remain authoritative, but experimental exact IDs can
    still surface explicitly as non-validated shadow candidates instead of
    being silently skipped.
    """
    row = con.execute(
        """
        SELECT strategy_id, orb_label, orb_minutes, filter_type, rr_target,
               win_rate, expectancy_r, sample_size, median_risk_points, source
        FROM (
            SELECT vs.strategy_id, vs.orb_label, vs.orb_minutes,
                   vs.filter_type, vs.rr_target, vs.win_rate,
                   vs.expectancy_r, vs.sample_size,
                   es.median_risk_points,
                   'validated' AS source,
                   0 AS source_rank
            FROM validated_setups vs
            LEFT JOIN experimental_strategies es
              ON vs.strategy_id = es.strategy_id
            WHERE vs.strategy_id = ?

            UNION ALL

            SELECT es.strategy_id, es.orb_label, es.orb_minutes,
                   es.filter_type, es.rr_target, es.win_rate,
                   es.expectancy_r, es.sample_size,
                   es.median_risk_points,
                   'experimental' AS source,
                   1 AS source_rank
            FROM experimental_strategies es
            WHERE es.strategy_id = ?
        )
        ORDER BY source_rank
        LIMIT 1
        """,
        [strategy_id, strategy_id],
    ).fetchone()

    if row is None:
        return None
    cols = [d[0] for d in con.description]
    return dict(zip(cols, row, strict=False))


def collect_trades(trading_day: date, db_path: Path, profile_filter: str | None = None) -> list[dict]:
    """Collect trades from prop_profiles deployed lanes.

    For each active profile's daily_lanes, looks up the exact strategy_id
    in validated_setups or experimental_strategies. Applies dollar gate.
    Only returns cost-positive trades.

    Args:
        profile_filter: if set, only show this profile (e.g. "apex_50k_manual").

    Source of truth: trading_app.prop_profiles.ACCOUNT_PROFILES (not live_config).
    """
    trades = []
    fitness_cache: dict[str, FitnessCheckResult] = {}

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        for pid, profile in ACCOUNT_PROFILES.items():
            if profile_filter and pid != profile_filter:
                continue
            lanes = effective_daily_lanes(profile)
            if ((profile_filter is None) and not profile.active) or not lanes:
                continue

            for lane in lanes:
                sid = lane.strategy_id
                instrument = lane.instrument

                variant = _fetch_exact_lane_variant(con, sid)
                if variant is None:
                    print(f"  WARNING: {sid} not in validated_setups/experimental_strategies — skipping", flush=True)
                    continue

                # Dollar gate
                passes, exp_d = _passes_dollar_gate(variant, instrument)
                if not passes:
                    print(f"  WARNING: {sid} failed dollar gate — skipping", flush=True)
                    continue

                # Fitness check
                fitness = _check_fitness(sid, db_path, fitness_cache)

                sm = lane.planned_stop_multiplier or profile.stop_multiplier

                # Deduplicate: if same strategy_id already in trades, merge profiles
                existing = next((t for t in trades if t["strategy_id"] == sid), None)
                if existing:
                    existing["profiles"].append(pid)
                    continue

                trades.append(
                    {
                        "session": variant["orb_label"],
                        "instrument": instrument,
                        "strategy_id": sid,
                        "aperture": variant.get("orb_minutes", 5),
                        "direction": _direction_rule(variant["filter_type"]),
                        "filter_desc": _filter_description(variant["filter_type"]),
                        "filter_type": variant["filter_type"],
                        "rr": variant["rr_target"],
                        "win_rate": variant["win_rate"],
                        "exp_r": variant["expectancy_r"],
                        "exp_dollars": exp_d,
                        "sample_size": variant["sample_size"],
                        "fitness": fitness.status,
                        "fitness_error": fitness.error,
                        "profile": pid,
                        "profiles": [pid],
                        "stop_mult": sm,
                        "orb_cap": lane.max_orb_size_pts,
                        "notes": (
                            (f"EXPERIMENTAL exact lane — not validated. {lane.execution_notes}".strip())
                            if variant.get("source") == "experimental"
                            else lane.execution_notes
                        ),
                    }
                )
    finally:
        con.close()

    return trades


def collect_opportunities(
    db_path: Path,
    deployed_sids: set[str],
) -> list[dict]:
    """Collect all validated strategies that pass gates but aren't deployed.

    Best per session x instrument (highest ExpR). Applies dollar gate.
    Skips PURGED/DECAY fitness. No look-ahead — uses only validated_setups
    and experimental_strategies (pre-computed, no future data).
    """
    active_instruments = tuple(sorted(ACTIVE_ORB_INSTRUMENTS))
    opportunities = []
    fitness_cache: dict[str, FitnessCheckResult] = {}

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        shelf_relation = deployable_validated_relation(con, alias="vs")
        # Best strategy per session x instrument, respecting family_rr_locks.
        # Join locks to pick the locked RR target per family (no RR snooping).
        rows = con.execute(
            f"""
            WITH locked AS (
                SELECT vs.strategy_id, vs.instrument, vs.orb_label, vs.orb_minutes,
                       vs.filter_type, vs.rr_target, vs.stop_multiplier,
                       vs.win_rate, vs.expectancy_r, vs.sample_size,
                       es.median_risk_points,
                       ROW_NUMBER() OVER (
                           PARTITION BY vs.instrument, vs.orb_label
                           ORDER BY vs.expectancy_r DESC
                       ) as rn
                FROM {shelf_relation}
                INNER JOIN family_rr_locks frl
                  ON vs.instrument = frl.instrument
                  AND vs.orb_label = frl.orb_label
                  AND vs.filter_type = frl.filter_type
                  AND vs.entry_model = frl.entry_model
                  AND vs.orb_minutes = frl.orb_minutes
                  AND vs.confirm_bars = frl.confirm_bars
                  AND vs.rr_target = frl.locked_rr
                LEFT JOIN experimental_strategies es
                  ON vs.strategy_id = es.strategy_id
                WHERE vs.expectancy_r > 0
                  AND vs.sample_size >= 100
                  AND vs.instrument IN (SELECT UNNEST(?::VARCHAR[]))
            )
            SELECT * FROM locked WHERE rn = 1
            ORDER BY orb_label, instrument
        """,
            [list(active_instruments)],
        ).fetchall()

        cols = [d[0] for d in con.description]

        for row in rows:
            variant = dict(zip(cols, row, strict=False))
            sid = variant["strategy_id"]
            instrument = variant["instrument"]

            # Skip already-deployed strategies
            if sid in deployed_sids:
                continue

            # Dollar gate
            passes, exp_d = _passes_dollar_gate(variant, instrument)
            if not passes:
                continue

            # Fitness check — skip DECAY only (PURGED is member-count heuristic, not fitness)
            fitness = _check_fitness(sid, db_path, fitness_cache)
            if fitness.status == "DECAY":
                continue

            opportunities.append(
                {
                    "session": variant["orb_label"],
                    "instrument": instrument,
                    "strategy_id": sid,
                    "aperture": variant.get("orb_minutes", 5),
                    "direction": _direction_rule(variant["filter_type"]),
                    "filter_desc": _filter_description(variant["filter_type"]),
                    "filter_type": variant["filter_type"],
                    "rr": variant["rr_target"],
                    "win_rate": variant["win_rate"],
                    "exp_r": variant["expectancy_r"],
                    "exp_dollars": exp_d,
                    "sample_size": variant["sample_size"],
                    "fitness": fitness.status,
                    "fitness_error": fitness.error,
                    "profile": "opportunity",
                    "stop_mult": variant.get("stop_multiplier", 1.0),
                    "orb_cap": None,
                    "notes": "",
                }
            )
    finally:
        con.close()

    return opportunities


def collect_manual_candidates(
    db_path: Path,
    exclude_sids: set[str],
) -> list[dict]:
    """Collect ALL positive validated strategies for manual trading.

    Includes REGIME tier (N >= 30) and PURGED fitness — everything the trader
    might want to manually execute. DECAY excluded (actively deteriorating).
    Strategies already in deployed or opportunities are excluded.
    Best per session x instrument x tier (highest ExpR).
    """
    active_instruments = tuple(sorted(ACTIVE_ORB_INSTRUMENTS))
    candidates = []
    fitness_cache: dict[str, FitnessCheckResult] = {}

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        shelf_relation = deployable_validated_relation(con, alias="vs")
        rows = con.execute(
            f"""
            SELECT vs.strategy_id, vs.instrument, vs.orb_label, vs.orb_minutes,
                   vs.filter_type, vs.rr_target, vs.stop_multiplier,
                   vs.win_rate, vs.expectancy_r, vs.sample_size,
                   vs.entry_model,
                   es.median_risk_points,
                   ROW_NUMBER() OVER (
                       PARTITION BY vs.instrument, vs.orb_label
                       ORDER BY vs.expectancy_r DESC
                   ) as rn
            FROM {shelf_relation}
            INNER JOIN family_rr_locks frl
              ON vs.instrument = frl.instrument
              AND vs.orb_label = frl.orb_label
              AND vs.filter_type = frl.filter_type
              AND vs.entry_model = frl.entry_model
              AND vs.orb_minutes = frl.orb_minutes
              AND vs.confirm_bars = frl.confirm_bars
              AND vs.rr_target = frl.locked_rr
            LEFT JOIN experimental_strategies es
              ON vs.strategy_id = es.strategy_id
            WHERE vs.expectancy_r > 0
              AND vs.sample_size >= 30
              AND vs.instrument IN (SELECT UNNEST(?::VARCHAR[]))
        """,
            [list(active_instruments)],
        ).fetchall()

        cols = [d[0] for d in con.description]

        for row in rows:
            variant = dict(zip(cols, row, strict=False))
            sid = variant["strategy_id"]
            instrument = variant["instrument"]

            # Skip already-shown strategies
            if sid in exclude_sids:
                continue

            # Best 3 per session x instrument
            if variant["rn"] > 3:
                continue

            # Dollar gate (soft — still show but mark)
            passes, exp_d = _passes_dollar_gate(variant, instrument)

            # Fitness check — keep PURGED (labeled), skip DECAY
            fitness = _check_fitness(sid, db_path, fitness_cache)
            if fitness.status == "DECAY":
                continue

            tier = "CORE" if variant["sample_size"] >= 100 else "REGIME"

            candidates.append(
                {
                    "session": variant["orb_label"],
                    "instrument": instrument,
                    "strategy_id": sid,
                    "aperture": variant.get("orb_minutes", 5),
                    "direction": _direction_rule(variant["filter_type"]),
                    "filter_desc": _filter_description(variant["filter_type"]),
                    "filter_type": variant["filter_type"],
                    "rr": variant["rr_target"],
                    "win_rate": variant["win_rate"],
                    "exp_r": variant["expectancy_r"],
                    "exp_dollars": exp_d if passes else None,
                    "sample_size": variant["sample_size"],
                    "fitness": fitness.status,
                    "fitness_error": fitness.error,
                    "profile": "manual",
                    "stop_mult": variant.get("stop_multiplier", 1.0),
                    "orb_cap": None,
                    "notes": f"{tier}" + (f" | ${exp_d:.0f}/trade" if passes and exp_d else ""),
                }
            )
    finally:
        con.close()

    return candidates


# ── HTML generation ───────────────────────────────────────────────────


def _direction_badge(direction: str) -> str:
    """HTML badge for direction constraint."""
    if direction == "LONG ONLY":
        return '<span class="badge badge-long">LONG ONLY</span>'
    if direction == "SHORT ONLY":
        return '<span class="badge badge-short">SHORT ONLY</span>'
    if direction == "CONT":
        return '<span class="badge badge-cont">CONT ONLY</span>'
    return ""


def _fitness_badge(fitness: str) -> str:
    """HTML badge for non-FIT fitness."""
    if fitness == "FIT":
        return ""
    cls = {
        "WATCH": "badge-watch",
        "DECAY": "badge-decay",
        "STALE": "badge-stale",
        "PURGED": "badge-purged",
        "UNKNOWN": "badge-unknown",
    }.get(fitness, "badge-unknown")
    return f' <span class="badge {cls}">{fitness}</span>'


def _status_badge_from_eligibility(trade: dict) -> dict:
    """Build HTML badge + pills + row-class + tooltip parts from canonical eligibility.

    Canonical consumer of build_eligibility_report output (the six elig_* keys
    attached by _enrich_trades_with_eligibility). Returns a dict with:
      badge_html        — main status badge HTML fragment
      pills_html        — STALE and/or HALF pills (may be empty string)
      row_class_suffix  — extra CSS class for the row (e.g. ' row-inactive')
      tooltip_parts     — list of strings to join into the row tooltip

    Mapping from OverallStatus to existing badge vocabulary:
      ELIGIBLE         → green check badge (matches old ACTIVE)
      INELIGIBLE       → dimmed INACTIVE badge + row-inactive class
      NEEDS_LIVE_DATA  → clock VERIFY badge (matches old CHECK)
      DATA_MISSING     → NEW amber DATA badge + row-inactive class
      UNKNOWN          → clock VERIFY badge with error in tooltip

    Pills (additive, appended after main badge):
      elig_stale=True        → STALE pill (validation older than 180d)
      elig_size_mult < 1.0   → HALF pill (calendar overlay reduces size)
    """
    overall = trade.get("elig_overall", "UNKNOWN")
    blocking = trade.get("elig_blocking", ())
    pending = trade.get("elig_pending", ())
    stale = trade.get("elig_stale", False)
    size_mult = float(trade.get("elig_size_mult", 1.0))
    freshness = trade.get("elig_freshness", "")
    elig_error = trade.get("elig_error")

    # ── Main status badge + row class ────────────────────────────────
    if overall == "ELIGIBLE":
        badge_html = '<span class="badge badge-filter-active">&#10003;</span>'
        row_class_suffix = ""
    elif overall == "INELIGIBLE":
        badge_html = '<span class="badge badge-filter-check">INACTIVE</span>'
        row_class_suffix = " row-inactive"
    elif overall == "NEEDS_LIVE_DATA":
        badge_html = '<span class="badge badge-filter-check">&#9201; VERIFY</span>'
        row_class_suffix = ""
    elif overall == "DATA_MISSING":
        badge_html = '<span class="badge badge-filter-missing">DATA</span>'
        row_class_suffix = " row-inactive"
    else:  # UNKNOWN or any future enum value we don't recognize
        badge_html = '<span class="badge badge-filter-check">&#9201; VERIFY</span>'
        row_class_suffix = ""

    # ── Pills (additive) ─────────────────────────────────────────────
    pills = []
    if stale:
        pills.append('<span class="pill pill-stale">STALE</span>')
    if size_mult < 1.0:
        pills.append('<span class="pill pill-half">HALF</span>')
    pills_html = "".join(pills)

    # ── Tooltip parts (merged into row tooltip by caller) ────────────
    tooltip_parts: list[str] = []
    if overall == "DATA_MISSING":
        tooltip_parts.append("feature data missing for this trading day")
    elif overall == "INELIGIBLE" and blocking:
        tooltip_parts.append("blocked by: " + "; ".join(blocking))
    elif overall == "NEEDS_LIVE_DATA" and pending:
        tooltip_parts.append("waiting on: " + "; ".join(pending))

    if overall == "UNKNOWN" and elig_error:
        tooltip_parts.append(f"eligibility error: {elig_error}")

    if freshness == "PRIOR_DAY":
        tooltip_parts.append("freshness: yesterday")
    elif freshness == "STALE":
        tooltip_parts.append("freshness: STALE - report may be inaccurate")

    return {
        "badge_html": badge_html,
        "pills_html": pills_html,
        "row_class_suffix": row_class_suffix,
        "tooltip_parts": tooltip_parts,
    }


def _next_session_label(session_times: dict) -> str | None:
    """Find the next upcoming session based on current Brisbane time."""
    now = datetime.now()
    now_minutes = now.hour * 60 + now.minute
    best_label = None
    best_delta = float("inf")
    for label, (h, m) in session_times.items():
        session_minutes = h * 60 + m
        delta = (session_minutes - now_minutes) % (24 * 60)
        if 0 < delta < best_delta:
            best_delta = delta
            best_label = label
    return best_label


def _build_session_cards(
    trades: list[dict],
    session_times: dict,
    profiles_used: dict,
    next_session: str | None = None,
    regime_ctx: dict[str, dict] | None = None,
) -> str:
    """Build unified session card HTML from a merged list of trades.

    Trades must have a 'section' key: 'deployed', 'opportunity', or 'manual'.
    Within each session card, rows are sorted: deployed first, then opportunity,
    then manual — and within each group by ExpR descending.
    """
    # Section sort priority: deployed=0, opportunity=1, manual=2
    _section_order = {"deployed": 0, "opportunity": 1, "manual": 2}

    sessions_used = sorted(
        set(t["session"] for t in trades),
        key=lambda s: _sort_key(*session_times.get(s, (0, 0))),
    )

    cards_html = ""
    for session in sessions_used:
        h, m = session_times.get(session, (0, 0))
        time_str = _format_time(h, m)
        event = SESSION_CATALOG.get(session, {}).get("event", "")
        session_trades = sorted(
            [t for t in trades if t["session"] == session],
            key=lambda t: (_section_order.get(t.get("section", "manual"), 2), -t["exp_r"]),
        )

        # Instrument badges for header
        session_instruments = sorted(set(t["instrument"] for t in session_trades))
        instr_badges_html = " ".join(f'<span class="badge badge-instr">{instr}</span>' for instr in session_instruments)

        # Count live + avail
        n_live = sum(1 for t in session_trades if t.get("section") == "deployed")
        n_avail = sum(1 for t in session_trades if t.get("section") == "opportunity")
        n_manual = sum(1 for t in session_trades if t.get("section") == "manual")
        count_parts = []
        if n_live:
            count_parts.append(f"{n_live} live")
        if n_avail:
            count_parts.append(f"{n_avail} avail")
        if n_manual:
            count_parts.append(f"{n_manual} manual")
        count_str = " + ".join(count_parts) if count_parts else "0 trades"

        # Regime banner for this session's instruments
        regime_bar_html = ""
        if regime_ctx:
            chips = []
            for instr in session_instruments:
                ctx = regime_ctx.get(instr)
                atr = ctx.get("atr_pct") if ctx else None
                if atr is not None:
                    if atr >= 70:
                        chip_cls = "regime-high"
                    elif atr >= 50:
                        chip_cls = "regime-normal"
                    else:
                        chip_cls = "regime-low"
                    chips.append(f'<span class="regime-chip {chip_cls}">{instr} ATR {atr:.0f}th pct</span>')
            if chips:
                regime_bar_html = '<div class="regime-bar">' + " ".join(chips) + "</div>"

        rows_html = ""
        for t in session_trades:
            section = t.get("section", "manual")
            exp_d_str = f"${t['exp_dollars']:+.2f}" if t["exp_dollars"] is not None else "n/a"
            dir_badge = _direction_badge(t["direction"])
            fit_badge = _fitness_badge(t["fitness"])
            fitness_title = ""
            if t.get("fitness_error"):
                fitness_title = f' title="{t["fitness_error"]}"'

            exp_r_class = "expr-high" if t["exp_r"] >= 0.20 else ""

            # Status badge
            if section == "deployed":
                status_badge = '<span class="badge badge-deployed">LIVE</span>'
                row_class = "row-deployed"
            elif section == "opportunity":
                status_badge = '<span class="badge badge-opp">AVAIL</span>'
                row_class = "row-opportunity"
            else:
                status_badge = '<span class="badge badge-manual">MANUAL</span>'
                row_class = "row-manual"

            # Filter status badge — canonical eligibility (delegates to
            # trading_app.eligibility.builder via _status_badge_from_eligibility)
            elig_view = _status_badge_from_eligibility(t)
            filter_status_badge = elig_view["badge_html"] + elig_view["pills_html"]
            row_class += elig_view["row_class_suffix"]
            elig_tooltip_parts = elig_view["tooltip_parts"]

            # Frequency column
            tpy = t.get("trades_per_year")
            if tpy is not None:
                if tpy > 50:
                    freq_str = f"~{tpy / 12:.0f}/mo"
                else:
                    freq_str = f"~{tpy:.0f}/yr"
            else:
                freq_str = "?"

            # Tier badge
            sample = t.get("sample_size", 0)
            if sample >= 100:
                tier_badge = f'<span class="badge badge-core">CORE</span> {sample}'
            elif sample >= 30:
                tier_badge = f'<span class="badge badge-regime">REGIME</span> {sample}'
            else:
                tier_badge = str(sample)

            # Stop column
            sm = t.get("stop_mult", 1.0)
            stop_str = f"{sm}x" if sm else "1.0x"

            # Tooltip for instrument cell: strategy_id for avail/manual, profile for deployed.
            # Also merges eligibility tooltip parts (blocked-by, waiting-on, freshness)
            # from the canonical eligibility report.
            tooltip_parts = []
            if section == "deployed":
                for pid in t.get("profiles", [t.get("profile", "")]):
                    pi = profiles_used.get(pid, {})
                    if pi:
                        tooltip_parts.append(f"{pi.get('firm', '?')} {pi.get('mode', '?')}")
                if t.get("orb_cap"):
                    tooltip_parts.append(f"Cap {t['orb_cap']:.0f}pts")
                if t.get("notes"):
                    tooltip_parts.append(t["notes"][:60])
            else:
                tooltip_parts.append(t.get("strategy_id", ""))
            tooltip_parts.extend(elig_tooltip_parts)
            tooltip = " | ".join(p for p in tooltip_parts if p)
            tooltip_attr = f' title="{tooltip}"' if tooltip else ""

            # Stats source label and tooltip
            src = t.get("stats_source", "all")
            src_badge = (
                f'<span class="stats-src stats-src-trailing">{src}</span>'
                if src != "all"
                else '<span class="stats-src stats-src-all">all</span>'
            )
            # Tooltip: show the OTHER stat for reference
            val_wr = t.get("validated_wr")
            val_expr = t.get("validated_expr")
            if src != "all" and val_wr is not None:
                wr_title = f' title="All-time: {val_wr:.0%} WR, {val_expr:+.3f} ExpR"'
            else:
                wr_title = ""

            rows_html += f"""
            <tr class="{row_class}">
                <td>{status_badge}</td>
                <td>{filter_status_badge}</td>
                <td class="instrument-cell"{tooltip_attr}>{t["instrument"]}</td>
                <td class="filter-cell">{t["filter_desc"]}</td>
                <td>{dir_badge if dir_badge else "ANY"}</td>
                <td>{t["rr"]:.1f}:1</td>
                <td>{stop_str}</td>
                <td{wr_title}>{t["win_rate"]:.0%} {src_badge}</td>
                <td class="{exp_r_class}"{wr_title}>{t["exp_r"]:+.3f}</td>
                <td class="dollars-cell">{exp_d_str}</td>
                <td class="freq-cell">{freq_str}</td>
                <td>{tier_badge}</td>
                <td{fitness_title}>{fit_badge if fit_badge else '<span class="fit-ok">FIT</span>'}</td>
            </tr>"""

        is_next = session == next_session
        next_cls = " next-session" if is_next else ""
        next_badge = ' <span class="badge badge-next">UP NEXT</span>' if is_next else ""
        card_class = f"session-card{next_cls}".strip()
        cards_html += f"""
        <div class="{card_class}">
            <div class="session-header">
                <div class="session-time">{time_str}</div>
                <div class="session-name">{session}{next_badge}</div>
                <div class="session-instruments">{instr_badges_html}</div>
                <div class="session-count">{count_str}</div>
                <div class="session-event">{event}</div>
            </div>
            {regime_bar_html}
            <table>
                <thead>
                    <tr>
                        <th>Status</th>
                        <th>Regime</th>
                        <th>Instr</th>
                        <th>Filter</th>
                        <th>Dir</th>
                        <th>RR</th>
                        <th>Stop</th>
                        <th>WR</th>
                        <th>ExpR</th>
                        <th>$/trade</th>
                        <th>Freq</th>
                        <th>Tier &amp; N</th>
                        <th>Fitness</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>"""

    return cards_html


# ── View B: Filter Universe Audit ─────────────────────────────────────
# Surfaces every filter in trading_app.config.ALL_FILTERS plus the ATR
# velocity overlay, cross-referenced against active validated_setups
# (routed count) and prop_profiles.ACCOUNT_PROFILES (deployed count).
# Anti-confirmation-bias: showing the full filter universe alongside the
# deployed slice exposes dead / stale / unrouted filters that would
# otherwise be invisible in View A.
#
# Design: docs/plans/2026-04-07-filter-universe-audit-design.md


def _build_filter_universe_rows(db_path: Path, trading_day: date) -> list[dict]:
    """Walk ALL_FILTERS + ATR_VELOCITY_OVERLAY and produce audit row dicts.

    One read-only DB connection + one aggregate query for routed counts.
    For each filter: reads the ClassVar metadata (VALIDATED_FOR,
    LAST_REVALIDATED, CONFIDENCE_TIER), counts active validated_setups
    strategies using it (routed), counts active profile lanes deploying
    it (deployed), derives a status badge (LIVE/ROUTED/DEAD).

    Returns a list of row dicts sorted by (deployed DESC, routed DESC,
    filter_type ASC). The sort order puts LIVE rows at the top and DEAD
    rows at the bottom so the trader's eye lands on the actionable rows
    first.

    Pure consumer of trading_app.config and trading_app.prop_profiles —
    zero re-encoded filter logic, zero calls to build_eligibility_report
    (View B is per-filter, not per-strategy).
    """
    from datetime import date as _date_type
    from datetime import datetime as _dt_type

    from trading_app.config import ALL_FILTERS, ATR_VELOCITY_OVERLAY
    from trading_app.prop_profiles import ACCOUNT_PROFILES

    # ── routed counts: one DB query ──────────────────────────────────
    routed_counts: dict[str, int] = {}
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        shelf_relation = deployable_validated_relation(con)
        rows = con.execute(
            f"""
            SELECT filter_type, COUNT(*)
            FROM {shelf_relation}
            GROUP BY filter_type
            """
        ).fetchall()
        for ft, n in rows:
            routed_counts[ft] = int(n)
    finally:
        con.close()

    # ── deployed counts: walk prop_profiles ──────────────────────────
    # Uses the canonical parse_strategy_id (which already strips _S075
    # suffix as of commit 35ae1fd). Lane → filter_type via the same
    # path used by View A, so counts stay consistent.
    deployed_counts: dict[str, int] = {}
    for _pid, profile in ACCOUNT_PROFILES.items():
        lanes = effective_daily_lanes(profile)
        if not profile.active or not lanes:
            continue
        for lane in lanes:
            try:
                dims = parse_strategy_id(lane.strategy_id)
            except Exception as exc:  # noqa: BLE001 — adapter boundary, surface loudly
                # Fail-loud per institutional-rigor rule 6 (no silent
                # failures). A malformed strategy_id would undercount
                # deployed lanes in View B; print a WARNING so the
                # trader notices while keeping the section renderable.
                print(
                    f"  WARNING: View B could not parse {lane.strategy_id}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            ft = dims["filter_type"]
            deployed_counts[ft] = deployed_counts.get(ft, 0) + 1

    # ── Walk ALL_FILTERS + overlay and build rows ────────────────────
    # Include the ATR velocity overlay as an extra entry. It lives
    # outside ALL_FILTERS but has the same ClassVar shape.
    entries: list[tuple[str, object]] = list(ALL_FILTERS.items())
    overlay_key = getattr(ATR_VELOCITY_OVERLAY, "filter_type", "ATR_VEL")
    entries.append((overlay_key, ATR_VELOCITY_OVERLAY))

    rows_out: list[dict] = []
    for key, filt in entries:
        cls = type(filt)
        class_name = cls.__name__

        # Read ClassVar metadata — all three are OPTIONAL (only ~6-12 of
        # 82 filters have them set). Treat absence as "not annotated".
        validated_for = getattr(cls, "VALIDATED_FOR", None) or ()
        last_revalidated = getattr(cls, "LAST_REVALIDATED", None)
        confidence_tier = getattr(cls, "CONFIDENCE_TIER", None)

        # Stale marker: use the canonical VALIDATION_FRESHNESS_DAYS
        # constant from trading_app.eligibility.builder (currently 180d).
        # Institutional-rigor rule 4: delegate, do not re-encode.
        is_stale = False
        last_rev_str = ""
        if last_revalidated is not None:
            if isinstance(last_revalidated, _dt_type):
                rev_date = last_revalidated.date()
            elif isinstance(last_revalidated, _date_type):
                rev_date = last_revalidated
            else:
                rev_date = None
            if rev_date is not None:
                last_rev_str = rev_date.isoformat()
                age_days = (trading_day - rev_date).days
                is_stale = age_days > VALIDATION_FRESHNESS_DAYS

        # Confidence tier: enum → string value. May be None (unannotated)
        # or a ConfidenceTier enum.
        tier_str = ""
        if confidence_tier is not None:
            tier_str = confidence_tier.value if hasattr(confidence_tier, "value") else str(confidence_tier)

        routed = routed_counts.get(key, 0)
        deployed = deployed_counts.get(key, 0)

        if deployed > 0:
            status = "LIVE"
        elif routed > 0:
            status = "ROUTED"
        else:
            status = "DEAD"

        rows_out.append(
            {
                "filter_type": key,
                "class_name": class_name,
                "description": _filter_description(key),
                "confidence_tier": tier_str,
                "validated_for": tuple(validated_for) if validated_for else (),
                "last_revalidated": last_rev_str,
                "is_stale": is_stale,
                "routed": routed,
                "deployed": deployed,
                "status": status,
            }
        )

    rows_out.sort(key=lambda r: (-r["deployed"], -r["routed"], r["filter_type"]))
    return rows_out


def _render_filter_universe_section(rows: list[dict]) -> str:
    """Pure function: list[row dict] → HTML fragment.

    Produces a collapsible <details> block with a summary line
    ('Filter Universe Audit — X total, Y routed, Z deployed') and a
    table of all rows, each with status badge + filter_type + class +
    description + tier + validated-for chips + revalidated date +
    STALE pill (if stale) + routed + deployed counts.

    Empty / missing metadata renders as '—' (em dash) rather than 'None'
    or a red warning — most filters (~70 of 82) are unannotated and
    this is the expected state, not a bug.
    """
    total = len(rows)
    routed_count = sum(1 for r in rows if r["routed"] > 0)
    deployed_count = sum(1 for r in rows if r["deployed"] > 0)
    live_lane_sum = sum(r["deployed"] for r in rows)
    summary_text = (
        f"Filter Universe Audit &mdash; {total} total, "
        f"{routed_count} routed, {deployed_count} deployed "
        f"({live_lane_sum} live lane{'s' if live_lane_sum != 1 else ''})"
    )

    body_rows = ""
    for r in rows:
        # Row class by status
        row_cls = {
            "LIVE": "row-live",
            "ROUTED": "row-routed",
            "DEAD": "row-dead",
        }.get(r["status"], "row-dead")

        # Status badge. NOTE: ROUTED intentionally reuses the existing
        # badge-filter-check amber palette (same visual meaning: "pay
        # attention, not immediately actionable") rather than introducing
        # a new dedicated class. Flagged in Gate 1 review as LOW-2 and
        # accepted.
        status_badge_cls = {
            "LIVE": "badge-filter-live",
            "ROUTED": "badge-filter-check",
            "DEAD": "badge-filter-dead",
        }.get(r["status"], "badge-filter-dead")
        status_badge = f'<span class="badge {status_badge_cls}">{r["status"]}</span>'

        # Validated-for: show up to 3 chips + "..." with full tuple in tooltip
        vf = r["validated_for"]
        if vf:
            chips = [f'<span class="vf-chip">{i}·{s}</span>' for i, s in list(vf)[:3]]
            vf_html = " ".join(chips)
            if len(vf) > 3:
                tooltip = "; ".join(f"{i}·{s}" for i, s in vf)
                vf_html += f' <span class="vf-more" title="{tooltip}">+{len(vf) - 3}</span>'
        else:
            vf_html = "&mdash;"

        # Last revalidated + optional STALE pill
        if r["last_revalidated"]:
            lr_html = r["last_revalidated"]
            if r["is_stale"]:
                lr_html += ' <span class="pill pill-stale">STALE</span>'
        else:
            lr_html = "&mdash;"

        tier_html = r["confidence_tier"] if r["confidence_tier"] else "&mdash;"

        body_rows += f"""
            <tr class="{row_cls}">
                <td>{status_badge}</td>
                <td class="filter-key">{r["filter_type"]}</td>
                <td class="filter-class">{r["class_name"]}</td>
                <td class="filter-desc">{r["description"]}</td>
                <td class="filter-tier">{tier_html}</td>
                <td class="filter-vf">{vf_html}</td>
                <td class="filter-lr">{lr_html}</td>
                <td class="filter-routed">{r["routed"]}</td>
                <td class="filter-deployed">{r["deployed"]}</td>
            </tr>"""

    return f"""
        <details class="filter-universe">
            <summary>{summary_text}</summary>
            <table class="filter-universe-table">
                <thead>
                    <tr>
                        <th>Status</th>
                        <th>Filter</th>
                        <th>Class</th>
                        <th>Description</th>
                        <th>Tier</th>
                        <th>Validated For</th>
                        <th>Revalidated</th>
                        <th>Routed</th>
                        <th>Deployed</th>
                    </tr>
                </thead>
                <tbody>{body_rows}
                </tbody>
            </table>
        </details>"""


def generate_html(
    trades: list[dict],
    session_times: dict,
    trading_day: date,
    opportunities: list[dict] | None = None,
    manual_candidates: list[dict] | None = None,
    regime_ctx: dict[str, dict] | None = None,
    filter_universe_rows: list[dict] | None = None,
) -> str:
    """Generate self-contained HTML trade sheet with unified timeline.

    filter_universe_rows: optional list of row dicts from
    _build_filter_universe_rows. When provided, a collapsible Filter
    Universe Audit section (View B) is rendered below the session cards.
    Passed in rather than computed here so generate_html stays a pure
    rendering function with no DB dependency.
    """

    day_name = trading_day.strftime("%A")
    date_str = trading_day.strftime("%d %b %Y")
    now_str = datetime.now().strftime("%H:%M")

    # Profile summary bar — compute per-profile EV and DD budget
    from trading_app.prop_profiles import PROP_FIRM_SPECS, get_account_tier

    profiles_used = {}
    for t in trades:
        pid = t.get("profile", "unknown")
        if pid not in profiles_used:
            prof = ACCOUNT_PROFILES.get(pid)
            if prof:
                spec = PROP_FIRM_SPECS.get(prof.firm, None)
                tier = get_account_tier(prof.firm, prof.account_size)
                auto_label = {"none": "MANUAL", "full": "AUTO", "semi": "SEMI"}.get(
                    spec.auto_trading if spec else "none", "?"
                )
                dll = tier.daily_loss_limit
                profiles_used[pid] = {
                    "firm": prof.firm.upper(),
                    "size": f"${prof.account_size // 1000}K",
                    "dd": tier.max_dd,
                    "dd_str": f"${tier.max_dd:,}",
                    "dll": dll,
                    "dll_str": f"${dll:,}" if dll else "none",
                    "stop": f"{prof.stop_multiplier}x",
                    "lanes": len(effective_daily_lanes(prof)),
                    "mode": auto_label,
                    "copies": prof.copies,
                    "ev_per_copy": 0.0,
                }

    # Sum EV per profile
    for t in trades:
        pid = t.get("profile", "unknown")
        if pid in profiles_used and t["exp_dollars"] is not None:
            profiles_used[pid]["ev_per_copy"] += t["exp_dollars"]

    profile_bar_html = ""
    for _pid, info in profiles_used.items():
        copies = info["copies"]
        copies_note = f" x{copies}" if copies > 1 else ""
        ev_1 = info["ev_per_copy"]
        ev_total = ev_1 * copies
        ev_line = f"EV ${ev_1:.0f}/day" if copies == 1 else f"EV ${ev_1:.0f}/day x{copies} = ${ev_total:.0f}"
        dll_line = f" | DLL {info['dll_str']}" if info["dll"] else ""
        lane_word = "lane" if info["lanes"] == 1 else "lanes"
        profile_bar_html += f"""
        <div class="profile-card profile-{info["mode"].lower()}">
            <strong>{info["firm"]} {info["size"]}{copies_note}</strong>
            <span class="profile-mode">{info["mode"]}</span>
            <div class="profile-detail">DD {info["dd_str"]}{dll_line} | Stop {info["stop"]} | {info["lanes"]} {lane_word}</div>
            <div class="profile-ev">{ev_line}</div>
        </div>"""

    # Section tags set by main() before calling generate_html()
    all_trades = list(trades) + (opportunities or []) + (manual_candidates or [])

    n_deployed = len(trades)
    n_opp = len(opportunities or [])
    n_manual = len(manual_candidates or [])

    # Find next upcoming session
    next_session = _next_session_label(session_times)

    # Build unified session cards
    cards_html = _build_session_cards(
        all_trades,
        session_times,
        profiles_used,
        next_session=next_session,
        regime_ctx=regime_ctx,
    )

    # Build View B — Filter Universe Audit section (optional, only when
    # rows were pre-computed by main() via _build_filter_universe_rows)
    filter_universe_html = _render_filter_universe_section(filter_universe_rows) if filter_universe_rows else ""

    # Instrument summary — all three sections
    instr_deployed_exp = {}
    instr_opp_exp = {}
    instr_manual_exp = {}
    for t in trades:
        if t["exp_dollars"] is not None:
            instr_deployed_exp[t["instrument"]] = instr_deployed_exp.get(t["instrument"], 0) + t["exp_dollars"]
    for t in opportunities or []:
        if t["exp_dollars"] is not None:
            instr_opp_exp[t["instrument"]] = instr_opp_exp.get(t["instrument"], 0) + t["exp_dollars"]
    for t in manual_candidates or []:
        if t["exp_dollars"] is not None:
            instr_manual_exp[t["instrument"]] = instr_manual_exp.get(t["instrument"], 0) + t["exp_dollars"]

    all_instruments = sorted(set(t["instrument"] for t in all_trades))
    deployed_total = sum(instr_deployed_exp.values())
    opp_total = sum(instr_opp_exp.values())
    manual_total = sum(instr_manual_exp.values())

    # Total EV across copies
    total_ev_with_copies = sum(info["ev_per_copy"] * info["copies"] for info in profiles_used.values())

    summary_html = ""
    for instr in all_instruments:
        dep = instr_deployed_exp.get(instr, 0)
        opp = instr_opp_exp.get(instr, 0)
        man = instr_manual_exp.get(instr, 0)
        dep_count = sum(1 for t in trades if t["instrument"] == instr)
        opp_count = sum(1 for t in (opportunities or []) if t["instrument"] == instr)
        man_count = sum(1 for t in (manual_candidates or []) if t["instrument"] == instr)

        count_parts = []
        if dep_count:
            count_parts.append(f"{dep_count} live")
        if opp_count:
            count_parts.append(f"{opp_count} avail")
        if man_count:
            count_parts.append(f"{man_count} manual")
        count_line = " + ".join(count_parts) if count_parts else "0"

        ev_parts = []
        if dep > 0:
            ev_parts.append(f"${dep:.0f} live")
        if opp > 0:
            ev_parts.append(f"+${opp:.0f} avail")
        if man > 0:
            ev_parts.append(f"+${man:.0f} manual")
        ev_line = " ".join(ev_parts) if ev_parts else "$0"

        summary_html += f"""
        <div class="summary-card">
            <div class="summary-instrument">{instr}</div>
            <div class="summary-count">{count_line}</div>
            <div class="summary-dollars">{ev_line}</div>
        </div>"""

    # Grand total card
    summary_html += f"""
        <div class="summary-card" style="border-color: #3fb950;">
            <div class="summary-instrument">TOTAL</div>
            <div class="summary-dollars">${deployed_total:.0f}/day live</div>
            <div class="summary-opp">+${opp_total:.0f} avail +${manual_total:.0f} manual</div>
            <div class="summary-count" style="margin-top:6px">${total_ev_with_copies:.0f}/day with copies</div>
        </div>"""

    fitness_errors = [t for t in all_trades if t.get("fitness_error")]
    fitness_warning_html = ""
    if fitness_errors:
        error_items = ""
        for t in fitness_errors[:10]:
            error_items += f"<li><code>{t['strategy_id']}</code>: {t['fitness_error']}</li>"
        more_note = ""
        if len(fitness_errors) > 10:
            more_note = f"<p>+ {len(fitness_errors) - 10} more fitness lookup errors.</p>"
        fitness_warning_html = f"""
    <div class="warning-box">
        <strong>Fitness lookup errors:</strong> {len(fitness_errors)} row(s) rendered as <code>UNKNOWN</code>.
        <ul>{error_items}</ul>
        {more_note}
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trade Sheet — {day_name} {date_str}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #0d1117;
        color: #e6edf3;
        padding: 20px;
        max-width: 1200px;
        margin: 0 auto;
    }}
    .header {{
        text-align: center;
        margin-bottom: 30px;
        padding: 20px;
        border-bottom: 2px solid #30363d;
    }}
    .header h1 {{
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 5px;
    }}
    .header .subtitle {{
        color: #8b949e;
        font-size: 14px;
    }}
    .header .date {{
        font-size: 20px;
        color: #58a6ff;
        margin-top: 5px;
    }}
    .entry-model-note {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 24px;
        font-size: 14px;
        color: #8b949e;
        text-align: center;
    }}
    .entry-model-note strong {{
        color: #58a6ff;
    }}
    .summary-row {{
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
        flex-wrap: wrap;
    }}
    .summary-card {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        flex: 1;
        min-width: 140px;
        text-align: center;
    }}
    .summary-instrument {{
        font-size: 18px;
        font-weight: 700;
        color: #58a6ff;
    }}
    .summary-count {{
        font-size: 14px;
        color: #8b949e;
        margin-top: 4px;
    }}
    .summary-dollars {{
        font-size: 16px;
        font-weight: 600;
        color: #3fb950;
        margin-top: 4px;
    }}
    .summary-opp {{
        font-size: 13px;
        color: #8b949e;
        margin-top: 2px;
    }}
    .session-card {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        margin-bottom: 16px;
        overflow: hidden;
    }}
    .session-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        background: #1c2128;
        border-bottom: 1px solid #30363d;
        flex-wrap: wrap;
    }}
    .session-time {{
        font-size: 22px;
        font-weight: 700;
        color: #f0883e;
        min-width: 90px;
    }}
    .session-name {{
        font-size: 16px;
        font-weight: 600;
        color: #e6edf3;
    }}
    .session-instruments {{
        display: flex;
        gap: 4px;
        align-items: center;
    }}
    .session-count {{
        font-size: 12px;
        color: #8b949e;
        padding: 2px 8px;
        background: #21262d;
        border-radius: 10px;
    }}
    .session-event {{
        font-size: 12px;
        color: #8b949e;
        margin-left: auto;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
    }}
    thead th {{
        text-align: left;
        padding: 8px 10px;
        font-size: 11px;
        text-transform: uppercase;
        color: #8b949e;
        border-bottom: 1px solid #30363d;
        font-weight: 600;
    }}
    tbody td {{
        padding: 8px 10px;
        font-size: 13px;
        border-bottom: 1px solid #21262d;
    }}
    tbody tr:last-child td {{
        border-bottom: none;
    }}
    tbody tr:hover {{
        background: #1c2128;
    }}
    .instrument-cell {{
        font-weight: 700;
        color: #58a6ff;
        cursor: help;
    }}
    .filter-cell {{
        color: #e6edf3;
    }}
    .dollars-cell {{
        font-weight: 600;
        color: #3fb950;
    }}
    .expr-high {{
        color: #3fb950;
        font-weight: 600;
    }}
    .fit-ok {{
        color: #3fb950;
        font-size: 12px;
    }}
    .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }}
    .badge-long {{
        background: #1f3a2a;
        color: #3fb950;
        border: 1px solid #3fb950;
    }}
    .badge-short {{
        background: #3d1f20;
        color: #f85149;
        border: 1px solid #f85149;
    }}
    .badge-cont {{
        background: #2a2f1f;
        color: #d29922;
        border: 1px solid #d29922;
    }}
    .badge-watch {{
        background: #3d2e1f;
        color: #d29922;
        border: 1px solid #d29922;
    }}
    .badge-decay {{
        background: #3d1f20;
        color: #f85149;
        border: 1px solid #f85149;
    }}
    .badge-stale {{
        background: #2d2d2d;
        color: #8b949e;
        border: 1px solid #8b949e;
    }}
    .badge-unknown {{
        background: #5c4712;
        color: #ffd58a;
        border: 1px solid #d29922;
    }}
    .badge-purged {{
        background: #2d2020;
        color: #8b5e5e;
        border: 1px solid #6e4444;
    }}
    .badge-deployed {{
        background: #1f3a2a;
        color: #3fb950;
        border: 1px solid #3fb950;
    }}
    .badge-opp {{
        background: #1a2a3a;
        color: #58a6ff;
        border: 1px solid #58a6ff;
    }}
    .badge-manual {{
        background: #3d2e1f;
        color: #d29922;
        border: 1px solid #d29922;
    }}
    .badge-core {{
        background: #1f3a2a;
        color: #3fb950;
        border: 1px solid #2ea043;
        font-size: 10px;
    }}
    .badge-regime {{
        background: #3d2e1f;
        color: #d29922;
        border: 1px solid #d29922;
        font-size: 10px;
    }}
    .badge-instr {{
        background: #1a2a3a;
        color: #58a6ff;
        border: 1px solid #388bfd;
        font-size: 10px;
        margin-right: 4px;
        padding: 2px 6px;
        border-radius: 4px;
    }}
    .row-deployed {{
        border-left: 3px solid #3fb950;
    }}
    .row-deployed td:first-child {{
        padding-left: 8px;
    }}
    .row-opportunity {{
        border-left: 3px solid #58a6ff;
    }}
    .row-opportunity td:first-child {{
        padding-left: 8px;
    }}
    .row-manual {{
        border-left: 3px solid #d29922;
    }}
    .row-manual td:first-child {{
        padding-left: 8px;
    }}
    .profile-bar {{
        display: flex;
        gap: 12px;
        margin-bottom: 24px;
        flex-wrap: wrap;
    }}
    .profile-card {{
        padding: 10px 16px;
        border-radius: 8px;
        border: 1px solid #30363d;
        background: #161b22;
        flex: 1;
        min-width: 200px;
    }}
    .profile-card strong {{ font-size: 14px; }}
    .profile-mode {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        margin-left: 8px;
    }}
    .profile-manual .profile-mode {{
        background: #1a3a1a;
        color: #3fb950;
        border: 1px solid #3fb950;
    }}
    .profile-auto .profile-mode {{
        background: #1a2a3a;
        color: #58a6ff;
        border: 1px solid #58a6ff;
    }}
    .profile-detail {{
        font-size: 12px;
        color: #8b949e;
        margin-top: 4px;
    }}
    .profile-ev {{
        font-size: 13px;
        font-weight: 600;
        color: #3fb950;
        margin-top: 4px;
    }}
    .warning-box {{
        background: #271b05;
        border: 1px solid #d29922;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 24px;
        color: #ffd58a;
        line-height: 1.5;
    }}
    .warning-box ul {{
        margin: 8px 0 0 20px;
    }}
    .warning-box code {{
        background: #3d2e1f;
        padding: 1px 4px;
        border-radius: 4px;
    }}
    .next-session {{
        border-color: #f0883e;
        box-shadow: 0 0 8px rgba(240, 136, 62, 0.3);
    }}
    .badge-next {{
        background: #3d2e1f;
        color: #f0883e;
        border: 1px solid #f0883e;
        margin-left: 8px;
    }}
    .footer {{
        text-align: center;
        padding: 20px;
        color: #484f58;
        font-size: 12px;
        border-top: 1px solid #30363d;
        margin-top: 20px;
    }}
    .legend {{
        display: flex;
        gap: 16px;
        justify-content: center;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }}
    .legend-item {{
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: #8b949e;
    }}
    .legend-swatch {{
        width: 14px;
        height: 14px;
        border-radius: 3px;
    }}
    .legend-swatch-live {{ background: #3fb950; }}
    .legend-swatch-avail {{ background: #58a6ff; }}
    .legend-swatch-manual {{ background: #d29922; }}
    .regime-bar {{
        display: flex;
        gap: 8px;
        padding: 4px 16px 8px;
        flex-wrap: wrap;
    }}
    .regime-chip {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }}
    .regime-high {{
        background: #1f3a2a;
        color: #3fb950;
        border: 1px solid #2ea043;
    }}
    .regime-normal {{
        background: #1a2a3a;
        color: #58a6ff;
        border: 1px solid #388bfd;
    }}
    .regime-low {{
        background: #3d1f20;
        color: #f85149;
        border: 1px solid #da3633;
    }}
    .badge-filter-active {{
        background: #1f3a2a;
        color: #3fb950;
        border: 1px solid #2ea043;
    }}
    .badge-filter-check {{
        background: #3d2e1f;
        color: #d29922;
        border: 1px solid #d29922;
    }}
    .badge-filter-missing {{
        background: #3d1f1f;
        color: #f0883e;
        border: 1px solid #f0883e;
    }}
    .pill {{
        display: inline-block;
        font-size: 9px;
        padding: 1px 5px;
        border-radius: 3px;
        margin-left: 3px;
        vertical-align: middle;
        font-weight: 600;
        letter-spacing: 0.3px;
    }}
    .pill-stale {{
        background: #3d2e1f;
        color: #d29922;
        border: 1px solid #d29922;
    }}
    .pill-half {{
        background: #3d2e1f;
        color: #f0883e;
        border: 1px solid #f0883e;
    }}
    /* View B: Filter Universe Audit section */
    .filter-universe {{
        margin: 24px 0;
        padding: 16px;
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
    }}
    .filter-universe summary {{
        cursor: pointer;
        font-size: 14px;
        font-weight: 600;
        color: #c9d1d9;
        padding: 6px 0;
    }}
    .filter-universe summary:hover {{
        color: #58a6ff;
    }}
    .filter-universe-table {{
        width: 100%;
        margin-top: 12px;
        border-collapse: collapse;
        font-size: 12px;
    }}
    .filter-universe-table th {{
        text-align: left;
        padding: 6px 8px;
        border-bottom: 1px solid #30363d;
        color: #8b949e;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 10px;
        letter-spacing: 0.5px;
    }}
    .filter-universe-table td {{
        padding: 5px 8px;
        border-bottom: 1px solid #21262d;
        vertical-align: top;
    }}
    .filter-universe-table .filter-key {{
        font-family: monospace;
        color: #79c0ff;
    }}
    .filter-universe-table .filter-class {{
        color: #8b949e;
        font-size: 11px;
    }}
    .filter-universe-table .filter-desc {{
        color: #c9d1d9;
    }}
    .filter-universe-table .filter-routed,
    .filter-universe-table .filter-deployed {{
        text-align: right;
        font-variant-numeric: tabular-nums;
        color: #c9d1d9;
    }}
    .row-live {{
        border-left: 3px solid #3fb950;
    }}
    .row-routed {{
        border-left: 3px solid #d29922;
    }}
    .row-dead {{
        border-left: 3px solid #484f58;
        opacity: 0.55;
    }}
    .badge-filter-live {{
        background: #1f3a2a;
        color: #3fb950;
        border: 1px solid #2ea043;
    }}
    .badge-filter-dead {{
        background: #21262d;
        color: #8b949e;
        border: 1px solid #484f58;
    }}
    .vf-chip {{
        display: inline-block;
        font-size: 10px;
        padding: 1px 5px;
        border-radius: 3px;
        margin-right: 3px;
        background: #21262d;
        color: #8b949e;
        border: 1px solid #30363d;
    }}
    .vf-more {{
        display: inline-block;
        font-size: 10px;
        padding: 1px 5px;
        border-radius: 3px;
        background: #0d1117;
        color: #8b949e;
        border: 1px dashed #30363d;
        cursor: help;
    }}
    .row-inactive {{
        opacity: 0.25;
    }}
    .row-inactive td {{
        text-decoration: line-through;
        color: #484f58 !important;
    }}
    .freq-cell {{
        font-size: 12px;
        color: #8b949e;
        white-space: nowrap;
    }}
    .stats-src {{
        font-size: 9px;
        padding: 1px 4px;
        border-radius: 3px;
        margin-left: 3px;
        vertical-align: middle;
    }}
    .stats-src-trailing {{
        background: #1f3a2a;
        color: #3fb950;
        border: 1px solid #2ea043;
    }}
    .stats-src-all {{
        background: #21262d;
        color: #484f58;
        border: 1px solid #30363d;
    }}
    @media print {{
        body {{ background: white; color: black; padding: 10px; }}
        .session-card {{ border: 1px solid #ccc; }}
        .session-header {{ background: #f0f0f0; }}
        .session-time {{ color: #d35400; }}
        .instrument-cell {{ color: #2980b9; }}
        .dollars-cell {{ color: #27ae60; }}
        .summary-card {{ border: 1px solid #ccc; }}
    }}
    @media (max-width: 768px) {{
        .session-header {{ flex-direction: column; gap: 4px; }}
        .session-event {{ margin-left: 0; }}
        table {{ font-size: 11px; }}
        thead th, tbody td {{ padding: 5px 6px; }}
    }}
</style>
</head>
<body>
    <div class="header">
        <h1>TRADE SHEET</h1>
        <div class="date">{day_name} {date_str}</div>
        <div class="subtitle">Generated {now_str} &mdash; {n_deployed} live + {n_opp} available + {n_manual} manual &mdash; All times Brisbane (AEST UTC+10)</div>
    </div>

    <div class="profile-bar">
        {profile_bar_html}
    </div>

    <div class="entry-model-note">
        All entries are <strong>E2 (stop-market)</strong> &mdash; place stop orders at ORB high/low.
        They trigger automatically on breakout. ORB = first 5 minutes after session start.
    </div>

    {fitness_warning_html}

    <div class="summary-row">
        {summary_html}
    </div>

    <div class="legend">
        <div class="legend-item"><div class="legend-swatch legend-swatch-live"></div> LIVE &mdash; deployed in prop profile</div>
        <div class="legend-item"><div class="legend-swatch legend-swatch-avail"></div> AVAIL &mdash; validated, passes gates</div>
        <div class="legend-item"><div class="legend-swatch legend-swatch-manual"></div> MANUAL &mdash; your discretion (incl. REGIME)</div>
    </div>

    {cards_html}

    {filter_universe_html}

    <div class="footer">
        Unified timeline &mdash; prop_profiles lanes + validated opportunities + manual candidates &mdash;
        hover instrument for strategy ID or profile &mdash; dollar gate applied where applicable
    </div>
</body>
</html>"""
    return html


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate daily trade sheet HTML")
    parser.add_argument("--date", type=str, default=None, help="Trading date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path. Default: trade_sheet.html")
    parser.add_argument("--db-path", type=Path, default=None, help="Path to gold.db")
    parser.add_argument("--no-open", action="store_true", help="Don't open in browser after generating")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Filter to one profile (e.g. apex_50k_manual). Default: all active.",
    )
    parser.add_argument(
        "--deployed-only",
        action="store_true",
        help="Only show deployed lanes (V1 behavior). Skip opportunities section.",
    )
    args = parser.parse_args()

    db_path = args.db_path or GOLD_DB_PATH
    trading_day = date.fromisoformat(args.date) if args.date else date.today()
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "trade_sheet.html"

    print("Trade Sheet Generator")
    print(f"  Date:   {trading_day}")
    print(f"  DB:     {db_path}")
    print(f"  Output: {output_path}")

    # Data freshness check
    try:
        con_check = duckdb.connect(str(db_path), read_only=True)
        try:
            for table, col in [("daily_features", "trading_day"), ("orb_outcomes", "trading_day")]:
                row = con_check.execute(f"SELECT MAX({col}) FROM {table}").fetchone()
                max_date = row[0] if row is not None else None
                days_stale = (trading_day - max_date.date()).days if max_date else 999
                status = "OK" if days_stale <= 2 else f"STALE ({days_stale} days old)"
                print(f"  {table}: {max_date.date() if max_date else 'EMPTY'} [{status}]")
        finally:
            con_check.close()
    except duckdb.IOException:
        print("  Data freshness: SKIP (DB locked by another process)")
    except (AttributeError, TypeError) as exc:
        print(f"  Data freshness: SKIP ({type(exc).__name__}: {exc})")
    print()

    # Resolve session times
    print("Resolving session times...")
    session_times = _resolve_session_times(trading_day)
    for label in sorted(session_times, key=lambda s: _sort_key(*session_times[s])):
        h, m = session_times[label]
        print(f"  {_format_time(h, m):>10}  {label}")
    print()

    # Collect deployed trades
    if args.profile:
        print(f"Filtering to profile: {args.profile}")
    print("Building deployed lanes...")
    trades = collect_trades(trading_day, db_path, profile_filter=args.profile)
    print(f"  {len(trades)} deployed trades across {len(set(t['instrument'] for t in trades))} instruments")

    # Collect opportunities (unless --deployed-only)
    opportunities = []
    manual_candidates = []
    if not args.deployed_only:
        print("Scanning validated opportunities...")
        deployed_sids = {t["strategy_id"] for t in trades}
        opportunities = collect_opportunities(db_path, deployed_sids)
        opp_instruments = set(t["instrument"] for t in opportunities) if opportunities else set()
        print(f"  {len(opportunities)} opportunities across {len(opp_instruments)} instruments")

        # Collect manual candidates (REGIME tier + PURGED, for manual trading)
        print("Scanning manual candidates...")
        shown_sids = deployed_sids | {t["strategy_id"] for t in opportunities}
        manual_candidates = collect_manual_candidates(db_path, shown_sids)
        manual_instruments = set(t["instrument"] for t in manual_candidates) if manual_candidates else set()
        print(f"  {len(manual_candidates)} manual candidates across {len(manual_instruments)} instruments")
    print()

    if not trades and not opportunities and not manual_candidates:
        print("ERROR: No trades found. Check DB and prop_profiles.py.")
        sys.exit(1)

    # Regime context
    print("Loading regime context...")
    try:
        regime_ctx = _get_regime_context(db_path)
        for instr, ctx in sorted(regime_ctx.items()):
            atr = ctx.get("atr_pct")
            if atr is not None:
                label = "HIGH" if atr >= 70 else "NORMAL" if atr >= 50 else "LOW"
                print(f"  {instr}: ATR {atr:.0f}th pct ({label}), as of {ctx.get('as_of')}")
            else:
                print(f"  {instr}: ATR data unavailable")
    except Exception as exc:
        print(f"  Regime context failed ({type(exc).__name__}: {exc}) — using CHECK for all")
        regime_ctx = {}

    # Tag trades with section and enrich with regime data
    for t in trades:
        t["section"] = "deployed"
    for t in opportunities:
        t["section"] = "opportunity"
    for t in manual_candidates:
        t["section"] = "manual"

    all_trades_for_enrichment = list(trades) + list(opportunities) + list(manual_candidates)
    if regime_ctx:
        _enrich_trades_with_regime(all_trades_for_enrichment, regime_ctx, db_path)

    # Canonical eligibility enrichment — single source of truth for filter
    # status is trading_app.eligibility.builder.build_eligibility_report,
    # which is also the live dashboard's signal source (Phase 3 of the
    # parent eligibility-context design). See
    # docs/plans/2026-04-07-trade-book-canonicalization-design.md.
    feature_rows = _prefetch_feature_rows(all_trades_for_enrichment, db_path)
    _enrich_trades_with_eligibility(all_trades_for_enrichment, trading_day, feature_rows)

    # View B — Filter Universe Audit (Phase 2 complete): compute audit
    # rows once from ALL_FILTERS + ATR velocity overlay + validated_setups
    # + prop_profiles. See
    # docs/plans/2026-04-07-filter-universe-audit-design.md.
    filter_universe_rows = _build_filter_universe_rows(db_path, trading_day)
    print(
        f"  {len(filter_universe_rows)} filters in universe "
        f"({sum(1 for r in filter_universe_rows if r['status'] == 'LIVE')} LIVE, "
        f"{sum(1 for r in filter_universe_rows if r['status'] == 'ROUTED')} ROUTED, "
        f"{sum(1 for r in filter_universe_rows if r['status'] == 'DEAD')} DEAD)",
        flush=True,
    )
    print()

    # Generate HTML
    html = generate_html(
        trades,
        session_times,
        trading_day,
        opportunities=opportunities or None,
        manual_candidates=manual_candidates or None,
        regime_ctx=regime_ctx or None,
        filter_universe_rows=filter_universe_rows,
    )
    output_path.write_text(html, encoding="utf-8")
    print(f"Written to {output_path}")

    # Open in browser
    if not args.no_open:
        webbrowser.open(str(output_path))
        print("Opened in browser.")


if __name__ == "__main__":
    main()
