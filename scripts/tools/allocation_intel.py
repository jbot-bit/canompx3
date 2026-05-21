"""Allocation intelligence — surface adjacent wins from current allocation state.

Run after every rebalance, or on session start, to surface:
  1. Active lanes (current live state)
  2. Displaced lanes (C8-PASSED but soft-gate rejected — next-profile candidates)
  3. Paused-reason histogram (distinguishes audit backlog from real fails)
  4. Session regime sweep with coverage gaps (HOT sessions undermapped)
  5. Allocation staleness (days since last rebalance)

Read-only. Reads canonical: lane allocation file + validated_setups.

Usage:
    python scripts/tools/allocation_intel.py
    python scripts/tools/allocation_intel.py --profile topstep_50k_mnq_auto
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.paths import GOLD_DB_PATH
from trading_app.prop_profiles import (
    ACCOUNT_PROFILES,
    legacy_lane_allocation_path,
    resolve_allocation_json,
)


def _tradeable_sessions() -> tuple[str, ...]:
    """Sessions allowed by at least one ACTIVE prop profile.

    More restrictive than SESSION_CATALOG (which includes diagnostic labels
    like BRISBANE_1025). Canonical source per `prop_profiles.ACCOUNT_PROFILES`.
    """
    sessions: set[str] = set()
    for prof in ACCOUNT_PROFILES.values():
        if prof.active and prof.allowed_sessions:
            sessions.update(prof.allowed_sessions)
    return tuple(sorted(sessions))


def _load_allocation(profile_id: str | None = None) -> dict:
    """Load allocation data via the canonical resolver.

    When ``profile_id`` is supplied, uses ``resolve_allocation_json`` for
    new-path-first + profile-mismatch guard semantics. Otherwise falls back
    to a direct read of the legacy single-profile file (no profile guard —
    used by the informational-only main() path).
    """
    if profile_id:
        result = resolve_allocation_json(profile_id)
        if result.data is None:
            raise FileNotFoundError(
                f"Allocation file missing or profile_id mismatch for {profile_id!r}"
            )
        return result.data
    legacy_path = legacy_lane_allocation_path()
    if not legacy_path.exists():
        raise FileNotFoundError(f"Allocation file missing: {legacy_path}")
    return json.loads(legacy_path.read_text())


def _staleness_days(rebalance_date: str | None) -> int | None:
    if not rebalance_date:
        return None
    try:
        return (date.today() - date.fromisoformat(rebalance_date)).days
    except ValueError:
        return None


def _active_lanes(alloc: dict) -> list[dict]:
    return list(alloc.get("lanes", []))


def _displaced(alloc: dict) -> list[dict]:
    return list(alloc.get("displaced", []))


def _paused(alloc: dict) -> list[dict]:
    return list(alloc.get("paused", []))


def _paused_histogram(paused: list[dict]) -> list[tuple[str, int]]:
    reasons: Counter[str] = Counter()
    for p in paused:
        r = p.get("reason") or p.get("pause_reason") or "unknown"
        first = r.split("|")[0].strip() if r else "unknown"
        reasons[first] += 1
    return reasons.most_common(10)


def _c8_status_for(con: duckdb.DuckDBPyConnection, sids: list[str]) -> dict[str, dict]:
    if not sids:
        return {}
    placeholders = ",".join(["?"] * len(sids))
    rows = con.execute(
        f"SELECT strategy_id, c8_oos_status, expectancy_r, sample_size FROM validated_setups WHERE strategy_id IN ({placeholders})",
        sids,
    ).fetchall()
    return {r[0]: {"c8": r[1], "expr": r[2], "n": r[3]} for r in rows}


def _compute_session_regimes(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """6-month trailing avg_r for unfiltered E2 RR1.0 outcomes by instrument x session.

    Mirrors the rebalancer's session-regime sweep. Computed inline so this script
    is self-contained (the rebalancer prints regimes to stdout but does not
    persist them to the lane allocation file).

    Filters to ACTIVE_ORB_INSTRUMENTS x _tradeable_sessions() -- ignores dead
    instruments (SIL/M2K/GC/MBT/MCL/M6E) and non-tradeable session labels
    (e.g., BRISBANE_1025 diagnostic).
    """
    active_instruments = tuple(ACTIVE_ORB_INSTRUMENTS)
    tradeable_sessions = _tradeable_sessions()
    placeholders_i = ",".join(["?"] * len(active_instruments))
    placeholders_s = ",".join(["?"] * len(tradeable_sessions))
    rows = con.execute(
        f"""
        SELECT symbol AS instrument, orb_label, AVG(pnl_r) AS avg_r, COUNT(*) AS n
        FROM orb_outcomes
        WHERE entry_model = 'E2' AND rr_target = 1.0 AND confirm_bars = 1 AND orb_minutes = 5
          AND pnl_r IS NOT NULL
          AND trading_day >= CURRENT_DATE - INTERVAL 180 DAY
          AND symbol IN ({placeholders_i})
          AND orb_label IN ({placeholders_s})
        GROUP BY symbol, orb_label
        HAVING COUNT(*) >= 20
        ORDER BY avg_r DESC
        """,
        list(active_instruments) + list(tradeable_sessions),
    ).fetchall()
    out: list[dict] = []
    for r in rows:
        avg_r = float(r[2])
        regime = "HOT" if avg_r >= 0.03 else ("COLD" if avg_r <= -0.03 else "FLAT")
        out.append({"instrument": r[0], "session": r[1], "avg_r": avg_r, "n": int(r[3]), "regime": regime})
    return out


def _hot_session_coverage(con: duckdb.DuckDBPyConnection, regimes: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in regimes:
        if r["regime"] != "HOT":
            continue
        row = con.execute(
            """
            SELECT COUNT(*) AS n, COALESCE(AVG(expectancy_r), 0) AS avg_expr,
                   COALESCE(SUM(CASE WHEN expectancy_r > 0.15 THEN 1 ELSE 0 END), 0) AS strong
            FROM validated_setups
            WHERE instrument = ? AND orb_label = ?
            """,
            [r["instrument"], r["session"]],
        ).fetchone()
        if row is None:
            continue
        out.append(
            {
                "instrument": r["instrument"],
                "session": r["session"],
                "avg_r": r["avg_r"],
                "validated_n": int(row[0]),
                "avg_expr": float(row[1]),
                "strong": int(row[2]),
            }
        )
    return sorted(out, key=lambda g: (g["validated_n"], -g["strong"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Allocation intelligence — adjacent-win surface")
    parser.add_argument("--profile", type=str, default=None, help="Profile filter (informational only)")
    args = parser.parse_args()

    alloc = _load_allocation(args.profile)
    rebal_date = alloc.get("rebalance_date")
    profile_id = alloc.get("profile_id", "unknown")
    if args.profile and args.profile != profile_id:
        print(f"WARN: requested profile={args.profile} but JSON profile={profile_id}")

    stale = _staleness_days(rebal_date)
    active = _active_lanes(alloc)
    displaced = _displaced(alloc)
    paused = _paused(alloc)

    print(f"# Allocation Intelligence — {profile_id}")
    print(f"Rebalance date: {rebal_date}  ({stale} days ago)" if stale is not None else f"Rebalance date: {rebal_date}")
    if stale is not None and stale >= 35:
        print("  ** WARNING: allocation >35 days stale; pre-session-check will warn **")
    print(f"Active lanes: {len(active)}  ·  Displaced: {len(displaced)}  ·  Paused: {len(paused)}")
    print()

    print("## 1. Active lanes")
    for ln in active:
        sid = ln.get("strategy_id", "?")
        annr = ln.get("annual_r") or ln.get("expected_annual_r") or "?"
        expr = ln.get("expectancy_r") or "?"
        print(f"  {sid:60s}  AnnR={annr}  ExpR={expr}")
    print()

    print("## 2. Displaced (soft-gate rejected - next-profile candidates)")
    if not displaced:
        print("  (none)")
    else:
        con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
        sids: list[str] = [str(d.get("strategy_id")) for d in displaced if d.get("strategy_id")]
        c8 = _c8_status_for(con, sids)
        for d in displaced:
            sid = d.get("strategy_id", "?")
            info = c8.get(sid, {})
            print(f"  {sid:60s}  ExpR={info.get('expr', '?')}  N={info.get('n', '?')}  C8={info.get('c8', '?')}")
        con.close()
    print()

    print("## 3. Paused reason histogram (top 10)")
    for reason, count in _paused_histogram(paused):
        print(f"  {count:5d}  {reason}")
    print()

    print("## 4. HOT sessions with thin validated coverage (undermapped scan targets)")
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    regimes = _compute_session_regimes(con)
    gaps = _hot_session_coverage(con, regimes)
    if not gaps:
        print("  (no HOT sessions or all well-covered)")
    else:
        for g in gaps[:10]:
            print(
                f"  {g['instrument']:4s} {g['session']:20s}  validated_n={g['validated_n']:4d}  strong={g['strong']:3d}  regime_avg_r={g['avg_r']:+.4f}"
            )
    con.close()
    print()

    print("## 5. Suggested next moves")
    if displaced:
        print(f"  - {len(displaced)} displaced lanes -> consider a 2nd correlation-orthogonal profile")
    chordia_missing = sum(c for r, c in _paused_histogram(paused) if "chordia" in r.lower() and "missing" in r.lower())
    if chordia_missing >= 50:
        print(f"  - {chordia_missing} Chordia-MISSING lanes -> batch audit pass would unlock inventory")
    if stale is not None and stale >= 7:
        print(f"  - Allocation is {stale} days old -> rebalance refresh recommended")
    undermapped = [g for g in gaps if g["validated_n"] <= 3]
    if undermapped:
        names = ", ".join(f"{g['instrument']} {g['session']}" for g in undermapped[:3])
        print(f"  - Undermapped HOT sessions -> directed scan candidates: {names}")


if __name__ == "__main__":
    main()
