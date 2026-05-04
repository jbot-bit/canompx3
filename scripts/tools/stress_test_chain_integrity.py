#!/usr/bin/env python3
"""Chain Integrity Stress Test v2 — finds siblings of the VOL trade population bug.

Tests the source-of-truth chain:
  daily_features -> discovery -> paper_trader -> execution_engine

for asymmetries in row dict construction, filter evaluation, cost model
arithmetic, and outcome replay.

Bug vectors targeted:
  V1: rel_vol dual computation divergence (build_daily_features vs discovery)
  V2: orb_minutes hardcode in paper_trader (always loads orb_minutes=5)
  V3: stop_multiplier chain (apply_tight_stop vs execution_engine stop adjust)
  V4: cross_atr three-code-path injection asymmetry
  V5: Wrong-but-non-NULL values passing fail-closed checks silently

Usage:
    python scripts/tools/stress_test_chain_integrity.py
    python scripts/tools/stress_test_chain_integrity.py --quick   # skip T3,T5
"""

import argparse
import inspect
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.cost_model import get_cost_spec
from pipeline.dst import SESSION_CATALOG
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import (
    ALL_FILTERS,
    CrossAssetATRFilter,
    OrbSizeFilter,
    VolumeFilter,
    apply_tight_stop,
)

UTC = ZoneInfo("UTC")

LOOKAHEAD_COLUMNS = {
    "double_break",
    "outcome",
    "mae_r",
    "mfe_r",
    "daily_high",
    "daily_low",
    "daily_close",
    "day_type",
}


def connect_ro():
    return duckdb.connect(str(GOLD_DB_PATH), read_only=True)


def print_section(title):
    print(f"\n{'_' * 60}")
    print(f"  {title}")
    print(f"{'_' * 60}")


def load_validated_strategies(con):
    rows = con.execute("""
        SELECT strategy_id, instrument, orb_label, orb_minutes,
               entry_model, rr_target, confirm_bars, filter_type,
               sample_size, win_rate, expectancy_r, status
        FROM validated_setups
        WHERE status NOT IN ('PURGED', 'RETIRED')
        ORDER BY instrument, orb_label, filter_type
    """).fetchall()
    cols = [
        "strategy_id",
        "instrument",
        "orb_label",
        "orb_minutes",
        "entry_model",
        "rr_target",
        "confirm_bars",
        "filter_type",
        "sample_size",
        "win_rate",
        "expectancy_r",
        "status",
    ]
    return [dict(zip(cols, r, strict=False)) for r in rows]


def load_daily_features(con, instrument, orb_minutes):
    rows = con.execute(
        """
        SELECT * FROM daily_features
        WHERE symbol = ? AND orb_minutes = ?
        ORDER BY trading_day
    """,
        [instrument, orb_minutes],
    ).fetchall()
    cols = [desc[0] for desc in con.description]
    return [dict(zip(cols, r, strict=False)) for r in rows]


def inject_cross_asset_atrs(con, features, instrument):
    cross_sources = {f.source_instrument for f in ALL_FILTERS.values() if isinstance(f, CrossAssetATRFilter)}
    source_atrs = {}
    for source in cross_sources:
        if source == instrument:
            continue
        rows = con.execute(
            """
            SELECT trading_day, atr_20_pct FROM daily_features
            WHERE symbol = ? AND orb_minutes = 5 AND atr_20_pct IS NOT NULL
        """,
            [source],
        ).fetchall()
        source_atrs[source] = {r[0]: float(r[1]) for r in rows}

    for row in features:
        td = row["trading_day"]
        for source, atr_map in source_atrs.items():
            val = atr_map.get(td)
            if val is not None:
                row[f"cross_atr_{source}_pct"] = val


# ================================================================
# T0: DB STATE SNAPSHOT
# ================================================================


def test_t0_db_snapshot(con):
    print_section("T0: DB STATE SNAPSHOT")
    findings = []
    tables = [
        "bars_1m",
        "bars_5m",
        "daily_features",
        "orb_outcomes",
        "experimental_strategies",
        "validated_setups",
        "edge_families",
    ]
    counts = {}
    for t in tables:
        try:
            counts[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except Exception as e:
            counts[t] = f"ERROR: {e}"
            findings.append(f"CRITICAL: Table {t} not accessible: {e}")

    print("  Table counts:")
    for t, n in counts.items():
        print(f"    {t:30s} {n:>12}")

    print("\n  Freshness:")
    for table in ["daily_features", "orb_outcomes"]:
        try:
            for sym, latest, days in con.execute(f"""
                SELECT symbol, MAX(trading_day), COUNT(DISTINCT trading_day)
                FROM {table} GROUP BY symbol ORDER BY symbol
            """).fetchall():
                print(f"    {table:25s} {sym:5s} latest={latest}  days={days}")
        except Exception:
            pass

    try:
        print("\n  Validated by status:")
        for status, n in con.execute("""
            SELECT status, COUNT(*) FROM validated_setups GROUP BY status ORDER BY status
        """).fetchall():
            print(f"    {status:20s} {n:>6}")
    except Exception:
        findings.append("CRITICAL: Cannot query validated_setups")

    verdict = "FAIL" if any("CRITICAL" in f for f in findings) else "PASS"
    print(f"\n  T0 VERDICT: {verdict}")
    return verdict, findings, counts


# ================================================================
# T1: ROW DICT FORENSICS
# ================================================================


def test_t1_row_dict_forensics(con, sample_size=30):
    print_section("T1: ROW DICT FORENSICS")
    findings = []

    sample_rows = con.execute(
        """
        SELECT d5.symbol, d5.trading_day
        FROM daily_features d5
        JOIN daily_features d30 ON d5.trading_day = d30.trading_day
          AND d5.symbol = d30.symbol
        WHERE d5.orb_minutes = 5 AND d30.orb_minutes = 30
        ORDER BY RANDOM() LIMIT ?
    """,
        [sample_size],
    ).fetchall()

    if not sample_rows:
        print("  No (5m, 30m) pairs found.")
        return "SKIP", ["No data"], {}

    key_diffs = defaultdict(int)
    total_checked = 0

    for instrument, td in sample_rows:
        row_5m = con.execute(
            """
            SELECT * FROM daily_features
            WHERE symbol = ? AND orb_minutes = 5 AND trading_day = ?
        """,
            [instrument, td],
        ).fetchone()
        row_30m = con.execute(
            """
            SELECT * FROM daily_features
            WHERE symbol = ? AND orb_minutes = 30 AND trading_day = ?
        """,
            [instrument, td],
        ).fetchone()
        if row_5m is None or row_30m is None:
            continue

        cols = [desc[0] for desc in con.description]
        dict_5m = dict(zip(cols, row_5m, strict=False))
        dict_30m = dict(zip(cols, row_30m, strict=False))
        total_checked += 1

        for col in cols:
            if col == "orb_minutes":
                continue
            v5, v30 = dict_5m.get(col), dict_30m.get(col)
            if v5 != v30 and not (v5 is None and v30 is None):
                key_diffs[col] += 1

    non_orb = {k: v for k, v in key_diffs.items() if not k.startswith("orb_") and not k.startswith("rel_vol_")}
    orb = {k: v for k, v in key_diffs.items() if k.startswith("orb_") or k.startswith("rel_vol_")}

    print(f"  Compared {total_checked} days: orb_minutes=5 vs 30")

    if non_orb:
        findings.append(f"WARN: {len(non_orb)} non-ORB columns differ between 5m/30m")
        print("\n  NON-ORB DIFFS (unexpected):")
        for col, cnt in sorted(non_orb.items(), key=lambda x: -x[1]):
            print(f"    {col:45s} {cnt}/{total_checked} days")

    print(f"\n  ORB columns that differ (expected): {len(orb)}")
    for col, cnt in sorted(orb.items(), key=lambda x: -x[1])[:10]:
        print(f"    {col:45s} {cnt}/{total_checked} days")

    if orb:
        findings.append(f"V2 EXPOSURE: {len(orb)} ORB columns differ 5m vs 30m. Paper trader always loads 5m.")

    non5m = con.execute("""
        SELECT orb_minutes, COUNT(*), GROUP_CONCAT(DISTINCT filter_type)
        FROM validated_setups
        WHERE status NOT IN ('PURGED', 'RETIRED') AND orb_minutes != 5
        GROUP BY orb_minutes
    """).fetchall()
    if non5m:
        for om, n, filters in non5m:
            findings.append(f"RED FLAG: {n} strategies with orb_minutes={om}")
            print(f"\n  RED FLAG: {n} validated at orb_minutes={om} (filters: {filters})")
    else:
        print("\n  All validated use orb_minutes=5 -- V2 not exposed.")

    cross_sources = {f.source_instrument for f in ALL_FILTERS.values() if isinstance(f, CrossAssetATRFilter)}
    if cross_sources:
        schema_cols = {r[0] for r in con.execute("DESCRIBE daily_features").fetchall()}
        print(f"\n  Cross-ATR ({cross_sources}):")
        for src in cross_sources:
            key = f"cross_atr_{src}_pct"
            print(f"    {key}: in_schema={key in schema_cols}")
            if key not in schema_cols:
                findings.append(f"V4: {key} not in schema -- injected at runtime")

    verdict = "FAIL" if any("RED FLAG" in f for f in findings) else ("WARN" if findings else "PASS")
    print(f"\n  T1 VERDICT: {verdict}")
    return verdict, findings, {"checked": total_checked, "orb_diff_cols": len(orb)}


# ================================================================
# T2: FILTER POPULATION MATCH
# ================================================================


def test_t2_filter_population_match(con):
    print_section("T2: FILTER POPULATION MATCH")
    findings = []

    strategies = load_validated_strategies(con)
    if not strategies:
        return "SKIP", ["No strategies"], {}

    print(f"  Testing {len(strategies)} validated strategies...")

    groups = defaultdict(list)
    for s in strategies:
        groups[(s["instrument"], s["orb_minutes"])].append(s)

    total_checked = 0
    total_match = 0
    total_mismatch = 0
    mismatch_details = []

    for (instrument, orb_minutes), group_strats in sorted(groups.items()):
        disc_features = load_daily_features(con, instrument, orb_minutes)
        inject_cross_asset_atrs(con, disc_features, instrument)

        if orb_minutes == 5:
            pt_features = disc_features
        else:
            pt_features = load_daily_features(con, instrument, 5)
            inject_cross_asset_atrs(con, pt_features, instrument)

        filter_groups = defaultdict(list)
        for s in group_strats:
            filter_groups[(s["filter_type"], s["orb_label"])].append(s)

        for (filter_type, orb_label), fg_strats in filter_groups.items():
            filt = ALL_FILTERS.get(filter_type)
            if filt is None:
                findings.append(f"ERROR: '{filter_type}' not in ALL_FILTERS")
                continue

            disc_days = {
                r["trading_day"]
                for r in disc_features
                if r.get(f"orb_{orb_label}_break_dir") is not None and filt.matches_row(r, orb_label)
            }

            pt_days = {
                r["trading_day"]
                for r in pt_features
                if r.get(f"orb_{orb_label}_break_dir") is not None and filt.matches_row(r, orb_label)
            }

            for s in fg_strats:
                total_checked += 1
                if disc_days == pt_days:
                    total_match += 1
                else:
                    total_mismatch += 1
                    delta = len(disc_days) - len(pt_days)
                    pct = (delta / len(disc_days) * 100) if disc_days else 0
                    if len(mismatch_details) < 50:
                        mismatch_details.append(
                            {
                                "id": s["strategy_id"],
                                "ft": filter_type,
                                "om": orb_minutes,
                                "nd": len(disc_days),
                                "np": len(pt_days),
                                "d": delta,
                                "pct": pct,
                                "sample": sorted(disc_days - pt_days)[:3],
                            }
                        )

    print(f"\n  Checked: {total_checked}  MATCH: {total_match}  MISMATCH: {total_mismatch}")

    if mismatch_details:
        print(f"\n  {'Strategy':50s} {'Filter':20s} {'OM':>3} {'Disc':>5} {'PT':>5} {'D':>5} {'%':>7}")
        for m in sorted(mismatch_details, key=lambda x: -abs(x["d"]))[:20]:
            print(
                f"  {m['id']:50s} {m['ft']:20s} {m['om']:>3} {m['nd']:>5} {m['np']:>5} {m['d']:>+5} {m['pct']:>+6.1f}%"
            )

        findings.append(f"RED FLAG: {total_mismatch}/{total_checked} mismatches")

        by_ft = defaultdict(int)
        for m in mismatch_details:
            by_ft[m["ft"]] += 1
        print("\n  By filter_type:")
        for ft, n in sorted(by_ft.items(), key=lambda x: -x[1]):
            print(f"    {ft:25s} {n:>5}")

    verdict = "FAIL" if total_mismatch > 0 else "PASS"
    print(f"\n  T2 VERDICT: {verdict}")
    return verdict, findings, {"checked": total_checked, "match": total_match, "mismatch": total_mismatch}


# ================================================================
# T3: REL_VOL CONCORDANCE
# ================================================================


def test_t3_rel_vol_concordance(con, per_group=5):
    print_section("T3: REL_VOL DB vs RECOMPUTED")
    findings = []
    tested = matched = mismatched = 0
    details = []

    for instrument in ACTIVE_ORB_INSTRUMENTS:
        for label in list(SESSION_CATALOG.keys())[:4]:
            rel_col = f"rel_vol_{label}"
            bts_col = f"orb_{label}_break_ts"
            bvol_col = f"orb_{label}_break_bar_volume"

            rows = con.execute(
                f"""
                SELECT trading_day, {rel_col}, {bts_col}, {bvol_col}
                FROM daily_features
                WHERE symbol = ? AND orb_minutes = 5
                  AND {rel_col} IS NOT NULL AND {bts_col} IS NOT NULL
                  AND {bvol_col} IS NOT NULL AND {bvol_col} > 0
                ORDER BY RANDOM() LIMIT ?
            """,
                [instrument, per_group],
            ).fetchall()

            for td, stored, bts, bvol in rows:
                try:
                    utc_ts = bts.astimezone(UTC) if hasattr(bts, "astimezone") else bts
                except Exception:
                    continue

                hist = con.execute(
                    """
                    SELECT volume FROM bars_1m
                    WHERE symbol = ?
                      AND EXTRACT(HOUR FROM (ts_utc AT TIME ZONE 'UTC')) = ?
                      AND EXTRACT(MINUTE FROM (ts_utc AT TIME ZONE 'UTC')) = ?
                      AND ts_utc < ?
                    ORDER BY ts_utc DESC LIMIT 20
                """,
                    [instrument, utc_ts.hour, utc_ts.minute, bts],
                ).fetchall()

                prior = [v for (v,) in hist if v and v > 0]
                if len(prior) < 5:
                    continue
                baseline = statistics.median(prior)
                if baseline <= 0:
                    continue

                computed = bvol / baseline
                tested += 1
                if abs(computed - stored) < 0.01:
                    matched += 1
                else:
                    mismatched += 1
                    if len(details) < 10:
                        details.append(f"{instrument} {td} {label}: stored={stored:.4f} computed={computed:.4f}")

    print(f"  Tested: {tested}  Matched: {matched}  Mismatched: {mismatched}")
    for d in details:
        print(f"    {d}")

    if mismatched > 0:
        findings.append(f"V1: {mismatched}/{tested} rel_vol diverge")

    # NULL coverage
    print("\n  NULL rel_vol on break-days (CME_REOPEN):")
    for inst in ACTIVE_ORB_INSTRUMENTS:
        null_n = con.execute(
            """
            SELECT COUNT(*) FROM daily_features
            WHERE symbol = ? AND orb_minutes = 5
              AND orb_CME_REOPEN_break_dir IS NOT NULL AND rel_vol_CME_REOPEN IS NULL
        """,
            [inst],
        ).fetchone()[0]
        total_n = con.execute(
            """
            SELECT COUNT(*) FROM daily_features
            WHERE symbol = ? AND orb_minutes = 5 AND orb_CME_REOPEN_break_dir IS NOT NULL
        """,
            [inst],
        ).fetchone()[0]
        print(f"    {inst}: {null_n}/{total_n}")
        if null_n > 0:
            findings.append(f"WARN: {inst} {null_n}/{total_n} NULL rel_vol on break-days")

    verdict = "FAIL" if mismatched > tested * 0.05 else ("WARN" if mismatched > 0 or findings else "PASS")
    print(f"\n  T3 VERDICT: {verdict}")
    return verdict, findings, {"tested": tested, "matched": matched, "mismatched": mismatched}


# ================================================================
# T4: ORB_MINUTES CROSS-CONTAMINATION
# ================================================================


def test_t4_orb_minutes_contamination(con):
    print_section("T4: ORB_MINUTES CROSS-CONTAMINATION")
    findings = []

    non5m = con.execute("""
        SELECT strategy_id, instrument, orb_label, orb_minutes, filter_type,
               sample_size, expectancy_r
        FROM validated_setups
        WHERE status NOT IN ('PURGED', 'RETIRED') AND orb_minutes != 5
    """).fetchall()

    if not non5m:
        print("  No validated strategies with orb_minutes != 5. V2 clean.")
        return "PASS", [], {"exposed": 0}

    cols = ["strategy_id", "instrument", "orb_label", "orb_minutes", "filter_type", "sample_size", "expectancy_r"]
    strats = [dict(zip(cols, r, strict=False)) for r in non5m]
    print(f"  EXPOSED: {len(strats)} strategies\n")
    for s in strats[:15]:
        print(
            f"  {s['strategy_id']:50s} om={s['orb_minutes']} {s['filter_type']:20s} "
            f"N={s['sample_size']} ExpR={s['expectancy_r']:+.4f}"
        )

    findings.append(f"V2: {len(strats)} strategies use orb_minutes!=5")
    print("\n  T4 VERDICT: FAIL")
    return "FAIL", findings, {"exposed": len(strats)}


# ================================================================
# T5: STOP_MULTIPLIER CHAIN
# ================================================================


def test_t5_stop_multiplier_chain(con, sample_size=20):
    print_section("T5: STOP_MULTIPLIER CHAIN REPLAY")
    findings = []

    s075 = con.execute(
        """
        SELECT strategy_id, instrument, orb_label, orb_minutes,
               entry_model, rr_target, confirm_bars
        FROM validated_setups
        WHERE status NOT IN ('PURGED', 'RETIRED') AND strategy_id LIKE '%S075%'
        ORDER BY RANDOM() LIMIT ?
    """,
        [sample_size],
    ).fetchall()

    if not s075:
        print("  No S0.75 strategies.")
        return "PASS", [], {"tested": 0}

    cols = ["strategy_id", "instrument", "orb_label", "orb_minutes", "entry_model", "rr_target", "confirm_bars"]
    strats = [dict(zip(cols, r, strict=False)) for r in s075]
    print(f"  Testing {len(strats)} S0.75 strategies...")

    tested = consistent = inconsistent = 0
    details = []

    for s in strats:
        cs = get_cost_spec(s["instrument"])
        outcomes = con.execute(
            """
            SELECT entry_price, stop_price, outcome, pnl_r, mae_r
            FROM orb_outcomes
            WHERE symbol = ? AND orb_label = ? AND orb_minutes = ?
              AND entry_model = ? AND rr_target = ? AND confirm_bars = ?
              AND entry_price IS NOT NULL AND mae_r IS NOT NULL
            ORDER BY RANDOM() LIMIT 30
        """,
            [s["instrument"], s["orb_label"], s["orb_minutes"], s["entry_model"], s["rr_target"], s["confirm_bars"]],
        ).fetchall()

        if not outcomes:
            continue

        o_dicts = [
            {"entry_price": e, "stop_price": st, "outcome": o, "pnl_r": p, "mae_r": m} for e, st, o, p, m in outcomes
        ]
        tight = apply_tight_stop(o_dicts, 0.75, cs)

        for orig, t in zip(o_dicts, tight, strict=False):
            tested += 1
            entry, stop, mae_r = orig["entry_price"], orig["stop_price"], orig["mae_r"]
            if not all([entry, stop, mae_r]):
                consistent += 1
                continue

            risk_pts = abs(entry - stop)
            if risk_pts <= 0:
                consistent += 1
                continue

            disc_killed = t["pnl_r"] != orig["pnl_r"]

            # Engine logic
            raw_risk_d = risk_pts * cs.point_value
            risk_d = raw_risk_d + cs.total_friction
            max_adv = mae_r * risk_d / cs.point_value
            eng_killed = max_adv >= 0.75 * risk_pts

            if disc_killed != eng_killed:
                inconsistent += 1
                if len(details) < 5:
                    details.append(f"{s['strategy_id'][:40]}: disc={disc_killed} eng={eng_killed}")
            else:
                consistent += 1

    print(f"  Tested: {tested}  Consistent: {consistent}  Inconsistent: {inconsistent}")
    for d in details:
        print(f"    {d}")

    if inconsistent > 0:
        findings.append(f"V3: {inconsistent}/{tested} divergences")

    verdict = "FAIL" if inconsistent > tested * 0.05 else ("WARN" if inconsistent > 0 else "PASS")
    print(f"\n  T5 VERDICT: {verdict}")
    return verdict, findings, {"tested": tested, "inconsistent": inconsistent}


# ================================================================
# T6: LOOKAHEAD STATIC AUDIT
# ================================================================


def test_t6_lookahead_audit():
    print_section("T6: LOOKAHEAD STATIC AUDIT")
    findings = []
    seen = set()
    total = clean = 0
    flagged = []

    for filt in ALL_FILTERS.values():
        cls = type(filt)
        if cls in seen:
            continue
        seen.add(cls)
        total += 1

        try:
            source = inspect.getsource(cls.matches_row)
        except (TypeError, OSError):
            continue

        accessed = set()
        for m in re.finditer(r'row(?:\.get)?\s*\(\s*f?["\']([^"\'{}]+)', source):
            accessed.add(m.group(1))
        for m in re.finditer(r'row\[f"orb_\{[^}]+\}_(\w+)"', source):
            accessed.add(m.group(1))
        for m in re.finditer(r'row\.get\(f"orb_\{[^}]+\}_(\w+)"', source):
            accessed.add(m.group(1))

        for key in accessed:
            for banned in LOOKAHEAD_COLUMNS:
                if banned in key:
                    flagged.append({"cls": cls.__name__, "key": key, "banned": banned})

        if not any(cls.__name__ == f["cls"] for f in flagged):
            clean += 1

    print(f"  Inspected: {total}  Clean: {clean}  Flagged: {total - clean}")
    for f in flagged:
        print(f"    {f['cls']:30s} '{f['key']}' (banned: '{f['banned']}')")
        findings.append(f"LOOKAHEAD: {f['cls']} accesses {f['key']}")

    # double_break specific
    for cls in seen:
        try:
            if "double_break" in inspect.getsource(cls):
                findings.append(f"BANNED: {cls.__name__} uses double_break")
                print(f"\n  BANNED: {cls.__name__} uses double_break")
        except (TypeError, OSError):
            pass

    verdict = "FAIL" if any("BANNED" in f for f in findings) else ("WARN" if flagged else "PASS")
    print(f"\n  T6 VERDICT: {verdict}")
    return verdict, findings, {"inspected": total, "flagged": len(flagged)}


# ================================================================
# T7: WRONG-VALUE INJECTION
# ================================================================


def test_t7_wrong_value_injection(con):
    print_section("T7: WRONG-VALUE INJECTION")
    findings = []

    # Test OrbSizeFilter with wrong orb_minutes
    size_filters = {k: f for k, f in ALL_FILTERS.items() if isinstance(f, OrbSizeFilter)}
    if size_filters:
        fkey, filt = next(iter(size_filters.items()))
        for inst in ACTIVE_ORB_INSTRUMENTS:
            for label in list(SESSION_CATALOG.keys())[:3]:
                sc = f"orb_{label}_size"
                dc = f"orb_{label}_break_dir"
                try:
                    r = con.execute(
                        f"""
                        SELECT d5.trading_day, d5.{sc}, d30.{sc}
                        FROM daily_features d5
                        JOIN daily_features d30
                          ON d5.trading_day = d30.trading_day AND d5.symbol = d30.symbol
                        WHERE d5.symbol = ? AND d5.orb_minutes = 5 AND d30.orb_minutes = 30
                          AND d5.{sc} IS NOT NULL AND d30.{sc} IS NOT NULL
                          AND d5.{dc} IS NOT NULL AND ABS(d5.{sc} - d30.{sc}) > 0.5
                        LIMIT 1
                    """,
                        [inst],
                    ).fetchone()
                except Exception:
                    continue
                if r:
                    td, s5, s30 = r
                    r5 = con.execute(
                        "SELECT * FROM daily_features WHERE symbol=? AND orb_minutes=5 AND trading_day=?", [inst, td]
                    ).fetchone()
                    r30 = con.execute(
                        "SELECT * FROM daily_features WHERE symbol=? AND orb_minutes=30 AND trading_day=?", [inst, td]
                    ).fetchone()
                    c = [d[0] for d in con.description]
                    d5 = dict(zip(c, r5, strict=False))
                    d30 = dict(zip(c, r30, strict=False))
                    res5 = filt.matches_row(d5, label)
                    res30 = filt.matches_row(d30, label)
                    print(f"  Size injection ({inst} {td} {label}):")
                    print(f"    5m={s5:.2f}->{res5}  30m={s30:.2f}->{res30}")
                    if res5 != res30:
                        findings.append("V5: OrbSizeFilter differs for 5m vs 30m")
                    break
            if findings:
                break

    # Test VolumeFilter fail-closed
    vol_filters = {k: f for k, f in ALL_FILTERS.items() if isinstance(f, VolumeFilter)}
    if vol_filters:
        fkey, filt = next(iter(vol_filters.items()))
        for inst in ACTIVE_ORB_INSTRUMENTS:
            for label in list(SESSION_CATALOG.keys())[:3]:
                try:
                    r = con.execute(
                        f"""
                        SELECT * FROM daily_features
                        WHERE symbol = ? AND orb_minutes = 5
                          AND orb_{label}_break_dir IS NOT NULL
                        LIMIT 1
                    """,
                        [inst],
                    ).fetchone()
                except Exception:
                    continue
                if r:
                    c = [d[0] for d in con.description]
                    rd = dict(zip(c, r, strict=False))
                    rv_key = f"rel_vol_{label}"

                    res_with = filt.matches_row(rd, label)
                    rd.pop(rv_key, None)
                    res_without = filt.matches_row(rd, label)

                    print(f"\n  VolumeFilter fail-closed ({inst} {label} {fkey}):")
                    print(f"    With rel_vol: {res_with}")
                    print(f"    Without key:  {res_without}")
                    if res_without is True:
                        findings.append("CRITICAL: VolumeFilter passes without rel_vol!")
                    else:
                        print("    Fail-closed confirmed")

                    rd[rv_key] = None
                    res_null = filt.matches_row(rd, label)
                    print(f"    With None:    {res_null}")
                    if res_null is True:
                        findings.append("CRITICAL: VolumeFilter passes with None rel_vol!")
                    break
            break

    verdict = "FAIL" if any("CRITICAL" in f for f in findings) else ("WARN" if findings else "PASS")
    print(f"\n  T7 VERDICT: {verdict}")
    return verdict, findings, {}


# ================================================================
# T8: COST MODEL ARITHMETIC
# ================================================================


def test_t8_cost_model(con):
    print_section("T8: COST MODEL ARITHMETIC")
    findings = []
    tested = matched = mismatched = 0

    for inst in ACTIVE_ORB_INSTRUMENTS:
        cs = get_cost_spec(inst)
        rows = con.execute(
            """
            SELECT trading_day, orb_label, entry_price, stop_price, exit_price,
                   outcome, pnl_r, entry_model
            FROM orb_outcomes
            WHERE symbol = ? AND entry_price IS NOT NULL
              AND stop_price IS NOT NULL AND exit_price IS NOT NULL
              AND pnl_r IS NOT NULL AND outcome IN ('win', 'loss')
            ORDER BY RANDOM() LIMIT 10
        """,
            [inst],
        ).fetchall()

        for td, ol, entry, stop, exit_px, _outcome, stored, em in rows:
            tested += 1
            direction = 1.0 if entry > stop else -1.0
            pnl_d = (exit_px - entry) * direction * cs.point_value - cs.total_friction
            risk_d = abs(entry - stop) * cs.point_value + cs.total_friction
            if risk_d <= 0:
                continue
            computed = pnl_d / risk_d
            diff = abs(computed - stored)
            if diff > 0.01:
                mismatched += 1
                if mismatched <= 5:
                    print(f"  MISMATCH: {inst} {td} {ol} {em}")
                    print(f"    stored={stored:.4f} computed={computed:.4f} D={diff:.4f}")
            else:
                matched += 1

    print(f"\n  Tested: {tested}  Matched: {matched}  Mismatched: {mismatched}")
    if mismatched > 0:
        findings.append(f"COST: {mismatched}/{tested} pnl_r differ >0.01R")

    verdict = "FAIL" if mismatched > tested * 0.1 else ("WARN" if mismatched > 0 else "PASS")
    print(f"\n  T8 VERDICT: {verdict}")
    return verdict, findings, {"tested": tested, "matched": matched, "mismatched": mismatched}


# ================================================================
# T9: STATISTICAL REVALIDATION
# ================================================================


def test_t9_statistical_revalidation(con):
    print_section("T9: STATISTICAL SPOT-CHECK")
    findings = []

    rows = con.execute("""
        SELECT strategy_id, p_value, fdr_adjusted_p, fdr_significant,
               discovery_k, sample_size, wfe, wfe_verdict,
               sharpe_ratio, expectancy_r
        FROM validated_setups
        WHERE status NOT IN ('PURGED', 'RETIRED') AND p_value IS NOT NULL
        ORDER BY RANDOM() LIMIT 20
    """).fetchall()

    if not rows:
        return "SKIP", [], {}

    cols = [
        "strategy_id",
        "p_value",
        "fdr_adjusted_p",
        "fdr_significant",
        "discovery_k",
        "sample_size",
        "wfe",
        "wfe_verdict",
        "sharpe_ratio",
        "expectancy_r",
    ]
    strats = [dict(zip(cols, r, strict=False)) for r in rows]

    tested = consistent = inconsistent = 0
    for s in strats:
        tested += 1
        issues = []

        if s["fdr_adjusted_p"] is not None and s["fdr_significant"] is not None:
            if (s["fdr_adjusted_p"] <= 0.05) != s["fdr_significant"]:
                issues.append(f"FDR: adj_p={s['fdr_adjusted_p']:.4f} sig={s['fdr_significant']}")

        if s["p_value"] is not None and s["fdr_adjusted_p"] is not None:
            if s["fdr_adjusted_p"] < s["p_value"] - 0.001:
                issues.append("BH: adj_p < raw_p")

        if s["sample_size"] is not None and s["sample_size"] < 30:
            issues.append(f"N={s['sample_size']}<30")

        if s["wfe"] is not None and s["wfe"] > 2.0:
            issues.append(f"WFE={s['wfe']:.2f}>2.0")

        if (
            s["sharpe_ratio"] is not None
            and s["expectancy_r"] is not None
            and s["sharpe_ratio"] * s["expectancy_r"] < 0
        ):
            issues.append("Sharpe/ExpR sign mismatch")

        if issues:
            inconsistent += 1
            if inconsistent <= 10:
                print(f"  {s['strategy_id'][:50]}:")
                for i in issues:
                    print(f"    ! {i}")
            findings.extend(issues)
        else:
            consistent += 1

    print(f"\n  Tested: {tested}  Consistent: {consistent}  Inconsistent: {inconsistent}")
    verdict = "FAIL" if inconsistent > tested * 0.2 else ("WARN" if inconsistent > 0 else "PASS")
    print(f"\n  T9 VERDICT: {verdict}")
    return verdict, findings, {"tested": tested, "inconsistent": inconsistent}


# ================================================================
# REPORT
# ================================================================


def print_report(results):
    print("\n")
    print("=" * 60)
    print("  CHAIN INTEGRITY STRESS TEST REPORT")
    print("=" * 60)
    print(f"  TIMESTAMP:    {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  DB:           {GOLD_DB_PATH}")
    print(f"  INSTRUMENTS:  {', '.join(ACTIVE_ORB_INSTRUMENTS)}")

    print(f"\n  {'Test':45s} {'Verdict':>8}")
    print(f"  {'=' * 55}")

    crit = warn = 0
    for name, (v, _, _) in results.items():
        tag = ""
        if v == "FAIL":
            tag = " << CRITICAL"
            crit += 1
        elif v == "WARN":
            tag = " << WARNING"
            warn += 1
        print(f"  {name:45s} {v:>8}{tag}")

    all_f = []
    for name, (_, findings, _) in results.items():
        for f in findings:
            all_f.append(f"[{name}] {f}")

    if all_f:
        print(f"\n  FINDINGS ({len(all_f)}):")
        print(f"  {'-' * 55}")
        for i, f in enumerate(all_f, 1):
            print(f"  {i:>3}. {f[:100]}")

    print("\n  BUG VECTORS:")
    print(f"  {'-' * 55}")
    for desc, pfx in [
        ("V1 rel_vol dual computation", "T3"),
        ("V2 orb_minutes hardcode", "T4"),
        ("V3 stop_multiplier chain", "T5"),
        ("V4 cross_atr asymmetry", "T1"),
        ("V5 wrong-value passthrough", "T7"),
    ]:
        matched = [v for n, (v, _, _) in results.items() if pfx in n]
        status = {"FAIL": "EXPOSED", "WARN": "SUSPECT", "PASS": "CLEAN", "SKIP": "NOT TESTED"}.get(
            matched[0] if matched else "SKIP", "?"
        )
        print(f"    {desc:35s} {status}")

    overall = "COMPROMISED" if crit > 0 else ("SUSPECT" if warn > 0 else "CLEAN")
    print(f"\n  {'=' * 55}")
    print(f"  OVERALL: {overall}  (critical={crit} warnings={warn})")
    print(f"  {'=' * 55}")
    return overall


def main():
    parser = argparse.ArgumentParser(description="Chain integrity stress test v2")
    parser.add_argument("--quick", action="store_true", help="Skip T3, T5")
    args = parser.parse_args()

    print("=" * 60)
    print("  CHAIN INTEGRITY STRESS TEST v2 -- 5 bug vectors")
    print("=" * 60)

    con = connect_ro()
    R = {}

    try:
        v, f, d = test_t0_db_snapshot(con)
        R["T0: DB Snapshot"] = (v, f, d)
        if v == "FAIL":
            print("\n  *** T0 CRITICAL -- halting ***")
            print_report(R)
            return 1

        v, f, d = test_t1_row_dict_forensics(con)
        R["T1: Row Dict Forensics"] = (v, f, d)

        v, f, d = test_t2_filter_population_match(con)
        R["T2: Filter Population Match"] = (v, f, d)

        if args.quick:
            R["T3: rel_vol Concordance"] = ("SKIP", ["--quick"], {})
        else:
            v, f, d = test_t3_rel_vol_concordance(con)
            R["T3: rel_vol Concordance"] = (v, f, d)

        v, f, d = test_t4_orb_minutes_contamination(con)
        R["T4: orb_minutes Contamination"] = (v, f, d)

        if args.quick:
            R["T5: stop_multiplier Chain"] = ("SKIP", ["--quick"], {})
        else:
            v, f, d = test_t5_stop_multiplier_chain(con)
            R["T5: stop_multiplier Chain"] = (v, f, d)

        v, f, d = test_t6_lookahead_audit()
        R["T6: Lookahead Audit"] = (v, f, d)

        v, f, d = test_t7_wrong_value_injection(con)
        R["T7: Wrong-Value Injection"] = (v, f, d)

        v, f, d = test_t8_cost_model(con)
        R["T8: Cost Model Arithmetic"] = (v, f, d)

        v, f, d = test_t9_statistical_revalidation(con)
        R["T9: Statistical Revalidation"] = (v, f, d)

    finally:
        con.close()

    overall = print_report(R)
    return 0 if overall == "CLEAN" else 1


if __name__ == "__main__":
    sys.exit(main())
