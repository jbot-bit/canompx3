"""
Grid search over strategy variants and save results to experimental_strategies.

For each combination of (orb_label, rr_target, confirm_bars, filter),
queries pre-computed orb_outcomes, computes performance metrics, and
writes results to experimental_strategies.

Usage:
    python trading_app/strategy_discovery.py --instrument MGC --start 2021-01-01 --end 2025-12-31
    python trading_app/strategy_discovery.py --instrument MGC --start 2021-01-01 --end 2025-12-31 --dry-run
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from datetime import date, timezone

from pipeline.log import get_logger
logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import duckdb

from pipeline.paths import GOLD_DB_PATH
from pipeline.init_db import ORB_LABELS
from pipeline.asset_configs import get_enabled_sessions
from pipeline.dst import (
    DST_AFFECTED_SESSIONS, is_winter_for_session, classify_dst_verdict,
)
from trading_app.config import get_filters_for_grid, ENTRY_MODELS, VolumeFilter
from trading_app.db_manager import init_trading_app_schema, compute_trade_day_hash
from trading_app.outcome_builder import RR_TARGETS, CONFIRM_BARS_OPTIONS

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

# Filter specificity ranking: higher = more specific = preferred as canonical
_FILTER_SPECIFICITY = {
    "ORB_G8": 5, "ORB_G6": 4, "ORB_G5_L12": 4, "ORB_G5": 3, "ORB_G4_L12": 3,
    "ORB_G4": 2, "DIR_LONG": 2, "DIR_SHORT": 2,
    "VOL_RV12_N20": 1, "NO_FILTER": 0,
}

_INSERT_SQL = """INSERT OR REPLACE INTO experimental_strategies
    (strategy_id, instrument, orb_label, orb_minutes,
     rr_target, confirm_bars, entry_model,
     filter_type, filter_params,
     sample_size, win_rate, avg_win_r, avg_loss_r,
     expectancy_r, sharpe_ratio, max_drawdown_r,
     median_risk_points, avg_risk_points,
     trades_per_year, sharpe_ann,
     yearly_results,
     entry_signals, scratch_count, early_exit_count,
     trade_day_hash, is_canonical, canonical_strategy_id,
     dst_winter_n, dst_winter_avg_r, dst_summer_n, dst_summer_avg_r, dst_verdict,
     validation_status, validation_notes,
     created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?,
            COALESCE(?, CURRENT_TIMESTAMP))"""

def _mark_canonical(strategies: list[dict]) -> None:
    """Mark canonical vs alias within each dedup group.

    Groups by (instrument, orb_label, entry_model, rr_target, confirm_bars, trade_day_hash).
    Within each group, the strategy with highest filter specificity is canonical;
    ties broken by filter_key alphabetically.
    """
    groups = defaultdict(list)
    for s in strategies:
        key = (s["instrument"], s["orb_label"], s["entry_model"],
               s["rr_target"], s["confirm_bars"], s["trade_day_hash"])
        groups[key].append(s)

    for group in groups.values():
        # Sort by specificity descending, then filter_key for determinism
        group.sort(key=lambda s: (
            -_FILTER_SPECIFICITY.get(s["filter_key"], -1),
            s["filter_key"],
        ))
        head = group[0]
        head["is_canonical"] = True
        head["canonical_strategy_id"] = head["strategy_id"]
        for alias in group[1:]:
            alias["is_canonical"] = False
            alias["canonical_strategy_id"] = head["strategy_id"]

def compute_metrics(outcomes: list[dict]) -> dict:
    """
    Compute performance metrics from a list of outcome rows.

    sample_size = wins + losses ONLY (scratches/early_exits excluded).
    entry_signals = wins + losses + scratches + early_exits (total entries).
    win_rate denominator = wins + losses.

    Returns dict with: sample_size, win_rate, avg_win_r, avg_loss_r,
    expectancy_r, sharpe_ratio, max_drawdown_r, median_risk_points,
    avg_risk_points, yearly_results, entry_signals, scratch_count,
    early_exit_count.
    """
    _empty = {
        "sample_size": 0,
        "win_rate": None,
        "avg_win_r": None,
        "avg_loss_r": None,
        "expectancy_r": None,
        "sharpe_ratio": None,
        "max_drawdown_r": None,
        "median_risk_points": None,
        "avg_risk_points": None,
        "trades_per_year": 0,
        "sharpe_ann": None,
        "yearly_results": "{}",
        "entry_signals": 0,
        "scratch_count": 0,
        "early_exit_count": 0,
    }
    if not outcomes:
        return dict(_empty)

    # Split wins/losses (scratches/early_exits excluded from W/L stats)
    wins = [o for o in outcomes if o["outcome"] == "win"]
    losses = [o for o in outcomes if o["outcome"] == "loss"]
    scratches = [o for o in outcomes if o["outcome"] == "scratch"]
    early_exits = [o for o in outcomes if o["outcome"] == "early_exit"]
    traded = [o for o in outcomes if o["outcome"] in ("win", "loss")]

    n_traded = len(traded)
    entry_signals = len(wins) + len(losses) + len(scratches) + len(early_exits)
    if n_traded == 0:
        result = dict(_empty)
        result["entry_signals"] = entry_signals
        result["scratch_count"] = len(scratches)
        result["early_exit_count"] = len(early_exits)
        return result

    win_rate = len(wins) / n_traded
    loss_rate = 1.0 - win_rate

    avg_win_r = (
        sum(o["pnl_r"] for o in wins) / len(wins) if wins else 0.0
    )
    avg_loss_r = (
        abs(sum(o["pnl_r"] for o in losses) / len(losses)) if losses else 0.0
    )

    # E = (WR * AvgWin_R) - (LR * AvgLoss_R)  [CANONICAL_LOGIC.txt section 4]
    expectancy_r = (win_rate * avg_win_r) - (loss_rate * avg_loss_r)

    # Sharpe ratio: mean(R) / std(R)
    r_values = [o["pnl_r"] for o in traded]
    mean_r = sum(r_values) / len(r_values)
    if len(r_values) > 1:
        variance = sum((r - mean_r) ** 2 for r in r_values) / (len(r_values) - 1)
        std_r = variance ** 0.5
        sharpe_ratio = mean_r / std_r if std_r > 0 else None
    else:
        sharpe_ratio = None

    # Max drawdown in R (cumulative R-equity curve)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for o in traded:
        cumulative += o["pnl_r"]
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)

    # Yearly breakdown
    yearly = {}
    for o in traded:
        year = str(o["trading_day"].year) if hasattr(o["trading_day"], "year") else str(o["trading_day"])[:4]
        if year not in yearly:
            yearly[year] = {"trades": 0, "wins": 0, "total_r": 0.0}
        yearly[year]["trades"] += 1
        if o["outcome"] == "win":
            yearly[year]["wins"] += 1
        yearly[year]["total_r"] += o["pnl_r"]

    # Compute per-year metrics
    for year_data in yearly.values():
        year_data["win_rate"] = round(
            year_data["wins"] / year_data["trades"], 4
        ) if year_data["trades"] > 0 else 0.0
        year_data["avg_r"] = round(
            year_data["total_r"] / year_data["trades"], 4
        ) if year_data["trades"] > 0 else 0.0
        year_data["total_r"] = round(year_data["total_r"], 4)

    # Annualized Sharpe = per_trade_sharpe * sqrt(trades_per_year)
    # Use actual date span (not count of distinct calendar years)
    if traded:
        trading_days = [o["trading_day"] for o in traded]
        min_day = min(trading_days)
        max_day = max(trading_days)
        if hasattr(min_day, "toordinal"):
            span_days = (max_day - min_day).days + 1
        else:
            from datetime import date as _date
            min_day = _date.fromisoformat(str(min_day)[:10])
            max_day = _date.fromisoformat(str(max_day)[:10])
            span_days = (max_day - min_day).days + 1
        years_span = max(span_days / 365.25, 0.25)  # floor at 3 months
    else:
        years_span = 0
    trades_per_year = (n_traded / years_span) if years_span > 0 else 0
    sharpe_ann = (
        sharpe_ratio * (trades_per_year ** 0.5)
        if sharpe_ratio is not None and trades_per_year > 0
        else None
    )

    # Risk stats (from entry_price and stop_price)
    risk_points_list = [
        abs(o["entry_price"] - o["stop_price"])
        for o in traded
        if o.get("entry_price") is not None and o.get("stop_price") is not None
    ]
    if risk_points_list:
        sorted_risks = sorted(risk_points_list)
        mid = len(sorted_risks) // 2
        if len(sorted_risks) % 2 == 0:
            median_risk = (sorted_risks[mid - 1] + sorted_risks[mid]) / 2
        else:
            median_risk = sorted_risks[mid]
        avg_risk = sum(risk_points_list) / len(risk_points_list)
    else:
        median_risk = None
        avg_risk = None

    return {
        "sample_size": n_traded,
        "win_rate": round(win_rate, 4),
        "avg_win_r": round(avg_win_r, 4),
        "avg_loss_r": round(avg_loss_r, 4),
        "expectancy_r": round(expectancy_r, 4),
        "sharpe_ratio": round(sharpe_ratio, 4) if sharpe_ratio is not None else None,
        "max_drawdown_r": round(max_dd, 4),
        "median_risk_points": round(median_risk, 4) if median_risk is not None else None,
        "avg_risk_points": round(avg_risk, 4) if avg_risk is not None else None,
        "trades_per_year": round(trades_per_year, 1),
        "sharpe_ann": round(sharpe_ann, 4) if sharpe_ann is not None else None,
        "yearly_results": json.dumps(yearly),
        "entry_signals": entry_signals,
        "scratch_count": len(scratches),
        "early_exit_count": len(early_exits),
    }

def compute_dst_split_from_outcomes(outcomes: list[dict], orb_label: str) -> dict:
    """Compute winter/summer split metrics from in-memory outcome list.

    Returns dict with: winter_n, winter_avg_r, summer_n, summer_avg_r, verdict.
    Returns verdict='CLEAN' for non-affected sessions.
    """
    if orb_label not in DST_AFFECTED_SESSIONS:
        return {
            "winter_n": None, "winter_avg_r": None,
            "summer_n": None, "summer_avg_r": None,
            "verdict": "CLEAN",
        }

    winter_rs = []
    summer_rs = []

    for o in outcomes:
        if o["outcome"] not in ("win", "loss"):
            continue
        td = o["trading_day"]
        if hasattr(td, 'date'):
            td = td.date()
        elif not isinstance(td, date):
            td = date.fromisoformat(str(td)[:10])

        is_w = is_winter_for_session(td, orb_label)
        if is_w is None:
            continue
        if is_w:
            winter_rs.append(o["pnl_r"])
        else:
            summer_rs.append(o["pnl_r"])

    winter_n = len(winter_rs)
    summer_n = len(summer_rs)
    winter_avg_r = sum(winter_rs) / winter_n if winter_n > 0 else None
    summer_avg_r = sum(summer_rs) / summer_n if summer_n > 0 else None

    verdict = classify_dst_verdict(winter_avg_r, summer_avg_r, winter_n, summer_n)

    return {
        "winter_n": winter_n,
        "winter_avg_r": round(winter_avg_r, 4) if winter_avg_r is not None else None,
        "summer_n": summer_n,
        "summer_avg_r": round(summer_avg_r, 4) if summer_avg_r is not None else None,
        "verdict": verdict,
    }


def make_strategy_id(
    instrument: str,
    orb_label: str,
    entry_model: str,
    rr_target: float,
    confirm_bars: int,
    filter_type: str,
) -> str:
    """Generate deterministic strategy ID.

    Format: {instrument}_{orb_label}_{entry_model}_RR{rr}_CB{cb}_{filter_type}
    Example: MGC_0900_E1_RR2.5_CB2_ORB_G4

    Components:
      instrument  - Trading instrument (MGC = Micro Gold Futures)
      orb_label   - ORB session time in Brisbane local (0900, 1000, 1800, etc.)
      entry_model - E1 (next bar open), E3 (limit retrace)
      RR          - Risk/Reward target (1.0 to 4.0)
      CB          - Confirm bars required (1 to 5)
      filter_type - ORB size filter (NO_FILTER, ORB_G4, ORB_L3, etc.)
    """
    return f"{instrument}_{orb_label}_{entry_model}_RR{rr_target}_CB{confirm_bars}_{filter_type}"

def _load_daily_features(con, instrument, orb_minutes, start_date, end_date):
    """Load all daily_features rows once into a list of dicts."""
    params = [instrument, orb_minutes]
    where = ["symbol = ?", "orb_minutes = ?"]
    if start_date:
        where.append("trading_day >= ?")
        params.append(start_date)
    if end_date:
        where.append("trading_day <= ?")
        params.append(end_date)

    rows = con.execute(
        f"SELECT * FROM daily_features WHERE {' AND '.join(where)} ORDER BY trading_day",
        params,
    ).fetchall()
    cols = [desc[0] for desc in con.description]
    return [dict(zip(cols, r)) for r in rows]

def _build_filter_day_sets(features, orb_labels, all_filters):
    """Pre-compute matching day sets for every (filter, orb) combo.

    Global exclusion: double-break days (both ORB high and low breached)
    are removed before any filter is applied.  Research shows avg pnl_r
    = -0.570 on double-break days vs +0.068 single-break (MGC 0900, N=18K).
    """
    result = {}
    for filter_key, strategy_filter in all_filters.items():
        for orb_label in orb_labels:
            days = set()
            for row in features:
                if row.get(f"orb_{orb_label}_break_dir") is None:
                    continue
                if row.get(f"orb_{orb_label}_double_break"):
                    continue
                if strategy_filter.matches_row(row, orb_label):
                    days.add(row["trading_day"])
            result[(filter_key, orb_label)] = days
    return result

def _ts_minute_key(ts):
    """Normalize a timestamp to UTC (year, month, day, hour, minute) tuple.

    DuckDB returns TIMESTAMPTZ as local timezone (e.g., Brisbane AEST+10),
    while Python datetime may use timezone.utc. Normalize to UTC for
    consistent comparison.
    """
    utc_ts = ts.astimezone(timezone.utc) if ts.tzinfo is not None else ts
    return (utc_ts.year, utc_ts.month, utc_ts.day, utc_ts.hour, utc_ts.minute)

def _compute_relative_volumes(con, features, instrument, orb_labels, all_filters):
    """
    Pre-compute relative volume at break bar for each (trading_day, orb_label).

    Enriches each feature row dict with rel_vol_{orb_label} key.
    Only runs if at least one VolumeFilter is in all_filters.
    Fail-closed: missing data -> rel_vol stays absent -> filter rejects.
    """
    import statistics

    # Determine max lookback needed across all volume filters
    vol_filters = [f for f in all_filters.values() if isinstance(f, VolumeFilter)]
    if not vol_filters:
        return
    max_lookback = max(f.lookback_days for f in vol_filters)

    # Step 1: Collect all break timestamps and unique UTC minutes-of-day
    break_ts_list = []
    unique_minutes = set()
    for row in features:
        for orb_label in orb_labels:
            break_ts = row.get(f"orb_{orb_label}_break_ts")
            if break_ts is not None and hasattr(break_ts, "hour"):
                break_ts_list.append(break_ts)
                utc_ts = break_ts.astimezone(timezone.utc) if break_ts.tzinfo is not None else break_ts
                unique_minutes.add(utc_ts.hour * 60 + utc_ts.minute)

    if not unique_minutes:
        return

    # Step 2: Load historical volumes for each unique minute-of-day
    # Keyed by minute-of-day, each entry is [(minute_key, volume), ...] sorted chronologically
    minute_history = {}
    for mod in sorted(unique_minutes):
        h, m = divmod(mod, 60)
        rows = con.execute(
            """SELECT ts_utc, volume FROM bars_1m
               WHERE symbol = ?
               AND EXTRACT(HOUR FROM (ts_utc AT TIME ZONE 'UTC')) = ?
               AND EXTRACT(MINUTE FROM (ts_utc AT TIME ZONE 'UTC')) = ?
               ORDER BY ts_utc""",
            [instrument, h, m],
        ).fetchall()
        minute_history[mod] = [(_ts_minute_key(ts), vol) for ts, vol in rows]

    # Step 3: Compute relative volume for each (day, orb_label) break
    for row in features:
        for orb_label in orb_labels:
            break_ts = row.get(f"orb_{orb_label}_break_ts")
            if break_ts is None:
                continue

            break_key = _ts_minute_key(break_ts)
            utc_ts = break_ts.astimezone(timezone.utc) if break_ts.tzinfo is not None else break_ts
            mod = utc_ts.hour * 60 + utc_ts.minute
            history = minute_history.get(mod, [])
            if not history:
                continue  # fail-closed

            # Find this break bar in the chronological history
            idx = None
            for j, (k, _) in enumerate(history):
                if k == break_key:
                    idx = j
                    break
            if idx is None:
                continue  # fail-closed: break bar not in bars_1m

            break_vol = history[idx][1]
            if break_vol is None or break_vol == 0:
                continue  # fail-closed

            # Take prior N entries (up to max_lookback)
            start = max(0, idx - max_lookback)
            prior_vols = [v for _, v in history[start:idx] if v > 0]

            if not prior_vols:
                continue  # fail-closed: no baseline

            baseline = statistics.median(prior_vols)
            if baseline <= 0:
                continue  # fail-closed

            row[f"rel_vol_{orb_label}"] = break_vol / baseline

def _load_outcomes_bulk(con, instrument, orb_minutes, orb_labels, entry_models):
    """
    Load all non-NULL outcomes in one query per (orb, entry_model).

    Returns dict keyed by (orb_label, entry_model, rr_target, confirm_bars)
    with value = list of outcome dicts.
    """
    grouped = {}
    for orb_label in orb_labels:
        for em in entry_models:
            rows = con.execute(
                """SELECT trading_day, rr_target, confirm_bars,
                          outcome, pnl_r, mae_r, mfe_r,
                          entry_price, stop_price
                   FROM orb_outcomes
                   WHERE symbol = ? AND orb_minutes = ?
                     AND orb_label = ? AND entry_model = ?
                     AND outcome IS NOT NULL
                   ORDER BY trading_day""",
                [instrument, orb_minutes, orb_label, em],
            ).fetchall()

            for r in rows:
                key = (orb_label, em, r[1], r[2])  # (orb, em, rr, cb)
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append({
                    "trading_day": r[0],
                    "outcome": r[3],
                    "pnl_r": r[4],
                    "mae_r": r[5],
                    "mfe_r": r[6],
                    "entry_price": r[7],
                    "stop_price": r[8],
                })

    return grouped

def run_discovery(
    db_path: Path | None = None,
    instrument: str = "MGC",
    start_date: date | None = None,
    end_date: date | None = None,
    orb_minutes: int = 5,
    dry_run: bool = False,
) -> int:
    """
    Grid search over all strategy variants.

    Bulk-loads data upfront (1 features query + 18 outcome queries),
    then iterates the grid in Python with no further DB reads.

    Returns count of strategies written.
    """
    if db_path is None:
        db_path = GOLD_DB_PATH

    if not dry_run:
        init_trading_app_schema(db_path=db_path)

    with duckdb.connect(str(db_path)) as con:
        # Determine which sessions to search
        sessions = get_enabled_sessions(instrument)
        if not sessions:
            sessions = ORB_LABELS  # fallback: all sessions
        logger.info(f"Sessions: {len(sessions)} enabled for {instrument}")

        # ---- Bulk load phase (all DB reads happen here) ----
        logger.info("Loading daily features...")
        features = _load_daily_features(con, instrument, orb_minutes, start_date, end_date)
        logger.info(f"  {len(features)} daily_features rows loaded")

        # Build union of all session-specific filters for bulk pre-computation
        all_grid_filters: dict = {}
        for s in sessions:
            all_grid_filters.update(get_filters_for_grid(instrument, s))

        logger.info("Computing relative volumes for volume filters...")
        _compute_relative_volumes(con, features, instrument, sessions, all_grid_filters)

        logger.info("Building filter/ORB day sets...")
        filter_days = _build_filter_day_sets(features, sessions, all_grid_filters)

        logger.info("Loading outcomes (bulk)...")
        outcomes_by_key = _load_outcomes_bulk(con, instrument, orb_minutes, sessions, ENTRY_MODELS)
        logger.info(f"  {sum(len(v) for v in outcomes_by_key.values())} outcome rows loaded")

        # ---- Grid iteration (pure Python, no DB reads) ----
        # Collect all strategies in memory first, then dedup before writing
        all_strategies = []  # list of (strategy_id, filter_key, trade_days, row_data)
        total_combos = 0
        for s in sessions:
            nf = len(get_filters_for_grid(instrument, s))
            total_combos += nf * len(RR_TARGETS) * len(CONFIRM_BARS_OPTIONS)  # E1
            total_combos += nf * len(RR_TARGETS) * 1                          # E3
        combo_idx = 0

        for orb_label in sessions:
            session_filters = get_filters_for_grid(instrument, orb_label)
            for filter_key, strategy_filter in session_filters.items():
                matching_day_set = filter_days[(filter_key, orb_label)]

                for em in ENTRY_MODELS:
                    for rr_target in RR_TARGETS:
                        for cb in CONFIRM_BARS_OPTIONS:
                            if em == "E3" and cb > 1:
                                continue
                            combo_idx += 1

                            if not matching_day_set:
                                continue

                            # Filter pre-loaded outcomes by matching days
                            all_outcomes = outcomes_by_key.get((orb_label, em, rr_target, cb), [])
                            outcomes = [o for o in all_outcomes if o["trading_day"] in matching_day_set]

                            if not outcomes:
                                continue

                            metrics = compute_metrics(outcomes)
                            if metrics["sample_size"] == 0:
                                continue
                            strategy_id = make_strategy_id(
                                instrument, orb_label, em, rr_target, cb, filter_key,
                            )
                            trade_days = sorted({o["trading_day"] for o in outcomes})
                            trade_day_hash = compute_trade_day_hash(trade_days)

                            # Compute DST regime split for affected sessions
                            dst_split = compute_dst_split_from_outcomes(outcomes, orb_label)

                            all_strategies.append({
                                "strategy_id": strategy_id,
                                "instrument": instrument,
                                "orb_label": orb_label,
                                "orb_minutes": orb_minutes,
                                "rr_target": rr_target,
                                "confirm_bars": cb,
                                "entry_model": em,
                                "filter_key": filter_key,
                                "filter_params": strategy_filter.to_json(),
                                "metrics": metrics,
                                "trade_day_hash": trade_day_hash,
                                "dst_split": dst_split,
                            })

                if combo_idx % 500 == 0:
                    logger.info(f"  Progress: {combo_idx}/{total_combos} combos, {len(all_strategies)} strategies")

        # ---- Dedup: mark canonical vs alias within each group ----
        logger.info(f"Dedup: {len(all_strategies)} strategies, computing canonical...")
        _mark_canonical(all_strategies)
        n_canonical = sum(1 for s in all_strategies if s["is_canonical"])
        n_alias = len(all_strategies) - n_canonical
        logger.info(f"  {n_canonical} canonical, {n_alias} aliases")

        # ---- Batch write ----
        if not dry_run:
            # Preserve existing created_at timestamps (INSERT OR REPLACE = DELETE+INSERT)
            existing_created = {}
            rows = con.execute(
                "SELECT strategy_id, created_at FROM experimental_strategies WHERE instrument = ?",
                [instrument],
            ).fetchall()
            existing_created = {r[0]: r[1] for r in rows}

            insert_batch = []
            for s in all_strategies:
                m = s["metrics"]
                dst = s["dst_split"]
                insert_batch.append([
                    s["strategy_id"], s["instrument"], s["orb_label"],
                    s["orb_minutes"], s["rr_target"], s["confirm_bars"],
                    s["entry_model"], s["filter_key"], s["filter_params"],
                    m["sample_size"], m["win_rate"],
                    m["avg_win_r"], m["avg_loss_r"],
                    m["expectancy_r"], m["sharpe_ratio"],
                    m["max_drawdown_r"],
                    m["median_risk_points"], m["avg_risk_points"],
                    m["trades_per_year"], m["sharpe_ann"],
                    m["yearly_results"],
                    m["entry_signals"], m["scratch_count"],
                    m["early_exit_count"],
                    s["trade_day_hash"], s["is_canonical"],
                    s["canonical_strategy_id"],
                    dst["winter_n"], dst["winter_avg_r"],
                    dst["summer_n"], dst["summer_avg_r"],
                    dst["verdict"],
                    None, None,  # Reset validation_status/notes
                    existing_created.get(s["strategy_id"]),  # Preserve or DEFAULT
                ])

                if len(insert_batch) >= 500:
                    con.executemany(_INSERT_SQL, insert_batch)
                    insert_batch = []

            if insert_batch:
                con.executemany(_INSERT_SQL, insert_batch)
            con.commit()

        total_strategies = len(all_strategies)
        logger.info(f"Discovered {total_strategies} strategies "
                    f"({n_canonical} canonical, {n_alias} aliases) "
                    f"from {total_combos} combos")
        if dry_run:
            logger.info("  (DRY RUN -- no data written)")

        return total_strategies

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Grid search over strategy variants"
    )
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--start", type=date.fromisoformat, help="Start date")
    parser.add_argument("--end", type=date.fromisoformat, help="End date")
    parser.add_argument("--orb-minutes", type=int, default=5, help="ORB duration")
    parser.add_argument("--dry-run", action="store_true", help="No DB writes")
    parser.add_argument("--db", type=str, default=None,
                        help="Database path (default: gold.db)")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else None

    run_discovery(
        db_path=db_path,
        instrument=args.instrument,
        start_date=args.start,
        end_date=args.end,
        orb_minutes=args.orb_minutes,
        dry_run=args.dry_run,
    )

if __name__ == "__main__":
    main()
