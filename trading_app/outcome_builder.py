"""
Pre-compute outcomes for all RR targets x confirm_bars combinations.

Populates the orb_outcomes table from daily_features + bars_1m data.
For each (trading_day, orb_label) with a break, computes outcomes at
multiple RR targets (1.0-4.0) and confirm_bars (1-5).

Usage:
    python trading_app/outcome_builder.py --instrument MGC --start 2024-01-01 --end 2024-12-31
    python trading_app/outcome_builder.py --instrument MGC --start 2024-01-01 --end 2024-12-31 --dry-run
"""

import sys
import time
from pathlib import Path
from datetime import date, datetime

from pipeline.log import get_logger
logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec, pnl_points_to_r, to_r_multiple, risk_in_dollars
from pipeline.init_db import ORB_LABELS
from pipeline.asset_configs import get_enabled_sessions
from pipeline.build_daily_features import compute_trading_day_utc_range
from trading_app.entry_rules import detect_entry_with_confirm_bars
from trading_app.config import ENTRY_MODELS, EARLY_EXIT_MINUTES
from trading_app.db_manager import init_trading_app_schema

# Grid parameters — see trading_app/config.py for full documentation
#
# RR_TARGETS: Risk-reward ratio multiples.
#   Target price = entry_price +/- (risk_points * rr_target * direction)
#   Risk = |entry_price - stop_price| where stop = opposite ORB level.
#   RR1.0 = target at 1x risk (high WR, low edge per trade)
#   RR4.0 = target at 4x risk (low WR, needs big moves)
#   Optimal: RR2.5 for 0900/1000, RR2.0 for 1800, RR1.5 for 2300
RR_TARGETS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

# CONFIRM_BARS_OPTIONS: How many consecutive 1-min closes outside ORB
#   before confirming a breakout signal and triggering entry.
#   CB1 = fastest entry, more fakeouts
#   CB5 = most confirmation, may miss momentum
#   NOTE: E3 (limit-at-ORB) always uses CB1 — higher CBs produce identical
#   entry prices (always ORB level) with 93-96% outcome overlap.
CONFIRM_BARS_OPTIONS = [1, 2, 3, 4, 5]

def _check_fill_bar_exit(
    bars_df: pd.DataFrame,
    entry_ts: datetime,
    entry_price: float,
    stop_price: float,
    target_price: float,
    break_dir: str,
    entry_model: str,
    cost_spec,
    rr_target: float,
) -> dict | None:
    """Check if the fill bar itself hits stop or target.

    Returns outcome dict if exit detected on fill bar, None otherwise.
    For E1: entry is at bar open, so full bar OHLC is post-fill.
    For E0/E3: entry is intra-bar at ORB level, check bar OHLC against levels.
    """
    fill_bar = bars_df[bars_df["ts_utc"] == pd.Timestamp(entry_ts)]
    if fill_bar.empty:
        return None

    bar = fill_bar.iloc[0]
    bar_high = bar["high"]
    bar_low = bar["low"]

    if break_dir == "long":
        hit_target = bar_high >= target_price
        hit_stop = bar_low <= stop_price
        favorable_pts = float(bar_high - entry_price)
        adverse_pts = float(entry_price - bar_low)
    else:
        hit_target = bar_low <= target_price
        hit_stop = bar_high >= stop_price
        favorable_pts = float(entry_price - bar_low)
        adverse_pts = float(bar_high - entry_price)

    if not hit_target and not hit_stop:
        return None

    exit_ts_val = bar["ts_utc"].to_pydatetime()
    result = {}

    if hit_target and hit_stop:
        # Ambiguous — conservative loss (matches existing convention)
        result["outcome"] = "loss"
        result["exit_ts"] = exit_ts_val
        result["exit_price"] = stop_price
        result["pnl_r"] = -1.0
    elif hit_target:
        risk_points = abs(entry_price - stop_price)
        result["outcome"] = "win"
        result["exit_ts"] = exit_ts_val
        result["exit_price"] = target_price
        result["pnl_r"] = round(
            to_r_multiple(cost_spec, entry_price, stop_price,
                          risk_points * rr_target),
            4,
        )
    else:
        result["outcome"] = "loss"
        result["exit_ts"] = exit_ts_val
        result["exit_price"] = stop_price
        result["pnl_r"] = -1.0

    # MAE/MFE for fill-bar exit: single-bar excursion
    result["mae_r"] = round(
        pnl_points_to_r(cost_spec, entry_price, stop_price, max(adverse_pts, 0.0)), 4
    )
    result["mfe_r"] = round(
        pnl_points_to_r(cost_spec, entry_price, stop_price, max(favorable_pts, 0.0)), 4
    )

    return result

def _compute_outcomes_all_rr(
    bars_df: pd.DataFrame,
    signal,
    orb_high: float,
    orb_low: float,
    break_dir: str,
    rr_targets: list[float],
    trading_day_end: datetime,
    cost_spec,
    entry_model: str = "E1",
    orb_label: str | None = None,
    break_ts=None,
) -> list[dict]:
    """Compute outcomes for ALL RR targets from a single pre-detected entry.

    Avoids redundant entry detection and DataFrame slicing across RR targets.
    Returns list of outcome dicts (one per RR target).
    """
    null_result = {
        "entry_ts": None, "entry_price": None, "stop_price": None,
        "target_price": None, "outcome": None, "exit_ts": None,
        "exit_price": None, "pnl_r": None,
        "risk_dollars": None, "pnl_dollars": None,
        "mae_r": None, "mfe_r": None,
    }

    if not signal.triggered:
        return [dict(null_result) for _ in rr_targets]

    # C3: at 1000 session only, reject slow breaks (confirm > 3 min after break_ts).
    # Data-verified: fast breaks (<=3 min) avg +0.213R vs slow (>3 min) -0.339R at 1000.
    # Note: 0900/1800 also show positive delta but not yet BH-validated cross-session.
    if (orb_label == "1000" and break_ts is not None
            and signal.confirm_bar_ts is not None):
        break_speed_min = (
            pd.Timestamp(signal.confirm_bar_ts) - pd.Timestamp(break_ts)
        ).total_seconds() / 60
        if break_speed_min > 3:
            return [dict(null_result) for _ in rr_targets]

    entry_price = signal.entry_price
    stop_price = signal.stop_price
    entry_ts = signal.entry_ts
    risk_points = abs(entry_price - stop_price)

    if risk_points <= 0:
        return [dict(null_result) for _ in rr_targets]

    # Per-contract dollar risk (same for all RR targets — risk is ORB-based)
    _risk_dollars = round(risk_in_dollars(cost_spec, entry_price, stop_price), 2)

    # Pre-compute target prices for all RR targets
    if break_dir == "long":
        target_prices = [entry_price + risk_points * rr for rr in rr_targets]
    else:
        target_prices = [entry_price - risk_points * rr for rr in rr_targets]

    # Pre-fetch fill bar ONCE
    fill_bar = bars_df[bars_df["ts_utc"] == pd.Timestamp(entry_ts)]
    fill_row = fill_bar.iloc[0] if not fill_bar.empty else None

    # Pre-slice post-entry bars ONCE
    post_entry = bars_df[
        (bars_df["ts_utc"] > pd.Timestamp(entry_ts))
        & (bars_df["ts_utc"] < pd.Timestamp(trading_day_end))
    ].sort_values("ts_utc")

    # Pre-compute shared numpy arrays from post-entry (same for all RR)
    has_post = not post_entry.empty
    if has_post:
        pe_highs = post_entry["high"].values
        pe_lows = post_entry["low"].values
        pe_closes = post_entry["close"].values
        if break_dir == "long":
            pe_hit_stop = pe_lows <= stop_price
            pe_favorable = pe_highs - entry_price
            pe_adverse = entry_price - pe_lows
        else:
            pe_hit_stop = pe_highs >= stop_price
            pe_favorable = entry_price - pe_lows
            pe_adverse = pe_highs - entry_price

        # Early exit: shared threshold detection
        early_exit_threshold = EARLY_EXIT_MINUTES.get(orb_label) if orb_label else None
        threshold_idx = None
        threshold_applies = False
        if early_exit_threshold is not None:
            elapsed = (post_entry["ts_utc"] - pd.Timestamp(entry_ts)).dt.total_seconds().values / 60.0
            threshold_mask = elapsed >= early_exit_threshold
            if threshold_mask.any():
                threshold_idx = int(np.argmax(threshold_mask))
                threshold_applies = True

    results = []
    for rr, target_price in zip(rr_targets, target_prices):
        result = {
            "entry_ts": entry_ts, "entry_price": entry_price,
            "stop_price": stop_price, "target_price": target_price,
            "outcome": None, "exit_ts": None, "exit_price": None,
            "pnl_r": None, "risk_dollars": _risk_dollars, "pnl_dollars": None,
            "mae_r": None, "mfe_r": None,
        }

        # --- Fill bar check ---
        if fill_row is not None:
            bar_high, bar_low = fill_row["high"], fill_row["low"]
            if break_dir == "long":
                hit_tgt = bar_high >= target_price
                hit_stp = bar_low <= stop_price
                fav_pts = float(bar_high - entry_price)
                adv_pts = float(entry_price - bar_low)
            else:
                hit_tgt = bar_low <= target_price
                hit_stp = bar_high >= stop_price
                fav_pts = float(entry_price - bar_low)
                adv_pts = float(bar_high - entry_price)

            if hit_tgt or hit_stp:
                exit_ts_val = fill_row["ts_utc"].to_pydatetime()
                if hit_tgt and hit_stp:
                    result.update(outcome="loss", exit_ts=exit_ts_val,
                                  exit_price=stop_price, pnl_r=-1.0)
                elif hit_tgt:
                    result.update(
                        outcome="win", exit_ts=exit_ts_val,
                        exit_price=target_price,
                        pnl_r=round(to_r_multiple(cost_spec, entry_price,
                                                  stop_price, risk_points * rr), 4),
                    )
                else:
                    result.update(outcome="loss", exit_ts=exit_ts_val,
                                  exit_price=stop_price, pnl_r=-1.0)
                result["mae_r"] = round(
                    pnl_points_to_r(cost_spec, entry_price, stop_price, max(adv_pts, 0.0)), 4)
                result["mfe_r"] = round(
                    pnl_points_to_r(cost_spec, entry_price, stop_price, max(fav_pts, 0.0)), 4)
                if result["pnl_r"] is not None:
                    result["pnl_dollars"] = round(result["pnl_r"] * _risk_dollars, 2)
                results.append(result)
                continue

        # --- No post-entry bars ---
        if not has_post:
            result["outcome"] = "scratch"
            result["mae_r"] = round(pnl_points_to_r(cost_spec, entry_price, stop_price, 0.0), 4)
            result["mfe_r"] = round(pnl_points_to_r(cost_spec, entry_price, stop_price, 0.0), 4)
            # pnl_dollars stays None for scratch (no trade)
            results.append(result)
            continue

        # --- Target check varies per RR; stop/favorable/adverse are shared ---
        if break_dir == "long":
            pe_hit_target = pe_highs >= target_price
        else:
            pe_hit_target = pe_lows <= target_price

        # --- Timed early exit (shared threshold, target-dependent check) ---
        if threshold_applies:
            any_prior_hit = (pe_hit_target[:threshold_idx] | pe_hit_stop[:threshold_idx])
            if not any_prior_hit.any():
                if break_dir == "long":
                    mtm_points = float(pe_closes[threshold_idx] - entry_price)
                else:
                    mtm_points = float(entry_price - pe_closes[threshold_idx])
                if mtm_points < 0:
                    result["outcome"] = "early_exit"
                    result["exit_ts"] = post_entry.iloc[threshold_idx]["ts_utc"].to_pydatetime()
                    result["exit_price"] = float(pe_closes[threshold_idx])
                    result["pnl_r"] = round(
                        to_r_multiple(cost_spec, entry_price, stop_price, mtm_points), 4)
                    max_fav = max(float(np.max(pe_favorable[:threshold_idx + 1])), 0.0)
                    max_adv = max(float(np.max(pe_adverse[:threshold_idx + 1])), 0.0)
                    result["mae_r"] = round(
                        pnl_points_to_r(cost_spec, entry_price, stop_price, max_adv), 4)
                    result["mfe_r"] = round(
                        pnl_points_to_r(cost_spec, entry_price, stop_price, max_fav), 4)
                    if result["pnl_r"] is not None:
                        result["pnl_dollars"] = round(result["pnl_r"] * _risk_dollars, 2)
                    results.append(result)
                    continue

        # --- Standard target/stop scan ---
        any_hit = pe_hit_target | pe_hit_stop
        if not any_hit.any():
            result["outcome"] = "scratch"
            max_fav = max(float(np.max(pe_favorable)), 0.0)
            max_adv = max(float(np.max(pe_adverse)), 0.0)
        else:
            first_hit_idx = int(np.argmax(any_hit))
            exit_ts_val = post_entry.iloc[first_hit_idx]["ts_utc"].to_pydatetime()
            if pe_hit_target[first_hit_idx] and pe_hit_stop[first_hit_idx]:
                result.update(outcome="loss", exit_ts=exit_ts_val,
                              exit_price=stop_price, pnl_r=-1.0)
            elif pe_hit_target[first_hit_idx]:
                result.update(
                    outcome="win", exit_ts=exit_ts_val, exit_price=target_price,
                    pnl_r=round(to_r_multiple(cost_spec, entry_price, stop_price,
                                              risk_points * rr), 4),
                )
            else:
                result.update(outcome="loss", exit_ts=exit_ts_val,
                              exit_price=stop_price, pnl_r=-1.0)
            max_fav = max(float(np.max(pe_favorable[:first_hit_idx + 1])), 0.0)
            max_adv = max(float(np.max(pe_adverse[:first_hit_idx + 1])), 0.0)

        result["mae_r"] = round(
            pnl_points_to_r(cost_spec, entry_price, stop_price, max_adv), 4)
        result["mfe_r"] = round(
            pnl_points_to_r(cost_spec, entry_price, stop_price, max_fav), 4)
        if result["pnl_r"] is not None:
            result["pnl_dollars"] = round(result["pnl_r"] * _risk_dollars, 2)
        results.append(result)

    return results

def compute_single_outcome(
    bars_df: pd.DataFrame,
    break_ts: datetime,
    orb_high: float,
    orb_low: float,
    break_dir: str,
    rr_target: float,
    confirm_bars: int,
    trading_day_end: datetime,
    cost_spec,
    entry_model: str = "E1",
    orb_label: str | None = None,
) -> dict:
    """
    Compute outcome for a single (rr_target, confirm_bars, entry_model) combination.

    Returns dict with keys matching orb_outcomes columns.
    """
    result = {
        "entry_ts": None,
        "entry_price": None,
        "stop_price": None,
        "target_price": None,
        "outcome": None,
        "exit_ts": None,
        "exit_price": None,
        "pnl_r": None,
        "risk_dollars": None,
        "pnl_dollars": None,
        "mae_r": None,
        "mfe_r": None,
    }

    # Detect entry with confirm bars
    signal = detect_entry_with_confirm_bars(
        bars_df=bars_df,
        orb_break_ts=break_ts,
        orb_high=orb_high,
        orb_low=orb_low,
        break_dir=break_dir,
        confirm_bars=confirm_bars,
        detection_window_end=trading_day_end,
        entry_model=entry_model,
    )

    if not signal.triggered:
        return result

    entry_price = signal.entry_price
    stop_price = signal.stop_price
    entry_ts = signal.entry_ts
    risk_points = abs(entry_price - stop_price)

    if risk_points <= 0:
        return result

    # Compute target price
    if break_dir == "long":
        target_price = entry_price + risk_points * rr_target
    else:
        target_price = entry_price - risk_points * rr_target

    result["entry_ts"] = entry_ts
    result["entry_price"] = entry_price
    result["stop_price"] = stop_price
    result["target_price"] = target_price

    # Per-contract dollar risk
    _risk_dollars = round(risk_in_dollars(cost_spec, entry_price, stop_price), 2)
    result["risk_dollars"] = _risk_dollars

    # Check fill bar for immediate exit (E1 and E3)
    fill_exit = _check_fill_bar_exit(
        bars_df, entry_ts, entry_price, stop_price, target_price,
        break_dir, entry_model, cost_spec, rr_target,
    )
    if fill_exit is not None:
        result.update(fill_exit)
        if result["pnl_r"] is not None:
            result["pnl_dollars"] = round(result["pnl_r"] * _risk_dollars, 2)
        return result

    # Scan bars forward from entry to determine outcome
    post_entry = bars_df[
        (bars_df["ts_utc"] > pd.Timestamp(entry_ts))
        & (bars_df["ts_utc"] < pd.Timestamp(trading_day_end))
    ].sort_values("ts_utc")

    if post_entry.empty:
        result["outcome"] = "scratch"
        result["mae_r"] = round(
            pnl_points_to_r(cost_spec, entry_price, stop_price, 0.0), 4
        )
        result["mfe_r"] = round(
            pnl_points_to_r(cost_spec, entry_price, stop_price, 0.0), 4
        )
        return result

    # --- Timed early exit: kill losers at N minutes after fill ---
    early_exit_threshold = EARLY_EXIT_MINUTES.get(orb_label) if orb_label else None

    highs = post_entry["high"].values
    lows = post_entry["low"].values
    closes = post_entry["close"].values

    if break_dir == "long":
        hit_target = highs >= target_price
        hit_stop = lows <= stop_price
        favorable = highs - entry_price
        adverse = entry_price - lows
    else:
        hit_target = lows <= target_price
        hit_stop = highs >= stop_price
        favorable = entry_price - lows
        adverse = highs - entry_price

    if early_exit_threshold is not None:
        elapsed = (post_entry["ts_utc"] - pd.Timestamp(entry_ts)).dt.total_seconds().values / 60.0
        threshold_mask = elapsed >= early_exit_threshold
        if threshold_mask.any():
            threshold_idx = int(np.argmax(threshold_mask))
            # Check if stop or target hit BEFORE the threshold bar
            any_prior_hit = (hit_target[:threshold_idx] | hit_stop[:threshold_idx])
            if not any_prior_hit.any():
                # Check MTM at threshold bar close
                if break_dir == "long":
                    mtm_points = float(closes[threshold_idx] - entry_price)
                else:
                    mtm_points = float(entry_price - closes[threshold_idx])
                if mtm_points < 0:
                    # Early exit: loser at threshold
                    result["outcome"] = "early_exit"
                    result["exit_ts"] = post_entry.iloc[threshold_idx]["ts_utc"].to_pydatetime()
                    result["exit_price"] = float(closes[threshold_idx])
                    result["pnl_r"] = round(
                        to_r_multiple(cost_spec, entry_price, stop_price, mtm_points), 4
                    )
                    max_favorable_points = max(float(np.max(favorable[:threshold_idx + 1])), 0.0)
                    max_adverse_points = max(float(np.max(adverse[:threshold_idx + 1])), 0.0)
                    result["mae_r"] = round(
                        pnl_points_to_r(cost_spec, entry_price, stop_price, max_adverse_points), 4
                    )
                    result["mfe_r"] = round(
                        pnl_points_to_r(cost_spec, entry_price, stop_price, max_favorable_points), 4
                    )
                    if result["pnl_r"] is not None:
                        result["pnl_dollars"] = round(result["pnl_r"] * _risk_dollars, 2)
                    return result
                # MTM >= 0 at threshold -> winner, fall through to normal scan

    any_hit = hit_target | hit_stop

    if not any_hit.any():
        # No target or stop hit — scratch
        result["outcome"] = "scratch"
        max_favorable_points = max(float(np.max(favorable)), 0.0)
        max_adverse_points = max(float(np.max(adverse)), 0.0)
    else:
        first_hit_idx = int(np.argmax(any_hit))
        # Use .iloc to preserve tz-aware timestamp (not .values which strips tz)
        exit_ts_val = post_entry.iloc[first_hit_idx]["ts_utc"].to_pydatetime()

        if hit_target[first_hit_idx] and hit_stop[first_hit_idx]:
            # Ambiguous bar — conservative: assume loss
            result["outcome"] = "loss"
            result["exit_ts"] = exit_ts_val
            result["exit_price"] = stop_price
            result["pnl_r"] = -1.0
        elif hit_target[first_hit_idx]:
            result["outcome"] = "win"
            result["exit_ts"] = exit_ts_val
            result["exit_price"] = target_price
            result["pnl_r"] = round(
                to_r_multiple(cost_spec, entry_price, stop_price,
                              risk_points * rr_target),
                4,
            )
        else:
            result["outcome"] = "loss"
            result["exit_ts"] = exit_ts_val
            result["exit_price"] = stop_price
            result["pnl_r"] = -1.0

        # MAE/MFE up to and including the exit bar (clamped >= 0)
        max_favorable_points = max(float(np.max(favorable[: first_hit_idx + 1])), 0.0)
        max_adverse_points = max(float(np.max(adverse[: first_hit_idx + 1])), 0.0)

    # MAE/MFE in R
    result["mae_r"] = round(
        pnl_points_to_r(cost_spec, entry_price, stop_price, max_adverse_points), 4
    )
    result["mfe_r"] = round(
        pnl_points_to_r(cost_spec, entry_price, stop_price, max_favorable_points), 4
    )

    # Dollar P&L
    if result["pnl_r"] is not None:
        result["pnl_dollars"] = round(result["pnl_r"] * _risk_dollars, 2)

    return result

def build_outcomes(
    db_path: Path | None = None,
    instrument: str = "MGC",
    start_date: date | None = None,
    end_date: date | None = None,
    orb_minutes: int = 5,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """
    Build orb_outcomes for all RR targets x confirm_bars.

    Returns count of rows written.
    """
    if db_path is None:
        db_path = GOLD_DB_PATH

    cost_spec = get_cost_spec(instrument)

    with duckdb.connect(str(db_path)) as con:
        from pipeline.db_config import configure_connection
        configure_connection(con, writing=True)

        # Ensure trading_app tables exist
        if not dry_run:
            init_trading_app_schema(db_path=db_path)

        # Build date filter
        date_clauses = []
        params = [instrument, orb_minutes]
        if start_date:
            date_clauses.append("AND trading_day >= ?")
            params.append(start_date)
        if end_date:
            date_clauses.append("AND trading_day <= ?")
            params.append(end_date)
        date_filter = " ".join(date_clauses)

        # Determine which sessions to build outcomes for
        sessions = get_enabled_sessions(instrument)
        if not sessions:
            sessions = ORB_LABELS  # fallback: all sessions
        logger.info(f"  Sessions: {len(sessions)} enabled for {instrument}")

        # Fetch all daily_features rows
        query = f"""
            SELECT trading_day, symbol, orb_minutes,
                   {', '.join(
                       f'orb_{lbl}_high, orb_{lbl}_low, orb_{lbl}_break_dir, orb_{lbl}_break_ts'
                       for lbl in sessions
                   )}
            FROM daily_features
            WHERE symbol = ? AND orb_minutes = ?
            {date_filter}
            ORDER BY trading_day
        """
        rows = con.execute(query, params).fetchall()
        col_names = [desc[0] for desc in con.description]

        total_written = 0
        total_days = len(rows)
        t0 = time.monotonic()

        # --- Batch bars loading: one query instead of ~N per-day queries ---
        # Pre-compute trading day boundaries for first and last day
        if total_days == 0:
            return 0
        first_day = dict(zip(col_names, rows[0]))["trading_day"]
        last_day = dict(zip(col_names, rows[-1]))["trading_day"]
        global_start, _ = compute_trading_day_utc_range(first_day)
        _, global_end = compute_trading_day_utc_range(last_day)

        logger.info(f"  Loading bars_1m for {instrument} ({global_start.date()} to {global_end.date()})...")
        all_bars_df = con.execute(
            """
            SELECT ts_utc, open, high, low, close, volume
            FROM bars_1m
            WHERE symbol = ?
            AND ts_utc >= ?::TIMESTAMPTZ
            AND ts_utc < ?::TIMESTAMPTZ
            ORDER BY ts_utc ASC
            """,
            [instrument, global_start.isoformat(), global_end.isoformat()],
        ).fetchdf()
        if not all_bars_df.empty:
            all_bars_df["ts_utc"] = pd.to_datetime(all_bars_df["ts_utc"], utc=True)
        all_ts = all_bars_df["ts_utc"].values  # numpy datetime64 array for searchsorted
        logger.info(f"  Loaded {len(all_bars_df):,} bars into memory")

        # Pre-load already-computed days for checkpoint/resume
        computed_days = set()
        if not dry_run and not force:
            computed_days = {
                r[0] for r in con.execute(
                    "SELECT DISTINCT trading_day FROM orb_outcomes "
                    "WHERE symbol = ? AND orb_minutes = ?",
                    [instrument, orb_minutes],
                ).fetchall()
            }
            if computed_days:
                logger.info(f"  Checkpoint: {len(computed_days)} days already computed, will skip")

        for day_idx, row in enumerate(rows):
            row_dict = dict(zip(col_names, row))
            trading_day = row_dict["trading_day"]
            symbol = row_dict["symbol"]

            # Skip days already computed (checkpoint/resume)
            if trading_day in computed_days:
                continue

            # Partition bars for this trading day using binary search (O(log n))
            td_start, td_end = compute_trading_day_utc_range(trading_day)
            start_idx = int(np.searchsorted(all_ts, pd.Timestamp(td_start).asm8, side="left"))
            end_idx = int(np.searchsorted(all_ts, pd.Timestamp(td_end).asm8, side="left"))
            bars_df = all_bars_df.iloc[start_idx:end_idx]

            if bars_df.empty:
                continue

            day_batch = []

            for orb_label in sessions:
                break_dir = row_dict.get(f"orb_{orb_label}_break_dir")
                break_ts = row_dict.get(f"orb_{orb_label}_break_ts")
                orb_high = row_dict.get(f"orb_{orb_label}_high")
                orb_low = row_dict.get(f"orb_{orb_label}_low")

                if break_dir is None or break_ts is None:
                    continue
                if orb_high is None or orb_low is None:
                    continue

                # Optimized: detect entry ONCE per (session, EM, CB),
                # then compute all 6 RR targets with shared bar slicing.
                for em in ENTRY_MODELS:
                    cb_options = [1] if em == "E3" else CONFIRM_BARS_OPTIONS
                    for cb in cb_options:
                        signal = detect_entry_with_confirm_bars(
                            bars_df=bars_df,
                            orb_break_ts=break_ts,
                            orb_high=orb_high,
                            orb_low=orb_low,
                            break_dir=break_dir,
                            confirm_bars=cb,
                            detection_window_end=td_end,
                            entry_model=em,
                        )

                        outcomes = _compute_outcomes_all_rr(
                            bars_df=bars_df,
                            signal=signal,
                            orb_high=orb_high,
                            orb_low=orb_low,
                            break_dir=break_dir,
                            rr_targets=RR_TARGETS,
                            trading_day_end=td_end,
                            cost_spec=cost_spec,
                            entry_model=em,
                            orb_label=orb_label,
                            break_ts=break_ts,
                        )

                        for rr_target, outcome in zip(RR_TARGETS, outcomes):
                            day_batch.append([
                                trading_day, symbol, orb_label, orb_minutes,
                                rr_target, cb, em,
                                outcome["entry_ts"], outcome["entry_price"],
                                outcome["stop_price"], outcome["target_price"],
                                outcome["outcome"], outcome["exit_ts"],
                                outcome["exit_price"], outcome["pnl_r"],
                                outcome["risk_dollars"], outcome["pnl_dollars"],
                                outcome["mae_r"], outcome["mfe_r"],
                            ])
                            total_written += 1

            # Batch insert all outcomes for this trading day
            if day_batch and not dry_run:
                batch_df = pd.DataFrame(  # noqa: F841 — used by DuckDB SQL below
                    day_batch,
                    columns=[
                        'trading_day', 'symbol', 'orb_label', 'orb_minutes',
                        'rr_target', 'confirm_bars', 'entry_model',
                        'entry_ts', 'entry_price', 'stop_price', 'target_price',
                        'outcome', 'exit_ts', 'exit_price', 'pnl_r',
                        'risk_dollars', 'pnl_dollars',
                        'mae_r', 'mfe_r',
                    ],
                )
                con.execute("""
                    INSERT OR REPLACE INTO orb_outcomes
                    SELECT trading_day, symbol, orb_label, orb_minutes,
                           rr_target, confirm_bars, entry_model,
                           entry_ts, entry_price, stop_price, target_price,
                           outcome, exit_ts, exit_price, pnl_r,
                           risk_dollars, pnl_dollars,
                           mae_r, mfe_r
                    FROM batch_df
                """)

            if (day_idx + 1) % 10 == 0:
                if not dry_run:
                    con.commit()
                elapsed = time.monotonic() - t0
                rate = (day_idx + 1) / elapsed
                remaining = (total_days - day_idx - 1) / rate if rate > 0 else 0
                print(
                    f"  {day_idx + 1}/{total_days} days "
                    f"({total_written} outcomes, "
                    f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)",
                    flush=True,
                )
                # Write heartbeat for external monitoring
                if not dry_run:
                    heartbeat_path = Path(db_path).parent / "outcome_builder.heartbeat"
                    heartbeat_path.write_text(
                        f"{datetime.now().isoformat()} | {instrument} | "
                        f"{day_idx + 1}/{total_days} | {trading_day}\n"
                    )

        if not dry_run:
            con.commit()

        elapsed = time.monotonic() - t0
        logger.info(f"Done: {total_written} outcomes for {total_days} trading days in {elapsed:.1f}s")
        if dry_run:
            logger.info("  (DRY RUN — no data written)")

        return total_written

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-compute ORB outcomes for all RR targets x confirm_bars"
    )
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--start", type=date.fromisoformat, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=date.fromisoformat, help="End date (YYYY-MM-DD)")
    parser.add_argument("--orb-minutes", type=int, default=5, help="ORB duration in minutes")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing")
    parser.add_argument("--force", action="store_true", help="Rebuild all days (ignore checkpoint)")
    args = parser.parse_args()

    build_outcomes(
        instrument=args.instrument,
        start_date=args.start,
        end_date=args.end,
        orb_minutes=args.orb_minutes,
        dry_run=args.dry_run,
        force=args.force,
    )

if __name__ == "__main__":
    main()
