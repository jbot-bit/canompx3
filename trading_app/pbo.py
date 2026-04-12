"""Probability of Backtest Overfitting (PBO) — Bailey et al. 2014.

Combinatorial Symmetric Cross-Validation (CSCV) approximation.
Partitions outcomes into S time blocks, tests all C(S, S/2) train/test
splits, and measures how often the IS-best strategy has negative OOS.

PBO > 0.50 = likely overfit (selection process is fragile).
PBO < 0.30 = robust (selection consistently picks a real edge).
"""

import logging
from collections import defaultdict
from itertools import combinations

from pipeline.cost_model import get_cost_spec
from trading_app.config import apply_tight_stop
from trading_app.validated_shelf import deployable_validated_relation

logger = logging.getLogger(__name__)


def compute_pbo(
    strategy_pnl: dict[str, list[tuple]],
    n_blocks: int = 8,
) -> dict:
    """Compute PBO for a set of strategies sharing the same trade days.

    Args:
        strategy_pnl: {strategy_id: [(trading_day, pnl_r), ...]} sorted by day.
            All strategies must share the same set of trading_days.
        n_blocks: Number of chronological time blocks (default 8 → C(8,4)=70 splits).

    Returns:
        dict with keys:
            pbo: float (0.0-1.0) or None if insufficient data
            n_splits: int — number of combinatorial splits tested
            n_negative_oos: int — splits where IS-best had negative OOS
            logit_pbo: float or None — log(PBO / (1 - PBO)), useful for ranking
    """
    if len(strategy_pnl) < 2:
        return {"pbo": None, "n_splits": 0, "n_negative_oos": 0, "logit_pbo": None}

    # Collect all unique trade days across strategies (should be identical)
    all_days = set()
    for days_list in strategy_pnl.values():
        all_days.update(d for d, _ in days_list)
    all_days_sorted = sorted(all_days)
    n_days = len(all_days_sorted)

    if n_days < n_blocks * 2:
        # Need at least 2 trades per block for meaningful splits
        return {"pbo": None, "n_splits": 0, "n_negative_oos": 0, "logit_pbo": None}

    # Build per-block pnl sums for each strategy
    block_size = n_days // n_blocks
    blocks = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size if i < n_blocks - 1 else n_days
        block_days = set(all_days_sorted[start:end])
        blocks.append(block_days)

    # Pre-compute block-level pnl sums: strategy_id -> [sum_block_0, ..., sum_block_S-1]
    strategy_ids = list(strategy_pnl.keys())
    block_sums = {}
    for sid, day_pnl in strategy_pnl.items():
        pnl_by_day = defaultdict(float)
        for day, pnl in day_pnl:
            pnl_by_day[day] += pnl
        sums = []
        for block_days in blocks:
            sums.append(sum(pnl_by_day.get(d, 0.0) for d in block_days))
        block_sums[sid] = sums

    # CSCV: for each C(S, S/2) split
    half = n_blocks // 2
    all_splits = list(combinations(range(n_blocks), half))
    n_negative_oos = 0

    for train_blocks in all_splits:
        test_blocks = tuple(i for i in range(n_blocks) if i not in train_blocks)

        # Find IS-best strategy (highest train sum)
        best_sid = None
        best_is_pnl = float("-inf")
        for sid in strategy_ids:
            is_pnl = sum(block_sums[sid][b] for b in train_blocks)
            if is_pnl > best_is_pnl:
                best_is_pnl = is_pnl
                best_sid = sid

        # Measure OOS performance of IS-best
        oos_pnl = sum(block_sums[best_sid][b] for b in test_blocks)
        if oos_pnl < 0:
            n_negative_oos += 1

    n_splits = len(all_splits)
    pbo = n_negative_oos / n_splits if n_splits > 0 else None

    # Logit transform (Bailey convention): log(PBO / (1-PBO))
    logit_pbo = None
    if pbo is not None and 0 < pbo < 1:
        import math

        logit_pbo = round(math.log(pbo / (1 - pbo)), 4)

    return {
        "pbo": round(pbo, 4) if pbo is not None else None,
        "n_splits": n_splits,
        "n_negative_oos": n_negative_oos,
        "logit_pbo": logit_pbo,
    }


def _get_eligible_days(con, instrument: str, orb_label: str, filter_obj) -> set:
    """Return set of trading_days where filter passes on daily_features.

    Loads daily_features for the instrument and evaluates the filter object's
    matches_row() on each row. Build-time only — not performance-critical.
    """
    df = con.execute(
        "SELECT * FROM daily_features WHERE symbol = ?",
        [instrument],
    ).fetchdf()

    eligible = set()
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        if filter_obj.matches_row(row_dict, orb_label):
            # Normalize pandas Timestamp → datetime.date to match orb_outcomes keys
            td = row_dict["trading_day"]
            eligible.add(td.date() if hasattr(td, "date") else td)
    return eligible


def compute_family_pbo(
    con,
    family_hash: str,
    instrument: str,
) -> dict:
    """Compute PBO for an edge family by loading member outcomes from DB.

    Filter-aware: applies the family's filter_type to daily_features before
    running CSCV, so PBO measures parameter stability on the filtered dataset
    where the edge actually exists. Without this, filter-gated instruments
    (MGC) get PBO=1.0 because the raw unfiltered dataset is uniformly negative.

    Args:
        con: DuckDB connection (read-only OK)
        family_hash: The edge family hash
        instrument: Instrument symbol

    Returns:
        dict from compute_pbo() — includes pbo, n_splits, n_negative_oos, logit_pbo
    """
    # Get member strategy_ids
    shelf_relation = deployable_validated_relation(con)
    members = con.execute(
        f"""SELECT strategy_id, orb_label, orb_minutes, entry_model,
                  rr_target, confirm_bars, filter_type, stop_multiplier
           FROM {shelf_relation}
           WHERE family_hash = ? AND instrument = ?""",
        [family_hash, instrument],
    ).fetchall()

    if len(members) < 2:
        return {"pbo": None, "n_splits": 0, "n_negative_oos": 0, "logit_pbo": None}

    # All members of an edge family share the same filter_type (enforced by
    # family_hash grouping — different filters → different trade days →
    # different family_hash).
    filter_type = members[0][6]
    orb_label = members[0][1]

    # Determine eligible trade days via filter application on daily_features.
    # For NO_FILTER, all days are eligible (skip the query).
    eligible_days = None
    if filter_type and filter_type != "NO_FILTER":
        from trading_app.config import ALL_FILTERS

        filter_obj = ALL_FILTERS.get(filter_type)
        if filter_obj is not None:
            eligible_days = _get_eligible_days(con, instrument, orb_label, filter_obj)
            if not eligible_days:
                logger.warning(
                    "PBO: filter %s yields 0 eligible days for %s %s",
                    filter_type,
                    instrument,
                    orb_label,
                )
                return {"pbo": None, "n_splits": 0, "n_negative_oos": 0, "logit_pbo": None}

    # Bulk-load raw outcomes for all member variants once, then apply any
    # stop-multiplier transforms per strategy id.
    raw_member_keys = {
        (_orb_label, orb_minutes, entry_model, rr_target, confirm_bars)
        for _sid, _orb_label, orb_minutes, entry_model, rr_target, confirm_bars, _ft, _sm in members
    }

    if not raw_member_keys:
        return {"pbo": None, "n_splits": 0, "n_negative_oos": 0, "logit_pbo": None}

    # Single query with DuckDB multi-column IN (VALUES syntax)
    values_rows = list(raw_member_keys)
    placeholders = ", ".join(["(?, ?, ?, ?, ?)"] * len(values_rows))
    flat_params = [instrument]
    for row in values_rows:
        flat_params.extend(row)

    rows = con.execute(
        f"""SELECT o.trading_day, o.pnl_r, o.mae_r, o.entry_price, o.stop_price, o.outcome,
                   o.orb_label, o.orb_minutes, o.entry_model, o.rr_target, o.confirm_bars
            FROM orb_outcomes o
            WHERE o.symbol = ?
              AND o.pnl_r IS NOT NULL
              AND (o.orb_label, o.orb_minutes, o.entry_model, o.rr_target, o.confirm_bars)
                  IN (VALUES {placeholders})
            ORDER BY o.trading_day""",
        flat_params,
    ).fetchall()

    raw_outcomes_by_key: dict[tuple, list[dict]] = defaultdict(list)
    for trading_day, pnl_r, mae_r, entry_price, stop_price, outcome, _orb_label, orb_minutes, entry_model, rr_target, confirm_bars in rows:
        if eligible_days is not None and trading_day not in eligible_days:
            continue
        key = (_orb_label, orb_minutes, entry_model, rr_target, confirm_bars)
        raw_outcomes_by_key[key].append(
            {
                "trading_day": trading_day,
                "pnl_r": pnl_r,
                "mae_r": mae_r,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "outcome": outcome,
            }
        )

    strategy_pnl = {}
    cost_spec = get_cost_spec(instrument)
    for sid, _orb_label, orb_minutes, entry_model, rr_target, confirm_bars, _ft, stop_multiplier in members:
        key = (_orb_label, orb_minutes, entry_model, rr_target, confirm_bars)
        outcomes = list(raw_outcomes_by_key.get(key, []))
        if not outcomes:
            continue
        if stop_multiplier is not None and stop_multiplier < 1.0:
            outcomes = apply_tight_stop(outcomes, float(stop_multiplier), cost_spec)
        strategy_pnl[sid] = [(o["trading_day"], o["pnl_r"]) for o in outcomes if o.get("pnl_r") is not None]

    if len(strategy_pnl) < 2:
        return {"pbo": None, "n_splits": 0, "n_negative_oos": 0, "logit_pbo": None}

    return compute_pbo(strategy_pnl)
