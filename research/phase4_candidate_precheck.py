"""Cheap read-only precheck for a Phase 4 hypothesis file.

Purpose:
- confirm the hypothesis scope resolves to at least one raw discovery combo
- show the exact accepted combo(s)
- report current experimental/validated row counts for the hypothesis SHA

This is a process gate, not a discovery writer. It reads only canonical
`daily_features` + `orb_outcomes` and the strategy tables for blast-radius
awareness. Use it before a real Phase 4 write so weak or mis-scoped
hypotheses die cheaply.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.asset_configs import get_enabled_sessions
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import (
    ENTRY_MODELS,
    SKIP_ENTRY_MODELS,
    STOP_MULTIPLIERS,
    WF_START_OVERRIDE,
    get_filters_for_grid,
    is_e2_lookahead_filter,
)
from trading_app.hypothesis_loader import (
    extract_scope_predicate,
    load_hypothesis_metadata,
)
from trading_app.outcome_builder import CONFIRM_BARS_OPTIONS, RR_TARGETS
from trading_app.strategy_discovery import (
    _build_filter_day_sets,
    _inject_hypothesis_filters,
    _load_daily_features,
    _load_outcomes_bulk,
)


@dataclass(frozen=True)
class AcceptedCombo:
    orb_label: str
    filter_key: str
    filter_type: str
    entry_model: str
    rr_target: float
    confirm_bars: int
    stop_multiplier: float
    n_preholdout_outcomes: int


def collect_scope_hits(
    *,
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    orb_minutes: int,
    holdout_date: date,
    hypothesis_file: Path,
) -> tuple[str, list[AcceptedCombo]]:
    meta = load_hypothesis_metadata(hypothesis_file)
    scope = extract_scope_predicate(meta, instrument=instrument)
    hypothesis_sha = str(meta["sha"])

    sessions = get_enabled_sessions(instrument)
    effective_start = WF_START_OVERRIDE.get(instrument)
    features = _load_daily_features(con, instrument, orb_minutes, effective_start, holdout_date)

    all_grid_filters: dict[str, object] = {}
    for session in sessions:
        all_grid_filters.update(get_filters_for_grid(instrument, session))

    hypothesis_extra_by_session: dict[str, dict[str, object]] = {s: {} for s in sessions}
    _inject_hypothesis_filters(
        scope_predicate=scope,
        sessions=sessions,
        all_grid_filters=all_grid_filters,
        hypothesis_extra_by_session=hypothesis_extra_by_session,
    )

    filter_days = _build_filter_day_sets(features, sessions, all_grid_filters)
    outcomes_by_key = _load_outcomes_bulk(
        con,
        instrument,
        orb_minutes,
        sessions,
        ENTRY_MODELS,
        holdout_date=holdout_date,
        start_date=effective_start,
    )

    hits: list[AcceptedCombo] = []
    for orb_label in sessions:
        if orb_label not in scope.allowed_sessions():
            continue
        session_filters = dict(get_filters_for_grid(instrument, orb_label))
        session_filters.update(hypothesis_extra_by_session.get(orb_label, {}))
        for filter_key, strategy_filter in session_filters.items():
            filter_type = getattr(strategy_filter, "filter_type", "")
            if filter_type not in scope.allowed_filter_types():
                continue
            matching_day_set = filter_days[(filter_key, orb_label)]
            for entry_model in ENTRY_MODELS:
                if entry_model in SKIP_ENTRY_MODELS:
                    continue
                if entry_model == "E2" and is_e2_lookahead_filter(filter_key):
                    continue
                if entry_model not in scope.allowed_entry_models():
                    continue
                for rr_target in RR_TARGETS:
                    if rr_target not in scope.allowed_rr_targets():
                        continue
                    for confirm_bars in CONFIRM_BARS_OPTIONS:
                        if entry_model in ("E2", "E3") and confirm_bars > 1:
                            continue
                        if confirm_bars not in scope.allowed_confirm_bars():
                            continue
                        if not matching_day_set:
                            continue
                        all_outcomes = outcomes_by_key.get((orb_label, entry_model, rr_target, confirm_bars), [])
                        outcomes = [row for row in all_outcomes if row["trading_day"] in matching_day_set]
                        if not outcomes:
                            continue
                        for stop_multiplier in STOP_MULTIPLIERS:
                            if stop_multiplier not in scope.allowed_stop_multipliers():
                                continue
                            if not scope.accepts(
                                orb_label=orb_label,
                                filter_type=filter_type,
                                entry_model=entry_model,
                                rr_target=float(rr_target),
                                confirm_bars=int(confirm_bars),
                                stop_multiplier=float(stop_multiplier),
                            ):
                                continue
                            hits.append(
                                AcceptedCombo(
                                    orb_label=orb_label,
                                    filter_key=filter_key,
                                    filter_type=filter_type,
                                    entry_model=entry_model,
                                    rr_target=float(rr_target),
                                    confirm_bars=int(confirm_bars),
                                    stop_multiplier=float(stop_multiplier),
                                    n_preholdout_outcomes=len(outcomes),
                                )
                            )
    return hypothesis_sha, hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Phase 4 hypothesis precheck")
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--orb-minutes", type=int, default=5)
    parser.add_argument("--holdout-date", type=date.fromisoformat, default=date(2026, 1, 1))
    parser.add_argument("--hypothesis-file", type=Path, required=True)
    args = parser.parse_args()

    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        sha, hits = collect_scope_hits(
            con=con,
            instrument=args.instrument,
            orb_minutes=args.orb_minutes,
            holdout_date=args.holdout_date,
            hypothesis_file=args.hypothesis_file,
        )
        exp_n = con.execute(
            "SELECT COUNT(*) FROM experimental_strategies WHERE hypothesis_file_sha = ?",
            [sha],
        ).fetchone()[0]
        val_n = con.execute(
            """
            SELECT COUNT(*) FROM validated_setups
            WHERE promoted_from IN (
                SELECT strategy_id
                FROM experimental_strategies
                WHERE hypothesis_file_sha = ?
            )
            """,
            [sha],
        ).fetchone()[0]

    print(f"hypothesis_sha={sha}")
    print(f"accepted_raw_trials={len(hits)}")
    print(f"experimental_rows_with_sha={exp_n}")
    print(f"validated_rows_from_sha={val_n}")
    for hit in hits:
        print(asdict(hit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
