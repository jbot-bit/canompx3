"""Canonical dataset builder for one meta-label sizing family."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.meta_labeling.feature_contract import (
    MetaLabelFeatureContract,
    get_feature_contract,
)


@dataclass(frozen=True)
class MetaLabelFamilySpec:
    """A single homogeneous primary-signal family for meta-labeling."""

    symbol: str
    orb_label: str
    entry_model: str
    rr_target: float
    confirm_bars: int
    direction: str | None = None
    filter_label: str = "NO_FILTER"
    orb_minutes: int = 5
    min_orb_size: float = 0.0

    @property
    def size_column(self) -> str:
        return f"orb_{self.orb_label}_size"

    @property
    def break_dir_column(self) -> str:
        return f"orb_{self.orb_label}_break_dir"


@dataclass(frozen=True)
class DatasetBundle:
    """Full and split datasets for a single family."""

    family: MetaLabelFamilySpec
    feature_contract: MetaLabelFeatureContract
    full_frame: pd.DataFrame
    development_frame: pd.DataFrame
    forward_frame: pd.DataFrame

    @property
    def train_frame(self) -> pd.DataFrame:
        """Backward-compatible alias: training lives inside development only."""
        return self.development_frame

    @property
    def holdout_frame(self) -> pd.DataFrame:
        """Backward-compatible alias: 2026+ is forward-monitor only."""
        return self.forward_frame


PHASE1_V1_FAMILY = MetaLabelFamilySpec(
    symbol="MNQ",
    orb_label="TOKYO_OPEN",
    entry_model="E2",
    rr_target=1.5,
    confirm_bars=1,
    direction="long",
    filter_label="NO_FILTER_LONG_ONLY",
    orb_minutes=5,
    min_orb_size=0.0,
)


def _resolve_db_path(db_path: Path | None) -> Path:
    return db_path if db_path is not None else GOLD_DB_PATH


def _base_query(contract: MetaLabelFeatureContract, family: MetaLabelFamilySpec) -> str:
    feature_sql = ",\n       ".join(f"d.{column}" for column in contract.all_features)
    return f"""
        SELECT
            o.trading_day,
            o.symbol,
            o.orb_label,
            o.entry_model,
            o.rr_target,
            o.confirm_bars,
            o.orb_minutes,
            o.outcome,
            o.pnl_r,
            d.{family.break_dir_column} AS trade_direction,
            {feature_sql}
        FROM orb_outcomes o
        JOIN daily_features d
          ON d.symbol = o.symbol
         AND d.trading_day = o.trading_day
         AND d.orb_minutes = o.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.entry_model = ?
          AND o.rr_target = ?
          AND o.confirm_bars = ?
          AND o.orb_minutes = ?
          AND o.outcome IN ('win', 'loss')
          AND d.{family.size_column} >= ?
        ORDER BY o.trading_day
    """


def build_family_dataset(
    family: MetaLabelFamilySpec = PHASE1_V1_FAMILY,
    db_path: Path | None = None,
) -> DatasetBundle:
    """Load the canonical dataset and enforce the sacred holdout split.

    Important: the 2026+ slice is preserved as untouched forward-monitor data,
    not as a sufficient validation set for model selection or promotion.
    """
    contract = get_feature_contract(family.orb_label)
    resolved_db = _resolve_db_path(db_path)
    query = _base_query(contract, family)

    with duckdb.connect(str(resolved_db), read_only=True) as con:
        configure_connection(con, writing=False)
        frame = con.execute(
            query,
            [
                family.symbol,
                family.orb_label,
                family.entry_model,
                family.rr_target,
                family.confirm_bars,
                family.orb_minutes,
                family.min_orb_size,
            ],
        ).fetchdf()

    if frame.empty:
        raise ValueError(f"No canonical rows found for family {family}")

    if family.direction is not None:
        frame = frame.loc[frame["trade_direction"] == family.direction].reset_index(drop=True)

    frame["trading_day"] = pd.to_datetime(frame["trading_day"]).dt.date
    frame["target"] = (frame["pnl_r"] > 0.0).astype(int)

    development_mask = frame["trading_day"] < HOLDOUT_SACRED_FROM
    development_frame = frame.loc[development_mask].reset_index(drop=True)
    forward_frame = frame.loc[~development_mask].reset_index(drop=True)

    if development_frame.empty:
        raise ValueError("Development frame is empty after sacred holdout split")

    development_expr = float(development_frame["pnl_r"].mean())
    if development_expr <= 0.0:
        raise ValueError(
            "Meta-labeling prerequisite violated: pre-2026 development ExpR "
            f"is non-positive ({development_expr:+.6f}R)"
        )

    return DatasetBundle(
        family=family,
        feature_contract=contract,
        full_frame=frame.reset_index(drop=True),
        development_frame=development_frame,
        forward_frame=forward_frame,
    )


def summarize_dataset(bundle: DatasetBundle) -> dict:
    """Return a compact summary for pre-fit review."""
    development = bundle.development_frame
    forward = bundle.forward_frame
    target_counts = development["target"].value_counts(dropna=False).to_dict()
    outcome_counts = development["outcome"].value_counts(dropna=False).to_dict()
    wins = int(target_counts.get(1, 0))
    non_positive = int(target_counts.get(0, 0))
    ratio = None if non_positive == 0 else round(wins / non_positive, 4)
    development_year_counts = {
        str(year): int(count)
        for year, count in development.groupby(
            development["trading_day"].map(lambda td: td.year)
        ).size().items()
    }
    forward_year_counts = {
        str(year): int(count)
        for year, count in forward.groupby(
            forward["trading_day"].map(lambda td: td.year)
        ).size().items()
    }
    forward_rows = int(len(forward))

    return {
        "family": {
            "symbol": bundle.family.symbol,
            "orb_label": bundle.family.orb_label,
            "entry_model": bundle.family.entry_model,
            "rr_target": bundle.family.rr_target,
            "confirm_bars": bundle.family.confirm_bars,
            "direction": bundle.family.direction,
            "filter_label": bundle.family.filter_label,
            "orb_minutes": bundle.family.orb_minutes,
            "min_orb_size": bundle.family.min_orb_size,
        },
        "holdout_date": HOLDOUT_SACRED_FROM.isoformat(),
        "full_rows": int(len(bundle.full_frame)),
        "development_rows_pre_2026": int(len(bundle.development_frame)),
        "forward_rows_2026_plus": forward_rows,
        "feature_count": len(bundle.feature_contract.all_features),
        "feature_list": list(bundle.feature_contract.all_features),
        "development_expr_r": round(float(development["pnl_r"].mean()), 6),
        "class_balance": {
            "win_count": wins,
            "non_positive_count": non_positive,
            "win_to_non_positive_ratio": ratio,
        },
        "development_outcomes": outcome_counts,
        "development_year_counts": development_year_counts,
        "forward_year_counts": forward_year_counts,
        "forward_assessment": {
            "status": "INFORMATIONAL_ONLY",
            "reason": (
                "Current 2026 forward slice is sacred and untouched, but too thin "
                "to serve as a standalone ML validation or promotion basis."
            ),
            "forward_rows": forward_rows,
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the canonical meta-label dataset for one homogeneous family"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Override DB path (defaults to pipeline.paths.GOLD_DB_PATH)",
    )
    args = parser.parse_args()

    bundle = build_family_dataset(db_path=args.db_path)
    print(json.dumps(summarize_dataset(bundle), indent=2, default=str))


if __name__ == "__main__":
    main()
