"""Fail-closed pre-trade feature contracts for meta-label sizing research."""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.session_guard import is_feature_safe

_BANNED_TOKENS = (
    "rel_vol_",
    "_break_dir",
    "_break_ts",
    "_break_delay",
    "_break_bar_",
    "_double_break",
    "daily_high",
    "daily_low",
    "daily_close",
    "day_type",
)

_LOCAL_SAFE_COLUMNS = frozenset(
    {
        "atr_vel_regime",
        "gap_type",
        "prev_day_direction",
        "day_of_week",
        "is_nfp_day",
        "is_opex_day",
        "is_friday",
        "is_monday",
        "is_tuesday",
    }
)


@dataclass(frozen=True)
class MetaLabelFeatureContract:
    """A pre-trade-safe feature contract for one target session."""

    target_session: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    banned_tokens: tuple[str, ...] = _BANNED_TOKENS

    @property
    def all_features(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features

    def validate(self) -> "MetaLabelFeatureContract":
        seen: set[str] = set()
        for column in self.all_features:
            if column in seen:
                raise ValueError(f"Duplicate feature in contract: {column}")
            seen.add(column)
            for token in self.banned_tokens:
                if token in column:
                    raise ValueError(
                        f"Feature '{column}' violates banned-token rule '{token}'"
                    )
            if not (
                column in _LOCAL_SAFE_COLUMNS
                or is_feature_safe(column, self.target_session)
            ):
                raise ValueError(
                    f"Feature '{column}' is not session-safe for {self.target_session}"
                )
        return self


_CME_REOPEN_V1 = MetaLabelFeatureContract(
    target_session="CME_REOPEN",
    numeric_features=(
        "atr_20",
        "atr_20_pct",
        "atr_vel_ratio",
        "gap_open_points",
        "prev_day_range",
        "orb_CME_REOPEN_size",
        "orb_CME_REOPEN_volume",
    ),
    categorical_features=(
        "atr_vel_regime",
        "gap_type",
        "prev_day_direction",
        "day_of_week",
        "is_nfp_day",
        "is_opex_day",
        "is_friday",
        "is_monday",
        "is_tuesday",
    ),
).validate()

_TOKYO_OPEN_V1 = MetaLabelFeatureContract(
    target_session="TOKYO_OPEN",
    numeric_features=(
        "atr_20",
        "atr_20_pct",
        "atr_vel_ratio",
        "gap_open_points",
        "prev_day_range",
        "orb_CME_REOPEN_size",
        "orb_CME_REOPEN_volume",
        "orb_TOKYO_OPEN_size",
        "orb_TOKYO_OPEN_volume",
    ),
    categorical_features=(
        "atr_vel_regime",
        "gap_type",
        "prev_day_direction",
        "day_of_week",
        "is_nfp_day",
        "is_opex_day",
        "is_friday",
        "is_monday",
        "is_tuesday",
    ),
).validate()


def get_feature_contract(target_session: str) -> MetaLabelFeatureContract:
    """Return the locked feature contract for a target session."""
    if target_session == "CME_REOPEN":
        return _CME_REOPEN_V1
    if target_session == "TOKYO_OPEN":
        return _TOKYO_OPEN_V1
    raise ValueError(
        f"No locked meta-label feature contract exists for session '{target_session}'"
    )
