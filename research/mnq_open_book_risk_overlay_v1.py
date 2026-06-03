#!/usr/bin/env python3
"""Risk-overlay test for MNQ NYSE_OPEN + US_DATA_1000 open books.

This follows the capital-fit failure in mnq_usdata_capital_fit_v1. It asks:
can structural overlays make the fixed open-book family account-fit at one MNQ
contract per leg under topstep_50k_mnq_auto?

Canonical inputs: orb_outcomes + daily_features + repo-owned risk/profile rules.
Selection window: trading_day < HOLDOUT_SACRED_FROM.
Monitoring window: trading_day >= HOLDOUT_SACRED_FROM, descriptive only.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.cost_model import get_cost_spec
from pipeline.paths import GOLD_DB_PATH
from research.best_own_strategy_scan_v1 import _filter_mask, _fmt, _strategy_name
from research.mnq_usdata_capital_fit_v1 import (
    DD_BUDGET_FRACTION,
    PROFILE_ID,
    SURVIVAL_FLOOR,
    _max_drawdown,
    _simulate_survival,
)
from research.mnq_usdata_rr_leg_choice_v1 import ANCHOR_LEG, _book_name
from trading_app.config import apply_tight_stop
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.prop_profiles import get_account_tier, get_profile
from trading_app.topstep_scaling_plan import lots_for_position, max_lots_for_xfa

SYMBOL = "MNQ"
ORB_MINUTES = 15
RESULT_DIR = Path("docs/audit/results")
PREREG_PATH = Path("docs/audit/hypotheses/2026-06-01-mnq-open-book-risk-overlay-v1.yaml")
RESULT_DOC = RESULT_DIR / "2026-06-01-mnq-open-book-risk-overlay-v1.md"
CANDIDATES_CSV = RESULT_DIR / "2026-06-01-mnq-open-book-risk-overlay-v1-candidates.csv"


@dataclass(frozen=True)
class BookShape:
    name: str
    us_data_rr: float
    us_data_filter: str


@dataclass(frozen=True)
class OverlaySpec:
    name: str
    stop_multiplier: float = 1.0
    risk_cap_dollars: float | None = None
    realized_loss_throttle: bool = False


BOOK_SHAPES = (
    BookShape("CURRENT_RR2_COST_LT10", 2.0, "COST_LT10"),
    BookShape("RAW_ANNUAL_RR2_NO_FILTER", 2.0, "NO_FILTER"),
    BookShape("LOW_DD_RR1_NO_FILTER", 1.0, "NO_FILTER"),
    BookShape("COMPROMISE_RR1_5_NO_FILTER", 1.5, "NO_FILTER"),
)

OVERLAYS = (
    OverlaySpec("RAW"),
    OverlaySpec("STOP_075", stop_multiplier=0.75),
    OverlaySpec("RISK_CAP_225", risk_cap_dollars=225.0),
    OverlaySpec("RISK_CAP_300", risk_cap_dollars=300.0),
    OverlaySpec("STOP_075_RISK_CAP_225", stop_multiplier=0.75, risk_cap_dollars=225.0),
    OverlaySpec("STOP_075_RISK_CAP_300", stop_multiplier=0.75, risk_cap_dollars=300.0),
    OverlaySpec("REALIZED_LOSS_THROTTLE", realized_loss_throttle=True),
)

DECLARED_K = len(BOOK_SHAPES) * len(OVERLAYS)


def _load_base(con: duckdb.DuckDBPyConnection, session: str, rr: float) -> pd.DataFrame:
    size_col = f"orb_{session}_size"
    volume_col = f"orb_{session}_volume"
    sql = f"""
        SELECT
            o.trading_day,
            o.entry_ts,
            o.exit_ts,
            o.outcome,
            COALESCE(o.pnl_r, o.ts_pnl_r, 0.0) AS pnl_r,
            o.pnl_r IS NULL AS pnl_r_was_null,
            o.entry_price,
            o.stop_price,
            o.risk_dollars,
            o.mae_r,
            df.{size_col} AS orb_size,
            df.{volume_col} AS orb_volume,
            df.atr_20_pct,
            df.atr_vel_ratio
        FROM orb_outcomes o
        INNER JOIN daily_features df
            ON o.symbol = df.symbol
           AND o.trading_day = df.trading_day
           AND o.orb_minutes = df.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.rr_target = ?
          AND o.confirm_bars = 1
          AND o.entry_model = 'E2'
        ORDER BY o.trading_day, o.entry_ts
    """
    df = con.execute(sql, [SYMBOL, session, ORB_MINUTES, rr]).fetchdf()
    if df.empty:
        return df
    df["symbol"] = SYMBOL
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    entry = df["entry_price"].astype(float)
    stop = df["stop_price"].astype(float)
    df["direction"] = np.select([entry > stop, entry < stop], ["long", "short"], default=None)
    df[size_col] = df["orb_size"]
    spec = get_cost_spec(SYMBOL)
    raw_risk = df["orb_size"].astype(float) * spec.point_value
    df["cost_ratio"] = spec.total_friction / (raw_risk + spec.total_friction)
    return df


def _load_leg(
    con: duckdb.DuckDBPyConnection,
    *,
    session: str,
    rr: float,
    filter_name: str,
    stop_multiplier: float,
    risk_cap_dollars: float | None,
) -> pd.DataFrame:
    base = _load_base(con, session, rr)
    if base.empty:
        return pd.DataFrame(columns=["trading_day", "entry_ts", "exit_ts", "pnl_r", "pnl_dollars_1ct", "risk_dollars"])
    mask = _filter_mask(base, filter_name, session).fillna(False)
    filtered = base[mask].copy()
    if risk_cap_dollars is not None:
        effective_risk = filtered["risk_dollars"].astype(float) * stop_multiplier
        filtered = filtered[effective_risk <= risk_cap_dollars].copy()
    if stop_multiplier < 1.0 and not filtered.empty:
        adjusted = apply_tight_stop(filtered.to_dict("records"), stop_multiplier, get_cost_spec(SYMBOL))
        filtered = pd.DataFrame(adjusted)
    filtered["risk_dollars"] = pd.to_numeric(filtered["risk_dollars"], errors="coerce").fillna(0.0)
    filtered["pnl_r"] = pd.to_numeric(filtered["pnl_r"], errors="coerce").fillna(0.0)
    filtered["pnl_dollars_1ct"] = filtered["pnl_r"].astype(float) * filtered["risk_dollars"].astype(float)
    filtered["leg"] = _strategy_name(session, ORB_MINUTES, rr, filter_name)
    return filtered[
        [
            "trading_day",
            "entry_ts",
            "exit_ts",
            "outcome",
            "pnl_r",
            "pnl_dollars_1ct",
            "risk_dollars",
            "pnl_r_was_null",
            "leg",
        ]
    ].copy()


def _apply_realized_loss_throttle(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    kept: list[pd.Series] = []
    for _day, day in trades.sort_values(["trading_day", "entry_ts"]).groupby("trading_day", sort=True):
        realized_losses: list[pd.Timestamp] = []
        for _, row in day.iterrows():
            entry_ts = pd.Timestamp(row["entry_ts"]) if pd.notna(row["entry_ts"]) else pd.Timestamp.max
            if any(loss_ts <= entry_ts for loss_ts in realized_losses):
                continue
            kept.append(row)
            exit_ts = pd.Timestamp(row["exit_ts"]) if pd.notna(row["exit_ts"]) else pd.Timestamp.max
            if float(row["pnl_dollars_1ct"]) < 0.0:
                realized_losses.append(exit_ts)
    if not kept:
        return trades.iloc[0:0].copy()
    return pd.DataFrame(kept).reset_index(drop=True)


def _daily_book(trades: pd.DataFrame, calendar: pd.Series) -> pd.DataFrame:
    days = sorted(pd.to_datetime(calendar).dt.date.unique())
    book = pd.DataFrame({"trading_day": days})
    if trades.empty:
        book["pnl_dollars_1ct"] = 0.0
        book["pnl_r"] = 0.0
        book["active_trades"] = 0
        return book
    grouped = (
        trades.groupby("trading_day", as_index=False)
        .agg(
            pnl_dollars_1ct=("pnl_dollars_1ct", "sum"),
            pnl_r=("pnl_r", "sum"),
            active_trades=("pnl_r", "count"),
        )
        .copy()
    )
    book = book.merge(grouped, on="trading_day", how="left")
    book[["pnl_dollars_1ct", "pnl_r", "active_trades"]] = book[
        ["pnl_dollars_1ct", "pnl_r", "active_trades"]
    ].fillna(0.0)
    book["active_trades"] = book["active_trades"].astype(int)
    return book


def _score_candidate(
    *,
    book_shape: BookShape,
    overlay: OverlaySpec,
    book: pd.DataFrame,
    trades_before_throttle: int,
    trades_after_overlay: int,
    null_pnl_count: int,
) -> dict[str, object]:
    profile = get_profile(PROFILE_ID)
    tier = get_account_tier(profile.firm, profile.account_size)
    dd_limit = float(tier.max_dd)
    daily_loss_limit = float(profile.daily_loss_dollars or tier.daily_loss_limit or dd_limit)
    freeze_at_balance = dd_limit + 100.0 if profile.is_express_funded else profile.account_size + dd_limit + 100.0
    day1_lots = max_lots_for_xfa(profile.account_size, 0.0) if profile.is_express_funded else math.inf
    is_book = book[book["trading_day"] < HOLDOUT_SACRED_FROM].copy()
    monitor = book[book["trading_day"] >= HOLDOUT_SACRED_FROM].copy()
    vals = is_book["pnl_dollars_1ct"].to_numpy(dtype=float)
    annual = float(np.nanmean(vals) * 252.0) if vals.size else float("nan")
    dd = _max_drawdown(vals)
    hist_belt_breaches = int(np.sum(vals <= -daily_loss_limit)) if vals.size else 0
    scaling_lots = lots_for_position(SYMBOL, 2)
    survival = _simulate_survival(
        vals,
        contracts_per_leg=1,
        dd_limit=dd_limit,
        daily_loss_limit=daily_loss_limit,
        freeze_at_balance=freeze_at_balance,
    )
    profile_safe = (
        annual > 0.0
        and scaling_lots <= day1_lots
        and math.isfinite(dd)
        and dd <= dd_limit * DD_BUDGET_FRACTION
        and hist_belt_breaches == 0
        and survival["operational_survival"] >= SURVIVAL_FLOOR
    )
    return {
        "candidate": f"{book_shape.name}__{overlay.name}",
        "book_shape": book_shape.name,
        "book": _book_name(book_shape.us_data_rr, book_shape.us_data_filter),
        "overlay": overlay.name,
        "us_data_rr": book_shape.us_data_rr,
        "us_data_filter": book_shape.us_data_filter,
        "stop_multiplier": overlay.stop_multiplier,
        "risk_cap_dollars": overlay.risk_cap_dollars,
        "realized_loss_throttle": overlay.realized_loss_throttle,
        "n_is_days": int(len(is_book)),
        "active_trade_days_is": int((is_book["active_trades"] > 0).sum()),
        "trades_before_throttle": trades_before_throttle,
        "trades_after_overlay": trades_after_overlay,
        "skipped_trades": trades_before_throttle - trades_after_overlay,
        "null_pnl_count": null_pnl_count,
        "annual_dollars": annual,
        "max_dd_dollars": dd,
        "worst_day_dollars": float(np.nanmin(vals)) if vals.size else float("nan"),
        "hist_daily_belt_breaches": hist_belt_breaches,
        "operational_survival": survival["operational_survival"],
        "trailing_dd_breach_mc": survival["trailing_dd_breach"],
        "daily_loss_breach_mc": survival["daily_loss_breach"],
        "mean_2026_dollars": float(monitor["pnl_dollars_1ct"].mean()) if len(monitor) else float("nan"),
        "profile_safe": profile_safe,
        "verdict": "NARROW" if profile_safe else "KILL",
    }


def run(db_path: Path = GOLD_DB_PATH) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with duckdb.connect(str(db_path), read_only=True) as con:
        anchor_calendar = _load_base(con, "NYSE_OPEN", 2.0)["trading_day"]
        us_calendars = {shape.name: _load_base(con, "US_DATA_1000", shape.us_data_rr)["trading_day"] for shape in BOOK_SHAPES}
        for shape in BOOK_SHAPES:
            for overlay in OVERLAYS:
                anchor = _load_leg(
                    con,
                    session="NYSE_OPEN",
                    rr=2.0,
                    filter_name="COST_LT10",
                    stop_multiplier=overlay.stop_multiplier,
                    risk_cap_dollars=overlay.risk_cap_dollars,
                )
                us_leg = _load_leg(
                    con,
                    session="US_DATA_1000",
                    rr=shape.us_data_rr,
                    filter_name=shape.us_data_filter,
                    stop_multiplier=overlay.stop_multiplier,
                    risk_cap_dollars=overlay.risk_cap_dollars,
                )
                trades = pd.concat([anchor, us_leg], ignore_index=True)
                trades_before = len(trades)
                if overlay.realized_loss_throttle:
                    trades = _apply_realized_loss_throttle(trades)
                calendar = pd.concat([anchor_calendar, us_calendars[shape.name]], ignore_index=True)
                book = _daily_book(trades, calendar)
                rows.append(
                    _score_candidate(
                        book_shape=shape,
                        overlay=overlay,
                        book=book,
                        trades_before_throttle=trades_before,
                        trades_after_overlay=len(trades),
                        null_pnl_count=int(trades["pnl_r_was_null"].sum()) if "pnl_r_was_null" in trades else 0,
                    )
                )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["profile_safe", "annual_dollars", "max_dd_dollars"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _write_table(rows: pd.DataFrame, columns: list[str]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    align = "| " + " | ".join("---" if col in {"candidate", "book", "overlay", "verdict"} else "---:" for col in columns) + " |"
    lines = [header, align]
    for _, row in rows.iterrows():
        parts: list[str] = []
        for col in columns:
            value = row[col]
            if col in {"candidate", "book", "overlay", "verdict"}:
                parts.append(f"`{value}`")
            elif col in {"n_is_days", "active_trade_days_is", "hist_daily_belt_breaches", "skipped_trades"}:
                parts.append(str(int(value)))
            else:
                parts.append(_fmt(value, 4))
        lines.append("| " + " | ".join(parts) + " |")
    return lines


def write_report(results: pd.DataFrame) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(CANDIDATES_CSV, index=False)
    profile = get_profile(PROFILE_ID)
    tier = get_account_tier(profile.firm, profile.account_size)
    safe = results[results["profile_safe"] == True]  # noqa: E712 - pandas scalar comparison
    winner = safe.iloc[0] if not safe.empty else None
    best_raw = results.sort_values("annual_dollars", ascending=False).iloc[0]
    best_dd = results[results["annual_dollars"] > 0].sort_values("max_dd_dollars", ascending=True).iloc[0]
    cols = [
        "candidate",
        "book",
        "overlay",
        "n_is_days",
        "active_trade_days_is",
        "skipped_trades",
        "annual_dollars",
        "max_dd_dollars",
        "worst_day_dollars",
        "hist_daily_belt_breaches",
        "operational_survival",
        "mean_2026_dollars",
        "verdict",
    ]
    lines = [
        "# MNQ Open Book Risk Overlay v1",
        "",
        f"**Prereg:** `{PREREG_PATH}`",
        "**Status:** conditional execution/filter/account-fit test; no deployment claim.",
        f"**Family K:** `{DECLARED_K}` fixed candidates.",
        "**Selection window:** `< 2026-01-01`; 2026 rows are descriptive only.",
        f"**Profile:** `{PROFILE_ID}`; daily belt `${float(profile.daily_loss_dollars or 0):.0f}`, max DD `${float(tier.max_dd):.0f}`.",
        "",
        "## Scope",
        "",
        (
            f"[MEASURED] Profile-safe overlay winner is `{winner['candidate']}` with annual "
            f"${_fmt(winner['annual_dollars'], 2)}, DD ${_fmt(winner['max_dd_dollars'], 2)}, "
            f"daily-belt breaches {int(winner['hist_daily_belt_breaches'])}, survival "
            f"{_fmt(winner['operational_survival'])}."
            if winner is not None
            else "[MEASURED] No overlay candidate is profile-safe under the active account constraints."
        ),
        f"[MEASURED] Highest annual-dollar candidate is `{best_raw['candidate']}` with annual ${_fmt(best_raw['annual_dollars'], 2)}, DD ${_fmt(best_raw['max_dd_dollars'], 2)}, daily-belt breaches {int(best_raw['hist_daily_belt_breaches'])}.",
        f"[MEASURED] Lowest positive-DD candidate is `{best_dd['candidate']}` with annual ${_fmt(best_dd['annual_dollars'], 2)}, DD ${_fmt(best_dd['max_dd_dollars'], 2)}, daily-belt breaches {int(best_dd['hist_daily_belt_breaches'])}.",
        "",
        "## Reproduction",
        "",
        f"- Front door: `python scripts/tools/prereg_front_door.py --hypothesis-file {PREREG_PATH} --execute --runner research/mnq_open_book_risk_overlay_v1.py --format text`",
        f"- Candidate CSV: `{CANDIDATES_CSV}`",
        "- DB mode: read-only canonical `gold.db` via `pipeline.paths.GOLD_DB_PATH`.",
        "",
        "## Candidate Ranking",
        "",
        *_write_table(results, cols),
        "",
        "## Grounding",
        "",
        "- `docs/institutional/conditional-edge-framework.md`: this is an execution/filter/allocator role test, so policy/account EV and drawdown matter more than selected-trade mean.",
        "- `docs/institutional/pre_registered_criteria.md` Criterion 11: funded deployment requires account-death Monte Carlo with prop-firm rules and >=70% 90-day survival.",
        "- `docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md`: risk analysis, portfolio construction, and bet-sizing are legitimate finite-data applications when trial count is tracked.",
        "- `TRADING_RULES.md`: active lanes use E2/CB1, 0.75x stop sizing, and explicit ORB/risk caps; stop-multiplier math is repo-canonical via `apply_tight_stop`.",
        "",
        "## Caveats",
        "",
        "- Stop-multiplier candidates use canonical MAE-based tight-stop math, but do not infer exact intra-bar kill timestamps.",
        "- Realized-loss throttle only skips later trades when a prior losing trade has an exit timestamp before the later entry. It is deliberately not combined with tight-stop candidates because tight-stop kill timestamps are unavailable.",
        "- Risk caps are structural fractions of the $450 daily belt, not optimized thresholds.",
        "- This is one-contract-per-leg account-fit research, not a live allocation patch.",
        "",
        "## Verdict",
        "",
        (
            "`NARROW` for the profile-safe winner. It still needs deployment-readiness translation before any live use."
            if winner is not None
            else "`KILL` for every row: risk caps can remove daily-belt breaches, but the two-leg book still exceeds the drawdown budget. The correct next move is lower-risk lane replacement, not more sizing on this two-leg book."
        ),
        "",
        "SURVIVED SCRUTINY: finite K=28, pre-entry risk caps, canonical stop-multiplier math, no 2026 tuning, no silent pnl-null dropout.",
        "DID NOT SURVIVE: no live/deployment claim from this research artifact alone.",
    ]
    RESULT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=GOLD_DB_PATH)
    args = parser.parse_args()
    results = run(args.db)
    write_report(results)
    print(f"wrote {len(results)} candidates to {CANDIDATES_CSV}")
    print(f"wrote report to {RESULT_DOC}")


if __name__ == "__main__":
    main()
