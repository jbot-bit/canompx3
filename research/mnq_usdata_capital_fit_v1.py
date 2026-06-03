#!/usr/bin/env python3
"""Capital-fit accounting for the MNQ US_DATA_1000 leg-choice books.

This is not a new alpha search. It fixes the 15 books from
mnq_usdata_rr_leg_choice_v1 and asks which one fits the active
topstep_50k_mnq_auto account constraints best after contract sizing.

Canonical inputs: orb_outcomes + daily_features + repo-owned prop profile rules.
Selection window: trading_day < HOLDOUT_SACRED_FROM.
Monitoring window: trading_day >= HOLDOUT_SACRED_FROM, descriptive only.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.paths import GOLD_DB_PATH
from research.best_own_strategy_scan_v1 import CandidateSpec, _filter_mask, _fmt, _load_base, _strategy_name
from research.mnq_open_book_validation_v1 import _objective_score
from research.mnq_usdata_rr_leg_choice_v1 import (
    ANCHOR_LEG,
    US_DATA_FILTERS,
    US_DATA_RRS,
    _book_name,
)
from research.mnq_usdata_rr_leg_choice_v1 import (
    BOOK_CSV as LEG_CHOICE_BOOK_CSV,
)
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.prop_profiles import get_account_tier, get_profile
from trading_app.topstep_scaling_plan import lots_for_position, max_lots_for_xfa

PROFILE_ID = "topstep_50k_mnq_auto"
SYMBOL = "MNQ"
ORB_MINUTES = 15
MAX_CONTRACTS_PER_LEG = 10
DD_BUDGET_FRACTION = 0.80
SURVIVAL_FLOOR = 0.70
HORIZON_DAYS = 90
N_PATHS = 10_000
SEED = 20_260_601

RESULT_DIR = Path("docs/audit/results")
PREREG_PATH = Path("docs/audit/hypotheses/2026-06-01-mnq-usdata-capital-fit-v1.yaml")
RESULT_DOC = RESULT_DIR / "2026-06-01-mnq-usdata-capital-fit-v1.md"
BOOKS_CSV = RESULT_DIR / "2026-06-01-mnq-usdata-capital-fit-v1-books.csv"
SIZING_CSV = RESULT_DIR / "2026-06-01-mnq-usdata-capital-fit-v1-sizing.csv"


def _max_drawdown(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return float("nan")
    curve = np.cumsum(vals)
    peak = np.maximum.accumulate(curve)
    return float((peak - curve).max())


def _load_leg(
    con: duckdb.DuckDBPyConnection,
    *,
    session: str,
    rr: float,
    filter_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = _load_base(con, session, ORB_MINUTES, rr)
    if base.empty:
        empty = pd.DataFrame(columns=["trading_day", "pnl_r", "pnl_dollars_1ct", "risk_dollars"])
        return base, empty
    mask = _filter_mask(base, filter_name, session).fillna(False)
    filtered = base[mask].copy()
    filtered["risk_dollars"] = pd.to_numeric(filtered["risk_dollars"], errors="coerce").fillna(0.0)
    filtered["pnl_dollars_1ct"] = filtered["pnl_r"].astype(float) * filtered["risk_dollars"].astype(float)
    return base, filtered[["trading_day", "pnl_r", "pnl_dollars_1ct", "risk_dollars"]].copy()


def _book_daily(calendar: pd.Series, legs: list[pd.DataFrame]) -> pd.DataFrame:
    book = pd.DataFrame({"trading_day": sorted(pd.to_datetime(calendar).dt.date.unique())})
    book["pnl_r"] = 0.0
    book["pnl_dollars_1ct"] = 0.0
    book["active_trades"] = 0
    for idx, leg in enumerate(legs, start=1):
        tmp = (
            leg.groupby("trading_day", as_index=False)
            .agg(
                **{
                    f"leg_{idx}_r": ("pnl_r", "sum"),
                    f"leg_{idx}_dollars": ("pnl_dollars_1ct", "sum"),
                    f"leg_{idx}_trades": ("pnl_r", "count"),
                }
            )
            .copy()
        )
        book = book.merge(tmp, on="trading_day", how="left")
        for col in (f"leg_{idx}_r", f"leg_{idx}_dollars", f"leg_{idx}_trades"):
            book[col] = book[col].fillna(0.0)
        book["pnl_r"] += book[f"leg_{idx}_r"]
        book["pnl_dollars_1ct"] += book[f"leg_{idx}_dollars"]
        book["active_trades"] += book[f"leg_{idx}_trades"].astype(int)
    return book


def _simulate_survival(
    values: np.ndarray,
    *,
    contracts_per_leg: int,
    dd_limit: float,
    daily_loss_limit: float,
    freeze_at_balance: float,
) -> dict[str, float]:
    vals = values[np.isfinite(values)] * contracts_per_leg
    if vals.size == 0:
        return {"operational_survival": 0.0, "trailing_dd_breach": 1.0, "daily_loss_breach": 1.0}
    rng = np.random.default_rng(SEED + contracts_per_leg)
    samples = rng.choice(vals, size=(N_PATHS, HORIZON_DAYS), replace=True)
    operational = 0
    dd_breaches = 0
    daily_breaches = 0
    for path in samples:
        balance = 0.0
        hwm = 0.0
        frozen = False
        breached = False
        for day_pnl in path:
            if day_pnl <= -daily_loss_limit:
                daily_breaches += 1
                breached = True
                break
            low_balance = balance + min(float(day_pnl), 0.0)
            if hwm - low_balance >= dd_limit:
                dd_breaches += 1
                breached = True
                break
            balance += float(day_pnl)
            if not frozen and balance > hwm:
                hwm = balance
                if hwm >= freeze_at_balance:
                    frozen = True
        if not breached:
            operational += 1
    return {
        "operational_survival": operational / N_PATHS,
        "trailing_dd_breach": dd_breaches / N_PATHS,
        "daily_loss_breach": daily_breaches / N_PATHS,
    }


def _score_book(book_name: str, book: pd.DataFrame, *, rr: float, filter_name: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    profile = get_profile(PROFILE_ID)
    tier = get_account_tier(profile.firm, profile.account_size)
    dd_limit = float(tier.max_dd)
    daily_loss_limit = float(profile.daily_loss_dollars or tier.daily_loss_limit or dd_limit)
    freeze_at_balance = dd_limit + 100.0 if profile.is_express_funded else profile.account_size + dd_limit + 100.0
    day1_lots = max_lots_for_xfa(profile.account_size, 0.0) if profile.is_express_funded else math.inf

    is_book = book[book["trading_day"] < HOLDOUT_SACRED_FROM].copy()
    monitor = book[book["trading_day"] >= HOLDOUT_SACRED_FROM].copy()
    vals_1ct = is_book["pnl_dollars_1ct"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for contracts in range(1, MAX_CONTRACTS_PER_LEG + 1):
        scaled = vals_1ct * contracts
        annual_dollars = float(np.nanmean(scaled) * 252.0) if scaled.size else float("nan")
        dd_dollars = _max_drawdown(scaled)
        worst_day = float(np.nanmin(scaled)) if scaled.size else float("nan")
        scaling_lots = lots_for_position(SYMBOL, contracts * 2)
        scaling_ok = scaling_lots <= day1_lots
        hist_dd_ok = math.isfinite(dd_dollars) and dd_dollars <= dd_limit * DD_BUDGET_FRACTION
        hist_daily_loss_breaches = int(np.sum(scaled <= -daily_loss_limit)) if scaled.size else 0
        survival = _simulate_survival(
            vals_1ct,
            contracts_per_leg=contracts,
            dd_limit=dd_limit,
            daily_loss_limit=daily_loss_limit,
            freeze_at_balance=freeze_at_balance,
        )
        profile_safe = (
            scaling_ok
            and hist_dd_ok
            and hist_daily_loss_breaches == 0
            and survival["operational_survival"] >= SURVIVAL_FLOOR
        )
        rows.append(
            {
                "strategy": book_name,
                "us_data_rr": rr,
                "us_data_filter": filter_name,
                "contracts_per_leg": contracts,
                "annual_dollars": annual_dollars,
                "max_dd_dollars": dd_dollars,
                "worst_day_dollars": worst_day,
                "hist_daily_loss_breaches": hist_daily_loss_breaches,
                "scaling_lots": scaling_lots,
                "scaling_ok": scaling_ok,
                "hist_dd_ok": hist_dd_ok,
                "operational_survival": survival["operational_survival"],
                "trailing_dd_breach_mc": survival["trailing_dd_breach"],
                "daily_loss_breach_mc": survival["daily_loss_breach"],
                "profile_safe": profile_safe,
            }
        )

    safe_rows = [row for row in rows if bool(row["profile_safe"])]
    best_safe = max(safe_rows, key=lambda row: (float(row["annual_dollars"]), int(row["contracts_per_leg"]))) if safe_rows else None
    base_metrics = {
        "strategy": book_name,
        "us_data_rr": rr,
        "us_data_filter": filter_name,
        "n_is_days": int(len(is_book)),
        "annual_dollars_1ct": float(np.nanmean(vals_1ct) * 252.0) if vals_1ct.size else float("nan"),
        "max_dd_dollars_1ct": _max_drawdown(vals_1ct),
        "worst_day_dollars_1ct": float(np.nanmin(vals_1ct)) if vals_1ct.size else float("nan"),
        "mean_2026_dollars_1ct": float(monitor["pnl_dollars_1ct"].mean()) if len(monitor) else float("nan"),
        "active_day_rate_is": float((is_book["active_trades"] > 0).mean()) if len(is_book) else float("nan"),
        "best_contracts_per_leg": int(best_safe["contracts_per_leg"]) if best_safe else 0,
        "profile_safe_annual_dollars": float(best_safe["annual_dollars"]) if best_safe else float("nan"),
        "profile_safe_max_dd_dollars": float(best_safe["max_dd_dollars"]) if best_safe else float("nan"),
        "profile_safe_survival": float(best_safe["operational_survival"]) if best_safe else 0.0,
        "profile_safe": best_safe is not None,
        "verdict": "NARROW" if best_safe is not None else "KILL",
    }
    base_metrics["capital_objective"] = (
        base_metrics["profile_safe_annual_dollars"] / base_metrics["profile_safe_max_dd_dollars"]
        if base_metrics["profile_safe"] and base_metrics["profile_safe_max_dd_dollars"] > 0
        else float("nan")
    )
    base_metrics["research_objective_score"] = _objective_score(
        {"annual_r": base_metrics["annual_dollars_1ct"], "dd": base_metrics["max_dd_dollars_1ct"]}
    )
    return base_metrics, rows


def run(db_path: Path = GOLD_DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    book_rows: list[dict[str, object]] = []
    sizing_rows: list[dict[str, object]] = []
    with duckdb.connect(str(db_path), read_only=True) as con:
        anchor_base, anchor = _load_leg(con, session="NYSE_OPEN", rr=2.0, filter_name="COST_LT10")
        for rr in US_DATA_RRS:
            for filter_name in US_DATA_FILTERS:
                us_base, us_leg = _load_leg(con, session="US_DATA_1000", rr=rr, filter_name=filter_name)
                calendar = pd.concat([anchor_base["trading_day"], us_base["trading_day"]], ignore_index=True)
                book = _book_daily(calendar, [anchor, us_leg])
                book_name = _book_name(rr, filter_name)
                book_row, size_rows = _score_book(book_name, book, rr=rr, filter_name=filter_name)
                book_row["leg_1"] = ANCHOR_LEG
                book_row["leg_2"] = _strategy_name("US_DATA_1000", ORB_MINUTES, rr, filter_name)
                book_rows.append(book_row)
                sizing_rows.extend(size_rows)
    books = pd.DataFrame(book_rows).sort_values(
        ["profile_safe", "capital_objective", "profile_safe_annual_dollars"],
        ascending=[False, False, False],
    )
    sizing = pd.DataFrame(sizing_rows)
    return books.reset_index(drop=True), sizing.reset_index(drop=True)


def _write_table(rows: pd.DataFrame, columns: list[str]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    align = "| " + " | ".join("---" if col in {"strategy", "us_data_filter", "verdict"} else "---:" for col in columns) + " |"
    lines = [header, align]
    for _, row in rows.iterrows():
        parts: list[str] = []
        for col in columns:
            value = row[col]
            if col in {"strategy", "us_data_filter", "verdict"}:
                parts.append(f"`{value}`")
            elif col in {"n_is_days", "best_contracts_per_leg"}:
                parts.append(str(int(value)))
            else:
                parts.append(_fmt(value, 4))
        lines.append("| " + " | ".join(parts) + " |")
    return lines


def write_report(books: pd.DataFrame, sizing: pd.DataFrame) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    safe_books = books[books["profile_safe"] == True]  # noqa: E712 - pandas scalar comparison
    raw_annual_winner = books.sort_values("annual_dollars_1ct", ascending=False).iloc[0]
    winner = safe_books.iloc[0] if not safe_books.empty else None
    annual_winner = (
        safe_books.sort_values("profile_safe_annual_dollars", ascending=False).iloc[0]
        if not safe_books.empty
        else None
    )
    current = books[(books["us_data_rr"] == 2.0) & (books["us_data_filter"] == "COST_LT10")].iloc[0]
    profile = get_profile(PROFILE_ID)
    tier = get_account_tier(profile.firm, profile.account_size)
    sizing.to_csv(SIZING_CSV, index=False)
    books.to_csv(BOOKS_CSV, index=False)
    lines = [
        "# MNQ US_DATA_1000 Capital Fit v1",
        "",
        f"**Prereg:** `{PREREG_PATH}`",
        f"**Leg-choice source:** `{LEG_CHOICE_BOOK_CSV}`",
        "**Status:** allocator/account-fit follow-up; no deployment claim.",
        "**Selection window:** `< 2026-01-01`; 2026 rows are descriptive only.",
        f"**Profile:** `{PROFILE_ID}`; daily belt `${float(profile.daily_loss_dollars or 0):.0f}`, max DD `${float(tier.max_dd):.0f}`.",
        "",
        "## Scope",
        "",
        (
            f"[MEASURED] Profile-safe capital winner is `{winner['strategy']}` at "
            f"{int(winner['best_contracts_per_leg'])} MNQ contracts per leg, annual "
            f"${_fmt(winner['profile_safe_annual_dollars'], 2)}, DD "
            f"${_fmt(winner['profile_safe_max_dd_dollars'], 2)}, survival "
            f"{_fmt(winner['profile_safe_survival'])}."
            if winner is not None
            else "[MEASURED] No book has a profile-safe contract size from 1-10 MNQ contracts per leg under the active account constraints."
        ),
        (
            f"[MEASURED] Profile-safe annual-dollar winner is `{annual_winner['strategy']}` "
            f"at {int(annual_winner['best_contracts_per_leg'])} contracts per leg, annual "
            f"${_fmt(annual_winner['profile_safe_annual_dollars'], 2)}."
            if annual_winner is not None
            else f"[MEASURED] Raw 1-contract annual-dollar winner is `{raw_annual_winner['strategy']}` with annual ${_fmt(raw_annual_winner['annual_dollars_1ct'], 2)}, but it is not profile-safe."
        ),
        f"[MEASURED] Current comparison row `{current['strategy']}` has best safe size {int(current['best_contracts_per_leg'])}, annual ${_fmt(current['profile_safe_annual_dollars'], 2)}, DD ${_fmt(current['profile_safe_max_dd_dollars'], 2)}, survival {_fmt(current['profile_safe_survival'])}.",
        "",
        "## Reproduction",
        "",
        f"- Front door: `python scripts/tools/prereg_front_door.py --hypothesis-file {PREREG_PATH} --execute --runner research/mnq_usdata_capital_fit_v1.py --format text`",
        f"- Book CSV: `{BOOKS_CSV}`",
        f"- Sizing CSV: `{SIZING_CSV}`",
        "- DB mode: read-only canonical `gold.db` via `pipeline.paths.GOLD_DB_PATH`.",
        "",
        "## Book Ranking",
        "",
        *_write_table(
            books,
            [
                "strategy",
                "us_data_rr",
                "us_data_filter",
                "n_is_days",
                "annual_dollars_1ct",
                "max_dd_dollars_1ct",
                "best_contracts_per_leg",
                "profile_safe_annual_dollars",
                "profile_safe_max_dd_dollars",
                "profile_safe_survival",
                "capital_objective",
                "mean_2026_dollars_1ct",
                "verdict",
            ],
        ),
        "",
        "## Grounding",
        "",
        "- `trading_app.prop_profiles`: active `topstep_50k_mnq_auto` account size, daily loss belt, and express-funded flag.",
        "- `trading_app.topstep_scaling_plan`: canonical MNQ micro-to-mini lot conversion and XFA day-one lot cap.",
        "- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md`: 2026 is descriptive only; no OOS tuning.",
        "",
        "## Caveats",
        "",
        "- Bootstrap is daily-close accounting; it does not replay intraday MAE/MFE path ordering.",
        "- This pass does not apply stop_multiplier=0.75, max-ORB-size caps, or a sequential daily-loss throttle; those are separate risk-overlay hypotheses.",
        "- Contract sizing is an allocator curve, not an alpha-family p-value test.",
        "- Any future `NARROW` row from this route would still inherit the upstream research-status limit from the leg-choice run; this is not live approval.",
        "",
        "## Verdict",
        "",
        (
            "`NARROW` for the profile-safe winner. Use this only as the next deployment-readiness design input, not as an auto-promotion."
            if winner is not None
            else "`KILL` for raw two-leg deployment under the active profile constraints. The next test should be a risk-overlay family: stop-multiplier, max-risk/day, or sequential daily-loss throttle."
        ),
        "",
        "SURVIVED SCRUTINY: fixed 15-book universe, repo-owned profile constraints, Topstep scaling cap, 2026 monitoring separation.",
        "DID NOT SURVIVE: no row becomes deployment-ready without a separate live/readiness translation.",
    ]
    RESULT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=GOLD_DB_PATH)
    args = parser.parse_args()
    books, sizing = run(args.db)
    write_report(books, sizing)
    print(f"wrote {len(books)} books to {BOOKS_CSV}")
    print(f"wrote {len(sizing)} sizing rows to {SIZING_CSV}")
    print(f"wrote report to {RESULT_DOC}")


if __name__ == "__main__":
    main()
