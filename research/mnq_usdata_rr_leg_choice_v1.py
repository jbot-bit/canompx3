#!/usr/bin/env python3
"""MNQ book leg-choice test for US_DATA_1000 RR/filter variants.

This is the narrow follow-up to mnq_open_book_validation_v1:

Fixed anchor leg:
  MNQ_NYSE_OPEN_O15_E2_RR2_COST_LT10

Variable second leg:
  MNQ_US_DATA_1000_O15_E2_RR{1,1.5,2}_{NO_FILTER,COST_LT08,10,12,15}

Canonical inputs: orb_outcomes + daily_features only.
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

from pipeline.asset_configs import ASSET_CONFIGS
from pipeline.paths import GOLD_DB_PATH
from research.best_own_strategy_scan_v1 import (
    FILTERS,
    ORB_MINUTES,
    RR_TARGETS,
    SYMBOL,
    CandidateSpec,
    _add_multiple_testing,
    _bh_qvalues,
    _fmt,
    _load_base,
    _score_candidate,
    _score_returns,
    _strategy_name,
)
from research.mnq_open_book_validation_v1 import _leg_correlation, _objective_score
from trading_app.dsr import compute_dsr, compute_sr0
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

RESULT_DIR = Path("docs/audit/results")
PREREG_PATH = Path("docs/audit/hypotheses/2026-06-01-mnq-usdata-rr-leg-choice-v1.yaml")
RESULT_DOC = RESULT_DIR / "2026-06-01-mnq-usdata-rr-leg-choice-v1.md"
BOOK_CSV = RESULT_DIR / "2026-06-01-mnq-usdata-rr-leg-choice-v1-books.csv"

ANCHOR_LEG = _strategy_name("NYSE_OPEN", 15, 2.0, "COST_LT10")
US_DATA_FILTERS = ("NO_FILTER", "COST_LT08", "COST_LT10", "COST_LT12", "COST_LT15")
US_DATA_RRS = (1.0, 1.5, 2.0)
DECLARED_K = len(US_DATA_FILTERS) * len(US_DATA_RRS)
UPSTREAM_EXPLORATORY_K = len(ASSET_CONFIGS[SYMBOL]["enabled_sessions"]) * len(ORB_MINUTES) * len(RR_TARGETS) * len(FILTERS)
BH_Q = 0.05
NO_THEORY_T_THRESHOLD = 3.79


def _portfolio_series(legs: list[pd.DataFrame]) -> pd.DataFrame:
    all_days = sorted({day for leg in legs for day in leg["trading_day"].tolist()})
    book = pd.DataFrame({"trading_day": all_days})
    book["pnl_r"] = 0.0
    for idx, leg in enumerate(legs, start=1):
        col = f"leg_{idx}"
        tmp = leg.groupby("trading_day", as_index=False)["pnl_r"].sum().rename(columns={"pnl_r": col})
        book = book.merge(tmp, on="trading_day", how="left")
        book[col] = book[col].fillna(0.0)
        book["pnl_r"] = book["pnl_r"] + book[col]
    return book


def _candidate_cells_and_series(db_path: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    filtered_by_name: dict[str, pd.DataFrame] = {}
    needed: set[tuple[str, int, float, str]] = {("NYSE_OPEN", 15, 2.0, "COST_LT10")}
    needed.update(("US_DATA_1000", 15, rr, filter_name) for rr in US_DATA_RRS for filter_name in US_DATA_FILTERS)

    with duckdb.connect(str(db_path), read_only=True) as con:
        for session, orb_minutes, rr, filter_name in sorted(needed):
            base = _load_base(con, session, orb_minutes, rr)
            spec = CandidateSpec(
                strategy=_strategy_name(session, orb_minutes, rr, filter_name),
                session=session,
                orb_minutes=orb_minutes,
                rr=rr,
                filter_name=filter_name,
            )
            row, filtered = _score_candidate(base, spec)
            rows.append(row)
            filtered_by_name[spec.strategy] = filtered

    # Also compute upstream Sharpe dispersion from the full broad family so
    # inherited DSR is not understated by this narrow follow-up.
    upstream_rows: list[dict[str, object]] = []
    with duckdb.connect(str(db_path), read_only=True) as con:
        for session in ASSET_CONFIGS[SYMBOL]["enabled_sessions"]:
            for orb_minutes in ORB_MINUTES:
                for rr in RR_TARGETS:
                    base = _load_base(con, session, orb_minutes, rr)
                    for filter_name in FILTERS:
                        spec = CandidateSpec(
                            strategy=_strategy_name(session, orb_minutes, rr, filter_name),
                            session=session,
                            orb_minutes=orb_minutes,
                            rr=rr,
                            filter_name=filter_name,
                        )
                        row, _ = _score_candidate(base, spec)
                        upstream_rows.append(row)
    _add_multiple_testing(upstream_rows, UPSTREAM_EXPLORATORY_K)
    upstream = pd.DataFrame(upstream_rows)
    cells = pd.DataFrame(rows)
    return upstream, filtered_by_name | {"__cells__": cells}


def _book_name(rr: float, filter_name: str) -> str:
    rr_token = str(int(rr)) if float(rr).is_integer() else str(rr).replace(".", "_")
    return f"NYOPEN_USDATA_RR{rr_token}_{filter_name}"


def _score_books(upstream: pd.DataFrame, filtered_by_name: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rr in US_DATA_RRS:
        for filter_name in US_DATA_FILTERS:
            leg_2 = _strategy_name("US_DATA_1000", 15, rr, filter_name)
            legs = [filtered_by_name[ANCHOR_LEG], filtered_by_name[leg_2]]
            book = _portfolio_series(legs)
            row = _score_returns(book, "pnl_r", strategy=_book_name(rr, filter_name))
            row.update(
                {
                    "family": "usdata_rr_leg_choice",
                    "leg_1": ANCHOR_LEG,
                    "leg_2": leg_2,
                    "us_data_rr": rr,
                    "us_data_filter": filter_name,
                    "leg_corr_is": _leg_correlation(legs),
                    "objective_score": _objective_score(row),
                }
            )
            rows.append(row)

    books = pd.DataFrame(rows)
    books["q_family"] = _bh_qvalues([float(p) for p in books["p"]])

    family_sharpes = books["sharpe"].astype(float).to_numpy()
    family_var_sr = float(np.nanvar(family_sharpes, ddof=1)) if np.isfinite(family_sharpes).sum() > 1 else 0.0
    family_sr0 = compute_sr0(DECLARED_K, family_var_sr)

    inherited_sharpes = np.concatenate(
        [
            upstream["sharpe"].astype(float).to_numpy(),
            family_sharpes,
        ]
    )
    inherited_var_sr = float(np.nanvar(inherited_sharpes[np.isfinite(inherited_sharpes)], ddof=1))
    inherited_sr0 = compute_sr0(UPSTREAM_EXPLORATORY_K + DECLARED_K, inherited_var_sr)

    for sr0_col, dsr_col, sr0 in (
        ("sr0_family", "dsr_family", family_sr0),
        ("sr0_inherited", "dsr_inherited", inherited_sr0),
    ):
        books[sr0_col] = sr0
        books[dsr_col] = [
            compute_dsr(float(row.sharpe), sr0, int(row.n_is), float(row.skewness), float(row.kurtosis_excess))
            if math.isfinite(float(row.sharpe)) and int(row.n_is) >= 2
            else 0.0
            for row in books.itertuples(index=False)
        ]

    books["verdict"] = books.apply(_verdict, axis=1)
    return books.sort_values(["objective_score", "annual_r", "t"], ascending=[False, False, False]).reset_index(drop=True)


def _verdict(row: pd.Series) -> str:
    passes_family = (
        float(row["q_family"]) < BH_Q
        and float(row["t"]) >= NO_THEORY_T_THRESHOLD
        and float(row["dsr_family"]) > 0.95
        and float(row["wfe"]) >= 0.50
        and bool(row["era_ok"])
    )
    if not passes_family:
        return "KILL"
    if float(row["dsr_inherited"]) > 0.95:
        return "CONTINUE"
    return "NARROW"


def _write_table(rows: pd.DataFrame, columns: list[str]) -> list[str]:
    if rows.empty:
        return ["_None._"]
    header = "| " + " | ".join(columns) + " |"
    align = "| " + " | ".join("---" if col in {"strategy", "us_data_filter", "verdict", "leg_2"} else "---:" for col in columns) + " |"
    lines = [header, align]
    for _, row in rows.iterrows():
        parts: list[str] = []
        for col in columns:
            value = row[col]
            if col in {"strategy", "us_data_filter", "verdict", "leg_2"}:
                parts.append(f"`{value}`")
            elif col in {"n_is", "n_2026"}:
                parts.append(str(int(value)))
            elif col in {"p", "q_family", "dsr_family", "dsr_inherited"}:
                parts.append(_fmt(value, 6))
            else:
                parts.append(_fmt(value))
        lines.append("| " + " | ".join(parts) + " |")
    return lines


def write_report(books: pd.DataFrame) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    objective_winner = books.iloc[0]
    annual_winner = books.sort_values("annual_r", ascending=False).iloc[0]
    current = books[(books["us_data_rr"] == 2.0) & (books["us_data_filter"] == "COST_LT10")].iloc[0]
    continue_count = int((books["verdict"] == "CONTINUE").sum())
    narrow_count = int((books["verdict"] == "NARROW").sum())
    kill_count = int((books["verdict"] == "KILL").sum())
    cols = [
        "strategy",
        "us_data_rr",
        "us_data_filter",
        "n_is",
        "annual_r",
        "dd",
        "objective_score",
        "t",
        "q_family",
        "dsr_family",
        "dsr_inherited",
        "wfe",
        "mean_2026",
        "leg_corr_is",
        "verdict",
    ]
    lines = [
        "# MNQ US_DATA_1000 RR Leg Choice v1",
        "",
        f"**Prereg:** `{PREREG_PATH.as_posix()}`",
        "**Status:** narrow post-exploratory follow-up; no deployment claim.",
        "**Canonical inputs:** `orb_outcomes` + `daily_features`; 2026 rows are descriptive only.",
        f"**Selection window:** `< {HOLDOUT_SACRED_FROM}`.",
        f"**Family K:** `{DECLARED_K}` US_DATA_1000 RR/filter books.",
        f"**Inherited exploratory burden:** `{UPSTREAM_EXPLORATORY_K}` upstream cells.",
        "",
        "## Scope",
        "",
        "[MEASURED] This test fixes the NYSE_OPEN anchor leg and only varies the US_DATA_1000 O15 E2 RR/filter leg.",
        f"[MEASURED] Objective-score winner is `{objective_winner['strategy']}` with annual R {_fmt(objective_winner['annual_r'])}, DD {_fmt(objective_winner['dd'])}, and objective {_fmt(objective_winner['objective_score'])}.",
        f"[MEASURED] Annual-only sensitivity winner is `{annual_winner['strategy']}` with annual R {_fmt(annual_winner['annual_r'])} and DD {_fmt(annual_winner['dd'])}.",
        f"[MEASURED] Current comparison row `{current['strategy']}` has annual R {_fmt(current['annual_r'])}, DD {_fmt(current['dd'])}, and objective {_fmt(current['objective_score'])}.",
        "",
        "## Reproduction",
        "",
        f"- Front door: `python scripts/tools/prereg_front_door.py --hypothesis-file {PREREG_PATH.as_posix()} --execute --runner research/mnq_usdata_rr_leg_choice_v1.py --format text`",
        f"- Book CSV: `{BOOK_CSV.as_posix()}`",
        "- DB mode: read-only canonical `gold.db` via `pipeline.paths.GOLD_DB_PATH`.",
        "",
        "## Candidate Books",
        "",
        *_write_table(books[cols], cols),
        "",
        f"Verdict counts: CONTINUE=`{continue_count}`, NARROW=`{narrow_count}`, KILL=`{kill_count}`.",
        "",
        "## Grounding",
        "",
        "- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`: DSR controls selection bias, non-normality, sample length, trial count, and cross-trial Sharpe dispersion.",
        "- `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md`: broad strategy searches need multiple-testing controls and high t-stat discipline.",
        "- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md`: 2026 is descriptive only; no OOS-tweaking.",
        "",
        "## Caveats",
        "",
        "- This is a post-exploratory follow-up, not clean first discovery.",
        "- `DSR_inherited` carries the prior broad scan, so family-positive rows remain `NARROW` unless inherited DSR clears.",
        "- The result is a research book comparison, not broker/account deployment sizing.",
        "",
        "## Verdict",
        "",
        f"`{objective_winner['verdict']}` for the objective-score winner. The data favors the lower-RR US_DATA_1000 leg for drawdown-adjusted book quality, but inherited DSR prevents replacement without another confirmation step.",
        "",
        "SURVIVED SCRUTINY: finite K, no-theory t>=3.79 gate, family BH, DSR family/inherited, WFE, era, 2026 descriptive separation.",
        "DID NOT SURVIVE: no candidate is deployment-ready from this result alone.",
        "CAVEATS: post-selection; no ONC de-correlation; arithmetic portfolio only.",
        "NEXT STEPS: if accepted, route a deployment-readiness design comparing current RR2 book versus RR1/RR1.5 candidate under account constraints.",
        "",
    ]
    RESULT_DOC.write_text("\n".join(lines), encoding="utf-8")


def run(db_path: Path = GOLD_DB_PATH) -> pd.DataFrame:
    upstream, payload = _candidate_cells_and_series(db_path)
    payload.pop("__cells__")
    filtered_by_name = payload
    if ANCHOR_LEG not in filtered_by_name:
        raise RuntimeError(f"Missing anchor leg series: {ANCHOR_LEG}")
    books = _score_books(upstream, filtered_by_name)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    books.to_csv(BOOK_CSV, index=False)
    write_report(books)
    return books


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=GOLD_DB_PATH)
    args = parser.parse_args()
    books = run(args.db)
    print(f"wrote {len(books)} books to {BOOK_CSV}")
    print(f"wrote report to {RESULT_DOC}")


if __name__ == "__main__":
    main()
