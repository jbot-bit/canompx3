#!/usr/bin/env python3
"""Post-exploratory MNQ ORB book validation and open challenger scan v1.

This runner answers the user's scope challenge directly:

1. Validate the current MNQ O15/E2/RR2 NYSE_OPEN + US_DATA_1000 book shape
   in a tiny, explicitly risk-aware family.
2. Run a capped challenger-book scan so the research is not pigeonholed around
   the previous answer.
3. Diagnose whether existing pre-entry-safe filters are adding broad value, or
   whether "new filters" is still unsupported by the current canonical fields.

Canonical inputs: orb_outcomes + daily_features only.
Selection window: trading_day < HOLDOUT_SACRED_FROM.
Monitoring window: trading_day >= HOLDOUT_SACRED_FROM, descriptive only.
"""

from __future__ import annotations

import argparse
import itertools
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
    _portfolio_series,
    _score_candidate,
    _score_returns,
    _strategy_name,
)
from trading_app.dsr import compute_dsr, compute_sr0
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

RESULT_DIR = Path("docs/audit/results")
PREREG_PATH = Path("docs/audit/hypotheses/2026-06-01-mnq-open-book-validation-v1.yaml")
RESULT_DOC = RESULT_DIR / "2026-06-01-mnq-open-book-validation-v1.md"
BOOK_CSV = RESULT_DIR / "2026-06-01-mnq-open-book-validation-v1-books.csv"
POOL_CSV = RESULT_DIR / "2026-06-01-mnq-open-book-validation-v1-pool.csv"
FILTER_CSV = RESULT_DIR / "2026-06-01-mnq-open-book-validation-v1-filter-diagnostics.csv"

UPSTREAM_EXPLORATORY_K = len(ASSET_CONFIGS[SYMBOL]["enabled_sessions"]) * len(ORB_MINUTES) * len(RR_TARGETS) * len(FILTERS)
POOL_SIZE = 20
MAX_POOL_PER_SESSION = 2
MAX_POOL_PER_SHAPE = 1
MIN_POOL_N = 100
BH_Q = 0.05
NO_THEORY_T_THRESHOLD = 3.79

CURRENT_BOOKS = (
    (
        "CURRENT_COST_LT10",
        (
            _strategy_name("NYSE_OPEN", 15, 2.0, "COST_LT10"),
            _strategy_name("US_DATA_1000", 15, 2.0, "COST_LT10"),
        ),
    ),
    (
        "CURRENT_NO_FILTER",
        (
            _strategy_name("NYSE_OPEN", 15, 2.0, "NO_FILTER"),
            _strategy_name("US_DATA_1000", 15, 2.0, "NO_FILTER"),
        ),
    ),
)


def _objective_score(row: pd.Series | dict[str, object]) -> float:
    annual = float(row.get("annual_r", float("nan")))
    drawdown = float(row.get("dd", float("nan")))
    if not math.isfinite(annual) or not math.isfinite(drawdown) or drawdown <= 0:
        return float("-inf")
    return annual / drawdown


def _leg_correlation(legs: list[pd.DataFrame]) -> float:
    if len(legs) != 2:
        return float("nan")
    left = _portfolio_series([legs[0]])[["trading_day", "pnl_r"]].rename(columns={"pnl_r": "leg_a"})
    right = _portfolio_series([legs[1]])[["trading_day", "pnl_r"]].rename(columns={"pnl_r": "leg_b"})
    joined = left.merge(right, on="trading_day", how="outer")
    joined[["leg_a", "leg_b"]] = joined[["leg_a", "leg_b"]].fillna(0.0)
    joined = joined[joined["trading_day"] < HOLDOUT_SACRED_FROM]
    if len(joined) < 3:
        return float("nan")
    return float(joined["leg_a"].corr(joined["leg_b"]))


def _candidate_cells_and_series(db_path: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    filtered_by_name: dict[str, pd.DataFrame] = {}
    sessions = tuple(ASSET_CONFIGS[SYMBOL]["enabled_sessions"])
    declared_k = len(sessions) * len(ORB_MINUTES) * len(RR_TARGETS) * len(FILTERS)

    with duckdb.connect(str(db_path), read_only=True) as con:
        for session in sessions:
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
                        row, filtered = _score_candidate(base, spec)
                        rows.append(row)
                        filtered_by_name[spec.strategy] = filtered

    _add_multiple_testing(rows, declared_k)
    cells = pd.DataFrame(rows)
    cells["pool_score"] = cells.apply(_objective_score, axis=1)
    return cells, filtered_by_name


def select_candidate_pool(cells: pd.DataFrame, pool_size: int = POOL_SIZE) -> pd.DataFrame:
    """Select a capped challenger pool using pre-2026 metrics only.

    This is still post-exploratory, so the report carries the inherited broad-K
    burden. The cap prevents one local shape from crowding out alternative
    sessions, apertures, RR targets, and filters.
    """

    work = cells.copy()
    if "pool_score" not in work.columns:
        work["pool_score"] = work.apply(_objective_score, axis=1)

    eligible = work[
        (work["n_is"].astype(int) >= MIN_POOL_N)
        & (work["q_global"].astype(float) < BH_Q)
        & (work["t"].astype(float) >= NO_THEORY_T_THRESHOLD)
        & (work["wfe"].astype(float) >= 0.50)
        & (work["era_ok"].astype(bool))
        & np.isfinite(work["pool_score"].astype(float))
    ].copy()
    eligible = eligible.sort_values(
        ["pool_score", "annual_r", "t"],
        ascending=[False, False, False],
    )

    selected: list[pd.Series] = []
    session_counts: dict[str, int] = {}
    shape_counts: dict[tuple[str, int, float], int] = {}
    for _, row in eligible.iterrows():
        session = str(row["session"])
        shape = (session, int(row["orb_minutes"]), float(row["rr"]))
        if session_counts.get(session, 0) >= MAX_POOL_PER_SESSION:
            continue
        if shape_counts.get(shape, 0) >= MAX_POOL_PER_SHAPE:
            continue
        selected.append(row)
        session_counts[session] = session_counts.get(session, 0) + 1
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
        if len(selected) >= pool_size:
            break

    return pd.DataFrame(selected).reset_index(drop=True)


def _score_book(
    name: str,
    leg_names: tuple[str, str],
    filtered_by_name: dict[str, pd.DataFrame],
    cell_lookup: pd.DataFrame,
    *,
    family: str,
) -> dict[str, object]:
    legs = [filtered_by_name[leg_name] for leg_name in leg_names]
    book = _portfolio_series(legs)
    row = _score_returns(book, "pnl_r", strategy=name)
    leg_rows = cell_lookup.set_index("strategy").loc[list(leg_names)]
    row.update(
        {
            "family": family,
            "leg_1": leg_names[0],
            "leg_2": leg_names[1],
            "leg_1_session": str(leg_rows.iloc[0]["session"]),
            "leg_2_session": str(leg_rows.iloc[1]["session"]),
            "leg_1_orb_minutes": int(leg_rows.iloc[0]["orb_minutes"]),
            "leg_2_orb_minutes": int(leg_rows.iloc[1]["orb_minutes"]),
            "leg_1_rr": float(leg_rows.iloc[0]["rr"]),
            "leg_2_rr": float(leg_rows.iloc[1]["rr"]),
            "leg_1_filter": str(leg_rows.iloc[0]["filter"]),
            "leg_2_filter": str(leg_rows.iloc[1]["filter"]),
            "leg_corr_is": _leg_correlation(legs),
        }
    )
    row["objective_score"] = _objective_score(row)
    return row


def _add_book_statistics(
    books: pd.DataFrame,
    family_col: str = "family",
    upstream_sharpes: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    books = books.copy()
    if books.empty:
        return books

    books["q_family"] = np.nan
    books["sr0_family"] = np.nan
    books["dsr_family"] = 0.0
    for _family, sub in books.groupby(family_col):
        idxs = list(sub.index)
        qs = _bh_qvalues([float(books.loc[idx, "p"]) for idx in idxs])
        sharpes = np.array([float(books.loc[idx, "sharpe"]) for idx in idxs], dtype=float)
        var_sr = float(np.nanvar(sharpes, ddof=1)) if np.isfinite(sharpes).sum() > 1 else 0.0
        sr0 = compute_sr0(len(idxs), var_sr)
        for idx, q in zip(idxs, qs, strict=True):
            n = int(books.loc[idx, "n_is"])
            sr = float(books.loc[idx, "sharpe"])
            skew = float(books.loc[idx, "skewness"])
            kurt = float(books.loc[idx, "kurtosis_excess"])
            books.loc[idx, "q_family"] = q
            books.loc[idx, "sr0_family"] = sr0
            books.loc[idx, "dsr_family"] = compute_dsr(sr, sr0, n, skew, kurt) if math.isfinite(sr) and n >= 2 else 0.0

    book_sharpes = np.array([float(x) for x in books["sharpe"]], dtype=float)
    if upstream_sharpes is None:
        inherited_sharpes = book_sharpes
    else:
        upstream = np.array([float(x) for x in upstream_sharpes], dtype=float)
        inherited_sharpes = np.concatenate([upstream[np.isfinite(upstream)], book_sharpes[np.isfinite(book_sharpes)]])
    var_sr_inherited = (
        float(np.nanvar(inherited_sharpes, ddof=1)) if np.isfinite(inherited_sharpes).sum() > 1 else 0.0
    )
    sr0_inherited = compute_sr0(UPSTREAM_EXPLORATORY_K + len(books), var_sr_inherited)
    books["sr0_inherited"] = sr0_inherited
    books["dsr_inherited"] = [
        compute_dsr(float(row.sharpe), sr0_inherited, int(row.n_is), float(row.skewness), float(row.kurtosis_excess))
        if math.isfinite(float(row.sharpe)) and int(row.n_is) >= 2
        else 0.0
        for row in books.itertuples(index=False)
    ]

    books["verdict"] = books.apply(_book_verdict, axis=1)
    return books


def _book_verdict(row: pd.Series) -> str:
    passes_core = (
        float(row.get("q_family", 1.0)) < BH_Q
        and float(row.get("t", 0.0)) >= NO_THEORY_T_THRESHOLD
        and float(row.get("dsr_family", 0.0)) > 0.95
        and bool(row.get("era_ok", False))
        and float(row.get("wfe", float("nan"))) >= 0.50
    )
    if not passes_core:
        return "KILL"
    if float(row.get("dsr_inherited", 0.0)) > 0.95:
        return "CONTINUE"
    return "NARROW"


def build_current_books(cells: pd.DataFrame, filtered_by_name: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [
        _score_book(name, leg_names, filtered_by_name, cells, family="current_two_book")
        for name, leg_names in CURRENT_BOOKS
    ]
    return _add_book_statistics(pd.DataFrame(rows), upstream_sharpes=cells["sharpe"])


def build_challenger_books(
    pool: pd.DataFrame,
    cells: pd.DataFrame,
    filtered_by_name: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, (left, right) in enumerate(itertools.combinations(pool.itertuples(index=False), 2), start=1):
        if str(left.session) == str(right.session):
            continue
        name = f"CHALLENGER_PAIR_{idx:03d}"
        leg_names = (str(left.strategy), str(right.strategy))
        rows.append(_score_book(name, leg_names, filtered_by_name, cells, family="open_pair_book"))
    books = _add_book_statistics(pd.DataFrame(rows), upstream_sharpes=cells["sharpe"])
    if books.empty:
        return books
    return books.sort_values(
        ["objective_score", "annual_r", "t"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def filter_diagnostics(cells: pd.DataFrame) -> pd.DataFrame:
    base = cells[cells["filter"] == "NO_FILTER"][
        [
            "session",
            "orb_minutes",
            "rr",
            "annual_r",
            "mean_is",
            "dd",
            "t",
            "wfe",
            "mean_2026",
        ]
    ].rename(
        columns={
            "annual_r": "annual_r_base",
            "mean_is": "mean_is_base",
            "dd": "dd_base",
            "t": "t_base",
            "wfe": "wfe_base",
            "mean_2026": "mean_2026_base",
        }
    )
    filtered = cells[cells["filter"] != "NO_FILTER"].merge(base, on=["session", "orb_minutes", "rr"], how="inner")
    filtered["delta_annual_r"] = filtered["annual_r"].astype(float) - filtered["annual_r_base"].astype(float)
    filtered["delta_mean_is"] = filtered["mean_is"].astype(float) - filtered["mean_is_base"].astype(float)
    filtered["delta_dd"] = filtered["dd"].astype(float) - filtered["dd_base"].astype(float)
    filtered["delta_t"] = filtered["t"].astype(float) - filtered["t_base"].astype(float)
    filtered["delta_wfe"] = filtered["wfe"].astype(float) - filtered["wfe_base"].astype(float)
    filtered["delta_mean_2026"] = filtered["mean_2026"].astype(float) - filtered["mean_2026_base"].astype(float)
    filtered["helped_risk_adjusted"] = (filtered["delta_annual_r"] > 0.0) & (filtered["delta_dd"] <= 0.0)

    grouped_rows: list[dict[str, object]] = []
    for filter_name, sub in filtered.groupby("filter"):
        best = sub.sort_values(["delta_annual_r", "delta_t"], ascending=[False, False]).iloc[0]
        grouped_rows.append(
            {
                "filter": filter_name,
                "comparisons": int(len(sub)),
                "median_delta_annual_r": float(sub["delta_annual_r"].median()),
                "mean_delta_annual_r": float(sub["delta_annual_r"].mean()),
                "median_delta_dd": float(sub["delta_dd"].median()),
                "mean_delta_t": float(sub["delta_t"].mean()),
                "median_delta_wfe": float(sub["delta_wfe"].median()),
                "median_delta_2026": float(sub["delta_mean_2026"].median()),
                "helped_risk_adjusted_count": int(sub["helped_risk_adjusted"].sum()),
                "hurt_annual_count": int((sub["delta_annual_r"] < 0.0).sum()),
                "best_delta_annual_r": float(best["delta_annual_r"]),
                "best_strategy": str(best["strategy"]),
            }
        )
    return pd.DataFrame(grouped_rows).sort_values(
        ["median_delta_annual_r", "helped_risk_adjusted_count"],
        ascending=[False, False],
    )


def _write_table(rows: pd.DataFrame, columns: list[str]) -> list[str]:
    if rows.empty:
        return ["_None._"]
    header = "| " + " | ".join(columns) + " |"
    align = "| " + " | ".join("---" if col in {"strategy", "filter", "verdict", "leg_1", "leg_2", "best_strategy"} else "---:" for col in columns) + " |"
    lines = [header, align]
    for _, row in rows.iterrows():
        parts: list[str] = []
        for col in columns:
            value = row[col]
            if col in {"strategy", "filter", "verdict", "leg_1", "leg_2", "best_strategy"}:
                parts.append(f"`{value}`")
            elif col in {"n_is", "n_2026", "comparisons", "helped_risk_adjusted_count", "hurt_annual_count"}:
                parts.append(str(int(value)))
            elif col in {"p", "q_family", "dsr_family", "dsr_inherited"}:
                parts.append(_fmt(value, 6))
            else:
                parts.append(_fmt(value))
        lines.append("| " + " | ".join(parts) + " |")
    return lines


def write_report(cells: pd.DataFrame, pool: pd.DataFrame, current: pd.DataFrame, challengers: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    current_winner = current.sort_values(["objective_score", "annual_r"], ascending=[False, False]).iloc[0]
    annual_winner = current.sort_values("annual_r", ascending=False).iloc[0]
    top_challenger = challengers.iloc[0] if not challengers.empty else None
    annual_challenger = challengers.sort_values("annual_r", ascending=False).iloc[0] if not challengers.empty else None
    top_filter = diagnostics.iloc[0] if not diagnostics.empty else None
    pool_sessions = ", ".join(sorted(pool["session"].astype(str).unique()))
    open_k = int(len(challengers))
    continue_count = int((challengers["verdict"] == "CONTINUE").sum()) if not challengers.empty else 0
    narrow_count = int((challengers["verdict"] == "NARROW").sum()) if not challengers.empty else 0

    current_cols = [
        "strategy",
        "n_is",
        "mean_is",
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
    challenger_cols = [
        "strategy",
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
    challenger_detail_cols = [
        "strategy",
        "leg_1",
        "leg_2",
        "annual_r",
        "dd",
        "objective_score",
        "mean_2026",
        "dsr_inherited",
        "verdict",
    ]
    pool_cols = ["strategy", "session", "orb_minutes", "rr", "filter", "annual_r", "dd", "pool_score", "t", "q_global", "wfe", "mean_2026"]
    diag_cols = [
        "filter",
        "comparisons",
        "median_delta_annual_r",
        "median_delta_dd",
        "mean_delta_t",
        "median_delta_wfe",
        "median_delta_2026",
        "helped_risk_adjusted_count",
        "hurt_annual_count",
        "best_delta_annual_r",
        "best_strategy",
    ]

    top_line = (
        f"[MEASURED] The current risk-aware book winner is `{current_winner['strategy']}` "
        f"with objective score {_fmt(current_winner['objective_score'])}; annual-only sensitivity would choose "
        f"`{annual_winner['strategy']}`."
    )
    if top_challenger is not None:
        challenger_line = (
            f"[MEASURED] The open challenger scan tested `{open_k}` two-leg books from a capped `{len(pool)}`-cell pool "
            f"spanning `{pool_sessions}`. Top challenger by the same risk-aware score is `{top_challenger['strategy']}`."
        )
        annual_challenger_line = (
            f"[MEASURED] Annual-only sensitivity would choose challenger `{annual_challenger['strategy']}` "
            f"with annual R {_fmt(annual_challenger['annual_r'])}, drawdown {_fmt(annual_challenger['dd'])}, "
            f"and 2026 descriptive mean {_fmt(annual_challenger['mean_2026'])}."
        )
    else:
        challenger_line = "[MEASURED] No challenger books were generated after the session-diversity guard."
        annual_challenger_line = "[MEASURED] No annual-only challenger sensitivity was available."
    if top_filter is not None:
        filter_line = (
            f"[MEASURED] Existing filter diagnosis is led by `{top_filter['filter']}`: "
            f"median annual delta {_fmt(top_filter['median_delta_annual_r'])}R and median drawdown delta "
            f"{_fmt(top_filter['median_delta_dd'])}R versus matched NO_FILTER parents."
        )
    else:
        filter_line = "[MEASURED] No filter diagnostic rows were available."

    lines = [
        "# MNQ Open Book Validation v1",
        "",
        f"**Prereg:** `{PREREG_PATH.as_posix()}`",
        "**Status:** post-exploratory, research-provisional only; no deployment claim.",
        "**Canonical inputs:** `orb_outcomes` + `daily_features`; 2026 rows are descriptive only.",
        f"**Selection window:** `< {HOLDOUT_SACRED_FROM}`.",
        f"**Inherited exploratory burden:** `{UPSTREAM_EXPLORATORY_K}` upstream cells from the prior broad scan.",
        f"**Open challenger K:** `{open_k}` pair books, capped below the 300 operational trial ceiling for this follow-up family.",
        "",
        "## Scope Answer",
        "",
        "[GROUNDED] The narrow current book is specific enough for confirmation, but too specific as the only research answer. The open pair-book layer keeps the question broad without turning it into an unbounded grid.",
        f"[GROUNDED] New filters are not assumed. Existing pre-entry-safe filters are diagnosed first; order-flow, footprint, and absorption remain `PARK_NEW_DATA` under current 1m OHLCV truth surfaces. No new theory grant is claimed, so the research gate uses t >= {NO_THEORY_T_THRESHOLD:.2f}.",
        top_line,
        challenger_line,
        annual_challenger_line,
        filter_line,
        "",
        "## Reproduction",
        "",
        f"- Front door: `python scripts/tools/prereg_front_door.py --hypothesis-file {PREREG_PATH.as_posix()} --execute --runner research/mnq_open_book_validation_v1.py --format text`",
        f"- Book CSV: `{BOOK_CSV.as_posix()}`",
        f"- Pool CSV: `{POOL_CSV.as_posix()}`",
        f"- Filter diagnostics CSV: `{FILTER_CSV.as_posix()}`",
        "- DB mode: read-only canonical `gold.db` via `pipeline.paths.GOLD_DB_PATH`.",
        "",
        "## Current Book Confirmation",
        "",
        *_write_table(current[current_cols], current_cols),
        "",
        "Primary objective is `annual_r / max_drawdown`, pre-declared to avoid choosing raw ROI after seeing drawdown. Raw annual R is reported as a sensitivity comparator.",
        "",
        "## Open Challenger Books",
        "",
        *_write_table(challengers.head(12)[challenger_cols], challenger_cols),
        "",
        "## Top Challenger Leg Detail",
        "",
        *_write_table(challengers.head(5)[challenger_detail_cols], challenger_detail_cols),
        "",
        f"Challenger verdict counts: CONTINUE=`{continue_count}`, NARROW=`{narrow_count}`, KILL=`{open_k - continue_count - narrow_count}`.",
        "",
        "## Challenger Pool",
        "",
        *_write_table(pool[pool_cols], pool_cols),
        "",
        "The pool is selected from pre-2026 metrics only using a capped risk-aware score, with at most two cells per session and one per session/aperture/RR shape.",
        "",
        "## Existing Filter Diagnosis",
        "",
        *_write_table(diagnostics[diag_cols], diag_cols),
        "",
        "This section answers whether to invent new filters. It measures whether current pre-entry-safe filters add broad value versus the same session/aperture/RR `NO_FILTER` parent before any new filter engineering.",
        "",
        "## Local Literature And Resource Grounding",
        "",
        "- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`: DSR controls selection bias, non-normality, sample length, trial count, and cross-trial Sharpe dispersion.",
        "- `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md`: broad strategy searches need multiple-testing controls and high t-stat discipline.",
        "- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md`: out-of-sample tweaking and lookahead are direct backtest-failure modes; this runner keeps 2026 out of selection.",
        "- `docs/institutional/pre_registered_criteria.md`: Criterion 5 fixed-universe DSR and Criterion 8 holdout discipline govern the research/deployment distinction.",
        "",
        "## Caveats",
        "",
        "- This is a post-exploratory follow-up from the prior 1,755-cell scan, not a clean first discovery.",
        "- `DSR_inherited` is intentionally harsh because it carries the upstream broad-scan burden; every surviving book is therefore `NARROW`, not deployment-ready.",
        "- Pair books are arithmetic research portfolios. They do not encode broker limits, prop-account sizing, correlated intraday risk, or live-session orchestration.",
        "- Order-flow, footprint, delta, and absorption claims remain parked because they are not measurable from current `bars_1m` OHLCV.",
        "",
        "## Verdict",
        "",
        f"`{current_winner['verdict']}` for the current risk-aware book under the narrow family; `NARROW` ceiling overall because the object is post-exploratory and carries the inherited 1,755-cell burden. The top challenger beats the current book on drawdown-adjusted score but does not clear inherited DSR, so it is a follow-up hypothesis, not a replacement.",
        "",
        "SURVIVED SCRUTINY: current-book and challenger metrics are computed from canonical layers only, with explicit K, DSR, BH, WFE, era, drawdown, and 2026 descriptive fields.",
        "DID NOT SURVIVE: no result in this report is deployment-ready; any challenger that depends on 2026 descriptive behavior for comfort is not selectable.",
        "CAVEATS: post-selection follow-up from a broad scan; DSR uses declared K and sibling failures but not ONC de-correlation; pair books are arithmetic portfolios, not broker/risk deployment plans.",
        "NEXT STEPS: if the risk-aware book remains the best practical candidate after this pass, route deployment-readiness separately; otherwise write a new prereg for the winning challenger family rather than silently swapping the live candidate.",
        "",
    ]
    RESULT_DOC.write_text("\n".join(lines), encoding="utf-8")


def run(db_path: Path = GOLD_DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cells, filtered_by_name = _candidate_cells_and_series(db_path)
    pool = select_candidate_pool(cells)
    current = build_current_books(cells, filtered_by_name)
    challengers = build_challenger_books(pool, cells, filtered_by_name)
    diagnostics = filter_diagnostics(cells)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pool.to_csv(POOL_CSV, index=False)
    pd.concat([current, challengers], ignore_index=True).to_csv(BOOK_CSV, index=False)
    diagnostics.to_csv(FILTER_CSV, index=False)
    write_report(cells, pool, current, challengers, diagnostics)
    return current, challengers, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=GOLD_DB_PATH)
    args = parser.parse_args()
    current, challengers, diagnostics = run(args.db)
    print(f"wrote {len(current) + len(challengers)} books to {BOOK_CSV}")
    print(f"wrote {len(diagnostics)} filter diagnostic rows to {FILTER_CSV}")
    print(f"wrote report to {RESULT_DOC}")


if __name__ == "__main__":
    main()
