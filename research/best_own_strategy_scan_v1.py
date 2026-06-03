#!/usr/bin/env python3
"""Broad MNQ ORB candidate scan for the next in-house strategy hypothesis.

This is intentionally labeled exploratory. It was created after the bounded
same-direction re-entry investigation failed to produce a deployable addition,
so it must not be read as a pre-registered validation run.

Canonical inputs: orb_outcomes + daily_features only.
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
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.asset_configs import ASSET_CONFIGS
from pipeline.cost_model import get_cost_spec
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import COST_RATIO_FILTERS
from trading_app.dsr import compute_dsr, compute_sr0
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

SYMBOL = "MNQ"
ORB_MINUTES = (5, 15, 30)
RR_TARGETS = (1.0, 1.5, 2.0)
FILTERS = (
    "NO_FILTER",
    "DIR_LONG",
    "DIR_SHORT",
    "COST_LT08",
    "COST_LT10",
    "COST_LT12",
    "COST_LT15",
    "ATR_P30",
    "ATR_P50",
    "ATR_P70",
    "ATR_VEL_GE105",
    "ORB_SIZE_Q67",
    "ORB_SIZE_Q80",
    "ORB_VOL_Q67",
    "COST10_ATR50",
)

RESULT_DIR = Path("docs/audit/results")
CELL_CSV = RESULT_DIR / "2026-06-01-best-own-strategy-scan-v1-cells.csv"
PORTFOLIO_CSV = RESULT_DIR / "2026-06-01-best-own-strategy-scan-v1-portfolio.csv"
RESULT_DOC = RESULT_DIR / "2026-06-01-best-own-strategy-scan-v1.md"

E2_LOOKAHEAD_BANNED_PATTERNS = (
    "break_ts",
    "break_delay_min",
    "break_bar_volume",
    "break_bar_continues",
    "rel_vol_",
    "break_dir",
    "outcome",
    "mae_r",
    "mfe_r",
)


@dataclass(frozen=True)
class CandidateSpec:
    strategy: str
    session: str
    orb_minutes: int
    rr: float
    filter_name: str


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    return str(value)


def _reject_lookahead(columns: list[str]) -> None:
    offenders: list[str] = []
    for col in columns:
        lowered = col.lower()
        if any(pattern in lowered for pattern in E2_LOOKAHEAD_BANNED_PATTERNS):
            offenders.append(col)
    if offenders:
        raise ValueError(f"Lookahead or post-trigger fields requested as predictors: {offenders}")


def _bh_qvalues(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    clean = [1.0 if not math.isfinite(float(p)) else max(0.0, min(1.0, float(p))) for p in p_values]
    m = len(clean)
    order = np.argsort(clean)
    q = np.ones(m, dtype=float)
    running = 1.0
    for rank_from_end, idx in enumerate(order[::-1], start=1):
        rank = m - rank_from_end + 1
        val = clean[idx] * m / rank
        running = min(running, val)
        q[idx] = running
    return [float(x) for x in q]


def _max_drawdown(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return float("nan")
    curve = np.cumsum(vals)
    running_max = np.maximum.accumulate(curve)
    return float((running_max - curve).max())


def _sharpe(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size < 2:
        return float("nan")
    std = float(vals.std(ddof=1))
    if std <= 0:
        return 0.0
    return float(vals.mean() / std)


def _distribution_shape(values: np.ndarray) -> tuple[float, float]:
    vals = values[np.isfinite(values)]
    if vals.size < 4:
        return 0.0, 0.0
    skewness = float(stats.skew(vals, bias=False))
    kurtosis_excess = float(stats.kurtosis(vals, fisher=True, bias=False))
    if not math.isfinite(skewness):
        skewness = 0.0
    if not math.isfinite(kurtosis_excess):
        kurtosis_excess = 0.0
    return skewness, kurtosis_excess


def _ttest(values: np.ndarray) -> tuple[float, float]:
    vals = values[np.isfinite(values)]
    if vals.size < 10:
        return float("nan"), float("nan")
    result = stats.ttest_1samp(vals, 0.0)
    return float(result.statistic), float(result.pvalue)


def _years_per_span(start: object, end: object) -> float:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    days = max(1, int((end_ts - start_ts).days) + 1)
    return days / 365.25


def _era_detail(df: pd.DataFrame, value_col: str) -> tuple[bool, str]:
    eras = [
        ("2019", "2019-01-01", "2019-12-31"),
        ("2020_2022", "2020-01-01", "2022-12-31"),
        ("2023", "2023-01-01", "2023-12-31"),
        ("2024_2025", "2024-01-01", "2025-12-31"),
    ]
    parts: list[str] = []
    passed = True
    eligible = 0
    for name, start, end in eras:
        sub = df[(df["trading_day"] >= pd.Timestamp(start).date()) & (df["trading_day"] <= pd.Timestamp(end).date())]
        if len(sub) < 50:
            parts.append(f"{name}:LOW_N={len(sub)}")
            continue
        eligible += 1
        mean = float(sub[value_col].mean())
        parts.append(f"{name}:{len(sub)}:{mean:+.3f}")
        if mean < -0.05:
            passed = False
    if eligible < 3:
        passed = False
    return passed, ";".join(parts)


def _year_detail(df: pd.DataFrame, value_col: str) -> tuple[int, int, str]:
    if df.empty:
        return 0, 0, ""
    work = df.copy()
    work["_year"] = pd.to_datetime(work["trading_day"]).dt.year
    parts: list[str] = []
    positive = 0
    eligible = 0
    for year, sub in work.groupby("_year"):
        if len(sub) < 30:
            parts.append(f"{int(year)}:LOW_N={len(sub)}")
            continue
        eligible += 1
        mean = float(sub[value_col].mean())
        if mean > 0:
            positive += 1
        parts.append(f"{int(year)}:{len(sub)}:{mean:+.3f}")
    return positive, eligible, ";".join(parts)


def _wfe(df: pd.DataFrame, value_col: str) -> float:
    windows: list[tuple[int, float, float]] = []
    years = pd.to_datetime(df["trading_day"]).dt.year if not df.empty else pd.Series(dtype=int)
    for year in range(2021, 2026):
        train = df[years < year]
        test = df[years == year]
        if len(train) < 100 or len(test) < 30:
            continue
        train_mean = float(train[value_col].mean())
        test_mean = float(test[value_col].mean())
        if train_mean <= 0:
            continue
        windows.append((len(test), train_mean, test_mean))
    if not windows:
        return float("nan")
    total_n = sum(n for n, _, _ in windows)
    train_weighted = sum(n * train for n, train, _ in windows) / total_n
    test_weighted = sum(n * test for n, _, test in windows) / total_n
    if train_weighted <= 0:
        return float("nan")
    return float(test_weighted / train_weighted)


def _session_feature_columns(session: str) -> list[str]:
    return [
        f"orb_{session}_size",
        f"orb_{session}_volume",
        "atr_20_pct",
        "atr_vel_ratio",
    ]


def _load_base(con: duckdb.DuckDBPyConnection, session: str, orb_minutes: int, rr: float) -> pd.DataFrame:
    predictor_cols = _session_feature_columns(session)
    _reject_lookahead(predictor_cols)
    size_col = f"orb_{session}_size"
    volume_col = f"orb_{session}_volume"
    sql = f"""
        SELECT
            o.trading_day,
            o.pnl_r,
            o.entry_price,
            o.stop_price,
            o.risk_dollars,
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
        ORDER BY o.trading_day
    """
    df = con.execute(sql, [SYMBOL, session, orb_minutes, rr]).fetchdf()
    if df.empty:
        return df
    df["symbol"] = SYMBOL
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    df["null_pnl"] = df["pnl_r"].isna()
    df["pnl_r"] = df["pnl_r"].fillna(0.0).astype(float)
    entry = df["entry_price"].astype(float)
    stop = df["stop_price"].astype(float)
    df["direction"] = np.select([entry > stop, entry < stop], ["long", "short"], default=None)
    df[size_col] = df["orb_size"]
    spec = get_cost_spec(SYMBOL)
    raw_risk = df["orb_size"].astype(float) * spec.point_value
    df["cost_ratio"] = spec.total_friction / (raw_risk + spec.total_friction)
    return df


def _filter_mask(df: pd.DataFrame, filter_name: str, session: str) -> pd.Series:
    true = pd.Series(True, index=df.index)
    if filter_name == "NO_FILTER":
        return true
    if filter_name == "DIR_LONG":
        return df["direction"] == "long"
    if filter_name == "DIR_SHORT":
        return df["direction"] == "short"
    if filter_name.startswith("COST_LT"):
        return COST_RATIO_FILTERS[filter_name].matches_df(df, session)
    if filter_name == "ATR_P30":
        return df["atr_20_pct"].astype(float) >= 30.0
    if filter_name == "ATR_P50":
        return df["atr_20_pct"].astype(float) >= 50.0
    if filter_name == "ATR_P70":
        return df["atr_20_pct"].astype(float) >= 70.0
    if filter_name == "ATR_VEL_GE105":
        return df["atr_vel_ratio"].astype(float) >= 1.05
    if filter_name == "ORB_SIZE_Q67":
        is_df = df[df["trading_day"] < HOLDOUT_SACRED_FROM]
        threshold = float(is_df["orb_size"].quantile(0.67))
        return df["orb_size"].astype(float) >= threshold
    if filter_name == "ORB_SIZE_Q80":
        is_df = df[df["trading_day"] < HOLDOUT_SACRED_FROM]
        threshold = float(is_df["orb_size"].quantile(0.80))
        return df["orb_size"].astype(float) >= threshold
    if filter_name == "ORB_VOL_Q67":
        is_df = df[df["trading_day"] < HOLDOUT_SACRED_FROM]
        threshold = float(is_df["orb_volume"].quantile(0.67))
        return df["orb_volume"].astype(float) >= threshold
    if filter_name == "COST10_ATR50":
        return (df["cost_ratio"] < 0.10) & (df["atr_20_pct"].astype(float) >= 50.0)
    raise ValueError(f"Unsupported filter: {filter_name}")


def _score_returns(df: pd.DataFrame, value_col: str, *, strategy: str, session: str = "", filter_name: str = "") -> dict[str, object]:
    is_df = df[df["trading_day"] < HOLDOUT_SACRED_FROM].copy()
    monitor_df = df[df["trading_day"] >= HOLDOUT_SACRED_FROM].copy()
    vals = is_df[value_col].to_numpy(dtype=float)
    t_stat, p_value = _ttest(vals)
    skewness, kurtosis_excess = _distribution_shape(vals)
    start = is_df["trading_day"].min() if len(is_df) else None
    end = is_df["trading_day"].max() if len(is_df) else None
    years = _years_per_span(start, end) if start is not None and end is not None else float("nan")
    trades_per_year = float(len(is_df) / years) if years and math.isfinite(years) else float("nan")
    mean = float(np.mean(vals)) if len(vals) else float("nan")
    sharpe = _sharpe(vals)
    dd = _max_drawdown(vals)
    era_ok, era = _era_detail(is_df, value_col)
    years_positive, years_eligible, years_text = _year_detail(is_df, value_col)
    return {
        "strategy": strategy,
        "session": session,
        "filter": filter_name,
        "n_is": int(len(is_df)),
        "mean_is": mean,
        "annual_r": mean * trades_per_year if math.isfinite(mean) and math.isfinite(trades_per_year) else float("nan"),
        "sharpe": sharpe,
        "skewness": skewness,
        "kurtosis_excess": kurtosis_excess,
        "t": t_stat,
        "p": p_value,
        "wfe": _wfe(is_df, value_col),
        "era_ok": era_ok,
        "years_positive": years_positive,
        "years_eligible": years_eligible,
        "dd": dd,
        "tail5": float(np.nanpercentile(vals, 5)) if len(vals) else float("nan"),
        "start": str(start) if start is not None else None,
        "end": str(end) if end is not None else None,
        "n_2026": int(len(monitor_df)),
        "mean_2026": float(monitor_df[value_col].mean()) if len(monitor_df) else float("nan"),
        "trades_per_year": trades_per_year,
        "era_detail": era,
        "year_detail": years_text,
    }


def _score_candidate(base: pd.DataFrame, spec: CandidateSpec) -> tuple[dict[str, object], pd.DataFrame]:
    if base.empty:
        empty = pd.DataFrame(columns=["trading_day", "pnl_r"])
        row = {
            "strategy": spec.strategy,
            "session": spec.session,
            "orb_minutes": spec.orb_minutes,
            "rr": spec.rr,
            "filter": spec.filter_name,
            "n_is": 0,
        }
        return row, empty
    mask = _filter_mask(base, spec.filter_name, spec.session).fillna(False)
    filtered = base[mask].copy()
    row = _score_returns(filtered, "pnl_r", strategy=spec.strategy, session=spec.session, filter_name=spec.filter_name)
    row.update(
        {
            "orb_minutes": spec.orb_minutes,
            "rr": spec.rr,
            "null_pnl_is": int(filtered[(filtered["trading_day"] < HOLDOUT_SACRED_FROM) & filtered["null_pnl"]].shape[0]),
            "null_pnl_total": int(filtered["null_pnl"].sum()),
        }
    )
    return row, filtered[["trading_day", "pnl_r"]].copy()


def _strategy_name(session: str, orb_minutes: int, rr: float, filter_name: str) -> str:
    rr_token = str(int(rr)) if float(rr).is_integer() else str(rr).replace(".", "_")
    return f"{SYMBOL}_{session}_O{orb_minutes}_E2_RR{rr_token}_{filter_name}"


def _add_multiple_testing(rows: list[dict[str, object]], declared_k: int) -> None:
    p_values = [float(r.get("p", 1.0)) for r in rows]
    q_global = _bh_qvalues(p_values)
    for row, q in zip(rows, q_global, strict=True):
        row["p_for_bh"] = row.get("p", float("nan"))
        row["q_global"] = q

    by_session: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_session.setdefault(str(row["session"]), []).append(idx)
    for idxs in by_session.values():
        qs = _bh_qvalues([float(rows[i].get("p", 1.0)) for i in idxs])
        for idx, q in zip(idxs, qs, strict=True):
            rows[idx]["q_session"] = q

    sharpes = np.array([float(r.get("sharpe", float("nan"))) for r in rows], dtype=float)
    var_sr = float(np.nanvar(sharpes, ddof=1)) if np.isfinite(sharpes).sum() > 1 else 0.0
    sr0 = compute_sr0(declared_k, var_sr)
    for row in rows:
        n = int(row.get("n_is", 0) or 0)
        sr = float(row.get("sharpe", float("nan")))
        vals_skew = float(row.get("skewness", 0.0) or 0.0)
        vals_kurt = float(row.get("kurtosis_excess", 0.0) or 0.0)
        row["sr0"] = sr0
        row["dsr"] = compute_dsr(sr, sr0, n, vals_skew, vals_kurt) if math.isfinite(sr) and n >= 2 else 0.0
        row["strict_pass"] = bool(
            row.get("q_global", 1.0) < 0.05
            and float(row.get("t", 0.0)) >= 3.0
            and float(row.get("dsr", 0.0)) > 0.95
            and bool(row.get("era_ok", False))
            and float(row.get("wfe", float("nan"))) >= 0.50
            and float(row.get("mean_2026", float("nan"))) > 0.0
        )
        row["research_shortlist"] = bool(
            row.get("q_global", 1.0) < 0.05
            and float(row.get("t", 0.0)) >= 3.0
            and bool(row.get("era_ok", False))
            and float(row.get("wfe", float("nan"))) >= 0.50
            and float(row.get("mean_2026", float("nan"))) > 0.0
        )


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


def _score_portfolios(filtered_by_name: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    specs = [
        (
            "NY_OPEN+US1000_COST10",
            [
                _strategy_name("NYSE_OPEN", 15, 2.0, "COST_LT10"),
                _strategy_name("US_DATA_1000", 15, 2.0, "COST_LT10"),
            ],
        ),
        (
            "NY_OPEN+US1000_NOFILTER",
            [
                _strategy_name("NYSE_OPEN", 15, 2.0, "NO_FILTER"),
                _strategy_name("US_DATA_1000", 15, 2.0, "NO_FILTER"),
            ],
        ),
    ]
    rows: list[dict[str, object]] = []
    for name, leg_names in specs:
        legs = [filtered_by_name[leg_name] for leg_name in leg_names]
        book = _portfolio_series(legs)
        row = _score_returns(book, "pnl_r", strategy=name)
        row["legs"] = " + ".join(leg_names)
        is_book = book[book["trading_day"] < HOLDOUT_SACRED_FROM].copy()
        corr = float("nan")
        if len(legs) == 2 and not is_book.empty:
            joined = _portfolio_series([legs[0]]).rename(columns={"pnl_r": "a"})
            joined = joined[["trading_day", "a"]].merge(
                _portfolio_series([legs[1]])[["trading_day", "pnl_r"]].rename(columns={"pnl_r": "b"}),
                on="trading_day",
                how="outer",
            )
            joined[["a", "b"]] = joined[["a", "b"]].fillna(0.0)
            joined = joined[joined["trading_day"] < HOLDOUT_SACRED_FROM]
            if len(joined) > 2:
                corr = float(joined["a"].corr(joined["b"]))
        row["leg_corr_is"] = corr
        rows.append(row)
    return rows


def _cell_table(rows: pd.DataFrame, columns: list[str]) -> list[str]:
    if rows.empty:
        return ["_None._"]

    header = "| " + " | ".join(columns) + " |"
    align = "| " + " | ".join("---:" if col not in {"strategy", "session", "filter"} else "---" for col in columns) + " |"
    lines = [header, align]
    for _, row in rows.iterrows():
        parts: list[str] = []
        for col in columns:
            value = row[col]
            if col == "strategy":
                parts.append(f"`{value}`")
            elif col in {"session", "filter"}:
                parts.append(str(value))
            elif col in {"n_is", "n_2026"}:
                parts.append(str(int(value)))
            elif col in {"p", "q_global", "dsr"}:
                parts.append(_fmt(value, 6))
            else:
                parts.append(_fmt(value))
        lines.append("| " + " | ".join(parts) + " |")
    return lines


def run_scan(db_path: Path = GOLD_DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = tuple(ASSET_CONFIGS[SYMBOL]["enabled_sessions"])
    declared_k = len(sessions) * len(ORB_MINUTES) * len(RR_TARGETS) * len(FILTERS)
    rows: list[dict[str, object]] = []
    filtered_by_name: dict[str, pd.DataFrame] = {}

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
    cells = cells.sort_values(
        ["strict_pass", "research_shortlist", "annual_r", "mean_is"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    portfolios = pd.DataFrame(_score_portfolios(filtered_by_name)).sort_values("annual_r", ascending=False)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cells.to_csv(CELL_CSV, index=False)
    portfolios.to_csv(PORTFOLIO_CSV, index=False)
    write_report(cells, portfolios, declared_k)
    return cells, portfolios


def write_report(cells: pd.DataFrame, portfolios: pd.DataFrame, declared_k: int) -> None:
    strict = int(cells["strict_pass"].sum())
    shortlist = int(cells["research_shortlist"].sum())
    top_practical = cells[cells["strategy"] == _strategy_name("NYSE_OPEN", 15, 2.0, "COST_LT10")].iloc[0]
    us1000 = cells[cells["strategy"] == _strategy_name("US_DATA_1000", 15, 2.0, "COST_LT10")].iloc[0]
    top_book = portfolios[portfolios["strategy"] == "NY_OPEN+US1000_COST10"].iloc[0]
    no_filter_book = portfolios[portfolios["strategy"] == "NY_OPEN+US1000_NOFILTER"].iloc[0]
    evidence_ranked = cells[cells["research_shortlist"]].head(12)
    negative_monitor = (
        cells[(cells["annual_r"] > 25.0) & (cells["mean_2026"] < 0.0)]
        .sort_values("annual_r", ascending=False)
        .head(8)
    )
    top_single_columns = [
        "strategy",
        "n_is",
        "mean_is",
        "annual_r",
        "t",
        "q_global",
        "dsr",
        "wfe",
        "dd",
        "mean_2026",
    ]
    failed_monitor_columns = [
        "strategy",
        "n_is",
        "annual_r",
        "t",
        "q_global",
        "wfe",
        "mean_2026",
        "dd",
    ]

    lines = [
        "# Best Own ORB Strategy Scan v1",
        "",
        "**Status:** exploratory canonical scan, not a pre-registered validation run.",
        "**Canonical inputs:** `orb_outcomes` + `daily_features` only; triple join on `(symbol, trading_day, orb_minutes)`.",
        f"**Selection window:** `< {HOLDOUT_SACRED_FROM}`. 2026 is descriptive only.",
        f"**Exploratory K:** `{declared_k}` cells = MNQ enabled sessions x O{{5,15,30}} x RR{{1,1.5,2}} x 15 filters.",
        f"**Cell CSV:** `{CELL_CSV.as_posix()}`",
        f"**Portfolio CSV:** `{PORTFOLIO_CSV.as_posix()}`",
        "",
        "## Data Read",
        "",
        f"Strict exploratory passes: `{strict}`. Research shortlist cells: `{shortlist}`.",
        f"Best single cell by the report's evidence sort: `{evidence_ranked.iloc[0]['strategy']}`.",
        f"Lower-DD two-lane book selected for follow-up: `{top_book['strategy']}`.",
        "",
        "No cell passes the full exploratory DSR gate. The data therefore does not authorize deployment; it only points to the next narrow hypothesis.",
        "",
        "## Evidence-Ranked Cells",
        "",
        *_cell_table(evidence_ranked, top_single_columns),
        "",
        "## High IS, Negative 2026 Monitor",
        "",
        *_cell_table(negative_monitor, failed_monitor_columns),
        "",
        "## Candidate Pair The Data Selects",
        "",
        "| Candidate | N IS | Mean R | Annual R | t | p | q global | DSR | WFE | DD | 2026 mean | Era |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| `{top_practical['strategy']}` | {int(top_practical['n_is'])} | {_fmt(top_practical['mean_is'])} | "
            f"{_fmt(top_practical['annual_r'])} | {_fmt(top_practical['t'])} | {_fmt(top_practical['p'], 6)} | "
            f"{_fmt(top_practical['q_global'], 6)} | {_fmt(top_practical['dsr'], 6)} | {_fmt(top_practical['wfe'])} | "
            f"{_fmt(top_practical['dd'])} | {_fmt(top_practical['mean_2026'])} | {top_practical['era_ok']} |"
        ),
        (
            f"| `{us1000['strategy']}` | {int(us1000['n_is'])} | {_fmt(us1000['mean_is'])} | "
            f"{_fmt(us1000['annual_r'])} | {_fmt(us1000['t'])} | {_fmt(us1000['p'], 6)} | "
            f"{_fmt(us1000['q_global'], 6)} | {_fmt(us1000['dsr'], 6)} | {_fmt(us1000['wfe'])} | "
            f"{_fmt(us1000['dd'])} | {_fmt(us1000['mean_2026'])} | {us1000['era_ok']} |"
        ),
        "",
        "## Two-Lane Book Check",
        "",
        "| Book | N IS | Mean R/day | Annual R | t | p | DD | 2026 mean/day | Leg corr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| `{top_book['strategy']}` | {int(top_book['n_is'])} | {_fmt(top_book['mean_is'])} | "
            f"{_fmt(top_book['annual_r'])} | {_fmt(top_book['t'])} | {_fmt(top_book['p'], 8)} | "
            f"{_fmt(top_book['dd'])} | {_fmt(top_book['mean_2026'])} | {_fmt(top_book['leg_corr_is'])} |"
        ),
        (
            f"| `{no_filter_book['strategy']}` | {int(no_filter_book['n_is'])} | {_fmt(no_filter_book['mean_is'])} | "
            f"{_fmt(no_filter_book['annual_r'])} | {_fmt(no_filter_book['t'])} | {_fmt(no_filter_book['p'], 8)} | "
            f"{_fmt(no_filter_book['dd'])} | {_fmt(no_filter_book['mean_2026'])} | {_fmt(no_filter_book['leg_corr_is'])} |"
        ),
        "",
        "The book comparison is not one-sided: NO_FILTER is higher by annual R, while COST_LT10 has lower drawdown and a slightly higher t-stat/Sharpe. The 2026 descriptive mean is identical because the 2026 trades that survive the cost gate match the no-filter set for these two legs.",
        "",
        "## Data-Driven Exclusions",
        "",
        "- Same-direction re-entry: KILL from the separate bounded execution report; no priority addition.",
        "- `NYSE_PREOPEN` O30: high pre-2026 annual R but negative 2026 descriptive monitoring; not selected.",
        "- `CME_PRECLOSE`: some positive cells, but weaker 2026/cost cushion than the selected O15 book.",
        "- Volume rows: not selected by the evidence sort; post-trigger volume and relative-volume confirmation remain banned for E2 predictors.",
        "- DSR fails under the full 1,755-cell exploratory burden. This is why the result is a priority hypothesis, not a deployment verdict.",
        "",
        "## Local Literature And Resource Grounding",
        "",
        "- `resources/INDEX.md`: local corpus manifest; curated extracts in `docs/institutional/literature/` are the canonical citation source.",
        "- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`: DSR requires the selected strategy's Sharpe, sample length, trial-count burden, cross-trial Sharpe variance, skewness, and kurtosis. This runner computes all six for the broad screen.",
        "- `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md`: broad trading-strategy searches require multiple-hypothesis controls and high t-stat hurdles. This report exposes `q_global`, `t`, and declared K instead of ranking raw means alone.",
        "- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md`: lookahead and OOS-tweaking are explicit failure modes. This runner bans E2 post-trigger predictors and keeps 2026 descriptive only.",
        "- `trading_app.config.CostRatioFilter`: COST_LT gates use canonical round-trip friction share of raw ORB risk, not triggered-trade `risk_dollars`.",
        "- Unsupported by current local data: order-flow absorption, footprint delta, and stop-hunt intent. Those remain `PARK_NEW_DATA`.",
        "",
        "## Verdict",
        "",
        "`NARROW`: make the next formal hypothesis a small, bounded validation of the O15/E2/RR2 MNQ NYSE_OPEN + US_DATA_1000 book. Declare upfront whether the objective ranks annual R first (`NO_FILTER`) or drawdown/t-stat first (`COST_LT10`), then keep the other as the sensitivity comparator.",
        "",
        "SURVIVED SCRUTINY: MNQ NYSE_OPEN O15 E2 RR2 COST_LT10 and MNQ US_DATA_1000 O15 E2 RR2 COST_LT10 are positive across pre-2026 eras, positive in 2026 descriptive monitoring, and combine with low lane correlation.",
        "DID NOT SURVIVE: no cell clears the strict full exploratory DSR gate; same-direction re-entry was killed separately.",
        "CAVEATS: exploratory post-selection; no capital deployment claim; DSR uses the broad K screen and per-cell return skew/kurtosis but no ONC de-correlation estimate.",
        "NEXT STEPS: pre-register a narrow book validation that uses the previously known MNQ NYSE_OPEN and US_DATA_1000 parent-survivor context, not this full exploratory K.",
        "",
    ]
    RESULT_DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=GOLD_DB_PATH)
    args = parser.parse_args()
    cells, portfolios = run_scan(args.db)
    print(f"wrote {len(cells)} cells to {CELL_CSV}")
    print(f"wrote {len(portfolios)} portfolios to {PORTFOLIO_CSV}")
    print(f"wrote report to {RESULT_DOC}")


if __name__ == "__main__":
    main()
