"""Stage B2 runner — MNQ US_DATA_1000 VWAP_MID_ALIGNED family-pooled Holm + clustered SE.

Bounded conditional-role executor for the single locked pre-reg at
``docs/audit/hypotheses/2026-05-17-mnq-usdata1000-vwapmid-family-pooled-oos-v1.yaml``
(quarantine path under ``drafts/`` also accepted).

Re-executes the 4-cell MNQ US_DATA_1000 VWAP_MID_ALIGNED family
(O15/O30 x RR1.0/RR1.5) under clustered standard errors at trading_day
level, applies the no-theory Chordia t>=3.79 hurdle AND the locked
Holm-Bonferroni K=4 FWER gate per cell. Both gates must pass independently.
The pooled t-stat is descriptive_only and never enters the verdict.

Route contract (per pre-reg ``not_done_by_this_pre_reg`` and Stage B2 plan):

- No writes to validated_setups, experimental_strategies, edge_families.
- No writes to lane_allocation.json, live_config.json, bot_state.json.
- No writes to docs/runtime/chordia_audit_log.yaml.
- No modification of prior PASS_CHORDIA chordia_audit_log.yaml entries.

Execution gate (BLOCKING — sole protection during quarantine):

The runner reads ``execution_gate.allowed_now`` from the pre-reg.
- ``allowed_now: false`` + no ``--dry-run`` -> exit non-zero before any DB read.
- ``allowed_now: false`` + ``--dry-run`` -> compute and print, write nothing.
- ``allowed_now: true`` + no ``--dry-run`` -> write the three artifact paths.
- ``allowed_now: true`` + ``--dry-run`` -> compute and print, write nothing.

A defensive ``assert`` at the artifact-write boundary repeats the gate check.

Outputs (only when gate open):

- ``docs/audit/results/<stem>.md``
- ``docs/audit/results/<stem>.per-cell.csv``
- ``docs/audit/results/<stem>.by-year.csv``
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import yaml
from scipy import stats as scipy_stats

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal
from research.oos_power import one_sample_power, power_verdict
from trading_app.chordia import CHORDIA_T_WITHOUT_THEORY, compute_chordia_t
from trading_app.config import ALL_FILTERS, WF_START_OVERRIDE, VWAPBreakDirectionFilter
from trading_app.eligibility.builder import parse_strategy_id
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_POOLED_T_ROLE = "DESCRIPTIVE_ONLY"
EXPECTED_TRIALS = 4
EXPECTED_K_FAMILY = 4
EXPECTED_FILTER_KEY = "VWAP_MID_ALIGNED"
EXPECTED_INSTRUMENT = "MNQ"
EXPECTED_SESSION = "US_DATA_1000"
EXPECTED_SCRATCH_POLICY = "realized-eod"
CLUSTER_SKEW_FLOOR = 30
RECONCILIATION_HALT_TOLERANCE = 0.50
RECONCILIATION_BLOCK_TOLERANCE = 0.10
HETEROGENEITY_FLIP_RATE_PCT_FLOOR = 25.0
CLUSTERING_INFLATION_WARNING_DELTA = 0.5
HOLM_VERDICT_PASS = "PASS_FAMILY_HOLM"
HOLM_VERDICT_FAIL = "FAIL_FAMILY"
ALLOWED_VERDICT_STRINGS = frozenset({HOLM_VERDICT_PASS, HOLM_VERDICT_FAIL})


# ---------------------------------------------------------------------------
# Pre-reg deserialization + sentinel chain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    strategy_id: str
    instrument: str
    orb_label: str
    orb_minutes: int
    entry_model: str
    confirm_bars: int
    rr_target: float
    filter_key: str
    prior_chordia_status: str
    prior_t_stat: float | None
    prior_audit_log_ref: str | None


@dataclass(frozen=True)
class Hypothesis:
    yaml_path: Path
    stem: str
    cells: tuple[CellSpec, ...]
    holm_thresholds: tuple[float, float, float, float]
    allowed_now: bool
    chordia_threshold_basis_t: float


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        sys.stderr.write(f"REFUSE: {msg}\n")
        raise SystemExit(2)


def _load_hypothesis(yaml_path: Path) -> Hypothesis:
    body = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    schema = body.get("primary_schema", {})
    scope = body.get("scope", {})
    data_policy = body.get("data_policy", {})
    gate = body.get("execution_gate", {})
    feature = body.get("feature_definition", {})

    _assert(
        schema.get("pooled_t_role_assert") == EXPECTED_POOLED_T_ROLE,
        f"primary_schema.pooled_t_role_assert must equal {EXPECTED_POOLED_T_ROLE!r}, "
        f"got {schema.get('pooled_t_role_assert')!r}. BLOCKING-B sentinel.",
    )
    _assert(
        int(body.get("total_expected_trials", -1)) == EXPECTED_TRIALS,
        f"total_expected_trials must equal {EXPECTED_TRIALS}, "
        f"got {body.get('total_expected_trials')!r}.",
    )
    _assert(
        int(schema.get("k_family", -1)) == EXPECTED_K_FAMILY,
        f"primary_schema.k_family must equal {EXPECTED_K_FAMILY}, "
        f"got {schema.get('k_family')!r}.",
    )
    _assert(
        str(scope.get("filter_type")) == EXPECTED_FILTER_KEY,
        f"scope.filter_type must equal {EXPECTED_FILTER_KEY!r}, "
        f"got {scope.get('filter_type')!r}.",
    )
    _assert(
        str(scope.get("instrument")) == EXPECTED_INSTRUMENT,
        f"scope.instrument must equal {EXPECTED_INSTRUMENT!r}, got {scope.get('instrument')!r}.",
    )
    _assert(
        str(scope.get("session")) == EXPECTED_SESSION,
        f"scope.session must equal {EXPECTED_SESSION!r}, got {scope.get('session')!r}.",
    )
    _assert(
        str(data_policy.get("scratch_policy")) == EXPECTED_SCRATCH_POLICY,
        f"data_policy.scratch_policy must equal {EXPECTED_SCRATCH_POLICY!r}, "
        f"got {data_policy.get('scratch_policy')!r}.",
    )
    _assert(
        str(feature.get("feature_name")) == EXPECTED_FILTER_KEY,
        f"feature_definition.feature_name must equal {EXPECTED_FILTER_KEY!r}, "
        f"got {feature.get('feature_name')!r}.",
    )

    # Canonical filter delegation self-check: the canonical instance must be a
    # VWAPBreakDirectionFilter with definition='orb_mid' (per pre-reg notes).
    filt = ALL_FILTERS.get(EXPECTED_FILTER_KEY)
    _assert(
        isinstance(filt, VWAPBreakDirectionFilter),
        f"ALL_FILTERS[{EXPECTED_FILTER_KEY!r}] is not a VWAPBreakDirectionFilter "
        f"(got {type(filt).__name__}).",
    )
    _assert(
        getattr(filt, "definition", None) == "orb_mid",
        f"ALL_FILTERS[{EXPECTED_FILTER_KEY!r}].definition must equal 'orb_mid', "
        f"got {getattr(filt, 'definition', None)!r}.",
    )

    raw_cells = schema.get("family_cells", [])
    _assert(
        isinstance(raw_cells, list) and len(raw_cells) == EXPECTED_TRIALS,
        f"primary_schema.family_cells must be a list of {EXPECTED_TRIALS} cells.",
    )
    parsed: list[CellSpec] = []
    for raw in raw_cells:
        for key in ("strategy_id", "orb_minutes", "rr_target", "filter"):
            _assert(
                key in raw,
                f"family_cells entry missing required field {key!r}: {raw!r}",
            )
        sid_parsed = parse_strategy_id(str(raw["strategy_id"]))
        _assert(
            sid_parsed["instrument"] == EXPECTED_INSTRUMENT
            and sid_parsed["orb_label"] == EXPECTED_SESSION
            and sid_parsed["filter_type"] == EXPECTED_FILTER_KEY,
            f"strategy_id {raw['strategy_id']!r} does not parse to MNQ/US_DATA_1000/VWAP_MID_ALIGNED.",
        )
        _assert(
            int(sid_parsed["orb_minutes"]) == int(raw["orb_minutes"])
            and abs(float(sid_parsed["rr_target"]) - float(raw["rr_target"])) < 1e-9,
            f"strategy_id {raw['strategy_id']!r} dimensions disagree with cell metadata.",
        )
        parsed.append(
            CellSpec(
                cell_id=str(raw.get("id", raw["strategy_id"])),
                strategy_id=str(raw["strategy_id"]),
                instrument=sid_parsed["instrument"],
                orb_label=sid_parsed["orb_label"],
                orb_minutes=int(sid_parsed["orb_minutes"]),
                entry_model=sid_parsed["entry_model"],
                confirm_bars=int(sid_parsed["confirm_bars"]),
                rr_target=float(sid_parsed["rr_target"]),
                filter_key=sid_parsed["filter_type"],
                prior_chordia_status=str(raw.get("prior_chordia_status", "UNAUDITED")),
                prior_t_stat=(
                    float(raw["prior_t_stat"])
                    if raw.get("prior_t_stat") is not None
                    else None
                ),
                prior_audit_log_ref=(
                    str(raw["prior_audit_log_ref"])
                    if raw.get("prior_audit_log_ref")
                    else None
                ),
            )
        )

    thresholds_block = schema.get("holm_bonferroni_thresholds", {}).get("thresholds", {})
    thresholds = (
        float(thresholds_block["alpha_prime_1"]),
        float(thresholds_block["alpha_prime_2"]),
        float(thresholds_block["alpha_prime_3"]),
        float(thresholds_block["alpha_prime_4"]),
    )
    # Defensive: Holm thresholds must be monotone non-decreasing by rank index.
    _assert(
        all(thresholds[i] <= thresholds[i + 1] for i in range(3)),
        f"Holm thresholds must be monotone non-decreasing, got {thresholds}.",
    )

    # Chordia threshold basis: extract numeric from prose. ASCII '>= 3.79' required.
    import re

    basis = str(schema.get("chordia_threshold_basis", ""))
    _assert(
        "≥" not in basis,
        "chordia_threshold_basis contains Unicode '>=' (U+2265); ASCII '>=' required.",
    )
    m = re.search(r"t_clustered\s*>=\s*(\d+\.\d+)", basis)
    if m is None:
        _assert(
            False,
            f"chordia_threshold_basis must contain ASCII 't_clustered >= <float>', got: {basis!r}",
        )
        raise AssertionError  # unreachable; satisfies narrowing
    chordia_t = float(m.group(1))
    _assert(
        abs(chordia_t - CHORDIA_T_WITHOUT_THEORY) < 1e-6,
        f"chordia_threshold_basis declares t>={chordia_t} which does not match "
        f"CHORDIA_T_WITHOUT_THEORY={CHORDIA_T_WITHOUT_THEORY}.",
    )

    allowed_now = bool(gate.get("allowed_now", False))

    return Hypothesis(
        yaml_path=yaml_path,
        stem=yaml_path.stem,
        cells=tuple(parsed),
        holm_thresholds=thresholds,
        allowed_now=allowed_now,
        chordia_threshold_basis_t=chordia_t,
    )


# ---------------------------------------------------------------------------
# Canonical IS / OOS load (mirrors chordia_strict_unlock_v1 join contract)
# ---------------------------------------------------------------------------


def _load_universe(
    con: duckdb.DuckDBPyConnection,
    cell: CellSpec,
    *,
    is_only: bool,
) -> pd.DataFrame:
    op = "<" if is_only else ">="
    start = WF_START_OVERRIDE.get(cell.instrument)
    start_clause = "AND o.trading_day >= ?" if start is not None else ""
    sql = f"""
        SELECT
            o.trading_day,
            o.symbol,
            o.orb_label,
            o.orb_minutes,
            o.entry_model,
            o.confirm_bars,
            o.rr_target,
            o.outcome,
            o.entry_price,
            o.target_price,
            o.stop_price,
            o.pnl_r,
            d.*
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.trading_day = d.trading_day
           AND o.symbol = d.symbol
           AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.entry_model = ?
          AND o.confirm_bars = ?
          AND o.rr_target = ?
          AND o.trading_day {op} ?
          AND o.outcome IS NOT NULL
          {start_clause}
        ORDER BY o.trading_day
    """
    params: list[Any] = [
        cell.instrument,
        cell.orb_label,
        cell.orb_minutes,
        cell.entry_model,
        cell.confirm_bars,
        cell.rr_target,
        HOLDOUT_SACRED_FROM,
    ]
    if start is not None:
        params.append(start)
    return con.execute(sql, params).df()


def _direction_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [
            "long" if tp > sp else "short"
            for tp, sp in zip(df["target_price"], df["stop_price"], strict=False)
        ],
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Per-frame metrics — clustered SE, naive t, direction split, per-year roll-up
# ---------------------------------------------------------------------------


def _two_sided_p_from_t(t_value: float, df: float) -> float:
    if not math.isfinite(t_value) or df <= 0:
        return float("nan")
    return float(2.0 * scipy_stats.t.sf(abs(t_value), df))


def _clustered_t(pnl: np.ndarray, cluster: np.ndarray) -> tuple[float, float, float, float]:
    """Return (coef, t_clustered, p_clustered, df) from an intercept-only OLS.

    Uses statsmodels OLS with ``cov_type='cluster'`` and
    ``cov_kwds={'groups': cluster}``. The cluster-robust degrees of freedom
    reported by statsmodels (Stata convention: G - 1 where G is the number
    of clusters) are used for the two-sided p-value.

    Returns ``(nan, nan, nan, 0)`` if N<2 or fewer than 2 distinct clusters.
    """
    n = len(pnl)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), 0.0
    groups = pd.Series(cluster)
    if groups.nunique() < 2:
        return float(np.mean(pnl)), float("nan"), float("nan"), 0.0
    y = np.asarray(pnl, dtype=float)
    X = np.ones((n, 1), dtype=float)
    model = sm.OLS(y, X)
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": groups.to_numpy()})
    coef = float(fit.params[0])
    se = float(fit.bse[0])
    if not math.isfinite(se) or se <= 0:
        return coef, float("nan"), float("nan"), 0.0
    df_used = float(getattr(fit, "df_resid", n - 1))
    # statsmodels uses cluster-robust df = G - 1 for inference under
    # cov_type='cluster'; surface that explicitly when available.
    cov_df = getattr(fit, "df_resid_inference", None)
    if cov_df is not None and math.isfinite(float(cov_df)) and float(cov_df) > 0:
        df_used = float(cov_df)
    t_clustered = coef / se
    p_clustered = _two_sided_p_from_t(t_clustered, df_used)
    return coef, float(t_clustered), float(p_clustered), float(df_used)


def _frame_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute all per-cell-per-period metrics on a fired-trade frame.

    Inputs: ``df`` with columns ``trading_day``, ``pnl_r`` (raw, NULL=scratch),
    ``outcome``, ``target_price``, ``stop_price``. Scratches are coerced to
    pnl=0.0 per realized-eod policy. Returns a metrics dict; empty inputs
    return NaN-filled metrics with N_trades=0.
    """
    out: dict[str, Any] = {
        "n_trades": 0,
        "n_unique_trading_days": 0,
        "cluster_size_mean": float("nan"),
        "cluster_size_max": 0,
        "scratch_n": 0,
        "null_non_scratch_n": 0,
        "expr": float("nan"),
        "std_r": float("nan"),
        "sharpe": float("nan"),
        "t_naive": float("nan"),
        "p_naive": float("nan"),
        "t_clustered": float("nan"),
        "p_clustered": float("nan"),
        "df_clustered": 0.0,
        "long_n": 0,
        "long_expr": float("nan"),
        "short_n": 0,
        "short_expr": float("nan"),
        "oos_dir_match": None,  # filled later for OOS frames
    }
    if df.empty:
        return out

    work = df.copy()
    scratch_mask = work["outcome"].astype(str).eq("scratch")
    null_mask = work["pnl_r"].isna()
    null_non_scratch = int((null_mask & ~scratch_mask).sum())
    work["pnl_eff"] = work["pnl_r"].fillna(0.0)
    # Realized-eod sanity: every scratch row must coerce to 0.0; assert that
    # the coercion left no surviving NaN that should not be there.
    assert work["pnl_eff"].notna().all(), "scratch coercion left residual NaN in pnl_eff"

    n = int(len(work))
    days = work["trading_day"]
    cluster_sizes = work.groupby("trading_day").size()
    n_clusters = int(cluster_sizes.size)
    mean_r = float(work["pnl_eff"].mean())
    std_r = float(work["pnl_eff"].std(ddof=1)) if n >= 2 else float("nan")
    sharpe = mean_r / std_r if n >= 2 and std_r > 0 else float("nan")
    t_naive = compute_chordia_t(sharpe, n) if n >= 2 and std_r > 0 else float("nan")
    p_naive = _two_sided_p_from_t(t_naive, n - 1) if math.isfinite(t_naive) else float("nan")

    _, t_clust, p_clust, df_clust = _clustered_t(
        work["pnl_eff"].to_numpy(),
        days.to_numpy(),
    )

    # Direction split (long-only vs short-only ExpR / N).
    long_n = short_n = 0
    long_expr = short_expr = float("nan")
    if {"target_price", "stop_price"}.issubset(work.columns):
        directions = _direction_series(work)
        for key in ("long", "short"):
            sub = work.loc[directions.eq(key), "pnl_eff"]
            sub_n = int(len(sub))
            if key == "long":
                long_n = sub_n
                long_expr = float(sub.mean()) if sub_n else float("nan")
            else:
                short_n = sub_n
                short_expr = float(sub.mean()) if sub_n else float("nan")

    out.update(
        {
            "n_trades": n,
            "n_unique_trading_days": n_clusters,
            "cluster_size_mean": float(cluster_sizes.mean()) if n_clusters else float("nan"),
            "cluster_size_max": int(cluster_sizes.max()) if n_clusters else 0,
            "scratch_n": int(scratch_mask.sum()),
            "null_non_scratch_n": null_non_scratch,
            "expr": mean_r,
            "std_r": std_r,
            "sharpe": sharpe,
            "t_naive": float(t_naive) if math.isfinite(t_naive) else float("nan"),
            "p_naive": p_naive,
            "t_clustered": t_clust,
            "p_clustered": p_clust,
            "df_clustered": df_clust,
            "long_n": long_n,
            "long_expr": long_expr,
            "short_n": short_n,
            "short_expr": short_expr,
        }
    )
    # Fail-closed on the silent-coercion class bug
    # (feedback_scratch_pnl_null_class_bug.md): non-scratch NULL pnl_r rows
    # must never reach pnl_eff via fillna(0). Realized-eod policy permits
    # NULL only on outcome=='scratch'; any other NULL is corrupt input.
    _assert(
        int(out["null_non_scratch_n"]) == 0,
        f"non-scratch NULL pnl_r rows in frame: "
        f"{int(out['null_non_scratch_n'])} (must be 0 under realized-eod policy)",
    )
    return out


def _yearly_rollup(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-year (calendar year) metrics for any year with >= 10 trades.

    Used for the by-year CSV. NULL pnl_r coerced to 0 (scratch policy).
    """
    if df.empty:
        return []
    work = df.copy()
    work["pnl_eff"] = work["pnl_r"].fillna(0.0)
    years = pd.to_datetime(work["trading_day"]).dt.year
    rows: list[dict[str, Any]] = []
    directions = (
        _direction_series(work)
        if {"target_price", "stop_price"}.issubset(work.columns)
        else None
    )
    for yr_key, idx in work.groupby(years).groups.items():
        year_int = int(yr_key)  # type: ignore[arg-type]
        sub = work.loc[idx]
        if len(sub) < 10:
            continue
        sub_dir = "mixed"
        if directions is not None:
            unique_dirs = directions.loc[idx].unique()
            if len(unique_dirs) == 1:
                sub_dir = str(unique_dirs[0])
        sub_metrics = _frame_metrics(sub.drop(columns=["pnl_eff"]) if "pnl_eff" in sub.columns else sub)
        rows.append(
            {
                "year": year_int,
                "n_trades": sub_metrics["n_trades"],
                "n_unique_trading_days": sub_metrics["n_unique_trading_days"],
                "expr": sub_metrics["expr"],
                "t_naive": sub_metrics["t_naive"],
                "t_clustered": sub_metrics["t_clustered"],
                "p_clustered": sub_metrics["p_clustered"],
                "direction": sub_dir,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Cell pipeline: load, filter, metrics, kill clauses, reconciliation
# ---------------------------------------------------------------------------


def _evaluate_cell(
    con: duckdb.DuckDBPyConnection,
    cell: CellSpec,
) -> dict[str, Any]:
    is_df = _load_universe(con, cell, is_only=True)
    oos_df = _load_universe(con, cell, is_only=False)

    def _apply_filter(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        mask = filter_signal(df, cell.filter_key, cell.orb_label).astype(bool)
        return df.loc[np.asarray(mask, dtype=bool)].copy()

    is_fired = _apply_filter(is_df)
    oos_fired = _apply_filter(oos_df)

    is_metrics = _frame_metrics(is_fired)
    oos_metrics = _frame_metrics(oos_fired)
    by_year = _yearly_rollup(is_fired)

    # OOS dir_match (descriptive, not verdict-driving per RULE 3.3).
    if oos_metrics["n_trades"] > 0 and math.isfinite(is_metrics["expr"]) and math.isfinite(oos_metrics["expr"]):
        oos_metrics["oos_dir_match"] = bool(
            (is_metrics["expr"] > 0) == (oos_metrics["expr"] > 0)
        )

    # OOS power tier on the IS effect size (per-cell). One-sample t-test power
    # on the OOS replay of the IS effect; delegates to canonical
    # `research.oos_power.one_sample_power` (NCP = d * sqrt(n)).
    oos_power_tier = "INSUFFICIENT_DATA"
    oos_power = float("nan")
    if (
        is_metrics["n_trades"] >= 2
        and oos_metrics["n_trades"] >= 2
        and math.isfinite(is_metrics["expr"])
        and math.isfinite(is_metrics["std_r"])
        and is_metrics["std_r"] > 0
    ):
        d_is = float(is_metrics["expr"]) / float(is_metrics["std_r"])
        n_oos = int(oos_metrics["n_trades"])
        try:
            oos_power = float(one_sample_power(d_is, n_oos, alpha=0.05))
            oos_power_tier = power_verdict(oos_power)
        except ValueError:
            oos_power_tier = "INSUFFICIENT_DATA"

    return {
        "cell": cell,
        "is_metrics": is_metrics,
        "oos_metrics": oos_metrics,
        "by_year": by_year,
        "oos_power": oos_power,
        "oos_power_tier": oos_power_tier,
        "is_fired_frame": is_fired,  # retained for pooled descriptive fit
    }


def _evaluate_cell_kills(
    cell_eval: dict[str, Any],
    chordia_t: float,
    holm_alpha_prime: float | None,
) -> dict[str, Any]:
    """Per-cell kill-clause evaluation.

    Returns dict with ``pass_chordia``, ``pass_holm`` (None when alpha_prime is None),
    ``cluster_skew_kill``, ``naive_flip_chordia``, ``naive_flip_holm``,
    ``expr_sign_kill``, ``cell_verdict_label``.
    """
    m = cell_eval["is_metrics"]
    n_clusters = int(m.get("n_unique_trading_days", 0) or 0)
    t_naive = float(m.get("t_naive", float("nan")))
    t_clust = float(m.get("t_clustered", float("nan")))
    p_naive = float(m.get("p_naive", float("nan")))
    p_clust = float(m.get("p_clustered", float("nan")))
    expr_is = float(m.get("expr", float("nan")))

    cluster_skew_kill = n_clusters < CLUSTER_SKEW_FLOOR
    pass_chordia = (
        math.isfinite(t_clust) and t_clust >= chordia_t and not cluster_skew_kill
    )
    pass_holm = None
    if holm_alpha_prime is not None:
        pass_holm = (
            math.isfinite(p_clust)
            and p_clust <= holm_alpha_prime
            and not cluster_skew_kill
        )

    naive_flip_chordia = (
        math.isfinite(t_naive)
        and math.isfinite(t_clust)
        and t_naive >= chordia_t
        and t_clust < chordia_t
    )
    naive_flip_holm = False
    if holm_alpha_prime is not None:
        naive_flip_holm = (
            math.isfinite(p_naive)
            and math.isfinite(p_clust)
            and p_naive <= holm_alpha_prime
            and p_clust > holm_alpha_prime
        )
    expr_sign_kill = math.isfinite(expr_is) and expr_is <= 0.0

    if cluster_skew_kill:
        cell_verdict_label = "UNVERIFIED_CLUSTER_SKEW"
    elif not math.isfinite(t_clust):
        cell_verdict_label = "UNVERIFIED_INSUFFICIENT_DATA"
    elif expr_sign_kill:
        cell_verdict_label = "FAIL_EXPR_NONPOSITIVE"
    elif naive_flip_chordia:
        cell_verdict_label = "FAIL_NAIVE_FLIP_CHORDIA"
    elif naive_flip_holm:
        cell_verdict_label = "FAIL_NAIVE_FLIP_HOLM"
    elif not pass_chordia:
        cell_verdict_label = "FAIL_CHORDIA"
    elif holm_alpha_prime is not None and not pass_holm:
        cell_verdict_label = "FAIL_HOLM"
    else:
        cell_verdict_label = "PASS_CELL"

    return {
        "pass_chordia": pass_chordia,
        "pass_holm": pass_holm,
        "cluster_skew_kill": cluster_skew_kill,
        "naive_flip_chordia": naive_flip_chordia,
        "naive_flip_holm": naive_flip_holm,
        "expr_sign_kill": expr_sign_kill,
        "cell_verdict_label": cell_verdict_label,
    }


def _load_audit_log_t_stats(repo_root: Path) -> dict[str, float]:
    log_path = repo_root / "docs" / "runtime" / "chordia_audit_log.yaml"
    if not log_path.exists():
        return {}
    body = yaml.safe_load(log_path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for entry in body.get("audits", []) or []:
        sid = entry.get("strategy_id")
        t_stat = entry.get("t_stat")
        if sid and t_stat is not None:
            try:
                out[str(sid)] = float(t_stat)
            except (TypeError, ValueError):
                continue
    return out


def _reconcile(
    cell: CellSpec,
    t_clustered: float,
    audit_log_t_stats: dict[str, float],
) -> dict[str, Any]:
    """Per-cell reconciliation vs chordia_audit_log.yaml prior t_stat.

    Halts the run if |delta_t| > 0.50 on any prior-PASS cell. Skips the
    O30 RR1.5 unaudited cell.
    """
    out: dict[str, Any] = {
        "applicable": False,
        "prior_t_stat": cell.prior_t_stat,
        "audit_log_t_stat": audit_log_t_stats.get(cell.strategy_id),
        "delta_t": None,
        "tier": "SKIPPED_UNAUDITED",
        "halt": False,
    }
    if cell.prior_chordia_status == "UNAUDITED" or cell.prior_t_stat is None:
        return out
    # Prefer the audit log value (canonical anchor per primary_schema
    # reconciliation_rule); fall back to pre-reg `prior_t_stat` if the log
    # entry is missing.
    anchor = out["audit_log_t_stat"] if out["audit_log_t_stat"] is not None else cell.prior_t_stat
    out["applicable"] = True
    if not math.isfinite(t_clustered):
        out["tier"] = "UNDEFINED_THIS_RUN"
        return out
    delta = float(t_clustered) - float(anchor)
    out["delta_t"] = delta
    abs_delta = abs(delta)
    if abs_delta <= RECONCILIATION_BLOCK_TOLERANCE:
        out["tier"] = "PASS_SILENT"
    elif abs_delta <= RECONCILIATION_HALT_TOLERANCE:
        out["tier"] = "PASS_WITH_BLOCK"
    else:
        out["tier"] = "HALT_DIVERGENCE"
        out["halt"] = True
    return out


# ---------------------------------------------------------------------------
# Holm rank + family verdict + H3 robustness
# ---------------------------------------------------------------------------


def _holm_assign(
    cell_evals: list[dict[str, Any]],
    thresholds: tuple[float, ...],
) -> list[float | None]:
    """Return alpha_prime for each cell (in input order) per ascending-p_clustered rank.

    The locked thresholds are indexed 0..K-1 matching alpha'_1..alpha'_K.
    Cells with non-finite p_clustered receive None (treated as fail upstream).
    """
    indexed = []
    for i, ev in enumerate(cell_evals):
        p = float(ev["is_metrics"].get("p_clustered", float("nan")))
        indexed.append((i, p))
    finite_only = [pair for pair in indexed if math.isfinite(pair[1])]
    finite_only.sort(key=lambda pair: pair[1])
    alpha_for: list[float | None] = [None] * len(cell_evals)
    for rank_idx, (orig_i, _) in enumerate(finite_only):
        if rank_idx < len(thresholds):
            alpha_for[orig_i] = thresholds[rank_idx]
    return alpha_for


def _family_verdict(per_cell_kills: list[dict[str, Any]]) -> str:
    pass_all = all(k.get("cell_verdict_label") == "PASS_CELL" for k in per_cell_kills)
    verdict = HOLM_VERDICT_PASS if pass_all else HOLM_VERDICT_FAIL
    assert verdict in ALLOWED_VERDICT_STRINGS, (
        f"family verdict {verdict!r} not in locked taxonomy "
        f"{sorted(ALLOWED_VERDICT_STRINGS)}; PASS_PARTIAL_HOLM is forbidden."
    )
    return verdict


def _h3_robustness(
    cell_evals: list[dict[str, Any]],
    k4_verdicts: list[str],
    cells: list[CellSpec],
) -> dict[str, Any]:
    prior_pass_idx = [
        i for i, c in enumerate(cells) if c.prior_chordia_status == "PASS_CHORDIA"
    ]
    if len(prior_pass_idx) != 3:
        return {"applicable": False, "h3_isolation_pass": None, "rows": []}
    k3_thresholds = (0.05 / 3, 0.05 / 2, 0.05 / 1)
    sub_evals = [cell_evals[i] for i in prior_pass_idx]
    alphas = _holm_assign(sub_evals, k3_thresholds)
    rows: list[dict[str, Any]] = []
    isolation_pass = True
    for local_idx, orig_idx in enumerate(prior_pass_idx):
        k4_verdict = k4_verdicts[orig_idx]
        ev = cell_evals[orig_idx]
        m = ev["is_metrics"]
        alpha_k3 = alphas[local_idx]
        p = float(m.get("p_clustered", float("nan")))
        n_clusters = int(m.get("n_unique_trading_days", 0) or 0)
        t_clust = float(m.get("t_clustered", float("nan")))
        expr_is = float(m.get("expr", float("nan")))
        if (
            alpha_k3 is None
            or not math.isfinite(p)
            or not math.isfinite(t_clust)
            or n_clusters < CLUSTER_SKEW_FLOOR
        ):
            k3_label = "UNVERIFIED"
        elif expr_is <= 0:
            k3_label = "FAIL_EXPR_NONPOSITIVE"
        elif t_clust < CHORDIA_T_WITHOUT_THEORY:
            k3_label = "FAIL_CHORDIA"
        elif p > alpha_k3:
            k3_label = "FAIL_HOLM"
        else:
            k3_label = "PASS_CELL"
        rows.append(
            {
                "strategy_id": cells[orig_idx].strategy_id,
                "k4_verdict": k4_verdict,
                "k3_alpha_prime": alpha_k3,
                "k3_verdict": k3_label,
                "expr_is": expr_is,
            }
        )
        if k3_label != k4_verdict:
            isolation_pass = False
        if expr_is <= 0:
            isolation_pass = False
    return {
        "applicable": True,
        "h3_isolation_pass": bool(isolation_pass),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Pooled descriptive fit (IS only)
# ---------------------------------------------------------------------------


def _pooled_descriptive(
    cell_evals: list[dict[str, Any]],
) -> dict[str, Any]:
    frames = []
    for ev in cell_evals:
        frame = ev["is_fired_frame"]
        if frame.empty:
            continue
        sub = frame[["trading_day", "outcome", "pnl_r"]].copy()
        sub["pnl_eff"] = sub["pnl_r"].fillna(0.0)
        frames.append(sub)
    if not frames:
        return {
            "n_trades": 0,
            "n_unique_trading_days": 0,
            "expr": float("nan"),
            "t_clustered": float("nan"),
            "p_clustered": float("nan"),
            "flip_rate_pct": float("nan"),
            "heterogeneity_ack": False,
        }
    pooled = pd.concat(frames, ignore_index=True)
    n = int(len(pooled))
    n_days = int(pooled["trading_day"].nunique())
    expr_pooled = float(pooled["pnl_eff"].mean())
    _, t_p, p_p, _ = _clustered_t(
        pooled["pnl_eff"].to_numpy(),
        pooled["trading_day"].to_numpy(),
    )
    # flip-rate over the 4 cells: % whose per-cell ExpR sign opposes pooled sign.
    pooled_sign = 1 if expr_pooled > 0 else (-1 if expr_pooled < 0 else 0)
    flips = 0
    valid = 0
    for ev in cell_evals:
        cell_expr = float(ev["is_metrics"].get("expr", float("nan")))
        if not math.isfinite(cell_expr) or pooled_sign == 0:
            continue
        valid += 1
        cell_sign = 1 if cell_expr > 0 else (-1 if cell_expr < 0 else 0)
        if cell_sign != 0 and cell_sign != pooled_sign:
            flips += 1
    flip_rate_pct = (flips / valid * 100.0) if valid else float("nan")
    het_ack = math.isfinite(flip_rate_pct) and flip_rate_pct >= HETEROGENEITY_FLIP_RATE_PCT_FLOOR
    return {
        "n_trades": n,
        "n_unique_trading_days": n_days,
        "expr": expr_pooled,
        "t_clustered": t_p,
        "p_clustered": p_p,
        "flip_rate_pct": flip_rate_pct,
        "heterogeneity_ack": het_ack,
    }


# ---------------------------------------------------------------------------
# Pooled OOS power on family-IS-effect-size
# ---------------------------------------------------------------------------


def _pooled_oos_power(cell_evals: list[dict[str, Any]], pooled_is: dict[str, Any]) -> dict[str, Any]:
    if not math.isfinite(pooled_is["expr"]) or pooled_is["n_trades"] < 2:
        return {"measured_pooled_power": float("nan"), "pooled_power_tier": "INSUFFICIENT_DATA"}
    # Pooled OOS = concat OOS frames.
    n_oos_total = sum(int(ev["oos_metrics"].get("n_trades", 0) or 0) for ev in cell_evals)
    if n_oos_total < 4:
        return {"measured_pooled_power": float("nan"), "pooled_power_tier": "INSUFFICIENT_DATA"}
    # Need a pooled IS std. Recompute from concatenated IS effective pnl.
    is_pnl_pieces = []
    for ev in cell_evals:
        frame = ev["is_fired_frame"]
        if frame.empty:
            continue
        is_pnl_pieces.append(frame["pnl_r"].fillna(0.0))
    if not is_pnl_pieces:
        return {"measured_pooled_power": float("nan"), "pooled_power_tier": "INSUFFICIENT_DATA"}
    pooled_pnl = pd.concat(is_pnl_pieces, ignore_index=True)
    pooled_std = float(pooled_pnl.std(ddof=1)) if len(pooled_pnl) >= 2 else float("nan")
    if not math.isfinite(pooled_std) or pooled_std <= 0:
        return {"measured_pooled_power": float("nan"), "pooled_power_tier": "INSUFFICIENT_DATA"}
    # One-sample power on the pooled IS effect size replayed on n_oos_total
    # observations; delegates to canonical `research.oos_power.one_sample_power`
    # (NCP = d * sqrt(n)). Previous two-sample-halved proxy understated power
    # by ~2x for one-sample comparisons.
    d_pooled = float(pooled_is["expr"]) / pooled_std
    try:
        power_value = float(one_sample_power(d_pooled, int(n_oos_total), alpha=0.05))
        return {
            "measured_pooled_power": power_value,
            "pooled_power_tier": power_verdict(power_value),
            "power_report": {
                "cohen_d": d_pooled,
                "n_oos": int(n_oos_total),
                "power": power_value,
                "alpha": 0.05,
                "form": "one_sample",
            },
        }
    except ValueError:
        return {"measured_pooled_power": float("nan"), "pooled_power_tier": "INSUFFICIENT_DATA"}


# ---------------------------------------------------------------------------
# Artifact writers (guarded by execution gate)
# ---------------------------------------------------------------------------


def _resolve_artifact_paths(hypothesis: Hypothesis) -> dict[str, Path]:
    # Strip optional ".draft" tail from stem so quarantine and promoted runs
    # land on identical artifact paths. Pre-reg cites the non-draft slug.
    stem = hypothesis.stem
    if stem.endswith(".draft"):
        stem = stem[:-6]
    base = ROOT / "docs" / "audit" / "results" / stem
    return {
        "md": base.with_suffix(".md"),
        "per_cell_csv": Path(str(base) + ".per-cell.csv"),
        "by_year_csv": Path(str(base) + ".by-year.csv"),
    }


def _fmt(value: Any, places: int = 4) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, (float,)):
        if math.isfinite(value):
            return f"{value:.{places}f}"
        return "nan"
    if value is None:
        return ""
    return str(value)


def _write_per_cell_csv(
    path: Path,
    cells: list[CellSpec],
    cell_evals: list[dict[str, Any]],
    kills: list[dict[str, Any]],
    alphas: list[float | None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "cell_id",
                "period",
                "N_trades",
                "N_unique_trading_days",
                "cluster_size_mean",
                "cluster_size_max",
                "ExpR",
                "Sharpe",
                "t_naive",
                "t_clustered",
                "p_naive",
                "p_clustered",
                "holm_adjusted_p_clustered",
                "holm_alpha_prime",
                "pass_fail_flag",
                "direction_split",
                "oos_dir_match",
                "oos_power_tier",
            ]
        )
        for cell, ev, kill, alpha in zip(cells, cell_evals, kills, alphas, strict=True):
            for period_label, metrics in (
                ("IS", ev["is_metrics"]),
                ("OOS", ev["oos_metrics"]),
            ):
                holm_adj = (
                    metrics.get("p_clustered")
                    if period_label == "IS" and alpha is not None
                    else ""
                )
                pass_flag = kill["cell_verdict_label"] if period_label == "IS" else "descriptive"
                direction_split = (
                    f"long_n={metrics.get('long_n')},long_expr={_fmt(metrics.get('long_expr'))};"
                    f"short_n={metrics.get('short_n')},short_expr={_fmt(metrics.get('short_expr'))}"
                )
                writer.writerow(
                    [
                        cell.strategy_id,
                        period_label,
                        metrics.get("n_trades", 0),
                        metrics.get("n_unique_trading_days", 0),
                        _fmt(metrics.get("cluster_size_mean")),
                        metrics.get("cluster_size_max", 0),
                        _fmt(metrics.get("expr")),
                        _fmt(metrics.get("sharpe")),
                        _fmt(metrics.get("t_naive"), 3),
                        _fmt(metrics.get("t_clustered"), 3),
                        _fmt(metrics.get("p_naive"), 5),
                        _fmt(metrics.get("p_clustered"), 5),
                        _fmt(holm_adj, 5) if isinstance(holm_adj, float) else "",
                        _fmt(alpha) if period_label == "IS" else "",
                        pass_flag,
                        direction_split,
                        "" if metrics.get("oos_dir_match") is None else str(metrics.get("oos_dir_match")).lower(),
                        ev["oos_power_tier"] if period_label == "OOS" else "",
                    ]
                )


def _write_by_year_csv(
    path: Path,
    cells: list[CellSpec],
    cell_evals: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "cell_id",
                "year",
                "N_trades",
                "N_unique_trading_days",
                "ExpR",
                "t_naive",
                "t_clustered",
                "p_clustered",
                "direction",
            ]
        )
        for cell, ev in zip(cells, cell_evals, strict=True):
            for row in ev["by_year"]:
                writer.writerow(
                    [
                        cell.strategy_id,
                        row["year"],
                        row["n_trades"],
                        row["n_unique_trading_days"],
                        _fmt(row["expr"]),
                        _fmt(row["t_naive"], 3),
                        _fmt(row["t_clustered"], 3),
                        _fmt(row["p_clustered"], 5),
                        row["direction"],
                    ]
                )


def _write_result_md(
    md_path: Path,
    hypothesis: Hypothesis,
    cells: list[CellSpec],
    cell_evals: list[dict[str, Any]],
    kills: list[dict[str, Any]],
    alphas: list[float | None],
    family_verdict: str,
    pooled_desc: dict[str, Any],
    pooled_power: dict[str, Any],
    h3_block: dict[str, Any],
    recon_blocks: list[dict[str, Any]],
    artifact_paths: dict[str, Path],
    clustering_inflation_warning: bool,
) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    chordia_t = hypothesis.chordia_threshold_basis_t
    flip_rate = pooled_desc["flip_rate_pct"]
    het_ack = bool(pooled_desc["heterogeneity_ack"])
    per_cell_csv_rel = artifact_paths["per_cell_csv"].relative_to(ROOT).as_posix()
    by_year_csv_rel = artifact_paths["by_year_csv"].relative_to(ROOT).as_posix()
    measured_power = pooled_power.get("measured_pooled_power", float("nan"))
    pooled_tier = pooled_power.get("pooled_power_tier", "INSUFFICIENT_DATA")
    pooled_t_role_lit = "descriptive_only"

    frontmatter_lines = [
        "---",
        "pooled_finding: true",
        "verdict_test: holm_bonferroni_per_cell_clustered_se",
        f"pooled_t_role: {pooled_t_role_lit}",
        f"per_cell_breakdown_path: {per_cell_csv_rel}",
        f"measured_pooled_power: {('%.4f' % measured_power) if math.isfinite(measured_power) else 'null'}",
        f"clustering_inflation_warning: {'true' if clustering_inflation_warning else 'false'}",
        f"flip_rate_pct: {('%.2f' % flip_rate) if math.isfinite(flip_rate) else 'null'}",
    ]
    if het_ack:
        frontmatter_lines.append("heterogeneity_ack: true")
    frontmatter_lines.append("---")
    frontmatter = "\n".join(frontmatter_lines)

    cell_table_rows = []
    for cell, ev, kill, alpha in zip(cells, cell_evals, kills, alphas, strict=True):
        m = ev["is_metrics"]
        cell_table_rows.append(
            "| `{sid}` | {n} | {nc} | {csm} | {csmax} | {expr} | {sharp} | {tn} | {tc} | {pn} | {pc} | {ap} | {verdict} |".format(
                sid=cell.strategy_id,
                n=m.get("n_trades", 0),
                nc=m.get("n_unique_trading_days", 0),
                csm=_fmt(m.get("cluster_size_mean")),
                csmax=m.get("cluster_size_max", 0),
                expr=_fmt(m.get("expr")),
                sharp=_fmt(m.get("sharpe")),
                tn=_fmt(m.get("t_naive"), 3),
                tc=_fmt(m.get("t_clustered"), 3),
                pn=_fmt(m.get("p_naive"), 5),
                pc=_fmt(m.get("p_clustered"), 5),
                ap=_fmt(alpha) if alpha is not None else "n/a",
                verdict=kill["cell_verdict_label"],
            )
        )

    dir_table_rows = []
    for cell, ev in zip(cells, cell_evals, strict=True):
        m = ev["is_metrics"]
        dir_table_rows.append(
            f"| `{cell.strategy_id}` | {m['long_n']} | {_fmt(m['long_expr'])} | "
            f"{m['short_n']} | {_fmt(m['short_expr'])} |"
        )

    recon_lines: list[str] = []
    for recon in recon_blocks:
        if not recon["applicable"] or recon["tier"] in {"PASS_SILENT", "SKIPPED_UNAUDITED"}:
            continue
        recon_lines.append(
            f"- `{recon['strategy_id']}`: this-run t_clustered vs anchor "
            f"(audit_log={_fmt(recon['audit_log_t_stat'], 3)}, pre-reg prior={_fmt(recon['prior_t_stat'], 3)}), "
            f"delta_t={_fmt(recon['delta_t'], 3)}, tier=`{recon['tier']}`. "
            "Candidate causes: scratch convention, cohort boundary, statsmodels API revision."
        )
    if not recon_lines:
        recon_lines.append("- All prior-PASS cells reconcile silently (|delta_t| <= 0.10).")

    h3_lines: list[str] = []
    if h3_block["applicable"]:
        h3_lines.append(
            f"- h3_isolation_pass: **{str(h3_block['h3_isolation_pass']).lower()}**"
        )
        h3_lines.append("")
        h3_lines.append("| Strategy | K=4 verdict | K=3 alpha' | K=3 verdict |")
        h3_lines.append("|---|---|---:|---|")
        for row in h3_block["rows"]:
            h3_lines.append(
                f"| `{row['strategy_id']}` | {row['k4_verdict']} | {_fmt(row['k3_alpha_prime'])} | {row['k3_verdict']} |"
            )
    else:
        h3_lines.append("- h3_isolation_pass: n/a (fewer than 3 prior-PASS cells).")

    body_lines = [
        "",
        "# MNQ US_DATA_1000 VWAP_MID_ALIGNED family-pooled Holm + clustered SE",
        "",
        f"**Pre-reg:** `{hypothesis.yaml_path.relative_to(ROOT).as_posix()}`",
        f"**Per-cell CSV:** `{per_cell_csv_rel}`",
        f"**By-year CSV:** `{by_year_csv_rel}`",
        f"**Canonical DB:** `{GOLD_DB_PATH}`",
        f"**statsmodels version:** `{statsmodels.__version__}`",
        f"**Holdout boundary (Mode A):** `trading_day >= {HOLDOUT_SACRED_FROM}`",
        f"**Cohort lower bound:** `WF_START_OVERRIDE['MNQ']={WF_START_OVERRIDE.get('MNQ')}`",
        "",
        "## Verdict",
        "",
        f"**MEASURED family verdict:** `{family_verdict}`",
        f"**MEASURED threshold applied:** `{chordia_t:.2f}`",
        f"**MEASURED loader has_theory:** `false`",
        "",
        "Both gates are necessary; neither subsumes the other. Per-cell clustered "
        f"t_clustered >= {chordia_t:.2f} (Chordia 2018 no-theory strict t-hurdle, ASCII) "
        "AND per-cell holm_adjusted_p_clustered <= alpha'_i (Holm-Bonferroni FWER at K=4, "
        "alpha=0.05). Cluster-skew floor N_unique_trading_days >= 30 and ExpR_IS > 0 are "
        "both required for any cell to pass.",
        "",
        "## Per-cell table (IS, Mode A, clustered SE at trading_day)",
        "",
        "| Strategy | N | N_days | Cluster mean | Cluster max | ExpR | Sharpe | t_naive | t_clustered | p_naive | p_clustered | alpha' | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        *cell_table_rows,
        "",
        "## Directional split (IS, RULE 12)",
        "",
        "| Strategy | Long N | Long ExpR | Short N | Short ExpR |",
        "|---|---:|---:|---:|---:|",
        *dir_table_rows,
        "",
        "## Pooled descriptive (informational; pooled_t_role = descriptive_only)",
        "",
        f"- pooled N_trades: **{pooled_desc['n_trades']}**",
        f"- pooled N_unique_trading_days: **{pooled_desc['n_unique_trading_days']}**",
        f"- pooled ExpR: **{_fmt(pooled_desc['expr'])}**",
        f"- pooled t_clustered: **{_fmt(pooled_desc['t_clustered'], 3)}**",
        f"- pooled p_clustered: **{_fmt(pooled_desc['p_clustered'], 5)}**",
        f"- flip_rate_pct (per-cell ExpR sign vs pooled sign): **{_fmt(flip_rate, 2)}**",
        f"- heterogeneity_ack: **{str(het_ack).lower()}** (required when flip_rate_pct >= 25)",
        f"- measured_pooled_power: **{_fmt(measured_power)}** (tier `{pooled_tier}`)",
        "",
        "## H3 robustness (K=3 re-rank on prior-PASS cells)",
        "",
        *h3_lines,
        "",
        "## Reconciliation vs chordia_audit_log.yaml prior t_stat",
        "",
        *recon_lines,
        "",
        "## Method notes",
        "",
        "- Canonical source only: `orb_outcomes` JOIN `daily_features` on `(trading_day, symbol, orb_minutes)`.",
        f"- Sacred holdout: `trading_day < {HOLDOUT_SACRED_FROM}` for IS, `>=` for descriptive OOS.",
        f"- Cohort lower bound: `WF_START_OVERRIDE['MNQ']={WF_START_OVERRIDE.get('MNQ')}` applied.",
        f"- Canonical filter delegation: `research.filter_utils.filter_signal(df, '{EXPECTED_FILTER_KEY}', '{EXPECTED_SESSION}')` "
        f"(definition='orb_mid', verified at runtime).",
        "- Realized-eod scratch policy: `pnl_r` NULL on `outcome='scratch'` rows coerced to 0.0 before any statistic.",
        "- Clustered SE: `statsmodels.regression.linear_model.OLS` intercept-only fit with "
        "`cov_type='cluster'`, `cov_kwds={'groups': trading_day}`.",
        "- Holm-Bonferroni ranking: cells sorted ascending on `p_clustered`, alpha'_i applied by rank index.",
        "- OOS power tier: per-cell and pooled via `research.oos_power.one_sample_power` (NCP = d * sqrt(n)); descriptive only (RULE 3.3).",
        "- No writes to `validated_setups`, `experimental_strategies`, `lane_allocation.json`, "
        "`chordia_audit_log.yaml`, `bot_state.json`, or `live_config.json`.",
        "",
        "## Harvey-Liu boundary statement (mandatory; pre-reg outputs_required_after_run)",
        "",
        "The Harvey-Liu Sharpe-haircut deflates IS Sharpe for multiple-testing inflation; it is "
        "NOT an OOS validation substitute. Deployment eligibility is decided by the downstream "
        "Stage B3 pre-reg, which applies BOTH (a) the Harvey-Liu haircut to IS Sharpe AND "
        "(b) allocator correlation gating. Both gates are orthogonal; both must clear independently.",
        "",
        "## Caveats",
        "",
        "- Pooled t is descriptive_only per BLOCKING-B compensating control "
        f"(`primary_schema.pooled_t_role_assert == {EXPECTED_POOLED_T_ROLE}`). It cannot rescue or "
        "override the per-cell Holm verdict.",
        "- `PASS_PARTIAL_HOLM` is not a verdict in this audit (K=4 all-or-nothing). 3-of-4 PASS "
        f"yields `{HOLM_VERDICT_FAIL}`.",
        "- OOS is descriptive at audit time (RULE 3.3 power-tier mandate); binary OOS gates are "
        "not applicable here.",
        "",
        "## Reproduction",
        "",
        "```",
        f"python research/vwap_mid_family_pooled_oos_v1.py --hypothesis-file {hypothesis.yaml_path.relative_to(ROOT).as_posix()}",
        "```",
        "",
    ]
    md_path.write_text(frontmatter + "\n" + "\n".join(body_lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage B2 family-pooled Holm-Bonferroni + clustered SE runner for "
            "MNQ US_DATA_1000 VWAP_MID_ALIGNED."
        )
    )
    parser.add_argument("--hypothesis-file", required=True, help="Path to pre-reg yaml.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print but write no artifacts (legal mode while execution_gate.allowed_now is false).",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    yaml_path = Path(args.hypothesis_file)
    if not yaml_path.is_absolute():
        yaml_path = ROOT / yaml_path
    yaml_path = yaml_path.resolve()
    if not yaml_path.exists():
        sys.stderr.write(f"REFUSE: pre-reg file not found: {yaml_path}\n")
        return 2

    hypothesis = _load_hypothesis(yaml_path)
    print(f"Pre-reg: {hypothesis.yaml_path.relative_to(ROOT).as_posix()}")
    print(f"Stem: {hypothesis.stem}")
    print(f"execution_gate.allowed_now: {hypothesis.allowed_now}")
    print(f"--dry-run: {args.dry_run}")

    # Execution gate (first leg). No DB read until this clears.
    if not hypothesis.allowed_now and not args.dry_run:
        sys.stderr.write(
            "REFUSE: execution_gate.allowed_now=False; pre-reg is in quarantine.\n"
            "        Promote out of `docs/audit/hypotheses/drafts/` via human review "
            "(set execution_gate.allowed_now=true, promoted_at, promoted_by) before "
            "running live. Re-run with --dry-run for sanity-check execution that "
            "writes no artifacts.\n"
        )
        return 2

    # All sentinel + canonical-delegation asserts have passed in _load_hypothesis.
    cells = list(hypothesis.cells)

    print("Connecting to gold.db read-only...")
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    audit_log_t_stats = _load_audit_log_t_stats(ROOT)

    cell_evals: list[dict[str, Any]] = []
    for cell in cells:
        ev = _evaluate_cell(con, cell)
        cell_evals.append(ev)

    # Reconciliation (halts before any kill clauses if a prior-PASS cell drifts >0.50).
    recon_blocks: list[dict[str, Any]] = []
    halt_reconciliation = False
    for cell, ev in zip(cells, cell_evals, strict=True):
        t_clustered = float(ev["is_metrics"].get("t_clustered", float("nan")))
        recon = _reconcile(cell, t_clustered, audit_log_t_stats)
        recon["strategy_id"] = cell.strategy_id
        recon_blocks.append(recon)
        if recon["halt"]:
            halt_reconciliation = True

    if halt_reconciliation:
        sys.stderr.write(
            "REFUSE: reconciliation halt — |delta_t| > 0.50 vs chordia_audit_log.yaml on at "
            "least one prior-PASS cell. Investigate upstream before writing artifacts.\n"
        )
        for recon in recon_blocks:
            if recon["halt"]:
                # Explicit `is not None` check — never use `or` here because
                # audit_log_t_stat == 0.0 is a legitimate (if degenerate)
                # anchor; `or` would silently swap in prior_t_stat instead.
                audit_log_t = recon.get("audit_log_t_stat")
                anchor = audit_log_t if audit_log_t is not None else recon["prior_t_stat"]
                sys.stderr.write(
                    f"        {recon['strategy_id']}: anchor "
                    f"{recon.get('audit_log_t_stat')}, this-run "
                    f"t={recon['delta_t'] + anchor:.3f}, "
                    f"delta={recon['delta_t']:+.3f}\n"
                )
        return 2

    # Holm assignment (ascending p_clustered rank -> locked alpha'_i thresholds).
    alphas = _holm_assign(cell_evals, hypothesis.holm_thresholds)

    # Per-cell kill clauses (gate A Chordia AND gate B Holm; both with cluster-skew floor).
    kills: list[dict[str, Any]] = []
    for cell, ev, alpha in zip(cells, cell_evals, alphas, strict=True):
        kill = _evaluate_cell_kills(ev, hypothesis.chordia_threshold_basis_t, alpha)
        kills.append(kill)

    family_verdict = _family_verdict(kills)

    pooled_desc = _pooled_descriptive(cell_evals)
    pooled_power = _pooled_oos_power(cell_evals, pooled_desc)

    h3_block = _h3_robustness(
        cell_evals,
        [k["cell_verdict_label"] for k in kills],
        cells,
    )

    # Clustering-inflation warning: any cell where t_clustered < t_naive - 0.5.
    clustering_inflation_warning = False
    for ev in cell_evals:
        tn = float(ev["is_metrics"].get("t_naive", float("nan")))
        tc = float(ev["is_metrics"].get("t_clustered", float("nan")))
        if math.isfinite(tn) and math.isfinite(tc) and (tn - tc) >= CLUSTERING_INFLATION_WARNING_DELTA:
            clustering_inflation_warning = True
            break

    # Per-cell summary printout (always, before any write).
    print("")
    print("Per-cell IS summary:")
    for cell, ev, kill, alpha in zip(cells, cell_evals, kills, alphas, strict=True):
        m = ev["is_metrics"]
        print(
            f"  {cell.strategy_id}: N={m['n_trades']} days={m['n_unique_trading_days']} "
            f"ExpR={_fmt(m['expr'])} t_naive={_fmt(m['t_naive'], 3)} "
            f"t_clustered={_fmt(m['t_clustered'], 3)} p_clustered={_fmt(m['p_clustered'], 5)} "
            f"alpha'={_fmt(alpha)} verdict={kill['cell_verdict_label']}"
        )
    print(f"\nFamily verdict: {family_verdict}")
    print(
        f"Pooled (descriptive): t_clustered={_fmt(pooled_desc['t_clustered'], 3)} "
        f"flip_rate_pct={_fmt(pooled_desc['flip_rate_pct'], 2)} "
        f"heterogeneity_ack={pooled_desc['heterogeneity_ack']}"
    )
    print(
        f"H3 isolation_pass: {h3_block.get('h3_isolation_pass')} "
        f"(applicable={h3_block.get('applicable')})"
    )
    print(f"Clustering inflation warning: {clustering_inflation_warning}")
    if math.isfinite(pooled_power.get("measured_pooled_power", float("nan"))):
        print(
            f"Pooled OOS power: {pooled_power['measured_pooled_power']:.4f} "
            f"(tier {pooled_power['pooled_power_tier']})"
        )

    # Artifact write decision (defensive gate re-check).
    artifact_paths = _resolve_artifact_paths(hypothesis)
    if args.dry_run or not hypothesis.allowed_now:
        for label, path in artifact_paths.items():
            print(f"WOULD WRITE: {path.relative_to(ROOT).as_posix()}  ({label})")
        if not hypothesis.allowed_now:
            print(
                "Execution gate closed; artifact writes refused. "
                "Re-run after pre-reg promotion (allowed_now=true) to emit the three artifacts."
            )
        return 0

    # Defensive re-assert at the artifact-write boundary.
    assert hypothesis.allowed_now is True, (
        "artifact-write boundary reached with execution_gate.allowed_now=False; "
        "BLOCKING-B sentinel violated"
    )

    _write_per_cell_csv(artifact_paths["per_cell_csv"], cells, cell_evals, kills, alphas)
    _write_by_year_csv(artifact_paths["by_year_csv"], cells, cell_evals)
    _write_result_md(
        artifact_paths["md"],
        hypothesis,
        cells,
        cell_evals,
        kills,
        alphas,
        family_verdict,
        pooled_desc,
        pooled_power,
        h3_block,
        recon_blocks,
        artifact_paths,
        clustering_inflation_warning,
    )

    for label, path in artifact_paths.items():
        print(f"WROTE: {path.relative_to(ROOT).as_posix()}  ({label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
