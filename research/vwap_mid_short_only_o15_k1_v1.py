"""Path A Stage 2 runner — MNQ US_DATA_1000 VWAP_MID_ALIGNED O15 RR1.5 SHORT-ONLY K=1.

Bounded single-cell single-direction K=1 executor for the locked draft pre-reg at
``docs/audit/hypotheses/drafts/2026-05-17-mnq-usdata1000-vwapmid-short-only-k1-v1.draft.yaml``.

Re-executes the SHORT-only subset of the RR1.5 O15 cell under clustered standard
errors at trading_day level, applies the Chordia 2018 no-theory strict t-hurdle
(t_clustered >= 3.79), and computes the four-gate deployment chain (Chordia /
Harvey-Liu IS-Sharpe haircut at K_effective=8 / cost-aware ExpR / pairwise
correlation vs current lane_allocation.json) as an INDEPENDENT report.

Audit verdict (PASS_K1_CHORDIA / FAIL_K1) is decided ON IS clustered-SE ALONE
per the pre-reg. The deployment_gate_chain is reported adjacent to but separate
from the audit verdict; a PASS_K1_CHORDIA cell can still fail deployment gates.

Route contract (per pre-reg ``not_done_by_this_pre_reg``):

- No writes to validated_setups, experimental_strategies, edge_families.
- No writes to lane_allocation.json, live_config.json, bot_state.json.
- No writes to docs/runtime/chordia_audit_log.yaml.
- No modification of prior PASS_CHORDIA chordia_audit_log.yaml entries.

Writes (gated on execution_gate.allowed_now=true; refuse otherwise):

- ``docs/audit/results/2026-05-17-mnq-usdata1000-vwapmid-short-only-k1-v1.md``
- ``docs/audit/results/2026-05-17-mnq-usdata1000-vwapmid-short-only-k1-v1.deployment.csv``

Even though these artifacts read only canonical layers, the act of writing them
under a quarantined (allowed_now=false) pre-reg produces an audit-evidence trail
that has not passed human review. Use ``--dry-run`` for review-mode inspection.
Canonical-layer writes (validated_setups / chordia_audit_log / lane_allocation)
are never performed by this runner regardless of mode.

Post-selection-inference contract (from pre-reg ``post_selection_disclosure``):

- Hypothesis generator: docs/audit/results/2026-05-17-mnq-usdata1000-vwapmid-family-pooled-oos-v1.md
- K_effective_informational = 8 (4 family cells x 2 directions inspected)
- n_trials declared = 1 (gate input for check_hypothesis_minbtl_compliance)
- Harvey-Liu haircut at K=8 is the pre-reg's deployment Gate 2 (SUPPRESSED until
  trading_app/ canonical helper lands — see HARVEY_LIU_HAIRCUT_SENTINEL below)
- Reconciliation expected: IS ExpR within 0.005 of 0.2572 (audit-integrity-failure > 0.030)
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
import statsmodels.api as sm
import yaml
from scipy import stats as scipy_stats

from pipeline.cost_model import COST_SPECS
from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal
from research.oos_power import one_sample_power, power_verdict
from trading_app.chordia import CHORDIA_T_WITHOUT_THEORY
from trading_app.config import ALL_FILTERS, VWAPBreakDirectionFilter, WF_START_OVERRIDE
from trading_app.eligibility.builder import parse_strategy_id
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.lane_correlation import RHO_REJECT_THRESHOLD, _pearson


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_INSTRUMENT = "MNQ"
EXPECTED_SESSION = "US_DATA_1000"
EXPECTED_FILTER_KEY = "VWAP_MID_ALIGNED"
EXPECTED_ORB_MINUTES = 15
EXPECTED_RR_TARGET = 1.5
EXPECTED_ENTRY_MODEL = "E2"
EXPECTED_CONFIRM_BARS = 1
EXPECTED_DIRECTION = "short"
EXPECTED_K_EFFECTIVE = 8
EXPECTED_N_TRIALS_DECLARED = 1
EXPECTED_SCRATCH_POLICY = "realized-eod"
EXPECTED_N_IS_REFERENCE = 354
EXPECTED_EXPR_IS_REFERENCE = 0.2572
RECONCILIATION_BLOCK_TOLERANCE_EXPR = 0.005
RECONCILIATION_HALT_TOLERANCE_EXPR = 0.030
CLUSTER_SKEW_FLOOR = 30

# Harvey-Liu Sharpe-haircut: NO CANONICAL HELPER in trading_app/ as of 2026-05-18.
# Inlining the multiplier (linear interpolation off Exhibit 6 Bonferroni K=5/K=10 anchors)
# would re-encode multiple-testing math that belongs in a shared canonical module —
# violation of .claude/rules/institutional-rigor.md § 4 (delegate to canonical sources;
# never re-encode). Gate 2 therefore emits the sentinel UNVERIFIED_NO_CANONICAL_HELPER
# until a helper lands in trading_app/ (e.g., trading_app.harvey_liu.haircut_ratio(k, alpha))
# with companion tests against the Exhibit 6 table. The audit verdict (H1 IS clustered SE)
# is unaffected — only the deployment_gate_chain Gate 2 column is suppressed.
HARVEY_LIU_HAIRCUT_SENTINEL = "UNVERIFIED_NO_CANONICAL_HELPER"
HARVEY_LIU_POST_HAIRCUT_SHARPE_FLOOR = 0.50  # pre-committed floor per pre-reg deployment_gate_chain.gates[1].threshold_floor


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
    direction: str
    expected_n_is: int
    expected_expr_is: float


@dataclass(frozen=True)
class Hypothesis:
    yaml_path: Path
    stem: str
    cell: CellSpec
    chordia_t: float
    k_effective: int
    allowed_now: bool


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        sys.stderr.write(f"REFUSE: {msg}\n")
        raise SystemExit(2)


def _strip_draft_stem(stem: str) -> str:
    if stem.endswith(".draft"):
        return stem[:-6]
    return stem


def _load_hypothesis(yaml_path: Path) -> Hypothesis:
    body = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    meta = body.get("metadata", {})
    scope = body.get("scope", {})
    schema = body.get("primary_schema", {})
    cell_block = schema.get("cell", {})
    data_policy = body.get("data_policy", {})
    feature = body.get("feature_definition", {})
    gate = body.get("execution_gate", {})
    psd = body.get("post_selection_disclosure", {})

    _assert(
        bool(psd.get("declared", False)),
        "post_selection_disclosure.declared must be true — this pre-reg is "
        "post-selection inference, not fresh K=1 discovery.",
    )
    _assert(
        int(psd.get("k_effective_informational", -1)) == EXPECTED_K_EFFECTIVE,
        f"post_selection_disclosure.k_effective_informational must equal {EXPECTED_K_EFFECTIVE}, "
        f"got {psd.get('k_effective_informational')!r}.",
    )
    _assert(
        int(schema.get("k_effective_informational", -1)) == EXPECTED_K_EFFECTIVE,
        f"primary_schema.k_effective_informational must equal {EXPECTED_K_EFFECTIVE}, "
        f"got {schema.get('k_effective_informational')!r}.",
    )
    _assert(
        int(meta.get("n_trials", -1)) == EXPECTED_N_TRIALS_DECLARED,
        f"metadata.n_trials must equal {EXPECTED_N_TRIALS_DECLARED}, got {meta.get('n_trials')!r}.",
    )
    _assert(
        int(body.get("total_expected_trials", -1)) == EXPECTED_N_TRIALS_DECLARED,
        f"total_expected_trials must equal {EXPECTED_N_TRIALS_DECLARED}, "
        f"got {body.get('total_expected_trials')!r}.",
    )
    _assert(
        str(scope.get("filter_type")) == EXPECTED_FILTER_KEY,
        f"scope.filter_type must equal {EXPECTED_FILTER_KEY!r}, got {scope.get('filter_type')!r}.",
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
        str(scope.get("direction")) == EXPECTED_DIRECTION,
        f"scope.direction must equal {EXPECTED_DIRECTION!r}, got {scope.get('direction')!r}.",
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

    # Theory-citation field-presence trap — must be absent from every hypothesis.
    for hyp in body.get("hypotheses", []) or []:
        _assert(
            "theory_citation" not in hyp,
            "hypotheses[].theory_citation MUST be absent (any truthy value flips loader "
            "has_theory=True and silently downgrades strict threshold 3.79 -> 3.00). "
            "See memory/feedback_chordia_theory_citation_field_presence_trap.md.",
        )

    # Canonical filter delegation self-check.
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

    sid = str(cell_block.get("strategy_id", ""))
    sid_parsed = parse_strategy_id(sid)
    _assert(
        sid_parsed["instrument"] == EXPECTED_INSTRUMENT
        and sid_parsed["orb_label"] == EXPECTED_SESSION
        and sid_parsed["filter_type"] == EXPECTED_FILTER_KEY
        and int(sid_parsed["orb_minutes"]) == EXPECTED_ORB_MINUTES
        and abs(float(sid_parsed["rr_target"]) - EXPECTED_RR_TARGET) < 1e-9
        and sid_parsed["entry_model"] == EXPECTED_ENTRY_MODEL
        and int(sid_parsed["confirm_bars"]) == EXPECTED_CONFIRM_BARS,
        f"strategy_id {sid!r} does not match expected dimensions.",
    )

    # Chordia threshold basis: ASCII '>= 3.79' only.
    basis = str(schema.get("chordia_threshold_basis", ""))
    _assert(
        "≥" not in basis,
        "chordia_threshold_basis contains Unicode '>=' (U+2265); ASCII '>=' required.",
    )
    import re

    m = re.search(r"t_clustered\s*>=\s*(\d+\.\d+)", basis)
    _assert(
        m is not None,
        f"chordia_threshold_basis must contain ASCII 't_clustered >= <float>', got: {basis!r}",
    )
    assert m is not None
    chordia_t = float(m.group(1))
    _assert(
        abs(chordia_t - CHORDIA_T_WITHOUT_THEORY) < 1e-6,
        f"chordia_threshold_basis declares t>={chordia_t} which does not match "
        f"CHORDIA_T_WITHOUT_THEORY={CHORDIA_T_WITHOUT_THEORY}.",
    )

    cell = CellSpec(
        cell_id=str(cell_block.get("id", sid)),
        strategy_id=sid,
        instrument=sid_parsed["instrument"],
        orb_label=sid_parsed["orb_label"],
        orb_minutes=int(sid_parsed["orb_minutes"]),
        entry_model=sid_parsed["entry_model"],
        confirm_bars=int(sid_parsed["confirm_bars"]),
        rr_target=float(sid_parsed["rr_target"]),
        filter_key=sid_parsed["filter_type"],
        direction=EXPECTED_DIRECTION,
        expected_n_is=int(cell_block.get("expected_N_IS", EXPECTED_N_IS_REFERENCE)),
        expected_expr_is=float(cell_block.get("expected_ExpR_IS", EXPECTED_EXPR_IS_REFERENCE)),
    )

    return Hypothesis(
        yaml_path=yaml_path,
        stem=yaml_path.stem,
        cell=cell,
        chordia_t=chordia_t,
        k_effective=EXPECTED_K_EFFECTIVE,
        allowed_now=bool(gate.get("allowed_now", False)),
    )


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


def _two_sided_p_from_t(t_value: float, df: float) -> float:
    if not math.isfinite(t_value) or df <= 0:
        return float("nan")
    return float(2.0 * scipy_stats.t.sf(abs(t_value), df))


def _clustered_t(pnl: np.ndarray, cluster: np.ndarray) -> tuple[float, float, float, float]:
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
    cov_df = getattr(fit, "df_resid_inference", None)
    if cov_df is not None and math.isfinite(float(cov_df)) and float(cov_df) > 0:
        df_used = float(cov_df)
    t_clustered = coef / se
    p_clustered = _two_sided_p_from_t(t_clustered, df_used)
    return coef, float(t_clustered), float(p_clustered), float(df_used)


def _restrict_to_direction(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    if df.empty:
        return df
    _assert(
        {"target_price", "stop_price"}.issubset(df.columns),
        "frame missing target_price/stop_price columns; cannot derive direction.",
    )
    directions = _direction_series(df)
    return df.loc[directions.eq(direction)].copy()


def _frame_metrics(df: pd.DataFrame) -> dict[str, Any]:
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
    }
    if df.empty:
        return out

    work = df.copy()
    scratch_mask = work["outcome"].astype(str).eq("scratch")
    null_mask = work["pnl_r"].isna()
    null_non_scratch = int((null_mask & ~scratch_mask).sum())
    work["pnl_eff"] = work["pnl_r"].fillna(0.0)
    assert work["pnl_eff"].notna().all(), "scratch coercion left residual NaN in pnl_eff"

    n = int(len(work))
    days = work["trading_day"]
    cluster_sizes = work.groupby("trading_day").size()
    n_clusters = int(cluster_sizes.size)
    mean_r = float(work["pnl_eff"].mean())
    std_r = float(work["pnl_eff"].std(ddof=1)) if n >= 2 else float("nan")
    sharpe = mean_r / std_r if n >= 2 and std_r > 0 else float("nan")

    # naive t-stat = mean / (std/sqrt(n)) = sharpe * sqrt(n).
    t_naive = sharpe * math.sqrt(n) if math.isfinite(sharpe) else float("nan")
    p_naive = _two_sided_p_from_t(t_naive, n - 1) if math.isfinite(t_naive) else float("nan")

    _, t_clust, p_clust, df_clust = _clustered_t(
        work["pnl_eff"].to_numpy(),
        days.to_numpy(),
    )

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
        }
    )
    _assert(
        null_non_scratch == 0,
        f"non-scratch NULL pnl_r rows in frame: {null_non_scratch} "
        "(must be 0 under realized-eod policy; "
        "see memory/feedback_scratch_pnl_null_class_bug.md)",
    )
    return out


def _yearly_rollup(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    work = df.copy()
    work["pnl_eff"] = work["pnl_r"].fillna(0.0)
    years = pd.to_datetime(work["trading_day"]).dt.year
    rows: list[dict[str, Any]] = []
    for yr_key, idx in work.groupby(years).groups.items():
        year_int = int(yr_key)  # type: ignore[arg-type]
        sub = work.loc[idx]
        if len(sub) < 10:
            continue
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
            }
        )
    return rows


def _evaluate_cell(con: duckdb.DuckDBPyConnection, cell: CellSpec) -> dict[str, Any]:
    is_df_all = _load_universe(con, cell, is_only=True)
    oos_df_all = _load_universe(con, cell, is_only=False)

    def _apply_filter(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        mask = filter_signal(df, cell.filter_key, cell.orb_label).astype(bool)
        return df.loc[np.asarray(mask, dtype=bool)].copy()

    is_fired_pooled = _apply_filter(is_df_all)
    oos_fired_pooled = _apply_filter(oos_df_all)

    is_fired = _restrict_to_direction(is_fired_pooled, cell.direction)
    oos_fired = _restrict_to_direction(oos_fired_pooled, cell.direction)

    is_metrics = _frame_metrics(is_fired)
    oos_metrics = _frame_metrics(oos_fired)
    by_year = _yearly_rollup(is_fired)

    oos_dir_match = None
    if (
        oos_metrics["n_trades"] > 0
        and math.isfinite(is_metrics["expr"])
        and math.isfinite(oos_metrics["expr"])
    ):
        oos_dir_match = bool((is_metrics["expr"] > 0) == (oos_metrics["expr"] > 0))

    oos_power = float("nan")
    oos_power_tier = "INSUFFICIENT_DATA"
    if (
        is_metrics["n_trades"] >= 2
        and oos_metrics["n_trades"] >= 2
        and math.isfinite(is_metrics["expr"])
        and math.isfinite(is_metrics["std_r"])
        and is_metrics["std_r"] > 0
    ):
        d_is = float(is_metrics["expr"]) / float(is_metrics["std_r"])
        try:
            oos_power = float(one_sample_power(d_is, int(oos_metrics["n_trades"]), alpha=0.05))
            oos_power_tier = power_verdict(oos_power)
        except ValueError:
            oos_power_tier = "INSUFFICIENT_DATA"

    return {
        "is_metrics": is_metrics,
        "oos_metrics": oos_metrics,
        "by_year": by_year,
        "oos_dir_match": oos_dir_match,
        "oos_power": oos_power,
        "oos_power_tier": oos_power_tier,
        "is_fired_frame": is_fired,
        "oos_fired_frame": oos_fired,
    }


def _evaluate_audit_verdict(cell_eval: dict[str, Any], chordia_t: float) -> dict[str, Any]:
    m = cell_eval["is_metrics"]
    n_clusters = int(m.get("n_unique_trading_days", 0) or 0)
    t_naive = float(m.get("t_naive", float("nan")))
    t_clust = float(m.get("t_clustered", float("nan")))
    expr_is = float(m.get("expr", float("nan")))

    cluster_skew_kill = n_clusters < CLUSTER_SKEW_FLOOR
    pass_chordia = (
        math.isfinite(t_clust) and t_clust >= chordia_t and not cluster_skew_kill
    )
    naive_flip_chordia = (
        math.isfinite(t_naive)
        and math.isfinite(t_clust)
        and t_naive >= chordia_t
        and t_clust < chordia_t
    )
    expr_sign_kill = math.isfinite(expr_is) and expr_is <= 0.0
    clustering_inflation_warning = (
        math.isfinite(t_naive)
        and math.isfinite(t_clust)
        and (t_naive - t_clust) >= 0.5
    )

    if cluster_skew_kill:
        verdict = "UNVERIFIED_CLUSTER_SKEW"
    elif not math.isfinite(t_clust):
        verdict = "UNVERIFIED_INSUFFICIENT_DATA"
    elif expr_sign_kill:
        verdict = "FAIL_EXPR_NONPOSITIVE"
    elif naive_flip_chordia:
        verdict = "FAIL_NAIVE_FLIP_CHORDIA"
    elif not pass_chordia:
        verdict = "FAIL_CHORDIA"
    else:
        verdict = "PASS_K1_CHORDIA"

    return {
        "pass_chordia": pass_chordia,
        "cluster_skew_kill": cluster_skew_kill,
        "naive_flip_chordia": naive_flip_chordia,
        "expr_sign_kill": expr_sign_kill,
        "clustering_inflation_warning": clustering_inflation_warning,
        "verdict": verdict,
    }


def _reconcile(cell: CellSpec, is_metrics: dict[str, Any]) -> dict[str, Any]:
    expr_is = float(is_metrics.get("expr", float("nan")))
    if not math.isfinite(expr_is):
        return {
            "tier": "UNVERIFIED_NO_DATA",
            "delta_expr": None,
            "halt": True,
            "expected_n_is": cell.expected_n_is,
            "expected_expr_is": cell.expected_expr_is,
        }
    delta = expr_is - cell.expected_expr_is
    abs_delta = abs(delta)
    halt = abs_delta > RECONCILIATION_HALT_TOLERANCE_EXPR
    if abs_delta <= RECONCILIATION_BLOCK_TOLERANCE_EXPR:
        tier = "PASS_SILENT"
    elif abs_delta <= RECONCILIATION_HALT_TOLERANCE_EXPR:
        tier = "PASS_WITH_BLOCK"
    else:
        tier = "HALT_DIVERGENCE"
    return {
        "tier": tier,
        "delta_expr": delta,
        "halt": halt,
        "expected_n_is": cell.expected_n_is,
        "expected_expr_is": cell.expected_expr_is,
    }


def _cost_aware_expr_gate(
    cell: CellSpec,
    is_metrics: dict[str, Any],
    con: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    """Gate 3: ExpR remains positive after subtracting per-trade friction expressed in R units.

    Friction in R = total_friction_usd / risk_dollars_per_R. risk_dollars_per_R is the average
    |entry_price - stop_price| * point_value across IS fired-short trades. Computed from
    orb_outcomes raw fields + COST_SPECS[instrument].
    """
    expr_is = float(is_metrics.get("expr", float("nan")))
    if not math.isfinite(expr_is):
        return {
            "value_pre_cost": float("nan"),
            "value_post_cost": float("nan"),
            "friction_in_r": float("nan"),
            "pass": False,
        }
    spec = COST_SPECS.get(cell.instrument)
    _assert(
        spec is not None,
        f"COST_SPECS missing entry for {cell.instrument!r}; cannot compute cost-aware gate.",
    )
    assert spec is not None
    total_friction_usd = float(spec.total_friction)
    point_value = float(spec.point_value)

    # Compute average risk_dollars across the IS fired-direction-restricted cohort via the same query path.
    start = WF_START_OVERRIDE.get(cell.instrument)
    start_clause = "AND o.trading_day >= ?" if start is not None else ""
    risk_sql = f"""
        WITH fired AS (
            SELECT o.entry_price, o.stop_price, o.target_price
            FROM orb_outcomes o
            WHERE o.symbol = ?
              AND o.orb_label = ?
              AND o.orb_minutes = ?
              AND o.entry_model = ?
              AND o.confirm_bars = ?
              AND o.rr_target = ?
              AND o.trading_day < ?
              AND o.outcome IS NOT NULL
              {start_clause}
        )
        SELECT AVG(ABS(entry_price - stop_price)) AS avg_risk_pts
        FROM fired
        WHERE target_price < stop_price  -- short direction (target below stop)
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
    row = con.execute(risk_sql, params).fetchone()
    avg_risk_pts = float(row[0]) if row and row[0] is not None else float("nan")
    if not math.isfinite(avg_risk_pts) or avg_risk_pts <= 0:
        return {
            "value_pre_cost": expr_is,
            "value_post_cost": float("nan"),
            "friction_in_r": float("nan"),
            "pass": False,
        }
    risk_dollars_per_R = avg_risk_pts * point_value
    friction_in_r = total_friction_usd / risk_dollars_per_R
    value_post_cost = expr_is - friction_in_r
    return {
        "value_pre_cost": expr_is,
        "value_post_cost": value_post_cost,
        "friction_in_r": friction_in_r,
        "pass": value_post_cost > 0.0,
    }


def _load_lane_allocation_json() -> dict[str, Any]:
    path = ROOT / "docs" / "runtime" / "lane_allocation.json"
    _assert(path.exists(), f"lane_allocation.json missing at {path}")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _correlation_gate(
    cell: CellSpec,
    cell_eval: dict[str, Any],
    con: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    """Gate 4: pairwise Pearson correlation of daily-pnl vs every lane in lane_allocation.json::lanes[]."""
    lane_alloc = _load_lane_allocation_json()
    lanes = lane_alloc.get("lanes", []) or []
    is_frame = cell_eval["is_fired_frame"]
    oos_frame = cell_eval["oos_fired_frame"]
    candidate_frames = [is_frame, oos_frame]
    candidate_concat = pd.concat(
        [f for f in candidate_frames if not f.empty], ignore_index=True
    )
    if candidate_concat.empty:
        return {
            "max_rho": float("nan"),
            "worst_lane": None,
            "n_lanes_compared": len(lanes),
            "threshold": RHO_REJECT_THRESHOLD,
            "pass": False,
            "pair_rows": [],
        }
    cand_daily = (
        candidate_concat.assign(pnl_eff=lambda df: df["pnl_r"].fillna(0.0))
        .groupby("trading_day")["pnl_eff"]
        .sum()
        .to_dict()
    )

    pair_rows: list[dict[str, Any]] = []
    max_rho = 0.0
    worst_lane: str | None = None
    for lane in lanes:
        strategy_id = lane.get("strategy_id")
        if not strategy_id:
            continue
        parsed = parse_strategy_id(str(strategy_id))
        start = WF_START_OVERRIDE.get(parsed["instrument"])
        start_clause = "AND o.trading_day >= ?" if start is not None else ""
        lane_sql = f"""
            SELECT o.trading_day, o.outcome, o.pnl_r, o.target_price, o.stop_price, d.*
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
              AND o.outcome IS NOT NULL
              {start_clause}
        """
        params: list[Any] = [
            parsed["instrument"],
            parsed["orb_label"],
            int(parsed["orb_minutes"]),
            parsed["entry_model"],
            int(parsed["confirm_bars"]),
            float(parsed["rr_target"]),
        ]
        if start is not None:
            params.append(start)
        lane_df = con.execute(lane_sql, params).df()
        if lane_df.empty:
            continue
        mask = filter_signal(lane_df, parsed["filter_type"], parsed["orb_label"]).astype(bool)
        lane_fired = lane_df.loc[np.asarray(mask, dtype=bool)].copy()
        if lane_fired.empty:
            continue
        lane_daily = (
            lane_fired.assign(pnl_eff=lambda df: df["pnl_r"].fillna(0.0))
            .groupby("trading_day")["pnl_eff"]
            .sum()
            .to_dict()
        )
        shared = sorted(set(cand_daily) & set(lane_daily))
        n_shared = len(shared)
        if n_shared < 5:
            rho = 0.0
        else:
            xs = [cand_daily[d] for d in shared]
            ys = [lane_daily[d] for d in shared]
            rho = float(_pearson(xs, ys))
        pair_rows.append(
            {
                "deployed_id": strategy_id,
                "shared_days": n_shared,
                "rho": rho,
            }
        )
        if abs(rho) > abs(max_rho):
            max_rho = rho
            worst_lane = strategy_id

    return {
        "max_rho": max_rho,
        "worst_lane": worst_lane,
        "n_lanes_compared": len(lanes),
        "threshold": RHO_REJECT_THRESHOLD,
        "pass": abs(max_rho) < RHO_REJECT_THRESHOLD,
        "pair_rows": pair_rows,
    }


def _deployment_gate_chain(
    hyp: Hypothesis,
    cell_eval: dict[str, Any],
    audit_verdict: dict[str, Any],
    con: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    """Build the four-gate deployment audit table.

    Returns a dict with one entry per gate (1..4): gate_name, measured_value, threshold,
    direction, pass_fail. Also returns chain_pass = AND over all four pass flags.
    """
    is_metrics = cell_eval["is_metrics"]

    # Gate 1: Chordia no-theory strict (verdict-aligned).
    t_clust = float(is_metrics.get("t_clustered", float("nan")))
    gate_1_pass = math.isfinite(t_clust) and t_clust >= hyp.chordia_t and audit_verdict["pass_chordia"]

    # Gate 2: Harvey-Liu IS Sharpe haircut at K_effective=8.
    # SUPPRESSED until trading_app.harvey_liu (or equivalent canonical helper) lands.
    # Re-encoding the Exhibit 6 multiplier inline here would violate
    # institutional-rigor.md § 4 (delegate to canonical sources, never re-encode).
    sharpe_raw = float(is_metrics.get("sharpe", float("nan")))
    gate_2_pass = False  # UNVERIFIED -> treated as non-passing for chain_pass purposes.

    # Gate 3: Cost-aware ExpR positive after COST_SPECS friction.
    cost_gate = _cost_aware_expr_gate(hyp.cell, is_metrics, con)
    gate_3_pass = bool(cost_gate["pass"])

    # Gate 4: Pairwise correlation vs current lane_allocation.json.
    corr_gate = _correlation_gate(hyp.cell, cell_eval, con)
    gate_4_pass = bool(corr_gate["pass"])

    rows = [
        {
            "gate_id": 1,
            "gate_name": "Chordia no-theory strict (t_clustered)",
            "measured_value": t_clust,
            "threshold": hyp.chordia_t,
            "direction": ">=",
            "pass_fail": "PASS" if gate_1_pass else "FAIL",
        },
        {
            "gate_id": 2,
            "gate_name": "Harvey-Liu IS Sharpe haircut at K_effective=8",
            "measured_value": HARVEY_LIU_HAIRCUT_SENTINEL,
            "threshold": HARVEY_LIU_POST_HAIRCUT_SHARPE_FLOOR,
            "direction": ">=",
            "pass_fail": HARVEY_LIU_HAIRCUT_SENTINEL,
            "extra": {
                "sharpe_raw": sharpe_raw,
                "haircut_ratio": HARVEY_LIU_HAIRCUT_SENTINEL,
                "k_effective": hyp.k_effective,
                "reason": (
                    "No canonical Harvey-Liu helper in trading_app/ as of 2026-05-18. "
                    "Inlining Exhibit 6 multipliers would violate institutional-rigor.md § 4. "
                    "Gate 2 will be re-enabled once trading_app.harvey_liu (or equivalent) lands "
                    "with companion tests against Harvey-Liu 2015 Exhibit 6."
                ),
            },
        },
        {
            "gate_id": 3,
            "gate_name": "Cost-aware ExpR positive (post-friction)",
            "measured_value": cost_gate["value_post_cost"],
            "threshold": 0.0,
            "direction": ">",
            "pass_fail": "PASS" if gate_3_pass else "FAIL",
            "extra": {
                "value_pre_cost": cost_gate["value_pre_cost"],
                "friction_in_r": cost_gate["friction_in_r"],
            },
        },
        {
            "gate_id": 4,
            "gate_name": "Pairwise correlation vs lane_allocation.json (max |rho|)",
            "measured_value": corr_gate["max_rho"],
            "threshold": corr_gate["threshold"],
            "direction": "<",
            "pass_fail": "PASS" if gate_4_pass else "FAIL",
            "extra": {
                "worst_lane": corr_gate["worst_lane"],
                "n_lanes_compared": corr_gate["n_lanes_compared"],
                "pair_rows": corr_gate["pair_rows"],
            },
        },
    ]
    chain_pass = gate_1_pass and gate_2_pass and gate_3_pass and gate_4_pass
    # Distinguish UNVERIFIED (no canonical helper) from FAIL (measured-fail).
    # If Gate 2 is the only non-PASS gate, the chain label is UNVERIFIED_NO_CANONICAL_HELPER
    # rather than FAIL — so a future reader does not mistake a missing-helper state for a
    # measured deployment-ineligible verdict.
    if not chain_pass and gate_1_pass and gate_3_pass and gate_4_pass and not gate_2_pass:
        chain_status = HARVEY_LIU_HAIRCUT_SENTINEL
    elif chain_pass:
        chain_status = "PASS"
    else:
        chain_status = "FAIL"
    return {
        "rows": rows,
        "chain_pass": bool(chain_pass),
        "chain_status": chain_status,
    }


def _fmt(value: Any, places: int = 4) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.{places}f}"
        return "nan"
    if value is None:
        return ""
    return str(value)


def _write_deployment_csv(path: Path, chain: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["gate_id", "gate_name", "measured_value", "threshold", "direction", "pass_fail"])
        for row in chain["rows"]:
            writer.writerow(
                [
                    row["gate_id"],
                    row["gate_name"],
                    _fmt(row["measured_value"], 4),
                    _fmt(row["threshold"], 4),
                    row["direction"],
                    row["pass_fail"],
                ]
            )


def _write_result_md(
    path: Path,
    hyp: Hypothesis,
    cell_eval: dict[str, Any],
    audit_verdict: dict[str, Any],
    reconciliation: dict[str, Any],
    chain: dict[str, Any],
) -> None:
    is_m = cell_eval["is_metrics"]
    oos_m = cell_eval["oos_metrics"]
    by_year = cell_eval["by_year"]

    deployment_csv_relative = Path(str(path).replace(".md", ".deployment.csv")).name

    frontmatter = (
        "---\n"
        "pooled_finding: false\n"
        "post_selection_inference: true\n"
        f"k_effective: {hyp.k_effective}\n"
        "verdict_test: chordia_strict_clustered_se_k1\n"
        f"measured_per_cell_power: {cell_eval['oos_power']:.4f}\n"
        f"clustering_inflation_warning: {str(audit_verdict['clustering_inflation_warning']).lower()}\n"
        f"deployment_gate_chain_pass: {str(chain['chain_pass']).lower()}\n"
        f"oos_power_tier: {cell_eval['oos_power_tier']}\n"
        "---\n\n"
    )

    title = "# MNQ US_DATA_1000 VWAP_MID_ALIGNED O15 RR1.5 SHORT-ONLY K=1 (post-selection)\n\n"

    deployment_table_md = "## Deployment gate chain (four orthogonal gates; all must clear for live deployment)\n\n"
    deployment_table_md += "| Gate | Name | Measured | Threshold | Direction | PASS/FAIL |\n"
    deployment_table_md += "|---:|---|---:|---:|:---:|:---:|\n"
    for row in chain["rows"]:
        deployment_table_md += (
            f"| {row['gate_id']} | {row['gate_name']} | "
            f"{_fmt(row['measured_value'], 4)} | "
            f"{_fmt(row['threshold'], 4)} | "
            f"{row['direction']} | {row['pass_fail']} |\n"
        )

    yearly_md = "\n## Year-by-year IS (adversarial split per RULE 12)\n\n"
    yearly_md += "| Year | N | N_days | ExpR | t_naive | t_clustered | p_clustered |\n"
    yearly_md += "|---:|---:|---:|---:|---:|---:|---:|\n"
    for row in by_year:
        yearly_md += (
            f"| {row['year']} | {row['n_trades']} | {row['n_unique_trading_days']} | "
            f"{_fmt(row['expr'])} | {_fmt(row['t_naive'], 3)} | "
            f"{_fmt(row['t_clustered'], 3)} | {_fmt(row['p_clustered'], 5)} |\n"
        )

    verdict_block = (
        "## Verdict\n\n"
        f"**MEASURED audit verdict:** `{audit_verdict['verdict']}`\n\n"
        f"**MEASURED threshold applied:** `{hyp.chordia_t:.2f}`\n\n"
        f"**MEASURED loader has_theory:** `false`\n\n"
        f"**MEASURED deployment_gate_chain_pass:** `{str(chain['chain_pass']).lower()}`\n\n"
        "The audit verdict is decided on IS clustered-SE alone (Chordia 2018 no-theory "
        "strict t-hurdle, ASCII >=). The deployment_gate_chain is reported adjacent but "
        "INDEPENDENT — a PASS_K1_CHORDIA cell that fails any deployment gate is not "
        "deployment-eligible until the failing gate is reconsidered.\n\n"
    )

    cell_table = (
        "## IS cell metrics (clustered SE at trading_day, SHORT-only)\n\n"
        "| Strategy | Direction | N | N_days | Cluster mean | Cluster max | ExpR | Sharpe | "
        "t_naive | t_clustered | p_naive | p_clustered |\n"
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        f"| `{hyp.cell.strategy_id}` | short | "
        f"{is_m['n_trades']} | {is_m['n_unique_trading_days']} | "
        f"{_fmt(is_m['cluster_size_mean'])} | {is_m['cluster_size_max']} | "
        f"{_fmt(is_m['expr'])} | {_fmt(is_m['sharpe'])} | "
        f"{_fmt(is_m['t_naive'], 3)} | {_fmt(is_m['t_clustered'], 3)} | "
        f"{_fmt(is_m['p_naive'], 5)} | {_fmt(is_m['p_clustered'], 5)} |\n\n"
    )

    oos_block = (
        "## OOS descriptive (NOT a verdict input per RULE 3.3)\n\n"
        f"- OOS N: **{oos_m['n_trades']}**\n"
        f"- OOS N_unique_trading_days: **{oos_m['n_unique_trading_days']}**\n"
        f"- OOS ExpR: **{_fmt(oos_m['expr'])}**\n"
        f"- OOS dir_match: **{cell_eval['oos_dir_match']}**\n"
        f"- measured_per_cell_power: **{cell_eval['oos_power']:.4f}** "
        f"(tier `{cell_eval['oos_power_tier']}`)\n"
        f"- Power-tier interpretation: dir_match outcome is "
        f"{'CAN_REFUTE_LEGITIMATE' if cell_eval['oos_power_tier'] == 'CAN_REFUTE' else 'INFORMATIONAL_ONLY (binary OOS gate not applicable; per RULE 3.3 + feedback_oos_power_floor.md + feedback_chordia_oos_park_vs_unverified_power_floor.md)'}.\n\n"
    )

    reconciliation_block = (
        "## Reconciliation vs Stage B3 generator (post-selection guard)\n\n"
        f"- Expected (Stage B3 directional split): N={reconciliation['expected_n_is']}, "
        f"ExpR={reconciliation['expected_expr_is']:.4f}\n"
        f"- Measured (this run): N={is_m['n_trades']}, ExpR={_fmt(is_m['expr'])}\n"
        f"- delta_ExpR: {_fmt(reconciliation['delta_expr']) if reconciliation['delta_expr'] is not None else 'n/a'}\n"
        f"- Reconciliation tier: **{reconciliation['tier']}**\n"
        f"- Tolerance bands: PASS_SILENT |delta_ExpR| <= {RECONCILIATION_BLOCK_TOLERANCE_EXPR}; "
        f"HALT_DIVERGENCE > {RECONCILIATION_HALT_TOLERANCE_EXPR}\n\n"
    )

    method_notes = (
        "## Method notes\n\n"
        "- Canonical source only: `orb_outcomes` JOIN `daily_features` on `(trading_day, symbol, orb_minutes)`.\n"
        "- Direction restriction extracted POST canonical filter delegation via "
        "`research.filter_utils.filter_signal(df, 'VWAP_MID_ALIGNED', 'US_DATA_1000')`. "
        "Short rows = `target_price < stop_price`.\n"
        f"- Sacred holdout: `trading_day < {HOLDOUT_SACRED_FROM}` for IS; `>=` for descriptive OOS.\n"
        f"- Cohort lower bound: `WF_START_OVERRIDE['MNQ']` applied.\n"
        "- Realized-eod scratch policy: `pnl_r` NULL on `outcome='scratch'` coerced to 0.0. "
        "Non-scratch NULL fails closed (feedback_scratch_pnl_null_class_bug.md).\n"
        "- Clustered SE: `statsmodels.OLS` intercept-only with `cov_type='cluster'`, "
        "`cov_kwds={'groups': trading_day}`.\n"
        "- OOS power: `research.oos_power.one_sample_power` (NCP = d * sqrt(n)); "
        "descriptive only per RULE 3.3.\n"
        "- Harvey-Liu haircut: deployment Gate 2 at K_effective=8 is currently "
        "**SUPPRESSED** (`UNVERIFIED_NO_CANONICAL_HELPER`). Inlining the Exhibit 6 "
        "multiplier in this runner would re-encode multiple-testing math in a research "
        "script (institutional-rigor.md § 4 violation). Gate 2 will be re-enabled once "
        "`trading_app.harvey_liu` (or equivalent canonical helper) lands with companion "
        "tests against Harvey-Liu 2015 Exhibit 6. IS-side deflation per "
        "`feedback_harvey_liu_haircut_not_oos_validation_substitute.md`; NOT an OOS "
        "substitute and NOT a correlation-gate substitute.\n"
        f"- Cost model: `pipeline.cost_model.COST_SPECS['{hyp.cell.instrument}']` queried at runtime.\n"
        f"- Correlation gate: `trading_app.lane_correlation.RHO_REJECT_THRESHOLD` queried at runtime "
        f"({RHO_REJECT_THRESHOLD:.2f}); pairwise vs every lane in `docs/runtime/lane_allocation.json::lanes[]`.\n"
        "- No writes to `validated_setups`, `experimental_strategies`, `lane_allocation.json`, "
        "`chordia_audit_log.yaml`, `bot_state.json`, or `live_config.json`.\n\n"
    )

    psd_block = (
        "## Post-selection inference disclosure\n\n"
        "This K=1 audit was NOT pre-registered before the data was inspected. The cell+direction "
        "was nominated by inspection of the Stage B3 family-pooled directional split table "
        "(`docs/audit/results/2026-05-17-mnq-usdata1000-vwapmid-family-pooled-oos-v1.md` "
        "Section 'Directional split (IS, RULE 12)'). 4 family cells x 2 directions = 8 "
        "cell-direction subsets were inspected before this cell was selected. The K_effective "
        "= 8 declaration honors Lopez de Prado false-strategy Eq.1 prior-trial Sharpe inflation. "
        "n_trials = 1 (declared gate input for `check_hypothesis_minbtl_compliance`) and "
        "K_effective_informational = 8 are intentionally distinct accountings.\n\n"
    )

    harvey_liu_boundary = (
        "## Harvey-Liu boundary statement\n\n"
        "The Harvey-Liu Sharpe-haircut applied in Gate 2 deflates IS Sharpe for multiple-testing "
        "inflation (Harvey & Liu 2015, resources/backtesting_dukepeople_liu.pdf Eq.5 p.14). It is "
        "an IS-side correction. It is NOT an OOS validation substitute "
        "(feedback_harvey_liu_haircut_not_oos_validation_substitute.md) and it is NOT an allocator "
        "correlation-gate substitute (feedback_harvey_liu_haircut_not_correlation_gate_substitute.md). "
        "Gate 2 and Gate 4 are orthogonal; both must clear independently for deployment eligibility.\n\n"
    )

    caveats = (
        "## Caveats\n\n"
        "- This is a POST-SELECTION confirmatory K=1, NOT a fresh K=1 discovery. Any downstream "
        "deflated-Sharpe / haircut math must use K_effective=8, not K=1.\n"
        f"- OOS measured power = {cell_eval['oos_power']:.4f} (tier "
        f"`{cell_eval['oos_power_tier']}`). Binary OOS gates are not applicable at audit-time per "
        "RULE 3.3 unless the tier is `CAN_REFUTE`.\n"
        "- Live deployment requires PASS on ALL FOUR gates above AND explicit human review per "
        "the Stage 4 of the surrounding plan. The audit verdict alone is necessary but insufficient.\n"
        "- The audit is read-only on canonical layers and runtime files.\n\n"
    )

    reproduction = (
        "## Reproduction\n\n"
        "```\n"
        f".venv/Scripts/python.exe research/vwap_mid_short_only_o15_k1_v1.py "
        f"--hypothesis-file {hyp.yaml_path.relative_to(ROOT).as_posix()}\n"
        "```\n\n"
        f"Deployment CSV companion: `{deployment_csv_relative}`\n"
    )

    body = (
        frontmatter
        + title
        + f"**Pre-reg:** `{hyp.yaml_path.relative_to(ROOT).as_posix()}`  \n"
        + f"**Deployment-gate CSV:** `{deployment_csv_relative}`  \n"
        + f"**Canonical DB:** `{GOLD_DB_PATH}`  \n"
        + f"**Holdout boundary (Mode A):** `trading_day >= {HOLDOUT_SACRED_FROM}`  \n"
        + f"**Cohort lower bound:** `WF_START_OVERRIDE['{EXPECTED_INSTRUMENT}']`  \n"
        + f"**K_effective_informational:** `{hyp.k_effective}` (post-selection)\n\n"
        + verdict_block
        + cell_table
        + deployment_table_md
        + yearly_md
        + oos_block
        + reconciliation_block
        + psd_block
        + harvey_liu_boundary
        + method_notes
        + caveats
        + reproduction
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hypothesis-file",
        required=True,
        help="Path to the locked draft pre-reg yaml (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print summary to stdout; do not write artifact files.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    hyp_path = Path(args.hypothesis_file)
    if not hyp_path.is_absolute():
        hyp_path = (ROOT / hyp_path).resolve()
    _assert(hyp_path.exists(), f"hypothesis file not found: {hyp_path}")

    hyp = _load_hypothesis(hyp_path)

    print(f"[run] hypothesis: {hyp.yaml_path.name}")
    print(f"[run] cell: {hyp.cell.strategy_id} | direction={hyp.cell.direction}")
    print(f"[run] chordia threshold: {hyp.chordia_t:.2f}")
    print(f"[run] K_effective: {hyp.k_effective}")
    print(f"[run] execution_gate.allowed_now: {hyp.allowed_now}")

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        cell_eval = _evaluate_cell(con, hyp.cell)
        is_m = cell_eval["is_metrics"]
        print(
            f"[run] IS: N={is_m['n_trades']} N_days={is_m['n_unique_trading_days']} "
            f"ExpR={is_m['expr']:.4f} Sharpe={is_m['sharpe']:.4f} "
            f"t_naive={is_m['t_naive']:.3f} t_clustered={is_m['t_clustered']:.3f}"
        )

        audit_verdict = _evaluate_audit_verdict(cell_eval, hyp.chordia_t)
        print(f"[run] audit verdict: {audit_verdict['verdict']}")
        if audit_verdict.get("clustering_inflation_warning"):
            print("[run] WARNING: clustering inflation (t_naive - t_clustered >= 0.5)")

        reconciliation = _reconcile(hyp.cell, is_m)
        print(
            f"[run] reconciliation: tier={reconciliation['tier']} "
            f"delta_ExpR={reconciliation['delta_expr']}"
        )
        if reconciliation["halt"]:
            sys.stderr.write(
                f"HALT: reconciliation divergence > {RECONCILIATION_HALT_TOLERANCE_EXPR} ExpR — "
                "investigate upstream before writing artifacts.\n"
            )
            return 3

        chain = _deployment_gate_chain(hyp, cell_eval, audit_verdict, con)
        for row in chain["rows"]:
            print(
                f"[gate-{row['gate_id']}] {row['gate_name']}: "
                f"measured={_fmt(row['measured_value'], 4)} "
                f"threshold={_fmt(row['threshold'], 4)} "
                f"direction={row['direction']} "
                f"-> {row['pass_fail']}"
            )
        print(f"[run] deployment_gate_chain_pass: {chain['chain_pass']}")

        if args.dry_run:
            print("[run] --dry-run set; not writing artifacts")
            return 0

        # Pre-reg execution_gate.allowed_now MUST be true before any artifact write.
        # Quarantined drafts (allowed_now=false) are reviewer-only; producing
        # result MD / deployment CSV under a false gate creates an audit-evidence
        # trail that did not pass human review. Re-execute with --dry-run to inspect
        # numbers, or promote the pre-reg out of drafts/ first.
        if not hyp.allowed_now:
            sys.stderr.write(
                "REFUSE: pre-reg execution_gate.allowed_now=false; artifacts blocked. "
                "Use --dry-run for review, or promote the pre-reg to "
                "docs/audit/hypotheses/ after human review.\n"
            )
            return 4

        stem = _strip_draft_stem(hyp.stem)
        base = ROOT / "docs" / "audit" / "results" / stem
        md_path = base.with_suffix(".md")
        csv_path = Path(str(base) + ".deployment.csv")

        _write_result_md(md_path, hyp, cell_eval, audit_verdict, reconciliation, chain)
        _write_deployment_csv(csv_path, chain)
        print(f"[run] wrote: {md_path.relative_to(ROOT)}")
        print(f"[run] wrote: {csv_path.relative_to(ROOT)}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
