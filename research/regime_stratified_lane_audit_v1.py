# scratch-policy: realized-eod
"""Regime-stratified era-stability audit on the 4 currently-deployed MNQ lanes.

Stage 4 of the MNQ era-stability audit. Implements the locked pre-reg at
``docs/audit/hypotheses/drafts/2026-05-17-mnq-deployed-lanes-regime-stratified-audit-v1.draft.yaml``.

Behavioural contract (per stage file
``docs/runtime/stages/regime-stratified-lane-audit-runner.md``):

- Reads the 4 deployed MNQ ``strategy_id`` values from
  ``docs/runtime/lane_allocation.json`` at runtime — never hardcoded
  (``feedback_allocator_orb_minutes_hardcode_2026_04_30.md`` class lesson).
- H1 = ``scipy.stats.chi2_contingency`` on per-eligible-session
  (regime x fired/not_fired) 4x2 table + logistic GLM robustness check.
- H2 = ``scipy.stats.f_oneway`` on per-trade ``pnl_r_effective`` grouped by
  regime + Tukey-HSD post-hoc when omnibus rejects.
- R0 (pre-2020 micro-launch) and R6 (sacred holdout) hard-asserted out of
  hypothesis-test input sets at runtime via ``HoldoutContaminationError`` —
  documentation-only exclusion rots silently
  (``feedback_e2_lookahead_drift_check_landed.md`` class lesson).
- K=8 sensitivity table emitted alongside K=2 primary verdict (Option 3).
- ``realized-eod`` scratch policy: ``pnl_r_effective`` populated;
  ``null_pnl_non_scratch`` count emitted (must be 0).
- Verdict feeds a FUTURE authorized step. This runner writes only the
  result MD + row-level CSV; zero production-code edits, zero allocator
  mutation, zero ``validated_setups`` write.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from research.filter_utils import filter_signal
from trading_app.config import WF_START_OVERRIDE
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM


ROOT = Path(__file__).resolve().parents[1]
LANE_ALLOCATION_PATH = ROOT / "docs" / "runtime" / "lane_allocation.json"
RESULTS_DIR = ROOT / "docs" / "audit" / "results"

MIN_N_PER_CELL = 30
K_FAMILY = 2
K_LANE_INFORMATIONAL = 8

FORWARD_MONITORING_LABEL = (
    "NOT SELECTION EVIDENCE per pre_registered_criteria.md Amendment 2.7 "
    "sacred holdout"
)


class HoldoutContaminationError(RuntimeError):
    """R0 or R6 leaked into a hypothesis test input set.

    Documentation-only exclusion of forbidden regimes is insufficient — it
    rots silently when a future maintainer mutates the input set. This
    error is the runtime hard-assert that enforces the prereg's
    ``input_regimes`` exclusion list.
    """


# ============================================================================
# Domain types
# ============================================================================


@dataclass(frozen=True)
class RegimeBucket:
    id: str
    start: date
    end: date
    role: str
    label: str

    def contains(self, trading_day: date) -> bool:
        return self.start <= trading_day <= self.end


@dataclass(frozen=True)
class Lane:
    strategy_id: str
    instrument: str
    session: str
    orb_minutes: int
    entry_model: str
    confirm_bars: int
    rr_target: float
    filter_type: str


@dataclass(frozen=True)
class AuditSpec:
    prereg_path: Path
    buckets: list[RegimeBucket]
    lanes: list[Lane]
    result_md: Path
    result_csv: Path
    is_buckets: list[RegimeBucket] = field(default_factory=list)
    r6_bucket: RegimeBucket | None = None


# ============================================================================
# Phase 1: load spec
# ============================================================================


def _parse_iso_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _load_audit_spec(prereg_path: Path) -> AuditSpec:
    body = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))

    regime_defs = body.get("regime_definitions", {})
    raw_buckets = regime_defs.get("buckets", [])
    if not raw_buckets:
        raise SystemExit("prereg missing regime_definitions.buckets")

    buckets: list[RegimeBucket] = []
    for raw in raw_buckets:
        end_raw = raw["end"]
        # R6 has end="cutoff (current date)" — substitute today's date.
        if isinstance(end_raw, str) and not end_raw[:4].isdigit():
            end_dt = date.today()
        else:
            end_dt = _parse_iso_date(end_raw)
        buckets.append(
            RegimeBucket(
                id=str(raw["id"]),
                start=_parse_iso_date(raw["start"]),
                end=end_dt,
                role=str(raw["role"]),
                label=str(raw["label"]),
            )
        )

    deployed = body.get("scope", {}).get("deployed_lanes", [])
    if not deployed:
        raise SystemExit("prereg missing scope.deployed_lanes")
    expected_strategy_ids = {str(d["strategy_id"]) for d in deployed}

    lanes = _load_deployed_mnq_lanes(LANE_ALLOCATION_PATH, expected_strategy_ids)

    is_buckets = [b for b in buckets if b.role == "IS_TEST_INPUT" or b.role == "CURRENT_REGIME_DECISION_INPUT"]
    r6 = next((b for b in buckets if b.role == "FORWARD_MONITOR_ONLY_SACRED_HOLDOUT"), None)

    stem = prereg_path.stem
    if stem.endswith(".draft"):
        stem = stem[: -len(".draft")]
    result_md = RESULTS_DIR / f"{stem}.md"
    result_csv = RESULTS_DIR / f"{stem}.csv"

    return AuditSpec(
        prereg_path=prereg_path,
        buckets=buckets,
        lanes=lanes,
        result_md=result_md,
        result_csv=result_csv,
        is_buckets=is_buckets,
        r6_bucket=r6,
    )


def _load_deployed_mnq_lanes(
    lane_alloc_path: Path,
    expected_strategy_ids: set[str],
) -> list[Lane]:
    """Read the 4 currently-deployed MNQ lanes from lane_allocation.json.

    Fails fast on count drift or strategy_id drift vs the prereg. Per
    ``feedback_allocator_orb_minutes_hardcode_2026_04_30.md`` the runner
    MUST source live deployment state from canonical JSON, not hardcode it.
    """
    if not lane_alloc_path.exists():
        raise SystemExit(f"lane_allocation.json missing at {lane_alloc_path}")
    payload = json.loads(lane_alloc_path.read_text(encoding="utf-8"))
    raw_lanes = payload.get("lanes", [])
    mnq_lanes = [lane for lane in raw_lanes if lane.get("instrument") == "MNQ"]
    if len(mnq_lanes) != len(expected_strategy_ids):
        raise SystemExit(
            f"MNQ deployed-lane count drift: prereg expected "
            f"{len(expected_strategy_ids)}, lane_allocation.json has "
            f"{len(mnq_lanes)}. Expected ids: {sorted(expected_strategy_ids)}. "
            f"Got: {sorted(lane.get('strategy_id', '?') for lane in mnq_lanes)}."
        )

    seen_ids = {str(lane["strategy_id"]) for lane in mnq_lanes}
    missing = expected_strategy_ids - seen_ids
    extra = seen_ids - expected_strategy_ids
    if missing or extra:
        raise SystemExit(
            f"MNQ deployed-lane strategy_id drift vs prereg. "
            f"Missing in lane_allocation.json: {sorted(missing)}. "
            f"Extra (not in prereg): {sorted(extra)}."
        )

    lanes: list[Lane] = []
    for raw in mnq_lanes:
        sid = str(raw["strategy_id"])
        # Parse entry_model / confirm_bars / rr_target from strategy_id since
        # lane_allocation.json doesn't carry them as explicit fields.
        from trading_app.eligibility.builder import parse_strategy_id

        dims = parse_strategy_id(sid)
        lanes.append(
            Lane(
                strategy_id=sid,
                instrument=dims["instrument"],
                session=dims["orb_label"],
                orb_minutes=int(raw.get("orb_minutes", dims["orb_minutes"])),
                entry_model=dims["entry_model"],
                confirm_bars=dims["confirm_bars"],
                rr_target=dims["rr_target"],
                filter_type=str(raw.get("filter_type", dims["filter_type"])),
            )
        )
    return lanes


# ============================================================================
# Phase 2: canonical-layer data pulls
# ============================================================================


def _load_eligible_sessions(
    con: duckdb.DuckDBPyConnection,
    lane: Lane,
    cohort_low: date,
    holdout_high: date,
) -> pd.DataFrame:
    """Return one row per UNIQUE (trading_day, orb_label) session-event.

    H1 DENOMINATOR GUARDRAIL (user-mandated): ``orb_outcomes`` has multiple
    rows per (trading_day, orb_label) — one per
    (entry_model x confirm_bars x rr_target x direction). For H1 fire-rate
    the denominator MUST be unique session-events, NOT joined-row count.
    Runtime assertion below catches any future regression that inflates
    the denominator. This catches both the daily-features-joins.md
    class-bug (sqrt(3)=1.73x t-inflation) and the multi-row-per-session
    class-bug specific to H1.
    """
    sql = """
        SELECT DISTINCT
            o.trading_day,
            o.orb_label,
            o.symbol,
            o.orb_minutes,
            d.*
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.trading_day = d.trading_day
           AND o.symbol = d.symbol
           AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.trading_day >= ?
          AND o.trading_day < ?
    """
    params = [
        lane.instrument,
        lane.session,
        lane.orb_minutes,
        cohort_low,
        holdout_high,
    ]
    df = con.execute(sql, params).df()

    if not df.empty:
        dup_count = int(df.duplicated(subset=["trading_day", "orb_label"]).sum())
        if dup_count != 0:
            raise RuntimeError(
                f"H1 denominator inflated: {dup_count} duplicate session-events "
                f"for {lane.strategy_id}. Expected unique (trading_day, orb_label) "
                f"rows after SELECT DISTINCT."
            )
    return df


def _compute_fire_mask(eligible_df: pd.DataFrame, lane: Lane) -> np.ndarray:
    if eligible_df.empty:
        return np.zeros(0, dtype=int)
    return filter_signal(eligible_df, lane.filter_type, lane.session)


def _load_fired_trades(
    con: duckdb.DuckDBPyConnection,
    lane: Lane,
    cohort_low: date,
    holdout_high: date | None,
) -> pd.DataFrame:
    """Return per-trade rows for the lane's exact tuple, with pnl_r_effective.

    ``pnl_r_effective`` realized-eod policy:
      - ``outcome IN ('win', 'loss')``     -> use ``pnl_r``
      - ``outcome == 'scratch'``           -> use ``pnl_r`` (post Stage 5
        outcome_builder fills realized-EOD MTM). NULL only in pathological
        empty-post-entry-bars case (per outcome_builder.py:619) — coerced
        to 0.0 as Criterion 13 'drop' fallback.
      - any other outcome with NULL pnl_r  -> execution-integrity bug.
        Counted via ``null_pnl_non_scratch`` and surfaced in the result MD.
    """
    end_clause = "AND o.trading_day < ?" if holdout_high is not None else ""
    sql = f"""
        SELECT
            o.trading_day,
            o.orb_label,
            o.symbol,
            o.orb_minutes,
            o.entry_model,
            o.confirm_bars,
            o.rr_target,
            o.outcome,
            o.pnl_r,
            o.entry_price,
            o.target_price,
            o.stop_price
        FROM orb_outcomes o
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.entry_model = ?
          AND o.confirm_bars = ?
          AND o.rr_target = ?
          AND o.trading_day >= ?
          {end_clause}
    """
    params: list[Any] = [
        lane.instrument,
        lane.session,
        lane.orb_minutes,
        lane.entry_model,
        lane.confirm_bars,
        lane.rr_target,
        cohort_low,
    ]
    if holdout_high is not None:
        params.append(holdout_high)
    df = con.execute(sql, params).df()

    if df.empty:
        df["pnl_r_effective"] = pd.Series(dtype=float)
        df["scratch"] = pd.Series(dtype=bool)
        df["null_pnl_non_scratch"] = pd.Series(dtype=bool)
        return df

    return _apply_realized_eod_policy(df)


def _apply_realized_eod_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Populate pnl_r_effective + flags per Criterion 13 realized-eod policy."""
    out = df.copy()
    outcome = out["outcome"].astype(str)
    scratch = outcome.eq("scratch")
    pnl_null = out["pnl_r"].isna()

    out["scratch"] = scratch
    out["null_pnl_non_scratch"] = pnl_null & ~scratch

    # realized-eod policy: scratch with realized-EOD pnl_r (post Stage 5)
    # already carries the MTM value. Pathological NULL is coerced to 0.0.
    # Non-scratch NULL pnl_r is an execution-integrity violation; it is
    # counted but NOT silently dropped — surfaced via null_pnl_non_scratch.
    out["pnl_r_effective"] = out["pnl_r"].fillna(0.0)
    return out


# ============================================================================
# Phase 3: regime stratification
# ============================================================================


def _assign_regime(trading_day: Any, buckets: list[RegimeBucket]) -> str | None:
    # Coerce any (datetime, Timestamp, np.datetime64, date, str) input to date.
    # date subclasses (datetime, Timestamp) pass the isinstance check; we
    # then call `.date()` only if it's NOT a plain date. Plain date instances
    # have no `.hour` attribute; this distinguishes them from datetime/Timestamp.
    raw: Any = trading_day
    if isinstance(raw, date) and not hasattr(raw, "hour"):
        td: date = raw
    else:
        td = pd.Timestamp(raw).date()
    for bucket in buckets:
        if bucket.contains(td):
            return bucket.id
    return None


def _stratify(
    df: pd.DataFrame,
    buckets: list[RegimeBucket],
    keep_regime_ids: list[str],
) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {rid: df.iloc[0:0].copy() for rid in keep_regime_ids}
    regime_series = df["trading_day"].apply(lambda td: _assign_regime(td, buckets))
    out: dict[str, pd.DataFrame] = {}
    for rid in keep_regime_ids:
        out[rid] = df.loc[regime_series.eq(rid)].copy()
    return out


def _drop_underpowered_cells(
    per_regime: dict[str, pd.DataFrame],
    min_n: int = MIN_N_PER_CELL,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Drop regimes whose row-count < min_n. NEVER widens sibling buckets."""
    kept: dict[str, pd.DataFrame] = {}
    dropped: list[str] = []
    for rid, frame in per_regime.items():
        if len(frame) < min_n:
            dropped.append(rid)
        else:
            kept[rid] = frame
    return kept, dropped


# ============================================================================
# Phase 4: hypothesis tests
# ============================================================================


def _assert_no_holdout_contamination(hyp_id: str, input_regimes: list[str]) -> None:
    forbidden = {"R0", "R6"}
    bad = forbidden & set(input_regimes)
    if bad:
        raise HoldoutContaminationError(
            f"{hyp_id} test input contains forbidden regime(s): {sorted(bad)}. "
            f"R0 = INFORMATIONAL_EXCLUDED; R6 = SACRED_HOLDOUT_FORWARD_MONITOR. "
            f"Authority: prereg regime_definitions + "
            f"trading_app.holdout_policy.HOLDOUT_SACRED_FROM."
        )


def _h1_chi_square(
    per_regime_eligible: dict[str, pd.DataFrame],
    fire_masks_per_regime: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Chi-square 4x2 on (regime x fired/not_fired). Small-cell fallback to
    Fisher's exact (pairwise 2x2 with Bonferroni) or chi-square with Yates.
    """
    regimes = sorted(per_regime_eligible.keys())
    table: list[list[int]] = []
    fire_counts: dict[str, tuple[int, int]] = {}
    for rid in regimes:
        fired_mask = fire_masks_per_regime[rid]
        n_total = int(len(fired_mask))
        n_fired = int(fired_mask.sum())
        n_not_fired = n_total - n_fired
        table.append([n_fired, n_not_fired])
        fire_counts[rid] = (n_fired, n_total)

    arr = np.array(table, dtype=int)

    if arr.shape[0] < 2 or arr.sum() == 0:
        return {
            "regimes": regimes,
            "table": arr.tolist(),
            "fire_counts": fire_counts,
            "chi2": float("nan"),
            "raw_p": float("nan"),
            "dof": 0,
            "expected": [[]],
            "small_cell": True,
            "small_cell_method": "INSUFFICIENT_REGIMES",
            "fallback_p": float("nan"),
            "pooled_fire_rate": float("nan"),
        }

    chi2_res = stats.chi2_contingency(arr, correction=False)
    chi2 = float(chi2_res[0])  # type: ignore[arg-type]
    raw_p = float(chi2_res[1])  # type: ignore[arg-type]
    dof = int(chi2_res[2])  # type: ignore[arg-type]
    expected = np.asarray(chi2_res[3])
    small_cell = bool((expected < 5).any())
    fallback_p = float("nan")
    fallback_method = "NONE"
    if small_cell:
        # Yates-corrected chi-square (works for any 2D table) is the
        # simplest universal fallback; Fisher's exact is reserved for the
        # 2x2 pairwise post-hoc step.
        try:
            fb_res = stats.chi2_contingency(arr, correction=True)
            fallback_p = float(fb_res[1])  # type: ignore[arg-type]
            fallback_method = "CHI2_YATES"
        except Exception:
            fallback_p = float("nan")
            fallback_method = "CHI2_YATES_FAILED"

    pooled_fired = int(arr[:, 0].sum())
    pooled_total = int(arr.sum())
    pooled_fire_rate = pooled_fired / pooled_total if pooled_total > 0 else float("nan")

    return {
        "regimes": regimes,
        "table": arr.tolist(),
        "fire_counts": fire_counts,
        "chi2": chi2,
        "raw_p": raw_p,
        "dof": dof,
        "expected": expected.tolist(),
        "small_cell": small_cell,
        "small_cell_method": fallback_method,
        "fallback_p": fallback_p,
        "pooled_fire_rate": pooled_fire_rate,
    }


def _h1_logistic_glm(
    fire_masks_per_regime: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Confirmatory logistic GLM on fired ~ C(regime). Best-effort: if
    statsmodels is unavailable, return a placeholder dict.
    """
    try:
        import statsmodels.formula.api as smf  # type: ignore
    except ImportError:
        return {"status": "STATSMODELS_UNAVAILABLE", "coefficients": {}}

    rows: list[tuple[int, str]] = []
    for rid, mask in fire_masks_per_regime.items():
        for val in mask:
            rows.append((int(val), rid))
    if not rows:
        return {"status": "EMPTY", "coefficients": {}}
    df = pd.DataFrame(rows, columns=["fired", "regime"])
    if df["fired"].nunique() < 2 or df["regime"].nunique() < 2:
        return {"status": "DEGENERATE", "coefficients": {}}

    try:
        model = smf.logit("fired ~ C(regime)", data=df).fit(disp=False)
        coefs = {str(k): float(v) for k, v in model.params.items()}
        pvals = {str(k): float(v) for k, v in model.pvalues.items()}
        return {
            "status": "FIT",
            "coefficients": coefs,
            "pvalues": pvals,
            "llf": float(model.llf),
            "aic": float(model.aic),
        }
    except Exception as exc:
        return {"status": f"FIT_FAILED: {type(exc).__name__}", "coefficients": {}}


def _check_anova_assumptions(per_regime_pnl: dict[str, np.ndarray]) -> dict[str, Any]:
    """Levene equal-variance + Shapiro-Wilk normality. Per user mandate
    sensitivity diagnostics fire when EITHER (a) Levene p<0.05 OR (b) any
    regime's Shapiro p<0.05 with that regime's N<500.
    """
    groups = [g for g in per_regime_pnl.values() if len(g) >= 2]
    if len(groups) < 2:
        return {"levene_p": float("nan"), "shapiro_flag": False, "trigger": False}
    try:
        _, levene_p = stats.levene(*groups)
    except Exception:
        levene_p = float("nan")

    shapiro_flag = False
    shapiro_per_regime: dict[str, float] = {}
    for rid, arr in per_regime_pnl.items():
        if 3 <= len(arr) < 5000:
            try:
                _, sp = stats.shapiro(arr)
                shapiro_per_regime[rid] = float(sp)
                if sp < 0.05 and len(arr) < 500:
                    shapiro_flag = True
            except Exception:
                shapiro_per_regime[rid] = float("nan")

    levene_flag = bool(math.isfinite(levene_p) and levene_p < 0.05)
    trigger = bool(levene_flag or shapiro_flag)
    return {
        "levene_p": float(levene_p),
        "shapiro_per_regime": shapiro_per_regime,
        "shapiro_flag": shapiro_flag,
        "levene_flag": levene_flag,
        "trigger": trigger,
    }


def _h2_anova(per_regime_pnl: dict[str, np.ndarray]) -> dict[str, Any]:
    """One-way ANOVA on per-trade pnl_r_effective grouped by regime.

    PRIMARY (prereg-locked): scipy.stats.f_oneway. Tukey-HSD post-hoc when
    omnibus p<0.01.

    SENSITIVITY (diagnostic only — does NOT replace primary, never flips
    verdict): Welch's ANOVA (alexandergovern) + Kruskal-Wallis emitted
    when assumption checks trigger.
    """
    groups_with_n = [(rid, arr) for rid, arr in per_regime_pnl.items() if len(arr) >= 2]
    if len(groups_with_n) < 2:
        return {
            "f_stat": float("nan"),
            "raw_p": float("nan"),
            "regimes": list(per_regime_pnl.keys()),
            "n_per_regime": {rid: int(len(arr)) for rid, arr in per_regime_pnl.items()},
            "mean_per_regime": {
                rid: float(np.mean(arr)) if len(arr) > 0 else float("nan")
                for rid, arr in per_regime_pnl.items()
            },
            "std_per_regime": {
                rid: float(np.std(arr, ddof=1)) if len(arr) >= 2 else float("nan")
                for rid, arr in per_regime_pnl.items()
            },
            "tukey": None,
            "assumptions": _check_anova_assumptions(per_regime_pnl),
            "sensitivity": None,
        }

    groups = [arr for _, arr in groups_with_n]
    try:
        f_stat, raw_p = stats.f_oneway(*groups)
    except Exception:
        f_stat, raw_p = float("nan"), float("nan")

    tukey = None
    if math.isfinite(raw_p) and raw_p < 0.01:
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd  # type: ignore

            values = np.concatenate(groups)
            labels = np.concatenate([np.array([rid] * len(arr)) for rid, arr in groups_with_n])
            tk = pairwise_tukeyhsd(values, labels, alpha=0.05)
            tukey = {
                "summary": str(tk.summary()),
                "pairs": list(tk.groupsunique),
            }
        except Exception as exc:
            tukey = {"status": f"TUKEY_FAILED: {type(exc).__name__}"}

    assumptions = _check_anova_assumptions(per_regime_pnl)
    sensitivity: dict[str, Any] | None = None
    if assumptions.get("trigger"):
        sensitivity = {}
        try:
            welch = stats.alexandergovern(*groups)
            sensitivity["welch_p"] = float(welch.pvalue)
            sensitivity["welch_statistic"] = float(welch.statistic)
        except Exception:
            sensitivity["welch_p"] = float("nan")
        try:
            kw_stat, kw_p = stats.kruskal(*groups)
            sensitivity["kruskal_p"] = float(kw_p)
            sensitivity["kruskal_stat"] = float(kw_stat)
        except Exception:
            sensitivity["kruskal_p"] = float("nan")
        sensitivity["note"] = (
            "DIAGNOSTIC ONLY - primary verdict is prereg ANOVA per Criterion 1 "
            "(pre-registered hypothesis). Sensitivity tests do NOT flip verdict; "
            "they flag prereg misspecification for the NEXT prereg cycle."
        )

    return {
        "f_stat": float(f_stat) if math.isfinite(f_stat) else float("nan"),
        "raw_p": float(raw_p) if math.isfinite(raw_p) else float("nan"),
        "regimes": [rid for rid, _ in groups_with_n],
        "n_per_regime": {rid: int(len(arr)) for rid, arr in per_regime_pnl.items()},
        "mean_per_regime": {
            rid: float(np.mean(arr)) if len(arr) > 0 else float("nan")
            for rid, arr in per_regime_pnl.items()
        },
        "std_per_regime": {
            rid: float(np.std(arr, ddof=1)) if len(arr) >= 2 else float("nan")
            for rid, arr in per_regime_pnl.items()
        },
        "tukey": tukey,
        "assumptions": assumptions,
        "sensitivity": sensitivity,
    }


# ============================================================================
# Phase 5: K=8 sensitivity + per-lane verdicts
# ============================================================================


def _bonferroni_k8(raw_p_by_cell: dict[tuple[str, str], float]) -> dict[tuple[str, str], dict[str, Any]]:
    """4 lanes x 2 hypotheses = 8 cells; raw_p x K=8 capped at 1.0.

    Tier: PASS if adj_p >= 0.05 ; WATCH if 0.01 <= adj_p < 0.05 ;
    FAIL if adj_p < 0.01.
    """
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, raw_p in raw_p_by_cell.items():
        if not math.isfinite(raw_p):
            adj = float("nan")
            tier = "NA"
        else:
            adj = min(raw_p * K_LANE_INFORMATIONAL, 1.0)
            if adj < 0.01:
                tier = "FAIL"
            elif adj < 0.05:
                tier = "WATCH"
            else:
                tier = "PASS"
        out[key] = {"raw_p": raw_p, "adj_p_k8": adj, "tier": tier}
    return out


def _per_lane_verdict(
    h1_raw_p: float,
    h2_raw_p: float,
    r5_expr: float,
    r5_n: int,
) -> tuple[str, str]:
    """K=2 family-level decision_rule per prereg.

    continue_if: H1 omnibus PASS and H2 omnibus PASS and R5 ExpR > 0
    park_if: any R5 ExpR <= 0 (N>=30) OR H1/H2 omnibus p<0.01
    kill_if: R5 ExpR < -0.10 (N>=30)  -- per-lane kill
    """
    if math.isfinite(r5_expr) and r5_n >= MIN_N_PER_CELL and r5_expr < -0.10:
        return "KILL", (
            f"R5 ExpR={r5_expr:.4f} < -0.10 with N={r5_n} >= {MIN_N_PER_CELL} "
            f"-> per-lane kill clause (per-lane kill escalation rule applies; "
            f"see K=8 sensitivity)."
        )

    omnibus_reject = (math.isfinite(h1_raw_p) and h1_raw_p < 0.01) or (
        math.isfinite(h2_raw_p) and h2_raw_p < 0.01
    )
    r5_nonpositive = math.isfinite(r5_expr) and r5_n >= MIN_N_PER_CELL and r5_expr <= 0.0

    if omnibus_reject or r5_nonpositive:
        bits = []
        if math.isfinite(h1_raw_p) and h1_raw_p < 0.01:
            bits.append(f"H1 omnibus p={h1_raw_p:.4f} < 0.01")
        if math.isfinite(h2_raw_p) and h2_raw_p < 0.01:
            bits.append(f"H2 omnibus p={h2_raw_p:.4f} < 0.01")
        if r5_nonpositive:
            bits.append(f"R5 ExpR={r5_expr:.4f} <= 0 at N={r5_n}")
        return "PARK", "; ".join(bits) if bits else "Decision rule park_if branch."

    return "CONTINUE", (
        f"H1 p={h1_raw_p:.4f}, H2 p={h2_raw_p:.4f}, R5 ExpR={r5_expr:.4f} (N={r5_n})."
    )


# ============================================================================
# Phase 6: emit MD + CSV
# ============================================================================


def _fmt(value: Any, places: int = 4) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.{places}f}"
    return "nan"


def _write_csv(spec: AuditSpec, rows: list[dict[str, Any]]) -> None:
    spec.result_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        spec.result_csv.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with spec.result_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(spec: AuditSpec, results: dict[str, Any]) -> None:
    spec.result_md.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("---")
    lines.append("pooled_finding: false")
    lines.append(f"prereg: {spec.prereg_path.relative_to(ROOT).as_posix()}")
    lines.append("---")
    lines.append("")
    lines.append("# MNQ Deployed-Lane Regime-Stratified Era-Stability Audit (v1)")
    lines.append("")
    lines.append(
        "Per-lane regime-stability audit on the 4 currently-deployed MNQ lanes "
        "from `docs/runtime/lane_allocation.json`. H1 = chi-square + logistic "
        "GLM on per-eligible-session fire-rate across R2/R3/R4/R5. H2 = one-way "
        "ANOVA on per-trade `pnl_r_effective` across R2/R3/R4/R5. R0 and R6 are "
        "runtime-asserted excluded from hypothesis-test inputs. R6 is reported "
        "in a separate forward-monitoring section."
    )
    lines.append("")

    # K=8 escalation pre-amble (mandatory per Option 3)
    lines.append("## K=8 multiplicity-escalation pre-amble")
    lines.append("")
    lines.append(
        "Primary verdict is at K=2 (family of H1 omnibus + H2 omnibus). The "
        "K=8 sensitivity table below decomposes per-lane x per-hypothesis "
        "(4 lanes x 2 hypotheses = 8 cells). Per the prereg's "
        "multiplicity_escalation_rule and per-lane_kill_escalation_rule: if "
        "ANY downstream consumer treats per-lane p-values as selection "
        "evidence (allocator-pause, capital reallocation, manual deploy/pause "
        "decision), the K=8 verdict MUST be examined and the more conservative "
        "verdict (K=8 if it differs from K=2) WINS. This pre-amble forecloses "
        "the 'we only used K=2 in the headline, so K=8 doesn't apply' loophole."
    )
    lines.append("")

    # Execution-integrity gate row
    null_nonscratch_total = int(results.get("null_pnl_non_scratch_total", 0))
    integrity_status = "PASS" if null_nonscratch_total == 0 else "FAIL"
    lines.append("## Execution-integrity gate")
    lines.append("")
    lines.append(
        f"- `null_pnl_non_scratch: {null_nonscratch_total}` "
        f"({integrity_status} - must be 0; if >0, audit FAILS "
        f"execution-integrity gate per Criterion 13)."
    )
    lines.append("- `scratch_policy: realized-eod` (Criterion 13 BINDING).")
    lines.append("")

    # dropped_regimes_per_lane log block
    lines.append("## dropped_regimes_per_lane (per-regime power floor)")
    lines.append("")
    drop_log = results.get("dropped_regimes_per_lane", {})
    if drop_log:
        for sid, dropped in drop_log.items():
            lines.append(f"- `{sid}`: H1_dropped={dropped['h1']}, H2_dropped={dropped['h2']}")
    else:
        lines.append("- (no regimes dropped)")
    lines.append("")

    # Per-lane verdict list
    lines.append("## Per-lane verdicts (K=2 primary)")
    lines.append("")
    lines.append("| lane | H1_p | H2_p | R5_ExpR | R5_N | verdict | rationale |")
    lines.append("|---|---|---|---|---|---|---|")
    for sid, lr in results["per_lane"].items():
        lines.append(
            f"| `{sid}` | {_fmt(lr['h1']['raw_p'], 5)} | "
            f"{_fmt(lr['h2']['raw_p'], 5)} | {_fmt(lr['r5_expr'])} | "
            f"{lr['r5_n']} | **{lr['verdict']}** | {lr['rationale']} |"
        )
    lines.append("")

    # H1 contingency tables + small-cell fallback + logistic GLM
    lines.append("## H1: fire-rate stability (chi-square 4x2 per lane)")
    lines.append("")
    for sid, lr in results["per_lane"].items():
        h1 = lr["h1"]
        lines.append(f"### `{sid}` -- H1")
        lines.append("")
        lines.append(f"Regimes used: {h1['regimes']}")
        lines.append(f"Contingency table (regime x [fired, not_fired]): {h1['table']}")
        lines.append(f"Expected cells: {h1.get('expected')}")
        lines.append(
            f"chi2 = {_fmt(h1['chi2'], 4)}, dof = {h1.get('dof')}, "
            f"raw_p = {_fmt(h1['raw_p'], 5)}"
        )
        if h1.get("small_cell"):
            lines.append(
                f"Small-cell fallback fired ({h1.get('small_cell_method')}); "
                f"fallback_p = {_fmt(h1.get('fallback_p'), 5)}"
            )
        glm = lr.get("h1_glm", {})
        lines.append(
            f"Logistic GLM status: {glm.get('status', 'UNKNOWN')}; "
            f"coefficients: {glm.get('coefficients', {})}"
        )
        lines.append("")

    # H2 ANOVA + Tukey-HSD + sensitivity diagnostics
    lines.append("## H2: ExpR stability (one-way ANOVA per lane)")
    lines.append("")
    for sid, lr in results["per_lane"].items():
        h2 = lr["h2"]
        lines.append(f"### `{sid}` -- H2")
        lines.append("")
        lines.append(f"Regimes used: {h2['regimes']}")
        lines.append(f"N per regime: {h2['n_per_regime']}")
        lines.append(f"Mean ExpR per regime: { {k: round(v, 4) for k, v in h2['mean_per_regime'].items()} }")
        lines.append(
            f"F = {_fmt(h2['f_stat'], 4)}, raw_p = {_fmt(h2['raw_p'], 5)}"
        )
        if h2.get("tukey"):
            lines.append("Tukey-HSD post-hoc (omnibus rejected):")
            lines.append("```")
            lines.append(str(h2["tukey"].get("summary", h2["tukey"])))
            lines.append("```")
        sens = h2.get("sensitivity")
        if sens:
            lines.append("")
            lines.append("#### H2 Sensitivity Diagnostics")
            lines.append("")
            lines.append(sens["note"])
            lines.append(
                f"- Welch's ANOVA p = {_fmt(sens.get('welch_p'), 5)}"
            )
            lines.append(
                f"- Kruskal-Wallis p = {_fmt(sens.get('kruskal_p'), 5)}"
            )
            lines.append(
                f"- Levene p = {_fmt(h2['assumptions'].get('levene_p'), 5)}, "
                f"shapiro_flag = {h2['assumptions'].get('shapiro_flag')}"
            )
        lines.append("")

    # K=8 sensitivity table
    lines.append("## K=8 sensitivity table (per-lane x per-hypothesis Bonferroni)")
    lines.append("")
    lines.append("| lane | hypothesis | raw_p | bonferroni_k8_p | k8_verdict_tier |")
    lines.append("|---|---|---|---|---|")
    for (sid, hyp), cell in results["k8_sensitivity"].items():
        lines.append(
            f"| `{sid}` | {hyp} | {_fmt(cell['raw_p'], 5)} | "
            f"{_fmt(cell['adj_p_k8'], 5)} | {cell['tier']} |"
        )
    lines.append("")

    # Forward-monitoring R6 section -- EXACT label required
    lines.append("## Forward monitoring (R6 sacred holdout)")
    lines.append("")
    lines.append(FORWARD_MONITORING_LABEL)
    lines.append("")
    lines.append("| lane | R6_N | R6_ExpR | R6_Sharpe | R6_fire_rate |")
    lines.append("|---|---|---|---|---|")
    fwd = results.get("forward_monitoring", {})
    for sid in results["per_lane"].keys():
        f = fwd.get(sid, {})
        lines.append(
            f"| `{sid}` | {f.get('n', 0)} | {_fmt(f.get('expr'))} | "
            f"{_fmt(f.get('sharpe'))} | {_fmt(f.get('fire_rate'))} |"
        )
    lines.append("")
    lines.append(
        "Forbidden uses of this section: ranking lanes; killing or pausing "
        "a lane; rescuing a lane that failed IS R5 kill clause; re-running "
        "the audit with different thresholds to rescue an R6 outcome."
    )
    lines.append("")

    # R0 informational section
    lines.append("## R0 (pre-2020 micro-launch) -- INFORMATIONAL_EXCLUDED")
    lines.append("")
    lines.append(
        "R0 is excluded from both H1 and H2 test inputs at runtime via "
        "`HoldoutContaminationError`. The window is reported here for "
        "completeness; it does NOT feed any verdict."
    )
    r0 = results.get("r0_informational", {})
    lines.append("")
    lines.append("| lane | R0_N_trades | R0_ExpR |")
    lines.append("|---|---|---|")
    for sid in results["per_lane"].keys():
        r = r0.get(sid, {})
        lines.append(f"| `{sid}` | {r.get('n', 0)} | {_fmt(r.get('expr'))} |")
    lines.append("")

    spec.result_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================
# Phase 7: orchestrator
# ============================================================================


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hypothesis-file",
        required=True,
        help="Path to the locked prereg YAML.",
    )
    return p


def _summarize_for_lane(
    lane: Lane,
    spec: AuditSpec,
    con: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    cohort_low = WF_START_OVERRIDE.get(lane.instrument)
    if cohort_low is None:
        raise SystemExit(f"WF_START_OVERRIDE missing for {lane.instrument!r}")

    is_regime_ids = [b.id for b in spec.is_buckets]

    eligible_full = _load_eligible_sessions(
        con, lane, cohort_low, HOLDOUT_SACRED_FROM
    )
    fire_mask_full = _compute_fire_mask(eligible_full, lane)
    eligible_full = eligible_full.assign(_fired=fire_mask_full)

    # Stratify eligible sessions by regime (IS only).
    eligible_per_regime = _stratify(
        eligible_full[["trading_day", "orb_label", "_fired"]],
        spec.is_buckets,
        is_regime_ids,
    )
    fire_masks_per_regime = {
        rid: df["_fired"].to_numpy(dtype=int) for rid, df in eligible_per_regime.items()
    }

    # Drop underpowered H1 cells (eligible_sessions < 30 in that regime).
    kept_h1, dropped_h1 = _drop_underpowered_cells(
        eligible_per_regime, MIN_N_PER_CELL
    )
    fire_masks_h1 = {rid: fire_masks_per_regime[rid] for rid in kept_h1}

    _assert_no_holdout_contamination("H1", list(kept_h1.keys()))
    h1_result = _h1_chi_square(kept_h1, fire_masks_h1)
    h1_glm = _h1_logistic_glm(fire_masks_h1)

    # H2: per-trade pnl_r_effective stratified.
    trades_full = _load_fired_trades(con, lane, cohort_low, HOLDOUT_SACRED_FROM)
    trades_per_regime = _stratify(trades_full, spec.is_buckets, is_regime_ids)
    kept_h2, dropped_h2 = _drop_underpowered_cells(
        trades_per_regime, MIN_N_PER_CELL
    )
    _assert_no_holdout_contamination("H2", list(kept_h2.keys()))
    pnl_per_regime = {
        rid: df["pnl_r_effective"].to_numpy(dtype=float) for rid, df in kept_h2.items()
    }
    h2_result = _h2_anova(pnl_per_regime)

    # R5 stats for kill clause.
    r5_trades = trades_per_regime.get("R5", pd.DataFrame())
    r5_pnl = r5_trades["pnl_r_effective"].to_numpy(dtype=float) if not r5_trades.empty else np.array([])
    r5_n = int(len(r5_pnl))
    r5_expr = float(np.mean(r5_pnl)) if r5_n > 0 else float("nan")

    verdict, rationale = _per_lane_verdict(
        h1_result["raw_p"], h2_result["raw_p"], r5_expr, r5_n
    )

    # Forward-monitor R6 (sacred holdout, descriptive only).
    fwd_eligible = _load_eligible_sessions(
        con, lane, HOLDOUT_SACRED_FROM, date(9999, 1, 1)
    )
    fwd_fire = _compute_fire_mask(fwd_eligible, lane)
    fwd_fire_rate = float(fwd_fire.mean()) if len(fwd_fire) > 0 else float("nan")
    fwd_trades = _load_fired_trades(con, lane, HOLDOUT_SACRED_FROM, None)
    fwd_n = int(len(fwd_trades))
    if fwd_n > 0:
        fwd_expr = float(fwd_trades["pnl_r_effective"].mean())
        fwd_std = float(fwd_trades["pnl_r_effective"].std(ddof=1)) if fwd_n >= 2 else float("nan")
        fwd_sharpe = fwd_expr / fwd_std if math.isfinite(fwd_std) and fwd_std > 0 else float("nan")
    else:
        fwd_expr = float("nan")
        fwd_sharpe = float("nan")

    # R0 informational (pre-2020 micro-launch).
    r0_bucket = next((b for b in spec.buckets if b.id == "R0"), None)
    if r0_bucket is not None:
        r0_trades_sql = """
            SELECT outcome, pnl_r
            FROM orb_outcomes
            WHERE symbol = ?
              AND orb_label = ?
              AND orb_minutes = ?
              AND entry_model = ?
              AND confirm_bars = ?
              AND rr_target = ?
              AND trading_day >= ?
              AND trading_day <= ?
        """
        r0_df = con.execute(
            r0_trades_sql,
            [
                lane.instrument,
                lane.session,
                lane.orb_minutes,
                lane.entry_model,
                lane.confirm_bars,
                lane.rr_target,
                r0_bucket.start,
                r0_bucket.end,
            ],
        ).df()
        if not r0_df.empty:
            r0_df = _apply_realized_eod_policy(r0_df)
            r0_n = int(len(r0_df))
            r0_expr = float(r0_df["pnl_r_effective"].mean())
        else:
            r0_n = 0
            r0_expr = float("nan")
    else:
        r0_n = 0
        r0_expr = float("nan")

    null_nonscratch_lane = int(trades_full["null_pnl_non_scratch"].sum()) if not trades_full.empty else 0

    return {
        "h1": h1_result,
        "h1_glm": h1_glm,
        "h2": h2_result,
        "verdict": verdict,
        "rationale": rationale,
        "r5_expr": r5_expr,
        "r5_n": r5_n,
        "dropped_h1": dropped_h1,
        "dropped_h2": dropped_h2,
        "forward_monitoring": {
            "n": fwd_n,
            "expr": fwd_expr,
            "sharpe": fwd_sharpe,
            "fire_rate": fwd_fire_rate,
        },
        "r0_informational": {"n": r0_n, "expr": r0_expr},
        "null_pnl_non_scratch": null_nonscratch_lane,
        "trades_per_regime": {rid: int(len(df)) for rid, df in trades_per_regime.items()},
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    prereg_path = Path(args.hypothesis_file).resolve()
    if not prereg_path.exists():
        sys.stderr.write(f"prereg file not found: {prereg_path}\n")
        return 2

    spec = _load_audit_spec(prereg_path)

    per_lane: dict[str, dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = []
    drop_log: dict[str, dict[str, list[str]]] = {}
    fwd: dict[str, dict[str, Any]] = {}
    r0_info: dict[str, dict[str, Any]] = {}
    null_total = 0
    raw_p_by_cell: dict[tuple[str, str], float] = {}

    from pipeline.paths import GOLD_DB_PATH

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        for lane in spec.lanes:
            lane_res = _summarize_for_lane(lane, spec, con)
            per_lane[lane.strategy_id] = lane_res
            drop_log[lane.strategy_id] = {
                "h1": lane_res["dropped_h1"],
                "h2": lane_res["dropped_h2"],
            }
            fwd[lane.strategy_id] = lane_res["forward_monitoring"]
            r0_info[lane.strategy_id] = lane_res["r0_informational"]
            null_total += int(lane_res["null_pnl_non_scratch"])
            raw_p_by_cell[(lane.strategy_id, "H1")] = lane_res["h1"]["raw_p"]
            raw_p_by_cell[(lane.strategy_id, "H2")] = lane_res["h2"]["raw_p"]

            # Per-lane per-regime CSV rows
            fire_counts = lane_res["h1"].get("fire_counts", {})
            for rid in [b.id for b in spec.is_buckets]:
                n_trades = lane_res["trades_per_regime"].get(rid, 0)
                n_fired, n_eligible = fire_counts.get(rid, (0, 0))
                fire_rate = (n_fired / n_eligible) if n_eligible > 0 else float("nan")
                csv_rows.append(
                    {
                        "strategy_id": lane.strategy_id,
                        "regime": rid,
                        "n_eligible_sessions": int(n_eligible),
                        "n_fired": int(n_fired),
                        "fire_rate": fire_rate,
                        "n_trades": int(n_trades),
                        "h1_dropped": rid in lane_res["dropped_h1"],
                        "h2_dropped": rid in lane_res["dropped_h2"],
                    }
                )
    finally:
        con.close()

    k8 = _bonferroni_k8(raw_p_by_cell)

    results = {
        "per_lane": per_lane,
        "k8_sensitivity": k8,
        "forward_monitoring": fwd,
        "r0_informational": r0_info,
        "null_pnl_non_scratch_total": null_total,
        "dropped_regimes_per_lane": drop_log,
    }

    _write_csv(spec, csv_rows)
    _write_markdown(spec, results)
    sys.stdout.write(
        f"OK: wrote {spec.result_md.relative_to(ROOT)} and "
        f"{spec.result_csv.relative_to(ROOT)}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
