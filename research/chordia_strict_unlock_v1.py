"""Bounded exact-lane Chordia strict-unlock runner.

See ``docs/specs/fast_lane_state_graph.md`` for the canonical fast-lane chain definition.

Executes a single preregistered exact-lane replay from
``docs/audit/hypotheses/*.yaml`` and emits:

- ``docs/audit/results/<stem>.md``
- ``docs/audit/results/<stem>.csv``

Route contract:
- bounded conditional-role runner only
- no writes to ``experimental_strategies``
- no writes to ``validated_setups``
- no writes to ``docs/runtime/chordia_audit_log.yaml``

This runner is intentionally narrow. It assumes the prereg defines one exact
lane at top-level ``scope`` and one hypothesis with ``expected_trial_count: 1``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal
from scripts.research.fast_lane_structural_hash import compute_structural_hash
from scripts.research.fast_lane_trial_ledger import (
    LedgerEntry,
    append_trial_ledger_entry,
    compute_trial_id,
    read_ledger,
)
from trading_app.chordia import (
    CHORDIA_T_WITHOUT_THEORY,
    chordia_threshold,
    compute_chordia_t,
)
from trading_app.config import ALL_FILTERS, WF_START_OVERRIDE, CrossAssetATRFilter
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.hypothesis_loader import check_mode_a_consistency, load_hypothesis_metadata
from trading_app.strategy_discovery import parse_stop_multiplier

OOS_DESCRIPTIVE_MIN_N = 30
DEFAULT_STOP_MULTIPLIER = 1.0

# FAST_LANE v5.1 gate thresholds — sourced from
# docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml § screen + § outcomes.
# Change these only by amending the template and bumping template_version.
FAST_LANE_V5_1_PROMOTE_T = 3.0       # screen.promote_threshold (2.5) + needs_more_band (0.5)
FAST_LANE_V5_1_NEEDS_MORE_T = 2.5    # screen.promote_threshold
FAST_LANE_V5_1_EXPR_MIN = 0.0        # screen.expr_min
FAST_LANE_V5_1_N_MIN = 50            # screen.n_IS_on_min
FAST_LANE_V5_1_FIRE_LO = 0.05        # screen.fire_rate_gate
FAST_LANE_V5_1_FIRE_HI = 0.95

SUPPORTED_TEMPLATE_VERSIONS = frozenset({"fast_lane_v5.1"})


def _normalize_writable_path(path: Path) -> Path:
    text = str(path)
    if text.startswith("/mnt/c/Users/"):
        return Path(text.replace("/mnt/c/Users/", "/mnt/c/users/", 1))
    return path


ROOT = _normalize_writable_path(Path(__file__).resolve().parents[1])
FAST_LANE_TRIAL_LEDGER_PATH = ROOT / "docs" / "runtime" / "fast_lane_trial_ledger.yaml"
FAST_LANE_RUNNER_ID = "research/chordia_strict_unlock_v1.py:fast_lane_v5.1"


@dataclass(frozen=True)
class Cell:
    hypothesis_file: Path
    hypothesis_name: str
    strategy_id: str
    instrument: str
    orb_label: str
    orb_minutes: int
    entry_model: str
    confirm_bars: int
    rr_target: float
    filter_key: str
    has_theory: bool
    theory_mode: str
    result_md: Path
    result_csv: Path
    result_summary_csv: Path
    # FAST_LANE v5.1 awareness. None for legacy heavyweight prereg (preserves
    # back-compat with every prereg authored before 2026-05-18). Unknown values
    # are rejected at load time (fail-closed per institutional-rigor § 6).
    template_version: str | None
    direction: str | None


def _resolve_output_paths(hypothesis_path: Path) -> tuple[Path, Path, Path]:
    stem = hypothesis_path.stem
    results_dir = ROOT / "docs" / "audit" / "results"
    return (
        results_dir / f"{stem}.md",
        results_dir / f"{stem}.csv",
        results_dir / f"{stem}.summary.csv",
    )


def _load_cell(hypothesis_path: Path) -> Cell:
    hypothesis_path = hypothesis_path.resolve()
    meta = load_hypothesis_metadata(hypothesis_path)
    check_mode_a_consistency(meta)

    body = yaml.safe_load(hypothesis_path.read_text(encoding="utf-8"))
    scope = body.get("scope", {})
    grounding = body.get("grounding", {})
    hypotheses = body.get("hypotheses", [])
    if not isinstance(scope, dict):
        raise SystemExit("Prereg top-level 'scope' must be a mapping for this bounded runner.")
    if not isinstance(hypotheses, list) or len(hypotheses) != 1:
        raise SystemExit("This bounded runner requires exactly one hypothesis entry.")

    # Fail-closed on non-default stop_multiplier. orb_outcomes does not store a
    # stop_multiplier column — its trade stream is built at the default 1.0 stop.
    # An S-suffixed strategy_id (e.g. *_S075) refers to a different physical
    # trade stream that requires outcome_builder rebuild at the target stop.
    # This runner has no such pathway; silently auditing the default-stop trades
    # under an S-suffixed id would compute a t-stat against the wrong cohort.
    sid = str(scope["strategy_id"])
    sid_stop = parse_stop_multiplier(sid)
    if sid_stop != DEFAULT_STOP_MULTIPLIER:
        sys.stderr.write(
            f"REFUSE: strategy {sid!r} has stop_multiplier={sid_stop} != {DEFAULT_STOP_MULTIPLIER}. "
            "This runner audits canonical orb_outcomes which is built at the default stop. "
            "Non-default stops require an outcome_builder rebuild at the target stop and a "
            "different runner. Refusing to run rather than audit the wrong trade stream.\n"
        )
        raise SystemExit(2)
    scope_stop = scope.get("stop_multiplier")
    if scope_stop is not None and float(scope_stop) != DEFAULT_STOP_MULTIPLIER:
        sys.stderr.write(
            f"REFUSE: prereg scope.stop_multiplier={scope_stop} != {DEFAULT_STOP_MULTIPLIER}. "
            "Same fail-closed reason as above.\n"
        )
        raise SystemExit(2)

    filter_status = grounding.get("filter_grounding_status", {}) if isinstance(grounding, dict) else {}
    # FAST_LANE v5.1: metadata.template_version routes the runner to a second
    # verdict branch. Missing field => None => heavyweight-only (back-compat).
    # Unknown non-None value => fail-closed (no silent fall-through).
    raw_template = meta.get("metadata", {}).get("template_version") if isinstance(meta.get("metadata"), dict) else None
    if raw_template is None:
        raw_template = body.get("metadata", {}).get("template_version")
    template_version: str | None
    if raw_template in (None, ""):
        template_version = None
    else:
        tv = str(raw_template).strip()
        if tv not in SUPPORTED_TEMPLATE_VERSIONS:
            sys.stderr.write(
                f"REFUSE: prereg metadata.template_version={tv!r} is not in supported set "
                f"{sorted(SUPPORTED_TEMPLATE_VERSIONS)}. Fail-closed per institutional-rigor § 6 "
                "(no silent fall-through to heavyweight when operator declared a template).\n"
            )
            raise SystemExit(2)
        template_version = tv
    direction = scope.get("direction")
    direction_str: str | None = str(direction).strip().lower() if direction is not None else None
    # Direction-enum guard. Template line 78 accepts only {pooled, long, short}.
    # Without this, an unknown direction silently bypasses gate 5 (per-direction
    # sign-check) via the non-pooled "else" branch, which would let an operator
    # accidentally PROMOTE a pooled cell that lacks per-direction sign evidence.
    # Fail-closed on unknown value per institutional-rigor § 6 (no silent failures).
    if template_version == "fast_lane_v5.1" and direction_str not in {"pooled", "long", "short"}:
        sys.stderr.write(
            f"REFUSE: prereg scope.direction={direction!r} (normalized {direction_str!r}) "
            f"is not in supported set {{pooled, long, short}}. FAST_LANE v5.1 gate 5 "
            "requires a known direction to route the per-direction sign-check.\n"
        )
        raise SystemExit(2)
    result_md, result_csv, result_summary_csv = _resolve_output_paths(hypothesis_path)
    return Cell(
        hypothesis_file=hypothesis_path,
        hypothesis_name=str(hypotheses[0].get("name", meta["name"])),
        strategy_id=str(scope["strategy_id"]),
        instrument=str(scope["instrument"]),
        orb_label=str(scope["session"]),
        orb_minutes=int(scope["orb_minutes"]),
        entry_model=str(scope["entry_model"]),
        confirm_bars=int(scope["confirm_bars"]),
        rr_target=float(scope["rr_target"]),
        filter_key=str(scope["filter_type"]),
        has_theory=bool(meta["has_theory"]),
        theory_mode=str(filter_status.get("verdict", "UNSPECIFIED")),
        result_md=result_md,
        result_csv=result_csv,
        result_summary_csv=result_summary_csv,
        template_version=template_version,
        direction=direction_str,
    )


def _load_universe(con: duckdb.DuckDBPyConnection, cell: Cell, *, is_only: bool) -> pd.DataFrame:
    op = "<" if is_only else ">="
    # Match canonical promoter cohort: strategy_discovery._load_outcomes_bulk applies
    # WF_START_OVERRIDE per instrument (e.g. MNQ/MES = 2020-01-01 micro-launch exclusion;
    # MGC = 2022-01-01 low-ATR regime). Without this, audit fires on pre-cutoff trades
    # that the promoter never saw, and audit N != validated_setups.sample_size.
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
            o.risk_dollars,
            o.pnl_dollars,
            o.mae_r,
            o.mfe_r,
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
          {start_clause}
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


def _inject_cross_asset_atr(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    *,
    filter_key: str,
    instrument: str,
) -> pd.DataFrame:
    filt_obj = ALL_FILTERS.get(filter_key)
    if not isinstance(filt_obj, CrossAssetATRFilter):
        return df
    source = filt_obj.source_instrument
    if source == instrument:
        return df
    src_rows = con.execute(
        """
        SELECT trading_day, atr_20_pct
        FROM daily_features
        WHERE symbol = ? AND orb_minutes = 5 AND atr_20_pct IS NOT NULL
        """,
        [source],
    ).fetchall()
    src_map = {td.date() if hasattr(td, "date") else td: float(pct) for td, pct in src_rows}
    col = f"cross_atr_{source}_pct"
    out = df.copy()
    out[col] = out["trading_day"].apply(lambda d: src_map.get(d.date() if hasattr(d, "date") else d))
    return out


def _direction_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        ["long" if tp > sp else "short" for tp, sp in zip(df["target_price"], df["stop_price"], strict=False)],
        index=df.index,
    )


def _nan_result(sample_label: str) -> dict[str, Any]:
    return {
        "sample": sample_label,
        "n_universe": 0,
        "n_fired": 0,
        "scratch_n": 0,
        "null_non_scratch_n": 0,
        "fire_rate": float("nan"),
        "expr": float("nan"),
        "policy_ev": float("nan"),
        "std_r": float("nan"),
        "sharpe": float("nan"),
        "t": float("nan"),
        "p_two_sided": float("nan"),
        "long_n": 0,
        "long_t": float("nan"),
        "long_expr": float("nan"),
        "short_n": 0,
        "short_t": float("nan"),
        "short_expr": float("nan"),
    }


def _evaluate_split(df: pd.DataFrame, cell: Cell, *, sample_label: str) -> tuple[dict[str, Any], pd.Series, pd.DataFrame]:
    if df.empty:
        return _nan_result(sample_label), pd.Series(dtype=bool), df

    fire_mask = pd.Series(filter_signal(df, cell.filter_key, cell.orb_label).astype(bool), index=df.index)
    fired = df.loc[fire_mask].copy()
    if fired.empty:
        result = _nan_result(sample_label)
        result["n_universe"] = int(len(df))
        result["fire_rate"] = 0.0
        return result, fire_mask, fired

    scratch_mask = fired["outcome"].astype(str).eq("scratch")
    null_mask = fired["pnl_r"].isna()
    fired["pnl_eff"] = fired["pnl_r"].fillna(0.0)
    n = int(len(fired))
    mean_r = float(fired["pnl_eff"].mean())
    std_r = float(fired["pnl_eff"].std(ddof=1)) if n >= 2 else float("nan")
    sharpe = mean_r / std_r if n >= 2 and std_r > 0 else float("nan")
    t_stat = compute_chordia_t(sharpe, n) if n >= 2 and std_r > 0 else float("nan")
    p_two = math.erfc(abs(t_stat) / math.sqrt(2.0)) if math.isfinite(t_stat) else float("nan")

    long_t = short_t = float("nan")
    long_n = short_n = 0
    long_expr = short_expr = float("nan")
    if {"target_price", "stop_price"}.issubset(fired.columns):
        directions = _direction_series(fired)
        for key in ("long", "short"):
            sub = fired.loc[directions.eq(key), "pnl_eff"]
            sub_n = int(len(sub))
            if key == "long":
                long_n = sub_n
            else:
                short_n = sub_n
            if sub_n == 0:
                continue
            sub_mean = float(sub.mean())
            sub_std = float(sub.std(ddof=1)) if sub_n >= 2 else float("nan")
            sub_t = compute_chordia_t(sub_mean / sub_std, sub_n) if sub_n >= 2 and sub_std > 0 else float("nan")
            if key == "long":
                long_expr = sub_mean
                long_t = sub_t
            else:
                short_expr = sub_mean
                short_t = sub_t

    result = {
        "sample": sample_label,
        "n_universe": int(len(df)),
        "n_fired": n,
        "scratch_n": int(scratch_mask.sum()),
        "null_non_scratch_n": int((null_mask & ~scratch_mask).sum()),
        "fire_rate": float(n / len(df)) if len(df) else float("nan"),
        "expr": mean_r,
        "policy_ev": float((fire_mask.astype(int) * df["pnl_r"].fillna(0.0)).mean()),
        "std_r": std_r,
        "sharpe": sharpe,
        "t": float(t_stat) if math.isfinite(t_stat) else float("nan"),
        "p_two_sided": p_two,
        "long_n": long_n,
        "long_t": long_t,
        "long_expr": long_expr,
        "short_n": short_n,
        "short_t": short_t,
        "short_expr": short_expr,
    }
    return result, fire_mask, fired


def _verdict(is_result: dict[str, Any], oos_result: dict[str, Any], threshold: float, has_theory: bool) -> tuple[str, str]:
    is_t = is_result["t"]
    is_n = is_result["n_fired"]
    is_expr = is_result["expr"]
    if not math.isfinite(is_t) or is_n < 2:
        return "SCAN_ABORT", "IS sample <2 trades or undefined t-stat."
    if is_t < threshold:
        return "FAIL_STRICT_CHORDIA", f"IS t={is_t:.3f} < {threshold:.2f}."
    if not math.isfinite(is_expr) or is_expr <= 0.0:
        return "FAIL_STRICT_CHORDIA", f"IS ExpR={is_expr:.4f} <= 0."
    if is_n < 100:
        return "FAIL_STRICT_CHORDIA", f"N_IS_on={is_n} < 100."

    oos_n = oos_result["n_fired"]
    oos_expr = oos_result["expr"]
    if oos_n >= OOS_DESCRIPTIVE_MIN_N and math.isfinite(oos_expr):
        if (is_expr > 0.0) == (oos_expr > 0.0):
            if has_theory and is_t < CHORDIA_T_WITHOUT_THEORY:
                return "PASS_PROTOCOL_A", (
                    f"IS clears theory threshold {threshold:.2f} with N={is_n} and ExpR={is_expr:.4f}; "
                    f"OOS sign matches at N_OOS={oos_n}."
                )
            return "PASS_CHORDIA", (
                f"IS clears strict threshold {threshold:.2f} with N={is_n} and ExpR={is_expr:.4f}; "
                f"OOS sign matches at N_OOS={oos_n}."
            )
        return "PARK", (
            f"IS gates clear but OOS sign opposes IS once N_OOS={oos_n} >= {OOS_DESCRIPTIVE_MIN_N}."
        )

    if has_theory and is_t < CHORDIA_T_WITHOUT_THEORY:
        return "PASS_PROTOCOL_A_OOS_UNDERPOWERED", (
            f"IS clears theory threshold {threshold:.2f}; OOS N={oos_n} < {OOS_DESCRIPTIVE_MIN_N}."
        )
    return "PASS_CHORDIA_OOS_UNDERPOWERED", (
        f"IS clears strict threshold {threshold:.2f}; OOS N={oos_n} < {OOS_DESCRIPTIVE_MIN_N}."
    )


def _fast_lane_verdict_v5_1(
    is_result: dict[str, Any],
    boundary: dict[str, Any],
    direction: str | None,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Compute FAST_LANE v5.1 verdict per TEMPLATE-fast-lane-v5.1.yaml gates.

    Gate order matches the template's outcomes precedence:
        1. Holdout boundary proof — any failure forces NEEDS-MORE (line 130-131).
        2-4. Fire-rate / ExpR / N gates — KILL band.
        5. Per-direction sign-check (pooled only) — NEEDS-MORE if missing or flipped.
        6. t-stat band — PROMOTE / NEEDS-MORE / KILL.

    First failing gate determines the verdict; later gates are recorded as
    "not evaluated" in the gate-rows table for operator audit.

    Returns
    -------
    verdict
        One of ``PROMOTE``, ``NEEDS-MORE``, ``KILL``.
    reason
        One-sentence rationale referencing the deciding gate.
    gate_rows
        List of dicts ``{name, threshold, observed, pass}`` for the result MD.
    """
    rows: list[dict[str, Any]] = []
    decided: tuple[str, str] | None = None

    def _record(name: str, threshold_text: str, observed_text: str, passed: bool | str) -> None:
        rows.append({
            "name": name,
            "threshold": threshold_text,
            "observed": observed_text,
            "pass": passed,
        })

    # Gate 1 — holdout boundary proof.
    proof = bool(boundary.get("holdout_boundary_proof"))
    proof_text = (
        f"max_IS={boundary.get('max_IS_trading_day') or '∅'} < "
        f"{boundary.get('holdout_boundary_value') or '2026-01-01'} ≤ "
        f"min_OOS={boundary.get('min_OOS_trading_day') or '∅'}"
    )
    _record("Holdout boundary proof", "max_IS < 2026-01-01 ≤ min_OOS", proof_text, proof)
    if not proof:
        decided = (
            "NEEDS-MORE",
            "Holdout boundary not proven — evidence uninterpretable (template § outcomes precedence: any holdout problem forces NEEDS-MORE).",
        )

    # Gate 2 — fire-rate band.
    fire = is_result.get("fire_rate")
    fire_ok = (
        isinstance(fire, (int, float))
        and math.isfinite(fire)
        and FAST_LANE_V5_1_FIRE_LO <= fire <= FAST_LANE_V5_1_FIRE_HI
    )
    fire_text = f"{fire:.4f}" if isinstance(fire, (int, float)) and math.isfinite(fire) else "nan"
    _record(
        "Fire-rate band",
        f"{FAST_LANE_V5_1_FIRE_LO:.2f} ≤ fire ≤ {FAST_LANE_V5_1_FIRE_HI:.2f}",
        fire_text,
        fire_ok if decided is None else "not evaluated",
    )
    if decided is None and not fire_ok:
        decided = (
            "KILL",
            f"Fire-rate {fire_text} outside [{FAST_LANE_V5_1_FIRE_LO:.2f}, {FAST_LANE_V5_1_FIRE_HI:.2f}] — degenerate filter, not a t-stat failure.",
        )

    # Gate 3 — ExpR_IS > 0.
    expr = is_result.get("expr")
    expr_ok = isinstance(expr, (int, float)) and math.isfinite(expr) and expr > FAST_LANE_V5_1_EXPR_MIN
    expr_text = f"{expr:.4f}" if isinstance(expr, (int, float)) and math.isfinite(expr) else "nan"
    _record(
        "ExpR_IS strict positive",
        f"> {FAST_LANE_V5_1_EXPR_MIN:.2f}",
        expr_text,
        expr_ok if decided is None else "not evaluated",
    )
    if decided is None and not expr_ok:
        decided = ("KILL", f"ExpR_IS={expr_text} ≤ {FAST_LANE_V5_1_EXPR_MIN:.2f}.")

    # Gate 4 — N_IS_on ≥ 50.
    n_is = int(is_result.get("n_fired") or 0)
    n_ok = n_is >= FAST_LANE_V5_1_N_MIN
    _record(
        "N_IS_on triage min",
        f"≥ {FAST_LANE_V5_1_N_MIN}",
        str(n_is),
        n_ok if decided is None else "not evaluated",
    )
    if decided is None and not n_ok:
        decided = ("KILL", f"N_IS_on={n_is} < {FAST_LANE_V5_1_N_MIN}.")

    # Gate 5 — per-direction sign-check (pooled lanes only).
    if direction == "pooled":
        long_raw = is_result.get("long_expr")
        short_raw = is_result.get("short_expr")
        long_f = float(long_raw) if isinstance(long_raw, (int, float)) and math.isfinite(float(long_raw)) else None
        short_f = float(short_raw) if isinstance(short_raw, (int, float)) and math.isfinite(float(short_raw)) else None
        if long_f is None or short_f is None:
            sign_pass: bool = False
            sign_obs = f"long_ExpR={long_raw!r}, short_ExpR={short_raw!r} (missing)"
        else:
            sign_pass = (long_f > 0.0) == (short_f > 0.0)
            sign_obs = (
                f"sign(long_ExpR={long_f:.4f})={'+' if long_f > 0.0 else '-'}, "
                f"sign(short_ExpR={short_f:.4f})={'+' if short_f > 0.0 else '-'}"
            )
        _record(
            "Per-direction sign-check (pooled)",
            "sign(long_ExpR) == sign(short_ExpR) AND both present",
            sign_obs,
            sign_pass if decided is None else "not evaluated",
        )
        if decided is None and not sign_pass:
            decided = (
                "NEEDS-MORE",
                "Pooled direction without per-direction sign-match; would bounce at pooled-finding-rule heavyweight stage — gated at triage.",
            )
    else:
        # single-direction lanes bypass per template line 128 (`single_direction_lanes` rule).
        _record(
            "Per-direction sign-check (pooled)",
            "n/a — single-direction lane bypass",
            f"direction={direction!r}",
            "bypass",
        )

    # Gate 6 — t-stat band.
    t_raw = is_result.get("t")
    t_f: float | None = float(t_raw) if isinstance(t_raw, (int, float)) and math.isfinite(float(t_raw)) else None
    t_text = f"{t_f:.3f}" if t_f is not None else "nan"
    if t_f is not None and t_f >= FAST_LANE_V5_1_PROMOTE_T:
        t_band = "PROMOTE"
    elif t_f is not None and t_f >= FAST_LANE_V5_1_NEEDS_MORE_T:
        t_band = "NEEDS-MORE"
    else:
        t_band = "KILL"
    _record(
        "t-stat band",
        f"≥ {FAST_LANE_V5_1_PROMOTE_T:.1f} PROMOTE / [{FAST_LANE_V5_1_NEEDS_MORE_T:.1f}, {FAST_LANE_V5_1_PROMOTE_T:.1f}) NEEDS-MORE / < {FAST_LANE_V5_1_NEEDS_MORE_T:.1f} KILL",
        f"t={t_text} → {t_band}",
        t_band if decided is None else "not evaluated",
    )
    if decided is None:
        if t_band == "PROMOTE":
            decided = ("PROMOTE", f"t={t_text} clears {FAST_LANE_V5_1_PROMOTE_T:.1f} PROMOTE band; all gates pass.")
        elif t_band == "NEEDS-MORE":
            decided = ("NEEDS-MORE", f"t={t_text} in NEEDS-MORE band [{FAST_LANE_V5_1_NEEDS_MORE_T:.1f}, {FAST_LANE_V5_1_PROMOTE_T:.1f}).")
        else:
            decided = ("KILL", f"t={t_text} < {FAST_LANE_V5_1_NEEDS_MORE_T:.1f}.")

    verdict, reason = decided
    return verdict, reason, rows


def _as_iso_day(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, _dt.date):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _split_boundary_metadata(is_df: pd.DataFrame, oos_df: pd.DataFrame) -> dict[str, Any]:
    """Compute holdout-boundary proof fields from the IS/OOS universe frames.

    Returns the latest IS trading day, the earliest OOS trading day, and a
    boolean proof that the realized split honored the canonical holdout
    boundary (``trading_app.holdout_policy.HOLDOUT_SACRED_FROM``).

    The proof requirement is ``max_IS_trading_day < HOLDOUT_SACRED_FROM <= min_OOS_trading_day``.
    When a split has zero rows the bound on that side is absent — the proof
    field reports ``True`` if the SQL boundary clause was the only thing
    excluding rows on that side (i.e. ``True`` when one side is empty AND
    the populated side respects its bound), ``False`` otherwise.
    """

    def _series_max(df: pd.DataFrame) -> Any:
        if df.empty or "trading_day" not in df.columns:
            return None
        return df["trading_day"].max()

    def _series_min(df: pd.DataFrame) -> Any:
        if df.empty or "trading_day" not in df.columns:
            return None
        return df["trading_day"].min()

    raw_max_is = _series_max(is_df)
    raw_min_oos = _series_min(oos_df)
    max_is = raw_max_is.date() if hasattr(raw_max_is, "date") else raw_max_is
    min_oos = raw_min_oos.date() if hasattr(raw_min_oos, "date") else raw_min_oos

    is_ok = (max_is is None) or (max_is < HOLDOUT_SACRED_FROM)
    oos_ok = (min_oos is None) or (min_oos >= HOLDOUT_SACRED_FROM)
    proof = bool(is_ok and oos_ok)

    return {
        "max_IS_trading_day": _as_iso_day(max_is),
        "min_OOS_trading_day": _as_iso_day(min_oos),
        "holdout_boundary_value": HOLDOUT_SACRED_FROM.isoformat(),
        "holdout_boundary_proof": proof,
    }


_SUMMARY_CSV_COLUMNS: tuple[str, ...] = (
    "sample",
    "n_universe",
    "n_fired",
    "fire_rate",
    "expr",
    "policy_ev",
    "sharpe",
    "t",
    "p_two_sided",
    "long_n",
    "long_expr",
    "long_t",
    "short_n",
    "short_expr",
    "short_t",
    "max_IS_trading_day",
    "min_OOS_trading_day",
    "holdout_boundary_value",
    "holdout_boundary_proof",
)


def _summary_csv_row(result: dict[str, Any], boundary: dict[str, Any]) -> list[Any]:
    return [
        result["sample"],
        result["n_universe"],
        result["n_fired"],
        result["fire_rate"],
        result["expr"],
        result["policy_ev"],
        result["sharpe"],
        result["t"],
        result["p_two_sided"],
        result["long_n"],
        result["long_expr"],
        result["long_t"],
        result["short_n"],
        result["short_expr"],
        result["short_t"],
        boundary["max_IS_trading_day"],
        boundary["min_OOS_trading_day"],
        boundary["holdout_boundary_value"],
        boundary["holdout_boundary_proof"],
    ]


def _write_summary_csv(
    summary_csv_path: Path,
    is_result: dict[str, Any],
    oos_result: dict[str, Any],
    boundary: dict[str, Any],
) -> None:
    """Emit a per-split summary CSV: two rows (IS, OOS), machine-readable.

    Distinct from the per-trade CSV at ``cell.result_csv`` — this surface is
    consumed by ``TEMPLATE-fast-lane-v5.yaml`` and any downstream calibration
    runner that needs per-direction ExpR and holdout boundary fields without
    parsing markdown.
    """
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(list(_SUMMARY_CSV_COLUMNS))
        writer.writerow(_summary_csv_row(is_result, boundary))
        writer.writerow(_summary_csv_row(oos_result, boundary))


def _write_csv(csv_path: Path, fired_frames: list[tuple[str, pd.DataFrame]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "sample",
                "trading_day",
                "strategy_id",
                "outcome",
                "pnl_r_raw",
                "pnl_r_effective",
                "scratch",
                "null_pnl_non_scratch",
                "target_price",
                "stop_price",
                "direction",
            ]
        )
        for sample, frame in fired_frames:
            if frame.empty:
                continue
            directions = _direction_series(frame) if {"target_price", "stop_price"}.issubset(frame.columns) else None
            for idx, row in frame.iterrows():
                raw = row["pnl_r"]
                scratch = str(row.get("outcome")) == "scratch"
                writer.writerow(
                    [
                        sample,
                        row["trading_day"].isoformat() if hasattr(row["trading_day"], "isoformat") else row["trading_day"],
                        "",
                        row.get("outcome", ""),
                        "" if pd.isna(raw) else float(raw),
                        float(row["pnl_eff"]),
                        int(scratch),
                        int(pd.isna(raw) and not scratch),
                        "" if pd.isna(row.get("target_price")) else float(row["target_price"]),
                        "" if pd.isna(row.get("stop_price")) else float(row["stop_price"]),
                        directions.loc[idx] if directions is not None else "unknown",
                    ]
                )


def _fmt(value: Any, places: int = 4) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.{places}f}"
    return "nan"


def _summary_row(label: str, result: dict[str, Any]) -> str:
    return (
        f"| {label} | {result['n_universe']} | {result['n_fired']} | {_fmt(result['fire_rate'] * 100, 2)}% | "
        f"{result['scratch_n']} | {result['null_non_scratch_n']} | {_fmt(result['expr'])} | {_fmt(result['policy_ev'])} | "
        f"{_fmt(result['sharpe'])} | {_fmt(result['t'], 3)} | {_fmt(result['p_two_sided'], 5)} |"
    )


def _fast_lane_block_lines(
    fl_verdict: str,
    fl_reason: str,
    fl_rows: list[dict[str, Any]],
    direction: str | None,
) -> list[str]:
    """Render the FAST_LANE v5.1 result-MD block.

    Sentinel-string ``## FAST_LANE v5.1 verdict (automated)`` is load-bearing:
    ``pipeline/check_drift.py::check_fast_lane_runner_template_routing`` greps
    for this exact heading and the ``**FAST_LANE verdict:**`` line to verify
    the runner did not silently bypass the v5.1 branch. Do not rename without
    updating that drift check.
    """
    lines = [
        "## FAST_LANE v5.1 verdict (automated)",
        "",
        f"**FAST_LANE verdict:** `{fl_verdict}`",
        "",
        fl_reason,
        "",
        "Computed by `_fast_lane_verdict_v5_1()` per "
        "`docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml` § screen + § outcomes. "
        "This block is automated; the heavyweight Chordia verdict above is independent and unchanged.",
        "",
        "### Gate table",
        "",
        "| # | Gate | Threshold | Observed | Pass |",
        "|---|---|---|---|---|",
    ]
    for idx, row in enumerate(fl_rows, start=1):
        passed = row["pass"]
        if isinstance(passed, bool):
            pass_text = "yes" if passed else "no"
        else:
            pass_text = str(passed)
        lines.append(
            f"| {idx} | {row['name']} | {row['threshold']} | {row['observed']} | {pass_text} |"
        )
    lines.append("")
    # Diagnostic note when the deciding gate is fire-rate (gate 2). This surfaces
    # the "degenerate filter, not t-stat failure" class that heavyweight Chordia
    # currently has no gate for. Per stage design § 7 self-check.
    fire_row = fl_rows[1] if len(fl_rows) >= 2 else None
    if (
        fire_row is not None
        and fire_row.get("name") == "Fire-rate band"
        and fire_row.get("pass") is False
    ):
        lines.extend([
            "> **Diagnostic note:** the fire-rate gate decided this verdict, not the t-stat band. "
            "A fire rate outside [0.05, 0.95] indicates a degenerate filter (firing on nearly every "
            "session or almost never) rather than a weak edge. The heavyweight Chordia verdict above "
            "does not currently apply a fire-rate gate — operator should not treat its t-stat verdict "
            "as the diagnostic signal here.",
            "",
        ])
    lines.extend([
        "### What PROMOTE authorizes",
        "",
        "- Authoring a heavyweight Chordia pre-reg for this lane (theory grant + clustered SE at trading_day + OOS power floor + era-stability section + DSR + Harvey-Liu haircut).",
        "- Nothing else.",
        "",
        "### What PROMOTE does NOT authorize",
        "",
        "- Capital allocation.",
        "- Writing this cell into `chordia_audit_log.yaml` as `PASS_CHORDIA`.",
        "- Sibling-cell rescue (other RR / CB / aperture / session variants need their own pre-reg).",
        "- Treating the FAST_LANE verdict as a substitute for paper-trade + SR-monitor validation.",
        "",
        f"_Scope direction at screen: `{direction!r}`. "
        "Pooled lanes require both per-direction ExpRs and same-sign to PROMOTE; single-direction lanes bypass that gate._",
        "",
    ])
    return lines


def _write_markdown(
    cell: Cell,
    threshold: float,
    verdict: str,
    verdict_reason: str,
    is_result: dict[str, Any],
    oos_result: dict[str, Any],
    fast_lane: tuple[str, str, list[dict[str, Any]]] | None = None,
) -> None:
    cell.result_md.parent.mkdir(parents=True, exist_ok=True)
    wf_start = WF_START_OVERRIDE.get(cell.instrument)
    lines = [
        f"# Chordia strict unlock audit — {cell.strategy_id}",
        "",
        f"**Prereq file:** `{cell.hypothesis_file.relative_to(ROOT)}`",
        f"**Result CSV:** `{cell.result_csv.relative_to(ROOT)}`",
        f"**Canonical DB:** `{GOLD_DB_PATH}`",
        "",
        "## Scope",
        "",
        f"Strict-Chordia unlock audit for the exact lane `{cell.strategy_id}`. "
        f"Tests whether the bounded canonical replay clears Chordia's strict t-stat hurdle "
        f"({threshold:.2f}, has_theory={cell.has_theory}) on canonical IS data, with "
        "descriptive OOS sign-match as a secondary gate. Single-lane K=1 confirmatory replay; "
        "no parameter sweeps, no filter variants, no instrument extensions.",
        "",
        "## Verdict",
        "",
        f"**MEASURED verdict:** `{verdict}`",
        "",
        verdict_reason,
        "",
        f"**MEASURED theory mode:** `{cell.theory_mode}`",
        f"**MEASURED threshold applied:** `{threshold:.2f}`",
        f"**MEASURED loader has_theory:** `{cell.has_theory}`",
        "",
        "## Split summary",
        "",
        "| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        _summary_row("IS", is_result),
        _summary_row("OOS", oos_result),
        "",
        "## Directional breakdown",
        "",
        "| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| IS | {is_result['long_n']} | {_fmt(is_result['long_expr'])} | {_fmt(is_result['long_t'], 3)} | "
            f"{is_result['short_n']} | {_fmt(is_result['short_expr'])} | {_fmt(is_result['short_t'], 3)} |"
        ),
        (
            f"| OOS | {oos_result['long_n']} | {_fmt(oos_result['long_expr'])} | {_fmt(oos_result['long_t'], 3)} | "
            f"{oos_result['short_n']} | {_fmt(oos_result['short_expr'])} | {_fmt(oos_result['short_t'], 3)} |"
        ),
        "",
        "## Method notes",
        "",
        "- Canonical source only: `orb_outcomes` joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.",
        f"- Sacred holdout boundary: `trading_day < {HOLDOUT_SACRED_FROM}` for IS, `>=` for descriptive OOS.",
        (
            f"- Cohort lower bound: `WF_START_OVERRIDE['{cell.instrument}']={wf_start}` applied "
            "to match canonical promoter (`trading_app/strategy_discovery._load_outcomes_bulk`)."
            if wf_start is not None
            else f"- Cohort lower bound: none (no `WF_START_OVERRIDE` entry for `{cell.instrument}`)."
        ),
        f"- Canonical filter delegation: `filter_signal(..., '{cell.filter_key}', '{cell.orb_label}')`.",
        "- Scratch handling: `pnl_r NULL -> 0.0` in the measured trade stream; scratch and null-non-scratch counts are reported separately.",
        "- No writes to `experimental_strategies`, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`.",
        "",
        "## Reproduction",
        "",
        "```",
        f"python research/chordia_strict_unlock_v1.py --hypothesis-file {cell.hypothesis_file.relative_to(ROOT)}",
        "```",
        "",
        "Outputs (overwritten in place):",
        "",
        f"- `{cell.result_md.relative_to(ROOT)}`",
        f"- `{cell.result_csv.relative_to(ROOT)}`",
        "",
        "## Caveats",
        "",
        "- Single-lane K=1 confirmatory replay; the strict t-stat hurdle does not include a "
        "search-family multiple-comparison correction. Survivorship/multiple-testing risk is "
        "carried by the upstream pre-registration, not this replay.",
        "- IS sample size in this audit reports `N_fired` (wins+losses+scratches with R=0). "
        "`validated_setups.sample_size` reports wins+losses only. Comparing the two t-stats "
        "directly is not like-for-like; reconcile via the scratch count reported above.",
        "- OOS window is descriptive only. Sign-match at `N_OOS >= 30` is a confirmatory gate, "
        "not a deployment criterion. PARK on OOS sign-flip means insufficient confirmation, "
        "not falsification.",
        "- Cross-asset enrichment (e.g., `cross_atr_MES_pct` for `X_MES_ATR60`) is computed "
        "in this runner from `daily_features.atr_20_pct` of the source instrument; verify "
        "the canonical promoter's enrichment path agrees before treating verdicts as "
        "directly comparable.",
        "",
    ]
    if fast_lane is not None:
        fl_verdict, fl_reason, fl_rows = fast_lane
        lines.extend(_fast_lane_block_lines(fl_verdict, fl_reason, fl_rows, cell.direction))
    cell.result_md.write_text("\n".join(lines), encoding="utf-8")


def _rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _combined_artifact_sha(paths: tuple[Path, ...]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"FAST_LANE trial ledger append requires YAML mapping prereg, got {type(data).__name__}")
    return data


def _fast_lane_structural_hash_inputs(scope: dict[str, Any]) -> dict[str, Any]:
    direction = str(scope.get("direction", "pooled")).strip().lower()
    direction_for_hash = {
        "pooled": "BOTH",
        "both": "BOTH",
        "long": "LONG",
        "short": "SHORT",
    }.get(direction)
    if direction_for_hash is None:
        raise ValueError(f"FAST_LANE trial ledger append: unsupported scope.direction {scope.get('direction')!r}")
    return {
        "instrument": scope["instrument"],
        "orb_label": scope["session"],
        "orb_minutes": int(scope["orb_minutes"]),
        "rr_target": float(scope["rr_target"]),
        "entry_model": scope["entry_model"],
        "confirm_bars": int(scope["confirm_bars"]),
        "filter_type": scope.get("filter_type", ""),
        "direction": direction_for_hash,
        "filter_threshold": scope.get("filter_threshold", ""),
    }


def _metadata_block(prereg: dict[str, Any]) -> dict[str, Any]:
    meta = prereg.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _ledger_pathway(raw: Any) -> str:
    token = str(raw or "A").strip().upper()
    return "B" if token.startswith("B") else "A"


def _ledger_testing_mode(raw: Any) -> str:
    token = str(raw or "individual").strip().lower()
    if token not in {"family", "individual"}:
        raise ValueError(f"FAST_LANE trial ledger append: testing_mode must be family|individual, got {raw!r}")
    return token


def _ledger_k_declared(raw: Any) -> int:
    if isinstance(raw, bool):
        raise TypeError("FAST_LANE trial ledger append: K_declared cannot be bool")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"FAST_LANE trial ledger append: K_declared must be int-like, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"FAST_LANE trial ledger append: K_declared must be >= 1, got {value}")
    return value


def _canonical_data_fingerprint(boundary: dict[str, Any]) -> str:
    payload = {
        "gold_db_path": str(GOLD_DB_PATH),
        "holdout_sacred_from": HOLDOUT_SACRED_FROM.isoformat(),
        "max_IS_trading_day": boundary.get("max_IS_trading_day"),
        "min_OOS_trading_day": boundary.get("min_OOS_trading_day"),
        "holdout_boundary_value": boundary.get("holdout_boundary_value"),
        "holdout_boundary_proof": boundary.get("holdout_boundary_proof"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _compute_trial_k_lineage(
    *,
    structural_hash: str,
    hash_inputs: dict[str, Any],
    ledger_rows: list[dict[str, Any]],
    k_declared: int,
    pooled_n: int,
) -> dict[str, Any]:
    k_lane = sum(1 for row in ledger_rows if row.get("structural_hash") == structural_hash)
    k_family = sum(
        1
        for row in ledger_rows
        if (
            (row.get("k_lineage") or {}).get("instrument") == hash_inputs.get("instrument")
            and (row.get("k_lineage") or {}).get("orb_label") == hash_inputs.get("orb_label")
            and (row.get("k_lineage") or {}).get("orb_minutes") == hash_inputs.get("orb_minutes")
        )
    )
    k_global = len(ledger_rows)
    e_max_n = max(int(pooled_n), 1)
    k_effective_minbtl = 2.0 * math.log(max(k_global, 2)) / (e_max_n * e_max_n)
    rho_hat_assumed = 0.5
    m_correlated = max(k_family, 1)
    n_hat = int(round(pooled_n * (rho_hat_assumed + (1.0 - rho_hat_assumed) * m_correlated)))
    return {
        "instrument": hash_inputs.get("instrument"),
        "orb_label": hash_inputs.get("orb_label"),
        "orb_minutes": hash_inputs.get("orb_minutes"),
        "K_global": k_global,
        "K_family": k_family,
        "K_lane": k_lane,
        "K_declared_in_prereg": int(k_declared),
        "K_effective_minBTL": k_effective_minbtl,
        "bh_fdr_passes": {
            "K_family": k_family <= 1,
            "K_lane": k_lane <= 1,
            "K_global": True,
        },
        "correlation_haircut_N_hat": n_hat,
        "rho_hat_assumed": rho_hat_assumed,
    }


def _append_fast_lane_trial_ledger_entry(
    cell: Cell,
    is_result: dict[str, Any],
    oos_result: dict[str, Any],
    boundary: dict[str, Any],
    fast_lane: tuple[str, str, list[dict[str, Any]]] | None,
    *,
    verdict: str,
    verdict_reason: str,
) -> None:
    if cell.template_version != "fast_lane_v5.1":
        return
    if fast_lane is None:
        raise RuntimeError("FAST_LANE trial ledger append requested but fast_lane verdict block is missing")

    prereg = _load_yaml_mapping(cell.hypothesis_file)
    scope = prereg.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("FAST_LANE trial ledger append requires prereg top-level scope mapping")
    meta = _metadata_block(prereg)
    hash_inputs = _fast_lane_structural_hash_inputs(scope)
    structural_hash = compute_structural_hash(hash_inputs)

    prereg_sha = _file_sha256(cell.hypothesis_file)
    result_artifact_sha = _combined_artifact_sha((cell.result_md, cell.result_csv, cell.result_summary_csv))
    data_fingerprint = _canonical_data_fingerprint(boundary)
    trial_id = compute_trial_id(
        prereg_sha=prereg_sha,
        runner_id=FAST_LANE_RUNNER_ID,
        result_artifact_sha=result_artifact_sha,
        canonical_data_fingerprint=data_fingerprint,
    )

    ledger_rows = list(read_ledger(FAST_LANE_TRIAL_LEDGER_PATH).get("entries", []))
    for row in ledger_rows:
        if row.get("trial_id") != trial_id:
            continue
        provenance = row.get("upstream_provenance") or {}
        if (
            row.get("prereg_sha") == prereg_sha
            and provenance.get("runner_id") == FAST_LANE_RUNNER_ID
            and provenance.get("result_artifact_sha") == result_artifact_sha
            and provenance.get("canonical_data_fingerprint") == data_fingerprint
        ):
            return
        raise RuntimeError(
            f"FAST_LANE trial ledger append refused: trial_id {trial_id!r} already exists with different provenance"
        )

    k_declared = _ledger_k_declared(meta.get("total_expected_trials", 1))
    fl_verdict, fl_reason, _ = fast_lane
    run_ts = _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    entry = LedgerEntry(
        run_id=f"chordia_strict_unlock_v1:{run_ts}:{trial_id}",
        trial_id=trial_id,
        run_timestamp_utc=run_ts,
        prereg_path=_rel_to_root(cell.hypothesis_file),
        prereg_sha=prereg_sha,
        structural_hash=structural_hash,
        template_version=cell.template_version,
        testing_mode=_ledger_testing_mode(meta.get("testing_mode", "individual")),
        pathway=_ledger_pathway(meta.get("pathway", "A")),
        K_declared=k_declared,
        k_lineage=_compute_trial_k_lineage(
            structural_hash=structural_hash,
            hash_inputs=hash_inputs,
            ledger_rows=ledger_rows,
            k_declared=k_declared,
            pooled_n=int(is_result.get("n_fired") or 0),
        ),
        n_hat=float(is_result.get("n_fired") or 0),
        upstream_provenance={
            "runner_id": FAST_LANE_RUNNER_ID,
            "result_md": _rel_to_root(cell.result_md),
            "result_csv": _rel_to_root(cell.result_csv),
            "result_summary_csv": _rel_to_root(cell.result_summary_csv),
            "result_artifact_sha": result_artifact_sha,
            "canonical_data_fingerprint": data_fingerprint,
        },
        outcome={
            "heavyweight_verdict": verdict,
            "heavyweight_reason": verdict_reason,
            "fast_lane_verdict": fl_verdict,
            "fast_lane_reason": fl_reason,
            "is_n_fired": int(is_result.get("n_fired") or 0),
            "is_t": is_result.get("t"),
            "is_expr": is_result.get("expr"),
            "is_fire_rate": is_result.get("fire_rate"),
            "oos_n_fired": int(oos_result.get("n_fired") or 0),
            "oos_t": oos_result.get("t"),
            "oos_expr": oos_result.get("expr"),
        },
    )
    append_trial_ledger_entry(FAST_LANE_TRIAL_LEDGER_PATH, entry)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one exact-lane Chordia strict-unlock prereg.")
    parser.add_argument("--hypothesis-file", required=True, help="Path to the prereg YAML file.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    hypothesis_path = Path(args.hypothesis_file)
    if not hypothesis_path.is_absolute():
        hypothesis_path = ROOT / hypothesis_path

    cell = _load_cell(hypothesis_path)
    threshold = chordia_threshold(cell.has_theory)
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    is_df = _inject_cross_asset_atr(
        con,
        _load_universe(con, cell, is_only=True),
        filter_key=cell.filter_key,
        instrument=cell.instrument,
    )
    oos_df = _inject_cross_asset_atr(
        con,
        _load_universe(con, cell, is_only=False),
        filter_key=cell.filter_key,
        instrument=cell.instrument,
    )
    is_result, _, is_fired = _evaluate_split(is_df, cell, sample_label="IS")
    oos_result, _, oos_fired = _evaluate_split(oos_df, cell, sample_label="OOS")
    verdict, verdict_reason = _verdict(is_result, oos_result, threshold, cell.has_theory)
    boundary = _split_boundary_metadata(is_df, oos_df)

    fast_lane: tuple[str, str, list[dict[str, Any]]] | None = None
    if cell.template_version == "fast_lane_v5.1":
        fast_lane = _fast_lane_verdict_v5_1(is_result, boundary, cell.direction)

    _write_csv(cell.result_csv, [("IS", is_fired), ("OOS", oos_fired)])
    _write_summary_csv(cell.result_summary_csv, is_result, oos_result, boundary)
    _write_markdown(cell, threshold, verdict, verdict_reason, is_result, oos_result, fast_lane=fast_lane)
    _append_fast_lane_trial_ledger_entry(
        cell,
        is_result,
        oos_result,
        boundary,
        fast_lane,
        verdict=verdict,
        verdict_reason=verdict_reason,
    )

    print(f"Strategy: {cell.strategy_id}")
    print(f"Prereg: {cell.hypothesis_file.relative_to(ROOT)}")
    print(f"Heavyweight Chordia verdict: {verdict}")
    print(f"Heavyweight reason: {verdict_reason}")
    if fast_lane is not None:
        fl_verdict, fl_reason, _ = fast_lane
        print(f"FAST_LANE v5.1 verdict: {fl_verdict}")
        print(f"FAST_LANE reason: {fl_reason}")
    print(f"Result MD: {cell.result_md.relative_to(ROOT)}")
    print(f"Result CSV: {cell.result_csv.relative_to(ROOT)}")
    print(f"Result Summary CSV: {cell.result_summary_csv.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
