"""Bounded exact-lane Chordia strict-unlock runner.

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
from trading_app.config import ALL_FILTERS, WF_START_OVERRIDE, CrossAssetATRFilter
from trading_app.chordia import (
    CHORDIA_T_WITHOUT_THEORY,
    chordia_threshold,
    compute_chordia_t,
)
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.hypothesis_loader import check_mode_a_consistency, load_hypothesis_metadata
from trading_app.strategy_discovery import parse_stop_multiplier


OOS_DESCRIPTIVE_MIN_N = 30
DEFAULT_STOP_MULTIPLIER = 1.0


def _normalize_writable_path(path: Path) -> Path:
    text = str(path)
    if text.startswith("/mnt/c/Users/"):
        return Path(text.replace("/mnt/c/Users/", "/mnt/c/users/", 1))
    return path


ROOT = _normalize_writable_path(Path(__file__).resolve().parents[1])


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


def _resolve_output_paths(hypothesis_path: Path) -> tuple[Path, Path]:
    stem = hypothesis_path.stem
    results_dir = ROOT / "docs" / "audit" / "results"
    return results_dir / f"{stem}.md", results_dir / f"{stem}.csv"


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
    result_md, result_csv = _resolve_output_paths(hypothesis_path)
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


def _write_markdown(
    cell: Cell,
    threshold: float,
    verdict: str,
    verdict_reason: str,
    is_result: dict[str, Any],
    oos_result: dict[str, Any],
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
    cell.result_md.write_text("\n".join(lines), encoding="utf-8")


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

    _write_csv(cell.result_csv, [("IS", is_fired), ("OOS", oos_fired)])
    _write_markdown(cell, threshold, verdict, verdict_reason, is_result, oos_result)

    print(f"Strategy: {cell.strategy_id}")
    print(f"Prereg: {cell.hypothesis_file.relative_to(ROOT)}")
    print(f"Verdict: {verdict}")
    print(f"Reason: {verdict_reason}")
    print(f"Result MD: {cell.result_md.relative_to(ROOT)}")
    print(f"Result CSV: {cell.result_csv.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
