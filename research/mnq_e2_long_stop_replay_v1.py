"""Clean E2 long-stop replay for prior-day geometry candidates.

This runner answers the deployable question that the contaminated
``PD_*`` E2 shelf rows do not answer:

    If a trader places a long stop at the ORB high after ORB formation,
    conditioned only on prior-day geometry and ORB midpoint, what happens?

It deliberately does not use ``orb_<session>_break_dir`` as a selector.
No writes to gold.db, ``validated_setups``, or runtime allocation state.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from pipeline.build_daily_features import compute_trading_day_utc_range
from pipeline.cost_model import get_cost_spec
from pipeline.dst import orb_utc_window
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import E2_SLIPPAGE_TICKS, WF_START_OVERRIDE
from trading_app.entry_rules import detect_break_touch
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.outcome_builder import _compute_outcomes_all_rr
from trading_app.entry_rules import _resolve_e2


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ReplaySpec:
    strategy_id: str
    instrument: str
    session: str
    orb_minutes: int
    rr_target: float
    filter_type: str


def _pd_context_state(row: pd.Series, spec: ReplaySpec) -> str | None:
    hi = row.get(f"orb_{spec.session}_high")
    lo = row.get(f"orb_{spec.session}_low")
    pdh = row.get("prev_day_high")
    pdl = row.get("prev_day_low")
    pdc = row.get("prev_day_close")
    atr = row.get("atr_20")
    if any(pd.isna(v) for v in (hi, lo, pdh, pdl)):
        return None
    orb_mid = (float(hi) + float(lo)) / 2.0

    if spec.filter_type == "PD_CLEAR_LONG":
        if pd.isna(pdc) or pd.isna(atr) or float(atr) <= 0:
            return None
        pivot = (float(pdh) + float(pdl) + float(pdc)) / 3.0
        inside_pdr = float(pdl) < orb_mid < float(pdh)
        near_pivot = abs(orb_mid - pivot) / float(atr) < 0.50
        return "TAKE_CLEAR_OF_CONGESTION" if not (inside_pdr or near_pivot) else "VETO_CONGESTED"

    if spec.filter_type == "PD_DISPLACE_LONG":
        if pd.isna(atr) or float(atr) <= 0:
            return None
        displaced = orb_mid < float(pdl) or abs(orb_mid - float(pdl)) / float(atr) < 0.15
        return "TAKE_DOWNSIDE_DISPLACEMENT" if displaced else "VETO_NO_DOWNSIDE_DISPLACEMENT"

    if spec.filter_type == "PD_GO_LONG":
        if pd.isna(pdc) or pd.isna(atr) or float(atr) <= 0:
            return None
        pivot = (float(pdh) + float(pdl) + float(pdc)) / 3.0
        downside_displacement = orb_mid < float(pdl) or abs(orb_mid - float(pdl)) / float(atr) < 0.15
        inside_pdr = float(pdl) < orb_mid < float(pdh)
        near_pivot = abs(orb_mid - pivot) / float(atr) < 0.50
        clear_of_congestion = not (inside_pdr or near_pivot)
        return "TAKE_GO_LONG_CONTEXT" if (downside_displacement or clear_of_congestion) else "VETO_NO_GO_LONG_CONTEXT"

    raise ValueError(f"unsupported clean prior-day filter_type: {spec.filter_type}")


def _summary(values: list[float], *, opportunities: int) -> dict[str, Any]:
    if not values:
        return {
            "n_fired": 0,
            "expr": math.nan,
            "policy_ev": 0.0,
            "std_r": math.nan,
            "sharpe": math.nan,
            "t": math.nan,
            "wins": 0,
            "losses": 0,
            "scratches": 0,
        }
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
        sharpe = mean / std if std > 0 else math.nan
        t_stat = sharpe * math.sqrt(n) if std > 0 else math.nan
    else:
        std = math.nan
        sharpe = math.nan
        t_stat = math.nan
    return {
        "n_fired": n,
        "expr": mean,
        "policy_ev": sum(values) / opportunities if opportunities else math.nan,
        "std_r": std,
        "sharpe": sharpe,
        "t": t_stat,
        "wins": sum(1 for v in values if v > 0),
        "losses": sum(1 for v in values if v < 0),
        "scratches": sum(1 for v in values if v == 0),
    }


def _load_features(con: duckdb.DuckDBPyConnection, spec: ReplaySpec) -> pd.DataFrame:
    start = WF_START_OVERRIDE.get(spec.instrument)
    start_clause = "AND trading_day >= ?" if start is not None else ""
    params: list[Any] = [spec.instrument, spec.orb_minutes]
    if start is not None:
        params.append(start)
    return con.execute(
        f"""
        SELECT *
        FROM daily_features
        WHERE symbol = ?
          AND orb_minutes = ?
          {start_clause}
        ORDER BY trading_day
        """,
        params,
    ).df()


def _load_day_bars(con: duckdb.DuckDBPyConnection, instrument: str, start_ts, end_ts) -> pd.DataFrame:
    return con.execute(
        """
        SELECT ts_utc, open, high, low, close
        FROM bars_1m
        WHERE symbol = ?
          AND ts_utc >= ?
          AND ts_utc < ?
        ORDER BY ts_utc
        """,
        [instrument, start_ts, end_ts],
    ).df()


def run(spec: ReplaySpec, *, db_path: Path = GOLD_DB_PATH) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    con = duckdb.connect(str(db_path), read_only=True)
    cost_spec = get_cost_spec(spec.instrument)
    rows: list[dict[str, Any]] = []
    try:
        features = _load_features(con, spec)
        for _, feat in features.iterrows():
            raw_trading_day = feat["trading_day"]
            trading_day = raw_trading_day.date() if hasattr(raw_trading_day, "date") else raw_trading_day
            state = _pd_context_state(feat, spec)
            context_pass = state is not None and state.startswith("TAKE")
            if not context_pass:
                continue

            orb_high = feat.get(f"orb_{spec.session}_high")
            orb_low = feat.get(f"orb_{spec.session}_low")
            if pd.isna(orb_high) or pd.isna(orb_low):
                continue

            orb_start, orb_end = orb_utc_window(trading_day, spec.session, spec.orb_minutes)
            day_start, day_end = compute_trading_day_utc_range(trading_day)
            bars = _load_day_bars(con, spec.instrument, orb_start, day_end)
            if bars.empty:
                continue

            touch = detect_break_touch(
                bars,
                float(orb_high),
                float(orb_low),
                "long",
                orb_end,
                day_end,
            )
            result = {
                "trading_day": trading_day,
                "split": "OOS" if trading_day >= HOLDOUT_SACRED_FROM else "IS",
                "context_state": state,
                "context_pass": True,
                "touched_long_stop": touch.touched,
                "entry_ts": None,
                "pnl_r": None,
                "outcome": None,
            }
            if touch.touched:
                signal = _resolve_e2(touch, slippage_ticks=E2_SLIPPAGE_TICKS, tick_size=cost_spec.tick_size)
                outcome = _compute_outcomes_all_rr(
                    bars,
                    signal,
                    float(orb_high),
                    float(orb_low),
                    "long",
                    [spec.rr_target],
                    day_end,
                    cost_spec,
                    entry_model="E2",
                    orb_label=spec.session,
                )[0]
                result.update(
                    entry_ts=outcome["entry_ts"],
                    pnl_r=0.0 if outcome["pnl_r"] is None else float(outcome["pnl_r"]),
                    outcome=outcome["outcome"],
                )
            rows.append(result)
    finally:
        con.close()

    df = pd.DataFrame(rows)
    summaries: dict[str, dict[str, Any]] = {}
    for split in ("IS", "OOS"):
        sub = df[df["split"] == split] if not df.empty else df
        values = [float(v) for v in sub["pnl_r"].dropna().tolist()] if not sub.empty else []
        summaries[split] = {
            "n_context": int(len(sub)) if not sub.empty else 0,
            **_summary(values, opportunities=int(len(sub)) if not sub.empty else 0),
        }
    return df, summaries


def _write_outputs(spec: ReplaySpec, df: pd.DataFrame, summaries: dict[str, dict[str, Any]]) -> tuple[Path, Path]:
    stem = f"2026-05-10-clean-long-stop-{spec.strategy_id.lower().replace('_', '-')}"
    result_dir = ROOT / "docs" / "audit" / "results"
    csv_path = result_dir / f"{stem}.csv"
    md_path = result_dir / f"{stem}.md"
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

    lines = [
        f"# Clean E2 long-stop replay — {spec.strategy_id}",
        "",
        "## Question",
        "",
        "Does the prior-day geometry context work when expressed as a live-placeable long stop, "
        "without using close-confirmed `break_dir` as a selector?",
        "",
        "## Method",
        "",
        "- Canonical layers: `bars_1m` + `daily_features`.",
        "- Entry: long E2 stop at ORB high after ORB formation, with canonical E2 slippage.",
        "- Selector: prior-day geometry state from ORB midpoint and prior-day levels only.",
        "- Forbidden: `orb_<session>_break_dir` selection, sibling rescue, threshold tuning.",
        "",
        "## Summary",
        "",
        "| Split | N context | N fired | ExpR fired | Policy EV/context | t | Wins | Losses | Scratches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, s in summaries.items():
        lines.append(
            f"| {split} | {s['n_context']} | {s['n_fired']} | {s['expr']:.4f} | "
            f"{s['policy_ev']:.4f} | {s['t']:.3f} | {s['wins']} | {s['losses']} | {s['scratches']} |"
        )
    lines.extend(
        [
            "",
            "## Classification Use",
            "",
            "This replay is a clean falsification surface for the prior-day long idea. "
            "It is not a live promotion by itself; deployment still requires the normal "
            "Criterion 4/8/9, additivity, runtime, SR, survival, and preflight gates.",
            "",
            f"CSV: `{csv_path.relative_to(ROOT)}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy-id", required=True)
    parser.add_argument("--instrument", default="MNQ")
    parser.add_argument("--session", required=True)
    parser.add_argument("--orb-minutes", type=int, default=5)
    parser.add_argument("--rr-target", type=float, required=True)
    parser.add_argument("--filter-type", choices=["PD_CLEAR_LONG", "PD_DISPLACE_LONG", "PD_GO_LONG"], required=True)
    args = parser.parse_args()
    spec = ReplaySpec(
        strategy_id=args.strategy_id,
        instrument=args.instrument,
        session=args.session,
        orb_minutes=args.orb_minutes,
        rr_target=args.rr_target,
        filter_type=args.filter_type,
    )
    df, summaries = run(spec)
    md_path, csv_path = _write_outputs(spec, df, summaries)
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
