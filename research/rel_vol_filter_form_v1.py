"""Locked rel_vol filter-form follow-on for MES/MGC.

Pre-reg:
  docs/audit/hypotheses/2026-04-21-rel-vol-filter-form-v1.yaml

Scope:
  - MES and MGC only
  - E2 / CB1 / O5 / RR1.5
  - rel_vol lineage only
  - canonical inputs only: orb_outcomes + daily_features + trading_app filters

This script does two things:
  1. Train per-lane rel_vol quintile thresholds on trading_day < 2024-01-01
  2. Evaluate the locked filter forms on 2024-2025 validation IS:
       F1 = Q5 only
       F2 = Q4 + Q5

Semi-OOS 2026-01-01..2026-04-19 is reported as informational only.
Fresh OOS starts 2026-04-22 and is not gated here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal
from research.pr48_sizer_rule_skeptical_reaudit_v1 import _sessions
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

PREREG_PATH = Path("docs/audit/hypotheses/2026-04-21-rel-vol-filter-form-v1.yaml")
RESULT_DOC = Path("docs/audit/results/2026-04-21-rel-vol-filter-form-v1.md")
INSTRUMENTS = ("MGC", "MES")
APERTURE = 5
RR = 1.5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
TRAIN_END = pd.Timestamp("2024-01-01")
VAL_START = TRAIN_END
VAL_END = pd.Timestamp(HOLDOUT_SACRED_FROM)
SEMI_OOS_START = VAL_END
SEMI_OOS_END = pd.Timestamp("2026-04-20")
FRESH_OOS_START = pd.Timestamp("2026-04-22")
MIN_TRAIN_PER_LANE = 100
BOOTSTRAP_BLOCK = 20
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 20260421
TAUTOLOGY_FILTERS = ("COST_LT12", "ORB_G5", "ATR_P50", "OVNRNG_50", "VWAP_MID_ALIGNED")
EXECUTION_VETO_REASON = (
    "Canonical `trading_app.config.VolumeFilter` marks `rel_vol` as `E2`-excluded: "
    "it uses break-bar volume and resolves at `BREAK_DETECTED`, which is unknown at "
    "E2 order placement."
)


@dataclass(frozen=True)
class FilterForm:
    key: str
    label: str
    fire_quintiles: frozenset[int]


@dataclass
class WindowMetrics:
    n_total: int
    n_fire: int
    fire_rate: float
    uniform_expr: float
    uniform_sr: float
    filter_expr: float
    filter_sr: float
    delta_expr: float
    delta_sr: float
    ci_lo: float | None
    ci_hi: float | None


@dataclass
class FormResult:
    form: FilterForm
    val: WindowMetrics
    semi_oos: WindowMetrics
    fresh_oos: WindowMetrics
    gate1_pass: bool
    gate2_pass: bool
    gate3_pass: bool
    gate4_pass: bool
    verdict: str
    fail_reasons: list[str]


FORMS = (
    FilterForm(key="F1_Q5_only", label="Q5 only", fire_quintiles=frozenset({5})),
    FilterForm(key="F2_Q4_plus_Q5", label="Q4 + Q5", fire_quintiles=frozenset({4, 5})),
)


def _fmt(x: float | None, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{x:+.{nd}f}"


def _fmt_plain(x: float | None, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{x:.{nd}f}"


def _fmt_pct(x: float | None, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{x:.{nd}%}"


def _sharpe(x: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return float("nan")
    return float(np.mean(x) / sd)


def _load_session_frame(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    session: str,
) -> pd.DataFrame:
    rel_col = f"rel_vol_{session}"
    size_col = f"orb_{session}_size"
    hi_col = f"orb_{session}_high"
    lo_col = f"orb_{session}_low"
    break_col = f"orb_{session}_break_dir"
    vwap_col = f"orb_{session}_vwap"
    sql = f"""
    SELECT
      o.trading_day,
      o.symbol,
      o.pnl_r,
      d.{rel_col} AS rel_vol,
      d.{size_col} AS orb_size,
      d.{hi_col} AS orb_high,
      d.{lo_col} AS orb_low,
      d.{break_col} AS break_dir,
      d.{vwap_col} AS session_vwap,
      d.atr_20_pct,
      d.overnight_range,
      d.{size_col},
      d.{hi_col},
      d.{lo_col},
      d.{break_col},
      d.{vwap_col}
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = '{instrument}'
      AND o.orb_label = '{session}'
      AND o.orb_minutes = {APERTURE}
      AND o.entry_model = '{ENTRY_MODEL}'
      AND o.confirm_bars = {CONFIRM_BARS}
      AND o.rr_target = {RR}
      AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day
    """
    df = con.sql(sql).to_df()
    if df.empty:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["session"] = session
    df["direction"] = df["break_dir"].astype(str)
    df["lane"] = df["session"] + "_" + df["direction"]
    return df


def _load_instrument(con: duckdb.DuckDBPyConnection, instrument: str) -> pd.DataFrame:
    frames = []
    for session in _sessions(con, instrument):
        frame = _load_session_frame(con, instrument, session)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"No rows for {instrument}")
    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values(["trading_day", "session", "direction"]).reset_index(drop=True)
    return full


def _train_thresholds(full: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    train = full.loc[full["trading_day"] < TRAIN_END].copy()
    thresholds: dict[str, np.ndarray] = {}
    for lane, lane_df in train.groupby("lane"):
        vals = lane_df["rel_vol"].dropna().astype(float).to_numpy()
        if len(vals) < MIN_TRAIN_PER_LANE:
            continue
        thresholds[lane] = np.quantile(vals, [0.2, 0.4, 0.6, 0.8])
    return thresholds, train


def _bucket(rel_vol: float, thresholds: np.ndarray) -> int:
    return int(np.searchsorted(thresholds, rel_vol, side="right") + 1)


def _apply_thresholds(df: pd.DataFrame, thresholds: dict[str, np.ndarray]) -> pd.DataFrame:
    kept = df.loc[df["lane"].isin(thresholds)].copy()
    if kept.empty:
        kept["quintile"] = pd.Series(dtype="Int64")
        return kept

    quintiles = np.full(len(kept), -1, dtype=int)
    for lane, lane_df in kept.groupby("lane"):
        t = thresholds[lane]
        rel = lane_df["rel_vol"].astype(float).to_numpy()
        idx = lane_df.index.to_numpy()
        lane_quint = np.full(len(lane_df), -1, dtype=int)
        valid = np.isfinite(rel)
        lane_quint[valid] = np.searchsorted(t, rel[valid], side="right") + 1
        quintiles[np.searchsorted(kept.index.to_numpy(), idx)] = lane_quint
    kept["quintile"] = pd.Series(quintiles, index=kept.index, dtype="Int64")
    kept.loc[kept["quintile"] < 1, "quintile"] = pd.NA
    return kept.reset_index(drop=True)


def _moving_block_delta_sharpe_ci(
    pnl: np.ndarray,
    fire_mask: np.ndarray,
    *,
    block_size: int,
    n_boot: int,
    seed: int,
) -> tuple[float | None, float | None]:
    n = len(pnl)
    if n < max(block_size * 2, 30):
        return None, None
    if fire_mask.sum() < 2:
        return None, None

    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block_size))
    base = np.arange(block_size)
    deltas = np.empty(n_boot, dtype=float)
    keep = 0
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([base + s for s in starts])[:n]
        pnl_boot = pnl[idx]
        fire_boot = fire_mask[idx]
        fired = pnl_boot[fire_boot]
        if len(fired) < 2:
            continue
        uniform_sr = _sharpe(pnl_boot)
        filter_sr = _sharpe(fired)
        if not np.isfinite(uniform_sr) or not np.isfinite(filter_sr):
            continue
        deltas[keep] = filter_sr - uniform_sr
        keep += 1
    if keep < 100:
        return None, None
    sample = deltas[:keep]
    return float(np.percentile(sample, 2.5)), float(np.percentile(sample, 97.5))


def _window_metrics(df: pd.DataFrame, form: FilterForm) -> WindowMetrics:
    n_total = len(df)
    if n_total == 0:
        return WindowMetrics(
            n_total=0,
            n_fire=0,
            fire_rate=float("nan"),
            uniform_expr=float("nan"),
            uniform_sr=float("nan"),
            filter_expr=float("nan"),
            filter_sr=float("nan"),
            delta_expr=float("nan"),
            delta_sr=float("nan"),
            ci_lo=None,
            ci_hi=None,
        )

    fire_mask = df["quintile"].isin(form.fire_quintiles).fillna(False).to_numpy(dtype=bool)
    pnl = df["pnl_r"].astype(float).to_numpy()
    fired = pnl[fire_mask]
    uniform_expr = float(np.mean(pnl))
    uniform_sr = _sharpe(pnl)
    if len(fired) == 0:
        return WindowMetrics(
            n_total=n_total,
            n_fire=0,
            fire_rate=0.0,
            uniform_expr=uniform_expr,
            uniform_sr=uniform_sr,
            filter_expr=float("nan"),
            filter_sr=float("nan"),
            delta_expr=float("nan"),
            delta_sr=float("nan"),
            ci_lo=None,
            ci_hi=None,
        )

    filter_expr = float(np.mean(fired))
    filter_sr = _sharpe(fired)
    delta_expr = filter_expr - uniform_expr
    delta_sr = filter_sr - uniform_sr
    ci_lo, ci_hi = _moving_block_delta_sharpe_ci(
        pnl,
        fire_mask,
        block_size=BOOTSTRAP_BLOCK,
        n_boot=BOOTSTRAP_B,
        seed=BOOTSTRAP_SEED,
    )
    return WindowMetrics(
        n_total=n_total,
        n_fire=int(fire_mask.sum()),
        fire_rate=float(fire_mask.mean()),
        uniform_expr=uniform_expr,
        uniform_sr=uniform_sr,
        filter_expr=filter_expr,
        filter_sr=filter_sr,
        delta_expr=delta_expr,
        delta_sr=delta_sr,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
    )


def _evaluate_form(
    val_df: pd.DataFrame,
    semi_df: pd.DataFrame,
    fresh_df: pd.DataFrame,
    form: FilterForm,
) -> FormResult:
    val = _window_metrics(val_df, form)
    semi = _window_metrics(semi_df, form)
    fresh = _window_metrics(fresh_df, form)

    gate1 = bool(np.isfinite(val.fire_rate) and 0.05 <= val.fire_rate <= 0.40)
    gate2 = bool(np.isfinite(val.filter_expr) and val.filter_expr >= 0.05)
    gate3 = bool(np.isfinite(val.filter_sr) and np.isfinite(val.uniform_sr) and val.filter_sr > val.uniform_sr)
    gate4 = bool(
        gate1
        and gate2
        and gate3
        and val.ci_lo is not None
        and val.ci_hi is not None
        and val.ci_lo > 0
    )

    fail_reasons: list[str] = []
    if not gate1:
        fail_reasons.append("Gate1 fire-rate outside [5%, 40%]")
    if not gate2:
        fail_reasons.append("Gate2 val-IS filter ExpR < +0.05R")
    if not gate3:
        fail_reasons.append("Gate3 filter Sharpe <= uniform Sharpe")
    if gate1 and gate2 and gate3 and not gate4:
        fail_reasons.append("Gate4 delta-Sharpe bootstrap CI includes 0")

    if gate4:
        verdict = f"FILTER_ALIVE_IS_{form.key.split('_')[0]}"
    elif gate1 and gate2 and gate3:
        verdict = "FILTER_RESEARCH_SURVIVOR"
    else:
        verdict = "FILTER_DEAD_IS"

    return FormResult(
        form=form,
        val=val,
        semi_oos=semi,
        fresh_oos=fresh,
        gate1_pass=gate1,
        gate2_pass=gate2,
        gate3_pass=gate3,
        gate4_pass=gate4,
        verdict=verdict,
        fail_reasons=fail_reasons,
    )


def _per_lane_metrics(df: pd.DataFrame, form: FilterForm) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lane, lane_df in df.groupby("lane"):
        m = _window_metrics(lane_df, form)
        rows.append(
            {
                "lane": lane,
                "n_total": m.n_total,
                "n_fire": m.n_fire,
                "fire_rate": m.fire_rate,
                "filter_expr": m.filter_expr,
                "filter_sr": m.filter_sr,
            }
        )
    return pd.DataFrame(rows).sort_values("lane").reset_index(drop=True)


def _per_year_metrics(df: pd.DataFrame, form: FilterForm) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, year_df in df.groupby(df["trading_day"].dt.year):
        m = _window_metrics(year_df, form)
        split = "train" if year_df["trading_day"].max() < VAL_START else "val"
        rows.append(
            {
                "year": int(year),
                "split": split,
                "n_total": m.n_total,
                "n_fire": m.n_fire,
                "fire_rate": m.fire_rate,
                "filter_expr": m.filter_expr,
                "filter_sr": m.filter_sr,
            }
        )
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def _per_direction_metrics(df: pd.DataFrame, form: FilterForm) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for direction, direction_df in df.groupby("direction"):
        m = _window_metrics(direction_df, form)
        rows.append(
            {
                "direction": direction,
                "n_total": m.n_total,
                "n_fire": m.n_fire,
                "fire_rate": m.fire_rate,
                "filter_expr": m.filter_expr,
                "filter_sr": m.filter_sr,
                "uniform_expr": m.uniform_expr,
                "uniform_sr": m.uniform_sr,
            }
        )
    return pd.DataFrame(rows).sort_values("direction").reset_index(drop=True)


def _binary_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) == 0:
        return float("nan")
    if a.sum() == 0 or a.sum() == len(a):
        return 0.0
    if b.sum() == 0 or b.sum() == len(b):
        return 0.0
    corr = np.corrcoef(a.astype(float), b.astype(float))[0, 1]
    return 0.0 if np.isnan(corr) else float(abs(corr))


def _tautology_table(df: pd.DataFrame, form: FilterForm) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for session, session_df in df.groupby("session"):
        candidate = session_df["quintile"].isin(form.fire_quintiles).fillna(False).to_numpy(dtype=int)
        for filter_key in TAUTOLOGY_FILTERS:
            deployed = filter_signal(session_df, filter_key, session)
            corr = _binary_corr(candidate, deployed)
            rows.append(
                {
                    "session": session,
                    "filter_key": filter_key,
                    "corr_abs": corr,
                    "flag": corr > 0.70,
                }
            )
    return pd.DataFrame(rows).sort_values(["session", "filter_key"]).reset_index(drop=True)


def _winner(results: tuple[FormResult, ...]) -> tuple[str, FormResult | None]:
    alive = [r for r in results if r.gate4_pass]
    if alive:
        best = max(alive, key=lambda r: r.val.filter_sr)
        return "CANDIDATE_READY_IS", best
    survivors = [r for r in results if r.gate1_pass and r.gate2_pass and r.gate3_pass]
    if survivors:
        best = max(survivors, key=lambda r: r.val.filter_sr)
        return "FILTER_RESEARCH_SURVIVOR", best
    return "FILTER_DEAD_IS", None


def _window_slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp | None = None) -> pd.DataFrame:
    mask = df["trading_day"] >= start
    if end is not None:
        mask &= df["trading_day"] < end
    return df.loc[mask].copy()


def _emit_instrument_section(
    lines: list[str],
    instrument: str,
    form_results: tuple[FormResult, ...],
    winner_status: str,
    winner: FormResult | None,
    full_kept: pd.DataFrame,
    val_df: pd.DataFrame,
    semi_df: pd.DataFrame,
    fresh_df: pd.DataFrame,
) -> None:
    lines.append(f"## {instrument}")
    lines.append("")
    lines.append(
        f"Kept lanes after train-threshold eligibility (`N_train >= {MIN_TRAIN_PER_LANE}`): "
        f"{full_kept['lane'].nunique()} lanes, {len(full_kept)} total trades across train/val/OOS."
    )
    lines.append("")
    lines.append(
        "| Form | Val N | Fire N | Fire Rate | Filter ExpR | Uniform ExpR | Filter SR | Uniform SR | ΔSR | Bootstrap 95% CI ΔSR | Gate1 | Gate2 | Gate3 | Gate4 | Verdict |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|")
    for res in form_results:
        ci = (
            f"[{_fmt_plain(res.val.ci_lo, 4)}, {_fmt_plain(res.val.ci_hi, 4)}]"
            if res.val.ci_lo is not None and res.val.ci_hi is not None
            else "NA"
        )
        lines.append(
            f"| {res.form.key} | {res.val.n_total} | {res.val.n_fire} | {res.val.fire_rate:.1%} | "
            f"{_fmt_plain(res.val.filter_expr,5)} | {_fmt_plain(res.val.uniform_expr,5)} | "
            f"{_fmt_plain(res.val.filter_sr,4)} | {_fmt_plain(res.val.uniform_sr,4)} | "
            f"{_fmt_plain(res.val.delta_sr,4)} | {ci} | "
            f"{'PASS' if res.gate1_pass else 'FAIL'} | "
            f"{'PASS' if res.gate2_pass else 'FAIL'} | "
            f"{'PASS' if res.gate3_pass else 'FAIL'} | "
            f"{'PASS' if res.gate4_pass else 'FAIL'} | "
            f"**{res.verdict}** |"
        )
    lines.append("")
    lines.append(f"Instrument summary: **{winner_status}**.")
    lines.append("")

    if winner is not None:
        lines.append(
            f"Winning form by locked rule: `{winner.form.key}` ({winner.form.label}) "
            f"with val-IS Sharpe `{_fmt_plain(winner.val.filter_sr,4)}`."
        )
        lines.append("")
    else:
        lines.append("No form cleared Gates 1-3, so there is no winning deployment form on val-IS.")
        lines.append("")

    lines.append("### Fail Reasons")
    lines.append("")
    for res in form_results:
        reasons = "; ".join(res.fail_reasons) if res.fail_reasons else "None"
        lines.append(f"- `{res.form.key}`: {reasons}")
    lines.append("")

    lines.append("### Semi-OOS (Informational Only)")
    lines.append("")
    lines.append(
        f"Window in pre-reg: `2026-01-01 .. 2026-04-19`. Available canonical data in this repo currently ends at "
        f"`{semi_df['trading_day'].max().date() if len(semi_df) else 'NA'}`."
    )
    lines.append("")
    lines.append("| Form | N | Fire N | Fire Rate | Filter ExpR | Uniform ExpR | Filter SR | Uniform SR |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for res in form_results:
        lines.append(
            f"| {res.form.key} | {res.semi_oos.n_total} | {res.semi_oos.n_fire} | "
            f"{_fmt_pct(res.semi_oos.fire_rate)} | {_fmt_plain(res.semi_oos.filter_expr,5)} | "
            f"{_fmt_plain(res.semi_oos.uniform_expr,5)} | {_fmt_plain(res.semi_oos.filter_sr,4)} | "
            f"{_fmt_plain(res.semi_oos.uniform_sr,4)} |"
        )
    lines.append("")

    lines.append("### Fresh OOS (Not Yet Accrued)")
    lines.append("")
    lines.append(
        f"Fresh OOS starts at `{FRESH_OOS_START.date()}`. Current kept-sample trades in that window: "
        f"`{len(fresh_df)}` total."
    )
    lines.append("")
    lines.append("| Form | N | Fire N | Fire Rate | Filter ExpR | Filter SR |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for res in form_results:
        lines.append(
            f"| {res.form.key} | {res.fresh_oos.n_total} | {res.fresh_oos.n_fire} | "
            f"{_fmt_pct(res.fresh_oos.fire_rate)} | {_fmt_plain(res.fresh_oos.filter_expr,5)} | "
            f"{_fmt_plain(res.fresh_oos.filter_sr,4)} |"
        )
    lines.append("")

    if winner is not None:
        lane_df = _per_lane_metrics(val_df, winner.form)
        year_df = _per_year_metrics(full_kept.loc[full_kept["trading_day"] < VAL_END], winner.form)
        dir_df = _per_direction_metrics(val_df, winner.form)
        taut_df = _tautology_table(val_df, winner.form)

        lines.append("### Winning Form Per-Lane (Val-IS)")
        lines.append("")
        lines.append("| Lane | Val N | Fire N | Fire Rate | Filter ExpR | Filter SR |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in lane_df.to_dict("records"):
            lines.append(
                f"| {row['lane']} | {row['n_total']} | {row['n_fire']} | {row['fire_rate']:.1%} | "
                f"{_fmt_plain(float(row['filter_expr']),5)} | {_fmt_plain(float(row['filter_sr']),4)} |"
            )
        lines.append("")

        lines.append("### Winning Form Per-Year (Full IS; 2024-2025 Are the Gated Val Years)")
        lines.append("")
        lines.append("| Year | Split | N | Fire N | Fire Rate | Filter ExpR | Filter SR |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in year_df.to_dict("records"):
            lines.append(
                f"| {row['year']} | {row['split']} | {row['n_total']} | {row['n_fire']} | "
                f"{row['fire_rate']:.1%} | {_fmt_plain(float(row['filter_expr']),5)} | "
                f"{_fmt_plain(float(row['filter_sr']),4)} |"
            )
        lines.append("")

        lines.append("### Winning Form Per-Direction (Val-IS)")
        lines.append("")
        lines.append("| Direction | Val N | Fire N | Fire Rate | Filter ExpR | Uniform ExpR | Filter SR | Uniform SR |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in dir_df.to_dict("records"):
            lines.append(
                f"| {row['direction']} | {row['n_total']} | {row['n_fire']} | {row['fire_rate']:.1%} | "
                f"{_fmt_plain(float(row['filter_expr']),5)} | {_fmt_plain(float(row['uniform_expr']),5)} | "
                f"{_fmt_plain(float(row['filter_sr']),4)} | {_fmt_plain(float(row['uniform_sr']),4)} |"
            )
        lines.append("")

        lines.append("### T0 Tautology Pre-Screen (Val-IS, Per Session)")
        lines.append("")
        lines.append(
            "Run exactly against the 5 pre-committed filter keys named in the YAML. "
            "Flag threshold is `|corr| > 0.70`."
        )
        lines.append("")
        lines.append("| Session | Filter | |corr| | Flag |")
        lines.append("|---|---|---:|---|")
        for row in taut_df.to_dict("records"):
            lines.append(
                f"| {row['session']} | {row['filter_key']} | {_fmt_plain(float(row['corr_abs']),3)} | "
                f"{'TAUTOLOGY' if bool(row['flag']) else 'OK'} |"
            )
        lines.append("")


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        max_data_day = con.execute(
            """
            SELECT MAX(trading_day)
            FROM orb_outcomes
            WHERE orb_minutes = 5
              AND entry_model = 'E2'
              AND confirm_bars = 1
              AND rr_target = 1.5
              AND pnl_r IS NOT NULL
            """
        ).fetchone()[0]

        instrument_payload: dict[str, dict[str, object]] = {}
        for instrument in INSTRUMENTS:
            full = _load_instrument(con, instrument)
            thresholds, train_df = _train_thresholds(full)
            kept = _apply_thresholds(full, thresholds)

            val_df = _window_slice(kept, VAL_START, VAL_END)
            semi_df = _window_slice(kept, SEMI_OOS_START, SEMI_OOS_END)
            fresh_df = _window_slice(kept, FRESH_OOS_START, None)

            form_results = tuple(_evaluate_form(val_df, semi_df, fresh_df, form) for form in FORMS)
            winner_status, winner = _winner(form_results)

            instrument_payload[instrument] = {
                "full": full,
                "train": train_df,
                "kept": kept,
                "val": val_df,
                "semi": semi_df,
                "fresh": fresh_df,
                "results": form_results,
                "winner_status": winner_status,
                "winner": winner,
            }
    finally:
        con.close()

    lines: list[str] = []
    lines.append("# rel_vol Filter Form v1\n")
    lines.append(f"**Pre-reg:** `{PREREG_PATH}`  ")
    lines.append("**Script:** `research/rel_vol_filter_form_v1.py`  ")
    lines.append("**Scope:** `MGC` + `MES`, `E2 / CB1 / O5 / RR1.5`, rel_vol lineage only.\n")
    lines.append("## What Was Actually Tested")
    lines.append("")
    lines.append(
        "Per instrument, thresholds were trained on `trading_day < 2024-01-01` per `(session, direction)` lane, "
        "with lanes below `N_train < 100` dropped entirely. The two locked forms were then evaluated on the "
        "true validation window `2024-01-01 .. 2025-12-31` only."
    )
    lines.append("")
    lines.append("- `F1_Q5_only`: trade only quintile 5.")
    lines.append("- `F2_Q4_plus_Q5`: trade only quintiles 4 and 5.")
    lines.append(
        "- Filter ExpR and filter Sharpe are computed on fired trades only; uniform baseline is computed on the same "
        "kept-lane universe, following the repo-standard on/off filter evaluation convention."
    )
    lines.append("- Gate 4 CI is a moving-block bootstrap of `filter Sharpe - uniform Sharpe` with `block_size=20`, `B=10000`.")
    lines.append("- Semi-OOS `2026-01-01 .. 2026-04-19` is reported as informational only; it is not used for gating.")
    lines.append("")
    lines.append("## Structural Veto")
    lines.append("")
    lines.append(
        f"{EXECUTION_VETO_REASON} This means the filter-form result below is a valid conditional research result, "
        "but it is **not** a deployable `E2` pre-entry filter in its current framing."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| Instrument | Winner Status | Winning Form | E2 Execution | Notes |")
    lines.append("|---|---|---|---|---|")
    for instrument in INSTRUMENTS:
        payload = instrument_payload[instrument]
        winner_status = payload["winner_status"]
        winner = payload["winner"]
        notes = "No form cleared Gates 1-3."
        winner_key = "NA"
        if winner is not None:
            winner_key = winner.form.key
            notes = (
                f"Val filter SR={_fmt_plain(winner.val.filter_sr,4)}, uniform SR={_fmt_plain(winner.val.uniform_sr,4)}, "
                f"fire rate={winner.val.fire_rate:.1%}."
            )
        lines.append(f"| {instrument} | **{winner_status}** | {winner_key} | **VETO** | {notes} |")
    lines.append("")

    lines.append(
        f"Max canonical data day available while running this script: `{max_data_day}`. "
        f"Fresh OOS window begins at `{FRESH_OOS_START.date()}`, so fresh-OOS accrual remains zero in this repo state."
    )
    lines.append("")

    for instrument in INSTRUMENTS:
        payload = instrument_payload[instrument]
        _emit_instrument_section(
            lines=lines,
            instrument=instrument,
            form_results=payload["results"],
            winner_status=payload["winner_status"],
            winner=payload["winner"],
            full_kept=payload["kept"],
            val_df=payload["val"],
            semi_df=payload["semi"],
            fresh_df=payload["fresh"],
        )

    lines.append("## Final Call")
    lines.append("")
    for instrument in INSTRUMENTS:
        payload = instrument_payload[instrument]
        winner_status = payload["winner_status"]
        winner = payload["winner"]
        if winner_status == "CANDIDATE_READY_IS" and winner is not None:
            lines.append(
                f"- `{instrument}`: `{winner.form.key}` is `CANDIDATE_READY_IS` mathematically, but `NOT_DEPLOYABLE_AS_E2_FILTER` "
                f"because `rel_vol` is break-bar-volume based. Fresh OOS confirmation is therefore moot unless the execution framing changes."
            )
        elif winner_status == "FILTER_RESEARCH_SURVIVOR" and winner is not None:
            lines.append(
                f"- `{instrument}`: winner is `{winner.form.key}`, but Gate 4 failed and the `E2` execution veto still applies."
            )
        else:
            lines.append(f"- `{instrument}`: `FILTER_DEAD_IS` under the locked form-space.")
    lines.append("")
    lines.append(
        "- Highest-EV next action: if this lineage stays open, reframe it as an execution-safe post-break role "
        "(for example an entry-model-switch or confirmation model that resolves after the break bar), not as an `E2` pre-entry filter."
    )
    lines.append("")

    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    RESULT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {RESULT_DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
