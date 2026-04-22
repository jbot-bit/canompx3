"""Read-only MNQ boundary geometry shortcut.

Scope:
- MNQ
- US_DATA_1000
- O5
- E2
- RR1.0
- long only

Computes clearance-to-PDH regimes on the fly from canonical daily_features and
orb_outcomes, then emits IS/OOS bucket stats and a row-level CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import duckdb
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = pd.Timestamp("2026-01-01")
RESULT_MD = ROOT / "docs" / "audit" / "results" / "2026-04-22-mnq-geometry-shortcut-v1.md"
RESULT_CSV = ROOT / "docs" / "audit" / "results" / "2026-04-22-mnq-geometry-shortcut-v1-rows.csv"


@dataclass(frozen=True)
class Scope:
    symbol: str = "MNQ"
    orb_label: str = "US_DATA_1000"
    orb_minutes: int = 5
    entry_model: str = "E2"
    confirm_bars: int = 1
    rr_target: float = 1.0
    break_dir: str = "long"


SCOPE = Scope()
BUCKET_ORDER = ["co_located_break", "choked", "mid_clearance", "open_air"]


def resolve_db_path() -> Path:
    local = ROOT / "gold.db"
    if local.exists():
        return local
    try:
        common_dir = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=ROOT,
            text=True,
        ).strip()
        candidate = Path(common_dir).resolve().parent / "gold.db"
        if candidate.exists():
            return candidate
    except Exception:
        pass
    raise FileNotFoundError("could not resolve canonical gold.db from worktree")


def load_scope_df(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    q = f"""
    SELECT
      o.trading_day,
      o.pnl_r,
      o.outcome,
      d.prev_day_high,
      d.prev_day_low,
      d.prev_day_close,
      d.orb_{SCOPE.orb_label}_high AS orb_high,
      d.orb_{SCOPE.orb_label}_low AS orb_low,
      d.orb_{SCOPE.orb_label}_break_dir AS break_dir
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
      AND o.pnl_r IS NOT NULL
      AND d.prev_day_high IS NOT NULL
      AND d.prev_day_low IS NOT NULL
      AND d.orb_{SCOPE.orb_label}_break_dir = ?
    ORDER BY o.trading_day
    """
    df = con.execute(
        q,
        [
            SCOPE.symbol,
            SCOPE.orb_label,
            SCOPE.orb_minutes,
            SCOPE.entry_model,
            SCOPE.confirm_bars,
            SCOPE.rr_target,
            SCOPE.break_dir,
        ],
    ).fetchdf()
    if df.empty:
        raise RuntimeError("no rows returned for scope")
    return df


def assign_bucket(clearance_r: float) -> str:
    if clearance_r <= 0.0:
        return "co_located_break"
    if clearance_r <= 1.0:
        return "choked"
    if clearance_r <= 2.0:
        return "mid_clearance"
    return "open_air"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["orb_risk"] = out["orb_high"].astype(float) - out["orb_low"].astype(float)
    out = out[out["orb_risk"] > 0].copy()
    out["clearance_r"] = (out["prev_day_high"].astype(float) - out["orb_high"].astype(float)) / out["orb_risk"]
    out["bucket"] = out["clearance_r"].map(assign_bucket)
    out["is_is"] = out["trading_day"] < HOLDOUT
    out["is_oos"] = ~out["is_is"]
    return out


def outcome_wr(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float((series == "win").mean())


def bucket_stats(df: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    return {
        "bucket": label,
        "n": int(len(df)),
        "win_rate": outcome_wr(df["outcome"]),
        "exp_r": float(df["pnl_r"].mean()) if len(df) else float("nan"),
        "total_r": float(df["pnl_r"].sum()) if len(df) else float("nan"),
    }


def compare_vs_rest(df: pd.DataFrame, bucket: str) -> tuple[float, float, float]:
    on = df[df["bucket"] == bucket]["pnl_r"].astype(float)
    off = df[df["bucket"] != bucket]["pnl_r"].astype(float)
    if len(on) < 2 or len(off) < 2:
        return float("nan"), float("nan"), float("nan")
    t_stat, p_val = stats.ttest_ind(on, off, equal_var=False)
    delta = float(on.mean() - off.mean())
    return delta, float(t_stat), float(p_val)


def render_table(rows: list[dict[str, float | int | str]]) -> str:
    lines = [
        "| bucket | N | WinRate | ExpR | Total_R |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        wr = "nan" if pd.isna(row["win_rate"]) else f"{100.0 * float(row['win_rate']):.1f}%"
        expr = "nan" if pd.isna(row["exp_r"]) else f"{float(row['exp_r']):+.4f}"
        total = "nan" if pd.isna(row["total_r"]) else f"{float(row['total_r']):+.1f}"
        lines.append(f"| {row['bucket']} | {row['n']} | {wr} | {expr} | {total} |")
    return "\n".join(lines)


def main() -> None:
    con = duckdb.connect(str(resolve_db_path()), read_only=True)
    raw = load_scope_df(con)
    df = prepare(raw)

    is_df = df[df["is_is"]].copy()
    oos_df = df[df["is_oos"]].copy()

    is_rows = [bucket_stats(is_df[is_df["bucket"] == b], b) for b in BUCKET_ORDER]
    oos_rows = [bucket_stats(oos_df[oos_df["bucket"] == b], b) for b in BUCKET_ORDER]
    baseline_is = bucket_stats(is_df, "baseline_all")
    baseline_oos = bucket_stats(oos_df, "baseline_all")

    comparisons = []
    for bucket in ["co_located_break", "choked", "open_air"]:
        delta, t_stat, p_val = compare_vs_rest(is_df, bucket)
        comparisons.append((bucket, delta, t_stat, p_val))

    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    RESULT_MD.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULT_CSV, index=False)

    lines: list[str] = []
    lines.append("# MNQ geometry shortcut — v1")
    lines.append("")
    lines.append("**Scope:** `MNQ / US_DATA_1000 / O5 / E2 / RR1.0 / CB1 / long`")
    lines.append("**Truth layers:** `orb_outcomes` + `daily_features` only")
    lines.append("**Feature:** `clearance_r = (prev_day_high - orb_high) / (orb_high - orb_low)`")
    lines.append("**Holdout split:** `2026-01-01`")
    lines.append("")
    lines.append("## IS bucket stats")
    lines.append("")
    lines.append(render_table(is_rows + [baseline_is]))
    lines.append("")
    lines.append("## OOS bucket stats")
    lines.append("")
    lines.append(render_table(oos_rows + [baseline_oos]))
    lines.append("")
    lines.append("## IS bucket-vs-rest Welch checks")
    lines.append("")
    lines.append("| bucket | delta_vs_rest | t_stat | p_value |")
    lines.append("|---|---:|---:|---:|")
    for bucket, delta, t_stat, p_val in comparisons:
        lines.append(f"| {bucket} | {delta:+.4f} | {t_stat:+.3f} | {p_val:.4f} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `co_located_break`: `clearance_r <= 0.0`")
    lines.append("- `choked`: `0.0 < clearance_r <= 1.0`")
    lines.append("- `mid_clearance`: `1.0 < clearance_r <= 2.0`")
    lines.append("- `open_air`: `clearance_r > 2.0`")
    lines.append("- Rows with `orb_risk <= 0` are excluded fail-closed.")
    lines.append(f"- Row-level audit CSV: `{RESULT_CSV.relative_to(ROOT)}`")
    lines.append("")
    lines.append("SURVIVED SCRUTINY:")
    lines.append("- read-only canonical query only")
    lines.append("- fixed holdout split")
    lines.append("- fixed geometry bins; no threshold tuning")
    lines.append("")
    lines.append("DID NOT SURVIVE:")
    lines.append("- nothing adjudicated here beyond the bounded geometry question")
    lines.append("")
    lines.append("CAVEATS:")
    lines.append("- single-lane shortcut only")
    lines.append("- Welch checks are descriptive; full family decision lives in the pre-reg framing")
    lines.append("")
    lines.append("NEXT STEPS:")
    lines.append("- if choked is materially weaker and co-located/open-air carry the lane, promote this into the first MNQ geometry family runner")

    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
