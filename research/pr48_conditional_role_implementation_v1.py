"""PR48 conditional-role implementation study v1.

Canonical inputs only:
- daily_features
- orb_outcomes

Bounded scope:
- instruments: MNQ, MES, MGC
- orb_minutes: 5
- entry_model: E2
- confirm_bars: 1
- rr_target: 1.5

Frozen-on-IS roles:
- parent
- q45_filter
- q5_filter
- continuous quintile sizer (0.5, 0.75, 1.0, 1.25, 1.5)

Outputs a markdown result doc for the role-aware implementation question.
No capital action.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.hypothesis_loader import check_mode_a_consistency, load_hypothesis_metadata

ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = ROOT / "docs" / "audit" / "hypotheses" / "2026-04-22-pr48-conditional-role-implementation-v1.yaml"
RESULT_DOC = ROOT / "docs" / "audit" / "results" / "2026-04-22-pr48-conditional-role-implementation-v1.md"

INSTRUMENTS = ("MNQ", "MES", "MGC")
ORB_MINUTES = 5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR_TARGET = 1.5
CONTINUOUS_WEIGHTS = {1: 0.5, 2: 0.75, 3: 1.0, 4: 1.25, 5: 1.5}


@dataclass(frozen=True)
class RoleSummary:
    name: str
    selected_n: int
    trade_share: float
    selected_avg_r: float
    policy_ev_per_opp: float
    avg_weight: float
    capital_normalized_ev: float


def _load_prereg_meta() -> tuple[dict, str]:
    meta = load_hypothesis_metadata(PREREG_PATH)
    check_mode_a_consistency(meta)
    body = yaml.safe_load(PREREG_PATH.read_text(encoding="utf-8"))
    commit_sha = str(body.get("metadata", {}).get("commit_sha", "UNSTAMPED"))
    return meta, commit_sha


def _list_sessions(con: duckdb.DuckDBPyConnection, symbol: str) -> list[str]:
    rows = con.execute(
        """
        SELECT DISTINCT orb_label FROM orb_outcomes
        WHERE symbol = ? AND orb_minutes = ? AND pnl_r IS NOT NULL
        ORDER BY orb_label
        """,
        [symbol, ORB_MINUTES],
    ).fetchall()
    return [r[0] for r in rows]


def _load_symbol(con: duckdb.DuckDBPyConnection, symbol: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for session in _list_sessions(con, symbol):
        rel_col = f"rel_vol_{session}"
        sql = f"""
        WITH df AS (
          SELECT d.trading_day, d.symbol, d.{rel_col} AS rel_vol
          FROM daily_features d
          WHERE d.symbol = '{symbol}' AND d.orb_minutes = {ORB_MINUTES}
        )
        SELECT o.trading_day, o.pnl_r, o.entry_price, o.stop_price, o.orb_label, df.rel_vol
        FROM orb_outcomes o
        JOIN df ON o.trading_day = df.trading_day AND o.symbol = df.symbol
        WHERE o.symbol = '{symbol}'
          AND o.orb_label = '{session}'
          AND o.orb_minutes = {ORB_MINUTES}
          AND o.entry_model = '{ENTRY_MODEL}'
          AND o.confirm_bars = {CONFIRM_BARS}
          AND o.rr_target = {RR_TARGET}
          AND o.pnl_r IS NOT NULL
        """
        sub = con.sql(sql).to_df()
        if sub.empty:
            continue
        sub["direction"] = np.where(sub["entry_price"] > sub["stop_price"], "long", "short")
        sub["lane"] = sub["orb_label"] + "_" + sub["direction"]
        sub["trading_day"] = pd.to_datetime(sub["trading_day"])
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["rel_vol", "pnl_r"]).reset_index(drop=True)


def _assign_is_quintiles(df: pd.DataFrame) -> pd.DataFrame:
    holdout = pd.Timestamp(HOLDOUT_SACRED_FROM)
    out = df.copy()
    is_df = out[out["trading_day"] < holdout]
    cuts: dict[str, np.ndarray] = {}
    for lane, g in is_df.groupby("lane"):
        cuts[lane] = np.quantile(g["rel_vol"].astype(float), [0.2, 0.4, 0.6, 0.8])

    def assign(row: pd.Series) -> int:
        q = cuts[row["lane"]]
        value = float(row["rel_vol"])
        if value <= q[0]:
            return 1
        if value <= q[1]:
            return 2
        if value <= q[2]:
            return 3
        if value <= q[3]:
            return 4
        return 5

    out["q_is"] = out.apply(assign, axis=1)
    out["w_parent"] = 1.0
    out["w_q45"] = np.where(out["q_is"] >= 4, 1.0, 0.0)
    out["w_q5"] = np.where(out["q_is"] == 5, 1.0, 0.0)
    out["w_cont"] = out["q_is"].map(CONTINUOUS_WEIGHTS).astype(float)
    return out


def _rank_slope(df: pd.DataFrame) -> tuple[int, float, float]:
    ranked = df.copy()
    ranked["rank_rel_vol"] = (
        ranked.groupby("lane")["rel_vol"].rank(method="average").div(ranked.groupby("lane")["lane"].transform("count"))
    )
    X = ranked[["rank_rel_vol"]].astype(float).copy()
    if ranked["lane"].nunique() > 1:
        X = pd.concat([X, pd.get_dummies(ranked["lane"], drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(ranked["pnl_r"].astype(float), X).fit(cov_type="HC3")
    return len(ranked), float(model.params["rank_rel_vol"]), float(model.tvalues["rank_rel_vol"])


def _summarize_role(df: pd.DataFrame, weight_col: str, name: str) -> RoleSummary:
    weights = df[weight_col].astype(float)
    selected = df.loc[weights > 0].copy()
    weighted_pnl = weights * df["pnl_r"].astype(float)
    avg_weight = float(weights.mean())
    capital_normalized = float(weighted_pnl.mean() / avg_weight) if avg_weight > 0 else float("nan")
    selected_mean = float(selected["pnl_r"].mean()) if not selected.empty else float("nan")
    return RoleSummary(
        name=name,
        selected_n=int((weights > 0).sum()),
        trade_share=float((weights > 0).mean()),
        selected_avg_r=selected_mean,
        policy_ev_per_opp=float(weighted_pnl.mean()),
        avg_weight=avg_weight,
        capital_normalized_ev=capital_normalized,
    )


def _quintile_means(df: pd.DataFrame) -> dict[int, tuple[int, float]]:
    grouped = df.groupby("q_is")["pnl_r"].agg(["count", "mean"]).reset_index()
    return {int(row["q_is"]): (int(row["count"]), float(row["mean"])) for _, row in grouped.iterrows()}


def _render_role_table(rows: list[RoleSummary]) -> list[str]:
    lines = [
        "| role | selected_n | trade_share | selected_avg_r | policy_ev_per_opp | avg_weight | capital_normalized_ev |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.name} | {row.selected_n} | {row.trade_share:.3f} | "
            f"{row.selected_avg_r:+.4f} | {row.policy_ev_per_opp:+.4f} | "
            f"{row.avg_weight:.3f} | {row.capital_normalized_ev:+.4f} |"
        )
    return lines


def main() -> int:
    prereg_meta, prereg_sha = _load_prereg_meta()
    holdout = pd.Timestamp(prereg_meta["holdout_date"])
    parts: list[str] = []
    parts.append("# PR48 conditional-role implementation v1\n")
    parts.append(f"**Pre-reg:** `{PREREG_PATH.relative_to(ROOT)}`")
    parts.append(f"**Pre-reg commit SHA:** `{prereg_sha}`")
    parts.append("**Canonical layers:** `daily_features`, `orb_outcomes`")
    parts.append(
        f"**Scope:** `{', '.join(INSTRUMENTS)}` x O{ORB_MINUTES} x {ENTRY_MODEL} x CB{CONFIRM_BARS} x RR{RR_TARGET} with IS-frozen role rules."
    )
    parts.append(f"**Sacred OOS window:** `{holdout.date().isoformat()}` onward (monitor only; thin window).")
    parts.append("")

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    latest_day = con.execute("SELECT MAX(trading_day) FROM orb_outcomes WHERE pnl_r IS NOT NULL").fetchone()[0]
    parts.append(f"**Latest canonical trading day in orb_outcomes:** `{latest_day}`")
    parts.append("")

    try:
        for symbol in INSTRUMENTS:
            raw = _load_symbol(con, symbol)
            raw = _assign_is_quintiles(raw)
            is_df = raw[raw["trading_day"] < holdout].copy()
            oos_df = raw[raw["trading_day"] >= holdout].copy()

            parts.append(f"## {symbol}")
            parts.append("")
            parts.append(
                f"- Range: `{raw['trading_day'].min().date()}` to `{raw['trading_day'].max().date()}`; "
                f"total N={len(raw)}, IS N={len(is_df)}, OOS N={len(oos_df)}, lanes={raw['lane'].nunique()}"
            )
            parts.append("")

            for era_name, era_df in (("IS", is_df), ("OOS", oos_df)):
                n_rank, beta, t_val = _rank_slope(era_df)
                roles = [
                    _summarize_role(era_df, "w_parent", "parent"),
                    _summarize_role(era_df, "w_q45", "q45_filter"),
                    _summarize_role(era_df, "w_q5", "q5_filter"),
                    _summarize_role(era_df, "w_cont", "continuous_sizer"),
                ]
                qmeans = _quintile_means(era_df)
                q5_minus_q1 = qmeans[5][1] - qmeans[1][1]
                parts.append(f"### {era_name}")
                parts.append("")
                parts.append(f"- Rank slope: N={n_rank}, beta={beta:+.5f}, t={t_val:+.3f}")
                parts.extend(_render_role_table(roles))
                parts.append("")
                parts.append(
                    "- Quintiles: "
                    + ", ".join(f"Q{q}: N={qmeans[q][0]}, avg={qmeans[q][1]:+.4f}" for q in sorted(qmeans))
                )
                parts.append(f"- Q5 minus Q1 mean spread: {q5_minus_q1:+.4f}R")
                parts.append("")
    finally:
        con.close()

    parts.append("## Interpretation guardrails")
    parts.append("")
    parts.append("- `selected_avg_r` is not enough. Conditional roles are judged on `policy_ev_per_opp` first.")
    parts.append(
        "- `capital_normalized_ev` is reported for the continuous sizer so a weight map is not mistaken for a binary filter."
    )
    parts.append(
        "- OOS from 2026-01-01 to latest canonical day is monitoring only; use it for direction and implementation sanity, not retuning."
    )
    parts.append("")

    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    RESULT_DOC.write_text("\n".join(parts), encoding="utf-8")
    print(f"WROTE {RESULT_DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
