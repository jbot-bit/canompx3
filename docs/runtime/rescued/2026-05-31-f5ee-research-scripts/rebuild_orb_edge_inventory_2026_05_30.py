#!/usr/bin/env python3
"""Mechanism-first ORB edge inventory rebuild.

Read-only canonical-layer run. Uses only orb_outcomes, daily_features, and
bars_1m-derived fields already stored in daily_features. Does not read
validated_setups, edge_families, or live_config.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import NormalDist

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import duckdb
import numpy as np
import pandas as pd

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, ASSET_CONFIGS
from pipeline.cost_model import COST_SPECS
from pipeline.paths import GOLD_DB_PATH

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - fallback for minimal local envs
    scipy_stats = None


HOLDOUT_DATE = date(2026, 1, 1)
INSTRUMENTS = ("MNQ", "MES", "MGC")
ENTRY_MODELS = ("E1", "E2")
ORB_MINUTES = (5, 15, 30)
RR_TARGETS = (1.0, 1.5, 2.0)
CONFIRM_BARS = 1
SIZE_THRESHOLDS = (0.08, 0.10, 0.12, 0.15)
OUT_DIR = Path("artifacts/research/orb_edge_inventory_2026_05_30")


@dataclass(frozen=True)
class ResultCell:
    family: str
    mechanism: str
    role: str
    variant: str
    instrument: str
    session: str
    entry_model: str
    orb_minutes: int
    rr_target: float
    k_family: int
    n_is: int
    exp_r_is: float | None
    t_stat: float | None
    p_value: float | None
    q_value: float | None
    wfe: float | None
    era_dead: bool
    n_oos: int
    exp_r_oos: float | None
    mean_cost_to_risk: float | None


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _case_expr(session_labels: list[str], suffix: str) -> str:
    parts = [f"WHEN '{s}' THEN d.{_q(f'orb_{s}_{suffix}')}" for s in session_labels]
    return "CASE o.orb_label " + " ".join(parts) + " END"


def load_rows() -> pd.DataFrame:
    sessions = sorted({s for inst in INSTRUMENTS for s in ASSET_CONFIGS[inst]["enabled_sessions"]})
    fields = {
        "orb_size": _case_expr(sessions, "size"),
        "break_dir": _case_expr(sessions, "break_dir"),
        "double_break": _case_expr(sessions, "double_break"),
        "orb_volume": _case_expr(sessions, "volume"),
    }
    sql = f"""
        SELECT
            o.trading_day,
            o.symbol AS instrument,
            o.orb_label AS session,
            o.orb_minutes,
            o.rr_target,
            o.confirm_bars,
            o.entry_model,
            o.outcome,
            CASE
                WHEN o.pnl_r IS NOT NULL THEN o.pnl_r
                WHEN lower(o.outcome) = 'scratch' THEN 0.0
                ELSE NULL
            END AS pnl_r_clean,
            o.risk_dollars,
            o.entry_ts,
            o.exit_ts,
            {fields["orb_size"]} AS orb_size,
            {fields["break_dir"]} AS break_dir,
            {fields["double_break"]} AS double_break,
            {fields["orb_volume"]} AS orb_volume,
            d.atr_vel_ratio,
            d.atr_20_pct
        FROM orb_outcomes o
        JOIN daily_features d
          ON d.symbol = o.symbol
         AND d.trading_day = o.trading_day
         AND d.orb_minutes = o.orb_minutes
        WHERE o.symbol IN {INSTRUMENTS}
          AND o.entry_model IN {ENTRY_MODELS}
          AND o.orb_minutes IN {ORB_MINUTES}
          AND o.rr_target IN {RR_TARGETS}
          AND o.confirm_bars = {CONFIRM_BARS}
          AND o.pnl_r IS NOT NULL OR (lower(o.outcome) = 'scratch')
    """
    # Parentheses around the OR branch keep the allowed-universe predicates binding.
    sql = sql.replace(
        "AND o.pnl_r IS NOT NULL OR (lower(o.outcome) = 'scratch')",
        "AND (o.pnl_r IS NOT NULL OR lower(o.outcome) = 'scratch')",
    )
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        df = con.execute(sql).fetchdf()
    finally:
        con.close()

    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    df["friction_dollars"] = df["instrument"].map(lambda x: COST_SPECS[x].total_friction)
    df["cost_to_risk"] = df["friction_dollars"] / df["risk_dollars"].replace(0, np.nan)
    df["is_holdout"] = df["trading_day"] >= HOLDOUT_DATE
    df = df[df["instrument"].isin(ACTIVE_ORB_INSTRUMENTS)].copy()

    df.sort_values(["instrument", "session", "orb_minutes", "trading_day"], inplace=True)
    group_cols = ["instrument", "session", "orb_minutes"]
    rolling = (
        df.drop_duplicates(["instrument", "session", "orb_minutes", "trading_day"])
        .sort_values(group_cols + ["trading_day"])
        .copy()
    )
    rolling["orb_volume_prior20_median"] = rolling.groupby(group_cols)["orb_volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=5).median()
    )
    rolling["orb_volume_ratio"] = rolling["orb_volume"] / rolling["orb_volume_prior20_median"].replace(0, np.nan)
    df = df.merge(
        rolling[group_cols + ["trading_day", "orb_volume_ratio"]],
        on=group_cols + ["trading_day"],
        how="left",
    )

    peer_key = ["trading_day", "session", "orb_minutes"]
    peer_base = (
        df[df["entry_model"] == "E2"]
        .drop_duplicates(peer_key + ["instrument"])
        [peer_key + ["instrument", "cost_to_risk", "atr_20_pct", "orb_volume_ratio"]]
        .copy()
    )
    peer_base["peer_cost_ok"] = peer_base["cost_to_risk"] <= 0.12
    peer_base["peer_atr_hi"] = peer_base["atr_20_pct"] >= 67
    peer_base["peer_part_hi"] = peer_base["orb_volume_ratio"] >= peer_base["orb_volume_ratio"].quantile(0.67)

    peer_rows = []
    for key_vals, grp in peer_base.groupby(peer_key):
        for inst in grp["instrument"].unique():
            peers = grp[grp["instrument"] != inst]
            peer_rows.append(
                {
                    "trading_day": key_vals[0],
                    "session": key_vals[1],
                    "orb_minutes": key_vals[2],
                    "instrument": inst,
                    "peer_cost_ok_count": int(peers["peer_cost_ok"].sum()),
                    "peer_available_count": int(len(peers)),
                    "peer_flow_ok_count": int((peers["peer_cost_ok"] & peers["peer_atr_hi"]).sum()),
                }
            )
    peer_df = pd.DataFrame(peer_rows)
    df = df.merge(peer_df, on=peer_key + ["instrument"], how="left")
    for col in ("peer_cost_ok_count", "peer_available_count", "peer_flow_ok_count"):
        df[col] = df[col].fillna(0).astype(int)
    return df


def two_t_p(mean: float, std: float, n: int) -> tuple[float | None, float | None]:
    if n < 2 or not np.isfinite(std) or std <= 0:
        return None, None
    t_stat = mean / (std / math.sqrt(n))
    if scipy_stats is not None:
        p = float(2.0 * scipy_stats.t.sf(abs(t_stat), df=n - 1))
    else:
        p = float(2.0 * (1.0 - NormalDist().cdf(abs(t_stat))))
    return float(t_stat), p


def sharpe_like(values: pd.Series) -> float | None:
    vals = values.dropna().astype(float)
    if len(vals) < 2:
        return None
    std = vals.std(ddof=1)
    if std <= 0 or not np.isfinite(std):
        return None
    return float(vals.mean() / std)


def wfe_for(sub: pd.DataFrame) -> float | None:
    is_df = sub[~sub["is_holdout"]].sort_values("trading_day")
    days = sorted(is_df["trading_day"].unique())
    if len(days) < 120:
        return None
    split_day = days[int(len(days) * 0.7)]
    train = is_df[is_df["trading_day"] < split_day]["pnl_r_clean"]
    test = is_df[is_df["trading_day"] >= split_day]["pnl_r_clean"]
    s_train = sharpe_like(train)
    s_test = sharpe_like(test)
    if s_train is None or s_train <= 0 or s_test is None:
        return None
    return float(s_test / s_train)


def era_dead(sub: pd.DataFrame) -> bool:
    is_df = sub[~sub["is_holdout"]].copy()
    if is_df.empty:
        return False

    def era(d: date) -> str:
        if d.year <= 2019:
            return "2015-2019"
        if d.year <= 2022:
            return "2020-2022"
        if d.year == 2023:
            return "2023"
        return "2024-2025"

    is_df["era"] = is_df["trading_day"].map(era)
    for _, grp in is_df.groupby("era"):
        if len(grp) >= 50 and grp["pnl_r_clean"].mean() < -0.05:
            return True
    return False


def summarize_subset(
    sub: pd.DataFrame,
    family: str,
    mechanism: str,
    role: str,
    variant: str,
    k_family: int,
) -> ResultCell | None:
    is_sub = sub[~sub["is_holdout"]]["pnl_r_clean"].dropna().astype(float)
    oos_sub = sub[sub["is_holdout"]]["pnl_r_clean"].dropna().astype(float)
    if is_sub.empty:
        return None
    mean = float(is_sub.mean())
    std = float(is_sub.std(ddof=1)) if len(is_sub) > 1 else float("nan")
    t_stat, p_value = two_t_p(mean, std, len(is_sub))
    first = sub.iloc[0]
    return ResultCell(
        family=family,
        mechanism=mechanism,
        role=role,
        variant=variant,
        instrument=str(first.instrument),
        session=str(first.session),
        entry_model=str(first.entry_model),
        orb_minutes=int(first.orb_minutes),
        rr_target=float(first.rr_target),
        k_family=k_family,
        n_is=int(len(is_sub)),
        exp_r_is=mean,
        t_stat=t_stat,
        p_value=p_value,
        q_value=None,
        wfe=wfe_for(sub),
        era_dead=era_dead(sub),
        n_oos=int(len(oos_sub)),
        exp_r_oos=float(oos_sub.mean()) if len(oos_sub) else None,
        mean_cost_to_risk=float(sub["cost_to_risk"].mean()) if sub["cost_to_risk"].notna().any() else None,
    )


def bh_adjust(cells: list[ResultCell]) -> list[ResultCell]:
    idx_ps = [(i, c.p_value) for i, c in enumerate(cells) if c.p_value is not None and np.isfinite(c.p_value)]
    if not idx_ps:
        return cells
    m = len(cells)
    ordered = sorted(idx_ps, key=lambda x: x[1])
    q_by_i: dict[int, float] = {}
    prev = 1.0
    for rank_from_end, (i, p) in enumerate(reversed(ordered), start=1):
        rank = m - rank_from_end + 1
        q = min(prev, p * m / rank)
        prev = q
        q_by_i[i] = float(min(q, 1.0))
    out = []
    for i, c in enumerate(cells):
        out.append(ResultCell(**{**c.__dict__, "q_value": q_by_i.get(i)}))
    return out


def build_cells(df: pd.DataFrame) -> list[ResultCell]:
    base_cols = ["instrument", "session", "entry_model", "orb_minutes", "rr_target"]
    groups = list(df.groupby(base_cols, dropna=False))
    k_base = len(groups)
    all_cells: list[ResultCell] = []

    for _, sub in groups:
        cell = summarize_subset(
            sub,
            "baseline_orb",
            "ORB continuation / stop cascade",
            "standalone",
            "NO_FILTER",
            k_base,
        )
        if cell:
            all_cells.append(cell)

    for threshold in SIZE_THRESHOLDS:
        fam_cells = []
        for _, sub in groups:
            fsub = sub[sub["cost_to_risk"] <= threshold]
            cell = summarize_subset(
                fsub,
                "size_friction",
                "ORB size relative to friction",
                "filter_conditioner",
                f"COST_TO_RISK_LE_{threshold:.2f}",
                k_base * len(SIZE_THRESHOLDS),
            )
            if cell:
                fam_cells.append(cell)
        all_cells.extend(fam_cells)

    # Tercile thresholds are fitted on pre-holdout only and applied unchanged.
    for feature, family, mechanism, role in [
        ("atr_vel_ratio", "volatility_state", "volatility expansion/contraction", "conditioner"),
        ("orb_volume_ratio", "session_participation", "ORB-window participation/liquidity", "conditioner"),
    ]:
        thresholds: dict[tuple, tuple[float, float]] = {}
        for key, sub in df[~df["is_holdout"]].groupby(["instrument", "session", "orb_minutes"], dropna=False):
            vals = sub[feature].dropna().astype(float)
            if len(vals) >= 90:
                thresholds[key] = (float(vals.quantile(1 / 3)), float(vals.quantile(2 / 3)))
        for tail in ("LOW_TERCILE", "HIGH_TERCILE"):
            fam_cells = []
            for _, sub in groups:
                first = sub.iloc[0]
                key = (first.instrument, first.session, first.orb_minutes)
                if key not in thresholds:
                    continue
                low, high = thresholds[key]
                if tail == "LOW_TERCILE":
                    fsub = sub[sub[feature] <= low]
                else:
                    fsub = sub[sub[feature] >= high]
                cell = summarize_subset(
                    fsub,
                    family,
                    mechanism,
                    role,
                    f"{feature}_{tail}",
                    k_base * 2,
                )
                if cell:
                    fam_cells.append(cell)
            all_cells.extend(fam_cells)

    for variant, predicate in [
        ("PEER_COST_OK_GE_1", lambda x: x["peer_cost_ok_count"] >= 1),
        ("PEER_COST_AND_ATR_OK_GE_1", lambda x: x["peer_flow_ok_count"] >= 1),
    ]:
        fam_cells = []
        for _, sub in groups:
            eligible = sub[sub["peer_available_count"] >= 1]
            fsub = eligible[predicate(eligible)]
            cell = summarize_subset(
                fsub,
                "cross_market_flow",
                "cross-market peer ORB size/volatility state",
                "confluence_allocator",
                variant,
                k_base * 2,
            )
            if cell:
                fam_cells.append(cell)
        all_cells.extend(fam_cells)

    # Apply BH within exact family+variant and then global across all tested cells.
    adjusted = []
    for _, chunk in pd.DataFrame([c.__dict__ for c in all_cells]).groupby(["family", "variant"], dropna=False):
        chunk_cells = [ResultCell(**{k: (None if pd.isna(v) else v) for k, v in row.items()}) for row in chunk.to_dict("records")]
        adjusted.extend(bh_adjust(chunk_cells))
    global_adjusted = bh_adjust(adjusted)
    return global_adjusted


def classify_cell(c: ResultCell) -> str:
    if c.n_is < 30:
        return "UNSUPPORTED_LOW_N"
    if c.exp_r_is is None or c.exp_r_is <= 0:
        return "UNSUPPORTED_NULL_OR_NEGATIVE"
    if c.q_value is None or c.q_value >= 0.05:
        return "UNSUPPORTED_BH_FAIL"
    if c.t_stat is None or c.t_stat < 3.0:
        return "RESEARCH_PROVISIONAL_TSTAT_LT3"
    if c.wfe is None or c.wfe < 0.50:
        return "RESEARCH_PROVISIONAL_WFE_FAIL"
    if c.era_dead:
        return "UNSUPPORTED_ERA_DEAD"
    if c.n_oos >= 30 and c.exp_r_oos is not None and c.exp_r_oos < 0:
        return "RESEARCH_PROVISIONAL_OOS_WARNING"
    if c.role != "standalone":
        return "RESEARCH_PROVISIONAL_ROLE_LIMITED"
    return "DEPLOYABLE_CANDIDATE_NEEDS_FORMAL_PREREG"


def make_reports(df: pd.DataFrame, cells: list[ResultCell]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cell_df = pd.DataFrame([c.__dict__ | {"classification": classify_cell(c)} for c in cells])
    cell_df.sort_values(["family", "instrument", "session", "exp_r_is"], ascending=[True, True, True, False], inplace=True)
    cell_df.to_csv(OUT_DIR / "cells.csv", index=False)

    family = (
        cell_df.groupby(["family", "mechanism", "role", "instrument", "session"], dropna=False)
        .agg(
            cells=("family", "size"),
            median_exp_r=("exp_r_is", "median"),
            mean_exp_r=("exp_r_is", "mean"),
            best_exp_r=("exp_r_is", "max"),
            bh_pass_cells=("q_value", lambda s: int((s < 0.05).sum())),
            t3_cells=("t_stat", lambda s: int((s >= 3.0).sum())),
            deployable_candidates=("classification", lambda s: int((s == "DEPLOYABLE_CANDIDATE_NEEDS_FORMAL_PREREG").sum())),
            provisional_cells=("classification", lambda s: int(s.astype(str).str.startswith("RESEARCH_PROVISIONAL").sum())),
            unsupported_cells=("classification", lambda s: int(s.astype(str).str.startswith("UNSUPPORTED").sum())),
            min_n_is=("n_is", "min"),
            max_n_is=("n_is", "max"),
            median_oos=("exp_r_oos", "median"),
        )
        .reset_index()
    )
    family.to_csv(OUT_DIR / "family_summary.csv", index=False)

    inventory = []
    for _, r in family.iterrows():
        if r.deployable_candidates > 0:
            status = "deployable-candidate"
        elif r.provisional_cells > 0 or (r.bh_pass_cells > 0 and r.t3_cells > 0):
            status = "research-provisional"
        else:
            status = "unsupported"
        inventory.append({**r.to_dict(), "status": status})
    inv_df = pd.DataFrame(inventory)
    inv_df.to_csv(OUT_DIR / "edge_inventory.csv", index=False)

    metadata = {
        "db_path": str(GOLD_DB_PATH),
        "holdout_date": HOLDOUT_DATE.isoformat(),
        "rows": int(len(df)),
        "pre_holdout_rows": int((~df["is_holdout"]).sum()),
        "holdout_rows": int(df["is_holdout"].sum()),
        "bars_1m_horizon": {},
        "trial_counts": {
            "global_cells_tested": int(len(cell_df)),
            "by_family_variant": {
                f"{family} / {variant}": int(n)
                for (family, variant), n in cell_df.groupby(["family", "variant"]).size().items()
            },
        },
        "active_orb_instruments": list(ACTIVE_ORB_INSTRUMENTS),
        "excluded": {
            "entry_models": ["E0", "E3"],
            "derived_truth_tables": ["validated_setups", "edge_families", "live_config"],
            "lookahead_features": ["rel_vol_{session}", "break_delay_min", "daily_close", "ts_* outcomes"],
        },
    }
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        for inst in INSTRUMENTS:
            metadata["bars_1m_horizon"][inst] = con.execute(
                "SELECT count(*), min(ts_utc), max(ts_utc) FROM bars_1m WHERE symbol = ?",
                [inst],
            ).fetchone()
    finally:
        con.close()
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, default=str, indent=2), encoding="utf-8")

    top_inv = inv_df.sort_values(["status", "median_exp_r"], ascending=[True, False])
    lines = [
        "# ORB Edge Inventory Rebuild - 2026-05-30",
        "",
        "Read-only canonical-layer run. Inputs: `orb_outcomes`, `daily_features`, `bars_1m` horizons only.",
        f"Holdout discipline: selection uses rows before `{HOLDOUT_DATE.isoformat()}`; 2026+ is descriptive.",
        "",
        "## Trial Counts",
        f"- Global tested cells: {len(cell_df):,}",
    ]
    for key, n in sorted(metadata["trial_counts"]["by_family_variant"].items()):
        lines.append(f"- {key}: {n:,}")
    lines.extend(["", "## Inventory By Family"])
    for status in ["deployable-candidate", "research-provisional", "unsupported"]:
        sub = top_inv[top_inv["status"] == status]
        lines.append(f"### {status}")
        if sub.empty:
            lines.append("- None")
            continue
        for _, r in sub.head(30).iterrows():
            lines.append(
                f"- {r.instrument} {r.session} {r.family}: median ExpR {r.median_exp_r:.3f}, "
                f"BH cells {int(r.bh_pass_cells)}/{int(r.cells)}, t>=3 cells {int(r.t3_cells)}, "
                f"N range {int(r.min_n_is)}-{int(r.max_n_is)}, OOS median {r.median_oos if pd.notna(r.median_oos) else 'NA'}"
            )
    lines.extend(
        [
            "",
            "## Nulls And Kills",
            "- Any family/session absent from deployable-candidate or research-provisional is unsupported in this run.",
            "- E0, E3, dead instruments, break-delay/speed, break-bar relative volume for E2, broad ML, gap/IBS/NR/EMA packages were excluded by prior NO-GO registry and not reopened.",
        ]
    )
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = load_rows()
    cells = build_cells(df)
    make_reports(df, cells)
    print(f"Wrote {OUT_DIR}")
    print(f"Rows: {len(df):,}; cells: {len(cells):,}")


if __name__ == "__main__":
    main()
