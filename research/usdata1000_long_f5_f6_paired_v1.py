#!/usr/bin/env python3
"""Focused paired verify for MNQ US_DATA_1000 long F5/F6 context states.

Locked by:
  docs/audit/hypotheses/2026-04-20-usdata1000-long-f5-f6-paired-v1.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from research.lib import bh_fdr, connect_db
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

INSTRUMENT = "MNQ"
SESSION = "US_DATA_1000"
RR_TARGETS = (1.0, 1.5)
BH_Q = 0.05
MIN_N = 100
T_ABS_MIN = 3.0
DELTA_ABS_MIN = 0.05
YEAR_MIN_GROUP = 10
OOS_MIN_GROUP = 30
HOLDOUT = pd.Timestamp(HOLDOUT_SACRED_FROM).date()

OUTPUT_DIR = Path("research/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "usdata1000_long_f5_f6_paired_v1.csv"
OUTPUT_MD = Path("docs/audit/results/2026-04-20-usdata1000-long-f5-f6-paired-v1.md")


@dataclass(frozen=True)
class HypothesisSpec:
    rr_target: float
    feature: str
    role: str
    expected_sign: int


HYPOTHESES = (
    HypothesisSpec(rr_target=1.0, feature="F5_BELOW_PDL", role="TAKE", expected_sign=1),
    HypothesisSpec(rr_target=1.5, feature="F5_BELOW_PDL", role="TAKE", expected_sign=1),
    HypothesisSpec(rr_target=1.0, feature="F6_INSIDE_PDR", role="AVOID", expected_sign=-1),
    HypothesisSpec(rr_target=1.5, feature="F6_INSIDE_PDR", role="AVOID", expected_sign=-1),
)


def _load_df() -> pd.DataFrame:
    sql = f"""
    SELECT
        o.trading_day,
        o.entry_price,
        o.stop_price,
        o.pnl_r,
        o.rr_target,
        d.atr_20,
        d.prev_day_high,
        d.prev_day_low,
        d.orb_{SESSION}_high AS orb_high,
        d.orb_{SESSION}_low AS orb_low
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = '{INSTRUMENT}'
      AND o.orb_label = '{SESSION}'
      AND o.orb_minutes = 5
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target IN ({", ".join(str(rr) for rr in RR_TARGETS)})
      AND o.pnl_r IS NOT NULL
      AND d.atr_20 IS NOT NULL
      AND d.atr_20 > 0
      AND d.prev_day_high IS NOT NULL
      AND d.prev_day_low IS NOT NULL
      AND d.orb_{SESSION}_high IS NOT NULL
      AND d.orb_{SESSION}_low IS NOT NULL
    ORDER BY o.trading_day
    """
    with connect_db() as con:
        df = con.execute(sql).fetchdf()

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["direction"] = np.where(df["entry_price"] > df["stop_price"], "long", "short")
    df = df[df["direction"] == "long"].copy()
    df["is_is"] = df["trading_day"].dt.date < HOLDOUT
    df["is_oos"] = ~df["is_is"]
    df["year"] = df["trading_day"].dt.year

    orb_mid = (df["orb_high"].astype(float) + df["orb_low"].astype(float)) / 2.0
    pdh = df["prev_day_high"].astype(float)
    pdl = df["prev_day_low"].astype(float)
    df["F5_BELOW_PDL"] = orb_mid < pdl
    df["F6_INSIDE_PDR"] = (orb_mid > pdl) & (orb_mid < pdh)
    return df


def _welch(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    res = stats.ttest_ind(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_var=False)
    return float(np.asarray(res.statistic)), float(np.asarray(res.pvalue))


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def _year_consistency(df_is: pd.DataFrame, feature: str, expected_sign: int) -> tuple[int, int]:
    eligible = 0
    consistent = 0
    for _, year_df in df_is.groupby("year"):
        on = year_df.loc[year_df[feature], "pnl_r"]
        off = year_df.loc[~year_df[feature], "pnl_r"]
        if len(on) < YEAR_MIN_GROUP or len(off) < YEAR_MIN_GROUP:
            continue
        eligible += 1
        delta = float(on.mean() - off.mean())
        if np.sign(delta) == expected_sign:
            consistent += 1
    return eligible, consistent


def _evaluate(df: pd.DataFrame, spec: HypothesisSpec) -> dict[str, object]:
    cell = df[df["rr_target"] == spec.rr_target].copy()
    is_df = cell[cell["is_is"]].copy()
    oos_df = cell[cell["is_oos"]].copy()

    on_is = is_df.loc[is_df[spec.feature], "pnl_r"]
    off_is = is_df.loc[~is_df[spec.feature], "pnl_r"]
    on_oos = oos_df.loc[oos_df[spec.feature], "pnl_r"]
    off_oos = oos_df.loc[~oos_df[spec.feature], "pnl_r"]

    exp_on_is = _safe_mean(on_is)
    exp_off_is = _safe_mean(off_is)
    delta_is = exp_on_is - exp_off_is
    t_is, p_is = _welch(on_is, off_is)

    exp_on_oos = _safe_mean(on_oos)
    exp_off_oos = _safe_mean(off_oos)
    delta_oos = exp_on_oos - exp_off_oos

    if len(on_oos) >= OOS_MIN_GROUP and len(off_oos) >= OOS_MIN_GROUP:
        oos_dir_match: bool | None = bool(np.sign(delta_oos) == spec.expected_sign)
    else:
        oos_dir_match = None

    eligible_years, years_consistent = _year_consistency(is_df, spec.feature, spec.expected_sign)
    role_coherent = (
        (spec.role == "TAKE" and delta_is >= DELTA_ABS_MIN and exp_on_is > 0)
        or (spec.role == "AVOID" and delta_is <= -DELTA_ABS_MIN and exp_off_is > 0)
    )

    return {
        "instrument": INSTRUMENT,
        "session": SESSION,
        "rr_target": spec.rr_target,
        "direction": "long",
        "feature": spec.feature,
        "role": spec.role,
        "expected_sign": spec.expected_sign,
        "n_is": int(len(is_df)),
        "n_on_is": int(len(on_is)),
        "n_off_is": int(len(off_is)),
        "exp_on_is": exp_on_is,
        "exp_off_is": exp_off_is,
        "delta_is": delta_is,
        "t_is": t_is,
        "p_is": p_is,
        "n_oos": int(len(oos_df)),
        "n_on_oos": int(len(on_oos)),
        "n_off_oos": int(len(off_oos)),
        "exp_on_oos": exp_on_oos,
        "exp_off_oos": exp_off_oos,
        "delta_oos": delta_oos,
        "oos_dir_match": oos_dir_match,
        "eligible_years": eligible_years,
        "years_consistent": years_consistent,
        "role_coherent": role_coherent,
    }


def _format_float(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    return str(value)


def main() -> None:
    df = _load_df()
    result_df = pd.DataFrame([_evaluate(df, spec) for spec in HYPOTHESES])

    pvals = result_df["p_is"].fillna(1.0).tolist()
    survivors = bh_fdr(pvals, q=BH_Q)
    ranked = sorted(enumerate(pvals), key=lambda item: item[1])
    q_values = [1.0] * len(pvals)
    running = 1.0
    for rank in range(len(ranked) - 1, -1, -1):
        idx, p_val = ranked[rank]
        q_val = min(running, p_val * len(ranked) / (rank + 1))
        running = q_val
        q_values[idx] = q_val
    result_df["q_family"] = q_values
    result_df["bh_survivor"] = [idx in survivors for idx in range(len(result_df))]

    result_df["passes_primary"] = (
        result_df["bh_survivor"]
        & (result_df["q_family"] < BH_Q)
        & (result_df["n_on_is"] >= MIN_N)
        & (result_df["n_off_is"] >= MIN_N)
        & result_df["role_coherent"]
        & (result_df["t_is"].abs() >= T_ABS_MIN)
        & (
            result_df["oos_dir_match"].isna()
            | result_df["oos_dir_match"].astype(bool)
        )
    )

    unique_roles_alive = int(result_df["passes_primary"].sum())
    family_verdict = "ALIVE" if unique_roles_alive >= 2 else "CONDITIONAL" if unique_roles_alive == 1 else "DEAD"

    result_df.to_csv(OUTPUT_CSV, index=False)

    lines = [
        "# US_DATA_1000 Long F5/F6 Paired Verify V1",
        "",
        "**Pre-reg:** `docs/audit/hypotheses/2026-04-20-usdata1000-long-f5-f6-paired-v1.yaml`",
        f"**Family verdict:** **{family_verdict}**",
        "",
        "## Scope",
        "",
        "- Instrument: MNQ",
        "- Session: US_DATA_1000",
        "- Direction: long",
        "- Aperture: O5",
        "- Entry model: E2",
        "- Confirm bars: CB1",
        "- RR targets: 1.0, 1.5",
        "- Hypotheses: F5_BELOW_PDL TAKE; F6_INSIDE_PDR AVOID",
        "",
        "## Resource grounding",
        "",
        "- `resources/Algorithmic_Trading_Chan.pdf`: bounded executable strategy families are valid research units when rules are explicit and objective.",
        "- `resources/Robert Carver - Systematic Trading.pdf`: useful signals can live as TAKE/AVOID conditioners rather than being forced into standalone systems.",
        "- `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf`: theory-first, small-family testing over broad post-hoc fishing.",
        "- `resources/Two_Million_Trading_Strategies_FDR.pdf`: honest family-level FDR remains mandatory even on small bounded families.",
        "",
        "## Family accounting",
        "",
        f"- Locked K: {len(HYPOTHESES)}",
        f"- Primary survivors: {unique_roles_alive}",
        "",
        "## Cell results",
        "",
    ]

    for _, row in result_df.sort_values(["feature", "rr_target"]).iterrows():
        lines.append(
            "- "
            f"RR{row['rr_target']:.1f} {row['feature']} {row['role']}: "
            f"IS on/off ExpR {_format_float(row['exp_on_is'])}/{_format_float(row['exp_off_is'])} | "
            f"delta {_format_float(row['delta_is'])} | "
            f"t={_format_float(row['t_is'], 2)} | q={_format_float(row['q_family'])} | "
            f"N_on/N_off={int(row['n_on_is'])}/{int(row['n_off_is'])} | "
            f"OOS delta={_format_float(row['delta_oos'])} | "
            f"dir_match={row['oos_dir_match']} | "
            f"verdict={'PASS' if bool(row['passes_primary']) else 'FAIL'}"
        )

    survivors_df = result_df[result_df["passes_primary"]].copy()
    fails_df = result_df[~result_df["passes_primary"]].copy()

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if not survivors_df.empty:
        for _, row in survivors_df.sort_values(["feature", "rr_target"]).iterrows():
            if row["role"] == "TAKE":
                lines.append(
                    "- "
                    f"{row['feature']} remains a TAKE state at RR{row['rr_target']:.1f}: "
                    f"washed-out below-PDL opens improve long trade quality."
                )
            else:
                lines.append(
                    "- "
                    f"{row['feature']} remains an AVOID state at RR{row['rr_target']:.1f}: "
                    f"inside-range opens continue to degrade long trade quality."
                )
    if not fails_df.empty:
        for _, row in fails_df.sort_values(["feature", "rr_target"]).iterrows():
            lines.append(
                "- "
                f"RR{row['rr_target']:.1f} {row['feature']} did not clear the bounded family gate."
            )

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is a context-family verify, not a standalone strategy claim.",
            "- 2026 OOS remains descriptive when a cell has fewer than 30 on-signal or 30 off-signal observations.",
            "- No capital, sizing, allocator, or live-filter action is authorized by this result alone.",
            "",
            "## Artefacts",
            "",
            f"- CSV: `{OUTPUT_CSV}`",
            f"- Script: `research/{Path(__file__).name}`",
        ]
    )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Family verdict={family_verdict} | primary_survivors={unique_roles_alive}")


if __name__ == "__main__":
    main()
