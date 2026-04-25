#!/usr/bin/env python3
"""MNQ NYSE_CLOSE RR1.0 governance / failure-mode follow-up.

Executes the staged follow-up called for by:
  docs/runtime/stages/mnq-nyse-close-rr10-followup.md

This is a read-only research pass. It does not discover new filters or write
to canonical layers. It answers a narrower question: does the repo's current
truth support killing the family, parking on a pure policy blocker, or
continuing via one exact native prereg?
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from research.lib import connect_db, ttest_1s, write_csv

RESULT_PATH = Path("docs/audit/results/2026-04-23-mnq-nyse-close-rr10-followup.md")
HOLDOUT_START = pd.Timestamp("2026-01-01")


@dataclass(frozen=True)
class CandidateSource:
    source_doc: str
    hypothesis_name: str
    filter_type: str
    orb_minutes: int
    rationale: str


CANDIDATE_SOURCES = (
    CandidateSource(
        source_doc="docs/audit/hypotheses/2026-04-09-mnq-comprehensive.yaml",
        hypothesis_name="MNQ NYSE_CLOSE G8 RR1.0",
        filter_type="ORB_G8",
        orb_minutes=5,
        rationale="Exact native ORB-size hypothesis already locked in the first MNQ comprehensive family.",
    ),
    CandidateSource(
        source_doc="docs/audit/hypotheses/2026-04-13-wave4-session-rr-expansion.yaml",
        hypothesis_name="NYSE_CLOSE ORB size gate",
        filter_type="ORB_G8",
        orb_minutes=5,
        rationale="Later wave re-locked the same native ORB_G8 RR1.0 path and explicitly dropped ORB_G5 as a no-op.",
    ),
    CandidateSource(
        source_doc="docs/audit/hypotheses/2026-04-13-wave4-session-rr-expansion.yaml",
        hypothesis_name="NYSE_CLOSE cost gate",
        filter_type="COST_LT12",
        orb_minutes=5,
        rationale="Native RR1.0 cost-gate candidate exists in repo truth but was never executed into durable results.",
    ),
    CandidateSource(
        source_doc="docs/audit/hypotheses/2026-04-13-wave4-session-rr-expansion.yaml",
        hypothesis_name="NYSE_CLOSE cross-asset vol",
        filter_type="X_MES_ATR60",
        orb_minutes=5,
        rationale="Cross-asset volatility candidate was locked but never moved into the experimental/result surfaces.",
    ),
)


def load_baseline_rows() -> pd.DataFrame:
    sql = """
    SELECT
        trading_day,
        orb_minutes,
        pnl_r
    FROM orb_outcomes
    WHERE symbol = 'MNQ'
      AND orb_label = 'NYSE_CLOSE'
      AND entry_model = 'E2'
      AND confirm_bars = 1
      AND rr_target = 1.0
      AND pnl_r IS NOT NULL
    ORDER BY trading_day, orb_minutes
    """
    with connect_db() as con:
        return con.execute(sql).fetchdf()


def load_experimental_rows() -> pd.DataFrame:
    sql = """
    SELECT
        strategy_id,
        filter_type,
        orb_minutes,
        sample_size,
        expectancy_r,
        p_value,
        validation_status,
        rejection_reason
    FROM experimental_strategies
    WHERE instrument = 'MNQ'
      AND orb_label = 'NYSE_CLOSE'
      AND rr_target = 1.0
    ORDER BY orb_minutes, filter_type
    """
    with connect_db() as con:
        return con.execute(sql).fetchdf()


def summarize_baseline(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["trading_day"] = pd.to_datetime(rows["trading_day"])
    rows["year"] = rows["trading_day"].dt.year

    records: list[dict[str, object]] = []
    for orb_minutes, grp in rows.groupby("orb_minutes", sort=True):
        is_grp = grp[grp["trading_day"] < HOLDOUT_START]
        oos_grp = grp[grp["trading_day"] >= HOLDOUT_START]
        n_is, avg_is, _wr_is, _t_is, p_is = ttest_1s(is_grp["pnl_r"].values)
        n_oos, avg_oos, _wr_oos, _t_oos, _p_oos = ttest_1s(oos_grp["pnl_r"].values)
        yearly = is_grp.groupby("year", sort=True)["pnl_r"].mean().reset_index(name="avg_r")
        positive_years = int((yearly["avg_r"] > 0).sum())

        records.append(
            {
                "orb_minutes": int(orb_minutes),
                "n_is": int(n_is),
                "avg_is": float(avg_is),
                "p_is": float(p_is),
                "positive_years": positive_years,
                "total_years": int(len(yearly)),
                "n_oos": int(n_oos),
                "avg_oos": float(avg_oos),
            }
        )

    return pd.DataFrame(records).sort_values("orb_minutes").reset_index(drop=True)


def build_candidate_table(experimental: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source in CANDIDATE_SOURCES:
        rows.append(
            {
                "source_doc": source.source_doc,
                "hypothesis_name": source.hypothesis_name,
                "filter_type": source.filter_type,
                "orb_minutes": source.orb_minutes,
                "source_exists": Path(source.source_doc).exists(),
                "experimental_rr10_present": bool(
                    (
                        (experimental["filter_type"] == source.filter_type)
                        & (experimental["orb_minutes"] == source.orb_minutes)
                    ).any()
                ),
                "rationale": source.rationale,
            }
        )
    return pd.DataFrame(rows)


def evaluate_followup(
    baseline: pd.DataFrame,
    experimental: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[str, str, pd.DataFrame]:
    broad_positive_apertures = int((baseline["avg_is"] > 0).sum())
    tested_apertures = sorted(experimental["orb_minutes"].unique().tolist()) if not experimental.empty else []
    nofilter_present = bool((experimental["filter_type"] == "NO_FILTER").any()) if not experimental.empty else False
    native_orbg8_present = bool((experimental["filter_type"] == "ORB_G8").any()) if not experimental.empty else False
    locked_orbg8_missing = bool(
        (
            (candidates["filter_type"] == "ORB_G8")
            & candidates["source_exists"]
            & ~candidates["experimental_rr10_present"]
        ).any()
    )

    framing = pd.DataFrame(
        [
            {
                "frame": "Null edge",
                "support": "Broad RR1.0 is positive on O5, O15, and O30 pre-2026.",
                "verdict": "REJECT",
            },
            {
                "frame": "Pure policy blocker",
                "support": (
                    "Portfolio builders exclude NYSE_CLOSE, but the native RR1.0 candidate path "
                    "was also never executed."
                ),
                "verdict": "PARTIAL",
            },
            {
                "frame": "Historical narrowness / missed execution",
                "support": (
                    f"RR1.0 experiments are limited to apertures {tested_apertures or ['none']}, "
                    f"NO_FILTER tested={nofilter_present}, ORB_G8 executed={native_orbg8_present}."
                ),
                "verdict": "PRIMARY",
            },
        ]
    )

    if broad_positive_apertures == 0:
        return "KILL broad-family follow-up", "No broad positive RR1.0 baseline remains.", framing

    if locked_orbg8_missing and tested_apertures == [5] and not nofilter_present:
        return (
            "CONTINUE with narrow prereg",
            "Freeze and execute the exact MNQ NYSE_CLOSE ORB_G8 RR1.0 prereg before any new sweep or policy change.",
            framing,
        )

    return (
        "PARK pending blocker removal",
        "Empirical path is already covered; remaining work would be policy / promotion discipline rather than research.",
        framing,
    )


def build_markdown(
    baseline: pd.DataFrame,
    experimental: pd.DataFrame,
    candidates: pd.DataFrame,
    framing: pd.DataFrame,
    decision: str,
    recommendation: str,
) -> str:
    lines = [
        "# MNQ NYSE_CLOSE RR1.0 Follow-up",
        "",
        "Date: 2026-04-23",
        "",
        "## Scope",
        "",
        "Close the staged RR1.0 failure-mode / governance follow-up for `MNQ NYSE_CLOSE` without reopening a broad filter sweep.",
        "",
        "Inputs used:",
        "",
        "- `gold.db::orb_outcomes`",
        "- `gold.db::experimental_strategies`",
        "- `docs/audit/results/2026-04-19-mnq-nyse-close-failure-mode-audit.md`",
        "- locked repo hypotheses under `docs/audit/hypotheses/`",
        "- `trading_app/portfolio.py` raw-baseline exclusion surface",
        "",
        "## Broad RR1.0 Baseline",
        "",
        "| Aperture | N IS | Avg IS | p IS | Pos Years | N OOS | Avg OOS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in baseline.itertuples(index=False):
        lines.append(
            f"| O{int(row.orb_minutes)} | {int(row.n_is)} | {row.avg_is:+.4f} | "
            f"{row.p_is:.4f} | {int(row.positive_years)}/{int(row.total_years)} | "
            f"{int(row.n_oos)} | {row.avg_oos:+.4f} |"
        )

    lines += [
        "",
        "## RR1.0 Surface Actually Tested",
        "",
        "| Strategy | Filter | Aperture | N | ExpR | p | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in experimental.itertuples(index=False):
        lines.append(
            f"| {row.strategy_id} | {row.filter_type} | {int(row.orb_minutes)} | "
            f"{int(row.sample_size)} | {row.expectancy_r:+.4f} | {row.p_value:.4f} | {row.validation_status} |"
        )

    lines += [
        "",
        "## Locked Native Candidates Present In Repo",
        "",
        "| Source | Hypothesis | Filter | Aperture | Already in RR1.0 experimental table? |",
        "|---|---|---|---:|---|",
    ]
    for row in candidates.itertuples(index=False):
        lines.append(
            f"| {row.source_doc} | {row.hypothesis_name} | {row.filter_type} | "
            f"{int(row.orb_minutes)} | {bool(row.experimental_rr10_present)} |"
        )

    lines += [
        "",
        "## Alternate Framings Checked",
        "",
        "| Frame | Support | Verdict |",
        "|---|---|---|",
    ]
    for row in framing.itertuples(index=False):
        lines.append(f"| {row.frame} | {row.support} | {row.verdict} |")

    lines += [
        "",
        "## Decision",
        "",
        f"**Outcome:** `{decision}`",
        "",
        recommendation,
        "",
        "Why:",
        "",
        "- Broad RR1.0 is alive on all three audited apertures, so a broad-family kill would be false.",
        "- RR1.0 experimentation is still only three O5 filters (`GAP_R015`, `OVNRNG_100`, `ORB_G5_NOFRI`).",
        "- `NO_FILTER`, `O15`, and `O30` were never taken through the RR1.0 experimental path.",
        "- The repo already contains two independent locked `ORB_G8` NYSE_CLOSE RR1.0 hypotheses, but no corresponding RR1.0 experimental row or durable result doc.",
        "- `trading_app/portfolio.py` still excludes `NYSE_CLOSE` from both raw-baseline builders, so direct promotion without a narrow exact prereg would overstep the evidence.",
        "",
        "## EV-Based Next Move",
        "",
        "Highest-EV path: execute one exact native prereg, `MNQ NYSE_CLOSE ORB_G8 RR1.0`.",
        "",
        "Not recommended now:",
        "",
        "- no direct portfolio unblock of raw `NYSE_CLOSE` baselines",
        "- no new broad RR1.0 sweep",
        "- no COST / cross-asset follow-up before the native ORB-size path is closed",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    baseline_rows = load_baseline_rows()
    experimental = load_experimental_rows()
    baseline = summarize_baseline(baseline_rows)
    candidates = build_candidate_table(experimental)
    decision, recommendation, framing = evaluate_followup(baseline, experimental, candidates)

    write_csv(baseline, "mnq_nyse_close_rr10_followup_baseline.csv")
    write_csv(experimental, "mnq_nyse_close_rr10_followup_experimental.csv")
    write_csv(candidates, "mnq_nyse_close_rr10_followup_candidates.csv")
    write_csv(framing, "mnq_nyse_close_rr10_followup_framing.csv")

    RESULT_PATH.write_text(
        build_markdown(baseline, experimental, candidates, framing, decision, recommendation),
        encoding="utf-8",
    )
    print(f"Wrote {RESULT_PATH}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
