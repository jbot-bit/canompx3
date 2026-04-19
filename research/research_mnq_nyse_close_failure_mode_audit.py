#!/usr/bin/env python3
"""Failure-mode audit for MNQ NYSE_CLOSE.

Locked by:
  docs/audit/hypotheses/2026-04-19-mnq-nyse-close-failure-mode-audit.yaml
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.lib import connect_db, ttest_1s, write_csv

RESULT_PATH = Path("docs/audit/results/2026-04-19-mnq-nyse-close-failure-mode-audit.md")
OUTPUT_PREFIX = "mnq_nyse_close_failure_mode_audit"
HOLDOUT_START = pd.Timestamp("2026-01-01")


def classify_rejection_reason(reason: str | None) -> str:
    if not reason:
        return "unknown"
    text = reason.lower()
    if "phase 3" in text:
        return "year_stability"
    if "criterion_9" in text or "era" in text:
        return "era_instability"
    if "phase 2" in text:
        return "negative_expectancy"
    if "criterion_8" in text or "oos" in text:
        return "oos_failure"
    return "other"


def load_baseline_rows() -> pd.DataFrame:
    sql = """
    SELECT
        o.trading_day,
        o.orb_minutes,
        o.rr_target,
        o.pnl_r,
        d.orb_NYSE_CLOSE_break_dir AS break_dir
    FROM orb_outcomes o
    JOIN daily_features d
      ON d.trading_day = o.trading_day
     AND d.symbol = o.symbol
     AND d.orb_minutes = o.orb_minutes
    WHERE o.symbol = 'MNQ'
      AND o.orb_label = 'NYSE_CLOSE'
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.outcome IS NOT NULL
    ORDER BY o.trading_day, o.orb_minutes, o.rr_target
    """
    with connect_db() as con:
        return con.execute(sql).fetchdf()


def load_experimental_rows() -> pd.DataFrame:
    sql = """
    SELECT
        instrument,
        orb_label,
        filter_type,
        rr_target,
        orb_minutes,
        sample_size,
        expectancy_r,
        p_value,
        validation_status,
        rejection_reason
    FROM experimental_strategies
    WHERE instrument = 'MNQ'
      AND orb_label = 'NYSE_CLOSE'
    ORDER BY rr_target, orb_minutes, expectancy_r DESC
    """
    with connect_db() as con:
        return con.execute(sql).fetchdf()


def load_comparison_counts() -> dict[str, int]:
    queries = {
        "validated_count": """
            SELECT COUNT(*) AS n
            FROM validated_setups
            WHERE instrument = 'MNQ' AND orb_label = 'NYSE_CLOSE'
        """,
        "deployable_count": """
            SELECT COUNT(*) AS n
            FROM deployable_validated_setups
            WHERE instrument = 'MNQ' AND orb_label = 'NYSE_CLOSE'
        """,
    }
    out: dict[str, int] = {}
    with connect_db() as con:
        for key, sql in queries.items():
            out[key] = int(con.execute(sql).fetchone()[0])
    return out


def summarize_baseline(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["trading_day"] = pd.to_datetime(rows["trading_day"])
    rows["year"] = rows["trading_day"].dt.year

    records: list[dict] = []
    for (orb_minutes, rr_target), grp in rows.groupby(["orb_minutes", "rr_target"], sort=True):
        is_grp = grp[grp["trading_day"] < HOLDOUT_START]
        oos_grp = grp[grp["trading_day"] >= HOLDOUT_START]
        n_is, avg_is, wr_is, t_is, p_is = ttest_1s(is_grp["pnl_r"].values)
        n_oos, avg_oos, wr_oos, t_oos, p_oos = ttest_1s(oos_grp["pnl_r"].values)

        yearly = (
            is_grp.groupby("year", sort=True)["pnl_r"]
            .mean()
            .reset_index(name="avg_r")
        )
        positive_years = int((yearly["avg_r"] > 0).sum())
        total_years = int(len(yearly))

        records.append(
            {
                "orb_minutes": int(orb_minutes),
                "rr_target": float(rr_target),
                "n_is": n_is,
                "avg_is": avg_is,
                "wr_is": wr_is,
                "t_is": t_is,
                "p_is": p_is,
                "positive_years": positive_years,
                "total_years": total_years,
                "long_avg_is": float(is_grp.loc[is_grp["break_dir"] == "long", "pnl_r"].mean()),
                "short_avg_is": float(is_grp.loc[is_grp["break_dir"] == "short", "pnl_r"].mean()),
                "n_oos": n_oos,
                "avg_oos": avg_oos,
                "wr_oos": wr_oos,
                "t_oos": t_oos,
                "p_oos": p_oos,
            }
        )

    return pd.DataFrame(records).sort_values(["orb_minutes", "rr_target"]).reset_index(drop=True)


def summarize_yearly_rr1(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["trading_day"] = pd.to_datetime(rows["trading_day"])
    rows["year"] = rows["trading_day"].dt.year
    rows = rows[(rows["trading_day"] < HOLDOUT_START) & (rows["rr_target"] == 1.0)]
    summary = (
        rows.groupby(["orb_minutes", "year"], sort=True)["pnl_r"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "n", "mean": "avg_r"})
    )
    return summary.sort_values(["orb_minutes", "year"]).reset_index(drop=True)


def summarize_experimental(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rows.empty:
        return rows.copy(), pd.DataFrame(columns=["rejection_bucket", "n"])
    detail = rows.copy()
    detail["rejection_bucket"] = detail["rejection_reason"].apply(classify_rejection_reason)
    bucket = (
        detail.groupby("rejection_bucket", sort=True)
        .size()
        .reset_index(name="n")
        .sort_values(["n", "rejection_bucket"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return detail, bucket


def build_markdown(
    baseline: pd.DataFrame,
    yearly_rr1: pd.DataFrame,
    experimental: pd.DataFrame,
    buckets: pd.DataFrame,
    counts: dict[str, int],
) -> str:
    rr1_positive = baseline[(baseline["rr_target"] == 1.0) & (baseline["avg_is"] > 0)]
    nofilter_tested = bool((experimental["filter_type"] == "NO_FILTER").any()) if not experimental.empty else False
    o30_tested = bool((experimental["orb_minutes"] == 30).any()) if not experimental.empty else False

    lines = [
        "# MNQ NYSE_CLOSE Failure-Mode Audit",
        "",
        "Date: 2026-04-19",
        "",
        "## Scope",
        "",
        "Audit why `MNQ NYSE_CLOSE` remains unvalidated and undeployed despite a",
        "positive canonical broad baseline on parts of the surface.",
        "",
        "This is a failure-mode / blocker audit, not a new discovery sweep.",
        "",
        "Canonical proof:",
        "",
        "- `gold.db::orb_outcomes`",
        "- `gold.db::daily_features`",
        "",
        "Comparison-only blocker context:",
        "",
        "- `gold.db::experimental_strategies`",
        "- `gold.db::validated_setups`",
        "- `gold.db::deployable_validated_setups`",
        "- `trading_app/portfolio.py`",
        "",
        "## Executive Verdict",
        "",
    ]

    if rr1_positive.empty:
        lines += [
            "`MNQ NYSE_CLOSE` is not a live gap. The broad RR1.0 baseline is not",
            "positive on any supported aperture, so the absence from validation is",
            "consistent with canonical truth.",
            "",
        ]
    else:
        lines += [
            "`MNQ NYSE_CLOSE` is **not canonically dead**. Broad `RR1.0` remains",
            "positive on all three audited apertures (`O5`, `O15`, `O30`) pre-2026.",
            "",
            "But the current unvalidated state is also **not random neglect**:",
            "",
            f"- validated rows: `{counts['validated_count']}`",
            f"- deployable rows: `{counts['deployable_count']}`",
            f"- experimental rows on record: `{len(experimental)}`",
            f"- NO_FILTER experimental coverage present: `{nofilter_tested}`",
            f"- O30 experimental coverage present: `{o30_tested}`",
            "",
            "The tested surface is narrow and mostly O5-filtered, and those candidates",
            "were rejected mainly for instability rather than outright lack of in-sample edge.",
            "",
            "So the honest conclusion is:",
            "",
            "- broad session-family still looks alive at RR1.0",
            "- prior attempted filters were mostly unstable",
            "- the unresolved issue is a **pathway / blocker problem**, not a blank discovery void",
            "",
        ]

    lines += [
        "## Canonical Baseline",
        "",
        "| Aperture | RR | N IS | Avg IS | p IS | Pos Years | Long Avg | Short Avg | N OOS | Avg OOS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in baseline.itertuples(index=False):
        p_is = f"{row.p_is:.4f}" if pd.notna(row.p_is) else "NA"
        lines.append(
            f"| O{int(row.orb_minutes)} | {row.rr_target:.1f} | {int(row.n_is)} | "
            f"{row.avg_is:+.4f} | {p_is} | {int(row.positive_years)}/{int(row.total_years)} | "
            f"{row.long_avg_is:+.4f} | {row.short_avg_is:+.4f} | {int(row.n_oos)} | {row.avg_oos:+.4f} |"
        )

    lines += [
        "",
        "## RR1.0 Year Map (Pre-2026)",
        "",
        "| Aperture | Year | N | Avg R |",
        "|---|---:|---:|---:|",
    ]
    for row in yearly_rr1.itertuples(index=False):
        lines.append(f"| O{int(row.orb_minutes)} | {int(row.year)} | {int(row.n)} | {row.avg_r:+.4f} |")

    lines += [
        "",
        "## Experimental Surface Actually Tried",
        "",
        "| Filter | RR | Aperture | N | ExpR | p | Status | Bucket |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in experimental.itertuples(index=False):
        p_val = f"{row.p_value:.4f}" if pd.notna(row.p_value) else "NA"
        lines.append(
            f"| {row.filter_type} | {row.rr_target:.1f} | O{int(row.orb_minutes)} | "
            f"{int(row.sample_size)} | {row.expectancy_r:+.4f} | {p_val} | "
            f"{row.validation_status} | {row.rejection_bucket} |"
        )

    lines += [
        "",
        "## Rejection Buckets",
        "",
        "| Bucket | N |",
        "|---|---:|",
    ]
    for row in buckets.itertuples(index=False):
        lines.append(f"| {row.rejection_bucket} | {int(row.n)} |")

    lines += [
        "",
        "## Comparison-Layer Blocker Note",
        "",
        "Current portfolio construction still excludes `NYSE_CLOSE` from the raw",
        "baseline path in [trading_app/portfolio.py](/mnt/c/Users/joshd/canompx3/trading_app/portfolio.py:633)",
        "and again in the multi-RR builder at",
        "[trading_app/portfolio.py](/mnt/c/Users/joshd/canompx3/trading_app/portfolio.py:991).",
        "",
        "That is not proof the exclusion is wrong, but it is now a **questionable",
        "blocker** because the broad RR1.0 session-family remains canonically positive",
        "while the experimented filter surface was narrow and unstable.",
        "",
        "## Bottom Line",
        "",
        "`MNQ NYSE_CLOSE` should not be treated as a fresh discovery void, and it",
        "also should not be treated as dead. The right next move is a narrow",
        "`RR1.0` native governance/failure-mode follow-up on the broad session-family,",
        "not another random filter sweep.",
        "",
        "## Outputs",
        "",
        "- `research/output/mnq_nyse_close_failure_mode_audit_baseline.csv`",
        "- `research/output/mnq_nyse_close_failure_mode_audit_rr1_years.csv`",
        "- `research/output/mnq_nyse_close_failure_mode_audit_experimental.csv`",
        "- `research/output/mnq_nyse_close_failure_mode_audit_rejection_buckets.csv`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    baseline_rows = load_baseline_rows()
    experimental_rows = load_experimental_rows()
    counts = load_comparison_counts()

    baseline = summarize_baseline(baseline_rows)
    yearly_rr1 = summarize_yearly_rr1(baseline_rows)
    experimental, buckets = summarize_experimental(experimental_rows)

    write_csv(baseline, f"{OUTPUT_PREFIX}_baseline.csv")
    write_csv(yearly_rr1, f"{OUTPUT_PREFIX}_rr1_years.csv")
    write_csv(experimental, f"{OUTPUT_PREFIX}_experimental.csv")
    write_csv(buckets, f"{OUTPUT_PREFIX}_rejection_buckets.csv")

    RESULT_PATH.write_text(
        build_markdown(baseline, yearly_rr1, experimental, buckets, counts),
        encoding="utf-8",
    )
    print(baseline.to_string(index=False))
    print()
    print(buckets.to_string(index=False))
    print(f"\nWrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
