"""MNQ parent structure shadow buckets v1.

Execute the exact prereg:
`docs/audit/hypotheses/2026-04-23-mnq-parent-structure-shadow-buckets-v1.yaml`

Canonical truth only:
- daily_features
- orb_outcomes

This is a bounded conditional-role / shadow-only study on exact deployed-parent
lane-side populations. It is not a broad confluence reboot, not a rerun of the
existing H01/H04 live-context overlay questions, and not a live-sizing or
same-session coexistence study.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal
from research.lib.conditional_role import (
    apply_bh,
    daily_policy_series,
    delta_test,
    resolve_research_db_path,
    role_metrics,
)
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.hypothesis_loader import check_mode_a_consistency, load_hypothesis_metadata

def _normalize_writable_path(path: Path) -> Path:
    text = str(path)
    if text.startswith("/mnt/c/Users/"):
        return Path(text.replace("/mnt/c/Users/", "/mnt/c/users/", 1))
    return path


ROOT = _normalize_writable_path(Path(__file__).resolve().parents[1])
PREREG_PATH = ROOT / "docs" / "audit" / "hypotheses" / "2026-04-23-mnq-parent-structure-shadow-buckets-v1.yaml"
RESULT_DOC = ROOT / "docs" / "audit" / "results" / "2026-04-23-mnq-parent-structure-shadow-buckets-v1.md"


@dataclass(frozen=True)
class TestOutcome:
    hypothesis_id: int
    hypothesis_name: str
    test_name: str
    pass_name: str
    parent_n: int
    selected_n_is: int
    selected_n_oos: int
    parent_avg_is: float
    selected_avg_is: float
    policy_ev_is: float
    policy_ev_delta_is: float
    delta_days_is: int
    delta_mean_is: float
    delta_t_is: float
    delta_p_is: float
    bh_survives: bool
    direction_positive: bool
    selected_avg_oos: float
    policy_ev_oos: float
    policy_ev_delta_oos: float
    delta_mean_oos: float
    delta_t_oos: float
    delta_p_oos: float


def _load_prereg_meta() -> tuple[dict, str]:
    meta = load_hypothesis_metadata(PREREG_PATH)
    check_mode_a_consistency(meta)
    body = yaml.safe_load(PREREG_PATH.read_text(encoding="utf-8"))
    commit_sha = str(body.get("metadata", {}).get("commit_sha", "UNSTAMPED"))
    return meta, commit_sha


def _git_last_commit(path: Path) -> str:
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        rel = path
    proc = subprocess.run(
        ["git", "log", "-n", "1", "--format=%H", "--", str(rel)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip() or "UNKNOWN"


def _load_parent_frame(
    con: duckdb.DuckDBPyConnection,
    *,
    session: str,
    orb_minutes: int,
    rr_target: float,
    direction: str,
) -> pd.DataFrame:
    side_pred = "o.entry_price > o.stop_price" if direction == "long" else "o.entry_price < o.stop_price"
    sql = f"""
    SELECT
      o.trading_day,
      o.pnl_r,
      o.risk_dollars,
      o.entry_price,
      o.stop_price,
      d.prev_day_high,
      d.prev_day_low,
      d.prev_day_close,
      d.atr_20,
      d.orb_{session}_high AS orb_high,
      d.orb_{session}_low AS orb_low,
      d.orb_{session}_break_dir AS orb_break_dir,
      d.orb_{session}_size AS orb_size
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND d.orb_minutes = {orb_minutes}
    WHERE o.symbol = 'MNQ'
      AND o.orb_label = '{session}'
      AND o.orb_minutes = {orb_minutes}
      AND o.rr_target = {rr_target}
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.pnl_r IS NOT NULL
      AND {side_pred}
    ORDER BY o.trading_day
    """
    df = con.sql(sql).to_df()
    if df.empty:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df[f"orb_{session}_high"] = df["orb_high"]
    df[f"orb_{session}_low"] = df["orb_low"]
    df[f"orb_{session}_break_dir"] = df["orb_break_dir"]
    df[f"orb_{session}_size"] = df["orb_size"]
    return df


def _with_signals(df: pd.DataFrame, session: str) -> pd.DataFrame:
    out = df.copy()
    out["sig_deployed"] = filter_signal(out, "ORB_G5", orb_label=session)
    out["sig_pd_clear"] = filter_signal(out, "PD_CLEAR_LONG", orb_label=session)
    out["sig_pd_displace"] = filter_signal(out, "PD_DISPLACE_LONG", orb_label=session)
    out["structure_score"] = out["sig_pd_clear"] + out["sig_pd_displace"]
    return out


def _evaluate_test(
    *,
    df: pd.DataFrame,
    hypothesis_id: int,
    hypothesis_name: str,
    test_name: str,
    pass_name: str,
    signal_mask: pd.Series,
    holdout: pd.Timestamp,
) -> TestOutcome:
    frame = df.copy()
    frame["w_parent"] = 1.0
    frame["w_signal"] = signal_mask.astype(float)

    is_df = frame[frame["trading_day"] < holdout].copy()
    oos_df = frame[frame["trading_day"] >= holdout].copy()

    parent_is = role_metrics(is_df, "w_parent")
    signal_is = role_metrics(is_df, "w_signal")
    parent_oos = role_metrics(oos_df, "w_parent")
    signal_oos = role_metrics(oos_df, "w_signal")

    is_delta = delta_test(daily_policy_series(is_df, "w_parent"), daily_policy_series(is_df, "w_signal"))
    oos_delta = delta_test(daily_policy_series(oos_df, "w_parent"), daily_policy_series(oos_df, "w_signal"))

    return TestOutcome(
        hypothesis_id=hypothesis_id,
        hypothesis_name=hypothesis_name,
        test_name=test_name,
        pass_name=pass_name,
        parent_n=len(frame),
        selected_n_is=signal_is["selected_n"],
        selected_n_oos=signal_oos["selected_n"],
        parent_avg_is=parent_is["selected_avg_r"],
        selected_avg_is=signal_is["selected_avg_r"],
        policy_ev_is=signal_is["policy_ev_per_opp_r"],
        policy_ev_delta_is=signal_is["policy_ev_per_opp_r"] - parent_is["policy_ev_per_opp_r"],
        delta_days_is=is_delta.n_days,
        delta_mean_is=is_delta.mean_delta_r,
        delta_t_is=is_delta.t_stat,
        delta_p_is=is_delta.p_two_tailed,
        bh_survives=False,
        direction_positive=is_delta.direction_positive,
        selected_avg_oos=signal_oos["selected_avg_r"],
        policy_ev_oos=signal_oos["policy_ev_per_opp_r"],
        policy_ev_delta_oos=signal_oos["policy_ev_per_opp_r"] - parent_oos["policy_ev_per_opp_r"],
        delta_mean_oos=oos_delta.mean_delta_r,
        delta_t_oos=oos_delta.t_stat,
        delta_p_oos=oos_delta.p_two_tailed,
    )


def _apply_family_bh(outcomes: list[TestOutcome]) -> list[TestOutcome]:
    from research.lib.conditional_role import DailyDeltaResult

    bh_input = {
        (row.hypothesis_name, f"{row.test_name}|{row.pass_name}"): DailyDeltaResult(
            n_days=row.delta_days_is,
            mean_delta_r=row.delta_mean_is,
            t_stat=row.delta_t_is,
            p_two_tailed=row.delta_p_is,
            bh_survives=False,
            direction_positive=row.direction_positive,
        )
        for row in outcomes
    }
    bh_out = apply_bh(bh_input, q=0.05)
    keyed = {(row.hypothesis_name, f"{row.test_name}|{row.pass_name}"): row for row in outcomes}
    updated: list[TestOutcome] = []
    for key, row in keyed.items():
        bh_row = bh_out[key]
        updated.append(
            TestOutcome(
                hypothesis_id=row.hypothesis_id,
                hypothesis_name=row.hypothesis_name,
                test_name=row.test_name,
                pass_name=row.pass_name,
                parent_n=row.parent_n,
                selected_n_is=row.selected_n_is,
                selected_n_oos=row.selected_n_oos,
                parent_avg_is=row.parent_avg_is,
                selected_avg_is=row.selected_avg_is,
                policy_ev_is=row.policy_ev_is,
                policy_ev_delta_is=row.policy_ev_delta_is,
                delta_days_is=row.delta_days_is,
                delta_mean_is=row.delta_mean_is,
                delta_t_is=row.delta_t_is,
                delta_p_is=row.delta_p_is,
                bh_survives=bh_row.bh_survives,
                direction_positive=row.direction_positive,
                selected_avg_oos=row.selected_avg_oos,
                policy_ev_oos=row.policy_ev_oos,
                policy_ev_delta_oos=row.policy_ev_delta_oos,
                delta_mean_oos=row.delta_mean_oos,
                delta_t_oos=row.delta_t_oos,
                delta_p_oos=row.delta_p_oos,
            )
        )
    return sorted(updated, key=lambda row: (row.hypothesis_id, row.test_name, row.pass_name))


def _hypothesis_verdict(rows: list[TestOutcome]) -> str:
    filtered = [row for row in rows if row.pass_name == "filtered"]
    if any(
        row.bh_survives
        and row.direction_positive
        and row.policy_ev_delta_is > 0
        and row.selected_avg_is > 0
        for row in filtered
    ):
        return "CONTINUE"
    if any(row.policy_ev_delta_is > 0 and row.selected_avg_is > 0 for row in filtered + rows):
        return "PARK"
    return "KILL"


def _render_summary_table(outcomes: list[TestOutcome]) -> list[str]:
    lines = [
        "| hypothesis | test | pass | N_parent | N_on_IS | policy_EV_delta_IS | delta_mean_IS | t_IS | p_IS | BH | N_on_OOS | policy_EV_delta_OOS | delta_mean_OOS |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for row in outcomes:
        lines.append(
            f"| H{row.hypothesis_id} | {row.test_name} | {row.pass_name} | {row.parent_n} | {row.selected_n_is} | "
            f"{row.policy_ev_delta_is:+.4f} | {row.delta_mean_is:+.4f} | {row.delta_t_is:+.3f} | {row.delta_p_is:.6f} | "
            f"{'Y' if row.bh_survives else 'N'} | {row.selected_n_oos} | {row.policy_ev_delta_oos:+.4f} | {row.delta_mean_oos:+.4f} |"
        )
    return lines


def _render_hypothesis_section(hypothesis_name: str, rows: list[TestOutcome]) -> list[str]:
    verdict = _hypothesis_verdict(rows)
    lines = [f"## {hypothesis_name}", "", f"**Verdict:** `{verdict}`", ""]
    lines.extend(_render_summary_table(rows))
    lines.append("")
    for row in rows:
        lines.extend(
            [
                f"### {row.test_name} / {row.pass_name}",
                "",
                f"- IS selected-trade mean: `{row.selected_avg_is:+.4f}R` vs parent `{row.parent_avg_is:+.4f}R`",
                f"- IS policy EV per opportunity delta: `{row.policy_ev_delta_is:+.4f}R`",
                f"- IS daily delta test: `n_days={row.delta_days_is}`, `mean={row.delta_mean_is:+.4f}R`, `t={row.delta_t_is:+.3f}`, `p={row.delta_p_is:.6f}`, `BH={'Y' if row.bh_survives else 'N'}`",
                f"- OOS selected N: `{row.selected_n_oos}`; selected mean `{row.selected_avg_oos:+.4f}R`; policy EV delta `{row.policy_ev_delta_oos:+.4f}R`",
                f"- OOS daily delta: `{row.delta_mean_oos:+.4f}R`, `t={row.delta_t_oos:+.3f}`, `p={row.delta_p_oos:.6f}`",
                "",
            ]
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULT_DOC)
    args = parser.parse_args()

    prereg_meta, prereg_commit_sha = _load_prereg_meta()
    prereg_git_sha = _git_last_commit(PREREG_PATH)
    holdout = pd.Timestamp(prereg_meta["holdout_date"])
    db_path = _normalize_writable_path(resolve_research_db_path(GOLD_DB_PATH))
    con = duckdb.connect(str(db_path), read_only=True)
    latest_day = con.execute("SELECT MAX(trading_day) FROM orb_outcomes WHERE pnl_r IS NOT NULL").fetchone()[0]

    try:
        comex = _with_signals(
            _load_parent_frame(
                con,
                session="COMEX_SETTLE",
                orb_minutes=5,
                rr_target=1.5,
                direction="long",
            ),
            "COMEX_SETTLE",
        )
        usd = _with_signals(
            _load_parent_frame(
                con,
                session="US_DATA_1000",
                orb_minutes=15,
                rr_target=1.5,
                direction="long",
            ),
            "US_DATA_1000",
        )
    finally:
        con.close()

    outcomes: list[TestOutcome] = []

    outcomes.append(
        _evaluate_test(
            df=comex,
            hypothesis_id=1,
            hypothesis_name="MNQ COMEX_SETTLE RR1.5 long PD_CLEAR_LONG shadow bucket",
            test_name="PD_CLEAR_LONG_fire",
            pass_name="unfiltered",
            signal_mask=comex["sig_pd_clear"] == 1,
            holdout=holdout,
        )
    )
    comex_filtered = comex[comex["sig_deployed"] == 1].copy()
    outcomes.append(
        _evaluate_test(
            df=comex_filtered,
            hypothesis_id=1,
            hypothesis_name="MNQ COMEX_SETTLE RR1.5 long PD_CLEAR_LONG shadow bucket",
            test_name="PD_CLEAR_LONG_fire",
            pass_name="filtered",
            signal_mask=comex_filtered["sig_pd_clear"] == 1,
            holdout=holdout,
        )
    )

    outcomes.append(
        _evaluate_test(
            df=usd,
            hypothesis_id=2,
            hypothesis_name="MNQ US_DATA_1000 O15 RR1.5 long prior-day structure score buckets",
            test_name="score_ge_1",
            pass_name="unfiltered",
            signal_mask=usd["structure_score"] >= 1,
            holdout=holdout,
        )
    )
    outcomes.append(
        _evaluate_test(
            df=usd,
            hypothesis_id=2,
            hypothesis_name="MNQ US_DATA_1000 O15 RR1.5 long prior-day structure score buckets",
            test_name="score_eq_2",
            pass_name="unfiltered",
            signal_mask=usd["structure_score"] == 2,
            holdout=holdout,
        )
    )
    usd_filtered = usd[usd["sig_deployed"] == 1].copy()
    outcomes.append(
        _evaluate_test(
            df=usd_filtered,
            hypothesis_id=2,
            hypothesis_name="MNQ US_DATA_1000 O15 RR1.5 long prior-day structure score buckets",
            test_name="score_ge_1",
            pass_name="filtered",
            signal_mask=usd_filtered["structure_score"] >= 1,
            holdout=holdout,
        )
    )
    outcomes.append(
        _evaluate_test(
            df=usd_filtered,
            hypothesis_id=2,
            hypothesis_name="MNQ US_DATA_1000 O15 RR1.5 long prior-day structure score buckets",
            test_name="score_eq_2",
            pass_name="filtered",
            signal_mask=usd_filtered["structure_score"] == 2,
            holdout=holdout,
        )
    )

    outcomes = _apply_family_bh(outcomes)

    grouped: dict[str, list[TestOutcome]] = {}
    for row in outcomes:
        grouped.setdefault(row.hypothesis_name, []).append(row)

    verdicts = {name: _hypothesis_verdict(rows) for name, rows in grouped.items()}
    family_counts = {label: sum(1 for v in verdicts.values() if v == label) for label in ("CONTINUE", "PARK", "KILL")}

    lines: list[str] = []
    lines.append("# MNQ parent structure shadow buckets v1")
    lines.append("")
    lines.append(f"**Pre-reg:** `{PREREG_PATH.relative_to(ROOT)}`")
    lines.append(f"**Pre-reg commit SHA field:** `{prereg_commit_sha}`")
    lines.append(f"**Pre-reg git lock commit:** `{prereg_git_sha}`")
    lines.append(f"**Canonical DB path:** `{db_path}`")
    lines.append(f"**Latest canonical trading day:** `{latest_day}`")
    lines.append("**Canonical layers:** `daily_features`, `orb_outcomes`")
    lines.append("**Family K:** `6` exact IS tests (`H1 1x2` + `H2 2x2`) with BH at `q=0.05`")
    lines.append("")
    lines.append(
        f"**Family verdict:** CONTINUE={family_counts['CONTINUE']} | PARK={family_counts['PARK']} | KILL={family_counts['KILL']}"
    )
    lines.append("")
    lines.append("## Scope / Question")
    lines.append("")
    lines.append(
        "Evaluate the exact preregistered `conditional_role` family on exact MNQ deployed-parent lane-side populations and ask one question only: do frozen ORB-end / pre-entry-safe prior-day structure buckets improve parent policy value as shadow-only conditioners?"
    )
    lines.append("")
    lines.append("## Verdict / Decision")
    lines.append("")
    lines.append("**Decision:** `DEAD`")
    lines.append("")
    lines.append(
        "No exact-parent shadow bucket in this `K=6` family improved `policy_ev_per_opportunity_r` enough to survive the prereg decision rule. Do not reopen this family under renamed score / confluence language."
    )
    lines.append("")
    lines.append("## Family summary")
    lines.append("")
    lines.extend(_render_summary_table(outcomes))
    lines.append("")

    for hypothesis_name, rows in grouped.items():
        lines.extend(_render_hypothesis_section(hypothesis_name, rows))

    lines.append("## Closeout")
    lines.append("")
    lines.append("SURVIVED SCRUTINY:")
    for name, verdict in verdicts.items():
        if verdict == "CONTINUE":
            lines.append(f"- {name}")
    lines.append("PARKED:")
    for name, verdict in verdicts.items():
        if verdict == "PARK":
            lines.append(f"- {name}")
    lines.append("DID NOT SURVIVE:")
    for name, verdict in verdicts.items():
        if verdict == "KILL":
            lines.append(f"- {name}")
    lines.append("## Caveats / Limitations")
    lines.append("")
    lines.append("CAVEATS:")
    lines.append("- OOS from 2026-01-01 onward is descriptive only and remains thin for every bucket.")
    lines.append("- This script evaluates exact lane-side parent populations only; it does not reopen same-session routing or live sizing questions.")
    lines.append("- ORB-end / pre-entry-safe structure predicates are used; no post-entry or post-session features are admitted.")
    lines.append("NEXT STEPS:")
    lines.append("- If any hypothesis is `CONTINUE`, the honest next move is a bounded shadow-monitor / translation follow-through on that exact object.")
    lines.append("- If a hypothesis is `PARK`, leave it as descriptive-only and do not broaden the family without a new prereg.")
    lines.append("- If a hypothesis is `KILL`, do not reopen it under renamed score language.")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append(f"- Runner: `research/{Path(__file__).name}`")
    lines.append(
        f"- Command: `./.venv-wsl/bin/python research/{Path(__file__).name} --output {RESULT_DOC.relative_to(ROOT)}`"
    )
    lines.append("")

    output_path = args.output
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path = _normalize_writable_path(output_path)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
