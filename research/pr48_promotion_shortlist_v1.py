"""PR48 promotion shortlist v1.

Bounded portfolio-layer shortlist:
- MES q45 executable
- MGC continuous executable
- MES+MGC duo versus raw parent duo
- MNQ continuous executable as shadow add-on

Canonical truth only:
- daily_features
- orb_outcomes
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.paths import GOLD_DB_PATH
from research.lib.conditional_role import (
    apply_bh,
    assign_is_quintiles,
    daily_dollar_series,
    delta_test,
    load_prereg_meta,
    load_symbol_frame,
    max_drawdown,
    role_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = ROOT / "docs" / "audit" / "hypotheses" / "2026-04-22-pr48-promotion-shortlist-v1.yaml"
RESULT_DOC = ROOT / "docs" / "audit" / "results" / "2026-04-22-pr48-promotion-shortlist-v1.md"

ORB_MINUTES = 5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR_TARGET = 1.5
DESIRED_CONT_WEIGHTS = {1: 0.5, 2: 0.75, 3: 1.0, 4: 1.25, 5: 1.5}
EXEC_CONT_CONTRACTS = {1: 0, 2: 1, 3: 1, 4: 1, 5: 2}


def _align(*series: pd.Series) -> list[pd.Series]:
    df = pd.concat(series, axis=1).fillna(0.0)
    return [df.iloc[:, i] for i in range(df.shape[1])]


def _combo_metrics(series: pd.Series) -> dict[str, float]:
    cumulative = series.cumsum()
    return {
        "total_dollars": float(series.sum()),
        "mean_daily_dollars": float(series.mean()),
        "max_dd_dollars": max_drawdown(series),
        "best_day_dollars": float(series.max()) if len(series) else 0.0,
        "worst_day_dollars": float(series.min()) if len(series) else 0.0,
        "n_days": int(len(series)),
        "end_equity_dollars": float(cumulative.iloc[-1]) if len(cumulative) else 0.0,
    }


def main() -> int:
    prereg_meta, prereg_sha = load_prereg_meta(PREREG_PATH)
    holdout = pd.Timestamp(prereg_meta["holdout_date"])
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    latest_day = con.execute("SELECT MAX(trading_day) FROM orb_outcomes WHERE pnl_r IS NOT NULL").fetchone()[0]

    try:
        mes = assign_is_quintiles(
            load_symbol_frame(
                con,
                symbol="MES",
                orb_minutes=ORB_MINUTES,
                entry_model=ENTRY_MODEL,
                confirm_bars=CONFIRM_BARS,
                rr_target=RR_TARGET,
            ),
            holdout=holdout,
            desired_weight_map=DESIRED_CONT_WEIGHTS,
            executable_contract_map=EXEC_CONT_CONTRACTS,
        )
        mgc = assign_is_quintiles(
            load_symbol_frame(
                con,
                symbol="MGC",
                orb_minutes=ORB_MINUTES,
                entry_model=ENTRY_MODEL,
                confirm_bars=CONFIRM_BARS,
                rr_target=RR_TARGET,
            ),
            holdout=holdout,
            desired_weight_map=DESIRED_CONT_WEIGHTS,
            executable_contract_map=EXEC_CONT_CONTRACTS,
        )
        mnq = assign_is_quintiles(
            load_symbol_frame(
                con,
                symbol="MNQ",
                orb_minutes=ORB_MINUTES,
                entry_model=ENTRY_MODEL,
                confirm_bars=CONFIRM_BARS,
                rr_target=RR_TARGET,
            ),
            holdout=holdout,
            desired_weight_map=DESIRED_CONT_WEIGHTS,
            executable_contract_map=EXEC_CONT_CONTRACTS,
        )
    finally:
        con.close()

    splits = {}
    for name, df in (("MES", mes), ("MGC", mgc), ("MNQ", mnq)):
        splits[name] = {
            "IS": df[df["trading_day"] < holdout].copy(),
            "OOS": df[df["trading_day"] >= holdout].copy(),
        }

    is_tests = {
        ("MES", "q45_exec"): delta_test(
            daily_dollar_series(splits["MES"]["IS"], "w_parent"),
            daily_dollar_series(splits["MES"]["IS"], "w_q45"),
        ),
        ("MGC", "cont_exec"): delta_test(
            daily_dollar_series(splits["MGC"]["IS"], "w_parent"),
            daily_dollar_series(splits["MGC"]["IS"], "contracts_cont_exec"),
        ),
    }

    mes_parent_is, mgc_parent_is = _align(
        daily_dollar_series(splits["MES"]["IS"], "w_parent"),
        daily_dollar_series(splits["MGC"]["IS"], "w_parent"),
    )
    mes_cand_is, mgc_cand_is = _align(
        daily_dollar_series(splits["MES"]["IS"], "w_q45"),
        daily_dollar_series(splits["MGC"]["IS"], "contracts_cont_exec"),
    )
    duo_parent_is = mes_parent_is + mgc_parent_is
    duo_candidate_is = mes_cand_is + mgc_cand_is
    is_tests[("DUO", "mes_q45_plus_mgc_cont_exec")] = delta_test(duo_parent_is, duo_candidate_is)

    duo_aligned_is, mnq_add_is = _align(
        duo_candidate_is.rename("duo"),
        daily_dollar_series(splits["MNQ"]["IS"], "contracts_cont_exec"),
    )
    trio_candidate_is = duo_aligned_is + mnq_add_is
    is_tests[("MNQ", "shadow_addon")] = delta_test(duo_aligned_is, trio_candidate_is)
    is_tests = apply_bh(is_tests, q=0.05)

    mes_parent_oos, mgc_parent_oos = _align(
        daily_dollar_series(splits["MES"]["OOS"], "w_parent"),
        daily_dollar_series(splits["MGC"]["OOS"], "w_parent"),
    )
    mes_cand_oos, mgc_cand_oos = _align(
        daily_dollar_series(splits["MES"]["OOS"], "w_q45"),
        daily_dollar_series(splits["MGC"]["OOS"], "contracts_cont_exec"),
    )
    duo_parent_oos = mes_parent_oos + mgc_parent_oos
    duo_candidate_oos = mes_cand_oos + mgc_cand_oos
    duo_aligned_oos, mnq_add_oos = _align(
        duo_candidate_oos.rename("duo"),
        daily_dollar_series(splits["MNQ"]["OOS"], "contracts_cont_exec"),
    )
    trio_candidate_oos = duo_aligned_oos + mnq_add_oos

    oos_checks = {
        ("MES", "q45_exec"): delta_test(
            daily_dollar_series(splits["MES"]["OOS"], "w_parent"),
            daily_dollar_series(splits["MES"]["OOS"], "w_q45"),
        ),
        ("MGC", "cont_exec"): delta_test(
            daily_dollar_series(splits["MGC"]["OOS"], "w_parent"),
            daily_dollar_series(splits["MGC"]["OOS"], "contracts_cont_exec"),
        ),
        ("DUO", "mes_q45_plus_mgc_cont_exec"): delta_test(duo_parent_oos, duo_candidate_oos),
        ("MNQ", "shadow_addon"): delta_test(duo_aligned_oos, trio_candidate_oos),
    }

    parts: list[str] = []
    parts.append("# PR48 promotion shortlist v1\n")
    parts.append(f"**Pre-reg:** `{PREREG_PATH.relative_to(ROOT)}`")
    parts.append(f"**Pre-reg commit SHA:** `{prereg_sha}`")
    parts.append("**Canonical layers:** `daily_features`, `orb_outcomes`")
    parts.append(f"**Scope:** O{ORB_MINUTES} x {ENTRY_MODEL} x CB{CONFIRM_BARS} x RR{RR_TARGET}")
    parts.append(
        "**Candidates:** `MES q45 executable`, `MGC continuous executable`, `MES+MGC duo`, `MNQ continuous executable shadow add-on`"
    )
    parts.append(
        "**Primary tests:** daily dollar delta vs the declared parent comparator, BH FDR at family `K=4` on IS."
    )
    parts.append(f"**Sacred OOS window:** `{holdout.date().isoformat()}` onward")
    parts.append(f"**Latest canonical trading day:** `{latest_day}`")
    parts.append("")

    parts.append("## IS shortlist tests")
    parts.append("")
    parts.append("| candidate | mean_daily_delta_$ | t | p_two_tailed | bh_survives | direction_positive |")
    parts.append("|---|---:|---:|---:|:---:|:---:|")
    for key in [
        ("MES", "q45_exec"),
        ("MGC", "cont_exec"),
        ("DUO", "mes_q45_plus_mgc_cont_exec"),
        ("MNQ", "shadow_addon"),
    ]:
        res = is_tests[key]
        label = f"{key[0]}:{key[1]}"
        parts.append(
            f"| {label} | {res.mean_delta_r:+,.2f} | {res.t_stat:+.3f} | {res.p_two_tailed:.4f} | "
            f"{'Y' if res.bh_survives else 'N'} | {'Y' if res.direction_positive else 'N'} |"
        )
    parts.append("")

    parts.append("## OOS direction checks")
    parts.append("")
    parts.append("| candidate | IS sign | OOS sign | OOS mean_daily_delta_$ | direction_match |")
    parts.append("|---|---:|---:|---:|:---:|")
    for key in [
        ("MES", "q45_exec"),
        ("MGC", "cont_exec"),
        ("DUO", "mes_q45_plus_mgc_cont_exec"),
        ("MNQ", "shadow_addon"),
    ]:
        is_res = is_tests[key]
        oos_res = oos_checks[key]
        label = f"{key[0]}:{key[1]}"
        parts.append(
            f"| {label} | {'+' if is_res.direction_positive else '-'} | "
            f"{'+' if oos_res.direction_positive else '-'} | {oos_res.mean_delta_r:+,.2f} | "
            f"{'Y' if (is_res.direction_positive == oos_res.direction_positive) else 'N'} |"
        )
    parts.append("")

    parts.append("## Candidate-level metrics")
    parts.append("")
    parts.append("| instrument | era | role | policy_ev_per_opp_r | daily_total_$ | daily_max_dd_$ |")
    parts.append("|---|---|---|---:|---:|---:|")
    for inst, role_name, weight_col in (
        ("MES", "parent", "w_parent"),
        ("MES", "q45_exec", "w_q45"),
        ("MGC", "parent", "w_parent"),
        ("MGC", "cont_exec", "contracts_cont_exec"),
        ("MNQ", "cont_exec_shadow", "contracts_cont_exec"),
    ):
        for era in ("IS", "OOS"):
            metrics = role_metrics(splits[inst][era], weight_col)
            parts.append(
                f"| {inst} | {era} | {role_name} | {metrics['policy_ev_per_opp_r']:+.4f} | "
                f"{metrics['daily_total_dollars']:+,.0f} | {metrics['daily_max_dd_dollars']:+,.0f} |"
            )
    parts.append("")

    parts.append("## Combo metrics")
    parts.append("")
    parts.append("| combo | era | total_$ | mean_daily_$ | max_dd_$ | best_day_$ | worst_day_$ |")
    parts.append("|---|---|---:|---:|---:|---:|---:|")
    combo_rows = {
        ("parent_duo", "IS"): _combo_metrics(duo_parent_is),
        ("candidate_duo", "IS"): _combo_metrics(duo_candidate_is),
        ("candidate_trio", "IS"): _combo_metrics(trio_candidate_is),
        ("parent_duo", "OOS"): _combo_metrics(duo_parent_oos),
        ("candidate_duo", "OOS"): _combo_metrics(duo_candidate_oos),
        ("candidate_trio", "OOS"): _combo_metrics(trio_candidate_oos),
    }
    for (combo_name, era), metrics in combo_rows.items():
        parts.append(
            f"| {combo_name} | {era} | {metrics['total_dollars']:+,.0f} | {metrics['mean_daily_dollars']:+,.2f} | "
            f"{metrics['max_dd_dollars']:+,.0f} | {metrics['best_day_dollars']:+,.0f} | {metrics['worst_day_dollars']:+,.0f} |"
        )
    parts.append("")

    parts.append("## Interpretation guardrails")
    parts.append("")
    parts.append("- This is a shortlist-action study, not a fresh discovery scan.")
    parts.append(
        "- Dollar deltas use canonical `risk_dollars` translation from `entry_price`, `stop_price`, and instrument cost specs."
    )
    parts.append("- MNQ remains a shadow add-on unless it improves the MES/MGC duo on fresh OOS, not just IS.")
    parts.append(
        "- OOS from 2026-01-01 to 2026-04-16 is thin; use it to decide continue vs shadow, not full deployment."
    )
    parts.append("")

    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    RESULT_DOC.write_text("\n".join(parts), encoding="utf-8")
    print(f"WROTE {RESULT_DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
