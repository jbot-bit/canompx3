"""PR48 role follow-through v1.

This is the bounded next step after the conditional-role implementation study.
It upgrades the question from descriptive role EVs to:

- daily policy-delta testing versus the parent
- BH FDR over the declared family K
- drawdown and dollar translation
- executable drag for the continuous map
- simple open-lot / concurrency diagnostics

Canonical truth only:
- daily_features
- orb_outcomes
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

from pipeline.cost_model import get_cost_spec, risk_in_dollars
from pipeline.paths import GOLD_DB_PATH
from research.lib.stats import bh_fdr, compute_metrics, ttest_1s
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.hypothesis_loader import check_mode_a_consistency, load_hypothesis_metadata
from trading_app.topstep_scaling_plan import lots_for_position

ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = ROOT / "docs" / "audit" / "hypotheses" / "2026-04-22-pr48-role-followthrough-v1.yaml"
RESULT_DOC = ROOT / "docs" / "audit" / "results" / "2026-04-22-pr48-role-followthrough-v1.md"

INSTRUMENTS = ("MNQ", "MES", "MGC")
ORB_MINUTES = 5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR_TARGET = 1.5
DESIRED_CONT_WEIGHTS = {1: 0.5, 2: 0.75, 3: 1.0, 4: 1.25, 5: 1.5}
# Single-account executable proxy for the desired continuous map.
# This is intentionally conservative: the lowest bucket drops to 0 contracts.
EXEC_CONT_CONTRACTS = {1: 0, 2: 1, 3: 1, 4: 1, 5: 2}
PRIMARY_CANDIDATES = ("q45_filter", "continuous_desired")


@dataclass(frozen=True)
class DailyDeltaResult:
    n_days: int
    mean_delta_r: float
    t_stat: float
    p_two_tailed: float
    bh_survives: bool
    direction_positive: bool


def _load_prereg_meta() -> tuple[dict, str]:
    meta = load_hypothesis_metadata(PREREG_PATH)
    check_mode_a_consistency(meta)
    body = yaml.safe_load(PREREG_PATH.read_text(encoding="utf-8"))
    commit_sha = str(body.get("metadata", {}).get("commit_sha", "UNSTAMPED"))
    return meta, commit_sha


def _list_sessions(con: duckdb.DuckDBPyConnection, symbol: str) -> list[str]:
    rows = con.execute(
        """
        SELECT DISTINCT orb_label
        FROM orb_outcomes
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
        SELECT o.trading_day, o.entry_ts, o.exit_ts, o.pnl_r, o.entry_price, o.stop_price, o.orb_label, df.rel_vol
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
        spec = get_cost_spec(symbol)
        sub["risk_dollars"] = sub.apply(
            lambda row, spec=spec: risk_in_dollars(spec, float(row["entry_price"]), float(row["stop_price"])),
            axis=1,
        )
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).dropna(subset=["rel_vol", "pnl_r"]).reset_index(drop=True)


def _assign_is_quintiles(df: pd.DataFrame, holdout: pd.Timestamp) -> pd.DataFrame:
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
    out["w_cont_desired"] = out["q_is"].map(DESIRED_CONT_WEIGHTS).astype(float)
    out["contracts_parent"] = 1
    out["contracts_q45"] = np.where(out["q_is"] >= 4, 1, 0)
    out["contracts_cont_exec"] = out["q_is"].map(EXEC_CONT_CONTRACTS).astype(int)
    return out


def _daily_policy_series(df: pd.DataFrame, weight_col: str) -> pd.Series:
    weighted = df["pnl_r"].astype(float) * df[weight_col].astype(float)
    return weighted.groupby(df["trading_day"]).sum().sort_index()


def _daily_dollar_series(df: pd.DataFrame, weight_col: str) -> pd.Series:
    weighted = df["pnl_r"].astype(float) * df["risk_dollars"].astype(float) * df[weight_col].astype(float)
    return weighted.groupby(df["trading_day"]).sum().sort_index()


def _max_drawdown(series: pd.Series) -> float:
    vals = series.to_numpy(dtype=float)
    if len(vals) == 0:
        return 0.0
    cumulative = np.cumsum(vals)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    return float(drawdown.max())


def _delta_test(parent_daily: pd.Series, cand_daily: pd.Series) -> DailyDeltaResult:
    aligned = pd.concat(
        [parent_daily.rename("parent"), cand_daily.rename("candidate")],
        axis=1,
    ).fillna(0.0)
    delta = (aligned["candidate"] - aligned["parent"]).to_numpy(dtype=float)
    n_days, mean_delta, _wr, t_stat, p_two = ttest_1s(delta, 0.0)
    return DailyDeltaResult(
        n_days=n_days,
        mean_delta_r=float(mean_delta),
        t_stat=float(t_stat),
        p_two_tailed=float(p_two),
        bh_survives=False,
        direction_positive=bool(mean_delta > 0) if not np.isnan(mean_delta) else False,
    )


def _role_metrics(df: pd.DataFrame, weight_col: str) -> dict[str, float]:
    selected = df[df[weight_col] > 0]
    trade_share = float((df[weight_col] > 0).mean())
    selected_avg = float(selected["pnl_r"].mean()) if not selected.empty else float("nan")
    policy_ev = float((df["pnl_r"].astype(float) * df[weight_col].astype(float)).mean())
    avg_weight = float(df[weight_col].mean())
    capital_norm = float(policy_ev / avg_weight) if avg_weight > 0 else float("nan")
    daily_r = _daily_policy_series(df, weight_col)
    daily_dollars = _daily_dollar_series(df, weight_col)
    r_metrics = compute_metrics(daily_r.tolist()) or {}
    return {
        "selected_n": int((df[weight_col] > 0).sum()),
        "trade_share": trade_share,
        "selected_avg_r": selected_avg,
        "policy_ev_per_opp_r": policy_ev,
        "avg_weight": avg_weight,
        "capital_normalized_ev_r": capital_norm,
        "daily_total_r": float(daily_r.sum()),
        "daily_max_dd_r": float(r_metrics.get("max_dd", 0.0)),
        "daily_total_dollars": float(daily_dollars.sum()),
        "daily_max_dd_dollars": _max_drawdown(daily_dollars),
    }


def _max_open_contracts_and_lots(df: pd.DataFrame, contract_col: str, instrument: str) -> tuple[int, int]:
    max_contracts = 0
    max_lots = 0
    for _day, day_df in df.groupby("trading_day"):
        trades = day_df[day_df[contract_col] > 0].copy()
        if trades.empty:
            continue
        events: list[tuple[pd.Timestamp, int, int]] = []
        for _, row in trades.iterrows():
            entry_ts = row["entry_ts"]
            exit_ts = row["exit_ts"]
            contracts = int(row[contract_col])
            if pd.isna(entry_ts):
                continue
            if pd.isna(exit_ts):
                exit_ts = entry_ts
            events.append((pd.Timestamp(entry_ts), 0, contracts))
            events.append((pd.Timestamp(exit_ts), 1, contracts))
        events.sort(key=lambda item: (item[0], item[1]))
        open_contracts = 0
        for _ts, event_type, contracts in events:
            if event_type == 0:
                open_contracts += contracts
            else:
                open_contracts = max(0, open_contracts - contracts)
            max_contracts = max(max_contracts, open_contracts)
            max_lots = max(max_lots, lots_for_position(instrument, open_contracts))
    return max_contracts, max_lots


def _apply_bh(results: dict[tuple[str, str], DailyDeltaResult]) -> None:
    ordered_keys = list(results.keys())
    p_values = [results[key].p_two_tailed for key in ordered_keys]
    reject_ix = bh_fdr(p_values, q=0.05)
    for i, key in enumerate(ordered_keys):
        res = results[key]
        results[key] = DailyDeltaResult(
            n_days=res.n_days,
            mean_delta_r=res.mean_delta_r,
            t_stat=res.t_stat,
            p_two_tailed=res.p_two_tailed,
            bh_survives=i in reject_ix,
            direction_positive=res.direction_positive,
        )


def main() -> int:
    prereg_meta, prereg_sha = _load_prereg_meta()
    holdout = pd.Timestamp(prereg_meta["holdout_date"])
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    latest_day = con.execute("SELECT MAX(trading_day) FROM orb_outcomes WHERE pnl_r IS NOT NULL").fetchone()[0]

    is_delta_tests: dict[tuple[str, str], DailyDeltaResult] = {}
    is_store: dict[str, dict[str, dict[str, float]]] = {}
    oos_store: dict[str, dict[str, dict[str, float]]] = {}
    exec_drag_store: dict[str, dict[str, float]] = {}
    concurrency_store: dict[str, dict[str, int]] = {}

    try:
        for symbol in INSTRUMENTS:
            raw = _load_symbol(con, symbol)
            raw = _assign_is_quintiles(raw, holdout)
            is_df = raw[raw["trading_day"] < holdout].copy()
            oos_df = raw[raw["trading_day"] >= holdout].copy()

            is_parent_daily = _daily_policy_series(is_df, "w_parent")
            oos_parent_daily = _daily_policy_series(oos_df, "w_parent")

            is_store[symbol] = {
                "parent": _role_metrics(is_df, "w_parent"),
                "q45_filter": _role_metrics(is_df, "w_q45"),
                "continuous_desired": _role_metrics(is_df, "w_cont_desired"),
            }
            oos_store[symbol] = {
                "parent": _role_metrics(oos_df, "w_parent"),
                "q45_filter": _role_metrics(oos_df, "w_q45"),
                "continuous_desired": _role_metrics(oos_df, "w_cont_desired"),
            }

            is_delta_tests[(symbol, "q45_filter")] = _delta_test(is_parent_daily, _daily_policy_series(is_df, "w_q45"))
            is_delta_tests[(symbol, "continuous_desired")] = _delta_test(
                is_parent_daily,
                _daily_policy_series(is_df, "w_cont_desired"),
            )

            oos_delta_q45 = _delta_test(oos_parent_daily, _daily_policy_series(oos_df, "w_q45"))
            oos_delta_cont = _delta_test(oos_parent_daily, _daily_policy_series(oos_df, "w_cont_desired"))
            oos_store[symbol]["q45_filter_delta"] = {
                "mean_delta_r": oos_delta_q45.mean_delta_r,
                "direction_positive": float(oos_delta_q45.direction_positive),
            }
            oos_store[symbol]["continuous_desired_delta"] = {
                "mean_delta_r": oos_delta_cont.mean_delta_r,
                "direction_positive": float(oos_delta_cont.direction_positive),
            }

            cont_exec_is = _role_metrics(is_df.assign(w_exec=is_df["contracts_cont_exec"].astype(float)), "w_exec")
            cont_exec_oos = _role_metrics(oos_df.assign(w_exec=oos_df["contracts_cont_exec"].astype(float)), "w_exec")
            exec_drag_store[symbol] = {
                "is_drag_policy_ev_r": cont_exec_is["policy_ev_per_opp_r"]
                - is_store[symbol]["continuous_desired"]["policy_ev_per_opp_r"],
                "oos_drag_policy_ev_r": cont_exec_oos["policy_ev_per_opp_r"]
                - oos_store[symbol]["continuous_desired"]["policy_ev_per_opp_r"],
                "is_exec_total_dollars": cont_exec_is["daily_total_dollars"],
                "oos_exec_total_dollars": cont_exec_oos["daily_total_dollars"],
            }

            q45_contracts, q45_lots = _max_open_contracts_and_lots(is_df.assign(c=is_df["contracts_q45"]), "c", symbol)
            exec_contracts, exec_lots = _max_open_contracts_and_lots(
                is_df.assign(c=is_df["contracts_cont_exec"]),
                "c",
                symbol,
            )
            concurrency_store[symbol] = {
                "q45_max_open_contracts_is": q45_contracts,
                "q45_max_open_lots_is": q45_lots,
                "cont_exec_max_open_contracts_is": exec_contracts,
                "cont_exec_max_open_lots_is": exec_lots,
            }
    finally:
        con.close()

    _apply_bh(is_delta_tests)

    parts: list[str] = []
    parts.append("# PR48 role follow-through v1\n")
    parts.append(f"**Pre-reg:** `{PREREG_PATH.relative_to(ROOT)}`")
    parts.append(f"**Pre-reg commit SHA:** `{prereg_sha}`")
    parts.append("**Canonical layers:** `daily_features`, `orb_outcomes`")
    parts.append(
        f"**Scope:** `{', '.join(INSTRUMENTS)}` x O{ORB_MINUTES} x {ENTRY_MODEL} x CB{CONFIRM_BARS} x RR{RR_TARGET}"
    )
    parts.append(
        "**Primary test:** daily policy delta versus parent for `Q4+Q5` and `continuous_desired`, "
        f"BH FDR at family `K={len(is_delta_tests)}`."
    )
    parts.append("**Diagnostics:** drawdown, dollar translation, executable drag, and max-open-lots proxy.")
    parts.append(f"**Sacred OOS window:** `{holdout.date().isoformat()}` onward.")
    parts.append(f"**Latest canonical trading day:** `{latest_day}`")
    parts.append("")

    for symbol in INSTRUMENTS:
        parts.append(f"## {symbol}")
        parts.append("")
        parts.append("### IS daily delta tests")
        parts.append("")
        parts.append("| candidate | mean_daily_delta_r | t | p_two_tailed | bh_survives | direction_positive |")
        parts.append("|---|---:|---:|---:|:---:|:---:|")
        for candidate in PRIMARY_CANDIDATES:
            res = is_delta_tests[(symbol, candidate)]
            parts.append(
                f"| {candidate} | {res.mean_delta_r:+.4f} | {res.t_stat:+.3f} | {res.p_two_tailed:.4f} | "
                f"{'Y' if res.bh_survives else 'N'} | {'Y' if res.direction_positive else 'N'} |"
            )
        parts.append("")

        parts.append("### Role metrics")
        parts.append("")
        parts.append(
            "| era | role | selected_n | trade_share | selected_avg_r | policy_ev_per_opp_r | "
            "capital_normalized_ev_r | daily_total_r | daily_max_dd_r | daily_total_dollars | daily_max_dd_dollars |"
        )
        parts.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for era_name, store in (("IS", is_store[symbol]), ("OOS", oos_store[symbol])):
            for role_name in ("parent", "q45_filter", "continuous_desired"):
                m = store[role_name]
                parts.append(
                    f"| {era_name} | {role_name} | {m['selected_n']} | {m['trade_share']:.3f} | "
                    f"{m['selected_avg_r']:+.4f} | {m['policy_ev_per_opp_r']:+.4f} | "
                    f"{m['capital_normalized_ev_r']:+.4f} | {m['daily_total_r']:+.2f} | "
                    f"{m['daily_max_dd_r']:+.2f} | {m['daily_total_dollars']:+,.0f} | {m['daily_max_dd_dollars']:+,.0f} |"
                )
        parts.append("")

        parts.append("### OOS direction match")
        parts.append("")
        parts.append(
            "| candidate | IS daily delta sign | OOS daily delta sign | OOS mean_daily_delta_r | direction_match |"
        )
        parts.append("|---|---:|---:|---:|:---:|")
        for candidate in PRIMARY_CANDIDATES:
            is_res = is_delta_tests[(symbol, candidate)]
            oos_delta = oos_store[symbol][f"{candidate}_delta"]
            oos_positive = bool(oos_delta["direction_positive"])
            parts.append(
                f"| {candidate} | {'+' if is_res.direction_positive else '-'} | "
                f"{'+' if oos_positive else '-'} | {oos_delta['mean_delta_r']:+.4f} | "
                f"{'Y' if (is_res.direction_positive == oos_positive) else 'N'} |"
            )
        parts.append("")

        drag = exec_drag_store[symbol]
        conc = concurrency_store[symbol]
        parts.append("### Executable drag and concurrency diagnostics")
        parts.append("")
        parts.append(
            f"- Continuous desired -> executable proxy policy EV drag: "
            f"IS `{drag['is_drag_policy_ev_r']:+.4f}R/opp`, OOS `{drag['oos_drag_policy_ev_r']:+.4f}R/opp`"
        )
        parts.append(
            f"- Continuous executable proxy total dollars: "
            f"IS `${drag['is_exec_total_dollars']:+,.0f}`, OOS `${drag['oos_exec_total_dollars']:+,.0f}`"
        )
        parts.append(
            f"- Q4+Q5 max open contracts/lots in IS: "
            f"{conc['q45_max_open_contracts_is']} contracts / {conc['q45_max_open_lots_is']} lots"
        )
        parts.append(
            f"- Continuous executable max open contracts/lots in IS: "
            f"{conc['cont_exec_max_open_contracts_is']} contracts / {conc['cont_exec_max_open_lots_is']} lots"
        )
        parts.append("")

    parts.append("## Interpretation guardrails")
    parts.append("")
    parts.append("- Primary inference is on daily policy delta versus parent, not selected-trade mean.")
    parts.append(
        "- Continuous desired sizing is the research object; executable drag is a diagnostic, not a separate promoted winner."
    )
    parts.append(
        "- OOS from 2026-01-01 to 2026-04-16 is still thin and should be treated as direction/implementation monitoring only."
    )
    parts.append("")

    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    RESULT_DOC.write_text("\n".join(parts), encoding="utf-8")
    print(f"WROTE {RESULT_DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
