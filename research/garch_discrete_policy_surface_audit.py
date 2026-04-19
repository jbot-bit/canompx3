"""Raw discrete policy-surface audit for garch as a trading/deployment input.

Purpose:
  Compare the main real-action trading policies for `garch_forecast_vol_pct`
  using canonical raw trade rows only:

    - trade only favorable regime
    - skip hostile regime
    - size up favorable regime
    - combine skip-low with double-high

This intentionally avoids fractional-weight abstractions. Every policy is a
real integer action count per trade: 0, 1, or 2.

Pre-registration:
  docs/audit/hypotheses/2026-04-16-garch-discrete-policy-surface-audit.yaml

Output:
  docs/audit/results/2026-04-16-garch-discrete-policy-surface-audit.md
"""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import duckdb
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from research import garch_normalized_sizing_audit as norm
from research import garch_regime_family_audit as fam

OUTPUT_MD = Path("docs/audit/results/2026-04-16-garch-discrete-policy-surface-audit.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

POLICIES = [
    "SESSION_TAKE_HIGH_ONLY",
    "GLOBAL_TAKE_HIGH_ONLY",
    "SESSION_SKIP_LOW_ONLY",
    "GLOBAL_SKIP_LOW_ONLY",
    "SESSION_HIGH_2X_ONLY",
    "GLOBAL_HIGH_2X_ONLY",
    "SESSION_CLIPPED_0_1_2",
    "GLOBAL_CLIPPED_0_1_2",
]


def action_contracts(policy: str, gp: float, session: str, profiles: dict[str, dict[str, bool]]) -> int:
    p = profiles.get(session, {"high_dir": False, "low_dir": False})
    session_high = bool(p.get("high_dir"))
    session_low = bool(p.get("low_dir"))

    if policy == "SESSION_TAKE_HIGH_ONLY":
        return 1 if (session_high and gp >= 70.0) or (not session_high) else 0
    if policy == "GLOBAL_TAKE_HIGH_ONLY":
        return 1 if gp >= 70.0 else 0
    if policy == "SESSION_SKIP_LOW_ONLY":
        return 0 if session_low and gp <= 30.0 else 1
    if policy == "GLOBAL_SKIP_LOW_ONLY":
        return 0 if gp <= 30.0 else 1
    if policy == "SESSION_HIGH_2X_ONLY":
        return 2 if session_high and gp >= 70.0 else 1
    if policy == "GLOBAL_HIGH_2X_ONLY":
        return 2 if gp >= 70.0 else 1
    if policy == "SESSION_CLIPPED_0_1_2":
        if session_high and gp >= 70.0:
            return 2
        if session_low and gp <= 30.0:
            return 0
        return 1
    if policy == "GLOBAL_CLIPPED_0_1_2":
        if gp >= 70.0:
            return 2
        if gp <= 30.0:
            return 0
        return 1
    raise ValueError(f"Unknown policy: {policy}")


def ann_sharpe(daily: pd.Series) -> float:
    daily = daily.astype(float)
    sd = daily.std(ddof=1)
    if len(daily) < 2 or sd <= 0:
        return 0.0
    return float((daily.mean() / sd) * math.sqrt(252.0))


def max_drawdown(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    equity = series.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min())


def summarize(df: pd.DataFrame, contracts_col: str) -> dict[str, float]:
    if len(df) == 0:
        return {
            "n_trades": 0,
            "active_trades": 0,
            "mean_contracts": 0.0,
            "exp_r": 0.0,
            "total_r": 0.0,
            "total_dollars": 0.0,
            "sharpe_r": 0.0,
            "max_dd_r": 0.0,
            "worst_day_dollars": 0.0,
            "worst_5day_dollars": 0.0,
            "max_daily_risk_dollars": 0.0,
        }

    work = df.copy()
    work["weighted_r"] = work["pnl_r"] * work[contracts_col]
    work["weighted_dollars"] = work["pnl_dollars"] * work[contracts_col]
    work["weighted_risk_dollars"] = work["risk_dollars"] * work[contracts_col]

    daily = (
        work.groupby("trading_day", as_index=True)[["weighted_r", "weighted_dollars", "weighted_risk_dollars"]]
        .sum()
        .sort_index()
    )
    roll5 = daily["weighted_dollars"].rolling(5).sum()

    active = work[work[contracts_col] > 0]
    return {
        "n_trades": int(len(work)),
        "active_trades": int(len(active)),
        "mean_contracts": float(work[contracts_col].mean()),
        "exp_r": float(work["weighted_r"].mean()),
        "total_r": float(work["weighted_r"].sum()),
        "total_dollars": float(work["weighted_dollars"].sum()),
        "sharpe_r": ann_sharpe(daily["weighted_r"]),
        "max_dd_r": max_drawdown(daily["weighted_r"]),
        "worst_day_dollars": float(daily["weighted_dollars"].min()),
        "worst_5day_dollars": float(roll5.min()) if roll5.notna().any() else 0.0,
        "max_daily_risk_dollars": float(daily["weighted_risk_dollars"].max()),
    }


def evaluate_scope(
    df: pd.DataFrame, scope: str, profiles: dict[str, dict[str, bool]]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    out_rows = []
    contribution_tables: dict[str, pd.DataFrame] = {}

    base = df.copy()
    base["contracts"] = 1
    base_is = summarize(base[~base["is_oos"]], "contracts")
    base_oos = summarize(base[base["is_oos"]], "contracts")
    base_full = summarize(base, "contracts")

    for policy in POLICIES:
        work = df.copy()
        work["contracts"] = [
            action_contracts(policy, float(gp), str(sess), profiles)
            for gp, sess in zip(work["gp"].astype(float), work["orb_label"].astype(str))
        ]

        is_metrics = summarize(work[~work["is_oos"]], "contracts")
        oos_metrics = summarize(work[work["is_oos"]], "contracts")
        full_metrics = summarize(work, "contracts")

        expr_delta_is = is_metrics["exp_r"] - base_is["exp_r"]
        expr_delta_oos = oos_metrics["exp_r"] - base_oos["exp_r"] if oos_metrics["n_trades"] else 0.0
        retention = expr_delta_oos / expr_delta_is if abs(expr_delta_is) > 1e-9 else float("nan")

        out_rows.append(
            {
                "scope": scope,
                "policy": policy,
                "active_pct": full_metrics["active_trades"] / full_metrics["n_trades"]
                if full_metrics["n_trades"]
                else 0.0,
                "mean_contracts": full_metrics["mean_contracts"],
                "full_delta_r": full_metrics["total_r"] - base_full["total_r"],
                "full_delta_dollars": full_metrics["total_dollars"] - base_full["total_dollars"],
                "full_sharpe_delta": full_metrics["sharpe_r"] - base_full["sharpe_r"],
                "full_max_dd_r_delta": full_metrics["max_dd_r"] - base_full["max_dd_r"],
                "worst_day_dollars_delta": full_metrics["worst_day_dollars"] - base_full["worst_day_dollars"],
                "worst_5day_dollars_delta": full_metrics["worst_5day_dollars"] - base_full["worst_5day_dollars"],
                "max_daily_risk_dollars_delta": full_metrics["max_daily_risk_dollars"]
                - base_full["max_daily_risk_dollars"],
                "is_exp_r_delta": expr_delta_is,
                "oos_exp_r_delta": expr_delta_oos,
                "oos_retention": retention,
            }
        )

        contrib = work.copy()
        contrib["base_dollars"] = contrib["pnl_dollars"]
        contrib["alt_dollars"] = contrib["pnl_dollars"] * contrib["contracts"]
        contrib["delta_dollars"] = contrib["alt_dollars"] - contrib["base_dollars"]
        contribution_tables[policy] = (
            contrib.groupby(["instrument", "orb_label"], as_index=False)[
                ["base_dollars", "alt_dollars", "delta_dollars"]
            ]
            .sum()
            .sort_values("delta_dollars", ascending=False)
            .reset_index(drop=True)
        )

    return pd.DataFrame(out_rows), contribution_tables


def emit(
    profiles: dict[str, dict[str, bool]],
    scope_results: pd.DataFrame,
    contrib_tables: dict[str, dict[str, pd.DataFrame]],
) -> None:
    lines = [
        "# Garch Discrete Policy Surface Audit",
        "",
        "**Date:** 2026-04-16",
        "**Pre-registration:** `docs/audit/hypotheses/2026-04-16-garch-discrete-policy-surface-audit.yaml`",
        "**Purpose:** compare raw discrete trading policies for `garch_forecast_vol_pct` on canonical trade rows only.",
        "",
        "**Grounding:**",
        "- `docs/institutional/literature/chan_2008_ch7_regime_switching.md`",
        "- `docs/institutional/mechanism_priors.md`",
        "- `docs/institutional/regime-and-rr-handling-framework.md`",
        "- `docs/audit/results/2026-04-16-garch-regime-family-audit.md`",
        "- `docs/audit/results/2026-04-16-garch-g0-preflight.md`",
        "",
        "**Raw-number rule:** all policy results are recomputed from canonical `orb_outcomes` + `daily_features` trade rows. No stored expectancy metadata is trusted.",
        "",
        "## Session directional support used by session-aware policies",
        "",
        "| Session | High directional support | Low directional support |",
        "|---|---|---|",
    ]
    for sess, p in sorted(profiles.items()):
        lines.append(f"| {sess} | {'Y' if p['high_dir'] else '.'} | {'Y' if p['low_dir'] else '.'} |")

    lines += [
        "",
        "## Policy definitions",
        "",
        "- `SESSION_TAKE_HIGH_ONLY`: trade only `gp>=70` in sessions with high-directional support; other sessions stay base `1x`.",
        "- `GLOBAL_TAKE_HIGH_ONLY`: trade only `gp>=70` everywhere.",
        "- `SESSION_SKIP_LOW_ONLY`: skip `gp<=30` in sessions with low-directional support; other sessions stay base `1x`.",
        "- `GLOBAL_SKIP_LOW_ONLY`: skip `gp<=30` everywhere.",
        "- `SESSION_HIGH_2X_ONLY`: double size on `gp>=70` in sessions with high-directional support; otherwise `1x`.",
        "- `GLOBAL_HIGH_2X_ONLY`: double size on `gp>=70` everywhere.",
        "- `SESSION_CLIPPED_0_1_2`: `2x` on high-supported `gp>=70`, `0x` on low-supported `gp<=30`, else `1x`.",
        "- `GLOBAL_CLIPPED_0_1_2`: `2x` on `gp>=70`, `0x` on `gp<=30`, else `1x` everywhere.",
        "",
        "All actions are raw integer counts per trade (`0`, `1`, or `2`). No fractional sizing is used here.",
        "",
    ]

    for scope in ["broad", "validated"]:
        sub = scope_results[scope_results["scope"] == scope].sort_values(
            ["full_delta_dollars", "full_sharpe_delta"], ascending=False
        )
        lines += [
            f"## {scope.title()} scope results",
            "",
            "| Policy | Active % | Mean contracts | Full Δ$ | Full ΔR | Sharpe Δ | MaxDD ΔR | Worst day Δ$ | Worst 5d Δ$ | Max daily risk Δ$ | IS ExpR Δ | OOS ExpR Δ | OOS retention |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in sub.iterrows():
            retain = "n/a" if pd.isna(r["oos_retention"]) else f"{r['oos_retention']:+.2f}"
            lines.append(
                f"| {r['policy']} | {r['active_pct']:.1%} | {r['mean_contracts']:.3f} | "
                f"{r['full_delta_dollars']:+.1f} | {r['full_delta_r']:+.1f} | {r['full_sharpe_delta']:+.3f} | "
                f"{r['full_max_dd_r_delta']:+.1f} | {r['worst_day_dollars_delta']:+.1f} | {r['worst_5day_dollars_delta']:+.1f} | "
                f"{r['max_daily_risk_dollars_delta']:+.1f} | {r['is_exp_r_delta']:+.4f} | {r['oos_exp_r_delta']:+.4f} | {retain} |"
            )

        best = sub.iloc[0]["policy"] if len(sub) else None
        if best is not None:
            lines += [
                "",
                f"### {scope.title()} best-policy contributions: `{best}`",
                "",
                "| Instrument | Session | Base $ | Alt $ | Δ$ |",
                "|---|---|---|---|---|",
            ]
            for _, r in contrib_tables[scope][best].head(15).iterrows():
                lines.append(
                    f"| {r['instrument']} | {r['orb_label']} | {r['base_dollars']:+.1f} | {r['alt_dollars']:+.1f} | {r['delta_dollars']:+.1f} |"
                )

    lines += [
        "",
        "## Reading the audit",
        "",
        "- `Full Δ$` and `Full ΔR` answer the total take-home question under raw discrete actions.",
        "- `Active %` shows how much of the book remains active; `Mean contracts` shows average raw action intensity.",
        "- `Sharpe Δ`, `MaxDD ΔR`, `Worst day Δ$`, and `Worst 5d Δ$` answer whether the policy improves or worsens path quality.",
        "- `Max daily risk Δ$` is a concentration proxy, not an account-breach simulation.",
        "- `OOS retention` compares OOS ExpR uplift to IS ExpR uplift. It is directional support, not clean deployment proof.",
        "",
        "## Caveats",
        "",
        "- This is still a backtest-side policy audit, not production proof.",
        "- Session-aware policies rely on raw family directional support from the regime-family audit; they do not discover new families here.",
        "- Profile translation, contract ceilings, and copier arithmetic are separate downstream questions.",
        "",
    ]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {OUTPUT_MD}")


def main() -> None:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    cells, _ = fam.build_cells()
    profiles = norm.session_profiles(cells)

    scope_frames = {}
    for scope in ["broad", "validated"]:
        rows = norm.load_scope_rows(con, scope)
        scope_frames[scope] = norm.load_scope_trades(con, rows)
    con.close()

    all_results = []
    all_contribs: dict[str, dict[str, pd.DataFrame]] = {}
    for scope, df in scope_frames.items():
        res, contrib = evaluate_scope(df, scope, profiles)
        all_results.append(res)
        all_contribs[scope] = contrib

    results_df = pd.concat(all_results, ignore_index=True)
    emit(profiles, results_df, all_contribs)


if __name__ == "__main__":
    main()
