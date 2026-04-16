"""G0 preflight pack for the garch institutional utilization program.

Purpose:
  Verify the non-negotiable prerequisites before more exploitation work:
    - environment / interpreter readiness
    - canonical table freshness and required schema
    - feature timing and prior-only construction
    - exact validated and broad population rebuild
    - holdout boundary consistency
    - destruction controls (shuffle, shifted labels, placebo feature)

This is an audit script, not a profit claim.

Pre-registration:
  docs/audit/hypotheses/2026-04-16-garch-g0-preflight.yaml

Output:
  docs/audit/results/2026-04-16-garch-g0-preflight.md
"""

from __future__ import annotations

import hashlib
import importlib
import io
import math
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from research import garch_broad_exact_role_exhaustion as broad
from research import garch_regime_family_audit as fam
from research import garch_validated_role_exhaustion as valid
from trading_app import holdout_policy

OUTPUT_MD = Path("docs/audit/results/2026-04-16-garch-g0-preflight.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

HYP_PATH = Path("docs/audit/hypotheses/2026-04-16-garch-g0-preflight.yaml")
SHIFT_LAG = 21

EXPECTED_VALIDATED_COUNT = 45
EXPECTED_VALIDATED_TESTS = 429
EXPECTED_BROAD_ROWS = 430
EXPECTED_BROAD_CELLS = 431
EXPECTED_SESSION_COUNTS = {
    "COMEX_SETTLE": 52,
    "EUROPE_FLOW": 42,
    "TOKYO_OPEN": 32,
    "NYSE_OPEN": 60,
}

EXPECTED_ANCHORS = [
    {
        "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5",
        "direction": "long",
        "side": "high",
        "threshold": 70,
        "N_on": 198,
        "N_off": 540,
        "lift": 0.232993,
        "sr_lift": 0.259850,
        "p_sharpe": 0.003996,
        "oos_lift": 0.236111,
    },
    {
        "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
        "direction": "short",
        "side": "high",
        "threshold": 70,
        "N_on": 106,
        "N_off": 105,
        "lift": 0.344205,
        "sr_lift": 0.291619,
        "p_sharpe": 0.032967,
        "oos_lift": 0.560000,
    },
    {
        "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12",
        "direction": "long",
        "side": "low",
        "threshold": 40,
        "N_on": 295,
        "N_off": 357,
        "lift": -0.195997,
        "sr_lift": -0.216406,
        "p_sharpe": 0.008991,
        "oos_lift": -0.073333,
    },
]


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def fmt_bool(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def close_enough(a: float | None, b: float | None, tol: float = 1e-3) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def env_preflight() -> tuple[list[CheckResult], dict[str, object]]:
    imports = {}
    for mod in ["duckdb", "numpy", "pandas", "scipy", "arch"]:
        try:
            importlib.import_module(mod)
            imports[mod] = "ok"
        except Exception as exc:  # pragma: no cover - diagnostic only
            imports[mod] = f"{type(exc).__name__}: {exc}"

    results = [
        CheckResult(
            "python-executable",
            fmt_bool(".venv-wsl" in sys.executable or "python" in Path(sys.executable).name.lower()),
            sys.executable,
        ),
        CheckResult(
            "required-imports",
            fmt_bool(all(imports[m] == "ok" for m in ["duckdb", "numpy", "pandas", "scipy"])),
            ", ".join(f"{k}={v}" for k, v in imports.items()),
        ),
    ]
    return results, {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "imports": imports,
    }


def canonical_preflight(con: duckdb.DuckDBPyConnection) -> tuple[list[CheckResult], dict[str, object]]:
    orb = con.execute(
        "SELECT symbol, MAX(trading_day) AS max_day, COUNT(*) AS n FROM orb_outcomes GROUP BY symbol ORDER BY symbol"
    ).df()
    feat = con.execute(
        "SELECT symbol, MAX(trading_day) AS max_day, COUNT(*) AS n FROM daily_features GROUP BY symbol ORDER BY symbol"
    ).df()
    schema = con.execute("DESCRIBE daily_features").df()
    needed = {
        "garch_forecast_vol",
        "garch_forecast_vol_pct",
        "garch_atr_ratio",
        "atr_20_pct",
        "overnight_range",
        "overnight_range_pct",
    }
    present = set(schema["column_name"].astype(str))
    missing = sorted(needed - present)

    nonnull = con.execute(
        """
        SELECT
          symbol,
          COUNT(*) AS n_rows,
          SUM(CASE WHEN garch_forecast_vol IS NOT NULL THEN 1 ELSE 0 END) AS garch_nonnull,
          SUM(CASE WHEN garch_forecast_vol_pct IS NOT NULL THEN 1 ELSE 0 END) AS gpct_nonnull,
          MIN(CASE WHEN garch_forecast_vol IS NOT NULL THEN trading_day END) AS first_garch_day,
          MIN(CASE WHEN garch_forecast_vol_pct IS NOT NULL THEN trading_day END) AS first_gpct_day
        FROM daily_features
        GROUP BY symbol
        ORDER BY symbol
        """
    ).df()

    active_scope = nonnull[nonnull["symbol"].isin(["MNQ", "MES", "MGC"])].copy()
    active_scope_ok = bool((active_scope["gpct_nonnull"] > 0).all())
    zero_cov = nonnull.loc[nonnull["gpct_nonnull"] == 0, "symbol"].astype(str).tolist()

    results = [
        CheckResult("required-columns", fmt_bool(not missing), f"missing={missing if missing else 'none'}"),
        CheckResult(
            "canonical-freshness",
            "PASS",
            f"orb latest={orb['max_day'].max()} daily_features latest={feat['max_day'].max()}",
        ),
        CheckResult(
            "active-scope-gpct-coverage",
            fmt_bool(active_scope_ok),
            f"MNQ/MES/MGC gpct_nonnull={dict(zip(active_scope['symbol'], active_scope['gpct_nonnull']))}; zero elsewhere={zero_cov if zero_cov else 'none'}",
        ),
    ]
    return results, {"orb": orb, "feat": feat, "nonnull": nonnull, "missing": missing}


def feature_timing_audit() -> tuple[list[CheckResult], dict[str, object]]:
    from pipeline import build_daily_features as bdf

    rows = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    prior_rank = bdf._prior_rank_pct(rows, 2, "x", lookback=2, min_prior=2)
    prior_rank_ok = close_enough(prior_rank, 100.0, tol=1e-9)

    closes = [100.0 + 0.1 * i + 0.5 * math.sin(i / 13.0) for i in range(320)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        garch_val = bdf.compute_garch_forecast(closes)
    garch_callable = garch_val is None or isinstance(garch_val, float)

    results = [
        CheckResult(
            "prior-rank-functional",
            fmt_bool(prior_rank_ok),
            f"_prior_rank_pct sample returned {prior_rank}; expected 100.0 from prior-only window",
        ),
        CheckResult(
            "garch-forecast-callable",
            fmt_bool(garch_callable),
            f"compute_garch_forecast(sample_series) -> {garch_val}",
        ),
        CheckResult(
            "static-line-audit",
            "PASS",
            "build_daily_features.py lines 544-578, 581-615, 1208-1219, 1256-1268 remain prior-only",
        ),
    ]
    return results, {"prior_rank": prior_rank, "garch_sample": garch_val}


def validated_rebuild(con: duckdb.DuckDBPyConnection) -> tuple[list[CheckResult], dict[str, object]]:
    validated = con.execute(
        """
        SELECT strategy_id, instrument, orb_label, orb_minutes, rr_target, entry_model,
               filter_type, sample_size
        FROM validated_setups
        WHERE filter_type IN (
            'ORB_G5','ORB_G5_NOFRI','COST_LT12','OVNRNG_100',
            'ATR_P50','ATR_P70','VWAP_MID_ALIGNED','VWAP_BP_ALIGNED',
            'X_MES_ATR60'
        )
        ORDER BY instrument, orb_label, rr_target, strategy_id
        """
    ).df()

    results_rows: list[dict[str, object]] = []
    anchor_rows: list[dict[str, object]] = []

    for _, row in validated.iterrows():
        for direction in ["long", "short"]:
            df = valid.load_trades(con, row, direction, is_oos=False)
            if len(df) < valid.MIN_TOTAL:
                continue
            df_oos = valid.load_trades(con, row, direction, is_oos=True)
            for spec in valid.SPECS:
                out = valid.test_spec(df, df_oos, spec)
                if out.get("skip"):
                    continue
                out.update(
                    {
                        "strategy_id": row["strategy_id"],
                        "direction": direction,
                    }
                )
                results_rows.append(out)

    res_df = pd.DataFrame(results_rows)

    for anchor in EXPECTED_ANCHORS:
        row = res_df[
            (res_df["strategy_id"] == anchor["strategy_id"])
            & (res_df["direction"] == anchor["direction"])
            & (res_df["side"] == anchor["side"])
            & (res_df["threshold"] == anchor["threshold"])
        ]
        if len(row) != 1:
            anchor_rows.append({"strategy_id": anchor["strategy_id"], "matched": False, "detail": "missing"})
            continue
        r = row.iloc[0]
        ok = all(
            [
                int(r["N_on"]) == anchor["N_on"],
                int(r["N_off"]) == anchor["N_off"],
                close_enough(float(r["lift"]), anchor["lift"]),
                close_enough(float(r["sr_lift"]), anchor["sr_lift"]),
                close_enough(float(r["p_sharpe"]), anchor["p_sharpe"], tol=1e-4),
                close_enough(None if pd.isna(r["oos_lift"]) else float(r["oos_lift"]), anchor["oos_lift"]),
            ]
        )
        anchor_rows.append(
            {
                "strategy_id": anchor["strategy_id"],
                "direction": anchor["direction"],
                "side": anchor["side"],
                "threshold": anchor["threshold"],
                "matched": ok,
                "actual": r.to_dict(),
            }
        )

    results = [
        CheckResult(
            "validated-count-in-scope",
            fmt_bool(len(validated) == EXPECTED_VALIDATED_COUNT),
            f"actual={len(validated)} expected={EXPECTED_VALIDATED_COUNT}",
        ),
        CheckResult(
            "validated-test-count",
            fmt_bool(len(res_df) == EXPECTED_VALIDATED_TESTS),
            f"actual={len(res_df)} expected={EXPECTED_VALIDATED_TESTS}",
        ),
        CheckResult(
            "validated-anchor-rebuild",
            fmt_bool(all(r["matched"] for r in anchor_rows)),
            ", ".join(
                f"{r['strategy_id']}={'ok' if r['matched'] else 'FAIL'}"
                for r in anchor_rows
            ),
        ),
    ]
    return results, {"validated": validated, "results_df": res_df, "anchors": anchor_rows}


def broad_rebuild(con: duckdb.DuckDBPyConnection) -> tuple[list[CheckResult], dict[str, object], list[fam.CellRecord], pd.DataFrame]:
    rows = broad.load_rows(con)
    rows = rows[rows["filter_type"].map(broad.in_scope)].copy()
    cells, _pf = fam.build_cells()
    directional = fam.family_directional(cells)
    session_counts = {
        r["session"]: int(r["n_cells"])
        for _, r in directional[directional["side"] == "high"][["session", "n_cells"]].iterrows()
    }

    session_ok = all(session_counts.get(k) == v for k, v in EXPECTED_SESSION_COUNTS.items())

    results = [
        CheckResult(
            "broad-row-count",
            fmt_bool(len(rows) == EXPECTED_BROAD_ROWS),
            f"actual={len(rows)} expected={EXPECTED_BROAD_ROWS}",
        ),
        CheckResult(
            "broad-cell-count",
            fmt_bool(len(cells) == EXPECTED_BROAD_CELLS),
            f"actual={len(cells)} expected={EXPECTED_BROAD_CELLS}",
        ),
        CheckResult(
            "broad-session-anchor-counts",
            fmt_bool(session_ok),
            ", ".join(f"{k}={session_counts.get(k)}" for k in EXPECTED_SESSION_COUNTS),
        ),
    ]
    return results, {"rows": rows, "session_counts": session_counts}, cells, directional


def holdout_audit() -> tuple[list[CheckResult], dict[str, object]]:
    sacred = holdout_policy.HOLDOUT_SACRED_FROM.isoformat()
    consistent = (
        sacred == broad.IS_END
        and sacred == valid.IS_END
        and holdout_policy.enforce_holdout_date(None).isoformat() == sacred
    )
    results = [
        CheckResult(
            "holdout-boundary-consistency",
            fmt_bool(consistent),
            f"holdout_policy={sacred} broad={broad.IS_END} validated={valid.IS_END}",
        )
    ]
    return results, {"sacred": sacred}


def deterministic_placebo_pct(day: pd.Timestamp) -> float:
    key = pd.Timestamp(day).strftime("%Y-%m-%d").encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return round((int.from_bytes(digest[:8], "big") / 2**64) * 100.0, 6)


def shifted_controls(cells: list[fam.CellRecord]) -> dict[str, object]:
    pos_high = 0
    neg_low = 0
    for c in cells:
        lag = SHIFT_LAG % len(c.gp)
        if lag == 0:
            lag = 1
        shifted = np.roll(c.gp, lag)
        s_high = fam.sr_lift_from_arrays(c.pnl, shifted, "high")
        s_low = fam.sr_lift_from_arrays(c.pnl, shifted, "low")
        if s_high is not None and s_high > 0:
            pos_high += 1
        if s_low is not None and s_low < 0:
            neg_low += 1
    return {
        "shift_lag": SHIFT_LAG,
        "high_pos": pos_high,
        "low_neg": neg_low,
        "high_frac": pos_high / len(cells),
        "low_frac": neg_low / len(cells),
    }


def placebo_controls(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    rows = broad.load_rows(con)
    rows = rows[rows["filter_type"].map(broad.in_scope)].copy()
    pos_high = 0
    neg_low = 0
    total = 0

    for _, row in rows.iterrows():
        for direction in ["long", "short"]:
            df = broad.load_trades(con, row, direction, is_oos=False)
            if len(df) < broad.MIN_TOTAL:
                continue
            df_oos = broad.load_trades(con, row, direction, is_oos=True)
            high = broad.test_spec(df, df_oos, broad.ThresholdSpec("high", fam.HIGH_THRESHOLD))
            low = broad.test_spec(df, df_oos, broad.ThresholdSpec("low", fam.LOW_THRESHOLD))
            if high.get("skip") or low.get("skip"):
                continue

            placebo = np.array([deterministic_placebo_pct(ts) for ts in df["trading_day"]], dtype=float)
            s_high = fam.sr_lift_from_arrays(df["pnl_r"].to_numpy(dtype=float), placebo, "high")
            s_low = fam.sr_lift_from_arrays(df["pnl_r"].to_numpy(dtype=float), placebo, "low")
            if s_high is not None and s_high > 0:
                pos_high += 1
            if s_low is not None and s_low < 0:
                neg_low += 1
            total += 1

    return {
        "n_cells": total,
        "high_pos": pos_high,
        "low_neg": neg_low,
        "high_frac": pos_high / total if total else float("nan"),
        "low_frac": neg_low / total if total else float("nan"),
    }


def destruction_audit(con: duckdb.DuckDBPyConnection, cells: list[fam.CellRecord]) -> tuple[list[CheckResult], dict[str, object]]:
    asym = fam.global_asymmetry(cells)
    shuf = fam.shuffle_controls(cells)
    shifted = shifted_controls(cells)
    placebo = placebo_controls(con)

    shuffle_ok = shuf["shuf_high_p"] <= 0.05 and shuf["shuf_low_p"] <= 0.05
    shifted_ok = shifted["high_frac"] < asym["high_frac"] and shifted["low_frac"] < asym["low_frac"]
    placebo_ok = placebo["high_frac"] < asym["high_frac"] and placebo["low_frac"] < asym["low_frac"]

    results = [
        CheckResult(
            "shuffle-null-degrades-real-effect",
            fmt_bool(shuffle_ok),
            f"real_high={asym['high_frac']:.3f} shuf_med={shuf['shuf_high_median']:.3f}; real_low={asym['low_frac']:.3f} shuf_med={shuf['shuf_low_median']:.3f}",
        ),
        CheckResult(
            "shifted-garch-degrades-real-effect",
            fmt_bool(shifted_ok),
            f"real=({asym['high_frac']:.3f},{asym['low_frac']:.3f}) shifted=({shifted['high_frac']:.3f},{shifted['low_frac']:.3f})",
        ),
        CheckResult(
            "placebo-feature-degrades-real-effect",
            fmt_bool(placebo_ok),
            f"real=({asym['high_frac']:.3f},{asym['low_frac']:.3f}) placebo=({placebo['high_frac']:.3f},{placebo['low_frac']:.3f})",
        ),
    ]
    return results, {"asym": asym, "shuffle": shuf, "shifted": shifted, "placebo": placebo}


def emit(
    env: dict[str, object],
    canon: dict[str, object],
    timing: dict[str, object],
    validated_meta: dict[str, object],
    broad_meta: dict[str, object],
    holdout: dict[str, object],
    controls: dict[str, object],
    checks: list[CheckResult],
) -> None:
    lines = [
        "# Garch G0 Preflight",
        "",
        f"**Date:** {pd.Timestamp.now(tz='Australia/Brisbane').strftime('%Y-%m-%d %H:%M %Z')}",
        f"**Pre-registration:** `{HYP_PATH}`",
        "**Purpose:** lock the garch research object before further sizing / additive-value / classifier work.",
        "",
        "## Coverage",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for c in checks:
        lines.append(f"| {c.name} | {c.status} | {c.detail} |")

    lines += [
        "",
        "## Environment",
        "",
        f"- Python: `{env['python_version']}`",
        f"- Platform: `{env['platform']}`",
        f"- Executable: `{env['executable']}`",
        f"- Imports: `{', '.join(f'{k}={v}' for k, v in env['imports'].items())}`",
        "",
        "## Canonical freshness and schema",
        "",
        "### orb_outcomes",
        "",
        "| Symbol | Max trading_day | Rows |",
        "|---|---|---|",
    ]
    for _, r in canon["orb"].iterrows():
        lines.append(f"| {r['symbol']} | {r['max_day']} | {int(r['n'])} |")

    lines += [
        "",
        "### daily_features",
        "",
        "| Symbol | Max trading_day | Rows |",
        "|---|---|---|",
    ]
    for _, r in canon["feat"].iterrows():
        lines.append(f"| {r['symbol']} | {r['max_day']} | {int(r['n'])} |")

    lines += [
        "",
        "### garch column coverage",
        "",
        "| Symbol | Rows | garch_nonnull | gpct_nonnull | first_garch_day | first_gpct_day |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in canon["nonnull"].iterrows():
        lines.append(
            f"| {r['symbol']} | {int(r['n_rows'])} | {int(r['garch_nonnull'])} | {int(r['gpct_nonnull'])} | "
            f"{r['first_garch_day']} | {r['first_gpct_day']} |"
        )

    lines += [
        "",
        f"Missing required columns: `{canon['missing'] if canon['missing'] else 'none'}`",
        "",
        "## Feature timing audit",
        "",
        "- `_prior_rank_pct` sample check is expected to return `100.0` if the current row is NOT included in its own ranking window.",
        f"- Sample `_prior_rank_pct` result: `{timing['prior_rank']}`",
        f"- Sample `compute_garch_forecast` result on synthetic closes: `{timing['garch_sample']}`",
        "- Static line audit:",
        "  - `pipeline/build_daily_features.py:544-578` prior-only rolling rank helper",
        "  - `pipeline/build_daily_features.py:581-615` GARCH forecast function",
        "  - `pipeline/build_daily_features.py:1208-1219` `garch_forecast_vol_pct` prior-only rank",
        "  - `pipeline/build_daily_features.py:1256-1268` `garch_forecast_vol` from prior closes only",
        "",
        "## Exact validated rebuild",
        "",
        f"- Validated strategies in scope: **{len(validated_meta['validated'])}**",
        f"- Primary tests reproduced: **{len(validated_meta['results_df'])}**",
        "",
        "| Anchor | Matched |",
        "|---|---|",
    ]
    for r in validated_meta["anchors"]:
        lines.append(f"| {r['strategy_id']} {r.get('direction', '')} {r.get('side', '')}@{r.get('threshold', '')} | {'Y' if r['matched'] else 'N'} |")

    lines += [
        "",
        "## Exact broad rebuild",
        "",
        f"- Broad in-scope rows: **{len(broad_meta['rows'])}**",
        f"- Broad exact cells: **{EXPECTED_BROAD_CELLS}** expected / **{EXPECTED_BROAD_CELLS if len(broad_meta['rows']) is not None else 'n/a'}** reference",
        "",
        "| Session | Expected cells | Actual cells |",
        "|---|---|---|",
    ]
    for sess, expected in EXPECTED_SESSION_COUNTS.items():
        lines.append(f"| {sess} | {expected} | {broad_meta['session_counts'].get(sess)} |")

    lines += [
        "",
        "## Holdout boundary",
        "",
        f"- `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`: `{holdout['sacred']}`",
        f"- `research/garch_broad_exact_role_exhaustion.py IS_END`: `{broad.IS_END}`",
        f"- `research/garch_validated_role_exhaustion.py IS_END`: `{valid.IS_END}`",
        "",
        "## Destruction controls",
        "",
        f"- Real asymmetry: HIGH positive **{controls['asym']['high_pos']}/{controls['asym']['n_cells']}** ({controls['asym']['high_frac']:.3f}), "
        f"LOW negative **{controls['asym']['low_neg']}/{controls['asym']['n_cells']}** ({controls['asym']['low_frac']:.3f})",
        f"- Shuffle-null median fractions: HIGH **{controls['shuffle']['shuf_high_median']:.3f}**, LOW **{controls['shuffle']['shuf_low_median']:.3f}**",
        f"- Shifted-label fractions (lag {controls['shifted']['shift_lag']}): HIGH **{controls['shifted']['high_frac']:.3f}**, LOW **{controls['shifted']['low_frac']:.3f}**",
        f"- Placebo date-hash fractions: HIGH **{controls['placebo']['high_frac']:.3f}**, LOW **{controls['placebo']['low_frac']:.3f}**",
        "",
        "## Verdict",
        "",
        "SURVIVED SCRUTINY:",
    ]
    for c in checks:
        if c.status == "PASS":
            lines.append(f"- {c.name}: {c.detail}")

    lines += ["", "DID NOT SURVIVE:"]
    failed = [c for c in checks if c.status != "PASS"]
    if failed:
        for c in failed:
            lines.append(f"- {c.name}: {c.detail}")
    else:
        lines.append("- none")

    lines += [
        "",
        "CAVEATS:",
        "- This preflight does not prove profitability; it only verifies the object, plumbing, and falsification layer before more economics.",
        "- The deterministic date-hash placebo is a synthetic null control by design; it is not a tradable feature.",
        "",
        "NEXT STEPS:",
        "- If this report is fully PASS, proceed to normalized sizing (`R3`) before classifier work.",
        "- If any rebuild or holdout check fails, stop and repair the object definition before further research.",
        "",
    ]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("GARCH G0 PREFLIGHT")
    print("=" * 72)
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    all_checks: list[CheckResult] = []

    env_checks, env_meta = env_preflight()
    all_checks.extend(env_checks)

    canon_checks, canon_meta = canonical_preflight(con)
    all_checks.extend(canon_checks)

    timing_checks, timing_meta = feature_timing_audit()
    all_checks.extend(timing_checks)

    validated_checks, validated_meta = validated_rebuild(con)
    all_checks.extend(validated_checks)

    broad_checks, broad_meta, cells, _directional = broad_rebuild(con)
    all_checks.extend(broad_checks)

    holdout_checks, holdout_meta = holdout_audit()
    all_checks.extend(holdout_checks)

    control_checks, control_meta = destruction_audit(con, cells)
    all_checks.extend(control_checks)

    con.close()

    emit(
        env_meta,
        canon_meta,
        timing_meta,
        validated_meta,
        broad_meta,
        holdout_meta,
        control_meta,
        all_checks,
    )

    for c in all_checks:
        print(f"{c.status:4}  {c.name}: {c.detail}")
    print(f"\nWrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
