"""Phase 2.9 framing audit — post-promotion robustness + A3 erratum.

Audits 4 MNQ COMEX_SETTLE Mode-A-discovered lanes (promoted 2026-04-11) against
three gates not explicit in the Phase 2.9 comprehensive multi-year scan:
  D1. Fire-rate stationarity across years 2019-2025 (chi-square goodness-of-fit)
  D2. Live OOS ExpR vs stored validated_setups.oos_exp_r arithmetic
  D3. Era stability per Criterion 9 (ExpR >= -0.05 per era with N>=50)
  D4. 2026 OOS t-stat and sign-match vs IS

Plus D5 (reporting only): full MNQ 2019-2025 unfiltered baseline — corrects
Phase 2.9 result doc line 295 A3 table which omitted MNQ 2019-2023 coverage
with a misleading "availability" caveat.

Pre-reg: docs/audit/hypotheses/2026-04-20-phase-2-9-framing-audit.yaml
Stage:   docs/runtime/stages/phase-2-9-framing-audit.md
Literature grounding:
  - bailey_et_al_2013_pseudo_mathematics.md (MinBTL, compensation effect)
  - harvey_liu_2015_backtesting.md (BHY under dependence)
  - pepelyshev_polunchenko_2015_cusum_sr.md (SR drift monitoring, cited future
    work; not applied here as OOS window is too short)
  - pre_registered_criteria.md Criteria 8 (OOS >= 0.40 IS) and 9 (era stability)

Canonical sources (never re-encoded):
  - pipeline.paths.GOLD_DB_PATH
  - trading_app.holdout_policy.HOLDOUT_SACRED_FROM
  - research.filter_utils.filter_signal (delegates to trading_app.config.ALL_FILTERS)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM


CELLS: list[dict] = [
    {
        "key": "OVNRNG_100_RR1.0",
        "filter": "OVNRNG_100",
        "rr": 1.0,
        "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100",
    },
    {
        "key": "OVNRNG_100_RR1.5",
        "filter": "OVNRNG_100",
        "rr": 1.5,
        "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
    },
    {
        "key": "X_MES_ATR60_RR1.0",
        "filter": "X_MES_ATR60",
        "rr": 1.0,
        "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60",
    },
    {
        "key": "X_MES_ATR60_RR1.5",
        "filter": "X_MES_ATR60",
        "rr": 1.5,
        "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60",
    },
]

ERAS: list[tuple[str, date, date]] = [
    ("era_2019_2020", date(2019, 1, 1), date(2020, 12, 31)),
    ("era_2021_2022", date(2021, 1, 1), date(2022, 12, 31)),
    ("era_2023",      date(2023, 1, 1), date(2023, 12, 31)),
    ("era_2024_2025", date(2024, 1, 1), date(2025, 12, 31)),
]


def load_trade_universe(con: duckdb.DuckDBPyConnection, rr_target: float) -> pd.DataFrame:
    """Load MNQ COMEX_SETTLE O5 E2 CB1 trades at given RR with filter-relevant columns.

    Triple-join on (trading_day, symbol, orb_minutes) per daily-features-joins.md.
    Separately LEFT JOIN MES atr_20_pct to inject cross_atr_MES_pct per the
    canonical _inject_cross_asset_atrs pattern in trading_app.strategy_discovery.
    """
    q = """
    WITH mnq_df AS (
        SELECT trading_day, symbol, orb_minutes,
               overnight_range, atr_20, atr_20_pct,
               orb_COMEX_SETTLE_break_dir
        FROM daily_features
        WHERE symbol = 'MNQ' AND orb_minutes = 5
    ),
    mes_atr AS (
        SELECT trading_day, atr_20_pct AS cross_atr_MES_pct
        FROM daily_features
        WHERE symbol = 'MES' AND orb_minutes = 5 AND atr_20_pct IS NOT NULL
    )
    SELECT o.trading_day,
           o.pnl_r,
           d.overnight_range,
           d.atr_20,
           d.atr_20_pct,
           d.orb_COMEX_SETTLE_break_dir,
           m.cross_atr_MES_pct
    FROM orb_outcomes o
    JOIN mnq_df d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    LEFT JOIN mes_atr m
      ON o.trading_day = m.trading_day
    WHERE o.symbol = 'MNQ'
      AND o.orb_label = 'COMEX_SETTLE'
      AND o.orb_minutes = 5
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target = ?
      AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day
    """
    df = con.execute(q, [rr_target]).df()
    # DuckDB returns trading_day as datetime64[us]; downstream diagnostics
    # compare against datetime.date (HOLDOUT_SACRED_FROM, era boundaries).
    # Cast to Python date for clean cross-type comparisons.
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    return df


def d1_firerate_stationarity(df: pd.DataFrame, fire_mask: np.ndarray) -> dict:
    """Chi-square goodness-of-fit for per-year fire counts under stationarity null.

    Null: filter fires at same rate every IS year.
    Reject at p < 0.05 -> filter semantic drifted (e.g., vol regime shift pushed
    more days over an absolute threshold).
    """
    work = df.copy()
    work["year"] = work["trading_day"].map(lambda d: d.year)
    work["fire"] = fire_mask
    is_df = work[work["trading_day"] < HOLDOUT_SACRED_FROM]
    table = is_df.groupby("year")["fire"].agg(["sum", "count"]).reset_index()
    table.columns = ["year", "fires", "trades"]
    total_fires = int(table["fires"].sum())
    total_trades = int(table["trades"].sum())
    if total_trades == 0 or total_fires == 0:
        return {
            "chi2": None,
            "dof": None,
            "p": None,
            "rate_overall": None,
            "per_year": table.to_dict("records"),
        }
    rate = total_fires / total_trades
    observed = table["fires"].to_numpy(dtype=float)
    expected = rate * table["trades"].to_numpy(dtype=float)
    mask = expected > 0
    if mask.sum() < 2:
        return {
            "chi2": None,
            "dof": None,
            "p": None,
            "rate_overall": float(rate),
            "per_year": table.to_dict("records"),
        }
    chi2 = float(((observed[mask] - expected[mask]) ** 2 / expected[mask]).sum())
    dof = int(mask.sum() - 1)
    p = float(stats.chi2.sf(chi2, dof))
    per_year = []
    for _, r in table.iterrows():
        trades_i = int(r["trades"])
        fires_i = int(r["fires"])
        per_year.append({
            "year": int(r["year"]),
            "fires": fires_i,
            "trades": trades_i,
            "fire_rate": fires_i / trades_i if trades_i else None,
        })
    return {
        "chi2": chi2,
        "dof": dof,
        "p": p,
        "rate_overall": float(rate),
        "per_year": per_year,
    }


def d2_oos_live_vs_stored(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    fire_mask: np.ndarray,
    strategy_id: str,
) -> dict:
    """Compare live canonical OOS ExpR to stored validated_setups.oos_exp_r.

    Integrity-guardian § 7: never trust metadata without verifying.
    Threshold: |delta| > 0.010 R -> flag metadata drift.
    """
    work = df.copy()
    work["fire"] = fire_mask
    oos = work[(work["trading_day"] >= HOLDOUT_SACRED_FROM) & (work["fire"] == 1)]
    live_exp_r = float(oos["pnl_r"].mean()) if len(oos) > 0 else None
    live_n = int(len(oos))

    stored = con.execute(
        "SELECT oos_exp_r, expectancy_r, sample_size, first_trade_day, last_trade_day "
        "FROM validated_setups WHERE strategy_id = ?",
        [strategy_id],
    ).fetchone()
    stored_oos = float(stored[0]) if stored and stored[0] is not None else None
    stored_is = float(stored[1]) if stored and stored[1] is not None else None
    stored_n = int(stored[2]) if stored and stored[2] is not None else None

    delta = None
    if live_exp_r is not None and stored_oos is not None:
        delta = live_exp_r - stored_oos
    return {
        "live_oos_exp_r": live_exp_r,
        "live_oos_n": live_n,
        "stored_oos_exp_r": stored_oos,
        "stored_is_exp_r": stored_is,
        "stored_sample_size": stored_n,
        "delta": delta,
        "pass_gate": (delta is not None and abs(delta) <= 0.010),
    }


def d3_era_stability(df: pd.DataFrame, fire_mask: np.ndarray) -> dict:
    """Era stability (Criterion 9). Pass requires every era with N>=50 to have
    ExpR >= -0.05."""
    work = df.copy()
    work["fire"] = fire_mask
    is_df = work[work["trading_day"] < HOLDOUT_SACRED_FROM]
    eras_out = []
    for name, start, end in ERAS:
        sub = is_df[
            (is_df["trading_day"] >= start)
            & (is_df["trading_day"] <= end)
            & (is_df["fire"] == 1)
        ]
        n = int(len(sub))
        expr = float(sub["pnl_r"].mean()) if n > 0 else None
        eras_out.append({
            "era": name,
            "start": str(start),
            "end": str(end),
            "n": n,
            "expr": expr,
        })
    breach = [
        e for e in eras_out
        if e["n"] >= 50 and e["expr"] is not None and e["expr"] < -0.05
    ]
    return {"eras": eras_out, "breach": breach, "pass_gate": len(breach) == 0}


def d4_oos_tstat(df: pd.DataFrame, fire_mask: np.ndarray, is_expr: float) -> dict:
    """2026 OOS t-stat + sign-match vs IS. RULE 3.1 backtesting-methodology.md."""
    work = df.copy()
    work["fire"] = fire_mask
    oos = work[(work["trading_day"] >= HOLDOUT_SACRED_FROM) & (work["fire"] == 1)]
    n = int(len(oos))
    if n < 5:
        return {
            "n": n,
            "expr": None,
            "sd": None,
            "t": None,
            "p": None,
            "sign_match": None,
            "status": "UNDERPOWERED_N_LT_5",
        }
    expr = float(oos["pnl_r"].mean())
    sd = float(oos["pnl_r"].std(ddof=1))
    t = expr * (n ** 0.5) / sd if sd > 0 else None
    p = float(stats.t.sf(abs(t), df=n - 1) * 2) if t is not None else None
    is_sign = 1 if is_expr >= 0 else -1
    oos_sign = 1 if expr >= 0 else -1
    sign_match = is_sign == oos_sign
    if sign_match:
        status = "SIGN_MATCH_N_LT_30_DIRECTIONAL_ONLY" if n < 30 else "SIGN_MATCH"
    else:
        status = "SIGN_FLIP"
    return {
        "n": n,
        "expr": expr,
        "sd": sd,
        "t": t,
        "p": p,
        "sign_match": sign_match,
        "status": status,
    }


def d5_full_mnq_baseline(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Full MNQ 2019-2025 COMEX_SETTLE O5 E2 CB1 RR1.0 unfiltered — A3 correction.

    Phase 2.9 result doc line 295 claimed "MNQ pre-2024 COMEX_SETTLE data has
    different sample_size availability (MNQ micro trades less continuous
    history before mid-2024)". Live DB shows ~249 rows/year from 2019 onwards.
    This query publishes the full baseline the doc omitted.
    """
    q = """
    SELECT CAST(EXTRACT(year FROM trading_day) AS INTEGER) AS yr,
           COUNT(*) AS n,
           AVG(pnl_r) AS expr,
           STDDEV_SAMP(pnl_r) AS sd
    FROM orb_outcomes
    WHERE symbol = 'MNQ'
      AND orb_label = 'COMEX_SETTLE'
      AND orb_minutes = 5
      AND entry_model = 'E2'
      AND confirm_bars = 1
      AND rr_target = 1.0
      AND pnl_r IS NOT NULL
      AND trading_day >= DATE '2019-01-01'
      AND trading_day < DATE '2026-01-01'
    GROUP BY yr
    ORDER BY yr
    """
    out = []
    for yr, n, expr, sd in con.execute(q).fetchall():
        n = int(n)
        expr = float(expr) if expr is not None else None
        sd = float(sd) if sd is not None else None
        t = None
        p = None
        if expr is not None and sd is not None and sd > 0 and n > 1:
            t = expr * (n ** 0.5) / sd
            p = float(stats.t.sf(abs(t), df=n - 1) * 2)
        out.append({"year": int(yr), "n": n, "expr": expr, "sd": sd, "t": t, "p": p})
    return out


def assign_verdict(d1: dict, d3: dict, d4: dict) -> str:
    if d4["sign_match"] is False:
        return "FREEZE"
    if not d3["pass_gate"]:
        return "FLAG_ERA_DEPENDENT"
    d1_pass = d1["p"] is None or d1["p"] >= 0.05
    if not d1_pass:
        return "WATCH_FIRE_RATE_DRIFT"
    if d4["status"] == "UNDERPOWERED_N_LT_5":
        return "DEFER_OOS_UNDERPOWERED"
    if d4["status"].endswith("DIRECTIONAL_ONLY"):
        return "CONTINUE_OOS_THIN"
    return "CONTINUE"


def _jsonify(o):
    if isinstance(o, (date,)):
        return str(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="research/output")
    parser.add_argument("--seed", type=int, default=20260420)
    args = parser.parse_args(argv)

    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    cutoff_row = con.execute(
        "SELECT MAX(trading_day) FROM orb_outcomes "
        "WHERE symbol = 'MNQ' AND orb_label = 'COMEX_SETTLE'"
    ).fetchone()
    cutoff = cutoff_row[0] if cutoff_row else None
    print(f"data_cutoff_mnq_comex_settle={cutoff}")
    print(f"holdout_sacred_from={HOLDOUT_SACRED_FROM}")

    results: dict = {
        "metadata": {
            "data_cutoff": str(cutoff),
            "holdout_sacred_from": str(HOLDOUT_SACRED_FROM),
            "seed": args.seed,
        }
    }

    for cell in CELLS:
        print(f"\n==== {cell['key']} ({cell['strategy_id']}) ====")
        df = load_trade_universe(con, cell["rr"])
        fire_mask = filter_signal(df, cell["filter"], orb_label="COMEX_SETTLE")
        assert len(fire_mask) == len(df), "fire_mask length mismatch"
        total_rows = len(df)
        total_fires = int(fire_mask.sum())
        print(f"total_rows={total_rows} total_fires={total_fires}")

        is_row_mask = df["trading_day"] < HOLDOUT_SACRED_FROM
        is_fires_mask = (fire_mask == 1) & is_row_mask
        is_n = int(is_fires_mask.sum())
        is_expr = float(df.loc[is_fires_mask, "pnl_r"].mean()) if is_n > 0 else 0.0
        print(f"IS: n={is_n} expr={is_expr:.4f}")

        d1 = d1_firerate_stationarity(df, fire_mask)
        d2 = d2_oos_live_vs_stored(con, df, fire_mask, cell["strategy_id"])
        d3 = d3_era_stability(df, fire_mask)
        d4 = d4_oos_tstat(df, fire_mask, is_expr)
        verdict = assign_verdict(d1, d3, d4)

        results[cell["key"]] = {
            "strategy_id": cell["strategy_id"],
            "filter": cell["filter"],
            "rr": cell["rr"],
            "is_n": is_n,
            "is_expr": is_expr,
            "total_rows": total_rows,
            "total_fires": total_fires,
            "d1_firerate_stationarity": d1,
            "d2_live_vs_stored_oos": d2,
            "d3_era_stability": d3,
            "d4_oos_tstat": d4,
            "verdict": verdict,
        }
        print(f"d1_chi2_p={d1['p']}")
        print(f"d2_delta={d2['delta']}  d2_pass={d2['pass_gate']}")
        print(f"d3_breaches={len(d3['breach'])}  d3_pass={d3['pass_gate']}")
        print(f"d4_n={d4['n']} d4_expr={d4['expr']} d4_t={d4['t']} d4_status={d4['status']}")
        print(f"verdict={verdict}")

    print("\n==== D5: full MNQ 2019-2025 unfiltered baseline (A3 correction) ====")
    d5 = d5_full_mnq_baseline(con)
    for r in d5:
        print(r)
    results["d5_full_mnq_baseline"] = d5

    main_csv = out_dir / "phase_2_9_framing_audit_main.csv"
    fields = [
        "cell", "strategy_id", "filter", "rr",
        "is_n", "is_expr",
        "d1_chi2_p", "d1_rate_overall",
        "d2_live_oos_expr", "d2_live_oos_n", "d2_stored_oos_expr", "d2_delta", "d2_pass",
        "d3_breaches", "d3_pass",
        "d4_n", "d4_expr", "d4_t", "d4_p", "d4_sign_match", "d4_status",
        "verdict",
    ]
    with open(main_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key, r in results.items():
            if key in ("metadata", "d5_full_mnq_baseline"):
                continue
            row = {
                "cell": key,
                "strategy_id": r["strategy_id"],
                "filter": r["filter"],
                "rr": r["rr"],
                "is_n": r["is_n"],
                "is_expr": round(r["is_expr"], 4),
                "d1_chi2_p": round(r["d1_firerate_stationarity"]["p"], 4)
                    if r["d1_firerate_stationarity"]["p"] is not None else "",
                "d1_rate_overall": round(r["d1_firerate_stationarity"]["rate_overall"], 4)
                    if r["d1_firerate_stationarity"]["rate_overall"] is not None else "",
                "d2_live_oos_expr": round(r["d2_live_vs_stored_oos"]["live_oos_exp_r"], 4)
                    if r["d2_live_vs_stored_oos"]["live_oos_exp_r"] is not None else "",
                "d2_live_oos_n": r["d2_live_vs_stored_oos"]["live_oos_n"],
                "d2_stored_oos_expr": round(r["d2_live_vs_stored_oos"]["stored_oos_exp_r"], 4)
                    if r["d2_live_vs_stored_oos"]["stored_oos_exp_r"] is not None else "",
                "d2_delta": round(r["d2_live_vs_stored_oos"]["delta"], 4)
                    if r["d2_live_vs_stored_oos"]["delta"] is not None else "",
                "d2_pass": r["d2_live_vs_stored_oos"]["pass_gate"],
                "d3_breaches": len(r["d3_era_stability"]["breach"]),
                "d3_pass": r["d3_era_stability"]["pass_gate"],
                "d4_n": r["d4_oos_tstat"]["n"],
                "d4_expr": round(r["d4_oos_tstat"]["expr"], 4)
                    if r["d4_oos_tstat"]["expr"] is not None else "",
                "d4_t": round(r["d4_oos_tstat"]["t"], 3)
                    if r["d4_oos_tstat"]["t"] is not None else "",
                "d4_p": round(r["d4_oos_tstat"]["p"], 4)
                    if r["d4_oos_tstat"]["p"] is not None else "",
                "d4_sign_match": r["d4_oos_tstat"]["sign_match"],
                "d4_status": r["d4_oos_tstat"]["status"],
                "verdict": r["verdict"],
            }
            w.writerow(row)
    print(f"\nwrote {main_csv}")

    json_path = out_dir / "phase_2_9_framing_audit.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=_jsonify)
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
