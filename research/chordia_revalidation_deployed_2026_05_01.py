"""Chordia revalidation of currently-deployed lanes (2026-05-01).

Pre-reg: docs/audit/hypotheses/2026-05-01-chordia-revalidation-deployed-lanes.yaml
Result : docs/audit/results/2026-05-01-chordia-revalidation-deployed-lanes.md

Defensive honesty gate per pre_registered_criteria.md Criterion 4. Per-lane
Pathway-B K=1 individual hypothesis tests on the 4 currently-deployed lanes
(rebalance 2026-04-18 lane_allocation snapshot, BEFORE the 2026-05-01
rebalance proposal that surfaces O15/O30 lanes from the PR #189 fix).

Per-lane verdicts only (pooled_finding: false). No allocator code touched.

Methodology: triple-join orb_outcomes x daily_features on
(trading_day, symbol, orb_minutes), apply canonical StrategyFilter.matches_df
to derive fire mask on IS data only (trading_day < HOLDOUT_SACRED_FROM),
compute per-trade R Sharpe, t = sharpe * sqrt(N) via canonical
trading_app.chordia.compute_chordia_t.

Scratch policy: COALESCE(pnl_r, 0.0) -- never WHERE pnl_r IS NOT NULL
(would silently drop ~0.3% of rows and inflate ExpR 10-45% per
memory/feedback_scratch_pnl_null_class_bug.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import duckdb
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal
from trading_app.chordia import (
    CHORDIA_T_WITH_THEORY,
    CHORDIA_T_WITHOUT_THEORY,
    chordia_threshold,
    compute_chordia_t,
)
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM


@dataclass(frozen=True)
class Lane:
    hyp_id: str
    strategy_id: str
    instrument: str
    orb_label: str
    orb_minutes: int
    entry_model: str
    confirm_bars: int
    rr_target: float
    filter_key: str
    has_theory: bool


# Locked at pre-reg time -- the 4 deployed lanes per
# docs/runtime/lane_allocation.json rebalance_date 2026-04-18.
#
# has_theory FLAG -- AUDITED 2026-05-01 (v3 steel-man pass) against actual
# local extracts in docs/institutional/literature/.
#
# Audit history:
#   v1 (pre-reg locked): all 4 has_theory=True citing "Fitschen Ch 5-7 + Chan".
#   v2 (after Fitschen PDF extraction): all 4 has_theory=False because Fitschen
#       Ch 5-7 cite was confirmed phantom (Ch 5=Exits, Ch 6=Filters with
#       OPPOSITE polarity, Ch 7=Money-Mgmt -- none cover session mechanisms).
#   v3 (this version, steel-man on Chan Ch 7 grounding):
#       - Chan 2013 Ch 7 p.155-157 grounds STOP-CASCADE BREAKOUT mechanism on
#         equity-index futures with verbatim "entry at the market open"
#         framing. The FSTX case study (p.156) is a EUROPEAN equity-index
#         future trading session-open momentum at Sharpe 1.4.
#       - Lanes whose (entry mechanism + instrument class + session class)
#         match the Chan Ch 7 + FSTX pattern get has_theory=True. Filter
#         overlay grounding is separate; the lane-as-whole inherits theory
#         status from the entry mechanism per the project's binary
#         has_theory model in trading_app/chordia.py.
#
# Per-lane v3 honest grounding:
#   H1 EUROPE_FLOW (MNQ, equity-index, EU session-open):
#       has_theory=True. Direct match to Chan FSTX example -- European-session
#       equity-index future at session open with stop-cascade breakout entry.
#       The strongest mechanism-to-literature match of the four lanes.
#   H2 COMEX_SETTLE (MNQ, equity-index, COMEX-close session):
#       has_theory=False. COMEX_SETTLE is a metals-market settlement window;
#       Chan Ch 7 stop-cascade-at-open doesn't apply to a close-driven
#       session. No literature support found in local extracts.
#   H3 NYSE_OPEN (MNQ, equity-index, US session-open):
#       has_theory=True. NYSE_OPEN is the canonical equity-index session
#       open. Chan Ch 7 p.155 names "entry at the market open" verbatim;
#       FSTX example generalizes by class.
#   H4 TOKYO_OPEN (MNQ, equity-index, Asian session-open):
#       has_theory=False. Equity-index session-open BUT Chan FSTX example
#       is European, not Asian. Generalization by region is a stretch
#       Chan doesn't make. Conservative reading: not grounded.
#
# See for full audit:
#   - docs/institutional/literature/fitschen_2013_path_of_least_resistance.md
#     (Ch 5/6/7 extension and citation-drift audit)
#   - docs/institutional/literature/chan_2013_ch7_intraday_momentum.md
#     (FSTX case study + stop-cascade mechanism)
#   - docs/runtime/chordia-revalidation-decision-audit-2026-05-01.md
#     (three-decision audit, Decision 3 = steel-man)
LANES: list[Lane] = [
    Lane("H1", "MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5",
         "MNQ", "EUROPE_FLOW", 5, "E2", 1, 1.5, "ORB_G5", True),
    Lane("H2", "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5",
         "MNQ", "COMEX_SETTLE", 5, "E2", 1, 1.5, "ORB_G5", False),
    Lane("H3", "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
         "MNQ", "NYSE_OPEN", 5, "E2", 1, 1.0, "COST_LT12", True),
    Lane("H4", "MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12",
         "MNQ", "TOKYO_OPEN", 5, "E2", 1, 1.5, "COST_LT12", False),
]


def load_lane_universe(con: duckdb.DuckDBPyConnection, lane: Lane) -> pd.DataFrame:
    """Triple-join orb_outcomes x daily_features for one lane on IS slice.

    Returns one row per trading_day where the lane could have fired
    (entry_model+rr+confirm_bars matches), with all daily_features columns
    needed for the canonical StrategyFilter to score the fire mask.

    Scratch policy: keeps pnl_r NULL rows; caller COALESCEs to 0.0.
    """
    sql = """
        SELECT
            o.trading_day,
            o.symbol,
            o.orb_label,
            o.orb_minutes,
            o.entry_model,
            o.confirm_bars,
            o.rr_target,
            o.target_price,
            o.stop_price,
            o.pnl_r,
            d.*
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.trading_day = d.trading_day
           AND o.symbol      = d.symbol
           AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol       = ?
          AND o.orb_label    = ?
          AND o.orb_minutes  = ?
          AND o.entry_model  = ?
          AND o.confirm_bars = ?
          AND o.rr_target    = ?
          AND o.trading_day  < ?
    """
    return con.execute(
        sql,
        [lane.instrument, lane.orb_label, lane.orb_minutes,
         lane.entry_model, lane.confirm_bars, lane.rr_target,
         HOLDOUT_SACRED_FROM],
    ).df()


def chordia_for_lane(con: duckdb.DuckDBPyConnection, lane: Lane) -> dict:
    """Apply canonical filter, compute Chordia t-stat on IS, return verdict dict."""
    df = load_lane_universe(con, lane)
    if df.empty:
        return {
            "hyp_id": lane.hyp_id, "strategy_id": lane.strategy_id,
            "verdict": "NO_DATA", "n_universe": 0, "n_fired": 0,
            "expr": float("nan"), "sharpe": float("nan"), "t": float("nan"),
            "long_t": float("nan"), "short_t": float("nan"),
            "has_theory": lane.has_theory,
            "threshold_protocol_a": CHORDIA_T_WITH_THEORY,
            "threshold_chordia_strict": CHORDIA_T_WITHOUT_THEORY,
        }

    # Apply canonical filter to derive fire mask -- never re-encode
    fire_mask = filter_signal(df, lane.filter_key, lane.orb_label).astype(bool)
    fired = df.loc[fire_mask].copy()

    # scratch-policy: COALESCE-to-zero per feedback_scratch_pnl_null_class_bug.md
    fired["pnl_r"] = fired["pnl_r"].fillna(0.0)

    n = len(fired)
    if n < 2:
        return {
            "hyp_id": lane.hyp_id, "strategy_id": lane.strategy_id,
            "verdict": "INSUFFICIENT_N", "n_universe": int(len(df)), "n_fired": n,
            "expr": float(fired["pnl_r"].mean()) if n else float("nan"),
            "sharpe": float("nan"), "t": float("nan"),
            "long_t": float("nan"), "short_t": float("nan"),
            "has_theory": lane.has_theory,
            "threshold_protocol_a": CHORDIA_T_WITH_THEORY,
            "threshold_chordia_strict": CHORDIA_T_WITHOUT_THEORY,
        }

    pnl = fired["pnl_r"]
    mean_r = float(pnl.mean())
    std_r = float(pnl.std(ddof=1))
    sharpe = mean_r / std_r if std_r > 0 else float("nan")
    t = compute_chordia_t(sharpe, n) if std_r > 0 else float("nan")

    # Direction derived from price geometry: orb_outcomes has no direction
    # column -- long iff target_price > stop_price. Per-lane-breakdown rule
    # (memory/feedback_per_lane_breakdown_required.md) requires we report
    # whether long and short legs agree in sign on the t-statistic.
    long_t = short_t = float("nan")
    long_n = short_n = 0
    if {"target_price", "stop_price"}.issubset(fired.columns):
        is_long = fired["target_price"] > fired["stop_price"]
        for mask, key in ((is_long, "long"), (~is_long, "short")):
            sub = fired.loc[mask, "pnl_r"]
            if len(sub) >= 2 and sub.std(ddof=1) > 0:
                d_sharpe = sub.mean() / sub.std(ddof=1)
                d_t = compute_chordia_t(d_sharpe, len(sub))
                if key == "long":
                    long_t = float(d_t); long_n = int(len(sub))
                else:
                    short_t = float(d_t); short_n = int(len(sub))

    # Use canonical chordia_threshold(has_theory) -- never re-encode the
    # WITH/WITHOUT switch. Per pre-reg failure_policy: lanes without
    # theory grounding face the strict 3.79 bar; verdict bands collapse
    # to FAIL_BOTH or PASS_CHORDIA only (no PROTOCOL_A band exists for
    # non-theory lanes per Chordia 2018).
    lane_threshold = chordia_threshold(lane.has_theory)
    if t < lane_threshold:
        verdict = "FAIL_BOTH"
    elif lane.has_theory and t < CHORDIA_T_WITHOUT_THEORY:
        verdict = "PASS_PROTOCOL_A"
    else:
        verdict = "PASS_CHORDIA"

    # Two-sided p approximation via large-N normal (informational only;
    # the t-statistic IS the gate, p is for context per Chordia 2018)
    p_two_sided = math.erfc(abs(t) / math.sqrt(2.0)) if math.isfinite(t) else float("nan")

    return {
        "hyp_id": lane.hyp_id,
        "strategy_id": lane.strategy_id,
        "n_universe": int(len(df)),
        "n_fired": int(n),
        "fire_rate": float(n / len(df)) if len(df) else float("nan"),
        "expr": mean_r,
        "std_r": std_r,
        "sharpe": sharpe,
        "t": float(t) if math.isfinite(t) else float("nan"),
        "p_two_sided": p_two_sided,
        "long_t": long_t,
        "long_n": long_n,
        "short_t": short_t,
        "short_n": short_n,
        "verdict": verdict,
        "has_theory": lane.has_theory,
        "threshold_protocol_a": CHORDIA_T_WITH_THEORY,
        "threshold_chordia_strict": CHORDIA_T_WITHOUT_THEORY,
    }


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    print(f"DB: {GOLD_DB_PATH}")
    print(f"IS cutoff (trading_day <): {HOLDOUT_SACRED_FROM}")
    print(f"Chordia thresholds: PROTOCOL_A>={CHORDIA_T_WITH_THEORY}, "
          f"CHORDIA_STRICT>={CHORDIA_T_WITHOUT_THEORY}")
    print()

    results = []
    for lane in LANES:
        r = chordia_for_lane(con, lane)
        results.append(r)

    # Headline table
    print("Per-lane verdict table (IS-only, < {}):".format(HOLDOUT_SACRED_FROM))
    print("-" * 130)
    print(f"{'Hyp':<4} {'Lane':<48} {'Theo':>5} {'Thr':>5} "
          f"{'N_fire':>7} {'Fire%':>7} {'ExpR':>8} {'t':>7} "
          f"{'long_t/N':>14} {'short_t/N':>14} {'Verdict':<16}")
    print("-" * 145)
    for r, lane in zip(results, LANES, strict=True):
        ln = f"{r.get('long_t', float('nan')):.3f}/{r.get('long_n', 0)}"
        sn = f"{r.get('short_t', float('nan')):.3f}/{r.get('short_n', 0)}"
        thr = chordia_threshold(lane.has_theory)
        print(
            f"{r['hyp_id']:<4} {r['strategy_id']:<48} "
            f"{str(lane.has_theory)[:5]:>5} {thr:>5.2f} "
            f"{r['n_fired']:>7} {r.get('fire_rate', float('nan')):>7.2%} "
            f"{r['expr']:>8.4f} "
            f"{r['t']:>7.3f} "
            f"{ln:>14} {sn:>14} "
            f"{r['verdict']:<16}"
        )
    print("-" * 130)

    # Summary
    fail = sum(1 for r in results if r["verdict"] == "FAIL_BOTH")
    pa = sum(1 for r in results if r["verdict"] == "PASS_PROTOCOL_A")
    pc = sum(1 for r in results if r["verdict"] == "PASS_CHORDIA")
    print(f"\nSummary: FAIL_BOTH={fail}, PASS_PROTOCOL_A={pa}, PASS_CHORDIA={pc}")

    return 0 if fail < 3 else 1  # >=3 FAIL_BOTH triggers re-rank per pre-reg


if __name__ == "__main__":
    raise SystemExit(main())
