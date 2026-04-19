"""MNQ NYSE_CLOSE LONG direction-locked broad validation — V1.

Pre-reg: docs/audit/hypotheses/2026-04-19-mnq-nyse-close-long-direction-locked-v1.yaml
Canonical sources: gold.db::orb_outcomes, pipeline.cost_model, deployable_validated_setups
Holdout: Mode A (2026-01-01 sacred). Pre-2026 selection; 2026 diagnostic only.

Gates (locked pre-run; see YAML):
  C3 BH-FDR q=0.05 across K=3 apertures
  C4 t >= 3.00 (with-theory)
  C6 WFE >= 0.50 (IS 2019-05-06..2023-12-31 vs pseudo-OOS 2024-01-01..2025-12-31)
  C7 N >= 100 per aperture
  C8 2026 avg_r >= 0 AND 2026/IS ratio >= 0.40
  C9 era stability: 2019-2022, 2023, 2024-2025 each ExpR >= -0.05 where N >= 50
  Adversarial: direction-mix parity, single-year dominance, correlation Jaccard <= 0.70
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Enable running from worktree or main
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.lib import bh_fdr, connect_db, query_df, write_csv  # noqa: E402

# ---------------------------------------------------------------------------
# LOCKED CONSTANTS (from pre-reg; no post-hoc relaxation)
# ---------------------------------------------------------------------------
HOLDOUT_BOUNDARY = "2026-01-01"
WF_IS_END = "2024-01-01"  # pre-2026 walk-forward IS end (pseudo-OOS = 2024-2025)
APERTURES = [5, 15, 30]
DIRECTIONS = ("long", "short")
RR_TARGET = 1.0
ORB_LABEL = "NYSE_CLOSE"
SYMBOL = "MNQ"
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
K_TOTAL = 3
BH_Q = 0.05
T_THRESHOLD = 3.00  # Criterion 4 with-theory
N_MIN_DEPLOYABLE = 100  # Criterion 7
WFE_MIN = 0.50  # Criterion 6
C8_OOS_RATIO_MIN = 0.40
C8_OOS_AVG_MIN = 0.0
C9_ERA_MIN = -0.05
C9_ERA_N_MIN = 50
SINGLE_YEAR_DOMINANCE_MAX = 0.50
LONG_SHORT_PARITY_SIGMA_GAP = 1.5  # LONG t must exceed SHORT t by >= this
CORR_JACCARD_HARD_FAIL = 0.70
CORR_JACCARD_WARN = 0.50
BOOTSTRAP_N = 10000
BOOTSTRAP_BLOCK_SIZE = 5
BOOTSTRAP_SEED = 42

OUTPUT_PREFIX = "mnq_nyse_close_long_direction_locked_v1"


# ---------------------------------------------------------------------------
# CANONICAL QUERIES
# ---------------------------------------------------------------------------
def _base_where() -> str:
    return (
        f"symbol='{SYMBOL}' AND orb_label='{ORB_LABEL}' "
        f"AND entry_model='{ENTRY_MODEL}' AND confirm_bars={CONFIRM_BARS} "
        f"AND rr_target={RR_TARGET} AND entry_price != stop_price"
    )


def _dir_case() -> str:
    # Derived direction per pre-flight B1: 0 degenerate rows on this cell.
    return "CASE WHEN entry_price > stop_price THEN 'long' ELSE 'short' END"


def fetch_trade_ledger(direction: str, aperture: int, pre_holdout: bool) -> pd.DataFrame:
    """Return per-trade rows with pnl_r and trading_day. Used for bootstrap/year/era/WF."""
    holdout_cmp = "<" if pre_holdout else ">="
    sql = f"""
    SELECT trading_day, pnl_r, risk_dollars,
           CAST(EXTRACT(year FROM trading_day) AS INT) AS yr
    FROM orb_outcomes
    WHERE {_base_where()}
      AND orb_minutes = ?
      AND {_dir_case()} = ?
      AND trading_day {holdout_cmp} '{HOLDOUT_BOUNDARY}'
    ORDER BY trading_day
    """
    return query_df(sql, [aperture, direction])


def fetch_aperture_summary(direction: str) -> pd.DataFrame:
    """Summary stats per aperture pre-2026 for one direction."""
    sql = f"""
    SELECT orb_minutes,
           COUNT(*) AS n,
           AVG(pnl_r) AS avg_r,
           STDDEV_SAMP(pnl_r) AS sd,
           AVG(pnl_r) / NULLIF(STDDEV_SAMP(pnl_r) / SQRT(COUNT(*)), 0) AS t_stat,
           AVG(CASE WHEN pnl_r > 0 THEN 1.0 ELSE 0.0 END) AS win_rate,
           COUNT(DISTINCT CAST(EXTRACT(year FROM trading_day) AS INT)) AS n_years,
           MIN(trading_day) AS min_day,
           MAX(trading_day) AS max_day,
           MEDIAN(risk_dollars) AS median_risk_dollars
    FROM orb_outcomes
    WHERE {_base_where()}
      AND {_dir_case()} = ?
      AND trading_day < '{HOLDOUT_BOUNDARY}'
    GROUP BY orb_minutes
    ORDER BY orb_minutes
    """
    return query_df(sql, [direction])


def fetch_oos_2026_summary() -> pd.DataFrame:
    """2026 diagnostic (LONG only, per-aperture). Reported, NOT used for selection."""
    sql = f"""
    SELECT orb_minutes,
           COUNT(*) AS n_oos,
           AVG(pnl_r) AS avg_r_oos,
           AVG(pnl_r) / NULLIF(STDDEV_SAMP(pnl_r) / SQRT(COUNT(*)), 0) AS t_oos
    FROM orb_outcomes
    WHERE {_base_where()}
      AND {_dir_case()} = 'long'
      AND trading_day >= '{HOLDOUT_BOUNDARY}'
    GROUP BY orb_minutes
    ORDER BY orb_minutes
    """
    return query_df(sql)


def fetch_year_table_long() -> pd.DataFrame:
    sql = f"""
    SELECT orb_minutes,
           CAST(EXTRACT(year FROM trading_day) AS INT) AS yr,
           COUNT(*) AS n,
           AVG(pnl_r) AS avg_r,
           SUM(pnl_r) AS sum_r
    FROM orb_outcomes
    WHERE {_base_where()}
      AND {_dir_case()} = 'long'
      AND trading_day < '{HOLDOUT_BOUNDARY}'
    GROUP BY orb_minutes, yr
    ORDER BY orb_minutes, yr
    """
    return query_df(sql)


def fetch_cross_instrument_diagnostic() -> pd.DataFrame:
    """T8: MES NYSE_CLOSE LONG same-condition for cross-instrument consistency. Informational."""
    sql = f"""
    SELECT symbol, orb_minutes,
           COUNT(*) AS n,
           AVG(pnl_r) AS avg_r,
           AVG(pnl_r) / NULLIF(STDDEV_SAMP(pnl_r) / SQRT(COUNT(*)), 0) AS t_stat
    FROM orb_outcomes
    WHERE symbol IN ('MES') AND orb_label='{ORB_LABEL}'
      AND entry_model='{ENTRY_MODEL}' AND confirm_bars={CONFIRM_BARS}
      AND rr_target={RR_TARGET} AND entry_price != stop_price
      AND {_dir_case()} = 'long'
      AND trading_day < '{HOLDOUT_BOUNDARY}'
    GROUP BY symbol, orb_minutes
    ORDER BY symbol, orb_minutes
    """
    return query_df(sql)


def fetch_live_mnq_lane_fire_days() -> dict[str, set]:
    """Return {strategy_id: set of trading_days} for each currently-deployable MNQ lane.

    Used for direction-locked candidate correlation via Jaccard over pre-2026 fire days.
    Uses ORB outcomes as the fire-day proxy at the (session, orb_minutes, rr_target) level.
    NOTE: this is an approximation — true fire days depend on each lane's filter_type.
    For a conservative correlation gate, approximating with session+aperture fire-days
    overstates overlap (worst-case), which is fail-closed safe.
    """
    sid_rows = query_df(
        """
        SELECT strategy_id, orb_label, orb_minutes, rr_target
        FROM deployable_validated_setups
        WHERE instrument = 'MNQ'
        """
    )
    result: dict[str, set] = {}
    with connect_db() as con:
        for _, row in sid_rows.iterrows():
            sql = f"""
            SELECT DISTINCT trading_day
            FROM orb_outcomes
            WHERE symbol='MNQ' AND orb_label=? AND orb_minutes=? AND rr_target=?
              AND entry_model='E2' AND confirm_bars=1
              AND trading_day < '{HOLDOUT_BOUNDARY}'
            """
            days = con.execute(sql, [row.orb_label, int(row.orb_minutes), float(row.rr_target)]).fetchdf()
            result[row.strategy_id] = set(pd.to_datetime(days.trading_day).dt.date)
    return result


# ---------------------------------------------------------------------------
# GATE HELPERS (pure, testable)
# ---------------------------------------------------------------------------
def p_from_t_two_tailed(t: float, n: int) -> float:
    """Two-tailed p-value from t-stat with df=n-1. scipy stats.t."""
    if np.isnan(t) or n < 2:
        return float("nan")
    from scipy import stats
    return float(2.0 * stats.t.sf(abs(t), df=n - 1))


def moving_block_bootstrap_p(pnl: np.ndarray, n_boot: int = BOOTSTRAP_N,
                             block_size: int = BOOTSTRAP_BLOCK_SIZE,
                             seed: int = BOOTSTRAP_SEED) -> float:
    """One-sided moving-block bootstrap p-value that observed mean > 0.

    Resampling preserves autocorrelation via contiguous blocks of pnl.
    Null: mean=0 (demeaned reference). We report: fraction of resampled
    means that exceed the observed mean, treating demeaned returns as the
    null distribution. Small p => observed mean is in the right tail of the null.
    """
    pnl = np.asarray(pnl, dtype=float)
    pnl = pnl[~np.isnan(pnl)]
    n = len(pnl)
    if n < block_size * 2:
        return float("nan")
    observed = pnl.mean()
    demeaned = pnl - observed  # null: E[x] = 0
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size
    ge_count = 0
    for _ in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        chunks = [demeaned[s:s + block_size] for s in starts]
        resampled = np.concatenate(chunks)[:n]
        if resampled.mean() >= observed:
            ge_count += 1
    return (ge_count + 1) / (n_boot + 1)


def single_year_dominance(year_sum_r: dict[int, float]) -> tuple[float, int | None]:
    """Return (max_share, dominant_year). Fails if max_share > 0.50."""
    total = sum(abs(v) for v in year_sum_r.values())
    if total == 0:
        return 0.0, None
    # Only positive-contribution dominance matters (negative-year drag isn't rescue-bias).
    positive_sum = sum(v for v in year_sum_r.values() if v > 0)
    if positive_sum == 0:
        return 0.0, None
    pos_only = {y: v for y, v in year_sum_r.items() if v > 0}
    y_max = max(pos_only, key=lambda k: pos_only[k])
    share = pos_only[y_max] / positive_sum
    return share, y_max


def era_stability(trades: pd.DataFrame, era_bounds: list[tuple[int, int]]) -> list[dict]:
    """Per-era ExpR + N per Criterion 9. era_bounds = [(start_yr, end_yr_inclusive)]."""
    out = []
    for start, end in era_bounds:
        sub = trades[(trades.yr >= start) & (trades.yr <= end)]
        n = len(sub)
        avg_r = float(sub.pnl_r.mean()) if n > 0 else float("nan")
        passes = True if n < C9_ERA_N_MIN else bool(avg_r >= C9_ERA_MIN)
        out.append({"era_start": start, "era_end": end, "n": n,
                    "avg_r": avg_r, "exempt": n < C9_ERA_N_MIN, "passes": passes})
    return out


def walk_forward_efficiency(trades: pd.DataFrame, is_end_date: str) -> dict:
    """WFE = (OOS mean / OOS std) / (IS mean / IS std). Criterion 6 threshold 0.50."""
    is_rows = trades[trades.trading_day < pd.Timestamp(is_end_date)]
    oos_rows = trades[trades.trading_day >= pd.Timestamp(is_end_date)]
    n_is, n_oos = len(is_rows), len(oos_rows)
    if n_is < 30 or n_oos < 30:
        return {"n_is": n_is, "n_oos": n_oos, "is_sharpe": float("nan"),
                "oos_sharpe": float("nan"), "wfe": float("nan"), "passes": False,
                "note": "insufficient sample"}
    is_sharpe = float(is_rows.pnl_r.mean() / is_rows.pnl_r.std(ddof=1))
    oos_sharpe = float(oos_rows.pnl_r.mean() / oos_rows.pnl_r.std(ddof=1))
    wfe = oos_sharpe / is_sharpe if is_sharpe != 0 else float("nan")
    return {"n_is": n_is, "n_oos": n_oos, "is_sharpe": is_sharpe,
            "oos_sharpe": oos_sharpe, "wfe": wfe,
            "passes": bool(not np.isnan(wfe) and wfe >= WFE_MIN), "note": ""}


def long_short_parity(long_t: float, short_t: float) -> dict:
    """Adversarial: SHORT must be at least 1.5 sigma WEAKER than LONG."""
    gap = long_t - short_t
    passes = bool(not np.isnan(gap) and gap >= LONG_SHORT_PARITY_SIGMA_GAP)
    short_significant = bool(not np.isnan(short_t) and abs(short_t) >= T_THRESHOLD)
    return {"long_t": long_t, "short_t": short_t, "gap_sigma": gap,
            "short_also_significant": short_significant,
            "passes": passes and not short_significant}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# DOLLAR EV ESTIMATE (display-only)
# ---------------------------------------------------------------------------
def dollar_ev_estimate(avg_r: float, n: int, years: float, median_risk_dollars: float,
                       copies: int = 5, contracts: int = 3) -> dict:
    """Rough annual gross dollar EV at specified scale. Display-only."""
    if any(np.isnan(x) for x in [avg_r, median_risk_dollars]):
        return {"per_trade_dollars": float("nan"), "annual_gross_at_scale": float("nan")}
    per_trade = avg_r * median_risk_dollars
    n_per_year = n / years if years > 0 else 0
    annual_per_contract_copy = n_per_year * per_trade
    annual_scale = annual_per_contract_copy * copies * contracts
    return {"per_trade_dollars": per_trade,
            "n_per_year": n_per_year,
            "annual_gross_at_scale": annual_scale}


# ---------------------------------------------------------------------------
# MAIN ORCHESTRATION
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 72)
    print("MNQ NYSE_CLOSE LONG direction-locked validation V1")
    print("Pre-reg: docs/audit/hypotheses/2026-04-19-mnq-nyse-close-long-direction-locked-v1.yaml")
    print(f"K_total={K_TOTAL} | BH q={BH_Q} | t>={T_THRESHOLD} | WFE>={WFE_MIN}")
    print(f"IS selection: trading_day < {HOLDOUT_BOUNDARY} | OOS diagnostic: >= {HOLDOUT_BOUNDARY}")
    print("=" * 72)

    # ---- Per-aperture LONG summary (pre-2026) ----
    long_summary = fetch_aperture_summary("long")
    short_summary = fetch_aperture_summary("short")
    oos_summary = fetch_oos_2026_summary()

    # Merge for convenience
    merged = long_summary.merge(
        short_summary[["orb_minutes", "n", "avg_r", "t_stat"]],
        on="orb_minutes", suffixes=("_long", "_short")
    ).merge(
        oos_summary, on="orb_minutes", how="left"
    )

    # ---- Compute p-values, BH-FDR, bootstrap ----
    p_values_parametric = []
    bootstrap_ps = []
    per_aperture_results = []

    for ap in APERTURES:
        row = merged[merged.orb_minutes == ap].iloc[0]
        n_long = int(row.n_long)
        t_long = float(row.t_stat_long)
        avg_long = float(row.avg_r_long)
        p_long = p_from_t_two_tailed(t_long, n_long)
        p_values_parametric.append(p_long)

        # Bootstrap null
        ledger_long = fetch_trade_ledger("long", ap, pre_holdout=True)
        boot_p = moving_block_bootstrap_p(ledger_long.pnl_r.to_numpy())
        bootstrap_ps.append(boot_p)

        # WFE on LONG pre-2026 trades
        ledger_long["trading_day"] = pd.to_datetime(ledger_long.trading_day)
        wfe_res = walk_forward_efficiency(ledger_long, WF_IS_END)

        # Year-sum for single-year dominance
        year_sum_raw = ledger_long.groupby("yr").pnl_r.sum().to_dict()
        year_sum: dict[int, float] = {int(k): float(v) for k, v in year_sum_raw.items()}
        dom_share, dom_year = single_year_dominance(year_sum)

        # Era stability
        era_bounds = [(2019, 2022), (2023, 2023), (2024, 2025)]
        era_res = era_stability(ledger_long, era_bounds)

        # Dollar EV
        median_risk = float(row.median_risk_dollars) if not pd.isna(row.median_risk_dollars) else float("nan")
        years_span = 7.0  # 2019-05-06..2025-12-31 ≈ 6.65, round for ladder readability
        ev = dollar_ev_estimate(avg_long, n_long, years_span, median_risk)

        per_aperture_results.append({
            "aperture": ap,
            "n_long": n_long, "avg_r_long": avg_long, "t_long": t_long,
            "p_long": p_long, "bootstrap_p": boot_p,
            "win_rate_long": float(row.win_rate),
            "median_risk_dollars": median_risk,
            "per_trade_dollars": ev["per_trade_dollars"],
            "annual_gross_at_scale": ev["annual_gross_at_scale"],
            "n_short": int(row.n_short), "avg_r_short": float(row.avg_r_short),
            "t_short": float(row.t_stat_short),
            "wfe": wfe_res["wfe"], "wfe_passes": wfe_res["passes"],
            "wfe_n_is": wfe_res["n_is"], "wfe_n_oos": wfe_res["n_oos"],
            "single_year_share": dom_share, "dominant_year": dom_year,
            "era_results": era_res,
            "n_oos_2026": int(row.n_oos) if not pd.isna(row.n_oos) else 0,
            "avg_r_oos_2026": float(row.avg_r_oos) if not pd.isna(row.avg_r_oos) else float("nan"),
        })

    # BH-FDR across the K=3 family
    bh_reject = bh_fdr(p_values_parametric, q=BH_Q)

    # ---- Cross-instrument T8 diagnostic ----
    cross_inst = fetch_cross_instrument_diagnostic()

    # ---- Correlation vs live MNQ lanes ----
    # For each aperture, compute candidate LONG fire-days set; Jaccard vs each live lane's fire-days.
    correlation_rows = []
    live_lane_days = fetch_live_mnq_lane_fire_days()
    print(f"\n[correlation] live MNQ lanes queried: {len(live_lane_days)}")
    for ap in APERTURES:
        cand_days = set(pd.to_datetime(
            fetch_trade_ledger("long", ap, pre_holdout=True).trading_day
        ).dt.date)
        for sid, days in live_lane_days.items():
            j = jaccard(cand_days, days)
            correlation_rows.append({
                "aperture": ap, "strategy_id": sid,
                "candidate_n_days": len(cand_days), "live_n_days": len(days),
                "jaccard": j,
                "warn": j >= CORR_JACCARD_WARN,
                "hard_fail": j >= CORR_JACCARD_HARD_FAIL,
            })
    corr_df = pd.DataFrame(correlation_rows)

    # ---- Assemble verdict per aperture ----
    verdicts = []
    for i, r in enumerate(per_aperture_results):
        c3_bh = (i in bh_reject)
        c4_t = r["t_long"] >= T_THRESHOLD
        c7_n = r["n_long"] >= N_MIN_DEPLOYABLE
        c6_wfe = r["wfe_passes"]

        # C8 2026 OOS gate
        is_avg = r["avg_r_long"]
        oos_avg = r["avg_r_oos_2026"]
        c8_oos_positive = (not np.isnan(oos_avg)) and (oos_avg >= C8_OOS_AVG_MIN)
        c8_ratio_ok = (not np.isnan(oos_avg)) and is_avg > 0 and (oos_avg / is_avg >= C8_OOS_RATIO_MIN)
        c8 = bool(c8_oos_positive and c8_ratio_ok)

        # C9 era stability
        c9 = all(e["passes"] for e in r["era_results"])

        # Adversarial gates
        parity = long_short_parity(r["t_long"], r["t_short"])
        g_parity = parity["passes"]
        g_single_year = r["single_year_share"] <= SINGLE_YEAR_DOMINANCE_MAX
        # Bootstrap null
        g_boot = (not np.isnan(r["bootstrap_p"])) and r["bootstrap_p"] <= 0.05
        # Correlation
        ap_corrs = corr_df[corr_df.aperture == r["aperture"]]
        g_corr = not bool(ap_corrs.hard_fail.any())

        gates = {
            "C3_bh_fdr": c3_bh, "C4_t_stat": c4_t, "C6_wfe": c6_wfe,
            "C7_sample_size": c7_n, "C8_2026_oos": c8, "C9_era_stability": c9,
            "adv_parity": g_parity, "adv_single_year": g_single_year,
            "adv_bootstrap": g_boot, "adv_correlation": g_corr,
        }
        all_required_pass = all([c3_bh, c4_t, c6_wfe, c7_n, c8, c9,
                                  g_parity, g_single_year, g_boot, g_corr])
        verdicts.append({
            "aperture": r["aperture"],
            "gates": gates,
            "strict_pass": all_required_pass,
            "parity_detail": parity,
        })

    # ---- Print verdict table ----
    print("\n" + "=" * 72)
    print("VERDICT TABLE (pre-registered gates applied)")
    print("=" * 72)
    print(f"{'Aper':>4} {'N':>5} {'avgR':>8} {'t':>6} {'BH':>4} {'C4':>3} {'C6':>3} {'C7':>3} {'C8':>3} {'C9':>3} {'Par':>4} {'Year':>5} {'Boot':>5} {'Cor':>4} | STRICT")
    for v, r in zip(verdicts, per_aperture_results):
        g = v["gates"]
        def _f(b): return "Y" if b else "N"
        print(f"{r['aperture']:>4} {r['n_long']:>5} {r['avg_r_long']:>+8.4f} {r['t_long']:>6.2f} "
              f"{_f(g['C3_bh_fdr']):>4} {_f(g['C4_t_stat']):>3} {_f(g['C6_wfe']):>3} "
              f"{_f(g['C7_sample_size']):>3} {_f(g['C8_2026_oos']):>3} {_f(g['C9_era_stability']):>3} "
              f"{_f(g['adv_parity']):>4} {_f(g['adv_single_year']):>5} {_f(g['adv_bootstrap']):>5} "
              f"{_f(g['adv_correlation']):>4} | "
              f"{'PASS' if v['strict_pass'] else 'FAIL'}")

    # ---- Cross-instrument T8 summary ----
    print("\n[T8 cross-instrument diagnostic — MES NYSE_CLOSE LONG pre-2026]")
    if len(cross_inst) > 0:
        for _, r in cross_inst.iterrows():
            print(f"  {r['symbol']} O{int(r['orb_minutes'])}: N={int(r['n'])}, "
                  f"avg_r={r['avg_r']:+.4f}, t={r['t_stat']:+.2f}")
    else:
        print("  No MES NYSE_CLOSE rows returned")

    # ---- Correlation warnings ----
    warn_rows = corr_df[corr_df.warn]
    hard_fail_rows = corr_df[corr_df.hard_fail]
    print(f"\n[correlation gate] Jaccard >= {CORR_JACCARD_WARN}: {len(warn_rows)} pairs; "
          f">= {CORR_JACCARD_HARD_FAIL}: {len(hard_fail_rows)} pairs")
    if not warn_rows.empty:
        print(warn_rows[["aperture", "strategy_id", "jaccard"]].to_string(index=False))

    # ---- Write outputs ----
    # Aperture summary
    ap_rows = []
    for v, r in zip(verdicts, per_aperture_results):
        base = {k: val for k, val in r.items() if k != "era_results"}
        base.update({f"gate_{k}": val for k, val in v["gates"].items()})
        base["strict_pass"] = v["strict_pass"]
        ap_rows.append(base)
    write_csv(pd.DataFrame(ap_rows), f"{OUTPUT_PREFIX}_apertures.csv")

    # Year table
    write_csv(fetch_year_table_long(), f"{OUTPUT_PREFIX}_years.csv")

    # Era table (per-aperture era rows flattened)
    era_flat = []
    for r in per_aperture_results:
        for e in r["era_results"]:
            era_flat.append({"aperture": r["aperture"], **e})
    write_csv(pd.DataFrame(era_flat), f"{OUTPUT_PREFIX}_eras.csv")

    # Direction parity
    parity_rows = []
    for v, r in zip(verdicts, per_aperture_results):
        parity_rows.append({"aperture": r["aperture"],
                             "t_long": r["t_long"], "t_short": r["t_short"],
                             "gap_sigma": v["parity_detail"]["gap_sigma"],
                             "short_also_significant": v["parity_detail"]["short_also_significant"],
                             "parity_passes": v["parity_detail"]["passes"]})
    write_csv(pd.DataFrame(parity_rows), f"{OUTPUT_PREFIX}_direction_parity.csv")

    # Correlation
    write_csv(corr_df, f"{OUTPUT_PREFIX}_correlation.csv")

    # ---- Final family verdict ----
    any_pass = any(v["strict_pass"] for v in verdicts)
    kill_conditions = [
        not any(v["gates"]["C3_bh_fdr"] for v in verdicts),
        not any(v["gates"]["C4_t_stat"] for v in verdicts),
        any(v["gates"]["adv_parity"] is False
            and v["parity_detail"]["short_also_significant"]
            for v in verdicts),
        any(r["single_year_share"] > SINGLE_YEAR_DOMINANCE_MAX for r in per_aperture_results),
        all((np.isnan(r["avg_r_oos_2026"]) or r["avg_r_oos_2026"] < 0) for r in per_aperture_results),
        not any(v["gates"]["C6_wfe"] for v in verdicts),
        any(v["gates"]["adv_correlation"] is False for v in verdicts),
    ]
    family_kill = any(kill_conditions)

    print("\n" + "=" * 72)
    if any_pass and not family_kill:
        print(f"FAMILY VERDICT: PROMOTE — {sum(v['strict_pass'] for v in verdicts)}/3 apertures pass all required gates")
    elif family_kill:
        print("FAMILY VERDICT: KILL — at least one pre-registered kill condition triggered")
    else:
        print("FAMILY VERDICT: NO-PROMOTE — no aperture passes all required gates (no hard-kill triggers either)")
    print("=" * 72)

    return 0 if not family_kill else 1


if __name__ == "__main__":
    sys.exit(main())
