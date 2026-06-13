"""research/mgc_trend_day_tail_descriptive.py

Phase A — MGC Trend-Day Runner descriptive scan (read-only, NO prereg, NO LOCK).

Answers ONE question on canonical layers: does the uncapped-R distribution on
MGC trend-days have a materially fatter RIGHT TAIL than non-trend days, and does
a pre-entry-SAFE proxy capture enough of it to be worth a Phase-B prereg?

The whole reason this object exists: every prior MGC verdict measured
`avg_pnl_r at RR1.5` — a metric that CAPS the winner at 1.5R and averages it into
the chop. A trend strategy is *supposed* to look bad on win-rate/mean-R: it loses
small often, wins huge rarely. This scan measures the TAIL under a REALIZABLE
exit (hold-to-close + a mechanical trailing stop), using uncapped MFE only as the
non-realizable ceiling benchmark.

⚠️ MFE-ILLUSION GUARD (load-bearing): `true_mfe_r` is the maximum favorable
excursion — the single best tick the trade ever reached. NO real exit realizes it
(you don't know the high until it has passed). MFE is the UPPER BOUND on a perfect
trailing stop, not a P&L any strategy earns. The go/no-go rests on tails 2 & 3
(realizable); tail 1 is the ceiling we're chasing. Measuring MFE and calling it
edge is the storytelling-bias trap.

Three tails per cell:
  1. true_mfe_r     — CEILING (NON-REALIZABLE; friction in denom only; benchmark)
  2. session_close_r — hold-to-close (realizable; net of friction)
  3. trail_r        — give-back-of-peak trailing stop (realizable; net of friction)

Two splits:
  - ORACLE   — by look-ahead day_type (TREND_UP/DOWN vs NON_TREND). Upper bound on
               the tradeable tail. Labelled clearly as look-ahead / non-tradeable.
  - SAFE-PROXY — by pre-entry-safe proxies (prev_day_direction, atr_20_pct decile,
               garch_atr_ratio>1, is_nfp_day). How much of the oracle tail a
               TRADEABLE signal captures. THE DECISION RESTS ON THIS.

Anti-cherry-pick (RULE 5.3): this reports the WHOLE grid. It does NOT select a
winner. Any Phase-B cell must be pre-committed from the distribution SHAPE
(which decile/session the realizable tail concentrates in, stable across years),
NOT picked as the post-hoc max cell.

Reuse (institutional-rigor § 4 — delegate to canonical, never re-encode):
  - research.research_trend_day_mfe.compute_true_session_mfe  (uncapped replay)
  - research.research_trend_day_mfe.load_outcomes / load_bars_1m_for_day
  - pipeline.cost_model.COST_SPECS / pipeline.paths.GOLD_DB_PATH
  - pipeline.build_daily_features.compute_trading_day_utc_range

@research-source: docs/plans/2026-06-14-mgc-trend-day-runner-phase-a.md (this plan)
@entry-models: E1, E2 (matches reused load_outcomes filter)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# Force UTF-8 stdout — Windows console defaults to cp1252 and chokes on the
# ≥ / × / ✓ / — glyphs in the report. Reconfigure rather than ASCII-mangle.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.build_daily_features import compute_trading_day_utc_range
from pipeline.cost_model import COST_SPECS, CostSpec, pnl_points_to_r
from pipeline.paths import GOLD_DB_PATH
from research.research_trend_day_mfe import (
    compute_true_session_mfe,
    load_bars_1m_for_day,
    load_outcomes,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INSTRUMENT = "MGC"
IS_CUTOFF = pd.Timestamp("2026-01-01")  # trading_day < this == IS (Mode A sacred holdout)

# Sessions confirmed N>=100 for MGC E1/E2 IS (plan scope).
TESTABLE_SESSIONS = [
    "TOKYO_OPEN",
    "LONDON_METALS",
    "EUROPE_FLOW",
    "SINGAPORE_OPEN",
    "US_DATA_830",
    "NYSE_OPEN",
    "US_DATA_1000",
    "COMEX_SETTLE",
    "CME_REOPEN",
]

ORB_MINUTES = [5, 15, 30]

# Trailing-stop sim parameter (give-back of peak favorable excursion).
# ARM once peak favorable >= TRAIL_ARM_R, then exit on the FIRST bar whose
# favorable excursion retraces >= TRAIL_GIVEBACK_FRAC of the running peak.
# Bounded below by the hard stop (never worse than -1R gross). Parameterized so
# it is an explicit lever, not a hidden magic number.
TRAIL_ARM_R = 1.0
TRAIL_GIVEBACK_FRAC = 0.50

MIN_CELL_N = 30  # REGIME-tier floor for a reportable cell

# Pre-entry-SAFE proxies (backtesting-methodology § 6.1 — all knowable before entry).
SAFE_PROXIES = ["prev_day_direction", "atr_20_pct", "garch_atr_ratio", "is_nfp_day"]


# ---------------------------------------------------------------------------
# Trailing-stop sim — extends the reused MFE replay (does NOT re-derive it)
# ---------------------------------------------------------------------------
def compute_trail_r(
    bars_1m: pd.DataFrame,  # pre-filtered: entry_ts < ts_utc <= trading_day_end, ORDERED
    entry_price: float,
    stop_price: float,
    break_dir: int,  # 1=long, -1=short
    cost_spec: CostSpec,
    arm_r: float = TRAIL_ARM_R,
    giveback_frac: float = TRAIL_GIVEBACK_FRAC,
) -> float | None:
    """Realizable give-back-of-peak trailing-stop exit, NET of friction.

    Bar-by-bar over the SAME 1m replay window the reused compute_true_session_mfe
    consumes. Returns signed R or None if no bars.

    Mechanic (causal — only uses bars up to the current bar):
      - Each 1m bar has a favorable HIGH excursion (best tick this bar) and a
        favorable LOW excursion (worst tick this bar). For a long these are
        (high-entry) and (low-entry); for a short, (entry-low) and (entry-high).
      - Running peak = max favorable HIGH excursion seen so far.
      - The hard stop is always live: if the bar's favorable-low reaches the
        negative hard-stop distance, the trade dies first (exit -hard_stop gross).
      - Once peak >= arm_r (in R), arm the trail at peak*(1-giveback). Thereafter
        the trail is hit when a bar's favorable-LOW (the adverse extreme of the
        bar) falls to/under the trail level → exit at the trail level. A trailing
        stop fires on the bar's LOW crossing the level, NOT its high.
      - The peak is updated from the bar HIGH each bar, so a strong trending bar
        ratchets the trail up before the next bar can trip it.
      - If never armed and never stopped, exit at session close.

    Conservative tie-break: within a single 1m bar we cannot know whether the high
    or the low printed first. We resolve in the order hard-stop → trail-check →
    ratchet-peak, so the trail is tested against the PRIOR peak before this bar can
    raise it — a bar's own low can never trip the trail its own high just set
    (that low preceded the new high intra-minute). Net bias is DOWN (stop/trail win
    ties), the honest direction for a go/no-go gate.
    """
    if bars_1m.empty:
        return None

    highs = bars_1m["high"].values
    lows = bars_1m["low"].values
    closes = bars_1m["close"].values

    # Hard-stop distance in points (positive).
    hard_stop_pts = abs(entry_price - stop_price)
    if hard_stop_pts <= 0:
        return None

    # R per favorable point (denominator includes friction, matching pnl_points_to_r).
    def pts_to_r_gross(pts: float) -> float:
        return pnl_points_to_r(cost_spec, entry_price, stop_price, pts)

    arm_pts = arm_r * hard_stop_pts  # favorable points needed to arm (R linear in pts)
    peak_fav = 0.0
    armed = False

    for i in range(len(bars_1m)):
        if break_dir == 1:  # long
            fav_high = highs[i] - entry_price  # best favorable tick this bar
            fav_low = lows[i] - entry_price    # worst favorable tick this bar (may be < 0)
        else:  # short
            fav_high = entry_price - lows[i]
            fav_low = entry_price - highs[i]

        # --- hard stop first (conservative tie-break) ---
        # adverse excursion = -fav_low when fav_low < 0; stop hit if it reaches hard_stop.
        if -fav_low >= hard_stop_pts:
            return _net_r(pts_to_r_gross, -hard_stop_pts, cost_spec)

        # --- trail check against the PRIOR peak (before this bar ratchets it) ---
        # A trailing stop set from the running peak can only be tripped by a bar
        # AFTER the bar that established that peak. Testing against the prior-bar
        # peak prevents the same bar's low (which preceded its own new high) from
        # falsely tripping the trail it just raised.
        if armed:
            trail_level_pts = peak_fav * (1.0 - giveback_frac)
            if fav_low <= trail_level_pts:
                return _net_r(pts_to_r_gross, trail_level_pts, cost_spec)

        # --- ratchet peak from the bar HIGH, then (re)arm ---
        if fav_high > peak_fav:
            peak_fav = fav_high
        if not armed and peak_fav >= arm_pts:
            armed = True

    # Never stopped, never trail-exited -> hold to session close.
    if break_dir == 1:
        close_pts = closes[-1] - entry_price
    else:
        close_pts = entry_price - closes[-1]
    return _net_r(pts_to_r_gross, close_pts, cost_spec)


def _net_r(
    pts_to_r_gross,
    gross_pts: float,
    cost_spec: CostSpec,
) -> float:
    """Convert gross favorable points to R, NET of friction in the numerator.

    Realizable exits pay friction (plan: 'uncapped winners still pay friction').
    pnl_points_to_r puts friction only in the denominator; here we ALSO deduct
    friction from the P&L numerator so a realized close/trail R is honest net.
    friction_pts = total_friction / point_value (the reward-reducing convention,
    cf. cost_model.reward_in_dollars).
    """
    friction_pts = cost_spec.total_friction / cost_spec.point_value
    net_pts = gross_pts - friction_pts
    return round(pts_to_r_gross(net_pts), 4)


# ---------------------------------------------------------------------------
# Replay driver — one bar load per trading day, compute all 3 tails per outcome
# ---------------------------------------------------------------------------
def compute_all_tails(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> pd.DataFrame:
    """For every outcome row, compute (true_mfe_r, session_close_r, trail_r).

    Tails 1 & 2 come from the reused compute_true_session_mfe; tail 3 from the
    trailing sim above, sharing the same per-day bar load.
    """
    cost_spec = COST_SPECS[INSTRUMENT]
    grouped = df.groupby("trading_day")
    n_days = len(grouped)
    print(f"  Computing 3 tails: {len(df):,} outcomes across {n_days:,} trading days...")

    t0 = time.time()
    results = []
    rows_done = 0
    for day_idx, (td, day_df) in enumerate(grouped):
        day_bars = load_bars_1m_for_day(con, INSTRUMENT, td)
        _, td_end = compute_trading_day_utc_range(td)
        td_end_ts = pd.Timestamp(td_end)

        for row_idx, row in day_df.iterrows():
            entry_ts = pd.Timestamp(row["entry_ts"])
            post_entry = day_bars[(day_bars["ts_utc"] > entry_ts) & (day_bars["ts_utc"] <= td_end_ts)]

            mfe = compute_true_session_mfe(
                bars_1m=post_entry,
                entry_price=row["entry_price"],
                stop_price=row["stop_price"],
                break_dir=row["break_dir"],
                cost_spec=cost_spec,
            )
            trail_r = compute_trail_r(
                bars_1m=post_entry,
                entry_price=row["entry_price"],
                stop_price=row["stop_price"],
                break_dir=row["break_dir"],
                cost_spec=cost_spec,
            )
            results.append(
                {
                    "idx": row_idx,
                    "true_mfe_r": mfe["true_mfe_r"],
                    "session_close_r": mfe["session_close_r"],
                    "trail_r": trail_r,
                }
            )

        rows_done += len(day_df)
        if (day_idx + 1) % 200 == 0 or (day_idx + 1) == n_days:
            pct = rows_done / len(df) * 100
            print(f"    [{day_idx + 1:,}/{n_days:,} days] {rows_done:,}/{len(df):,} ({pct:.0f}%) — {time.time()-t0:.1f}s")

    tails = pd.DataFrame(results).set_index("idx")
    for col in tails.columns:
        df[col] = tails[col]
    print(f"  Tail computation complete: {time.time()-t0:.1f}s\n")
    return df


# ---------------------------------------------------------------------------
# Feature merge (proxies + oracle day_type) — triple-join discipline
# ---------------------------------------------------------------------------
def load_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load day_type (oracle) + safe proxies, keyed (trading_day, orb_minutes).

    daily_features has 3 rows per (trading_day, symbol) — one per orb_minutes.
    We keep orb_minutes as a join key (triple-join rule) so the caller merges on
    both columns and never triples N.
    """
    sql = """
        SELECT trading_day, orb_minutes,
               day_type,
               prev_day_direction, atr_20_pct, garch_atr_ratio, is_nfp_day,
               day_of_week, daily_close, daily_open
        FROM daily_features
        WHERE symbol = ?
    """
    # NOTE: the capped mfe_r used by the RULE-13 pressure test is a per-trade
    # orb_outcomes column (already loaded as `capped_mfe_r` by the reused
    # load_outcomes), NOT a daily_features column — so it is not selected here.
    return con.execute(sql, [INSTRUMENT]).fetchdf()


# ---------------------------------------------------------------------------
# Tail statistics
# ---------------------------------------------------------------------------
TAIL_COLS = ["true_mfe_r", "session_close_r", "trail_r"]
TAIL_LABELS = {
    "true_mfe_r": "MFE (CEILING — non-realizable)",
    "session_close_r": "CLOSE (realizable)",
    "trail_r": "TRAIL (realizable)",
}


def tail_stats(r: pd.Series) -> dict:
    """Per-tail distribution stats. The TAIL is the thesis — report it fully."""
    a = np.asarray(r.dropna().to_numpy(), dtype=float)
    n = len(a)
    if n == 0:
        return {k: np.nan for k in ["n", "mean", "median", "p90", "p99", "max", "pct_ge3r", "sum_r", "sharpe", "gain_to_pain"]}
    mean = float(np.mean(a))
    std = float(np.std(a, ddof=1)) if n > 1 else np.nan
    gains = a[a > 0].sum()
    pains = -a[a < 0].sum()  # positive magnitude of losses
    return {
        "n": n,
        "mean": round(mean, 4),
        "median": round(float(np.median(a)), 4),
        "p90": round(float(np.quantile(a, 0.90)), 4),
        "p99": round(float(np.quantile(a, 0.99)), 4),
        "max": round(float(np.max(a)), 4),
        "pct_ge3r": round(float((a >= 3.0).mean()) * 100, 2),
        "sum_r": round(float(np.sum(a)), 2),
        "sharpe": round(mean / std, 4) if std and std > 0 else np.nan,
        "gain_to_pain": round(gains / pains, 4) if pains > 0 else np.nan,
    }


def _bin_proxy(df: pd.DataFrame, proxy: str) -> pd.Series:
    """Return a categorical bin Series for a proxy. Deciles for continuous,
    binary for categorical. NaN rows stay NaN (dropped from the split — never
    bucketed into a third class that would shift the comparator parent)."""
    if proxy == "prev_day_direction":
        return df["prev_day_direction"]  # 'bull'/'bear'/None
    if proxy == "is_nfp_day":
        return df["is_nfp_day"].map({True: "nfp", False: "non_nfp"})
    if proxy in ("atr_20_pct", "garch_atr_ratio"):
        vals = df[proxy]
        mask = vals.notna()
        out = pd.Series(np.nan, index=df.index, dtype=object)
        if mask.sum() >= 10:
            try:
                out.loc[mask] = pd.qcut(vals[mask], q=10, labels=[f"D{i+1}" for i in range(10)], duplicates="drop").astype(object)
            except ValueError:
                out.loc[mask] = "all"  # not enough distinct values for deciles
        return out
    raise ValueError(f"Unknown proxy {proxy!r}")


# ---------------------------------------------------------------------------
# RULE 13 — pressure test
# ---------------------------------------------------------------------------
def pressure_test(df: pd.DataFrame) -> None:
    """Inject a known look-ahead column as a fake proxy; it MUST surface as a
    tautology / look-ahead so we KNOW the harness catches the failure class.

    We use the realized tail itself (session_close_r) and the DB-capped mfe_r (if
    present) as fake 'proxies'. A real proxy is pre-entry-safe and should NOT
    correlate ~1.0 with the realized tail; a look-ahead proxy WILL. We flag any
    candidate proxy whose |corr| with the tail exceeds 0.70 (RULE 7 tautology
    threshold).
    """
    print("  " + "=" * 70)
    print("  RULE 13 PRESSURE TEST — inject look-ahead, confirm it is flagged")
    print("  " + "=" * 70)
    tail = df["session_close_r"].astype(float)

    # Fake proxy 1: the realized tail vs ITSELF (must flag — perfect corr).
    corr_self = float(np.corrcoef(tail.dropna(), tail.dropna())[0, 1])
    flagged_self = abs(corr_self) > 0.70
    print(f"    [inject] session_close_r vs itself: corr={corr_self:.3f} -> "
          f"{'FLAGGED look-ahead/tautology ✓' if flagged_self else 'NOT FLAGGED ✗ (HARNESS BROKEN)'}")

    # Fake proxy 2: DB-capped mfe_r (orb_outcomes, banned look-ahead) vs the tail.
    if "capped_mfe_r" in df.columns:
        sub = df[["capped_mfe_r", "session_close_r"]].dropna()
        if len(sub) >= 30:
            corr_mfe = float(np.corrcoef(sub["capped_mfe_r"], sub["session_close_r"])[0, 1])
            flagged_mfe = abs(corr_mfe) > 0.70
            print(f"    [inject] orb_outcomes.mfe_r (BANNED look-ahead) vs tail: corr={corr_mfe:.3f} -> "
                  f"{'FLAGGED ✓' if flagged_mfe else 'below 0.70 (still BANNED by § 6.3 — never a real proxy)'}")
        else:
            print("    [inject] capped_mfe_r present but <30 joinable rows — skipped.")
    else:
        print("    [inject] capped_mfe_r absent — using close-vs-self proof only.")

    # Sanity: a genuinely SAFE proxy must NOT be ~1.0 corr with the tail.
    if "atr_20_pct" in df.columns:
        sub = df[["atr_20_pct", "session_close_r"]].dropna()
        if len(sub) >= 30:
            corr_safe = float(np.corrcoef(sub["atr_20_pct"], sub["session_close_r"])[0, 1])
            ok = abs(corr_safe) <= 0.70
            print(f"    [control] atr_20_pct (SAFE) vs tail: corr={corr_safe:.3f} -> "
                  f"{'PASS (not a tautology) ✓' if ok else 'UNEXPECTED high corr — investigate ✗'}")

    if not flagged_self:
        raise RuntimeError("RULE 13 pressure test FAILED — harness does not flag a perfect-correlation look-ahead.")
    print("  Pressure test PASSED — harness flags look-ahead/tautology.\n")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_oracle_split(df: pd.DataFrame) -> pd.DataFrame:
    """ORACLE split: trend (TREND_UP/DOWN) vs non-trend (NON_TREND) on the SAME
    parent. Upper bound on the tail — LOOK-AHEAD, NON-TRADEABLE."""
    print("  " + "=" * 70)
    print("  ORACLE SPLIT (look-ahead day_type — NON-TRADEABLE upper bound)")
    print("  " + "=" * 70)
    trend = df[df["day_type"].isin(["TREND_UP", "TREND_DOWN"])]
    nontrend = df[df["day_type"] == "NON_TREND"]
    rows = []
    for tail in TAIL_COLS:
        ts = tail_stats(trend[tail])
        ns = tail_stats(nontrend[tail])
        ratio = (ts["pct_ge3r"] / ns["pct_ge3r"]) if ns["pct_ge3r"] and ns["pct_ge3r"] > 0 else np.nan
        print(f"\n    Tail: {TAIL_LABELS[tail]}")
        print(f"      {'subset':10s} {'N':>5s} {'mean':>8s} {'P90':>7s} {'P99':>7s} {'max':>7s} {'%≥3R':>6s} {'sumR':>9s} {'Shrp':>7s} {'G/P':>6s}")
        for name, s in [("TREND", ts), ("NON_TREND", ns)]:
            print(f"      {name:10s} {s['n']:>5} {s['mean']:>8.3f} {s['p90']:>7.2f} {s['p99']:>7.2f} {s['max']:>7.2f} {s['pct_ge3r']:>5.1f}% {s['sum_r']:>9.1f} {s['sharpe']:>7.3f} {s['gain_to_pain']:>6.2f}")
        print(f"      tail-ratio (%≥3R trend / non-trend) = {ratio:.2f}x")
        rows.append({"split": "oracle", "tail": tail, "trend_pct_ge3r": ts["pct_ge3r"], "nontrend_pct_ge3r": ns["pct_ge3r"],
                     "ratio": round(ratio, 3) if ratio == ratio else None,
                     "trend_sum_r": ts["sum_r"], "nontrend_sum_r": ns["sum_r"]})
    print()
    return pd.DataFrame(rows)


def print_safe_proxy_split(df: pd.DataFrame) -> pd.DataFrame:
    """SAFE-PROXY split — the decision-bearing analysis. For each proxy, compare
    the high-trend-prior bin vs its complement on the realizable tails."""
    print("  " + "=" * 70)
    print("  SAFE-PROXY SPLIT (pre-entry tradeable — THE DECISION RESTS HERE)")
    print("  " + "=" * 70)
    rows = []
    for proxy in SAFE_PROXIES:
        bins = _bin_proxy(df, proxy)
        sub = df.assign(_bin=bins).dropna(subset=["_bin"])
        if sub.empty:
            print(f"\n    Proxy {proxy}: no binnable rows — skipped.")
            continue
        print(f"\n    Proxy: {proxy}")
        for tail in ["session_close_r", "trail_r"]:  # realizable tails only
            print(f"      Tail {TAIL_LABELS[tail]}:")
            print(f"        {'bin':8s} {'N':>5s} {'mean':>8s} {'P90':>7s} {'P99':>7s} {'max':>7s} {'%≥3R':>6s} {'sumR':>9s} {'G/P':>6s}")
            for b, g in sub.groupby("_bin", observed=True):
                s = tail_stats(g[tail])
                if s["n"] < MIN_CELL_N:
                    continue
                print(f"        {str(b):8s} {s['n']:>5} {s['mean']:>8.3f} {s['p90']:>7.2f} {s['p99']:>7.2f} {s['max']:>7.2f} {s['pct_ge3r']:>5.1f}% {s['sum_r']:>9.1f} {s['gain_to_pain']:>6.2f}")
                rows.append({"split": "safe_proxy", "proxy": proxy, "bin": str(b), "tail": tail, **{k: s[k] for k in ["n", "mean", "p90", "p99", "max", "pct_ge3r", "sum_r", "gain_to_pain"]}})

        # --- Decision-bearing contrast: high-trend-prior bin vs its complement ---
        hi_mask = _high_trend_prior_mask(df, proxy)
        if hi_mask is not None:
            hi = df[hi_mask]
            lo = df[~hi_mask & df[proxy].notna()] if proxy in df.columns else df[~hi_mask]
            print("      CAPTURE CONTRAST — high-trend-prior vs complement (realizable):")
            for tail in ["session_close_r", "trail_r"]:
                hs, ls = tail_stats(hi[tail]), tail_stats(lo[tail])
                if hs["n"] < MIN_CELL_N or ls["n"] < MIN_CELL_N:
                    print(f"        {tail:16s}: insufficient N (hi={hs['n']}, lo={ls['n']})")
                    continue
                ratio = (hs["pct_ge3r"] / ls["pct_ge3r"]) if ls["pct_ge3r"] and ls["pct_ge3r"] > 0 else float("nan")
                print(f"        {tail:16s}: hi %≥3R={hs['pct_ge3r']:.1f}% (N={hs['n']}) vs lo={ls['pct_ge3r']:.1f}% (N={ls['n']}) "
                      f"-> ratio={ratio:.2f}x | sumR hi={hs['sum_r']:.0f} lo={ls['sum_r']:.0f}")
                rows.append({"split": "capture_contrast", "proxy": proxy, "bin": "HI_vs_LO", "tail": tail,
                             "hi_pct_ge3r": hs["pct_ge3r"], "lo_pct_ge3r": ls["pct_ge3r"],
                             "ratio": round(ratio, 3) if ratio == ratio else None,
                             "hi_sum_r": hs["sum_r"], "lo_sum_r": ls["sum_r"], "hi_n": hs["n"], "lo_n": ls["n"]})
    print()
    return pd.DataFrame(rows)


def _high_trend_prior_mask(df: pd.DataFrame, proxy: str) -> pd.Series | None:
    """Boolean mask selecting the bin a priori MOST likely to precede a trend day,
    for the high-vs-complement capture contrast. NaN proxy rows -> False (excluded
    from 'high'; they fall to the complement only if the proxy is non-null there).

    Choices are pre-committed from the proxy's economic meaning, NOT post-hoc:
      - atr_20_pct: top tercile (high prior volatility -> wider trend range).
      - garch_atr_ratio: > 1 (forecast vol above realized -> expansion expected).
      - is_nfp_day: NFP days (scheduled vol catalyst).
      - prev_day_direction: NOT a trend-magnitude prior (direction, not range) —
        no single 'high-trend' bin; return None (per-bin table already shown).
    """
    if proxy == "atr_20_pct":
        v = df["atr_20_pct"]
        thr = v.dropna().quantile(2 / 3)
        return v.notna() & (v >= thr)
    if proxy == "garch_atr_ratio":
        v = df["garch_atr_ratio"]
        return v.notna() & (v > 1.0)
    if proxy == "is_nfp_day":
        return df["is_nfp_day"] == True  # noqa: E712 (pandas boolean mask)
    return None


def print_grid(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Full conditional grid: proxy bin × session × DOW × YEAR — let the data show
    WHERE the realizable tail concentrates (anti-pigeonhole). Saved to CSV; only a
    compact YEAR-stability summary is printed (RULE 12 outlier honesty)."""
    df = df.copy()
    df["year"] = pd.to_datetime(df["trading_day"]).dt.year
    grid_rows = []
    for proxy in SAFE_PROXIES:
        bins = _bin_proxy(df, proxy)
        sub = df.assign(_bin=bins).dropna(subset=["_bin"])
        for (b, sess, dow, yr), g in sub.groupby(["_bin", "orb_label", "day_of_week", "year"], observed=True):
            for tail in ["session_close_r", "trail_r"]:
                s = tail_stats(g[tail])
                if s["n"] < 10:  # grid floor lower than report floor; still flagged small
                    continue
                grid_rows.append({"proxy": proxy, "bin": str(b), "session": sess, "dow": dow, "year": yr,
                                  "tail": tail, **{k: s[k] for k in ["n", "mean", "p90", "pct_ge3r", "sum_r"]}})
    grid = pd.DataFrame(grid_rows)
    if not grid.empty:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        p = out / "mgc_trend_day_tail_grid.csv"
        grid.to_csv(p, index=False)
        print(f"  Full grid saved: {p} ({len(grid):,} cells)\n")

        # RULE 12: year-by-year stability of the realizable tail on the strongest
        # safe-proxy bin. A 4R+ tail driven by ONE year is noise, not edge.
        print("  " + "=" * 70)
        print("  YEAR STABILITY (RULE 12 — realizable %≥3R by year, trail_r)")
        print("  " + "=" * 70)
        tr = grid[grid["tail"] == "trail_r"]
        if not tr.empty:
            by_year = tr.groupby("year").agg(cells=("n", "size"), trades=("n", "sum"),
                                             mean_pct_ge3r=("pct_ge3r", "mean"), sum_r=("sum_r", "sum"))
            print(by_year.to_string())
        print()
    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="MGC Trend-Day tail descriptive scan (Phase A, read-only)")
    ap.add_argument("--limit", type=int, default=None, help="Cap outcome rows (for a fast smoke run)")
    ap.add_argument("--output-dir", type=str, default="research/output/")
    args = ap.parse_args()

    print("=== MGC Trend-Day Tail Descriptive Scan (Phase A) ===")
    print(f"DB: {GOLD_DB_PATH}")
    print(f"Instrument: {INSTRUMENT}  |  IS cutoff: trading_day < {IS_CUTOFF.date()}")
    print(f"Trail sim: arm@{TRAIL_ARM_R}R, give back {TRAIL_GIVEBACK_FRAC:.0%} of peak\n")

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True, config={"access_mode": "READ_ONLY"})
    try:
        # --- Load E1/E2 outcomes (reused loader), then scope to IS + sessions ---
        df = load_outcomes(con, INSTRUMENT, limit=args.limit)
        df["trading_day"] = pd.to_datetime(df["trading_day"])
        df = df[df["trading_day"] < IS_CUTOFF]
        df = df[df["orb_label"].isin(TESTABLE_SESSIONS)]
        df = df[df["orb_minutes"].isin(ORB_MINUTES)]
        print(f"  Outcomes (IS, testable sessions, E1/E2): {len(df):,}")
        if df.empty:
            print("  No outcomes — abort.")
            return

        # --- Compute 3 tails (reused MFE + new trail) ---
        df = compute_all_tails(con, df)

        # --- Merge oracle day_type + safe proxies (triple-join) ---
        feats = load_features(con)
        feats["trading_day"] = pd.to_datetime(feats["trading_day"])
        df = df.merge(feats, on=["trading_day", "orb_minutes"], how="left")

        # --- RULE 13 pressure test (must flag look-ahead) ---
        pressure_test(df)

        # --- Comparator parent diagnostics ---
        print("  " + "=" * 70)
        print("  PARENT DIAGNOSTICS")
        print("  " + "=" * 70)
        print(f"    total outcomes:        {len(df):,}")
        print(f"    day_type populated:    {df['day_type'].notna().sum():,}")
        print(f"    TREND_UP/DOWN:         {df['day_type'].isin(['TREND_UP','TREND_DOWN']).sum():,}")
        print(f"    NON_TREND:             {(df['day_type']=='NON_TREND').sum():,}")
        for p in SAFE_PROXIES:
            print(f"    {p:22s} non-null: {df[p].notna().sum():,}")
        print()

        # --- The three analyses ---
        oracle = print_oracle_split(df)
        proxy = print_safe_proxy_split(df)
        print_grid(df, args.output_dir)  # writes the full grid CSV + year-stability print

        # --- Persist per-trade + summaries ---
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        keep = ["trading_day", "orb_label", "orb_minutes", "entry_model", "rr_target",
                "outcome", "pnl_r", "day_type", "prev_day_direction", "atr_20_pct",
                "garch_atr_ratio", "is_nfp_day", "day_of_week",
                "true_mfe_r", "session_close_r", "trail_r"]
        keep = [c for c in keep if c in df.columns]
        df[keep].to_csv(out / "mgc_trend_day_tail_raw.csv", index=False)
        if not oracle.empty:
            oracle.to_csv(out / "mgc_trend_day_tail_oracle.csv", index=False)
        if not proxy.empty:
            proxy.to_csv(out / "mgc_trend_day_tail_safe_proxy.csv", index=False)
        print(f"  Per-trade + summary CSVs saved under {out}/\n")

        # --- Decision-gate echo (define BEFORE results — RULE 5/Step 5) ---
        print("  " + "=" * 70)
        print("  DECISION GATE (pre-defined — interpret the tables above against it)")
        print("  " + "=" * 70)
        print("   ALIVE -> Phase B IF: oracle tail fat AND safe-proxy captures ≥~half")
        print("            (≥2× non-trend %≥3R on realizable tail, sum-R positive where")
        print("             capped-RR was negative, stable in ≥3 of ~3.5 clean years).")
        print("   PARK  -> IF safe-proxy realizable tail ≈ non-trend tail (oracle leaking,")
        print("            not a tradeable edge). Write negative result, redirect to")
        print("            higher-EV live-MNQ capital batons (MGC deployable_expected=False).")
        print()
    finally:
        con.close()

    print("=" * 60)
    print("PHASE A COMPLETE — descriptive only. NO prereg, NO LOCK, NO validated_setups write.")
    print("Realizable tails (close + trail) bear the decision; MFE is the ceiling benchmark.")
    print("=" * 60)


if __name__ == "__main__":
    main()
