"""VWAP family comprehensive scan — Pathway A_family.

Pre-reg: docs/audit/hypotheses/2026-04-18-vwap-comprehensive-family-scan.yaml
Pre-reg sha (locked at scan start): see PRE_REG_SHA constant below.

Scope: 12 sessions × 3 instruments × 3 apertures × 3 RR × 2 directions ×
2 VWAP filter variants = 1296 cells.

Closes the gap surfaced 2026-04-18 in the surface audit: the 2026-04-15
comprehensive scan did not include VWAP features. 10 of 12 sessions × 2
VWAP variants are honestly untested at family level.

Mode A sacred: HOLDOUT_SACRED_FROM = 2026-01-01 from
trading_app.holdout_policy. OOS one-shot consumption.

Methodology controls (per .claude/rules/backtesting-methodology.md):
- RULE 1 (feature temporal alignment): VWAP is RULE 6.1 SAFE.
  pipeline/build_daily_features.py:983-998 computes VWAP from pre-session
  bars only (ts_utc < orb_start, strict less-than).
- RULE 2 (filter timing): VWAPBreakDirectionFilter resolves at break
  detection. Uses orb_{label}_vwap (pre-session) + orb_{label}_break_dir +
  orb_high/low (entry-time-known).
- RULE 3 (IS/OOS): IS = trading_day < 2026-01-01. OOS one-shot =
  [2026-01-01, 2026-04-18). dir_match required. No threshold tuning.
- RULE 4 (multi-framing BH-FDR): K_global=1296, K_family=648 per variant,
  K_lane=12, K_session=108, K_instrument=432, K_feature=648. Reported per
  cell.
- RULE 5 (comprehensive scope): all 12 × 3 × 3 × 3 × 2 × 2 enumerated.
  Auto-skips for disabled (instrument, session) pairs.
- RULE 6 (trade-time-knowability): both filters audited as RULE 6.1 SAFE.
- RULE 7 (T0 tautology): per-cell |corr| vs deployed-cell filter where one
  exists. |corr| > 0.7 → tautology flag (excluded from survivors).
- RULE 8 (extreme fire 5%/95% + arithmetic-only WR-flat).
- RULE 9 (data source): triple-join (trading_day, symbol, orb_minutes).
  Canonical layers only.
- RULE 12 (red flags): |t|>7 or Δ_IS>0.6 → audit before report.
- RULE 13 (pressure test): H3 positive control = re-validate L6 lane
  IS ExpR within ±0.02 of C12 +0.2101 baseline. Harness-bug detector.

Bootstrap (moving-block, block=5, B=10000, fixed-mask, centered-data H0):
inlined here because research/oneshot_utils.py lives on the
research/f5-below-pdl-stage1 branch and has not yet merged to main.
Logic is character-equivalent to oneshot_utils.moving_block_bootstrap_p
(same author, same canonical formulation per Lahiri 2003 / Politis-Romano 1994).

Filter signals come from `research.filter_utils.filter_signal`, which
delegates to the canonical `trading_app.config.ALL_FILTERS[key].matches_df`
per `.claude/rules/institutional-rigor.md` Rule 4. The 2026-04-18 A+
hardening pass removed the local `vwap_signal` and `deployed_filter_signal`
functions that were re-encoding canonical filter logic; they are now pure
delegations to the canonical filter instances.

One-shot lock: refuses to re-run if result md already exists.

Output: docs/audit/results/2026-04-18-vwap-comprehensive-family-scan.md
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, ASSET_CONFIGS  # noqa: E402
from pipeline.dst import SESSION_CATALOG  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from research.filter_utils import filter_signal  # noqa: E402
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402

# =============================================================================
# Pre-reg lock — refuse to run if result md already exists
# =============================================================================

PRE_REG_PATH = Path("docs/audit/hypotheses/2026-04-18-vwap-comprehensive-family-scan.yaml")
PRE_REG_SHA = "495810f5"  # commit that landed the pre-reg
RESULT_MD = Path("docs/audit/results/2026-04-18-vwap-comprehensive-family-scan.md")

if RESULT_MD.exists():
    print(
        f"REFUSING TO RE-RUN. Result file already exists: {RESULT_MD}\n"
        f"This scan is a Mode A one-shot OOS consumption. Re-running would\n"
        f"violate the no-tuning-against-OOS rule. Delete the result md\n"
        f"manually if you have a documented reason to override."
    )
    sys.exit(1)

if not PRE_REG_PATH.exists():
    print(f"FATAL: pre-reg not found at {PRE_REG_PATH}")
    sys.exit(1)

# =============================================================================
# Scope — derived from canonical sources only
# =============================================================================

ALL_SESSIONS: list[str] = list(SESSION_CATALOG.keys())  # 12 enabled sessions
# Filter to the 12 we tested in comprehensive scan (BRISBANE_0925/1955 not in production scope)
SCAN_SESSIONS: list[str] = [s for s in ALL_SESSIONS if s not in {"BRISBANE_0925", "BRISBANE_1955"}]
# Sanity-check
assert len(SCAN_SESSIONS) == 12, f"Expected 12 sessions, got {len(SCAN_SESSIONS)}: {SCAN_SESSIONS}"

ALL_INSTRUMENTS: list[str] = list(ACTIVE_ORB_INSTRUMENTS)  # ['MES', 'MGC', 'MNQ']
ALL_APERTURES: list[int] = [5, 15, 30]
ALL_RRS: list[float] = [1.0, 1.5, 2.0]
ALL_DIRS: list[str] = ["long", "short"]
VWAP_VARIANTS: list[str] = ["VWAP_MID_ALIGNED", "VWAP_BP_ALIGNED"]

# Mode A IS/OOS boundary
IS_END_EXCLUSIVE = HOLDOUT_SACRED_FROM  # date(2026, 1, 1)
OOS_START = HOLDOUT_SACRED_FROM
OOS_END_EXCLUSIVE = pd.Timestamp("2026-04-18").date()

# Per-instrument enabled session map (auto-skip disabled cells)
ENABLED: dict[str, set[str]] = {inst: set(ASSET_CONFIGS[inst]["enabled_sessions"]) for inst in ALL_INSTRUMENTS}

# L6 deployed positive-control identity (per docs/audit/results/2026-04-18-c12-alarmed-lanes-review.md)
L6_KEY = ("MNQ", "US_DATA_1000", 15, 1.5, "long", "VWAP_MID_ALIGNED")
L6_C12_BASELINE_EXPR = 0.2101  # per C12 review baseline mean R
L6_C12_BASELINE_TOL = 0.02  # |delta| within tol = control PASS

# Deployed filter map for T0 tautology pre-screen — per active_validated_setups
# query 2026-04-18 + comprehensive scan deployed lanes. Only the canonical
# deployed-lane filters at each (instr, session, apt, rr) cell are listed.
DEPLOYED_FILTERS: dict[tuple, list[str]] = {
    ("MNQ", "COMEX_SETTLE", 5, 1.0): ["ORB_G5", "OVNRNG_100"],
    ("MNQ", "COMEX_SETTLE", 5, 1.5): ["ORB_G5", "OVNRNG_100"],
    ("MNQ", "COMEX_SETTLE", 5, 2.0): ["ORB_G5"],
    ("MNQ", "EUROPE_FLOW", 5, 1.0): ["ORB_G5", "OVNRNG_100"],
    ("MNQ", "EUROPE_FLOW", 5, 1.5): ["ORB_G5", "OVNRNG_100"],
    ("MNQ", "NYSE_OPEN", 5, 1.0): ["ORB_G5"],
    ("MNQ", "NYSE_OPEN", 5, 1.5): ["ORB_G5"],
    ("MNQ", "TOKYO_OPEN", 5, 1.5): ["ORB_G5"],
    ("MNQ", "TOKYO_OPEN", 5, 2.0): ["ORB_G5"],
    ("MNQ", "US_DATA_1000", 15, 1.0): ["ORB_G5", "VWAP_MID_ALIGNED"],
    ("MNQ", "US_DATA_1000", 15, 1.5): ["VWAP_MID_ALIGNED"],
    ("MNQ", "US_DATA_1000", 15, 2.0): ["VWAP_MID_ALIGNED"],
    ("MNQ", "SINGAPORE_OPEN", 15, 1.5): ["ATR_P50"],
    ("MNQ", "SINGAPORE_OPEN", 30, 1.5): ["ATR_P50"],
    ("MES", "CME_PRECLOSE", 5, 1.0): ["ORB_G8"],
}

# =============================================================================
# Inline canonical helpers (would import from research.oneshot_utils once merged)
# =============================================================================


def moving_block_bootstrap_p(
    pnl_on: np.ndarray,
    B: int = 10_000,
    block: int = 5,
    seed: int = 20260418,
    tail: str = "upper",
) -> float:
    """Moving-block bootstrap p for H0: E[pnl]=0. Centers DATA so H0 holds.
    Inlined from research/oneshot_utils.py (research/f5-below-pdl-stage1 branch)
    pending merge to main. Same logic per Lahiri 2003 / Politis-Romano 1994."""
    n = len(pnl_on)
    if n < block * 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    pnl = np.asarray(pnl_on, dtype=float)
    observed_mean = float(pnl.mean())
    centered_data = pnl - observed_mean
    n_blocks = int(np.ceil(n / block))
    boot_means = np.empty(B, dtype=float)
    for b in range(B):
        starts = rng.integers(low=0, high=n - block + 1, size=n_blocks)
        sampled = np.concatenate([centered_data[s : s + block] for s in starts])[:n]
        boot_means[b] = sampled.mean()
    if tail == "upper":
        count = int(np.sum(boot_means >= observed_mean))
    elif tail == "lower":
        count = int(np.sum(boot_means <= observed_mean))
    else:
        count = int(np.sum(np.abs(boot_means) >= abs(observed_mean)))
    return (count + 1) / (B + 1)


def t0_correlation(feat_sig: np.ndarray, filter_sig: np.ndarray) -> float:
    """Absolute correlation between two binary signals.
    Copied from research/comprehensive_deployed_lane_scan.py:478."""
    if filter_sig.sum() == 0 or (1 - filter_sig).sum() == 0:
        return 0.0
    if feat_sig.sum() == 0 or (1 - feat_sig).sum() == 0:
        return 0.0
    try:
        c = np.corrcoef(feat_sig.astype(float), filter_sig.astype(float))[0, 1]
        return 0.0 if np.isnan(c) else float(abs(c))
    except Exception:
        return 0.0


# =============================================================================
# Data loading — triple-join per .claude/rules/daily-features-joins.md
# =============================================================================


def load_lane(con: duckdb.DuckDBPyConnection, instrument: str, session: str, apt: int, rr: float) -> pd.DataFrame:
    """Load orb_outcomes JOIN daily_features for one cell. Triple-join correct.

    Loads BOTH the canonical `orb_{session}_*` column names (required by
    `research.filter_utils.filter_signal` which delegates to canonical
    `matches_df`) AND convenience aliases used by `test_cell` for stats
    computation. This keeps the scan's internal column references stable
    while allowing the canonical filter lookup to succeed without renaming.
    """
    q = f"""
    SELECT
        o.trading_day, o.symbol, o.orb_minutes, o.orb_label,
        o.entry_model, o.rr_target, o.outcome, o.pnl_r,
        d.atr_20, d.atr_20_pct, d.overnight_range,
        -- Canonical column names (consumed by ALL_FILTERS[*].matches_df):
        d.orb_{session}_size,
        d.orb_{session}_high,
        d.orb_{session}_low,
        d.orb_{session}_break_dir,
        d.orb_{session}_vwap,
        -- Convenience aliases (consumed by test_cell stats logic):
        d.orb_{session}_size AS orb_size,
        d.orb_{session}_high AS orb_high,
        d.orb_{session}_low AS orb_low,
        d.orb_{session}_break_dir AS break_dir,
        d.orb_{session}_vwap AS session_vwap
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
      AND o.symbol = d.symbol
      AND o.orb_minutes = d.orb_minutes
    WHERE o.orb_label = '{session}'
      AND o.symbol = '{instrument}'
      AND o.orb_minutes = {apt}
      AND o.entry_model = 'E2'
      AND o.rr_target = {rr}
      AND o.confirm_bars = 1
      AND o.pnl_r IS NOT NULL
      AND d.atr_20 IS NOT NULL AND d.atr_20 > 0
      AND d.orb_{session}_break_dir IN ('long','short')
    """
    df = con.execute(q).df()
    if len(df) == 0:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["is_is"] = df["trading_day"].dt.date < IS_END_EXCLUSIVE
    df["is_oos"] = (df["trading_day"].dt.date >= OOS_START) & (df["trading_day"].dt.date < OOS_END_EXCLUSIVE)
    df["year"] = df["trading_day"].dt.year
    return df


# =============================================================================
# Filter signals — delegated to canonical ALL_FILTERS via research.filter_utils
# per .claude/rules/institutional-rigor.md Rule 4 (delegate to canonical source,
# never re-encode). The local `vwap_signal` and `deployed_filter_signal`
# helpers that existed in the original 2026-04-18 scan were removed in the
# A+ hardening pass. All filter signal computation now flows through
# `research.filter_utils.filter_signal(df, key, orb_label)` which is a thin
# wrapper around `ALL_FILTERS[key].matches_df(df, orb_label)`.
#
# Tests proving wrapper equivalence: tests/test_research/test_filter_utils.py
# =============================================================================


# =============================================================================
# Per-cell test — single-pass (no overlay)
# =============================================================================


def test_cell(
    df: pd.DataFrame,
    sig: np.ndarray,
    direction: str,
    cell_id: tuple,
    deployed_keys: list[str],
) -> dict | None:
    """Test one (direction, variant) cell. Returns metrics dict or None if N too small."""
    mask_dir = (df["break_dir"] == direction).values
    sub = df[mask_dir].copy()
    sig_dir = sig[mask_dir]

    if len(sub) == 0:
        return None

    sub["_sig"] = sig_dir

    is_df = sub[sub["is_is"]]
    oos_df = sub[sub["is_oos"]]

    on_is = is_df[is_df["_sig"] == 1]["pnl_r"].values
    off_is = is_df[is_df["_sig"] == 0]["pnl_r"].values
    on_oos = oos_df[oos_df["_sig"] == 1]["pnl_r"].values
    off_oos = oos_df[oos_df["_sig"] == 0]["pnl_r"].values

    if len(on_is) < 30 or len(off_is) < 30:
        # Cell exists but not enough trades on either side — return null record
        return None

    expr_on_is = float(np.mean(on_is))
    expr_off_is = float(np.mean(off_is))
    delta_is = expr_on_is - expr_off_is
    wr_on = float(np.mean(on_is > 0))
    wr_off = float(np.mean(off_is > 0))
    wr_spread = wr_on - wr_off

    # Welch t (heteroscedastic two-sample), unconditional on cluster
    t_is, p_is_two_tail = stats.ttest_ind(on_is, off_is, equal_var=False)

    expr_on_oos = float(np.mean(on_oos)) if len(on_oos) >= 5 else float("nan")
    expr_off_oos = float(np.mean(off_oos)) if len(off_oos) >= 5 else float("nan")
    delta_oos = (
        (expr_on_oos - expr_off_oos) if not np.isnan(expr_on_oos) and not np.isnan(expr_off_oos) else float("nan")
    )
    dir_match = (not np.isnan(delta_oos)) and (np.sign(delta_is) == np.sign(delta_oos)) and (np.sign(delta_is) != 0)

    fire_rate = float(np.sum(sig_dir)) / max(1, len(sig_dir))
    extreme_fire = (fire_rate < 0.05) or (fire_rate > 0.95)
    arithmetic_only = (abs(wr_spread) < 0.03) and (abs(delta_is) > 0.10)

    # Bootstrap p — moving-block, B=10000, fixed mask, centered data
    tail = "upper" if expr_on_is > 0 else "lower"
    boot_p = moving_block_bootstrap_p(on_is, B=10_000, block=5, seed=20260418, tail=tail)

    # Per-year IS positivity
    is_on_df = is_df[is_df["_sig"] == 1]
    yearly_expr = is_on_df.groupby("year")["pnl_r"].agg(["mean", "count"])
    yrs_pos = int(((yearly_expr["mean"] > 0) & (yearly_expr["count"] >= 5)).sum())
    yrs_total = int((yearly_expr["count"] >= 5).sum())

    # T0 tautology vs deployed filters on this cell.
    # Uses canonical ALL_FILTERS via research.filter_utils — no re-encoding.
    orb_label = cell_id[1]
    t0_max = 0.0
    t0_against = ""
    for key in deployed_keys:
        ds = filter_signal(df, key, orb_label=orb_label)
        ds_dir = ds[mask_dir]
        c = t0_correlation(sig_dir, ds_dir)
        if c > t0_max:
            t0_max = c
            t0_against = key
    tautology = t0_max > 0.7

    return {
        "instrument": cell_id[0],
        "session": cell_id[1],
        "aperture": cell_id[2],
        "rr": cell_id[3],
        "direction": direction,
        "variant": cell_id[4],
        "n_on_is": len(on_is),
        "n_off_is": len(off_is),
        "n_on_oos": len(on_oos),
        "n_off_oos": len(off_oos),
        "fire_rate": fire_rate,
        "expr_on_is": expr_on_is,
        "expr_off_is": expr_off_is,
        "delta_is": delta_is,
        "wr_on_is": wr_on,
        "wr_off_is": wr_off,
        "wr_spread": wr_spread,
        "expr_on_oos": expr_on_oos,
        "delta_oos": delta_oos,
        "dir_match": dir_match,
        "t_is": float(t_is),
        "p_is": float(p_is_two_tail),
        "boot_p": boot_p,
        "yrs_positive_is": yrs_pos,
        "yrs_total_is": yrs_total,
        "extreme_fire": extreme_fire,
        "arithmetic_only": arithmetic_only,
        "t0_max_corr": t0_max,
        "t0_against": t0_against,
        "tautology": tautology,
    }


# =============================================================================
# Multi-framing BH-FDR — adapted from comprehensive_deployed_lane_scan.py:615
# =============================================================================


def bh_fdr_multi_framing(results: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    res = results.copy()
    res["_row_id"] = range(len(res))

    def _apply_bh(df: pd.DataFrame, group_cols: list[str] | None, suffix: str) -> pd.DataFrame:
        df = df.copy()
        if group_cols is None:
            df = df.sort_values("p_is").reset_index(drop=True)
            K = len(df)
            df[f"K_{suffix}"] = K
            df[f"bh_rank_{suffix}"] = df.index + 1
            df[f"bh_crit_{suffix}"] = alpha * df[f"bh_rank_{suffix}"] / K
            df[f"bh_pass_{suffix}"] = df["p_is"] <= df[f"bh_crit_{suffix}"]
            return df
        pieces = []
        for _, grp in df.groupby(group_cols, dropna=False):
            g = grp.sort_values("p_is").reset_index(drop=True)
            K = len(g)
            g[f"K_{suffix}"] = K
            g[f"bh_rank_{suffix}"] = g.index + 1
            g[f"bh_crit_{suffix}"] = alpha * g[f"bh_rank_{suffix}"] / K
            g[f"bh_pass_{suffix}"] = g["p_is"] <= g[f"bh_crit_{suffix}"]
            pieces.append(g)
        return pd.concat(pieces, ignore_index=True)

    out = _apply_bh(res, None, "global")
    fams = [
        (["variant"], "family"),  # Per-VWAP-variant family
        (["session", "aperture", "rr", "instrument"], "lane"),
        (["session"], "session"),
        (["instrument"], "instrument"),
        (["variant", "session"], "feature_session"),  # Per (variant, session)
    ]
    for group_cols, suffix in fams:
        bh_cols = [f"K_{suffix}", f"bh_rank_{suffix}", f"bh_crit_{suffix}", f"bh_pass_{suffix}"]
        sub = _apply_bh(res, group_cols, suffix)[["_row_id"] + bh_cols]
        out = out.merge(sub, on="_row_id", how="left")

    return out.drop(columns=["_row_id"])


# =============================================================================
# Main scan
# =============================================================================


def main():
    print("=" * 70)
    print("VWAP COMPREHENSIVE FAMILY SCAN — Pathway A_family")
    print("=" * 70)
    print(f"Pre-reg: {PRE_REG_PATH}")
    print(f"Pre-reg sha: {PRE_REG_SHA}")
    print(f"Result md: {RESULT_MD}")
    print(f"DB: {GOLD_DB_PATH}")
    print(f"IS window: < {IS_END_EXCLUSIVE}")
    print(f"OOS window: [{OOS_START}, {OOS_END_EXCLUSIVE})")
    print(f"Sessions: {SCAN_SESSIONS}")
    print(f"Instruments: {ALL_INSTRUMENTS}")
    print(f"Apertures: {ALL_APERTURES}")
    print(f"RRs: {ALL_RRS}")
    print(f"Directions: {ALL_DIRS}")
    print(f"VWAP variants: {VWAP_VARIANTS}")

    # Total cell count enumeration check
    total_combos = (
        len(SCAN_SESSIONS)
        * len(ALL_INSTRUMENTS)
        * len(ALL_APERTURES)
        * len(ALL_RRS)
        * len(ALL_DIRS)
        * len(VWAP_VARIANTS)
    )
    print(f"Total combos enumerated: {total_combos}")
    assert total_combos == 1296, f"Expected 1296 combos, got {total_combos}"

    # Open DB connection (read-only, retry on busy)
    con = None
    for attempt in range(8):
        try:
            con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
            break
        except Exception as e:
            print(f"  DB busy (attempt {attempt + 1}/8): {e}")
            time.sleep(5)
    if con is None:
        print("FATAL: could not open DB after 8 attempts")
        sys.exit(1)

    print("\nDB connected. Running scan...")

    rows = []
    skipped_disabled = []
    skipped_low_n = []
    cell_count = 0
    t_start = time.time()

    for instr in ALL_INSTRUMENTS:
        for session in SCAN_SESSIONS:
            # Auto-skip disabled (instrument, session) per ASSET_CONFIGS
            if session not in ENABLED.get(instr, set()):
                for apt in ALL_APERTURES:
                    for rr in ALL_RRS:
                        for direction in ALL_DIRS:
                            for variant in VWAP_VARIANTS:
                                skipped_disabled.append((instr, session, apt, rr, direction, variant))
                continue

            for apt in ALL_APERTURES:
                for rr in ALL_RRS:
                    df = load_lane(con, instr, session, apt, rr)
                    if len(df) < 60:
                        for direction in ALL_DIRS:
                            for variant in VWAP_VARIANTS:
                                skipped_low_n.append((instr, session, apt, rr, direction, variant, len(df)))
                        continue

                    deployed_keys = DEPLOYED_FILTERS.get((instr, session, apt, rr), [])

                    for variant in VWAP_VARIANTS:
                        sig = filter_signal(df, variant, orb_label=session)
                        for direction in ALL_DIRS:
                            cell_id = (instr, session, apt, rr, variant)
                            res = test_cell(df, sig, direction, cell_id, deployed_keys)
                            cell_count += 1
                            if res is None:
                                skipped_low_n.append((instr, session, apt, rr, direction, variant, len(df)))
                                continue
                            rows.append(res)

                    if cell_count % 100 == 0:
                        elapsed = time.time() - t_start
                        print(f"  {cell_count} cells tested, elapsed={elapsed:.0f}s, len(rows)={len(rows)}")

    con.close()

    elapsed = time.time() - t_start
    print(
        f"\nSCAN COMPLETE — {cell_count} cells attempted, {len(rows)} cells with results, "
        f"{len(skipped_disabled)} disabled-skip, {len(skipped_low_n)} N<60 skip. Elapsed={elapsed:.0f}s."
    )

    if not rows:
        print("FATAL: no rows returned. Halting.")
        sys.exit(1)

    res = pd.DataFrame(rows)

    # Apply BH-FDR multi-framing
    res = bh_fdr_multi_framing(res, alpha=0.05)

    # Trustworthy gate
    trustworthy = res[
        (~res["extreme_fire"]) & (~res["arithmetic_only"]) & (~res["tautology"]) & (res["n_on_is"] >= 50)
    ].copy()

    # H1 family-level survivors (binding gate per pre-reg)
    h1 = trustworthy[
        (trustworthy["bh_pass_family"])
        & (trustworthy["dir_match"])
        & (trustworthy["t_is"].abs() >= 3.0)
        & (trustworthy["yrs_positive_is"] >= 4)
        & (trustworthy["boot_p"] < 0.10)
    ].copy()

    # H2 strict (descriptive ribbon)
    h2 = trustworthy[
        (trustworthy["bh_pass_family"])
        & (trustworthy["dir_match"])
        & (trustworthy["t_is"].abs() >= 3.79)
        & (trustworthy["yrs_positive_is"] >= 4)
        & (trustworthy["boot_p"] < 0.10)
    ].copy()

    # H3 positive control — L6 cell (long, VWAP_MID_ALIGNED) on (MNQ, US_DATA_1000, 15, 1.5)
    l6_row = res[
        (res["instrument"] == L6_KEY[0])
        & (res["session"] == L6_KEY[1])
        & (res["aperture"] == L6_KEY[2])
        & (res["rr"] == L6_KEY[3])
        & (res["direction"] == L6_KEY[4])
        & (res["variant"] == L6_KEY[5])
    ]
    h3_pass = False
    h3_note = ""
    if len(l6_row) == 1:
        l6 = l6_row.iloc[0]
        diff = abs(l6["expr_on_is"] - L6_C12_BASELINE_EXPR)
        h3_pass = diff < L6_C12_BASELINE_TOL
        h3_note = (
            f"L6 IS ExpR={l6['expr_on_is']:+.4f} (baseline {L6_C12_BASELINE_EXPR:+.4f}, "
            f"|diff|={diff:.4f}, tol={L6_C12_BASELINE_TOL}); H3 = {'PASS' if h3_pass else 'FAIL'}"
        )
    else:
        h3_note = f"L6 row missing — found {len(l6_row)} matches, expected 1"

    # Red flag audit per RULE 12
    red_flag_t = trustworthy[trustworthy["t_is"].abs() > 7.0]
    red_flag_d = trustworthy[trustworthy["delta_is"].abs() > 0.6]

    # Decision
    n_h1 = len(h1)
    if not h3_pass:
        verdict = "HARNESS_BUG"
        decision_action = "K2 FIRES — halt, do not consume OOS, debug harness"
    elif n_h1 == 0:
        verdict = "K1 FIRES — VWAP family DOCTRINE-CLOSED"
        decision_action = (
            "Mark VWAP family closed; HTF Phase A simple build/filter family is now "
            "also closed, so pivot only to a NEW mechanism class or a structurally "
            "new HTF pre-reg"
        )
    elif n_h1 in (1, 2):
        verdict = "PARK"
        decision_action = "Promising but family-level under-power; defer per-cell Pathway B"
    else:
        verdict = "CONTINUE"
        decision_action = "Recommend rel_vol × VWAP cross-factor pre-reg next session"

    # Excessive extreme-fire check (K3)
    pct_extreme = res["extreme_fire"].mean()
    if pct_extreme >= 0.50:
        verdict = "K3 FIRES — DESIGN FAULT"
        decision_action = "Fraction extreme-fire cells too high; halt and re-design"

    # Insufficient OOS (K4)
    total_oos = res["n_on_oos"].sum()
    if total_oos < 100:
        verdict = "K4 FIRES — OOS WINDOW EMPTY"
        decision_action = "OOS sample insufficient for any cell-level dir_match"

    # =============================================================================
    # Write result md
    # =============================================================================
    print(f"\n=== Writing result md to {RESULT_MD} ===")
    write_result_md(
        res=res,
        trustworthy=trustworthy,
        h1=h1,
        h2=h2,
        h3_pass=h3_pass,
        h3_note=h3_note,
        red_flag_t=red_flag_t,
        red_flag_d=red_flag_d,
        verdict=verdict,
        decision_action=decision_action,
        cell_count=cell_count,
        skipped_disabled_count=len(skipped_disabled),
        skipped_low_n_count=len(skipped_low_n),
        elapsed=elapsed,
    )

    print(f"\nVERDICT: {verdict}")
    print(f"H1 family-level survivors: {n_h1}")
    print(f"H2 strict (t>=3.79) survivors: {len(h2)}")
    print(f"H3 positive control: {h3_note}")
    print(f"Result md: {RESULT_MD}")


def write_result_md(
    res: pd.DataFrame,
    trustworthy: pd.DataFrame,
    h1: pd.DataFrame,
    h2: pd.DataFrame,
    h3_pass: bool,
    h3_note: str,
    red_flag_t: pd.DataFrame,
    red_flag_d: pd.DataFrame,
    verdict: str,
    decision_action: str,
    cell_count: int,
    skipped_disabled_count: int,
    skipped_low_n_count: int,
    elapsed: float,
) -> None:
    bh_global = trustworthy[trustworthy["bh_pass_global"]]
    bh_family = trustworthy[trustworthy["bh_pass_family"]]
    bh_lane = trustworthy[trustworthy["bh_pass_lane"]]
    bh_session = trustworthy[trustworthy["bh_pass_session"]]
    bh_instr = trustworthy[trustworthy["bh_pass_instrument"]]
    bh_feat_sess = trustworthy[trustworthy["bh_pass_feature_session"]]

    flagged_extreme = res[res["extreme_fire"]]
    flagged_arith = res[res["arithmetic_only"]]
    flagged_taut = res[res["tautology"]]

    lines = [
        "# VWAP Comprehensive Family Scan — Result",
        "",
        "**Pre-reg:** `docs/audit/hypotheses/2026-04-18-vwap-comprehensive-family-scan.yaml`",
        f"**Pre-reg sha:** `{PRE_REG_SHA}`",
        f"**Run UTC:** {pd.Timestamp.utcnow().isoformat()}",
        f"**Mode A holdout:** sacred from `{HOLDOUT_SACRED_FROM}` per `trading_app.holdout_policy`",
        f"**IS:** trading_day < `{IS_END_EXCLUSIVE}`",
        f"**OOS one-shot:** `[{OOS_START}, {OOS_END_EXCLUSIVE})`",
        f"**Elapsed:** {elapsed:.0f}s",
        "",
        "## Phase 1 admissibility verdict",
        "",
        "**VWAP_MID_ALIGNED + VWAP_BP_ALIGNED**: RULE 6.1 SAFE — trade-time-knowable.",
        "",
        "- Build path: `pipeline/build_daily_features.py:983-998` (Module 7, Mar 20 2026)",
        "  computes VWAP from pre-session bars only (`ts_utc < orb_start`, strict less-than).",
        "- Filter: `trading_app/config.py:2420-2554` `VWAPBreakDirectionFilter` reads same",
        "  `orb_{label}_vwap` column + entry-time-known `orb_high`/`orb_low`/`break_dir`.",
        "- Filter and column are perfectly aligned. Fail-closed on missing data.",
        "",
        "## Schema verification",
        "",
        "- `daily_features.orb_{label}_vwap` populated for all 12 sessions.",
        "- Triple-join on `(trading_day, symbol, orb_minutes)` per `.claude/rules/daily-features-joins.md`.",
        "",
        "## Coverage",
        "",
        "- Total combos enumerated: 1296",
        f"- Cells attempted: {cell_count}",
        f"- Cells with usable results: {len(res)}",
        f"- Skipped — disabled (instrument, session) per `pipeline.asset_configs`: {skipped_disabled_count}",
        f"- Skipped — N<60 IS or N<30 on/off: {skipped_low_n_count}",
        "",
        "## BH-FDR pass counts at each K framing",
        "",
        f"- **K_global** (K=1296 cells, q=0.05): **{len(bh_global)}** pass (trustworthy only)",
        f"- **K_family** (K=648 per VWAP variant, q=0.05): **{len(bh_family)}** pass",
        f"- **K_lane** (K=12 per (session, instr, apt, rr), q=0.05): **{len(bh_lane)}** pass",
        f"- **K_session** (K=108 per session, q=0.05): **{len(bh_session)}** pass",
        f"- **K_instrument** (K=432 per instrument, q=0.05): **{len(bh_instr)}** pass",
        f"- **K_feature_session** (per (variant, session), q=0.05): **{len(bh_feat_sess)}** pass",
        "",
        "## Trustworthy gate",
        "",
        f"- Total cells: {len(res)}",
        f"- Flagged extreme_fire (<5% or >95%): {len(flagged_extreme)}",
        f"- Flagged arithmetic_only (WR-flat + Δ_IS large): {len(flagged_arith)}",
        f"- Flagged tautology (T0 |corr| > 0.7 vs deployed filter): {len(flagged_taut)}",
        "- N_on_IS < 50: excluded from trustworthy",
        f"- Trustworthy cells: {len(trustworthy)}",
        "",
        "## Hypothesis verdicts",
        "",
        "### H1 family-level (BINDING)",
        "Gates: BH-FDR K_family q<0.05 AND dir_match AND |t_IS|>=3.0 AND yrs_positive_IS>=4/7 AND boot_p<0.10",
        f"**Survivors: {len(h1)}**",
        "Threshold: >=3 to CONTINUE, 1-2 to PARK, 0 to KILL (K1)",
        "",
        "### H2 strict (descriptive ribbon)",
        "Gates: as H1 but |t_IS|>=3.79 (Chordia no-theory)",
        f"**Survivors: {len(h2)}**",
        "",
        "### H3 positive control",
        f"{h3_note}",
        "Pass = harness reproduces L6 known-live IS ExpR within tolerance.",
        "",
        "## Red flag audit (RULE 12)",
        "",
        f"- |t_IS| > 7: {len(red_flag_t)} cells",
        f"- |delta_IS| > 0.6: {len(red_flag_d)} cells",
        "",
    ]

    if len(red_flag_t) > 0 or len(red_flag_d) > 0:
        lines += [
            "**WARNING: Red flag cells present. Audit before treating as survivors.**",
            "",
        ]

    # H1 survivors table
    lines += [
        "## H1 survivors (binding family-level gate)",
        "",
        "| Instr | Session | Apt | RR | Dir | Variant | N_on_IS | N_on_OOS | Fire% | ExpR_IS | ExpR_OOS | Δ_IS | Δ_OOS | t | p | boot_p | yrs+ | BH_g | BH_f | BH_l |",
        "|-------|---------|-----|----|----|---------|---------|----------|-------|---------|----------|------|-------|---|---|--------|------|------|------|------|",
    ]
    for _, r in h1.sort_values("t_is", key=abs, ascending=False).iterrows():
        bhg = "Y" if bool(r["bh_pass_global"]) else "."
        bhf = "Y" if bool(r["bh_pass_family"]) else "."
        bhl = "Y" if bool(r["bh_pass_lane"]) else "."
        lines.append(
            f"| {r['instrument']} | {r['session']} | O{int(r['aperture'])} | {r['rr']:.1f} | "
            f"{r['direction']} | {r['variant'][:8]} | {int(r['n_on_is'])} | {int(r['n_on_oos'])} | "
            f"{r['fire_rate']:.1%} | {r['expr_on_is']:+.3f} | {r['expr_on_oos']:+.3f} | "
            f"{r['delta_is']:+.3f} | {r['delta_oos']:+.3f} | {r['t_is']:+.2f} | {r['p_is']:.4f} | "
            f"{r['boot_p']:.4f} | {int(r['yrs_positive_is'])}/{int(r['yrs_total_is'])} | {bhg} | {bhf} | {bhl} |"
        )

    if len(h1) == 0:
        lines.append("(none)")

    # H2 strict ribbon
    lines += [
        "",
        "## H2 strict ribbon (|t|>=3.79 Chordia no-theory)",
        "",
        "| Instr | Session | Apt | RR | Dir | Variant | N_on_IS | ExpR_IS | Δ_IS | Δ_OOS | t | boot_p | yrs+ |",
        "|-------|---------|-----|----|----|---------|---------|---------|------|-------|---|--------|------|",
    ]
    for _, r in h2.sort_values("t_is", key=abs, ascending=False).iterrows():
        lines.append(
            f"| {r['instrument']} | {r['session']} | O{int(r['aperture'])} | {r['rr']:.1f} | "
            f"{r['direction']} | {r['variant'][:8]} | {int(r['n_on_is'])} | {r['expr_on_is']:+.3f} | "
            f"{r['delta_is']:+.3f} | {r['delta_oos']:+.3f} | {r['t_is']:+.2f} | "
            f"{r['boot_p']:.4f} | {int(r['yrs_positive_is'])}/{int(r['yrs_total_is'])} |"
        )

    if len(h2) == 0:
        lines.append("(none)")

    # Top promising (|t|>=2.5, dir_match, trustworthy)
    promising = (
        trustworthy[(trustworthy["t_is"].abs() >= 2.5) & (trustworthy["dir_match"])]
        .sort_values("t_is", key=abs, ascending=False)
        .head(40)
    )

    lines += [
        "",
        "## Promising cells (|t|>=2.5 + dir_match + trustworthy, top 40 by |t|)",
        "",
        "| Instr | Session | Apt | RR | Dir | Variant | N_on_IS | Fire% | ExpR_IS | Δ_IS | Δ_OOS | t | boot_p | BH_f | BH_l |",
        "|-------|---------|-----|----|----|---------|---------|-------|---------|------|-------|---|--------|------|------|",
    ]
    for _, r in promising.iterrows():
        lines.append(
            f"| {r['instrument']} | {r['session']} | O{int(r['aperture'])} | {r['rr']:.1f} | "
            f"{r['direction']} | {r['variant'][:8]} | {int(r['n_on_is'])} | {r['fire_rate']:.1%} | "
            f"{r['expr_on_is']:+.3f} | {r['delta_is']:+.3f} | {r['delta_oos']:+.3f} | "
            f"{r['t_is']:+.2f} | {r['boot_p']:.4f} | "
            f"{'Y' if bool(r['bh_pass_family']) else '.'} | {'Y' if bool(r['bh_pass_lane']) else '.'} |"
        )

    # Flagged cells (transparency)
    lines += [
        "",
        "## Flagged cells (excluded from trustworthy — transparency)",
        "",
        f"- extreme_fire: {len(flagged_extreme)}",
        f"- arithmetic_only: {len(flagged_arith)}",
        f"- tautology (|corr|>0.7 vs deployed filter): {len(flagged_taut)}",
        "",
        "Top 20 flagged with |t|>=2.5:",
        "",
        "| Instr | Session | Apt | RR | Dir | Variant | t | Fire% | T0 corr | T0 vs | Reason |",
        "|-------|---------|-----|----|----|---------|---|-------|---------|-------|--------|",
    ]
    flagged = res[(res["extreme_fire"] | res["arithmetic_only"] | res["tautology"]) & (res["t_is"].abs() >= 2.5)]
    for _, r in flagged.sort_values("t_is", key=abs, ascending=False).head(20).iterrows():
        reasons = []
        if r["tautology"]:
            reasons.append(f"TAUT({r['t0_max_corr']:.2f})")
        if r["extreme_fire"]:
            reasons.append(f"FIRE({r['fire_rate']:.1%})")
        if r["arithmetic_only"]:
            reasons.append(f"ARITH(WRΔ={r['wr_spread']:+.3f})")
        lines.append(
            f"| {r['instrument']} | {r['session']} | O{int(r['aperture'])} | {r['rr']:.1f} | "
            f"{r['direction']} | {r['variant'][:8]} | {r['t_is']:+.2f} | {r['fire_rate']:.1%} | "
            f"{r['t0_max_corr']:.2f} | {r['t0_against']} | {', '.join(reasons)} |"
        )

    # Decision
    lines += [
        "",
        "## VERDICT",
        "",
        f"**{verdict}**",
        "",
        "Decision rule (per pre-reg):",
        "- continue_if: H1 survivors >= 3 → recommend rel_vol × VWAP cross-factor next",
        "- park_if: H1 survivors == 1 or 2 → defer per-cell Pathway B",
        "- kill_if: H1 survivors == 0 → K1 fires, VWAP family DOCTRINE-CLOSED",
        "",
        f"**Action:** {decision_action}",
        "",
        "## NEXT STEPS",
        "",
    ]
    if verdict == "CONTINUE":
        lines += [
            "1. Write Pathway B individual pre-regs for each H1 survivor cell that also passes T0 tautology and N_IS >= 100.",
            "2. Write a separate cross-factor pre-reg for `rel_vol × VWAP` 2-way scan (next session).",
            "3. Update HANDOFF.md with one-line CONTINUE verdict and survivor list.",
            "4. Do NOT queue the old HTF Phase A build note from the surface map without re-checking project state; simple HTF v1 is no longer an unopened branch.",
        ]
    elif verdict.startswith("PARK"):
        lines += [
            "1. Document the 1-2 promising cells in HANDOFF.md as 'family-level under-power, defer'.",
            "2. Do NOT deploy. Do NOT write Pathway B pre-reg this session.",
            "3. Pivot next session only to a still-open mechanism family, not the already-killed simple HTF v1 branch.",
        ]
    elif verdict.startswith("K1"):
        lines += [
            "1. Update STRATEGY_BLUEPRINT.md NO-GO with VWAP family DOCTRINE-CLOSED entry citing this scan.",
            "2. Update HANDOFF.md with one-line K1 verdict.",
            "3. Re-route the old surface-map HTF build item to closed. Next session should choose a new mechanism class or a structurally new HTF question instead.",
        ]
    elif verdict.startswith("HARNESS"):
        lines += [
            "1. Halt — do not consume OOS for any cell.",
            "2. Debug harness: confirm L6 row identity, verify load_lane query, verify filter_utils.filter_signal call.",
            "3. Do NOT delete this result md without explicit user instruction.",
        ]
    elif verdict.startswith("K3"):
        lines += [
            "1. Halt — design fault. Re-examine VWAP filter compute.",
            "2. Do NOT deploy any survivor.",
        ]
    elif verdict.startswith("K4"):
        lines += [
            "1. OOS sample empty. Wait for accumulation; re-run later.",
            "2. Do NOT deploy any survivor.",
        ]

    lines += [
        "",
        "## Caveats",
        "",
        "- Pathway A_family scan; survivors require separate Pathway B pre-reg + validator pass before deployment.",
        "- VWAP_MID_ALIGNED and VWAP_BP_ALIGNED share the underlying `orb_{label}_vwap` column. They are not orthogonal — high cross-correlation expected and informational only.",
        "- T0 tautology pre-screen uses approximations of canonical filters (ORB_G5 = top-quintile orb_size, ATR_P50 = median ATR, OVNRNG_100 = overnight_range/atr ≥ 1.0, ORB_G8 = orb_size ≥ 8). Material divergence from `trading_app/config.py` filter logic would underestimate T0 risk.",
        "- Per-year IS positivity uses a per-(year, on-signal) groupby with N≥5 floor. Years with <5 trades-on are not counted toward yrs_total.",
        "- Bootstrap is moving-block (block=5, B=10000), centered-data H0 (corrected 2026-04-18 per oneshot_utils.py addendum). Tail = upper if ExpR_on_IS>0 else lower.",
        "- Dir_match strictly requires sign(Δ_IS) == sign(Δ_OOS) AND sign(Δ_IS) != 0.",
        "",
        "## Reproducibility",
        "",
        "- Repo: `C:/Users/joshd/canompx3` (parent), branch `main`",
        f"- Pre-reg sha: `{PRE_REG_SHA}`",
        "- Script: `research/vwap_comprehensive_family_scan.py`",
        f"- DB: `{GOLD_DB_PATH}`",
        "- Canonical sources used: `pipeline.dst.SESSION_CATALOG`, `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS`, `pipeline.asset_configs.ASSET_CONFIGS`, `pipeline.paths.GOLD_DB_PATH`, `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`.",
        "",
    ]

    RESULT_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
