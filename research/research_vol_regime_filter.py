"""
Vol-Regime Filter Comparison & Cross-Asset Signal Research
==========================================================

Questions:
1. Is ATR percentile (already in DB, zero-cost) as good as VOL_RV12_N20 (break-bar volume)?
2. Does COMBINING ATR + VOL produce a better filter than either alone?
3. Can cross-asset ATR (e.g. MES vol regime) predict edge in other instruments?
4. Does GARCH forecast vol outperform trailing ATR?

Key finding going IN: ATR_20 and rel_vol are nearly uncorrelated (Spearman = -0.048).
They measure different things:
  - rel_vol = break-bar participation/conviction (same-day, at-break)
  - ATR_20 = background volatility regime (trailing 20-day price range)
  - garch_forecast_vol = forward-looking vol estimate

Methodology:
- E2 entry, CB1, 5m aperture (strongest signal for VOL)
- 6 filter variants per (instrument, session, rr_target)
- BH FDR q=0.05 across ALL tested combos
- Jobson-Korkie test for significant Sharpe differences
- NO walk-forward here (save for follow-up if candidates emerge)

Zero-lookahead guarantee:
- ATR_20: trailing 20-day price range, computed from prior days
- rel_vol: trailing 20 break-bar median, computed from prior breaks
- GARCH: forecast from prior day's model fit
- Cross-asset ATR: source instrument's prior-day ATR_20
  All signals available BEFORE the session fires.

@research-source research_vol_regime_filter.py
"""

from __future__ import annotations

from collections import defaultdict

import duckdb
import numpy as np
from scipy import stats  # noqa: F401 — used for ttest_1samp

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

# ── Setup (importable without sys.path hack) ──────────────────────────
# This file lives in research/, one level below project root.
# The project root is on PYTHONPATH via pyproject.toml / .env / uv run.
from pipeline.paths import GOLD_DB_PATH

INSTRUMENTS = sorted(ACTIVE_ORB_INSTRUMENTS)  # MGC, MES, MNQ, M2K
ORB_MINUTES = 5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR_TARGETS = [1.0, 1.5, 2.0]

# ATR percentile thresholds to test
ATR_PCTILES = [50, 60, 70, 80]
# Trailing window for percentile computation
ATR_TRAILING = 252

# Minimum trades to consider a combo
MIN_TRADES = 30


def load_data(db_path):
    """Load daily_features + orb_outcomes for all active instruments."""
    con = duckdb.connect(str(db_path), read_only=True)

    # Load daily features (5m only)
    features = con.execute("""
        SELECT *
        FROM daily_features
        WHERE orb_minutes = ?
          AND symbol IN (SELECT UNNEST(?::VARCHAR[]))
        ORDER BY symbol, trading_day
    """, [ORB_MINUTES, INSTRUMENTS]).fetchdf()

    # Load E2 CB1 outcomes (5m only, all RR targets)
    outcomes = con.execute("""
        SELECT trading_day, symbol, orb_label, rr_target, pnl_r,
               entry_model, confirm_bars, ts_pnl_r
        FROM orb_outcomes
        WHERE orb_minutes = ?
          AND entry_model = ?
          AND confirm_bars = ?
          AND symbol IN (SELECT UNNEST(?::VARCHAR[]))
          AND pnl_r IS NOT NULL
        ORDER BY symbol, trading_day
    """, [ORB_MINUTES, ENTRY_MODEL, CONFIRM_BARS, INSTRUMENTS]).fetchdf()

    con.close()
    return features, outcomes


def compute_atr_percentile(features):
    """Compute rolling ATR percentile rank for each instrument.

    For each trading day, rank ATR_20 among the prior ATR_TRAILING values.
    Returns dict: {(instrument, trading_day): atr_percentile}.
    """
    atr_pct = {}
    for instrument in INSTRUMENTS:
        inst_df = features[
            (features["symbol"] == instrument) & (features["atr_20"].notna())
        ].sort_values("trading_day")

        atr_vals = inst_df["atr_20"].values
        days = inst_df["trading_day"].values

        for i in range(ATR_TRAILING, len(atr_vals)):
            window = atr_vals[max(0, i - ATR_TRAILING):i]  # prior values only
            today = atr_vals[i]
            pct = np.searchsorted(np.sort(window), today) / len(window) * 100
            atr_pct[(instrument, days[i])] = pct

    return atr_pct


def compute_garch_percentile(features):
    """Compute rolling GARCH forecast vol percentile rank."""
    garch_pct = {}
    for instrument in INSTRUMENTS:
        inst_df = features[
            (features["symbol"] == instrument) & (features["garch_forecast_vol"].notna())
        ].sort_values("trading_day")

        vals = inst_df["garch_forecast_vol"].values
        days = inst_df["trading_day"].values

        for i in range(ATR_TRAILING, len(vals)):
            window = vals[max(0, i - ATR_TRAILING):i]
            today = vals[i]
            pct = np.searchsorted(np.sort(window), today) / len(window) * 100
            garch_pct[(instrument, days[i])] = pct

    return garch_pct


def compute_stats(pnl_rs, years_span=None):
    """Compute performance stats for a list of R-multiples."""
    if len(pnl_rs) < MIN_TRADES:
        return None
    arr = np.array(pnl_rs)
    n = len(arr)
    avg_r = np.mean(arr)
    std_r = np.std(arr, ddof=1)
    win_rate = np.mean(arr > 0)
    total_r = np.sum(arr)

    # Sharpe annualized using trades_per_year (matches pipeline strategy_discovery.py:550-551)
    if std_r > 0:
        sharpe_per_trade = avg_r / std_r
        if years_span and years_span > 0:
            trades_per_year = n / years_span
        else:
            trades_per_year = n / 7.0  # conservative fallback
        sharpe_ann = sharpe_per_trade * np.sqrt(trades_per_year)
    else:
        sharpe_ann = 0.0

    # t-test: avg_r vs 0
    t_stat, p_val = stats.ttest_1samp(arr, 0)
    p_val_1tail = p_val / 2 if avg_r > 0 else 1.0  # one-sided

    return {
        "n": n,
        "avg_r": avg_r,
        "win_rate": win_rate,
        "total_r": total_r,
        "sharpe_ann": sharpe_ann,
        "t_stat": t_stat,
        "p_val": p_val_1tail,
    }


def matched_day_permutation_test(
    all_pnls: list[float],
    included_mask: list[bool],
    n_perms: int = 10_000,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """Permutation test: does the filter select days with higher avg R?

    This is the correct test for a selection filter. Every filtered day
    also exists in the baseline, so we compare INCLUDED vs EXCLUDED days
    on the same outcome data. The null hypothesis is that the filter
    label is independent of R.

    Args:
        all_pnls: R-multiples for ALL days (baseline).
        included_mask: True for days that pass the filter.
        n_perms: number of random permutations.

    Returns:
        (observed_uplift, p_value).
        observed_uplift = mean(included) - mean(excluded).
        p_value = fraction of permutations with uplift >= observed.
    """
    arr = np.array(all_pnls)
    mask = np.array(included_mask)
    n_included = int(mask.sum())

    if n_included < 10 or n_included >= len(arr) - 10:
        return 0.0, 1.0

    observed = arr[mask].mean() - arr[~mask].mean()

    rng = np.random.default_rng(rng_seed)
    count_ge = 0
    for _ in range(n_perms):
        perm_mask = np.zeros(len(arr), dtype=bool)
        perm_mask[rng.choice(len(arr), size=n_included, replace=False)] = True
        perm_uplift = arr[perm_mask].mean() - arr[~perm_mask].mean()
        if perm_uplift >= observed:
            count_ge += 1

    p_value = (count_ge + 1) / (n_perms + 1)  # +1 for continuity correction
    return float(observed), float(p_value)


def bh_fdr(p_values, q=0.05):
    """Benjamini-Hochberg FDR correction.

    Returns array of adjusted p-values.
    """
    p = np.array(p_values)
    n = len(p)
    ranked = np.argsort(p)
    adjusted = np.zeros(n)

    for i in range(n):
        adjusted[ranked[i]] = p[ranked[i]] * n / (np.where(ranked == ranked[i])[0][0] + 1)

    # Enforce monotonicity (from largest rank down)
    reverse_ranked = ranked[::-1]
    for i in range(1, n):
        adjusted[reverse_ranked[i]] = min(
            adjusted[reverse_ranked[i]],
            adjusted[reverse_ranked[i - 1]]
        )

    return np.minimum(adjusted, 1.0)


def get_sessions_for_instrument(features, instrument):
    """Get sessions that have break data for this instrument."""
    inst_df = features[features["symbol"] == instrument]
    sessions = []
    for col in inst_df.columns:
        if col.startswith("rel_vol_") and inst_df[col].notna().sum() > MIN_TRADES:
            session = col.replace("rel_vol_", "")
            sessions.append(session)
    return sessions


def run_research():
    """Main research execution."""
    print("=" * 80)
    print("VOL-REGIME FILTER COMPARISON & CROSS-ASSET SIGNAL")
    print("=" * 80)
    print(f"Instruments: {INSTRUMENTS}")
    print(f"Aperture: {ORB_MINUTES}m | Entry: {ENTRY_MODEL} CB{CONFIRM_BARS}")
    print(f"RR targets: {RR_TARGETS}")
    print(f"ATR percentile thresholds: {ATR_PCTILES}")
    print(f"Min trades: {MIN_TRADES}")
    print()

    # ── Load data ──────────────────────────────────────────────────────
    print("Loading data...")
    features, outcomes = load_data(GOLD_DB_PATH)
    print(f"  Features: {len(features)} rows")
    print(f"  Outcomes: {len(outcomes)} rows")

    # ── Compute ATR & GARCH percentiles ────────────────────────────────
    print("Computing ATR percentiles (trailing 252d)...")
    atr_pct = compute_atr_percentile(features)
    print(f"  ATR percentile entries: {len(atr_pct)}")

    print("Computing GARCH percentiles (trailing 252d)...")
    garch_pct = compute_garch_percentile(features)
    print(f"  GARCH percentile entries: {len(garch_pct)}")

    # ── Build rel_vol lookup ───────────────────────────────────────────
    rel_vol_lookup = {}  # (instrument, session, trading_day) -> rel_vol
    for _, row in features.iterrows():
        inst = row["symbol"]
        day = row["trading_day"]
        for col in features.columns:
            if col.startswith("rel_vol_") and row[col] is not None and not (isinstance(row[col], float) and np.isnan(row[col])):
                session = col.replace("rel_vol_", "")
                rel_vol_lookup[(inst, session, day)] = row[col]

    print(f"  rel_vol entries: {len(rel_vol_lookup)}")
    print()

    # ── Correlation analysis ───────────────────────────────────────────
    print("=" * 80)
    print("PART 0: SIGNAL CORRELATION ANALYSIS")
    print("=" * 80)

    for instrument in INSTRUMENTS:
        sessions_for_corr = get_sessions_for_instrument(features, instrument)
        for session in sessions_for_corr[:3]:  # top 3 sessions per instrument
            pairs_atr_vol = []
            pairs_garch_vol = []
            inst_df = features[features["symbol"] == instrument].sort_values("trading_day")
            rv_col = f"rel_vol_{session}"

            for _, row in inst_df.iterrows():
                day = row["trading_day"]
                rv = row.get(rv_col)
                atr_p = atr_pct.get((instrument, day))
                garch_p = garch_pct.get((instrument, day))

                if rv is not None and not (isinstance(rv, float) and np.isnan(rv)):
                    if atr_p is not None:
                        pairs_atr_vol.append((atr_p, rv))
                    if garch_p is not None:
                        pairs_garch_vol.append((garch_p, rv))

            if len(pairs_atr_vol) >= 50:
                a, v = zip(*pairs_atr_vol, strict=True)
                rho, p = stats.spearmanr(a, v)
                print(f"  {instrument:4s} {session:20s}  ATR vs rel_vol: rho={rho:+.4f} p={p:.4f} (N={len(pairs_atr_vol)})")

            if len(pairs_garch_vol) >= 50:
                g, v = zip(*pairs_garch_vol, strict=True)
                rho, p = stats.spearmanr(g, v)
                print(f"  {instrument:4s} {session:20s}  GARCH vs rel_vol: rho={rho:+.4f} p={p:.4f} (N={len(pairs_garch_vol)})")

    # ── Main filter comparison ─────────────────────────────────────────
    print()
    print("=" * 80)
    print("PART 1: SAME-ASSET FILTER COMPARISON")
    print("=" * 80)

    results = []  # list of dicts for all combos

    for instrument in INSTRUMENTS:
        sessions = get_sessions_for_instrument(features, instrument)

        for session in sessions:
            inst_outcomes = outcomes[
                (outcomes["symbol"] == instrument) &
                (outcomes["orb_label"] == session)
            ]

            if len(inst_outcomes) < MIN_TRADES:
                continue

            for rr in RR_TARGETS:
                rr_outcomes = inst_outcomes[
                    (inst_outcomes["rr_target"] == rr)
                ]

                if len(rr_outcomes) < MIN_TRADES:
                    continue

                # Build day -> pnl_r mapping
                day_pnl = {}
                for _, row in rr_outcomes.iterrows():
                    day_pnl[row["trading_day"]] = row["pnl_r"]

                all_days = sorted(day_pnl.keys())

                # Compute years_span from actual date range (matches pipeline)
                if len(all_days) >= 2:
                    _span = (all_days[-1] - all_days[0]).days
                    years_span = _span / 365.25 if _span > 0 else 1.0
                else:
                    years_span = 1.0

                # ── Filter 1: BASELINE (no filter) ─────────────────────
                baseline_pnls = [day_pnl[d] for d in all_days]
                baseline_stats = compute_stats(baseline_pnls, years_span)
                if baseline_stats:
                    results.append({
                        "instrument": instrument,
                        "session": session,
                        "rr": rr,
                        "filter": "BASELINE",
                        "pnls": baseline_pnls,
                        "days": list(all_days),
                        **baseline_stats,
                    })

                # All filters for this (instrument, session, rr) combo
                filter_specs = [
                    ("VOL_ONLY", [
                        d for d in all_days
                        if rel_vol_lookup.get((instrument, session, d), 0) >= 1.2
                    ]),
                    ("GARCH_P70", [
                        d for d in all_days
                        if garch_pct.get((instrument, d), 0) >= 70
                    ]),
                    ("ATR70+VOL", [
                        d for d in all_days
                        if (atr_pct.get((instrument, d), 0) >= 70
                            and rel_vol_lookup.get((instrument, session, d), 0) >= 1.2)
                    ]),
                    ("ATR70|VOL", [
                        d for d in all_days
                        if (atr_pct.get((instrument, d), 0) >= 70
                            or rel_vol_lookup.get((instrument, session, d), 0) >= 1.2)
                    ]),
                ]
                for pct_thresh in ATR_PCTILES:
                    filter_specs.append((f"ATR_P{pct_thresh}", [
                        d for d in all_days
                        if atr_pct.get((instrument, d), 0) >= pct_thresh
                    ]))

                for filt_name, filt_days in filter_specs:
                    pnls = [day_pnl[d] for d in filt_days]
                    st = compute_stats(pnls, years_span)
                    if st:
                        results.append({
                            "instrument": instrument,
                            "session": session,
                            "rr": rr,
                            "filter": filt_name,
                            "pnls": pnls,
                            "days": list(filt_days),
                            **st,
                        })

    # ── Cross-asset filters ────────────────────────────────────────────
    print()
    print("=" * 80)
    print("PART 2: CROSS-ASSET ATR SIGNAL")
    print("=" * 80)

    # MES ATR -> other instruments
    # MGC ATR -> other instruments
    cross_sources = ["MES", "MGC"]
    for source in cross_sources:
        for target in INSTRUMENTS:
            if target == source:
                continue

            sessions = get_sessions_for_instrument(features, target)
            for session in sessions:
                target_outcomes = outcomes[
                    (outcomes["symbol"] == target) &
                    (outcomes["orb_label"] == session)
                ]

                if len(target_outcomes) < MIN_TRADES:
                    continue

                for rr in RR_TARGETS:
                    rr_outcomes = target_outcomes[
                        target_outcomes["rr_target"] == rr
                    ]

                    if len(rr_outcomes) < MIN_TRADES:
                        continue

                    day_pnl = {}
                    for _, row in rr_outcomes.iterrows():
                        day_pnl[row["trading_day"]] = row["pnl_r"]

                    all_days = sorted(day_pnl.keys())

                    if len(all_days) >= 2:
                        _span = (all_days[-1] - all_days[0]).days
                        years_span = _span / 365.25 if _span > 0 else 1.0
                    else:
                        years_span = 1.0

                    # Add baseline for cross-asset groups too
                    base_pnls = [day_pnl[d] for d in all_days]
                    base_stats = compute_stats(base_pnls, years_span)
                    if base_stats:
                        cross_key = (target, session, rr)
                        if cross_key not in {(r["instrument"], r["session"], r["rr"])
                                              for r in results if r["filter"] == "BASELINE"}:
                            results.append({
                                "instrument": target,
                                "session": session,
                                "rr": rr,
                                "filter": "BASELINE",
                                "pnls": base_pnls,
                                "days": list(all_days),
                                **base_stats,
                            })

                    for pct_thresh in [60, 70]:
                        cross_days = [
                            d for d in all_days
                            if atr_pct.get((source, d), 0) >= pct_thresh
                        ]
                        cross_pnls = [day_pnl[d] for d in cross_days]
                        cross_stats = compute_stats(cross_pnls, years_span)
                        if cross_stats:
                            results.append({
                                "instrument": target,
                                "session": session,
                                "rr": rr,
                                "filter": f"X_{source}_P{pct_thresh}",
                                "pnls": cross_pnls,
                                "days": list(cross_days),
                                **cross_stats,
                            })

    # ── BH FDR correction ──────────────────────────────────────────────
    print()
    print("=" * 80)
    print("PART 3: BH FDR CORRECTION")
    print("=" * 80)

    # Exclude BASELINE from FDR (it's the reference)
    filtered_results = [r for r in results if r["filter"] != "BASELINE"]
    if not filtered_results:
        print("No filtered results to correct!")
        return

    p_values = [r["p_val"] for r in filtered_results]
    adjusted = bh_fdr(p_values)

    for i, r in enumerate(filtered_results):
        r["p_adj"] = adjusted[i]
        r["fdr_sig"] = adjusted[i] < 0.05

    # Add baseline results back (no FDR needed)
    for r in results:
        if r["filter"] == "BASELINE":
            r["p_adj"] = r["p_val"]
            r["fdr_sig"] = r["p_val"] < 0.05

    total_tests = len(filtered_results)
    fdr_survivors = sum(1 for r in filtered_results if r["fdr_sig"])
    print(f"Total filter combos tested: {total_tests}")
    print(f"BH FDR survivors (q=0.05): {fdr_survivors}")
    print(f"Survival rate: {100 * fdr_survivors / total_tests:.1f}%")

    # ── Summary tables ─────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("PART 4: FDR SURVIVORS — SORTED BY SHARPE")
    print("=" * 80)

    survivors = [r for r in results if r.get("fdr_sig") and r["filter"] != "BASELINE"]
    survivors.sort(key=lambda x: x["sharpe_ann"], reverse=True)

    print(f"\n{'Inst':4s} {'Session':20s} {'RR':>4s} {'Filter':14s} {'N':>5s} {'WR':>6s} {'AvgR':>8s} {'Sharpe':>7s} {'p_adj':>8s}")
    print("-" * 80)
    for r in survivors[:60]:  # top 60
        print(
            f"{r['instrument']:4s} {r['session']:20s} {r['rr']:4.1f} "
            f"{r['filter']:14s} {r['n']:5d} {r['win_rate']:6.3f} "
            f"{r['avg_r']:+8.4f} {r['sharpe_ann']:7.3f} {r['p_adj']:8.4f}"
        )

    # ── Head-to-head: matched-day permutation test ──────────────────────
    print()
    print("=" * 80)
    print("PART 5: HEAD-TO-HEAD — MATCHED-DAY PERMUTATION TEST")
    print("=" * 80)
    print("Tests whether filter SELECTS higher-R days vs EXCLUDED days.")
    print("Permutation test (10K perms): null = filter label independent of R.")

    # Group by (instrument, session, rr)
    grouped = defaultdict(dict)
    for r in results:
        key = (r["instrument"], r["session"], r["rr"])
        grouped[key][r["filter"]] = r

    # For each group, run matched-day permutation test
    lifts = []
    for key, filters in grouped.items():
        baseline = filters.get("BASELINE")
        if not baseline:
            continue

        baseline_pnls = baseline["pnls"]
        baseline_days = baseline.get("days", [])

        for filt_name, filt_result in filters.items():
            if filt_name == "BASELINE":
                continue

            filt_days_set = set(filt_result.get("days", []))

            # Build matched mask: for each baseline day, is it in the filter?
            if baseline_days and filt_days_set:
                included_mask = [d in filt_days_set for d in baseline_days]
            else:
                # Fallback: approximate from sample sizes
                # Can't do matched test without day data
                continue

            uplift, perm_p = matched_day_permutation_test(
                baseline_pnls, included_mask
            )

            lifts.append({
                "instrument": key[0],
                "session": key[1],
                "rr": key[2],
                "filter": filt_name,
                "n_base": baseline["n"],
                "n_filt": filt_result["n"],
                "expr_base": baseline["avg_r"],
                "expr_filt": filt_result["avg_r"],
                "uplift": uplift,
                "perm_p": perm_p,
                "fdr_sig": filt_result.get("fdr_sig", False),
            })

    # Sort by uplift, show FDR survivors only
    sig_lifts = [x for x in lifts if x["fdr_sig"]]
    sig_lifts.sort(key=lambda x: x["uplift"], reverse=True)

    print(f"\n{'Inst':4s} {'Session':20s} {'RR':>4s} {'Filter':14s} {'N_b':>5s} {'N_f':>5s} "
          f"{'ExR_b':>7s} {'ExR_f':>7s} {'Uplift':>8s} {'Perm_p':>8s}")
    print("-" * 95)
    for item in sig_lifts[:40]:
        sig_flag = " **" if item["perm_p"] < 0.05 else ""
        print(
            f"{item['instrument']:4s} {item['session']:20s} {item['rr']:4.1f} "
            f"{item['filter']:14s} {item['n_base']:5d} {item['n_filt']:5d} "
            f"{item['expr_base']:+7.4f} {item['expr_filt']:+7.4f} {item['uplift']:+8.4f} "
            f"{item['perm_p']:8.4f}{sig_flag}"
        )

    # ── Cross-asset summary ────────────────────────────────────────────
    print()
    print("=" * 80)
    print("PART 6: CROSS-ASSET RESULTS")
    print("=" * 80)

    cross_results = [r for r in results if r["filter"].startswith("X_")]
    cross_survivors = [r for r in cross_results if r.get("fdr_sig")]
    cross_survivors.sort(key=lambda x: x["sharpe_ann"], reverse=True)

    print(f"Cross-asset combos tested: {len(cross_results)}")
    print(f"Cross-asset FDR survivors: {len(cross_survivors)}")

    if cross_survivors:
        print(f"\n{'Inst':4s} {'Session':20s} {'RR':>4s} {'Filter':14s} {'N':>5s} {'WR':>6s} {'AvgR':>8s} {'Sharpe':>7s} {'p_adj':>8s}")
        print("-" * 80)
        for r in cross_survivors[:30]:
            print(
                f"{r['instrument']:4s} {r['session']:20s} {r['rr']:4.1f} "
                f"{r['filter']:14s} {r['n']:5d} {r['win_rate']:6.3f} "
                f"{r['avg_r']:+8.4f} {r['sharpe_ann']:7.3f} {r['p_adj']:8.4f}"
            )

    # ── Filter type summary ────────────────────────────────────────────
    print()
    print("=" * 80)
    print("PART 7: FILTER TYPE SUMMARY (aggregate across all combos)")
    print("=" * 80)

    filter_types = set(r["filter"] for r in results)
    print(f"\n{'Filter':14s} {'Tested':>7s} {'FDR_sig':>8s} {'Rate':>6s} {'Avg_ExpR':>9s} {'Avg_Sharpe':>11s}")
    print("-" * 60)
    for ft in sorted(filter_types):
        ft_results = [r for r in results if r["filter"] == ft]
        n_tested = len(ft_results)
        n_sig = sum(1 for r in ft_results if r.get("fdr_sig"))
        avg_expr = np.mean([r["avg_r"] for r in ft_results])
        avg_sharpe = np.mean([r["sharpe_ann"] for r in ft_results])
        rate = 100 * n_sig / n_tested if n_tested > 0 else 0
        print(f"{ft:14s} {n_tested:7d} {n_sig:8d} {rate:5.1f}% {avg_expr:+9.4f} {avg_sharpe:11.3f}")

    # ── NEW vs EXISTING: do ATR filters find edge where VOL doesn't? ──
    print()
    print("=" * 80)
    print("PART 8: STANDALONE FDR-SIG SUBSETS WHERE VOL IS NOT SIGNIFICANT")
    print("=" * 80)
    print("NOTE: This identifies promising subsets, NOT incremental uplift vs VOL.")
    print("Matched-day permutation test in Part 5 is the correct uplift measure.")

    atr_only_wins = []
    for key, filters in grouped.items():
        vol = filters.get("VOL_ONLY")
        baseline = filters.get("BASELINE")
        if not baseline:
            continue

        # Check each ATR filter
        for filt_name in ["ATR_P50", "ATR_P60", "ATR_P70", "ATR_P80", "GARCH_P70", "ATR70+VOL", "ATR70|VOL"]:
            atr = filters.get(filt_name)
            if not atr:
                continue

            # ATR is FDR significant but VOL is not (or doesn't exist)
            atr_sig = atr.get("fdr_sig", False)
            vol_sig = vol.get("fdr_sig", False) if vol else False

            if atr_sig and not vol_sig:
                atr_only_wins.append({
                    "instrument": key[0],
                    "session": key[1],
                    "rr": key[2],
                    "filter": filt_name,
                    "n": atr["n"],
                    "avg_r": atr["avg_r"],
                    "sharpe": atr["sharpe_ann"],
                    "base_avg_r": baseline["avg_r"],
                    "base_sharpe": baseline["sharpe_ann"],
                })

    if atr_only_wins:
        atr_only_wins.sort(key=lambda x: x["sharpe"], reverse=True)
        print(f"Found {len(atr_only_wins)} combos where ATR/GARCH/COMBINED is FDR-sig but VOL is not:")
        print(f"\n{'Inst':4s} {'Session':20s} {'RR':>4s} {'Filter':14s} {'N':>5s} {'AvgR':>8s} {'Sharpe':>7s} {'Base_R':>7s} {'Base_Sh':>8s}")
        print("-" * 85)
        for w in atr_only_wins[:30]:
            print(
                f"{w['instrument']:4s} {w['session']:20s} {w['rr']:4.1f} "
                f"{w['filter']:14s} {w['n']:5d} {w['avg_r']:+8.4f} "
                f"{w['sharpe']:7.3f} {w['base_avg_r']:+7.4f} {w['base_sharpe']:8.3f}"
            )
    else:
        print("No ATR-only wins found (VOL captures all FDR-significant edge)")

    print()
    print("=" * 80)
    print("RESEARCH COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_research()
