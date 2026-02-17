#!/usr/bin/env python3
"""
Cross-Instrument Lead-Lag Research (MGC x MES x MNQ).

Read-only analysis: queries daily_features to test whether watching
multiple instruments gives predictive signal for ORB breakouts.

Five analysis blocks:
  1. Session-Sequential Lead-Lag (conditional probability of break direction)
  2. Concordance Filter (all 3 break same direction vs split)
  3. Gold Leads Equities (MGC 2300 -> MES/MNQ 0030 deep dive)
  4. Divergence Signal (instruments disagree at shared sessions)
  5. Volatility Regime Sync (ATR-20 quartile buckets)

Honest framing: MES and MNQ are both US equity indices --heavily correlated
by default. The interesting question is whether gold (MGC) provides independent
signal, and whether cross-instrument concordance/divergence is a useful filter.

Usage:
    python scripts/research_cross_instrument.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH

# =========================================================================
# Configuration
# =========================================================================

INSTRUMENTS = ["MGC", "MES", "MNQ"]

# Sessions per instrument (from asset_configs.py, fixed ORB labels only)
INSTRUMENT_SESSIONS = {
    "MGC": ["0900", "1000", "1100", "1800", "2300"],
    "MES": ["0900", "1000", "1100", "1800", "0030"],
    "MNQ": ["0900", "1000", "1100", "1800", "0030"],
}

SHARED_SESSIONS = ["0900", "1000", "1100", "1800"]
ALL_SESSIONS = ["0900", "1000", "1100", "1800", "2300", "0030"]

MIN_SAMPLE = 30
P_THRESHOLD = 0.05

# =========================================================================
# Data Loading
# =========================================================================

def load_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load daily_features for all 3 instruments, orb_minutes=5 only."""
    query = """
    SELECT
        trading_day, symbol, atr_20,
        orb_0900_break_dir,  orb_0900_size,  orb_0900_outcome,  orb_0900_double_break,
        orb_1000_break_dir,  orb_1000_size,  orb_1000_outcome,  orb_1000_double_break,
        orb_1100_break_dir,  orb_1100_size,  orb_1100_outcome,  orb_1100_double_break,
        orb_1800_break_dir,  orb_1800_size,  orb_1800_outcome,  orb_1800_double_break,
        orb_2300_break_dir,  orb_2300_size,  orb_2300_outcome,  orb_2300_double_break,
        orb_0030_break_dir,  orb_0030_size,  orb_0030_outcome,  orb_0030_double_break
    FROM daily_features
    WHERE symbol IN ('MGC', 'MES', 'MNQ')
      AND orb_minutes = 5
    ORDER BY trading_day, symbol
    """
    return con.execute(query).fetchdf()

def build_wide_df(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot so each trading_day has one row with columns per instrument.

    Uses vectorized merge instead of row-by-row iteration.
    """
    sessions = ALL_SESSIONS
    suffixes = ["break_dir", "size", "outcome", "double_break"]

    # Rename orb_ columns to plain session columns for each instrument slice
    dfs = {}
    for inst in INSTRUMENTS:
        sub = df[df["symbol"] == inst].copy()
        rename_map = {"atr_20": f"{inst}_atr_20"}
        for sess in sessions:
            for suf in suffixes:
                rename_map[f"orb_{sess}_{suf}"] = f"{inst}_{sess}_{suf}"
        sub = sub.rename(columns=rename_map)
        keep_cols = ["trading_day"] + list(rename_map.values())
        dfs[inst] = sub[[c for c in keep_cols if c in sub.columns]]

    # Merge all three on trading_day
    wide = dfs["MGC"]
    for inst in ["MES", "MNQ"]:
        wide = wide.merge(dfs[inst], on="trading_day", how="inner")

    return wide

# =========================================================================
# Statistical helpers
# =========================================================================

def chi2_or_fisher(a: int, b: int, c: int, d: int):
    """2x2 contingency test. Returns (statistic, p_value, test_name)."""
    table = np.array([[a, b], [c, d]])
    if table.min() < 5:
        _, p = stats.fisher_exact(table)
        return None, p, "Fisher"
    chi2, p, _, _ = stats.chi2_contingency(table, correction=True)
    return chi2, p, "Chi2"

def sig_flag(n: int, p: float | None) -> str:
    """Return flag string for result quality."""
    flags = []
    if n < MIN_SAMPLE:
        flags.append(f"INSUFFICIENT (N={n})")
    if p is not None and p > P_THRESHOLD:
        flags.append(f"NS (p={p:.3f})")
    return ", ".join(flags) if flags else "SIG"

# =========================================================================
# Analysis 1: Session-Sequential Lead-Lag
# =========================================================================

def analysis_1_lead_lag(wide: pd.DataFrame) -> list[dict]:
    """For each session pair where A finishes before B, compute lift."""
    session_pairs = [
        ("0900", "1000"), ("0900", "1100"),
        ("1000", "1100"), ("1100", "1800"),
        ("1800", "2300"), ("1800", "0030"),
        ("2300", "0030"),
    ]

    results = []
    for l_sess, f_sess in session_pairs:
        for l_inst in INSTRUMENTS:
            if l_sess not in INSTRUMENT_SESSIONS.get(l_inst, []):
                continue
            for f_inst in INSTRUMENTS:
                if f_sess not in INSTRUMENT_SESSIONS.get(f_inst, []):
                    continue

                l_col = f"{l_inst}_{l_sess}_break_dir"
                f_col = f"{f_inst}_{f_sess}_break_dir"
                if l_col not in wide.columns or f_col not in wide.columns:
                    continue

                for direction in ["long", "short"]:
                    l_is_dir = (wide[l_col] == direction)
                    f_is_dir = (wide[f_col] == direction)
                    f_has_break = wide[f_col].notna()

                    # Baseline P(follower breaks direction)
                    baseline_n = int(f_has_break.sum())
                    if baseline_n == 0:
                        continue
                    baseline_rate = f_is_dir.sum() / baseline_n

                    # Conditional P(follower dir | leader dir)
                    cond_n = int(l_is_dir.sum())
                    if cond_n == 0 or baseline_rate == 0:
                        continue
                    cond_rate = (l_is_dir & f_is_dir).sum() / cond_n
                    lift = cond_rate / baseline_rate

                    # Contingency table
                    a = int((l_is_dir & f_is_dir).sum())
                    b = int((l_is_dir & ~f_is_dir).sum())
                    c = int((~l_is_dir & f_is_dir).sum())
                    d = int((~l_is_dir & ~f_is_dir).sum())
                    _, p_val, test_name = chi2_or_fisher(a, b, c, d)

                    results.append({
                        "leader": f"{l_inst} {l_sess}",
                        "follower": f"{f_inst} {f_sess}",
                        "direction": direction,
                        "baseline": baseline_rate,
                        "conditional": cond_rate,
                        "lift": lift,
                        "N_cond": cond_n,
                        "p_value": p_val,
                        "test": test_name,
                        "flag": sig_flag(cond_n, p_val),
                    })

    return results

def print_analysis_1(results: list[dict]):
    print("\n--- ANALYSIS 1: SESSION-SEQUENTIAL LEAD-LAG ---")
    print("Does instrument A breaking at session X predict instrument B"
          " breaking same direction at later session Y?")
    print()

    if not results:
        print("  No valid pairs found.")
        return

    results.sort(key=lambda r: abs(r["lift"] - 1.0), reverse=True)

    header = (f"{'Leader':<14} {'Follower':<14} {'Dir':<6} "
              f"{'Base':>6} {'Cond':>6} {'Lift':>6} {'N':>5} "
              f"{'p':>7} {'Flag':<20}")
    print(header)
    print("-" * len(header))

    for r in results[:40]:
        print(f"{r['leader']:<14} {r['follower']:<14} {r['direction']:<6} "
              f"{r['baseline']:>6.1%} {r['conditional']:>6.1%} {r['lift']:>6.2f} "
              f"{r['N_cond']:>5} {r['p_value']:>7.4f} {r['flag']:<20}")

# =========================================================================
# Analysis 2: Concordance Filter
# =========================================================================

def analysis_2_concordance(wide: pd.DataFrame) -> dict:
    """At shared sessions, classify concordance and compute win rates."""
    results = {}

    for sess in SHARED_SESSIONS:
        dir_cols = [f"{inst}_{sess}_break_dir" for inst in INSTRUMENTS]
        out_cols = [f"{inst}_{sess}_outcome" for inst in INSTRUMENTS]
        if not all(c in wide.columns for c in dir_cols + out_cols):
            continue

        # Vectorized: count longs/shorts per day
        is_long = pd.DataFrame({inst: wide[f"{inst}_{sess}_break_dir"] == "long"
                                for inst in INSTRUMENTS})
        is_short = pd.DataFrame({inst: wide[f"{inst}_{sess}_break_dir"] == "short"
                                 for inst in INSTRUMENTS})
        has_break = is_long | is_short

        n_long = is_long.sum(axis=1)
        n_short = is_short.sum(axis=1)
        n_active = has_break.sum(axis=1)

        # Classify days
        conc3 = (n_active == 3) & ((n_long == 3) | (n_short == 3))
        maj2 = ~conc3 & (n_active >= 2) & ((n_long >= 2) | (n_short >= 2))
        split = ~conc3 & ~maj2 & (n_active >= 2)

        # Majority direction
        maj_dir = np.where(n_long >= n_short, "long", "short")

        sess_data = {}
        for cat_name, cat_mask in [("concordant_3", conc3), ("majority_2", maj2), ("split", split)]:
            sub = wide[cat_mask]
            if len(sub) == 0:
                sess_data[cat_name] = {"n_days": 0}
                continue

            # Collect per-instrument outcomes
            maj_wins, maj_total, min_wins, min_total = 0, 0, 0, 0
            all_wins, all_total = 0, 0

            for inst in INSTRUMENTS:
                inst_dir = sub[f"{inst}_{sess}_break_dir"]
                inst_out = sub[f"{inst}_{sess}_outcome"]
                valid = inst_out.isin(["win", "loss"])
                is_win = inst_out == "win"

                if cat_name != "split":
                    inst_maj_dir = pd.Series(maj_dir[cat_mask], index=sub.index)
                    on_majority = (inst_dir == inst_maj_dir) & valid
                    on_minority = (inst_dir != inst_maj_dir) & inst_dir.isin(["long", "short"]) & valid

                    maj_wins += int((is_win & on_majority).sum())
                    maj_total += int(on_majority.sum())
                    min_wins += int((is_win & on_minority).sum())
                    min_total += int(on_minority.sum())

                all_wins += int((is_win & valid).sum())
                all_total += int(valid.sum())

            sess_data[cat_name] = {
                "n_days": len(sub),
                "maj_wr": maj_wins / maj_total if maj_total > 0 else None,
                "maj_n": maj_total,
                "min_wr": min_wins / min_total if min_total > 0 else None,
                "min_n": min_total,
                "all_wr": all_wins / all_total if all_total > 0 else None,
                "all_n": all_total,
            }

        # Baseline
        bl_wins, bl_total = 0, 0
        for inst in INSTRUMENTS:
            oc = wide[f"{inst}_{sess}_outcome"]
            v = oc.isin(["win", "loss"])
            bl_wins += int((oc == "win").sum())
            bl_total += int(v.sum())
        sess_data["baseline"] = {
            "wr": bl_wins / bl_total if bl_total > 0 else None,
            "n": bl_total,
        }

        results[sess] = sess_data

    return results

def print_analysis_2(results: dict):
    print("\n--- ANALYSIS 2: CONCORDANCE FILTER ---")
    print("When all 3 instruments break the same direction, are outcomes better?")
    print()

    for sess in SHARED_SESSIONS:
        if sess not in results:
            continue
        data = results[sess]
        print(f"  Session {sess}:")

        for cat, label in [("concordant_3", "3/3 Concordant"),
                           ("majority_2", "2/1 Majority"),
                           ("split", "Split/Mixed")]:
            d = data[cat]
            if d["n_days"] == 0:
                print(f"    {label}: N=0")
                continue

            if cat != "split":
                maj_str = f"{d['maj_wr']:.1%}" if d["maj_wr"] is not None else "N/A"
                min_str = f"{d['min_wr']:.1%}" if d["min_wr"] is not None else "N/A"
                flag = sig_flag(d["maj_n"], None)
                print(f"    {label}: {d['n_days']:>3} days "
                      f"| Majority WR={maj_str} (N={d['maj_n']}) "
                      f"| Minority WR={min_str} (N={d['min_n']}) "
                      f"| {flag}")
            else:
                wr_str = f"{d['all_wr']:.1%}" if d["all_wr"] is not None else "N/A"
                print(f"    {label}: {d['n_days']:>3} days | WR={wr_str} (N={d['all_n']})")

        bl = data.get("baseline", {})
        if bl.get("wr") is not None:
            print(f"    Baseline (unconditional): WR={bl['wr']:.1%} (N={bl['n']})")
        print()

# =========================================================================
# Analysis 3: Gold Leads Equities (MGC 2300 -> MES/MNQ 0030)
# =========================================================================

def analysis_3_gold_leads(wide: pd.DataFrame) -> dict:
    """Deep dive: MGC 2300 break -> MES/MNQ 0030 outcomes."""
    results = {}

    mgc_dir = "MGC_2300_break_dir"
    mgc_size = "MGC_2300_size"
    mgc_db = "MGC_2300_double_break"
    mgc_atr = "MGC_atr_20"

    if mgc_dir not in wide.columns:
        return results

    for follower in ["MES", "MNQ"]:
        f_dir = f"{follower}_0030_break_dir"
        f_out = f"{follower}_0030_outcome"
        if f_dir not in wide.columns:
            continue

        fr = {}

        for d in ["long", "short"]:
            opp = "short" if d == "long" else "long"
            mgc_is_d = wide[mgc_dir] == d
            f_same = wide[f_dir] == d
            f_opp = wide[f_dir] == opp

            # Follow rate
            n_mgc = int(mgc_is_d.sum())
            follow_rate = (mgc_is_d & f_same).sum() / n_mgc if n_mgc > 0 else None

            # Aligned win rate
            aligned = mgc_is_d & f_same
            a_out = wide.loc[aligned, f_out]
            a_valid = a_out.isin(["win", "loss"])
            a_n = int(a_valid.sum())
            a_wr = (a_out == "win").sum() / a_n if a_n > 0 else None

            # Contrarian win rate
            opposed = mgc_is_d & f_opp
            c_out = wide.loc[opposed, f_out]
            c_valid = c_out.isin(["win", "loss"])
            c_n = int(c_valid.sum())
            c_wr = (c_out == "win").sum() / c_n if c_n > 0 else None

            fr[f"mgc_{d}"] = {
                "follow_rate": follow_rate, "follow_n": n_mgc,
                "aligned_wr": a_wr, "aligned_n": a_n,
                "contrarian_wr": c_wr, "contrarian_n": c_n,
            }

        # Double-break filter
        mgc_double = wide[mgc_db] == True  # noqa: E712
        db_out = wide.loc[mgc_double, f_out]
        db_v = db_out.isin(["win", "loss"])
        db_n = int(db_v.sum())
        fr["after_double_break"] = {
            "wr": (db_out == "win").sum() / db_n if db_n > 0 else None,
            "n": db_n,
        }

        # ORB size conditional (large vs small relative to ATR)
        has_data = wide[mgc_dir].notna() & wide[mgc_size].notna() & wide[mgc_atr].notna()
        sub = wide[has_data].copy()
        if len(sub) > 0:
            ratio = sub[mgc_size] / sub[mgc_atr]
            med = float(ratio.median())
            for label, mask in [("large_orb", ratio >= med), ("small_orb", ratio < med)]:
                s_out = sub.loc[mask, f_out]
                s_v = s_out.isin(["win", "loss"])
                s_n = int(s_v.sum())
                fr[label] = {
                    "wr": (s_out == "win").sum() / s_n if s_n > 0 else None,
                    "n": s_n, "median_ratio": med,
                }

        results[follower] = fr

    return results

def print_analysis_3(results: dict, wide: pd.DataFrame):
    print("\n--- ANALYSIS 3: GOLD LEADS EQUITIES (MGC 2300 -> MES/MNQ 0030) ---")
    print("Key asymmetry: MGC 2300 resolves ~90 min before MES/MNQ 0030")
    print()

    if not results:
        print("  No data (MGC 2300 or MES/MNQ 0030 columns missing).")
        return

    # Baseline 0030 win rates
    for f in ["MES", "MNQ"]:
        oc = f"{f}_0030_outcome"
        if oc in wide.columns:
            vals = wide[oc]
            v = vals.isin(["win", "loss"])
            wr = (vals == "win").sum() / v.sum() if v.sum() > 0 else 0
            print(f"  {f} 0030 baseline WR: {wr:.1%} (N={v.sum()})")
    print()

    for follower, data in results.items():
        print(f"  {follower} 0030 after MGC 2300:")
        for d in ["long", "short"]:
            key = f"mgc_{d}"
            if key not in data:
                continue
            r = data[key]
            fr = f"{r['follow_rate']:.1%}" if r["follow_rate"] is not None else "N/A"
            awr = f"{r['aligned_wr']:.1%}" if r["aligned_wr"] is not None else "N/A"
            cwr = f"{r['contrarian_wr']:.1%}" if r["contrarian_wr"] is not None else "N/A"
            print(f"    MGC {d}: follow_rate={fr} (N={r['follow_n']}) "
                  f"| aligned WR={awr} (N={r['aligned_n']}, {sig_flag(r['aligned_n'], None)}) "
                  f"| contrarian WR={cwr} (N={r['contrarian_n']}, {sig_flag(r['contrarian_n'], None)})")

        db = data.get("after_double_break", {})
        db_wr = f"{db['wr']:.1%}" if db.get("wr") is not None else "N/A"
        print(f"    After MGC 2300 double-break (choppy): "
              f"WR={db_wr} (N={db.get('n', 0)}, {sig_flag(db.get('n', 0), None)})")

        for sl in ["large_orb", "small_orb"]:
            s = data.get(sl, {})
            s_wr = f"{s['wr']:.1%}" if s.get("wr") is not None else "N/A"
            med = s.get("median_ratio")
            med_str = f" (split at {med:.3f} ATR)" if med is not None else ""
            print(f"    MGC 2300 {sl}{med_str}: "
                  f"{follower} 0030 WR={s_wr} (N={s.get('n', 0)}, {sig_flag(s.get('n', 0), None)})")
        print()

# =========================================================================
# Analysis 4: Divergence Signal
# =========================================================================

def analysis_4_divergence(wide: pd.DataFrame) -> dict:
    """When instruments disagree at shared sessions, what happens next?"""
    results = {}

    for sess in SHARED_SESSIONS:
        dir_cols = {i: f"{i}_{sess}_break_dir" for i in INSTRUMENTS}
        if not all(c in wide.columns for c in dir_cols.values()):
            continue

        # Next session for each instrument
        next_sess = {}
        for inst in INSTRUMENTS:
            sl = INSTRUMENT_SESSIONS[inst]
            if sess in sl:
                idx = sl.index(sess)
                if idx + 1 < len(sl):
                    next_sess[inst] = sl[idx + 1]

        mgc_d = wide[dir_cols["MGC"]]
        mes_d = wide[dir_cols["MES"]]
        mnq_d = wide[dir_cols["MNQ"]]

        mgc_active = mgc_d.isin(["long", "short"])
        mes_active = mes_d.isin(["long", "short"])
        mnq_active = mnq_d.isin(["long", "short"])
        all_active = mgc_active & mes_active & mnq_active

        # Gold vs equities: MGC opposite to MES+MNQ (who agree)
        gold_div = all_active & (mgc_d != mes_d) & (mes_d == mnq_d)

        # Equities disagree
        eq_div = mes_active & mnq_active & (mes_d != mnq_d)

        # Concordant (all 3 same direction)
        concordant = all_active & (mgc_d == mes_d) & (mes_d == mnq_d)

        sess_results = {}
        for div_name, div_mask, desc in [
            ("gold_vs_equities", gold_div, "MGC opposite to MES+MNQ"),
            ("equities_disagree", eq_div, "MES and MNQ opposite"),
        ]:
            n_div = int(div_mask.sum())
            n_conc = int(concordant.sum())

            next_outcomes = {}
            for inst, ns in next_sess.items():
                oc = f"{inst}_{ns}_outcome"
                if oc not in wide.columns:
                    continue

                div_out = wide.loc[div_mask, oc]
                dv = div_out.isin(["win", "loss"])
                d_n = int(dv.sum())
                d_wr = (div_out == "win").sum() / d_n if d_n > 0 else None

                conc_out = wide.loc[concordant, oc]
                cv = conc_out.isin(["win", "loss"])
                c_n = int(cv.sum())
                c_wr = (conc_out == "win").sum() / c_n if c_n > 0 else None

                next_outcomes[f"{inst}_{ns}"] = {
                    "div_wr": d_wr, "div_n": d_n,
                    "conc_wr": c_wr, "conc_n": c_n,
                }

            sess_results[div_name] = {
                "desc": desc,
                "n_divergent_days": n_div,
                "n_concordant_days": n_conc,
                "next_session_outcomes": next_outcomes,
            }

        results[sess] = sess_results

    return results

def print_analysis_4(results: dict):
    print("\n--- ANALYSIS 4: DIVERGENCE SIGNAL ---")
    print("When instruments disagree at a shared session, what happens next?")
    print()

    for sess, sess_data in results.items():
        print(f"  Session {sess}:")
        for div_data in sess_data.values():
            n_div = div_data["n_divergent_days"]
            n_conc = div_data["n_concordant_days"]
            print(f"    {div_data['desc']}: {n_div} divergent days, {n_conc} concordant days")

            for target, od in div_data["next_session_outcomes"].items():
                d_wr = f"{od['div_wr']:.1%}" if od["div_wr"] is not None else "N/A"
                c_wr = f"{od['conc_wr']:.1%}" if od["conc_wr"] is not None else "N/A"
                print(f"      -> {target}: "
                      f"Divergent WR={d_wr} (N={od['div_n']}, {sig_flag(od['div_n'], None)}) "
                      f"| Concordant WR={c_wr} (N={od['conc_n']}, {sig_flag(od['conc_n'], None)})")
        print()

# =========================================================================
# Analysis 5: Volatility Regime Sync
# =========================================================================

def analysis_5_vol_regime(wide: pd.DataFrame) -> dict:
    """Bucket days by ATR-20 percentile rank, compute session win rates."""
    results = {}

    # Compute ATR-20 quartile per instrument (vectorized)
    for inst in INSTRUMENTS:
        col = f"{inst}_atr_20"
        if col not in wide.columns:
            continue
        q25 = wide[col].quantile(0.25)
        q75 = wide[col].quantile(0.75)
        wide[f"{inst}_vol_q"] = np.where(
            wide[col] >= q75, "high",
            np.where(wide[col] <= q25, "low", "mid"),
        )

    vol_qs = [f"{i}_vol_q" for i in INSTRUMENTS]
    if not all(c in wide.columns for c in vol_qs):
        return results

    # Vectorized regime classification
    all_high = ((wide[vol_qs[0]] == "high") &
                (wide[vol_qs[1]] == "high") &
                (wide[vol_qs[2]] == "high"))
    all_low = ((wide[vol_qs[0]] == "low") &
               (wide[vol_qs[1]] == "low") &
               (wide[vol_qs[2]] == "low"))
    has_all = (wide[vol_qs[0]].notna() &
               wide[vol_qs[1]].notna() &
               wide[vol_qs[2]].notna())
    mixed = has_all & ~all_high & ~all_low

    for regime_name, mask in [("all_high", all_high), ("all_low", all_low), ("mixed", mixed)]:
        sub = wide[mask]
        n_days = len(sub)

        session_wrs = {}
        for sess in SHARED_SESSIONS:
            for inst in INSTRUMENTS:
                oc = f"{inst}_{sess}_outcome"
                if oc not in wide.columns:
                    continue

                vals = sub[oc]
                v = vals.isin(["win", "loss"])
                n = int(v.sum())
                wr = (vals == "win").sum() / n if n > 0 else None

                # Baseline
                all_vals = wide[oc]
                all_v = all_vals.isin(["win", "loss"])
                all_wr = (all_vals == "win").sum() / all_v.sum() if all_v.sum() > 0 else None

                session_wrs[f"{inst}_{sess}"] = {"wr": wr, "n": n, "baseline_wr": all_wr}

        results[regime_name] = {"n_days": n_days, "session_wrs": session_wrs}

    return results

def print_analysis_5(results: dict):
    print("\n--- ANALYSIS 5: VOLATILITY REGIME SYNC ---")
    print("Does synchronized high/low volatility across asset classes change ORB edge?")
    print()

    labels = {
        "all_high": "All 3 High-Vol (top 25% ATR-20)",
        "all_low": "All 3 Low-Vol (bottom 25% ATR-20)",
        "mixed": "Mixed Vol Regimes",
    }

    for rn in ["all_high", "all_low", "mixed"]:
        if rn not in results:
            continue
        data = results[rn]
        print(f"  {labels[rn]} ({data['n_days']} days):")

        for key, wd in sorted(data["session_wrs"].items()):
            wr = f"{wd['wr']:.1%}" if wd["wr"] is not None else "  N/A"
            bl = f"{wd['baseline_wr']:.1%}" if wd["baseline_wr"] is not None else "N/A"
            flag = sig_flag(wd["n"], None)
            lift = ""
            if wd["wr"] is not None and wd["baseline_wr"] and wd["baseline_wr"] > 0:
                lift = f"lift={wd['wr'] / wd['baseline_wr']:.2f}"
            print(f"    {key:<12}: WR={wr:>5} (N={wd['n']:>3}, {flag:<20}) "
                  f"baseline={bl} {lift}")
        print()

# =========================================================================
# Honest Summary
# =========================================================================

def compute_mes_mnq_concordance(wide: pd.DataFrame) -> dict:
    """Compute baseline MES-MNQ concordance at shared sessions."""
    concordance = {}
    for sess in SHARED_SESSIONS:
        mes = wide[f"MES_{sess}_break_dir"]
        mnq = wide[f"MNQ_{sess}_break_dir"]
        both = mes.notna() & mnq.notna()
        sub_mes, sub_mnq = mes[both], mnq[both]
        if len(sub_mes) == 0:
            continue
        same = int((sub_mes == sub_mnq).sum())
        concordance[sess] = {"same": same, "total": len(sub_mes), "rate": same / len(sub_mes)}
    return concordance

def print_honest_summary(lead_lag: list[dict], gold_leads: dict,
                         mes_mnq_corr: dict, n_days: int):
    print("\n" + "=" * 60)
    print("HONEST SUMMARY")
    print("=" * 60)

    print("\nBASELINE EXPECTATION:")
    for sess, d in mes_mnq_corr.items():
        print(f"  MES/MNQ concordance at {sess}: "
              f"{d['rate']:.0%} ({d['same']}/{d['total']})")
    print("  (They're the same asset class -- high concordance expected, not a signal)")

    survived = []
    not_survived = []

    # Only highlight lead-lag with meaningful lift (>1.05 or <0.95)
    # With N=500+, even trivial lifts pass p<0.05
    LIFT_THRESHOLD = 0.05
    for r in lead_lag:
        desc = f"{r['leader']} {r['direction']} -> {r['follower']}: lift={r['lift']:.2f}"
        meaningful = abs(r["lift"] - 1.0) >= LIFT_THRESHOLD
        if r["N_cond"] >= MIN_SAMPLE and r["p_value"] < P_THRESHOLD and meaningful:
            survived.append(desc)
        elif r["N_cond"] < MIN_SAMPLE:
            not_survived.append(f"{desc} [N={r['N_cond']}, insufficient]")
        elif r["p_value"] >= P_THRESHOLD:
            not_survived.append(f"{desc} [p={r['p_value']:.3f}, NS]")
        elif not meaningful:
            not_survived.append(f"{desc} [lift too small]")

    for follower, data in gold_leads.items():
        for d in ["long", "short"]:
            key = f"mgc_{d}"
            if key not in data:
                continue
            r = data[key]
            if r["aligned_wr"] is not None and r["aligned_n"] >= MIN_SAMPLE:
                survived.append(f"MGC 2300 {d} -> {follower} 0030 aligned: "
                                f"WR={r['aligned_wr']:.1%} (N={r['aligned_n']})")
            elif r["aligned_n"] > 0:
                not_survived.append(f"MGC 2300 {d} -> {follower} 0030: "
                                    f"N={r['aligned_n']}, insufficient")

    print(f"\nSURVIVED SCRUTINY (N>={MIN_SAMPLE}, p<{P_THRESHOLD}):")
    if survived:
        for s in survived:
            print(f"  + {s}")
    else:
        print("  (none)")

    print("\nDID NOT SURVIVE:")
    if not_survived:
        for s in not_survived[:20]:
            print(f"  - {s}")
        if len(not_survived) > 20:
            print(f"  ... and {len(not_survived) - 20} more")
    else:
        print("  (none)")

    print(f"\nCAVEATS:")
    print(f"  1. {n_days} overlapping days is thin --many conditional slices have N < 50")
    print(f"  2. MES ~ MNQ --their concordance is a baseline fact, not a discovery")
    print(f"  3. Multiple comparisons --testing ~50 combinations inflates false positives")
    print(f"  4. Feb 2024 - Feb 2026 includes specific market conditions (may not generalize)")
    print(f"  5. Outcomes use daily_features break_dir/outcome (parameter-free, RR=1.0)")

# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 60)
    print("CROSS-INSTRUMENT LEAD-LAG RESEARCH")
    print("=" * 60)

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    df = load_features(con)
    con.close()

    instruments_found = sorted(df["symbol"].unique())
    print(f"Instruments: {instruments_found}")

    # Overlapping trading days (all 3 present)
    day_counts = df.groupby("trading_day")["symbol"].nunique()
    overlap_days = day_counts[day_counts == 3].index
    print(f"Overlapping trading days (all 3): {len(overlap_days)}")

    if len(overlap_days) == 0:
        print("FATAL: No overlapping days. Check daily_features data.")
        sys.exit(1)

    df_overlap = df[df["trading_day"].isin(overlap_days)]
    print(f"Period: {df_overlap['trading_day'].min()} to {df_overlap['trading_day'].max()}")
    print()

    wide = build_wide_df(df_overlap)

    ll = analysis_1_lead_lag(wide)
    print_analysis_1(ll)

    conc = analysis_2_concordance(wide)
    print_analysis_2(conc)

    gl = analysis_3_gold_leads(wide)
    print_analysis_3(gl, wide)

    div = analysis_4_divergence(wide)
    print_analysis_4(div)

    vol = analysis_5_vol_regime(wide)
    print_analysis_5(vol)

    mes_mnq_corr = compute_mes_mnq_concordance(wide)
    print_honest_summary(ll, gl, mes_mnq_corr, len(overlap_days))

if __name__ == "__main__":
    main()
