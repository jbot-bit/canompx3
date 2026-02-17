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

Honest framing: MES and MNQ are both US equity indices — heavily correlated
by default. The interesting question is whether gold (MGC) provides independent
signal, and whether cross-instrument concordance/divergence is a useful filter.

Usage:
    python scripts/research_cross_instrument.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH

# =========================================================================
# Configuration
# =========================================================================

INSTRUMENTS = ["MGC", "MES", "MNQ"]

# Sessions per instrument (from asset_configs.py enabled_sessions,
# filtered to fixed ORB labels relevant for cross-instrument analysis)
INSTRUMENT_SESSIONS = {
    "MGC": ["0900", "1000", "1100", "1800", "2300"],
    "MES": ["0900", "1000", "1100", "1800", "0030"],
    "MNQ": ["0900", "1000", "1100", "1800", "0030"],
}

# Shared sessions (all 3 instruments have these)
SHARED_SESSIONS = ["0900", "1000", "1100", "1800"]

# Session temporal order within a Brisbane trading day
SESSION_ORDER = {"0900": 0, "1000": 1, "1100": 2, "1800": 3, "2300": 4, "0030": 5}

# Significance thresholds
MIN_SAMPLE = 30
P_THRESHOLD = 0.05

# =========================================================================
# Data Loading
# =========================================================================


def load_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load daily_features for all 3 instruments, orb_minutes=5 only."""
    query = """
    SELECT
        trading_day, symbol,
        atr_20,
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
    """Pivot so each trading_day row has columns for all instruments.

    Output columns: trading_day, MGC_0900_break_dir, MES_0900_break_dir, etc.
    """
    rows = []
    for td, grp in df.groupby("trading_day"):
        row = {"trading_day": td}
        for _, r in grp.iterrows():
            sym = r["symbol"]
            row[f"{sym}_atr_20"] = r["atr_20"]
            for sess in ["0900", "1000", "1100", "1800", "2300", "0030"]:
                row[f"{sym}_{sess}_break_dir"] = r[f"orb_{sess}_break_dir"]
                row[f"{sym}_{sess}_size"] = r[f"orb_{sess}_size"]
                row[f"{sym}_{sess}_outcome"] = r[f"orb_{sess}_outcome"]
                row[f"{sym}_{sess}_double_break"] = r[f"orb_{sess}_double_break"]
        rows.append(row)
    wide = pd.DataFrame(rows)
    return wide


# =========================================================================
# Statistical helpers
# =========================================================================


def conditional_prob(condition_mask: pd.Series, event_mask: pd.Series):
    """Compute P(event | condition) with sample size."""
    joint = condition_mask & event_mask
    n_cond = condition_mask.sum()
    if n_cond == 0:
        return None, 0
    return joint.sum() / n_cond, int(n_cond)


def chi2_or_fisher(a: int, b: int, c: int, d: int):
    """Chi-squared test on 2x2 contingency table; Fisher exact if any cell < 5.

    Table layout:
        [[a, b],   a = condition & event, b = condition & ~event
         [c, d]]   c = ~condition & event, d = ~condition & ~event

    Returns (statistic, p_value, test_name).
    """
    table = np.array([[a, b], [c, d]])
    if table.min() < 5:
        _, p = stats.fisher_exact(table)
        return None, p, "Fisher"
    chi2, p, _, _ = stats.chi2_contingency(table, correction=True)
    return chi2, p, "Chi2"


def significance_flag(n: int, p: float | None) -> str:
    """Return flag string for result quality."""
    flags = []
    if n < MIN_SAMPLE:
        flags.append(f"INSUFFICIENT (N={n})")
    if p is not None and p > P_THRESHOLD:
        flags.append(f"NS (p={p:.3f})")
    if not flags:
        return "SIG"
    return ", ".join(flags)


# =========================================================================
# Analysis 1: Session-Sequential Lead-Lag
# =========================================================================


def analysis_1_lead_lag(wide: pd.DataFrame) -> list[dict]:
    """For each session pair where A finishes before B, compute lift."""
    # Define valid lead-lag pairs: (leader_instrument, leader_session,
    #                                follower_instrument, follower_session)
    pairs = []

    # Within-session-sequence pairs (earlier session -> later session)
    for leader_sess, follower_sess in [
        ("0900", "1000"), ("0900", "1100"),
        ("1000", "1100"), ("1100", "1800"),
        ("1800", "2300"), ("1800", "0030"),
        ("2300", "0030"),
    ]:
        for l_inst in INSTRUMENTS:
            for f_inst in INSTRUMENTS:
                # Check both instruments have the sessions
                if (leader_sess in INSTRUMENT_SESSIONS.get(l_inst, []) and
                        follower_sess in INSTRUMENT_SESSIONS.get(f_inst, [])):
                    pairs.append((l_inst, leader_sess, f_inst, follower_sess))

    results = []
    for l_inst, l_sess, f_inst, f_sess in pairs:
        l_col = f"{l_inst}_{l_sess}_break_dir"
        f_col = f"{f_inst}_{f_sess}_break_dir"

        if l_col not in wide.columns or f_col not in wide.columns:
            continue

        for direction in ["long", "short"]:
            # Baseline: P(follower breaks direction)
            f_has_break = wide[f_col].notna()
            f_is_dir = wide[f_col] == direction
            baseline_n = f_has_break.sum()
            baseline_rate = f_is_dir.sum() / baseline_n if baseline_n > 0 else None

            # Conditional: P(follower breaks direction | leader broke direction)
            l_is_dir = wide[l_col] == direction
            cond_rate, cond_n = conditional_prob(l_is_dir, f_is_dir)

            if cond_rate is None or baseline_rate is None or baseline_rate == 0:
                continue

            lift = cond_rate / baseline_rate

            # Chi-squared / Fisher test
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
                "flag": significance_flag(cond_n, p_val),
            })

    return results


def print_analysis_1(results: list[dict]):
    """Print Analysis 1 results."""
    print("\n--- ANALYSIS 1: SESSION-SEQUENTIAL LEAD-LAG ---")
    print(
        "Question: Does instrument A breaking at session X predict "
        "instrument B breaking at later session Y?"
    )
    print()

    if not results:
        print("  No valid pairs found.")
        return

    # Sort by absolute lift distance from 1.0
    results.sort(key=lambda r: abs(r["lift"] - 1.0), reverse=True)

    header = (
        f"{'Leader':<14} {'Follower':<14} {'Dir':<6} "
        f"{'Base':>6} {'Cond':>6} {'Lift':>6} {'N':>5} "
        f"{'p':>7} {'Flag':<20}"
    )
    print(header)
    print("-" * len(header))

    for r in results[:40]:  # Top 40 by lift magnitude
        print(
            f"{r['leader']:<14} {r['follower']:<14} {r['direction']:<6} "
            f"{r['baseline']:>6.1%} {r['conditional']:>6.1%} {r['lift']:>6.2f} "
            f"{r['N_cond']:>5} {r['p_value']:>7.4f} {r['flag']:<20}"
        )


# =========================================================================
# Analysis 2: Concordance Filter
# =========================================================================


def analysis_2_concordance(wide: pd.DataFrame) -> dict:
    """At shared sessions, classify concordance and compute win rates."""
    results = {}

    for sess in SHARED_SESSIONS:
        cols = {inst: f"{inst}_{sess}_break_dir" for inst in INSTRUMENTS}
        outcome_cols = {inst: f"{inst}_{sess}_outcome" for inst in INSTRUMENTS}

        # Skip if columns missing
        if not all(c in wide.columns for c in cols.values()):
            continue

        sess_results = {"concordant_3": [], "majority_2": [], "split": []}

        for _, row in wide.iterrows():
            dirs = {inst: row.get(cols[inst]) for inst in INSTRUMENTS}
            outcomes = {inst: row.get(outcome_cols[inst]) for inst in INSTRUMENTS}

            # Filter to instruments that actually broke
            active = {k: v for k, v in dirs.items() if v in ("long", "short")}
            if len(active) < 2:
                continue

            long_count = sum(1 for v in active.values() if v == "long")
            short_count = sum(1 for v in active.values() if v == "short")

            if len(active) == 3 and (long_count == 3 or short_count == 3):
                category = "concordant_3"
                majority_dir = "long" if long_count == 3 else "short"
            elif long_count >= 2 or short_count >= 2:
                category = "majority_2"
                majority_dir = "long" if long_count >= short_count else "short"
            else:
                category = "split"
                majority_dir = None

            for inst in INSTRUMENTS:
                if dirs[inst] not in ("long", "short"):
                    continue
                is_majority = dirs[inst] == majority_dir if majority_dir else None
                is_win = outcomes[inst] == "win"
                sess_results[category].append({
                    "instrument": inst,
                    "direction": dirs[inst],
                    "is_majority": is_majority,
                    "is_win": is_win,
                    "outcome": outcomes[inst],
                })

        results[sess] = sess_results

    return results


def print_analysis_2(results: dict, wide: pd.DataFrame):
    """Print Analysis 2 results."""
    print("\n--- ANALYSIS 2: CONCORDANCE FILTER ---")
    print(
        "Question: When all 3 instruments break the same direction, "
        "are outcomes better?"
    )
    print()

    for sess in SHARED_SESSIONS:
        if sess not in results:
            continue

        print(f"  Session {sess}:")
        data = results[sess]

        for category, label in [
            ("concordant_3", "3/3 Concordant"),
            ("majority_2", "2/1 Majority"),
            ("split", "Split/Mixed"),
        ]:
            entries = data[category]
            if not entries:
                print(f"    {label}: N=0")
                continue

            wins = sum(1 for e in entries if e["is_win"])
            valid = sum(1 for e in entries if e["outcome"] in ("win", "loss"))
            n_days = len(entries)

            # Majority-side vs minority-side win rates
            if category in ("concordant_3", "majority_2"):
                maj_entries = [e for e in entries if e["is_majority"]]
                min_entries = [e for e in entries if e["is_majority"] is False]

                maj_valid = [e for e in maj_entries if e["outcome"] in ("win", "loss")]
                min_valid = [e for e in min_entries if e["outcome"] in ("win", "loss")]

                maj_wr = (
                    sum(1 for e in maj_valid if e["is_win"]) / len(maj_valid)
                    if maj_valid else None
                )
                min_wr = (
                    sum(1 for e in min_valid if e["is_win"]) / len(min_valid)
                    if min_valid else None
                )

                maj_str = f"{maj_wr:.1%}" if maj_wr is not None else "N/A"
                min_str = f"{min_wr:.1%}" if min_wr is not None else "N/A"

                flag = significance_flag(len(maj_valid), None)
                print(
                    f"    {label}: N={n_days:>4} trades "
                    f"| Majority WR={maj_str} (N={len(maj_valid)}) "
                    f"| Minority WR={min_str} (N={len(min_valid)}) "
                    f"| {flag}"
                )
            else:
                wr = wins / valid if valid > 0 else None
                wr_str = f"{wr:.1%}" if wr is not None else "N/A"
                print(f"    {label}: N={n_days:>4} trades | WR={wr_str} (N={valid})")

        # Baseline: unconditional win rate at this session
        baseline_wins = 0
        baseline_valid = 0
        for inst in INSTRUMENTS:
            oc = f"{inst}_{sess}_outcome"
            if oc in wide.columns:
                vals = wide[oc].dropna()
                baseline_wins += (vals == "win").sum()
                baseline_valid += vals.isin(["win", "loss"]).sum()

        if baseline_valid > 0:
            print(
                f"    Baseline (unconditional): WR={baseline_wins/baseline_valid:.1%} "
                f"(N={baseline_valid})"
            )
        print()


# =========================================================================
# Analysis 3: Gold Leads Equities (MGC 2300 -> MES/MNQ 0030)
# =========================================================================


def analysis_3_gold_leads(wide: pd.DataFrame) -> dict:
    """Deep dive: MGC 2300 break -> MES/MNQ 0030 outcomes."""
    results = {}

    mgc_dir_col = "MGC_2300_break_dir"
    mgc_size_col = "MGC_2300_size"
    mgc_db_col = "MGC_2300_double_break"
    mgc_atr_col = "MGC_atr_20"

    if mgc_dir_col not in wide.columns:
        return results

    for follower in ["MES", "MNQ"]:
        f_dir_col = f"{follower}_0030_break_dir"
        f_outcome_col = f"{follower}_0030_outcome"

        if f_dir_col not in wide.columns:
            continue

        follower_results = {}

        # --- Same-direction alignment ---
        for mgc_dir in ["long", "short"]:
            mask_mgc = wide[mgc_dir_col] == mgc_dir
            mask_follower_dir = wide[f_dir_col] == mgc_dir
            mask_follower_opp = wide[f_dir_col] == ("short" if mgc_dir == "long" else "long")

            # P(follower same dir | MGC dir)
            same_rate, same_n = conditional_prob(mask_mgc, mask_follower_dir)

            # Win rate when following MGC's direction
            aligned = mask_mgc & mask_follower_dir
            aligned_outcomes = wide.loc[aligned, f_outcome_col]
            aligned_valid = aligned_outcomes.isin(["win", "loss"])
            aligned_wins = (aligned_outcomes == "win").sum()
            aligned_n = aligned_valid.sum()
            aligned_wr = aligned_wins / aligned_n if aligned_n > 0 else None

            # Win rate going opposite (contrarian)
            opposed = mask_mgc & mask_follower_opp
            opp_outcomes = wide.loc[opposed, f_outcome_col]
            opp_valid = opp_outcomes.isin(["win", "loss"])
            opp_wins = (opp_outcomes == "win").sum()
            opp_n = opp_valid.sum()
            opp_wr = opp_wins / opp_n if opp_n > 0 else None

            follower_results[f"mgc_{mgc_dir}"] = {
                "follow_rate": same_rate,
                "follow_n": same_n,
                "aligned_wr": aligned_wr,
                "aligned_n": aligned_n,
                "contrarian_wr": opp_wr,
                "contrarian_n": opp_n,
            }

        # --- Double-break (choppy) filter ---
        mgc_double = wide[mgc_db_col] == True  # noqa: E712
        follower_after_double = wide.loc[mgc_double, f_outcome_col]
        db_valid = follower_after_double.isin(["win", "loss"])
        db_wins = (follower_after_double == "win").sum()
        db_n = db_valid.sum()

        follower_results["after_double_break"] = {
            "wr": db_wins / db_n if db_n > 0 else None,
            "n": db_n,
        }

        # --- ORB size conditional (large vs small relative to ATR) ---
        has_both = wide[mgc_dir_col].notna() & wide[mgc_size_col].notna() & wide[mgc_atr_col].notna()
        subset = wide[has_both].copy()
        if len(subset) > 0:
            subset["mgc_size_atr"] = subset[mgc_size_col] / subset[mgc_atr_col]
            median_ratio = subset["mgc_size_atr"].median()

            for size_label, size_mask in [
                ("large_orb", subset["mgc_size_atr"] >= median_ratio),
                ("small_orb", subset["mgc_size_atr"] < median_ratio),
            ]:
                size_outcomes = subset.loc[size_mask, f_outcome_col]
                sv = size_outcomes.isin(["win", "loss"])
                sw = (size_outcomes == "win").sum()
                sn = sv.sum()

                follower_results[size_label] = {
                    "wr": sw / sn if sn > 0 else None,
                    "n": sn,
                    "median_ratio": median_ratio,
                }

        results[follower] = follower_results

    return results


def print_analysis_3(results: dict, wide: pd.DataFrame):
    """Print Analysis 3 deep dive."""
    print("\n--- ANALYSIS 3: GOLD LEADS EQUITIES (MGC 2300 -> MES/MNQ 0030) ---")
    print(
        "Key asymmetry: MGC 2300 resolves ~90 min before MES/MNQ 0030"
    )
    print()

    if not results:
        print("  No data available (MGC 2300 or MES/MNQ 0030 columns missing).")
        return

    # Baseline 0030 win rates
    for follower in ["MES", "MNQ"]:
        oc = f"{follower}_0030_outcome"
        if oc in wide.columns:
            vals = wide[oc].dropna()
            valid = vals.isin(["win", "loss"])
            wr = (vals == "win").sum() / valid.sum() if valid.sum() > 0 else 0
            print(f"  {follower} 0030 baseline WR: {wr:.1%} (N={valid.sum()})")

    print()

    for follower, data in results.items():
        print(f"  {follower} 0030 after MGC 2300:")

        for mgc_dir in ["long", "short"]:
            key = f"mgc_{mgc_dir}"
            if key not in data:
                continue
            d = data[key]
            fr = f"{d['follow_rate']:.1%}" if d["follow_rate"] is not None else "N/A"
            awr = f"{d['aligned_wr']:.1%}" if d["aligned_wr"] is not None else "N/A"
            cwr = f"{d['contrarian_wr']:.1%}" if d["contrarian_wr"] is not None else "N/A"

            a_flag = significance_flag(d["aligned_n"], None)
            c_flag = significance_flag(d["contrarian_n"], None)

            print(
                f"    MGC {mgc_dir}: follow_rate={fr} (N={d['follow_n']}) "
                f"| aligned WR={awr} (N={d['aligned_n']}, {a_flag}) "
                f"| contrarian WR={cwr} (N={d['contrarian_n']}, {c_flag})"
            )

        # Double-break
        db = data.get("after_double_break", {})
        db_wr = f"{db['wr']:.1%}" if db.get("wr") is not None else "N/A"
        db_flag = significance_flag(db.get("n", 0), None)
        print(
            f"    After MGC 2300 double-break (choppy): "
            f"WR={db_wr} (N={db.get('n', 0)}, {db_flag})"
        )

        # Size conditional
        for size_label in ["large_orb", "small_orb"]:
            s = data.get(size_label, {})
            s_wr = f"{s['wr']:.1%}" if s.get("wr") is not None else "N/A"
            s_flag = significance_flag(s.get("n", 0), None)
            med = s.get("median_ratio")
            med_str = f" (split at {med:.3f} ATR)" if med is not None else ""
            print(
                f"    MGC 2300 {size_label}{med_str}: "
                f"{follower} 0030 WR={s_wr} (N={s.get('n', 0)}, {s_flag})"
            )

        print()


# =========================================================================
# Analysis 4: Divergence Signal
# =========================================================================


def analysis_4_divergence(wide: pd.DataFrame) -> dict:
    """When instruments disagree at shared sessions, what happens next?"""
    results = {}

    for sess_idx, sess in enumerate(SHARED_SESSIONS):
        cols = {inst: f"{inst}_{sess}_break_dir" for inst in INSTRUMENTS}
        if not all(c in wide.columns for c in cols.values()):
            continue

        # Find next session for each instrument
        next_sessions = {}
        for inst in INSTRUMENTS:
            inst_sessions = INSTRUMENT_SESSIONS[inst]
            if sess in inst_sessions:
                idx = inst_sessions.index(sess)
                if idx + 1 < len(inst_sessions):
                    next_sessions[inst] = inst_sessions[idx + 1]

        divergence_types = {
            "gold_vs_equities": {
                "desc": "MGC opposite to MES+MNQ",
                "check": lambda dirs: (
                    dirs.get("MGC") is not None
                    and dirs.get("MES") is not None
                    and dirs.get("MNQ") is not None
                    and dirs["MGC"] != dirs["MES"]
                    and dirs["MES"] == dirs["MNQ"]
                ),
            },
            "equities_disagree": {
                "desc": "MES and MNQ opposite directions",
                "check": lambda dirs: (
                    dirs.get("MES") is not None
                    and dirs.get("MNQ") is not None
                    and dirs["MES"] != dirs["MNQ"]
                    and dirs["MES"] in ("long", "short")
                    and dirs["MNQ"] in ("long", "short")
                ),
            },
        }

        sess_results = {}
        for div_name, div_info in divergence_types.items():
            div_days = []
            conc_days = []

            for _, row in wide.iterrows():
                dirs = {inst: row.get(cols[inst]) for inst in INSTRUMENTS}
                active = {k: v for k, v in dirs.items() if v in ("long", "short")}

                if len(active) < 2:
                    continue

                if div_info["check"](dirs):
                    div_days.append(row)
                elif len(active) == 3 and len(set(active.values())) == 1:
                    conc_days.append(row)

            # Next-session outcomes for divergent days
            div_next_outcomes = {}
            for inst, next_sess in next_sessions.items():
                oc = f"{inst}_{next_sess}_outcome"
                if oc not in wide.columns:
                    continue

                div_outs = [r.get(oc) for r in div_days if r.get(oc) in ("win", "loss")]
                conc_outs = [r.get(oc) for r in conc_days if r.get(oc) in ("win", "loss")]

                div_wr = sum(1 for o in div_outs if o == "win") / len(div_outs) if div_outs else None
                conc_wr = sum(1 for o in conc_outs if o == "win") / len(conc_outs) if conc_outs else None

                div_next_outcomes[f"{inst}_{next_sess}"] = {
                    "div_wr": div_wr,
                    "div_n": len(div_outs),
                    "conc_wr": conc_wr,
                    "conc_n": len(conc_outs),
                }

            sess_results[div_name] = {
                "desc": div_info["desc"],
                "n_divergent_days": len(div_days),
                "n_concordant_days": len(conc_days),
                "next_session_outcomes": div_next_outcomes,
            }

        results[sess] = sess_results

    return results


def print_analysis_4(results: dict):
    """Print Analysis 4 results."""
    print("\n--- ANALYSIS 4: DIVERGENCE SIGNAL ---")
    print(
        "Question: When instruments disagree at a shared session, "
        "what happens at the next session?"
    )
    print()

    for sess, sess_data in results.items():
        print(f"  Session {sess}:")
        for div_name, div_data in sess_data.items():
            desc = div_data["desc"]
            n_div = div_data["n_divergent_days"]
            n_conc = div_data["n_concordant_days"]

            print(f"    {desc}: {n_div} divergent days, {n_conc} concordant days")

            for target, out_data in div_data["next_session_outcomes"].items():
                d_wr = f"{out_data['div_wr']:.1%}" if out_data["div_wr"] is not None else "N/A"
                c_wr = f"{out_data['conc_wr']:.1%}" if out_data["conc_wr"] is not None else "N/A"
                d_flag = significance_flag(out_data["div_n"], None)
                c_flag = significance_flag(out_data["conc_n"], None)
                print(
                    f"      -> {target}: "
                    f"Divergent WR={d_wr} (N={out_data['div_n']}, {d_flag}) "
                    f"| Concordant WR={c_wr} (N={out_data['conc_n']}, {c_flag})"
                )
        print()


# =========================================================================
# Analysis 5: Volatility Regime Sync
# =========================================================================


def analysis_5_vol_regime(wide: pd.DataFrame) -> dict:
    """Bucket days by ATR-20 percentile rank, compute session win rates."""
    results = {}

    # Compute ATR-20 quartile per instrument
    for inst in INSTRUMENTS:
        col = f"{inst}_atr_20"
        if col in wide.columns:
            valid = wide[col].dropna()
            if len(valid) > 0:
                q25 = valid.quantile(0.25)
                q75 = valid.quantile(0.75)
                wide[f"{inst}_vol_q"] = np.where(
                    wide[col] >= q75, "high",
                    np.where(wide[col] <= q25, "low", "mid"),
                )
            else:
                wide[f"{inst}_vol_q"] = None

    # Classify synchronized regimes
    vol_cols = [f"{inst}_vol_q" for inst in INSTRUMENTS]
    if not all(c in wide.columns for c in vol_cols):
        return results

    regimes = {
        "all_high": lambda r: all(r.get(f"{i}_vol_q") == "high" for i in INSTRUMENTS),
        "all_low": lambda r: all(r.get(f"{i}_vol_q") == "low" for i in INSTRUMENTS),
        "mixed": lambda r: (
            not all(r.get(f"{i}_vol_q") == "high" for i in INSTRUMENTS)
            and not all(r.get(f"{i}_vol_q") == "low" for i in INSTRUMENTS)
            and all(r.get(f"{i}_vol_q") is not None for i in INSTRUMENTS)
        ),
    }

    for regime_name, regime_check in regimes.items():
        regime_mask = wide.apply(regime_check, axis=1)
        regime_subset = wide[regime_mask]
        n_days = len(regime_subset)

        session_wrs = {}
        for sess in SHARED_SESSIONS:
            for inst in INSTRUMENTS:
                oc = f"{inst}_{sess}_outcome"
                if oc not in wide.columns:
                    continue

                vals = regime_subset[oc].dropna()
                valid = vals.isin(["win", "loss"])
                wins = (vals == "win").sum()
                n = valid.sum()
                wr = wins / n if n > 0 else None

                # Baseline (all days)
                all_vals = wide[oc].dropna()
                all_valid = all_vals.isin(["win", "loss"])
                all_wr = (all_vals == "win").sum() / all_valid.sum() if all_valid.sum() > 0 else None

                session_wrs[f"{inst}_{sess}"] = {
                    "wr": wr,
                    "n": n,
                    "baseline_wr": all_wr,
                }

        results[regime_name] = {
            "n_days": n_days,
            "session_wrs": session_wrs,
        }

    return results


def print_analysis_5(results: dict):
    """Print Analysis 5 results."""
    print("\n--- ANALYSIS 5: VOLATILITY REGIME SYNC ---")
    print(
        "Question: Does synchronized high/low volatility across asset "
        "classes change ORB edge?"
    )
    print()

    regime_labels = {
        "all_high": "All 3 High-Vol (top 25% ATR-20)",
        "all_low": "All 3 Low-Vol (bottom 25% ATR-20)",
        "mixed": "Mixed Vol Regimes",
    }

    for regime_name in ["all_high", "all_low", "mixed"]:
        if regime_name not in results:
            continue
        data = results[regime_name]
        label = regime_labels[regime_name]
        print(f"  {label} ({data['n_days']} days):")

        for key, wr_data in sorted(data["session_wrs"].items()):
            wr = f"{wr_data['wr']:.1%}" if wr_data["wr"] is not None else "N/A"
            bl = f"{wr_data['baseline_wr']:.1%}" if wr_data["baseline_wr"] is not None else "N/A"
            flag = significance_flag(wr_data["n"], None)
            lift = ""
            if wr_data["wr"] is not None and wr_data["baseline_wr"] is not None and wr_data["baseline_wr"] > 0:
                lift_val = wr_data["wr"] / wr_data["baseline_wr"]
                lift = f"lift={lift_val:.2f}"
            print(
                f"    {key:<12}: WR={wr:>5} (N={wr_data['n']:>3}, {flag:<20}) "
                f"baseline={bl} {lift}"
            )
        print()


# =========================================================================
# Honest Summary
# =========================================================================


def compute_mes_mnq_correlation(wide: pd.DataFrame) -> dict:
    """Compute baseline MES-MNQ concordance at shared sessions."""
    concordance = {}
    for sess in SHARED_SESSIONS:
        mes_col = f"MES_{sess}_break_dir"
        mnq_col = f"MNQ_{sess}_break_dir"
        if mes_col not in wide.columns or mnq_col not in wide.columns:
            continue
        both_present = wide[mes_col].notna() & wide[mnq_col].notna()
        subset = wide[both_present]
        if len(subset) == 0:
            continue
        same = (subset[mes_col] == subset[mnq_col]).sum()
        concordance[sess] = {"same": int(same), "total": len(subset), "rate": same / len(subset)}
    return concordance


def print_honest_summary(
    lead_lag_results: list[dict],
    concordance_results: dict,
    gold_leads_results: dict,
    mes_mnq_corr: dict,
    n_days: int,
):
    """Print honest summary of what survived scrutiny."""
    print("\n" + "=" * 60)
    print("HONEST SUMMARY")
    print("=" * 60)

    # MES-MNQ baseline concordance
    print("\nBASELINE EXPECTATION:")
    for sess, data in mes_mnq_corr.items():
        print(
            f"  MES/MNQ concordance at {sess}: "
            f"{data['rate']:.0%} ({data['same']}/{data['total']})"
        )
    print("  (They're the same asset class — high concordance is expected, not a signal)")

    # Survived scrutiny
    survived = []
    not_survived = []

    for r in lead_lag_results:
        desc = f"{r['leader']} {r['direction']} -> {r['follower']}: lift={r['lift']:.2f}"
        if r["N_cond"] >= MIN_SAMPLE and r["p_value"] < P_THRESHOLD:
            survived.append(desc)
        elif r["N_cond"] < MIN_SAMPLE:
            not_survived.append(f"{desc} [N={r['N_cond']}, insufficient sample]")
        elif r["p_value"] >= P_THRESHOLD:
            not_survived.append(f"{desc} [p={r['p_value']:.3f}, not significant]")

    # Gold-leads results
    for follower, data in gold_leads_results.items():
        for mgc_dir in ["long", "short"]:
            key = f"mgc_{mgc_dir}"
            if key not in data:
                continue
            d = data[key]
            if d["aligned_wr"] is not None and d["aligned_n"] >= MIN_SAMPLE:
                survived.append(
                    f"MGC 2300 {mgc_dir} -> {follower} 0030 aligned: "
                    f"WR={d['aligned_wr']:.1%} (N={d['aligned_n']})"
                )
            elif d["aligned_n"] > 0:
                not_survived.append(
                    f"MGC 2300 {mgc_dir} -> {follower} 0030 aligned: "
                    f"N={d['aligned_n']}, insufficient sample"
                )

    print(f"\nSURVIVED SCRUTINY (N>={MIN_SAMPLE}, p<{P_THRESHOLD}):")
    if survived:
        for s in survived:
            print(f"  + {s}")
    else:
        print("  (none)")

    print(f"\nDID NOT SURVIVE:")
    if not_survived:
        for s in not_survived[:20]:  # Cap at 20 to avoid wall of text
            print(f"  - {s}")
        if len(not_survived) > 20:
            print(f"  ... and {len(not_survived) - 20} more")
    else:
        print("  (none)")

    print(f"\nCAVEATS:")
    print(f"  1. {n_days} overlapping days is thin — many conditional slices have N < 50")
    print(f"  2. MES ~ MNQ — their concordance is a baseline fact, not a discovery")
    print(f"  3. Multiple comparisons — testing ~50 combinations inflates false positives")
    print(f"  4. Feb 2024 - Feb 2026 includes specific market conditions (may not generalize)")
    print(f"  5. Outcomes use daily_features break_dir/outcome (parameter-free, RR=1.0)")


# =========================================================================
# Main
# =========================================================================


def main():
    print("=" * 60)
    print("CROSS-INSTRUMENT LEAD-LAG RESEARCH")
    print("=" * 60)

    # Connect read-only
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    # Load data
    df = load_features(con)
    con.close()

    # Check instrument coverage
    instruments_found = sorted(df["symbol"].unique())
    print(f"Instruments: {instruments_found}")

    # Find overlapping trading days (all 3 instruments present)
    day_counts = df.groupby("trading_day")["symbol"].nunique()
    overlap_days = day_counts[day_counts == 3].index
    print(f"Overlapping trading days (all 3): {len(overlap_days)}")

    if len(overlap_days) == 0:
        print("FATAL: No overlapping days found. Check daily_features data.")
        sys.exit(1)

    df_overlap = df[df["trading_day"].isin(overlap_days)]
    min_day = df_overlap["trading_day"].min()
    max_day = df_overlap["trading_day"].max()
    print(f"Period: {min_day} to {max_day}")
    print()

    # Build wide dataframe
    wide = build_wide_df(df_overlap)

    # Run analyses
    lead_lag_results = analysis_1_lead_lag(wide)
    print_analysis_1(lead_lag_results)

    concordance_results = analysis_2_concordance(wide)
    print_analysis_2(concordance_results, wide)

    gold_leads_results = analysis_3_gold_leads(wide)
    print_analysis_3(gold_leads_results, wide)

    divergence_results = analysis_4_divergence(wide)
    print_analysis_4(divergence_results)

    vol_results = analysis_5_vol_regime(wide)
    print_analysis_5(vol_results)

    # Honest summary
    mes_mnq_corr = compute_mes_mnq_correlation(wide)
    print_honest_summary(
        lead_lag_results,
        concordance_results,
        gold_leads_results,
        mes_mnq_corr,
        len(overlap_days),
    )


if __name__ == "__main__":
    main()
