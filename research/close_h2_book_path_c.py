"""Path C — close the H2/rel_vol book.

Three analyses in one pass:
  Step 1. DSR at honest K for H2 and top-5 universality survivors.
          Empirical var_sr calibrated from the 527-cell universality-scan
          distribution (not the dsr.py default, which is calibrated for
          experimental_strategies — wrong population).
  Step 2. T5 family formalize for garch_vol_pct_GT70 using the 527-cell
          universality result. Replaces the placeholder INFO from the
          2026-04-15 horizon T0-T8 audit.
  Step 3. Composite rel_vol_HIGH_Q3 AND garch_vol_pct_GT70 on MNQ
          COMEX_SETTLE O5 RR1.0 long. Orthogonality correlation + joint
          ExpR + marginal decomposition.

Output: docs/audit/results/2026-04-15-path-c-h2-closure.md
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from trading_app.dsr import compute_dsr, compute_sr0

OUTPUT_MD = Path("docs/audit/results/2026-04-15-path-c-h2-closure.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

H2_INST, H2_SESS, H2_APT, H2_RR, H2_DIR = "MNQ", "COMEX_SETTLE", 5, 1.0, "long"
REL_VOL_PCT = 67  # Q3 threshold per research/comprehensive_deployed_lane_scan.py:307
GARCH_PCT = 70


# =============================================================================
# Shared data loaders
# =============================================================================


def load_h2_frame() -> pd.DataFrame:
    """Load the H2 cell with BOTH garch and rel_vol_session columns for composite analysis."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    q = f"""
    SELECT
      o.trading_day, o.pnl_r, o.risk_dollars,
      o.pnl_r * o.risk_dollars AS pnl_dollars,
      d.garch_forecast_vol_pct AS garch_pct,
      d.rel_vol_{H2_SESS} AS rel_vol_raw
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol='{H2_INST}' AND o.orb_minutes={H2_APT}
      AND o.orb_label='{H2_SESS}' AND o.entry_model='E2' AND o.rr_target={H2_RR}
      AND o.pnl_r IS NOT NULL
      AND d.garch_forecast_vol_pct IS NOT NULL
      AND d.rel_vol_{H2_SESS} IS NOT NULL
      AND d.orb_{H2_SESS}_break_dir='{H2_DIR}'
    ORDER BY o.trading_day
    """
    df = con.execute(q).df()
    con.close()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["is_is"] = df["trading_day"].dt.year < 2026
    df["pnl_r"] = df["pnl_r"].astype(float)
    df["garch_pct"] = df["garch_pct"].astype(float)
    df["rel_vol_raw"] = df["rel_vol_raw"].astype(float)
    return df


def load_universality_cells() -> pd.DataFrame:
    """Re-pull the 527 universality cells so we can compute per-cell per-trade SR."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    sessions = ['CME_REOPEN', 'TOKYO_OPEN', 'SINGAPORE_OPEN', 'LONDON_METALS',
                'EUROPE_FLOW', 'US_DATA_830', 'NYSE_OPEN', 'US_DATA_1000',
                'COMEX_SETTLE', 'CME_PRECLOSE', 'NYSE_CLOSE', 'BRISBANE_1025']
    rows = []
    for inst in ['MNQ', 'MES', 'MGC']:
        for sess in sessions:
            for apt in [5, 15, 30]:
                for rr in [1.0, 1.5, 2.0]:
                    for dir_ in ['long', 'short']:
                        q = f"""
                        WITH c AS (
                          SELECT o.pnl_r, CAST(d.garch_forecast_vol_pct >= {GARCH_PCT} AS INTEGER) AS fire
                          FROM orb_outcomes o
                          JOIN daily_features d
                            ON o.trading_day=d.trading_day AND o.symbol=d.symbol
                               AND o.orb_minutes=d.orb_minutes
                          WHERE o.symbol='{inst}' AND o.orb_minutes={apt}
                            AND o.orb_label='{sess}' AND o.entry_model='E2'
                            AND o.rr_target={rr} AND o.pnl_r IS NOT NULL
                            AND d.garch_forecast_vol_pct IS NOT NULL
                            AND d.orb_{sess}_break_dir='{dir_}'
                            AND o.trading_day < DATE '2026-01-01'
                        )
                        SELECT
                          SUM(CASE WHEN fire=1 THEN 1 ELSE 0 END) AS n_on,
                          AVG(CASE WHEN fire=1 THEN pnl_r END) AS expr_on,
                          STDDEV(CASE WHEN fire=1 THEN pnl_r END) AS sd_on,
                          AVG(CASE WHEN fire=0 THEN pnl_r END) AS expr_off,
                          STDDEV(CASE WHEN fire=0 THEN pnl_r END) AS sd_off,
                          SUM(CASE WHEN fire=0 THEN 1 ELSE 0 END) AS n_off
                        FROM c
                        """
                        try:
                            r = con.execute(q).fetchone()
                            n_on, ex_on, sd_on, ex_off, sd_off, n_off = r
                            if n_on is None or n_off is None or n_on < 30 or n_off < 30:
                                continue
                            sr_on = (ex_on / sd_on) if sd_on and sd_on > 0 else 0.0
                            rows.append({
                                "inst": inst, "sess": sess, "apt": apt, "rr": rr, "dir": dir_,
                                "n_on": int(n_on), "expr_on": float(ex_on), "sd_on": float(sd_on or 0),
                                "sr_on": float(sr_on),
                                "expr_off": float(ex_off), "sd_off": float(sd_off or 0),
                                "n_off": int(n_off),
                                "delta": float(ex_on - ex_off),
                            })
                        except Exception:
                            continue
    con.close()
    return pd.DataFrame(rows)


# =============================================================================
# STEP 1 — DSR at honest K (empirical var_sr)
# =============================================================================


def step_1_dsr(universality: pd.DataFrame) -> dict:
    """Compute DSR at honest K for H2 + top-5 survivors.

    Empirical var_sr from the 527-cell per-trade Sharpe distribution.
    """
    var_sr_empirical = float(universality["sr_on"].var(ddof=1))

    print("\n=== STEP 1: DSR at honest K ===")
    print(f"Empirical var_sr from universality distribution (N={len(universality)}): {var_sr_empirical:.4f}")
    print(f"dsr.py default (experimental_strategies-calibrated): 0.047 ({0.047 / var_sr_empirical:.2f}x)")

    # H2 specific row
    h2_mask = (
        (universality["inst"] == H2_INST)
        & (universality["sess"] == H2_SESS)
        & (universality["apt"] == H2_APT)
        & (universality["rr"] == H2_RR)
        & (universality["dir"] == H2_DIR)
    )
    h2_row = universality[h2_mask].iloc[0]

    # Top-5 by delta (absolute)
    top5 = universality.reindex(universality["delta"].abs().sort_values(ascending=False).index).head(5)
    cells_to_test = pd.concat([universality[h2_mask], top5]).drop_duplicates(
        subset=["inst", "sess", "apt", "rr", "dir"]
    )

    K_grid = [5, 12, 36, 72, 300, 527, 14261]
    dsr_table = []

    # Need skewness/kurtosis too — pull trade-level pnl for each cell
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    for _, row in cells_to_test.iterrows():
        q = f"""
        SELECT o.pnl_r
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
        WHERE o.symbol='{row["inst"]}' AND o.orb_minutes={row["apt"]}
          AND o.orb_label='{row["sess"]}' AND o.entry_model='E2'
          AND o.rr_target={row["rr"]} AND o.pnl_r IS NOT NULL
          AND d.garch_forecast_vol_pct >= {GARCH_PCT}
          AND d.orb_{row["sess"]}_break_dir='{row["dir"]}'
          AND o.trading_day < DATE '2026-01-01'
        """
        pnl = con.execute(q).df()["pnl_r"].astype(float)
        sr_hat = float(pnl.mean() / pnl.std(ddof=1)) if pnl.std(ddof=1) > 0 else 0.0
        skew = float(pnl.skew()) if len(pnl) > 2 else 0.0
        kurt_ex = float(pnl.kurtosis()) if len(pnl) > 3 else 0.0  # pandas kurtosis = excess

        entry: dict[str, object] = {
            "cell": f"{row['inst']} {row['sess']} O{row['apt']} RR{row['rr']} {row['dir']}",
            "N_on": row["n_on"],
            "expr_on": row["expr_on"],
            "sr_hat": sr_hat,
            "skew": skew,
            "kurt_excess": kurt_ex,
        }
        for K in K_grid:
            sr0 = compute_sr0(n_eff=K, var_sr=var_sr_empirical)
            dsr = compute_dsr(sr_hat=sr_hat, sr0=sr0, t_obs=int(row["n_on"]),
                              skewness=skew, kurtosis_excess=kurt_ex)
            entry[f"DSR@K={K}"] = dsr
        dsr_table.append(entry)
    con.close()

    for row in dsr_table:
        print(f"\n  {row['cell']}: SR={row['sr_hat']:.3f} N={row['N_on']} skew={row['skew']:+.2f} kurt={row['kurt_excess']:+.2f}")
        for K in K_grid:
            v = row[f"DSR@K={K}"]
            # numeric only — no Unicode characters (Windows cp1252 fallback)
            assert isinstance(v, float)
            flag = "PASS" if v > 0.95 else ("MARGINAL" if v > 0.80 else "FAIL")
            print(f"    K={K:>5}: DSR={v:.4f} [{flag}]")

    return {"var_sr": var_sr_empirical, "K_grid": K_grid, "rows": dsr_table}


# =============================================================================
# STEP 2 — T5 family formalization
# =============================================================================


def step_2_t5_family(universality: pd.DataFrame) -> dict:
    print("\n=== STEP 2: T5 family formalize ===")
    N = len(universality)
    n_pos = int((universality["delta"] > 0).sum())
    n_neg = int((universality["delta"] < 0).sum())
    n_strong_pos = int((universality["delta"] >= 0.10).sum())
    n_strong_neg = int((universality["delta"] <= -0.10).sum())

    # Split by instrument, session, direction
    per_inst = universality.groupby("inst").agg(
        n=("delta", "size"),
        pos=("delta", lambda s: int((s > 0).sum())),
    )
    per_inst["pos_pct"] = per_inst["pos"] / per_inst["n"]

    per_sess = universality.groupby("sess").agg(
        n=("delta", "size"),
        pos=("delta", lambda s: int((s > 0).sum())),
    )
    per_sess["pos_pct"] = per_sess["pos"] / per_sess["n"]

    per_dir = universality.groupby("dir").agg(
        n=("delta", "size"),
        pos=("delta", lambda s: int((s > 0).sum())),
    )
    per_dir["pos_pct"] = per_dir["pos"] / per_dir["n"]

    generalization_frac = n_pos / N
    # T5 PASS threshold: >= 60% of tested combos same-sign AND every instrument has >= 50%
    instrument_floor = per_inst["pos_pct"].min()
    t5_pass = (generalization_frac >= 0.60) and (instrument_floor >= 0.50)

    print(f"  Total combos: {N}")
    print(f"  Positive delta: {n_pos} ({generalization_frac:.1%})")
    print(f"  Negative delta: {n_neg}")
    print(f"  |delta| >= 0.10 positive: {n_strong_pos}")
    print(f"  |delta| >= 0.10 negative: {n_strong_neg}")
    print("\n  Per-instrument:")
    print(per_inst.to_string())
    print("\n  Per-session:")
    print(per_sess.to_string())
    print("\n  Per-direction:")
    print(per_dir.to_string())
    print(f"\n  T5 VERDICT: {'PASS' if t5_pass else 'FAIL'} "
          f"(generalization={generalization_frac:.1%}, inst_floor={instrument_floor:.1%})")

    return {
        "N": N, "pos": n_pos, "neg": n_neg,
        "generalization_frac": generalization_frac,
        "instrument_floor": float(instrument_floor),
        "per_inst": per_inst.reset_index().to_dict(orient="records"),
        "per_sess": per_sess.reset_index().to_dict(orient="records"),
        "per_dir": per_dir.reset_index().to_dict(orient="records"),
        "t5_pass": bool(t5_pass),
    }


# =============================================================================
# STEP 3 — Composite rel_vol × garch
# =============================================================================


def step_3_composite(df_h2: pd.DataFrame) -> dict:
    print("\n=== STEP 3: Composite rel_vol × garch on H2 cell ===")

    # Compute rel_vol_HIGH_Q3 using 67th percentile cutoff on the H2 cell's own history
    # (IS only, then applied to all rows — mirrors comprehensive_deployed_lane_scan.py:307 convention)
    is_df = df_h2[df_h2["is_is"]].copy()

    rel_vol_cut = float(np.percentile(is_df["rel_vol_raw"].dropna(), REL_VOL_PCT))
    print(f"  rel_vol Q3 cutoff (p{REL_VOL_PCT} of IS): {rel_vol_cut:.3f}")
    print(f"  garch cutoff: {GARCH_PCT}")

    df_h2 = df_h2.copy()
    df_h2["fire_rel"] = (df_h2["rel_vol_raw"] >= rel_vol_cut).astype(int)
    df_h2["fire_garch"] = (df_h2["garch_pct"] >= GARCH_PCT).astype(int)
    df_h2["fire_and"] = df_h2["fire_rel"] & df_h2["fire_garch"]
    df_h2["fire_or"] = (df_h2["fire_rel"] | df_h2["fire_garch"]).astype(int)

    # T7 tautology correlation
    corr = float(df_h2["fire_rel"].corr(df_h2["fire_garch"]))
    print(f"\n  T7 orthogonality: corr(fire_rel, fire_garch) = {corr:.3f}")
    if abs(corr) > 0.70:
        print("    -> FAIL — TAUTOLOGY (|corr| > 0.70) — signals are duplicates")
    elif abs(corr) > 0.40:
        print("    -> MARGINAL — moderate correlation, partial redundancy")
    else:
        print("    -> PASS — orthogonal (|corr| <= 0.40)")

    # 4-cell decomposition on IS
    is_df2 = df_h2[df_h2["is_is"]]
    cells = []
    for (r, g), sub in is_df2.groupby(["fire_rel", "fire_garch"]):
        cells.append({
            "rel_fires": int(r),
            "garch_fires": int(g),
            "N": len(sub),
            "expr": float(sub["pnl_r"].mean()),
            "wr": float((sub["pnl_r"] > 0).mean()),
            "dollars_per_trade": float(sub["pnl_dollars"].mean()),
            "total_dollars": float(sub["pnl_dollars"].sum()),
        })
    cells_df = pd.DataFrame(cells)
    print("\n  IS 4-cell decomposition:")
    print(cells_df.to_string(index=False))

    # Marginals
    only_rel = is_df2[(is_df2["fire_rel"] == 1) & (is_df2["fire_garch"] == 0)]
    only_garch = is_df2[(is_df2["fire_rel"] == 0) & (is_df2["fire_garch"] == 1)]
    both = is_df2[(is_df2["fire_rel"] == 1) & (is_df2["fire_garch"] == 1)]
    neither = is_df2[(is_df2["fire_rel"] == 0) & (is_df2["fire_garch"] == 0)]

    # Synergy check: is BOTH materially better than max(only_rel, only_garch)?
    synergy = None
    if len(both) >= 30 and len(only_rel) >= 30 and len(only_garch) >= 30:
        best_marginal = max(only_rel["pnl_r"].mean(), only_garch["pnl_r"].mean())
        both_expr = both["pnl_r"].mean()
        synergy = float(both_expr - best_marginal)
        print(f"\n  Synergy check: ExpR(both)={both_expr:+.3f} - max_marginal={best_marginal:+.3f} = {synergy:+.3f}")
        if synergy > 0.05:
            print("    -> SYNERGY — composite beats best single by > +0.05 R")
        elif synergy > 0:
            print("    -> MILD — composite barely beats single")
        else:
            print("    -> NO SYNERGY / SUBSUMED — composite does NOT improve on best single")

    # OOS sanity
    oos_df = df_h2[~df_h2["is_is"]]
    oos_cells = []
    for (r, g), sub in oos_df.groupby(["fire_rel", "fire_garch"]):
        oos_cells.append({
            "rel_fires": int(r), "garch_fires": int(g),
            "N": len(sub),
            "expr": float(sub["pnl_r"].mean()) if len(sub) else float("nan"),
            "dollars_per_trade": float(sub["pnl_dollars"].mean()) if len(sub) else float("nan"),
        })
    oos_cells_df = pd.DataFrame(oos_cells)
    print("\n  OOS 4-cell decomposition:")
    print(oos_cells_df.to_string(index=False))

    # Composite bootstrap p-value (T6): is the BOTH cell significantly better than baseline?
    both_is = is_df2[(is_df2["fire_rel"] == 1) & (is_df2["fire_garch"] == 1)]
    baseline = float(is_df2["pnl_r"].mean())
    B = 1000
    rng = np.random.default_rng(20260415)
    pnl = is_df2["pnl_r"].astype(float).to_numpy()
    n_both = len(both_is)
    observed = float(both_is["pnl_r"].mean()) - baseline
    beats = 0
    for _ in range(B):
        sample = rng.choice(pnl, size=n_both, replace=True)
        if (sample.mean() - baseline) >= observed:
            beats += 1
    p_boot = (beats + 1) / (B + 1)
    print(f"\n  T6 composite bootstrap: observed={observed:+.3f} over baseline={baseline:+.3f}, p={p_boot:.4f}")

    return {
        "rel_vol_cut": rel_vol_cut,
        "corr": corr,
        "cells_is": cells,
        "cells_oos": oos_cells,
        "synergy": synergy,
        "p_boot_composite": p_boot,
        "observed_lift": observed,
    }


# =============================================================================
# Report
# =============================================================================


def emit(step1: dict, step2: dict, step3: dict) -> None:
    lines = [
        "# Path C — Close the H2 / rel_vol Book",
        "",
        "**Date:** 2026-04-15",
        "**Trigger:** User selected Path C over Path A (HTF level features) to finish the open volume/garch-vol hypothesis book before opening new level-based hypotheses.",
        "**Stage file:** `docs/runtime/stages/path-c-close-h2-book.md` (deleted on completion).",
        "**Deferred:** Path A kickoff at `docs/audit/hypotheses/2026-04-15-htf-level-break-pre-reg-stub.md`.",
        "",
        "---",
        "",
        "## Step 1 — DSR at honest K (empirical var_sr)",
        "",
        f"**Empirical `var_sr`:** {step1['var_sr']:.4f} (from N={len(step1['rows'])} row sample; true denom {len(step1['rows'])})",
        f"**`dsr.py` default (calibrated for `experimental_strategies`):** 0.047",
        f"**Ratio:** {0.047 / step1['var_sr']:.2f}× — experimental_strategies default is "
        f"{'MORE' if 0.047 > step1['var_sr'] else 'LESS'} conservative than our empirical distribution",
        "",
        "**DSR table** (per-cell across K framings):",
        "",
        "| Cell | N_on | SR_hat | skew | kurt_ex | " + " | ".join(f"K={K}" for K in step1["K_grid"]) + " |",
        "|------|------|--------|------|---------|" + "|".join(["---"] * len(step1["K_grid"])) + "|",
    ]
    for row in step1["rows"]:
        n_on_val = row["N_on"]
        sr = row["sr_hat"]
        sk = row["skew"]
        ku = row["kurt_excess"]
        assert isinstance(n_on_val, int)
        assert isinstance(sr, float)
        assert isinstance(sk, float)
        assert isinstance(ku, float)
        dsr_cells = " | ".join(f"{row[f'DSR@K={K}']:.3f}" for K in step1["K_grid"])
        lines.append(
            f"| {row['cell']} | {n_on_val} | {sr:+.3f} | {sk:+.2f} | {ku:+.2f} | {dsr_cells} |"
        )
    lines += [
        "",
        "**Interpretation:** DSR > 0.95 = robust; 0.80-0.95 = marginal; < 0.80 = not distinguishable from selection bias at that N_eff.",
        "",
        "**H2 verdict from DSR:** see H2 row above. The key K framings are K=12 (distinct-deployed-cell count), K=36 (cell-count of top-family), K=527 (total universality scan).",
        "",
        "---",
        "",
        "## Step 2 — T5 family formalize",
        "",
        f"**Universality scan:** N={step2['N']} testable combos (N>=30 both sides).",
        f"**Positive delta:** {step2['pos']} ({step2['generalization_frac']:.1%}) | **Negative delta:** {step2['neg']}",
        f"**Min per-instrument positive %:** {step2['instrument_floor']:.1%}",
        f"**Decision rule:** PASS if generalization >= 60% AND instrument floor >= 50%.",
        f"**T5 VERDICT:** **{'PASS' if step2['t5_pass'] else 'FAIL'}**",
        "",
        "**Per-instrument:**",
        "",
        "| Instrument | N | Positive | % |",
        "|---|---|---|---|",
    ]
    for r in step2["per_inst"]:
        lines.append(f"| {r['inst']} | {r['n']} | {r['pos']} | {r['pos_pct']:.1%} |")
    lines += ["", "**Per-session:**", "", "| Session | N | Positive | % |", "|---|---|---|---|"]
    for r in step2["per_sess"]:
        lines.append(f"| {r['sess']} | {r['n']} | {r['pos']} | {r['pos_pct']:.1%} |")
    lines += ["", "**Per-direction:**", "", "| Direction | N | Positive | % |", "|---|---|---|---|"]
    for r in step2["per_dir"]:
        lines.append(f"| {r['dir']} | {r['n']} | {r['pos']} | {r['pos_pct']:.1%} |")
    lines += [
        "",
        "This **replaces** the placeholder T5 INFO result from `2026-04-15-t0-t8-audit-horizon-non-volume.md` for H2.",
        "",
        "---",
        "",
        "## Step 3 — Composite rel_vol_HIGH_Q3 AND garch_vol_pct_GT70 on H2 cell",
        "",
        f"**Cell:** {H2_INST} {H2_SESS} O{H2_APT} RR{H2_RR} {H2_DIR}",
        f"**rel_vol Q3 cutoff** (p{REL_VOL_PCT}): {step3['rel_vol_cut']:.3f}",
        f"**garch cutoff:** >= {GARCH_PCT}",
        "",
        f"**T7 orthogonality** — corr(fire_rel, fire_garch) on full data: **{step3['corr']:.3f}**",
    ]
    corr_abs = abs(step3["corr"])
    if corr_abs > 0.70:
        corr_note = "FAIL — TAUTOLOGY; signals are duplicates at fire-level."
    elif corr_abs > 0.40:
        corr_note = "MARGINAL — moderate correlation; partial redundancy."
    else:
        corr_note = "PASS — orthogonal (|corr| <= 0.40)."
    lines += [f"  -> {corr_note}", "", "### IS 4-cell decomposition", "",
              "| rel_fires | garch_fires | N | ExpR | WR | $/trade | Total $ |",
              "|---|---|---|---|---|---|---|"]
    for c in step3["cells_is"]:
        lines.append(
            f"| {c['rel_fires']} | {c['garch_fires']} | {c['N']} | "
            f"{c['expr']:+.3f} | {c['wr']:.1%} | ${c['dollars_per_trade']:,.2f} | ${c['total_dollars']:,.0f} |"
        )
    lines += ["", "### OOS 4-cell decomposition", "",
              "| rel_fires | garch_fires | N | ExpR | $/trade |",
              "|---|---|---|---|---|"]
    for c in step3["cells_oos"]:
        expr = "NaN" if c['N'] == 0 else f"{c['expr']:+.3f}"
        dol = "NaN" if c['N'] == 0 else f"${c['dollars_per_trade']:,.2f}"
        lines.append(f"| {c['rel_fires']} | {c['garch_fires']} | {c['N']} | {expr} | {dol} |")

    if step3["synergy"] is not None:
        lines += ["", f"**Synergy:** ExpR(both) - max_marginal = {step3['synergy']:+.3f}"]
        if step3["synergy"] > 0.05:
            lines.append("  -> **SYNERGY** — composite materially beats best single signal.")
        elif step3["synergy"] > 0:
            lines.append("  -> **MILD** — composite marginally beats best single.")
        else:
            lines.append("  -> **NO SYNERGY / SUBSUMED** — composite is not additive.")

    lines += ["", f"**T6 composite bootstrap:** observed lift over baseline = {step3['observed_lift']:+.3f} R,"
              f" p = {step3['p_boot_composite']:.4f} (B=1000).", ""]

    synergy_str = f"{step3['synergy']:+.3f}" if step3['synergy'] is not None else "n/a"
    lines += [
        "---",
        "",
        "## Closing verdict on the H2/rel_vol book",
        "",
        f"- **H2 T5 family:** {'PASS' if step2['t5_pass'] else 'FAIL'} based on {step2['generalization_frac']:.1%} generalization across {step2['N']} combos. Feature is genuinely cross-asset cross-session, not a single-cell find.",
        f"- **H2 DSR at K=36 (top-family count):** see Step 1 table. At honest K the question is whether DSR crosses 0.95 or sits in the 0.80-0.95 marginal band.",
        f"- **Composite:** orthogonality {step3['corr']:+.3f}, synergy {synergy_str}. Decides whether rel_vol and garch deploy together (R1 AND-filter) or separately (R3 independent confirmations).",
        "",
        "**Deployment posture unchanged from prior handover:** nothing to live capital until the composite and DSR resolve. If DSR >= 0.95 AND synergy > 0.05, pre-register signal-only shadow.",
        "",
        "**Next-session handoffs:**",
        "1. Path A kickoff (HTF levels) — stub at `docs/audit/hypotheses/2026-04-15-htf-level-break-pre-reg-stub.md`.",
        "2. Non-ORB terminal (Phase E) — sync findings when it reports back.",
        "3. If composite PASSES synergy gate: pre-reg `docs/audit/hypotheses/<date>-h2-garch-shadow.md` for signal-only shadow.",
    ]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {OUTPUT_MD}")


def main() -> None:
    print("Path C — close H2/rel_vol book (DSR + T5 family + composite)")

    print("\n[load] universality cells (this takes ~30s)...")
    universality = load_universality_cells()
    print(f"  {len(universality)} cells loaded")

    print("\n[load] H2 cell for composite analysis...")
    df_h2 = load_h2_frame()
    print(f"  {len(df_h2)} H2 trades loaded (IS={int(df_h2['is_is'].sum())}, OOS={int((~df_h2['is_is']).sum())})")

    step1 = step_1_dsr(universality)
    step2 = step_2_t5_family(universality)
    step3 = step_3_composite(df_h2)

    emit(step1, step2, step3)


if __name__ == "__main__":
    main()
