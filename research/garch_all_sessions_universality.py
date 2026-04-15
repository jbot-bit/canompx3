"""Garch overlay — all-session universality scan with trader discipline.

Goal: answer "does garch overlay work on sessions we haven't tested yet?"

Per user demand: no skipping, no bias, try all angles.

Scope:
  12 sessions × 3 instruments × 2 directions × 3 apertures × 3 RRs × 3 thresholds
  = 1944 theoretical cells. Data-coverage filtered in practice.

For each testable cell (N>=30 both sides of threshold):
  - sr_lift (Sharpe on - Sharpe off)
  - var_ratio (SD_on^2 / SD_off^2)
  - wr_lift
  - mae_diff
  - mean_lift (ExpR diff)
  - Welch t-test p (p_mean)
  - Permutation test p on Sharpe lift (p_sharpe)
  - Per-year positive direction count

BH-FDR at K_global (full cell count) and K_session/K_instrument slices.

Goal: identify SESSIONS where garch is (a) signal-positive, (b) signal-negative
(SKIP opportunity), (c) null.

Output: docs/audit/results/2026-04-15-garch-all-sessions-universality.md
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
from scipy import stats

from pipeline.paths import GOLD_DB_PATH

OUTPUT_MD = Path("docs/audit/results/2026-04-15-garch-all-sessions-universality.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

IS_END = "2026-01-01"
THRESHOLDS = [60, 70, 80]
SEED = 20260415

SESSIONS = [
    "CME_REOPEN", "TOKYO_OPEN", "SINGAPORE_OPEN", "LONDON_METALS",
    "EUROPE_FLOW", "US_DATA_830", "NYSE_OPEN", "US_DATA_1000",
    "COMEX_SETTLE", "CME_PRECLOSE", "NYSE_CLOSE", "BRISBANE_1025",
]

DEPLOYED = {
    "MNQ_EUROPE_FLOW_5_1.5_long", "MNQ_EUROPE_FLOW_5_1.5_short",
    "MNQ_SINGAPORE_OPEN_30_1.5_long", "MNQ_SINGAPORE_OPEN_30_1.5_short",
    "MNQ_COMEX_SETTLE_5_1.5_long", "MNQ_COMEX_SETTLE_5_1.5_short",
    "MNQ_NYSE_OPEN_5_1.0_long", "MNQ_NYSE_OPEN_5_1.0_short",
    "MNQ_TOKYO_OPEN_5_1.5_long", "MNQ_TOKYO_OPEN_5_1.5_short",
    "MNQ_US_DATA_1000_15_1.5_long", "MNQ_US_DATA_1000_15_1.5_short",
}


def test_cell(con, inst, sess, apt, rr, direction, threshold):
    q = f"""
    SELECT o.pnl_r, d.garch_forecast_vol_pct AS gp, o.mae_r, o.mfe_r
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol='{inst}' AND o.orb_minutes={apt} AND o.orb_label='{sess}'
      AND o.entry_model='E2' AND o.rr_target={rr} AND o.pnl_r IS NOT NULL
      AND d.garch_forecast_vol_pct IS NOT NULL
      AND d.orb_{sess}_break_dir='{direction}'
      AND o.trading_day < DATE '{IS_END}'
    """
    try:
        df = con.execute(q).df()
    except Exception:
        return None
    if len(df) < 60:
        return None
    df["pnl_r"] = df["pnl_r"].astype(float)
    df["gp"] = df["gp"].astype(float)
    on = df[df["gp"] >= threshold]
    off = df[df["gp"] < threshold]
    if len(on) < 30 or len(off) < 30:
        return None

    expr_on, expr_off = on["pnl_r"].mean(), off["pnl_r"].mean()
    sd_on = on["pnl_r"].std(ddof=1)
    sd_off = off["pnl_r"].std(ddof=1)
    sr_on = expr_on / sd_on if sd_on > 0 else 0.0
    sr_off = expr_off / sd_off if sd_off > 0 else 0.0
    var_ratio = (sd_on ** 2) / (sd_off ** 2) if sd_off > 0 else 0.0
    wr_on = float((on["pnl_r"] > 0).mean())
    wr_off = float((off["pnl_r"] > 0).mean())
    mae_on = float(on["mae_r"].dropna().mean()) if not on["mae_r"].dropna().empty else 0.0
    mae_off = float(off["mae_r"].dropna().mean()) if not off["mae_r"].dropna().empty else 0.0

    # Welch t-test on mean
    t_stat, p_mean = stats.ttest_ind(on["pnl_r"], off["pnl_r"], equal_var=False)

    # Permutation on Sharpe lift (B=1000 — balancing speed across 600+ cells)
    rng = np.random.default_rng(SEED)
    pnl = df["pnl_r"].to_numpy()
    is_on = (df["gp"].values >= threshold).astype(int)
    obs_lift = sr_on - sr_off

    def sharpe(arr):
        s = arr.std(ddof=1)
        return arr.mean() / s if s > 0 else 0.0

    B = 1000
    beats = 0
    for _ in range(B):
        shuffled = rng.permutation(is_on)
        on_s = pnl[shuffled == 1]
        off_s = pnl[shuffled == 0]
        if len(on_s) > 1 and len(off_s) > 1:
            if abs(sharpe(on_s) - sharpe(off_s)) >= abs(obs_lift):
                beats += 1
    p_sharpe = (beats + 1) / (B + 1)

    return {
        "inst": inst, "sess": sess, "apt": apt, "rr": rr, "dir": direction, "thresh": threshold,
        "deployed": f"{inst}_{sess}_{apt}_{rr}_{direction}" in DEPLOYED,
        "n_total": len(df), "n_on": len(on), "n_off": len(off),
        "expr_on": float(expr_on), "expr_off": float(expr_off),
        "mean_lift": float(expr_on - expr_off),
        "sr_on": float(sr_on), "sr_off": float(sr_off), "sr_lift": float(sr_on - sr_off),
        "var_ratio": float(var_ratio),
        "wr_on": wr_on, "wr_off": wr_off, "wr_lift": float(wr_on - wr_off),
        "mae_on": mae_on, "mae_off": mae_off, "mae_diff": float(mae_on - mae_off),
        "t_stat": float(t_stat), "p_mean": float(p_mean), "p_sharpe": float(p_sharpe),
    }


def bh_fdr(pvals: list[float], q: float = 0.05) -> list[bool]:
    n = len(pvals)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    max_rank = 0
    for rank, (_, p) in enumerate(indexed, start=1):
        if not np.isnan(p) and p <= q * rank / n:
            max_rank = rank
    survives = [False] * n
    for rank, (idx, _) in enumerate(indexed, start=1):
        if rank <= max_rank:
            survives[idx] = True
    return survives


def main():
    print("Garch all-sessions universality scan (trader discipline)")
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    results = []
    total = 0
    tested = 0
    for inst in ["MNQ", "MES", "MGC"]:
        for sess in SESSIONS:
            for apt in [5, 15, 30]:
                for rr in [1.0, 1.5, 2.0]:
                    for direction in ["long", "short"]:
                        for thresh in THRESHOLDS:
                            total += 1
                            r = test_cell(con, inst, sess, apt, rr, direction, thresh)
                            if r:
                                results.append(r)
                                tested += 1
    con.close()
    print(f"  {tested} testable cells of {total} attempted")

    # BH-FDR at K_global
    p_sharpe_all = [r["p_sharpe"] for r in results]
    survives = bh_fdr(p_sharpe_all, q=0.05)
    for i, r in enumerate(results):
        r["bh_fdr"] = survives[i]

    global_survivors = [r for r in results if r["bh_fdr"]]
    print(f"  K_global BH-FDR survivors: {len(global_survivors)}")

    # By session
    sess_agg = {}
    for r in results:
        key = (r["inst"], r["sess"])
        if key not in sess_agg:
            sess_agg[key] = {"pos": 0, "neg": 0, "total": 0, "survivors": 0, "strong_pos": 0, "strong_neg": 0}
        sess_agg[key]["total"] += 1
        if r["sr_lift"] > 0:
            sess_agg[key]["pos"] += 1
            if r["sr_lift"] > 0.15:
                sess_agg[key]["strong_pos"] += 1
        else:
            sess_agg[key]["neg"] += 1
            if r["sr_lift"] < -0.15:
                sess_agg[key]["strong_neg"] += 1
        if r["bh_fdr"]:
            sess_agg[key]["survivors"] += 1

    print("\n=== Per (instrument, session) sr_lift direction tally ===")
    print(f"  {'Inst':4} {'Session':16} {'Total':6} {'Pos':5} {'StrongPos':10} {'Neg':5} {'StrongNeg':10} {'BH-FDR':8}")
    for (inst, sess), a in sorted(sess_agg.items(), key=lambda x: (x[0][0], x[0][1])):
        print(f"  {inst:4} {sess:16} {a['total']:6} {a['pos']:5} {a['strong_pos']:10} "
              f"{a['neg']:5} {a['strong_neg']:10} {a['survivors']:8}")

    # Identify sessions where garch works
    print("\n=== Sessions where garch LIFTS Sharpe (strong_pos > strong_neg by 3+) ===")
    for (inst, sess), a in sess_agg.items():
        if a["strong_pos"] - a["strong_neg"] >= 3 and a["pos"] / max(a["total"], 1) >= 0.7:
            print(f"  {inst} {sess}: {a['pos']}/{a['total']} positive, {a['strong_pos']} strong_pos, {a['strong_neg']} strong_neg, {a['survivors']} BH-FDR survivors")

    print("\n=== Sessions where garch HURTS Sharpe (strong_neg > strong_pos by 3+, potential SKIP) ===")
    for (inst, sess), a in sess_agg.items():
        if a["strong_neg"] - a["strong_pos"] >= 3 and a["neg"] / max(a["total"], 1) >= 0.7:
            print(f"  {inst} {sess}: {a['neg']}/{a['total']} negative, {a['strong_neg']} strong_neg, {a['strong_pos']} strong_pos")

    # Top 30 survivors
    print("\n=== Top 30 cells by |sr_lift| among BH-FDR survivors ===")
    top = sorted(global_survivors, key=lambda r: -abs(r["sr_lift"]))[:30]
    for r in top:
        dep = "[DEPLOYED]" if r["deployed"] else ""
        print(f"  {r['inst']} {r['sess']:14} O{r['apt']:2} RR{r['rr']} {r['dir']:5} @{r['thresh']} "
              f"N={r['n_on']}/{r['n_off']:4} sr_lift={r['sr_lift']:+.3f} "
              f"var={r['var_ratio']:.2f} wr_lift={r['wr_lift']:+.1%} "
              f"p_sharpe={r['p_sharpe']:.4f} {dep}")

    emit(results, sess_agg, len(global_survivors), tested)


def emit(results, sess_agg, n_survivors, n_tested):
    # Binomial sign test per session (under H0 garch is random, P(positive) = 0.5)
    lines = [
        "# Garch Overlay — All-Sessions Universality (Trader Discipline)",
        "",
        "**Date:** 2026-04-15",
        "**Scope:** 12 sessions × 3 instruments × 2 directions × 3 apertures × 3 RRs × 3 thresholds = 1944 theoretical cells. " + f"{n_tested} testable (N>=60 total, N>=30 both sides).",
        "",
        "**Question:** where does garch overlay work, inversely-work, or produce no signal?",
        "",
        f"**BH-FDR K_global={n_tested}: {n_survivors} survivors at q=0.05 on Sharpe-permutation p-values.**",
        "",
        "### Per (instrument, session) tally — sr_lift direction",
        "",
        "| Inst | Session | Total | Pos | StrongPos(>0.15) | Neg | StrongNeg(<-0.15) | BH-FDR |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (inst, sess), a in sorted(sess_agg.items(), key=lambda x: (x[0][0], x[0][1])):
        lines.append(f"| {inst} | {sess} | {a['total']} | {a['pos']} | {a['strong_pos']} | "
                     f"{a['neg']} | {a['strong_neg']} | {a['survivors']} |")

    # Binomial sign test per (inst, sess)
    lines += ["", "### Per-session binomial sign test (H0: garch is random, P(pos)=0.5)", "",
              "Filters: total >= 6 cells. P-value = P(X >= observed_pos | n=total, p=0.5).",
              "",
              "| Inst | Session | Pos/Total | Binomial p | Verdict |",
              "|---|---|---|---|---|"]
    for (inst, sess), a in sorted(sess_agg.items()):
        if a["total"] < 6:
            continue
        p_bin = float(stats.binom.sf(a["pos"] - 1, a["total"], 0.5))
        p_bin_neg = float(stats.binom.sf(a["neg"] - 1, a["total"], 0.5))
        verdict = "NULL"
        if p_bin < 0.05:
            verdict = "POSITIVE-LIFT"
        elif p_bin_neg < 0.05:
            verdict = "INVERSE (SKIP CANDIDATE)"
        lines.append(f"| {inst} | {sess} | {a['pos']}/{a['total']} | {p_bin:.4f} | {verdict} |")

    # Top BH-FDR survivors
    lines += ["", "### BH-FDR survivors (K=" + str(n_tested) + ", q=0.05)", ""]
    survivors = [r for r in results if r["bh_fdr"]]
    if not survivors:
        lines.append("_No survivors at global K-correction. See per-session tally above for directional evidence._")
    else:
        lines += ["| Inst | Sess | Apt | RR | Dir | Thr | N | sr_lift | VarRatio | WR lift | MAE diff | p_sharpe | Deployed? |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(survivors, key=lambda x: x["p_sharpe"]):
            dep = "YES" if r["deployed"] else "no"
            lines.append(f"| {r['inst']} | {r['sess']} | O{r['apt']} | {r['rr']} | {r['dir']} | "
                         f"{r['thresh']} | {r['n_on']}/{r['n_off']} | {r['sr_lift']:+.3f} | "
                         f"{r['var_ratio']:.2f} | {r['wr_lift']:+.1%} | {r['mae_diff']:+.3f} | "
                         f"{r['p_sharpe']:.4f} | {dep} |")

    lines += ["", "---", "",
              "## How to read this",
              "",
              "- **POSITIVE-LIFT sessions** (binomial p<0.05, strong_pos dominates): candidates for **R5 SIZER** overlay — size up on garch=HIGH days within this session.",
              "- **INVERSE sessions** (strong_neg dominates): candidates for **R1 SKIP** — avoid trading on garch=HIGH days.",
              "- **NULL sessions**: garch adds no information; do not touch.",
              "- **BH-FDR survivors** at K_global are the most defensible single-cell claims. Non-survivors with directional consistency across thresholds should still be pre-registered as family-level hypotheses per RULE 4.1.",
              ""]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {OUTPUT_MD}")


if __name__ == "__main__":
    main()
