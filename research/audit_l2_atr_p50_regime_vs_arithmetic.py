"""
L2 ATR_P50 filter — vol-regime gate, arithmetic cost-gate, or real edge?

Question
--------
`MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` is the only non-cost-gate
filter class in the current `topstep_50k_mnq_auto` deployed portfolio
(other 5 lanes use ORB_G5 or COST_LT12). PR #71 classified L1 ORB_G5 as
ARITHMETIC_ONLY and left L2 ATR_P50 as follow-up (separate filter class,
separate audit).

This audit decomposes L2's ATR_P50 filter three ways:

1. RULE 8.2 ARITHMETIC_ONLY classification (same as PR #71).
2. Two-proportion z on WR directly (real directional content vs cost-amp).
3. ORB-size correlation: is ATR_P50 effectively a relabeled cost-gate?
   If ATR-fire days also have systematically larger ORBs, the ExpR lift
   may be mechanical cost amplification, not volatility-regime signal.
4. ORB_G5 comparison: does ATR_P50 fire have substantial overlap with
   ORB_G5 fire? If corr(ATR_P50_fire, ORB_G5_fire) > 0.70 on the same
   lane universe, RULE 7 flags the filter as TAUTOLOGY (duplicate gate).

Classification
--------------
Confirmatory audit per RULE 10 — no new pre-reg required (no write to
`experimental_strategies` or `validated_setups`).

Canonical sources
-----------------
- Filter spec: `trading_app/config.py:2832-2836 ATR_P50 = OwnATRPercentileFilter(min_pct=50.0)`.
- Feature: `daily_features.atr_20_pct` (rolling 252d percentile of
  ATR(20), pre-computed; `build_daily_features.py`).
- Pre-session-knowable (STARTUP-resolved per `describe()` atom).
- IS: `trading_day < 2026-01-01` (Mode A sacred holdout).

Literature grounding
--------------------
- **Chan 2008 Ch 7** (`chan_2008_ch7_regime_switching.md`) — volatility-
  regime-conditional strategy activation. Predicts that filtering by
  volatility percentile can improve directional content IF the underlying
  strategy is momentum-regime-sensitive.
- **Chordia et al 2018** — K_family framing.
- **Bailey-LdP 2014** — multiple-testing haircut framing.

Usage
-----
    python research/audit_l2_atr_p50_regime_vs_arithmetic.py

Emits `docs/audit/results/2026-04-21-l2-atr-p50-regime-vs-arithmetic-audit.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
from scipy import stats

from pipeline.paths import GOLD_DB_PATH

ATR_P50_MIN = 50.0  # canonical from trading_app/config.py:2835
ORB_G5_MIN_PTS = 5.0

RULE_82_WR_SPREAD_PP = 3.0
RULE_82_EXPR_DELTA = 0.10
TAUTOLOGY_CORR = 0.70
POWER_FLOOR = 30

# Lane spec: the sole deployed ATR_P50 lane
SYMBOL = "MNQ"
SESSION = "SINGAPORE_OPEN"
ORB_MINUTES = 15
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR_TARGET = 1.5
HOLDOUT = "2026-01-01"


@dataclass
class GroupStats:
    label: str
    n: int
    wr_pct: float
    exp_r: float
    sd_r: float
    mean_orb_pts: float
    median_orb_pts: float
    mean_atr_pct: float
    pnls: np.ndarray


def fetch_rows() -> list[tuple]:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        size_col = f"orb_{SESSION}_size"
        q = f"""
        SELECT
            o.pnl_r,
            o.outcome,
            d.{size_col} AS orb_size_pts,
            d.atr_20_pct AS atr_pct
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.entry_model = ?
          AND o.confirm_bars = ?
          AND o.rr_target = ?
          AND o.trading_day < CAST(? AS DATE)
          AND o.entry_ts IS NOT NULL
          AND o.pnl_r IS NOT NULL
          AND d.{size_col} IS NOT NULL
          AND d.atr_20_pct IS NOT NULL
        """
        return con.execute(
            q,
            [
                SYMBOL,
                SESSION,
                ORB_MINUTES,
                ENTRY_MODEL,
                CONFIRM_BARS,
                RR_TARGET,
                HOLDOUT,
            ],
        ).fetchall()
    finally:
        con.close()


def group_stats(rows: list[tuple], label: str) -> GroupStats:
    n = len(rows)
    if n == 0:
        empty = np.array([], dtype=float)
        return GroupStats(label, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, empty)
    pnls = np.array([float(r[0]) for r in rows], dtype=float)
    wins = sum(1 for r in rows if r[1] == "win")
    orb_sizes = np.array([float(r[2]) for r in rows], dtype=float)
    atr_pcts = np.array([float(r[3]) for r in rows], dtype=float)
    return GroupStats(
        label=label,
        n=n,
        wr_pct=100.0 * wins / n,
        exp_r=float(pnls.mean()),
        sd_r=float(pnls.std(ddof=1)) if n > 1 else 0.0,
        mean_orb_pts=float(orb_sizes.mean()),
        median_orb_pts=float(np.median(orb_sizes)),
        mean_atr_pct=float(atr_pcts.mean()),
        pnls=pnls,
    )


def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    if not 0 < p_pool < 1:
        return float("nan"), 1.0
    se = float(np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)))
    if se == 0.0:
        return float("nan"), 1.0
    z = (p1 - p2) / se
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t_val, p_val = stats.ttest_ind(a, b, equal_var=False)
    return float(t_val), float(p_val)


def render_md(
    rows: list[tuple],
    f: GroupStats,
    nf: GroupStats,
    welch_t: float,
    welch_p: float,
    z_wr: float,
    p_wr: float,
    orb_corr: float,
    tautology_fire: tuple[int, int, int, int, float],
) -> str:
    d_wr = f.wr_pct - nf.wr_pct
    d_er = f.exp_r - nf.exp_r
    d_orb = f.mean_orb_pts - nf.mean_orb_pts
    size_ratio = f.mean_orb_pts / nf.mean_orb_pts if nf.mean_orb_pts > 0 else float("nan")

    # RULE 8.2 classification
    wr_under = abs(d_wr) < RULE_82_WR_SPREAD_PP
    expr_over = abs(d_er) > RULE_82_EXPR_DELTA
    arith_flag = wr_under and expr_over

    wr_significant = (not np.isnan(p_wr)) and p_wr < 0.05
    behavioral_flag = wr_significant and abs(d_wr) >= RULE_82_WR_SPREAD_PP

    overlap_n, atr_only, orb_only, neither, tautology_corr = tautology_fire

    md: list[str] = []
    md.append("# L2 MNQ SINGAPORE_OPEN ATR_P50 — regime-gate vs arithmetic-gate confirmatory audit")
    md.append("")
    md.append("**Date:** 2026-04-21")
    md.append("**Branch:** `research/l2-atr-p50-arithmetic-vs-regime-audit`")
    md.append("**Script:** `research/audit_l2_atr_p50_regime_vs_arithmetic.py`")
    md.append(
        "**Parent claim:** `docs/runtime/lane_allocation.json` 2026-04-18 rebalance — "
        "lane deployed with annual_r=44.0, ExpR=+0.2407R, N=137, WR=53.3%, status HOT/DEPLOY."
    )
    md.append(
        "**Rule:** `backtesting-methodology.md § RULE 8.2` ARITHMETIC_ONLY; "
        "§ RULE 7 TAUTOLOGY; `research-truth-protocol.md § 10` confirmatory audit."
    )
    md.append("**Classification:** confirmatory audit (no new pre-reg required).")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## TL;DR")
    md.append("")
    if behavioral_flag:
        md.append(
            f"**ATR_P50 shows real directional content on L2.** Δ_WR = {d_wr:+.2f}pp "
            f"at p = {p_wr:.4f}; WR spread is statistically non-zero. NOT arithmetic_only."
        )
    elif arith_flag:
        md.append(
            f"**ATR_P50 fires RULE 8.2 ARITHMETIC_ONLY on L2.** Δ_WR = {d_wr:+.2f}pp "
            f"(z-p={p_wr:.3f}, below 3pp threshold), Δ_ExpR = {d_er:+.4f}R (above 0.10 "
            f"threshold). WR spread statistically indistinguishable from zero; the "
            f"observed ExpR lift is consistent with cost amplification, not volatility-"
            f"regime directional gating."
        )
    else:
        md.append(
            f"**ATR_P50 passes neither RULE 8.2 flag nor RULE 8.2 behavioral flag cleanly.** "
            f"Δ_WR = {d_wr:+.2f}pp (z-p={p_wr:.3f}), Δ_ExpR = {d_er:+.4f}R. Not arithmetic; "
            f"not conclusively behavioral either at p < 0.05. Needs narrower follow-up."
        )
    md.append("")
    if tautology_corr >= TAUTOLOGY_CORR:
        md.append(
            f"**RULE 7 TAUTOLOGY flag: corr(ATR_P50_fire, ORB_G5_fire) = {tautology_corr:.3f} "
            f"≥ {TAUTOLOGY_CORR}.** ATR_P50 fire events are near-duplicates of ORB_G5 fire "
            f"events on this lane — the filter is effectively a relabeled cost-gate, not "
            f"an independent volatility-regime signal."
        )
    else:
        md.append(
            f"**No RULE 7 TAUTOLOGY with ORB_G5**: fire-event correlation = {tautology_corr:.3f} "
            f"(< {TAUTOLOGY_CORR} threshold). ATR_P50 is a distinct gate from ORB_G5 on this lane."
        )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Scope")
    md.append("")
    md.append(
        f"- Lane: {SYMBOL} × {SESSION} × {ENTRY_MODEL} × CB={CONFIRM_BARS} × "
        f"O{ORB_MINUTES} × RR={RR_TARGET} × ATR_P50"
    )
    md.append(f"- IS window: trading_day < {HOLDOUT} (Mode A holdout)")
    md.append(f"- Total N (eligible, atr_20_pct + orb_size known): **{len(rows):,}**")
    md.append(
        "- Source: canonical `orb_outcomes` ⨝ `daily_features` on (trading_day, symbol, "
        "orb_minutes)"
    )
    md.append(
        f"- Filter spec (canonical `trading_app/config.py:2832-2836`): "
        f"`atr_20_pct >= {ATR_P50_MIN:g}` (pre-session, STARTUP-resolved)"
    )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## RULE 8.2 decomposition")
    md.append("")
    md.append("| Group | N | WR | ExpR | σ(R) | mean_ORB | median_ORB | mean_ATR_pct |")
    md.append("|---|---|---|---|---|---|---|---|")
    md.append(
        f"| FIRE (atr_pct ≥ {ATR_P50_MIN:g}) | {f.n:,} | {f.wr_pct:.2f}% | "
        f"{f.exp_r:+.4f}R | {f.sd_r:.3f} | {f.mean_orb_pts:.2f} | "
        f"{f.median_orb_pts:.2f} | {f.mean_atr_pct:.2f} |"
    )
    md.append(
        f"| NON-FIRE (atr_pct < {ATR_P50_MIN:g}) | {nf.n:,} | {nf.wr_pct:.2f}% | "
        f"{nf.exp_r:+.4f}R | {nf.sd_r:.3f} | {nf.mean_orb_pts:.2f} | "
        f"{nf.median_orb_pts:.2f} | {nf.mean_atr_pct:.2f} |"
    )
    md.append("")
    md.append(
        f"- **Δ_WR** = {d_wr:+.2f}pp "
        f"({'< 3pp threshold' if abs(d_wr) < RULE_82_WR_SPREAD_PP else '≥ 3pp threshold'})"
    )
    md.append(
        f"- **Two-proportion z on WR** = {z_wr:+.3f}, p = {p_wr:.4f} → "
        f"{'WR spread statistically zero' if p_wr >= 0.05 else 'WR spread significant at α=0.05'}"
    )
    md.append(
        f"- **Δ_ExpR** = {d_er:+.4f}R "
        f"({'> 0.10 threshold' if abs(d_er) > RULE_82_EXPR_DELTA else '≤ 0.10 threshold'})"
    )
    md.append(f"- **Welch t** on pnl_r = {welch_t:+.3f}, p = {welch_p:.4f}")
    md.append(f"- **mean ORB size fire - non-fire** = {d_orb:+.2f} pts (ratio = {size_ratio:.2f}x)")
    md.append(f"- **RULE 8.2**: |wr_spread|={abs(d_wr):.2f}pp, |Δ_ExpR|={abs(d_er):.4f}R")
    md.append(
        f"- **Verdict**: "
        f"{'ARITHMETIC_ONLY (cost-screen shape)' if arith_flag and not behavioral_flag else ('behavioral edge' if behavioral_flag else 'inconclusive at p=0.05')}"
    )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Does ATR correlate with ORB size? (arithmetic-confound check)")
    md.append("")
    md.append(
        "If ATR_P50 fire days have systematically larger ORBs, the ExpR lift could be "
        "mechanical cost amplification (same pattern as ORB_G5 on L1 per PR #71), "
        "not volatility-regime directional content."
    )
    md.append("")
    md.append(f"- Mean ORB size on ATR_P50 fire days: **{f.mean_orb_pts:.2f} pts**")
    md.append(f"- Mean ORB size on ATR_P50 non-fire days: **{nf.mean_orb_pts:.2f} pts**")
    md.append(
        f"- Size ratio (fire / non-fire): **{size_ratio:.2f}x** "
        f"(compare to L1 ORB_G5 ratio = 4.73x per PR #71)"
    )
    if size_ratio >= 1.5:
        md.append(
            "- Ratio ≥ 1.5 — sizeable arithmetic confound present. ATR_P50 is partially "
            "a size-proxy. Interpret Δ_ExpR accordingly."
        )
    else:
        md.append(
            "- Ratio < 1.5 — arithmetic confound is small. ATR_P50's ExpR lift is less "
            "plausibly a size artifact than ORB_G5's was on L1."
        )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## RULE 7 TAUTOLOGY check vs ORB_G5")
    md.append("")
    md.append(
        "For each lane trade, classify ATR_P50 fire and ORB_G5 fire. Compute overlap "
        "and Pearson correlation of the two binary fire vectors."
    )
    md.append("")
    md.append("| | ORB_G5 fire | ORB_G5 non-fire | row total |")
    md.append("|---|---|---|---|")
    md.append(f"| ATR_P50 fire | {overlap_n} | {atr_only} | {overlap_n + atr_only} |")
    md.append(f"| ATR_P50 non-fire | {orb_only} | {neither} | {orb_only + neither} |")
    md.append(f"| column total | {overlap_n + orb_only} | {atr_only + neither} | {len(rows)} |")
    md.append("")
    md.append(f"- Pearson correlation (binary fire vectors): **{tautology_corr:+.4f}**")
    md.append(f"- RULE 7 threshold: |corr| > {TAUTOLOGY_CORR} → flag TAUTOLOGY")
    if tautology_corr >= TAUTOLOGY_CORR:
        md.append(
            "- **TAUTOLOGY FLAGGED** — ATR_P50 on this lane duplicates the ORB_G5 gating. "
            "Would kill the additive-overlay hypothesis in H4 framing."
        )
    else:
        md.append(
            "- **No tautology** — ATR_P50 is operationally distinct from ORB_G5 on this lane."
        )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Literature grounding")
    md.append("")
    md.append(
        "- **Chan 2008 Ch 7** (`docs/institutional/literature/chan_2008_ch7_regime_switching.md`) — "
        "volatility-regime-conditional strategy activation. The canonical prior for "
        "\"ATR percentile gates a momentum strategy\" is that HIGH vol = trending "
        "regime = higher WR on a momentum-continuation lane. This audit tests "
        "whether ATR_P50 fits that pattern on L2."
    )
    md.append(
        "- **Chordia et al 2018** (`docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md`) "
        "— K_family and t-threshold framing. K is small here (1 lane, 1 filter, 1 "
        "aperture); the relevant bar is Chordia strict t ≥ 3.79 (no prior literature "
        "directly validates ATR_P50 as a specific pre-session filter on E2 CB1 RR1.5, "
        "so we have no theory-backed pathway claim to relax the threshold)."
    )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Follow-up (not in this PR)")
    md.append("")
    md.append(
        "1. If RULE 8.2 arithmetic-only fired, propose metadata reclassification on "
        "`OwnATRPercentileFilter.CONFIDENCE_TIER` to reflect cost-gate class rather "
        "than regime-gate class (per institutional-rigor.md § 5)."
    )
    md.append(
        "2. If behavioral content confirmed, the lane's H4 hypothesis status is "
        "reinforced — but the RR=1.0 / RR=2.0 variants should be audited before "
        "generalizing."
    )
    md.append(
        "3. If TAUTOLOGY flagged, ATR_P50 should be replaced in L2's filter spec "
        "with whichever of ORB_G5 / ATR_P50 has tighter confidence evidence (separate "
        "audit required)."
    )
    md.append(
        "4. Cross-session ATR_P50 audit (other sessions × MNQ × O15 × RR=1.5) to "
        "test whether the L2 finding generalizes or is session-specific. Separate PR."
    )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Reproduction")
    md.append("")
    md.append("```bash")
    md.append("python research/audit_l2_atr_p50_regime_vs_arithmetic.py")
    md.append("```")
    md.append("")
    md.append(
        "Writes this document to "
        "`docs/audit/results/2026-04-21-l2-atr-p50-regime-vs-arithmetic-audit.md`."
    )
    md.append("")
    return "\n".join(md)


def main() -> None:
    rows = fetch_rows()
    if not rows:
        raise RuntimeError("L2 IS universe is empty")

    fire_rows = [r for r in rows if float(r[3]) >= ATR_P50_MIN]
    nofire_rows = [r for r in rows if float(r[3]) < ATR_P50_MIN]

    f = group_stats(fire_rows, "fire")
    nf = group_stats(nofire_rows, "nofire")

    t_val, p_val = welch(f.pnls, nf.pnls)

    wins_f = int(round(f.wr_pct * f.n / 100))
    wins_nf = int(round(nf.wr_pct * nf.n / 100))
    z_wr, p_wr = two_proportion_z(wins_f, f.n, wins_nf, nf.n)

    # Tautology check vs ORB_G5 on same universe
    atr_fire_vec = np.array([1 if float(r[3]) >= ATR_P50_MIN else 0 for r in rows])
    orb_fire_vec = np.array([1 if float(r[2]) >= ORB_G5_MIN_PTS else 0 for r in rows])
    overlap_n = int(((atr_fire_vec == 1) & (orb_fire_vec == 1)).sum())
    atr_only = int(((atr_fire_vec == 1) & (orb_fire_vec == 0)).sum())
    orb_only = int(((atr_fire_vec == 0) & (orb_fire_vec == 1)).sum())
    neither = int(((atr_fire_vec == 0) & (orb_fire_vec == 0)).sum())
    # Pearson correlation of two binary vectors == phi coefficient
    if atr_fire_vec.std() > 0 and orb_fire_vec.std() > 0:
        corr = float(np.corrcoef(atr_fire_vec, orb_fire_vec)[0, 1])
    else:
        corr = float("nan")

    md = render_md(
        rows,
        f,
        nf,
        t_val,
        p_val,
        z_wr,
        p_wr,
        corr,
        (overlap_n, atr_only, orb_only, neither, corr),
    )
    out_path = Path(
        "docs/audit/results/2026-04-21-l2-atr-p50-regime-vs-arithmetic-audit.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")

    print("=== L2 ATR_P50 REGIME-vs-ARITHMETIC AUDIT ===")
    print(
        f"FIRE   : N={f.n:>4} WR={f.wr_pct:.2f}% ExpR={f.exp_r:+.4f}R mean_ORB={f.mean_orb_pts:.2f}pts mean_ATR_pct={f.mean_atr_pct:.2f}"
    )
    print(
        f"NONFIRE: N={nf.n:>4} WR={nf.wr_pct:.2f}% ExpR={nf.exp_r:+.4f}R mean_ORB={nf.mean_orb_pts:.2f}pts mean_ATR_pct={nf.mean_atr_pct:.2f}"
    )
    print(f"Welch t={t_val:+.3f} p={p_val:.4f}")
    print(
        f"d_WR={f.wr_pct - nf.wr_pct:+5.2f}pp z_WR_p={p_wr:.4f}   d_ExpR={f.exp_r - nf.exp_r:+.4f}R"
    )
    print(f"RULE 7 corr(ATR_P50, ORB_G5 fire vectors) = {corr:.4f}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
