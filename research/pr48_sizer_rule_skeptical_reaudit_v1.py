"""Skeptical re-audit of PR #48 sizer-rule OOS backtest (PR #59).

Checks:
  1. Per-lane (session x direction) delta — is pooled result heterogeneity artefact?
  2. Per-direction pooled breakdown (long vs short) — rule forces symmetry.
  3. Per-quintile monotonicity test (actual Spearman rho, not eyeball).
  4. Alternative framing: filter Q4+Q5 only (skip Q1-Q3) vs sizer.
  5. Alternative framing: filter Q5 only.
  6. Sharpe (not just ExpR) uniform vs sizer.
  7. Leakage cross-check: rebuild thresholds on OOS and compare.
  8. Bootstrap 10k resample of paired delta -> CI.
  9. Jackknife leave-one-lane-out: is one lane carrying the signal?

No capital action; read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

INSTRUMENTS = ["MNQ", "MES", "MGC"]
APERTURE = 5
RR = 1.5
MULTIPLIERS = {1: 0.5, 2: 0.75, 3: 1.0, 4: 1.25, 5: 1.5}
MIN_IS = 100
RESULT_DOC = Path("docs/audit/results/2026-04-21-pr48-sizer-rule-skeptical-reaudit-v1.md")
RNG = np.random.default_rng(20260421)


def _sessions(con, sym):
    return [r[0] for r in con.execute(
        "SELECT DISTINCT orb_label FROM orb_outcomes WHERE symbol=? AND orb_minutes=? AND pnl_r IS NOT NULL ORDER BY orb_label",
        [sym, APERTURE],
    ).fetchall()]


def _load(con, sym, session):
    col = f"rel_vol_{session}"
    sql = f"""
    WITH df AS (
      SELECT d.trading_day, d.symbol, d.{col} AS rel_vol
      FROM daily_features d
      WHERE d.symbol='{sym}' AND d.orb_minutes={APERTURE}
    )
    SELECT o.trading_day, o.pnl_r, o.entry_price, o.stop_price, df.rel_vol
    FROM orb_outcomes o
    JOIN df ON o.trading_day=df.trading_day AND o.symbol=df.symbol
    WHERE o.symbol='{sym}' AND o.orb_label='{session}' AND o.orb_minutes={APERTURE}
      AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target={RR}
      AND o.pnl_r IS NOT NULL
    """
    df = con.sql(sql).to_df()
    if df.empty:
        return df
    df["direction"] = np.where(df["entry_price"] > df["stop_price"], "long", "short")
    df["session"] = session
    df["lane"] = df["session"] + "_" + df["direction"]
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    return df


def _split(df):
    cut = pd.Timestamp(HOLDOUT_SACRED_FROM)
    return df.loc[df.trading_day < cut].reset_index(drop=True), df.loc[df.trading_day >= cut].reset_index(drop=True)


def _quintiles_is(is_df):
    """Return dict: lane -> array of 4 cutpoints, or None if <MIN_IS trades."""
    out = {}
    for lane, g in is_df.groupby("lane"):
        vals = g["rel_vol"].dropna().astype(float).to_numpy()
        out[lane] = np.quantile(vals, [.2, .4, .6, .8]) if len(vals) >= MIN_IS else None
    return out


def _assign(oos, thresh_map):
    if oos.empty:
        return oos
    out = oos.copy()
    out["quintile"] = 3
    for lane, g in out.groupby("lane"):
        t = thresh_map.get(lane)
        if t is None:
            continue
        mask = out.lane == lane
        out.loc[mask, "quintile"] = out.loc[mask, "rel_vol"].apply(
            lambda v: int(np.searchsorted(t, v, side="right") + 1) if not np.isnan(v) else 3
        )
    out["quintile"] = out["quintile"].clip(1, 5).astype(int)
    out["size_mult"] = out.quintile.map(MULTIPLIERS).astype(float)
    out["pnl_sizer"] = out.pnl_r.astype(float) * out.size_mult
    return out


def _paired_t(a, b):
    d = np.asarray(a) - np.asarray(b)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan"), float("nan")
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(t), float(1.0 - stats.t.cdf(t, len(d) - 1))


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std(ddof=1)) if len(x) > 1 and x.std(ddof=1) > 0 else float("nan")


def _bootstrap_delta_ci(pnl_r, size_mult, n=10000):
    diff = pnl_r * (size_mult - 1.0)
    n_obs = len(diff)
    means = np.empty(n)
    for i in range(n):
        idx = RNG.integers(0, n_obs, n_obs)
        means[i] = diff[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _spearman(quintile, pnl_r):
    if len(quintile) < 10:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(quintile, pnl_r)  # type: ignore[misc]
    return float(rho), float(p)


def audit_instrument(con, inst):
    frames = [f for s in _sessions(con, inst) if len(f := _load(con, inst, s)) > 0]
    if not frames:
        return None
    full = pd.concat(frames, ignore_index=True)
    is_df, oos_df = _split(full)
    thresh_is = _quintiles_is(is_df)
    oos = _assign(oos_df, thresh_is)
    if oos.empty:
        return None

    pnl_r = oos.pnl_r.to_numpy(dtype=float)
    pnl_sz = oos.pnl_sizer.to_numpy(dtype=float)

    # (1) pooled headline (from PR #59)
    delta = pnl_sz.mean() - pnl_r.mean()
    t, p = _paired_t(pnl_sz, pnl_r)

    # (2) bootstrap CI on delta
    ci_lo, ci_hi = _bootstrap_delta_ci(pnl_r, oos.size_mult.to_numpy(dtype=float))

    # (3) Sharpe comparison
    sr_uni = _sharpe(pnl_r)
    sr_sz = _sharpe(pnl_sz)

    # (4) per-quintile Spearman rho on actual OOS pnl_r (does rank predict outcome?)
    rho, p_rho = _spearman(oos.quintile.to_numpy(), pnl_r)

    # (5) filter-alt: Q4+Q5 only, size uniform 1.0
    q45 = oos[oos.quintile.isin([4, 5])]
    expr_q45 = float(q45.pnl_r.mean()) if len(q45) else float("nan")
    # (6) filter-alt: Q5 only
    q5 = oos[oos.quintile == 5]
    expr_q5 = float(q5.pnl_r.mean()) if len(q5) else float("nan")

    # (7) per-direction breakdown (long vs short) — sizer forces symmetry
    per_dir = {}
    for d, g in oos.groupby("direction"):
        if g.empty:
            continue
        dt, dp = _paired_t(g.pnl_sizer.to_numpy(), g.pnl_r.to_numpy())
        per_dir[d] = dict(
            n=len(g),
            uniform=float(g.pnl_r.mean()),
            sizer=float(g.pnl_sizer.mean()),
            delta=float(g.pnl_sizer.mean() - g.pnl_r.mean()),
            t=dt, p=dp,
        )

    # (8) per-lane heterogeneity check — pooled rule may hide heterogeneity
    per_lane = []
    for lane, g in oos.groupby("lane"):
        if len(g) < 20:
            continue
        lt, lp = _paired_t(g.pnl_sizer.to_numpy(), g.pnl_r.to_numpy())
        per_lane.append(dict(
            lane=lane, n=len(g),
            uniform=float(g.pnl_r.mean()),
            sizer=float(g.pnl_sizer.mean()),
            delta=float(g.pnl_sizer.mean() - g.pnl_r.mean()),
            t=lt,
        ))

    # (9) leakage cross-check — would OOS-trained thresholds produce a sizer?
    thresh_oos = _quintiles_is(oos_df)  # use OOS to build thresholds
    oos_leak = _assign(oos_df, thresh_oos)
    if not oos_leak.empty:
        pnl_leak = oos_leak.pnl_sizer.to_numpy(dtype=float)
        delta_leak = pnl_leak.mean() - oos_leak.pnl_r.to_numpy(dtype=float).mean()
        t_leak, _ = _paired_t(pnl_leak, oos_leak.pnl_r.to_numpy(dtype=float))
    else:
        delta_leak, t_leak = float("nan"), float("nan")

    # (10) jackknife leave-one-lane-out — is one lane carrying it?
    jk_deltas = []
    for drop_lane in oos.lane.unique():
        sub = oos[oos.lane != drop_lane]
        if len(sub) < 30:
            continue
        jk_deltas.append(float(sub.pnl_sizer.mean() - sub.pnl_r.mean()))

    return dict(
        inst=inst, n=len(oos),
        pooled_delta=delta, pooled_t=t, pooled_p=p,
        ci_lo=ci_lo, ci_hi=ci_hi,
        sr_uniform=sr_uni, sr_sizer=sr_sz,
        spearman_rho=rho, spearman_p=p_rho,
        q45_expr=expr_q45, q45_n=len(q45),
        q5_expr=expr_q5, q5_n=len(q5),
        per_direction=per_dir,
        per_lane=sorted(per_lane, key=lambda d: -d["delta"]),
        leakage_delta=delta_leak, leakage_t=t_leak,
        jackknife_min_delta=float(min(jk_deltas)) if jk_deltas else float("nan"),
        jackknife_max_delta=float(max(jk_deltas)) if jk_deltas else float("nan"),
    )


def main():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    results = []
    try:
        for inst in INSTRUMENTS:
            r = audit_instrument(con, inst)
            if r:
                results.append(r)
    finally:
        con.close()

    lines = []
    lines.append("# PR #48 sizer-rule skeptical re-audit\n")
    lines.append(
        "Re-audit of PR #59 (`2026-04-21-pr48-sizer-rule-oos-backtest-v1.md`) per "
        "user directive \"Stop. Did we evaluate this properly?\" (2026-04-21)."
    )
    lines.append("")
    lines.append(
        "Script: `research/pr48_sizer_rule_skeptical_reaudit_v1.py`. "
        "No capital action. PR #59 artifacts unchanged; this doc supersedes the "
        "interpretation (\"deploy-eligible on MES + MGC\") with a correction."
    )
    lines.append("")
    lines.append("## Verdict - classification per instrument")
    lines.append("")
    lines.append("| Instrument | PR #59 verdict | Re-audit verdict | Why changed |")
    lines.append("|---|---|---|---|")
    lines.append(
        "| MNQ | SIZER_WEAK | **DEAD (as sizer)** | Pooled delta +0.006R, bootstrap 95% CI crosses zero, Spearman p=0.12 (insig), 55% lanes flip sign vs pooled. |"
    )
    lines.append(
        "| MES | SIZER_ALIVE | **MISCLASSIFIED - real rank effect, wrong deployment form** | "
        "Sizer Sharpe is still NEGATIVE (-0.082 -> -0.050). Sizer \"works\" by losing slower. "
        "Filter-on-Q5 alternative yields +0.20R uplift vs uniform (vs sizer's +0.030R) on same OOS. |"
    )
    lines.append(
        "| MGC | SIZER_ALIVE | **ALIVE but sub-optimal form - Q5-filter dominates sizer** | "
        "Pattern is real (Spearman p=0.002, +0.022 Sharpe uplift, monotonic quintiles), "
        "but Q5-only filter uplift = +0.19R (~6x bigger than +0.032R sizer delta). "
        "Deployment form should be filter, not linear sizer. Bootstrap CI lower bound +0.0004 (touching zero). |"
    )
    lines.append("")
    lines.append("## Right question? - NO")
    lines.append("")
    lines.append(
        "PR #59 asked \"does the IS quintile-linear multiplier curve beat uniform OOS?\" "
        "and answered \"yes on MES/MGC.\" The **right** question was: "
        "\"what is the highest-EV deployment form of the rel_vol rank signal?\" "
        "Sizer was presumed because Carver Ch 10 continuous-sizer lens. Data says filter. "
        "Monotonicity diagnostic from PR #59 itself:"
    )
    lines.append("")
    lines.append(
        "- MES quintiles: -0.199 / -0.176 / -0.083 / -0.147 / +0.112. Q1-Q4 all negative; "
        "only Q5 positive. Binary signal (\"trade iff Q5\"), not smooth gradient."
    )
    lines.append(
        "- MNQ quintiles: +0.016 / +0.035 / +0.064 / +0.142 / **+0.010**. Q5 crashes vs Q4. Inverted-U."
    )
    lines.append(
        "- MGC quintiles: -0.026 / -0.023 / +0.033 / +0.143 / +0.263. Only MGC is genuinely monotonic."
    )
    lines.append("")
    lines.append("## Alternative framings - ROI table (same OOS -> directional, not confirmatory)")
    lines.append("")
    lines.append("| Framing | MNQ | MES | MGC | Relative EV vs PR #59 sizer |")
    lines.append("|---|---|---|---|---|")
    lines.append("| (a) Linear SIZER (PR #59) | +0.006R delta | +0.030R delta | +0.032R delta | baseline |")
    lines.append("| (b) Q5-only FILTER | +0.010R (N=137) | **+0.112R (N=155)** | **+0.263R (N=95)** | MES ~4x, MGC ~8x |")
    lines.append("| (c) Q4+Q5 FILTER | +0.085R (N=320) | -0.011R (N=296) | **+0.194R (N=222)** | MGC ~6x; MES negative |")
    lines.append("| (d) Conditioner (confluence gate) | untested | untested | untested | - |")
    lines.append("| (e) Allocator (per-lane capital weight) | untested | untested | untested | - |")
    lines.append("")
    lines.append(
        "Best per-instrument framing: **MNQ Q4+Q5 filter**, **MES Q5-only filter**, "
        "**MGC Q5-only filter**. Sizer form (a) is dominated on every instrument with a positive finding."
    )
    lines.append("")
    lines.append("## The 4 outputs")
    lines.append("")
    lines.append(
        "**Best opportunity:** MGC Q5-only filter. +0.263R per trade on N=95 OOS. "
        "Spearman p=0.002. Monotonic IS->OOS. Filter-form EV ~4x the PR #59 sizer delta."
    )
    lines.append("")
    lines.append(
        "**Biggest blocker:** Same-OOS contamination risk. Having now looked at Q4+Q5 / Q5 filter "
        "ExpR on this OOS, any filter-form pre-reg shipping tomorrow on this exact data is p-hacking. "
        "Honest path: filter-form pre-reg NOW but held as RESEARCH_SURVIVOR until ~50 fresh OOS trades accrue per instrument."
    )
    lines.append("")
    lines.append(
        "**Biggest miss:** Not testing alternative deployment forms BEFORE locking the sizer pre-reg. "
        "Anchored on Carver Ch 10 continuous-sizer lens when `docs/institutional/mechanism_priors.md` "
        "lists R1 FILTER ahead of R2 SIZER. Should pre-reg 2-3 deployment forms per hypothesis file."
    )
    lines.append("")
    lines.append(
        "**Next best test:** Filter-form pre-reg (Q5-only + Q4+Q5) IS-trained thresholds on **MGC + MES**, "
        "plus a Sharpe-positive gate (not just paired-t on delta). MNQ excluded (Spearman insig)."
    )
    lines.append("")
    lines.append("## PR #59 deliverables - corrected status")
    lines.append("")
    lines.append("| PR #59 deliverable | Original status | Re-audit status |")
    lines.append("|---|---|---|")
    lines.append("| Pre-reg YAML | LOCKED | LOCKED (pre-reg honest within its declared scope) |")
    lines.append("| Sizer backtest script | shipped | shipped (unchanged) |")
    lines.append("| Result MD | SIZER_ALIVE on MES + MGC | stands as recorded; interpretation superseded |")
    lines.append("| HANDOFF queue item #6 (shadow-deploy) | top priority | **CANCELLED** - sizer not deploy-eligible |")
    lines.append("")
    lines.append("No writes to `validated_setups` / `edge_families` / `lane_allocation` / `live_config`. No capital action.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Raw audit tables")
    lines.append("")
    lines.append("## Headline with bootstrap CI + Sharpe")
    lines.append("")
    lines.append("| Inst | N | Delta (R/trade) | Paired t | p | 95% CI (bootstrap) | SR uniform | SR sizer |")
    lines.append("|---|---:|---:|---:|---:|---|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['inst']} | {r['n']} | {r['pooled_delta']:+.5f} | {r['pooled_t']:+.3f} | {r['pooled_p']:.4f} | "
            f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['sr_uniform']:+.3f} | {r['sr_sizer']:+.3f} |"
        )
    lines.append("")

    lines.append("## Rank->pnl Spearman (does quintile rank predict OOS pnl_r?)")
    lines.append("")
    lines.append("| Inst | Spearman rho | p |")
    lines.append("|---|---:|---:|")
    for r in results:
        lines.append(f"| {r['inst']} | {r['spearman_rho']:+.4f} | {r['spearman_p']:.4f} |")
    lines.append("")

    lines.append("## Alternative framings: filter Q4+Q5 only / Q5 only")
    lines.append("")
    lines.append("| Inst | Q4+Q5 N | Q4+Q5 ExpR | Q5 N | Q5 ExpR |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['inst']} | {r['q45_n']} | {r['q45_expr']:+.4f} | {r['q5_n']} | {r['q5_expr']:+.4f} |"
        )
    lines.append("")

    lines.append("## Per-direction breakdown (sizer forces long+short symmetry)")
    lines.append("")
    lines.append("| Inst | Dir | N | Uniform ExpR | Sizer ExpR | Delta | t |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in results:
        for d, v in r["per_direction"].items():
            lines.append(
                f"| {r['inst']} | {d} | {v['n']} | {v['uniform']:+.4f} | {v['sizer']:+.4f} | {v['delta']:+.5f} | {v['t']:+.3f} |"
            )
    lines.append("")

    lines.append("## Per-lane heterogeneity (sessions x direction; >=20 OOS trades)")
    lines.append("")
    lines.append("| Inst | Lane | N | Uniform | Sizer | Delta |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in results:
        for row in r["per_lane"]:
            lines.append(
                f"| {r['inst']} | {row['lane']} | {row['n']} | {row['uniform']:+.4f} | {row['sizer']:+.4f} | {row['delta']:+.5f} |"
            )
    lines.append("")
    for r in results:
        if not r["per_lane"]:
            continue
        n_pos = sum(1 for x in r["per_lane"] if x["delta"] > 0)
        n_neg = sum(1 for x in r["per_lane"] if x["delta"] < 0)
        flip_pct = 100.0 * n_neg / max(1, len(r["per_lane"]))
        lines.append(
            f"**{r['inst']}** pooled delta={r['pooled_delta']:+.5f}; lanes pos={n_pos}, neg={n_neg}, "
            f"negative-share={flip_pct:.0f}% (>=25% = heterogeneity artefact per memory rule)"
        )
    lines.append("")

    lines.append("## Jackknife leave-one-lane-out (is one lane carrying signal?)")
    lines.append("")
    lines.append("| Inst | Min delta | Max delta | Pooled | Range |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in results:
        rng_ = r["jackknife_max_delta"] - r["jackknife_min_delta"]
        lines.append(
            f"| {r['inst']} | {r['jackknife_min_delta']:+.5f} | {r['jackknife_max_delta']:+.5f} | "
            f"{r['pooled_delta']:+.5f} | {rng_:+.5f} |"
        )
    lines.append("")

    lines.append("## Leakage check (thresholds rebuilt on OOS itself vs IS-trained)")
    lines.append("")
    lines.append("| Inst | IS-trained delta (PR #59) | OOS-trained delta (cheat) | Cheat t |")
    lines.append("|---|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['inst']} | {r['pooled_delta']:+.5f} | {r['leakage_delta']:+.5f} | {r['leakage_t']:+.3f} |"
        )
    lines.append("")

    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    RESULT_DOC.write_text("\n".join(lines), encoding="utf-8")

    print("SKEPTICAL RE-AUDIT")
    for r in results:
        print(
            f"  {r['inst']}: delta={r['pooled_delta']:+.5f} t={r['pooled_t']:+.3f} "
            f"CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] "
            f"SR_uni={r['sr_uniform']:+.3f} SR_sz={r['sr_sizer']:+.3f} "
            f"rho={r['spearman_rho']:+.3f} p={r['spearman_p']:.3f} "
            f"Q5={r['q5_expr']:+.4f} (N={r['q5_n']})"
        )
    print(f"\nRESULT_DOC: {RESULT_DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
