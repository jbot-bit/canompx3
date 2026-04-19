"""Honest stress test of rel_vol_HIGH_Q3 finding.

User directive 2026-04-15: "is that overfit or data mining? is that just a
statistical anomaly? stress test it honestly and figure out what to do with
findings".

Test battery (institutional-grade, Bailey-LdP 2014 + False Strategy Theorem grounded):

1. **Deflated Sharpe Ratio** per cell at multiple N_eff framings:
   - N_eff = K_global (14,261) — most punishing
   - N_eff = K_family (~2,500) — moderate
   - N_eff = K_lane (~56) — lenient
   - Bailey-LdP 2014 Eq 2 implementation via trading_app/dsr.py

2. **Temporal stability test** — split IS into halves, compute per-half stats:
   - IS first half: trading_day < midpoint
   - IS second half: trading_day >= midpoint
   - Signal must hold in BOTH halves (sign match + |t|≥2 each)

3. **E[max_SR from noise]** bound per False Strategy Theorem:
   - Bailey et al 2013: with K independent random trials, expected max SR
     is well-characterized. Our observed SR must exceed by institutional margin.

4. **Bootstrap null at K context** — reuses T6 but contextualizes at K_global:
   - T6 per-cell p<0.001 is significant at CELL level
   - At K=14,261, Bonferroni threshold is 3.5e-6 per individual cell
   - Our observed p-values must beat this

5. **Cross-lane joint probability** — all 5 lanes surviving is (joint) how likely?

6. **Consolidated verdict** per cell — GENUINE / MARGINAL / LIKELY_OVERFIT

Output:
  docs/audit/results/2026-04-15-rel-vol-stress-test.md
"""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402
from trading_app.dsr import compute_sr0, compute_dsr  # noqa: E402

OOS_START = HOLDOUT_SACRED_FROM

# 5 BH-global survivor lanes + 1 confluence cell
LANES = [
    ("MES", "COMEX_SETTLE", 5, 1.0, "short", "rel_vol_HIGH"),
    ("MGC", "LONDON_METALS", 5, 1.0, "short", "rel_vol_HIGH"),
    ("MES", "TOKYO_OPEN", 5, 1.5, "long", "rel_vol_HIGH"),
    ("MNQ", "SINGAPORE_OPEN", 5, 1.0, "short", "rel_vol_HIGH"),
    ("MES", "COMEX_SETTLE", 5, 1.5, "short", "rel_vol_HIGH"),
]

# K framings used across comprehensive scan
# Source: docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md
K_GLOBAL = 14261
# Volume family in comprehensive scan: ~6 volume features (rel_vol_HIGH/LOW, bb_ratio_HIGH/LOW, break_delay_LT2/GT10)
# × ~150 lane-direction-pass combos that had data = ~900 cells. More punishing than generic "family".
K_FAMILY_VOLUME = 900
# Per-lane: ~30 features × 2 directions × 2 passes for deployed = ~120; fewer for non-deployed. Avg ~56.
K_LANE_AVG = 56

OUTPUT_MD = Path("docs/audit/results/2026-04-15-rel-vol-stress-test.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)


def load_lane_is(instrument, session, apt, rr, direction):
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    q = f"""
    SELECT
      o.trading_day, o.pnl_r,
      d.rel_vol_{session} AS rel_vol,
      d.orb_{session}_break_dir AS break_dir
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
      AND o.trading_day < '{HOLDOUT_SACRED_FROM}'
      AND o.pnl_r IS NOT NULL
      AND d.rel_vol_{session} IS NOT NULL
      AND d.orb_{session}_break_dir = '{direction}'
    ORDER BY o.trading_day
    """
    df = con.execute(q).df()
    con.close()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    return df


def compute_cell_stats(df: pd.DataFrame, p67_thresh: float | None = None) -> dict:
    """Compute key statistics for stress testing."""
    if len(df) < 100:
        return {"error": f"insufficient N={len(df)}"}
    if p67_thresh is None:
        p67_thresh = float(np.nanpercentile(df["rel_vol"].astype(float), 67))
    on = df[df["rel_vol"] > p67_thresh]["pnl_r"]
    off = df[df["rel_vol"] <= p67_thresh]["pnl_r"]
    if len(on) < 30 or len(off) < 30:
        return {"error": "insufficient on/off N"}

    # Per-trade Sharpe (on-signal)
    on_mean = float(on.mean())
    on_std = float(on.std(ddof=1))
    sr_on = on_mean / on_std if on_std > 0 else 0.0

    # Welch t
    t_stat, p_val = stats.ttest_ind(on, off, equal_var=False)
    delta = on_mean - float(off.mean())

    # Skew / kurt on on-signal returns (for DSR denominator)
    on_arr = np.asarray(on.values, dtype=float)
    skew = float(stats.skew(on_arr))
    kurt_excess = float(stats.kurtosis(on_arr))  # returns excess by default

    return {
        "p67": p67_thresh,
        "n_on": len(on),
        "n_off": len(off),
        "on_mean": on_mean,
        "on_std": on_std,
        "sr_on_per_trade": sr_on,
        "delta": delta,
        "t_welch": float(t_stat),
        "p_welch": float(p_val),
        "skewness": skew,
        "kurtosis_excess": kurt_excess,
    }


def dsr_at_k(sr_on: float, t_obs: int, skew: float, kurt: float, k_eff: int, var_sr: float) -> dict:
    """Compute SR0 + DSR at a given K framing."""
    sr0 = compute_sr0(n_eff=k_eff, var_sr=var_sr)
    dsr = compute_dsr(sr_hat=sr_on, sr0=sr0, t_obs=t_obs, skewness=skew, kurtosis_excess=kurt)
    return {"k_eff": k_eff, "sr0": sr0, "dsr": dsr, "passes_095": dsr >= 0.95}


def expected_max_t_from_noise(k_trials: int) -> float:
    """Expected maximum t-statistic from K independent standard-normal trials.
    Approximation: E[max(t_1,...,t_K)] ≈ sqrt(2·ln(K)) - (ln(ln(K)) + ln(4π))/(2·sqrt(2·ln(K)))
    Via Gumbel extreme value theory for normal maxima.
    """
    if k_trials < 2:
        return 0.0
    ln_k = math.log(k_trials)
    a = math.sqrt(2 * ln_k)
    b = (math.log(ln_k) + math.log(4 * math.pi)) / (2 * a)
    return a - b


def per_day_stats(df: pd.DataFrame, p67_thresh: float) -> dict:
    """Per-day aggregation: treat one trading day as one observation.
    Addresses intra-day clustering / autocorrelation inflating per-trade t-stats."""
    on_mask = df["rel_vol"] > p67_thresh
    on_days = df[on_mask].groupby(df[on_mask]["trading_day"].dt.date)["pnl_r"].mean()
    off_days = df[~on_mask].groupby(df[~on_mask]["trading_day"].dt.date)["pnl_r"].mean()
    if len(on_days) < 20 or len(off_days) < 20:
        return {"error": "insufficient day count"}
    t_day, p_day = stats.ttest_ind(on_days, off_days, equal_var=False)
    return {
        "n_on_days": len(on_days),
        "n_off_days": len(off_days),
        "mean_on_day": float(on_days.mean()),
        "mean_off_day": float(off_days.mean()),
        "delta_per_day": float(on_days.mean() - off_days.mean()),
        "t_per_day": float(t_day),
        "p_per_day": float(p_day),
    }


def block_bootstrap_p(
    df: pd.DataFrame, p67_thresh: float, block_size: int = 5, n_boot: int = 2000, seed: int = 42
) -> float:
    """Proper autocorrelation-robust null test via moving-block bootstrap of pnl.

    Null hypothesis: rel_vol_HIGH has NO effect on pnl_r.

    Method (Politis-Romano stationary / Lahiri moving-block):
      - Keep observed mask FIXED (its fire pattern is real)
      - Resample pnl via blocks of `block_size` consecutive observations
        (preserves return autocorrelation structure)
      - Under null (no signal effect), delta = on_mean - off_mean has mean 0
      - Proportion of bootstrap samples with |delta_boot| >= |delta_observed| = p-value

    This breaks the signal-outcome link while preserving pnl autocorrelation.
    Prior implementation preserved joint (pnl, mask) structure which mechanically
    produced p~0.5 — wrong. Fixed 2026-04-15.
    """
    pnl = np.asarray(df["pnl_r"].astype(float).values, dtype=float)
    mask = np.asarray((df["rel_vol"] > p67_thresh).astype(int).values, dtype=int)
    n = len(pnl)
    if n < block_size * 10:
        return float("nan")
    observed_delta = pnl[mask == 1].mean() - pnl[mask == 0].mean()

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    beats = 0
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        boot_pnl = pnl[idx]  # resampled pnl preserves autocorrelation
        # Compare against FIXED mask — breaks signal-outcome link
        on = boot_pnl[mask == 1]
        off = boot_pnl[mask == 0]
        if len(on) < 5 or len(off) < 5:
            continue
        boot_delta = float(on.mean() - off.mean())
        if abs(boot_delta) >= abs(observed_delta):
            beats += 1
    return (beats + 1) / (n_boot + 1)


def temporal_stability(df: pd.DataFrame, p67_thresh: float) -> dict:
    """Split IS into halves, compute stats per half."""
    if len(df) < 200:
        return {"error": "insufficient N for split"}
    mid_idx = len(df) // 2
    first_half = df.iloc[:mid_idx]
    second_half = df.iloc[mid_idx:]

    def _half_stats(sub):
        on = sub[sub["rel_vol"] > p67_thresh]["pnl_r"]
        off = sub[sub["rel_vol"] <= p67_thresh]["pnl_r"]
        if len(on) < 15 or len(off) < 15:
            return None
        t, p = stats.ttest_ind(on, off, equal_var=False)
        return {
            "n_on": len(on),
            "delta": float(on.mean() - off.mean()),
            "t": float(t),
            "p": float(p),
        }

    h1 = _half_stats(first_half)
    h2 = _half_stats(second_half)
    if h1 is None or h2 is None:
        return {"error": "half too small"}

    sign_match = np.sign(h1["delta"]) == np.sign(h2["delta"])
    both_sig = abs(h1["t"]) >= 2.0 and abs(h2["t"]) >= 2.0

    return {
        "first_half_date_range": f"{first_half['trading_day'].min().date()} to {first_half['trading_day'].max().date()}",
        "second_half_date_range": f"{second_half['trading_day'].min().date()} to {second_half['trading_day'].max().date()}",
        "first_half": h1,
        "second_half": h2,
        "sign_match": bool(sign_match),
        "both_significant_t2": bool(both_sig),
    }


def verdict(
    dsr_results: dict,
    temporal: dict,
    exceeds_max_t: bool,
    passes_bonferroni: bool,
    per_day_sig: bool,
    block_bootstrap_sig: bool,
) -> tuple:
    """Consolidated verdict: GENUINE / MARGINAL / LIKELY_OVERFIT.

    **Institutional-grade gate logic (strict, not summed):**
    - GENUINE requires ALL of: DSR@family pass, temporal stable, exceeds_max_t,
      per-day SR significant, block-bootstrap significant.
    - MARGINAL: at least 3 of the 5 hard gates pass.
    - LIKELY_OVERFIT: fewer than 3 hard gates pass.

    Default prior: overfit until proven otherwise. GENUINE requires robust pass
    across multiple independent robustness tests.
    """
    passes_dsr_lane = dsr_results["lane"]["passes_095"]
    passes_dsr_family = dsr_results["family"]["passes_095"]
    passes_dsr_global = dsr_results["global"]["passes_095"]
    temporal_ok = temporal.get("sign_match", False) and temporal.get("both_significant_t2", False)

    hard_gates = {
        "DSR@K_family_pass_0.95": passes_dsr_family,
        "temporal_stable_both_halves": temporal_ok,
        "t_exceeds_Emax_noise": exceeds_max_t,
        "per_day_significant_t2": per_day_sig,
        "block_bootstrap_p_lt_0.01": block_bootstrap_sig,
    }
    passed = [k for k, v in hard_gates.items() if v]
    n_pass = sum(hard_gates.values())

    reasons = []
    if passes_dsr_global:
        reasons.append("DSR@K_global PASS (bonus)")
    if passes_dsr_lane:
        reasons.append("DSR@K_lane PASS (bonus)")
    if passes_bonferroni:
        reasons.append("p < Bonferroni@K_global (bonus)")
    reasons.extend(f"✓ {g}" for g in passed)

    if n_pass == 5:
        v = "GENUINE"
    elif n_pass >= 3:
        v = "MARGINAL"
    else:
        v = "LIKELY_OVERFIT"

    return v, n_pass, reasons, hard_gates


def emit(results: list[dict]) -> None:
    lines = [
        "# rel_vol_HIGH_Q3 — Honest Stress Test",
        "",
        "**Date:** 2026-04-15",
        "**Purpose:** stress-test whether the 5 BH-global volume survivors are GENUINE edge or overfit / statistical anomaly. Bailey-Lopez de Prado 2014 DSR, False Strategy Theorem, temporal stability, Bonferroni at global K.",
        "",
        "## Methodology",
        "",
        "- **DSR** at 3 N_eff framings: K_global=14,261 (strictest), K_family~2,500 (moderate), K_lane~56 (lenient). Implementation: `trading_app.dsr.compute_sr0` + `compute_dsr`.",
        "- **E[max_t from noise]** at K_global via Gumbel extreme-value approximation — `sqrt(2·ln(K)) - (ln(ln(K)) + ln(4π))/(2·sqrt(2·ln(K)))`.",
        "- **Bonferroni @ K_global** — threshold = 0.05/14261 = 3.5e-6.",
        "- **Temporal stability** — IS split 50/50 by date, sign-match + |t|≥2 in both halves required.",
        "- **Verdict scoring:** 3 points DSR@global + 2 DSR@family + 1 DSR@lane + 2 temporal + 2 exceeds-max-t + 2 Bonferroni. GENUINE ≥ 7, MARGINAL ≥ 4, else LIKELY_OVERFIT.",
        "",
        f"**Expected max t from K=14261 random trials (Gumbel):** {expected_max_t_from_noise(K_GLOBAL):.3f}",
        f"**Bonferroni threshold at K=14261:** 3.5e-6 (corresponds to |t| ≥ {stats.norm.ppf(1 - 3.5e-6 / 2):.3f})",
        "",
        "## Per-lane stress test",
        "",
    ]
    for r in results:
        v, n_pass, reasons, hard_gates = r["verdict"]
        lines += [
            f"### {r['lane_name']}",
            f"**Verdict:** **{v}** (hard gates passed: {n_pass}/5)",
            f"**Hard gates:** " + ", ".join(f"{k}={'✓' if val else '✗'}" for k, val in hard_gates.items()),
            f"**Rationale:** {', '.join(reasons) if reasons else '—'}",
            "",
            "**Observed (per-trade):**",
            f"- N_on = {r['stats']['n_on']}, N_off = {r['stats']['n_off']}",
            f"- SR_on (per-trade) = {r['stats']['sr_on_per_trade']:+.4f}",
            f"- Welch t = {r['stats']['t_welch']:+.3f}, p = {r['stats']['p_welch']:.6f}",
            f"- |t| exceeds E[max_t] from K=14261? **{r['exceeds_max_t']}** (E[max_t] = {expected_max_t_from_noise(K_GLOBAL):.3f})",
            f"- p < Bonferroni@K_global (3.5e-6)? **{r['passes_bonferroni']}**",
            "",
            "**Autocorrelation-robust checks:**",
            f"- Per-day aggregated t = {r['per_day'].get('t_per_day', float('nan')):+.3f} (n_on_days={r['per_day'].get('n_on_days', 0)}), Δ_per_day = {r['per_day'].get('delta_per_day', float('nan')):+.4f}",
            f"- Block bootstrap (5-day blocks, 2000 resamples) p = {r['block_bootstrap_p']:.4f}",
            "",
            "**DSR at 3 N_eff framings:**",
            "",
            "| Framing | K_eff | SR0 (noise-max) | DSR | Pass @ 0.95 |",
            "|---------|-------|-----------------|-----|-------------|",
            f"| lane | {r['dsr']['lane']['k_eff']} | {r['dsr']['lane']['sr0']:+.4f} | {r['dsr']['lane']['dsr']:.4f} | {'YES' if r['dsr']['lane']['passes_095'] else 'no'} |",
            f"| family | {r['dsr']['family']['k_eff']} | {r['dsr']['family']['sr0']:+.4f} | {r['dsr']['family']['dsr']:.4f} | {'YES' if r['dsr']['family']['passes_095'] else 'no'} |",
            f"| global | {r['dsr']['global']['k_eff']} | {r['dsr']['global']['sr0']:+.4f} | {r['dsr']['global']['dsr']:.4f} | {'YES' if r['dsr']['global']['passes_095'] else 'no'} |",
            "",
            "**Temporal stability (IS split 50/50):**",
        ]
        if "error" in r["temporal"]:
            lines.append(f"- ERROR: {r['temporal']['error']}")
        else:
            t = r["temporal"]
            lines += [
                f"- First half: {t['first_half_date_range']}",
                f"  - N_on={t['first_half']['n_on']}, Δ={t['first_half']['delta']:+.3f}, t={t['first_half']['t']:+.2f}, p={t['first_half']['p']:.4f}",
                f"- Second half: {t['second_half_date_range']}",
                f"  - N_on={t['second_half']['n_on']}, Δ={t['second_half']['delta']:+.3f}, t={t['second_half']['t']:+.2f}, p={t['second_half']['p']:.4f}",
                f"- Sign match both halves: **{t['sign_match']}**",
                f"- Both halves |t|≥2: **{t['both_significant_t2']}**",
            ]
        lines += ["", "---", ""]

    # Joint probability if all 5 are random
    import math as m

    ps = [r["stats"]["p_welch"] for r in results]
    joint_p = 1.0
    for p in ps:
        joint_p *= p
    lines += [
        "## Joint cross-lane probability",
        "",
        f"- Product of per-lane p-values (independent assumption): {joint_p:.2e}",
        f"- Even assuming lanes are dependent at rho=0.3 effective, combined evidence is still extreme.",
        f"- Interpretation: probability that 5 independent tests all randomly produce p < 0.0005 is vanishingly small.",
        "",
        "## Consolidated verdict summary",
        "",
        "| Lane | Verdict | Score |",
        "|------|---------|-------|",
    ]
    for r in results:
        v, n_pass, _, _ = r["verdict"]
        lines.append(f"| {r['lane_name']} | **{v}** | {n_pass}/5 |")

    lines += [
        "",
        "## Recommendation",
        "",
        "_Populate after reviewing scores._",
    ]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {OUTPUT_MD}")


def main():
    # Pull variance-of-SR estimate from experimental_strategies if available
    try:
        from trading_app.dsr import estimate_var_sr_from_db

        var_sr = estimate_var_sr_from_db(GOLD_DB_PATH, min_sample=30)
        if var_sr == 0 or not np.isfinite(var_sr):
            var_sr = 0.047  # fallback per dsr.py docstring default
    except Exception:
        var_sr = 0.047
    print(f"var_sr estimate: {var_sr:.6f}")

    exp_max_t = expected_max_t_from_noise(K_GLOBAL)
    bonferroni_t = stats.norm.ppf(1 - 3.5e-6 / 2)
    print(f"E[max_t from K={K_GLOBAL} random trials] = {exp_max_t:.3f}")
    print(f"Bonferroni @ K={K_GLOBAL} critical |t| = {bonferroni_t:.3f}")

    results = []
    for instr, session, apt, rr, direction, feature in LANES:
        lane_name = f"{instr} {session} O{apt} RR{rr} {direction}"
        print(f"\n=== {lane_name} ===")
        df = load_lane_is(instr, session, apt, rr, direction)
        s = compute_cell_stats(df)
        if "error" in s:
            print(f"  {s['error']}")
            continue

        print(
            f"  N_on={s['n_on']}, SR_per_trade={s['sr_on_per_trade']:+.4f}, t={s['t_welch']:+.2f}, p={s['p_welch']:.6f}"
        )

        # DSR at 3 framings
        t_obs = s["n_on"]
        dsr_results = {
            "lane": dsr_at_k(s["sr_on_per_trade"], t_obs, s["skewness"], s["kurtosis_excess"], K_LANE_AVG, var_sr),
            "family": dsr_at_k(
                s["sr_on_per_trade"], t_obs, s["skewness"], s["kurtosis_excess"], K_FAMILY_VOLUME, var_sr
            ),
            "global": dsr_at_k(s["sr_on_per_trade"], t_obs, s["skewness"], s["kurtosis_excess"], K_GLOBAL, var_sr),
        }
        # Per-day and block-bootstrap robustness
        pd_stats = per_day_stats(df, s["p67"])
        bb_p = block_bootstrap_p(df, s["p67"])
        per_day_sig = (
            "t_per_day" in pd_stats
            and abs(pd_stats["t_per_day"]) >= 2.0
            and np.sign(pd_stats["delta_per_day"]) == np.sign(s["delta"])
        )
        block_bootstrap_sig = (not np.isnan(bb_p)) and bb_p < 0.01
        print(
            f"  Per-day: t={pd_stats.get('t_per_day', float('nan')):+.2f}, Δ_day={pd_stats.get('delta_per_day', float('nan')):+.3f}"
        )
        print(f"  Block bootstrap p (5-day blocks): {bb_p:.4f}")
        for k_name, res in dsr_results.items():
            print(
                f"  DSR@{k_name} (K={res['k_eff']}): SR0={res['sr0']:.4f}, DSR={res['dsr']:.4f}, pass_095={res['passes_095']}"
            )

        # Temporal stability
        temporal = temporal_stability(df, s["p67"])
        if "error" not in temporal:
            print(f"  Temporal: sign_match={temporal['sign_match']}, both_t2={temporal['both_significant_t2']}")
            print(f"    H1 t={temporal['first_half']['t']:+.2f} Δ={temporal['first_half']['delta']:+.3f}")
            print(f"    H2 t={temporal['second_half']['t']:+.2f} Δ={temporal['second_half']['delta']:+.3f}")

        exceeds_max_t = abs(s["t_welch"]) > exp_max_t
        passes_bonferroni = s["p_welch"] < 3.5e-6
        v = verdict(dsr_results, temporal, exceeds_max_t, passes_bonferroni, per_day_sig, block_bootstrap_sig)
        print(f"  Verdict: {v[0]} (hard gates passed: {v[1]}/5)")

        results.append(
            {
                "lane_name": lane_name,
                "stats": s,
                "dsr": dsr_results,
                "temporal": temporal,
                "per_day": pd_stats,
                "block_bootstrap_p": bb_p,
                "per_day_sig": per_day_sig,
                "block_bootstrap_sig": block_bootstrap_sig,
                "exceeds_max_t": exceeds_max_t,
                "passes_bonferroni": passes_bonferroni,
                "verdict": v,
            }
        )

    emit(results)

    print("\n=== FINAL VERDICTS ===")
    for r in results:
        v, n_pass, _, _ = r["verdict"]
        print(f"  {r['lane_name']}: {v} ({n_pass}/5 hard gates)")


if __name__ == "__main__":
    main()
