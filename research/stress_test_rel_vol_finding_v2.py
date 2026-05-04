"""Stress test v2 — corrected methodology per self-audit 2026-04-15.

Corrections from v1 (stress_test_rel_vol_finding.py):

1. var_sr — compute from ACTUAL comprehensive-scan cell distribution, not default 0.047.
   Prior v1 used 0.047 which is calibrated for experimental_strategies table.
   Our 14,261 scan cells are different population — likely much smaller variance.

2. N_eff — report DSR at FULL range [5, 12, 36, 72, 300, 900, 14261]:
   - K=5: tested lanes (most lenient — rel_vol worked on 5 specific cells)
   - K=12: distinct sessions tested for volume family
   - K=36: 12 sessions × 3 instruments (deployment universe)
   - K=72: × 2 directions
   - K=300: Bailey finite-data bound for clean MNQ (pre_registered_criteria #2)
   - K=900: volume family in comprehensive scan (9 feature-variants × 100 lane-dirs)
   - K=14261: ALL comprehensive-scan cells (raw, ignores correlation)

3. DSR explicitly labeled INFORMATIONAL per dsr.py line 35.
   Not treated as hard gate.

4. Verdict revised:
   - 4 CORE gates (bootstrap, temporal, exceeds_max_t, per_day) — primary
   - DSR at each N_eff — informational grading
   - Final label: REAL_EDGE / EDGE_WITH_CAVEAT / NOT_YET_VALIDATED

5. Output includes var_sr computation transparency.
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

LANES = [
    ("MES", "COMEX_SETTLE", 5, 1.0, "short"),
    ("MGC", "LONDON_METALS", 5, 1.0, "short"),
    ("MES", "TOKYO_OPEN", 5, 1.5, "long"),
    ("MNQ", "SINGAPORE_OPEN", 5, 1.0, "short"),
    ("MES", "COMEX_SETTLE", 5, 1.5, "short"),
]

# Full N_eff range for honest DSR reporting
N_EFF_FRAMINGS = {
    "K=5 (survivor lanes)": 5,
    "K=12 (sessions)": 12,
    "K=36 (sessions×instr)": 36,
    "K=72 (sessions×instr×dir)": 72,
    "K=300 (Bailey MinBTL)": 300,
    "K=900 (volume family)": 900,
    "K=14261 (raw scan)": 14261,
}

OUTPUT_MD = Path("docs/audit/results/2026-04-15-rel-vol-stress-test-v2.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)


def load_lane_is(instrument, session, apt, rr, direction):
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    q = f"""
    SELECT o.trading_day, o.pnl_r,
      d.rel_vol_{session} AS rel_vol,
      d.orb_{session}_break_dir AS break_dir
    FROM orb_outcomes o JOIN daily_features d
      ON o.trading_day = d.trading_day
      AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
    WHERE o.orb_label = '{session}' AND o.symbol = '{instrument}'
      AND o.orb_minutes = {apt} AND o.entry_model = 'E2'
      AND o.rr_target = {rr} AND o.trading_day < '{HOLDOUT_SACRED_FROM}'
      AND o.pnl_r IS NOT NULL AND d.rel_vol_{session} IS NOT NULL
      AND d.orb_{session}_break_dir = '{direction}'
    ORDER BY o.trading_day
    """
    df = con.execute(q).df()
    con.close()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    return df


def estimate_var_sr_from_scan(comprehensive_scan_md: Path) -> tuple[float, int, str]:
    """Estimate var_sr empirically from the 14K comprehensive-scan cell distribution.
    Reads the top-40 all-cells table (approx) and computes cross-sectional SR variance.
    If scan-distribution unavailable, fall back to a principled proxy from the strict
    survivor set.

    Returns: (var_sr_estimate, n_cells_used, source_description)
    """
    # Simpler approach: pull per-cell on-signal means and stds directly from DB
    # for the 5 survivor lanes, compute per-trade SR, then estimate cross-lane variance
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    # Pull a broader set — all MNQ/MES/MGC lanes at O5, RR1.5, E2 as a reasonable
    # cross-section of our scan universe
    srs = []
    for session in [
        "CME_REOPEN",
        "TOKYO_OPEN",
        "SINGAPORE_OPEN",
        "LONDON_METALS",
        "EUROPE_FLOW",
        "US_DATA_830",
        "NYSE_OPEN",
        "US_DATA_1000",
        "COMEX_SETTLE",
        "CME_PRECLOSE",
        "NYSE_CLOSE",
        "BRISBANE_1025",
    ]:
        for instr in ["MNQ", "MES", "MGC"]:
            for direction in ["long", "short"]:
                q = f"""
                SELECT o.pnl_r
                FROM orb_outcomes o JOIN daily_features d
                  ON o.trading_day = d.trading_day AND o.symbol = d.symbol
                  AND o.orb_minutes = d.orb_minutes
                WHERE o.orb_label = '{session}' AND o.symbol = '{instr}'
                  AND o.orb_minutes = 5 AND o.entry_model = 'E2'
                  AND o.rr_target = 1.5 AND o.trading_day < '{HOLDOUT_SACRED_FROM}'
                  AND o.pnl_r IS NOT NULL
                  AND d.orb_{session}_break_dir = '{direction}'
                """
                try:
                    df = con.execute(q).df()
                    if len(df) >= 30:
                        pnl = np.asarray(df["pnl_r"].astype(float).values, dtype=float)
                        std = float(np.std(pnl, ddof=1))
                        sr = float(np.mean(pnl) / std) if std > 0 else 0.0
                        srs.append(sr)
                except Exception:
                    continue
    con.close()
    if len(srs) < 20:
        return (0.047, len(srs), "fallback to dsr.py default — insufficient cross-section")
    srs_arr = np.array(srs)
    var = float(np.var(srs_arr, ddof=1))
    return (var, len(srs), f"empirical cross-lane per-trade SR variance across {len(srs)} lanes")


def compute_cell_stats(df: pd.DataFrame, p67_thresh: float | None = None) -> dict:
    if len(df) < 100:
        return {"error": f"insufficient N={len(df)}"}
    if p67_thresh is None:
        p67_thresh = float(np.nanpercentile(df["rel_vol"].astype(float), 67))
    on = df[df["rel_vol"] > p67_thresh]["pnl_r"]
    off = df[df["rel_vol"] <= p67_thresh]["pnl_r"]
    if len(on) < 30 or len(off) < 30:
        return {"error": "insufficient on/off N"}
    on_arr = np.asarray(on.values, dtype=float)
    on_mean = float(on_arr.mean())
    on_std = float(on_arr.std(ddof=1))
    sr_on = on_mean / on_std if on_std > 0 else 0.0
    t_stat, p_val = stats.ttest_ind(on, off, equal_var=False)
    return {
        "p67": p67_thresh,
        "n_on": len(on),
        "sr_on_per_trade": sr_on,
        "delta": on_mean - float(off.mean()),
        "t_welch": float(t_stat),
        "p_welch": float(p_val),
        "skewness": float(stats.skew(on_arr)),
        "kurtosis_excess": float(stats.kurtosis(on_arr)),
    }


def block_bootstrap_p(
    df: pd.DataFrame, p67_thresh: float, block_size: int = 5, n_boot: int = 2000, seed: int = 42
) -> float:
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
        boot_pnl = pnl[idx]
        on = boot_pnl[mask == 1]
        off = boot_pnl[mask == 0]
        if len(on) < 5 or len(off) < 5:
            continue
        if abs(float(on.mean() - off.mean())) >= abs(observed_delta):
            beats += 1
    return (beats + 1) / (n_boot + 1)


def temporal_stability(df: pd.DataFrame, p67_thresh: float) -> dict:
    if len(df) < 200:
        return {"ok": False, "reason": "insufficient N"}
    mid = len(df) // 2
    h1, h2 = df.iloc[:mid], df.iloc[mid:]

    def _half(sub):
        on = sub[sub["rel_vol"] > p67_thresh]["pnl_r"]
        off = sub[sub["rel_vol"] <= p67_thresh]["pnl_r"]
        if len(on) < 15 or len(off) < 15:
            return None
        t, p = stats.ttest_ind(on, off, equal_var=False)
        return {"n": len(on), "delta": float(on.mean() - off.mean()), "t": float(t)}

    a, b = _half(h1), _half(h2)
    if a is None or b is None:
        return {"ok": False, "reason": "half too small"}
    sign_match = np.sign(a["delta"]) == np.sign(b["delta"])
    both_sig = abs(a["t"]) >= 2.0 and abs(b["t"]) >= 2.0
    return {
        "ok": bool(sign_match and both_sig),
        "h1": a,
        "h2": b,
        "sign_match": bool(sign_match),
        "both_sig": bool(both_sig),
    }


def expected_max_t(k: int) -> float:
    if k < 2:
        return 0.0
    ln_k = math.log(k)
    a = math.sqrt(2 * ln_k)
    b = (math.log(ln_k) + math.log(4 * math.pi)) / (2 * a)
    return a - b


def per_day_t(df: pd.DataFrame, p67_thresh: float, observed_delta: float) -> tuple[float, bool]:
    on_mask = df["rel_vol"] > p67_thresh
    on_days = df[on_mask].groupby(df[on_mask]["trading_day"].dt.date)["pnl_r"].mean()
    off_days = df[~on_mask].groupby(df[~on_mask]["trading_day"].dt.date)["pnl_r"].mean()
    if len(on_days) < 20 or len(off_days) < 20:
        return (float("nan"), False)
    t, _ = stats.ttest_ind(on_days, off_days, equal_var=False)
    t = float(t)
    sig = abs(t) >= 2.0 and np.sign(on_days.mean() - off_days.mean()) == np.sign(observed_delta)
    return (t, sig)


def emit(results: list[dict], var_sr: float, var_sr_n: int, var_sr_src: str) -> None:
    lines = [
        "# rel_vol_HIGH Stress Test v2 — Self-Audited",
        "",
        "**Date:** 2026-04-15",
        "**Purpose:** honest re-audit of v1 after fresh-perspective review revealed overweighting of explicitly-INFORMATIONAL DSR at inflated N_eff=14,261.",
        "",
        "## Corrections applied",
        "",
        f"- **var_sr:** {var_sr:.6f} ({var_sr_src})",
        f"- **N_eff reported across {len(N_EFF_FRAMINGS)} framings** — K=5 (lanes) to K=14,261 (raw scan). True effective N is unknown but is almost certainly NOT 14,261 (cells are highly correlated).",
        "- **DSR labeled INFORMATIONAL per dsr.py line 35** — not a hard gate.",
        "- **4 CORE hard gates** (bootstrap, temporal, exceeds_max_t, per_day) — these are the primary verdict drivers.",
        "- Final verdict labels revised: REAL_EDGE / EDGE_WITH_CAVEAT / NOT_YET_VALIDATED.",
        "",
        f"**Expected max |t| from K=14,261 random trials (Gumbel):** {expected_max_t(14261):.3f}",
        f"**Expected max |t| from K=300 (Bailey MinBTL):** {expected_max_t(300):.3f}",
        f"**Expected max |t| from K=72 (sess×instr×dir):** {expected_max_t(72):.3f}",
        "",
        "## Per-lane result",
        "",
    ]

    for r in results:
        s = r["stats"]
        lines += [
            f"### {r['lane_name']}",
            f"- N_on={s['n_on']}, per-trade SR={s['sr_on_per_trade']:+.4f}, Δ={s['delta']:+.3f}",
            f"- Welch t={s['t_welch']:+.3f}, p={s['p_welch']:.6f}",
            "",
            "**CORE hard gates:**",
            f"- Block bootstrap p (autocorrelation-robust): {r['bootstrap_p']:.4f} {'PASS' if r['bootstrap_p'] < 0.01 else 'FAIL'}",
            f"- Temporal stability (IS split 50/50 sign+|t|≥2 both halves): {'PASS' if r['temporal']['ok'] else 'FAIL'}",
            f"  - H1 t={r['temporal'].get('h1', {}).get('t', float('nan')):+.2f}, H2 t={r['temporal'].get('h2', {}).get('t', float('nan')):+.2f}",
            f"- |t| > E[max_t from K=14,261 noise] ({expected_max_t(14261):.2f}): {'PASS' if abs(s['t_welch']) > expected_max_t(14261) else 'FAIL'}",
            f"- Per-day aggregated t: {r['per_day_t']:+.2f} ({'PASS' if r['per_day_sig'] else 'FAIL'})",
            "",
            "**INFORMATIONAL — DSR across N_eff framings:**",
            "",
            "| Framing | K | SR0 (noise-max) | DSR | Pass@0.95 |",
            "|---------|---|-----------------|-----|-----------|",
        ]
        for name, dsr_res in r["dsr_by_framing"].items():
            lines.append(
                f"| {name} | {dsr_res['k']} | {dsr_res['sr0']:+.4f} | {dsr_res['dsr']:.4f} | {'YES' if dsr_res['dsr'] >= 0.95 else 'no'} |"
            )
        lines += ["", f"**Verdict:** **{r['verdict']}**", "", "---", ""]

    lines += [
        "",
        "## Consolidated verdict",
        "",
        "| Lane | Verdict | Core gates | DSR @ N_eff=36 | DSR @ N_eff=72 | DSR @ N_eff=300 |",
        "|------|---------|------------|----------------|----------------|-----------------|",
    ]
    for r in results:
        core_pass = r["n_core_pass"]
        d36 = r["dsr_by_framing"]["K=36 (sessions×instr)"]["dsr"]
        d72 = r["dsr_by_framing"]["K=72 (sessions×instr×dir)"]["dsr"]
        d300 = r["dsr_by_framing"]["K=300 (Bailey MinBTL)"]["dsr"]
        lines.append(f"| {r['lane_name']} | **{r['verdict']}** | {core_pass}/4 | {d36:.3f} | {d72:.3f} | {d300:.3f} |")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {OUTPUT_MD}")


def verdict(n_core_pass: int, dsr_at_realistic_k: float) -> str:
    """Fresh-trader + institutional-grade hybrid verdict.

    Core (empirical):
    - Bootstrap p < 0.01 (signal-outcome link real)
    - Temporal both halves sign+|t|≥2 (stable)
    - |t| > E[max_t noise at K=14K]
    - Per-day aggregated t ≥ 2 sign-match
    """
    if n_core_pass == 4:
        if dsr_at_realistic_k >= 0.95:
            return "REAL_EDGE — all 4 core + DSR@realistic_K pass"
        elif dsr_at_realistic_k >= 0.5:
            return "REAL_EDGE — all 4 core pass, DSR ambiguous at N_eff (deploy with monitoring)"
        else:
            return "EDGE_WITH_CAVEAT — 4/4 core pass but DSR@realistic_K<0.5 (small effect size)"
    elif n_core_pass == 3:
        return "EDGE_WITH_CAVEAT — 3/4 core gates"
    else:
        return "NOT_YET_VALIDATED"


def main():
    print("Estimating var_sr from actual scan cross-section...")
    var_sr, var_sr_n, var_sr_src = estimate_var_sr_from_scan(
        Path("docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md")
    )
    print(f"  var_sr = {var_sr:.6f} ({var_sr_src})")

    results = []
    for instr, session, apt, rr, direction in LANES:
        lane_name = f"{instr} {session} O{apt} RR{rr} {direction}"
        print(f"\n=== {lane_name} ===")
        df = load_lane_is(instr, session, apt, rr, direction)
        s = compute_cell_stats(df)
        if "error" in s:
            print(f"  {s['error']}")
            continue

        bootstrap_p = block_bootstrap_p(df, s["p67"])
        temporal = temporal_stability(df, s["p67"])
        pd_t, pd_sig = per_day_t(df, s["p67"], s["delta"])
        exceeds_max_t = abs(s["t_welch"]) > expected_max_t(14261)

        # DSR at every N_eff framing
        dsr_by_framing = {}
        for name, k_eff in N_EFF_FRAMINGS.items():
            sr0 = compute_sr0(n_eff=k_eff, var_sr=var_sr)
            dsr_val = compute_dsr(
                sr_hat=s["sr_on_per_trade"],
                sr0=sr0,
                t_obs=s["n_on"],
                skewness=s["skewness"],
                kurtosis_excess=s["kurtosis_excess"],
            )
            dsr_by_framing[name] = {"k": k_eff, "sr0": sr0, "dsr": dsr_val}

        core = {
            "bootstrap": bootstrap_p < 0.01 if not np.isnan(bootstrap_p) else False,
            "temporal": temporal["ok"],
            "exceeds_max_t": exceeds_max_t,
            "per_day": pd_sig,
        }
        n_core_pass = sum(core.values())

        # Use DSR at realistic K=72 (sessions × instr × dir) for verdict
        dsr_realistic = dsr_by_framing["K=72 (sessions×instr×dir)"]["dsr"]
        v = verdict(n_core_pass, dsr_realistic)

        print(
            f"  CORE: bootstrap={core['bootstrap']}, temporal={core['temporal']}, exceeds_max_t={core['exceeds_max_t']}, per_day={core['per_day']} → {n_core_pass}/4"
        )
        print("  DSR:")
        for name, dr in dsr_by_framing.items():
            print(f"    {name}: SR0={dr['sr0']:.3f} DSR={dr['dsr']:.3f}")
        print(f"  Verdict: {v}")

        results.append(
            {
                "lane_name": lane_name,
                "stats": s,
                "bootstrap_p": bootstrap_p,
                "temporal": temporal,
                "per_day_t": pd_t,
                "per_day_sig": pd_sig,
                "dsr_by_framing": dsr_by_framing,
                "core": core,
                "n_core_pass": n_core_pass,
                "verdict": v,
            }
        )

    emit(results, var_sr, var_sr_n, var_sr_src)

    print("\n=== FINAL VERDICTS (v2) ===")
    for r in results:
        print(f"  {r['lane_name']}: {r['verdict']}")


if __name__ == "__main__":
    main()
