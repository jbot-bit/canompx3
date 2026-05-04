"""PR #51 5 CANDIDATE_READY cells — Deflated Sharpe Ratio audit (Bailey-LdP 2014 Eq. 2).

PR #51 `2ed62dc3` locked 5 CANDIDATE_READY cells on the MNQ unfiltered-baseline
cross-family (K=105). Phase 0 pre_registered_criteria.md requires 12 criteria;
PR #51 verified H1 / C6 / C8 / C9 but did NOT compute C5 (DSR >= 0.95).

This is a confirmatory audit, not new discovery — no new pre-reg per
research-truth-protocol.md § 10. Computes DSR per Bailey-López de Prado 2014
Eq. 2 for each of the 5 cells against the full 105-trial family variance.

Grounding:
- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`
- Phase 0 C5 threshold: DSR >= 0.95 (pre_registered_criteria.md)

No capital action.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from math import e as EULER_E
from pathlib import Path

import duckdb
import numpy as np
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

RESULT_DOC = Path("docs/audit/results/2026-04-21-mnq-pr51-dsr-audit-v1.md")
INSTRUMENT = "MNQ"
APERTURES = [5, 15, 30]
RRS = [1.0, 1.5, 2.0]
MIN_N_IS = 100
DSR_THRESHOLD = 0.95

# PR #51 CANDIDATE_READY cells (apt, rr, session)
PR51_CANDIDATES = [
    (5, 1.0, "NYSE_OPEN"),
    (5, 1.5, "NYSE_OPEN"),
    (15, 1.0, "NYSE_OPEN"),
    (15, 1.0, "US_DATA_1000"),
    (15, 1.5, "US_DATA_1000"),
]

EULER_GAMMA = 0.5772156649


@dataclass
class CellStats:
    aperture: int
    rr: float
    session: str
    n_is: int
    mean_pnl_r: float
    std_pnl_r: float
    sr_trade: float  # trade-level Sharpe (non-annualized)
    skewness: float
    kurtosis: float  # Fisher kurtosis + 3 = Pearson (Bailey uses γ₄ Pearson)


def _list_sessions(con: duckdb.DuckDBPyConnection, orb_minutes: int) -> list[str]:
    rows = con.execute(
        """
        SELECT DISTINCT orb_label FROM orb_outcomes
        WHERE symbol = ? AND orb_minutes = ? AND entry_model = 'E2'
          AND confirm_bars = 1 AND pnl_r IS NOT NULL
        ORDER BY orb_label
        """,
        [INSTRUMENT, orb_minutes],
    ).fetchall()
    return [r[0] for r in rows]


def _load_cell_is(con: duckdb.DuckDBPyConnection, orb_minutes: int, rr: float, session: str) -> np.ndarray:
    sql = """
    SELECT pnl_r
    FROM orb_outcomes
    WHERE symbol = ? AND orb_label = ? AND orb_minutes = ?
      AND entry_model = 'E2' AND confirm_bars = 1 AND rr_target = ?
      AND pnl_r IS NOT NULL
      AND trading_day < ?
    """
    rows = con.execute(sql, [INSTRUMENT, session, orb_minutes, rr, HOLDOUT_SACRED_FROM]).fetchall()
    return np.array([r[0] for r in rows], dtype=float)


def _cell_stats(apt: int, rr: float, session: str, vals: np.ndarray) -> CellStats | None:
    if len(vals) < MIN_N_IS:
        return None
    m = float(vals.mean())
    s = float(vals.std(ddof=1))
    if s == 0:
        return None
    sr = m / s
    # scipy.stats.skew default is Fisher-Pearson (bias-adjusted). Bailey uses γ̂₃ skewness estimator.
    skew = float(stats.skew(vals, bias=False))
    # Bailey γ̂₄ is Pearson (non-excess) kurtosis. scipy.stats.kurtosis default is FISHER (excess),
    # set fisher=False to get Pearson (non-excess) kurtosis.
    kurt = float(stats.kurtosis(vals, fisher=False, bias=False))
    return CellStats(
        aperture=apt,
        rr=rr,
        session=session,
        n_is=len(vals),
        mean_pnl_r=m,
        std_pnl_r=s,
        sr_trade=sr,
        skewness=skew,
        kurtosis=kurt,
    )


def _bailey_sr0(v_sr: float, n_trials: int) -> float:
    """SR_0 rejection threshold per Bailey-LdP 2014 Eq. 1 footnote:
    SR_0 = sqrt(V[SR]) * ((1-γ)·Z⁻¹[1 - 1/N] + γ·Z⁻¹[1 - 1/(N·e)])
    """
    z_inv_1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z_inv_2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * EULER_E))
    return float(np.sqrt(v_sr) * ((1.0 - EULER_GAMMA) * z_inv_1 + EULER_GAMMA * z_inv_2))


def _dsr(sr: float, sr_0: float, T: int, skew: float, kurt: float) -> float:
    """Bailey-LdP 2014 Eq. 2: DSR = Z[((SR - SR_0)·sqrt(T-1)) / sqrt(1 - γ₃·SR + (γ₄-1)/4·SR²)]"""
    denom_sq = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr**2)
    if denom_sq <= 0:
        return float("nan")
    denom = np.sqrt(denom_sq)
    z_stat = ((sr - sr_0) * np.sqrt(T - 1)) / denom
    return float(stats.norm.cdf(z_stat))


def _sanity_bailey_worked_example() -> None:
    """Replicate Bailey-LdP 2014 worked example p 9-10 to validate implementation.

    Paper inputs: N=100, V[SR]=1/2 (annualized), T=1250, SR_annual=2.5, γ̂₃=-3, γ̂₄=10.
    Paper-reported: SR_0 ≈ 0.1132 (non-annualized, 250 obs/yr), DSR ≈ 0.9004.
    Non-annualized SR = 2.5/sqrt(250); V[SR] non-annualized = (1/2) / 250.
    """
    N = 100
    v_sr_ann = 0.5
    v_sr_daily = v_sr_ann / 250.0
    sr_ann = 2.5
    sr_daily = sr_ann / np.sqrt(250.0)
    T = 1250
    skew = -3.0
    kurt = 10.0  # Pearson (non-excess)

    sr_0 = _bailey_sr0(v_sr_daily, N)
    dsr = _dsr(sr_daily, sr_0, T, skew, kurt)
    print("[sanity] Bailey-LdP 2014 worked example (p 9-10):")
    print(f"  SR_0 computed = {sr_0:.4f} (paper says ~0.1132)")
    print(f"  DSR computed  = {dsr:.4f} (paper says ~0.9004)")
    ok_sr0 = abs(sr_0 - 0.1132) < 0.005
    ok_dsr = abs(dsr - 0.9004) < 0.01
    if not (ok_sr0 and ok_dsr):
        raise SystemExit(
            f"Sanity check FAILED — implementation diverges from Bailey paper. "
            f"SR_0 delta={abs(sr_0 - 0.1132):.4f}, DSR delta={abs(dsr - 0.9004):.4f}"
        )
    print("  [PASS] implementation matches paper within tolerance\n")


def main() -> int:
    _sanity_bailey_worked_example()

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    all_cells: list[CellStats] = []
    candidate_cells: dict[tuple[int, float, str], CellStats] = {}
    try:
        for apt in APERTURES:
            sessions = _list_sessions(con, apt)
            for rr in RRS:
                for s in sessions:
                    vals = _load_cell_is(con, apt, rr, s)
                    stat = _cell_stats(apt, rr, s, vals)
                    if stat is not None:
                        all_cells.append(stat)
                        if (apt, rr, s) in PR51_CANDIDATES:
                            candidate_cells[(apt, rr, s)] = stat
    finally:
        con.close()

    # Family: only cells meeting N_IS >= 100 (matches PR #51 K_family inclusion rule)
    n_family = len(all_cells)
    print(f"Family K = {n_family} cells (MNQ unfiltered baseline, N_IS >= {MIN_N_IS})")
    if n_family != 105:
        print(
            f"  WARNING: reproduced family K = {n_family}; PR #51 reported K=105. "
            "Small divergence acceptable (db refresh); audit continues."
        )

    # V[{SR_n}] across the family (bias-corrected)
    sr_array = np.array([c.sr_trade for c in all_cells], dtype=float)
    v_sr = float(sr_array.var(ddof=1))
    mean_sr = float(sr_array.mean())
    sr_0 = _bailey_sr0(v_sr, n_family)
    print(f"Family SR mean  = {mean_sr:+.5f}")
    print(f"Family V[SR]    = {v_sr:.5f}")
    print(f"Family SR_0     = {sr_0:+.5f} (DSR rejection threshold, trade-level)")
    print("")

    # Per-cell DSR for the 5 PR #51 CANDIDATE_READYs
    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    parts.append("# MNQ PR #51 5 CANDIDATE_READY cells — Deflated Sharpe Ratio audit v1\n")
    parts.append(
        "**Authority:** Bailey-López de Prado 2014 Eq. 2 "
        "(`docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`).\n"
    )
    parts.append(f"**Phase 0 C5 threshold:** DSR >= {DSR_THRESHOLD}.\n")
    parts.append(
        "**Confirmatory audit** (no new discovery, no pre-reg required per research-truth-protocol.md § 10).\n"
    )
    parts.append("## Family context")
    parts.append("")
    parts.append(f"- Family (re-derived from PR #51 axes): **K = {n_family}** cells with N_IS >= {MIN_N_IS}")
    parts.append(f"- Family mean trade SR: `{mean_sr:+.5f}`")
    parts.append(f"- Family variance V[SR]: `{v_sr:.5f}`")
    parts.append(f"- Bailey SR_0 rejection threshold (trade-level): `{sr_0:+.5f}`")
    parts.append("")
    parts.append("## DSR per CANDIDATE_READY cell")
    parts.append("")
    lines = [
        "| Apt | RR | Session | N_IS | Trade SR | Skew (γ₃) | Kurt (γ₄ Pearson) | DSR | Phase 0 C5 |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]

    results: list[tuple[tuple[int, float, str], float, bool]] = []
    for key in PR51_CANDIDATES:
        stat = candidate_cells.get(key)
        if stat is None:
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | - | - | - | - | - | MISSING |")
            continue
        dsr_val = _dsr(stat.sr_trade, sr_0, stat.n_is, stat.skewness, stat.kurtosis)
        verdict = "PASS" if dsr_val >= DSR_THRESHOLD else "FAIL"
        results.append((key, dsr_val, verdict == "PASS"))
        lines.append(
            f"| {stat.aperture} | {stat.rr} | {stat.session} | {stat.n_is} | "
            f"{stat.sr_trade:+.5f} | {stat.skewness:+.3f} | {stat.kurtosis:.3f} | "
            f"{dsr_val:.4f} | **{verdict}** |"
        )
    parts.append("\n".join(lines))
    parts.append("")

    n_pass = sum(1 for _, _, p in results if p)
    n_fail = len(results) - n_pass
    parts.append("## Summary")
    parts.append("")
    parts.append(f"- CANDIDATEs tested: {len(results)}")
    parts.append(f"- DSR PASS (>= {DSR_THRESHOLD}): **{n_pass}**")
    parts.append(f"- DSR FAIL (< {DSR_THRESHOLD}): {n_fail}")
    parts.append("")
    parts.append("## Interpretation")
    parts.append("")
    if n_pass == len(results):
        parts.append(
            f"All {len(results)} PR #51 CANDIDATE_READY cells pass Phase 0 C5 (DSR >= {DSR_THRESHOLD}). "
            "Claim strengthened: the 5 cells clear every Phase 0 criterion that can be computed "
            "without 2026 OOS maturity + Monte Carlo account survival + live Shiryaev-Roberts drift. "
            "Shadow-deployment design is the next bounded step."
        )
    elif n_pass == 0:
        parts.append(
            f"NONE of the {len(results)} PR #51 CANDIDATE_READY cells pass Phase 0 C5 "
            f"(DSR < {DSR_THRESHOLD}). These cells passed H1/C6/C8/C9 but fail C5 — Bailey-LdP "
            "2014 DSR corrects for family selection bias + non-normality. Re-classify as "
            "RESEARCH_SURVIVOR pending C5. Deployment paused."
        )
    else:
        parts.append(
            f"{n_pass} of {len(results)} PR #51 CANDIDATE_READY cells pass Phase 0 C5 "
            f"(DSR >= {DSR_THRESHOLD}); {n_fail} fail. Only the passing cells remain fully "
            "Phase 0-compliant. Failing cells must be re-classified as RESEARCH_SURVIVOR."
        )
    parts.append("")
    parts.append("## Methodology notes")
    parts.append("")
    parts.append(
        "- Trade-level SR (not annualized). Bailey Eq. 2 is scale-invariant — SR and SR_0 in the "
        "same units cancel, so trade-level vs daily-level does not change DSR as long as both "
        "sides use the same convention."
    )
    parts.append(
        "- Skewness via `scipy.stats.skew(bias=False)` — the bias-adjusted Fisher-Pearson estimator, matches Bailey γ̂₃."
    )
    parts.append(
        "- Kurtosis via `scipy.stats.kurtosis(fisher=False, bias=False)` — Pearson (non-excess) "
        "kurtosis, matches Bailey γ̂₄."
    )
    parts.append("- Family V[SR] computed with ddof=1 (bias-corrected sample variance).")
    parts.append(
        "- Independence assumption: treats the 105 cells as independent trials. This is "
        "conservative — per Bailey Exhibit 4, correlated trials yield a smaller effective N, "
        "which LOWERS SR_0, which RAISES DSR. If DSR passes at N=105 independent, it passes "
        "at any smaller effective N. If DSR fails at N=105, an independence correction would "
        "not rescue it (effective N would still be <= 105)."
    )
    parts.append(
        "- Sanity check: Bailey-LdP 2014 worked example (pp 9-10) reproduced in `_sanity_bailey_worked_example()`."
    )
    parts.append("")
    parts.append("## Not done by this result")
    parts.append("")
    parts.append("- No writes to validated_setups / edge_families / lane_allocation / live_config.")
    parts.append("- No deployment or capital action.")
    parts.append("- Does NOT compute C11 (90-day account-death Monte Carlo) or C12 (live Shiryaev-Roberts).")
    parts.append("- Does NOT re-test MES/MGC (PR #53 + PR #55 canonical for those).")
    parts.append("")
    parts.append("## Canonical run output")
    parts.append("")
    parts.append("See terminal output on commit; all per-cell stats logged.")

    RESULT_DOC.write_text("\n".join(parts), encoding="utf-8")

    print(f"CANDIDATEs tested: {len(results)}")
    print(f"DSR PASS: {n_pass}")
    print(f"DSR FAIL: {n_fail}")
    print(f"RESULT_DOC: {RESULT_DOC}")
    print("\nPer-cell DSR:")
    for key, dsr_val, passed in results:
        tag = "PASS" if passed else "FAIL"
        print(f"  MNQ {key[0]}m RR={key[1]} {key[2]}: DSR={dsr_val:.4f} [{tag}]")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
