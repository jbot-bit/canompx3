"""
Pre-velocity × atr_vel_regime interaction descriptive scan.

Question
--------
Within each (instrument, session, aperture) cell, is the pre-velocity-alignment
edge (Δ_ExpR_aligned - Δ_ExpR_opposed) significantly LARGER in Expanding (high-vol)
regime than in Contracting (low-vol) regime?

Mechanism prior
---------------
Chan 2013 Ch 7 stop-cascade (verbatim p.155-156, see
docs/institutional/literature/chan_2013_ch7_intraday_momentum.md):
  "There is an additional cause of momentum that is mainly applicable to the
   short time frame: the triggering of stops. Such triggers often lead to the
   so-called breakout strategies."
  "[A]t the shortest possible time scale, the imbalance of the bid and ask
   sizes, the changes in order flow, or the aforementioned nonuniform
   distribution of stop orders can all induce momentum in prices."

Prediction: stop-cascade strength scales with stop-density per price unit;
Expanding regime increases stop-density via dealer hedging + larger ranges.
Therefore alignment edge should be GREATER in Expanding than Contracting.
Sign predicted: interaction = delta_E - delta_C > 0.

Pre-registration
----------------
Locked at docs/audit/hypotheses/2026-04-26-pre-velocity-vol-regime-interaction-descriptive.yaml
- 12 sessions × 3 instruments × 3 apertures = 108 cells max
- Power floor RULE 3.2: every sub-cell (E_aligned, E_opposed, C_aligned, C_opposed) >= 30
- BH-FDR q=0.05, K_global = number of adequately-powered cells
- Mode A IS only (trading_day < 2026-01-01)

Test stat
---------
Difference-in-differences Welch t-test (one-sided, mechanism-predicted sign):
  delta_E = mean(R | E, aligned) - mean(R | E, opposed)
  delta_C = mean(R | C, aligned) - mean(R | C, opposed)
  interaction = delta_E - delta_C
  var(interaction) = sum_g var(R_g) / n_g  (4 independent groups)
  t = interaction / sqrt(var(interaction))
  df via Welch-Satterthwaite over 4 terms
  p_oneside = 1 - cdf(t)  if t > 0, else > 0.5

Classification: descriptive read-only (RULE 10) — no writes to
experimental_strategies / validated_setups.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
from scipy import stats

from pipeline.paths import GOLD_DB_PATH

SESSIONS = (
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
)
INSTRUMENTS = ("MNQ", "MES", "MGC")
APERTURES = (5, 15, 30)

ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR_TARGET = 1.5
HOLDOUT = "2026-01-01"
MIN_N_SUBCELL = 30  # RULE 3.2 power floor per sub-cell

PREREG = "docs/audit/hypotheses/2026-04-26-pre-velocity-vol-regime-interaction-descriptive.yaml"
COMMIT_SHA_AT_LOCK = "c8129a47"


@dataclass
class CellResult:
    symbol: str
    session: str
    orb_minutes: int
    n_e_aligned: int
    n_e_opposed: int
    n_c_aligned: int
    n_c_opposed: int
    delta_e: float  # delta_ExpR in Expanding
    delta_c: float  # delta_ExpR in Contracting
    interaction: float  # delta_e - delta_c
    welch_t: float
    welch_p_oneside: float
    welch_df: float
    powered: bool
    note: str = ""


def fetch_cell_rows(
    con: duckdb.DuckDBPyConnection, symbol: str, session: str, orb_minutes: int
) -> list[tuple[float, float, float, float, str]]:
    """Returns list of (pnl_r, entry_price, stop_price, pre_velocity, regime)."""
    pv_col = f"orb_{session}_pre_velocity"
    q = f"""
    SELECT
        o.pnl_r,
        o.entry_price,
        o.stop_price,
        d.{pv_col} AS pre_velocity,
        d.atr_vel_regime AS regime
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
      AND d.{pv_col} IS NOT NULL
      AND d.atr_vel_regime IS NOT NULL
    """
    return con.execute(
        q,
        [symbol, session, orb_minutes, ENTRY_MODEL, CONFIRM_BARS, RR_TARGET, HOLDOUT],
    ).fetchall()


def classify_subcell(entry_price: float | None, stop_price: float | None, pre_vel: float | None, regime: str | None) -> str | None:
    """Return one of 'E_aligned'/'E_opposed'/'C_aligned'/'C_opposed' or None to drop."""
    if pre_vel == 0.0 or pre_vel is None:
        return None
    if regime not in ("Expanding", "Contracting"):
        return None  # Stable bucket excluded per prereg
    if entry_price is None or stop_price is None:
        return None
    is_long = entry_price > stop_price
    aligned = (pre_vel > 0 and is_long) or (pre_vel < 0 and not is_long)
    prefix = "E" if regime == "Expanding" else "C"
    return f"{prefix}_aligned" if aligned else f"{prefix}_opposed"


def diff_in_diff_welch(
    r_e_a: np.ndarray, r_e_o: np.ndarray, r_c_a: np.ndarray, r_c_o: np.ndarray
) -> tuple[float, float, float]:
    """Welch difference-of-differences t, df, one-sided p (H1: interaction > 0)."""
    means = [r.mean() for r in (r_e_a, r_e_o, r_c_a, r_c_o)]
    vars_ = [r.var(ddof=1) if r.size > 1 else float("nan") for r in (r_e_a, r_e_o, r_c_a, r_c_o)]
    ns = [r.size for r in (r_e_a, r_e_o, r_c_a, r_c_o)]
    delta_e = means[0] - means[1]
    delta_c = means[2] - means[3]
    interaction = delta_e - delta_c
    se_terms = [vars_[i] / ns[i] for i in range(4)]
    var_interact = sum(se_terms)
    if var_interact <= 0 or any(np.isnan(s) for s in se_terms):
        return float("nan"), float("nan"), float("nan")
    se = float(np.sqrt(var_interact))
    t = interaction / se
    # Welch-Satterthwaite df for 4-term composite
    df_num = var_interact**2
    df_den = sum((se_terms[i] ** 2) / max(ns[i] - 1, 1) for i in range(4))
    df = df_num / df_den if df_den > 0 else float("nan")
    p_oneside = 1.0 - stats.t.cdf(t, df) if not np.isnan(df) else float("nan")
    return float(t), float(df), float(p_oneside)


def run_cell(con, symbol: str, session: str, orb_minutes: int) -> CellResult:
    rows = fetch_cell_rows(con, symbol, session, orb_minutes)
    buckets: dict[str, list[float]] = {
        "E_aligned": [],
        "E_opposed": [],
        "C_aligned": [],
        "C_opposed": [],
    }
    for pnl_r, entry_price, stop_price, pre_vel, regime in rows:
        sub = classify_subcell(entry_price, stop_price, pre_vel, regime)
        if sub is None:
            continue
        buckets[sub].append(float(pnl_r))

    arr = {k: np.array(v, dtype=float) for k, v in buckets.items()}
    ns = {k: int(arr[k].size) for k in arr}
    powered = all(ns[k] >= MIN_N_SUBCELL for k in arr)

    if not powered:
        return CellResult(
            symbol=symbol, session=session, orb_minutes=orb_minutes,
            n_e_aligned=ns["E_aligned"], n_e_opposed=ns["E_opposed"],
            n_c_aligned=ns["C_aligned"], n_c_opposed=ns["C_opposed"],
            delta_e=float("nan"), delta_c=float("nan"),
            interaction=float("nan"), welch_t=float("nan"),
            welch_p_oneside=float("nan"), welch_df=float("nan"),
            powered=False, note=f"underpowered (min sub-cell N={min(ns.values())} < {MIN_N_SUBCELL})",
        )

    delta_e = float(arr["E_aligned"].mean() - arr["E_opposed"].mean())
    delta_c = float(arr["C_aligned"].mean() - arr["C_opposed"].mean())
    interaction = delta_e - delta_c
    t, df, p_one = diff_in_diff_welch(
        arr["E_aligned"], arr["E_opposed"], arr["C_aligned"], arr["C_opposed"]
    )
    return CellResult(
        symbol=symbol, session=session, orb_minutes=orb_minutes,
        n_e_aligned=ns["E_aligned"], n_e_opposed=ns["E_opposed"],
        n_c_aligned=ns["C_aligned"], n_c_opposed=ns["C_opposed"],
        delta_e=delta_e, delta_c=delta_c, interaction=interaction,
        welch_t=t, welch_p_oneside=p_one, welch_df=df, powered=True,
    )


def bh_fdr_oneside(results: list[CellResult], q: float = 0.05) -> list[bool]:
    """BH-FDR on one-sided p-values; returns survivor mask aligned to results order."""
    pvals = [r.welch_p_oneside for r in results]
    n = len(pvals)
    if n == 0:
        return []
    idx_sorted = sorted(range(n), key=lambda i: pvals[i] if not np.isnan(pvals[i]) else 1.0)
    thresholds = [(i + 1) / n * q for i in range(n)]
    max_k = -1
    for i, idx in enumerate(idx_sorted):
        p = pvals[idx]
        if not np.isnan(p) and p <= thresholds[i]:
            max_k = i
    survive = [False] * n
    if max_k >= 0:
        for i in range(max_k + 1):
            survive[idx_sorted[i]] = True
    return survive


def render_md(results: list[CellResult]) -> str:
    powered = [r for r in results if r.powered]
    powered_sorted = sorted(powered, key=lambda r: -r.welch_t)  # largest t first
    bh_survive = bh_fdr_oneside(powered, q=0.05)
    survivors = [r for r, s in zip(powered, bh_survive) if s]

    md = []
    md.append("# Pre-velocity × atr_vel_regime interaction — descriptive\n")
    md.append("**Date:** 2026-04-26")
    md.append(f"**Pre-reg:** `{PREREG}` (committed at `{COMMIT_SHA_AT_LOCK}`)")
    md.append("**Script:** `research/audit_pre_velocity_vol_regime_interaction.py`")
    md.append("**Classification:** DESCRIPTIVE DIAGNOSTIC (RULE 10 carve-out — read-only)\n")
    md.append("---\n")

    md.append("## Mechanism prior (locked before run)\n")
    md.append(
        "Chan 2013 Ch 7 stop-cascade (verbatim p.155-156, "
        "`docs/institutional/literature/chan_2013_ch7_intraday_momentum.md`): "
        "stop-density-driven momentum scales with regime volatility. "
        "Predicted sign: interaction = Δ_ExpR_E − Δ_ExpR_C **> 0** (one-sided).\n"
    )

    md.append("## Coverage\n")
    md.append(f"- Cells attempted: {len(SESSIONS) * len(INSTRUMENTS) * len(APERTURES)}")
    md.append(f"- Cells with results emitted: {len(results)}")
    md.append(f"- Cells adequately powered (each sub-cell N ≥ {MIN_N_SUBCELL}): **{len(powered)}**")
    md.append(f"- Cells underpowered: {len(results) - len(powered)}\n")

    md.append("## Headline\n")
    md.append(f"- Adequately-powered cells: **{len(powered)}**")
    p005 = sum(1 for r in powered if not np.isnan(r.welch_p_oneside) and r.welch_p_oneside < 0.05)
    md.append(f"- Cells with raw one-sided p<0.05: **{p005}**")
    md.append(f"- BH-FDR survivors at q=0.05, K_global={len(powered)}: **{len(survivors)}**\n")
    md.append("\n")

    if survivors:
        md.append("### Survivors (mechanism-predicted sign)\n")
        md.append("| Instrument | Session | O | t | p_one | interaction (Δ_E−Δ_C) | Δ_E | Δ_C |")
        md.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(survivors, key=lambda x: -x.welch_t):
            md.append(f"| {r.symbol} | {r.session} | {r.orb_minutes} | {r.welch_t:+.2f} | {r.welch_p_oneside:.4f} | {r.interaction:+.3f}R | {r.delta_e:+.3f}R | {r.delta_c:+.3f}R |")
        md.append("")
    else:
        md.append("### No BH-FDR survivors at K_global\n")
        md.append("**Verdict: NULL.** Pre-velocity × atr_vel_regime interaction is not detectable as a universal effect in the canonical 12 × 3 × 3 cross-section under Mode A IS. Per the prereg kill criterion, this descriptive closes the cluster.\n")

    md.append("\n## Per-instrument BH-FDR (K_family)\n")
    md.append("| Instrument | N_powered | p<0.05 (raw, one-sided) | BH-FDR survivors @ q=0.05 |")
    md.append("|---|---|---|---|")
    for inst in INSTRUMENTS:
        sub = [r for r in powered if r.symbol == inst]
        sub_p005 = sum(1 for r in sub if not np.isnan(r.welch_p_oneside) and r.welch_p_oneside < 0.05)
        sub_bh = bh_fdr_oneside(sub, q=0.05)
        sub_survive = sum(sub_bh)
        md.append(f"| {inst} | {len(sub)} | {sub_p005} | {sub_survive} |")
    md.append("")

    md.append("\n## Top |t| cells (descriptive only, no claim)\n")
    md.append("| Instrument | Session | O | N_E_a | N_E_o | N_C_a | N_C_o | Δ_E | Δ_C | interaction | t | p_one | df |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in powered_sorted[:25]:
        md.append(
            f"| {r.symbol} | {r.session} | {r.orb_minutes} | "
            f"{r.n_e_aligned} | {r.n_e_opposed} | {r.n_c_aligned} | {r.n_c_opposed} | "
            f"{r.delta_e:+.3f}R | {r.delta_c:+.3f}R | {r.interaction:+.3f}R | "
            f"{r.welch_t:+.2f} | {r.welch_p_oneside:.4f} | {r.welch_df:.0f} |"
        )
    md.append("")

    md.append("\n## Underpowered cells (RULE 3.2)\n")
    underp = [r for r in results if not r.powered]
    md.append(f"- Count: {len(underp)}")
    md.append("- Excluded from K_global pool. Listed in CSV companion.\n")

    return "\n".join(md)


def main() -> None:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    results: list[CellResult] = []
    for symbol in INSTRUMENTS:
        for session in SESSIONS:
            for orb_minutes in APERTURES:
                try:
                    r = run_cell(con, symbol, session, orb_minutes)
                    results.append(r)
                except duckdb.BinderException:
                    # Session may not have a populated pre_velocity column for this combo
                    continue
    con.close()

    out_md = Path("docs/audit/results/2026-04-26-pre-velocity-vol-regime-interaction-descriptive.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_md(results), encoding="utf-8")

    # CSV companion (full table including underpowered)
    out_csv = out_md.with_suffix(".csv")
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("instrument,session,orb_minutes,n_e_aligned,n_e_opposed,n_c_aligned,n_c_opposed,delta_e,delta_c,interaction,welch_t,welch_p_oneside,welch_df,powered,note\n")
        for r in results:
            f.write(
                f"{r.symbol},{r.session},{r.orb_minutes},"
                f"{r.n_e_aligned},{r.n_e_opposed},{r.n_c_aligned},{r.n_c_opposed},"
                f"{r.delta_e:.6f},{r.delta_c:.6f},{r.interaction:.6f},"
                f"{r.welch_t:.6f},{r.welch_p_oneside:.6f},{r.welch_df:.6f},"
                f"{r.powered},{r.note}\n"
            )

    print(f"wrote {out_md}")
    print(f"wrote {out_csv}")
    powered_n = sum(1 for r in results if r.powered)
    survivors = sum(bh_fdr_oneside([r for r in results if r.powered]))
    print(f"powered={powered_n}  bh_survivors={survivors}")


if __name__ == "__main__":
    main()
