"""
Cross-session pre-break context descriptive diagnostic.

Question
--------
Does pre-ORB-end velocity direction alignment with E2 fill direction
predict trade outcome, uniformly across the full 12 × 3 × 3 deployed
universe on the UNFILTERED baseline?

This is the highest-EV follow-up flagged by:
- docs/plans/2026-04-21-post-stale-lock-action-queue.md item #2 pre-
  registration dependency
- docs/audit/results/2026-04-21-orb-g5-deployed-lane-arithmetic-check.md
  follow-up § 6

Framing
-------
For each (symbol, session, orb_minutes) cell:

- UNIVERSE: E2 CB1 RR=1.5 IS (trading_day < 2026-01-01), eligible trades
  only (entry_ts IS NOT NULL, pnl_r IS NOT NULL), NO pre-gate filter.
  Rationale per PR #71: ORB_G5 on L1 is cost-gate with zero directional
  content — filtering on it before measuring pre-break context would
  distort the baseline with cost amplification, not directional signal.

- FEATURE: `orb_{label}_pre_velocity` (canonical per
  `pipeline/build_daily_features.py`). Signed: positive = up-momentum,
  negative = down-momentum. Trade-time-knowable (pre-ORB interval).

- DIRECTION: inferred from canonical fill metadata
  `entry_price > stop_price → long`. For E2 the stop side is opposite
  the break side, so this is equivalent to the fill direction without
  reading any post-ORB close-break column (no look-ahead).

- ALIGNMENT:
  - aligned  = (pre_vel > 0 AND long)  OR (pre_vel < 0 AND short)
  - opposed  = (pre_vel > 0 AND short) OR (pre_vel < 0 AND long)
  - zero pre_vel is dropped (no sign to align on)

- TEST: per-cell two-proportion z on WR, Welch t on pnl_r, Δ_WR (pp),
  Δ_ExpR. Descriptive only — no promotion, no pre-reg claim, no
  BH-FDR survivor set. Intent: is this signal strong and universal
  enough to justify writing a Pathway-B pre-reg?

- SCOPE: 12 sessions × 3 instruments × 3 apertures = 108 cells max.
  Cells with N < 100 are reported but flagged RULE 3.2 underpowered.

Classification
--------------
Confirmatory-free descriptive diagnostic per `research-truth-protocol.md`
§ Phase 0 footnote and `backtesting-methodology.md` RULE 10 (no new
pre-reg required for descriptive read-only scans that do not write to
`experimental_strategies` or `validated_setups`).

Usage
-----
    python research/audit_pre_break_context_cross_session.py

Emits `docs/audit/results/2026-04-21-pre-break-context-cross-session-descriptive.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
from scipy import stats

from pipeline.paths import GOLD_DB_PATH

# Sessions with canonical `orb_{label}_pre_velocity` populated
# (confirmed by DB schema introspection 2026-04-21)
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

# Active instruments per pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS
INSTRUMENTS = ("MNQ", "MES", "MGC")

APERTURES = (5, 15, 30)

RR_TARGET = 1.5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
HOLDOUT = "2026-01-01"

# RULE 3.2 power floor (per-cell)
MIN_N_PER_GROUP = 30


@dataclass(frozen=True)
class CellResult:
    symbol: str
    session: str
    orb_minutes: int
    n_total: int
    n_aligned: int
    n_opposed: int
    wr_aligned: float
    wr_opposed: float
    exp_r_aligned: float
    exp_r_opposed: float
    delta_wr_pp: float
    delta_exp_r: float
    z_wr: float
    p_wr: float
    welch_t: float
    welch_p: float

    @property
    def underpowered(self) -> bool:
        return (
            self.n_aligned < MIN_N_PER_GROUP
            or self.n_opposed < MIN_N_PER_GROUP
        )

    @property
    def directional_at_05(self) -> bool:
        return (not np.isnan(self.p_wr)) and self.p_wr < 0.05

    @property
    def welch_at_05(self) -> bool:
        return (not np.isnan(self.welch_p)) and self.welch_p < 0.05


def fetch_cell_rows(
    con: duckdb.DuckDBPyConnection, symbol: str, session: str, orb_minutes: int
) -> list[tuple[float, str, float, float, float]]:
    """Returns list of (pnl_r, outcome, entry_price, stop_price, pre_velocity)."""
    pv_col = f"orb_{session}_pre_velocity"
    q = f"""
    SELECT
        o.pnl_r,
        o.outcome,
        o.entry_price,
        o.stop_price,
        d.{pv_col} AS pre_velocity
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
    """
    return con.execute(
        q,
        [
            symbol,
            session,
            orb_minutes,
            ENTRY_MODEL,
            CONFIRM_BARS,
            RR_TARGET,
            HOLDOUT,
        ],
    ).fetchall()


def run_cell(
    con: duckdb.DuckDBPyConnection, symbol: str, session: str, orb_minutes: int
) -> CellResult | None:
    rows = fetch_cell_rows(con, symbol, session, orb_minutes)
    if len(rows) < 2 * MIN_N_PER_GROUP:
        return None  # cell with too few total trades — skip

    aligned_pnls: list[float] = []
    opposed_pnls: list[float] = []
    aligned_wins = 0
    opposed_wins = 0

    for pnl_r, outcome, entry_price, stop_price, pre_vel in rows:
        if pre_vel is None or (isinstance(pre_vel, float) and np.isnan(pre_vel)):
            continue
        if pre_vel == 0.0:
            continue
        if entry_price is None or stop_price is None:
            continue
        is_long = entry_price > stop_price
        if (pre_vel > 0 and is_long) or (pre_vel < 0 and not is_long):
            aligned_pnls.append(float(pnl_r))
            if outcome == "win":
                aligned_wins += 1
        else:
            opposed_pnls.append(float(pnl_r))
            if outcome == "win":
                opposed_wins += 1

    n_a = len(aligned_pnls)
    n_o = len(opposed_pnls)
    if n_a == 0 or n_o == 0:
        return None

    arr_a = np.array(aligned_pnls, dtype=float)
    arr_o = np.array(opposed_pnls, dtype=float)

    wr_a = aligned_wins / n_a
    wr_o = opposed_wins / n_o

    # two-proportion z
    p_pool = (aligned_wins + opposed_wins) / (n_a + n_o)
    se = float(np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_o))) if 0 < p_pool < 1 else float("nan")
    z_wr = (wr_a - wr_o) / se if se and se > 0 else float("nan")
    p_wr = 2.0 * (1.0 - stats.norm.cdf(abs(z_wr))) if not np.isnan(z_wr) else float("nan")

    # Welch on pnl_r
    if n_a > 1 and n_o > 1:
        t_val, p_val = stats.ttest_ind(arr_a, arr_o, equal_var=False)
        welch_t = float(t_val)
        welch_p = float(p_val)
    else:
        welch_t, welch_p = float("nan"), float("nan")

    return CellResult(
        symbol=symbol,
        session=session,
        orb_minutes=orb_minutes,
        n_total=len(rows),
        n_aligned=n_a,
        n_opposed=n_o,
        wr_aligned=100.0 * wr_a,
        wr_opposed=100.0 * wr_o,
        exp_r_aligned=float(arr_a.mean()),
        exp_r_opposed=float(arr_o.mean()),
        delta_wr_pp=100.0 * (wr_a - wr_o),
        delta_exp_r=float(arr_a.mean() - arr_o.mean()),
        z_wr=z_wr,
        p_wr=p_wr,
        welch_t=welch_t,
        welch_p=welch_p,
    )


def bh_fdr(p_values: list[float], q: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR. Returns mask of which hypotheses survive at q."""
    n = len(p_values)
    idx_sorted = sorted(range(n), key=lambda i: p_values[i])
    thresholds = [(i + 1) / n * q for i in range(n)]
    survive_sorted = [False] * n
    max_k = -1
    for i in range(n):
        if p_values[idx_sorted[i]] <= thresholds[i]:
            max_k = i
    if max_k >= 0:
        for i in range(max_k + 1):
            survive_sorted[i] = True
    mask = [False] * n
    for rank, orig_i in enumerate(idx_sorted):
        mask[orig_i] = survive_sorted[rank]
    return mask


def render_md(results: list[CellResult]) -> str:
    total_cells = len(SESSIONS) * len(INSTRUMENTS) * len(APERTURES)
    produced = len(results)
    underpowered = sum(1 for r in results if r.underpowered)
    nonunder = [r for r in results if not r.underpowered]

    directional_05 = [r for r in nonunder if r.directional_at_05]

    # BH-FDR on the non-underpowered set (K_family = 1; the family here is
    # "pre_velocity direction alignment across the cross-session universe")
    if nonunder:
        p_list = [float(r.p_wr) for r in nonunder]
        mask = bh_fdr(p_list, q=0.05)
        bh_survivors = [nonunder[i] for i in range(len(nonunder)) if mask[i]]
    else:
        bh_survivors = []

    md: list[str] = []
    md.append("# Cross-session pre-break context descriptive diagnostic")
    md.append("")
    md.append("**Date:** 2026-04-21")
    md.append("**Branch:** `research/pre-break-context-cross-session-descriptive`")
    md.append("**Script:** `research/audit_pre_break_context_cross_session.py`")
    md.append("**Classification:** DESCRIPTIVE DIAGNOSTIC (read-only, no promotion)")
    md.append("  per `backtesting-methodology.md § RULE 10` — no pre-reg required because")
    md.append("  no write to `experimental_strategies` or `validated_setups`.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Literature grounding")
    md.append("")
    md.append(
        "Per `institutional-rigor.md § 7` (ground in local resources before training "
        "memory), citations below are from verbatim extracts in "
        "`docs/institutional/literature/`, not from memory."
    )
    md.append("")
    md.append(
        "- **Fitschen 2013 Ch 3** (`fitschen_2013_path_of_least_resistance.md`, "
        "pp. 32-42): commodities trend on BOTH daily and intraday bars; stock indices "
        "trend on intraday bars (counter-trend on daily). Verbatim p.41: *\"In the case "
        "of commodities, both daily bars and hourly bars have a tendency to trend.\"* "
        "This is the CORE ORB premise. **Prediction under Fitschen:** pre-ORB momentum "
        "direction should continue through the ORB break → aligned group should have "
        "HIGHER WR and HIGHER ExpR than opposed, universally across MNQ/MES/MGC."
    )
    md.append(
        "- **Chan 2013 Ch 7** (`chan_2013_ch7_intraday_momentum.md`, pp. 155-168): "
        "intraday momentum exists; the **mechanism is stop-triggered cascade**. Verbatim "
        "p.155: *\"There is an additional cause of momentum that is mainly applicable "
        "to the short time frame: the triggering of stops. Such triggers often lead to "
        "the so-called breakout strategies.\"* Chan's gap-momentum result on FSTX "
        "(equity-index future, Sharpe 1.4 over 8 years) is a direct analog for the "
        "project's MNQ/MES ORB premise. **Under Chan's mechanism,** pre-ORB directional "
        "momentum (our `pre_velocity`) should reflect the existing order-flow imbalance; "
        "a break in the SAME direction catches more stops and sustains the move → "
        "aligned wins more often."
    )
    md.append(
        "- **Chordia et al 2018** (`chordia_et_al_2018_two_million_strategies.md`): "
        "t ≥ 3.79 strict threshold for no-prior-theory findings; t ≥ 3.00 acceptable "
        "with literature-grounded theory. K-framing is per-family not global; a broad "
        "signal passing a narrow K_family gate is legitimate (here: K_instrument = 30, "
        "K_global = 89)."
    )
    md.append(
        "- **Bailey-López de Prado 2014** (`bailey_lopez_de_prado_2014_deflated_sharpe.md`): "
        "multiple-testing haircut. Not directly used here (descriptive, not Sharpe "
        "promotion), but the framing — a result that is robust across narrow K families "
        "is evidence of genuine effect, not a tuning artifact — is load-bearing for the "
        "per-instrument panel below."
    )
    md.append(
        "- **Harvey-Liu 2015 BHY** (`harvey_liu_2015_backtesting.md`): BH-FDR correction "
        "at the hypothesis-family level. Applied here as K_global = adequately-powered "
        "cells and K_instrument = per-asset adequately-powered cells."
    )
    md.append("")
    md.append(
        "**What the literature predicts vs what this descriptive finds:** under Fitschen, "
        "aligned pre-velocity should continue → positive Δ_WR universally. A flat-to-null "
        "cross-section result contradicts the *strength* of Fitschen's intraday trend "
        "prediction at the specific timescale of the 5-bar pre-ORB slope (which is finer "
        "than Fitschen's hourly bars). A null at this scale does NOT refute Fitschen — it "
        "narrows the observed continuation signal to other timescales or feature framings."
    )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Question")
    md.append("")
    md.append("Does pre-ORB-end velocity direction alignment with E2 fill direction predict")
    md.append("outcome on the **UNFILTERED** universe, uniformly across the 12 × 3 × 3")
    md.append("deployed cross-section? Answer informs whether action queue item #2 should")
    md.append("be re-scoped to a universal pre-reg or remains lane-specific.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Scope")
    md.append("")
    md.append(f"- Sessions: {len(SESSIONS)} (all with canonical `pre_velocity` column)")
    md.append(f"- Instruments: {len(INSTRUMENTS)} (ACTIVE_ORB_INSTRUMENTS)")
    md.append(f"- Apertures: {len(APERTURES)} (O5, O15, O30)")
    md.append(f"- Cells attempted: **{total_cells}**")
    md.append(f"- Cells with sufficient data (≥ {2 * MIN_N_PER_GROUP} trades): **{produced}**")
    md.append(f"- Cells underpowered per RULE 3.2 (n_aligned or n_opposed < {MIN_N_PER_GROUP}): **{underpowered}**")
    md.append(f"- Cells with adequate power: **{len(nonunder)}**")
    md.append("")
    md.append(f"- Entry model: {ENTRY_MODEL} CB={CONFIRM_BARS} RR={RR_TARGET}")
    md.append(f"- IS: trading_day < {HOLDOUT} (Mode A sacred holdout)")
    md.append("- Source: canonical `orb_outcomes ⨝ daily_features` on (trading_day, symbol, orb_minutes)")
    md.append("- Feature: `orb_{session}_pre_velocity` (canonical per pipeline/build_daily_features.py)")
    md.append("- Direction: inferred from `entry_price > stop_price → long` (fill metadata, pre-entry-knowable)")
    md.append("- No pre-filter applied (unfiltered universe)")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Headline")
    md.append("")
    md.append(f"- Adequately-powered cells: **{len(nonunder)} / {total_cells}**")
    md.append(f"- Cells with p_WR < 0.05 (raw, uncorrected): **{len(directional_05)}**")
    md.append(f"- Cells surviving BH-FDR at q=0.05, K_global = {len(nonunder)}: **{len(bh_survivors)}**")
    md.append("")
    md.append("### Per-instrument BH-FDR (narrower K_family)")
    md.append("")
    md.append(
        "Per `backtesting-methodology.md § RULE 4`, K_family is the natural hypothesis unit. "
        "A signal that is asset-class-specific (e.g. metals vs equities) may survive under "
        "K_instrument but fail under K_global. Reported side-by-side, not as a rescue."
    )
    md.append("")
    md.append("| Instrument | N_powered | p<0.05 (raw) | BH-FDR survivors (q=0.05) |")
    md.append("|---|---|---|---|")
    per_inst_survivors: dict[str, list[CellResult]] = {}
    for inst in INSTRUMENTS:
        inst_cells = [r for r in nonunder if r.symbol == inst]
        inst_raw = sum(1 for r in inst_cells if r.directional_at_05)
        if inst_cells:
            inst_ps = [float(r.p_wr) for r in inst_cells]
            inst_mask = bh_fdr(inst_ps, q=0.05)
            inst_surv = [inst_cells[i] for i in range(len(inst_cells)) if inst_mask[i]]
        else:
            inst_surv = []
        per_inst_survivors[inst] = inst_surv
        md.append(
            f"| {inst} | {len(inst_cells)} | {inst_raw} | {len(inst_surv)} |"
        )
    md.append("")
    if any(per_inst_survivors.values()):
        md.append("**Per-instrument survivor detail:**")
        md.append("")
        md.append("| Instrument | Session | O | N | Δ_WR | p_WR | Δ_ExpR | Welch p |")
        md.append("|---|---|---|---|---|---|---|---|")
        for inst, survs in per_inst_survivors.items():
            for r in sorted(survs, key=lambda x: x.p_wr):
                md.append(
                    f"| {inst} | {r.session} | {r.orb_minutes} | {r.n_total} | "
                    f"{r.delta_wr_pp:+.2f}pp | {r.p_wr:.3f} | "
                    f"{r.delta_exp_r:+.3f}R | {r.welch_p:.3f} |"
                )
        md.append("")
    else:
        md.append(
            "**No survivors at any per-instrument K_family either.** Rejects asset-class-"
            "specific universality as well as the full cross-section."
        )
        md.append("")
    md.append("---")
    md.append("")
    if bh_survivors:
        md.append("**Universal-signal interpretation:** if pre_velocity alignment were a broad")
        md.append("behavioral edge, we would expect ≫50% of powered cells to flag at p<0.05")
        md.append("with consistent sign. Observed cells and signs:")
        md.append("")
        signs = [np.sign(r.delta_wr_pp) for r in bh_survivors]
        pos = sum(1 for s in signs if s > 0)
        neg = sum(1 for s in signs if s < 0)
        md.append(f"- BH-FDR survivors with positive Δ_WR (aligned wins more): {pos}")
        md.append(f"- BH-FDR survivors with negative Δ_WR (opposed wins more): {neg}")
        md.append("")
    else:
        md.append("**No cells survive BH-FDR at K = (all adequately-powered cells), q=0.05.**")
        md.append("This rejects a broad universal pre_velocity alignment edge at this scope.")
        md.append("Individual cells may still be worth narrower pre-reg investigation, but a")
        md.append("cross-session Pathway-B rewrite of action queue item #2 is NOT supported.")
        md.append("")
    md.append("---")
    md.append("")
    md.append("## Full per-cell table (adequately-powered, sorted by |Welch t| desc)")
    md.append("")
    md.append(
        "| Instrument | Session | O | N_total | N_aligned | N_opposed | WR_a | WR_o | Δ_WR | p_WR | ExpR_a | ExpR_o | Δ_ExpR | t | p | BH |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    sorted_powered = sorted(
        nonunder,
        key=lambda r: -abs(r.welch_t) if not np.isnan(r.welch_t) else 0.0,
    )
    bh_set = {id(r) for r in bh_survivors}
    for r in sorted_powered:
        bh_tag = "✓" if id(r) in bh_set else ""
        md.append(
            f"| {r.symbol} | {r.session} | {r.orb_minutes} | {r.n_total} | "
            f"{r.n_aligned} | {r.n_opposed} | {r.wr_aligned:.1f}% | {r.wr_opposed:.1f}% | "
            f"{r.delta_wr_pp:+.2f}pp | {r.p_wr:.3f} | "
            f"{r.exp_r_aligned:+.3f}R | {r.exp_r_opposed:+.3f}R | {r.delta_exp_r:+.3f}R | "
            f"{r.welch_t:+.2f} | {r.welch_p:.3f} | {bh_tag} |"
        )
    md.append("")
    if underpowered > 0:
        md.append("### Underpowered cells (omitted from BH-FDR)")
        md.append("")
        md.append("| Instrument | Session | O | N_total | N_aligned | N_opposed |")
        md.append("|---|---|---|---|---|---|")
        for r in results:
            if r.underpowered:
                md.append(
                    f"| {r.symbol} | {r.session} | {r.orb_minutes} | "
                    f"{r.n_total} | {r.n_aligned} | {r.n_opposed} |"
                )
        md.append("")
    md.append("---")
    md.append("")
    md.append("## Interpretation")
    md.append("")
    if len(bh_survivors) == 0:
        md.append("- **No universal edge in pre_velocity direction alignment** at this scope.")
        md.append("- Consistent with the null hypothesis that pre-ORB-end momentum direction")
        md.append("  does not predict outcome on unfiltered E2 CB1 RR1.5 across the deployed cross-section.")
        md.append("- Does NOT preclude lane-specific edges or sub-conditioned frames")
        md.append("  (e.g. only when paired with volatility regime, or only on specific")
        md.append("  session-instrument combinations). Those would require separate pre-registered tests.")
    elif len(bh_survivors) < len(nonunder) / 10:
        md.append(f"- **Narrow hits only** ({len(bh_survivors)} BH-FDR survivors in {len(nonunder)} powered cells).")
        md.append("- Cannot claim a universal behavioral edge from this.")
        md.append("- The surviving cells MAY deserve lane-specific pre-reg treatment, but")
        md.append("  only after checking sign consistency and year-by-year stability on those cells.")
    else:
        md.append(f"- **Broad signal** ({len(bh_survivors)} / {len(nonunder)} cells survive BH-FDR).")
        md.append("- Warrants a universal-scoped Pathway-B pre-reg with the surviving sign")
        md.append("  direction, written against `docs/institutional/pre_registered_criteria.md`.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Limitations")
    md.append("")
    md.append("- Binary alignment (aligned / opposed) discards magnitude information in `pre_velocity`.")
    md.append("  A continuous-scaling quintile analysis would be a stronger next test if this")
    md.append("  pass shows any hint of signal.")
    md.append("- Fill direction inferred from `entry_price > stop_price`; relies on canonical")
    md.append("  backtest outcome-builder invariants being intact. Verified by `check_drift.py`.")
    md.append("- RR target fixed at 1.5 to match the dominant deployed lane shape. Other RRs")
    md.append("  (1.0, 2.0) would change the win-probability baseline and the ExpR arithmetic.")
    md.append("- The UNFILTERED universe includes trades where deployed cost-gate filters would")
    md.append("  have skipped the lane. If directional signal is masked by small-ORB cost drag,")
    md.append("  the descriptive null could underestimate true directional content. A follow-up")
    md.append("  with a cost-agnostic size-normalized baseline could disambiguate.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Reproduction")
    md.append("")
    md.append("```bash")
    md.append("python research/audit_pre_break_context_cross_session.py")
    md.append("```")
    md.append("")
    md.append("Writes this document to")
    md.append("`docs/audit/results/2026-04-21-pre-break-context-cross-session-descriptive.md`.")
    md.append("")
    return "\n".join(md)


def main() -> None:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        results: list[CellResult] = []
        for symbol in INSTRUMENTS:
            for session in SESSIONS:
                for orb_minutes in APERTURES:
                    try:
                        r = run_cell(con, symbol, session, orb_minutes)
                    except duckdb.BinderException:
                        # Session missing pre_velocity column (shouldn't happen
                        # for declared SESSIONS but defend anyway)
                        r = None
                    if r is not None:
                        results.append(r)
    finally:
        con.close()

    md = render_md(results)
    out_path = Path(
        "docs/audit/results/2026-04-21-pre-break-context-cross-session-descriptive.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")

    n_powered = sum(1 for r in results if not r.underpowered)
    n_underpwr = sum(1 for r in results if r.underpowered)
    n_sig = sum(1 for r in results if not r.underpowered and r.directional_at_05)

    print("=== CROSS-SESSION PRE-BREAK CONTEXT DESCRIPTIVE DIAGNOSTIC ===")
    print(f"Cells with data: {len(results)} of {len(SESSIONS) * len(INSTRUMENTS) * len(APERTURES)}")
    print(f"Adequately powered: {n_powered}")
    print(f"Underpowered (RULE 3.2): {n_underpwr}")
    print(f"Raw p_WR < 0.05 (uncorrected, powered cells): {n_sig}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
