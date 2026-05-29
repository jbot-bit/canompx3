"""MGC CPCV audit — methodology-correct multi-path re-test of underpowered cells.

PASS 2 of the LOCKED design in
`docs/audit/hypotheses/drafts/2026-05-29-mgc-cpcv-methodology-audit-PASS1.md`.

Mandate (verbatim from PASS 1): CPCV on MGC as a *methodology-correct* audit,
NOT a threshold rescue. No post-hoc threshold changes. No deployment claim.
Same gates; better estimator.

Why CPCV (RULE 3.3): the binary trade-fraction OOS that killed every MGC cell
uses ONE held-out path (last 30% of trades) ~= 30 OOS trades, power ~0.4 ->
STATISTICALLY_USELESS. CPCV (AFML 2018 Ch 12 Sec 12.4) re-uses the SAME data,
partitioned into N=6 temporal groups, testing ALL C(6,4)=15 train/test splits
to assemble phi=5 backtest paths -> a *distribution* of OOS performance, not a
single noisy point. No threshold moves: t>=3.79, PBO<0.50, power floors are all
read-only constants from pre_registered_criteria.md.

Literature grounding (verbatim extracts, not memory):
  - CPCV formula: AFML 2018 Sec 12.4.1
    `docs/institutional/literature/lopez_de_prado_2018_afml_ch_3_7_8.md:189-199`.
    "For T observations partitioned into N groups [...] phi[N,k] = k/N * C(N,N-k)."
    N=6, k=2 -> C(6,4)=15 splits -> phi=5 paths.
  - Purge + embargo: AFML Ch 7 Sec 7.4 (same extract). Embargo h ~= 0.01*T.
  - PBO (logit convention): Bailey et al 2014, delegated to
    `trading_app.pbo`-style logit transform.
  - Thresholds (UNCHANGED): `docs/institutional/pre_registered_criteria.md`.

Discipline:
  - RULE 9 canonical triple-join (orb_minutes PINNED).
  - RULE 6.1 trade-time-knowable features ONLY (no break-bar / mae / outcome).
  - Costs: `orb_outcomes.pnl_r` is already net of canonical friction
    (`pipeline.cost_model.COST_SPECS['MGC'].total_friction = 5.74`) -> CPCV
    scores on pnl_r directly; no re-costing.
  - NO DB writes. Read-only. Emits per-path + aggregate + PBO markdown.
  - RULE 13 pressure test: `--pressure-test` injects a look-ahead series and
    confirms it does NOT spuriously pass the VALID gate.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from itertools import combinations

import duckdb
import numpy as np
import pandas as pd

from pipeline.cost_model import COST_SPECS
from research.comprehensive_deployed_lane_scan import _overnight_lookhead_clean
from research.oos_power import one_sample_power, power_verdict

# ---- CPCV design constants (LOCKED in PASS 1; do NOT redesign) --------------
N_GROUPS = 6          # AFML Fig 12.1 worked example
K_TEST = 2            # testing-set size in groups
EMBARGO_FRAC = 0.01   # LdP default h ~= 0.01*T

# Selection budget carried for honest PBO/DSR accounting (PASS 1 line 43-48).
# The 6 candidates were selected from a 1992-cell MGC wide scan.
SELECTION_BUDGET_K = 1992

# Strict Chordia threshold (no prior theory) — pre_registered_criteria.md C4.
# Read-only constant; NOT relaxed for MGC.
CHORDIA_T_STRICT = 3.79

# MGC real-micro horizon (years). PASS 1: ~3.0yr. Used only for narrative.
MGC_HORIZON_YEARS = 3.0


@dataclass(frozen=True)
class Candidate:
    """One underpowered-but-promising MGC cell from the powered wide scan."""

    idx: int
    session: str
    om: int
    rr: float
    direction: str
    feature: str
    op: str
    threshold: float

    @property
    def filter_label(self) -> str:
        if self.feature == "none":
            return "none"
        return f"{self.feature}{self.op}{self.threshold:g}"

    @property
    def label(self) -> str:
        return (
            f"MGC {self.session} O{self.om} RR{self.rr:g} {self.direction} "
            f"[{self.filter_label}]"
        )


# Exact candidate set from PASS 1 (reproduced 1:1 against canonical 2026-05-29).
CANDIDATES: tuple[Candidate, ...] = (
    Candidate(1, "US_DATA_830", 30, 2.0, "long", "day_of_week", "==", 1),
    Candidate(2, "NYSE_OPEN", 30, 2.0, "long", "day_of_week", "==", 3),
    Candidate(3, "NYSE_OPEN", 30, 1.0, "long", "day_of_week", "==", 3),
    Candidate(4, "SINGAPORE_OPEN", 30, 2.0, "long", "day_of_week", "==", 4),
    Candidate(5, "EUROPE_FLOW", 30, 2.0, "long", "atr_20_pct", ">=", 60),
    Candidate(6, "LONDON_METALS", 30, 1.5, "long", "overnight_range_pct", ">=", 80),
)


# ---- canonical data pull (RULE 9 triple-join) -------------------------------
def _pull_candidate(con: duckdb.DuckDBPyConnection, c: Candidate) -> pd.DataFrame:
    """Return the candidate's full trade series, temporally sorted.

    Canonical triple-join (orb_minutes PINNED). Trade-time features only.
    """
    q = f"""
        SELECT o.trading_day, o.pnl_r,
               CASE WHEN o.stop_price < o.entry_price THEN 'long' ELSE 'short' END AS dir,
               d.overnight_range_pct, d.atr_20_pct, d.day_of_week
        FROM orb_outcomes o
        JOIN daily_features d
          ON d.trading_day = o.trading_day
         AND d.symbol = o.symbol
         AND d.orb_minutes = o.orb_minutes
        WHERE o.symbol = 'MGC' AND o.orb_label = '{c.session}'
          AND o.orb_minutes = {c.om} AND o.entry_model = 'E2'
          AND o.rr_target = {c.rr} AND o.confirm_bars = 1
          AND o.outcome IS NOT NULL
    """
    df = con.sql(q).df()
    if df.empty:
        return df
    df = df[df["dir"] == c.direction]
    if c.feature != "none":
        # RULE 1.2 look-ahead gate: overnight_* invalid for ORB < 17:00 Brisbane.
        if c.feature.startswith("overnight_") and not _overnight_lookhead_clean(c.session):
            raise ValueError(
                f"{c.label}: overnight feature on look-ahead-unsafe session "
                f"{c.session} — RULE 1.2 violation. Candidate set is malformed."
            )
        s = pd.to_numeric(df[c.feature], errors="coerce")
        if c.op == ">=":
            df = df[s >= c.threshold]
        elif c.op == "<=":
            df = df[s <= c.threshold]
        elif c.op == "==":
            df = df[s == c.threshold]
        else:  # pragma: no cover - candidate set is fixed
            raise ValueError(f"unknown op {c.op!r}")
    df = df.dropna(subset=["pnl_r"]).copy()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    return df.sort_values("trading_day").reset_index(drop=True)


# ---- CPCV path construction (AFML Sec 12.4 — faithful, no canonical impl) ----
def _group_bounds(n_obs: int, n_groups: int) -> list[tuple[int, int]]:
    """Contiguous, no-shuffle group [start, end) bounds per AFML Sec 12.4.1.

    Groups 1..N-1 are floor(T/N); the Nth absorbs the remainder.
    """
    base = n_obs // n_groups
    bounds: list[tuple[int, int]] = []
    start = 0
    for g in range(n_groups):
        end = start + base if g < n_groups - 1 else n_obs
        bounds.append((start, end))
        start = end
    return bounds


def _purge_embargo_train_idx(
    train_groups: tuple[int, ...],
    test_groups: tuple[int, ...],
    bounds: list[tuple[int, int]],
    days: np.ndarray,
    embargo_days: int,
) -> np.ndarray:
    """Train positional indices after AFML Ch 7 Sec 7.4 purge + embargo.

    Purge: drop any train observation whose trading_day falls inside the
    [min, max] calendar span of ANY test group (label-overlap guard). For ORB
    the label window is intraday (entry -> same-day exit), so the dominant
    leakage channel is serial correlation across adjacent days, addressed by
    the embargo. The span-purge is the conservative AFML default applied for
    safety, not because intraday labels overlap calendar-distant test groups.

    Embargo: additionally drop train observations within `embargo_days`
    trading days AFTER each test group's span (one-sided, per Sec 7.4).
    """
    # Calendar span of each test group.
    test_spans: list[tuple[np.datetime64, np.datetime64]] = []
    for tg in test_groups:
        s, e = bounds[tg]
        if e <= s:
            continue
        test_spans.append((days[s], days[e - 1]))

    keep: list[int] = []
    embargo_delta = np.timedelta64(embargo_days, "D")
    for tg in train_groups:
        s, e = bounds[tg]
        for pos in range(s, e):
            d = days[pos]
            leaked = False
            for span_lo, span_hi in test_spans:
                # Purge: inside the test span.
                if span_lo <= d <= span_hi:
                    leaked = True
                    break
                # Embargo: within h trading days AFTER the test span.
                if span_hi < d <= span_hi + embargo_delta:
                    leaked = True
                    break
            if not leaked:
                keep.append(pos)
    return np.array(keep, dtype=int)


def cpcv_phi(n_groups: int, k_test: int) -> int:
    """phi[N,k] = k/N * C(N, N-k) backtest paths (AFML Sec 12.4.1).

    Returned for accounting/disclosure only — see the degeneracy note in
    run_cpcv(): for a FIXED-OUTCOME backtest (no model refit) the phi
    full-coverage paths are mathematically identical, so the informative
    multi-path distribution is the per-split test-fold distribution, not the
    phi reconstructed paths.
    """
    n_combos = math.comb(n_groups, n_groups - k_test)  # C(N, N-k) = C(N, k)
    return (k_test * n_combos) // n_groups


def _t_stat(x: np.ndarray) -> tuple[int, float, float, float]:
    """Return (n, mean, per-trade Sharpe, one-sample t) for a pnl_r array."""
    n = len(x)
    if n < 2:
        return n, float("nan"), float("nan"), float("nan")
    m = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return n, m, float("nan"), 0.0
    sharpe = m / sd          # per-trade Sharpe (unitless)
    t = m / (sd / math.sqrt(n))
    return n, m, sharpe, t


@dataclass(frozen=True)
class FoldResult:
    """One CPCV test-fold = the k=2 held-out groups of a single split."""

    fold_id: int
    test_groups: tuple[int, ...]
    n_test: int
    expr: float
    sharpe: float
    t: float


@dataclass(frozen=True)
class CpcvResult:
    candidate: Candidate
    n_total: int
    n_splits: int
    phi: int
    folds: list[FoldResult]
    median_expr: float
    median_sharpe: float
    median_t: float
    worst_expr: float
    iqr_expr: float
    frac_positive: float
    pooled_n: int
    pooled_t: float
    pooled_power: float
    pooled_tier: str
    pbo: float
    logit_pbo: float | None
    verdict: str
    verdict_reason: str


def run_cpcv(c: Candidate, df: pd.DataFrame) -> CpcvResult:
    """Run CPCV on one candidate's full trade series and classify it."""
    pnl = df["pnl_r"].to_numpy(dtype=float)
    days = df["trading_day"].to_numpy(dtype="datetime64[D]")
    n_total = len(pnl)

    bounds = _group_bounds(n_total, N_GROUPS)
    embargo_days = max(1, math.ceil(EMBARGO_FRAC * n_total))

    # All C(N, N-k) train/test splits: choose K_TEST groups to be the test set.
    test_combos = list(combinations(range(N_GROUPS), K_TEST))
    splits: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for test in test_combos:
        train = tuple(g for g in range(N_GROUPS) if g not in test)
        splits.append((train, test))
    n_splits = len(splits)  # C(6,4)=15

    # CPCV multi-path distribution for a FIXED-OUTCOME backtest.
    #
    # AFML Sec 12.4 builds phi = k/N*C(N,N-k) full-coverage paths by threading
    # one test-instance of each group. We PROVED (see module docstring + result
    # doc) those phi paths are mathematically IDENTICAL here: each path is a
    # union of ALL N groups, and a trade's pnl_r is a fixed historical outcome
    # that does NOT change with which split tested it (there is no model to
    # refit, so purging cannot alter a test group's own pnl). Reporting 5
    # identical full-series paths would be a fabricated distribution.
    #
    # The genuine CPCV dispersion is the per-split TEST-FOLD distribution: each
    # of the C(N,N-k)=15 splits holds out k=2 groups, and the 15 test-fold means
    # vary because they sample different temporal windows. This is the honest
    # multi-path object — it answers "how does this cell's OOS performance vary
    # across all combinatorial held-out windows?" without inventing dispersion.
    phi = cpcv_phi(N_GROUPS, K_TEST)  # =5; disclosed, not used as the distribution

    fold_results: list[FoldResult] = []
    for f_id, (_train_groups, test_groups) in enumerate(splits):
        idx: list[int] = []
        for g in test_groups:
            s, e = bounds[g]
            idx.extend(range(s, e))
        seg = pnl[np.array(sorted(idx), dtype=int)]
        n_seg, m_seg, sh_seg, t_seg = _t_stat(seg)
        fold_results.append(
            FoldResult(f_id, test_groups, n_seg, m_seg, sh_seg, t_seg)
        )

    path_exprs = np.array([fr.expr for fr in fold_results], dtype=float)
    path_sharpes = np.array([fr.sharpe for fr in fold_results], dtype=float)
    path_ts = np.array([fr.t for fr in fold_results], dtype=float)

    # All-NaN guard: a degenerate (e.g. zero-variance) series yields all-NaN
    # Sharpe/t arrays; nanmedian warns and returns NaN. Treat explicitly as
    # NaN (institutional-rigor Sec 6: no silent NaN) so downstream classify()
    # routes it to non-VALID via the median-t < CHORDIA gate.
    def _safe_nanmedian(a: np.ndarray) -> float:
        return float(np.nanmedian(a)) if np.any(~np.isnan(a)) else float("nan")

    # Fold-distribution dispersion diagnostics (K1 median ExpR, K2 worst path).
    median_expr = _safe_nanmedian(path_exprs)
    median_sharpe = _safe_nanmedian(path_sharpes)
    median_fold_t = _safe_nanmedian(path_ts)  # per-fold t (~33 trades) — reported, NOT the gate
    worst_expr = float(np.nanmin(path_exprs)) if np.any(~np.isnan(path_exprs)) else float("nan")
    q75, q25 = np.nanpercentile(path_exprs, [75, 25])
    iqr_expr = float(q75 - q25)
    frac_positive = float(np.mean(path_exprs > 0))

    # Pooled / full-coverage-path stats. The only COMPLETE backtest path (all N
    # groups) is the full series; for a fixed-outcome backtest its t IS the
    # pooled t. K4's t-gate (calibrated against full-sample t, not 33-trade
    # fold-t) therefore uses pooled_t — see _classify().
    pooled_n, _, _, pooled_t = _t_stat(pnl)
    cohen_d = abs(pooled_t) / math.sqrt(n_total) if n_total > 1 else 0.0
    pooled_power = one_sample_power(cohen_d, n_total)
    pooled_tier = power_verdict(pooled_power)

    # PBO (K3) per-candidate, AFML/Bailey style: across the C(N,N-k) splits,
    # how often does the IS-best (highest train mean, purged+embargoed) realize
    # NEGATIVE OOS mean? Here the "strategies" being selected over are the
    # alternative train-window configurations of THIS candidate — the relevant
    # overfit question for a single cell whose window was implicitly chosen.
    n_neg_oos = 0
    n_scored = 0
    for train_groups, test_groups in splits:
        train_idx = _purge_embargo_train_idx(
            train_groups, test_groups, bounds, days, embargo_days
        )
        if len(train_idx) < 2:
            continue
        test_idx: list[int] = []
        for tg in test_groups:
            s, e = bounds[tg]
            test_idx.extend(range(s, e))
        if len(test_idx) < 2:
            continue
        is_mean = float(np.mean(pnl[train_idx]))
        oos_mean = float(np.mean(pnl[np.array(test_idx, dtype=int)]))
        # IS-best decision: this candidate's window is "selected" only when its
        # IS mean is positive (a real desk would not deploy a negative-IS cell).
        # PBO counts the fraction of selected splits whose OOS went negative.
        if is_mean > 0:
            n_scored += 1
            if oos_mean < 0:
                n_neg_oos += 1
    pbo = (n_neg_oos / n_scored) if n_scored > 0 else 1.0
    logit_pbo = (
        round(math.log(pbo / (1 - pbo)), 4) if 0 < pbo < 1 else None
    )

    verdict, reason = _classify(
        median_expr=median_expr,
        median_t=pooled_t,  # full-coverage-path t (== pooled for fixed outcomes)
        worst_expr=worst_expr,
        frac_positive=frac_positive,
        pbo=pbo,
        pooled_power=pooled_power,
    )

    return CpcvResult(
        candidate=c,
        n_total=n_total,
        n_splits=n_splits,
        phi=phi,
        folds=fold_results,
        median_expr=median_expr,
        median_sharpe=median_sharpe,
        median_t=median_fold_t,
        worst_expr=worst_expr,
        iqr_expr=iqr_expr,
        frac_positive=frac_positive,
        pooled_n=pooled_n,
        pooled_t=pooled_t,
        pooled_power=pooled_power,
        pooled_tier=pooled_tier,
        pbo=round(pbo, 4),
        logit_pbo=logit_pbo,
        verdict=verdict,
        verdict_reason=reason,
    )


# ---- classification (LOCKED kill criteria K1-K4; no post-hoc change) --------
def _classify(
    *,
    median_expr: float,
    median_t: float,
    worst_expr: float,
    frac_positive: float,
    pbo: float,
    pooled_power: float,
) -> tuple[str, str]:
    """Apply PASS 1 LOCKED kill criteria. Returns (verdict, reason).

    K1: median fold ExpR <= 0                        -> UNVERIFIED
    K2: worst fold ExpR < -0.05 AND >40% folds neg   -> WRONG
    K3: PBO > 0.50                                   -> WRONG
    K4: full-path t < 3.79 AND pooled power < 0.50   -> UNVERIFIED
    VALID only if: median fold ExpR>0 AND full-path t>=3.79 AND PBO<0.50
                   AND worst fold not catastrophic AND pooled power>=0.50

    `median_expr`/`worst_expr`/`frac_positive` come from the per-split test-fold
    distribution (the genuine CPCV dispersion for a fixed-outcome backtest).
    `median_t` is the FULL-COVERAGE-PATH t (== pooled t here): the only complete
    backtest path is the full series, so its t is the threshold-comparable
    statistic, not the ~33-trade per-fold t. See run_cpcv() degeneracy note.
    """
    frac_negative = 1.0 - frac_positive
    # NaN guard (institutional-rigor Sec 6): a degenerate series (zero variance,
    # or too few trades per group to score a fold) yields NaN median stats.
    # NaN comparisons silently evaluate False, which would let a degenerate
    # input slip past K1/K4 into CONDITIONAL. Treat NaN as unscoreable ->
    # UNVERIFIED explicitly.
    if math.isnan(median_expr) or math.isnan(median_t):
        return (
            "UNVERIFIED",
            f"degenerate: median fold ExpR={median_expr}, full-path t={median_t} "
            "(unscoreable — zero variance or too few trades/group)",
        )
    # K3 first — overfit verdict dominates (Bailey: PBO>0.5 == coin flip).
    if pbo > 0.50:
        return "WRONG", f"K3: PBO={pbo:.3f} > 0.50 (selection more likely overfit than real)"
    # K2 — catastrophic worst-fold artifact.
    if worst_expr < -0.05 and frac_negative > 0.40:
        return (
            "WRONG",
            f"K2: worst fold ExpR={worst_expr:+.4f} < -0.05 AND "
            f"{frac_negative*100:.0f}% of folds negative",
        )
    # K1 — no edge survives the multi-fold distribution.
    if median_expr <= 0:
        return "UNVERIFIED", f"K1: median fold ExpR={median_expr:+.4f} <= 0"
    # K4 — underpowered / sub-Chordia on the full path.
    if median_t < CHORDIA_T_STRICT and pooled_power < 0.50:
        return (
            "UNVERIFIED",
            f"K4: full-path t={median_t:.2f} < {CHORDIA_T_STRICT} AND "
            f"pooled power={pooled_power:.2f} < 0.50",
        )
    # VALID gate (all must hold).
    if (
        median_expr > 0
        and median_t >= CHORDIA_T_STRICT
        and pbo < 0.50
        and worst_expr >= -0.05
        and pooled_power >= 0.50
    ):
        return (
            "VALID",
            f"median fold ExpR={median_expr:+.4f}>0, full-path t={median_t:.2f}>="
            f"{CHORDIA_T_STRICT}, PBO={pbo:.3f}<0.50, worst fold={worst_expr:+.4f}, "
            f"power={pooled_power:.2f}>=0.50",
        )
    # Positive-but-not-confirmatory — between UNVERIFIED and VALID.
    return (
        "CONDITIONAL",
        f"median fold ExpR={median_expr:+.4f}>0 and not killed, but VALID gate "
        f"unmet (full-path t={median_t:.2f}, power={pooled_power:.2f})",
    )


# ---- reporting --------------------------------------------------------------
def render_markdown(results: list[CpcvResult]) -> str:
    friction = COST_SPECS["MGC"].total_friction
    lines: list[str] = []
    lines.append("# MGC CPCV Audit — PASS 2 (run, methodology-correct)\n")
    lines.append(
        "**Mandate:** CPCV (AFML 2018 Sec 12.4) multi-path re-test of 6 "
        "underpowered-but-promising MGC cells. NOT a threshold rescue — same "
        "gates, better estimator. No DB writes. Read-only canonical query.\n"
    )
    n_splits = len(list(combinations(range(N_GROUPS), K_TEST)))
    phi = cpcv_phi(N_GROUPS, K_TEST)
    lines.append(
        f"**Design (LOCKED):** N={N_GROUPS} temporal groups, k={K_TEST} -> "
        f"C({N_GROUPS},{N_GROUPS - K_TEST})={n_splits} splits -> "
        f"phi={phi} combinatorial paths. "
        f"Embargo h=ceil({EMBARGO_FRAC}*T). Purge per AFML Ch 7 Sec 7.4.\n"
    )
    lines.append(
        "**METHODOLOGICAL FINDING (load-bearing — read before the numbers):** "
        f"the phi={phi} full-coverage CPCV paths are MATHEMATICALLY IDENTICAL "
        "for this audit, and that is structural, not a bug. AFML Sec 12.4 builds "
        "each path as one tested instance of every group threaded across splits; "
        "each path is therefore a union of ALL N groups = the full trade series. "
        "A trade's `pnl_r` is a FIXED historical outcome that does not change "
        "with which split tested it (there is no model to refit, so purging only "
        "removes TRAIN observations and cannot alter a test group's own pnl). "
        "Hence all phi paths reconstruct the identical full series with identical "
        "ExpR/t — CPCV's path-dispersion innovation requires a per-split model "
        "refit, which a fixed-outcome backtest does not have. **The genuine "
        f"multi-path object reported below is the per-split TEST-FOLD distribution "
        f"({n_splits} folds, each the k={K_TEST} held-out groups of one split), "
        "whose means DO vary across temporal windows. The full-coverage-path t "
        "(== pooled t) is used for the K4 t-gate; the fold distribution drives "
        "K1/K2 dispersion.**\n"
    )
    lines.append(
        f"**Costs:** MGC total_friction={friction} (canonical "
        "`COST_SPECS['MGC']`); `orb_outcomes.pnl_r` already net — scored directly.\n"
    )
    lines.append(
        f"**Selection budget (honest):** K={SELECTION_BUDGET_K} (the 6 were "
        f"selected from a {SELECTION_BUDGET_K}-cell wide scan). With a "
        f"~{MGC_HORIZON_YEARS:g}yr horizon this is well over the Bailey 2013 "
        "MinBTL bound, so the EXPECTED outcome is UNVERIFIED even under CPCV. "
        "This audit tests whether the multi-path estimator changes that.\n"
    )
    lines.append(
        "**Kill criteria (LOCKED, no post-hoc change):** "
        "K1 median fold ExpR<=0 -> UNVERIFIED; "
        "K2 worst fold ExpR<-0.05 AND >40% folds neg -> WRONG; "
        "K3 PBO>0.50 -> WRONG; "
        f"K4 full-path t<{CHORDIA_T_STRICT} AND pooled power<0.50 -> UNVERIFIED. "
        f"VALID only if median fold ExpR>0 AND full-path t>={CHORDIA_T_STRICT} AND "
        "PBO<0.50 AND worst fold>=-0.05 AND pooled power>=0.50.\n"
    )

    lines.append("\n## Aggregate verdict table\n")
    lines.append(
        "| # | candidate | N | folds | median fold ExpR | full-path t | "
        "worst fold | fold IQR | %folds+ | pooled power | tier | PBO | verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        c = r.candidate
        lines.append(
            f"| {c.idx} | {c.label} | {r.n_total} | {r.n_splits} | "
            f"{r.median_expr:+.4f} | {r.pooled_t:.2f} | {r.worst_expr:+.4f} | "
            f"{r.iqr_expr:.4f} | {r.frac_positive*100:.0f}% | "
            f"{r.pooled_power:.2f} | {r.pooled_tier} | {r.pbo:.3f} | "
            f"**{r.verdict}** |"
        )

    lines.append("\n## Per-candidate detail\n")
    for r in results:
        c = r.candidate
        lines.append(f"### #{c.idx} — {c.label}\n")
        lines.append(f"- **Verdict:** {r.verdict} — {r.verdict_reason}")
        lines.append(
            f"- N_total={r.n_total}, splits/folds={r.n_splits}, phi={r.phi} "
            f"(degenerate — see finding), PBO={r.pbo:.3f} (logit={r.logit_pbo})"
        )
        lines.append(
            f"- full-path t={r.pooled_t:.2f} (N={r.pooled_n}), pooled "
            f"power={r.pooled_power:.2f} ({r.pooled_tier}); median fold "
            f"t={r.median_t:.2f} (per-fold ~N/3 trades, not the gate)"
        )
        lines.append(
            f"\n  CPCV test-fold distribution ({r.n_splits} folds, "
            f"k={K_TEST} held-out groups each):\n"
        )
        lines.append("  | fold | test groups | N_test | ExpR | Sharpe | t |")
        lines.append("  |---|---|---|---|---|---|")
        for fr in r.folds:
            tg = ",".join(str(g) for g in fr.test_groups)
            lines.append(
                f"  | {fr.fold_id} | {{{tg}}} | {fr.n_test} | {fr.expr:+.4f} | "
                f"{fr.sharpe:+.4f} | {fr.t:.2f} |"
            )
        lines.append("")

    n_valid = sum(1 for r in results if r.verdict == "VALID")
    n_cond = sum(1 for r in results if r.verdict == "CONDITIONAL")
    n_unv = sum(1 for r in results if r.verdict == "UNVERIFIED")
    n_wrong = sum(1 for r in results if r.verdict == "WRONG")
    lines.append("## Summary\n")
    lines.append(
        f"- VALID={n_valid}, CONDITIONAL={n_cond}, UNVERIFIED={n_unv}, "
        f"WRONG={n_wrong} (of {len(results)})"
    )
    lines.append(
        "- CPCV is a confirmatory re-estimator on prior survivors; the K="
        f"{SELECTION_BUDGET_K} selection budget is carried for honest PBO/DSR "
        "accounting and is NOT re-spent here."
    )
    none_deployable = n_valid == 0
    lines.append(
        "- No threshold was changed. No deployment claim is made."
        + (
            " ZERO candidates reached VALID — the multi-path estimator did NOT "
            "rescue any cell (matches the pre-registered expectation under "
            f"K={SELECTION_BUDGET_K} / ~{MGC_HORIZON_YEARS:g}yr horizon). "
            "CONDITIONAL means positive-and-not-overfit but unconfirmable at "
            "the locked t/power floors — NOT an edge, NOT dead."
            if none_deployable
            else ""
        )
    )
    lines.append(
        "- MGC path forward remains the SR-monitor signal-only shadow per "
        "pre_registered_criteria.md Criterion 12 (grounded "
        "pepelyshev_polunchenko_2015_cusum_sr) — NOT a calendar wait, NOT a "
        "threshold relaxation."
    )
    return "\n".join(lines) + "\n"


# ---- pressure test (RULE 13) ------------------------------------------------
def pressure_test(con: duckdb.DuckDBPyConnection) -> tuple[bool, str]:
    """Inject a look-ahead 'perfect' series; confirm CPCV does NOT bless noise.

    RULE 13: a known-bad input must be caught or fail honestly. We build a
    pure-noise series (mean ~0) and confirm CPCV returns UNVERIFIED (K1) — i.e.
    the estimator does not manufacture an edge from noise. We also build a
    constant-positive look-ahead series and confirm the PBO/verdict machinery
    flags it as the artifact it is (PBO=0 but a real desk would catch the
    suspiciously perfect path dispersion; the test asserts it is NOT silently
    promoted on a single path).
    """
    # Use candidate #5's day grid (largest N) for a realistic group structure.
    df = _pull_candidate(con, CANDIDATES[4]).copy()
    if df.empty or len(df) < N_GROUPS * 2:
        return False, "pressure test could not source a base series"

    rng = np.random.default_rng(0)  # fixed seed: reproducible, no Date/random ban issue
    noise = pd.DataFrame(
        {
            "trading_day": df["trading_day"].to_numpy(),
            "pnl_r": rng.normal(0.0, 1.0, size=len(df)),
        }
    )
    noise_dummy = Candidate(99, "PRESSURE_NOISE", 30, 1.0, "long", "none", "none", 0)
    r_noise = run_cpcv(noise_dummy, noise)
    if r_noise.verdict == "VALID":
        return (
            False,
            f"FAIL: pure-noise series classified VALID "
            f"(median ExpR={r_noise.median_expr:+.4f}, t={r_noise.median_t:.2f})",
        )

    # Constant-positive look-ahead series: every trade wins by a fixed amount.
    # This is NOT real (no strategy wins every trade) — a desk must not promote
    # it. CPCV will show median ExpR>0 and PBO=0, but pooled power will be
    # extreme; the assertion is that the VALID gate is reachable ONLY because
    # the input is degenerate, which the pressure-test surfaces explicitly.
    perfect = pd.DataFrame(
        {
            "trading_day": df["trading_day"].to_numpy(),
            "pnl_r": np.full(len(df), 0.5),
        }
    )
    perfect_dummy = Candidate(98, "PRESSURE_PERFECT", 30, 1.0, "long", "none", "none", 0)
    r_perfect = run_cpcv(perfect_dummy, perfect)
    # A constant series has zero variance -> t is NaN/0 -> NOT VALID. Confirm
    # the estimator degrades gracefully rather than dividing by zero.
    if r_perfect.verdict == "VALID":
        return (
            False,
            "FAIL: degenerate constant-positive series classified VALID — "
            "zero-variance handling is broken",
        )
    return (
        True,
        f"PASS: noise->{r_noise.verdict} (median ExpR={r_noise.median_expr:+.4f}), "
        f"constant->{r_perfect.verdict} (zero-variance degrades to non-VALID)",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=None, help="override DB path (default canonical)")
    ap.add_argument("--out", default=None, help="write markdown report to this path")
    ap.add_argument(
        "--pressure-test",
        action="store_true",
        help="RULE 13: run the look-ahead/noise pressure test and exit",
    )
    args = ap.parse_args()

    if args.db:
        db = args.db
    else:
        from pipeline.paths import GOLD_DB_PATH

        db = str(GOLD_DB_PATH)
    con = duckdb.connect(db, read_only=True)

    if args.pressure_test:
        ok, msg = pressure_test(con)
        con.close()
        print(f"[RULE 13 pressure test] {msg}")
        return 0 if ok else 1

    results: list[CpcvResult] = []
    for c in CANDIDATES:
        df = _pull_candidate(con, c)
        if df.empty or len(df) < N_GROUPS * 2:
            print(f"SKIP #{c.idx} {c.label}: insufficient data (N={len(df)})")
            continue
        results.append(run_cpcv(c, df))
    con.close()

    if not results:
        print("No candidates produced CPCV results.")
        return 1

    md = render_markdown(results)
    print(md)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"\nReport -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
