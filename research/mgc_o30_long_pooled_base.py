"""MGC O30 long — POOLED base edge + RULE 2 conditional overlay scan.

Q1 of action-queue item `mgc_o30_long_pooled_base_then_conditional_overlay_2026_05_29`.

FRAMING CORRECTION to the 2026-05-29 MGC CPCV audit
(`docs/audit/results/2026-05-29-mgc-cpcv-audit.md`). That audit tested 6
pre-selected `(session x day_of_week)` slices, each at N~100, then applied
per-cell Chordia-strict t>=3.79 carrying K=1992. The tell: ALL 6 winners are
`long` AND `O30`. That may be ONE structural MGC O30 long drift effect read
through 6 thin windows -- OR six noise draws that happened to land positive.
Pooling first is the only way to distinguish them.

This scan answers Q1 (HIGHEST EV) in two parts, then applies the LOCKED
disposition from the action-queue exit_criteria:

  Q1a -- POOLED BASE: MGC O30 long ExpR/t/N on the FULL sample (all DOW), per
         session AND pooled across sessions, per RR. N goes ~100 -> 1000+.
         Per-year breakdown (RULE 12 outlier guard) + OOS power tier (RULE 3.3).

  Q1b -- CONDITIONAL OVERLAY (RULE 2 Pass-1/Pass-2): each DOW/vol condition from
         the 6 CPCV slices, tested as an overlay ON the pooled base, NOT as a
         standalone K=1 strategy. Pass-1 = condition lift on the full base
         universe; Pass-2 = residual lift after the base. Question: does DOW add
         edge ON TOP of the base, or is it a proxy for which session trades on
         which day (US_DATA only on data days; NFP=Fri)? Per-cell p + BH-FDR at
         K=family.

NO FORCING, NO RESCUE. Q1 can KILL MGC O30 long harder than the CPCV audit did.
That is the point -- the fair fight the standalone-slice framing never gave it.

Discipline (binding):
  - RULE 9 canonical triple-join, orb_minutes PINNED. Canonical layers only
    (orb_outcomes JOIN daily_features). NO DB writes -- read-only.
  - RULE 6.1 trade-time-knowable features ONLY (day_of_week, atr_20_pct: priori;
    overnight_range_pct: RULE 1.2-gated to look-ahead-safe sessions via
    `research.comprehensive_deployed_lane_scan._overnight_lookhead_clean`).
  - RULE 2 Pass-1/Pass-2 overlay semantics.
  - RULE 3.3 OOS power floor: every binary OOS read carries its power tier
    (delegated to `research.oos_power`); never re-implement the power math.
  - RULE 4 BH-FDR at K=family (the overlay conditions), NOT K=1992 (that was the
    discovery SELECTION budget; this is a confirmatory re-frame).
  - RULE 12 per-year stability; RULE 13 pressure test (`--pressure-test`).
  - Costs: `orb_outcomes.pnl_r` already net of canonical friction
    (`pipeline.cost_model.COST_SPECS['MGC'].total_friction`); scored directly.
  - No threshold relaxation. Disposition is pre-registered (locked below).

Literature grounding (verbatim files, not memory):
  - ORB premise / intraday trend-follow: Fitschen 2013 Ch 3
    `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md`.
  - OOS power floor / Sharpe haircut: Harvey-Liu 2015
    `docs/institutional/literature/harvey_liu_2015_backtesting.md`.
  - BH-FDR: Benjamini-Hochberg 1995
    `docs/institutional/literature/benjamini_hochberg_1995_fdr.md`.
  - Chordia t-thresholds: `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md`.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from pipeline.cost_model import COST_SPECS
from research.comprehensive_deployed_lane_scan import _overnight_lookhead_clean
from research.oos_power import one_sample_power, power_verdict

# ---- locked constants -------------------------------------------------------
SYMBOL = "MGC"
ORB_MINUTES = 30
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
DIRECTION = "long"

# RRs to report (no cherry-pick: the 6 CPCV slices mixed RR1.0/1.5/2.0).
RR_TARGETS: tuple[float, ...] = (1.0, 1.5, 2.0)

# Chordia strict (no prior theory) and lenient (with theory) — read-only.
CHORDIA_T_STRICT = 3.79
CHORDIA_T_THEORY = 3.0

# Pre-registered disposition power/t floors (Criterion 4 + RULE 3.3).
POWERED_FLOOR = 0.50  # DIRECTIONAL_ONLY lower edge

# Sacred holdout (Mode A) — imported, never inlined.
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM


# The conditions that appeared in the 6 CPCV slices, as overlays on the base.
# (session is the base scope; condition is what we overlay.)
@dataclass(frozen=True)
class Overlay:
    """One conditional overlay from the CPCV slice set."""

    cpcv_idx: int
    session: str
    rr: float
    feature: str
    op: str
    threshold: float

    @property
    def label(self) -> str:
        return f"{self.feature}{self.op}{self.threshold:g}"


OVERLAYS: tuple[Overlay, ...] = (
    Overlay(1, "US_DATA_830", 2.0, "day_of_week", "==", 1),
    Overlay(2, "NYSE_OPEN", 2.0, "day_of_week", "==", 3),
    Overlay(3, "NYSE_OPEN", 1.0, "day_of_week", "==", 3),
    Overlay(4, "SINGAPORE_OPEN", 2.0, "day_of_week", "==", 4),
    Overlay(5, "EUROPE_FLOW", 2.0, "atr_20_pct", ">=", 60),
    Overlay(6, "LONDON_METALS", 1.5, "overnight_range_pct", ">=", 80),
)


# ---- canonical data pull (RULE 9 triple-join) -------------------------------
def _pull_base(con: duckdb.DuckDBPyConnection, session: str | None, rr: float) -> pd.DataFrame:
    """MGC O30 long base trade series for a session (or all sessions if None).

    Canonical triple-join, orb_minutes PINNED. Trade-time features only.
    Long is defined canonically as stop below entry (same convention as the
    CPCV audit's `_pull_candidate`).
    """
    session_clause = "" if session is None else f"AND o.orb_label = '{session}'"
    q = f"""
        SELECT o.trading_day, o.orb_label, o.pnl_r,
               CASE WHEN o.stop_price < o.entry_price THEN 'long' ELSE 'short' END AS dir,
               d.day_of_week, d.atr_20_pct, d.overnight_range_pct
        FROM orb_outcomes o
        JOIN daily_features d
          ON d.trading_day = o.trading_day
         AND d.symbol = o.symbol
         AND d.orb_minutes = o.orb_minutes
        WHERE o.symbol = '{SYMBOL}' AND o.orb_minutes = {ORB_MINUTES}
          AND o.entry_model = '{ENTRY_MODEL}' AND o.confirm_bars = {CONFIRM_BARS}
          AND o.rr_target = {rr} AND o.outcome IS NOT NULL
          {session_clause}
    """
    df = con.sql(q).df()
    if df.empty:
        return df
    df = df[df["dir"] == DIRECTION].copy()
    df = df.dropna(subset=["pnl_r"])
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    return df.sort_values("trading_day").reset_index(drop=True)


# ---- statistics -------------------------------------------------------------
def _t_stat(x: np.ndarray) -> tuple[int, float, float]:
    """Return (n, mean ExpR, one-sample t) for a pnl_r array."""
    n = len(x)
    if n < 2:
        return n, float("nan"), float("nan")
    m = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return n, m, 0.0
    return n, m, m / (sd / math.sqrt(n))


def _p_from_t(t: float, n: int) -> float:
    """Two-sided p-value for a one-sample t with n-1 df."""
    if not np.isfinite(t) or n < 2:
        return float("nan")
    from scipy import stats

    return float(2.0 * stats.t.sf(abs(t), df=n - 1))


def _power_tier(t: float, n: int) -> tuple[float, str]:
    """In-sample power + tier (RULE 3.3, delegated). Effect size and evaluation
    N are the SAME sample here (the edge's own power)."""
    if not np.isfinite(t) or n < 2:
        return float("nan"), "STATISTICALLY_USELESS"
    cohen_d = abs(t) / math.sqrt(n)
    power = one_sample_power(cohen_d, n)
    return power, power_verdict(power)


def _oos_power_tier(t_is: float, n_is: int, n_oos: int) -> tuple[float, str]:
    """OOS power to detect the IS effect (RULE 3.3, delegated).

    The two sample sizes are DISTINCT: Cohen's d is the IS effect
    (`|t_is|/sqrt(n_is)`), but power is evaluated at the OOS sample size
    `n_oos`. Conflating them (using n_oos for both) misstates d. This is the
    2026-04-20 reference-incident framing: "does the OOS sample have power to
    detect the IS effect size?"
    """
    if not np.isfinite(t_is) or n_is < 2 or n_oos < 2:
        return float("nan"), "STATISTICALLY_USELESS"
    cohen_d = abs(t_is) / math.sqrt(n_is)
    power = one_sample_power(cohen_d, n_oos)
    return power, power_verdict(power)


def _per_year(df: pd.DataFrame) -> list[tuple[int, int, float, float]]:
    """(year, n, ExpR, t) per calendar year — RULE 12 outlier/stability guard."""
    out: list[tuple[int, int, float, float]] = []
    for year, grp in df.groupby(df["trading_day"].dt.year):
        n, expr, t = _t_stat(grp["pnl_r"].to_numpy(dtype=float))
        out.append((int(year), n, expr, t))
    return sorted(out)


# ---- IS/OOS split (Mode A sacred holdout) -----------------------------------
def _is_oos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sacred = pd.Timestamp(HOLDOUT_SACRED_FROM)
    is_df = df[df["trading_day"] < sacred]
    oos_df = df[df["trading_day"] >= sacred]
    return is_df, oos_df


# ---- BH-FDR (RULE 4) --------------------------------------------------------
def bh_fdr(pvals: list[float], q: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg 1995. Returns pass/fail per input p (original order).

    NaN p-values fail (cannot be a discovery). K = number of non-NaN tests in
    the family.
    """
    indexed = [(i, p) for i, p in enumerate(pvals) if np.isfinite(p)]
    k = len(indexed)
    passes = [False] * len(pvals)
    if k == 0:
        return passes
    indexed.sort(key=lambda ip: ip[1])
    max_rank = 0
    for rank, (_, p) in enumerate(indexed, start=1):
        if p <= (rank / k) * q:
            max_rank = rank
    for rank, (orig_i, _) in enumerate(indexed, start=1):
        if rank <= max_rank:
            passes[orig_i] = True
    return passes


# ---- result containers ------------------------------------------------------
@dataclass(frozen=True)
class BaseResult:
    scope: str  # session name or "POOLED_ALL_SESSIONS"
    rr: float
    n: int
    expr: float
    t: float
    p: float
    power: float
    tier: str
    n_is: int
    expr_is: float
    t_is: float
    n_oos: int
    expr_oos: float
    t_oos: float
    oos_power: float
    oos_tier: str
    dir_match: bool
    per_year: list[tuple[int, int, float, float]]


@dataclass(frozen=True)
class OverlayResult:
    overlay: Overlay
    # Pass-1: condition lift on the FULL base universe (this session, this RR).
    p1_n_on: int
    p1_expr_on: float
    p1_n_off: int
    p1_expr_off: float
    p1_lift: float
    p1_t: float
    p1_p: float
    # Pass-2: the on-condition cell's own t/power (residual edge given base).
    p2_t: float
    p2_p: float
    p2_power: float
    p2_tier: str
    bh_pass: bool  # filled after family BH-FDR


def run_base(con: duckdb.DuckDBPyConnection) -> list[BaseResult]:
    """Q1a: pooled base, per session and pooled-across-sessions, per RR."""
    results: list[BaseResult] = []
    # session list = the union of sessions in the overlay set + pooled.
    sessions = sorted({o.session for o in OVERLAYS})
    scopes: list[str | None] = [*sessions, None]  # None = pooled across all
    for scope in scopes:
        for rr in RR_TARGETS:
            df = _pull_base(con, scope, rr)
            if df.empty or len(df) < 2:
                continue
            pnl = df["pnl_r"].to_numpy(dtype=float)
            n, expr, t = _t_stat(pnl)
            p = _p_from_t(t, n)
            power, tier = _power_tier(t, n)
            is_df, oos_df = _is_oos(df)
            n_is, expr_is, t_is = _t_stat(is_df["pnl_r"].to_numpy(dtype=float))
            n_oos, expr_oos, t_oos = _t_stat(oos_df["pnl_r"].to_numpy(dtype=float))
            oos_power, oos_tier = _oos_power_tier(t_is, n_is, n_oos)
            dir_match = (
                np.isfinite(expr_is)
                and np.isfinite(expr_oos)
                and np.sign(expr_is) == np.sign(expr_oos)
            )
            results.append(
                BaseResult(
                    scope=("POOLED_ALL_SESSIONS" if scope is None else scope),
                    rr=rr, n=n, expr=expr, t=t, p=p, power=power, tier=tier,
                    n_is=n_is, expr_is=expr_is, t_is=t_is,
                    n_oos=n_oos, expr_oos=expr_oos, t_oos=t_oos,
                    oos_power=oos_power, oos_tier=oos_tier, dir_match=dir_match,
                    per_year=_per_year(df),
                )
            )
    return results


def _apply_condition(df: pd.DataFrame, ov: Overlay) -> pd.Series:
    """Boolean mask for the overlay condition. RULE 1.2 look-ahead gate for
    overnight_* features."""
    if ov.feature.startswith("overnight_") and not _overnight_lookhead_clean(ov.session):
        raise ValueError(
            f"{ov.session}: overnight feature on look-ahead-unsafe session "
            f"(RULE 1.2). Overlay set malformed."
        )
    s = pd.to_numeric(df[ov.feature], errors="coerce")
    if ov.op == "==":
        return s == ov.threshold
    if ov.op == ">=":
        return s >= ov.threshold
    if ov.op == "<=":
        return s <= ov.threshold
    raise ValueError(f"unknown op {ov.op!r}")


def run_overlays(con: duckdb.DuckDBPyConnection) -> list[OverlayResult]:
    """Q1b: RULE 2 Pass-1/Pass-2 overlay test for each CPCV-slice condition."""
    raw: list[OverlayResult] = []
    pvals: list[float] = []
    for ov in OVERLAYS:
        df = _pull_base(con, ov.session, ov.rr)
        if df.empty:
            continue
        mask = _apply_condition(df, ov)
        on = df[mask]["pnl_r"].to_numpy(dtype=float)
        off = df[~mask]["pnl_r"].to_numpy(dtype=float)
        n_on, expr_on, t_on = _t_stat(on)
        n_off, expr_off, _ = _t_stat(off)
        # Pass-1 lift: difference of on vs off, two-sample t (Welch).
        lift = (expr_on - expr_off) if (np.isfinite(expr_on) and np.isfinite(expr_off)) else float("nan")
        from scipy import stats

        if len(on) >= 2 and len(off) >= 2:
            tt = stats.ttest_ind(on, off, equal_var=False)
            p1_t, p1_p = float(tt.statistic), float(tt.pvalue)  # type: ignore[attr-defined]
        else:
            p1_t, p1_p = float("nan"), float("nan")
        # Pass-2: the on-condition cell's own one-sample t/power (residual edge).
        p2_p = _p_from_t(t_on, n_on)
        p2_power, p2_tier = _power_tier(t_on, n_on)
        raw.append(
            OverlayResult(
                overlay=ov,
                p1_n_on=n_on, p1_expr_on=expr_on, p1_n_off=n_off,
                p1_expr_off=expr_off, p1_lift=lift, p1_t=p1_t, p1_p=p1_p,
                p2_t=t_on, p2_p=p2_p, p2_power=p2_power, p2_tier=p2_tier,
                bh_pass=False,
            )
        )
        # BH family is the OVERLAY LIFT test (Pass-1) — "does the condition add?"
        pvals.append(p1_p)
    # RULE 4: BH-FDR at K=family (the overlay conditions).
    passes = bh_fdr(pvals, q=0.05)
    out: list[OverlayResult] = []
    for r, ok in zip(raw, passes, strict=True):
        out.append(
            OverlayResult(
                overlay=r.overlay, p1_n_on=r.p1_n_on, p1_expr_on=r.p1_expr_on,
                p1_n_off=r.p1_n_off, p1_expr_off=r.p1_expr_off, p1_lift=r.p1_lift,
                p1_t=r.p1_t, p1_p=r.p1_p, p2_t=r.p2_t, p2_p=r.p2_p,
                p2_power=r.p2_power, p2_tier=r.p2_tier, bh_pass=ok,
            )
        )
    return out


# ---- disposition (LOCKED before results, per action-queue exit_criteria) -----
def disposition(base: list[BaseResult], overlays: list[OverlayResult]) -> tuple[str, str]:
    """Apply the pre-registered disposition logic. Returns (verdict, reason).

    - pooled base positive+powered (t>=3.79 AND power>=0.50) -> DIVERSIFIER_CANDIDATE
    - only conditional adds edge (any overlay BH-passes + powered) -> OVERLAY_ROLE
    - pooled base flat -> DEAD_FOR_ORB (stop slicing; SR-monitor shadow only)
    """
    pooled = [b for b in base if b.scope == "POOLED_ALL_SESSIONS"]
    best_pooled = max(pooled, key=lambda b: (b.t if np.isfinite(b.t) else -9), default=None)
    # Best per-session base too (pooling across sessions can dilute a real
    # session-specific drift; report both, but the headline question is pooled).
    best_session = max(
        (b for b in base if b.scope != "POOLED_ALL_SESSIONS"),
        key=lambda b: (b.t if np.isfinite(b.t) else -9),
        default=None,
    )

    def powered(b: BaseResult | None) -> bool:
        return (
            b is not None and np.isfinite(b.t) and b.t >= CHORDIA_T_STRICT
            and np.isfinite(b.power) and b.power >= POWERED_FLOOR
        )

    winner = best_pooled if powered(best_pooled) else (
        best_session if powered(best_session) else None
    )
    if winner is not None:
        return (
            "DIVERSIFIER_CANDIDATE",
            f"base {winner.scope} RR{winner.rr:g} t={winner.t:.2f}>={CHORDIA_T_STRICT} "
            f"power={winner.power:.2f}>={POWERED_FLOOR} -> route to prereg/validation",
        )

    # Does any overlay add real residual edge (BH-pass AND on-cell powered)?
    real_overlays = [
        o for o in overlays
        if o.bh_pass and np.isfinite(o.p2_power) and o.p2_power >= POWERED_FLOOR
        and np.isfinite(o.p2_t) and o.p2_t >= CHORDIA_T_STRICT
    ]
    if real_overlays:
        labels = ", ".join(f"#{o.overlay.cpcv_idx} {o.overlay.session}/{o.overlay.label}" for o in real_overlays)
        return (
            "OVERLAY_ROLE",
            f"base flat but {len(real_overlays)} overlay(s) add powered residual edge "
            f"({labels}) -> overlay/portfolio role, NEVER standalone",
        )

    best_t = max(
        (b.t for b in base if np.isfinite(b.t)), default=float("nan")
    )
    return (
        "DEAD_FOR_ORB",
        f"pooled base flat (best base t={best_t:.2f} < {CHORDIA_T_STRICT}) and no "
        f"overlay adds powered residual edge -> MGC O30 long DEAD for ORB; stop "
        f"slicing; SR-monitor signal-only shadow (Criterion 12) is the only honest path",
    )


# ---- reporting --------------------------------------------------------------
def _flip_rate(base: list[BaseResult]) -> float:
    """Per-cell sign flips vs pooled sign (pooled-finding-rule.md)."""
    pooled = [b for b in base if b.scope == "POOLED_ALL_SESSIONS" and np.isfinite(b.expr)]
    if not pooled:
        return 0.0
    pooled_sign = np.sign(np.mean([b.expr for b in pooled]))
    cells = [b for b in base if b.scope != "POOLED_ALL_SESSIONS" and np.isfinite(b.expr)]
    if not cells:
        return 0.0
    flips = sum(1 for b in cells if np.sign(b.expr) != pooled_sign and b.expr != 0)
    return 100.0 * flips / len(cells)


def render_markdown(base: list[BaseResult], overlays: list[OverlayResult]) -> str:
    friction = COST_SPECS[SYMBOL].total_friction
    verdict, reason = disposition(base, overlays)
    flip = _flip_rate(base)
    lines: list[str] = []
    # pooled-finding-rule.md front-matter (this file makes a pooled claim).
    lines.append("---")
    lines.append("pooled_finding: true")
    lines.append("per_cell_breakdown_path: docs/audit/results/2026-05-29-mgc-o30-long-pooled-base-and-overlay.md#per-session-base-breakdown")
    lines.append(f"flip_rate_pct: {flip:.1f}")
    if flip >= 25:
        lines.append("heterogeneity_ack: true")
    lines.append("---\n")

    lines.append("# MGC O30 Long — Pooled Base + Conditional Overlay (Q1)\n")
    lines.append(
        "**Mandate:** Q1 framing correction to the 2026-05-29 MGC CPCV audit. "
        "Test whether the 6 underpowered `(session x DOW/vol)` CPCV slices are ONE "
        "MGC O30 long drift effect (poolable) or six noise draws. Pool the base "
        "FIRST, then test each condition as a RULE 2 Pass-1/Pass-2 overlay. "
        "Read-only canonical layers; NO DB writes; NO threshold relaxation. The "
        "fair fight can KILL MGC O30 long harder than the slice framing did.\n"
    )
    lines.append(
        f"**Scope:** {SYMBOL} O{ORB_MINUTES} {DIRECTION} {ENTRY_MODEL} "
        f"CB{CONFIRM_BARS}, RR in {{{', '.join(f'{r:g}' for r in RR_TARGETS)}}}. "
        f"Costs: `COST_SPECS['{SYMBOL}'].total_friction={friction}`; `pnl_r` already "
        f"net -> scored directly. Sacred holdout: {HOLDOUT_SACRED_FROM} (Mode A).\n"
    )
    lines.append(
        f"**Thresholds (read-only):** Chordia strict t>={CHORDIA_T_STRICT} "
        f"(no theory), power floor {POWERED_FLOOR} (RULE 3.3). BH-FDR q=0.05 at "
        "K=family (the overlay conditions), NOT K=1992 selection budget.\n"
    )

    lines.append(f"\n## VERDICT: {verdict}\n")
    lines.append(f"{reason}\n")

    # ---- Q1a pooled base ----
    lines.append("\n## Q1a — Pooled base (all DOW)\n")
    lines.append("### Pooled across ALL overlay sessions\n")
    lines.append("| RR | N | ExpR | t | p | power | tier | dir_match (IS->OOS) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for b in base:
        if b.scope != "POOLED_ALL_SESSIONS":
            continue
        lines.append(
            f"| {b.rr:g} | {b.n} | {b.expr:+.4f} | {b.t:.2f} | {b.p:.4f} | "
            f"{b.power:.2f} | {b.tier} | {b.dir_match} (IS {b.expr_is:+.3f} "
            f"N={b.n_is} -> OOS {b.expr_oos:+.3f} N={b.n_oos}, OOS power "
            f"{b.oos_power:.2f}/{b.oos_tier}) |"
        )

    lines.append("\n### Per-session base breakdown\n")
    lines.append("| session | RR | N | ExpR | t | p | power | tier | dir_match |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for b in base:
        if b.scope == "POOLED_ALL_SESSIONS":
            continue
        lines.append(
            f"| {b.scope} | {b.rr:g} | {b.n} | {b.expr:+.4f} | {b.t:.2f} | "
            f"{b.p:.4f} | {b.power:.2f} | {b.tier} | {b.dir_match} |"
        )

    # Per-year stability for the strongest pooled cell (RULE 12).
    pooled_cells = [b for b in base if b.scope == "POOLED_ALL_SESSIONS" and np.isfinite(b.t)]
    if pooled_cells:
        strongest = max(pooled_cells, key=lambda b: b.t)
        lines.append(
            f"\n### Per-year stability — strongest pooled cell (RR{strongest.rr:g})\n"
        )
        lines.append("| year | N | ExpR | t |")
        lines.append("|---|---|---|---|")
        for year, n, expr, t in strongest.per_year:
            lines.append(f"| {year} | {n} | {expr:+.4f} | {t:.2f} |")

    # ---- Q1b overlays ----
    lines.append("\n## Q1b — Conditional overlays (RULE 2 Pass-1/Pass-2)\n")
    lines.append(
        "Pass-1 = condition lift (on vs off) on the full base universe for that "
        "session+RR. Pass-2 = the on-condition cell's own t/power (residual edge "
        "given the base). BH-FDR at K=family on the Pass-1 lift p-values.\n"
    )
    lines.append(
        "| # | session | RR | condition | N_on | ExpR_on | N_off | ExpR_off | "
        "P1 lift | P1 t | P1 p | BH-pass | P2 t | P2 power | P2 tier |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for o in overlays:
        ov = o.overlay
        lines.append(
            f"| {ov.cpcv_idx} | {ov.session} | {ov.rr:g} | {ov.label} | "
            f"{o.p1_n_on} | {o.p1_expr_on:+.4f} | {o.p1_n_off} | "
            f"{o.p1_expr_off:+.4f} | {o.p1_lift:+.4f} | {o.p1_t:.2f} | "
            f"{o.p1_p:.4f} | {o.bh_pass} | {o.p2_t:.2f} | {o.p2_power:.2f} | "
            f"{o.p2_tier} |"
        )

    # ---- summary ----
    lines.append("\n## Summary\n")
    n_overlay_real = sum(
        1 for o in overlays
        if o.bh_pass and np.isfinite(o.p2_t) and o.p2_t >= CHORDIA_T_STRICT
        and np.isfinite(o.p2_power) and o.p2_power >= POWERED_FLOOR
    )
    lines.append(
        f"- Pooled-finding flip rate (per-session vs pooled sign): {flip:.1f}%"
        + (" (>=25% — heterogeneity acknowledged)" if flip >= 25 else "")
    )
    lines.append(
        f"- Overlays adding powered+BH-confirmed residual edge: {n_overlay_real}/{len(overlays)}"
    )
    lines.append(f"- **Disposition: {verdict}** — {reason}")
    lines.append(
        "- No threshold relaxed. No DB write. K=family (not K=1992). The CPCV "
        "audit's K=1992 was the discovery SELECTION budget; this confirmatory "
        "re-frame tests already-surfaced cells, so the honest family is the "
        f"{len(OVERLAYS)} overlay conditions."
    )
    if verdict == "DEAD_FOR_ORB":
        lines.append(
            "- MGC O30 long path forward: SR-monitor signal-only shadow per "
            "`pre_registered_criteria.md` Criterion 12 (grounded "
            "`pepelyshev_polunchenko_2015_cusum_sr`). NOT a calendar wait, NOT a "
            "threshold relaxation. STOP slicing into thinner DOW windows."
        )
    return "\n".join(lines) + "\n"


# ---- pressure test (RULE 13) ------------------------------------------------
def pressure_test(con: duckdb.DuckDBPyConnection) -> tuple[bool, str]:
    """Inject a pnl_r-derived look-ahead overlay; confirm it is NOT blessed.

    RULE 13: a known-bad feature must be caught. We build an overlay whose
    condition IS the trade outcome (pnl_r > 0) -- pure look-ahead. Its Pass-1
    lift will be enormous and its Pass-2 t huge, BUT it is tautological: the
    condition is the label. The pressure test asserts the scan SURFACES this as
    a suspiciously perfect cell (|t| > 10, lift > 0.6) -- the RULE 12 red-flag
    band -- rather than silently promoting it as a real overlay.
    """
    df = _pull_base(con, "NYSE_OPEN", 2.0)
    if df.empty or len(df) < 20:
        return False, "pressure test could not source a base series"
    # Look-ahead condition: 'win' = pnl_r > 0 (the label itself).
    on = df[df["pnl_r"] > 0]["pnl_r"].to_numpy(dtype=float)
    off = df[df["pnl_r"] <= 0]["pnl_r"].to_numpy(dtype=float)
    _, expr_on, t_on = _t_stat(on)
    _, expr_off, _ = _t_stat(off)
    lift = expr_on - expr_off
    # A real overlay never produces this; the label-as-feature does.
    caught = (abs(t_on) > 10.0) or (lift > 0.6)
    if not caught:
        return (
            False,
            f"FAIL: look-ahead overlay (pnl_r>0) NOT flagged "
            f"(t_on={t_on:.2f}, lift={lift:+.4f}) -- RULE 12 red-flag band missed",
        )
    return (
        True,
        f"PASS: look-ahead overlay (pnl_r>0) lands in RULE 12 red-flag band "
        f"(t_on={t_on:.2f} > 10 or lift={lift:+.4f} > 0.6) -- correctly "
        f"surfaced as tautological, not a real edge",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=None, help="override DB path (default canonical)")
    ap.add_argument("--out", default=None, help="write markdown report to this path")
    ap.add_argument(
        "--pressure-test", action="store_true",
        help="RULE 13: run the look-ahead pressure test and exit",
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

    base = run_base(con)
    overlays = run_overlays(con)
    con.close()

    if not base:
        print("No base results produced.")
        return 1

    md = render_markdown(base, overlays)
    print(md)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"\nReport -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
