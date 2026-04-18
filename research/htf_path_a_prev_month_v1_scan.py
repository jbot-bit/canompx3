#!/usr/bin/env python3
"""HTF Path A prev-month v1 family scan (Pathway A, K_family=24).

Pre-registered at:
  docs/audit/hypotheses/2026-04-18-htf-path-a-prev-month-v1.yaml

Reads ONLY canonical tables (daily_features + orb_outcomes). No writes to
validated_setups or experimental_strategies. Emits a single result markdown.

Holdout: Mode A, boundary imported from trading_app.holdout_policy
(not hardcoded). IS = trading_day < HOLDOUT_SACRED_FROM; OOS is the
complement within the data window.

Output: docs/audit/results/2026-04-18-htf-path-a-prev-month-v1-scan.md

Usage:
  DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/htf_path_a_prev_month_v1_scan.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
from scipy import stats as _sstats

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

# =========================================================================
# PRE-REGISTERED FAMILY (locked by YAML; DO NOT modify without re-prereg)
# =========================================================================

YAML_PATH = PROJECT_ROOT / "docs/audit/hypotheses/2026-04-18-htf-path-a-prev-month-v1.yaml"
RESULT_PATH = PROJECT_ROOT / "docs/audit/results/2026-04-18-htf-path-a-prev-month-v1-scan.md"

INSTRUMENTS = ("MNQ", "MES")
SESSIONS = ("TOKYO_OPEN", "EUROPE_FLOW", "NYSE_OPEN")
ORB_MINUTES = 15
RR_TARGETS = (1.5, 2.0)
DIRECTIONS = ("long", "short")
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1

# BH-FDR framings
K_GLOBAL = 24       # whole run
K_FAMILY = 24       # one family
K_INSTRUMENT = 12   # per instrument
K_SESSION = 8       # per session
K_DIRECTION = 12    # per direction

BH_Q = 0.05

# Criterion 9 eras (Amendment 2.8 horizon: MNQ/MES real-micro 2019-05-06+)
ERAS: tuple[tuple[str, date, date], ...] = (
    ("2019-2020", date(2019, 1, 1), date(2020, 12, 31)),
    ("2021-2022", date(2021, 1, 1), date(2022, 12, 31)),
    ("2023",      date(2023, 1, 1), date(2023, 12, 31)),
    ("2024-2025", date(2024, 1, 1), date(2025, 12, 31)),
)

# Gate thresholds (all from authority documents — see YAML)
FIRE_RATE_MIN = 0.05   # RULE 8.1
FIRE_RATE_MAX = 0.95   # RULE 8.1
N_ON_MIN = 50          # RULE 4.1 discovery-candidacy floor
CHORDIA_T = 3.79       # Criterion 4 no-theory
WFE_MIN = 0.50         # Criterion 6
WFE_LEAKAGE = 0.95     # RULE 3.2 leakage-suspect
ERA_EXPR_MIN = -0.05   # Criterion 9
ERA_N_EXEMPT = 50      # Criterion 9 exemption threshold
TAUTOLOGY_MAX = 0.70   # RULE 7 (T0)
AO_WR_SPREAD = 0.03    # RULE 8.2
AO_DELTA_IS = 0.10     # RULE 8.2


# =========================================================================
# DATA STRUCTURES
# =========================================================================

@dataclass
class CellResult:
    instrument: str
    session: str
    direction: str
    rr: float
    predicate: str

    # Base (unfiltered, same lane / direction)
    n_is_base: int = 0
    n_oos_base: int = 0
    expr_is_base: float | None = None
    expr_oos_base: float | None = None
    wr_is_base: float | None = None

    # On (HTF filter fires)
    n_is_on: int = 0
    n_oos_on: int = 0
    expr_is_on: float | None = None
    expr_oos_on: float | None = None
    wr_is_on: float | None = None
    std_is_on: float | None = None
    std_oos_on: float | None = None
    sharpe_is_on: float | None = None
    sharpe_oos_on: float | None = None

    # Derived
    delta_is: float | None = None
    delta_oos: float | None = None
    dir_match: bool | None = None
    fire_rate_is: float | None = None
    t_stat: float | None = None
    raw_p: float | None = None
    wfe: float | None = None

    # Flags
    tautology_corr: float | None = None
    is_tautology: bool = False
    is_arithmetic_only: bool = False

    # Era stability
    era_expr: dict[str, dict[str, Any]] = field(default_factory=dict)
    era_stable: bool | None = None

    # BH-FDR q-values
    q_global: float | None = None
    q_family: float | None = None
    q_instrument: float | None = None
    q_session: float | None = None
    q_direction: float | None = None
    passes_bh_family: bool = False

    verdict: str = "UNKNOWN"


# =========================================================================
# SQL HELPERS
# =========================================================================

def _predicate_sql(session: str, direction: str) -> str:
    if direction == "long":
        return (
            f"d.orb_{session}_break_dir = 'long' "
            f"AND d.prev_month_high IS NOT NULL "
            f"AND d.orb_{session}_high > d.prev_month_high"
        )
    return (
        f"d.orb_{session}_break_dir = 'short' "
        f"AND d.prev_month_low IS NOT NULL "
        f"AND d.orb_{session}_low < d.prev_month_low"
    )


def _load_cell_trades(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    session: str,
    direction: str,
    rr: float,
    on_filter: bool,
) -> list[tuple[date, float, str, float | None]]:
    """Returns [(trading_day, pnl_r, outcome, orb_size)] for the cell.

    Direction is resolved via daily_features.orb_{session}_break_dir (orb_outcomes
    has no direction column — trades inherit the break direction by construction).
    """
    filter_sql = f"AND ({_predicate_sql(session, direction)})" if on_filter else ""
    sql = f"""
        SELECT o.trading_day,
               o.pnl_r,
               o.outcome,
               d.orb_{session}_size AS orb_size
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
          AND d.orb_{session}_break_dir = ?
          {filter_sql}
        ORDER BY o.trading_day
    """
    return con.execute(
        sql,
        [instrument, session, ORB_MINUTES, ENTRY_MODEL, CONFIRM_BARS, rr, direction],
    ).fetchall()


def _tautology_corr(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    session: str,
    direction: str,
    holdout: date,
) -> float | None:
    """Pearson corr between HTF-fire binary and orb_size (proxy for ORB_G5 family).

    Computed on IS window daily_features rows where orb_size is non-NULL.
    Direction-specific predicate. Returns None on insufficient data or
    degenerate variance.
    """
    predicate = _predicate_sql(session, direction)
    sql = f"""
        SELECT
          CASE WHEN ({predicate}) THEN 1 ELSE 0 END AS fire,
          d.orb_{session}_size AS osize
        FROM daily_features d
        WHERE d.symbol = ?
          AND d.orb_minutes = ?
          AND d.trading_day < ?
          AND d.orb_{session}_size IS NOT NULL
          AND d.orb_{session}_break_dir IS NOT NULL
    """
    rows = con.execute(sql, [instrument, ORB_MINUTES, holdout]).fetchall()
    if len(rows) < 30:
        return None
    fires = [r[0] for r in rows]
    sizes = [r[1] for r in rows]
    if sum(fires) == 0 or sum(fires) == len(fires):
        return None  # zero-variance binary
    try:
        r, _ = _sstats.pearsonr(fires, sizes)
        if math.isnan(r):
            return None
        return float(r)
    except Exception:
        return None


# =========================================================================
# STATISTICS
# =========================================================================

def _summarise(trades: list[tuple[date, float, str, Any]]) -> dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {"n": 0, "expr": None, "wr": None, "std": None, "sharpe": None, "pnl": []}
    pnl = [float(t[1]) for t in trades if t[1] is not None]
    wins = sum(1 for t in trades if t[2] == "win")
    if not pnl:
        return {"n": n, "expr": None, "wr": None, "std": None, "sharpe": None, "pnl": []}
    expr = sum(pnl) / len(pnl)
    wr = wins / n
    if len(pnl) > 1:
        std = (sum((p - expr) ** 2 for p in pnl) / (len(pnl) - 1)) ** 0.5
    else:
        std = None
    sharpe = (expr / std) if (std and std > 0) else None
    return {"n": n, "expr": expr, "wr": wr, "std": std, "sharpe": sharpe, "pnl": pnl}


def _t_test(pnl: list[float]) -> tuple[float | None, float | None]:
    if len(pnl) < 2:
        return None, None
    mean = sum(pnl) / len(pnl)
    var = sum((p - mean) ** 2 for p in pnl) / (len(pnl) - 1)
    std = var ** 0.5
    if std == 0:
        return None, None
    se = std / math.sqrt(len(pnl))
    t = mean / se
    p = 2.0 * (1.0 - _sstats.t.cdf(abs(t), df=len(pnl) - 1))
    return float(t), float(p)


def _bh_fdr(pvalues: list[tuple[Any, float]]) -> dict[Any, float]:
    """Benjamini-Hochberg q-values. Returns {key: q} for every input key.

    q_i = min over j >= i of (p_j * m / rank_j). Same monotone adjustment as
    statsmodels.stats.multitest.multipletests(method='fdr_bh').
    """
    if not pvalues:
        return {}
    clean = [(k, p) for k, p in pvalues if p is not None and not math.isnan(p)]
    if not clean:
        return {k: None for k, _ in pvalues}  # type: ignore[return-value]
    sorted_p = sorted(clean, key=lambda x: x[1])
    m = len(sorted_p)
    adjusted: dict[Any, float] = {}
    running_min = float("inf")
    for i in range(m - 1, -1, -1):
        key, p = sorted_p[i]
        raw_q = p * m / (i + 1)
        running_min = min(running_min, raw_q)
        adjusted[key] = min(1.0, running_min)
    # Keys with NaN / None p remain undefined; callers treat as not-surviving
    for key, p in pvalues:
        adjusted.setdefault(key, None)  # type: ignore[arg-type]
    return adjusted


# =========================================================================
# CELL EVALUATION
# =========================================================================

def _build_cell(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    session: str,
    direction: str,
    rr: float,
    holdout: date,
) -> CellResult:
    predicate = "pmh_break_long" if direction == "long" else "pml_break_short"
    cell = CellResult(instrument, session, direction, rr, predicate)

    # Base (direction-aligned ORB without HTF filter)
    base_trades = _load_cell_trades(con, instrument, session, direction, rr, on_filter=False)
    base_is_t = [t for t in base_trades if t[0] < holdout]
    base_oos_t = [t for t in base_trades if t[0] >= holdout]
    base_is = _summarise(base_is_t)
    base_oos = _summarise(base_oos_t)
    cell.n_is_base = base_is["n"]
    cell.n_oos_base = base_oos["n"]
    cell.expr_is_base = base_is["expr"]
    cell.expr_oos_base = base_oos["expr"]
    cell.wr_is_base = base_is["wr"]

    # On (HTF filter fires)
    on_trades = _load_cell_trades(con, instrument, session, direction, rr, on_filter=True)
    on_is_t = [t for t in on_trades if t[0] < holdout]
    on_oos_t = [t for t in on_trades if t[0] >= holdout]
    on_is = _summarise(on_is_t)
    on_oos = _summarise(on_oos_t)
    cell.n_is_on = on_is["n"]
    cell.n_oos_on = on_oos["n"]
    cell.expr_is_on = on_is["expr"]
    cell.expr_oos_on = on_oos["expr"]
    cell.wr_is_on = on_is["wr"]
    cell.std_is_on = on_is["std"]
    cell.std_oos_on = on_oos["std"]
    cell.sharpe_is_on = on_is["sharpe"]
    cell.sharpe_oos_on = on_oos["sharpe"]

    if cell.expr_is_on is not None and cell.expr_is_base is not None:
        cell.delta_is = cell.expr_is_on - cell.expr_is_base
    if cell.expr_oos_on is not None and cell.expr_oos_base is not None:
        cell.delta_oos = cell.expr_oos_on - cell.expr_oos_base
    if cell.n_oos_on >= 5 and cell.delta_is is not None and cell.delta_oos is not None:
        cell.dir_match = (cell.delta_is >= 0) == (cell.delta_oos >= 0)

    if cell.n_is_base > 0:
        cell.fire_rate_is = cell.n_is_on / cell.n_is_base

    cell.t_stat, cell.raw_p = _t_test(on_is["pnl"])

    if cell.sharpe_is_on and cell.sharpe_is_on > 0 and cell.sharpe_oos_on is not None:
        cell.wfe = cell.sharpe_oos_on / cell.sharpe_is_on

    # Arithmetic-only (RULE 8.2)
    if (
        cell.wr_is_on is not None
        and cell.wr_is_base is not None
        and cell.delta_is is not None
    ):
        wr_spread = abs(cell.wr_is_on - cell.wr_is_base)
        if wr_spread < AO_WR_SPREAD and abs(cell.delta_is) > AO_DELTA_IS:
            cell.is_arithmetic_only = True

    # Era stability (Criterion 9)
    for era_name, era_start, era_end in ERAS:
        era_tr = [t for t in on_is_t if era_start <= t[0] <= era_end]
        n = len(era_tr)
        if n == 0:
            cell.era_expr[era_name] = {"n": 0, "expr": None, "exempt": True}
            continue
        era_pnl = [float(t[1]) for t in era_tr if t[1] is not None]
        era_expr = sum(era_pnl) / len(era_pnl) if era_pnl else None
        exempt = n < ERA_N_EXEMPT
        cell.era_expr[era_name] = {"n": n, "expr": era_expr, "exempt": exempt}
    cell.era_stable = all(
        e["exempt"] or (e["expr"] is not None and e["expr"] >= ERA_EXPR_MIN)
        for e in cell.era_expr.values()
    )

    return cell


def _assign_bh(results: list[CellResult]) -> None:
    def _key(c: CellResult) -> tuple:
        return (c.instrument, c.session, c.direction, c.rr)

    family_pv = [(_key(c), c.raw_p) for c in results]
    q_family = _bh_fdr(family_pv)

    q_inst: dict = {}
    for inst in INSTRUMENTS:
        group = [(_key(c), c.raw_p) for c in results if c.instrument == inst]
        q_inst.update(_bh_fdr(group))
    q_sess: dict = {}
    for sess in SESSIONS:
        group = [(_key(c), c.raw_p) for c in results if c.session == sess]
        q_sess.update(_bh_fdr(group))
    q_dir: dict = {}
    for dirn in DIRECTIONS:
        group = [(_key(c), c.raw_p) for c in results if c.direction == dirn]
        q_dir.update(_bh_fdr(group))

    for c in results:
        k = _key(c)
        c.q_family = q_family.get(k)
        c.q_global = c.q_family
        c.q_instrument = q_inst.get(k)
        c.q_session = q_sess.get(k)
        c.q_direction = q_dir.get(k)
        c.passes_bh_family = c.q_family is not None and c.q_family < BH_Q


def _assign_verdict(cell: CellResult) -> None:
    reasons: list[str] = []
    if cell.fire_rate_is is None or not (FIRE_RATE_MIN <= cell.fire_rate_is <= FIRE_RATE_MAX):
        reasons.append(f"fire_rate={cell.fire_rate_is}")
    if cell.n_is_on < N_ON_MIN:
        reasons.append(f"N_on<{N_ON_MIN}")
    if cell.raw_p is None or cell.raw_p >= 0.05:
        reasons.append(f"raw_p>=0.05 ({cell.raw_p})")
    if cell.t_stat is None or abs(cell.t_stat) < CHORDIA_T:
        reasons.append(f"|t|<{CHORDIA_T} ({cell.t_stat})")
    if cell.dir_match is False:
        reasons.append("dir_mismatch")
    elif cell.dir_match is None and cell.n_oos_on >= 5:
        reasons.append("dir_match_unknown")
    if cell.is_tautology:
        reasons.append(f"tautology corr={cell.tautology_corr:.3f}")
    if cell.is_arithmetic_only:
        reasons.append("arithmetic_only")
    if cell.wfe is None:
        reasons.append("WFE_none")
    elif cell.wfe < WFE_MIN:
        reasons.append(f"WFE<{WFE_MIN} ({cell.wfe:.3f})")
    elif cell.wfe > WFE_LEAKAGE:
        reasons.append(f"WFE>{WFE_LEAKAGE} LEAKAGE_SUSPECT ({cell.wfe:.3f})")
    if cell.era_stable is False:
        reasons.append("era_unstable")
    if not cell.passes_bh_family:
        reasons.append("BH_family_fail")

    if not reasons:
        cell.verdict = "PASS"
    elif reasons == ["BH_family_fail"]:
        cell.verdict = "PARK (BH-FDR family only)"
    elif reasons == ["dir_mismatch"] and cell.n_oos_on >= 30:
        cell.verdict = "PARK (OOS direction flip, N_OOS>=30)"
    else:
        cell.verdict = "KILL: " + "; ".join(reasons)


def _family_verdict(results: list[CellResult]) -> str:
    passed = [c for c in results if c.verdict == "PASS"]
    n_total = len(results)
    n_pass = len(passed)
    if n_pass == 0:
        return "FAMILY KILL (FK1: zero cells pass all gates + BH_family)"
    if n_pass > int(0.833 * n_total):
        return f"FAMILY KILL (FK2: uniform-survivor pattern, {n_pass}/{n_total} — pipeline artefact)"
    if n_pass >= 5:
        top_keys = {(c.session, c.direction) for c in passed[:5]}
        if len(top_keys) == 1:
            return "FAMILY KILL (FK3: all top-5 survivors share (session, direction))"
    if passed:
        worst_delta = max(abs(c.delta_is or 0.0) for c in passed)
        if worst_delta > 0.6:
            return f"FAMILY CAUTION (FK4: best delta_IS={worst_delta:.3f} > 0.6 red flag)"
    return f"PARTIAL PASS ({n_pass}/{n_total} cells cleared every gate)"


# =========================================================================
# RESULT MARKDOWN
# =========================================================================

def _fmt(value: Any, fmt: str = ".3f", none: str = "—") -> str:
    if value is None:
        return none
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if math.isnan(value):
            return none
        return format(value, fmt)
    return str(value)


def _emit_result_md(
    results: list[CellResult],
    family_verdict: str,
    holdout: date,
    db_path: Path,
    scan_head_sha: str,
) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append("# HTF Path A prev-month v1 — Family Scan Results")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append(f"**Scan script HEAD SHA:** `{scan_head_sha}`")
    lines.append(f"**Canonical DB:** `{db_path}`")
    lines.append(f"**Holdout (Mode A):** `trading_day >= {holdout.isoformat()}` — imported from `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`")
    lines.append(f"**Pre-registration:** `docs/audit/hypotheses/2026-04-18-htf-path-a-prev-month-v1.yaml`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Instruments: {list(INSTRUMENTS)}")
    lines.append(f"- Sessions: {list(SESSIONS)}")
    lines.append(f"- Aperture: O{ORB_MINUTES}")
    lines.append(f"- RR targets: {list(RR_TARGETS)}")
    lines.append(f"- Directions / cells: {list(DIRECTIONS)} (pmh_break_long / pml_break_short)")
    lines.append(f"- Entry model: {ENTRY_MODEL} confirm_bars={CONFIRM_BARS}")
    lines.append(f"- **Total cells:** {len(results)}")
    lines.append("")
    lines.append("## BH-FDR framings (informational)")
    lines.append("")
    lines.append("| Framing | K | Role |")
    lines.append("|---|---:|---|")
    lines.append(f"| K_global     | {K_GLOBAL}     | Informational — equal to K_family under single-family run |")
    lines.append(f"| K_family     | {K_FAMILY}     | **PRIMARY promotion gate** (q < {BH_Q}) |")
    lines.append(f"| K_instrument | {K_INSTRUMENT} | Informational — per instrument across sess×dir×RR |")
    lines.append(f"| K_session    | {K_SESSION}    | Informational — per session across inst×dir×RR |")
    lines.append(f"| K_direction  | {K_DIRECTION}  | Informational — per direction across inst×sess×RR |")
    lines.append("")
    lines.append("## Family verdict")
    lines.append("")
    lines.append(f"**{family_verdict}**")
    lines.append("")

    # Summary stats table
    lines.append("## Per-cell stats")
    lines.append("")
    lines.append(
        "| # | inst | session | dir | RR | N_is_base | N_is_on | fire% | ExpR_base_IS | ExpR_on_IS | ΔIS | ExpR_on_OOS | ΔOOS | dir_match | t | raw p | q_family | q_inst | q_sess | q_dir |"
    )
    lines.append(
        "|--|-----|---------|-----|----|----:|----:|----:|----:|----:|----:|----:|----:|:--:|----:|----:|----:|----:|----:|----:|"
    )
    for i, c in enumerate(results, 1):
        fr_pct = f"{c.fire_rate_is * 100:.1f}" if c.fire_rate_is is not None else "—"
        lines.append(
            "| "
            + " | ".join([
                str(i),
                c.instrument,
                c.session,
                c.direction,
                f"{c.rr:g}",
                _fmt(c.n_is_base, "d"),
                _fmt(c.n_is_on, "d"),
                fr_pct,
                _fmt(c.expr_is_base),
                _fmt(c.expr_is_on),
                _fmt(c.delta_is),
                _fmt(c.expr_oos_on),
                _fmt(c.delta_oos),
                _fmt(c.dir_match),
                _fmt(c.t_stat, ".2f"),
                _fmt(c.raw_p, ".4f"),
                _fmt(c.q_family, ".3f"),
                _fmt(c.q_instrument, ".3f"),
                _fmt(c.q_session, ".3f"),
                _fmt(c.q_direction, ".3f"),
            ])
            + " |"
        )
    lines.append("")

    # WFE + era + flags table
    lines.append("## WFE, era stability, flags")
    lines.append("")
    lines.append(
        "| # | inst | session | dir | RR | Sharpe_IS | Sharpe_OOS | WFE | era_stable | tautology corr | tautology? | arithmetic_only |"
    )
    lines.append(
        "|--|-----|---------|-----|----|----:|----:|----:|:--:|----:|:--:|:--:|"
    )
    for i, c in enumerate(results, 1):
        lines.append(
            "| "
            + " | ".join([
                str(i),
                c.instrument,
                c.session,
                c.direction,
                f"{c.rr:g}",
                _fmt(c.sharpe_is_on, ".3f"),
                _fmt(c.sharpe_oos_on, ".3f"),
                _fmt(c.wfe, ".3f"),
                _fmt(c.era_stable),
                _fmt(c.tautology_corr, ".3f"),
                _fmt(c.is_tautology),
                _fmt(c.is_arithmetic_only),
            ])
            + " |"
        )
    lines.append("")

    # Per-era detail
    lines.append("## Era stability detail (per cell, IS-on trades)")
    lines.append("")
    lines.append(
        "| # | inst | session | dir | RR | " + " | ".join(f"{e[0]} (n, ExpR)" for e in ERAS) + " |"
    )
    lines.append("|--|-----|---------|-----|----|" + "|".join(["----:" for _ in ERAS]) + "|")
    for i, c in enumerate(results, 1):
        era_cells = []
        for era_name, _, _ in ERAS:
            e = c.era_expr.get(era_name, {"n": 0, "expr": None})
            exempt_mark = "*" if e.get("exempt") else ""
            era_cells.append(f"n={e['n']}, ExpR={_fmt(e['expr'])}{exempt_mark}")
        lines.append(
            "| "
            + " | ".join([str(i), c.instrument, c.session, c.direction, f"{c.rr:g}"] + era_cells)
            + " |"
        )
    lines.append("")
    lines.append("*exempt = N < 50 (Criterion 9 threshold)*")
    lines.append("")

    # Per-cell verdict table
    lines.append("## Verdict per cell")
    lines.append("")
    lines.append("| # | inst | session | dir | RR | verdict |")
    lines.append("|--|-----|---------|-----|----|---|")
    for i, c in enumerate(results, 1):
        lines.append(f"| {i} | {c.instrument} | {c.session} | {c.direction} | {c.rr:g} | {c.verdict} |")
    lines.append("")

    # Methodology notes
    lines.append("## Methodology notes")
    lines.append("")
    lines.append(
        "- Direction resolved via `daily_features.orb_{session}_break_dir` "
        "(orb_outcomes has no direction column; the trade inherits break direction "
        "by construction). `long` = up-break, `short` = down-break."
    )
    lines.append(
        "- Base (unfiltered) cell = same (instrument, session, direction, RR, E2, cb=1) "
        "lane without the HTF predicate. Delta_IS / Delta_OOS = ExpR_on − ExpR_base."
    )
    lines.append(
        "- t-test: one-sample two-tailed vs 0 on per-trade pnl_r of IS-on trades."
    )
    lines.append(
        "- BH-FDR: classic Benjamini-Hochberg monotone q-value computation. "
        "K_family is the primary promotion gate (q < 0.05); other framings reported "
        "for honest disclosure per backtesting-methodology.md RULE 4."
    )
    lines.append(
        "- Tautology (T0): Pearson correlation of HTF-fire binary against "
        "`orb_{session}_size` (continuous) over IS-window daily_features rows. "
        "Proxy for the ORB_G family of size-based filters. |corr| > 0.70 → flagged."
    )
    lines.append(
        "- Arithmetic-only (RULE 8.2): `|WR_on − WR_base| < 0.03` AND `|Δ_IS| > 0.10`. "
        "Indicates cost-screen mechanism, not a WR-predictor."
    )
    lines.append(
        "- WFE = Sharpe_OOS_on / Sharpe_IS_on (both per-trade Sharpe on the same "
        "scale; annualisation cancels in the ratio). <0.50 fails Criterion 6; "
        ">0.95 flagged LEAKAGE_SUSPECT per RULE 3.2."
    )
    lines.append(
        "- Era bins per Criterion 9: 2019-2020, 2021-2022, 2023, 2024-2025. "
        "Eras with N < 50 on IS-on trades are exempt."
    )
    lines.append(
        "- Look-ahead: `prev_month_*` populated by canonical "
        "`pipeline.build_daily_features._apply_htf_level_fields` from fully-closed "
        "prior calendar month only. Drift check 59 (HTF integrity) passes with "
        "all 4 divergence classes caught in pressure test (commit 668d2680)."
    )
    lines.append(
        "- No writes to `validated_setups` or `experimental_strategies`. Read-only scan."
    )
    lines.append("")

    lines.append("## Reproduction")
    lines.append("")
    lines.append("```")
    lines.append(f"DUCKDB_PATH={db_path} python research/htf_path_a_prev_month_v1_scan.py")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


# =========================================================================
# MAIN
# =========================================================================

def _git_head_sha() -> str:
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
        return out
    except Exception:
        return "unknown"


def main() -> int:
    db = GOLD_DB_PATH
    if not db.exists():
        print(f"FATAL: DB not found at {db}", file=sys.stderr)
        return 2

    holdout = HOLDOUT_SACRED_FROM
    if not YAML_PATH.exists():
        print(f"FATAL: pre-reg YAML not found at {YAML_PATH}", file=sys.stderr)
        return 2

    head_sha = _git_head_sha()
    print(f"Canonical DB: {db}")
    print(f"Holdout (Mode A): trading_day >= {holdout}")
    print(f"Pre-reg: {YAML_PATH}")
    print(f"HEAD SHA: {head_sha}")
    print()

    results: list[CellResult] = []
    with duckdb.connect(str(db), read_only=True) as con:
        for instrument in INSTRUMENTS:
            for session in SESSIONS:
                for direction in DIRECTIONS:
                    for rr in RR_TARGETS:
                        cell = _build_cell(con, instrument, session, direction, rr, holdout)
                        results.append(cell)

        # T0 tautology correlation — computed once per (inst, sess, dir), RR-invariant
        t0_cache: dict = {}
        for c in results:
            k = (c.instrument, c.session, c.direction)
            if k not in t0_cache:
                t0_cache[k] = _tautology_corr(con, c.instrument, c.session, c.direction, holdout)
            c.tautology_corr = t0_cache[k]
            if c.tautology_corr is not None and abs(c.tautology_corr) > TAUTOLOGY_MAX:
                c.is_tautology = True

    _assign_bh(results)
    for c in results:
        _assign_verdict(c)
    fam_verdict = _family_verdict(results)

    md = _emit_result_md(results, fam_verdict, holdout, db, head_sha)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(md, encoding="utf-8")
    print(f"wrote {RESULT_PATH} ({len(md)} bytes)")
    print()
    print(f"family verdict: {fam_verdict}")
    n_pass = sum(1 for c in results if c.verdict == "PASS")
    n_park = sum(1 for c in results if c.verdict.startswith("PARK"))
    n_kill = sum(1 for c in results if c.verdict.startswith("KILL"))
    print(f"cells: PASS={n_pass} PARK={n_park} KILL={n_kill}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
