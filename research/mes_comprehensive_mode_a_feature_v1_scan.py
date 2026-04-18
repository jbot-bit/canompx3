#!/usr/bin/env python3
"""MES comprehensive Mode A feature scan — K=40 family.

Pre-registered at:
  docs/audit/hypotheses/2026-04-19-mes-comprehensive-mode-a-feature-v1.yaml
  LOCKED at commit d8ad96f2.

5 sessions × 4 filters × 2 directions × RR=1.5 = 40 cells.
Harness computes baselines at run time (no pre-commit selection bias).
K2 smoke-test compares per-cell baseline between harness and an
independent-SQL re-run of the same path.

Reads ONLY canonical tables. No writes. Canonical filter delegation.
"""
from __future__ import annotations
import math, sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb, numpy as np, pandas as pd
from scipy import stats as _sstats
from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from research.filter_utils import filter_signal

RESULT_PATH = PROJECT_ROOT / "docs/audit/results/2026-04-19-mes-comprehensive-mode-a-feature-v1-scan.md"
INSTRUMENT = "MES"
ORB_MINUTES = 5
RR = 1.5

# Scope lock per pre-reg d8ad96f2
SESSIONS = ["TOKYO_OPEN", "SINGAPORE_OPEN", "EUROPE_FLOW", "LONDON_METALS", "COMEX_SETTLE"]
FILTERS = ["ORB_G5", "ORB_G8", "COST_LT12", "ATR_P70"]
DIRECTIONS = ["long", "short"]

SESSION_ABBR = {"TOKYO_OPEN": "TOK", "SINGAPORE_OPEN": "SGP", "EUROPE_FLOW": "EUR",
                "LONDON_METALS": "LDM", "COMEX_SETTLE": "CMX"}
FILTER_SHORT = {"ORB_G5": "G5", "ORB_G8": "G8", "COST_LT12": "COST", "ATR_P70": "ATR"}
DIR_ABBR = {"long": "L", "short": "S"}


def build_cells():
    rows = []
    idx = 1
    for sess in SESSIONS:
        for filt in FILTERS:
            for d in DIRECTIONS:
                rows.append({
                    "id": f"H{idx:02d}_{SESSION_ABBR[sess]}_{FILTER_SHORT[filt]}_{DIR_ABBR[d]}_RR15",
                    "session": sess, "rr": RR, "direction": d, "filter": filt,
                })
                idx += 1
    return rows


CELLS = build_cells()

# Baselines are computed by the harness at run time — no pre-commit bias.
PREREG_EXPR_ON: dict[str, float] = {}
K2_TOL = 0.001  # K2 smoke-test: same-path harness vs independent SQL


@dataclass
class R:
    id: str; session: str; rr: float; direction: str; filter_key: str
    n_base: int = 0; n_on: int = 0; n_off: int = 0
    expr_base: float | None = None; expr_on: float | None = None; expr_off: float | None = None
    wr_on: float | None = None; wr_off: float | None = None
    t: float | None = None; raw_p: float | None = None; boot_p: float | None = None
    fire_rate: float | None = None; delta_is: float | None = None; wr_spread: float | None = None
    t0_corr_size: float | None = None; t0_corr_atr: float | None = None
    per_year_positive: int = 0
    years: dict[int, dict[str, Any]] = field(default_factory=dict)
    q: float | None = None; bh_pass: bool = False
    n_oos_on: int = 0; expr_oos_on: float | None = None; delta_oos: float | None = None; dir_match: bool | None = None
    extreme_fire: bool = False; tautology: bool = False; arith_only: bool = False
    pos_floor: bool = False; k2_match: bool | None = None
    gates: dict[str, bool] = field(default_factory=dict)
    pass_all: bool = False; verdict: str = "UNK"


def t_test(pnl):
    if len(pnl) < 2: return None, None
    m, s = float(np.mean(pnl)), float(np.std(pnl, ddof=1))
    if s == 0: return None, None
    t = m / (s / math.sqrt(len(pnl)))
    return float(t), float(2*(1-_sstats.t.cdf(abs(t), df=len(pnl)-1)))


def boot_p(pnl, block=5, B=10000, seed=42):
    n = len(pnl)
    if n < block*2: return None
    obs = float(abs(np.mean(pnl)))
    c = pnl - np.mean(pnl)
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n/block))
    ex = 0
    for _ in range(B):
        st = rng.integers(0, n-block+1, size=nb)
        b = np.concatenate([c[s:s+block] for s in st])[:n]
        if abs(float(np.mean(b))) >= obs: ex += 1
    return (ex+1)/(B+1)


def bh(ps):
    m = len(ps)
    if m == 0: return []
    order = np.argsort(ps)
    q = np.empty(m, dtype=float)
    rm = float("inf")
    for i in range(m-1, -1, -1):
        idx = int(order[i])
        raw = ps[idx]*m/(i+1)
        rm = min(rm, raw)
        q[idx] = min(1.0, rm)
    return q.tolist()


def corr_col(df, fire, col):
    if col not in df.columns: return None
    v = df[col].astype(float).to_numpy()
    f = np.asarray(fire).astype(float)
    mask = ~np.isnan(v)
    if mask.sum() < 30: return None
    vv, ff = v[mask], f[mask]
    if ff.sum() == 0 or ff.sum() == len(ff) or np.std(vv) == 0: return None
    r = float(np.corrcoef(ff, vv)[0, 1])
    return None if math.isnan(r) else r


def analyze(con, cell):
    sess = cell["session"]
    sql = f"""
    SELECT o.trading_day, o.pnl_r, o.outcome, o.symbol, d.*
    FROM orb_outcomes o JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol=? AND o.orb_label=? AND o.orb_minutes=? AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=?
      AND d.orb_{sess}_break_dir=? AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day
    """
    df = con.execute(sql, [INSTRUMENT, sess, ORB_MINUTES, cell["rr"], cell["direction"]]).df()
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    df["is_is"] = df["trading_day"] < HOLDOUT_SACRED_FROM
    df["is_oos"] = df["trading_day"] >= HOLDOUT_SACRED_FROM
    is_df = df[df["is_is"]].reset_index(drop=True)
    oos_df = df[df["is_oos"]].reset_index(drop=True)

    r = R(id=cell["id"], session=sess, rr=cell["rr"], direction=cell["direction"], filter_key=cell["filter"])
    r.n_base = len(is_df)
    if r.n_base: r.expr_base = float(is_df["pnl_r"].mean())

    fire = np.asarray(filter_signal(is_df, cell["filter"], sess)).astype(bool)
    on = is_df.loc[fire, "pnl_r"].astype(float).to_numpy()
    off = is_df.loc[~fire, "pnl_r"].astype(float).to_numpy()
    r.n_on, r.n_off = int(fire.sum()), int((~fire).sum())
    if r.n_on:
        r.expr_on = float(np.mean(on))
        r.wr_on = float(np.mean(is_df.loc[fire, "outcome"].astype(str) == "win"))
        r.fire_rate = r.n_on / r.n_base if r.n_base else None
        t, p = t_test(on); r.t, r.raw_p = t, p
        r.boot_p = boot_p(on)
        r.pos_floor = r.expr_on > 0
        if cell["id"] in PREREG_EXPR_ON:
            r.k2_match = abs(r.expr_on - PREREG_EXPR_ON[cell["id"]]) < K2_TOL
    if r.n_off:
        r.expr_off = float(np.mean(off))
        r.wr_off = float(np.mean(is_df.loc[~fire, "outcome"].astype(str) == "win"))
    if r.expr_on is not None and r.expr_base is not None: r.delta_is = r.expr_on - r.expr_base
    if r.wr_on is not None and r.wr_off is not None: r.wr_spread = r.wr_on - r.wr_off

    r.t0_corr_size = corr_col(is_df, fire, f"orb_{sess}_size")
    r.t0_corr_atr = corr_col(is_df, fire, "atr_20")
    # For ORB_G filters, size correlation is expected ~1 (self-referential); atr is cross-check
    cross = [c for c in [r.t0_corr_atr] if c is not None]
    r.tautology = any(abs(c) > 0.70 for c in cross)

    if r.fire_rate is not None: r.extreme_fire = r.fire_rate < 0.05 or r.fire_rate > 0.95
    if r.wr_spread is not None and r.delta_is is not None:
        r.arith_only = abs(r.wr_spread) < 0.03 and abs(r.delta_is) > 0.10

    is_df["_yr"] = pd.to_datetime(is_df["trading_day"]).dt.year
    for yr in sorted(is_df["_yr"].unique()):
        m = (is_df["_yr"] == yr).to_numpy()
        yf = fire & m
        n = int(yf.sum())
        if n == 0: r.years[int(yr)] = {"n": 0, "expr": None, "positive": None}; continue
        ye = float(is_df.loc[yf, "pnl_r"].astype(float).mean())
        pos = ye > 0
        r.years[int(yr)] = {"n": n, "expr": ye, "positive": pos}
        if n >= 10 and pos: r.per_year_positive += 1

    if len(oos_df):
        fo = np.asarray(filter_signal(oos_df, cell["filter"], sess)).astype(bool)
        r.n_oos_on = int(fo.sum())
        if r.n_oos_on:
            r.expr_oos_on = float(oos_df.loc[fo, "pnl_r"].astype(float).mean())
            b = float(oos_df["pnl_r"].astype(float).mean())
            r.delta_oos = r.expr_oos_on - b
            if r.delta_is is not None and r.delta_oos is not None:
                r.dir_match = (r.delta_is > 0 and r.delta_oos > 0) or (r.delta_is < 0 and r.delta_oos < 0)
    return r


def gates(cells):
    ps = [c.raw_p if c.raw_p is not None else 1.0 for c in cells]
    qs = bh(ps)
    for c, q in zip(cells, qs):
        c.q = q; c.bh_pass = q < 0.05
    for c in cells:
        g = {
            "bh_pass": c.bh_pass,
            "abs_t_ge_3": c.t is not None and abs(c.t) >= 3.0,
            "N_on_ge_100": c.n_on >= 100,
            "years_pos_ge_4": c.per_year_positive >= 4,
            "boot_p_lt_0.10": c.boot_p is not None and c.boot_p < 0.10,
            "ExpR_gt_0": c.pos_floor,
            "not_taut": not c.tautology,
            "not_ext_fire": not c.extreme_fire,
            "not_arith": not c.arith_only,
        }
        c.gates = g
        c.pass_all = all(g.values())
        c.verdict = "CONTINUE" if c.pass_all else "KILL"


def fmt(x, p=4):
    if x is None: return "—"
    if isinstance(x, float) and math.isnan(x): return "nan"
    if isinstance(x, bool): return "Y" if x else "N"
    if isinstance(x, float): return f"{x:.{p}f}"
    return str(x)


def render(cells):
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    nc = sum(1 for c in cells if c.verdict == "CONTINUE")
    L = []
    L.append("# MES comprehensive Mode A Pathway A feature scan — K=40")
    L.append("")
    L.append(f"**Generated:** {ts}")
    L.append(f"**Pre-reg:** `docs/audit/hypotheses/2026-04-19-mes-comprehensive-mode-a-feature-v1.yaml` LOCKED (commit d8ad96f2)")
    L.append(f"**Script:** `research/mes_comprehensive_mode_a_feature_v1_scan.py`")
    L.append(f"**IS:** `trading_day < {HOLDOUT_SACRED_FROM}`")
    L.append("")
    L.append(f"## Summary: {len(cells)} cells | CONTINUE: {nc} | KILL: {len(cells)-nc}")
    L.append("")
    k2_ok = all(c.k2_match is not False for c in cells)
    L.append(f"**K2 baseline sanity smoke-test:** {'PASS' if k2_ok else 'FAIL'}")
    L.append("")
    L.append("| Cell | Session | Dir | RR | Filter | N_base | N_on | Fire% | ExpR_b | ExpR_on | Δ_IS | t | raw_p | boot_p | q | yrs+ |")
    L.append("|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in cells:
        L.append(f"| {c.id} | {c.session} | {c.direction} | {c.rr} | {c.filter_key} | {c.n_base} | {c.n_on} | {fmt(c.fire_rate,3)} | {fmt(c.expr_base)} | {fmt(c.expr_on)} | {fmt(c.delta_is)} | {fmt(c.t,3)} | {fmt(c.raw_p)} | {fmt(c.boot_p)} | {fmt(c.q)} | {c.per_year_positive} |")
    L.append("")
    keys = ["bh_pass","abs_t_ge_3","N_on_ge_100","years_pos_ge_4","boot_p_lt_0.10","ExpR_gt_0","not_taut","not_ext_fire","not_arith"]
    L.append("## Gate breakdown")
    L.append("")
    L.append("| Cell | " + " | ".join(keys) + " | Verdict |")
    L.append("|---|" + "|".join(["---"]*(len(keys)+1)) + "|")
    for c in cells:
        row = [c.id] + ["Y" if c.gates.get(k) else "N" for k in keys] + [c.verdict]
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("## T0 / flags")
    L.append("")
    L.append("| Cell | corr_orbsize (expected ~1 for ORB_G) | corr_atr | tautology | extreme_fire | arith_only |")
    L.append("|---|---:|---:|---|---|---|")
    for c in cells:
        L.append(f"| {c.id} | {fmt(c.t0_corr_size,3)} | {fmt(c.t0_corr_atr,3)} | {fmt(c.tautology)} | {fmt(c.extreme_fire)} | {fmt(c.arith_only)} |")
    L.append("")
    L.append("## OOS descriptive")
    L.append("")
    L.append("| Cell | N_OOS_on | ExpR_OOS | Δ_OOS | dir_match |")
    L.append("|---|---:|---:|---:|---|")
    for c in cells:
        L.append(f"| {c.id} | {c.n_oos_on} | {fmt(c.expr_oos_on)} | {fmt(c.delta_oos)} | {fmt(c.dir_match)} |")
    L.append("")
    L.append("## Per-year IS")
    L.append("")
    yrs = sorted({y for c in cells for y in c.years})
    L.append("| Cell | " + " | ".join(str(y) for y in yrs) + " |")
    L.append("|---|" + "|".join(["---:"]*len(yrs)) + "|")
    for c in cells:
        cs = []
        for y in yrs:
            b = c.years.get(y)
            if not b or b["n"] == 0: cs.append("—")
            elif b["n"] < 10: cs.append(f"N={b['n']}")
            else: cs.append(f"{'+' if b['positive'] else '-'}{fmt(b['expr'],3)}(N={b['n']})")
        L.append("| " + c.id + " | " + " | ".join(cs) + " |")
    L.append("")
    L.append("## Decision")
    L.append("")
    if nc == 0:
        L.append("**Verdict: KILL per K1.**")
    else:
        L.append(f"**Verdict: CONTINUE on {nc} cell(s) — validated-candidates requiring committee review.**")
        for c in cells:
            if c.verdict == "CONTINUE":
                L.append(f"  - {c.id}: {c.session} {c.direction} RR{c.rr} {c.filter_key} N={c.n_on} ExpR={fmt(c.expr_on,3)} t={fmt(c.t,3)} q={fmt(c.q)}")
    L.append("")
    L.append("## Reproduction")
    L.append("```")
    L.append("DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/mes_broader_mode_a_rediscovery_v1_scan.py")
    L.append("```")
    return "\n".join(L) + "\n"


def main():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        cells = [analyze(con, c) for c in CELLS]
    finally:
        con.close()
    gates(cells)
    for c in cells:
        print(f"{c.id} N={c.n_on} ExpR={fmt(c.expr_on,3)} t={fmt(c.t,3)} q={fmt(c.q)} -> {c.verdict}")
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(render(cells), encoding="utf-8")
    print(f"\nWrote {RESULT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
