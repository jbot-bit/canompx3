"""MES + MGC filter-overlay family scan — Pathway A with BH-FDR.

Pre-reg: docs/audit/hypotheses/2026-04-21-mes-mgc-filter-overlay-family-v1.yaml

Cross-instrument completion of PR #53 (MES + MGC unfiltered-baseline null,
K=176, 0 survivors). Tests whether 5 canonical filters from one-mechanism-per-
family rescue any MES/MGC cell at 5m aperture. Scope excludes Asian-window
sessions (TOKYO_OPEN, SINGAPORE_OPEN, CME_REOPEN, BRISBANE_1025) per
OvernightRangeAbsFilter's overnight_range look-ahead rule.

K_family <= 210. Canonical filter delegation via research.filter_utils.filter_signal
(no re-encoding). Triple-join on (trading_day, symbol, orb_minutes) per RULE 9.
Fire-rate gate [0.05, 0.95] per RULE 8.1. Downstream C6/C8/C9 mandatory.

No capital action.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

PREREG_PATH = "docs/audit/hypotheses/2026-04-21-mes-mgc-filter-overlay-family-v1.yaml"
RESULT_DOC = Path("docs/audit/results/2026-04-21-mes-mgc-filter-overlay-family-v1.md")

APERTURE = 5
RRS = [1.0, 1.5, 2.0]
MIN_N_ON_IS = 100
MIN_N_OOS = 50
BH_FDR_ALPHA = 0.05
FIRE_MIN = 0.05
FIRE_MAX = 0.95

SESSIONS_BY_INSTRUMENT = {
    "MES": [
        "NYSE_OPEN",
        "US_DATA_830",
        "US_DATA_1000",
        "COMEX_SETTLE",
        "CME_PRECLOSE",
        "NYSE_CLOSE",
        "LONDON_METALS",
        "EUROPE_FLOW",
    ],
    "MGC": [
        "NYSE_OPEN",
        "US_DATA_830",
        "US_DATA_1000",
        "COMEX_SETTLE",
        "LONDON_METALS",
        "EUROPE_FLOW",
    ],
}

FILTERS = [
    "COST_LT12",
    "ORB_G5",
    "OVNRNG_50",
    "ATR_P50",
    "VWAP_MID_ALIGNED",
]


@dataclass
class CellResult:
    instrument: str
    session: str
    rr: float
    filter_key: str
    n_total_is: int
    n_on_is: int
    fire_rate: float
    net_expr_on_is: float
    t_is: float
    p_one_tailed_is: float
    q_bh: float
    wfe: float
    n_on_oos: int
    net_expr_on_oos: float
    ratio_oos_is: float
    era_min_expr: float
    era_n_fail: int
    h1_t_pass: bool
    h1_fdr_pass: bool
    fire_pass: bool
    c6_pass: bool
    c8_pass: bool
    c9_pass: bool
    verdict: str


def _load_cell_with_features(
    con: duckdb.DuckDBPyConnection, instrument: str, session: str, rr: float
) -> pd.DataFrame:
    """Triple-joined orb_outcomes x daily_features for one (instrument, session, RR) cell."""
    sql = """
    SELECT o.trading_day, o.symbol, o.orb_minutes, o.pnl_r, d.*
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = ?
      AND o.orb_label = ?
      AND o.orb_minutes = ?
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target = ?
      AND o.pnl_r IS NOT NULL
    """
    df = con.execute(sql, [instrument, session, APERTURE, rr]).df()
    if len(df) == 0:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    return df


def _t_mean(vals: np.ndarray) -> tuple[float, float]:
    if len(vals) < 2 or np.all(vals == vals[0]):
        return float("nan"), float("nan")
    m = vals.mean()
    se = vals.std(ddof=1) / np.sqrt(len(vals))
    if se == 0:
        return float("nan"), float("nan")
    t = m / se
    p = float(1.0 - stats.t.cdf(t, len(vals) - 1))
    return float(t), p


def _wfe(is_df: pd.DataFrame, n_folds: int = 3) -> float:
    if len(is_df) < n_folds * 50:
        return float("nan")
    df = is_df.sort_values("trading_day").reset_index(drop=True)
    edges = np.linspace(0, len(df), n_folds + 1, dtype=int)
    is_mean = df["pnl_r"].mean()
    if is_mean == 0 or np.isnan(is_mean):
        return float("nan")
    means = []
    for i in range(1, n_folds):
        slc = df.iloc[edges[i] : edges[i + 1]]
        if len(slc) < 10:
            continue
        means.append(float(slc["pnl_r"].mean()))
    return float(np.mean(means) / is_mean) if means else float("nan")


def _era_stability(is_df: pd.DataFrame) -> tuple[float, int]:
    if is_df.empty:
        return float("nan"), 0
    df = is_df.copy()
    df["year"] = df["trading_day"].dt.year
    agg = df.groupby("year")["pnl_r"].agg(["count", "mean"]).reset_index()
    elig = agg[agg["count"] >= 50]
    if len(elig) == 0:
        return float("nan"), 0
    return float(elig["mean"].min()), int((elig["mean"] < -0.05).sum())


def _bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg q-values with monotonicity."""
    p_arr = np.array([p if not np.isnan(p) else 1.0 for p in p_values])
    n = len(p_arr)
    if n == 0:
        return []
    order = np.argsort(p_arr)
    ranks = np.arange(1, n + 1)
    q = p_arr[order] * n / ranks
    for i in range(len(q) - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    q_final = np.empty_like(q)
    q_final[order] = q
    return q_final.tolist()


def _test_cell(
    instrument: str,
    session: str,
    rr: float,
    filter_key: str,
    df_full: pd.DataFrame,
) -> CellResult | None:
    if df_full.empty:
        return None

    # Canonical filter signal (fail-closed: NaN coerced to 0)
    fire = filter_signal(df_full, filter_key, session)
    df = df_full.copy()
    df["_fire"] = fire

    is_df = df.loc[df["trading_day"] < pd.Timestamp(HOLDOUT_SACRED_FROM)].reset_index(drop=True)
    oos_df = df.loc[df["trading_day"] >= pd.Timestamp(HOLDOUT_SACRED_FROM)].reset_index(drop=True)

    n_total_is = len(is_df)
    if n_total_is == 0:
        return None

    is_on = is_df.loc[is_df["_fire"] == 1].reset_index(drop=True)
    n_on_is = len(is_on)
    fire_rate = n_on_is / n_total_is if n_total_is > 0 else float("nan")

    if n_on_is < MIN_N_ON_IS:
        # SCAN_ABORT — too few on-days to run downstream gates
        return CellResult(
            instrument=instrument,
            session=session,
            rr=rr,
            filter_key=filter_key,
            n_total_is=n_total_is,
            n_on_is=n_on_is,
            fire_rate=fire_rate,
            net_expr_on_is=float("nan"),
            t_is=float("nan"),
            p_one_tailed_is=float("nan"),
            q_bh=float("nan"),
            wfe=float("nan"),
            n_on_oos=0,
            net_expr_on_oos=float("nan"),
            ratio_oos_is=float("nan"),
            era_min_expr=float("nan"),
            era_n_fail=0,
            h1_t_pass=False,
            h1_fdr_pass=False,
            fire_pass=(FIRE_MIN <= fire_rate <= FIRE_MAX) if not np.isnan(fire_rate) else False,
            c6_pass=False,
            c8_pass=False,
            c9_pass=False,
            verdict="SCAN_ABORT",
        )

    vals_is = is_on["pnl_r"].astype(float).to_numpy()
    net_is = float(vals_is.mean())
    t_is, p_is = _t_mean(vals_is)

    oos_on = oos_df.loc[oos_df["_fire"] == 1].reset_index(drop=True)
    n_on_oos = len(oos_on)
    if n_on_oos > 0:
        net_oos = float(oos_on["pnl_r"].mean())
        ratio = net_oos / net_is if net_is != 0 else float("nan")
    else:
        net_oos, ratio = float("nan"), float("nan")

    wfe = _wfe(is_on)
    era_min, era_nfail = _era_stability(is_on)

    fire_pass = FIRE_MIN <= fire_rate <= FIRE_MAX
    h1_t_pass = (
        net_is > 0
        and not np.isnan(t_is)
        and t_is >= 3.0
        and not np.isnan(p_is)
        and p_is < 0.05
        and fire_pass
    )
    c6_pass = not np.isnan(wfe) and wfe >= 0.50
    c8_pass = not np.isnan(ratio) and ratio >= 0.40 and n_on_oos >= MIN_N_OOS
    c9_pass = not np.isnan(era_min) and era_nfail == 0

    return CellResult(
        instrument=instrument,
        session=session,
        rr=rr,
        filter_key=filter_key,
        n_total_is=n_total_is,
        n_on_is=n_on_is,
        fire_rate=fire_rate,
        net_expr_on_is=net_is,
        t_is=t_is,
        p_one_tailed_is=p_is,
        q_bh=float("nan"),
        wfe=wfe,
        n_on_oos=n_on_oos,
        net_expr_on_oos=net_oos,
        ratio_oos_is=ratio,
        era_min_expr=era_min,
        era_n_fail=era_nfail,
        h1_t_pass=h1_t_pass,
        h1_fdr_pass=False,
        fire_pass=fire_pass,
        c6_pass=c6_pass,
        c8_pass=c8_pass,
        c9_pass=c9_pass,
        verdict="TBD",
    )


def _assign_verdicts(cells: list[CellResult]) -> None:
    # Only count cells that ran (not SCAN_ABORT) in BH-FDR K
    runnable = [c for c in cells if c.verdict != "SCAN_ABORT"]
    p_values = [c.p_one_tailed_is for c in runnable]
    qs = _bh_fdr(p_values)
    for c, q in zip(runnable, qs):
        c.q_bh = q
        c.h1_fdr_pass = c.h1_t_pass and not np.isnan(q) and q < BH_FDR_ALPHA
        if not (c.h1_t_pass and c.h1_fdr_pass):
            c.verdict = "KILL_IS"
        elif c.c6_pass and c.c8_pass and c.c9_pass:
            c.verdict = "CANDIDATE_READY"
        else:
            c.verdict = "RESEARCH_SURVIVOR"


def _render_table(cells: list[CellResult]) -> str:
    lines = [
        "| Inst | Session | RR | Filter | Fire% | N_on | Net ExpR | t | q(BH) | WFE | N_OOS | Net OOS | OOS/IS | Worst Yr | Verdict |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def f(x: float, spec: str = "+.4f") -> str:
        return f"{x:{spec}}" if not np.isnan(x) else "-"

    for c in sorted(
        cells,
        key=lambda x: (
            x.instrument,
            x.session,
            x.rr,
            x.filter_key,
        ),
    ):
        lines.append(
            f"| {c.instrument} | {c.session} | {c.rr} | {c.filter_key} | "
            f"{f(c.fire_rate * 100 if not np.isnan(c.fire_rate) else float('nan'), '.1f')} | "
            f"{c.n_on_is} | {f(c.net_expr_on_is)} | {f(c.t_is, '+.3f')} | "
            f"{f(c.q_bh, '.4f')} | {f(c.wfe, '.3f')} | {c.n_on_oos} | "
            f"{f(c.net_expr_on_oos)} | {f(c.ratio_oos_is, '+.2f')} | "
            f"{f(c.era_min_expr)} | **{c.verdict}** |"
        )
    return "\n".join(lines)


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    cells: list[CellResult] = []
    try:
        for instrument, sessions in SESSIONS_BY_INSTRUMENT.items():
            for session in sessions:
                for rr in RRS:
                    df = _load_cell_with_features(con, instrument, session, rr)
                    for f_key in FILTERS:
                        r = _test_cell(instrument, session, rr, f_key, df)
                        if r is not None:
                            cells.append(r)
    finally:
        con.close()

    _assign_verdicts(cells)
    cr = [c for c in cells if c.verdict == "CANDIDATE_READY"]
    rs = [c for c in cells if c.verdict == "RESEARCH_SURVIVOR"]
    kill = [c for c in cells if c.verdict == "KILL_IS"]
    abort = [c for c in cells if c.verdict == "SCAN_ABORT"]

    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    parts.append("# MES + MGC filter-overlay family — v1\n")
    parts.append(f"**Pre-reg:** `{PREREG_PATH}`\n")
    parts.append("**Script:** `research/mes_mgc_filter_overlay_family_v1.py`\n")
    k_runnable = len(cells) - len(abort)
    parts.append(
        f"**Family K (runnable):** {k_runnable} cells (SCAN_ABORT excluded from BH-FDR); "
        f"total enumerated {len(cells)}.\n"
    )
    parts.append(
        "**Gates:** H1 (t>=+3.0, BH-FDR q<0.05, fire_rate in [5%, 95%]), "
        f"C6 (WFE>=0.50), C8 (OOS/IS>=0.40, N_OOS>={MIN_N_OOS}), C9 (era stability)\n"
    )
    parts.append(
        "**Canonical filter delegation:** research.filter_utils.filter_signal. "
        "No filter logic re-encoded.\n"
    )
    parts.append("## Summary counts")
    parts.append("")
    parts.append(f"- CANDIDATE_READY: **{len(cr)}**")
    parts.append(f"- RESEARCH_SURVIVOR: {len(rs)}")
    parts.append(f"- KILL_IS: {len(kill)}")
    parts.append(f"- SCAN_ABORT (fire rate below N_on_IS floor): {len(abort)}")
    parts.append("")
    parts.append("## CANDIDATE_READY cells")
    parts.append("")
    parts.append(_render_table(cr) if cr else "_None._")
    parts.append("")
    parts.append("## RESEARCH_SURVIVOR cells (for reference)")
    parts.append("")
    parts.append(_render_table(rs) if rs else "_None._")
    parts.append("")
    parts.append("## Full result table (all runnable cells)")
    parts.append("")
    parts.append(_render_table([c for c in cells if c.verdict != "SCAN_ABORT"]))
    parts.append("")
    parts.append("## SCAN_ABORT (fire rate < N_on_IS threshold)")
    parts.append("")
    if abort:
        lines = ["| Inst | Session | RR | Filter | Fire% | N_on |", "|---|---|---:|---|---:|---:|"]
        for c in sorted(abort, key=lambda x: (x.instrument, x.session, x.rr, x.filter_key)):
            fr = c.fire_rate * 100 if not np.isnan(c.fire_rate) else float("nan")
            fr_s = f"{fr:.1f}" if not np.isnan(fr) else "-"
            lines.append(
                f"| {c.instrument} | {c.session} | {c.rr} | {c.filter_key} | {fr_s} | {c.n_on_is} |"
            )
        parts.append("\n".join(lines))
    else:
        parts.append("_None._")
    parts.append("")
    parts.append("## Interpretation")
    parts.append("")
    if cr:
        parts.append(
            f"- {len(cr)} CANDIDATE_READY cell(s) survived all four gates. Pre-reg decision rule: "
            "viable MES/MGC filter-overlay lane. Confirmatory-validation stage would be the next "
            "pre-reg — this scan does NOT deploy."
        )
    elif rs:
        parts.append(
            f"- Zero CANDIDATE_READY, {len(rs)} RESEARCH_SURVIVOR. Signal exists in IS but "
            "downstream gates fail. Watch list, no action."
        )
    else:
        parts.append(
            "- Zero CANDIDATE_READY, zero RESEARCH_SURVIVOR. Pre-reg decision rule triggered: "
            "MES/MGC single-filter ORB at 5m aperture is DEAD. Future work must change aperture, "
            "entry model, or filter composition."
        )
    parts.append("")
    parts.append("## Not done by this result")
    parts.append("")
    parts.append("- No writes to validated_setups / lane_allocation / edge_families.")
    parts.append("- No deployment or capital action.")
    parts.append("- No multi-filter combinations tested (future pre-regs).")
    parts.append("- No apertures other than 5m tested (future pre-regs if survivors emerge).")
    parts.append("- No MNQ re-test (PR #51 canonical for MNQ).")

    RESULT_DOC.write_text("\n".join(parts), encoding="utf-8")

    print(f"Total enumerated: {len(cells)}")
    print(f"Family K (runnable): {k_runnable}")
    print(f"CANDIDATE_READY: {len(cr)}")
    print(f"RESEARCH_SURVIVOR: {len(rs)}")
    print(f"KILL_IS: {len(kill)}")
    print(f"SCAN_ABORT: {len(abort)}")
    print(f"RESULT_DOC: {RESULT_DOC}")
    if cr:
        print("\nCANDIDATE_READY cells:")
        for c in sorted(cr, key=lambda x: -x.net_expr_on_is):
            print(
                f"  {c.instrument} {c.session} RR={c.rr} {c.filter_key}: "
                f"fire={c.fire_rate * 100:.1f}% N_on={c.n_on_is} "
                f"NetExpR={c.net_expr_on_is:+.4f} t={c.t_is:+.3f} "
                f"q={c.q_bh:.4f} WFE={c.wfe:.2f} OOS/IS={c.ratio_oos_is:+.2f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
