"""Stage 1 replay for the locked MNQ wide-rel-IB v2 pre-registration.

Pre-registration: docs/audit/hypotheses/2026-04-18-mnq-wide-rel-ib-v2.yaml
Design doc:      docs/plans/2026-04-18-mnq-wide-rel-ib-v2-design.md
Distinctness:    docs/audit/results/2026-04-18-mes-wide-ib-distinctness-audit.md

Scope (LOCKED per pre-reg; no widening permitted):
  Instruments: MNQ only
  Sessions: CME_PRECLOSE, TOKYO_OPEN
  Aperture: O5
  Entry: E2 CB=1 stop_mult=1.0
  RR: 1.0, 1.5, 2.0
  Cells: K=6 total

Candidate filter: WIDE_REL_1.0X_AND_G5 (conjunction of orb_size/rolling20>=1.0 AND orb_size>=5.0)
Baseline: ORB_G5 alone
Metric: delta_ExpR = ExpR(WIDE_REL AND G5) - ExpR(G5 alone), per cell, IS and OOS

Primary pass rule (all must hold per-cell):
  1. BH-FDR q<0.05 at K=6 on Welch t-test p-value of delta
  2. Chordia t >= 3.79 (WITHOUT-theory threshold)
  3. WFE >= 0.50 across 5 expanding folds on IS
  4. N_OOS >= 30
  5. OOS direction match: sign(delta_IS) == sign(delta_OOS)
  6. OOS effect ratio: delta_OOS / delta_IS >= 0.40

Tautology guards (Rule 7, canonical metric = BOOLEAN fire correlation):
  Kill cell if |corr(cell_fire, alt_filter_fire)| > 0.70 for any deployed alternative.
  Alternatives per lane (excluding G5 which is a conjunction component):
    MNQ CME_PRECLOSE: X_MES_ATR60
    MNQ TOKYO_OPEN:   COST_LT12

Output: docs/audit/results/2026-04-18-mnq-wide-rel-ib-v2-replay.md
        research/output/mnq_wide_rel_ib_v2_replay.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import date
from pathlib import Path

import duckdb
from scipy.stats import ttest_ind_from_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pipeline.cost_model import COST_SPECS
from pipeline.paths import GOLD_DB_PATH

HYPOTHESIS_FILE = "docs/audit/hypotheses/2026-04-18-mnq-wide-rel-ib-v2.yaml"
DESIGN_DOC = "docs/plans/2026-04-18-mnq-wide-rel-ib-v2-design.md"

OUTPUT_MD = Path("docs/audit/results/2026-04-18-mnq-wide-rel-ib-v2-replay.md")
OUTPUT_JSON = Path("research/output/mnq_wide_rel_ib_v2_replay.json")

# Pre-reg locked constants — no tuning permitted
INSTRUMENT = "MNQ"
SESSIONS = ("CME_PRECLOSE", "TOKYO_OPEN")
APERTURE = 5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
STOP_MULTIPLIER = 1.0
RR_TARGETS = (1.0, 1.5, 2.0)
HOLDOUT = date(2026, 1, 1)
TRAILING_WINDOW_DAYS = 20
WIDE_REL_THRESHOLD = 1.0
G5_THRESHOLD = 5.0

# Pass thresholds (LOCKED)
BH_FDR_Q = 0.05
K_FAMILY = 6
CHORDIA_T_WITHOUT_THEORY = 3.79
WFE_MIN = 0.50
N_OOS_MIN = 30
EFF_RATIO_MIN = 0.40
TAUTOLOGY_RHO = 0.70
NUM_WFE_FOLDS = 5

# Deployed-alternative filters to check tautology against (per lane)
# G5 excluded — it is a conjunction component of the candidate.
ALT_FILTERS_PER_LANE = {
    "CME_PRECLOSE": ["X_MES_ATR60"],
    "TOKYO_OPEN": ["COST_LT12"],
}

# X_MES_ATR60 filter: MES atr_20_pct on same trading_day >= 70
X_MES_ATR60_THRESHOLD = 70.0

# COST_LT12: MNQ cost ratio < 12
COST_LT12_THRESHOLD = 12.0


def _load_candidate_fires(con: duckdb.DuckDBPyConnection, session: str) -> dict:
    """Return {trading_day: (wide_rel_fire_bool, g5_fire_bool)} for this session."""
    q = f"""
    WITH df AS (
      SELECT trading_day, orb_{session}_size AS sz
      FROM daily_features WHERE symbol='{INSTRUMENT}' AND orb_minutes={APERTURE}
        AND orb_{session}_size IS NOT NULL
    ),
    lagged AS (
      SELECT trading_day, sz,
             AVG(sz) OVER (ORDER BY trading_day
                           ROWS BETWEEN {TRAILING_WINDOW_DAYS} PRECEDING
                                    AND 1 PRECEDING) AS rolling20
      FROM df WHERE sz IS NOT NULL
    )
    SELECT trading_day, sz, rolling20 FROM lagged
    """
    out: dict[date, tuple[bool, bool]] = {}
    for td, sz, r20 in con.execute(q).fetchall():
        if r20 is None or r20 == 0:
            continue
        wide = (sz / r20) >= WIDE_REL_THRESHOLD
        g5 = sz >= G5_THRESHOLD
        out[td] = (wide, g5)
    return out


def _load_x_mes_atr60_fires(con: duckdb.DuckDBPyConnection) -> dict:
    """Return {trading_day: x_mes_atr60_fire_bool} (MES atr_20_pct >= 70 on same day)."""
    q = """
    SELECT trading_day, atr_20_pct
    FROM daily_features
    WHERE symbol='MES' AND orb_minutes=5 AND atr_20_pct IS NOT NULL
    """
    return {td: (p is not None and p >= X_MES_ATR60_THRESHOLD) for td, p in con.execute(q).fetchall()}


def _load_cost_lt12_fires(con: duckdb.DuckDBPyConnection, session: str) -> dict:
    """Return {trading_day: cost_lt12_fire_bool} for MNQ at this session."""
    cost_spec = COST_SPECS[INSTRUMENT]
    friction = cost_spec.total_friction
    pv = cost_spec.point_value
    q = f"""
    SELECT trading_day, orb_{session}_size
    FROM daily_features
    WHERE symbol='{INSTRUMENT}' AND orb_minutes={APERTURE}
      AND orb_{session}_size IS NOT NULL
    """
    out: dict[date, bool] = {}
    for td, sz in con.execute(q).fetchall():
        if sz is None or sz <= 0:
            out[td] = False
            continue
        raw_risk = sz * pv
        cost_ratio_pct = 100.0 * friction / (raw_risk + friction)
        out[td] = cost_ratio_pct < COST_LT12_THRESHOLD
    return out


def _load_trades(con: duckdb.DuckDBPyConnection, session: str, rr: float) -> list:
    """Return list of (trading_day, pnl_r) for this cell, in chronological order."""
    q = f"""
    SELECT trading_day, pnl_r FROM orb_outcomes
    WHERE symbol='{INSTRUMENT}' AND orb_label='{session}' AND orb_minutes={APERTURE}
      AND entry_model='{ENTRY_MODEL}' AND confirm_bars={CONFIRM_BARS}
      AND rr_target={rr} AND pnl_r IS NOT NULL
    ORDER BY trading_day
    """
    return con.execute(q).fetchall()


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, v ** 0.5


def _fire_correlation(a: list[bool], b: list[bool]) -> float:
    """Pearson correlation of two boolean sequences (fire indicators)."""
    n = len(a)
    if n == 0 or n != len(b):
        return 0.0
    ai = [1 if x else 0 for x in a]
    bi = [1 if x else 0 for x in b]
    ma = sum(ai) / n
    mb = sum(bi) / n
    num = sum((ai[i] - ma) * (bi[i] - mb) for i in range(n))
    den_a = sum((x - ma) ** 2 for x in ai) ** 0.5
    den_b = sum((x - mb) ** 2 for x in bi) ** 0.5
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)


def _bh_fdr(pvals: list[tuple[str, float]], k: int, q: float) -> list[tuple[str, float, float, bool]]:
    """Return [(cell, p, threshold, pass)] sorted ascending by p."""
    sorted_p = sorted(pvals, key=lambda x: x[1])
    out: list[tuple[str, float, float, bool]] = []
    for i, (cell, p) in enumerate(sorted_p, 1):
        threshold = (i / k) * q
        out.append((cell, p, threshold, p <= threshold))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="MNQ wide-rel-IB v2 pre-reg replay.")
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--output-json", default=str(OUTPUT_JSON))
    args = parser.parse_args()

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        row = con.execute("SELECT MAX(trading_day) FROM daily_features").fetchone()
        if row is None or row[0] is None:
            raise RuntimeError("daily_features is empty")
        as_of: date = row[0]
        print(f"[replay] as_of: {as_of}")

        # Load candidate fires (WIDE_REL, G5) per session
        fires_candidate = {s: _load_candidate_fires(con, s) for s in SESSIONS}
        # Load alt-filter fires
        x_mes_fires = _load_x_mes_atr60_fires(con)
        cost_lt12_fires_per_session = {s: _load_cost_lt12_fires(con, s) for s in SESSIONS}

        # Load trades per cell
        trades_per_cell: dict[tuple[str, float], list] = {}
        for session in SESSIONS:
            for rr in RR_TARGETS:
                trades_per_cell[(session, rr)] = _load_trades(con, session, rr)
    finally:
        con.close()

    # Tautology pre-gate (Rule 7)
    print("\n=== Tautology pre-gate (Rule 7 canonical fire correlation) ===")
    tautology_per_lane: dict[str, dict[str, float]] = {}
    tautology_killed: set[str] = set()
    for session in SESSIONS:
        # Candidate fire = WIDE_REL AND G5 for days that are in fires_candidate
        days = sorted(fires_candidate[session].keys())
        cand_fires = [fires_candidate[session][d][0] and fires_candidate[session][d][1] for d in days]
        tautology_per_lane[session] = {}
        for alt in ALT_FILTERS_PER_LANE[session]:
            if alt == "X_MES_ATR60":
                alt_fires = [x_mes_fires.get(d, False) for d in days]
            elif alt == "COST_LT12":
                alt_fires = [cost_lt12_fires_per_session[session].get(d, False) for d in days]
            else:
                raise ValueError(f"unknown alt filter: {alt}")
            rho = _fire_correlation(cand_fires, alt_fires)
            tautology_per_lane[session][alt] = rho
            verdict = "DUPLICATE" if abs(rho) > TAUTOLOGY_RHO else "distinct"
            print(f"  MNQ {session} vs {alt}: rho={rho:+.4f} -> {verdict}")
            if abs(rho) > TAUTOLOGY_RHO:
                tautology_killed.add(session)

    # Per-cell evaluation
    print("\n=== Per-cell evaluation ===")
    cell_results: list[dict] = []
    for session in SESSIONS:
        for rr in RR_TARGETS:
            cell_key = f"{session}_RR{rr}"
            trades = trades_per_cell[(session, rr)]
            fires = fires_candidate[session]

            # IS trades split by WIDE_REL (given G5 true)
            wide_and_g5_is: list[float] = []
            g5_only_is: list[float] = []
            wide_and_g5_oos: list[float] = []
            g5_only_oos: list[float] = []
            for td, pnl in trades:
                fire = fires.get(td)
                if fire is None:
                    continue
                wide, g5 = fire
                if not g5:
                    continue  # baseline requires G5
                if td < HOLDOUT:
                    if wide:
                        wide_and_g5_is.append(pnl)
                    else:
                        g5_only_is.append(pnl)
                else:
                    if wide:
                        wide_and_g5_oos.append(pnl)
                    else:
                        g5_only_oos.append(pnl)

            m_cand_is, sd_cand_is = _mean_sd(wide_and_g5_is)
            m_base_is, sd_base_is = _mean_sd(g5_only_is)
            delta_is = m_cand_is - m_base_is
            t_stat_is, p_is = ttest_ind_from_stats(
                m_cand_is, sd_cand_is, len(wide_and_g5_is),
                m_base_is, sd_base_is, len(g5_only_is),
                equal_var=False,
            ) if len(wide_and_g5_is) >= 2 and len(g5_only_is) >= 2 else (0.0, 1.0)

            m_cand_oos, _ = _mean_sd(wide_and_g5_oos)
            m_base_oos, _ = _mean_sd(g5_only_oos)
            delta_oos = m_cand_oos - m_base_oos
            n_oos_cand = len(wide_and_g5_oos)

            eff_ratio = (delta_oos / delta_is) if abs(delta_is) > 1e-9 else float("nan")
            dir_match = (delta_is > 0 and delta_oos > 0) or (delta_is < 0 and delta_oos < 0)

            # WFE proxy: annualized Sharpe of (WIDE+G5) OOS over IS
            def _ann_sharpe(xs: list[float]) -> float:
                if not xs:
                    return 0.0
                m, sd = _mean_sd(xs)
                if sd == 0:
                    return 0.0
                return (m / sd) * (252 ** 0.5)

            sr_cand_is = _ann_sharpe(wide_and_g5_is)
            sr_cand_oos = _ann_sharpe(wide_and_g5_oos)
            wfe = sr_cand_oos / sr_cand_is if abs(sr_cand_is) > 1e-9 else 0.0

            # Tautology kill flag
            in_tautology_kill = session in tautology_killed

            # Per-criterion pass checks (pre BH-FDR)
            pass_t = abs(t_stat_is) >= CHORDIA_T_WITHOUT_THEORY
            pass_wfe = wfe >= WFE_MIN
            pass_n_oos = n_oos_cand >= N_OOS_MIN
            pass_dir = dir_match
            pass_eff = (eff_ratio == eff_ratio) and eff_ratio >= EFF_RATIO_MIN  # NaN!=NaN so this filters NaN

            cell_results.append({
                "cell": cell_key,
                "session": session,
                "rr": rr,
                "n_wide_g5_is": len(wide_and_g5_is),
                "n_g5_only_is": len(g5_only_is),
                "expr_wide_g5_is": m_cand_is,
                "expr_g5_only_is": m_base_is,
                "delta_is": delta_is,
                "t_stat_is": t_stat_is,
                "p_is": p_is,
                "n_oos_wide_g5": n_oos_cand,
                "n_oos_g5_only": len(g5_only_oos),
                "expr_wide_g5_oos": m_cand_oos,
                "expr_g5_only_oos": m_base_oos,
                "delta_oos": delta_oos,
                "eff_ratio": eff_ratio,
                "dir_match": dir_match,
                "wfe": wfe,
                "pass_t_3_79": pass_t,
                "pass_wfe_0_50": pass_wfe,
                "pass_n_oos_30": pass_n_oos,
                "pass_dir_match": pass_dir,
                "pass_eff_ratio_0_40": pass_eff,
                "in_tautology_kill_lane": in_tautology_kill,
            })
            print(f"  {cell_key}: IS N={len(wide_and_g5_is)}/{len(g5_only_is)} delta={delta_is:+.4f} t={t_stat_is:+.2f} p={p_is:.4f} | OOS N={n_oos_cand} delta={delta_oos:+.4f} eff={eff_ratio:+.3f} dir_match={dir_match} WFE={wfe:+.3f}")

    # BH-FDR at K=6
    bh_results = _bh_fdr([(c["cell"], c["p_is"]) for c in cell_results], K_FAMILY, BH_FDR_Q)
    bh_pass = {cell: passed for cell, _p, _th, passed in bh_results}
    for c in cell_results:
        c["pass_bh_fdr_k6"] = bh_pass[c["cell"]]
        # Overall per-cell primary verdict
        c["primary_pass"] = (
            c["pass_bh_fdr_k6"]
            and c["pass_t_3_79"]
            and c["pass_wfe_0_50"]
            and c["pass_n_oos_30"]
            and c["pass_dir_match"]
            and c["pass_eff_ratio_0_40"]
            and not c["in_tautology_kill_lane"]
        )

    n_passing = sum(1 for c in cell_results if c["primary_pass"])
    if n_passing >= 4:
        family_verdict = "STRONG_PASS"
    elif n_passing >= 2:
        family_verdict = "STANDARD_PASS"
    elif n_passing == 1:
        family_verdict = "MARGINAL"
    else:
        family_verdict = "NULL"
    print(f"\n=== Family verdict: {family_verdict} ({n_passing}/6 cells pass primary) ===")

    # Emit MD
    lines = [
        "# MNQ Wide-Rel-IB v2 — Stage 1 Replay",
        "",
        f"**Date:** {date.today().isoformat()}",
        f"**As-of trading day:** {as_of.isoformat()}",
        f"**Pre-reg:** `{HYPOTHESIS_FILE}`",
        f"**Design:** `{DESIGN_DOC}`",
        f"**Family verdict:** **{family_verdict}** ({n_passing}/6 cells pass primary)",
        "",
        "## Tautology pre-gate (Rule 7 canonical fire correlation)",
        "",
        "| Lane | Alt filter | fire_rho | |rho|>0.70? | Verdict |",
        "|---|---|---:|:---:|:---:|",
    ]
    for session in SESSIONS:
        for alt, rho in tautology_per_lane[session].items():
            ex = abs(rho) > TAUTOLOGY_RHO
            lines.append(f"| MNQ {session} | {alt} | {rho:+.4f} | {'YES' if ex else 'no'} | {'DUPLICATE_FILTER — kill' if ex else 'distinct'} |")

    lines += [
        "",
        "## Per-cell primary evaluation",
        "",
        "| Cell | IS N(W+G5/G5only) | ExpR IS (W+G5 / G5only) | delta_IS | t_IS | p_IS | BH K=6 | OOS N | delta_OOS | eff_ratio | dir_match | WFE | PRIMARY |",
        "|---|---|---|---:|---:|---:|:---:|---:|---:|---:|:---:|---:|:---:|",
    ]
    for c in cell_results:
        primary = "PASS" if c["primary_pass"] else "FAIL"
        lines.append(
            f"| {c['cell']} | {c['n_wide_g5_is']}/{c['n_g5_only_is']} | "
            f"{c['expr_wide_g5_is']:+.4f} / {c['expr_g5_only_is']:+.4f} | "
            f"{c['delta_is']:+.4f} | {c['t_stat_is']:+.2f} | {c['p_is']:.4f} | "
            f"{'PASS' if c['pass_bh_fdr_k6'] else 'FAIL'} | "
            f"{c['n_oos_wide_g5']} | {c['delta_oos']:+.4f} | {c['eff_ratio']:+.3f} | "
            f"{'PASS' if c['pass_dir_match'] else 'FAIL'} | "
            f"{c['wfe']:+.3f} | **{primary}** |"
        )

    lines += [
        "",
        "## Per-criterion breakdown",
        "",
        "| Cell | BH K=6 | t>=3.79 | WFE>=0.50 | N_OOS>=30 | dir_match | eff>=0.40 | tautology_ok | PRIMARY |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for c in cell_results:
        lines.append(
            f"| {c['cell']} | "
            f"{'PASS' if c['pass_bh_fdr_k6'] else 'FAIL'} | "
            f"{'PASS' if c['pass_t_3_79'] else 'FAIL'} | "
            f"{'PASS' if c['pass_wfe_0_50'] else 'FAIL'} | "
            f"{'PASS' if c['pass_n_oos_30'] else 'FAIL'} | "
            f"{'PASS' if c['pass_dir_match'] else 'FAIL'} | "
            f"{'PASS' if c['pass_eff_ratio_0_40'] else 'FAIL'} | "
            f"{'PASS' if not c['in_tautology_kill_lane'] else 'FAIL'} | "
            f"**{'PASS' if c['primary_pass'] else 'FAIL'}** |"
        )

    lines += [
        "",
        "## BH-FDR K=6 rank table",
        "",
        "| rank | cell | p_IS | threshold | pass |",
        "|---:|---|---:|---:|:---:|",
    ]
    for cell, p, threshold, passed in bh_results:
        lines.append(f"| — | {cell} | {p:.6f} | {threshold:.6f} | {'PASS' if passed else 'FAIL'} |")

    lines += [
        "",
        f"## Family verdict: **{family_verdict}**",
        "",
        f"Cells passing primary: {n_passing}/6",
        "",
        "Verdict semantics per design doc:",
        "- STRONG_PASS (4-6 pass) — proceed to Stage 2, propose promotion",
        "- STANDARD_PASS (2-3 pass) — promote passing cells only",
        "- MARGINAL (1 pass) — reevaluate under allocator correlation gate",
        "- NULL (0 pass) — close family, no rescue",
        "",
    ]
    md = "\n".join(lines) + "\n"
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(md, encoding="utf-8")
    print(f"[replay] wrote {args.output_md}")

    payload = {
        "as_of": as_of.isoformat(),
        "hypothesis_file": HYPOTHESIS_FILE,
        "family_verdict": family_verdict,
        "n_passing": n_passing,
        "tautology_per_lane": tautology_per_lane,
        "cell_results": cell_results,
        "bh_fdr": [{"cell": c, "p": p, "threshold": th, "pass": pa} for c, p, th, pa in bh_results],
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[replay] wrote {args.output_json}")


if __name__ == "__main__":
    main()
