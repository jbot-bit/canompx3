#!/usr/bin/env python3
"""Mode A audit of 23 RETIRED validated_setups — look for wrongly-retired lanes.

Correction 2 of 9 from 2026-04-19 adversarial self-audit. Phase 3 audited
only the 38 active lanes. If any of the 23 retired lanes was retired under
Mode B numbers, they might be alive under strict Mode A. Survivorship bias
in my own re-validation work if I only check winners.

Method: same methodology as Phase 3
(research/mode_a_revalidation_active_setups.py) but with
`WHERE LOWER(status) = 'retired'`. For each retired lane compute Mode A
N, ExpR, Sharpe_ann, WR, per-year, then flag any lane that:
  - Has Mode A ExpR > 0.10 AND
  - Has Mode A N >= 100 AND
  - Has Mode A t >= 3.00 (Pathway A Chordia-with-theory bar) AND
  - Has years_positive (Mode A, absolute count) >= 4
These flags IDENTIFY CANDIDATES for committee review to consider reinstating.

Does NOT auto-reinstate. Does NOT mutate validated_setups.

Output: docs/audit/results/2026-04-19-mode-a-audit-of-retired-setups.md
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as _sstats

from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS, CrossAssetATRFilter
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from research.filter_utils import filter_signal

RESULT_PATH = PROJECT_ROOT / "docs/audit/results/2026-04-19-mode-a-audit-of-retired-setups.md"


@dataclass
class RetiredAudit:
    strategy_id: str
    instrument: str
    orb_label: str
    orb_minutes: int
    rr_target: float
    entry_model: str
    confirm_bars: int
    filter_type: str | None
    direction: str = "long"
    retired_at: Any = None
    retirement_reason: str | None = None

    # Stored (Mode B)
    stored_n: int = 0
    stored_expr: float | None = None
    stored_sharpe: float | None = None

    # Mode A canonical
    mode_a_n: int = 0
    mode_a_expr: float | None = None
    mode_a_sharpe: float | None = None
    mode_a_wr: float | None = None
    mode_a_t: float | None = None
    mode_a_raw_p: float | None = None
    mode_a_years_positive: int = 0
    mode_a_years_total: int = 0

    # Verdict
    revive_candidate: bool = False
    revive_reasons: list[str] = field(default_factory=list)
    stay_retired_reasons: list[str] = field(default_factory=list)


def direction_from_execution_spec(spec: str | None) -> str:
    if not spec:
        return "long"
    return "short" if "short" in str(spec).lower() else "long"


def load_retired(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    rows = con.execute("""
        SELECT strategy_id, instrument, orb_label, orb_minutes, rr_target,
               entry_model, confirm_bars, filter_type, sample_size,
               expectancy_r, sharpe_ann, win_rate, retired_at, retirement_reason,
               execution_spec
        FROM validated_setups
        WHERE LOWER(status) = 'retired'
        ORDER BY retired_at DESC NULLS LAST, instrument, orb_label
    """).fetchall()
    cols = ["strategy_id", "instrument", "orb_label", "orb_minutes", "rr_target",
            "entry_model", "confirm_bars", "filter_type", "sample_size",
            "expectancy_r", "sharpe_ann", "win_rate", "retired_at", "retirement_reason",
            "execution_spec"]
    return [dict(zip(cols, r)) for r in rows]


def compute_mode_a(con: duckdb.DuckDBPyConnection, spec: dict[str, Any]) -> RetiredAudit:
    direction = direction_from_execution_spec(spec.get("execution_spec"))
    audit = RetiredAudit(
        strategy_id=spec["strategy_id"],
        instrument=spec["instrument"],
        orb_label=spec["orb_label"],
        orb_minutes=spec["orb_minutes"],
        rr_target=spec["rr_target"],
        entry_model=spec["entry_model"],
        confirm_bars=spec["confirm_bars"],
        filter_type=spec["filter_type"],
        direction=direction,
        retired_at=spec["retired_at"],
        retirement_reason=spec["retirement_reason"],
        stored_n=spec["sample_size"] or 0,
        stored_expr=spec["expectancy_r"],
        stored_sharpe=spec["sharpe_ann"],
    )

    # Instrument must be in ACTIVE list for the re-validation to be meaningful
    from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
    if spec["instrument"] not in ACTIVE_ORB_INSTRUMENTS:
        audit.stay_retired_reasons.append(
            f"Instrument {spec['instrument']} not in ACTIVE_ORB_INSTRUMENTS (dead-for-ORB). Retirement-reason preserved."
        )
        return audit

    sess = spec["orb_label"]
    # Must check column exists for this session (SESSION_CATALOG has diff sessions per instrument)
    test_col = f"orb_{sess}_break_dir"
    # Check via DESCRIBE once (cheap)
    cols = [c[0] for c in con.sql("DESCRIBE daily_features").fetchall()]
    if test_col not in cols:
        audit.stay_retired_reasons.append(
            f"daily_features has no column `{test_col}` — session not tracked for this instrument or sessions list diverged."
        )
        return audit

    sql = f"""
        SELECT o.trading_day, o.pnl_r, o.outcome, o.symbol, d.*
        FROM orb_outcomes o
        JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
        WHERE o.symbol=? AND o.orb_label=? AND o.orb_minutes=?
          AND o.entry_model=? AND o.confirm_bars=? AND o.rr_target=?
          AND d.{test_col}=? AND o.pnl_r IS NOT NULL AND o.trading_day < ?
        ORDER BY o.trading_day
    """
    df = con.execute(sql, [spec["instrument"], sess, spec["orb_minutes"],
                            spec["entry_model"], spec["confirm_bars"], spec["rr_target"],
                            direction, HOLDOUT_SACRED_FROM]).df()
    if len(df) == 0:
        audit.stay_retired_reasons.append("0 IS rows after JOIN — strategy spec may not match any orb_outcomes data.")
        return audit

    # Cross-asset ATR injection for X_MES_ATR60 etc.
    filter_type = spec.get("filter_type")
    if filter_type and filter_type in ALL_FILTERS:
        filt_obj = ALL_FILTERS[filter_type]
        if isinstance(filt_obj, CrossAssetATRFilter):
            source = filt_obj.source_instrument
            if source != spec["instrument"]:
                src_rows = con.execute("""
                    SELECT trading_day, atr_20_pct FROM daily_features
                    WHERE symbol=? AND orb_minutes=5 AND atr_20_pct IS NOT NULL
                """, [source]).fetchall()
                src_map = {}
                for td, pct in src_rows:
                    key = td.date() if hasattr(td, "date") else td
                    src_map[key] = float(pct)
                col = f"cross_atr_{source}_pct"
                df[col] = df["trading_day"].apply(lambda d: src_map.get(d.date() if hasattr(d, "date") else d))

    # Apply filter
    if filter_type and filter_type != "UNFILTERED":
        try:
            fire = np.asarray(filter_signal(df, filter_type, sess)).astype(bool)
        except (KeyError, ValueError) as e:
            audit.stay_retired_reasons.append(f"filter_signal failed: {e}")
            return audit
        df_on = df.loc[fire].reset_index(drop=True)
    else:
        df_on = df

    if len(df_on) == 0:
        audit.stay_retired_reasons.append("0 IS rows after filter — sample-unviable under Mode A.")
        return audit

    pnl = df_on["pnl_r"].astype(float).to_numpy()
    audit.mode_a_n = len(pnl)
    audit.mode_a_expr = float(np.mean(pnl))
    audit.mode_a_wr = float(np.mean(df_on["outcome"].astype(str) == "win"))
    if audit.mode_a_n > 1:
        std = float(np.std(pnl, ddof=1))
        if std > 0:
            se = std / math.sqrt(audit.mode_a_n)
            audit.mode_a_t = audit.mode_a_expr / se
            audit.mode_a_raw_p = float(2.0 * (1.0 - _sstats.t.cdf(abs(audit.mode_a_t), df=audit.mode_a_n - 1)))
            # annualized sharpe — per-cell trade-per-year estimate
            df_on["_year"] = pd.to_datetime(df_on["trading_day"]).dt.year
            n_years = max(1, len(df_on["_year"].unique()))
            trades_per_year = audit.mode_a_n / n_years
            sharpe_per_trade = audit.mode_a_expr / std
            audit.mode_a_sharpe = sharpe_per_trade * math.sqrt(trades_per_year)

    # Per-year positive
    if "_year" not in df_on.columns:
        df_on["_year"] = pd.to_datetime(df_on["trading_day"]).dt.year
    for yr in sorted(df_on["_year"].unique()):
        ym = df_on.loc[df_on["_year"] == yr]
        if len(ym) >= 10:
            audit.mode_a_years_total += 1
            if float(ym["pnl_r"].astype(float).mean()) > 0:
                audit.mode_a_years_positive += 1

    # Revive candidate logic
    reasons_revive = []
    reasons_stay = []
    if audit.mode_a_n < 100:
        reasons_stay.append(f"Mode A N={audit.mode_a_n} < 100 (below deployable floor)")
    if audit.mode_a_expr is None or audit.mode_a_expr <= 0.05:
        reasons_stay.append(f"Mode A ExpR={audit.mode_a_expr} <= 0.05 (below modest-edge floor)")
    if audit.mode_a_t is None or abs(audit.mode_a_t) < 3.00:
        reasons_stay.append(f"Mode A |t|={audit.mode_a_t} < 3.00 (below Chordia with-theory bar)")
    if audit.mode_a_years_positive < 4:
        reasons_stay.append(f"Mode A years_positive={audit.mode_a_years_positive} < 4 (below per-year stability floor)")

    if not reasons_stay:
        reasons_revive.append("Passes all 4 revive-candidate criteria: N>=100 AND ExpR>0.05 AND |t|>=3.00 AND years_positive>=4")
        audit.revive_candidate = True
    audit.revive_reasons = reasons_revive
    audit.stay_retired_reasons = reasons_stay

    return audit


def _fmt(x, p=4):
    if x is None: return "—"
    if isinstance(x, float):
        if math.isnan(x): return "nan"
        return f"{x:.{p}f}"
    return str(x)


def render(audits: list[RetiredAudit]) -> str:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n_revive = sum(1 for a in audits if a.revive_candidate)
    L: list[str] = []
    L.append("# Mode A audit of retired validated_setups — 2026-04-19")
    L.append("")
    L.append(f"**Generated:** {ts}")
    L.append(f"**Script:** `research/mode_a_audit_retired_setups.py`")
    L.append(f"**IS boundary:** `trading_day < {HOLDOUT_SACRED_FROM}` (Mode A)")
    L.append("")
    L.append("## Motivation")
    L.append("")
    L.append("Phase 3 of the 2026-04-19 overnight session re-validated the 38 ACTIVE lanes under Mode A. The session's adversarial self-audit (Correction 2 of 9) identified that 23 RETIRED lanes were NOT similarly re-validated. If any was retired on Mode B numbers where Mode A shows a viable edge, that's survivorship bias in the re-validation work itself.")
    L.append("")
    L.append("This audit recomputes Mode A IS stats for every retired lane and flags any that passes ALL of:")
    L.append("")
    L.append("- Mode A N >= 100 (deployable sample)")
    L.append("- Mode A ExpR > 0.05 (modest positive edge)")
    L.append("- Mode A |t| >= 3.00 (Chordia with-theory bar)")
    L.append("- Mode A years_positive >= 4 (per-year stability, absolute count, N>=10 per year)")
    L.append("")
    L.append("Flagged lanes are REVIVE CANDIDATES for committee review, NOT automatic reinstatements.")
    L.append("")
    L.append(f"## Summary")
    L.append("")
    L.append(f"- Total retired lanes audited: **{len(audits)}**")
    L.append(f"- Revive candidates (pass all 4 criteria): **{n_revive}**")
    L.append(f"- Stay-retired (fails at least one criterion): **{len(audits) - n_revive}**")
    L.append("")
    L.append("## Per-lane Mode A audit")
    L.append("")
    L.append("| Instr | Session | Om | RR | Filter | Dir | Retired | Mode-A N | Mode-A ExpR | Mode-A Sharpe | Mode-A t | Yrs+ | Verdict |")
    L.append("|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|")
    for a in audits:
        verdict = "**REVIVE CANDIDATE**" if a.revive_candidate else "stay-retired"
        ret_d = str(a.retired_at)[:10] if a.retired_at else "—"
        L.append(
            f"| {a.instrument} | {a.orb_label} | {a.orb_minutes} | {a.rr_target} | "
            f"{a.filter_type or 'UNFILTERED'} | {a.direction} | {ret_d} | "
            f"{a.mode_a_n} | {_fmt(a.mode_a_expr)} | {_fmt(a.mode_a_sharpe, 2)} | "
            f"{_fmt(a.mode_a_t, 2)} | {a.mode_a_years_positive}/{a.mode_a_years_total} | {verdict} |"
        )
    L.append("")
    if n_revive > 0:
        L.append("## Revive candidates — detail")
        L.append("")
        for a in audits:
            if not a.revive_candidate:
                continue
            L.append(f"### {a.instrument} {a.orb_label} O{a.orb_minutes} RR{a.rr_target} {a.filter_type or 'UNFILTERED'} {a.direction}")
            L.append(f"- `strategy_id`: `{a.strategy_id}`")
            L.append(f"- Retirement date: {a.retired_at}")
            L.append(f"- Retirement reason (at the time): {a.retirement_reason}")
            L.append(f"- Stored (Mode-B era): N={a.stored_n} ExpR={_fmt(a.stored_expr)} Sharpe_ann={_fmt(a.stored_sharpe, 2)}")
            L.append(f"- Mode A canonical: N={a.mode_a_n} ExpR={_fmt(a.mode_a_expr)} Sharpe_ann={_fmt(a.mode_a_sharpe, 2)} WR={_fmt(a.mode_a_wr, 3)}")
            L.append(f"- Mode A |t|={_fmt(a.mode_a_t, 2)} raw_p={_fmt(a.mode_a_raw_p)} years_positive={a.mode_a_years_positive}/{a.mode_a_years_total}")
            L.append(f"- **Action:** committee review — consider reinstating OR pre-reg re-validation under current regime with Carver forecast-combiner framework if the retirement reason was regime-specific.")
            L.append("")
    L.append("## Stay-retired — summary of stay reasons")
    L.append("")
    reason_counts: dict[str, int] = {}
    for a in audits:
        if a.revive_candidate:
            continue
        key = a.stay_retired_reasons[0] if a.stay_retired_reasons else "no reason recorded"
        # Simplify the key
        if "not in ACTIVE_ORB_INSTRUMENTS" in key: key = "instrument not active"
        elif "N=0 after filter" in key or "0 IS rows" in key: key = "0 rows post-filter"
        elif "Mode A N=" in key: key = "Mode A N<100 (sample-thin)"
        elif "Mode A ExpR=" in key: key = "Mode A ExpR <= 0.05"
        elif "Mode A |t|=" in key: key = "Mode A |t|<3.00"
        elif "Mode A years_positive=" in key: key = "Mode A years_positive<4"
        elif "filter_signal failed" in key: key = "filter failure"
        elif "no column" in key: key = "session not in schema"
        else: key = "other"
        reason_counts[key] = reason_counts.get(key, 0) + 1
    L.append("| Primary reason to stay retired | Count |")
    L.append("|---|---:|")
    for k, v in sorted(reason_counts.items(), key=lambda x: -x[1]):
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("## Reproduction")
    L.append("")
    L.append("```")
    L.append("DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/mode_a_audit_retired_setups.py")
    L.append("```")
    L.append("")
    L.append("Read-only. No writes to validated_setups or experimental_strategies.")
    L.append("")
    return "\n".join(L) + "\n"


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        retired = load_retired(con)
        print(f"Loaded {len(retired)} retired validated_setups")
        audits = []
        for i, spec in enumerate(retired, 1):
            a = compute_mode_a(con, spec)
            audits.append(a)
            flag = "REVIVE" if a.revive_candidate else " "
            print(f"  {flag:6} {i:2}/{len(retired)} {a.instrument:4} {a.orb_label:14} "
                  f"O{a.orb_minutes} RR{a.rr_target} {(a.filter_type or 'UNF'):<20} "
                  f"{a.direction:<5} N={a.mode_a_n:4} ExpR={_fmt(a.mode_a_expr, 3)} t={_fmt(a.mode_a_t, 2)}")
    finally:
        con.close()

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(render(audits), encoding="utf-8")
    print(f"\nWrote {RESULT_PATH.relative_to(PROJECT_ROOT)}")
    n_revive = sum(1 for a in audits if a.revive_candidate)
    print(f"Revive candidates: {n_revive} / {len(audits)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
