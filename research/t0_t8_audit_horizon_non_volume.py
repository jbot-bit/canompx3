"""T0-T8 audit on 5 non-volume horizon candidates from 2026-04-15 comprehensive scan.

Source: docs/handoffs/2026-04-15-session-handover.md § Tier 1.
Prior scan: docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md.

Cells (all survived comprehensive BH_family at K=14261 with dir_match):
  H1. MES LONDON_METALS O30 RR1.5 long  ovn_range_pct_GT80   (t=+3.54, ΔOOS=+0.690)
  H2. MNQ COMEX_SETTLE O5  RR1.0 long   garch_vol_pct_GT70   (t=+3.18, ΔOOS=+0.236)
  H3. MNQ BRISBANE_1025 O30 RR2.0 long  is_monday            (t=+3.27, ΔOOS=+0.854)
  H4. MNQ COMEX_SETTLE O15 RR1.0 short  dow_thu              (t=+3.27, ΔOOS=+0.552)
  H5. MES COMEX_SETTLE O30 RR1.0 long   ovn_took_pdh_SKIP    (t=-3.82)  # SKIP signal

Look-ahead gates per .claude/rules/backtesting-methodology.md RULE 1.2:
  - ovn_* features valid only for ORB starting ≥ 17:00 Brisbane:
      LONDON_METALS 17:00 ✓, EUROPE_FLOW 18:00 ✓, COMEX_SETTLE 04:30 (next day) ✓
  - garch_* forecast at prior close — always valid ✓
  - is_monday / day_of_week — calendar — always valid ✓

T0 custom: tautology proxies EXCLUDE the cell's own feature (self-correlation trivially 1.0).
T4 custom: feature-class-specific threshold grid; binary features return INFO (no theta).
T1-T3, T5-T8: import from t0_t8_audit_prior_day_patterns unchanged.

Output: docs/audit/results/2026-04-15-t0-t8-audit-horizon-non-volume.md
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from research.t0_t8_audit_prior_day_patterns import (  # type: ignore  # noqa: E402
    Pattern,
    TestResult,
    load_pattern,
    t1_wr_monotonicity,
    t2_is_baseline,
    t3_oos_wfe,
    t5_family,
    t6_null_floor,
    t7_per_year,
    t8_cross_instrument,
)

OUTPUT_MD = Path("docs/audit/results/2026-04-15-t0-t8-audit-horizon-non-volume.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Cell definitions
# =============================================================================


@dataclass
class HorizonCell:
    name: str
    description: str
    instrument: str
    session: str
    aperture: int
    rr: float
    direction: str
    feature_class: str  # 'ovn_range_pct', 'garch_vol_pct', 'is_monday', 'dow_thu', 'ovn_took_pdh'
    feature_sql: str
    expected_sign: str
    self_proxy: str  # which T0 proxy IS this feature (exclude from tautology check)


CELLS: list[HorizonCell] = [
    HorizonCell(
        name="H1_MES_LONDON_METALS_O30_RR1.5_long_ovn_range_pct_GT80",
        description="MES LONDON_METALS O30 RR1.5 LONG ovn_range_pct≥80 — overnight vol expansion predicts continuation",
        instrument="MES",
        session="LONDON_METALS",
        aperture=30,
        rr=1.5,
        direction="long",
        feature_class="ovn_range_pct",
        feature_sql="CAST((d.overnight_range_pct >= 80) AS INTEGER)",
        expected_sign="positive",
        self_proxy="ovn80_fire",
    ),
    HorizonCell(
        name="H2_MNQ_COMEX_SETTLE_O5_RR1.0_long_garch_vol_pct_GT70",
        description="MNQ COMEX_SETTLE O5 RR1.0 LONG garch_forecast_vol_pct≥70 — forward vol forecast",
        instrument="MNQ",
        session="COMEX_SETTLE",
        aperture=5,
        rr=1.0,
        direction="long",
        feature_class="garch_vol_pct",
        feature_sql="CAST((d.garch_forecast_vol_pct >= 70) AS INTEGER)",
        expected_sign="positive",
        self_proxy="",  # not a deployed proxy — no self-overlap
    ),
    HorizonCell(
        name="H3_MNQ_BRISBANE_1025_O30_RR2.0_long_is_monday",
        description="MNQ BRISBANE_1025 O30 RR2.0 LONG is_monday — Monday-open effect",
        instrument="MNQ",
        session="BRISBANE_1025",
        aperture=30,
        rr=2.0,
        direction="long",
        feature_class="is_monday",
        feature_sql="CAST(d.is_monday AS INTEGER)",
        expected_sign="positive",
        self_proxy="",
    ),
    HorizonCell(
        name="H4_MNQ_COMEX_SETTLE_O15_RR1.0_short_dow_thu",
        description="MNQ COMEX_SETTLE O15 RR1.0 SHORT dow_thu — Thursday effect",
        instrument="MNQ",
        session="COMEX_SETTLE",
        aperture=15,
        rr=1.0,
        direction="short",
        feature_class="dow_thu",
        feature_sql="CAST((d.day_of_week = 3) AS INTEGER)",
        expected_sign="positive",
        self_proxy="",
    ),
    HorizonCell(
        name="H5_MES_COMEX_SETTLE_O30_RR1.0_long_ovn_took_pdh_SKIP",
        description="MES COMEX_SETTLE O30 RR1.0 LONG — SKIP when ovn_took_pdh (continuation already spent)",
        instrument="MES",
        session="COMEX_SETTLE",
        aperture=30,
        rr=1.0,
        direction="long",
        feature_class="ovn_took_pdh",
        feature_sql="CAST(COALESCE(d.overnight_took_pdh, false) AS INTEGER)",
        expected_sign="negative",  # SKIP signal: feature=1 → lower ExpR
        self_proxy="",
    ),
]


# =============================================================================
# Custom T0 — exclude self-proxy from tautology check
# =============================================================================


def t0_tautology_horizon(cell: HorizonCell, df: pd.DataFrame) -> TestResult:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    days = tuple(str(d.date()) for d in df["trading_day"].unique())
    if not days:
        con.close()
        return TestResult("T0_tautology", "no_data", "INFO", "no trading days")
    days_sql = "','".join(days)
    q = f"""
    SELECT trading_day,
           CAST(prev_day_range/NULLIF(atr_20,0) >= 1.05 AS INT) AS pdr_r105_fire,
           CAST(ABS(gap_open_points)/NULLIF(atr_20,0) >= 0.015 AS INT) AS gap_r015_fire,
           CAST(atr_20_pct >= 70 AS INT) AS atr70_fire,
           CAST(overnight_range_pct >= 80 AS INT) AS ovn80_fire
    FROM daily_features
    WHERE symbol='{cell.instrument}' AND orb_minutes={cell.aperture}
      AND trading_day IN ('{days_sql}')
    """
    proxies = con.execute(q).df()
    con.close()
    proxies["trading_day"] = pd.to_datetime(proxies["trading_day"])
    merged = df[["trading_day", "feature"]].drop_duplicates().merge(proxies, on="trading_day", how="left")

    all_proxies = ["pdr_r105_fire", "gap_r015_fire", "atr70_fire", "ovn80_fire"]
    proxies_to_check = [p for p in all_proxies if p != cell.self_proxy]

    corrs: dict[str, float] = {}
    for col in proxies_to_check:
        valid = merged[["feature", col]].dropna()
        if len(valid) > 10:
            c = valid["feature"].corr(valid[col])
            corrs[col] = float(c) if not np.isnan(c) else 0.0

    max_corr = max((abs(v) for v in corrs.values()), default=0.0)
    max_filt = max(corrs, key=lambda k: abs(corrs[k])) if corrs else "none"
    note = f"excluded self_proxy={cell.self_proxy or 'none'}; corrs={corrs}"

    if max_corr > 0.70:
        return TestResult(
            "T0_tautology", f"{max_corr:.3f} vs {max_filt}", "FAIL", f"DUPLICATE_FILTER with {max_filt}; {note}"
        )
    return TestResult("T0_tautology", f"max |corr|={max_corr:.3f} ({max_filt})", "PASS", note)


# =============================================================================
# Custom T4 — feature-class-specific sensitivity
# =============================================================================


def t4_sensitivity_horizon(cell: HorizonCell, df: pd.DataFrame) -> TestResult:
    is_df = df[df["is_is"]].copy()
    if cell.feature_class == "ovn_range_pct":
        # Test threshold at 70/80/90
        grid = [70, 80, 90]
        col = "overnight_range_pct"
    elif cell.feature_class == "garch_vol_pct":
        grid = [60, 70, 80]
        col = "garch_forecast_vol_pct"
    else:
        return TestResult(
            "T4_sensitivity", "N/A", "INFO", f"binary feature ({cell.feature_class}) — no theta grid applicable"
        )

    # Pull underlying column for IS days
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    days = tuple(str(d.date()) for d in is_df["trading_day"].unique())
    if not days:
        con.close()
        return TestResult("T4_sensitivity", "no_IS_data", "FAIL", "empty IS frame")
    days_sql = "','".join(days)
    q = f"""
    SELECT trading_day, {col} AS raw
    FROM daily_features
    WHERE symbol='{cell.instrument}' AND orb_minutes={cell.aperture}
      AND trading_day IN ('{days_sql}')
    """
    raw = con.execute(q).df()
    con.close()
    raw["trading_day"] = pd.to_datetime(raw["trading_day"])
    merged = is_df.merge(raw, on="trading_day", how="left")

    deltas = []
    for th in grid:
        fire = (merged["raw"] >= th).astype(int)
        on = merged.loc[fire == 1, "pnl_r"]
        off = merged.loc[fire == 0, "pnl_r"]
        if len(on) >= 30 and len(off) >= 30:
            deltas.append(float(on.mean() - off.mean()))
        else:
            deltas.append(float("nan"))

    detail_vals = ", ".join(f"θ={g}:Δ={d:+.3f}" for g, d in zip(grid, deltas))

    if any(np.isnan(deltas)):
        return TestResult("T4_sensitivity", detail_vals, "INFO", f"N<30 at some threshold in grid {grid}")
    signs = [np.sign(d) for d in deltas]
    if not all(s == signs[0] for s in signs):
        return TestResult(
            "T4_sensitivity", detail_vals, "FAIL", "sign flips across threshold grid — PARAMETER_SENSITIVE"
        )
    mid_mag = abs(deltas[1])
    adj_min = min(abs(deltas[0]), abs(deltas[2]))
    if mid_mag > 0 and adj_min < 0.25 * mid_mag:
        return TestResult(
            "T4_sensitivity", detail_vals, "FAIL", "adjacent-threshold magnitude < 25% primary — knife-edge"
        )
    return TestResult("T4_sensitivity", detail_vals, "PASS", "signs consistent, magnitudes within 25% band")


# =============================================================================
# Audit runner
# =============================================================================


def audit_cell(cell: HorizonCell) -> dict:
    print(f"\n=== {cell.name} ===")
    pat = Pattern(
        name=cell.name,
        description=cell.description,
        instrument=cell.instrument,
        session=cell.session,
        aperture=cell.aperture,
        rr=cell.rr,
        direction=cell.direction,
        feature_sql=cell.feature_sql,
        theta=0.0,  # unused in our T4 override
        expected_sign=cell.expected_sign,
    )
    df = load_pattern(pat)
    n_total = len(df)
    n_on = int(df["feature"].sum())
    print(f"  loaded {n_total} trades, {n_on} on-signal")

    tests = {
        "T0": t0_tautology_horizon(cell, df),
        "T1": t1_wr_monotonicity(pat, df),
        "T2": t2_is_baseline(pat, df),
        "T3": t3_oos_wfe(pat, df),
        "T4": t4_sensitivity_horizon(cell, df),
        "T5": t5_family(pat),
        "T6": t6_null_floor(pat, df),
        "T7": t7_per_year(pat, df),
        "T8": t8_cross_instrument(pat),
    }

    for tname, tr in tests.items():
        print(f"  {tname} {tr.name}: {tr.pass_status} — {tr.detail}")

    return {"cell": cell, "pattern": pat, "n_total": n_total, "n_on": n_on, "tests": tests}


def emit(results: list[dict]) -> None:
    lines = [
        "# T0-T8 Audit — 5 Non-Volume Horizon Candidates",
        "",
        "**Date:** 2026-04-15",
        "**Source:** `docs/handoffs/2026-04-15-session-handover.md` § Tier 1",
        "**Prior scan:** `docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md`",
        "**Audit protocol:** `.claude/rules/quant-audit-protocol.md` Steps 3-5",
        "**Template:** `research/t0_t8_audit_mgc_level_cells.py` (imports helpers from `t0_t8_audit_prior_day_patterns.py`)",
        "",
        "**Pre-reg posture:** Confirmatory T0-T8 on prior-scan BH_family survivors — per `backtesting-methodology.md` RULE 10, no new pre-reg required.",
        "",
        "**Look-ahead clearance:**",
        "- `overnight_*` features require ORB start ≥ 17:00 Brisbane — LONDON_METALS (17:00), COMEX_SETTLE (04:30 next day) both clear.",
        "- `garch_forecast_vol_pct` forecast at prior close — always trade-time-knowable.",
        "- `is_monday` / `day_of_week` calendar — always trade-time-knowable.",
        "",
        "**Custom test notes:**",
        "- T0 excludes the cell's own deployed-filter proxy to avoid 100% self-correlation.",
        "- T4 uses feature-class-specific threshold grids for percentile features; binary features return INFO.",
        "- T8 twin for MES↔MNQ is same-asset-class (equity index pair) — valid. MGC has no such twin in this cell set.",
        "",
    ]

    for r in results:
        cell = r["cell"]
        tests = r["tests"]
        pc = sum(1 for tr in tests.values() if tr.pass_status == "PASS")
        fc = sum(1 for tr in tests.values() if tr.pass_status == "FAIL")
        ic = len(tests) - pc - fc

        if fc == 0 and pc >= 5:
            verdict = "**VALIDATED** — deploy candidate. Pre-reg Stage 1 binary filter."
        elif fc == 1 and pc >= 5:
            verdict = "**CONDITIONAL** — one fail; acceptable if non-load-bearing (e.g., thin OOS N)."
        elif fc >= 2:
            verdict = "**KILL / DOWNGRADE** — multiple failures."
        else:
            verdict = "**INFO_HEAVY** — too many INFO; need more data."

        lines += [
            f"## {cell.name}",
            f"**Description:** {cell.description}",
            f"**Scope:** {cell.instrument} | {cell.session} | O{cell.aperture} | RR{cell.rr} | {cell.direction} | expected={cell.expected_sign}",
            f"**Feature class:** `{cell.feature_class}`",
            f"**N_total:** {r['n_total']} | **N_on_signal:** {r['n_on']}",
            "",
            "| Test | Value | Status | Detail |",
            "|------|-------|--------|--------|",
        ]
        for tname, tr in tests.items():
            val = str(tr.value) if isinstance(tr.value, str) else f"{tr.value}"
            lines.append(f"| {tname} {tr.name} | {val} | **{tr.pass_status}** | {tr.detail} |")
        lines += [
            "",
            f"**Counts:** {pc} PASS, {fc} FAIL, {ic} INFO",
            f"### Verdict: {verdict}",
            "",
            "---",
            "",
        ]

    lines += [
        "## Summary Table",
        "",
        "| Cell | P/F/I | Verdict |",
        "|------|-------|---------|",
    ]
    for r in results:
        cell = r["cell"]
        tests = r["tests"]
        pc = sum(1 for tr in tests.values() if tr.pass_status == "PASS")
        fc = sum(1 for tr in tests.values() if tr.pass_status == "FAIL")
        ic = len(tests) - pc - fc
        if fc == 0 and pc >= 5:
            v = "VALIDATED"
        elif fc == 1 and pc >= 5:
            v = "CONDITIONAL"
        elif fc >= 2:
            v = "KILL/DOWNGRADE"
        else:
            v = "INFO_HEAVY"
        lines.append(f"| {cell.name} | {pc}P/{fc}F/{ic}I | {v} |")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {OUTPUT_MD}")


def main() -> None:
    print(f"T0-T8 audit on {len(CELLS)} non-volume horizon candidates\n")
    results = []
    for cell in CELLS:
        try:
            results.append(audit_cell(cell))
        except Exception as e:
            import traceback

            print(f"  ERROR on {cell.name}: {e}")
            traceback.print_exc()
    emit(results)

    print("\n=== FINAL VERDICTS ===")
    for r in results:
        cell = r["cell"]
        tests = r["tests"]
        pc = sum(1 for tr in tests.values() if tr.pass_status == "PASS")
        fc = sum(1 for tr in tests.values() if tr.pass_status == "FAIL")
        if fc == 0 and pc >= 5:
            v = "VALIDATED"
        elif fc == 1 and pc >= 5:
            v = "CONDITIONAL"
        elif fc >= 2:
            v = "KILL/DOWNGRADE"
        else:
            v = "INFO_HEAVY"
        print(f"  {cell.name}: {v} ({pc}P/{fc}F)")


if __name__ == "__main__":
    main()
