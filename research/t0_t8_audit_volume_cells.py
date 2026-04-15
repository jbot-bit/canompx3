"""T0-T8 audit on 4 volume-family cells from 2026-04-15 comprehensive + confluence scans.

Cells audited (per final synthesis):
  V1. MES COMEX_SETTLE O5 RR1.0 short rel_vol_HIGH_Q3            (BH-global, t=+4.89)
  V2. MES TOKYO_OPEN O5 RR1.5 long rel_vol_HIGH_Q3               (BH-global, t=+4.46)
  V3. MNQ SINGAPORE_OPEN O5 RR1.0 short rel_vol_HIGH_Q3          (BH-global, t=+4.27)
  V4. MNQ COMEX_SETTLE O5 RR1.5 short rel_vol_HIGH_Q3_AND_F6_INSIDE_PDR (confluence, t=+3.51, Δ_OOS=+0.276 dir_match)

Each cell's volume threshold is computed per-lane from IS P67 (top tertile)
and hardcoded into the feature_sql for stable reproducibility.

Reuses T0-T8 test functions from research/t0_t8_audit_prior_day_patterns.py.

Output:
  docs/audit/results/2026-04-15-t0-t8-audit-volume-cells.md

Reference:
  - Source scans:
    - docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md (single-factor)
    - docs/audit/results/2026-04-15-volume-confluence-scan.md (confluence)
  - Protocol: .claude/rules/quant-audit-protocol.md
  - Volume theory: Aronson Ch 6 (to extract)
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

import duckdb  # noqa: E402
import numpy as np  # noqa: E402

from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402
from research.t0_t8_audit_prior_day_patterns import (  # type: ignore  # noqa: E402
    Pattern,
    audit_pattern,
)

OUTPUT_MD = Path("docs/audit/results/2026-04-15-t0-t8-audit-volume-cells.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)


def compute_rel_vol_p67(session: str, instrument: str, apt: int, rr: float) -> float:
    """Compute IS P67 of rel_vol for the lane — matches comprehensive scan threshold."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    q = f"""
    SELECT d.rel_vol_{session} AS rel_vol
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
      AND o.symbol = d.symbol
      AND o.orb_minutes = d.orb_minutes
    WHERE o.orb_label = '{session}'
      AND o.symbol = '{instrument}'
      AND o.orb_minutes = {apt}
      AND o.entry_model = 'E2'
      AND o.rr_target = {rr}
      AND o.trading_day < '{HOLDOUT_SACRED_FROM}'
      AND o.pnl_r IS NOT NULL
      AND d.atr_20 IS NOT NULL AND d.atr_20 > 0
      AND d.rel_vol_{session} IS NOT NULL
      AND d.orb_{session}_break_dir IN ('long', 'short')
    """
    df = con.execute(q).df()
    con.close()
    if len(df) < 20:
        raise ValueError(f"Insufficient IS rel_vol data for {instrument} {session} O{apt} RR{rr}")
    return float(np.nanpercentile(df["rel_vol"].astype(float), 67))


@dataclass
class VolumeCell:
    name: str
    description: str
    instrument: str
    session: str
    aperture: int
    rr: float
    direction: str
    expected_sign: str  # 'positive' if we expect on-signal > off-signal

    # Composite support
    with_level_feature: str | None = None  # "F6_INSIDE_PDR" if confluence cell


def build_volume_feature_sql(session: str, p67_threshold: float, with_level: str | None) -> str:
    rel_vol_ge = f"(d.rel_vol_{session} > {p67_threshold:.6f})"
    if with_level is None:
        return f"CAST({rel_vol_ge} AS INTEGER)"
    mid = f"(d.orb_{session}_high + d.orb_{session}_low)/2.0"
    if with_level == "F6_INSIDE_PDR":
        level_expr = f"({mid} > d.prev_day_low AND {mid} < d.prev_day_high)"
        return f"CAST({rel_vol_ge} AND {level_expr} AS INTEGER)"
    raise ValueError(f"unsupported level feature: {with_level}")


CELLS = [
    VolumeCell(
        name="V1_MES_COMEX_SETTLE_O5_RR1.0_short_REL_VOL_HIGH",
        description="MES COMEX_SETTLE O5 RR1.0 SHORT with rel_vol > P67 — BH-global survivor t=+4.89",
        instrument="MES", session="COMEX_SETTLE", aperture=5, rr=1.0, direction="short",
        expected_sign="positive",
    ),
    VolumeCell(
        name="V2_MES_TOKYO_OPEN_O5_RR1.5_long_REL_VOL_HIGH",
        description="MES TOKYO_OPEN O5 RR1.5 LONG with rel_vol > P67 — BH-global survivor t=+4.46",
        instrument="MES", session="TOKYO_OPEN", aperture=5, rr=1.5, direction="long",
        expected_sign="positive",
    ),
    VolumeCell(
        name="V3_MNQ_SINGAPORE_OPEN_O5_RR1.0_short_REL_VOL_HIGH",
        description="MNQ SINGAPORE_OPEN O5 RR1.0 SHORT with rel_vol > P67 — BH-global survivor t=+4.27",
        instrument="MNQ", session="SINGAPORE_OPEN", aperture=5, rr=1.0, direction="short",
        expected_sign="positive",
    ),
    VolumeCell(
        name="V4_MNQ_COMEX_SETTLE_O5_RR1.5_short_REL_VOL_HIGH_AND_F6",
        description="MNQ COMEX_SETTLE O5 RR1.5 SHORT with rel_vol > P67 AND F6_INSIDE_PDR — confluence survivor t=+3.51, Δ_OOS=+0.276",
        instrument="MNQ", session="COMEX_SETTLE", aperture=5, rr=1.5, direction="short",
        expected_sign="positive",
        with_level_feature="F6_INSIDE_PDR",
    ),
]


def build_patterns() -> list[Pattern]:
    patterns = []
    for cell in CELLS:
        p67 = compute_rel_vol_p67(cell.session, cell.instrument, cell.aperture, cell.rr)
        fsql = build_volume_feature_sql(cell.session, p67, cell.with_level_feature)
        p = Pattern(
            name=cell.name,
            description=cell.description + f" (rel_vol threshold P67={p67:.3f})",
            instrument=cell.instrument,
            session=cell.session,
            aperture=cell.aperture,
            rr=cell.rr,
            direction=cell.direction,
            feature_sql=fsql,
            theta=0.0,  # binary feature, no theta sensitivity (T4 skipped)
            expected_sign=cell.expected_sign,
        )
        patterns.append(p)
    return patterns


def emit(results: list[dict]) -> None:
    lines = [
        "# T0-T8 Audit — Volume Cells from 2026-04-15 Scans",
        "",
        "**Date:** 2026-04-15",
        "**Source scans:** comprehensive-deployed-lane-scan.md + volume-confluence-scan.md",
        "**Protocol:** `.claude/rules/quant-audit-protocol.md` Steps 3-5",
        "",
    ]
    for r in results:
        p = r["pattern"]
        tests = r["tests"]
        pc = sum(1 for tr in tests.values() if tr.pass_status == "PASS")
        fc = sum(1 for tr in tests.values() if tr.pass_status == "FAIL")
        ic = len(tests) - pc - fc
        if fc == 0 and pc >= 5:
            verdict = "**VALIDATED**"
        elif fc == 1 and pc >= 5:
            verdict = "**CONDITIONAL**"
        elif fc >= 2:
            verdict = "**KILL_DOWNGRADE**"
        else:
            verdict = "**INFO_HEAVY**"
        lines += [
            f"## {p.name}",
            f"**Description:** {p.description}",
            f"**Scope:** {p.instrument} | {p.session} | O{p.aperture} | RR{p.rr} | {p.direction} | expected={p.expected_sign}",
            f"**N_total:** {r['n_total']} | **N_on_signal:** {r['n_on']}",
            "",
            "| Test | Value | Status | Detail |",
            "|------|-------|--------|--------|",
        ]
        for tname, tr in tests.items():
            lines.append(f"| {tname} {tr.name} | {tr.value} | **{tr.pass_status}** | {tr.detail} |")
        lines += [
            "",
            f"**Counts:** {pc} PASS, {fc} FAIL, {ic} INFO",
            f"### Verdict: {verdict}",
            "",
            "---",
            "",
        ]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {OUTPUT_MD}")


def main():
    patterns = build_patterns()
    print(f"Auditing {len(patterns)} volume cells with full T0-T8 battery\n")
    results = []
    for p in patterns:
        print(f"\n=== {p.name} ===")
        try:
            r = audit_pattern(p)
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
    emit(results)

    # Summary
    print("\n=== FINAL VERDICTS ===")
    for r in results:
        p = r["pattern"]
        tests = r["tests"]
        pc = sum(1 for tr in tests.values() if tr.pass_status == "PASS")
        fc = sum(1 for tr in tests.values() if tr.pass_status == "FAIL")
        if fc == 0 and pc >= 5:
            verdict = "VALIDATED"
        elif fc == 1 and pc >= 5:
            verdict = "CONDITIONAL"
        elif fc >= 2:
            verdict = "KILL_DOWNGRADE"
        else:
            verdict = "INFO_HEAVY"
        print(f"  {p.name}: {verdict} ({pc}P/{fc}F)")


if __name__ == "__main__":
    main()
