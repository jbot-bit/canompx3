"""T0-T8 audit on 4 new MGC level survivors from mgc_level_scan.py.

Cells (discovered 2026-04-15 MGC level scan, all |t|>=2.5 + dir_match):
  L1. MGC LONDON_METALS O30 RR1.5 long  F2_NEAR_PDL_30  (Δ_OOS=+1.046 — strongest OOS)
  L2. MGC EUROPE_FLOW    O30 RR1.5 short F2_NEAR_PDL_30 (Δ_OOS=+0.612)
  L3. MGC NYSE_OPEN      O15 RR1.0 short F6_INSIDE_PDR  (Δ_OOS=+0.190)
  L4. MGC TOKYO_OPEN     O30 RR2.0 long  F6_INSIDE_PDR  (Δ_OOS=-0.239, SKIP)

These are NEW discoveries beyond prior verified MGC SINGAPORE_OPEN F3_NEAR_PIVOT
(which remains the strongest confirmed cell at CONDITIONAL × 2 RRs).

Reuses t0_t8_audit_prior_day_patterns.py Pattern + audit_pattern().

Output:
  docs/audit/results/2026-04-15-t0-t8-audit-mgc-level-cells.md
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

from research.t0_t8_audit_prior_day_patterns import (  # type: ignore  # noqa: E402
    Pattern,
    audit_pattern,
)

OUTPUT_MD = Path("docs/audit/results/2026-04-15-t0-t8-audit-mgc-level-cells.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class LevelCell:
    name: str
    description: str
    session: str
    aperture: int
    rr: float
    direction: str
    feature_name: str  # e.g. "F2_NEAR_PDL_30", "F6_INSIDE_PDR"
    expected_sign: str  # "positive" for TAKE, "negative" for SKIP


def build_feature_sql(session: str, feature: str) -> tuple[str, float]:
    """Return (feature_sql, theta) per mega-script convention."""
    mid = f"(d.orb_{session}_high + d.orb_{session}_low)/2.0"
    pivot = "(d.prev_day_high + d.prev_day_low + d.prev_day_close)/3.0"

    if feature.startswith("F1_NEAR_PDH_"):
        theta = int(feature.split("_")[-1]) / 100.0
        return (f"CAST((ABS({mid} - d.prev_day_high) / d.atr_20 < {theta}) AS INTEGER)", theta)
    if feature.startswith("F2_NEAR_PDL_"):
        theta = int(feature.split("_")[-1]) / 100.0
        return (f"CAST((ABS({mid} - d.prev_day_low) / d.atr_20 < {theta}) AS INTEGER)", theta)
    if feature.startswith("F3_NEAR_PIVOT_"):
        theta = int(feature.split("_")[-1]) / 100.0
        return (f"CAST((ABS({mid} - {pivot}) / d.atr_20 < {theta}) AS INTEGER)", theta)
    if feature == "F4_ABOVE_PDH":
        return (f"CAST(({mid} > d.prev_day_high) AS INTEGER)", 0.0)
    if feature == "F5_BELOW_PDL":
        return (f"CAST(({mid} < d.prev_day_low) AS INTEGER)", 0.0)
    if feature == "F6_INSIDE_PDR":
        return (f"CAST(({mid} > d.prev_day_low AND {mid} < d.prev_day_high) AS INTEGER)", 0.0)
    raise ValueError(f"unsupported feature: {feature}")


CELLS = [
    LevelCell(
        name="L1_MGC_LONDON_METALS_O30_RR1.5_long_F2_NEAR_PDL_30",
        description="MGC LONDON_METALS O30 RR1.5 LONG F2_NEAR_PDL_30 — Δ_OOS=+1.046 strongest MGC TAKE",
        session="LONDON_METALS", aperture=30, rr=1.5, direction="long",
        feature_name="F2_NEAR_PDL_30", expected_sign="positive",
    ),
    LevelCell(
        name="L2_MGC_EUROPE_FLOW_O30_RR1.5_short_F2_NEAR_PDL_30",
        description="MGC EUROPE_FLOW O30 RR1.5 SHORT F2_NEAR_PDL_30 — Δ_OOS=+0.612",
        session="EUROPE_FLOW", aperture=30, rr=1.5, direction="short",
        feature_name="F2_NEAR_PDL_30", expected_sign="positive",
    ),
    LevelCell(
        name="L3_MGC_NYSE_OPEN_O15_RR1.0_short_F6_INSIDE_PDR",
        description="MGC NYSE_OPEN O15 RR1.0 SHORT F6_INSIDE_PDR — Δ_OOS=+0.190 TAKE",
        session="NYSE_OPEN", aperture=15, rr=1.0, direction="short",
        feature_name="F6_INSIDE_PDR", expected_sign="positive",
    ),
    LevelCell(
        name="L4_MGC_TOKYO_OPEN_O30_RR2.0_long_F6_INSIDE_PDR",
        description="MGC TOKYO_OPEN O30 RR2.0 LONG F6_INSIDE_PDR — Δ_OOS=-0.239 SKIP",
        session="TOKYO_OPEN", aperture=30, rr=2.0, direction="long",
        feature_name="F6_INSIDE_PDR", expected_sign="negative",
    ),
    # Cross-RR family check on L1 — does the signal hold at RR1.0 and RR2.0?
    # This tests RULE: family concordance across RR. Per-RR deployment gate
    # should require at least 2 of 3 RRs validate to deploy any.
    LevelCell(
        name="L1RR10_MGC_LONDON_METALS_O30_RR1.0_long_F2_NEAR_PDL_30",
        description="Cross-RR check: MGC LONDON_METALS O30 RR1.0 LONG F2_NEAR_PDL_30",
        session="LONDON_METALS", aperture=30, rr=1.0, direction="long",
        feature_name="F2_NEAR_PDL_30", expected_sign="positive",
    ),
    LevelCell(
        name="L1RR20_MGC_LONDON_METALS_O30_RR2.0_long_F2_NEAR_PDL_30",
        description="Cross-RR check: MGC LONDON_METALS O30 RR2.0 LONG F2_NEAR_PDL_30",
        session="LONDON_METALS", aperture=30, rr=2.0, direction="long",
        feature_name="F2_NEAR_PDL_30", expected_sign="positive",
    ),
]


def build_patterns() -> list[Pattern]:
    patterns = []
    for cell in CELLS:
        fsql, theta = build_feature_sql(cell.session, cell.feature_name)
        p = Pattern(
            name=cell.name,
            description=cell.description,
            instrument="MGC",
            session=cell.session,
            aperture=cell.aperture,
            rr=cell.rr,
            direction=cell.direction,
            feature_sql=fsql,
            theta=theta,
            expected_sign=cell.expected_sign,
        )
        patterns.append(p)
    return patterns


def emit(results: list[dict]) -> None:
    lines = [
        "# T0-T8 Audit — 4 New MGC Level Cells",
        "",
        "**Date:** 2026-04-15",
        "**Source:** `docs/audit/results/2026-04-15-mgc-level-scan.md` promising list",
        "**Protocol:** `.claude/rules/quant-audit-protocol.md` Steps 3-5",
        "",
        "**MGC-specific caveats (no bias applied):**",
        "- Data window 2022-06 to 2026-04 (~3.8 years) — fewer years for T7 per-year than MES/MNQ",
        "- OOS window 3 months (same as all instruments) — T3 thin at low fire rates",
        "- T8 cross-instrument twin defaults to MNQ in existing code (equity vs gold — wrong asset class)",
        "- 2026 MGC regime shift flagged on prior M1 cell (WFE=0.33) — monitor",
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
            f"**Scope:** MGC | {p.session} | O{p.aperture} | RR{p.rr} | {p.direction} | expected={p.expected_sign}",
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
    print(f"T0-T8 audit on {len(patterns)} new MGC level cells\n")
    results = []
    for p in patterns:
        print(f"\n=== {p.name} ===")
        try:
            r = audit_pattern(p)
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
    emit(results)

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
