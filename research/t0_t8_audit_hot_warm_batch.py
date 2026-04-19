"""T0-T8 batch audit on ALL remaining HOT + WARM cells from mega-exploration.

Extends `research/t0_t8_audit_prior_day_patterns.py` (which covered 3 O5 patterns
P1/P2/P3) to the full set of mega-exploration survivors with holdout dir match.

Input:
  docs/audit/results/2026-04-15-prior-day-features-orb-mega-exploration.md

Output:
  docs/audit/results/2026-04-15-t0-t8-audit-hot-warm-batch.md

This is CONFIRMATORY audit (mega-exploration was exploratory), not new discovery.
No new pre-registration needed — this stress-tests previously-catalogued cells
against the same T0-T8 battery already applied to P1/P2/P3.

References:
  - `.claude/rules/quant-audit-protocol.md` Steps 3-5
  - `docs/audit/results/2026-04-15-prior-day-features-orb-mega-exploration.md`
  - `docs/audit/results/2026-04-15-t0-t8-audit-o5-patterns.md` (the 3 prior)
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8 stdout on Windows so Δ / ≥ print cleanly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from research.t0_t8_audit_prior_day_patterns import (  # type: ignore  # noqa: E402
    Pattern,
    audit_pattern,
)

MEGA_MD = Path("docs/audit/results/2026-04-15-prior-day-features-orb-mega-exploration.md")
OUTPUT_MD = Path("docs/audit/results/2026-04-15-t0-t8-audit-hot-warm-batch.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

# Already-audited cells (from t0_t8_audit_prior_day_patterns.py)
ALREADY_AUDITED = {
    ("MNQ", "NYSE_CLOSE", 5, 1.5, "long", "F3_NEAR_PIVOT_15"),
    ("MES", "NYSE_CLOSE", 5, 1.5, "short", "F1_NEAR_PDH_15"),
    ("MNQ", "US_DATA_1000", 5, 1.0, "long", "F5_BELOW_PDL"),
}


# =============================================================================
# Pattern factory — builds feature_sql from signal name
# =============================================================================


def build_feature_sql(signal: str, session: str) -> tuple[str, float]:
    """Return (feature_sql, theta) for a given signal name.

    Signal names per mega script:
      F1_NEAR_PDH_{15,30,50}, F2_NEAR_PDL_{15,30,50}, F3_NEAR_PIVOT_{15,30,50}
      F4_ABOVE_PDH, F5_BELOW_PDL, F6_INSIDE_PDR, F7_GAP_UP, F8_GAP_DOWN
    """
    mid = f"(d.orb_{session}_high + d.orb_{session}_low)/2.0"
    pivot = "(d.prev_day_high + d.prev_day_low + d.prev_day_close)/3.0"

    if signal.startswith("F1_NEAR_PDH_"):
        theta = int(signal.split("_")[-1]) / 100.0
        return (f"CAST((ABS({mid} - d.prev_day_high) / d.atr_20 < {theta}) AS INTEGER)", theta)
    if signal.startswith("F2_NEAR_PDL_"):
        theta = int(signal.split("_")[-1]) / 100.0
        return (f"CAST((ABS({mid} - d.prev_day_low) / d.atr_20 < {theta}) AS INTEGER)", theta)
    if signal.startswith("F3_NEAR_PIVOT_"):
        theta = int(signal.split("_")[-1]) / 100.0
        return (f"CAST((ABS({mid} - {pivot}) / d.atr_20 < {theta}) AS INTEGER)", theta)
    if signal == "F4_ABOVE_PDH":
        return (f"CAST(({mid} > d.prev_day_high) AS INTEGER)", 0.0)
    if signal == "F5_BELOW_PDL":
        return (f"CAST(({mid} < d.prev_day_low) AS INTEGER)", 0.0)
    if signal == "F6_INSIDE_PDR":
        return (f"CAST(({mid} > d.prev_day_low AND {mid} < d.prev_day_high) AS INTEGER)", 0.0)
    if signal == "F7_GAP_UP":
        return ("CAST((d.gap_type = 'gap_up') AS INTEGER)", 0.0)
    if signal == "F8_GAP_DOWN":
        return ("CAST((d.gap_type = 'gap_down') AS INTEGER)", 0.0)
    raise ValueError(f"unknown signal: {signal}")


def expected_sign_from_delta(delta_is: float) -> str:
    return "positive" if delta_is > 0 else "negative"


# =============================================================================
# Parser for mega-exploration markdown
# =============================================================================


def parse_mega_md() -> list[Pattern]:
    """Extract all HOT + WARM cells with dir_match=Y, excluding P1/P2/P3."""
    text = MEGA_MD.read_text(encoding="utf-8")
    patterns: list[Pattern] = []
    seen: set[tuple] = set()

    # Pattern for HOT/WARM bullet lines in the "## HOT cells" and "## WARM cells" sections
    # Format: "- MNQ US_DATA_1000 O5 RR1.0 long F5_BELOW_PDL: t_cl=+4.24 Δ_IS=+0.337 ..."
    bullet_re = re.compile(
        r"^- ([A-Z]{3}) ([A-Z_0-9]+) O(\d+) RR([\d.]+) (long|short) (F\d_[A-Z_0-9]+?):"
        r"\s+t_cl=([+-][\d.]+)"
        r"\s+\u0394_IS=([+-][\d.]+)",
        re.MULTILINE,
    )

    for m in bullet_re.finditer(text):
        instr, session, apt, rr, direction, signal, t_cl, delta_is = m.groups()
        key = (instr, session, int(apt), float(rr), direction, signal)
        if key in ALREADY_AUDITED or key in seen:
            continue
        seen.add(key)

        try:
            feature_sql, theta = build_feature_sql(signal, session)
        except ValueError as e:
            print(f"[skip] {key}: {e}")
            continue

        exp_sign = expected_sign_from_delta(float(delta_is))
        name = f"{instr}_{session}_O{apt}_RR{rr}_{direction}_{signal}"
        patterns.append(
            Pattern(
                name=name,
                description=f"{signal} {direction.upper()} on {session} {instr} (mega t_cl={t_cl})",
                instrument=instr,
                session=session,
                aperture=int(apt),
                rr=float(rr),
                direction=direction,
                feature_sql=feature_sql,
                theta=theta,
                expected_sign=exp_sign,
            )
        )

    return patterns


# =============================================================================
# Reporting
# =============================================================================


def classify_verdict(tests: dict) -> str:
    pass_count = sum(1 for tr in tests.values() if tr.pass_status == "PASS")
    fail_count = sum(1 for tr in tests.values() if tr.pass_status == "FAIL")
    if fail_count == 0 and pass_count >= 5:
        return "VALIDATED"
    if fail_count == 1 and pass_count >= 5:
        return "CONDITIONAL"
    if fail_count >= 2:
        return "KILL_DOWNGRADE"
    return "INFO_HEAVY"


def emit(all_results: list[dict]) -> None:
    lines = [
        "# T0-T8 Batch Audit — All HOT + WARM Mega-Exploration Survivors",
        "",
        "**Date:** 2026-04-15",
        "**Source:** `docs/audit/results/2026-04-15-prior-day-features-orb-mega-exploration.md`",
        "**Excludes:** P1, P2, P3 (already audited in `2026-04-15-t0-t8-audit-o5-patterns.md`)",
        "**Audit protocol:** `.claude/rules/quant-audit-protocol.md` Steps 3-5",
        "",
        "## Summary Table",
        "",
        "| Cell | Verdict | P/F/I | N_on | ExpR_on | Expected_Sign | Key FAILs |",
        "|------|---------|-------|------|---------|---------------|-----------|",
    ]

    for r in all_results:
        p = r["pattern"]
        tests = r["tests"]
        verdict = classify_verdict(tests)
        pc = sum(1 for tr in tests.values() if tr.pass_status == "PASS")
        fc = sum(1 for tr in tests.values() if tr.pass_status == "FAIL")
        ic = len(tests) - pc - fc
        # Extract ExpR from T2
        t2 = tests.get("T2")
        expr_str = "N/A"
        if t2 and "ExpR=" in str(t2.value):
            m = re.search(r"ExpR=([+-][\d.]+)", str(t2.value))
            if m:
                expr_str = m.group(1)
        fails = [tname for tname, tr in tests.items() if tr.pass_status == "FAIL"]
        lines.append(
            f"| {p.name} | **{verdict}** | {pc}P/{fc}F/{ic}I | {r['n_on']} | {expr_str} | {p.expected_sign} | {','.join(fails) or '—'} |"
        )

    lines += ["", "---", "", "## Per-Cell Detail", ""]

    for r in all_results:
        p = r["pattern"]
        tests = r["tests"]
        lines += [
            f"### {p.name}",
            f"**Scope:** {p.instrument} | {p.session} | O{p.aperture} | RR{p.rr} | {p.direction} | signal={p.description}",
            f"**N_total:** {r['n_total']} | **N_on_signal:** {r['n_on']}",
            "",
            "| Test | Value | Status | Detail |",
            "|------|-------|--------|--------|",
        ]
        pc = 0
        fc = 0
        for tname, tr in tests.items():
            val = str(tr.value)
            lines.append(f"| {tname} {tr.name} | {val} | **{tr.pass_status}** | {tr.detail} |")
            if tr.pass_status == "PASS":
                pc += 1
            elif tr.pass_status == "FAIL":
                fc += 1
        ic = len(tests) - pc - fc
        verdict = classify_verdict(tests)
        lines += [
            "",
            f"**Test counts:** {pc} PASS, {fc} FAIL, {ic} INFO",
            f"### Verdict: **{verdict}**",
            "",
            "---",
            "",
        ]

    # Aggregated counts
    verdicts = [classify_verdict(r["tests"]) for r in all_results]
    counts = {v: verdicts.count(v) for v in set(verdicts)}
    lines = (
        lines[:9]
        + [
            "**Verdict totals:** "
            + ", ".join(
                f"{v}={counts.get(v, 0)}" for v in ["VALIDATED", "CONDITIONAL", "KILL_DOWNGRADE", "INFO_HEAVY"]
            ),
            "",
        ]
        + lines[9:]
    )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {OUTPUT_MD}")


def main():
    patterns = parse_mega_md()
    print(f"Loaded {len(patterns)} unaudited HOT+WARM cells from mega-exploration")
    for p in patterns:
        print(f"  {p.name}")

    all_results = []
    for i, p in enumerate(patterns, 1):
        print(f"\n[{i}/{len(patterns)}] {p.name}")
        try:
            r = audit_pattern(p)
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    emit(all_results)

    verdicts = [classify_verdict(r["tests"]) for r in all_results]
    print("\n=== VERDICT SUMMARY ===")
    for v in ["VALIDATED", "CONDITIONAL", "KILL_DOWNGRADE", "INFO_HEAVY"]:
        n = verdicts.count(v)
        print(f"  {v}: {n}")


if __name__ == "__main__":
    main()
