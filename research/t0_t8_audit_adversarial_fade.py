"""T0-T8 adversarial-fade audit — test if SKIP signals become FADE signals.

Hypothesis: for each cell where taking the first ORB break in direction X loses
money reliably (validated in mega-exploration + T0-T8 batch), the OPPOSITE
direction should make money — because the same mechanism (level rejection)
that hurts longs helps shorts and vice versa.

Input: ALL 30 HOT/WARM cells from mega-exploration (same source as batch audit).
Transformation: flip the cell's direction. long → short and short → long.
Keep instrument, session, aperture, RR, signal, theta the same.
Re-audit with T0-T8 at the new direction.

A FADE validates if:
- Expected sign flips to positive (was negative on original direction)
- T2 is_baseline PASS / CONDITIONAL
- T6 null floor bootstrap PASS
- T7 per-year PASS (≥70% years matching new positive direction)
- T8 cross-instrument PASS (twin matches)
- Additional: OOS direction matches IS (is_oos delta sign)

Output:
  docs/audit/results/2026-04-15-t0-t8-adversarial-fade-audit.md

References:
  - Source cells: docs/audit/results/2026-04-15-prior-day-features-orb-mega-exploration.md
  - Prior audits:
    - docs/audit/results/2026-04-15-t0-t8-audit-o5-patterns.md
    - docs/audit/results/2026-04-15-t0-t8-audit-hot-warm-batch.md
  - Mechanism priors: docs/institutional/mechanism_priors.md H6/H7 (level rejection)
  - Protocol: .claude/rules/quant-audit-protocol.md Steps 3-5
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from research.t0_t8_audit_prior_day_patterns import (  # type: ignore  # noqa: E402
    Pattern,
    audit_pattern,
)
from research.t0_t8_audit_hot_warm_batch import build_feature_sql  # type: ignore  # noqa: E402

MEGA_MD = Path("docs/audit/results/2026-04-15-prior-day-features-orb-mega-exploration.md")
OUTPUT_MD = Path("docs/audit/results/2026-04-15-t0-t8-adversarial-fade-audit.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)


def flip_direction(d: str) -> str:
    return "short" if d == "long" else "long"


def flip_sign(s: str) -> str:
    return "positive" if s == "negative" else "negative"


def parse_all_cells() -> list[tuple[str, str, int, float, str, str, float, float]]:
    """Extract all HOT + WARM cells from mega MD. Returns tuples:
    (instrument, session, aperture, rr, direction, signal, t_cl, delta_is)."""
    text = MEGA_MD.read_text(encoding="utf-8")
    bullet_re = re.compile(
        r"^- ([A-Z]{3}) ([A-Z_0-9]+) O(\d+) RR([\d.]+) (long|short) (F\d_[A-Z_0-9]+?):"
        r"\s+t_cl=([+-][\d.]+)"
        r"\s+\u0394_IS=([+-][\d.]+)",
        re.MULTILINE,
    )
    out = []
    for m in bullet_re.finditer(text):
        instr, session, apt, rr, direction, signal, t_cl, delta_is = m.groups()
        out.append((instr, session, int(apt), float(rr), direction, signal, float(t_cl), float(delta_is)))
    return out


def build_fade_patterns() -> list[tuple[Pattern, dict]]:
    """For each SKIP-candidate cell (negative delta_is), build the opposite-direction
    Pattern as a FADE candidate.

    Positive-delta cells are already TAKE candidates in the original direction —
    we skip flipping those (flipping a winning long→short would be negative-EV).
    """
    cells = parse_all_cells()
    patterns = []
    for instr, session, apt, rr, direction, signal, t_cl, delta_is in cells:
        if delta_is > 0:
            # Already a TAKE candidate in original direction — no fade to test
            continue
        # Flip direction; expected sign now POSITIVE (we expect to gain where original lost)
        fade_dir = flip_direction(direction)
        try:
            feature_sql, theta = build_feature_sql(signal, session)
        except ValueError as e:
            print(f"[skip] bad signal {signal}: {e}")
            continue

        name = f"FADE_{instr}_{session}_O{apt}_RR{rr}_{fade_dir}_{signal}"
        desc = (
            f"FADE of {direction.upper()} SKIP signal: take {fade_dir.upper()} "
            f"when {signal} fires (original lost {delta_is:+.3f} @ t_cl={t_cl:+.2f})"
        )
        p = Pattern(
            name=name,
            description=desc,
            instrument=instr,
            session=session,
            aperture=apt,
            rr=rr,
            direction=fade_dir,
            feature_sql=feature_sql,
            theta=theta,
            expected_sign="positive",
        )
        patterns.append(
            (
                p,
                {
                    "orig_direction": direction,
                    "orig_delta_is": delta_is,
                    "orig_t_cl": t_cl,
                },
            )
        )
    return patterns


def classify_fade_verdict(tests: dict) -> str:
    """Fade-specific verdict. We want:
    - T2 PASS or INFO (need N≥30 on-signal)
    - T6 PASS (beats null)
    - T7 PASS (per-year consistency)
    - T8 PASS (cross-instrument) — softer requirement
    - ExpR_on > 0 (the fade actually pays positive)

    Stricter than regular audit because we're looking for NEW deployable direction.
    """
    fail_count = sum(1 for tr in tests.values() if tr.pass_status == "FAIL")

    # Need T2 at least INFO (N≥30), T6 PASS, T7 PASS or INFO
    t2 = tests.get("T2")
    t6 = tests.get("T6")
    t7 = tests.get("T7")

    t2_ok = t2 and t2.pass_status in ("PASS", "INFO") and "N=" in str(t2.value)
    t6_pass = t6 and t6.pass_status == "PASS"
    t7_ok = t7 and t7.pass_status in ("PASS", "INFO")

    # Extract ExpR
    expr_on = None
    if t2 and "ExpR=" in str(t2.value):
        m = re.search(r"ExpR=([+-][\d.]+)", str(t2.value))
        if m:
            expr_on = float(m.group(1))

    if expr_on is None or expr_on <= 0:
        return "FADE_FAILS — no positive ExpR"
    if not (t2_ok and t6_pass and t7_ok):
        return "FADE_WEAK — missing core tests"
    if fail_count == 0:
        return "FADE_VALIDATED"
    if fail_count == 1:
        return "FADE_CONDITIONAL"
    return "FADE_CONDITIONAL" if fail_count <= 2 else "FADE_REJECTED"


def extract_expr(tests: dict) -> str:
    t2 = tests.get("T2")
    if t2 and "ExpR=" in str(t2.value):
        m = re.search(r"ExpR=([+-][\d.]+)", str(t2.value))
        if m:
            return m.group(1)
    return "N/A"


def emit(all_results: list[dict]) -> None:
    lines = [
        "# T0-T8 Adversarial Fade Audit",
        "",
        "**Date:** 2026-04-15",
        "**Hypothesis:** SKIP signals are latent FADE signals. Take the opposite direction.",
        "**Source cells:** HOT + WARM from mega-exploration with negative delta_is",
        "**Protocol:** `.claude/rules/quant-audit-protocol.md` Steps 3-5",
        "",
    ]

    # Summary
    verdicts = []
    for r in all_results:
        v = classify_fade_verdict(r["tests"])
        verdicts.append(v)

    counts = {v: verdicts.count(v) for v in set(verdicts)}
    lines += [
        "## Verdict Summary",
        "",
        "| Verdict | Count |",
        "|---------|-------|",
    ]
    for v in sorted(counts, key=lambda x: -counts[x]):
        lines.append(f"| {v} | {counts[v]} |")
    lines.append("")

    lines += [
        "## Summary Table (sorted by fade ExpR)",
        "",
        "| Fade Cell | Orig Dir | ExpR (fade) | Verdict | N_on | Key FAILs |",
        "|-----------|----------|-------------|---------|------|-----------|",
    ]

    # Sort by ExpR desc
    annotated = []
    for r, v in zip(all_results, verdicts):
        expr_str = extract_expr(r["tests"])
        try:
            expr_num = float(expr_str)
        except Exception:
            expr_num = -999.0
        annotated.append((expr_num, r, v, expr_str))
    annotated.sort(key=lambda x: -x[0])

    for expr_num, r, v, expr_str in annotated:
        p = r["pattern"]
        fails = [tname for tname, tr in r["tests"].items() if tr.pass_status == "FAIL"]
        orig_dir = "long" if p.direction == "short" else "short"
        lines.append(f"| {p.name} | {orig_dir} | {expr_str} | **{v}** | {r['n_on']} | {','.join(fails) or '—'} |")

    lines += ["", "---", "", "## Per-Cell Detail (fade validated / conditional only)", ""]

    for expr_num, r, v, expr_str in annotated:
        if not v.startswith("FADE_VALIDATED") and not v.startswith("FADE_CONDITIONAL"):
            continue
        p = r["pattern"]
        lines += [
            f"### {p.name}",
            f"**Fade scope:** {p.instrument} | {p.session} | O{p.aperture} | RR{p.rr} | {p.direction}",
            f"**Signal:** {p.feature_sql[:120]}...",
            f"**N_total:** {r['n_total']} | **N_on_signal:** {r['n_on']}",
            f"**Fade ExpR:** {expr_str}",
            "",
            "| Test | Value | Status | Detail |",
            "|------|-------|--------|--------|",
        ]
        for tname, tr in r["tests"].items():
            lines.append(f"| {tname} {tr.name} | {tr.value} | **{tr.pass_status}** | {tr.detail} |")
        lines += ["", f"### Verdict: **{v}**", "", "---", ""]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {OUTPUT_MD}")


def main():
    fade_patterns = build_fade_patterns()
    print(f"Testing FADE on {len(fade_patterns)} cells (SKIP→FADE direction flip)")

    all_results = []
    for i, (p, meta) in enumerate(fade_patterns, 1):
        print(f"\n[{i}/{len(fade_patterns)}] {p.name}")
        print(f"  orig: {meta['orig_direction']} lost {meta['orig_delta_is']:+.3f}")
        try:
            r = audit_pattern(p)
            r["meta"] = meta
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    emit(all_results)

    # Aggregated summary
    verdicts = [classify_fade_verdict(r["tests"]) for r in all_results]
    print("\n=== FADE VERDICT SUMMARY ===")
    for v in sorted(set(verdicts), key=lambda x: -verdicts.count(x)):
        print(f"  {v}: {verdicts.count(v)}")


if __name__ == "__main__":
    main()
