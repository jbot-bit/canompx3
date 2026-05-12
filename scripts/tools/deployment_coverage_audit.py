"""Deployment coverage audit — orphan validated strategies.

Read-only audit. Joins three sources of truth:
  1. validated_setups (gold.db) -- the deployable pool (status='active')
  2. trading_app.prop_profiles.ACCOUNT_PROFILES -- broker profile whitelists
  3. trading_app.lane_allocator.compute_lane_scores -- canonical annual_r

Emits a markdown report ranking every active validated strategy by its
routability across the full profile space (active + inactive). Does not
mutate prop_profiles.py, validated_setups, or lane_allocation.json.

Per CLAUDE.md "audit-first-default-for-research-layers" + Volatile Data Rule.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import duckdb

from pipeline.dst import SESSION_CATALOG
from pipeline.paths import GOLD_DB_PATH
from trading_app.lane_allocator import LaneScore, compute_lane_scores
from trading_app.prop_profiles import ACCOUNT_PROFILES, AccountProfile

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "docs" / "audit" / "results" / "2026-05-12-deployment-coverage-orphans.md"


def _profile_admits(profile: AccountProfile, instrument: str, session: str) -> bool:
    inst_ok = profile.allowed_instruments is None or instrument in profile.allowed_instruments
    sess_ok = profile.allowed_sessions is None or session in profile.allowed_sessions
    return inst_ok and sess_ok


def _classify(score: LaneScore, profiles: Iterable[AccountProfile]) -> tuple[str, list[str], list[str]]:
    """Return (orphan_class, admitting_active_ids, admitting_inactive_ids)."""
    active_admit, inactive_admit = [], []
    for p in profiles:
        if _profile_admits(p, score.instrument, score.orb_label):
            (active_admit if p.active else inactive_admit).append(p.profile_id)

    if active_admit:
        return "ROUTABLE_ACTIVE", active_admit, inactive_admit
    if inactive_admit:
        return "ROUTABLE_DORMANT", active_admit, inactive_admit

    # No profile admits. Distinguish firm-gap (firm has the instrument but not the session)
    # from no-firm (no profile in our roster touches this instrument at all).
    inst_seen_by_firm: set[str] = set()
    for p in profiles:
        if p.allowed_instruments is None or score.instrument in p.allowed_instruments:
            inst_seen_by_firm.add(p.firm)
    return ("ORPHAN_FIRM_GAP" if inst_seen_by_firm else "ORPHAN_NO_FIRM", [], [])


def _minimal_fix(score: LaneScore, profiles: Iterable[AccountProfile]) -> str:
    """Cheapest profile edit that would unlock this strategy."""
    # 1) profile already admits the instrument, only missing the session
    inst_only = [
        p
        for p in profiles
        if (p.allowed_instruments is None or score.instrument in p.allowed_instruments)
        and (p.allowed_sessions is not None and score.orb_label not in p.allowed_sessions)
    ]
    if inst_only:
        # prefer active profiles, then topstep (current live broker)
        inst_only.sort(key=lambda p: (not p.active, p.firm != "topstep", p.profile_id))
        p = inst_only[0]
        flag = "ACTIVE" if p.active else "INACTIVE"
        return f"add session '{score.orb_label}' to {p.profile_id} [{flag}]"

    # 2) no profile has the instrument -- need a new profile at some firm
    return f"NEW PROFILE required for instrument '{score.instrument}' (no roster profile admits it)"


def _dead_whitelist(profile: AccountProfile, instr_session_population: set[tuple[str, str]]) -> list[str]:
    """Sessions in profile.allowed_sessions with no active strategies for any of profile's instruments."""
    if profile.allowed_sessions is None:
        return []
    instruments = profile.allowed_instruments or {i for i, _ in instr_session_population}
    dead = []
    for sess in sorted(profile.allowed_sessions):
        if sess not in SESSION_CATALOG:
            dead.append(f"{sess} [NOT IN SESSION_CATALOG]")
            continue
        has_any = any((inst, sess) in instr_session_population for inst in instruments)
        if not has_any:
            dead.append(sess)
    return dead


def _format_row(score: LaneScore, klass: str, fix: str) -> str:
    return (
        f"| {score.strategy_id} | {score.instrument} | {score.orb_label} | "
        f"{score.annual_r_estimate:.1f} | {score.trailing_n} | {score.trailing_expr:.3f} | "
        f"{klass} | {fix} |"
    )


def main() -> None:
    rebalance_date = date.today()
    profiles = list(ACCOUNT_PROFILES.values())

    print(f"Computing lane scores for rebalance_date={rebalance_date} ...")
    scores = compute_lane_scores(rebalance_date=rebalance_date)
    print(f"  {len(scores)} validated strategies scored.")

    # Population of (instrument, session) actually present in validated_setups
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    pop_rows = con.execute(
        "SELECT DISTINCT instrument, orb_label FROM validated_setups WHERE status='active'"
    ).fetchall()
    population: set[tuple[str, str]] = {(i, s) for i, s in pop_rows}
    con.close()

    # Classify every score
    classified: list[tuple[LaneScore, str, list[str], list[str]]] = []
    counts: dict[str, int] = defaultdict(int)
    annual_by_class: dict[str, float] = defaultdict(float)
    for s in scores:
        klass, active_admit, inactive_admit = _classify(s, profiles)
        classified.append((s, klass, active_admit, inactive_admit))
        counts[klass] += 1
        annual_by_class[klass] += s.annual_r_estimate

    # ----- Build report -----
    lines: list[str] = []
    lines.append("# Deployment Coverage Audit — Orphan Validated Strategies")
    lines.append("")
    lines.append(f"**Rebalance date:** {rebalance_date}")
    lines.append(f"**Total active validated strategies:** {len(scores)}")
    lines.append(
        f"**Profiles in roster:** {len(profiles)} "
        f"({sum(1 for p in profiles if p.active)} active, "
        f"{sum(1 for p in profiles if not p.active)} inactive)"
    )
    lines.append("")
    lines.append("Read-only audit. Does not mutate prop_profiles, validated_setups, or lane_allocation.json.")
    lines.append("annual_r computed via canonical `trading_app.lane_allocator.compute_lane_scores`.")
    lines.append("")
    lines.append("## Scope and limitations (read first)")
    lines.append("")
    lines.append(
        "- **`annual_r` is live, not stale.** Numbers recomputed every run from the canonical "
        "trailing-window formula in `lane_allocator.py:373`. Any prior cited value (e.g. memory or "
        "plan documents) may differ — trust this report."
    )
    lines.append(
        "- **`minimal_fix` = cheapest single-whitelist edit, NOT an activation recommendation.** "
        "It identifies the profile already closest to admitting the strategy. It does NOT verify "
        "broker-fit, cost-model viability, or whether the profile is inactive for capital reasons."
    )
    lines.append(
        '- **`ROUTABLE_DORMANT` does not mean "deploy this".** A profile may be inactive because '
        "of capital, broker rules, or a deliberate prior decision. This audit only proves the "
        "(instrument, session) pair is whitelisted by an inactive profile — activation is a "
        "separate decision."
    )
    lines.append(
        "- **No new statistical claims.** This audit makes zero new p-value or significance "
        "assertions. Every annual_r already passed validation upstream — we're only joining sources."
    )
    lines.append("")

    # Notable findings (computed honestly from the data, not assumed)
    lines.append("## Notable findings")
    lines.append("")
    findings: list[str] = []

    # 1. Profiles with zero deployable strategies as currently configured
    for p in profiles:
        admits_count = sum(1 for s in scores if _profile_admits(p, s.instrument, s.orb_label))
        if admits_count == 0:
            findings.append(
                f"- **`{p.profile_id}`** ({'active' if p.active else 'inactive'}, firm={p.firm}) "
                f"admits ZERO active validated strategies as currently configured. "
                f"Its (instrument, session) whitelist does not intersect validated_setups."
            )

    # 2. Active profiles missing major edge: rank by sum-annual_r they DON'T admit
    for p in [pp for pp in profiles if pp.active]:
        blocked = [s for s in scores if not _profile_admits(p, s.instrument, s.orb_label)]
        blocked_sum = sum(s.annual_r_estimate for s in blocked)
        if blocked_sum > 0:
            findings.append(
                f"- Active profile **`{p.profile_id}`** does not admit "
                f"{len(blocked)}/{len(scores)} validated strategies "
                f"(sum annual_r blocked = {blocked_sum:.1f}R)."
            )

    # 3. Active routing concentration
    active_inst_set: set[str] = set()
    active_sess_set: set[str] = set()
    for p in profiles:
        if p.active:
            if p.allowed_instruments is not None:
                active_inst_set |= p.allowed_instruments
            if p.allowed_sessions is not None:
                active_sess_set |= p.allowed_sessions
    findings.append(
        f"- Active-profile reach: instruments={sorted(active_inst_set)} "
        f"sessions={sorted(active_sess_set)}. "
        f"Strategies on instruments outside that set cannot be live-traded today."
    )

    for f in findings:
        lines.append(f)
    lines.append("")

    # Summary
    lines.append("## Summary by routing class")
    lines.append("")
    lines.append("| Class | Count | Sum annual_r |")
    lines.append("|---|---:|---:|")
    for klass in ["ROUTABLE_ACTIVE", "ROUTABLE_DORMANT", "ORPHAN_FIRM_GAP", "ORPHAN_NO_FIRM"]:
        lines.append(f"| {klass} | {counts[klass]} | {annual_by_class[klass]:.1f} |")
    lines.append("")

    # Top orphans by annual_r
    orphan_ranked = sorted(
        [(s, k) for s, k, _, _ in classified if k.startswith("ORPHAN") or k == "ROUTABLE_DORMANT"],
        key=lambda t: -t[0].annual_r_estimate,
    )

    lines.append("## Top 30 non-active-routable strategies by annual_r")
    lines.append("")
    lines.append(
        "Strategies the live broker (topstep_50k_mnq_auto) cannot trade. "
        "Rank order = how much $/yr edge sits behind a profile-config decision."
    )
    lines.append("")
    lines.append("| strategy_id | instrument | session | annual_r | N | ExpR | class | minimal_fix |")
    lines.append("|---|---|---|---:|---:|---:|---|---|")
    for s, k in orphan_ranked[:30]:
        lines.append(_format_row(s, k, _minimal_fix(s, profiles)))
    lines.append("")

    # Per-firm coverage matrix
    lines.append("## Per-firm coverage matrix")
    lines.append("")
    lines.append("Best annual_r reachable by each profile across (instrument, session) pairs in validated_setups.")
    lines.append("`-` = profile does not admit that pair.")
    lines.append("")
    pairs = sorted(population)
    header = "| profile | active | " + " | ".join(f"{i}/{s}" for i, s in pairs) + " |"
    sep = "|---|---|" + "|".join(["---:"] * len(pairs)) + "|"
    lines.append(header)
    lines.append(sep)

    # Best annual_r per (profile, instrument, session)
    best: dict[tuple[str, str, str], float] = {}
    for s, _k, _a, _i in classified:
        for p in profiles:
            if _profile_admits(p, s.instrument, s.orb_label):
                key = (p.profile_id, s.instrument, s.orb_label)
                if s.annual_r_estimate > best.get(key, float("-inf")):
                    best[key] = s.annual_r_estimate

    for p in sorted(profiles, key=lambda x: (not x.active, x.profile_id)):
        cells = []
        for inst, sess in pairs:
            v = best.get((p.profile_id, inst, sess))
            cells.append(f"{v:.1f}" if v is not None else "-")
        lines.append(f"| {p.profile_id} | {'YES' if p.active else 'no'} | " + " | ".join(cells) + " |")
    lines.append("")

    # Dead-whitelist entries
    lines.append("## Dead whitelist entries")
    lines.append("")
    lines.append(
        "Sessions allowed by a profile but with zero active validated strategies for that profile's instruments. "
        "= empty capacity in the whitelist; tightening cost-free."
    )
    lines.append("")
    any_dead = False
    for p in sorted(profiles, key=lambda x: x.profile_id):
        dead = _dead_whitelist(p, population)
        if dead:
            any_dead = True
            lines.append(f"- **{p.profile_id}** ({'active' if p.active else 'inactive'}): {', '.join(dead)}")
    if not any_dead:
        lines.append("_None — every whitelisted session has at least one active strategy._")
    lines.append("")

    # Cross-broker arbitrage
    lines.append("## Cross-broker arbitrage")
    lines.append("")
    lines.append(
        "Strategies blocked on every active profile but routable on ≥1 inactive profile. "
        "Activation candidates ranked by annual_r."
    )
    lines.append("")
    arb = sorted(
        [(s, _i) for s, k, _a, _i in classified if k == "ROUTABLE_DORMANT"],
        key=lambda t: -t[0].annual_r_estimate,
    )
    if arb:
        lines.append("| strategy_id | instrument | session | annual_r | N | inactive profiles that admit |")
        lines.append("|---|---|---|---:|---:|---|")
        for s, inactive_ids in arb[:25]:
            lines.append(
                f"| {s.strategy_id} | {s.instrument} | {s.orb_label} | "
                f"{s.annual_r_estimate:.1f} | {s.trailing_n} | {', '.join(inactive_ids)} |"
            )
    else:
        lines.append("_No dormant-only strategies — every validated edge is either active-routable or fully orphaned._")
    lines.append("")

    # Minimal-delta fix queue
    lines.append("## Minimal-delta fix queue")
    lines.append("")
    lines.append(
        "Ranked by sum-of-annual_r unlocked per single profile edit. "
        "**Each edit is gated on broker-fit + cost-model verification** — this list is awareness, not approval."
    )
    lines.append("")
    fix_unlock: dict[str, list[LaneScore]] = defaultdict(list)
    for s, k, _a, _i in classified:
        if k in ("ROUTABLE_ACTIVE",):
            continue
        fix = _minimal_fix(s, profiles)
        fix_unlock[fix].append(s)
    fix_ranked = sorted(
        fix_unlock.items(),
        key=lambda kv: -sum(s.annual_r_estimate for s in kv[1]),
    )
    lines.append("| fix | strategies unlocked | sum annual_r | top strategy |")
    lines.append("|---|---:|---:|---|")
    for fix, ss in fix_ranked[:20]:
        total = sum(s.annual_r_estimate for s in ss)
        top = max(ss, key=lambda s: s.annual_r_estimate)
        lines.append(f"| {fix} | {len(ss)} | {total:.1f} | {top.strategy_id} ({top.annual_r_estimate:.1f}) |")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(
        "Generated by `scripts/tools/deployment_coverage_audit.py`. Re-run to refresh; output is deterministic."
    )
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {REPORT_PATH.relative_to(REPO_ROOT)}")
    print(
        f"  ROUTABLE_ACTIVE={counts['ROUTABLE_ACTIVE']}  "
        f"ROUTABLE_DORMANT={counts['ROUTABLE_DORMANT']}  "
        f"ORPHAN_FIRM_GAP={counts['ORPHAN_FIRM_GAP']}  "
        f"ORPHAN_NO_FIRM={counts['ORPHAN_NO_FIRM']}"
    )


if __name__ == "__main__":
    main()
