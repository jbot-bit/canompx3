"""Deployable-shelf gap scanner.

Read-only diagnostic. Joins five canonical surfaces and prints a ranked
unlock queue per account profile:

  1. ``trading_app.prop_profiles``  — profile firm/allowed lists/exclusivity prose
  2. ``trading_app.prop_profiles.load_allocation_lanes`` — current allocator state
  3. ``pipeline.db_contracts.deployable_validated_relation`` — validated shelf
  4. ``trading_app.chordia.load_chordia_audit_log`` + ``chordia_verdict_label``
  5. ``trading_app.live.broker_factory.VALID_BROKERS`` — routability map

No re-encoding: every doctrine value (Chordia thresholds, fitness labels,
audit log fields, firm bans) comes from the canonical helper. No invented
multipliers. ``annual_r`` is taken from the DB or surfaced as UNKNOWN; we
never fabricate a value. Chordia ranking is the canonical 4-class label
in fixed order (PASS_CHORDIA > PASS_PROTOCOL_A > FAIL_CHORDIA > FAIL_BOTH
> MISSING), not a linear interpolation between thresholds.

Falsification cases verified at build time (see verification script):

  * ``tradeify_50k_type_b --instrument MGC``: every MGC LONDON_METALS row
    must compute to a FAIL_* verdict under the canonical chordia function
    (the K1 family included).
  * ``topstep_50k_mnq_auto --instrument MGC``: profile sets
    ``allowed_instruments=frozenset({"MNQ"})``, so MGC rows must NOT
    appear in the ranked queue (they appear only under
    ``--include-blocked`` annotated as instrument_blocked).

Usage::

    python scripts/tools/deployable_shelf_gap.py --profile <id>
    python scripts/tools/deployable_shelf_gap.py --all-profiles --format json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import duckdb

# Make repo root importable when invoked directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS  # noqa: E402
from pipeline.db_contracts import deployable_validated_relation  # noqa: E402
from pipeline.dst import SESSION_CATALOG  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from trading_app.chordia import (  # noqa: E402
    ChordiaAuditLog,
    chordia_verdict_label,
    compute_chordia_t,
    load_chordia_audit_log,
)
from trading_app.live.broker_factory import VALID_BROKERS  # noqa: E402
from trading_app.prop_profiles import (  # noqa: E402
    ACCOUNT_PROFILES,
    PROP_FIRM_SPECS,
    AccountProfile,
    effective_daily_lanes,
    get_profile,
    resolve_allocation_json,
)

# Fixed firm→broker map. Mirrors the dispatch logic in
# trading_app/live/broker_factory.create_broker_components but reads only
# the mapping — no auth calls. If broker_factory adds a firm, extend this
# map (drift detected by routability=False on the new firm).
_FIRM_BROKER_MAP: dict[str, str] = {
    "topstep": "projectx",
    "tradeify": "tradovate",
    "mffu": "tradovate",
    "bulenox": "rithmic",
    "self_funded": "rithmic",
}

# Canonical Chordia verdict ordering (best → worst). Used as primary sort
# key. PASS_CHORDIA (strict, t>=3.79) ranks above PASS_PROTOCOL_A
# (theory-supported t>=3.00) which ranks above failures.
_VERDICT_ORDER: dict[str, int] = {
    "PASS_CHORDIA": 0,
    "PASS_PROTOCOL_A": 1,
    "FAIL_CHORDIA": 2,
    "FAIL_BOTH": 3,
    "MISSING": 4,
    "UNKNOWN": 5,
}

# Power-floor minimum sample size for "PASS but UNVERIFIED" warning. From
# pre_registered_criteria.md Criterion 3 (Bailey MinBTL) — the minimum
# trade count below which a t-stat pass is not yet reliable evidence.
_POWER_FLOOR_N: int = 50

# Exclusivity prose patterns. Read from AccountProfile.notes; tradeify
# notes use "no cross-firm sharing" verbiage (see prop_profiles.py:286).
# Patterns are anchored to whole-word matches to avoid false positives.
_EXCLUSIVITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bexclusive\s+to\s+(\w+)\b", re.IGNORECASE),
    re.compile(r"\bno\s+cross-firm\s+sharing\b", re.IGNORECASE),
    re.compile(r"\bbot\s+must\s+be\s+exclusive\b", re.IGNORECASE),
)


# =========================================================================
# Frozen dataclasses
# =========================================================================


@dataclass(frozen=True)
class ProfileFilter:
    """Resolved profile predicates for the candidate join."""

    profile_id: str
    firm: str
    active: bool
    allowed_instruments: frozenset[str]
    allowed_sessions: frozenset[str]
    firm_banned: frozenset[str]
    exclusivity_clause: bool
    notes_excerpt: str


@dataclass(frozen=True)
class AllocationState:
    """Lanes already-committed on this profile, by status."""

    deployed: frozenset[str]
    removed: frozenset[str]  # status not in {DEPLOY, PROVISIONAL}


@dataclass(frozen=True)
class CandidateRow:
    """One row from deployable_validated_setups."""

    strategy_id: str
    instrument: str
    orb_label: str
    orb_minutes: int | None
    entry_model: str | None
    rr_target: float | None
    confirm_bars: int | None
    filter_type: str | None
    sharpe_ratio: float | None
    sample_size: int | None
    win_rate: float | None
    expectancy_r: float | None
    trades_per_year: float | None


@dataclass(frozen=True)
class RankedRow:
    candidate: CandidateRow
    chordia_verdict: str
    has_theory: bool
    t_stat: float | None
    audit_age_days: int | None
    audit_stale: bool
    deployed_here: bool
    deployed_other_profiles: tuple[str, ...]
    exclusivity_blocked: bool
    routable_if_active: bool
    instrument_blocked: bool
    session_blocked: bool
    removed_intentionally: bool
    power_warning: bool
    flags: tuple[str, ...]


@dataclass(frozen=True)
class ScanReport:
    profile: ProfileFilter
    today: date
    db_path: str
    db_signature: str
    audit_log_sha256: str
    deployable_view_row_count: int
    ranked: tuple[RankedRow, ...]
    scope_notes: tuple[str, ...] = field(default_factory=tuple)


# =========================================================================
# Pure functions
# =========================================================================


def _resolve_allowed_instruments(profile: AccountProfile, firm_banned: frozenset[str]) -> frozenset[str]:
    """Expand profile.allowed_instruments using canonical ACTIVE set.

    ``allowed_instruments=None`` semantically means "all active ORB
    instruments, less firm bans" — per prop_profiles.AccountProfile
    docstring at line 94. Firm bans always apply.
    """
    if profile.allowed_instruments is None:
        base = frozenset(ACTIVE_ORB_INSTRUMENTS)
    else:
        base = frozenset(profile.allowed_instruments)
    return base - firm_banned


def _resolve_allowed_sessions(profile: AccountProfile) -> frozenset[str]:
    if profile.allowed_sessions is None:
        return frozenset(SESSION_CATALOG.keys())
    return frozenset(profile.allowed_sessions)


def _exclusivity_clause(notes: str) -> bool:
    for pat in _EXCLUSIVITY_PATTERNS:
        if pat.search(notes or ""):
            return True
    return False


def load_profile_filter(profile_id: str) -> ProfileFilter:
    profile = get_profile(profile_id)
    firm_spec = PROP_FIRM_SPECS.get(profile.firm)
    firm_banned = frozenset(firm_spec.banned_instruments) if firm_spec else frozenset()
    allowed_instruments = _resolve_allowed_instruments(profile, firm_banned)
    allowed_sessions = _resolve_allowed_sessions(profile)
    notes_excerpt = (profile.notes or "")[:160].replace("\n", " ")
    return ProfileFilter(
        profile_id=profile.profile_id,
        firm=profile.firm,
        active=profile.active,
        allowed_instruments=allowed_instruments,
        allowed_sessions=allowed_sessions,
        firm_banned=firm_banned,
        exclusivity_clause=_exclusivity_clause(profile.notes or ""),
        notes_excerpt=notes_excerpt,
    )


def load_allocation_state(profile_id: str) -> AllocationState:
    """Read this profile's allocation status sets.

    Reads the raw JSON via ``resolve_allocation_json`` (Stage 1b authority
    inversion) so we can distinguish DEPLOY/PROVISIONAL from
    PAUSE/STALE/REMOVED — distinctions ``effective_daily_lanes`` collapses.
    Fail-closed-as-empty on missing/corrupt/profile-mismatch.
    """
    result = resolve_allocation_json(profile_id)
    data = result.data
    if data is None:
        return AllocationState(deployed=frozenset(), removed=frozenset())

    lanes = data.get("lanes")
    if not isinstance(lanes, list):
        return AllocationState(deployed=frozenset(), removed=frozenset())

    deployed: set[str] = set()
    removed: set[str] = set()
    for entry in lanes:
        sid = entry.get("strategy_id")
        if not sid:
            continue
        status = entry.get("status", "")
        if status in ("DEPLOY", "PROVISIONAL"):
            deployed.add(sid)
        else:
            removed.add(sid)
    return AllocationState(deployed=frozenset(deployed), removed=frozenset(removed))


def load_candidate_pool(profile: ProfileFilter, db_path: Path) -> tuple[list[CandidateRow], int]:
    """Read deployable_validated_setups filtered to profile-allowed instruments.

    Returns (rows, total_view_row_count). The total is reported in the
    scan header so operators can see how much of the shelf the filter
    narrowed.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rel = deployable_validated_relation(con, alias="v")
        total = con.execute(f"SELECT COUNT(*) FROM {rel}").fetchone()[0]

        sql = f"""
            SELECT
                v.strategy_id,
                v.instrument,
                v.orb_label,
                v.orb_minutes,
                v.entry_model,
                v.rr_target,
                v.confirm_bars,
                v.filter_type,
                v.sharpe_ratio,
                v.sample_size,
                v.win_rate,
                v.expectancy_r,
                v.trades_per_year
            FROM {rel}
            WHERE v.instrument = ANY($insts)
        """
        rows = con.execute(sql, {"insts": list(profile.allowed_instruments)}).fetchall()
    finally:
        con.close()

    candidates: list[CandidateRow] = []
    for r in rows:
        candidates.append(
            CandidateRow(
                strategy_id=r[0],
                instrument=r[1],
                orb_label=r[2],
                orb_minutes=r[3],
                entry_model=r[4],
                rr_target=r[5],
                confirm_bars=r[6],
                filter_type=r[7],
                sharpe_ratio=r[8],
                sample_size=r[9],
                win_rate=r[10],
                expectancy_r=r[11],
                trades_per_year=r[12],
            )
        )
    return candidates, int(total)


def resolve_chordia_state(
    candidate: CandidateRow,
    audit_log: ChordiaAuditLog,
    today: date,
) -> tuple[str, bool, float | None, int | None, bool]:
    """Return (verdict, has_theory, t_stat, audit_age_days, audit_stale).

    Prefers a stored audit verdict; falls back to the canonical
    ``chordia_verdict_label`` computed from sharpe + N + theory grant.
    Never re-implements threshold math.
    """
    has_theory = audit_log.has_theory(candidate.strategy_id)
    stored = audit_log.verdict(candidate.strategy_id)
    if stored is not None:
        verdict = stored
    else:
        verdict = chordia_verdict_label(candidate.sharpe_ratio, candidate.sample_size, has_theory)

    if candidate.sharpe_ratio is not None and candidate.sample_size is not None and candidate.sample_size >= 2:
        try:
            t_stat: float | None = compute_chordia_t(candidate.sharpe_ratio, candidate.sample_size)
        except ValueError:
            t_stat = None
    else:
        t_stat = None

    age = audit_log.audit_age_days(candidate.strategy_id, today)
    audit_stale = age is not None and age > audit_log.audit_freshness_days
    return verdict, has_theory, t_stat, age, audit_stale


def detect_exclusivity_blocks(
    current: ProfileFilter,
    all_profiles: dict[str, AccountProfile],
) -> dict[str, list[str]]:
    """Return strategy_id → list of other profile_ids that hold it under an exclusivity clause."""
    blocks: dict[str, list[str]] = {}
    for other_id, other in all_profiles.items():
        if other_id == current.profile_id:
            continue
        if other.firm == current.firm:
            # Same-firm sharing is allowed even under exclusivity prose.
            continue
        other_excl = _exclusivity_clause(other.notes or "")
        if not (other_excl or current.exclusivity_clause):
            continue
        for lane in effective_daily_lanes(other):
            blocks.setdefault(lane.strategy_id, []).append(other_id)
    return blocks


def deployed_elsewhere(
    current_profile_id: str,
    all_profiles: dict[str, AccountProfile],
) -> dict[str, list[str]]:
    """Return strategy_id → list of OTHER profile_ids that already deploy it (any firm)."""
    out: dict[str, list[str]] = {}
    for pid, p in all_profiles.items():
        if pid == current_profile_id:
            continue
        for lane in effective_daily_lanes(p):
            out.setdefault(lane.strategy_id, []).append(pid)
    return out


def resolve_routability(firm: str) -> bool:
    broker = _FIRM_BROKER_MAP.get(firm)
    return broker is not None and broker in VALID_BROKERS


def _db_signature(db_path: Path) -> str:
    """Return a short identity for the DB file (mtime + size).

    Hashing the full DB file is too slow for a diagnostic; mtime+size is
    sufficient to detect "same DB across two runs vs different".
    """
    try:
        st = db_path.stat()
    except FileNotFoundError:
        return "missing"
    return f"size={st.st_size},mtime={int(st.st_mtime)}"


def _audit_log_sha256() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    p = repo_root / "docs" / "runtime" / "chordia_audit_log.yaml"
    if not p.exists():
        return "missing"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def rank_candidates(
    profile: ProfileFilter,
    candidates: list[CandidateRow],
    audit_log: ChordiaAuditLog,
    allocation: AllocationState,
    exclusivity_blocks: dict[str, list[str]],
    deployed_others: dict[str, list[str]],
    routable: bool,
    today: date,
) -> list[RankedRow]:
    ranked: list[RankedRow] = []
    for cand in candidates:
        verdict, has_theory, t_stat, age, audit_stale = resolve_chordia_state(cand, audit_log, today)
        deployed_here = cand.strategy_id in allocation.deployed
        removed = cand.strategy_id in allocation.removed
        instrument_blocked = cand.instrument not in profile.allowed_instruments
        session_blocked = cand.orb_label not in profile.allowed_sessions
        deployed_other = tuple(sorted(deployed_others.get(cand.strategy_id, [])))
        excl_blocked = bool(exclusivity_blocks.get(cand.strategy_id))

        passes_chordia = verdict in ("PASS_CHORDIA", "PASS_PROTOCOL_A")
        n = cand.sample_size or 0
        power_warning = passes_chordia and n < _POWER_FLOOR_N

        flags: list[str] = []
        if deployed_here:
            flags.append("deployed_here")
        if removed:
            flags.append("removed_in_alloc")
        if instrument_blocked:
            flags.append("instrument_blocked")
        if session_blocked:
            flags.append("session_blocked")
        if excl_blocked:
            flags.append(
                f"exclusive_to={','.join(sorted({pid for pid in exclusivity_blocks.get(cand.strategy_id, [])}))}"
            )
        if deployed_other and not excl_blocked:
            flags.append(f"deployed_other={','.join(deployed_other)}")
        if audit_stale:
            flags.append("audit_stale")
        if power_warning:
            flags.append(f"power_below_N{_POWER_FLOOR_N}")
        if not routable:
            flags.append("not_routable")

        ranked.append(
            RankedRow(
                candidate=cand,
                chordia_verdict=verdict,
                has_theory=has_theory,
                t_stat=t_stat,
                audit_age_days=age,
                audit_stale=audit_stale,
                deployed_here=deployed_here,
                deployed_other_profiles=deployed_other,
                exclusivity_blocked=excl_blocked,
                routable_if_active=routable,
                instrument_blocked=instrument_blocked,
                session_blocked=session_blocked,
                removed_intentionally=removed,
                power_warning=power_warning,
                flags=tuple(flags),
            )
        )

    def sort_key(r: RankedRow) -> tuple:
        # Primary: canonical chordia verdict ordering.
        # Secondary: annual_r proxy (sample-size × ExpR — DB-derived, no fabricated constants).
        # We do NOT invent annual_r when expectancy_r is null; those rows
        # sort last by carrying -inf as the proxy.
        n = r.candidate.sample_size or 0
        e = r.candidate.expectancy_r
        proxy = (e * n) if (e is not None and n > 0) else -math.inf
        # Tertiary: t-stat desc.
        t = r.t_stat if r.t_stat is not None else -math.inf
        return (_VERDICT_ORDER.get(r.chordia_verdict, 99), -proxy, -t)

    ranked.sort(key=sort_key)
    return ranked


# =========================================================================
# Filtering after rank (CLI flag handling)
# =========================================================================


def filter_ranked(
    ranked: list[RankedRow],
    instrument: str | None,
    include_blocked: bool,
    include_deployed: bool,
    limit: int,
) -> list[RankedRow]:
    out: list[RankedRow] = []
    for r in ranked:
        if instrument and r.candidate.instrument != instrument:
            continue
        if not include_blocked and (r.instrument_blocked or r.session_blocked or r.exclusivity_blocked):
            continue
        if not include_deployed and r.deployed_here:
            continue
        out.append(r)
        if len(out) >= limit:
            break
    return out


# =========================================================================
# Rendering
# =========================================================================


def render_markdown(report: ScanReport, rows: list[RankedRow]) -> str:
    p = report.profile
    lines: list[str] = []
    lines.append(f"# Deployable-Shelf Gap — {p.profile_id}")
    lines.append("")
    lines.append(f"- firm: `{p.firm}` (routable: `{resolve_routability(p.firm)}`)")
    lines.append(f"- active: `{p.active}`")
    lines.append(f"- allowed_instruments: `{sorted(p.allowed_instruments)}`")
    lines.append(f"- allowed_sessions: `{sorted(p.allowed_sessions)}`")
    lines.append(f"- firm_banned: `{sorted(p.firm_banned)}`")
    lines.append(f"- exclusivity_clause: `{p.exclusivity_clause}`")
    lines.append(f"- deployable_view_total_rows: `{report.deployable_view_row_count}`")
    lines.append(f"- db_signature: `{report.db_signature}`")
    lines.append(f"- audit_log_sha256: `{report.audit_log_sha256}`")
    lines.append(f"- today: `{report.today.isoformat()}`")
    lines.append("")
    for note in report.scope_notes:
        lines.append(f"> {note}")
    if report.scope_notes:
        lines.append("")

    if not rows:
        lines.append("_(no candidates after filters)_")
        return "\n".join(lines)

    header = (
        "| # | verdict | strategy_id | inst | session | orb | entry | rr | filter | "
        "sharpe | N | t | ExpR | audit_age | flags |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for idx, r in enumerate(rows, start=1):
        c = r.candidate
        sharpe = f"{c.sharpe_ratio:.3f}" if c.sharpe_ratio is not None else "—"
        n = str(c.sample_size) if c.sample_size is not None else "—"
        t = f"{r.t_stat:.3f}" if r.t_stat is not None else "—"
        e = f"{c.expectancy_r:.3f}" if c.expectancy_r is not None else "UNKNOWN"
        age = f"{r.audit_age_days}d" if r.audit_age_days is not None else "—"
        flags = ",".join(r.flags) or "—"
        lines.append(
            f"| {idx} | {r.chordia_verdict} | `{c.strategy_id}` | {c.instrument} | {c.orb_label} | "
            f"{c.orb_minutes or '—'} | {c.entry_model or '—'} | {c.rr_target or '—'} | "
            f"{c.filter_type or '—'} | {sharpe} | {n} | {t} | {e} | {age} | {flags} |"
        )
    return "\n".join(lines)


def render_json(report: ScanReport, rows: list[RankedRow]) -> str:
    out: dict = {
        "profile_id": report.profile.profile_id,
        "firm": report.profile.firm,
        "active": report.profile.active,
        "allowed_instruments": sorted(report.profile.allowed_instruments),
        "allowed_sessions": sorted(report.profile.allowed_sessions),
        "firm_banned": sorted(report.profile.firm_banned),
        "exclusivity_clause": report.profile.exclusivity_clause,
        "routable_if_active": resolve_routability(report.profile.firm),
        "deployable_view_total_rows": report.deployable_view_row_count,
        "db_signature": report.db_signature,
        "audit_log_sha256": report.audit_log_sha256,
        "today": report.today.isoformat(),
        "scope_notes": list(report.scope_notes),
        "rows": [
            {
                "strategy_id": r.candidate.strategy_id,
                "instrument": r.candidate.instrument,
                "orb_label": r.candidate.orb_label,
                "orb_minutes": r.candidate.orb_minutes,
                "entry_model": r.candidate.entry_model,
                "rr_target": r.candidate.rr_target,
                "confirm_bars": r.candidate.confirm_bars,
                "filter_type": r.candidate.filter_type,
                "sharpe_ratio": r.candidate.sharpe_ratio,
                "sample_size": r.candidate.sample_size,
                "win_rate": r.candidate.win_rate,
                "expectancy_r": r.candidate.expectancy_r,
                "trades_per_year": r.candidate.trades_per_year,
                "chordia_verdict": r.chordia_verdict,
                "has_theory": r.has_theory,
                "t_stat": r.t_stat,
                "audit_age_days": r.audit_age_days,
                "audit_stale": r.audit_stale,
                "deployed_here": r.deployed_here,
                "deployed_other_profiles": list(r.deployed_other_profiles),
                "exclusivity_blocked": r.exclusivity_blocked,
                "instrument_blocked": r.instrument_blocked,
                "session_blocked": r.session_blocked,
                "removed_intentionally": r.removed_intentionally,
                "power_warning": r.power_warning,
                "flags": list(r.flags),
            }
            for r in rows
        ],
    }
    return json.dumps(out, indent=2, default=str)


# =========================================================================
# Top-level scan
# =========================================================================


def run_scan(
    profile_id: str,
    today: date,
    db_path: Path,
) -> ScanReport:
    profile = load_profile_filter(profile_id)
    allocation = load_allocation_state(profile_id)
    candidates, total_rows = load_candidate_pool(profile, db_path)
    audit_log = load_chordia_audit_log()
    exclusivity = detect_exclusivity_blocks(profile, ACCOUNT_PROFILES)
    deployed_others = deployed_elsewhere(profile_id, ACCOUNT_PROFILES)
    routable = resolve_routability(profile.firm)
    ranked = rank_candidates(
        profile=profile,
        candidates=candidates,
        audit_log=audit_log,
        allocation=allocation,
        exclusivity_blocks=exclusivity,
        deployed_others=deployed_others,
        routable=routable,
        today=today,
    )
    scope_notes: list[str] = [
        "Candidate pool restricted to deployable_validated_setups "
        "(promoted+active+deployment_scope='deployable'). Filtered to "
        "profile-allowed instruments only.",
        "Chordia ranking uses canonical chordia_verdict_label(). Thresholds: "
        "t>=3.79 (PASS_CHORDIA), 3.00<=t<3.79 with theory (PASS_PROTOCOL_A), "
        "3.00<=t<3.79 without theory (FAIL_CHORDIA), t<3.00 (FAIL_BOTH).",
        "Secondary sort = sample_size × expectancy_r (DB-derived). Rows with "
        "null expectancy_r sort last; no annual_r is fabricated.",
        f"Power-floor warning fires when canonical PASS verdict has N<{_POWER_FLOOR_N}.",
    ]
    return ScanReport(
        profile=profile,
        today=today,
        db_path=str(db_path),
        db_signature=_db_signature(db_path),
        audit_log_sha256=_audit_log_sha256(),
        deployable_view_row_count=total_rows,
        ranked=tuple(ranked),
        scope_notes=tuple(scope_notes),
    )


# =========================================================================
# CLI
# =========================================================================


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--profile", help="Single profile_id to scan")
    group.add_argument("--all-profiles", action="store_true", help="Scan every profile in ACCOUNT_PROFILES")
    parser.add_argument("--instrument", choices=sorted(ACTIVE_ORB_INSTRUMENTS))
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--include-blocked",
        action="store_true",
        help="Include rows blocked by instrument/session/exclusivity (for diagnosis)",
    )
    parser.add_argument(
        "--include-deployed",
        action="store_true",
        help="Include rows already deployed on this profile",
    )
    parser.add_argument(
        "--today",
        help="ISO date used for Chordia audit-age (default = today)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.today:
        try:
            today = date.fromisoformat(args.today)
        except ValueError:
            print(f"ERROR: --today must be ISO YYYY-MM-DD, got {args.today!r}", file=sys.stderr)
            return 2
    else:
        today = datetime.now(tz=None).date()

    db_path = Path(GOLD_DB_PATH)
    if not db_path.exists():
        print(f"ERROR: canonical DB missing at {db_path}", file=sys.stderr)
        return 3

    if args.all_profiles:
        profile_ids = list(ACCOUNT_PROFILES.keys())
    else:
        if args.profile not in ACCOUNT_PROFILES:
            print(
                f"ERROR: unknown profile {args.profile!r}. Known: {sorted(ACCOUNT_PROFILES.keys())}",
                file=sys.stderr,
            )
            return 2
        profile_ids = [args.profile]

    chunks: list[str] = []
    for pid in profile_ids:
        report = run_scan(pid, today, db_path)
        rows = filter_ranked(
            list(report.ranked),
            instrument=args.instrument,
            include_blocked=args.include_blocked,
            include_deployed=args.include_deployed,
            limit=args.limit,
        )
        if args.format == "json":
            chunks.append(render_json(report, rows))
        else:
            chunks.append(render_markdown(report, rows))

    if args.format == "json" and len(chunks) > 1:
        # Combine multiple JSON reports into a single array.
        combined = "[" + ",".join(chunks) + "]"
        print(combined)
    else:
        print("\n\n".join(chunks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
