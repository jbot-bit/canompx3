"""Triage validated_setups backlog — rank-and-draft fast-lane preregs for untouched lanes.

Enumerates active rows in ``validated_setups``, filters to lanes with no
existing fast-lane result MD under ``docs/audit/results/``, scores each
candidate using transparent components grounded in canonical helpers, and
emits ranked FAST_LANE v5.1 prereg drafts under
``docs/audit/hypotheses/drafts/<slug>.draft.yaml`` (LHP quarantine zone).

Pure inventory expansion of the PROMOTE queue's upstream side. The script
NEVER:

- Writes to ``experimental_strategies``, ``validated_setups``, or any
  table in ``gold.db``.
- Touches ``current profile allocation``,
  ``docs/runtime/chordia_audit_log.yaml``, or files under
  ``trading_app/live/*``.
- Promotes drafts out of ``drafts/`` into the active ``hypotheses/``
  directory. That is the operator's call.

Scoring components (sum-to-1 weights — see ``SCORE_WEIGHTS``)
-------------------------------------------------------------
- ``pooled_t_headroom``: ``max(0, (sharpe_proxy_t - PROMOTE_THRESHOLD)) / sharpe_proxy_t``.
  Sharpe-proxy t = ``sharpe_ratio * sqrt(sample_size)`` (one-sample one-tail).
  Above ``PROMOTE_THRESHOLD=2.5`` (v5.1 template) is the binding heuristic
  per ``chordia_et_al_2018_two_million_strategies.md:78`` implied-t band.
- ``n_adequacy``: ``min(1.0, sample_size / N_ADEQUACY_TARGET)``. Linear-
  headroom anchor; ``N_ADEQUACY_TARGET=200`` for parity with the cherry-
  pick ranker (which scores ranked-already-PROMOTEd entries on the same
  scale).
- ``oos_power_readiness``: canonical
  ``research.oos_power.one_sample_power(d, n_oos, alpha=0.05)`` evaluated
  against the OOS N parsed by querying ``orb_outcomes`` for the lane's
  post-holdout window. Returns 0 when N_OOS < ``OOS_N_FLOOR=30`` (RULE 3.3).
- ``era_stability``: 1.0 when ``all_years_positive == true`` AND
  ``years_tested >= ERA_STABILITY_MIN_YEARS=5``, 0.0 otherwise. Coarse
  proxy until cherry-pick journal accumulates richer signal (n>=3 entries
  per ``feedback_n3_same_class_doctrine_threshold.md``).
- ``non_artifact``: 1.0 unless ``filter_type`` is in the known-artifact
  set (currently empty — placeholder for future flagging).

Canonical delegation
--------------------
- ``trading_app.eligibility.builder.parse_strategy_id`` for strategy_id
  decomposition (per ``feedback_aperture_overlay_canonical_parser.md``).
- ``research.oos_power.one_sample_power`` + ``power_verdict`` for OOS
  power calculation (RULE 3.3).
- ``pipeline.paths.GOLD_DB_PATH`` for DB connection.
- ``pipeline.dst.SESSION_CATALOG`` membership for session validation.

Doctrine grounding
------------------
- Plan: ``C:/Users/joshd/.claude/plans/or-linknin-them-togehr-delegated-gizmo.md``
- Stage: ``docs/runtime/stages/2026-05-19-triage-validated-setups.md``
- Methodology: ``.claude/rules/backtesting-methodology.md`` § RULE 3.3
  (OOS power floor)
- Template: ``docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml``
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "docs" / "audit" / "results"
DRAFTS_DIR = REPO_ROOT / "docs" / "audit" / "hypotheses" / "drafts"
TEMPLATE_PATH = REPO_ROOT / "docs" / "audit" / "hypotheses" / "TEMPLATE-fast-lane-v5.1.yaml"

# Template anchors — these mirror the v5.1 template constants.
PROMOTE_THRESHOLD: float = 2.5  # v5.1 promote_threshold
N_ADEQUACY_TARGET: int = 200  # parity with cherry_pick_ranker
OOS_N_FLOOR: int = 30  # RULE 3.3 power floor
ERA_STABILITY_MIN_YEARS: int = 5  # coarse proxy

SCORE_WEIGHTS: dict[str, float] = {
    "pooled_t_headroom": 0.35,
    "n_adequacy": 0.15,
    "oos_power_readiness": 0.25,
    "era_stability": 0.15,
    "non_artifact": 0.10,
}

ARTIFACT_FILTER_TYPES: frozenset[str] = frozenset()


# Title pattern in fast-lane / heavyweight result MDs:
#   # Chordia strict unlock audit — <STRATEGY_ID>
# Mirrors the pattern used by the cherry-pick journal enricher.
_RESULT_MD_TITLE_RE = re.compile(
    r"^#\s+Chordia\s+(?:strict\s+unlock|heavyweight)\s+audit\s*[—-]\s*(?P<sid>[A-Z0-9_.]+)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class TriageCandidate:
    """One validated_setups row scored for fast-lane prereg drafting."""

    strategy_id: str
    instrument: str
    orb_label: str
    orb_minutes: int
    entry_model: str
    confirm_bars: int
    rr_target: float
    filter_type: str
    sample_size: int
    expectancy_r: float
    sharpe_ratio: float
    years_tested: int
    all_years_positive: bool
    oos_n: int
    oos_power: float
    score_components: dict[str, float]

    @property
    def total_score(self) -> float:
        return sum(SCORE_WEIGHTS[k] * self.score_components[k] for k in SCORE_WEIGHTS if k in self.score_components)


# ---------- result-MD index ----------


def collect_seen_strategy_ids(results_dir: Path) -> set[str]:
    """Walk docs/audit/results/ and collect every strategy_id stamped in a result MD title.

    A lane is considered "untouched by fast-lane" when its strategy_id does
    not appear as the title of ANY result MD in this directory. Heavyweight
    Chordia MDs also count (they share the same title pattern) — if a lane
    has a heavyweight verdict, the operator already worked it.
    """
    if not results_dir.exists():
        return set()
    seen: set[str] = set()
    for md in results_dir.glob("*.md"):
        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        m = _RESULT_MD_TITLE_RE.search(text)
        if m is not None:
            seen.add(m.group("sid"))
    return seen


# ---------- DB query ----------


def query_active_validated_setups(con: Any) -> list[dict[str, Any]]:
    """Return active validated_setups rows shaped for scoring.

    Read-only over canonical layer. Filters status='active'.
    """
    rows = con.execute(
        """
        SELECT
            strategy_id, instrument, orb_label, orb_minutes, entry_model,
            confirm_bars, rr_target, filter_type, sample_size, expectancy_r,
            sharpe_ratio, years_tested, all_years_positive
        FROM validated_setups
        WHERE status = 'active'
        """
    ).fetchall()
    cols = [
        "strategy_id",
        "instrument",
        "orb_label",
        "orb_minutes",
        "entry_model",
        "confirm_bars",
        "rr_target",
        "filter_type",
        "sample_size",
        "expectancy_r",
        "sharpe_ratio",
        "years_tested",
        "all_years_positive",
    ]
    return [dict(zip(cols, r, strict=True)) for r in rows]


def query_oos_n(con: Any, row: dict[str, Any]) -> int:
    """Count OOS trades for a lane (trading_day >= HOLDOUT_SACRED_FROM).

    Uses ``orb_outcomes`` joined to ``daily_features`` on triple-key
    (trading_day, symbol, orb_minutes) per
    ``.claude/rules/daily-features-joins.md``. Filter application is
    deliberately scoped to the lane's exact dimensions — we do not apply
    the filter_type clause (this is a coarse N count for power estimation,
    not a precise replay).
    """
    from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

    result = con.execute(
        """
        SELECT COUNT(*) AS n
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.trading_day = d.trading_day
           AND o.symbol = d.symbol
           AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.entry_model = ?
          AND o.confirm_bars = ?
          AND o.rr_target = ?
          AND o.trading_day >= ?
        """,
        [
            row["instrument"],
            row["orb_label"],
            row["orb_minutes"],
            row["entry_model"],
            row["confirm_bars"],
            row["rr_target"],
            HOLDOUT_SACRED_FROM,
        ],
    ).fetchone()
    return int(result[0]) if result else 0


# ---------- scoring ----------


def compute_pooled_t_headroom(sharpe: float, n: int) -> float:
    """Sharpe-proxy headroom over PROMOTE_THRESHOLD, normalized by t. Bounded [0, 1).

    Sharpe-proxy t = ``sharpe * sqrt(n)`` (one-sample one-tail). validated_setups
    stores per-trade Sharpe (not annualized) so this is the canonical
    pooled-sample t under H0=0.
    """
    if not isinstance(sharpe, (int, float)) or math.isnan(float(sharpe)):
        return 0.0
    if n <= 0:
        return 0.0
    t = float(sharpe) * math.sqrt(n)
    if t <= PROMOTE_THRESHOLD:
        return 0.0
    return (t - PROMOTE_THRESHOLD) / t


def compute_n_adequacy(n: int) -> float:
    """Linear-headroom anchor at N_ADEQUACY_TARGET. Bounded [0, 1]."""
    if n <= 0:
        return 0.0
    return min(1.0, n / N_ADEQUACY_TARGET)


def compute_oos_power_readiness(sharpe: float, is_n: int, oos_n: int) -> tuple[float, float]:
    """Return (power_score, raw_power). 0.0 below OOS_N_FLOOR; canonical helper above.

    Cohen's d = ``sharpe`` (one-sample d == Sharpe when std cancels — same
    derivation as ``cherry_pick_ranker.compute_oos_power_readiness``).
    """
    if oos_n < OOS_N_FLOOR or is_n <= 0:
        return 0.0, 0.0
    if not isinstance(sharpe, (int, float)) or math.isnan(float(sharpe)):
        return 0.0, 0.0
    try:
        from research.oos_power import one_sample_power
    except ImportError:
        return 0.0, 0.0
    try:
        power = one_sample_power(abs(float(sharpe)), oos_n, alpha=0.05)
    except (ValueError, ZeroDivisionError):
        return 0.0, 0.0
    power = float(power)
    if math.isnan(power):
        return 0.0, 0.0
    bounded = max(0.0, min(1.0, power))
    return bounded, bounded


def compute_era_stability(years_tested: int, all_years_positive: bool) -> float:
    if years_tested >= ERA_STABILITY_MIN_YEARS and all_years_positive:
        return 1.0
    return 0.0


def compute_non_artifact(filter_type: str) -> float:
    return 0.0 if filter_type in ARTIFACT_FILTER_TYPES else 1.0


def score_candidate(row: dict[str, Any], oos_n: int) -> TriageCandidate:
    """Score one validated_setups row. Pure -- no DB or fs side effects."""
    sharpe = row.get("sharpe_ratio") or 0.0
    n_is = int(row.get("sample_size") or 0)
    filter_type = str(row.get("filter_type") or "NONE")

    headroom = compute_pooled_t_headroom(sharpe, n_is)
    n_adq = compute_n_adequacy(n_is)
    power_score, raw_power = compute_oos_power_readiness(sharpe, n_is, oos_n)
    era = compute_era_stability(int(row.get("years_tested") or 0), bool(row.get("all_years_positive")))
    non_art = compute_non_artifact(filter_type)

    return TriageCandidate(
        strategy_id=str(row["strategy_id"]),
        instrument=str(row["instrument"]),
        orb_label=str(row["orb_label"]),
        orb_minutes=int(row["orb_minutes"]),
        entry_model=str(row["entry_model"]),
        confirm_bars=int(row["confirm_bars"]),
        rr_target=float(row["rr_target"]),
        filter_type=filter_type,
        sample_size=n_is,
        expectancy_r=float(row.get("expectancy_r") or 0.0),
        sharpe_ratio=float(sharpe),
        years_tested=int(row.get("years_tested") or 0),
        all_years_positive=bool(row.get("all_years_positive")),
        oos_n=oos_n,
        oos_power=raw_power,
        score_components={
            "pooled_t_headroom": headroom,
            "n_adequacy": n_adq,
            "oos_power_readiness": power_score,
            "era_stability": era,
            "non_artifact": non_art,
        },
    )


# ---------- draft emission ----------


def _slugify_strategy_id(strategy_id: str) -> str:
    """Convert a strategy_id to a filename-safe slug.

    Lowercase, dots -> nothing, underscores -> hyphens, RR-target dots
    collapsed (RR1.0 -> rr10) to mirror existing prereg slug conventions
    (compare ``2026-05-18-mes-cmepreclose-e2-rr10-cb1-costlt15-...``).
    """
    s = strategy_id.lower()
    s = s.replace(".", "")
    s = s.replace("_", "-")
    return s


def build_draft_filename(candidate: TriageCandidate, today: date) -> str:
    slug = _slugify_strategy_id(candidate.strategy_id)
    return f"{today.isoformat()}-triage-{slug}-fast-lane-v51.draft.yaml"


def build_draft_yaml(candidate: TriageCandidate, *, today: date) -> str:
    """Construct a fast-lane v5.1 prereg draft yaml as a single string.

    Hand-rolled YAML (not safe_dump round-trip) so the operator gets a
    review-friendly layout with the same comment scaffolding as the
    template. ``triage_provenance`` block is mandatory; Check #165
    enforces ``source_validated_setup_strategy_id`` presence.
    """
    lookahead_e2 = (
        f"  lookahead_banned_if_e2:\n"
        f"    - 'orb_{candidate.orb_minutes}_break_ts'\n"
        f"    - 'orb_{candidate.orb_minutes}_break_delay_min'\n"
        f"    - 'orb_{candidate.orb_minutes}_break_bar_continues'\n"
        f"    - 'orb_{candidate.orb_minutes}_break_bar_volume'\n"
        f"    - 'orb_{candidate.orb_minutes}_break_dir (when used as predictor)'\n"
        f"    - 'rel_vol_{candidate.orb_label} (numerator is break_bar_volume — same look-ahead class)'\n"
    )

    components = candidate.score_components
    return (
        f"# TRIAGE-GENERATED FAST_LANE v5.1 prereg — UNVALIDATED DRAFT\n"
        f"#\n"
        f"# Source: scripts/research/triage_validated_setups.py (2026-05-19 cycle)\n"
        f"# Source validated_setups strategy_id: {candidate.strategy_id}\n"
        f"# rank_score={candidate.total_score:.4f} "
        f"oos_n={candidate.oos_n} oos_power={candidate.oos_power:.3f}\n"
        f"#\n"
        f"# Operator review checklist before promoting drafts/ -> hypotheses/:\n"
        f"#   1. [ ] /nogo {_slugify_strategy_id(candidate.strategy_id)} returns NOT_FOUND\n"
        f"#   2. [ ] Lane still tradeable (instrument/session/aperture in active config)\n"
        f"#   3. [ ] Sample size + Sharpe still plausible at current cohort lower bound\n"
        f"#   4. [ ] If --ground-via-mcp is desired, run llm_hypothesis_proposer with\n"
        f"#         the candidate strategy_id BEFORE promoting\n"
        f"#\n"
        f"metadata:\n"
        f"  name: '{candidate.strategy_id.lower()}_fast_lane'\n"
        f'  purpose: "Triage-generated fast-lane screen — does this validated lane clear v5.1 promote_threshold?"\n'
        f'  date_locked: "{today.isoformat()}+10:00"\n'
        f"  pathway: 'B_individual'\n"
        f"  testing_mode: 'individual'\n"
        f"  n_trials: 1\n"
        f"  template_version: 'fast_lane_v5.1'\n"
        f"  supersedes: 'fast_lane_v5'\n"
        f"  is_triage_screen: true\n"
        f"  promotion_target: 'heavyweight_chordia_prereg'\n"
        f'  validation_status_explicit: "NOT_VALIDATED — triage queue-prioritisation screen, not calibrated"\n'
        f"  nogo_check:\n"
        f"    result: 'PENDING_OPERATOR'\n"
        f'    method: "operator MUST run /nogo before promoting drafts/ -> hypotheses/"\n'
        f"\n"
        f"triage_provenance:\n"
        f"  source: 'scripts/research/triage_validated_setups.py'\n"
        f"  source_validated_setup_strategy_id: '{candidate.strategy_id}'\n"
        f"  rank_score: {candidate.total_score:.4f}\n"
        f"  score_components:\n"
        f"    pooled_t_headroom: {components.get('pooled_t_headroom', 0.0):.4f}\n"
        f"    n_adequacy: {components.get('n_adequacy', 0.0):.4f}\n"
        f"    oos_power_readiness: {components.get('oos_power_readiness', 0.0):.4f}\n"
        f"    era_stability: {components.get('era_stability', 0.0):.4f}\n"
        f"    non_artifact: {components.get('non_artifact', 0.0):.4f}\n"
        f"  validated_setups_snapshot:\n"
        f"    sample_size: {candidate.sample_size}\n"
        f"    expectancy_r: {candidate.expectancy_r:.4f}\n"
        f"    sharpe_ratio: {candidate.sharpe_ratio:.4f}\n"
        f"    years_tested: {candidate.years_tested}\n"
        f"    all_years_positive: {str(candidate.all_years_positive).lower()}\n"
        f"  canonical_oos_count:\n"
        f"    n_oos: {candidate.oos_n}\n"
        f"    oos_power: {candidate.oos_power:.4f}\n"
        f"    holdout_floor: '2026-01-01 (HOLDOUT_SACRED_FROM)'\n"
        f"\n"
        f"scope:\n"
        f"  instrument: {candidate.instrument}\n"
        f"  strategy_id: '{candidate.strategy_id}'\n"
        f"  session: {candidate.orb_label}\n"
        f"  orb_minutes: {candidate.orb_minutes}\n"
        f"  entry_model: {candidate.entry_model}\n"
        f"  confirm_bars: {candidate.confirm_bars}\n"
        f"  rr_target: {candidate.rr_target}\n"
        f"  filter_type: '{candidate.filter_type}'\n"
        f"  filter_source: \"trading_app.config.ALL_FILTERS['{candidate.filter_type}']\"\n"
        f"  direction: pooled\n"
        f"\n"
        f"holdout:\n"
        f"  holdout_date: '2026-01-01'\n"
        f"  enforce_via: 'trading_app.holdout_policy.enforce_holdout_date'\n"
        f"  violation_outcome: 'NEEDS-MORE'\n"
        f'  holdout_clean_definition: "max_IS_trading_day < min_OOS_trading_day AND min_OOS_trading_day >= holdout_date AND no override token in runner invocation"\n'
        f"\n"
        f"data_policy:\n"
        f"  scratch_policy: 'realized-eod'\n"
        f"  canonical_layers_only: true\n"
        f"  lookahead_banned_always: [mae_r, mfe_r, outcome, pnl_r]\n"
        f"{lookahead_e2}"
        f"  e2_lookahead_banned_if_e2: true\n"
        f"\n"
        f"screen:\n"
        f"  metric: 't_IS'\n"
        f"  promote_threshold: 2.5\n"
        f"  expr_min: 0.0\n"
        f"  n_IS_on_min: 50\n"
        f"  needs_more_band: 0.5\n"
        f"  fire_rate_gate:\n"
        f'    kill_if: "fire_rate < 0.05 OR fire_rate > 0.95"\n'
    )


def write_draft(
    candidate: TriageCandidate,
    *,
    drafts_dir: Path,
    today: date,
    overwrite: bool = False,
) -> Path:
    """Write the v5.1 draft yaml to ``drafts_dir/<filename>``. Returns the path."""
    drafts_dir.mkdir(parents=True, exist_ok=True)
    out = drafts_dir / build_draft_filename(candidate, today)
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"Draft already exists at {out}. Pass --overwrite to replace, or rename the existing file."
        )
    out.write_text(build_draft_yaml(candidate, today=today), encoding="utf-8")
    return out


# ---------- CLI ----------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="triage_validated_setups",
        description=(
            "Rank validated_setups lanes with no fast-lane MD yet, emit "
            "v5.1 prereg drafts under docs/audit/hypotheses/drafts/. "
            "Read-only on canonical layers; write-only to drafts/."
        ),
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top-ranked candidates to draft (default: 10).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Score and rank but do NOT write draft files. Default behaviour.",
    )
    p.add_argument(
        "--write",
        action="store_true",
        help="Persist draft files to docs/audit/hypotheses/drafts/.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing draft files. Default: refuse to overwrite.",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Override results dir (test seam).",
    )
    p.add_argument(
        "--drafts-dir",
        type=Path,
        default=DRAFTS_DIR,
        help="Override drafts dir (test seam).",
    )
    p.add_argument(
        "--instrument",
        choices=["MNQ", "MES", "MGC"],
        default=None,
        help="Narrow scoring to one instrument.",
    )
    return p


def _connect_read_only() -> Any:
    import duckdb

    from pipeline.paths import GOLD_DB_PATH

    return duckdb.connect(str(GOLD_DB_PATH), read_only=True)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    seen = collect_seen_strategy_ids(args.results_dir)

    try:
        con = _connect_read_only()
    except Exception as exc:
        print(f"ERROR: cannot open gold.db: {exc}", file=sys.stderr)
        return 5

    try:
        rows = query_active_validated_setups(con)
        if args.instrument:
            rows = [r for r in rows if r["instrument"] == args.instrument]
        untouched = [r for r in rows if r["strategy_id"] not in seen]

        candidates: list[TriageCandidate] = []
        for r in untouched:
            try:
                oos_n = query_oos_n(con, r)
            except Exception:
                oos_n = 0
            candidates.append(score_candidate(r, oos_n))
    finally:
        con.close()

    candidates.sort(key=lambda c: c.total_score, reverse=True)

    print(f"{'rank':>4} {'score':>6} {'sharpe':>6} {'n_IS':>5} {'oos_n':>5} {'pow':>5} {'years':>5}  strategy_id")
    for i, c in enumerate(candidates[: args.top_k], start=1):
        print(
            f"{i:>4} {c.total_score:>6.3f} {c.sharpe_ratio:>6.2f} "
            f"{c.sample_size:>5} {c.oos_n:>5} {c.oos_power:>5.2f} "
            f"{c.years_tested:>5}  {c.strategy_id}"
        )

    if args.write and not args.dry_run:
        today = date.today()
        for c in candidates[: args.top_k]:
            try:
                path = write_draft(c, drafts_dir=args.drafts_dir, today=today, overwrite=args.overwrite)
                rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
                print(f"DRAFT_WRITTEN: {rel}")
            except FileExistsError as exc:
                print(f"SKIPPED: {exc}", file=sys.stderr)
    else:
        print(f"\nDRY RUN — {min(args.top_k, len(candidates))} draft(s) NOT written. Pass --write to persist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
