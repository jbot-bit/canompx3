"""Cherry-pick ranker — FAST_LANE PROMOTE survivors ranked by heavyweight-Chordia pass probability.

Reads ``docs/runtime/promote_queue.yaml`` (derived state, rebuilt by
``scripts/research/fast_lane_promote_queue.py``), filters to QUEUED entries
(PROMOTE survivors with no heavyweight pre-reg yet authored, no revocation,
no park), then scores each by transparent components:

- deflation_headroom: ``max(0, pooled_t - HEAVYWEIGHT_T_THRESHOLD) / pooled_t``
  Pre-clustered-SE headroom above the strict t=3.79 threshold (Chordia 2018,
  see ``pre_registered_criteria.md`` Criterion 4). Naive t will deflate under
  clustered-SE at trading_day per
  ``feedback_clustered_se_trading_day_pooled_finding_guard.md``; the gap is
  the binding budget for that deflation.
- n_adequacy: ``min(1.0, pooled_n / N_ADEQUACY_TARGET)`` -- linear up to a
  reasonable clustered-SE-floor reference (``N_unique_trading_days >= 30``
  per ``feedback_n_unique_trading_days_floor_clustered_se.md``; pooled_n is
  trade count, so the target is set above the floor).
- oos_power_readiness: result of ``research.oos_power.oos_ttest_power``
  evaluated against the OOS row parsed from the result MD. Returns 0.0 when
  N_OOS < OOS_N_FLOOR (the same N>=30 floor cited in RULE 3.3); otherwise
  the computed power value in [0, 1].
- dir_match: 1.0 if sign(is_expr) == sign(oos_expr), 0.0 otherwise. Pure
  indicator -- the magnitude question is delegated to oos_power_readiness.
- non_artifact: 1.0 if ``pooling_artifact`` is False in the queue entry,
  0.0 otherwise (both-sides-KILL pooled rows already fall to REVOKED, so
  this is defensive against future relaxation).
- era_stability_proxy: weight=0 in v1 by design. Reserved for future journal
  feedback; manually adjusted only -- no auto-tuning per
  ``feedback_meta_tooling_n1_tunnel_2026_05_01.md``.

Output: CSV at ``docs/runtime/cherry_pick_ranking_<date>.csv`` (--write flag)
plus stdout table of top-N. Read-only by default (--dry-run).

This script does NOT mutate ``chordia_audit_log.yaml``, ``validated_setups``,
``lane_allocation.json``, or any file under ``trading_app/live/``. The ranker
is a descriptive surface; the operator decides whether to bridge a candidate
into heavyweight via the companion ``fast_lane_to_heavyweight_bridge.py``.

Canonical-inline-copy contract
------------------------------
``HEAVYWEIGHT_T_THRESHOLD`` mirrors ``pre_registered_criteria.md`` Criterion 4
no-theory threshold (Chordia 2018 verbatim, ``literature/chordia_et_al_2018
_two_million_strategies.md:20``). Parity enforced by
``pipeline.check_drift.check_cherry_pick_ranker_threshold_parity`` (Check
#160). See ``pipeline/canonical_inline_copies.py`` registry entry
``cherry_pick_ranker_heavyweight_t_threshold`` for full registration.

Doctrine grounding
------------------
- Plan: ``C:/Users/joshd/.claude/plans/or-linknin-them-togehr-delegated-gizmo.md``
- Stage file: ``docs/runtime/stages/2026-05-19-cherry-pick-ranker.md``
- Bug class: ``memory/feedback_canonical_inline_copy_parity_bug_class.md``
  (5th confirmed instance, 2026-05-19)
- Power floor: ``.claude/rules/backtesting-methodology.md`` RULE 3.3 + canonical
  helper ``research/oos_power.py``
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMOTE_QUEUE = REPO_ROOT / "docs" / "runtime" / "promote_queue.yaml"
RESULTS_DIR = REPO_ROOT / "docs" / "audit" / "results"
RANKING_DIR = REPO_ROOT / "docs" / "runtime"

# Canonical-inline-copy: HEAVYWEIGHT_T_THRESHOLD mirrors pre_registered_criteria.md
# Criterion 4 no-theory threshold (Chordia 2018 verbatim Tier 1 at
# literature/chordia_et_al_2018_two_million_strategies.md:20). DO NOT EDIT THIS
# CONSTANT without amending the canonical doctrine; Check #160 enforces parity.
HEAVYWEIGHT_T_THRESHOLD: float = 3.79

# N adequacy target -- linear-headroom anchor for the n_adequacy component.
# Chosen above the clustered-SE N_unique_trading_days >= 30 floor (a trades-per-
# day ratio of ~6-7 typical on the active sessions reaches this) so that a
# pooled_n at the floor scores near full credit. Not canonical -- internal to
# this ranker's design; tune via journal lessons (manually).
N_ADEQUACY_TARGET: int = 200

# OOS N floor below which oos_power_readiness is forced to 0.0. Mirrors RULE
# 3.3's underpowered-OOS cliff (N < 30 -> STATISTICALLY_USELESS). Not canonical-
# inlined -- the canonical source is .claude/rules/backtesting-methodology.md
# § 3.2 and POWER_TIERS in research/oos_power.py.
OOS_N_FLOOR: int = 30

# Default score-skip threshold. A QUEUED entry with score < SCORE_SKIP_FLOOR
# is descriptively flagged for operator review (-skip column True) but still
# emitted in the CSV. Pure heuristic; tune via journal lessons.
SCORE_SKIP_FLOOR: float = 0.40


# Score component weights. Sum to 1.0 by construction; era_stability_proxy
# left at 0.0 in v1 by design (no journal data yet -- meta-tooling-on-n=1
# trap). Adjust manually after >=3 iterations land in the journal.
WEIGHTS: dict[str, float] = {
    "deflation_headroom": 0.30,
    "n_adequacy": 0.15,
    "oos_power_readiness": 0.25,
    "dir_match": 0.15,
    "non_artifact": 0.10,
    "era_stability_proxy": 0.05,
}


# Match the "| OOS | ..." row from the canonical fast-lane result MD split
# summary. Tolerant of NaN markers and negative numbers. Mirrors structure of
# the IS-row regex in fast_lane_promote_queue.py.
_OOS_RE = re.compile(
    r"^\|\s*OOS\s*\|\s*(?P<nu>[-+0-9.nNaA]+)\s*\|\s*(?P<nf>[-+0-9.nNaA]+)\s*"
    r"\|\s*(?P<fp>[-+0-9.nNaA]+)%\s*\|\s*\S+\s*\|\s*\S+\s*"
    r"\|\s*(?P<expr>[-+0-9.eEnNaA]+)\s*\|\s*\S+\s*\|\s*\S+\s*"
    r"\|\s*(?P<t>[-+0-9.eEnNaA]+)\s*\|",
    re.MULTILINE,
)


def _parse_float(s: str) -> float:
    s = s.strip().lower()
    if s in {"nan", "", "n/a"}:
        return float("nan")
    return float(s)


def _parse_int(s: str) -> int:
    s = s.strip().lower()
    if s in {"nan", "", "n/a"}:
        return 0
    return int(float(s))


@dataclass(frozen=True)
class OOSStats:
    """OOS row parsed from the result MD split summary."""

    n_oos: int
    expr_oos: float
    t_oos: float


def parse_oos_row(result_md_path: Path) -> OOSStats | None:
    """Parse the OOS split-summary row from a fast-lane result MD.

    Returns None when the file is missing or the OOS row is not present.
    Single-direction or lanes that fired zero OOS trades are returned with
    n_oos=0, expr_oos=NaN, t_oos=NaN -- the caller decides how to score.
    """
    if not result_md_path.exists():
        return None
    text = result_md_path.read_text(encoding="utf-8")
    m = _OOS_RE.search(text)
    if m is None:
        return None
    try:
        return OOSStats(
            n_oos=_parse_int(m.group("nf")),
            expr_oos=_parse_float(m.group("expr")),
            t_oos=_parse_float(m.group("t")),
        )
    except ValueError:
        return None


@dataclass(frozen=True)
class ScoreBreakdown:
    """Per-component score for a candidate -- transparent and audit-grade."""

    deflation_headroom: float
    n_adequacy: float
    oos_power_readiness: float
    dir_match: float
    non_artifact: float
    era_stability_proxy: float

    @property
    def total(self) -> float:
        return (
            WEIGHTS["deflation_headroom"] * self.deflation_headroom
            + WEIGHTS["n_adequacy"] * self.n_adequacy
            + WEIGHTS["oos_power_readiness"] * self.oos_power_readiness
            + WEIGHTS["dir_match"] * self.dir_match
            + WEIGHTS["non_artifact"] * self.non_artifact
            + WEIGHTS["era_stability_proxy"] * self.era_stability_proxy
        )


def compute_deflation_headroom(pooled_t: float) -> float:
    """Headroom of pooled_t above HEAVYWEIGHT_T_THRESHOLD, normalized by t.

    Returns 0.0 if pooled_t <= HEAVYWEIGHT_T_THRESHOLD. Bounded [0, 1).
    """
    if math.isnan(pooled_t) or pooled_t <= HEAVYWEIGHT_T_THRESHOLD:
        return 0.0
    return (pooled_t - HEAVYWEIGHT_T_THRESHOLD) / pooled_t


def compute_n_adequacy(pooled_n: int) -> float:
    """Linear-headroom anchor capped at N_ADEQUACY_TARGET. Bounded [0, 1]."""
    if pooled_n <= 0:
        return 0.0
    return min(1.0, pooled_n / N_ADEQUACY_TARGET)


def compute_oos_power_readiness(
    pooled_expr: float, pooled_t: float, pooled_n: int, oos_stats: OOSStats | None
) -> float:
    """One-sample power of the OOS sample to detect the IS effect at alpha=0.05.

    Returns 0.0 if OOS is unavailable, OOS N < OOS_N_FLOOR, or pooled IS stats
    are degenerate. Otherwise delegates to ``research.oos_power.one_sample_power``.

    Cohen's d = |t_IS| / sqrt(N_IS). Derivation: canonical runner emits
    one-sample t = mean * sqrt(N) / std, so std = mean * sqrt(N) / t and
    d = |mean| / std = |t| / sqrt(N). The IS std cancels; only IS t and N needed.
    """
    if oos_stats is None:
        return 0.0
    if oos_stats.n_oos < OOS_N_FLOOR:
        return 0.0
    if math.isnan(pooled_expr) or math.isnan(pooled_t) or pooled_n <= 0:
        return 0.0
    if abs(pooled_t) < 1e-9:
        return 0.0
    cohen_d = abs(pooled_t) / math.sqrt(pooled_n)
    try:
        from research.oos_power import one_sample_power
    except ImportError:
        return 0.0
    try:
        power = one_sample_power(cohen_d, oos_stats.n_oos, alpha=0.05)
    except (ValueError, ZeroDivisionError):
        return 0.0
    power = float(power)
    if math.isnan(power):
        return 0.0
    return max(0.0, min(1.0, power))


def compute_dir_match(pooled_expr: float, oos_stats: OOSStats | None) -> float:
    """1.0 if IS and OOS ExpR have the same sign, 0.0 otherwise.

    Returns 0.0 when OOS data is missing or either ExpR is NaN.
    """
    if oos_stats is None:
        return 0.0
    if math.isnan(pooled_expr) or math.isnan(oos_stats.expr_oos):
        return 0.0
    if pooled_expr == 0.0 or oos_stats.expr_oos == 0.0:
        return 0.0
    return 1.0 if (pooled_expr > 0) == (oos_stats.expr_oos > 0) else 0.0


def compute_non_artifact(pooling_artifact: bool) -> float:
    """1.0 if not a pooling artifact, 0.0 otherwise."""
    return 0.0 if pooling_artifact else 1.0


@dataclass(frozen=True)
class RankedCandidate:
    """A scored fast-lane PROMOTE survivor, ready for operator review."""

    strategy_id: str
    direction: str
    pooled_t: float
    pooled_n: int
    pooled_expr: float
    pooling_artifact: bool
    oos_n: int
    oos_expr: float
    dir_match: bool
    score: ScoreBreakdown
    skip_recommended: bool
    result_md: str

    @property
    def total_score(self) -> float:
        return self.score.total


def _load_queue_entries(queue_path: Path) -> list[dict[str, Any]]:
    """Load promote_queue.yaml entries. Returns [] when file missing/invalid."""
    if not queue_path.exists():
        return []
    try:
        payload = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return []
    if not isinstance(payload, dict):
        return []
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict)]


def rank_queue_entries(
    queue_entries: list[dict[str, Any]],
    *,
    results_dir: Path | None = None,
) -> list[RankedCandidate]:
    """Score every QUEUED entry (PROMOTE survivor, no heavyweight, no revocation, no park)."""
    rd = results_dir if results_dir is not None else RESULTS_DIR
    ranked: list[RankedCandidate] = []
    for entry in queue_entries:
        status = entry.get("status")
        if status != "QUEUED":
            continue
        strategy_id = entry.get("strategy_id")
        if not isinstance(strategy_id, str):
            continue

        # Explicit defaults (not `or`-fallback) so a legitimate 0/0.0 value
        # from the queue cache is preserved -- prior `or float("nan")` falsy-
        # coerced a runtime 0 into NaN, which is harmless for current data
        # but would silently mask a degenerate t=0 fast-lane result. Code-
        # review A- residual close, 2026-05-19.
        def _opt_float(key: str) -> float:
            v = entry.get(key)
            if v is None:
                return float("nan")
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")

        def _opt_int(key: str) -> int:
            v = entry.get(key)
            if v is None:
                return 0
            try:
                return int(v)
            except (TypeError, ValueError):
                return 0

        pooled_t = _opt_float("pooled_t")
        pooled_n = _opt_int("pooled_n")
        pooled_expr = _opt_float("pooled_expr")
        pooling_artifact = bool(entry.get("pooling_artifact") or False)
        direction = str(entry.get("direction") or "pooled")

        # Result MD path is repo-relative in the queue cache; resolve here.
        result_md_rel = entry.get("result_md")
        oos: OOSStats | None = None
        if isinstance(result_md_rel, str):
            result_md_abs = (REPO_ROOT / result_md_rel).resolve()
            # Tolerate test fixtures that pass a different results_dir.
            if not result_md_abs.exists():
                result_md_abs = rd / Path(result_md_rel).name
            oos = parse_oos_row(result_md_abs)

        breakdown = ScoreBreakdown(
            deflation_headroom=compute_deflation_headroom(pooled_t),
            n_adequacy=compute_n_adequacy(pooled_n),
            oos_power_readiness=compute_oos_power_readiness(
                pooled_expr, pooled_t, pooled_n, oos
            ),
            dir_match=compute_dir_match(pooled_expr, oos),
            non_artifact=compute_non_artifact(pooling_artifact),
            era_stability_proxy=0.0,
        )
        ranked.append(
            RankedCandidate(
                strategy_id=strategy_id,
                direction=direction,
                pooled_t=pooled_t,
                pooled_n=pooled_n,
                pooled_expr=pooled_expr,
                pooling_artifact=pooling_artifact,
                oos_n=oos.n_oos if oos is not None else 0,
                oos_expr=oos.expr_oos if oos is not None else float("nan"),
                dir_match=bool(breakdown.dir_match),
                score=breakdown,
                skip_recommended=breakdown.total < SCORE_SKIP_FLOOR,
                result_md=str(result_md_rel) if result_md_rel else "",
            )
        )

    ranked.sort(key=lambda c: c.total_score, reverse=True)
    return ranked


CSV_COLUMNS = [
    "rank",
    "strategy_id",
    "direction",
    "pooled_t",
    "pooled_n",
    "pooled_expr",
    "oos_n",
    "oos_expr",
    "dir_match",
    "pooling_artifact",
    "score",
    "deflation_headroom",
    "n_adequacy",
    "oos_power_readiness",
    "score_dir_match",
    "non_artifact",
    "era_stability_proxy",
    "skip_recommended",
    "result_md",
]


def _row_for(candidate: RankedCandidate, rank: int) -> dict[str, str]:
    s = candidate.score
    return {
        "rank": str(rank),
        "strategy_id": candidate.strategy_id,
        "direction": candidate.direction,
        "pooled_t": f"{candidate.pooled_t:.4f}",
        "pooled_n": str(candidate.pooled_n),
        "pooled_expr": f"{candidate.pooled_expr:.4f}",
        "oos_n": str(candidate.oos_n),
        "oos_expr": (
            "NaN" if math.isnan(candidate.oos_expr) else f"{candidate.oos_expr:.4f}"
        ),
        "dir_match": str(candidate.dir_match),
        "pooling_artifact": str(candidate.pooling_artifact),
        "score": f"{candidate.total_score:.4f}",
        "deflation_headroom": f"{s.deflation_headroom:.4f}",
        "n_adequacy": f"{s.n_adequacy:.4f}",
        "oos_power_readiness": f"{s.oos_power_readiness:.4f}",
        "score_dir_match": f"{s.dir_match:.4f}",
        "non_artifact": f"{s.non_artifact:.4f}",
        "era_stability_proxy": f"{s.era_stability_proxy:.4f}",
        "skip_recommended": str(candidate.skip_recommended),
        "result_md": candidate.result_md,
    }


def write_csv(ranked: list[RankedCandidate], out_path: Path) -> None:
    """Write the ranking CSV. Creates parent directory if needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for i, c in enumerate(ranked, start=1):
            writer.writerow(_row_for(c, i))


def format_stdout_table(ranked: list[RankedCandidate], top_n: int) -> str:
    """Render a compact stdout table of top-N candidates."""
    if not ranked:
        return "No QUEUED candidates in promote_queue.yaml."
    lines = [
        f"{'rank':>4} {'score':>6} {'t':>6} {'n':>5} {'oos_n':>5} {'dir':>4} "
        f"{'skip':>4}  strategy_id"
    ]
    for i, c in enumerate(ranked[:top_n], start=1):
        lines.append(
            f"{i:>4} {c.total_score:>6.3f} {c.pooled_t:>6.2f} {c.pooled_n:>5} "
            f"{c.oos_n:>5} {('Y' if c.dir_match else 'N'):>4} "
            f"{('Y' if c.skip_recommended else 'N'):>4}  {c.strategy_id}"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cherry_pick_ranker",
        description=(
            "Rank FAST_LANE PROMOTE survivors by heavyweight-Chordia pass "
            "probability. Read-only by default; --write emits CSV."
        ),
    )
    p.add_argument(
        "--queue",
        type=Path,
        default=PROMOTE_QUEUE,
        help="Path to promote_queue.yaml (default: docs/runtime/promote_queue.yaml).",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Path to fast-lane result MDs (default: docs/audit/results/).",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of rows to print to stdout (default: 10).",
    )
    p.add_argument(
        "--write",
        action="store_true",
        help=(
            "Write CSV to docs/runtime/cherry_pick_ranking_<date>.csv. "
            "Default is dry-run (stdout only)."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    entries = _load_queue_entries(args.queue)
    ranked = rank_queue_entries(entries, results_dir=args.results_dir)
    print(format_stdout_table(ranked, args.top_n))
    if args.write:
        out = RANKING_DIR / f"cherry_pick_ranking_{date.today().isoformat()}.csv"
        write_csv(ranked, out)
        rel = out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out
        print(f"\nWrote {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
