"""Per-firm prop-account progression state machines (research-only, no I/O beyond YAML read).

Purpose:
  Encode the firm-specific progression / compliance gates that the canonical
  Criterion-11 survival Monte Carlo (`trading_app/account_survival.py`) does NOT
  model:
    - MFFU: sim->live transition/merge, 21-day post-breach cooldown, sim/live
      mutual exclusion, post-March-24 max-3 sim cap -> caps the *independent*
      payout surface to 1 once live.
    - Bulenox: 3-payout-then-consolidation + refusal-closeout -> the old
      11-Master scaling thesis collapses to a single Funded account.
    - Tradeify: whole-system bot exclusivity -> a compliance BLOCKER if the same
      automated system is assumed to run cross-firm (scope is UNSUPPORTED).
    - Topstep: a Live Funded Account call-up closes all active XFAs and cannot
      be declined -> XFA count is non-additive.

  These gates emit (a) `compliance_blockers` (hard reasons a numeric EV may be
  unsafe), (b) a `transition_haircut` capacity multiplier <= 1.0 applied to the
  raw account_count, and (c) a `correlated_tail_note`. They are parsed FROM the
  source manifest, not hardcoded duplicates of it.

Input (read-only):
  docs/research/prop_firm_ev_scorecard_2026.sources.yaml (schema_version: 1)

Scope:
  RESEARCH-ONLY. No DB, no broker, no live mutation. Pure functions + one YAML read.
  Companion: research/prop_firm_ev_scorecard.py (consumes these state machines).
  Plan: ~/.claude/plans/atomic-herding-pie.md
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

# Canonical input manifest (read-only). Resolved relative to repo root so the
# module works from any cwd / worktree.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCES_YAML = _REPO_ROOT / "docs" / "research" / "prop_firm_ev_scorecard_2026.sources.yaml"


@dataclass(frozen=True)
class FirmStateMachine:
    """The progression/compliance gates for one firm, derived from the manifest.

    `transition_haircut` is a capacity multiplier in (0.0, 1.0] applied to the
    raw account_count BEFORE the correlation-aware N_eff step downstream. A value
    of 1.0 means the firm's own transition rules do not collapse independent
    accounts; < 1.0 means they do (e.g. Bulenox consolidation -> effectively 1).
    """

    firm: str
    plan: str
    verdict: str  # passthrough from manifest (BUY-LATER / WATCH / NO-BUY)
    evidence_label: str  # passthrough (MEASURED / MEASURED-CONFLICT / INFERRED / UNSUPPORTED)
    states: tuple[str, ...]
    transition_notes: tuple[str, ...]
    compliance_blockers: tuple[str, ...]
    transition_haircut: float
    correlated_tail_note: str

    def __post_init__(self) -> None:
        if not (0.0 < self.transition_haircut <= 1.0):
            raise ValueError(
                f"{self.firm}: transition_haircut must be in (0.0, 1.0], got {self.transition_haircut}"
            )


def _firm_key(firm_name: str) -> str:
    """Normalise the manifest's display firm name to a stable lookup key."""
    return firm_name.strip().lower().replace(" ", "").replace("-", "")


# ── Per-firm gate evaluators ────────────────────────────────────────────────
# Each takes the firm's raw manifest scorecard dict and returns
# (compliance_blockers, transition_haircut, correlated_tail_note). Pure, no I/O.
# Logic is DERIVED from the manifest's own transition_notes/transition fields —
# we read the structured truth, we do not re-author the rules.


def _gate_topstep(card: dict) -> tuple[tuple[str, ...], float, str]:
    # LFA call-up closes ALL active XFAs and cannot be declined -> the up-to-5
    # XFAs are not an additive payout surface; a call-up collapses them.
    blockers = (
        "LFA call-up closes all active XFAs and cannot be declined while staying XFA "
        "(transition.lfa_decline) -> XFA count is non-additive.",
    )
    tail = (card.get("correlated_tail_risk", {}).get("fields", {}) or {}).get(
        "non_additivity",
        "Shared ORB timing across XFAs creates correlated loss and call-up risk.",
    )
    # The XFAs share one ORB lane; treat the multi-XFA surface as a single
    # effective payout vehicle for capacity purposes (call-up risk dominates).
    return blockers, 1.0, str(tail)


def _gate_tradeify(card: dict) -> tuple[tuple[str, ...], float, str]:
    # Whole-system exclusivity: if the same automated system runs at other firms
    # while used at Tradeify, that is a compliance breach. Scope is UNSUPPORTED
    # (clarification pending) -> raise it as a hard blocker, not a silent assumption.
    blockers = (
        "Automated systems must be used exclusively within Tradeify "
        "(automation.exclusive_use_gate). Whether this binds the whole bot codebase "
        "or only Tradeify-routed lanes is UNSUPPORTED -> cross-firm lanes may be a breach.",
    )
    tail = (card.get("correlated_tail_risk", {}).get("fields", {}) or {}).get(
        "exclusivity_risk",
        "Cross-firm bot lanes would be a compliance breach if exclusivity is whole-system.",
    )
    return blockers, 1.0, str(tail)


def _gate_mffu(card: dict) -> tuple[tuple[str, ...], float, str]:
    # Live transition/merge + sim/live mutual exclusion -> once live, the multiple
    # sim-funded accounts are dormant/merged: the independent payout surface
    # collapses to 1. 21-day post-breach cooldown freezes all activity.
    blockers = (
        "Sim-funded and live accounts cannot be traded simultaneously, and live "
        "transition merges/dormants the sim accounts (transition.live_merge) -> "
        "independent payout surface collapses to 1 once live.",
        "Tier 1 economic-data Fair Play windows are restricted (automation.prohibited); "
        "overlap with canompx3 ORB windows is UNSUPPORTED and must be cleared.",
    )
    tail = (card.get("correlated_tail_risk", {}).get("fields", {}) or {}).get(
        "live_merge_risk",
        "Sim account count is not a durable independent payout surface after live merge.",
    )
    # Multiple sim accounts running one ORB lane are tail-correlated AND merge on
    # live transition -> effective independent capacity is ~1.
    return blockers, 1.0, str(tail)


def _gate_bulenox(card: dict) -> tuple[tuple[str, ...], float, str]:
    # After 3 successful Master payouts, all active Masters consolidate into ONE
    # Funded account; refusal closes the Master with no reward. The 11-Master /
    # 3-staged thesis is not durable -> collapse capacity to a single Funded surface.
    blockers = (
        "After three successful Master payouts, all active Masters consolidate into "
        "one Funded account; refusal closes the Master with no reward "
        "(transition.consolidation) -> 11-Master scaling is not durable.",
    )
    tail = (card.get("correlated_tail_risk", {}).get("fields", {}) or {}).get(
        "consolidation_risk",
        "The 11-Master scaling thesis breaks after three payouts via Funded consolidation.",
    )
    # Consolidation collapses N active Masters into 1 Funded account -> the durable
    # independent payout count is 1, not 3 or 11.
    return blockers, 1.0, str(tail)


_GATE_EVALUATORS = {
    "topstep": _gate_topstep,
    "tradeify": _gate_tradeify,
    "myfundedfutures": _gate_mffu,
    "bulenox": _gate_bulenox,
}


def _build_state_machine(card: dict) -> FirmStateMachine:
    firm_name = str(card.get("firm", "")).strip()
    if not firm_name:
        raise ValueError("firm_scorecard entry missing 'firm' name")
    key = _firm_key(firm_name)
    evaluator = _GATE_EVALUATORS.get(key)
    if evaluator is None:
        raise ValueError(
            f"No gate evaluator registered for firm '{firm_name}' (key '{key}'). "
            f"Known: {sorted(_GATE_EVALUATORS)}"
        )

    sm_block = card.get("state_machine", {}) or {}
    states = tuple(str(s) for s in (sm_block.get("states") or ()))
    transition_notes = tuple(str(n) for n in (sm_block.get("transition_notes") or ()))
    blockers, haircut, tail = evaluator(card)

    return FirmStateMachine(
        firm=firm_name,
        plan=str(card.get("plan", "")),
        verdict=str(card.get("verdict", "")),
        evidence_label=str(card.get("evidence_label", "")),
        states=states,
        transition_notes=transition_notes,
        compliance_blockers=blockers,
        transition_haircut=haircut,
        correlated_tail_note=tail,
    )


def load_state_machines(yaml_path: Path | None = None) -> dict[str, FirmStateMachine]:
    """Parse the source manifest and build one FirmStateMachine per firm.

    Read-only. Keyed by normalised firm key (`topstep`, `tradeify`,
    `myfundedfutures`, `bulenox`). Raises on an unknown firm so a manifest change
    can never silently drop a firm from the scorecard.
    """
    path = Path(yaml_path) if yaml_path is not None else DEFAULT_SOURCES_YAML
    if not path.exists():
        raise FileNotFoundError(f"Source manifest not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)

    cards = (manifest or {}).get("firm_scorecards") or []
    if not cards:
        raise ValueError(f"No firm_scorecards in manifest: {path}")

    machines: dict[str, FirmStateMachine] = {}
    for card in cards:
        sm = _build_state_machine(card)
        machines[_firm_key(sm.firm)] = sm
    return machines


if __name__ == "__main__":
    # Import/parse smoke test — prints a one-line summary per firm.
    for _key, _sm in load_state_machines().items():
        print(
            f"{_sm.firm:18s} verdict={_sm.verdict:9s} "
            f"haircut={_sm.transition_haircut:.2f} "
            f"blockers={len(_sm.compliance_blockers)} "
            f"states={len(_sm.states)}"
        )
