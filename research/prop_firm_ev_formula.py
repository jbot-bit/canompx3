"""Prop-Firm EV Simulator — Stage 3: the EV combiner + NO_NUMERIC_EV seam.

Read-only, capital-ADJACENT (informs *which prop accounts to buy*), ZERO live /
DB / broker mutation. This is the combiner half of the EV scorecard pre-registered
in ``docs/research/prop_firm_ev_scorecard_2026.md``. Stages 1 + 2 are IMPORTED, not
re-encoded:

- Stage 1 (``research.prop_firm_state_machine``) — per-firm progression/compliance
  gates: ``compliance_blockers``, ``transition_haircut``, ``correlated_tail_note``,
  ``evidence_label`` (the ``MEASURED-CONFLICT`` fee flag).
- Stage 2 (``research.prop_firm_ev_scorecard``) — the survival + reach-payout
  adapter: ``survival_prob`` / ``reach_payout_prob`` with honest ``None``s, run with
  ``write_state=False`` (the zero-mutation seam).

Design contract (committed pre-reg doc + source manifest; the cited master plan
``~/.claude/plans/atomic-herding-pie.md`` is NOT on disk — flagged, not in scope to
fix here):

- Canonical EV formula (doc:33-40): four legs (survive, reach, fee/price,
  transition+compliance haircuts) that can each independently be missing or
  conflicting. The contract rule is absolute (doc:40): "No numeric EV is assigned
  unless official fee, payout, transition, and compliance fields are all present and
  non-conflicting." Under current ground truth EVERY firm is ``NoNumericEv`` — that
  is the correct honest output, encoded as structured fail-closed reasons, never a
  fabricated number or a silent ``EV=0``.
- Result is a TAGGED UNION (``NumericEv | NoNumericEv``) — a single all-Optional
  dataclass would re-create the dead/lying-field anti-pattern (institutional-rigor
  §5). The caller pattern-matches.
- The correlated-tail haircut uses the Frechet-Hoeffding WORST-DEPENDENCE bound on
  the joint ruin probability (no per-path correlation datum exists; fabricating a rho
  would violate institutional-rigor §6). Cite ``carver_2015_ch11_portfolios.md``
  (correlation-aware, never simple addition — doc:46).

RULE 11 (``docs/audit/results/``) governs *discovery* scans that write
``validated_setups``; this is decision-support — no ``orb_outcomes`` enumeration, no
BH-FDR, no DB write — so Phase-0 pre-reg / MinBTL gates do NOT apply (confirmed).
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path

# Make `research` importable when run directly as `python research/...py` (no
# package context). Mirrors research/prop_firm_state_machine.py:37.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.prop_firm_ev_scorecard import (
    DEPLOYED_PROFILE_IDS,
    SurvivalAdapterRecord,
    evaluate_survival_adapter,
)
from research.prop_firm_state_machine import (
    DEFAULT_SOURCES_YAML,
    FirmStateMachine,
    _firm_key,
    load_state_machines,
)

# Sibling of the contract doc + manifest. Decision-support artifact, NOT a
# discovery-scan result under docs/audit/results/ (RULE 11 does not apply).
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_PATH = (
    _REPO_ROOT / "docs" / "research" / "prop_firm_ev_scorecard_2026.results.md"
)

# The fee-conflict sentinel emitted by Stage 1 from the manifest's evidence_label.
_FEE_CONFLICT_LABEL = "MEASURED-CONFLICT"


class NoNumericEvReason(Enum):
    """Minimal complete set of causes a firm cannot receive a numeric EV.

    Collected exhaustively per firm (never short-circuited) so the scorecard is
    actionable — e.g. Tradeify must carry FEE_CONFLICT AND MISSING_REACH_LEG AND
    COMPLIANCE_BLOCKER at once.
    """

    MISSING_SURVIVE_LEG = "survival_prob is None (no AccountProfile for the firm)"
    MISSING_REACH_LEG = "reach_payout_prob is None (no profit_target encoded)"
    FEE_CONFLICT = "fee/price evidence_label is MEASURED-CONFLICT (must not be averaged)"
    MISSING_FEE = "required fee field absent or unparseable"
    MISSING_PAYOUT_CAP = "no capped_payout_capacity resolvable from payout.fields"
    COMPLIANCE_BLOCKER = "FirmStateMachine.compliance_blockers is non-empty"
    UNSUPPORTED_FIELD_BLOCKS = "an unsupported_field gates a required leg"


@dataclass(frozen=True)
class NumericEv:
    """A firm that cleared all four legs — every multiplicand surfaced, not just EV.

    ``ev`` is the combined dollar expected value. Each component is carried so a
    buy-decision reader sees the full decomposition (never just the product). Only
    constructed when survive, reach, fee, and payout-cap are all present and the fee
    label is non-conflicting and no compliance blocker fires.
    """

    profile_id: str
    firm: str
    ev: float
    # Every multiplicand surfaced, never just the product.
    survive_prob: float
    reach_prob: float
    reach_is_coarse: bool
    capped_payout_capacity: float
    n_eff: float
    fees: float
    transition_haircut: float
    compliance_haircut: float
    correlated_tail_haircut: float
    tail_is_bound: bool  # Frechet upper-bound flag (mirrors Stage-2 reach_is_coarse).
    notes: tuple[str, ...]


@dataclass(frozen=True)
class NoNumericEv:
    """A firm that cannot receive a numeric EV — structured, fail-closed reasons.

    ``reasons`` is >=1, ordered, de-duped, and NOT short-circuited (all applicable
    causes collected). ``detail`` carries one human string per reason. The legs that
    WERE resolvable are surfaced (``survive_prob`` / ``reach_prob``) for the report,
    but NO ``ev`` field exists — a missing leg can never be misread as ``EV=0.0``.
    """

    profile_id: str
    firm: str
    reasons: tuple[NoNumericEvReason, ...]
    detail: tuple[str, ...]
    survive_prob: float | None
    reach_prob: float | None


EvResult = NumericEv | NoNumericEv


# ── Payout-cap resolver (prose, not a number) ───────────────────────────────
# The manifest payout.fields caps are free-text English, heterogeneous and
# size-nested (e.g. Topstep "50K $2,000; 100K $3,000; 150K $5,000."). They are NOT
# parsable floats read directly. This resolver mirrors Stage 2's resolve_profit_target
# fail-closed contract: parse for the profile's account size, return None-with-reason
# on genuine absence, RAISE on malformed-but-present, NEVER fabricate a number.

# Which payout.fields key holds the dollar cap, per firm. The size-nested prose
# (Topstep) is parsed for the matching account-size token; flat prose (Tradeify /
# MFFU / Bulenox) yields a single dollar figure.
_PAYOUT_CAP_FIELD = {
    "topstep": "standard_xfa_caps",
    "tradeify": "select_flex_150k_cap",
    "myfundedfutures": "max_request",
    "bulenox": "first_three_caps",
}

# "50K", "100K", "150K" -> 50000, ... — for matching size-nested prose to the
# profile's account_size.
_SIZE_TOKEN_RE = re.compile(r"(\d+)K\b", re.IGNORECASE)
# "$2,000" / "$1,500" / "$5,000." — a dollar amount in the prose.
_DOLLAR_RE = re.compile(r"\$([\d,]+)")


@dataclass(frozen=True)
class PayoutCapResolution:
    """Outcome of resolving a firm's per-request payout cap from prose.

    ``cap`` is the dollar cap for the profile's account size, or ``None`` when the
    prose encodes no cap for that size. ``reason`` explains a ``None``; ``source``
    names the payout.fields key a non-None cap came from.
    """

    cap: float | None
    reason: str | None
    source: str | None


def _parse_dollar(token: str, firm: str, field: str) -> float:
    """Coerce a matched ``$N,NNN`` token to float, raising on malformed data.

    A dollar token present in canonical prose must be a real positive number.
    Anything else is a bug in the manifest, not a missing-data case — fail loud
    (mirrors Stage 2 ``_coerce_target``).
    """
    cleaned = token.replace(",", "")
    try:
        value = float(cleaned)
    except ValueError as exc:  # pragma: no cover — regex guarantees digits; defensive.
        raise ValueError(
            f"payout cap for ({firm}, {field}) is malformed: {token!r}; expected a dollar amount"
        ) from exc
    if value <= 0:
        raise ValueError(f"payout cap for ({firm}, {field}) is non-positive: {value}")
    return value


def resolve_payout_cap(firm: str, account_size: int) -> PayoutCapResolution:
    """Resolve a firm's per-request payout cap from the manifest payout prose.

    Handles BOTH the size-nested shape (Topstep: "50K $2,000; 100K $3,000; ...",
    parsed for the matching size token) and the flat shape (a single dollar figure
    in the field). Returns ``None`` (with a reason) when no cap is encoded for the
    profile's account size — true breadth varies by firm — and RAISES ``ValueError``
    only when a cap is PRESENT but malformed (a real manifest bug that must surface).
    Never defaults a number.
    """
    key = _firm_key(firm)
    machines = load_state_machines()
    sm = machines.get(key)
    if sm is None:
        return PayoutCapResolution(cap=None, reason=f"unknown_firm_{key}", source=None)

    field = _PAYOUT_CAP_FIELD.get(key)
    if field is None:
        return PayoutCapResolution(
            cap=None, reason=f"no_payout_cap_field_registered_for_{key}", source=None
        )

    raw = _firm_payout_field(firm, field)
    if raw is None:
        return PayoutCapResolution(
            cap=None, reason=f"payout_field_{field}_absent", source=None
        )

    size_label = f"{account_size // 1000}K"
    size_tokens = _SIZE_TOKEN_RE.findall(raw)
    dollar_tokens = _DOLLAR_RE.findall(raw)

    # Size-NESTED prose pairs MULTIPLE sizes with MULTIPLE dollars in one string
    # (Topstep: "50K $2,000; 100K $3,000; 150K $5,000."). A single dollar means FLAT
    # prose, even when a lone "NK" token appears as an incidental plan-name suffix
    # (MFFU "...Flex 50K.", Tradeify "150K Select Flex...") — verified against the
    # manifest. Using token-PRESENCE alone mis-classifies those flats and the
    # dollar-must-follow-size guard raises spuriously (the bug this discriminator
    # closes). The honest signal is len>1 on BOTH axes.
    is_nested = len(size_tokens) > 1 and len(dollar_tokens) > 1
    if is_nested:
        # Pair each "NK" token with the dollar amount that follows it; REQUIRE a match
        # for the profile's account size — never fall back to a different size's figure
        # (the Stage-2 by_size trap).
        for m in _SIZE_TOKEN_RE.finditer(raw):
            if m.group(0).upper() == size_label.upper():
                tail = raw[m.end() :]
                dollar = _DOLLAR_RE.search(tail)
                if dollar is None:
                    raise ValueError(
                        f"payout cap for ({firm}, {field}) names size {size_label} "
                        f"but no dollar amount follows it: {raw!r}"
                    )
                return PayoutCapResolution(
                    cap=_parse_dollar(dollar.group(1), firm, field),
                    reason=None,
                    source=f"payout.fields.{field}[{size_label}]",
                )
        # Size-nested prose that does NOT cover this account size — honest None.
        return PayoutCapResolution(
            cap=None,
            reason=f"no_cap_for_size_{size_label}_in_{field}",
            source=None,
        )

    # Flat prose: a single dollar figure. If the prose names a SINGLE size token that
    # does NOT match the profile's account size, the cap is for a different size — do
    # not mis-attach it (Tradeify's 150K cap on a 50K profile). Genuine absence here.
    if len(size_tokens) == 1 and size_tokens[0].upper() != f"{account_size // 1000}":
        return PayoutCapResolution(
            cap=None,
            reason=f"flat_cap_names_size_{size_tokens[0]}K_not_{size_label}_in_{field}",
            source=None,
        )

    dollar = _DOLLAR_RE.search(raw)
    if dollar is None:
        return PayoutCapResolution(
            cap=None, reason=f"no_dollar_amount_in_{field}", source=None
        )
    return PayoutCapResolution(
        cap=_parse_dollar(dollar.group(1), firm, field),
        reason=None,
        source=f"payout.fields.{field}",
    )


def _firm_card(firm: str) -> dict:
    """Read one firm's full scorecard dict from the manifest (read-only). {} if absent."""
    import yaml

    with DEFAULT_SOURCES_YAML.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    want = _firm_key(firm)
    for card in (manifest or {}).get("firm_scorecards") or []:
        if _firm_key(str(card.get("firm", ""))) == want:
            return card if isinstance(card, dict) else {}
    return {}


def _firm_payout_field(firm: str, field: str) -> str | None:
    """Read one payout.fields value from the manifest (read-only). None if absent."""
    fields = (_firm_card(firm).get("payout", {}) or {}).get("fields", {}) or {}
    raw = fields.get(field)
    return str(raw) if raw is not None else None


# ── Fee-leg resolver (the EV "fee/price" leg — purchase price to buy the account) ─
# The fee leg of the EV formula (doc:33-40) is the firm's PURCHASE PRICE. Verified
# against the manifest: every firm captures the official combine/eval purchase price
# only as an ``unsupported_fields`` entry (Topstep "Trading Combine purchase price by
# tier", Tradeify "Dashboard checkout price", MFFU/Bulenox "purchase price"), while
# ``fees.fields`` holds activation/eval/API/data fees. The resolver mirrors the
# payout-cap fail-closed contract: returns a dollar when an evaluation/activation
# purchase fee is MEASURED and non-conflicting, FEE_CONFLICT when the fees
# evidence_label is MEASURED-CONFLICT, MISSING_FEE when no purchase price is encoded,
# and UNSUPPORTED_FIELD_BLOCKS when an unsupported_field explicitly names the price.

# The fees.fields key holding the up-front purchase/evaluation fee, per firm. Topstep
# is intentionally absent — its combine purchase price lives only in unsupported_fields.
_FEE_PRICE_FIELD = {
    "tradeify": None,  # conflict handled via evidence_label; no clean dollar.
    "myfundedfutures": "evaluation_fee",
    "bulenox": "master_activation_fee",
}
# Substrings that, in an unsupported_fields entry, mean the PURCHASE PRICE itself is
# unsupported -> the fee leg is gated (UNSUPPORTED_FIELD_BLOCKS).
_PRICE_UNSUPPORTED_MARKERS = ("purchase price", "checkout price", "combine purchase")


@dataclass(frozen=True)
class FeeResolution:
    """Outcome of resolving a firm's EV fee leg (the purchase price).

    Exactly one of (``fee`` non-None) / (``conflict``) / (``unsupported``) /
    (``missing``) describes the outcome; ``reason`` explains a non-numeric result.
    """

    fee: float | None
    conflict: bool
    unsupported: bool
    reason: str | None


def resolve_fee_leg(firm: str) -> FeeResolution:
    """Resolve the EV fee leg (purchase price) for a firm from the manifest.

    Fail-closed, mirrors ``resolve_payout_cap``: never fabricates a price. A
    MEASURED-CONFLICT fees label is a conflict (not averaged); a purchase price named
    in ``unsupported_fields`` gates the leg; otherwise a clean evaluation/activation
    fee dollar is returned, or MISSING when none is encoded.
    """
    key = _firm_key(firm)
    card = _firm_card(firm)
    fees_block = card.get("fees", {}) or {}

    # Conflict takes precedence — a conflicting price must never be coerced to a number.
    if str(fees_block.get("evidence_label", "")) == _FEE_CONFLICT_LABEL:
        return FeeResolution(
            fee=None, conflict=True, unsupported=False, reason="fees_evidence_label_conflict"
        )

    # Purchase price flagged unsupported anywhere on the card -> the leg is gated.
    unsupported_entries = list(card.get("unsupported_fields", []) or []) + list(
        fees_block.get("unsupported_fields", []) or []
    )
    for entry in unsupported_entries:
        low = str(entry).lower()
        if any(marker in low for marker in _PRICE_UNSUPPORTED_MARKERS):
            return FeeResolution(
                fee=None,
                conflict=False,
                unsupported=True,
                reason=f"purchase_price_unsupported: {entry}",
            )

    field = _FEE_PRICE_FIELD.get(key)
    if field is None:
        return FeeResolution(
            fee=None, conflict=False, unsupported=False, reason=f"no_fee_price_field_for_{key}"
        )
    raw = (fees_block.get("fields", {}) or {}).get(field)
    if raw is None:
        return FeeResolution(
            fee=None, conflict=False, unsupported=False, reason=f"fee_field_{field}_absent"
        )
    dollar = _DOLLAR_RE.search(str(raw))
    if dollar is None:
        return FeeResolution(
            fee=None, conflict=False, unsupported=False, reason=f"no_dollar_amount_in_{field}"
        )
    return FeeResolution(
        fee=_parse_dollar(dollar.group(1), firm, field),
        conflict=False,
        unsupported=False,
        reason=None,
    )


# ── Frechet correlated-tail haircut (fail-closed worst-dependence bound) ─────


@dataclass(frozen=True)
class FrechetTailBound:
    """Frechet-Hoeffding bounds on the joint ruin probability of k firms.

    For marginal survive probabilities ``s_i`` the joint "all survive" probability is
    bounded ``[max(0, sum(s_i) - (k-1)), min(s_i)]`` (worst .. best dependence). The
    joint tail (>= 1 ruin) is ``1 - joint_survive``, so ``tail in [1 - min(s_i),
    1 - max(0, sum(s_i) - (k-1))]`` (least tail .. MOST tail). The fail-closed haircut
    is the MOST-tail / worst-dependence endpoint. The independence point estimate
    ``1 - prod(s_i)`` sits strictly inside the interval (it is not an endpoint).
    """

    haircut: float  # 1 - max(0, sum(s_i) - (k-1)) — worst-dependence (fail-closed).
    tail_least: float  # 1 - min(s_i) — best dependence (least tail).
    tail_independence: float  # 1 - prod(s_i) — independence point estimate.
    tail_most: float  # == haircut (named for report clarity).
    is_bound: bool


def frechet_tail_haircut(survive_probs: tuple[float, ...]) -> FrechetTailBound:
    """Worst-dependence joint-tail haircut for k co-firing firms (Carver Ch11).

    No per-path correlation datum exists (the engine exposes only p05/p50/p95
    marginals), so a fabricated rho is forbidden (institutional-rigor §6). The
    Frechet-Hoeffding bound is the honest, distribution-free fail-closed envelope:
    we charge the MOST-tail endpoint (worst dependence), report the full interval,
    and flag ``is_bound=True``.
    """
    if not survive_probs:
        raise ValueError("frechet_tail_haircut requires >=1 survive probability")
    for s in survive_probs:
        if not (0.0 <= s <= 1.0):
            raise ValueError(f"survive probability out of [0,1]: {s}")

    k = len(survive_probs)
    joint_survive_lower = max(0.0, sum(survive_probs) - (k - 1))  # worst dependence.
    joint_survive_upper = min(survive_probs)  # best dependence.
    prod = math.prod(survive_probs)

    tail_most = 1.0 - joint_survive_lower  # MOST tail — the fail-closed haircut.
    tail_least = 1.0 - joint_survive_upper  # least tail.
    tail_independence = 1.0 - prod  # independence point estimate (interior).

    return FrechetTailBound(
        haircut=tail_most,
        tail_least=tail_least,
        tail_independence=tail_independence,
        tail_most=tail_most,
        is_bound=True,
    )


# ── N_eff capacity ──────────────────────────────────────────────────────────


def compute_n_eff(raw_count: float, transition_haircut: float, *, compliance_blocked: bool) -> float:
    """Compliance-gated effective independent payout capacity.

    Every firm's transition gate collapses capacity to ~1 (Topstep call-up
    non-additive; Tradeify exclusivity; MFFU live-merge; Bulenox consolidation), so
    the count is capped at 1. No correlation datum justifies a fractional discounted
    count, so none is invented. ``transition_haircut`` stays as the live seam for when
    a firm's haircut later drops below 1.0. Compliance-blocked firms collapse to 0.0
    (belt-and-suspenders — they already return NoNumericEv, so n_eff never silently
    yields EV=0).
    """
    if compliance_blocked:
        return 0.0
    return min(raw_count * transition_haircut, 1.0)


# ── The combiner ────────────────────────────────────────────────────────────


def combine_ev(
    sm: FirmStateMachine,
    survival: SurvivalAdapterRecord | None,
    *,
    profile_id: str,
    account_size: int,
) -> EvResult:
    """Combine Stage 1 + Stage 2 into a NumericEv or a fail-closed NoNumericEv.

    Collects ALL applicable NO_NUMERIC_EV reasons (never short-circuits). Only emits a
    NumericEv when survive, reach, fee (non-conflicting), and payout-cap are all
    present and no compliance blocker fires. ``ev`` is computed only on that all-clear
    path — the single arithmetic branch.
    """
    reasons: list[NoNumericEvReason] = []
    detail: list[str] = []

    survive_prob = survival.survival_prob if survival is not None else None
    reach_prob = survival.reach_payout_prob if survival is not None else None
    reach_is_coarse = bool(survival.reach_is_coarse) if survival is not None else False

    # Leg 1 — survive.
    if survive_prob is None:
        reasons.append(NoNumericEvReason.MISSING_SURVIVE_LEG)
        detail.append(
            "survive: no AccountProfile for firm "
            f"{sm.firm!r} -> survival_prob is None"
            if survival is None
            else "survive: survival_prob is None"
        )

    # Leg 2 — reach.
    if reach_prob is None:
        reasons.append(NoNumericEvReason.MISSING_REACH_LEG)
        why = (
            survival.target_reason
            if survival is not None and survival.target_reason
            else "no_profit_target_encoded"
        )
        detail.append(f"reach: reach_payout_prob is None ({why})")

    # Leg 3 — fee / price (the purchase price). Resolved fail-closed: distinguishes
    # conflict, unsupported-gated, and genuinely-missing from a clean dollar.
    fee_res = resolve_fee_leg(sm.firm)
    if fee_res.conflict:
        reasons.append(NoNumericEvReason.FEE_CONFLICT)
        detail.append(
            f"fee: fees.evidence_label is {_FEE_CONFLICT_LABEL} -> conflicting prices "
            "must not be averaged into an EV"
        )
    elif fee_res.unsupported:
        reasons.append(NoNumericEvReason.UNSUPPORTED_FIELD_BLOCKS)
        detail.append(f"fee: {fee_res.reason}")
    elif fee_res.fee is None:
        reasons.append(NoNumericEvReason.MISSING_FEE)
        detail.append(f"fee: {fee_res.reason}")

    # Leg 4 — payout cap (prose resolver).
    cap_res = resolve_payout_cap(sm.firm, account_size)
    if cap_res.cap is None:
        reasons.append(NoNumericEvReason.MISSING_PAYOUT_CAP)
        detail.append(f"payout_cap: {cap_res.reason}")

    # Compliance gate (Stage 1).
    compliance_blocked = bool(sm.compliance_blockers)
    if compliance_blocked:
        reasons.append(NoNumericEvReason.COMPLIANCE_BLOCKER)
        detail.append(
            f"compliance: {len(sm.compliance_blockers)} blocker(s) -> "
            f"{sm.compliance_blockers[0]}"
        )

    if reasons:
        # De-dupe while preserving order (a leg can in principle add a reason twice).
        seen: set[NoNumericEvReason] = set()
        ordered: list[NoNumericEvReason] = []
        for r in reasons:
            if r not in seen:
                seen.add(r)
                ordered.append(r)
        return NoNumericEv(
            profile_id=profile_id,
            firm=sm.firm,
            reasons=tuple(ordered),
            detail=tuple(detail),
            survive_prob=survive_prob,
            reach_prob=reach_prob,
        )

    # ── All-clear arithmetic path (the only branch that computes a number) ──
    # Reaching here means survive/reach/cap/fee are all non-None and fee is
    # non-conflicting and no compliance blocker fired. mypy/readers: the asserts
    # narrow the Optionals for the arithmetic below.
    assert survive_prob is not None and reach_prob is not None and cap_res.cap is not None
    assert fee_res.fee is not None  # MISSING_FEE would have returned NoNumericEv above.

    n_eff = compute_n_eff(1.0, sm.transition_haircut, compliance_blocked=False)
    fees = fee_res.fee
    # The correlated-tail haircut is a CROSS-FIRM (k>=2) phenomenon — the Frechet
    # joint-ruin bound over the surviving firms. For a single firm there is no
    # correlation, and the single-firm ruin (1 - survive_prob) is ALREADY priced into
    # the survive_prob multiplicand below; charging frechet_tail_haircut((survive,)) =
    # 1 - survive here would DOUBLE-COUNT it. So the per-firm correlated-tail haircut
    # is 0.0 — the bound stays dormant until a portfolio of >=2 numeric-eligible firms
    # is scored (current reality: never, every firm is NoNumericEv). is_bound flags
    # that the (dormant) machinery is the Frechet envelope, not a fabricated rho.
    correlated_tail_haircut = 0.0
    tail_is_bound = True
    compliance_haircut = 0.0  # no blocker on this path (would be NoNumericEv otherwise).

    capped_capacity = cap_res.cap * n_eff
    ev = (
        survive_prob * reach_prob * capped_capacity
        - fees
        - compliance_haircut
        - correlated_tail_haircut
    )

    return NumericEv(
        profile_id=profile_id,
        firm=sm.firm,
        ev=ev,
        survive_prob=survive_prob,
        reach_prob=reach_prob,
        reach_is_coarse=reach_is_coarse,
        capped_payout_capacity=capped_capacity,
        n_eff=n_eff,
        fees=fees,
        transition_haircut=sm.transition_haircut,
        compliance_haircut=compliance_haircut,
        correlated_tail_haircut=correlated_tail_haircut,
        tail_is_bound=tail_is_bound,
        notes=(f"payout_cap_source={cap_res.source}",),
    )


def evaluate_firm_ev(
    profile_id: str,
    *,
    as_of_date: date | None = None,
    n_paths: int = 10_000,
    seed: int = 0,
) -> EvResult:
    """Evaluate one deployed profile end-to-end (Stage 1 + Stage 2 -> EV).

    Read-only: Stage 2 is invoked with ``write_state=False`` (zero mutation). MFFU has
    no profile, so it is handled by ``evaluate_all_firm_ev`` from the manifest side.
    """
    survival = evaluate_survival_adapter(
        profile_id, as_of_date=as_of_date, n_paths=n_paths, seed=seed
    )
    machines = load_state_machines()
    sm = machines.get(_firm_key(survival.firm))
    if sm is None:  # pragma: no cover — every deployed firm has a state machine.
        raise ValueError(f"No FirmStateMachine for firm {survival.firm!r}")
    return combine_ev(
        sm, survival, profile_id=profile_id, account_size=survival.account_size
    )


def evaluate_all_firm_ev(
    *,
    as_of_date: date | None = None,
    n_paths: int = 10_000,
    seed: int = 0,
) -> list[EvResult]:
    """Evaluate EVERY firm in the manifest — deployed profiles + manifest-only firms.

    Deployed profiles (Topstep / Tradeify / Bulenox) run the full Stage-2 survival
    adapter. Manifest-only firms with no AccountProfile (MFFU) get a NoNumericEv with
    a None survival record so MISSING_SURVIVE_LEG is surfaced honestly rather than the
    firm being silently dropped from the scorecard.
    """
    machines = load_state_machines()
    deployed_by_firm: dict[str, str] = {}
    results: list[EvResult] = []

    # Map each deployed profile to its firm key (read-only profile lookup).
    from trading_app.prop_profiles import get_profile

    for pid in DEPLOYED_PROFILE_IDS:
        deployed_by_firm[_firm_key(get_profile(pid).firm)] = pid

    for key, sm in machines.items():
        pid = deployed_by_firm.get(key)
        if pid is not None:
            results.append(
                evaluate_firm_ev(pid, as_of_date=as_of_date, n_paths=n_paths, seed=seed)
            )
        else:
            # Manifest-only firm (no AccountProfile, e.g. MFFU). Account size comes
            # from the manifest plan text would be fragile; the survive leg is None
            # regardless, so account_size only matters for the payout-cap resolver.
            account_size = _manifest_account_size(sm)
            results.append(
                combine_ev(
                    sm,
                    None,
                    profile_id=f"{key}_manifest_only",
                    account_size=account_size,
                )
            )
    return results


def _manifest_account_size(sm: FirmStateMachine) -> int:
    """Best-effort account size for a manifest-only firm (for the payout resolver).

    MFFU's reviewed sleeve is Flex 50K. Defaulting to 50000 here only feeds the
    payout-cap resolver; the survive leg is None for manifest-only firms regardless,
    so the firm is NoNumericEv whatever the cap resolves to. Documented, not silent.
    """
    return 50_000


# ── Portfolio correlated-tail bound (cross-firm, k>=2) ──────────────────────


def portfolio_tail_bound(results: list[EvResult]) -> FrechetTailBound | None:
    """Frechet joint-ruin bound across the NUMERIC-eligible firms (k>=2).

    This is the LIVE home of ``frechet_tail_haircut``: the per-firm combiner correctly
    charges 0.0 (single-firm ruin is already in the survive multiplicand), and the
    correlated-tail risk only exists ACROSS a portfolio of >=2 numeric-eligible firms.
    Returns ``None`` when fewer than two firms are numeric-eligible (current reality:
    every firm is NoNumericEv, so this is always None — honest, not a gap). When >=2
    firms clear, it reports the worst-dependence joint-ruin haircut + the full interval
    + the independence point estimate (Carver 2015 Ch11 — never simple addition).
    """
    survive_probs = tuple(r.survive_prob for r in results if isinstance(r, NumericEv))
    if len(survive_probs) < 2:
        return None
    return frechet_tail_haircut(survive_probs)


# ── CLI + report ────────────────────────────────────────────────────────────


def _format_line(result: EvResult) -> str:
    """One per-firm scorecard line."""
    if isinstance(result, NumericEv):
        return (
            f"{result.firm:<16} NUMERIC ev={result.ev:.2f} "
            f"(survive={result.survive_prob:.3f} reach={result.reach_prob:.3f} "
            f"cap={result.capped_payout_capacity:.0f} n_eff={result.n_eff:.2f} "
            f"fees={result.fees:.2f} tail_haircut={result.correlated_tail_haircut:.3f})"
        )
    reasons = ",".join(r.name for r in result.reasons)
    return f"{result.firm:<16} NO_NUMERIC_EV [{reasons}]"


def _build_report(results: list[EvResult], *, n_paths: int, seed: int) -> str:
    """Render the scorecard markdown artifact (decision-support, not a discovery scan)."""
    lines: list[str] = [
        "# Prop-Firm EV Scorecard — Results (Stage 3)",
        "",
        "**Research-only, capital-ADJACENT (informs *which prop accounts to buy*).** "
        "Zero DB / broker / live mutation. Generated by "
        "`research/prop_firm_ev_formula.py`.",
        "",
        f"Monte-Carlo paths: `{n_paths}` · seed: `{seed}`. Survival adapter run with "
        "`write_state=False` (zero-mutation).",
        "",
        "EV formula (contract doc:33-40): four legs (survive · reach · fee/price · "
        "transition+compliance haircuts). **No numeric EV is assigned unless official "
        "fee, payout, transition, and compliance fields are all present and "
        "non-conflicting.** Every NO_NUMERIC_EV below is the correct honest output, "
        "not a defect.",
        "",
        "Correlated-tail haircut: Frechet-Hoeffding worst-dependence bound on joint "
        "ruin (Carver 2015 Ch11 — correlation-aware, never simple addition). Dormant "
        "until two firms are simultaneously numeric-eligible.",
        "",
        "## Per-firm scorecard",
        "",
        "| Firm | Verdict | Survive | Reach | Reasons |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        if isinstance(r, NumericEv):
            lines.append(
                f"| {r.firm} | NUMERIC ev={r.ev:.2f} | {r.survive_prob:.3f} | "
                f"{r.reach_prob:.3f} | — |"
            )
        else:
            survive = f"{r.survive_prob:.3f}" if r.survive_prob is not None else "None"
            reach = f"{r.reach_prob:.3f}" if r.reach_prob is not None else "None"
            reasons = ", ".join(rr.name for rr in r.reasons)
            lines.append(f"| {r.firm} | NO_NUMERIC_EV | {survive} | {reach} | {reasons} |")

    lines.extend(["", "## Reason detail", ""])
    for r in results:
        lines.append(f"### {r.firm}")
        if isinstance(r, NumericEv):
            lines.append(
                f"- NUMERIC ev={r.ev:.2f}; cap={r.capped_payout_capacity:.0f}; "
                f"n_eff={r.n_eff:.2f}; tail_haircut={r.correlated_tail_haircut:.3f}"
            )
        else:
            for d in r.detail:
                lines.append(f"- {d}")
        lines.append("")

    # Portfolio correlated-tail bound (cross-firm, k>=2). Dormant until >=2 firms are
    # numeric-eligible — current reality: every firm is NoNumericEv, so this is None.
    lines.extend(["## Portfolio correlated-tail bound", ""])
    bound = portfolio_tail_bound(results)
    numeric_count = sum(1 for r in results if isinstance(r, NumericEv))
    if bound is None:
        lines.append(
            f"- Dormant: {numeric_count} firm(s) numeric-eligible (<2). The Frechet "
            "joint-ruin bound activates only across a portfolio of two or more "
            "numeric-eligible firms — no fabricated correlation is ever assumed."
        )
    else:
        lines.append(
            f"- joint-ruin tail interval [least={bound.tail_least:.3f}, "
            f"most={bound.tail_most:.3f}]; independence point estimate="
            f"{bound.tail_independence:.3f}; fail-closed haircut (worst dependence)="
            f"{bound.haircut:.3f} (Carver 2015 Ch11)."
        )
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Dry-run by default — prints the scorecard, writes nothing unless asked."""
    parser = argparse.ArgumentParser(
        description="Prop-Firm EV Scorecard (Stage 3 combiner) — research-only, read-only."
    )
    parser.add_argument("--n-paths", type=int, default=10_000, help="Monte-Carlo paths.")
    parser.add_argument("--seed", type=int, default=0, help="Monte-Carlo seed.")
    parser.add_argument(
        "--write-report",
        dest="write_report",
        action="store_true",
        default=False,
        help=f"Write the scorecard to {DEFAULT_REPORT_PATH.name} (default: dry-run print).",
    )
    parser.add_argument(
        "--no-write-report",
        dest="write_report",
        action="store_false",
        help="Explicit dry-run (the default).",
    )
    args = parser.parse_args(argv)

    results = evaluate_all_firm_ev(n_paths=args.n_paths, seed=args.seed)
    for r in results:
        print(_format_line(r))

    report = _build_report(results, n_paths=args.n_paths, seed=args.seed)
    if args.write_report:
        DEFAULT_REPORT_PATH.write_text(report, encoding="utf-8")
        print(f"\nReport written: {DEFAULT_REPORT_PATH}")
    else:
        print("\n(dry-run — report not written; pass --write-report to persist)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
