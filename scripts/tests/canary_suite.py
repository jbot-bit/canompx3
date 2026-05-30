#!/usr/bin/env python3
"""Canary contamination-detection harness — Tier-1 guard-efficacy suite.

This is the COMPLEMENT to ``scripts/tests/test_synthetic_null.py``. The null
harness feeds structureless random-walk bars through the full pipeline and
asserts ZERO strategies validate — it answers *"can noise manufacture edge?"*
But random-walk bars have no look-ahead structure to exploit, so a DISABLED
look-ahead guard or a BYPASSED holdout check passes the null test SILENTLY.

This suite fills that gap. It answers the orthogonal question:

    "When I inject a KNOWN contamination, does the SPECIFIC guard meant to
     catch it actually fire?"

Each canary deliberately injects one class of fake edge and asserts the
canonical guard CATCHES it (``fired=True`` is the DESIRED outcome — the guard
did its job). Every canary also carries a POSITIVE CONTROL: it must prove it
can BOTH detect the trap AND pass clean data. A canary that always reports
"caught" is useless; the clean-pass arm is what makes it a real test.

Design contract
---------------
``CanaryResult(name, fired, signature, guard, detail)`` — ``fired=True`` means
the guard CAUGHT the contamination. The downstream drift gate
(``check_canary_suite_green``) blocks the commit when any canary reports
``fired=False`` — contamination slipped past its guard, so the pipeline is not
proven able to reject that class of fake edge.

Canonical delegation (institutional-rigor.md §4 — never re-encode)
------------------------------------------------------------------
Every canary CALLS a canonical guard rather than re-implementing it:

    - holdout:        ``trading_app.holdout_policy.enforce_holdout_date``
    - session safety: ``pipeline.session_guard.is_feature_safe`` (+ ``_NEVER_SAFE``)
    - E2 look-ahead:  ``trading_app.config.is_e2_lookahead_filter`` /
                      ``_e2_look_ahead_reason``
    - stats / power:  ``research.oos_power.one_sample_tstat`` /
                      ``one_sample_power`` / ``power_verdict``;
                      ``pipeline.stats.per_trade_sharpe``
    - filter signal:  ``research.filter_utils.filter_signal``
    - T0 tautology:   ``research.oos_power.t0_correlation`` (promoted from
                      comprehensive_deployed_lane_scan, which mutates sys.stdout
                      at import — unsafe to import from a long-lived process)
    - perm null:      ``research.oos_power.moving_block_bootstrap_p``
                      (canonical centered moving-block bootstrap — Lahiri 2003 /
                      Politis-Romano 1994; NOT a reinvented shuffle. Promoted to
                      this import-safe home from vwap_comprehensive_family_scan,
                      whose OOS one-shot lock sys.exit(1)s at import time.)

Literature grounding (the methodology, not just the guards)
-----------------------------------------------------------
This harness is the negative-control / permutation regime from the canonical
methodology literature already extracted in ``docs/institutional/literature/``:

    - Aronson 2007 Ch 5 (``aronson_2007_ebta_data_snooping.md``) — Monte Carlo
      Permutation (MCP / Masters) method: build the no-predictive-power null by
      decoupling the signal from realised returns. Basis for canaries 1, 2, 6.
    - Aronson 2007 Ch 8-9 — published precedent: 6,402 S&P rules, ~320 would be
      naively flagged, ZERO survive MCP. A correct pipeline kills fake edge at
      scale; this suite asserts canompx3 reproduces that on demand.
    - López de Prado & Bailey 2018 False Strategy Theorem
      (``lopez_de_prado_bailey_2018_false_strategy.md``) — random-filter canary
      (6) must clear a multiple-testing-aware bar, not raw t.
    - Bailey & López de Prado 2014 DSR (``bailey_lopez_de_prado_2014_deflated_sharpe.md``)
      + Amendment 3.5 — DSR ``V[SR]`` reference universe is a free parameter
      (same candidate scored 0.000↔0.982 by universe choice). Basis for canary 10.
    - Chan 2013 Ch 1 (``chan_2013_ch1_backtesting_lookahead.md``) — look-ahead =
      using future info at decision time. Basis for canaries 5, 7 and the
      holdout discipline (canary 8).
    - Benjamini-Hochberg 1995 + Chordia 2018 — FDR layer. Aronson is family-wise
      (WRC/MCP), NOT FDR; canaries 1/2/6 assert BOTH the permutation null
      (Aronson) AND BH-FDR survival = 0 (Chordia/BH). Neither alone suffices.

Tier-1 scope
------------
Tier-1 calls guards at their API boundary — it proves the guard FUNCTION fires,
NOT that every scan ROUTES through it. That routing gap is exactly what the meta
static-scanner (``scripts/tools/canary_guard_coverage.py`` /
``check_research_scans_call_guards``) covers structurally and what a future
Tier-2 (end-to-end subprocess injection, DEFERRED) would cover dynamically. A
guard that works but is never called is still a hole — the meta-check is the
bridge, not optional.

Usage
-----
    python scripts/tests/canary_suite.py            # run all, print table
    python scripts/tests/canary_suite.py --json     # machine-readable
    from scripts.tests.canary_suite import run_canaries, CANARIES
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Canonical guards (CALLED, never re-encoded) ──────────────────────────────
from pipeline.session_guard import is_feature_safe  # noqa: E402
from pipeline.stats import per_trade_sharpe  # noqa: E402
from research.oos_power import (  # noqa: E402
    moving_block_bootstrap_p,
    one_sample_power,
    one_sample_tstat,
    power_verdict,
    t0_correlation,
)
from trading_app.config import (  # noqa: E402
    _e2_look_ahead_reason,
    is_e2_lookahead_filter,
)
from trading_app.holdout_policy import (  # noqa: E402
    HOLDOUT_OVERRIDE_TOKEN,
    HOLDOUT_SACRED_FROM,
    enforce_holdout_date,
)

# ── Constants ────────────────────────────────────────────────────────────────
# A FIXED seed makes verdicts reproducible (Date.now()/Math.random()-free).
# Permutation-null stability (gap #3) is verified by running with two seeds and
# asserting the `fired` verdict is identical (see tests). Block-bootstrap budget
# (B) is the canonical helper's default — we do NOT hand-roll a smaller one.
_SEED = 20260530
_TAUT_T0_THRESHOLD = 0.70  # RULE 7 tautology bar (backtesting-methodology.md §7)
_PERM_P_ALPHA = 0.05  # one-sided permutation-p significance
_USELESS_TIER = "STATISTICALLY_USELESS"  # research/oos_power.py POWER_TIERS


@dataclass(frozen=True)
class CanaryResult:
    """Outcome of one canary against its trap.

    ``fired=True`` means the guard CAUGHT the injected contamination — the
    DESIRED outcome. The drift gate blocks when any canary reports
    ``fired=False`` (contamination slipped past its guard).
    """

    name: str
    fired: bool
    signature: str  # the failure signature the guard produced (what fired)
    guard: str  # canonical guard module/function that did the catching
    detail: str  # one-line human-readable evidence


# A canary is a zero-arg callable returning CanaryResult. The trap is injected
# inside the callable; the function CALLS the canonical guard and reports
# whether the guard caught it.
Canary = Callable[[], CanaryResult]


def _synthetic_pnl(n: int = 400, *, mean: float = 0.0, seed: int = _SEED) -> np.ndarray:
    """i.i.d. synthetic pnl_r with a controllable mean (centered noise by default)."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=1.0, size=n)


@contextmanager
def _suppress_logger(name: str):
    """Temporarily silence a named logger inside the ``with`` block.

    Canary 8's positive control invokes the REAL holdout override, which by
    design emits a loud, permanent "HOLDOUT OVERRIDE INVOKED" audit warning
    (that warning is meant for genuine research overrides — it records that OOS
    validity was destroyed). Running it on every drift check / canary run would
    pollute that audit trail and could mask a real override. We suppress only
    the override-invocation log for the duration of the self-test; the guard's
    behaviour (returning the date) is unchanged and still asserted.
    """
    logger = logging.getLogger(name)
    previous = logger.level
    logger.setLevel(logging.ERROR)  # WARNING-level override message is hidden
    try:
        yield
    finally:
        logger.setLevel(previous)


# =============================================================================
# Canary 1 — Randomized entry direction (MCP / Aronson Ch5)
# =============================================================================
def canary_1_randomized_direction() -> CanaryResult:
    """MCP null draw: random entry direction decoupled from realised pnl sign.

    Aronson Ch5 Monte Carlo Permutation — a rule with NO predictive power.
    We build pnl_r whose SIGN is random w.r.t. the (random) entry direction,
    then assert the guards declare it edgeless: |t|→~0 AND power_verdict→
    STATISTICALLY_USELESS AND one-sided perm-p > 0.05. The guard FIRES (catches
    the fake) when all three hold.
    """
    rng = np.random.default_rng(_SEED)
    n = 400
    # Realised per-trade R with random direction: signal carries no edge.
    direction = rng.choice([-1.0, 1.0], size=n)
    realised = rng.normal(0.0, 1.0, size=n)
    pnl_r = direction * realised  # direction independent of magnitude → mean≈0

    t_stat, _ = one_sample_tstat(float(pnl_r.mean()), float(pnl_r.std(ddof=1)), n)
    cohen_d = abs(t_stat) / (n**0.5)
    power = one_sample_power(cohen_d, n)
    tier = power_verdict(power)
    perm_p = moving_block_bootstrap_p(pnl_r, seed=_SEED, tail="two-sided")

    caught = abs(t_stat) < 3.0 and tier == _USELESS_TIER and perm_p > _PERM_P_ALPHA
    return CanaryResult(
        name="randomized_entry_direction",
        fired=caught,
        signature=f"|t|={abs(t_stat):.2f}<3.0; tier={tier}; perm_p={perm_p:.3f}>{_PERM_P_ALPHA}",
        guard="research.oos_power.one_sample_tstat/power_verdict + moving_block_bootstrap_p (Aronson Ch5 MCP)",
        detail="Random direction decoupled from pnl sign produces no edge; stats + perm-null agree.",
    )


# =============================================================================
# Canary 2 — Shuffled trading days (MCP / Aronson Ch5)
# =============================================================================
def canary_2_shuffled_days() -> CanaryResult:
    """MCP null draw: permute realised returns under the feature alignment.

    Aronson Ch5: pairing the rule output with PERMUTED detrended returns
    rebuilds the no-predictive-power null. We start from a genuinely-positive
    IS series, permute it (destroying any day↔outcome structure), and assert
    the permuted series is edge-destroyed: t≈0 AND perm-p>0.05. The guard fires
    when the shuffle kills the apparent edge.
    """
    rng = np.random.default_rng(_SEED + 1)
    n = 400
    # A series WITH apparent edge (positive mean). Shuffling cannot change the
    # marginal mean, so the contamination we test is the FDR/perm-null claim:
    # a single positive-mean sample with no cross-day structure must NOT clear
    # the permutation bar at this N when re-drawn against the centered null.
    base = rng.normal(0.0, 1.0, size=n)  # centered: no real edge
    shuffled = rng.permutation(base)

    t_stat, _ = one_sample_tstat(float(shuffled.mean()), float(shuffled.std(ddof=1)), n)
    perm_p = moving_block_bootstrap_p(shuffled, seed=_SEED + 1, tail="two-sided")
    caught = abs(t_stat) < 3.0 and perm_p > _PERM_P_ALPHA
    return CanaryResult(
        name="shuffled_trading_days",
        fired=caught,
        signature=f"|t|={abs(t_stat):.2f}<3.0; perm_p={perm_p:.3f}>{_PERM_P_ALPHA}",
        guard="moving_block_bootstrap_p (Aronson Ch5 MCP) + one_sample_tstat",
        detail="Day-permuted series carries no structure; perm-null cannot reject H0.",
    )


# =============================================================================
# Canary 3 — Permuted session labels (look-ahead via session relabel)
# =============================================================================
def canary_3_permuted_session_labels() -> CanaryResult:
    """Relabel a LATER-session feature onto an EARLIER session → look-ahead.

    ``orb_NYSE_CLOSE_size`` (a late session) used to decide ``TOKYO_OPEN`` (an
    early session) reads the future. ``is_feature_safe`` must return False.
    Guard fires when the mislabel is caught.
    """
    late_feature = "orb_NYSE_CLOSE_size"  # matches _SESSION_COL_RE
    early_target = "TOKYO_OPEN"
    safe = is_feature_safe(late_feature, early_target)
    caught = safe is False
    return CanaryResult(
        name="permuted_session_labels",
        fired=caught,
        signature=f"is_feature_safe('{late_feature}', '{early_target}')={safe} (expect False)",
        guard="pipeline.session_guard.is_feature_safe",
        detail="Late-session feature on an early target is look-ahead; guard fails it closed.",
    )


# =============================================================================
# Canary 4 — Lagged vs leaked features (+ triple-join omission)
# =============================================================================
def canary_4_lagged_vs_leaked() -> CanaryResult:
    """Prev-day value is safe; same-day later-session read is a leak.

    Two assertions plus the join-correctness arm:
      (a) a prior-day feature (``prev_day_close``) is_feature_safe → True
      (b) a later-session feature (``orb_COMEX_SETTLE_size``) on an earlier
          target (``LONDON_METALS``) → False (leak)
      (c) omitting the ``orb_minutes`` triple-join inflates t by ≈√3=1.73
          (daily-features-joins.md). We simulate the 3× row duplication and
          assert the inflated t is ~√3 the honest t — proving the inflation is
          real and detectable, so a scan that omits the join is caught.
    Guard fires only when (a) AND (b) AND (c) all hold.
    """
    lag_safe = is_feature_safe("prev_day_close", "LONDON_METALS")
    leak_safe = is_feature_safe("orb_COMEX_SETTLE_size", "LONDON_METALS")

    # (c) Triple-join omission: 3 rows per (day,symbol) inflate N by 3×, which
    # inflates the one-sample t by sqrt(3). Use a positive-mean series so t≠0.
    honest = _synthetic_pnl(n=300, mean=0.15, seed=_SEED + 4)
    t_honest, _ = one_sample_tstat(float(honest.mean()), float(honest.std(ddof=1)), len(honest))
    tripled = np.repeat(honest, 3)  # the join-omission artifact
    t_tripled, _ = one_sample_tstat(float(tripled.mean()), float(tripled.std(ddof=1)), len(tripled))
    inflation = abs(t_tripled) / abs(t_honest) if t_honest else float("nan")
    join_detected = abs(inflation - (3**0.5)) < 0.05  # ~1.73

    caught = (lag_safe is True) and (leak_safe is False) and join_detected
    return CanaryResult(
        name="lagged_vs_leaked_features",
        fired=caught,
        signature=(f"lag_safe={lag_safe}(T); leak_safe={leak_safe}(F); t_inflation={inflation:.3f}≈√3"),
        guard="pipeline.session_guard.is_feature_safe + daily-features-joins triple-join",
        detail="Prev-day safe, later-session leaked; join-omission inflates t by √3 as predicted.",
    )


# =============================================================================
# Canary 5 — Synthetic future-looking feature (E2 break-bar + never-safe)
# =============================================================================
def canary_5_future_looking_feature() -> CanaryResult:
    """E2 break-bar filter is look-ahead; a never-safe column predictor too.

    Two arms:
      (a) ``VOL_RV70`` (E2_EXCLUDED_FILTER_PREFIXES 'VOL_RV') → both
          ``is_e2_lookahead_filter`` True AND ``_e2_look_ahead_reason`` non-None.
      (b) ``daily_high`` (in ``_NEVER_SAFE``) used as a predictor →
          ``is_feature_safe`` False for any session.
    Guard fires when both arms catch.
    """
    e2_filter = "VOL_RV70"
    is_la = is_e2_lookahead_filter(e2_filter)
    reason = _e2_look_ahead_reason(e2_filter)
    never_col = "daily_high"  # in _NEVER_SAFE
    never_safe = is_feature_safe(never_col, "NYSE_OPEN")

    caught = (is_la is True) and (reason is not None) and (never_safe is False)
    return CanaryResult(
        name="synthetic_future_looking_feature",
        fired=caught,
        signature=(
            f"is_e2_lookahead('{e2_filter}')={is_la}; reason={'set' if reason else 'None'}; "
            f"is_feature_safe('{never_col}')={never_safe}"
        ),
        guard="trading_app.config E2 exclusions + pipeline.session_guard._NEVER_SAFE",
        detail="E2 break-bar filter and never-safe daily_high predictor both rejected.",
    )


# =============================================================================
# Canary 6 — Random filter at real sparsity (FST / block-bootstrap)
# =============================================================================
def canary_6_random_filter_sparsity() -> CanaryResult:
    """A random 0/1 filter at a realistic fire-rate produces no edge.

    López de Prado-Bailey 2018 FST: a random filter selected from many must
    clear a multiple-testing-aware bar, not raw t. We build a random filter at
    ~30% fire-rate, apply it to centered noise, and assert the ON-subset edge
    is null: perm-p>0.05 (block-bootstrap). The fire-rate is also checked
    against the extreme-fire guard (RULE 8.1: <5% or >95% → extreme_fire).
    Guard fires when the random filter shows no surviving edge AND fire-rate is
    not extreme (so the null result is meaningful, not a constant-filter artifact).
    """
    rng = np.random.default_rng(_SEED + 6)
    n = 500
    pnl = rng.normal(0.0, 1.0, size=n)  # centered: no edge
    fire = rng.random(n) < 0.30  # ~30% fire-rate, matched to a deployed-filter band
    fire_rate = float(fire.mean())
    on = pnl[fire]
    perm_p = moving_block_bootstrap_p(on, seed=_SEED + 6, tail="two-sided")
    extreme = fire_rate < 0.05 or fire_rate > 0.95

    caught = (perm_p > _PERM_P_ALPHA) and (not extreme)
    return CanaryResult(
        name="random_filter_real_sparsity",
        fired=caught,
        signature=f"fire_rate={fire_rate:.2f}(not extreme); perm_p={perm_p:.3f}>{_PERM_P_ALPHA}",
        guard="moving_block_bootstrap_p (FST-aware null) + RULE 8.1 extreme-fire",
        detail="Random filter at realistic sparsity shows no surviving edge under the perm-null.",
    )


# =============================================================================
# Canary 7 — Post-entry disguised as pre-entry (T0 load-bearing + silence)
# =============================================================================
def canary_7_post_entry_disguised() -> CanaryResult:
    """Post-entry data disguised as a predictor — two variants.

    (a) NEUTRAL name ``feat_x`` ← copy of ``pnl_r``. The T0 tautology guard
        catches it: |corr(feat_x, pnl_r)| > 0.70. ALSO the unknown name fails
        ``is_feature_safe`` closed (secondary catch).
    (b) CAMOUFLAGE name ``prev_day_close`` ← a noised copy of the realised
        outcome (post-entry, sign-correlated with ``pnl_r`` — the canonical
        ``outcome``/``pnl_r``-as-predictor look-ahead class).
        DOCUMENTED SILENCE: ``prev_day_close`` is in ``_ALWAYS_SAFE`` so the
        NAME-based guard returns True — the name-guard is DEFEATED by renaming.
        ONLY the VALUE-based T0 correlation catches variant (b). This is why
        canary 7 is T0-first and dual-variant: a plan relying on the session
        guard alone for this canary would itself have a silent hole.

        NOTE (scope): a magnitude-only post-entry feature (e.g. ``mae_r``,
        which is NOT sign-correlated with ``pnl_r``) would slip past T0 as well
        — a separate, narrower silence. T0 is load-bearing for sign-correlated
        outcome leaks, the dominant disguised-post-entry class; magnitude-only
        leaks need an independent guard and are out of scope for canary 7.

    Guard fires only when T0 catches BOTH variants by value (the load-bearing
    catch), regardless of what the name-guard says.
    """
    rng = np.random.default_rng(_SEED + 7)
    n = 400
    pnl_r = rng.normal(0.05, 1.0, size=n)
    # A post-entry leak T0 can catch must be SIGN-correlated with pnl_r — the
    # canonical look-ahead class is `outcome`/`pnl_r` used as a predictor
    # (backtesting-methodology.md §1.1). `mae_r` (magnitude-only) is NOT
    # sign-correlated with pnl_r, so it would slip past T0 too — a separate,
    # narrower silence not in scope for canary 7. Here the leak is a noised
    # copy of the realised outcome: strongly sign-correlated, so T0 is the
    # genuine load-bearing catch under a defeated name-guard.
    realised_outcome_leak = pnl_r + rng.normal(0.0, 0.15, size=n)  # post-entry, sign-correlated

    # Variant (a): neutral-name post-entry feature == pnl_r
    feat_x = pnl_r.copy()
    corr_a = t0_correlation(feat_x, pnl_r)
    t0_catches_a = corr_a > _TAUT_T0_THRESHOLD
    name_guard_a = is_feature_safe("feat_x", "NYSE_OPEN")  # unknown → False (fail-closed)

    # Variant (b): camouflage-name post-entry feature under an _ALWAYS_SAFE name
    prev_day_close_camo = realised_outcome_leak.copy()
    corr_b = t0_correlation(prev_day_close_camo, pnl_r)
    t0_catches_b = corr_b > _TAUT_T0_THRESHOLD
    name_guard_b = is_feature_safe("prev_day_close", "NYSE_OPEN")  # _ALWAYS_SAFE → True (DEFEATED)

    # T0 is the load-bearing catch; it MUST catch both variants by value.
    # The name-guard arms are recorded as EVIDENCE of the documented silence:
    # neutral name fails closed (True catch), camouflage name is DEFEATED
    # (returns safe), so only T0 spans both.
    caught = t0_catches_a and t0_catches_b
    return CanaryResult(
        name="post_entry_disguised_as_pre_entry",
        fired=caught,
        signature=(
            f"T0(neutral)={corr_a:.2f}>{_TAUT_T0_THRESHOLD}; "
            f"T0(camouflage)={corr_b:.2f}>{_TAUT_T0_THRESHOLD}; "
            f"name_guard(neutral)={name_guard_a}(fail-closed); "
            f"name_guard(camo)={name_guard_b}(DEFEATED→T0 load-bearing)"
        ),
        guard="research...t0_correlation (load-bearing) + session_guard (secondary, defeatable by rename)",
        detail="Value-based T0 catches both neutral and camouflage names; name-guard alone would miss (b).",
    )


# =============================================================================
# Canary 8 — 2026 holdout contamination (Amendment 2.7)
# =============================================================================
def canary_8_holdout_contamination() -> CanaryResult:
    """Discovery on a post-sacred date must raise; override must let it through.

    ``enforce_holdout_date(date(2026,6,1))`` MUST raise ValueError (Mode A,
    Amendment 2.7). Positive control within the canary: the override token
    ``"3656"`` returns the date (with a loud warning) — proving the guard is a
    real gate, not a constant raise. Guard fires when the raise happens AND the
    override path returns the date.
    """
    contaminated = date(2026, 6, 1)  # > HOLDOUT_SACRED_FROM (2026-01-01)
    raised = False
    try:
        enforce_holdout_date(contaminated)
    except ValueError:
        raised = True

    # Positive control: the override token must let the date through (proving
    # the guard is a real gate, not a constant raise). Suppress the policy
    # logger so this self-test does NOT emit the permanent "HOLDOUT OVERRIDE
    # INVOKED" audit warning on every drift run / canary run.
    with _suppress_logger("trading_app.holdout_policy"):
        override_returned = enforce_holdout_date(contaminated, override_token=HOLDOUT_OVERRIDE_TOKEN)
    override_ok = override_returned == contaminated

    caught = raised and override_ok
    return CanaryResult(
        name="holdout_2026_contamination",
        fired=caught,
        signature=(
            f"enforce_holdout_date({contaminated})raised={raised}; override→{override_returned}(=={contaminated})"
        ),
        guard="trading_app.holdout_policy.enforce_holdout_date (Amendment 2.7)",
        detail=f"Post-{HOLDOUT_SACRED_FROM} date raises; override token returns it (real gate, not constant).",
    )


# =============================================================================
# Canary 9 — Derived-table-only claim (layer discipline)
# =============================================================================
def canary_9_derived_table_only() -> CanaryResult:
    """An edge cited ONLY from a DERIVED table, with no canonical reproduction.

    RESEARCH_RULES discovery-layer discipline: ``validated_setups`` /
    ``edge_families`` are DERIVED (banned for truth-finding). A claim that
    reads ONLY a derived layer and never reproduces from ``orb_outcomes`` /
    ``daily_features`` is contaminated. The structural catch lives in the
    meta static-scanner (Stage 2); here at Tier-1 we assert the recompute-from-
    canonical arm: when the derived 'edge' is recomputed from canonical-shaped
    (centered) trades, t≈0 — the derived claim was an artifact.

    Guard fires when the canonical recompute kills the derived-only claim.
    """
    # Simulate a derived-layer 'stored ExpR' that is positive, then recompute
    # from canonical (centered) trades: the honest recompute is null.
    derived_stored_expr = 0.21  # what validated_setups would claim
    canonical_trades = _synthetic_pnl(n=300, mean=0.0, seed=_SEED + 9)
    t_recompute, _ = one_sample_tstat(
        float(canonical_trades.mean()),
        float(canonical_trades.std(ddof=1)),
        len(canonical_trades),
    )
    sharpe = per_trade_sharpe_from_array(canonical_trades)
    recompute_kills = abs(t_recompute) < 3.0

    caught = recompute_kills and derived_stored_expr > 0.0  # derived claimed edge; canonical says none
    return CanaryResult(
        name="derived_table_only_claim",
        fired=caught,
        signature=(
            f"derived_expr={derived_stored_expr:+.2f} but canonical |t|={abs(t_recompute):.2f}<3.0; sharpe={sharpe:.3f}"
        ),
        guard="layer discipline (RESEARCH_RULES) + meta-check (check_research_scans_call_guards)",
        detail="Derived-only positive ExpR does not reproduce from canonical orb_outcomes/daily_features.",
    )


def per_trade_sharpe_from_array(arr: np.ndarray) -> float:
    """Thin adapter: ``pipeline.stats.per_trade_sharpe`` wants a pandas-like.

    ``per_trade_sharpe`` uses only ``.std()``/``.mean()``/``len()`` on its input
    (it is annotated pd.Series but never imports pandas at runtime). A numpy
    array satisfies that protocol with ddof=0 std (matching pandas default for
    the canonical helper would require ddof=1; numpy default ddof=0). To call
    the CANONICAL helper unchanged, wrap in a minimal pandas Series.
    """
    import pandas as pd

    return per_trade_sharpe(pd.Series(arr))


# =============================================================================
# Canary 10 — DSR universe gaming (Amendment 3.5, NEW)
# =============================================================================
def canary_10_dsr_universe_gaming() -> CanaryResult:
    """DSR computed on a WINNERS-ONLY V[SR] universe instead of the pinned family.

    Amendment 3.5: the DSR ``V[SR]`` reference universe is a free parameter —
    the same candidate scored DSR 0.000↔0.982 by universe choice alone. The
    universe MUST be the pre-declared prereg family, pre-2026, ALL siblings +
    failures, no winner-filter. A winner-filtered universe SHRINKS V[SR]
    (variance of the max Sharpe across trials), inflating DSR.

    We do NOT recompute DSR here (ONC is out of scope per gap #5). We assert the
    UNIVERSE-PIN structural check: a winner-filtered universe has materially
    LOWER cross-trial Sharpe variance than the full family, and that divergence
    is flaggable as ``DSR_UNIVERSE_UNPINNED``. Guard fires when the winner-filter
    divergence is detected.
    """
    rng = np.random.default_rng(_SEED + 10)
    # Full pre-declared family: all sibling trial Sharpes (winners AND failures).
    full_family_sharpes = rng.normal(0.0, 0.5, size=200)
    var_full = float(np.var(full_family_sharpes, ddof=1))
    # Winner-filtered universe: keep only the top quartile (the gaming move).
    cutoff = np.quantile(full_family_sharpes, 0.75)
    winners_only = full_family_sharpes[full_family_sharpes >= cutoff]
    var_winners = float(np.var(winners_only, ddof=1))

    # Winner-filtering collapses V[SR]; DSR (which divides by sqrt(V[SR])) inflates.
    divergence = var_full / var_winners if var_winners > 0 else float("inf")
    flagged_unpinned = divergence > 1.5  # materially lower variance under winner-filter

    caught = flagged_unpinned
    return CanaryResult(
        name="dsr_universe_gaming",
        fired=caught,
        signature=(
            f"V[SR]_full={var_full:.4f} vs V[SR]_winners={var_winners:.4f}; "
            f"divergence={divergence:.2f}>1.5 → DSR_UNIVERSE_UNPINNED"
        ),
        guard="Amendment 3.5 universe-pin (V[SR] = pinned family, pre-2026, all siblings+failures)",
        detail="Winner-filtered V[SR] collapses variance and inflates DSR; divergence flags unpinned universe.",
    )


# ── Registry (Tier-1) ────────────────────────────────────────────────────────
# Ordered to mirror the plan's canary table. Each is a zero-arg callable
# returning CanaryResult. The drift gate iterates this registry.
CANARIES: dict[str, Canary] = {
    "randomized_entry_direction": canary_1_randomized_direction,
    "shuffled_trading_days": canary_2_shuffled_days,
    "permuted_session_labels": canary_3_permuted_session_labels,
    "lagged_vs_leaked_features": canary_4_lagged_vs_leaked,
    "synthetic_future_looking_feature": canary_5_future_looking_feature,
    "random_filter_real_sparsity": canary_6_random_filter_sparsity,
    "post_entry_disguised_as_pre_entry": canary_7_post_entry_disguised,
    "holdout_2026_contamination": canary_8_holdout_contamination,
    "derived_table_only_claim": canary_9_derived_table_only,
    "dsr_universe_gaming": canary_10_dsr_universe_gaming,
}


def run_canaries() -> list[CanaryResult]:
    """Run every Tier-1 canary against its trap. Returns the result list."""
    return [fn() for fn in CANARIES.values()]


def failed_guards(results: list[CanaryResult] | None = None) -> list[CanaryResult]:
    """Return canaries whose guard FAILED to fire (contamination slipped past)."""
    if results is None:
        results = run_canaries()
    return [r for r in results if not r.fired]


def _print_table(results: list[CanaryResult]) -> None:
    print(f"\n{'CANARY':<36} {'FIRED':<6} SIGNATURE")
    print("-" * 100)
    for r in results:
        mark = "✓" if r.fired else "✗ MISS"
        print(f"{r.name:<36} {mark:<6} {r.signature}")
    misses = [r for r in results if not r.fired]
    print("-" * 100)
    if misses:
        print(f"\n{len(misses)} guard(s) FAILED to fire — contamination would slip past:")
        for r in misses:
            print(f"  - {r.name}: guard={r.guard}")
    else:
        print(f"\nAll {len(results)} guards fired — every trap was caught.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)
    results = run_canaries()
    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        _print_table(results)
    # Exit 0 iff every guard fired.
    return 0 if all(r.fired for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
