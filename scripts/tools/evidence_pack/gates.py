"""Twelve-criteria gate evaluator (one pure function per locked criterion).

Each helper returns a GateResult. Source of truth:
``docs/institutional/pre_registered_criteria.md``. DSR is wired as
CROSS_CHECK_ONLY per v2 amendments (N_eff unresolved). Criterion 4
(Chordia t >= 3.79) is wired as severity benchmark, not hard bar, per
the same amendments. Per design doc § 7 + stage criteria 7-8.

These helpers do NOT read files or DBs themselves; they receive parsed
inputs from the orchestrator and emit GateResult records.
"""

from __future__ import annotations

from typing import Any

from .manifest import GateResult


def gate_pre_registered(prereg_path: str | None, prereg_sha: str | None) -> GateResult:
    """Criterion 1 — Pre-registered hypothesis file exists with commit SHA."""

    if not prereg_path:
        return GateResult(
            name="C1_pre_registered",
            status="FAIL",
            value="missing",
            threshold="path resolves",
            source="—",
            note="Prereg path did not resolve.",
        )
    if not prereg_sha:
        return GateResult(
            name="C1_pre_registered",
            status="UNCOMPUTED",
            value=prereg_path,
            threshold="frontmatter commit_sha present",
            source=prereg_path,
            note="Prereg present but no frontmatter commit_sha.",
        )
    return GateResult(
        name="C1_pre_registered",
        status="PASS",
        value=prereg_sha,
        threshold="path resolves AND commit_sha present",
        source=prereg_path,
    )


def gate_minbtl(total_trials: int | None, minbtl_bound: int = 300) -> GateResult:
    """Criterion 2 — MinBTL bound. Pre-reg total_expected_trials <= bound."""

    if total_trials is None:
        return GateResult(
            name="C2_minbtl",
            status="UNCOMPUTED",
            value=None,
            threshold=minbtl_bound,
            source="prereg.total_expected_trials",
            note="Prereg did not declare total_expected_trials.",
        )
    status = "PASS" if total_trials <= minbtl_bound else "FAIL"
    return GateResult(
        name="C2_minbtl",
        status=status,
        value=total_trials,
        threshold=minbtl_bound,
        source="prereg.total_expected_trials",
    )


def gate_bh_fdr(p_value: float | None, q: float = 0.05, k: int | None = None) -> GateResult:
    """Criterion 3 — BH FDR. Survives at q=0.05 against declared K."""

    if p_value is None or k is None:
        return GateResult(
            name="C3_bh_fdr",
            status="UNCOMPUTED",
            value=p_value,
            threshold=q,
            source="result.p_value, prereg.k_global",
            note="Missing p_value or K — cannot evaluate.",
        )
    # BH-FDR rank-1 approximation: pass iff p_value <= q / k
    status = "PASS" if p_value <= q / max(k, 1) else "FAIL"
    return GateResult(
        name="C3_bh_fdr",
        status=status,
        value=p_value,
        threshold=q / max(k, 1),
        source="result.p_value, prereg.k_global",
        note=f"q={q}, K={k}",
    )


def gate_chordia_t(t_stat: float | None, threshold: float = 3.79) -> GateResult:
    """Criterion 4 — Chordia severity benchmark, NOT a hard gate.

    Per pre_registered_criteria.md v2 Amendment 2.2: 'Chordia t-threshold
    reframed as severity benchmark, not hard bar.' Returns CROSS_CHECK_ONLY
    so the verdict is not gated by it. Per stage acceptance criterion #8.
    """

    if t_stat is None:
        return GateResult(
            name="C4_chordia_severity",
            status="UNCOMPUTED",
            value=None,
            threshold=threshold,
            source="result.t_stat",
            note="Severity benchmark only; not a hard gate.",
        )
    return GateResult(
        name="C4_chordia_severity",
        status="CROSS_CHECK_ONLY",
        value=t_stat,
        threshold=threshold,
        source="result.t_stat",
        note="Severity benchmark per Amendment 2.2; not a hard gate.",
    )


def gate_dsr(dsr_value: float | None, threshold: float = 0.95) -> GateResult:
    """Criterion 5 — DSR. CROSS_CHECK_ONLY per v2 Amendment 2.1.

    Per pre_registered_criteria.md: 'DSR downgraded from binding to
    cross-check (N_eff unresolved).' Stage acceptance criterion #7 requires
    DSR gate to never return PASS/FAIL. ALWAYS CROSS_CHECK_ONLY (or
    UNCOMPUTED if missing).
    """

    if dsr_value is None:
        return GateResult(
            name="C5_dsr",
            status="UNCOMPUTED",
            value=None,
            threshold=threshold,
            source="trading_app.dsr.compute_dsr",
            note="N_eff unresolved per pre_registered_criteria.md Amendment 2.1.",
        )
    return GateResult(
        name="C5_dsr",
        status="CROSS_CHECK_ONLY",
        value=dsr_value,
        threshold=threshold,
        source="trading_app.dsr.compute_dsr",
        note="Cross-check only per Amendment 2.1; N_eff unresolved.",
    )


def gate_wfe(wfe: float | None, threshold: float = 0.50) -> GateResult:
    """Criterion 6 — Walk-forward efficiency >= 0.50."""

    if wfe is None:
        return GateResult(
            name="C6_wfe",
            status="UNCOMPUTED",
            value=None,
            threshold=threshold,
            source="result.wfe",
        )
    status = "PASS" if wfe >= threshold else "FAIL"
    return GateResult(
        name="C6_wfe",
        status=status,
        value=wfe,
        threshold=threshold,
        source="result.wfe",
    )


def gate_n_trades(n: int | None, threshold: int = 100) -> GateResult:
    """Criterion 7 — N trades >= 100."""

    if n is None:
        return GateResult(
            name="C7_n_trades",
            status="UNCOMPUTED",
            value=None,
            threshold=threshold,
            source="result.n_trades",
        )
    status = "PASS" if n >= threshold else "FAIL"
    return GateResult(
        name="C7_n_trades",
        status=status,
        value=n,
        threshold=threshold,
        source="result.n_trades",
    )


def gate_oos_2026(oos_status: str | None) -> GateResult:
    """Criterion 8 — 2026 OOS positive (per Amendment 2.7 Mode A).

    Accepts the canonical strategy_validator._evaluate_criterion_8_oos
    status strings: PASSED, NO_OOS_DATA, INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH,
    or any FAIL variant.
    """

    if oos_status is None:
        return GateResult(
            name="C8_oos_2026",
            status="UNCOMPUTED",
            value=None,
            threshold="PASSED",
            source="trading_app.strategy_validator._evaluate_criterion_8_oos",
        )
    canonical_pass = {"PASSED", "NO_OOS_DATA", "INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH"}
    status = "PASS" if oos_status in canonical_pass else "FAIL"
    return GateResult(
        name="C8_oos_2026",
        status=status,
        value=oos_status,
        threshold="PASSED (or pass-through reasons)",
        source="trading_app.strategy_validator._evaluate_criterion_8_oos",
    )


def gate_era_stability(min_era_expr: float | None, n_min: int = 50, threshold: float = -0.05) -> GateResult:
    """Criterion 9 — No era ExpR < -0.05 with N >= 50."""

    if min_era_expr is None:
        return GateResult(
            name="C9_era_stability",
            status="UNCOMPUTED",
            value=None,
            threshold=threshold,
            source="result.min_era_expr",
        )
    status = "PASS" if min_era_expr >= threshold else "FAIL"
    return GateResult(
        name="C9_era_stability",
        status=status,
        value=min_era_expr,
        threshold=threshold,
        source=f"result.min_era_expr (N>={n_min})",
    )


def gate_micro_only(first_day: str | None, micro_launch: str = "2019-05-06") -> GateResult:
    """Criterion 10 — first_trade_day >= MICRO launch (no parent-proxy era)."""

    if first_day is None:
        return GateResult(
            name="C10_micro_only",
            status="UNCOMPUTED",
            value=None,
            threshold=micro_launch,
            source="validated_setups.first_trade_day",
        )
    status = "PASS" if first_day >= micro_launch else "FAIL"
    return GateResult(
        name="C10_micro_only",
        status=status,
        value=first_day,
        threshold=micro_launch,
        source="validated_setups.first_trade_day",
    )


def gate_account_death_mc(survival_pct: float | None, threshold: float = 0.70) -> GateResult:
    """Criterion 11 — Account-death MC survival >= 70%. Profile-level."""

    if survival_pct is None:
        return GateResult(
            name="C11_account_death_mc",
            status="UNCOMPUTED",
            value=None,
            threshold=threshold,
            source="trading_app.account_survival",
            note="Profile-level check; activates after deploy.",
        )
    status = "PASS" if survival_pct >= threshold else "FAIL"
    return GateResult(
        name="C11_account_death_mc",
        status=status,
        value=survival_pct,
        threshold=threshold,
        source="trading_app.account_survival",
    )


def gate_sr_monitor(sr_status: str | None) -> GateResult:
    """Criterion 12 — Shiryaev-Roberts monitor. Auto-activates on deploy."""

    if sr_status is None:
        return GateResult(
            name="C12_sr_monitor",
            status="UNCOMPUTED",
            value=None,
            threshold="CONTINUE",
            source="trading_app.sr_monitor",
            note="Auto-activates on deploy.",
        )
    status = "PASS" if sr_status == "CONTINUE" else "FAIL"
    return GateResult(
        name="C12_sr_monitor",
        status=status,
        value=sr_status,
        threshold="CONTINUE",
        source="trading_app.sr_monitor",
    )


def evaluate_all(inputs: dict[str, Any]) -> tuple[GateResult, ...]:
    """Run all twelve gates in canonical order.

    ``inputs`` is the orchestrator-built dict of parsed values. Missing keys
    yield UNCOMPUTED, never silent PASS.
    """

    return (
        gate_pre_registered(inputs.get("prereg_path"), inputs.get("prereg_sha")),
        gate_minbtl(inputs.get("total_trials")),
        gate_bh_fdr(inputs.get("p_value"), k=inputs.get("k_global")),
        gate_chordia_t(inputs.get("t_stat")),
        gate_dsr(inputs.get("dsr")),
        gate_wfe(inputs.get("wfe")),
        gate_n_trades(inputs.get("n_trades")),
        gate_oos_2026(inputs.get("oos_status")),
        gate_era_stability(inputs.get("min_era_expr")),
        gate_micro_only(inputs.get("first_trade_day")),
        gate_account_death_mc(inputs.get("survival_pct")),
        gate_sr_monitor(inputs.get("sr_status")),
    )
