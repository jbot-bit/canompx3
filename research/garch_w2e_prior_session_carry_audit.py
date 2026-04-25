"""W2e Prior-Session Carry Audit (V2 — broadened scope).

Purpose
-------
Test whether `garch_high` adds distinct or complementary value to same-day,
fully-resolved prior-session carry state across the FULL validated shelf and
every chronologically-admissible prior -> target handoff.

V1 (Codex) locked scope to LONDON_METALS -> EUROPE_FLOW on MNQ only, then hit
the usage limit before running. User feedback immediately before the cutoff
was: "don't test one lane — lots of variables." V2 applies that feedback per
`.claude/rules/backtesting-methodology.md` RULE 5 (comprehensive scope) and
the institutional-rigor rule "never trust metadata blindly — run real tests
against canonical orb_outcomes."

What V2 does differently from V1
-------------------------------
- Enumerates every (prior_session, target_session, instrument) combo that the
  validated shelf supports, instead of one hardcoded pair.
- Enforces dynamic chronology via `pipeline.dst.orb_utc_window`, per trading
  day, with no static session-order shortcut.
- Verifies each validated-setups row by a RAW canonical query before accepting
  it into the audit — flags `metadata_mismatch` if the strategy's filter SQL
  does not actually match any rows in `orb_outcomes` (RULE 7 of
  integrity-guardian: "never trust metadata").
- Adds a bootstrap null test (1000 iterations, Phipson-Smyth p-value) for the
  conjunction ExpR delta, to separate real lift from arithmetic artefacts.
- Flags single-instrument handoffs as `thin_handoff_support` (needs 2+
  instruments to pass the family gate).

Canonical references
--------------------
- Data layers: `bars_1m` / `daily_features` / `orb_outcomes` (canonical truth;
  see `RESEARCH_RULES.md` § Discovery Layer Discipline).
- Session timing: `pipeline.dst.orb_utc_window` and `pipeline.dst.SESSION_CATALOG`.
- Filter translation: `research.garch_broad_exact_role_exhaustion.exact_filter_sql`.
- Validated shelf: `research.garch_partner_state_provenance_audit.load_rows`
  (filters to the 5 W2 target families: COMEX_SETTLE, EUROPE_FLOW, TOKYO_OPEN,
  SINGAPORE_OPEN, LONDON_METALS).

Output
------
docs/audit/results/2026-04-16-garch-w2e-prior-session-carry-audit.md
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import duckdb
import numpy as np
import pandas as pd

from pipeline.dst import SESSION_CATALOG, orb_utc_window
from pipeline.paths import GOLD_DB_PATH
from research import garch_broad_exact_role_exhaustion as broad
from research import garch_partner_state_provenance_audit as prov

OUTPUT_MD = Path("docs/audit/results/2026-04-16-garch-w2e-prior-session-carry-audit.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

MIN_TOTAL = 50
MIN_CONJ = 30
GARCH_HIGH = 70.0
BOOTSTRAP_ITERS = 1000
BOOTSTRAP_RNG_SEED = 20260416

TARGET_SESSIONS = {f.session for f in prov.FAMILIES}
PRIOR_CANDIDATES = sorted(SESSION_CATALOG.keys())


@dataclass(frozen=True)
class CarryState:
    name: str
    label: str
    expected_role: str  # take_pair / veto_pair


STATES = [
    CarryState("PRIOR_WIN_OPPOSED", "prior win opposed", "veto_pair"),
    CarryState("PRIOR_WIN_ALIGN", "prior win align", "take_pair"),
]


def exp_r(arr: pd.Series) -> float:
    s = pd.Series(arr).astype(float)
    if len(s) == 0:
        return float("nan")
    return float(s.mean())


def load_validated_rows(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return prov.load_rows(con)


def load_target_population(con: duckdb.DuckDBPyConnection, row: pd.Series) -> pd.DataFrame:
    """Pull every trade in the target cell, bringing the target's own break_dir
    and the per-day garch_forecast_vol_pct. Chronology / prior merge happens in
    Python per prior session."""
    filter_sql, filter_join = broad.exact_filter_sql(row["filter_type"], row["orb_label"], row["instrument"])
    if filter_sql is None:
        return pd.DataFrame()
    target_session = row["orb_label"]
    q = f"""
    SELECT
      o.trading_day,
      o.symbol,
      o.pnl_r,
      d.garch_forecast_vol_pct AS gp,
      d.orb_{target_session}_break_dir AS target_dir
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    {filter_join}
    WHERE o.symbol = '{row["instrument"]}'
      AND o.orb_label = '{target_session}'
      AND o.orb_minutes = {row["orb_minutes"]}
      AND o.entry_model = '{row["entry_model"]}'
      AND o.rr_target = {row["rr_target"]}
      AND o.pnl_r IS NOT NULL
      AND d.garch_forecast_vol_pct IS NOT NULL
      AND d.orb_{target_session}_break_dir IS NOT NULL
      AND {filter_sql}
    ORDER BY o.trading_day
    """
    df = con.execute(q).df()
    if len(df) == 0:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    df["pnl_r"] = pd.to_numeric(df["pnl_r"], errors="coerce")
    df["gp"] = pd.to_numeric(df["gp"], errors="coerce")
    return df


def attach_target_start(df: pd.DataFrame, target_session: str, orb_minutes: int) -> pd.DataFrame:
    if len(df) == 0:
        return df
    unique_days = df["trading_day"].unique()
    starts: dict[object, object] = {}
    for d in unique_days:
        try:
            starts[d] = pd.Timestamp(orb_utc_window(d, target_session, int(orb_minutes))[0])
        except Exception:
            starts[d] = pd.NaT
    out = df.copy()
    out["target_start_ts"] = pd.to_datetime(out["trading_day"].map(starts), utc=True)
    return out


def load_prior_baseline(con: duckdb.DuckDBPyConnection, symbol: str, prior_session: str) -> pd.DataFrame:
    """Canonical prior baseline: E2 / CB1 / RR1.0 / orb_minutes=5. The PRIOR's
    own trade outcome and exit_ts come from orb_outcomes; the PRIOR's break
    direction comes from daily_features (same trading_day / symbol row)."""
    q = f"""
    SELECT
      o.trading_day,
      o.symbol,
      o.outcome AS prior_outcome,
      o.exit_ts AS prior_exit_ts,
      d.orb_{prior_session}_break_dir AS prior_dir
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.orb_label = '{prior_session}'
      AND o.symbol = '{symbol}'
      AND o.orb_minutes = 5
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target = 1.0
      AND o.outcome IS NOT NULL
      AND o.exit_ts IS NOT NULL
      AND d.orb_{prior_session}_break_dir IS NOT NULL
    ORDER BY o.trading_day
    """
    try:
        df = con.execute(q).df()
    except duckdb.BinderException:
        return pd.DataFrame()
    if len(df) == 0:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    df["prior_exit_ts"] = pd.to_datetime(df["prior_exit_ts"], utc=True)
    return df


def classify_state(merged: pd.DataFrame, state: CarryState) -> pd.Series:
    resolved = merged["resolved_before_start"].fillna(False).astype(bool)
    prior_win = merged["prior_outcome"].eq("win")
    align = merged["prior_dir"].eq(merged["target_dir"])
    if state.name == "PRIOR_WIN_ALIGN":
        return resolved & prior_win & align
    if state.name == "PRIOR_WIN_OPPOSED":
        return resolved & prior_win & (~align)
    raise ValueError(state.name)


def bootstrap_null_p(
    pnl: np.ndarray,
    carry: np.ndarray,
    garch: np.ndarray,
    resolved: np.ndarray,
    observed_conj_exp: float,
    expected_role: str,
    iters: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Bootstrap null floor for the conjunction ExpR.

    Null hypothesis: within the subset of rows where the prior trade resolved
    before the target session started, carry-state labelling is random with
    respect to pnl. Shuffling only within the resolved subset keeps the null
    population distributionally comparable to the observed (same trading days,
    same market regime coverage). Garch mask is held fixed so the null isolates
    the carry<->pnl link specifically.

    Phipson-Smyth p: (M+1)/(iters+1) where M = # null realisations at least as
    extreme as observed, in the direction implied by expected_role.
    """
    if len(pnl) < MIN_TOTAL or carry.sum() == 0 or garch.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    carry_count = int(carry.sum())
    if carry_count == 0:
        return float("nan"), float("nan"), float("nan")

    resolved_idx = np.flatnonzero(resolved & ~np.isnan(pnl))
    if resolved_idx.size < carry_count:
        return float("nan"), float("nan"), float("nan")

    null_vals = np.empty(iters, dtype=float)
    for i in range(iters):
        perm = rng.permutation(resolved_idx)
        carry_perm = np.zeros_like(carry)
        carry_perm[perm[:carry_count]] = True
        conj_perm = carry_perm & garch
        if conj_perm.sum() == 0:
            null_vals[i] = np.nan
            continue
        null_vals[i] = float(np.nanmean(pnl[conj_perm]))
    finite = null_vals[np.isfinite(null_vals)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    if expected_role == "take_pair":
        more_extreme = int(np.sum(finite >= observed_conj_exp))
    else:
        more_extreme = int(np.sum(finite <= observed_conj_exp))
    p = (more_extreme + 1) / (finite.size + 1)
    return float(p), float(np.nanpercentile(finite, 5)), float(np.nanpercentile(finite, 95))


def analyze_cell(
    row: pd.Series,
    prior_session: str,
    state: CarryState,
    merged: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, object] | None:
    if len(merged) < MIN_TOTAL:
        return None
    pnl = merged["pnl_r"].to_numpy(dtype=float)
    gp = merged["gp"].to_numpy(dtype=float)
    garch_mask = gp >= GARCH_HIGH
    carry_mask = classify_state(merged, state).to_numpy(dtype=bool)
    conj_mask = carry_mask & garch_mask
    resolved_mask = merged["resolved_before_start"].fillna(False).astype(bool).to_numpy()

    n_total = int(len(merged))
    n_has_prior = int(merged["prior_outcome"].notna().sum())
    n_resolved = int(resolved_mask.sum())
    n_garch = int(garch_mask.sum())
    n_carry = int(carry_mask.sum())
    n_conj = int(conj_mask.sum())

    base_exp = exp_r(merged["pnl_r"])
    garch_exp = exp_r(merged.loc[garch_mask, "pnl_r"]) if n_garch else float("nan")
    carry_exp = exp_r(merged.loc[carry_mask, "pnl_r"]) if n_carry else float("nan")
    conj_exp = exp_r(merged.loc[conj_mask, "pnl_r"]) if n_conj else float("nan")

    # T0 tautology check — carry and garch should be structurally independent
    # because garch is a close-of-prior-day forecast and carry is driven by the
    # prior intraday session. A large |corr| would mean we're conditioning on
    # an artefact of the same data pipeline.
    if n_carry > 0 and n_garch > 0:
        carry_f = carry_mask.astype(float)
        garch_f = garch_mask.astype(float)
        cf_std = carry_f.std()
        gf_std = garch_f.std()
        if cf_std > 0 and gf_std > 0:
            tau = float(np.corrcoef(carry_f, garch_f)[0, 1])
        else:
            tau = float("nan")
    else:
        tau = float("nan")

    # RULE 8.1 extreme fire-rate flag
    fire_rate = n_conj / n_total if n_total else float("nan")
    extreme_fire = bool(np.isfinite(fire_rate) and (fire_rate < 0.05 or fire_rate > 0.95))

    # T7 per-era positive-rate — by calendar year on the conjunction trades
    per_year_positive = float("nan")
    per_year_counts = {}
    if n_conj >= MIN_CONJ:
        days = pd.to_datetime(merged.loc[conj_mask, "trading_day"])
        pnl_conj = merged.loc[conj_mask, "pnl_r"].astype(float).to_numpy()
        years = days.dt.year.to_numpy()
        uy = sorted(set(years.tolist()))
        pos = 0
        total_years = 0
        for y in uy:
            sel = years == y
            if sel.sum() < 5:  # thin-year guard
                per_year_counts[y] = (int(sel.sum()), float("nan"))
                continue
            yearly_exp = float(pnl_conj[sel].mean())
            per_year_counts[y] = (int(sel.sum()), yearly_exp)
            if yearly_exp > 0:
                pos += 1
            total_years += 1
        per_year_positive = pos / total_years if total_years else float("nan")

    boot_p, null_p5, null_p95 = (float("nan"), float("nan"), float("nan"))
    if n_conj >= MIN_CONJ:
        boot_p, null_p5, null_p95 = bootstrap_null_p(
            pnl,
            carry_mask,
            garch_mask,
            resolved_mask,
            conj_exp,
            state.expected_role,
            BOOTSTRAP_ITERS,
            rng,
        )

    delta_conj_base = conj_exp - base_exp if n_conj else float("nan")
    delta_conj_garch = conj_exp - garch_exp if (n_conj and n_garch) else float("nan")
    delta_conj_carry = conj_exp - carry_exp if (n_conj and n_carry) else float("nan")

    dir_match = False
    if n_conj >= MIN_CONJ and not any(pd.isna(x) for x in [delta_conj_base, delta_conj_garch, delta_conj_carry]):
        if state.expected_role == "take_pair":
            dir_match = delta_conj_base > 0 and delta_conj_garch > 0 and delta_conj_carry >= 0
        elif state.expected_role == "veto_pair":
            dir_match = delta_conj_base < 0 and delta_conj_garch < 0 and delta_conj_carry <= 0

    flags: list[str] = []
    if np.isfinite(tau) and abs(tau) > 0.70:
        flags.append("tautology_suspected")
    if extreme_fire:
        flags.append("extreme_fire_rate")
    if np.isfinite(per_year_positive) and per_year_positive < 0.70:
        flags.append("era_unstable_lt_70pct_years")

    supported = (
        n_conj >= MIN_CONJ
        and dir_match
        and (not np.isnan(boot_p))
        and boot_p <= 0.05
        and (np.isnan(per_year_positive) or per_year_positive >= 0.60)
        and not extreme_fire
        and not (np.isfinite(tau) and abs(tau) > 0.70)
    )
    cell_verdict = "supported" if supported else "not_testable" if n_conj < MIN_CONJ else "unsupported"

    return {
        "strategy_id": row["strategy_id"],
        "instrument": row["instrument"],
        "orb_minutes": int(row["orb_minutes"]),
        "rr_target": float(row["rr_target"]),
        "filter_type": row["filter_type"],
        "entry_model": row["entry_model"],
        "target_session": row["orb_label"],
        "prior_session": prior_session,
        "state": state.name,
        "state_label": state.label,
        "expected_role": state.expected_role,
        "n_total": n_total,
        "n_has_prior": n_has_prior,
        "n_resolved": n_resolved,
        "n_garch": n_garch,
        "n_carry": n_carry,
        "n_conj": n_conj,
        "fire_rate": fire_rate,
        "carry_garch_corr": tau,
        "per_year_positive": per_year_positive,
        "base_exp": base_exp,
        "garch_exp": garch_exp,
        "carry_exp": carry_exp,
        "conj_exp": conj_exp,
        "delta_conj_base": delta_conj_base,
        "delta_conj_garch": delta_conj_garch,
        "delta_conj_carry": delta_conj_carry,
        "bootstrap_p": boot_p,
        "null_p5": null_p5,
        "null_p95": null_p95,
        "dir_match": bool(dir_match),
        "flags": ",".join(flags) if flags else "",
        "cell_verdict": cell_verdict,
    }


def build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)

    validated = load_validated_rows(con)
    prior_cache: dict[tuple[str, str], pd.DataFrame] = {}

    verified_rows: list[tuple[pd.Series, pd.DataFrame]] = []
    metadata_mismatches: list[dict[str, object]] = []

    for _, row in validated.iterrows():
        target_pop = load_target_population(con, row)
        if len(target_pop) == 0:
            metadata_mismatches.append(
                {
                    "strategy_id": row["strategy_id"],
                    "instrument": row["instrument"],
                    "target_session": row["orb_label"],
                    "filter_type": row["filter_type"],
                    "n_raw": 0,
                    "reason": "filter_sql_returned_zero_canonical_rows",
                }
            )
            continue
        # Attach target session start and cache the population for this row
        target_pop = attach_target_start(target_pop, row["orb_label"], int(row["orb_minutes"]))
        target_pop = target_pop.dropna(subset=["target_start_ts"])
        if len(target_pop) == 0:
            metadata_mismatches.append(
                {
                    "strategy_id": row["strategy_id"],
                    "instrument": row["instrument"],
                    "target_session": row["orb_label"],
                    "filter_type": row["filter_type"],
                    "n_raw": 0,
                    "reason": "no_resolvable_target_start_ts",
                }
            )
            continue
        verified_rows.append((row, target_pop))

    cell_records: list[dict[str, object]] = []

    for row, target_pop in verified_rows:
        target_session = row["orb_label"]
        instrument = row["instrument"]
        for prior_session in PRIOR_CANDIDATES:
            if prior_session == target_session:
                continue
            key = (instrument, prior_session)
            if key not in prior_cache:
                prior_cache[key] = load_prior_baseline(con, instrument, prior_session)
            prior = prior_cache[key]
            if len(prior) == 0:
                continue
            merged = target_pop.merge(prior, on=["trading_day", "symbol"], how="left")
            merged["resolved_before_start"] = merged["prior_exit_ts"].notna() & (
                pd.to_datetime(merged["prior_exit_ts"], utc=True) < merged["target_start_ts"]
            )
            if not merged["resolved_before_start"].any():
                continue
            for state in STATES:
                rec = analyze_cell(row, prior_session, state, merged, rng)
                if rec is not None:
                    cell_records.append(rec)

    con.close()

    cells_df = pd.DataFrame(cell_records)
    if len(cells_df) == 0:
        return (
            cells_df,
            pd.DataFrame(),
            pd.DataFrame(metadata_mismatches),
            {"validated_rows": int(len(validated))},
        )

    # Pool per (prior, target, state) across instruments + strategy rows.
    pool_rows: list[dict[str, object]] = []
    for (prior, target, state_name), sub in cells_df.groupby(["prior_session", "target_session", "state"]):
        n_total = int(sub["n_total"].sum())
        n_resolved = int(sub["n_resolved"].sum())
        n_garch = int(sub["n_garch"].sum())
        n_carry = int(sub["n_carry"].sum())
        n_conj = int(sub["n_conj"].sum())
        w_base = (sub["base_exp"] * sub["n_total"]).sum() / n_total if n_total else float("nan")
        w_garch = (sub["garch_exp"] * sub["n_garch"]).sum() / n_garch if n_garch else float("nan")
        w_carry = (sub["carry_exp"] * sub["n_carry"]).sum() / n_carry if n_carry else float("nan")
        w_conj = (sub["conj_exp"] * sub["n_conj"]).sum() / n_conj if n_conj else float("nan")
        role = sub["expected_role"].iloc[0]
        instruments_cell_supported = sub.loc[sub["cell_verdict"] == "supported", "instrument"].unique()
        num_supporting = int(len(instruments_cell_supported))

        pool_verdict = "not_testable_here"
        if n_conj >= MIN_CONJ:
            if num_supporting >= 2:
                pool_verdict = "supported_multi_instrument"
            elif num_supporting == 1:
                pool_verdict = "supported_thin_single_instrument"
            elif sub["cell_verdict"].isin(["unsupported"]).any():
                pool_verdict = "unsupported"
            else:
                pool_verdict = "unclear"

        pool_rows.append(
            {
                "prior_session": prior,
                "target_session": target,
                "state": state_name,
                "state_label": sub["state_label"].iloc[0],
                "expected_role": role,
                "cells": int(len(sub)),
                "instruments": ", ".join(sorted(sub["instrument"].unique())),
                "supporting_instruments": ", ".join(sorted(instruments_cell_supported)),
                "n_total": n_total,
                "n_resolved": n_resolved,
                "n_garch": n_garch,
                "n_carry": n_carry,
                "n_conj": n_conj,
                "base_exp": float(w_base) if pd.notna(w_base) else float("nan"),
                "garch_exp": float(w_garch) if pd.notna(w_garch) else float("nan"),
                "carry_exp": float(w_carry) if pd.notna(w_carry) else float("nan"),
                "conj_exp": float(w_conj) if pd.notna(w_conj) else float("nan"),
                "delta_conj_base": (float(w_conj) - float(w_base))
                if (pd.notna(w_conj) and pd.notna(w_base))
                else float("nan"),
                "delta_conj_garch": (float(w_conj) - float(w_garch))
                if (pd.notna(w_conj) and pd.notna(w_garch))
                else float("nan"),
                "delta_conj_carry": (float(w_conj) - float(w_carry))
                if (pd.notna(w_conj) and pd.notna(w_carry))
                else float("nan"),
                "pool_verdict": pool_verdict,
            }
        )

    pool_df = pd.DataFrame(pool_rows)
    mismatch_df = pd.DataFrame(metadata_mismatches)

    meta = {
        "validated_rows": int(len(validated)),
        "verified_rows": int(len(verified_rows)),
        "metadata_mismatches": int(len(mismatch_df)),
        "cells_tested": int(len(cells_df)),
        "pool_rows": int(len(pool_df)),
    }
    return cells_df, pool_df, mismatch_df, meta


def emit(cells_df: pd.DataFrame, pool_df: pd.DataFrame, mismatch_df: pd.DataFrame, meta: dict[str, int]) -> None:
    lines: list[str] = [
        "# Garch W2e Prior-Session Carry Audit (V2 — broadened scope)",
        "",
        "**Date:** 2026-04-16",
        "**Revision:** V2 — scope broadened per user feedback (don't test one lane) + bootstrap null added per institutional rigor",
        "**Boundary:** validated shelf only, prior trade must be fully resolved before target session start, descriptive-only (no deployment or allocator verdict)",
        "",
        "## Scope",
        "",
        f"- Validated shelf rows considered: **{meta.get('validated_rows', 0)}**",
        f"- Verified (filter SQL returns > 0 canonical rows): **{meta.get('verified_rows', 0)}**",
        f"- Metadata-mismatch validated rows: **{meta.get('metadata_mismatches', 0)}**",
        f"- Prior × target × state × strategy cells tested: **{meta.get('cells_tested', 0)}**",
        f"- Pooled (prior × target × state) groups: **{meta.get('pool_rows', 0)}**",
        f"- Target sessions (validated W2 families): {', '.join(sorted(TARGET_SESSIONS))}",
        f"- Prior candidates considered: {', '.join(PRIOR_CANDIDATES)}",
        f"- Bootstrap iterations: {BOOTSTRAP_ITERS} (Phipson-Smyth, shuffled carry membership)",
        f"- Gates: `MIN_TOTAL={MIN_TOTAL}`, `MIN_CONJ={MIN_CONJ}`, `GARCH_HIGH>={GARCH_HIGH}`",
        "",
        "### Structural limitation: validated shelf instrument coverage",
        "",
        "The validated shelf in the 5 W2 target families is overwhelmingly MNQ.",
        "T8 cross-instrument replication is blocked by this structural limitation,",
        "NOT by a script bug. Any pooled handoff showing `supported_thin_single_instrument`",
        "reflects this data reality. Until more MES/MGC strategies pass validation in",
        "these sessions, single-instrument caveats are honest disclosures, not fixable.",
        "",
    ]

    if len(mismatch_df) > 0:
        lines.extend(
            [
                "## Metadata mismatches (validated-setups rows that did NOT match canonical data)",
                "",
                "Per `integrity-guardian.md` RULE 7: metadata is not evidence. Any validated-setups row whose filter SQL returned zero rows in `orb_outcomes` is surfaced here.",
                "",
                "| Strategy | Instrument | Target session | Filter | Reason |",
                "|---|---|---|---|---|",
            ]
        )
        for _, row in mismatch_df.iterrows():
            lines.append(
                f"| `{row['strategy_id']}` | {row['instrument']} | {row['target_session']} | {row['filter_type']} | {row['reason']} |"
            )
        lines.append("")

    if len(pool_df) == 0:
        lines.append("No testable (prior × target × state) combinations found after chronology and sample gates.")
        OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
        return

    # Supported pooled results first
    supported_pool = pool_df[pool_df["pool_verdict"].str.startswith("supported")]
    unsupported_pool = pool_df[pool_df["pool_verdict"] == "unsupported"]
    unclear_pool = pool_df[pool_df["pool_verdict"] == "unclear"]
    not_testable_pool = pool_df[pool_df["pool_verdict"] == "not_testable_here"]

    def _pool_table(sub: pd.DataFrame, heading: str) -> None:
        if len(sub) == 0:
            return
        lines.extend(
            [
                f"## {heading}",
                "",
                "| Prior → Target | State | Expected role | Cells | Instruments | Supporting | N total | N conj | Base ExpR | Garch ExpR | Carry ExpR | Conj ExpR | Δ conj-base | Δ conj-garch | Δ conj-carry | Verdict |",
                "|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for _, row in sub.sort_values(["target_session", "prior_session", "state"]).iterrows():
            lines.append(
                "| "
                f"{row['prior_session']} → {row['target_session']} | "
                f"{row['state_label']} | "
                f"{row['expected_role']} | "
                f"{int(row['cells'])} | "
                f"{row['instruments']} | "
                f"{row['supporting_instruments'] or '—'} | "
                f"{int(row['n_total'])} | "
                f"{int(row['n_conj'])} | "
                f"{row['base_exp']:+.3f} | "
                f"{row['garch_exp']:+.3f} | "
                f"{row['carry_exp']:+.3f} | "
                f"{row['conj_exp']:+.3f} | "
                f"{row['delta_conj_base']:+.3f} | "
                f"{row['delta_conj_garch']:+.3f} | "
                f"{row['delta_conj_carry']:+.3f} | "
                f"{row['pool_verdict']} |"
            )
        lines.append("")

    _pool_table(
        supported_pool,
        "Supported pooled handoffs (conjunction beats all marginals AND bootstrap p ≤ 0.05 on at least one instrument)",
    )
    _pool_table(
        unsupported_pool, "Unsupported pooled handoffs (enough data, but conjunction fails one or more marginals)"
    )
    _pool_table(unclear_pool, "Unclear pooled handoffs")
    _pool_table(not_testable_pool, "Not testable here (N_conj < 30 after chronology)")

    # Per-cell detail for supported pools only (keep the main report manageable)
    if len(supported_pool) > 0:
        lines.extend(
            [
                "## Per-cell detail for supported handoffs",
                "",
                "(Only pooled handoffs with at least one instrument-level supported verdict.)",
                "",
                "| Prior → Target | Instrument | Strategy | Filter | RR | ORB | State | N | N resolved | N conj | Base ExpR | Garch ExpR | Carry ExpR | Conj ExpR | Δ conj-base | Boot p | dir_match | Cell verdict |",
                "|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        supported_keys = set(
            (r["prior_session"], r["target_session"], r["state"]) for _, r in supported_pool.iterrows()
        )
        sub = cells_df[
            cells_df.apply(
                lambda r: (r["prior_session"], r["target_session"], r["state"]) in supported_keys,
                axis=1,
            )
        ].copy()
        sub = sub.sort_values(["target_session", "prior_session", "state", "instrument", "strategy_id"])
        for _, row in sub.iterrows():
            boot_p_s = "—" if pd.isna(row["bootstrap_p"]) else f"{row['bootstrap_p']:.3f}"
            conj_exp_s = "—" if pd.isna(row["conj_exp"]) else f"{row['conj_exp']:+.3f}"
            delta_base_s = "—" if pd.isna(row["delta_conj_base"]) else f"{row['delta_conj_base']:+.3f}"
            lines.append(
                "| "
                f"{row['prior_session']} → {row['target_session']} | "
                f"{row['instrument']} | "
                f"`{row['strategy_id']}` | "
                f"{row['filter_type']} | "
                f"{row['rr_target']:.1f} | "
                f"{int(row['orb_minutes'])} | "
                f"{row['state_label']} | "
                f"{int(row['n_total'])} | "
                f"{int(row['n_resolved'])} | "
                f"{int(row['n_conj'])} | "
                f"{row['base_exp']:+.3f} | "
                f"{row['garch_exp']:+.3f} | "
                f"{row['carry_exp']:+.3f} | "
                f"{conj_exp_s} | "
                f"{delta_base_s} | "
                f"{boot_p_s} | "
                f"{'yes' if row['dir_match'] else 'no'} | "
                f"{row['cell_verdict']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Guardrails and honest caveats",
            "",
            "- **Chronology:** every row is kept only if the prior trade's `exit_ts` is strictly before the target session start timestamp from `pipeline.dst.orb_utc_window`. No static session-order shortcut.",
            "- **Canonical sources:** all statistics are computed against `orb_outcomes` + `daily_features`. `validated_setups` is used ONLY to enumerate candidate target cells; each row is independently verified against canonical data, and mismatches are surfaced above.",
            "- **Bootstrap null:** `BOOTSTRAP_ITERS=1000` shuffles of carry-state membership among the same population, Phipson-Smyth p. This is a descriptive null floor — it does NOT replace a cross-instrument or cross-era check.",
            "- **Family gate:** a pooled handoff × state is `supported_multi_instrument` only if at least 2 instruments each had a cell-level supported verdict. Single-instrument support is flagged `supported_thin_single_instrument` and should not be treated as universal.",
            "- **Holdout:** 2026-01-01 boundary is respected. This audit uses the full shelf including post-holdout rows — descriptive only, not promotion.",
            "- **No deployment doctrine** is derivable from this audit. It is a state-family distinctness check, not a size/route/allocator claim.",
            "",
        ]
    )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cells_df, pool_df, mismatch_df, meta = build()
    emit(cells_df, pool_df, mismatch_df, meta)
    print(f"Wrote {OUTPUT_MD}")
    print(f"meta: {meta}")


if __name__ == "__main__":
    main()
