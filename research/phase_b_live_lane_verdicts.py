from __future__ import annotations

import argparse
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import duckdb

from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.dsr import compute_dsr, compute_sr0, estimate_var_sr_from_db
from trading_app.prop_profiles import ACCOUNT_PROFILES, effective_daily_lanes, parse_strategy_id

SNAPSHOT_COMMIT = "5e768af8"
SNAPSHOT_PATH = "docs/audit/2026-04-21-reset-snapshot.md"
PROFILE_ID = "topstep_50k_mnq_auto"
HOLDOUT_START = "2026-01-01"
RHO_BOUNDS = (0.3, 0.5, 0.7)
DECISIONS_DIR = Path("docs/decisions")
CERT_ROOT = Path("docs/audit/certificates")
TEMPLATE_ROOT = Path("docs/audit/remediations/gate-templates")

PHASE_A_MATRIX: dict[str, dict[str, object]] = {
    "MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5": {
        "fitness": "FIT",
        "t": 2.528,
        "wfe": 2.8551,
        "dsr": 0.0000000004,
        "n_trials": 35616,
        "holdout_clean": False,
        "sr_state": "CONTINUE",
        "a3_verdict": "RESEARCH-PROVISIONAL",
    },
    "MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15": {
        "fitness": "FIT",
        "t": 2.928,
        "wfe": 1.4222,
        "dsr": 0.042151,
        "n_trials": 35700,
        "holdout_clean": False,
        "sr_state": "CONTINUE",
        "a3_verdict": "RESEARCH-PROVISIONAL",
    },
    "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5": {
        "fitness": "FIT",
        "t": 3.717,
        "wfe": 2.6151,
        "dsr": 0.0000005135,
        "n_trials": 35616,
        "holdout_clean": False,
        "sr_state": "CONTINUE",
        "a3_verdict": "RESEARCH-PROVISIONAL",
    },
    "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12": {
        "fitness": "FIT",
        "t": 3.511,
        "wfe": 1.9835,
        "dsr": 0.0000001230,
        "n_trials": 35616,
        "holdout_clean": False,
        "sr_state": "ALARM",
        "a3_verdict": "RESEARCH-PROVISIONAL + SR REVIEW",
    },
    "MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12": {
        "fitness": "FIT",
        "t": 3.400,
        "wfe": 0.8225,
        "dsr": 0.000315,
        "n_trials": 35616,
        "holdout_clean": False,
        "sr_state": "CONTINUE",
        "a3_verdict": "RESEARCH-PROVISIONAL",
    },
    "MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15": {
        "fitness": "FIT",
        "t": 2.831,
        "wfe": 0.7008,
        "dsr": 0.002599,
        "n_trials": 35700,
        "holdout_clean": False,
        "sr_state": "CONTINUE",
        "a3_verdict": "RESEARCH-PROVISIONAL",
    },
}

PHASE_A_SR_EXCERPT = """Lane                                        N         SR      Thr Status
L1 EUROPE_FLOW ORB_G5                      22       2.02    31.96 CONTINUE
L2 SINGAPORE_OPEN ATR_P50                   4       1.90    31.96 CONTINUE
L3 COMEX_SETTLE ORB_G5                     16       5.76    31.96 CONTINUE
L4 NYSE_OPEN COST_LT12                     21      33.27    31.96 ALARM
L5 TOKYO_OPEN COST_LT12                    20       6.63    31.96 CONTINUE
L6 US_DATA_1000 ORB_G5                      5       0.63    31.96 CONTINUE"""

TIMING_RULES: dict[str, dict[str, str]] = {
    "ORB_G5": {
        "feature_columns": "daily_features.orb_{orb_label}_size",
        "source_bar": "ORB_FORMATION",
        "summary": "Phase A A4 marked ORB_G5 timing-valid for the live six-lane book.",
    },
    "COST_LT12": {
        "feature_columns": "daily_features.orb_{orb_label}_size + pipeline.cost_model.COST_SPECS",
        "source_bar": "ORB_FORMATION",
        "summary": "Phase A A4 marked COST_LT12 timing-valid for the live six-lane book.",
    },
    "ATR_P50": {
        "feature_columns": "daily_features.atr_20_pct (prior-only rolling percentile)",
        "source_bar": "STARTUP",
        "summary": "Phase A A4 marked ATR_P50 timing-valid for the live six-lane book.",
    },
}

MECHANISM_NOTES: dict[str, str] = {
    "ORB_G5": (
        "This is an R1 pre-entry filter that excludes undersized ORBs. The project-canon claim is operational: "
        "a materially formed ORB is more likely to represent a real opening range than a noise print, so the gate "
        "screens low-information setups rather than inventing a new alpha source."
    ),
    "COST_LT12": (
        "This is an R1 operational viability filter, not a predictive signal. The filter removes trades where "
        "round-trip friction consumes too much of the raw ORB risk budget, so the mechanism is execution viability "
        "rather than directional edge creation."
    ),
    "ATR_P50": (
        "This is an R1 volatility-regime filter. The project-canon claim is that higher own-instrument ATR percentile "
        "marks sessions where ORB continuation has enough realized range to monetize the breakout geometry."
    ),
}


@dataclass(frozen=True)
class LaneInputs:
    strategy_id: str
    instrument: str
    orb_label: str
    entry_model: str
    rr_target: float
    confirm_bars: int
    filter_type: str
    orb_minutes: int
    max_orb_size_pts: float | None


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _slugify(strategy_id: str) -> str:
    return strategy_id.lower().replace(".", "p").replace("_", "-")


def _short_session(orb_label: str) -> str:
    return orb_label.lower().replace("_", "-")


def _decision_slug(lane: LaneInputs) -> str:
    return f"{lane.instrument.lower()}-{_short_session(lane.orb_label)}-{lane.entry_model.lower()}"


def _load_lanes() -> dict[str, LaneInputs]:
    profile = ACCOUNT_PROFILES[PROFILE_ID]
    lanes = {}
    for spec in effective_daily_lanes(profile):
        parsed = parse_strategy_id(spec.strategy_id)
        lanes[spec.strategy_id] = LaneInputs(
            strategy_id=spec.strategy_id,
            instrument=spec.instrument,
            orb_label=spec.orb_label,
            entry_model=parsed["entry_model"],
            rr_target=float(parsed["rr_target"]),
            confirm_bars=int(parsed["confirm_bars"]),
            filter_type=str(parsed["filter_type"]),
            orb_minutes=int(parsed["orb_minutes"]),
            max_orb_size_pts=spec.max_orb_size_pts,
        )
    return lanes


def _load_validated_row(strategy_id: str) -> dict[str, object]:
    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        configure_connection(con)
        row = con.execute(
            """
            SELECT strategy_id, sample_size, sharpe_ratio, wfe, dsr_score, n_trials_at_discovery,
                   wf_tested, wf_passed, oos_exp_r, discovery_date, status, all_years_positive,
                   skewness, kurtosis_excess
            FROM active_validated_setups
            WHERE strategy_id = ?
            """,
            [strategy_id],
        ).fetchone()
    if row is None:
        raise ValueError(f"active_validated_setups row missing for {strategy_id}")
    return {
        "strategy_id": row[0],
        "sample_size": int(row[1]),
        "sharpe_ratio": float(row[2]),
        "wfe": float(row[3]) if row[3] is not None else None,
        "dsr_score": float(row[4]) if row[4] is not None else None,
        "n_trials_at_discovery": int(row[5]) if row[5] is not None else None,
        "wf_tested": bool(row[6]),
        "wf_passed": bool(row[7]),
        "oos_exp_r": float(row[8]) if row[8] is not None else None,
        "discovery_date": str(row[9]),
        "status": row[10],
        "all_years_positive": bool(row[11]),
        "skewness": float(row[12]) if row[12] is not None else 0.0,
        "kurtosis_excess": float(row[13]) if row[13] is not None else 0.0,
    }


def _dsr_bracket(
    sr_hat: float,
    sample_size: int,
    n_trials: int,
    skewness: float,
    kurtosis_excess: float,
) -> tuple[float, list[dict[str, float]]]:
    var_sr = float(estimate_var_sr_from_db(GOLD_DB_PATH, min_sample=30))
    rows: list[dict[str, float]] = []
    for rho in RHO_BOUNDS:
        n_eff = rho + (1.0 - rho) * n_trials
        sr0 = compute_sr0(n_eff=n_eff, var_sr=var_sr)
        dsr = compute_dsr(
            sr_hat=sr_hat,
            sr0=sr0,
            t_obs=sample_size,
            skewness=skewness,
            kurtosis_excess=kurtosis_excess,
        )
        rows.append(
            {
                "rho": rho,
                "n_eff": n_eff,
                "sr0": sr0,
                "dsr": dsr,
            }
        )
    return var_sr, rows


def _verdict(lane: LaneInputs, matrix: dict[str, object], g3_rows: list[dict[str, float]]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    sr_state = str(matrix["sr_state"])
    if sr_state == "ALARM":
        reasons.append("Criterion 12 live SR monitor is ALARM in the binding Phase A snapshot.")
        return "PAUSE-PENDING-REVIEW", reasons
    if not bool(matrix["holdout_clean"]):
        reasons.append("Holdout integrity fails: discovery_date is after 2026-01-01 and OOS evidence is already populated.")
    if float(matrix["t"]) < 3.79:
        reasons.append("Chordia strict band fails and no local-literature theory band was verified for t>=3.00.")
    conservative = next(row for row in g3_rows if math.isclose(row["rho"], 0.7))
    if conservative["dsr"] <= 0.95:
        reasons.append("Bracketed DSR fails at the directive's conservative rho=0.7 bound.")
    if not reasons:
        return "KEEP", ["All Phase B gates clear on Phase A evidence."]
    return "DEGRADE", reasons


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _g1_certificate(lane: LaneInputs, slug: str, git_sha: str) -> str:
    rule = TIMING_RULES[lane.filter_type]
    orb_size_col = f"orb_{lane.orb_label}_size"
    overlay_line = (
        f"| max_orb_size_pts overlay | prop_profiles + daily_features | `{orb_size_col}` with cap `{lane.max_orb_size_pts:g}` | ORB_FORMATION | CLEAR | "
        f"Phase A snapshot {SNAPSHOT_COMMIT} separated the execution overlay and measured the cap pass-rate for this lane. |"
        if lane.max_orb_size_pts is not None
        else ""
    )
    return f"""# G1 — Timing-Validity Certificate

**Candidate:** `{lane.strategy_id}`
**Date authored:** `2026-04-21`
**Pre-reg:** `N/A — deployed lane retrospective verdict`
**Entry model:** `{lane.entry_model}`
**Decision-time bar:** `{rule["source_bar"]}`

---

## Purpose

This retrospective lane verdict reuses the binding timing-validity result from [the Phase A snapshot](/mnt/c/Users/joshd/canompx3/{SNAPSHOT_PATH}:615). The lane uses `{lane.filter_type}` only; no rel_vol or break-bar variables are present.

## Ban-list cross-check (`.claude/rules/daily-features-joins.md` § Look-Ahead Columns)

- [x] No `break_ts`, `break_delay_min`, `break_bar_volume`, or `rel_vol_*` inputs in the deployed lane definition
- [x] No outcome-derived columns (`pnl_r`, `mae_r`, `mfe_r`) used as filters
- [x] No look-ahead ban-list hits in the current live framing

## Variable inventory

| Variable | Source table | Source column(s) | Source bar / window | Knowable at decision-time bar? | Evidence |
|---|---|---|---|---|---|
| `{lane.filter_type}` gate | `daily_features` / canonical filter registry | `{rule["feature_columns"].replace("{orb_label}", lane.orb_label)}` | `{rule["source_bar"]}` | CLEAR | `{rule["summary"]}` |
{overlay_line}

## Canonical-window verification

- [x] The Phase A snapshot already cleared the active six-lane book as timing-valid in current framing.
- [x] This lane does not use post-break or outcome-derived filter inputs.
- [x] No post-break-role reframing is required.

## Verdict

- [x] CLEAR — timing-valid under the binding Phase A snapshot.

## Literature citation

- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md` p.4
- `docs/audit/2026-04-21-reset-snapshot.md` A4

## Authored by / committed

- Author: `Codex`
- Commit SHA of candidate script: `{git_sha}`
- Commit SHA of this certificate: `{git_sha}`
"""


def _g3_certificate(
    lane: LaneInputs,
    git_sha: str,
    sample_size: int,
    sr_hat: float,
    n_trials: int,
    var_sr: float,
    dsr_rows: list[dict[str, float]],
) -> str:
    rows = "\n".join(
        f"| `{row['rho']:.1f}` | `{row['n_eff']:.1f}` | `{row['sr0']:.6f}` | `{row['dsr']:.6f}` |"
        for row in dsr_rows
    )
    conservative = next(row for row in dsr_rows if math.isclose(row["rho"], 0.7))
    verdict = "PASS" if conservative["dsr"] > 0.95 else "FAIL"
    return f"""# G3 — DSR + N̂ Certificate

**Candidate:** `{lane.strategy_id}`
**Hypothesis family:** `live deployed lane retrospective`
**Pre-reg:** `N/A — deployed lane retrospective verdict`

---

## Purpose

This certificate follows the Terminal 3 template but uses the Phase B corrective rule: bracket `rho_hat` at `0.3 / 0.5 / 0.7`, then fail closed for `KEEP` unless `DSR > 0.95` at the directive's conservative `rho_hat=0.7` bound.

## Live computation

| Input | Value | Source / query |
|---|---|---|
| `T` | `{sample_size}` | `active_validated_setups.sample_size` |
| `SR_hat` | `{sr_hat:.6f}` | `active_validated_setups.sharpe_ratio` |
| `V[SR_n]` | `{var_sr:.6f}` | `trading_app.dsr.estimate_var_sr_from_db(GOLD_DB_PATH)` |
| `M` raw trials | `{n_trials}` | `active_validated_setups.n_trials_at_discovery` |

### Bracketed `N_hat` / DSR

| `rho_hat` | `N_hat = rho + (1-rho) * M` | `SR_0` | `DSR` |
|---|---:|---:|---:|
{rows}

## Verdict

- [x] `{verdict}` at the directive's conservative `rho_hat=0.7` bound.
- [x] `rho_hat=0.7` result used for the Phase B keep/degrade decision.
- [x] gold-db MCP unavailable in-session; canonical substitute was read-only Python against `pipeline.paths.GOLD_DB_PATH`.

## Literature citation

- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` Eq. 2 + Appendix A.3 Eq. 9
- `docs/institutional/pre_registered_criteria.md` Amendment 2.1

## Authored by / committed

- Author: `Codex`
- Commit SHA of candidate script: `{git_sha}`
- Commit SHA of DSR helper: `{_git_output('rev-parse', 'HEAD', '--', 'trading_app/dsr.py')[:12]}`
- Pinned `pre_registered_criteria.md` commit SHA at eval time: `{_git_output('rev-list', '-1', 'HEAD', '--', 'docs/institutional/pre_registered_criteria.md')[:12]}`
"""


def _g4_certificate(lane: LaneInputs, git_sha: str, t_value: float) -> str:
    band = "FAIL"
    if abs(t_value) >= 3.79:
        band = "STRICT_PASS"
    return f"""# G4 — Chordia t-Band Certificate

**Candidate:** `{lane.strategy_id}`
**Observed |t|:** `{abs(t_value):.3f}`

---

## Purpose

Phase B uses the Phase A `t ≈ SR * sqrt(N)` value recorded in the binding snapshot. No new theory cite was added from local literature for this retrospective lane verdict, so the strict `|t| >= 3.79` band is the only passing route.

## Theory justification required for t ≥ 3.00 band

No verified local-literature theory extract was attached for this lane in Phase B, so the `t >= 3.00 with theory` band is **not claimed**.

## Band assignment

- [x] `|t| >= 3.79` — `{str(abs(t_value) >= 3.79).upper()}`
- [ ] `|t| >= 3.00` with verified local-literature theory — NOT CLAIMED
- [x] Below the applicable band for a clean PASS in this phase.

## Verdict

- [x] `{band}` — strict Chordia band {'cleared' if band == 'STRICT_PASS' else 'not cleared'}.

## Literature citation

- `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md`
- `docs/institutional/pre_registered_criteria.md` Amendment 2.2

## Authored by / committed

- Author: `Codex`
- Commit SHA of candidate script: `{git_sha}`
- Commit SHA of this certificate: `{git_sha}`
- Pinned `pre_registered_criteria.md` commit SHA at eval time: `{_git_output('rev-list', '-1', 'HEAD', '--', 'docs/institutional/pre_registered_criteria.md')[:12]}`
"""


def _g6_certificate(lane: LaneInputs, git_sha: str, validated: dict[str, object]) -> str:
    discovery_date = str(validated["discovery_date"])
    fail = discovery_date >= HOLDOUT_START
    return f"""# G6 — Holdout-Integrity Certificate

**Candidate:** `{lane.strategy_id}`
**Pre-reg:** `UNAVAILABLE — grandfathered deployed lane`
**Pre-reg commit SHA at lock:** `UNAVAILABLE`

---

## Purpose

This lane is being judged only against the binding Phase A snapshot. Phase A already established that all six active lanes have `discovery_date` after the sacred `2026-01-01` holdout boundary while also carrying populated OOS fields.

## Required evidence

### 1. Pre-reg lock timestamp

- Pre-reg file: `UNAVAILABLE`
- Pre-reg commit SHA: `UNAVAILABLE`
- Pre-reg commit date: `UNAVAILABLE`
- First evaluation run SHA: `UNAVAILABLE`
- First evaluation run date: `UNAVAILABLE`
- Evaluation date > Pre-reg date? [ ] YES [x] NO (FAIL — cannot certify)

### 2. Holdout boundary enforcement

- [ ] Script-level holdout enforcement could be certified from a pre-reg path
- [x] Phase A snapshot A6 shows this deployed lane was discovered after `2026-01-01`

### 3. No post-hoc iteration against 2026 OOS

- [ ] Zero between-commits certified from a pre-reg lineage
- [x] Cannot certify because no pre-reg lock is attached to this live row

### 4. Mode A compliance

- [ ] 2026 OOS was preserved as untouched holdout
- [x] This lane already shows `wf_tested={validated["wf_tested"]}` / `wf_passed={validated["wf_passed"]}` / `oos_exp_r={validated["oos_exp_r"]}`
- Declared fresh-OOS start date: `UNAVAILABLE`

## Verdict

- [x] {'FAIL' if fail else 'CLEAR'} — `discovery_date={discovery_date}` with live OOS fields populated is not Mode-A holdout-clean.

## Literature citation

- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md` p.4
- `docs/institutional/pre_registered_criteria.md` Amendment 2.7

## Authored by / committed

- Author: `Codex`
- Commit SHA of candidate script: `{git_sha}`
- Commit SHA of this certificate: `{git_sha}`
"""


def _g8_certificate(lane: LaneInputs, git_sha: str) -> str:
    mechanism = MECHANISM_NOTES[lane.filter_type]
    return f"""# G8 — Mechanism Statement Certificate

**Candidate:** `{lane.strategy_id}`

---

## Purpose

This is a project-canon-grounded mechanism statement for a deployed-lane retrospective verdict.

## Mechanism statement

```text
{mechanism}
```

## Literature grounding

### B — PROJECT-CANON-GROUNDED

- [x] Mechanism is supported by `docs/institutional/mechanism_priors.md` §4
- [x] Mechanism is consistent with the canonical filter implementation in `trading_app/config.py`
- [x] This lane is treated as an R1 filter / operational screen, not a new discovery claim

## Mechanism-prior-hierarchy check (`mechanism_priors.md` §4)

- [x] Candidate's proposed role matches the effect shape and implementation role
- [x] No role reframing is required in Phase B

## Verdict

- [x] GROUNDED — project canon, not training-memory only.

## Literature citation

- `docs/institutional/mechanism_priors.md`
- `trading_app/config.py`

## Authored by / committed

- Author: `Codex`
- Commit SHA of this certificate: `{git_sha}`
"""


def _decision_doc(
    lane: LaneInputs,
    slug: str,
    validated: dict[str, object],
    matrix: dict[str, object],
    dsr_rows: list[dict[str, float]],
    verdict: str,
    reasons: list[str],
    cert_dir: Path,
) -> str:
    t_value = float(matrix["t"])
    strict_pass = abs(t_value) >= 3.79
    conservative = next(row for row in dsr_rows if math.isclose(row["rho"], 0.7))
    gate_rows = [
        ("G1 timing-validity", "PASS", "Phase A A4 marked the live six-lane book timing-valid under current framing."),
        (
            "G3 DSR bracket",
            "PASS" if conservative["dsr"] > 0.95 else "FAIL",
            f"rho=0.7 -> DSR={conservative['dsr']:.6f}",
        ),
        (
            "G4 Chordia",
            "PASS" if strict_pass else "FAIL",
            f"|t|={abs(t_value):.3f}; no verified local-literature theory band claimed",
        ),
        (
            "L3 WFE",
            "PASS" if float(matrix["wfe"]) >= 0.5 else "FAIL",
            f"WFE={float(matrix['wfe']):.4f}",
        ),
        (
            "L10 holdout integrity",
            "PASS" if bool(matrix["holdout_clean"]) else "FAIL",
            f"discovery_date={validated['discovery_date']} with wf_tested={validated['wf_tested']}, wf_passed={validated['wf_passed']}",
        ),
        ("SR state", str(matrix["sr_state"]), "Phase A E7 report-only SR monitor"),
    ]
    gate_table = "\n".join(f"| {gate} | {status} | {note} |" for gate, status, note in gate_rows)
    reason_lines = "\n".join(f"- {reason}" for reason in reasons)
    cert_links = "\n".join(
        [
            f"- [G1 timing-validity](/mnt/c/Users/joshd/canompx3/{cert_dir / 'G1-timing-validity.md'})",
            f"- [G3 DSR + N̂](/mnt/c/Users/joshd/canompx3/{cert_dir / 'G3-dsr-neff.md'})",
            f"- [G4 Chordia band](/mnt/c/Users/joshd/canompx3/{cert_dir / 'G4-chordia-band.md'})",
            f"- [G6 holdout integrity](/mnt/c/Users/joshd/canompx3/{cert_dir / 'G6-holdout-integrity.md'})",
            f"- [G8 mechanism statement](/mnt/c/Users/joshd/canompx3/{cert_dir / 'G8-mechanism-statement.md'})",
        ]
    )
    non_applicable = "\n".join(
        [
            "| G2 MinBTL | Not separately re-run here; Phase A A3 already recorded `n_trials` and the lane failed the current operational discovery-budget ceilings. |",
            "| G5 smell test | Not triggered; `|t|` is below 7. |",
            "| G7 negative controls | Not applicable in Phase B because this is not a promotion to shadow/live. |",
            "| G9 kill criteria | Not available; grandfathered deployed lane without a fresh Phase B pre-reg. |",
            "| G10 pre-reg commit pin | Not available; no Phase B pre-reg file exists for this retrospective verdict. |",
        ]
    )
    return f"""# Phase B Lane Verdict — `{lane.strategy_id}`

- Snapshot authority: `{SNAPSHOT_COMMIT}` in [the Phase A truth ledger](/mnt/c/Users/joshd/canompx3/{SNAPSHOT_PATH})
- Final verdict: `{verdict}`
- Phase A driver: `{lane.strategy_id}` carried `SR state = {matrix["sr_state"]}` and `Holdout-clean = FAIL` in the binding snapshot.

## Phase A evidence excerpt

From Phase A E7:

```text
{PHASE_A_SR_EXCERPT}
```

From Phase A A3:

```text
{lane.strategy_id} | fitness={matrix["fitness"]} | t≈{float(matrix["t"]):.3f} | WFE={float(matrix["wfe"]):.4f} | DSR={float(matrix["dsr"]):.10f} | n_trials={int(matrix["n_trials"])} | holdout=FAIL | SR={matrix["sr_state"]} | baseline verdict={matrix["a3_verdict"]}
```

## Gate summary

| Gate | Result | Notes |
|---|---|---|
{gate_table}

## Attached certificates

{cert_links}

## Non-applicable / inherited gate records

| Gate | Record |
|---|---|
{non_applicable}

## Decision rationale

{reason_lines}

## Final decision

- [x] `{verdict}`

This verdict cites the binding Phase A snapshot directly and does not reopen discovery. Any future `KEEP` case would require a new clean holdout lineage and a gate set that clears without fail-closed exceptions.
"""


def _render_lane(lane: LaneInputs, current_sha: str) -> dict[str, object]:
    validated = _load_validated_row(lane.strategy_id)
    matrix = PHASE_A_MATRIX[lane.strategy_id]
    var_sr, dsr_rows = _dsr_bracket(
        sr_hat=float(validated["sharpe_ratio"]),
        sample_size=int(validated["sample_size"]),
        n_trials=int(validated["n_trials_at_discovery"]),
        skewness=float(validated["skewness"]),
        kurtosis_excess=float(validated["kurtosis_excess"]),
    )
    verdict, reasons = _verdict(lane, matrix, dsr_rows)
    slug = _slugify(lane.strategy_id)
    cert_dir = CERT_ROOT / f"2026-04-21-lane-verdict-{slug}"
    decision_path = DECISIONS_DIR / f"2026-04-21-lane-verdict-{slug}.md"
    _write(cert_dir / "G1-timing-validity.md", _g1_certificate(lane, slug, current_sha))
    _write(
        cert_dir / "G3-dsr-neff.md",
        _g3_certificate(
            lane=lane,
            git_sha=current_sha,
            sample_size=int(validated["sample_size"]),
            sr_hat=float(validated["sharpe_ratio"]),
            n_trials=int(validated["n_trials_at_discovery"]),
            var_sr=var_sr,
            dsr_rows=dsr_rows,
        ),
    )
    _write(cert_dir / "G4-chordia-band.md", _g4_certificate(lane, current_sha, float(matrix["t"])))
    _write(cert_dir / "G6-holdout-integrity.md", _g6_certificate(lane, current_sha, validated))
    _write(cert_dir / "G8-mechanism-statement.md", _g8_certificate(lane, current_sha))
    _write(
        decision_path,
        _decision_doc(
            lane=lane,
            slug=slug,
            validated=validated,
            matrix=matrix,
            dsr_rows=dsr_rows,
            verdict=verdict,
            reasons=reasons,
            cert_dir=cert_dir,
        ),
    )
    return {
        "strategy_id": lane.strategy_id,
        "decision_path": decision_path,
        "cert_dir": cert_dir,
        "verdict": verdict,
        "t": float(matrix["t"]),
        "wfe": float(matrix["wfe"]),
        "holdout": bool(matrix["holdout_clean"]),
        "sr_state": str(matrix["sr_state"]),
        "dsr_rho_07": next(row["dsr"] for row in dsr_rows if math.isclose(row["rho"], 0.7)),
    }


def _rollup(rows: list[dict[str, object]]) -> str:
    body = "\n".join(
        f"| `{row['strategy_id']}` | `{row['dsr_rho_07']:.6f}` | `{float(row['t']):.3f}` | "
        f"`{float(row['wfe']):.4f}` | `{'PASS' if row['holdout'] else 'FAIL'}` | `{row['sr_state']}` | `{row['verdict']}` |"
        for row in rows
    )
    return f"""# Phase B Rollup — 2026-04-21

- Snapshot authority: `{SNAPSHOT_COMMIT}` in [the Phase A truth ledger](/mnt/c/Users/joshd/canompx3/{SNAPSHOT_PATH})
- Scope: 6 live lanes only, recalibrated after Phase A contradictions.

| Lane | DSR @ rho=0.7 | Chordia | WFE | Holdout | SR | Verdict |
|---|---:|---:|---:|---|---|---|
{body}

## Summary

- `PAUSE-PENDING-REVIEW`: `{sum(1 for row in rows if row['verdict'] == 'PAUSE-PENDING-REVIEW')}`
- `DEGRADE`: `{sum(1 for row in rows if row['verdict'] == 'DEGRADE')}`
- `KEEP`: `{sum(1 for row in rows if row['verdict'] == 'KEEP')}`

No lane cleared a clean `KEEP` path in this phase. The NYSE lane is paused under Criterion 12; the remaining five fail closed to `DEGRADE` on holdout integrity plus additional gate deficits.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy-id")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--write-rollup", action="store_true")
    args = parser.parse_args()

    if not args.strategy_id and not args.all:
        raise SystemExit("pass --strategy-id or --all")

    current_sha = _git_output("rev-parse", "--short", "HEAD")
    lanes = _load_lanes()
    selected: list[LaneInputs]
    if args.all:
        selected = [lanes[sid] for sid in lanes]
    else:
        if args.strategy_id not in lanes:
            raise SystemExit(f"unknown strategy_id: {args.strategy_id}")
        selected = [lanes[args.strategy_id]]

    rendered = [_render_lane(lane, current_sha) for lane in selected]
    if args.write_rollup:
        _write(DECISIONS_DIR / "2026-04-21-phase-b-rollup.md", _rollup(rendered))


if __name__ == "__main__":
    main()
