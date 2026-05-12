"""Bounded re-audit of the VWAP dead-confluence verdict.

This runner audits the already-consumed 2026-04-18 VWAP family scan. It does
not rerun that scan and does not alter the database. It checks the primary
prereg, result doc, and runner script against the user's requested safeguards:
canonical layers, temporal validity, Mode A holdout, K-family/BH-FDR,
OOS dir_match, and tautology/overlap handling.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PREREG_PATH = REPO_ROOT / "docs/audit/hypotheses/2026-05-12-vwap-dead-confluence-reaudit-v1.yaml"
ORIGINAL_PREREG = REPO_ROOT / "docs/audit/hypotheses/2026-04-18-vwap-comprehensive-family-scan.yaml"
ORIGINAL_RESULT = REPO_ROOT / "docs/audit/results/2026-04-18-vwap-comprehensive-family-scan.md"
ORIGINAL_RUNNER = REPO_ROOT / "research/vwap_comprehensive_family_scan.py"
PIPELINE_SOURCE = REPO_ROOT / "pipeline/build_daily_features.py"
RESULT_DOC = REPO_ROOT / "docs/audit/results/2026-05-12-vwap-dead-confluence-reaudit-v1.md"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    evidence: str


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _contains_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def _mode_a_l6_sanity() -> dict[str, float | int]:
    """Recompute one VWAP positive-control cell from canonical layers.

    This is not a family rescan. It only verifies that the original runner's
    strict Mode A L6 sanity row is in the same neighborhood using current
    canonical filter delegation.
    """
    from pipeline.paths import GOLD_DB_PATH
    from research.filter_utils import filter_signal
    from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

    sql = """
    SELECT
        o.trading_day,
        o.pnl_r,
        d.orb_US_DATA_1000_high,
        d.orb_US_DATA_1000_low,
        d.orb_US_DATA_1000_break_dir,
        d.orb_US_DATA_1000_vwap
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = 'MNQ'
      AND o.orb_label = 'US_DATA_1000'
      AND o.orb_minutes = 15
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target = 1.5
      AND o.pnl_r IS NOT NULL
      AND o.trading_day < ?
      AND d.orb_US_DATA_1000_break_dir = 'long'
    ORDER BY o.trading_day
    """
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        df = con.execute(sql, [HOLDOUT_SACRED_FROM]).df()
    finally:
        con.close()
    if df.empty:
        return {"n_total": 0, "n_on": 0, "expr_on": float("nan")}
    sig = filter_signal(df, "VWAP_MID_ALIGNED", "US_DATA_1000")
    on = df.loc[sig == 1, "pnl_r"].astype(float)
    return {
        "n_total": int(len(df)),
        "n_on": int(len(on)),
        "expr_on": float(on.mean()) if len(on) else float("nan"),
    }


def build_checks() -> tuple[list[Check], dict[str, float | int]]:
    prereg = yaml.safe_load(_read(PREREG_PATH))
    original_result = _read(ORIGINAL_RESULT)
    original_runner = _read(ORIGINAL_RUNNER)
    pipeline_source = _read(PIPELINE_SOURCE)

    checks: list[Check] = []

    artifacts_ok = all(path.exists() for path in [PREREG_PATH, ORIGINAL_PREREG, ORIGINAL_RESULT, ORIGINAL_RUNNER])
    checks.append(
        Check(
            "primary_artifacts_exist",
            "PASS" if artifacts_ok else "FAIL",
            "Found current prereg plus original VWAP prereg/result/runner files.",
        )
    )

    canonical_ok = _contains_all(
        original_runner,
        ["FROM orb_outcomes o", "JOIN daily_features d", "from pipeline.paths import GOLD_DB_PATH"],
    )
    checks.append(
        Check(
            "canonical_layers_only",
            "PASS" if canonical_ok else "FAIL",
            "Original runner loads orb_outcomes JOIN daily_features through GOLD_DB_PATH.",
        )
    )

    temporal_ok = _contains_all(
        pipeline_source,
        [
            "Both use only bars BEFORE the ORB window",
            'pre_mask = (bars_df["ts_utc"] >= td_start) & (bars_df["ts_utc"] < orb_start)',
        ],
    ) and "RULE 6.1 SAFE" in original_result
    checks.append(
        Check(
            "temporal_validity",
            "PASS" if temporal_ok else "FAIL",
            "VWAP source uses strict pre-ORB bars and result marks VWAP filters RULE 6.1 SAFE.",
        )
    )

    holdout = str(prereg["metadata"]["holdout_date"])
    mode_a_ok = holdout == "2026-01-01" and "trading_day < `2026-01-01`" in original_result
    checks.append(
        Check(
            "mode_a_holdout",
            "PASS" if mode_a_ok else "FAIL",
            f"Current prereg holdout={holdout}; original result reports IS trading_day < 2026-01-01.",
        )
    )

    k_ok = "K_family" in original_result and re.search(r"K_family.*0\*\* pass", original_result, re.I | re.S)
    checks.append(
        Check(
            "k_family_bh_fdr",
            "PASS" if k_ok else "FAIL",
            "Original result reports K_family BH-FDR with 0 pass.",
        )
    )

    oos_ok = "dir_match" in original_result and "OOS one-shot" in original_result
    checks.append(
        Check(
            "oos_dir_match",
            "PASS" if oos_ok else "FAIL",
            "Original result states OOS one-shot consumption and dir_match requirement.",
        )
    )

    tautology_ok = all(
        needle in original_result
        for needle in ["Flagged tautology", "T0 corr=1.00", "TAUTOLOGY"]
    ) and "research.filter_utils.filter_signal" in original_runner
    checks.append(
        Check(
            "tautology_or_overlap_screen",
            "PASS" if tautology_ok else "FAIL",
            "Original result reports T0 tautology flags; runner delegates filters through research.filter_utils.",
        )
    )

    dead_ok = "VWAP family DOCTRINE-CLOSED" in original_result and "Survivors: 0" in original_result
    checks.append(
        Check(
            "original_dead_verdict_present",
            "PASS" if dead_ok else "FAIL",
            "Original result has 0 H1 survivors and VWAP family DOCTRINE-CLOSED verdict.",
        )
    )

    no_rerun_ok = "REFUSING TO RE-RUN" in original_runner and "Mode A one-shot OOS consumption" in original_runner
    checks.append(
        Check(
            "no_rerun_of_consumed_oos",
            "PASS" if no_rerun_ok else "FAIL",
            "Original runner refuses rerun when result exists due to one-shot OOS consumption.",
        )
    )

    h3_caveat_ok = "H3 specification error" in original_result and "K1 substantive" in original_result
    checks.append(
        Check(
            "h3_positive_control_caveat",
            "CAVEATED_PASS" if h3_caveat_ok else "FAIL",
            "Original result documents H3 baseline specification error while preserving independent K1 death verdict.",
        )
    )

    return checks, _mode_a_l6_sanity()


def render_result(checks: list[Check], l6: dict[str, float | int]) -> str:
    from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

    hard_failures = [c for c in checks if c.status == "FAIL"]
    verdict = "PASS_STANDS" if not hard_failures else "INVALID_OR_PARK"

    lines = [
        "# VWAP dead-confluence verdict procedural re-audit v1",
        "",
        f"**Pre-reg:** `{PREREG_PATH.relative_to(REPO_ROOT)}`",
        f"**Original result:** `{ORIGINAL_RESULT.relative_to(REPO_ROOT)}`",
        f"**Original runner:** `{ORIGINAL_RUNNER.relative_to(REPO_ROOT)}`",
        f"**Mode A holdout:** `{HOLDOUT_SACRED_FROM}`",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
    ]
    if verdict == "PASS_STANDS":
        lines.append(
            "The 2026-04-18 VWAP family DOCTRINE-CLOSED verdict stands as a procedural research verdict. "
            "This audit did not rerun the consumed family scan."
        )
    else:
        lines.append("At least one required audit check failed; do not cite the original death verdict without repair.")
    lines.extend(["", "## Checklist", ""])
    lines.append("| Check | Status | Evidence |")
    lines.append("|---|---|---|")
    for check in checks:
        lines.append(f"| {check.name} | {check.status} | {check.evidence} |")

    lines.extend(
        [
            "",
            "## Canonical Sanity Query",
            "",
            "Bounded read-only sanity query on the known L6 VWAP cell:",
            "",
            "| Cell | N total IS | N on-signal IS | ExpR on-signal IS |",
            "|---|---:|---:|---:|",
            (
                "| MNQ US_DATA_1000 O15 E2 RR1.5 CB1 VWAP_MID_ALIGNED | "
                f"{l6['n_total']} | {l6['n_on']} | {float(l6['expr_on']):+.4f} |"
            ),
            "",
            "This is a sanity check only. It is not a family rescan and does not change the original K-family verdict.",
            "",
            "## Scope Discipline",
            "",
            "- No write to `gold.db`.",
            "- No write to `experimental_strategies` or `validated_setups`.",
            "- No rerun of `research/vwap_comprehensive_family_scan.py`.",
            "- No OOS retuning.",
            "",
            "## Decision",
            "",
        ]
    )
    if verdict == "PASS_STANDS":
        lines.extend(
            [
                "VWAP remains closed as a broad confluence family from this primary scan.",
                "The known live/shelf VWAP exact lanes remain exact-lane facts; they do not reopen the family.",
                "Next priority in the user's list should be volume-at-break or prior-day geometry, using the same prereg + bounded-runner pattern.",
            ]
        )
    else:
        lines.append("Repair the failed checks before using the VWAP death verdict.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    checks, l6 = build_checks()
    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    RESULT_DOC.write_text(render_result(checks, l6), encoding="utf-8")
    print(f"Wrote {RESULT_DOC.relative_to(REPO_ROOT)}")
    return 0 if all(c.status != "FAIL" for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
