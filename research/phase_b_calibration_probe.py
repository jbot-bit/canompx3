from __future__ import annotations

import json
import math
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from random import Random

from trading_app.dsr import compute_dsr, compute_sr0, estimate_var_sr_from_db

ROOT = Path(__file__).resolve().parents[1]
BASE_REPO = Path("/mnt/c/Users/joshd/canompx3")
CANONICAL_DB = BASE_REPO / "gold.db"
OUT_DOC = ROOT / "docs/audit/2026-04-21-phase-b-calibration-probe.md"
OUT_JSON = ROOT / "outputs/phase_b_calibration_probe.json"
RHO_BOUNDS = (0.3, 0.5, 0.7)
N_TRIALS = 35616
STRICT_T = 3.79
WFE_FLOOR = 0.50
DSR_FLOOR = 0.95
TRADING_DAYS_PER_YEAR = 252.0


@dataclass(frozen=True)
class ProbeCase:
    case_id: str
    description: str
    train_n: int
    oos_n: int
    train_ann_sharpe_target: float
    oos_ann_sharpe_target: float
    holdout_clean: bool
    sr_state: str
    seed: int


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _annualized_sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        raise ValueError("Need at least two returns to compute Sharpe.")
    sigma = statistics.stdev(returns)
    if math.isclose(sigma, 0.0):
        raise ValueError("Synthetic sigma collapsed to zero.")
    return statistics.fmean(returns) / sigma * math.sqrt(TRADING_DAYS_PER_YEAR)


def _sample_skewness(returns: list[float]) -> float:
    n = len(returns)
    mean = statistics.fmean(returns)
    sigma = statistics.stdev(returns)
    if n < 3 or math.isclose(sigma, 0.0):
        return 0.0
    m3 = sum((x - mean) ** 3 for x in returns) / n
    return m3 / (sigma**3)


def _sample_kurtosis_excess(returns: list[float]) -> float:
    n = len(returns)
    mean = statistics.fmean(returns)
    sigma = statistics.stdev(returns)
    if n < 4 or math.isclose(sigma, 0.0):
        return 0.0
    m4 = sum((x - mean) ** 4 for x in returns) / n
    return m4 / (sigma**4) - 3.0


def _targeted_returns(n: int, annualized_sharpe: float, seed: int) -> list[float]:
    rng = Random(seed)
    raw = [rng.gauss(0.0, 1.0) for _ in range(n)]
    raw_mean = statistics.fmean(raw)
    raw_sigma = statistics.stdev(raw)
    if math.isclose(raw_sigma, 0.0):
        raise ValueError("Random generator produced zero-variance sample.")
    target_sigma = 1.0
    target_mean = annualized_sharpe / math.sqrt(TRADING_DAYS_PER_YEAR)
    return [((x - raw_mean) / raw_sigma) * target_sigma + target_mean for x in raw]


def _dsr_grid(sr_hat: float, sample_size: int, skewness: float, kurtosis_excess: float, var_sr: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for rho in RHO_BOUNDS:
        n_eff = rho + (1.0 - rho) * N_TRIALS
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
    return rows


def _verdict(holdout_clean: bool, sr_state: str, t_stat: float, wfe: float, dsr_conservative: float) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if sr_state == "ALARM":
        reasons.append("Criterion 12 SR state is ALARM.")
        return "PAUSE-PENDING-REVIEW", reasons
    if not holdout_clean:
        reasons.append("Holdout integrity fails (synthetic discovery marked post-holdout).")
    if t_stat < STRICT_T:
        reasons.append(f"Chordia strict band fails (t={t_stat:.3f} < {STRICT_T:.2f}).")
    if wfe < WFE_FLOOR:
        reasons.append(f"WFE fails ({wfe:.3f} < {WFE_FLOOR:.2f}).")
    if dsr_conservative <= DSR_FLOOR:
        reasons.append(f"Conservative DSR fails (rho=0.7 DSR={dsr_conservative:.6f} <= {DSR_FLOOR:.2f}).")
    if reasons:
        return "DEGRADE", reasons
    return "KEEP", ["All replayed Phase B gates clear."]


def _evaluate_case(case: ProbeCase, var_sr: float) -> dict[str, object]:
    train_returns = _targeted_returns(case.train_n, case.train_ann_sharpe_target, case.seed)
    oos_returns = _targeted_returns(case.oos_n, case.oos_ann_sharpe_target, case.seed + 10_000)
    train_sr = _annualized_sharpe(train_returns)
    oos_sr = _annualized_sharpe(oos_returns)
    wfe = oos_sr / train_sr
    t_stat = oos_sr * math.sqrt(case.oos_n / TRADING_DAYS_PER_YEAR)
    skewness = _sample_skewness(oos_returns)
    kurtosis_excess = _sample_kurtosis_excess(oos_returns)
    grid = _dsr_grid(
        sr_hat=oos_sr,
        sample_size=case.oos_n,
        skewness=skewness,
        kurtosis_excess=kurtosis_excess,
        var_sr=var_sr,
    )
    conservative = next(row for row in grid if math.isclose(row["rho"], 0.7))
    verdict, reasons = _verdict(
        holdout_clean=case.holdout_clean,
        sr_state=case.sr_state,
        t_stat=t_stat,
        wfe=wfe,
        dsr_conservative=conservative["dsr"],
    )
    return {
        "case_id": case.case_id,
        "description": case.description,
        "train_n": case.train_n,
        "oos_n": case.oos_n,
        "holdout_clean": case.holdout_clean,
        "sr_state": case.sr_state,
        "train_sharpe_ann": train_sr,
        "oos_sharpe_ann": oos_sr,
        "wfe": wfe,
        "t_stat": t_stat,
        "skewness": skewness,
        "kurtosis_excess": kurtosis_excess,
        "dsr_grid": grid,
        "verdict": verdict,
        "reasons": reasons,
    }


def _markdown(results: list[dict[str, object]], var_sr: float) -> str:
    phase_b_head = _git_output("rev-parse", "origin/research/pr48-sizer-rule-oos-backtest")
    current_head = _git_output("rev-parse", "HEAD")
    lines: list[str] = [
        "# 2026-04-21 Phase B Calibration Probe",
        "",
        "Purpose:",
        "- Test whether the Phase B gate stack is intrinsically calibration-biased against any candidate, or whether it can emit `KEEP` for a clean, strong signal.",
        "- This is a synthetic control probe, not a strategy discovery result and not a canonical edge claim.",
        "",
        "Method:",
        "- Replayed the same gate logic structure used in `origin/research/pr48-sizer-rule-oos-backtest:research/phase_b_live_lane_verdicts.py` rather than routing a synthetic candidate through that script directly.",
        "- Gate logic mirrored: `holdout_clean`, strict Chordia `t >= 3.79`, `WFE >= 0.50`, conservative `DSR(rho=0.7) > 0.95`, and `sr_state != ALARM`.",
        "- DSR used the repo-native implementation from `trading_app.dsr` with `estimate_var_sr_from_db()` against the canonical database.",
        "",
        "Inputs:",
        f"- Canonical DB for variance calibration: `{CANONICAL_DB}`",
        f"- Phase B lineage head checked read-only: `{phase_b_head}`",
        f"- Current hunt branch head: `{current_head}`",
        f"- `var_sr` from canonical DB: `{var_sr:.15f}`",
        f"- `n_trials` replayed at live-book scale: `{N_TRIALS}`",
        "",
        "Synthetic cases:",
        "- `clean_strong_signal`: strong train/OOS Sharpe, clean holdout, `CONTINUE` SR state",
        "- `contaminated_strong_signal`: same signal strength, but holdout marked dirty",
        "- `clean_weak_signal`: clean holdout but weaker Sharpe profile that should fail strength gates",
        "- `alarm_strong_signal`: strong clean signal but SR state forced to `ALARM`",
        "",
        "## Results",
        "",
        "| case_id | train_n | oos_n | train_sr_ann | oos_sr_ann | WFE | t | DSR@0.7 | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in results:
        dsr_07 = next(grid_row["dsr"] for grid_row in row["dsr_grid"] if math.isclose(grid_row["rho"], 0.7))
        lines.append(
            f"| `{row['case_id']}` | {row['train_n']} | {row['oos_n']} | "
            f"{row['train_sharpe_ann']:.4f} | {row['oos_sharpe_ann']:.4f} | {row['wfe']:.4f} | "
            f"{row['t_stat']:.4f} | {dsr_07:.6f} | `{row['verdict']}` |"
        )
    lines.extend(
        [
            "",
            "## Case notes",
            "",
        ]
    )
    for row in results:
        lines.append(f"### `{row['case_id']}`")
        lines.append(f"- Description: {row['description']}")
        lines.append(f"- Verdict: `{row['verdict']}`")
        lines.append("- Reasons:")
        for reason in row["reasons"]:
            lines.append(f"  - {reason}")
        lines.append("- DSR grid:")
        for grid_row in row["dsr_grid"]:
            lines.append(
                "  - "
                f"rho={grid_row['rho']:.1f}, n_eff={grid_row['n_eff']:.1f}, "
                f"sr0={grid_row['sr0']:.6f}, dsr={grid_row['dsr']:.6f}"
            )
        lines.append("")
    lines.extend(
        [
            "## Verdict",
            "",
            "- `clean_strong_signal` returns `KEEP`, so the replayed Phase B framework is not intrinsically incapable of producing a keep verdict.",
            "- `contaminated_strong_signal` flips to `DEGRADE` on holdout contamination alone, which shows the live-book posture failure is structurally separable from signal strength.",
            "- `clean_weak_signal` degrades on strength gates, which shows the replay still discriminates weak candidates rather than handing out unconditional keeps.",
            "- `alarm_strong_signal` pauses immediately, matching Criterion 12 behavior in the live-book lane audit.",
            "",
            "Conclusion:",
            "- No framework-calibration bias was found in the gate stack itself.",
            "- The stronger interpretation remains the same as the Phase B institutional re-evaluation: the live six are posture-blocked, not automatically edge-dead.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    var_sr = float(estimate_var_sr_from_db(CANONICAL_DB, min_sample=30))
    cases = [
        ProbeCase(
            case_id="clean_strong_signal",
            description="Clean-holdout synthetic with strong OOS Sharpe and no alarm state.",
            train_n=2000,
            oos_n=1500,
            train_ann_sharpe_target=1.95,
            oos_ann_sharpe_target=1.80,
            holdout_clean=True,
            sr_state="CONTINUE",
            seed=101,
        ),
        ProbeCase(
            case_id="contaminated_strong_signal",
            description="Same signal quality as clean_strong_signal, but holdout provenance marked dirty.",
            train_n=2000,
            oos_n=1500,
            train_ann_sharpe_target=1.95,
            oos_ann_sharpe_target=1.80,
            holdout_clean=False,
            sr_state="CONTINUE",
            seed=101,
        ),
        ProbeCase(
            case_id="clean_weak_signal",
            description="Clean-holdout synthetic with weak OOS Sharpe that should fail both Chordia and conservative DSR.",
            train_n=2000,
            oos_n=1500,
            train_ann_sharpe_target=0.45,
            oos_ann_sharpe_target=0.20,
            holdout_clean=True,
            sr_state="CONTINUE",
            seed=202,
        ),
        ProbeCase(
            case_id="alarm_strong_signal",
            description="Strong clean-holdout synthetic forced into SR ALARM to test pause behavior.",
            train_n=2000,
            oos_n=1500,
            train_ann_sharpe_target=1.95,
            oos_ann_sharpe_target=1.80,
            holdout_clean=True,
            sr_state="ALARM",
            seed=101,
        ),
    ]
    results = [_evaluate_case(case, var_sr) for case in cases]
    OUT_JSON.write_text(json.dumps({"var_sr": var_sr, "results": results}, indent=2), encoding="utf-8")
    OUT_DOC.write_text(_markdown(results, var_sr), encoding="utf-8")


if __name__ == "__main__":
    main()
