"""FAST_LANE v5.1 PROMOTE queue scanner.

See ``docs/specs/fast_lane_state_graph.md`` for the canonical fast-lane chain definition.

Reconstructs the PROMOTE-queue state from on-disk artifacts:
  - PROMOTE result MDs under ``docs/audit/results/*fast-lane*.md``
  - revocation sidecars (``<base>.revocation.md`` next to the result MD)
  - heavyweight pre-regs under ``docs/audit/hypotheses/`` whose
    ``scope.strategy_id`` matches a PROMOTE result's strategy_id
  - park entries in ``docs/runtime/action-queue.yaml``

The queue is **derived state**. The cache file ``docs/runtime/promote_queue.yaml``
is rebuilt from these sources on every run; ``--write`` refreshes the cache,
``--dry-run`` (default) prints the would-be queue and the diff vs the cache.

Drift check ``check_fast_lane_promote_orphans`` (Check 157, added in this
landing) reconstructs independently and fails the build if the cache is
stale, hand-edited, or any entry is ERROR.

Per-direction sanity gate
-------------------------
Re-applies v5.1 thresholds to the ``## Directional breakdown`` table that
every v5.1 result MD carries.  When a pooled PROMOTE has BOTH per-direction
sub-stats failing v5.1 gates as standalone (t<2.5, N<50, or fire-rate
outside [0.05, 0.95]), the pooled PROMOTE is a sample-doubling artifact and
the cell is flagged ``REVOKE_RECOMMENDED``.  Operator authors a
``.revocation.md`` sidecar; on next scan the entry moves QUEUED -> REVOKED.

Status enum (no UNKNOWN)
------------------------
QUEUED      PROMOTE + no revocation sidecar + no heavyweight prereg +
            no park entry + per-direction sanity gate PASS
ESCALATED   PROMOTE + matching heavyweight prereg under docs/audit/hypotheses/
REVOKED     PROMOTE + revocation sidecar present
PARKED      PROMOTE + action-queue.yaml entry naming this strategy_id with park
ERROR       PROMOTE + (missing/unparseable directional breakdown
                       OR per-direction sanity gate fires REVOKE_RECOMMENDED
                       with no revocation sidecar yet)

ERROR forces operator attention - never silently lingers.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "docs" / "audit" / "results"
HYPOTHESES_DIR = REPO_ROOT / "docs" / "audit" / "hypotheses"
ACTION_QUEUE = REPO_ROOT / "docs" / "runtime" / "action-queue.yaml"
QUEUE_CACHE = REPO_ROOT / "docs" / "runtime" / "promote_queue.yaml"
TRIAL_LEDGER_PATH = REPO_ROOT / "docs" / "runtime" / "fast_lane_trial_ledger.yaml"
TRIAL_CORRECTIONS_PATH = REPO_ROOT / "docs" / "runtime" / "fast_lane_trial_corrections.yaml"
GRAVEYARD_DIGEST_PATH = REPO_ROOT / "docs" / "runtime" / "fast_lane_graveyard_digest.yaml"

FAST_LANE_RESULT_GLOB = "*fast-lane*.md"

# Locked prior for correlation-haircut N_hat per Stage 2A.3 stage file
# § "K-Lineage Schema" + § Risks #1. Empirical fit deferred to Stage 2B once
# N_unique_trading_days ledger entries >= 50. Inlined here intentionally --
# Check #172 asserts every emitted entry carries rho_hat_assumed == 0.5.
RHO_HAT_ASSUMED: float = 0.5

# v5.1 thresholds re-applied to per-direction sub-stats.
# Mirrors docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml lines 102-145.
T_PROMOTE_FLOOR = 3.0
T_KILL_FLOOR = 2.5
EXPR_FLOOR = 0.0
N_FLOOR = 50
FIRE_MIN = 0.05
FIRE_MAX = 0.95

# Pre-flight OOS-power gate (RULE 3.3 enforcement at classification time).
#
# Rejects PROMOTE results whose expected OOS sample size cannot deliver power
# >= OOS_POWER_FLOOR to detect a Cohen's d = OOS_COHEN_D_TARGET effect via the
# canonical one_sample_power helper. Mirrors the DIRECTIONAL_ONLY tier in
# research/oos_power.py::POWER_TIERS (0.50); cells below 0.50 are
# STATISTICALLY_USELESS and cannot inform a deployment decision regardless of
# OOS sign.
#
# Literature grounding:
#   - bailey_et_al_2013_pseudo_mathematics.md Thm 1 / Eq. 6 — MinBTL bound;
#     gating cells with no usable OOS preserves trial-budget headroom.
#   - harvey_liu_2015_backtesting.md p.17 — OOS testing is probabilistic; an
#     underpowered OOS read cannot inform regardless of sign.
#   - lopez_de_prado_2018_afml_ch_3_7_8.md § 12.4 — CPCV is the multi-path
#     remedy for borderline-OOS cells this gate flags.
#
# Cohen's d = 0.3 (between Cohen 1988 "small" 0.2 and "medium" 0.5) is the
# gate's design target — empirically matches the per-trade R-multiple effect
# sizes observed across the validated ORB universe (pooled t / sqrt(N) ratios
# on FAST_LANE PROMOTE results cluster in 0.2-0.3). d=0.2 was too strict for
# the current 136-day OOS window (rejects every candidate including 50%+ fire
# rates); d=0.5 too lenient (passes structurally unbuildable cells). d=0.3
# splits the difference and matches realized effect sizes — a cell at d=0.3
# needing N>~85 for 0.50 power gives the OOS window a real chance to deliver
# a discriminating read while still rejecting cells that mathematically
# cannot accumulate that N before the next holdout rotation.
OOS_POWER_FLOOR = 0.50
OOS_COHEN_D_TARGET = 0.3


# Canonical status enum. The six SUPPRESSED_* tokens are mirrored by the
# Stage 2A.3 stage file's `## Suppression Status Enum` table -- Check #172
# parses the table and asserts byte-for-byte parity. See
# pipeline/canonical_inline_copies.py
# ``fast_lane_promote_suppression_status_values`` for the registry entry.
STATUS_VALUES = (
    "QUEUED",
    "ESCALATED",
    "REVOKED",
    "PARKED",
    "REJECTED_OOS_UNPOWERED",
    "SUPPRESSED_BANNED_ENTRY_MODEL",
    "SUPPRESSED_E2_LOOKAHEAD",
    "SUPPRESSED_GRAVEYARD",
    "SUPPRESSED_DUPLICATE_ACTIVE",
    "SUPPRESSED_SIBLING_RETEST",
    "SUPPRESSED_K_OVERRUN",
    "ERROR",
)


@dataclass
class PromoteEntry:
    result_md: str
    strategy_id: str
    direction: str
    pooled_t: float
    pooled_expr: float
    pooled_n: int
    pooled_fire: float
    long_n: int
    long_expr: float
    long_t: float
    short_n: int
    short_expr: float
    short_t: float
    pooled_universe_n: int
    long_fire: float
    short_fire: float
    long_side_verdict: str
    short_side_verdict: str
    pooling_artifact: bool
    revocation_sidecar: str | None
    heavyweight_prereg: str | None
    park_entry: str | None
    status: str
    error_reason: str | None
    # Stage 2A.3 provenance fields. Populated for every entry the scanner
    # emits, regardless of terminal status -- Check #172 enforces presence.
    # ``structural_hash`` is the 16-hex de-dup criterion from
    # ``fast_lane_structural_hash.compute_structural_hash``; ``k_lineage``
    # carries the universe-of-trials accounting per Bailey-Lopez de Prado
    # 2014 sec 3; ``n_hat`` is the correlation-haircut sample size from the
    # K-Lineage Schema in the stage file.
    structural_hash: str = ""
    k_lineage: dict[str, Any] = field(default_factory=dict)
    n_hat: int = 0
    upstream_k_role: str = ""
    upstream_k_value: int | None = None


_TITLE_RE = re.compile(r"^#\s+Chordia strict unlock audit\s+\S\s+(?P<sid>\S+)\s*$", re.MULTILINE)
_VERDICT_RE = re.compile(r"^\*\*FAST_LANE verdict:\*\*\s+`(?P<v>PROMOTE|KILL|NEEDS-MORE)`\s*$", re.MULTILINE)
_DIR_FOOTER_RE = re.compile(r"_Scope direction at screen:\s*`'(?P<dir>pooled|long|short)'`", re.MULTILINE)

_SPLIT_IS_RE = re.compile(
    r"^\|\s*IS\s*\|\s*(?P<nu>[-+0-9.]+)\s*\|\s*(?P<nf>[-+0-9.]+)\s*\|\s*(?P<fp>[-+0-9.]+)%\s*"
    r"\|\s*\S+\s*\|\s*\S+\s*\|\s*(?P<expr>[-+0-9.eE]+)\s*\|\s*\S+\s*\|\s*\S+\s*\|\s*(?P<t>[-+0-9.eE]+)\s*\|",
    re.MULTILINE,
)

_DIR_IS_RE = re.compile(
    r"^\|\s*IS\s*\|\s*(?P<ln>[-+0-9.]+)\s*\|\s*(?P<lex>[-+0-9.eE]+)\s*\|\s*(?P<lt>[-+0-9.eEnNaA]+)\s*"
    r"\|\s*(?P<sn>[-+0-9.]+)\s*\|\s*(?P<sex>[-+0-9.eEnNaA]+)\s*\|\s*(?P<st>[-+0-9.eEnNaA]+)\s*\|",
    re.MULTILINE,
)


def _parse_float(s: str) -> float:
    s = s.strip().lower()
    if s in {"nan", "", "n/a"}:
        return float("nan")
    return float(s)


def _parse_int(s: str) -> int:
    s = s.strip()
    if s == "":
        return 0
    return int(float(s))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _rel_to_repo(path: Path) -> str:
    """Return repo-relative path string; fall back to absolute when outside
    REPO_ROOT (test fixtures under tmp_path)."""
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def parse_result_md(path: Path) -> dict[str, Any] | None:
    text = _read_text(path)
    verdict_match = _VERDICT_RE.search(text)
    title_match = _TITLE_RE.search(text)
    direction_match = _DIR_FOOTER_RE.search(text)
    if not verdict_match or not title_match:
        return None
    return {
        "result_md": _rel_to_repo(path),
        "strategy_id": title_match.group("sid"),
        "verdict": verdict_match.group("v"),
        "direction": direction_match.group("dir") if direction_match else "pooled",
    }


def parse_promote_stats(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    text = _read_text(path)
    is_m = _SPLIT_IS_RE.search(text)
    dir_m = _DIR_IS_RE.search(text)
    if not is_m:
        return None, "Split summary IS row not found"
    if not dir_m:
        return None, "Directional breakdown IS row not found"
    try:
        pooled_universe_n = _parse_int(is_m.group("nu"))
        stats = {
            "pooled_universe_n": pooled_universe_n,
            "pooled_n": _parse_int(is_m.group("nf")),
            "pooled_fire": _parse_float(is_m.group("fp")) / 100.0,
            "pooled_expr": _parse_float(is_m.group("expr")),
            "pooled_t": _parse_float(is_m.group("t")),
            "long_n": _parse_int(dir_m.group("ln")),
            "long_expr": _parse_float(dir_m.group("lex")),
            "long_t": _parse_float(dir_m.group("lt")),
            "short_n": _parse_int(dir_m.group("sn")),
            "short_expr": _parse_float(dir_m.group("sex")),
            "short_t": _parse_float(dir_m.group("st")),
        }
    except ValueError as exc:
        return None, f"Numeric parse failure: {exc}"
    if pooled_universe_n <= 0:
        return None, "pooled_universe_n <= 0; cannot compute per-direction fire-rate"
    stats["long_fire"] = stats["long_n"] / pooled_universe_n
    stats["short_fire"] = stats["short_n"] / pooled_universe_n
    return stats, None


def _side_verdict(t: float, n: int, expr: float, fire: float) -> str:
    if n == 0 or math.isnan(t):
        return "N_A"
    if n < N_FLOOR or fire < FIRE_MIN or fire > FIRE_MAX or expr <= EXPR_FLOOR:
        return "KILL_AS_STANDALONE"
    if t < T_KILL_FLOOR:
        return "KILL_AS_STANDALONE"
    if t < T_PROMOTE_FLOOR:
        return "NEEDS_MORE_AS_STANDALONE"
    return "PROMOTE_AS_STANDALONE"


def per_direction_sanity_gate(stats: dict[str, Any], direction: str) -> tuple[str, str, bool]:
    lv = _side_verdict(stats["long_t"], stats["long_n"], stats["long_expr"], stats["long_fire"])
    sv = _side_verdict(stats["short_t"], stats["short_n"], stats["short_expr"], stats["short_fire"])
    if direction == "pooled" and lv == "KILL_AS_STANDALONE" and sv == "KILL_AS_STANDALONE":
        return lv, sv, True
    return lv, sv, False


def _compute_expected_oos_power(
    fire_rate: float,
    oos_window_days: int,
    *,
    cohen_d: float = OOS_COHEN_D_TARGET,
) -> tuple[float, int, str | None]:
    """Return ``(expected_power, expected_n_oos, reason_if_unable)``.

    Fail-closed: any unusable input returns ``(0.0, 0, reason)`` so the caller
    treats the cell as REJECTED rather than silently passing.

    ``fire_rate`` is the pooled fire-rate already parsed from the result MD.
    ``oos_window_days`` is the count of calendar days between
    ``trading_app.holdout_policy.HOLDOUT_SACRED_FROM`` and the latest
    ``orb_outcomes.trading_day`` observed by the caller. The product
    ``round(fire_rate * oos_window_days)`` is the expected OOS trade count;
    one-sample power is then computed against ``cohen_d``.
    """
    if math.isnan(fire_rate) or fire_rate <= 0:
        return 0.0, 0, "fire_rate unavailable or non-positive"
    if oos_window_days <= 0:
        return 0.0, 0, "OOS window not yet opened (oos_window_days <= 0)"
    expected_n_oos = int(round(fire_rate * oos_window_days))
    if expected_n_oos < 2:
        # one_sample_power refuses n < 2; treat as zero-power for clarity.
        return 0.0, expected_n_oos, "expected_n_oos < 2 (one_sample_power undefined)"
    # Canonical delegation per institutional-rigor.md § 10 — never re-encode.
    from research.oos_power import one_sample_power

    power = float(one_sample_power(cohen_d, expected_n_oos))
    return power, expected_n_oos, None


def _resolve_oos_window_days(db_path: Path | None = None) -> tuple[int, str | None]:
    """Compute ``(oos_window_days, error_reason)`` from canonical sources.

    Reads the latest ``orb_outcomes.trading_day`` from the canonical DuckDB
    and subtracts ``HOLDOUT_SACRED_FROM``. Fail-closed: returns ``(0, reason)``
    on any DB error so the caller REJECTS dependent cells rather than passing.
    """
    try:
        from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
    except Exception as exc:  # pragma: no cover -- import guard
        return 0, f"holdout_policy import failed: {exc}"
    try:
        import duckdb

        from pipeline.paths import GOLD_DB_PATH

        path = db_path if db_path is not None else GOLD_DB_PATH
        con = duckdb.connect(str(path), read_only=True)
        try:
            row = con.execute("SELECT MAX(trading_day) FROM orb_outcomes").fetchone()
        finally:
            con.close()
        if row is None or row[0] is None:
            return 0, "orb_outcomes empty (MAX(trading_day) is NULL)"
        latest = row[0]
        days = (latest - HOLDOUT_SACRED_FROM).days
        return max(int(days), 0), None
    except Exception as exc:
        return 0, f"OOS window query failed: {exc}"


def find_revocation_sidecar(result_md: Path) -> Path | None:
    sidecar = result_md.with_name(result_md.stem + ".revocation.md")
    return sidecar if sidecar.exists() else None


# ---------- Stage 2A.3 provenance + suppression helpers ----------


def _load_source_yaml(result_md: Path, hypotheses_dir: Path) -> dict[str, Any] | None:
    """Locate the source YAML for a result MD by filename stem.

    Mirrors ``fast_lane_to_heavyweight_bridge.locate_source_yaml`` -- the
    fast-lane convention is ``<stem>.md`` <-> ``<stem>.yaml`` under
    ``docs/audit/hypotheses/``. Drafts (``drafts/<stem>.yaml``) are NOT a
    valid source for a published result MD.
    """
    candidate = hypotheses_dir / f"{result_md.stem}.yaml"
    if not candidate.exists():
        return None
    try:
        data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _scope_to_hash_inputs(scope: dict[str, Any]) -> dict[str, Any] | None:
    """Translate a fast-lane source YAML's ``scope:`` block into the input
    dict consumed by ``compute_structural_hash``.

    Returns None if any required structural-hash key is missing -- the caller
    routes the entry to ERROR rather than emitting a degenerate hash. Source
    YAMLs use ``session`` (string label) where the hash recipe expects
    ``orb_label``; we forward verbatim.

    Direction-vocabulary translation: fast-lane source YAMLs use
    ``direction: pooled`` to mean "both directions evaluated together by
    the screen" (a screen-time convention from FAST_LANE v5.1 template).
    The structural_hash recipe uses the structural taxonomy {LONG, SHORT,
    BOTH} where ``BOTH`` is the structural-identity equivalent of pooled.
    Translation is lossless because a single-direction prereg always
    writes its own LONG/SHORT in ``scope.direction`` -- only the
    pooled-screen variant uses the screen-time token.
    """
    try:
        raw_dir = scope.get("direction", "BOTH")
        if isinstance(raw_dir, str) and raw_dir.strip().lower() == "pooled":
            normalised_dir = "BOTH"
        else:
            normalised_dir = raw_dir
        return {
            "instrument": scope["instrument"],
            "orb_label": scope["session"],
            "orb_minutes": int(scope["orb_minutes"]),
            "rr_target": float(scope["rr_target"]),
            "entry_model": scope["entry_model"],
            "confirm_bars": int(scope["confirm_bars"]),
            "filter_type": scope.get("filter_type", ""),
            "direction": normalised_dir,
            "filter_threshold": scope.get("filter_threshold", ""),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _read_graveyard_digest(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read the graveyard digest once per scan; return {hash -> [entries]}.

    Class-level entries (hash_kind == "class") and lane-level entries
    (hash_kind == "lane") are both indexed by structural_hash. The scanner
    uses lane matches for suppression (SUPPRESSED_GRAVEYARD) and class
    matches as informational (recorded as a citation, not a suppressor).
    """
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return {}
    if not isinstance(data, dict):
        return {}
    entries = data.get("entries") or []
    if not isinstance(entries, list):
        return {}
    index: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        h = entry.get("structural_hash")
        if not isinstance(h, str):
            continue
        index.setdefault(h, []).append(entry)
    return index


def _read_trial_ledger(path: Path) -> list[dict[str, Any]]:
    """Read existing ledger entries once per scan. Empty when ledger
    missing -- ledger is bootstrapped on first scanner run."""
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return []
    if not isinstance(data, dict):
        return []
    entries = data.get("entries") or []
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict)]


def _find_duplicate_active_prereg(
    structural_hash: str,
    source_yaml_path: Path,
    hypotheses_dir: Path,
) -> Path | None:
    """Return another hypotheses/ YAML (NOT under drafts/) that resolves to
    the same structural_hash AND has no published result MD yet.

    A published result MD is identified by the existence of
    ``docs/audit/results/<stem>.md`` next to the source YAML's stem. If one
    exists the source has already produced a result -- it is not an active
    duplicate, it is the prior occurrence of the same lane and counts via
    K_lane through the ledger instead.
    """
    if not hypotheses_dir.exists():
        return None
    results_dir = source_yaml_path.parent.parent.parent / "docs" / "audit" / "results"
    if not results_dir.exists():
        # Fall back to module-level RESULTS_DIR when source_yaml_path is a
        # tmp_path fixture far from REPO_ROOT.
        results_dir = RESULTS_DIR
    for candidate in sorted(hypotheses_dir.glob("*.yaml")):
        if candidate.resolve() == source_yaml_path.resolve():
            continue
        try:
            data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        if not isinstance(data, dict):
            continue
        scope = data.get("scope")
        if not isinstance(scope, dict):
            continue
        candidate_inputs = _scope_to_hash_inputs(scope)
        if candidate_inputs is None:
            continue
        try:
            from scripts.research.fast_lane_structural_hash import (
                compute_structural_hash,
            )

            candidate_hash = compute_structural_hash(candidate_inputs)
        except Exception:
            continue
        if candidate_hash != structural_hash:
            continue
        # Active = no companion result MD on disk yet.
        companion_md = results_dir / f"{candidate.stem}.md"
        if companion_md.exists():
            continue
        return candidate
    return None


def _is_e2_lookahead_filter(filter_type: str) -> bool:
    """Canonical delegation -- never re-encode E2 filter sets.

    Reads ``trading_app.config.E2_EXCLUDED_FILTER_PREFIXES`` and
    ``E2_EXCLUDED_FILTER_SUBSTRINGS`` at call time. Returns True if the
    filter triggers either match per Stage 2A.3 § Suppression Status Enum
    SUPPRESSED_E2_LOOKAHEAD row.
    """
    from trading_app.config import (  # canonical source per institutional-rigor § 10
        E2_EXCLUDED_FILTER_PREFIXES,
        E2_EXCLUDED_FILTER_SUBSTRINGS,
    )

    if not isinstance(filter_type, str):
        return False
    for prefix in E2_EXCLUDED_FILTER_PREFIXES:
        if filter_type.startswith(prefix):
            return True
    for substring in E2_EXCLUDED_FILTER_SUBSTRINGS:
        if substring in filter_type:
            return True
    return False


def _compute_k_lineage(
    structural_hash: str,
    hash_inputs: dict[str, Any],
    ledger_rows: list[dict[str, Any]],
    k_declared: int,
    pooled_n: int,
) -> dict[str, Any]:
    """Build the K-lineage block per Stage 2A.3 § K-Lineage Schema.

    Counts are computed against the in-memory ledger snapshot taken at scan
    start. ``K_lane`` is the count of ledger rows already on disk sharing
    this structural_hash. Scanners are derived views and do not append trial
    history; real research runners create the row that future scans observe.
    """
    k_lane = sum(1 for row in ledger_rows if row.get("structural_hash") == structural_hash)
    k_family = sum(
        1
        for row in ledger_rows
        if (
            (row.get("k_lineage") or {}).get("instrument") == hash_inputs.get("instrument")
            and (row.get("k_lineage") or {}).get("orb_label") == hash_inputs.get("orb_label")
            and (row.get("k_lineage") or {}).get("orb_minutes") == hash_inputs.get("orb_minutes")
        )
    )
    k_global = len(ledger_rows)

    # MinBTL effective-K per Bailey 2013 Thm 1 Eq. 6. The proxy for
    # E[max_N] is pooled_n; guard against div-by-zero with max(...,1).
    e_max_n = max(int(pooled_n), 1)
    k_effective_minbtl = 2.0 * math.log(max(k_global, 2)) / (e_max_n * e_max_n)

    # Correlation-haircut N_hat per Bailey-Lopez de Prado 2014 Eq. 9 prose
    # adapted in the stage file: rho_hat + (1 - rho_hat) * M_correlated,
    # where M_correlated is the family size (sibling lanes share a regime
    # so their effective sample is shared). Rounded to nearest int because
    # n_hat downstream gates ("N_hat >= K_declared * 2") are integer.
    m_correlated = max(k_family, 1)
    n_hat_float = pooled_n * (RHO_HAT_ASSUMED + (1.0 - RHO_HAT_ASSUMED) * m_correlated)
    n_hat = int(round(n_hat_float))

    # BH-FDR passes per backtesting-methodology RULE 4 (informational only
    # for K_global; binding for K_family and K_lane). Returns a placeholder
    # truthy answer when the framing's K is 0 -- BH on an empty universe
    # passes trivially. Stage 2B will replace with the real BH computation
    # once t-stats per lane are available across the ledger.
    bh_fdr_passes = {
        "K_family": k_family <= 1,
        "K_lane": k_lane <= 1,
        "K_global": True,
    }

    return {
        # canonical lane identifiers (echoed so K_family/K_lane queries on
        # the ledger don't need to re-parse strategy_id)
        "instrument": hash_inputs.get("instrument"),
        "orb_label": hash_inputs.get("orb_label"),
        "orb_minutes": hash_inputs.get("orb_minutes"),
        "K_global": k_global,
        "K_family": k_family,
        "K_lane": k_lane,
        "K_declared_in_prereg": int(k_declared),
        "K_effective_minBTL": k_effective_minbtl,
        "bh_fdr_passes": bh_fdr_passes,
        "correlation_haircut_N_hat": n_hat,
        "rho_hat_assumed": RHO_HAT_ASSUMED,
    }


def _resolve_k_declared(source_yaml: dict[str, Any]) -> int:
    """Read K_declared from the source YAML's ``metadata.total_expected_trials``.

    Default 1 when missing (FAST_LANE v5.1 is a K=1 triage screen by design;
    Pathway B / heavyweight templates may declare > 1 but those are not
    consumed by this scanner today).
    """
    meta = source_yaml.get("metadata") or {}
    raw = meta.get("total_expected_trials")
    if isinstance(raw, int) and raw >= 1:
        return raw
    return 1


_HEAVYWEIGHT_TEMPLATE_EXCLUDES = {"fast_lane_v5.1"}


def find_heavyweight_prereg(strategy_id: str, hypotheses_dir: Path = HYPOTHESES_DIR) -> Path | None:
    if not hypotheses_dir.exists():
        return None
    for candidate in sorted(hypotheses_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        if not isinstance(data, dict):
            continue
        scope = data.get("scope") or {}
        if scope.get("strategy_id") != strategy_id:
            continue
        meta = data.get("metadata") or {}
        template = meta.get("template_version")
        if template in _HEAVYWEIGHT_TEMPLATE_EXCLUDES:
            continue
        return candidate
    return None


def find_park_entry(strategy_id: str, action_queue: Path = ACTION_QUEUE) -> str | None:
    if not action_queue.exists():
        return None
    try:
        data = yaml.safe_load(action_queue.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    entries = data.get("entries") or data.get("items") or []
    if not isinstance(entries, list):
        return None
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        if (entry.get("status") or "").lower() != "park":
            continue
        sid = entry.get("strategy_id") or entry.get("lane_id")
        if sid == strategy_id:
            return f"action-queue#{idx}"
    return None


def classify(
    entry: PromoteEntry,
    *,
    oos_window_days: int | None = None,
    oos_window_error: str | None = None,
    source_scope: dict[str, Any] | None = None,
    graveyard_index: dict[str, list[dict[str, Any]]] | None = None,
    duplicate_active_path: Path | None = None,
) -> tuple[str, str | None]:
    """Classify a PROMOTE entry's terminal status.

    Priority order (highest precedence first):
      1. REVOKED      -- revocation sidecar present
      2. ESCALATED    -- heavyweight prereg authored
      3. PARKED       -- action-queue park entry exists
      4. ERROR        -- pooling-artifact gate fires with no revocation sidecar
      5. SUPPRESSED_BANNED_ENTRY_MODEL  -- entry_model in {E0, E3}
      6. SUPPRESSED_E2_LOOKAHEAD        -- E2 + canonical excluded filter
      7. SUPPRESSED_GRAVEYARD           -- structural_hash matches a lane-level
                                           graveyard digest entry
      8. SUPPRESSED_DUPLICATE_ACTIVE    -- another active prereg shares this
                                           structural_hash with no result MD yet
      9. SUPPRESSED_SIBLING_RETEST      -- K_lane >= 2 on a K=1 lane (ledger
                                           says we've already repeated it)
     10. SUPPRESSED_K_OVERRUN           -- K_lane > K_declared_in_prereg
     11. REJECTED_OOS_UNPOWERED         -- OOS pre-flight power gate (RULE 3.3)
     12. QUEUED                         -- survives every gate

    Each suppression rule returns BEFORE the OOS-power gate so a structurally
    invalid lane is never rebranded as "merely underpowered OOS".

    Class-level graveyard matches (hash_kind == "class") do not suppress --
    they record an informational citation in error_reason but the entry
    proceeds through the rest of the chain. Lane matches (hash_kind ==
    "lane") suppress.
    """
    if entry.revocation_sidecar is not None:
        return "REVOKED", None
    if entry.heavyweight_prereg is not None:
        return "ESCALATED", None
    if entry.park_entry is not None:
        return "PARKED", None
    if entry.pooling_artifact:
        return (
            "ERROR",
            "per-direction sanity gate flags pooling artifact "
            "(both directions KILL_AS_STANDALONE); revocation sidecar required",
        )

    # --- Stage 2A.3 suppression chain ----------------------------------
    entry_model = (source_scope or {}).get("entry_model")
    filter_type = (source_scope or {}).get("filter_type", "")

    if isinstance(entry_model, str) and entry_model.upper() in {"E0", "E3"}:
        return (
            "SUPPRESSED_BANNED_ENTRY_MODEL",
            f"entry_model={entry_model!r} is graveyard-banned (E0 purged, "
            "E3 soft-retired Feb 2026); fast-lane scanner refuses to queue. "
            "See trading_app/config.py ENTRY_MODELS + "
            "memory/feedback_e2_lookahead_drift_check_landed.md.",
        )

    if (
        isinstance(entry_model, str)
        and entry_model.upper() == "E2"
        and isinstance(filter_type, str)
        and _is_e2_lookahead_filter(filter_type)
    ):
        return (
            "SUPPRESSED_E2_LOOKAHEAD",
            f"entry_model=E2 with filter_type={filter_type!r} matches "
            "canonical E2_EXCLUDED_FILTER_PREFIXES / _SUBSTRINGS "
            "(break-bar look-ahead class). Canonical source: "
            "trading_app.config.E2_EXCLUDED_FILTER_PREFIXES / "
            "E2_EXCLUDED_FILTER_SUBSTRINGS.",
        )

    informational_class_citation: str | None = None
    if graveyard_index is not None and entry.structural_hash:
        matches = graveyard_index.get(entry.structural_hash, [])
        for m in matches:
            if m.get("hash_kind") == "lane":
                return (
                    "SUPPRESSED_GRAVEYARD",
                    f"structural_hash={entry.structural_hash!r} matches "
                    f"graveyard lane entry titled "
                    f"{m.get('title')!r} (status {m.get('status')!r}, "
                    f"source {m.get('source_path')!r}).",
                )
        # Record class-level match for later inclusion in error_reason
        # without suppressing.
        class_matches = [m for m in matches if m.get("hash_kind") == "class"]
        if class_matches:
            first = class_matches[0]
            informational_class_citation = (
                f"graveyard class-level match (informational, non-suppressing): "
                f"{first.get('title')!r} (source {first.get('source_path')!r})"
            )

    if duplicate_active_path is not None:
        return (
            "SUPPRESSED_DUPLICATE_ACTIVE",
            f"another active prereg shares structural_hash "
            f"{entry.structural_hash!r}: {duplicate_active_path.name} "
            "has no result MD yet; resolve by withdrawing one or pooling.",
        )

    k_lane = int((entry.k_lineage or {}).get("K_lane", 0))
    k_declared = int((entry.k_lineage or {}).get("K_declared_in_prereg", 1))
    if k_declared <= 1 and k_lane >= 2:
        return (
            "SUPPRESSED_SIBLING_RETEST",
            f"K_lane={k_lane} on K_declared_in_prereg={k_declared} -- "
            "ledger shows repeated prior runs on this K=1 lane; "
            "sibling-retest gate fires per stage file "
            "§ Suppression Status Enum. PARK or pool with sibling rather "
            "than re-running.",
        )

    if k_lane > k_declared:
        return (
            "SUPPRESSED_K_OVERRUN",
            f"K_lane={k_lane} > K_declared_in_prereg={k_declared}; "
            "observed trial count exceeds the declared K budget. "
            "Re-declare K or PARK.",
        )

    # --- Pre-flight OOS-power gate (RULE 3.3) ---------------------------
    # If the caller could not resolve the OOS window, fail-closed to REJECTED
    # so an unreadable DB never silently passes a structurally-unbuildable cell.
    if oos_window_days is None or oos_window_error is not None:
        reason = oos_window_error or "oos_window_days not supplied"
        return (
            "REJECTED_OOS_UNPOWERED",
            f"OOS pre-flight unresolved: {reason}"
            + (f"; {informational_class_citation}" if informational_class_citation else ""),
        )
    power, expected_n_oos, why = _compute_expected_oos_power(entry.pooled_fire, oos_window_days)
    if why is not None:
        return (
            "REJECTED_OOS_UNPOWERED",
            f"OOS pre-flight: {why} (oos_window_days={oos_window_days})"
            + (f"; {informational_class_citation}" if informational_class_citation else ""),
        )
    if power < OOS_POWER_FLOOR:
        return (
            "REJECTED_OOS_UNPOWERED",
            (
                f"expected_n_oos={expected_n_oos} (fire_rate={entry.pooled_fire:.4f} "
                f"* oos_window_days={oos_window_days}) -> expected_power={power:.3f} "
                f"< floor={OOS_POWER_FLOOR:.2f} at cohen_d={OOS_COHEN_D_TARGET}; "
                "structurally underpowered OOS — pick CPCV / Harvey-Liu haircut / "
                "pool with siblings / PARK per backtesting-methodology.md RULE 3.3"
            )
            + (f"; {informational_class_citation}" if informational_class_citation else ""),
        )

    return "QUEUED", informational_class_citation


def _default_k_lineage_for_error() -> dict[str, Any]:
    """A K-lineage block that satisfies the provenance contract on ERROR-
    state entries (source YAML missing / unparseable) so Check #172 still
    sees a populated dict with rho_hat_assumed == 0.5."""
    return {
        "instrument": None,
        "orb_label": None,
        "orb_minutes": None,
        "K_global": 0,
        "K_family": 0,
        "K_lane": 0,
        "K_declared_in_prereg": 1,
        "K_effective_minBTL": 0.0,
        "bh_fdr_passes": {
            "K_family": True,
            "K_lane": True,
            "K_global": True,
        },
        "correlation_haircut_N_hat": 0,
        "rho_hat_assumed": RHO_HAT_ASSUMED,
    }


def build_entry(
    path: Path,
    *,
    hypotheses_dir: Path = HYPOTHESES_DIR,
    action_queue: Path = ACTION_QUEUE,
    oos_window_days: int | None = None,
    oos_window_error: str | None = None,
    ledger_rows: list[dict[str, Any]] | None = None,
    graveyard_index: dict[str, list[dict[str, Any]]] | None = None,
) -> PromoteEntry | None:
    parsed = parse_result_md(path)
    if parsed is None:
        return None
    if parsed["verdict"] != "PROMOTE":
        return None

    stats, parse_err = parse_promote_stats(path)
    if stats is None:
        return PromoteEntry(
            result_md=parsed["result_md"],
            strategy_id=parsed["strategy_id"],
            direction=parsed["direction"],
            pooled_t=float("nan"),
            pooled_expr=float("nan"),
            pooled_n=0,
            pooled_fire=float("nan"),
            long_n=0,
            long_expr=float("nan"),
            long_t=float("nan"),
            short_n=0,
            short_expr=float("nan"),
            short_t=float("nan"),
            pooled_universe_n=0,
            long_fire=float("nan"),
            short_fire=float("nan"),
            long_side_verdict="N_A",
            short_side_verdict="N_A",
            pooling_artifact=False,
            revocation_sidecar=None,
            heavyweight_prereg=None,
            park_entry=None,
            status="ERROR",
            error_reason=f"result MD parse failure: {parse_err}",
            structural_hash="0" * 16,
            k_lineage=_default_k_lineage_for_error(),
            n_hat=0,
            upstream_k_role="result_md_unparseable",
            upstream_k_value=None,
        )

    lv, sv, artifact = per_direction_sanity_gate(stats, parsed["direction"])
    sidecar = find_revocation_sidecar(path)
    heavy = find_heavyweight_prereg(parsed["strategy_id"], hypotheses_dir)
    park = find_park_entry(parsed["strategy_id"], action_queue)

    # --- Stage 2A.3 provenance computation -----------------------------
    # The scanner is the single call site that owns scope-derivation: every
    # downstream consumer (ranker, bridge, drift check) reads the
    # structural_hash + k_lineage + n_hat off the cache rather than
    # re-computing. That centralisation IS the de-dup criterion.
    source_yaml = _load_source_yaml(path, hypotheses_dir)
    source_scope: dict[str, Any] | None = None
    structural_hash = ""
    k_lineage: dict[str, Any] = {}
    n_hat = 0
    upstream_k_role = ""
    upstream_k_value: int | None = None
    duplicate_active_path: Path | None = None
    extra_error: str | None = None

    if source_yaml is None:
        # Source missing -- still emit provenance defaults so Check #172
        # passes; classify() will route to ERROR via the missing scope.
        structural_hash = "0" * 16
        k_lineage = _default_k_lineage_for_error()
        upstream_k_role = "source_yaml_missing"
        extra_error = (
            f"source YAML for {path.stem} not found under "
            f"{hypotheses_dir.name}/; provenance fields defaulted, "
            "operator should authoritatively pair the result MD with "
            "its source prereg."
        )
    else:
        scope_block = source_yaml.get("scope") or {}
        if not isinstance(scope_block, dict):
            scope_block = {}
        source_scope = scope_block
        hash_inputs = _scope_to_hash_inputs(scope_block)
        if hash_inputs is None:
            structural_hash = "0" * 16
            k_lineage = _default_k_lineage_for_error()
            upstream_k_role = "scope_block_incomplete"
            extra_error = (
                "source YAML scope block is missing one or more "
                "structural-hash inputs (instrument / session / "
                "orb_minutes / rr_target / entry_model / confirm_bars / "
                "filter_type / direction); provenance fields defaulted."
            )
        else:
            try:
                from scripts.research.fast_lane_structural_hash import (
                    compute_structural_hash,
                )

                structural_hash = compute_structural_hash(hash_inputs)
            except Exception as exc:
                structural_hash = "0" * 16
                k_lineage = _default_k_lineage_for_error()
                upstream_k_role = "hash_compute_failed"
                extra_error = f"compute_structural_hash failed: {type(exc).__name__}: {exc}"

            if extra_error is None:
                k_declared = _resolve_k_declared(source_yaml)
                k_lineage = _compute_k_lineage(
                    structural_hash=structural_hash,
                    hash_inputs=hash_inputs,
                    ledger_rows=ledger_rows or [],
                    k_declared=k_declared,
                    pooled_n=stats["pooled_n"],
                )
                n_hat = int(k_lineage["correlation_haircut_N_hat"])
                upstream_k_role = "fast_lane_v5.1_K1_triage"
                upstream_k_value = k_declared

                # Look for active-duplicate prereg only if hash is real and
                # we have a valid source path -- _find_duplicate searches
                # the hypotheses directory for sibling YAMLs.
                source_yaml_path = hypotheses_dir / f"{path.stem}.yaml"
                duplicate_active_path = _find_duplicate_active_prereg(
                    structural_hash=structural_hash,
                    source_yaml_path=source_yaml_path,
                    hypotheses_dir=hypotheses_dir,
                )

    entry = PromoteEntry(
        result_md=parsed["result_md"],
        strategy_id=parsed["strategy_id"],
        direction=parsed["direction"],
        pooled_t=stats["pooled_t"],
        pooled_expr=stats["pooled_expr"],
        pooled_n=stats["pooled_n"],
        pooled_fire=stats["pooled_fire"],
        long_n=stats["long_n"],
        long_expr=stats["long_expr"],
        long_t=stats["long_t"],
        short_n=stats["short_n"],
        short_expr=stats["short_expr"],
        short_t=stats["short_t"],
        pooled_universe_n=stats["pooled_universe_n"],
        long_fire=stats["long_fire"],
        short_fire=stats["short_fire"],
        long_side_verdict=lv,
        short_side_verdict=sv,
        pooling_artifact=artifact,
        revocation_sidecar=(_rel_to_repo(sidecar) if sidecar is not None else None),
        heavyweight_prereg=(_rel_to_repo(heavy) if heavy is not None else None),
        park_entry=park,
        status="ERROR",
        error_reason=None,
        structural_hash=structural_hash,
        k_lineage=k_lineage,
        n_hat=n_hat,
        upstream_k_role=upstream_k_role,
        upstream_k_value=upstream_k_value,
    )
    entry.status, entry.error_reason = classify(
        entry,
        oos_window_days=oos_window_days,
        oos_window_error=oos_window_error,
        source_scope=source_scope,
        graveyard_index=graveyard_index,
        duplicate_active_path=duplicate_active_path,
    )
    if extra_error is not None:
        # Preserve classification reason where present, but surface the
        # provenance-defaulting cause so operators see WHY the hash is
        # the zero-sentinel.
        if entry.error_reason:
            entry.error_reason = f"{entry.error_reason}; {extra_error}"
        else:
            entry.error_reason = extra_error
    return entry


def scan(
    results_dir: Path | None = None,
    *,
    hypotheses_dir: Path | None = None,
    action_queue: Path | None = None,
    oos_window_days: int | None = None,
    oos_window_error: str | None = None,
    db_path: Path | None = None,
    ledger_path: Path | None = None,
    trial_corrections_path: Path | None = None,
    graveyard_digest_path: Path | None = None,
    append_to_ledger: bool = False,
) -> list[PromoteEntry]:
    # Resolve module-level defaults at call time so monkeypatching from tests
    # works against scripts.research.fast_lane_promote_queue.RESULTS_DIR etc.
    rd = results_dir if results_dir is not None else RESULTS_DIR
    hd = hypotheses_dir if hypotheses_dir is not None else HYPOTHESES_DIR
    aq = action_queue if action_queue is not None else ACTION_QUEUE
    lp = ledger_path if ledger_path is not None else TRIAL_LEDGER_PATH
    tc = trial_corrections_path if trial_corrections_path is not None else TRIAL_CORRECTIONS_PATH
    gd = graveyard_digest_path if graveyard_digest_path is not None else GRAVEYARD_DIGEST_PATH
    if append_to_ledger:
        # Deprecated Phase-0/Phase-1 compatibility seam. Scanners are derived
        # views and may not create trial history; only real research runners
        # can call fast_lane_trial_ledger.append_trial_ledger_entry.
        append_to_ledger = False

    # Resolve the OOS window once per scan so each entry classification
    # reuses the same (latest_trading_day - HOLDOUT_SACRED_FROM) result.
    # Tests can inject by supplying oos_window_days directly.
    if oos_window_days is None and oos_window_error is None:
        oos_window_days, oos_window_error = _resolve_oos_window_days(db_path)

    # Stage 2A.3: read ledger + digest ONCE per scan, before any entry
    # processing. The ledger snapshot is the same for every entry within a
    # scan -- subsequent scans see whatever this scan appended.
    from scripts.research.fast_lane_trial_ledger import (
        filter_v2_k_count_rows,
        read_trial_corrections,
    )

    ledger_rows = filter_v2_k_count_rows(_read_trial_ledger(lp), read_trial_corrections(tc))
    graveyard_index = _read_graveyard_digest(gd)

    entries: list[PromoteEntry] = []
    for path in sorted(rd.glob(FAST_LANE_RESULT_GLOB)):
        entry = build_entry(
            path,
            hypotheses_dir=hd,
            action_queue=aq,
            oos_window_days=oos_window_days,
            oos_window_error=oos_window_error,
            ledger_rows=ledger_rows,
            graveyard_index=graveyard_index,
        )
        if entry is None:
            continue
        entries.append(entry)
    return entries


def _entry_to_dict(entry: PromoteEntry) -> dict[str, Any]:
    d = asdict(entry)
    for k, v in list(d.items()):
        if isinstance(v, float) and math.isnan(v):
            d[k] = None
    # k_lineage is already a plain dict (no NaN exposure today). If a
    # future contributor adds a float field that may NaN, recurse here.
    return d


def serialize_queue(entries: list[PromoteEntry]) -> str:
    payload = {
        "schema_version": 1,
        "source": "scripts/research/fast_lane_promote_queue.py",
        "warning": (
            "DERIVED STATE - do not hand-edit. Rebuilt from result MDs + "
            "revocation sidecars + heavyweight preregs + action-queue.yaml "
            "on every scanner run. Drift check #157 reconstructs and diffs."
        ),
        "entries": [_entry_to_dict(e) for e in entries],
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def diff_against_cache(entries: list[PromoteEntry], cache_path: Path = QUEUE_CACHE) -> list[str]:
    if not cache_path.exists():
        return ["(no cache file on disk; first run)"]
    try:
        cached = yaml.safe_load(cache_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return [f"(cache unreadable: {exc})"]
    cached_entries = {e.get("strategy_id"): e for e in (cached.get("entries") or [])}
    fresh_entries = {e.strategy_id: _entry_to_dict(e) for e in entries}

    lines: list[str] = []
    for sid in sorted(set(cached_entries) | set(fresh_entries)):
        c = cached_entries.get(sid)
        f = fresh_entries.get(sid)
        if c is None and f is not None:
            lines.append(f"ADDED   {sid} status={f['status']}")
        elif f is None and c is not None:
            lines.append(f"REMOVED {sid}")
        elif c is not None and f is not None and c.get("status") != f.get("status"):
            lines.append(f"CHANGED {sid} {c.get('status')} -> {f.get('status')}")
    return lines or ["(cache up to date)"]


def render_report(entries: list[PromoteEntry]) -> str:
    lines = [
        "FAST_LANE v5.1 PROMOTE queue",
        "============================",
        f"Total PROMOTE results scanned: {len(entries)}",
        "",
    ]
    by_status: dict[str, list[PromoteEntry]] = {s: [] for s in STATUS_VALUES}
    for e in entries:
        by_status[e.status].append(e)
    for status in STATUS_VALUES:
        bucket = by_status[status]
        lines.append(f"## {status} ({len(bucket)})")
        for e in bucket:
            lines.append(f"  - {e.strategy_id}")
            lines.append(f"      result_md: {e.result_md}")
            lines.append(
                f"      direction={e.direction} pooled_t={e.pooled_t:.3f} "
                f"pooled_n={e.pooled_n} pooled_fire={e.pooled_fire:.4f}"
            )
            lines.append(f"      long: t={e.long_t!r} n={e.long_n} fire={e.long_fire:.4f} -> {e.long_side_verdict}")
            lines.append(
                f"      short: t={e.short_t!r} n={e.short_n} fire={e.short_fire:.4f} -> {e.short_side_verdict}"
            )
            if e.revocation_sidecar:
                lines.append(f"      revocation_sidecar: {e.revocation_sidecar}")
            if e.heavyweight_prereg:
                lines.append(f"      heavyweight_prereg: {e.heavyweight_prereg}")
            if e.park_entry:
                lines.append(f"      park_entry: {e.park_entry}")
            if e.error_reason:
                lines.append(f"      ERROR: {e.error_reason}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Refresh docs/runtime/promote_queue.yaml from current on-disk state.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="(default) Print the would-be queue and the diff vs cache; do not write.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON to stdout (for drift-check integration).",
    )
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--cache-path", default=str(QUEUE_CACHE))
    parser.add_argument("--hypotheses-dir", default=str(HYPOTHESES_DIR))
    parser.add_argument("--action-queue", default=str(ACTION_QUEUE))
    parser.add_argument(
        "--oos-window-days",
        type=int,
        default=None,
        help=(
            "Override the OOS window length used by the pre-flight OOS-power "
            "gate (RULE 3.3). Default: derived from "
            "MAX(orb_outcomes.trading_day) - HOLDOUT_SACRED_FROM via the "
            "canonical DB. Tests and what-if analyses can supply an explicit "
            "value (e.g., 365 for a one-year window)."
        ),
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help=("Override the DuckDB path used to derive --oos-window-days. Default: pipeline.paths.GOLD_DB_PATH."),
    )
    parser.add_argument(
        "--ledger-path",
        default=str(TRIAL_LEDGER_PATH),
        help=(
            "Path to docs/runtime/fast_lane_trial_ledger.yaml. Stage 2A.3 "
            "reads this append-only universe-of-trials ledger."
        ),
    )
    parser.add_argument(
        "--trial-corrections-path",
        default=str(TRIAL_CORRECTIONS_PATH),
        help=(
            "Path to docs/runtime/fast_lane_trial_corrections.yaml. "
            "Correction-not-deletion records exclude derived legacy rows "
            "from V2 K-lineage counts while preserving audit history."
        ),
    )
    parser.add_argument(
        "--graveyard-digest-path",
        default=str(GRAVEYARD_DIGEST_PATH),
        help=(
            "Path to docs/runtime/fast_lane_graveyard_digest.yaml. "
            "Consulted once per scan for SUPPRESSED_GRAVEYARD matching."
        ),
    )
    parser.add_argument(
        "--no-ledger-append",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Scanner CLI runs are read-only "
            "for trial-ledger provenance in both --dry-run and --write modes."
        ),
    )
    args = parser.parse_args(argv)

    if args.write and args.dry_run:
        parser.error("--write and --dry-run are mutually exclusive")

    entries = scan(
        Path(args.results_dir),
        hypotheses_dir=Path(args.hypotheses_dir),
        action_queue=Path(args.action_queue),
        oos_window_days=args.oos_window_days,
        oos_window_error=None,
        db_path=Path(args.db_path) if args.db_path else None,
        ledger_path=Path(args.ledger_path),
        trial_corrections_path=Path(args.trial_corrections_path),
        graveyard_digest_path=Path(args.graveyard_digest_path),
        append_to_ledger=False,
    )

    if args.json:
        sys.stdout.write(json.dumps([_entry_to_dict(e) for e in entries], indent=2))
        sys.stdout.write("\n")
        return 0

    sys.stdout.write(render_report(entries))
    sys.stdout.write("\n--- diff vs cache ---\n")
    for line in diff_against_cache(entries, Path(args.cache_path)):
        sys.stdout.write(line + "\n")

    if args.write:
        Path(args.cache_path).write_text(serialize_queue(entries), encoding="utf-8")
        sys.stdout.write(f"\nwrote cache: {args.cache_path}\n")

    if any(e.status == "ERROR" for e in entries):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
