"""Scanner integration tests for Stage 2A.3 suppression rules.

Six tests, one per row in
``docs/runtime/stages/2026-05-20-fast-lane-anti-fp-2a3-scanner-bridge-wiring.md``
§ "Suppression Status Enum". Each test:

  1. Builds a synthetic result MD with PROMOTE verdict + sane stats so the
     scanner survives the OOS-power gate (or supplies oos_window_days large
     enough to give power >= 0.5).
  2. Builds a matching source YAML under ``hypotheses_dir`` so the scanner
     can compute structural_hash.
  3. Optionally seeds ledger entries / graveyard digest / sibling YAMLs to
     trigger one suppression rule.
  4. Calls ``scan(append_to_ledger=False)`` and asserts the emitted entry's
     status is the expected SUPPRESSED_* value and provenance fields are
     populated.

Builds use the canonical FAST_LANE v5.1 result-MD format because the
scanner's parser is strict; rather than reimplement the format, we
generate small fixtures that match the regexes in
``fast_lane_promote_queue.parse_result_md / parse_promote_stats``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.research.fast_lane_promote_queue import scan
from scripts.research.fast_lane_structural_hash import compute_structural_hash

# Large OOS window so the OOS-power gate passes when stats are sane;
# tests that intentionally trip the pre-power suppression chain do not
# care about this, but baseline-QUEUED tests do.
LARGE_OOS_WINDOW_DAYS = 3650


def _make_result_md(
    tmp_path: Path,
    *,
    stem: str,
    strategy_id: str,
    direction: str = "long",
    pooled_n: int = 220,
    pooled_t: float = 3.20,
    pooled_expr: float = 0.18,
    fire_pct: float = 14.5,
    long_n: int = 220,
    long_expr: float = 0.18,
    long_t: float = 3.20,
    short_n: int = 0,
    short_expr: float = 0.0,
    short_t: float = 0.0,
    pooled_universe_n: int = 1500,
) -> Path:
    """Emit a minimal result MD matching the scanner's regex shape."""
    # Default short-side stats are zero -- on a single-direction lane this
    # is realistic; the per-direction sanity gate will mark short_n=0 as
    # N_A, the long side as PROMOTE_AS_STANDALONE (if t >= 3.0), and the
    # pooled row will fall through to QUEUED.
    md = textwrap.dedent(
        f"""\
        # Chordia strict unlock audit · {strategy_id}

        **FAST_LANE verdict:** `PROMOTE`

        ## Split summary

        | row | nu | nf | fp | a | b | expr | c | d | t |
        |-----|-----|-----|-----|---|---|------|---|---|---|
        | IS  | {pooled_universe_n} | {pooled_n} | {fire_pct}% | _ | _ | {pooled_expr} | _ | _ | {pooled_t} |
        | OOS | 50 | 30 | 6.0% | _ | _ | 0.10 | _ | _ | 2.10 |

        ## Directional breakdown

        | row | ln | lex | lt | sn | sex | st |
        |-----|-----|-----|----|----|-----|----|
        | IS  | {long_n} | {long_expr} | {long_t} | {short_n} | {short_expr} | {short_t} |

        _Scope direction at screen: `'{direction}'`_
        """
    )
    path = tmp_path / f"{stem}.md"
    path.write_text(md, encoding="utf-8")
    return path


def _make_source_yaml(
    tmp_path: Path,
    *,
    stem: str,
    strategy_id: str,
    instrument: str = "MGC",
    session: str = "LONDON_METALS",
    orb_minutes: int = 30,
    entry_model: str = "E1",
    confirm_bars: int = 2,
    rr_target: float = 1.0,
    direction: str = "long",
    filter_type: str = "ATR_P50",
    extras: dict[str, Any] | None = None,
) -> Path:
    """Emit a v5.1-shaped source YAML under tmp_path/hypotheses/."""
    scope = {
        "instrument": instrument,
        "strategy_id": strategy_id,
        "session": session,
        "orb_minutes": orb_minutes,
        "entry_model": entry_model,
        "confirm_bars": confirm_bars,
        "rr_target": rr_target,
        "direction": direction,
        "filter_type": filter_type,
    }
    metadata: dict[str, Any] = {
        "theory_grant": False,
        "name": stem.replace("-", "_"),
        "total_expected_trials": 1,
        "testing_mode": "individual",
        "template_version": "fast_lane_v5.1",
    }
    if extras:
        scope.update(extras)
    data = {"metadata": metadata, "scope": scope}
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir(parents=True, exist_ok=True)
    out = hyp_dir / f"{stem}.yaml"
    out.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return out


def _make_action_queue(tmp_path: Path) -> Path:
    aq_path = tmp_path / "action-queue.yaml"
    aq_path.write_text("entries: []\n", encoding="utf-8")
    return aq_path


def _make_empty_ledger(tmp_path: Path) -> Path:
    lp = tmp_path / "fast_lane_trial_ledger.yaml"
    lp.write_text(
        "do_not_hand_edit: true\nschema_version: 1\nentries: []\n",
        encoding="utf-8",
    )
    return lp


def _make_digest(tmp_path: Path, entries: list[dict[str, Any]]) -> Path:
    dp = tmp_path / "fast_lane_graveyard_digest.yaml"
    payload = {
        "schema_version": 1,
        "do_not_hand_edit": True,
        "entries": entries,
    }
    dp.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return dp


def _scope_to_inputs(scope: dict[str, Any]) -> dict[str, Any]:
    raw_dir = scope.get("direction", "BOTH")
    if isinstance(raw_dir, str) and raw_dir.strip().lower() == "pooled":
        raw_dir = "BOTH"
    return {
        "instrument": scope["instrument"],
        "orb_label": scope["session"],
        "orb_minutes": int(scope["orb_minutes"]),
        "rr_target": float(scope["rr_target"]),
        "entry_model": scope["entry_model"],
        "confirm_bars": int(scope["confirm_bars"]),
        "filter_type": scope.get("filter_type", ""),
        "direction": raw_dir,
        "filter_threshold": scope.get("filter_threshold", ""),
    }


# ---- 1. SUPPRESSED_BANNED_ENTRY_MODEL ------------------------------------


def test_suppressed_banned_entry_model_e3(tmp_path):
    """An E3 entry triggers SUPPRESSED_BANNED_ENTRY_MODEL."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    stem = "2026-05-20-e3-banned-fast-lane-v1"
    _make_result_md(
        results_dir,
        stem=stem,
        strategy_id="MGC_LONDON_METALS_E3_RR1.0_CB1_ATR70_25",
    )
    _make_source_yaml(
        tmp_path,
        stem=stem,
        strategy_id="MGC_LONDON_METALS_E3_RR1.0_CB1_ATR70_25",
        entry_model="E3",
        confirm_bars=1,
    )

    entries = scan(
        results_dir,
        hypotheses_dir=tmp_path / "hypotheses",
        action_queue=_make_action_queue(tmp_path),
        ledger_path=_make_empty_ledger(tmp_path),
        graveyard_digest_path=_make_digest(tmp_path, []),
        oos_window_days=LARGE_OOS_WINDOW_DAYS,
        append_to_ledger=False,
    )

    assert len(entries) == 1
    e = entries[0]
    assert e.status == "SUPPRESSED_BANNED_ENTRY_MODEL", f"got {e.status} reason={e.error_reason}"
    assert len(e.structural_hash) == 16
    assert e.k_lineage.get("rho_hat_assumed") == 0.5
    assert e.n_hat > 0


# ---- 2. SUPPRESSED_E2_LOOKAHEAD ------------------------------------------


def test_suppressed_e2_lookahead(tmp_path):
    """E2 + canonical excluded-prefix filter triggers SUPPRESSED_E2_LOOKAHEAD.

    Uses a real ALL_FILTERS entry that matches E2_EXCLUDED_FILTER_PREFIXES so
    structural_hash compute succeeds; the gate then fires inside classify().
    """
    from trading_app.config import ALL_FILTERS, E2_EXCLUDED_FILTER_PREFIXES

    # Pick the first ALL_FILTERS member matching a canonical excluded
    # prefix; the scanner imports the same canonical tuple at runtime so
    # this test always exercises the live prefix set (never re-encoded).
    filter_type = next(f for f in sorted(ALL_FILTERS) if any(f.startswith(p) for p in E2_EXCLUDED_FILTER_PREFIXES))

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    stem = "2026-05-20-e2-lookahead-fast-lane-v1"
    sid = f"MGC_LONDON_METALS_E2_RR1.0_CB1_{filter_type}"
    _make_result_md(results_dir, stem=stem, strategy_id=sid)
    _make_source_yaml(
        tmp_path,
        stem=stem,
        strategy_id=sid,
        entry_model="E2",
        confirm_bars=1,
        filter_type=filter_type,
    )

    entries = scan(
        results_dir,
        hypotheses_dir=tmp_path / "hypotheses",
        action_queue=_make_action_queue(tmp_path),
        ledger_path=_make_empty_ledger(tmp_path),
        graveyard_digest_path=_make_digest(tmp_path, []),
        oos_window_days=LARGE_OOS_WINDOW_DAYS,
        append_to_ledger=False,
    )

    assert len(entries) == 1
    e = entries[0]
    assert e.status == "SUPPRESSED_E2_LOOKAHEAD", f"got {e.status} reason={e.error_reason}"
    assert len(e.structural_hash) == 16


# ---- 3. SUPPRESSED_GRAVEYARD ---------------------------------------------


def test_suppressed_graveyard_lane_match(tmp_path):
    """A lane-level graveyard hash match triggers SUPPRESSED_GRAVEYARD."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    stem = "2026-05-20-grave-fast-lane-v1"
    sid = "MGC_LONDON_METALS_E1_RR1.0_CB2_ATR_P50"
    _make_result_md(results_dir, stem=stem, strategy_id=sid)
    source_yaml = _make_source_yaml(tmp_path, stem=stem, strategy_id=sid)

    # Compute the structural_hash the scanner will compute, then seed the
    # graveyard digest with a lane-level entry sharing that hash.
    scope = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))["scope"]
    h = compute_structural_hash(_scope_to_inputs(scope))
    digest_entries = [
        {
            "source_path": "chatgpt_bundle/06_RD_GRAVEYARD.md",
            "title": "Synthetic graveyard lane match for tests",
            "status": "DEAD",
            "hash_kind": "lane",
            "structural_hash": h,
            "lane_inputs": {},
        }
    ]

    entries = scan(
        results_dir,
        hypotheses_dir=tmp_path / "hypotheses",
        action_queue=_make_action_queue(tmp_path),
        ledger_path=_make_empty_ledger(tmp_path),
        graveyard_digest_path=_make_digest(tmp_path, digest_entries),
        oos_window_days=LARGE_OOS_WINDOW_DAYS,
        append_to_ledger=False,
    )

    assert len(entries) == 1
    e = entries[0]
    assert e.status == "SUPPRESSED_GRAVEYARD", f"got {e.status} reason={e.error_reason}"
    assert e.structural_hash == h


# ---- 4. SUPPRESSED_DUPLICATE_ACTIVE --------------------------------------


def test_suppressed_duplicate_active(tmp_path):
    """A sibling YAML with the same hash + no result MD triggers
    SUPPRESSED_DUPLICATE_ACTIVE."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    stem = "2026-05-20-dup-fast-lane-v1"
    sid = "MGC_LONDON_METALS_E1_RR1.0_CB2_ATR_P50"
    _make_result_md(results_dir, stem=stem, strategy_id=sid)
    _make_source_yaml(tmp_path, stem=stem, strategy_id=sid)

    # Add a sibling yaml under hypotheses/ with the same scope but no
    # companion result MD. The scanner's _find_duplicate_active_prereg
    # walks the same dir and reports the sibling as an active duplicate.
    _make_source_yaml(
        tmp_path,
        stem="2026-05-20-dup-twin-fast-lane-v1",
        strategy_id=sid,  # same strategy_id is fine; the gate is on hash
    )

    entries = scan(
        results_dir,
        hypotheses_dir=tmp_path / "hypotheses",
        action_queue=_make_action_queue(tmp_path),
        ledger_path=_make_empty_ledger(tmp_path),
        graveyard_digest_path=_make_digest(tmp_path, []),
        oos_window_days=LARGE_OOS_WINDOW_DAYS,
        append_to_ledger=False,
    )

    assert len(entries) == 1
    e = entries[0]
    assert e.status == "SUPPRESSED_DUPLICATE_ACTIVE", f"got {e.status} reason={e.error_reason}"


# ---- 5. SUPPRESSED_SIBLING_RETEST ----------------------------------------


def test_suppressed_sibling_retest_k_lane_ge_2(tmp_path):
    """K_lane >= 2 (>= 2 prior ledger entries) triggers SUPPRESSED_SIBLING_RETEST."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    stem = "2026-05-20-sib-fast-lane-v1"
    sid = "MGC_LONDON_METALS_E1_RR1.0_CB2_ATR_P50"
    _make_result_md(results_dir, stem=stem, strategy_id=sid)
    source_yaml = _make_source_yaml(tmp_path, stem=stem, strategy_id=sid)

    # Seed the ledger with 2 prior entries sharing this structural_hash.
    scope = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))["scope"]
    h = compute_structural_hash(_scope_to_inputs(scope))

    ledger_path = tmp_path / "fast_lane_trial_ledger.yaml"
    ledger_payload = {
        "do_not_hand_edit": True,
        "schema_version": 1,
        "entries": [
            {
                "run_id": "synthetic-1",
                "run_timestamp_utc": "2026-05-19T01:00:00Z",
                "prereg_path": "docs/audit/hypotheses/synthetic-prior-1.yaml",
                "prereg_sha": "a" * 16,
                "structural_hash": h,
                "template_version": "fast_lane_v5.1",
                "testing_mode": "individual",
                "pathway": "A",
                "K_declared": 1,
                "holdout_policy": "mode_A",
                "holdout_sacred_from": "2026-01-01",
                "k_lineage": {
                    "instrument": scope["instrument"],
                    "orb_label": scope["session"],
                    "orb_minutes": int(scope["orb_minutes"]),
                },
                "n_hat": 100.0,
                "upstream_provenance": {},
                "outcome": {},
            },
            {
                "run_id": "synthetic-2",
                "run_timestamp_utc": "2026-05-19T02:00:00Z",
                "prereg_path": "docs/audit/hypotheses/synthetic-prior-2.yaml",
                "prereg_sha": "b" * 16,
                "structural_hash": h,
                "template_version": "fast_lane_v5.1",
                "testing_mode": "individual",
                "pathway": "A",
                "K_declared": 1,
                "holdout_policy": "mode_A",
                "holdout_sacred_from": "2026-01-01",
                "k_lineage": {
                    "instrument": scope["instrument"],
                    "orb_label": scope["session"],
                    "orb_minutes": int(scope["orb_minutes"]),
                },
                "n_hat": 100.0,
                "upstream_provenance": {},
                "outcome": {},
            },
        ],
    }
    ledger_path.write_text(
        "do_not_hand_edit: true\nschema_version: 1\n"
        + yaml.safe_dump(
            {"entries": ledger_payload["entries"]},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    entries = scan(
        results_dir,
        hypotheses_dir=tmp_path / "hypotheses",
        action_queue=_make_action_queue(tmp_path),
        ledger_path=ledger_path,
        graveyard_digest_path=_make_digest(tmp_path, []),
        oos_window_days=LARGE_OOS_WINDOW_DAYS,
        append_to_ledger=False,
    )

    assert len(entries) == 1
    e = entries[0]
    assert e.status == "SUPPRESSED_SIBLING_RETEST", f"got {e.status} reason={e.error_reason}"
    assert e.k_lineage["K_lane"] >= 2


# ---- 6. SUPPRESSED_K_OVERRUN ---------------------------------------------


def test_suppressed_k_overrun(tmp_path):
    """N_hat >= K_declared * 2 triggers SUPPRESSED_K_OVERRUN.

    The synthesis: large pooled_n -> large N_hat under rho_hat=0.5. With
    K_declared=1, the threshold is 2; we need N_hat >= 2. Empty ledger
    means K_family = 0 -> M_correlated = max(K_family, 1) = 1 ->
    N_hat = pooled_n * (0.5 + 0.5*1) = pooled_n. So any pooled_n >= 2
    will trip. Use pooled_n = 220 which trivially exceeds 2.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    stem = "2026-05-20-koverrun-fast-lane-v1"
    sid = "MGC_LONDON_METALS_E1_RR1.0_CB2_ATR_P50"
    _make_result_md(results_dir, stem=stem, strategy_id=sid, pooled_n=220)
    _make_source_yaml(tmp_path, stem=stem, strategy_id=sid)
    # The default _resolve_k_declared returns K_declared=1 because the
    # synthetic source YAML's metadata.total_expected_trials=1. n_hat=220
    # >> K_declared * 2 = 2 -> SUPPRESSED_K_OVERRUN fires.

    entries = scan(
        results_dir,
        hypotheses_dir=tmp_path / "hypotheses",
        action_queue=_make_action_queue(tmp_path),
        ledger_path=_make_empty_ledger(tmp_path),
        graveyard_digest_path=_make_digest(tmp_path, []),
        oos_window_days=LARGE_OOS_WINDOW_DAYS,
        append_to_ledger=False,
    )

    assert len(entries) == 1
    e = entries[0]
    assert e.status == "SUPPRESSED_K_OVERRUN", f"got {e.status} reason={e.error_reason}"
    assert e.n_hat >= 2
