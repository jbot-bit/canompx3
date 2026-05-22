# Fast Lane V2 Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all current fast-lane scan and walk dry-run paths provably non-mutating before expanding the automated trade finder/maker/verifier.

**Architecture:** Preserve current scanner classification behavior, but make ledger append authority explicit and fail-closed. Scanner CLI dry-runs, scanner CLI cache writes, direct `scan()` defaults, and walk dry-runs must all be read-only for trial-ledger provenance until Phase 1 moves trial-event append authority to real research execution.

**Tech Stack:** Python, pytest, YAML runtime artifacts, existing `scripts/research/fast_lane_promote_queue.py`, existing `scripts/tools/fast_lane_walk.py`.

---

## File Structure

- Modify `tests/test_research/test_fast_lane_promote_queue_suppression.py`: add integration tests proving `fast_lane_promote_queue.py --dry-run`, `fast_lane_promote_queue.py --write`, and direct `scan()` default leave the trial ledger byte-for-byte unchanged.
- Modify `tests/test_tools/test_fast_lane_walk.py`: add an orchestrator test proving `run_chain(dry_run=True)` passes `--no-ledger-append` to the promote queue step while preserving non-write behavior for other steps.
- Modify `scripts/research/fast_lane_promote_queue.py`: make direct `scan()` default read-only and make CLI always call `scan(... append_to_ledger=False)`.
- Modify `scripts/tools/fast_lane_walk.py`: add per-step dry-run argument handling so promote queue receives `--no-ledger-append` when walk dry-run composes it.
- Update `HANDOFF.md`: record Phase 0 implementation status and next gates.

## Task 1: Scanner CLI and Default Scan Are Non-Mutating

**Files:**
- Modify: `tests/test_research/test_fast_lane_promote_queue_suppression.py`
- Modify: `scripts/research/fast_lane_promote_queue.py`

- [x] **Step 1: Write the failing scanner CLI test**

Add imports:

```python
import io
from contextlib import redirect_stdout
```

Change the existing import:

```python
from scripts.research.fast_lane_promote_queue import main, scan
```

Add this test near the helper/integration tests:

```python
def test_main_dry_run_does_not_append_to_trial_ledger(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    stem = "2026-05-21-cli-dry-run-fast-lane-v1"
    _make_result_md(
        results_dir,
        stem=stem,
        strategy_id="MGC_LONDON_METALS_E1_RR1.0_CB2_ATR_P50_30",
    )
    _make_source_yaml(
        tmp_path,
        stem=stem,
        strategy_id="MGC_LONDON_METALS_E1_RR1.0_CB2_ATR_P50_30",
    )
    ledger_path = _make_empty_ledger(tmp_path)
    before = ledger_path.read_bytes()

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "--dry-run",
                "--results-dir",
                str(results_dir),
                "--cache-path",
                str(tmp_path / "promote_queue.yaml"),
                "--hypotheses-dir",
                str(tmp_path / "hypotheses"),
                "--action-queue",
                str(_make_action_queue(tmp_path)),
                "--ledger-path",
                str(ledger_path),
                "--graveyard-digest-path",
                str(_make_digest(tmp_path, [])),
                "--oos-window-days",
                str(LARGE_OOS_WINDOW_DAYS),
            ]
        )

    assert rc == 0
    assert "FAST_LANE v5.1 PROMOTE queue" in buf.getvalue()
    assert ledger_path.read_bytes() == before
```

- [x] **Step 2: Run the scanner test and verify it fails red**

Run:

```bash
./.venv-wsl/bin/python -m pytest tests/test_research/test_fast_lane_promote_queue_suppression.py::test_main_dry_run_does_not_append_to_trial_ledger -q
```

Expected: FAIL because the ledger bytes changed.

- [x] **Step 3: Implement the minimal CLI fix**

In `scripts/research/fast_lane_promote_queue.py`, compute append authority after parsing args:

```python
    append_to_ledger = args.write and not args.no_ledger_append
```

Then pass it to `scan`:

```python
        append_to_ledger=append_to_ledger,
```

This makes CLI dry-run read-only without changing the lower-level `scan()` default.

- [x] **Step 4: Run the scanner test and verify green**

Run:

```bash
./.venv-wsl/bin/python -m pytest tests/test_research/test_fast_lane_promote_queue_suppression.py::test_main_dry_run_does_not_append_to_trial_ledger -q
```

Expected: PASS.

- [x] **Step 5: Write the failing scanner CLI write-mode test**

Add this test:

```python
def test_main_write_refreshes_cache_without_appending_trial_ledger(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    stem = "2026-05-21-cli-write-fast-lane-v1"
    _make_result_md(results_dir, stem=stem, strategy_id="MGC_LONDON_METALS_E1_RR1.0_CB2_ATR_P50_30")
    _make_source_yaml(tmp_path, stem=stem, strategy_id="MGC_LONDON_METALS_E1_RR1.0_CB2_ATR_P50_30")
    ledger_path = _make_empty_ledger(tmp_path)
    cache_path = tmp_path / "promote_queue.yaml"
    before = ledger_path.read_bytes()

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "--write",
            "--results-dir", str(results_dir),
            "--cache-path", str(cache_path),
            "--hypotheses-dir", str(tmp_path / "hypotheses"),
            "--action-queue", str(_make_action_queue(tmp_path)),
            "--ledger-path", str(ledger_path),
            "--graveyard-digest-path", str(_make_digest(tmp_path, [])),
            "--oos-window-days", str(LARGE_OOS_WINDOW_DAYS),
        ])

    assert rc == 0
    assert cache_path.exists()
    assert "wrote cache:" in buf.getvalue()
    assert ledger_path.read_bytes() == before
```

- [x] **Step 6: Verify scanner CLI write-mode fails red**

Run:

```bash
./.venv-wsl/bin/python -m pytest tests/test_research/test_fast_lane_promote_queue_suppression.py::test_main_write_refreshes_cache_without_appending_trial_ledger -q
```

Expected before fix: FAIL because `--write` appends trial-ledger rows.

- [x] **Step 7: Write the failing direct `scan()` default test**

Add this test:

```python
def test_scan_default_is_read_only_for_trial_ledger(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    stem = "2026-05-21-scan-default-fast-lane-v1"
    _make_result_md(results_dir, stem=stem, strategy_id="MGC_LONDON_METALS_E1_RR1.0_CB2_ATR_P50_30")
    _make_source_yaml(tmp_path, stem=stem, strategy_id="MGC_LONDON_METALS_E1_RR1.0_CB2_ATR_P50_30")
    ledger_path = _make_empty_ledger(tmp_path)
    before = ledger_path.read_bytes()

    entries = scan(
        results_dir,
        hypotheses_dir=tmp_path / "hypotheses",
        action_queue=_make_action_queue(tmp_path),
        ledger_path=ledger_path,
        graveyard_digest_path=_make_digest(tmp_path, []),
        oos_window_days=LARGE_OOS_WINDOW_DAYS,
    )

    assert len(entries) == 1
    assert entries[0].status != "ERROR"
    assert ledger_path.read_bytes() == before
```

- [x] **Step 8: Verify direct `scan()` default fails red**

Run:

```bash
./.venv-wsl/bin/python -m pytest tests/test_research/test_fast_lane_promote_queue_suppression.py::test_scan_default_is_read_only_for_trial_ledger -q
```

Expected before fix: FAIL because direct `scan()` defaults to appending.

- [x] **Step 9: Implement the broader fail-closed scanner fix**

Change `scan()` default:

```python
    append_to_ledger: bool = False,
```

Then make CLI calls always read-only for ledger provenance:

```python
        append_to_ledger=False,
```

Update `--no-ledger-append` help text to say it is a deprecated compatibility flag because scanner CLI runs are read-only in both `--dry-run` and `--write` modes.

- [x] **Step 10: Run scanner write/default tests and verify green**

Run:

```bash
./.venv-wsl/bin/python -m pytest tests/test_research/test_fast_lane_promote_queue_suppression.py::test_main_write_refreshes_cache_without_appending_trial_ledger tests/test_research/test_fast_lane_promote_queue_suppression.py::test_scan_default_is_read_only_for_trial_ledger -q
```

Expected: PASS.

## Task 2: Walk Dry-Run Propagates Scanner Read-Only Mode

**Files:**
- Modify: `tests/test_tools/test_fast_lane_walk.py`
- Modify: `scripts/tools/fast_lane_walk.py`

- [x] **Step 1: Write the failing walk orchestrator test**

Add this test next to `test_run_chain_dry_run_strips_write_flags`:

```python
def test_run_chain_dry_run_adds_no_ledger_append_to_promote_queue() -> None:
    seen_argv: dict[str, list[str]] = {}

    def capture(label: str) -> Callable[[list[str]], int]:
        def _inner(argv: list[str]) -> int:
            seen_argv[label] = list(argv)
            return 0

        return _inner

    steps = [
        ("promote_queue", capture("promote_queue"), ["--write"]),
        ("cherry_pick_ranker", capture("ranker"), ["--write", "--write-journal"]),
        ("status_rollup", capture("status"), ["--write"]),
    ]

    overall_rc, _ = run_chain(steps=steps, dry_run=True)

    assert overall_rc == 0
    assert seen_argv["promote_queue"] == ["--no-ledger-append"]
    assert seen_argv["ranker"] == []
    assert seen_argv["status"] == []
```

- [x] **Step 2: Run the walk test and verify it fails red**

Run:

```bash
./.venv-wsl/bin/python -m pytest tests/test_tools/test_fast_lane_walk.py::test_run_chain_dry_run_adds_no_ledger_append_to_promote_queue -q
```

Expected: FAIL because promote queue receives `[]`.

- [x] **Step 3: Implement dry-run argument normalization**

In `scripts/tools/fast_lane_walk.py`, add:

```python
def _dry_run_argv(label: str, argv: list[str]) -> list[str]:
    effective_argv = [a for a in argv if not a.startswith("--write")]
    if label == "promote_queue" and "--no-ledger-append" not in effective_argv:
        effective_argv.append("--no-ledger-append")
    return effective_argv
```

Then replace the inline dry-run stripping block in `run_chain` with:

```python
        effective_argv = list(argv)
        if dry_run:
            effective_argv = _dry_run_argv(label, effective_argv)
```

- [x] **Step 4: Run the walk test and verify green**

Run:

```bash
./.venv-wsl/bin/python -m pytest tests/test_tools/test_fast_lane_walk.py::test_run_chain_dry_run_adds_no_ledger_append_to_promote_queue -q
```

Expected: PASS.

## Task 3: Regression Gates

**Files:**
- No production edits unless a test exposes a real failure.

- [x] **Step 1: Run targeted test files**

Run:

```bash
./.venv-wsl/bin/python -m pytest tests/test_research/test_fast_lane_promote_queue_suppression.py tests/test_tools/test_fast_lane_walk.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run relevant drift gate if available**

Current evidence:

```text
./.venv-wsl/bin/python pipeline/check_drift.py --fast --quiet
FAIL: Active native trade-window provenance matches canonical recomputation (count=844)
PASS: FAST_LANE PROMOTE queue: no orphan PROMOTEs, no ERROR entries, cache up to date
PASS: Fast-lane status roll-up reconstruction parity
PASS: Fast-lane trial ledger append-only
PASS: Fast-lane graveyard digest parity
PASS: Fast-lane promote queue provenance present
SUMMARY: drift_detected violations=844 passed=134
```

This drift failure is outside Phase 0 and matches the concurrent lane-allocation terminal's reported baseline carry-over. Do not mark Phase 0 fully verified until repo-wide drift is clean or the baseline exception is formally accepted by the operator.

Run:

```bash
./.venv-wsl/bin/python pipeline/check_drift.py
```

Expected: pass, or stop and report the exact failing check. Do not claim done if this fails.

- [ ] **Step 3: Verify working tree and commit**

Run:

```bash
git status --short
git add docs/superpowers/plans/2026-05-21-fast-lane-v2-phase-0.md tests/test_research/test_fast_lane_promote_queue_suppression.py tests/test_tools/test_fast_lane_walk.py scripts/research/fast_lane_promote_queue.py scripts/tools/fast_lane_walk.py HANDOFF.md
git commit -m "fix(fast-lane): make dry-runs non-mutating"
git status --short
```

Expected: commit succeeds and the tree is clean unless unrelated user changes appear.

## Phase 0 Exit Criteria

- `fast_lane_promote_queue.py --dry-run` does not mutate `docs/runtime/fast_lane_trial_ledger.yaml`.
- `fast_lane_promote_queue.py --write` refreshes `promote_queue.yaml` without mutating `docs/runtime/fast_lane_trial_ledger.yaml`.
- Direct `scan()` defaults to read-only for trial-ledger provenance.
- `fast_lane_walk.py --dry-run` passes `--no-ledger-append` into the promote queue step.
- Back-to-back dry-runs do not create timestamp-based ledger rows.
- The current historical ledger is not edited or cleaned in this phase.
- Any future correction for polluted scanner rows must be an explicit correction/exclusion artifact with audit provenance.
