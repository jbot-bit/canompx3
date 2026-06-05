# Fix: anchor the self-funded margin-guard marker check to the ACCOUNT_TIERS dict

**Date:** 2026-06-05
**Scope:** `pipeline/check_drift.py` check (b) of `check_prop_caps_do_not_leak_into_self_funded`, plus one new test in `tests/test_pipeline/test_check_drift_self_funded_sizing.py`.
**Origin:** Code-review finding on PR #343 (scored 90/100; also raised as an unresolved inline comment on the PR).

## Problem

`check_prop_caps_do_not_leak_into_self_funded` check (b) verifies the
`@margin-guard-not-earnings-cap` marker covers the `self_funded` ACCOUNT_TIERS
block. Current logic:

```python
marker_line = next((i for i, ln in enumerate(lines) if marker in ln), None)  # FIRST match
first_tier = min(self_funded_tier_lines)
if marker_line > first_tier:   # only checks marker precedes the first tier
    <violation>
```

`next()` takes the **first** marker occurrence anywhere in the file. A second,
unrelated occurrence of the exact marker string was later added at
`trading_app/prop_profiles.py:139` (inside the `self_imposed_dd_dollars` field
comment in `AccountProfile`). Because line 139 < the first self_funded tier
(~line 604), the check is satisfied by the decoy **regardless of whether the
real ACCOUNT_TIERS-block marker (line 597) exists.**

Consequence — **fail-open**: deleting the real block marker still passes the
guard, defeating the docstring promise that "a new self_funded tier added
without the marker fails loud." The existing test
`test_guard_passes_on_real_repo_state` passes today for the wrong reason (the
decoy satisfies it).

## Fix (Option B — anchor to the ACCOUNT_TIERS dict)

Require a marker occurrence **inside the `ACCOUNT_TIERS` dict span and before
the first `self_funded` tier**, rather than the first marker anywhere in the file.

Algorithm:
1. Locate the dict-open line: the line matching `ACCOUNT_TIERS` and ending in `= {`
   (the real file: `ACCOUNT_TIERS: dict[...] = {` at line 508). Use a tolerant
   match (`"ACCOUNT_TIERS" in ln and ln.rstrip().endswith("{")`).
   - If not found → fail loud ("could not locate ACCOUNT_TIERS dict opening …
     layout changed"), consistent with the existing layout-changed guards.
2. `first_tier = min(self_funded_tier_lines)` (unchanged).
3. Find marker occurrences (ALL, not just first):
   `marker_lines = [i for i, ln in enumerate(lines) if marker in ln]`.
4. **Anchored requirement:** at least one marker line `m` must satisfy
   `dict_open < m < first_tier`. (Strictly greater than dict_open so a marker
   far above the dict — the line-139 decoy — does not qualify; strictly less
   than first_tier so it introduces the block, preserving the existing
   "marker too late" behaviour.)
   - If none qualify → violation. The message names the failure precisely:
     marker absent from / not introducing the ACCOUNT_TIERS self_funded block.

The `marker_line is None` (marker absent entirely) branch is folded into the
anchored check: if no marker lies in the qualifying span, it is a violation
whether the marker is wholly absent or only present as a decoy elsewhere.

### Why Option B over a proximity window (Option A)

Anchoring to the dict span is robust to blank lines / extra comment lines
between the marker and the first tier (the real file has 7 comment lines
between them). A fixed line-window (`marker within N lines of first tier`)
would be brittle to that gap and to future edits. Option B uses the
`ACCOUNT_TIERS = {` boundary already present in the file.

## Behaviour matrix (after fix)

| Source state | marker in dict span? | Result |
|---|---|---|
| Real repo (marker @597, decoy @139) | yes (@597) | PASS |
| Real block marker deleted, decoy @139 kept | no | **FAIL** (was PASS — the bug) |
| Marker present, but only ABOVE the dict | no | FAIL |
| Marker present but AFTER first tier | no | FAIL (preserved) |
| Marker wholly absent | no | FAIL (preserved) |
| No self_funded tiers (layout vanished) | n/a | FAIL loud (preserved) |
| ACCOUNT_TIERS dict opening not found | n/a | FAIL loud (new) |

## Tests

Add to `tests/test_pipeline/test_check_drift_self_funded_sizing.py`:

- **New RED test** `test_guard_fails_when_marker_only_outside_dict`: a temp
  source where the `@margin-guard-not-earnings-cap` marker appears ONLY above
  the `ACCOUNT_TIERS = {` line (a decoy in an unrelated comment/docstring),
  with NO marker inside the dict. The dict's self_funded tiers carry no marker.
  Assert the guard now produces a violation. This is the exact line-139
  scenario in miniature; it must FAIL before the fix and PASS (i.e. correctly
  flag the violation) after.

Keep all existing (b) tests green:
- `test_guard_passes_on_well_formed_temp` (`_GOOD_SRC` has the marker inside the
  dict, above the first tier → still passes).
- `test_guard_fails_when_marker_absent`, `…_marker_after_first_tier`,
  `…_doctrine_file_missing`, `…_no_self_funded_tiers` — all preserved.

The `_GOOD_SRC`/`_NO_MARKER_SRC`/`_MARKER_TOO_LATE_SRC` fixtures already place
the marker inside (or relative to) the `ACCOUNT_TIERS = {` dict, so they remain
valid under the anchored logic.

## Non-goals

- No change to structural checks (c) / (d) or to the runtime resolver layer.
- No change to `prop_profiles.py` (the real marker @597 is already correctly
  inside the dict; the decoy @139 is legitimate in its own context and stays).
- No change to the doctrine `.md` file or the marker comment wording (those were
  the two sub-threshold review findings, explicitly out of scope here).

## Verification

1. `pytest tests/test_pipeline/test_check_drift_self_funded_sizing.py -q` — all
   green incl. the new test.
2. Confirm RED-before-green: temporarily revert the check body, run the new
   test, see it fail; restore.
3. Run the single drift check standalone against the real repo to confirm it
   still passes clean (marker @597 is inside the dict).
4. Full drift / pre-commit gate before any integration.

## Integration

Tier: this is a drift-check (capital-protecting truth layer) change → **Tier B**.
No push without explicit operator GO. We do not PR — direct integration on the
owning branch per project doctrine.
