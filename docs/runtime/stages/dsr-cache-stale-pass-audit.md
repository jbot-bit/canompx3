# Stage: DSR drift-cache stale-PASS audit (Codex high-risk finding)

task: Verify and (if real) fix whether the DSR reference-universe-lock drift check can serve a STALE PASS after its own verdict logic in pipeline/check_drift.py changes.
mode: TRUTH AUDIT (not yet IMPLEMENTATION — finding is PLAUSIBLE, NOT PROVEN)
status: OPEN — next session

## Origin

Codex adversarial high-risk review (2026-06-04, range `96b2554e..11c204e1`,
verdict **needs-attention**, one [high] finding). The review itself is the first
run of the new `/highrisk-review` command (landed `11c204e1` on main).

Codex ran in a sandbox where MANY probe commands were DECLINED (`exit -1` in the
task log) — so its conclusion is an **inference from the diff**, not an
execution-verified fact. Treat as a claim requiring proof/disproof
(institutional-rigor §11: never trust metadata — verify by execution).

## The claim (Codex, verbatim gist)

`pipeline/check_drift.py:124-129` — the "DSR reference-universe lock" cache entry
declares `file_deps: []` (line 125) and only hashes `docs/audit/hypotheses/*.yaml`
(tree_deps). Codex says the verdict ALSO depends on the module-level
`_ALLOWED_DSR_TRIALS_DERIVATION` constant (VERIFIED present at check_drift.py:3906,
enforced at 4011-4014 — ~3,800 lines from the cache entry, same file), which is
NOT in the key → changing the DSR allowed-derivations (or the check body) could leave an old
PASS live until the meta-recheck happens to sample that label. Impact: a blocking
Bailey–López de Prado DSR/multiplicity gate reporting green on stale logic =
exactly the over-claim class (cf.
[[feedback_drift_cache_fast_lane_db_verdict_input_stale_pass_requires_db_false_trap_2026_06_03]]).

## The counter-claim already IN the code (must be tested, not trusted)

`check_drift.py:119-123` comment asserts the defense already exists: *"the
verdict's allowed derivations come from the module-level
`_ALLOWED_DSR_TRIALS_DERIVATION` constant in THIS file, which the cache key cannot
miss (an edit to check_drift.py is a code change that re-runs the whole suite).
No fixed-file or git-state inputs."*

**LOAD-BEARING UNKNOWN:** is "an edit to check_drift.py re-runs the whole suite"
actually true? i.e. does the cache layer key on (or invalidate by) check_drift.py's
own content / a self-hash / pre-commit running the checker uncached when it
changes? If YES → finding is FALSE (comment is correct, no fix needed, maybe add
a regression test to lock it). If NO → finding is REAL → fix required.

## TRUTH-AUDIT steps (operator spec, 2026-06-04 — Tier B, DO NOT PATCH NOW)

Operator instruction verbatim: "Verify whether the DSR drift-check cache can serve
stale PASS after changes to pipeline/check_drift.py logic. No merge/push without
approval."

1. **Reproduce current cache behavior.** Run the DSR check, confirm it caches a
   PASS (warm HIT). Read the cache-key construction (`_cache_key`/`read_pass`/
   version+label hashing, ~check_drift.py:60-135): EXACTLY what goes into the key
   (file_deps + tree_deps + label + version) and whether check_drift.py's own
   content / a suite-level self-hash is an input anywhere.
2. **Change `_ALLOWED_DSR_TRIALS_DERIVATION` (line 3906) or equivalent DSR verdict
   logic in a temporary branch/test** (known-violation injection,
   integrity-guardian §7) — NOT on main.
3. **Prove whether the cache key changes or forces a cold run.** Read the actual
   re-run verdict. Also inspect the meta-recheck (`_CACHE_HITS_THIS_RUN` + the
   1/run sampled cold re-run): does it bound staleness to ≤1 commit, or can a
   stale PASS persist across many commits?
4. **If stale PASS is real:** official fix (no band-aid) — add
   `pipeline/check_drift.py` / an extracted DSR-policy module to this entry's
   `file_deps`, OR make the check non-cacheable; + add a regression test.
5. **Run:** `python -m pytest tests/test_pipeline/test_drift_cache.py -q` AND a
   focused cold/warm drift check for the DSR label.
6. **Report blast radius and exact diff.** No merge/push without approval.

## IF REAL — official fix (institutional-rigor §4, no band-aid)

Per Codex recommendation + repo's own cache doctrine:
- Add `pipeline/check_drift.py` (or a narrowly-extracted DSR-policy module) to
  THIS entry's `file_deps`, OR mark the DSR check non-cacheable.
- Prefer extracting `_ALLOWED_DSR_TRIALS_DERIVATION` + the DSR verdict logic into
  a small module so the dep is precise (avoids re-running ALL cached checks on
  every unrelated check_drift.py edit — token/perf cost).
- Add a regression test: changing `_ALLOWED_DSR_TRIALS_DERIVATION` (or the DSR
  check body) MUST change the cache key OR force a cold run. Mirror the existing
  anti-regression test added for the FAST_LANE un-wire (`ca0a2b7e`/`690c52f1`).

## Verification gate (MANDATORY before claiming done)

- `python -m pytest tests/test_pipeline/test_drift_cache.py -q` (show output)
- Focused cold/warm check for the DSR label behaviour
- `python pipeline/check_drift.py` passes
- Dead-code sweep
- **Adversarial-audit gate (.claude/rules/adversarial-audit-gate.md):** this is a
  truth-layer pipeline/ CRIT/HIGH fix → dispatch evidence-auditor (independent
  context) BEFORE the fix is considered closed. Codex itself is one independent
  pass; the evidence-auditor is the second, repo-native one.

## Blast Radius

- pipeline/check_drift.py — the drift-cache key + DSR check entry. Read by the
  pre-commit hook (every commit) and CI. A wrong fix that over-broadens file_deps
  makes MANY cached checks cold on every check_drift.py edit (perf regression,
  ~adds tens of seconds to every commit). A wrong fix that under-scopes leaves the
  stale-PASS hole open.
- tests/test_pipeline/test_drift_cache.py — companion tests; must gain a DSR-key
  regression.
- NO gold.db schema, NO trading_app, NO live/capital path touched. But the GATE
  itself guards capital-promotion decisions (DSR clearance) → treat as truth-layer.

## scope_lock
- pipeline/check_drift.py
- tests/test_pipeline/test_drift_cache.py

## Do NOT
- Do NOT fix based on Codex's inference alone — PROVE it first (step 2 execution).
- Do NOT trust the line 119-123 comment as evidence — test it.
- Do NOT over-broaden file_deps to the whole check_drift.py without weighing the
  per-commit perf cost (extract a module instead).
- Tier B: capital-adjacent truth gate — present design proposal + wait for go
  before editing.
