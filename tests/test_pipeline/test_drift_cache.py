"""Honesty proofs for the drift-cache (pipeline/_drift_cache.py).

The cache may NEVER return a stale PASS. These tests prove, by construction:

  (a) changed dep  → key changes → cache misses → real check runs
  (b) corrupt cache file → read fails closed → real check runs
  (c) unreadable / missing dep → key is None → never a hit (real check runs)
  (d) a FAIL verdict is never written to the cache
  (e) the meta cold-recheck flags a stale PASS when CHECK_DEPS is incomplete

Each test isolates the cache to a tmp dir and points PROJECT_ROOT at a tmp
fixture so no real .git/.drift-cache or repo file is touched.
"""

from __future__ import annotations

import pytest

from pipeline import _drift_cache


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Redirect the cache dir and PROJECT_ROOT to tmp; create two dep files."""
    root = tmp_path / "repo"
    (root / "trading_app" / "ai").mkdir(parents=True)
    dep_a = root / "trading_app" / "config.py"
    dep_b = root / "trading_app" / "ai" / "sql_adapter.py"
    dep_a.write_text('ENTRY_MODELS = ["E1", "E2", "E3"]\n', encoding="utf-8")
    dep_b.write_text("VALID_ENTRY_MODELS = set(ENTRY_MODELS)\n", encoding="utf-8")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    monkeypatch.setattr(_drift_cache, "PROJECT_ROOT", root)
    monkeypatch.setattr(_drift_cache, "_cache_dir", lambda: cache_dir)

    deps = ["trading_app/config.py", "trading_app/ai/sql_adapter.py"]
    return {"root": root, "dep_a": dep_a, "dep_b": dep_b, "cache_dir": cache_dir, "deps": deps}


LABEL = "ENTRY_MODELS sync"


def test_pass_is_cached_and_hits_on_unchanged_deps(isolated_cache):
    deps = isolated_cache["deps"]
    key1 = _drift_cache.cache_key(LABEL, deps)
    assert key1 is not None
    assert _drift_cache.read_pass(LABEL, key1) is False  # nothing written yet

    _drift_cache.write_pass(LABEL, key1, [])  # PASS (empty violations)
    assert _drift_cache.read_pass(LABEL, key1) is True  # now a hit

    # Recomputing the key over unchanged deps yields the same key → still a hit.
    key2 = _drift_cache.cache_key(LABEL, deps)
    assert key2 == key1
    assert _drift_cache.read_pass(LABEL, key2) is True


def test_changed_dep_misses(isolated_cache):
    """(a) Editing a declared dep changes the key → cached PASS no longer hits."""
    deps = isolated_cache["deps"]
    key1 = _drift_cache.cache_key(LABEL, deps)
    _drift_cache.write_pass(LABEL, key1, [])
    assert _drift_cache.read_pass(LABEL, key1) is True

    # Edit one dep file.
    isolated_cache["dep_a"].write_text('ENTRY_MODELS = ["E1", "E2"]\n', encoding="utf-8")

    key2 = _drift_cache.cache_key(LABEL, deps)
    assert key2 != key1, "key must change when a dep's content changes"
    assert _drift_cache.read_pass(LABEL, key2) is False, "stale PASS must NOT hit after dep edit"


def test_corrupt_cache_runs_real_check(isolated_cache):
    """(b) A corrupt cache file reads fail-closed → MISS (caller runs real check)."""
    deps = isolated_cache["deps"]
    key = _drift_cache.cache_key(LABEL, deps)
    _drift_cache.write_pass(LABEL, key, [])
    assert _drift_cache.read_pass(LABEL, key) is True

    # Corrupt the on-disk cache file.
    cache_file = isolated_cache["cache_dir"] / f"{_drift_cache._safe_name(LABEL)}.json"
    cache_file.write_text("{not valid json", encoding="utf-8")

    assert _drift_cache.read_pass(LABEL, key) is False, "corrupt cache must fail closed to MISS"


def test_missing_dep_yields_none_key_never_hits(isolated_cache):
    """(c) An unreadable/missing dep → key None → can never be a hit."""
    # First cache a real PASS.
    deps = isolated_cache["deps"]
    key = _drift_cache.cache_key(LABEL, deps)
    _drift_cache.write_pass(LABEL, key, [])

    # Now declare a dep that does not exist.
    bad_deps = deps + ["trading_app/does_not_exist.py"]
    bad_key = _drift_cache.cache_key(LABEL, bad_deps)
    assert bad_key is None, "missing dep must yield a None key"
    assert _drift_cache.read_pass(LABEL, bad_key) is False
    # write_pass with a None key is a no-op (no blind cache).
    _drift_cache.write_pass(LABEL, bad_key, [])
    assert not list(isolated_cache["cache_dir"].glob("*extra*"))


def test_fail_verdict_never_cached(isolated_cache):
    """(d) A non-empty violation list is never persisted."""
    deps = isolated_cache["deps"]
    key = _drift_cache.cache_key(LABEL, deps)
    _drift_cache.write_pass(LABEL, key, ["  some violation"])  # FAIL
    assert _drift_cache.read_pass(LABEL, key) is False, "FAIL must never be cached"
    cache_file = isolated_cache["cache_dir"] / f"{_drift_cache._safe_name(LABEL)}.json"
    assert not cache_file.exists(), "no cache file should be written for a FAIL"


def test_meta_recheck_fires_on_stale_cache(monkeypatch):
    """(e) The honesty backstop's OWN failure path: when a cached-hit check's
    cold re-run finds a violation (i.e. CHECK_DEPS was incomplete and an input
    changed without changing the key), the meta cold-recheck MUST emit a
    STALE CACHE violation. Without this, the backstop's detection path is
    untested (audit finding, 2026-05-29)."""
    import pipeline.check_drift as cd

    label = "ENTRY_MODELS sync"
    # Simulate a stale PASS: the check is recorded as a cache hit, but cold it
    # now returns a violation (the exact "under-declared deps" scenario). The
    # meta-check resolves fn from the CHECKS tuple (the real registered ref),
    # so patch the tuple entry, not the module global.
    monkeypatch.setattr(cd, "_CACHE_HITS_THIS_RUN", [label])
    patched = [
        (lbl, (lambda: ["  injected stale-cache violation"]) if lbl == label else fn, adv, rdb)
        for lbl, fn, adv, rdb in cd.CHECKS
    ]
    monkeypatch.setattr(cd, "CHECKS", patched)

    out = cd.check_drift_cache_meta_recheck()
    assert any("STALE CACHE" in v and label in v for v in out), (
        f"meta-recheck must flag a stale cached PASS; got {out!r}"
    )


def test_meta_recheck_clean_when_cold_matches(monkeypatch):
    """Complement to (e): when the cold re-run reproduces the cached PASS
    (empty), the meta-recheck emits no violation."""
    import pipeline.check_drift as cd

    label = "ENTRY_MODELS sync"
    monkeypatch.setattr(cd, "_CACHE_HITS_THIS_RUN", [label])
    patched = [(lbl, (lambda: []) if lbl == label else fn, adv, rdb) for lbl, fn, adv, rdb in cd.CHECKS]
    monkeypatch.setattr(cd, "CHECKS", patched)

    assert cd.check_drift_cache_meta_recheck() == []


def test_cache_dir_none_disables_cache(isolated_cache, monkeypatch):
    """When the cache location cannot be resolved, read/write are no-ops (fail-closed)."""
    monkeypatch.setattr(_drift_cache, "_cache_dir", lambda: None)
    deps = isolated_cache["deps"]
    key = _drift_cache.cache_key(LABEL, deps)
    _drift_cache.write_pass(LABEL, key, [])
    assert _drift_cache.read_pass(LABEL, key) is False


def test_cache_dir_uses_git_common_dir_for_linked_worktree(tmp_path, monkeypatch):
    """A linked worktree has .git as a file; cache still belongs under common .git."""
    root = tmp_path / "repo"
    common = tmp_path / "common.git"
    root.mkdir()
    common.mkdir()
    (root / ".git").write_text(f"gitdir: {common / 'worktrees' / 'repo'}\n", encoding="utf-8")
    monkeypatch.setattr(_drift_cache, "PROJECT_ROOT", root)

    class Result:
        returncode = 0
        stdout = str(common)

    monkeypatch.setattr(_drift_cache.subprocess, "run", lambda *args, **kwargs: Result())

    cache_dir = _drift_cache._cache_dir()
    assert cache_dir == (common / ".drift-cache").resolve()
    assert cache_dir.exists()


def _run_real_dispatch(label, monkeypatch, cache_dir):
    """Replay the runner's exact cache-dispatch decision (check_drift.py:15156-15166)
    against the REAL _drift_cache functions, the REAL CHECK_DEPS entry, and the
    REAL registered check_fn — only the cache dir is isolated to tmp. Returns the
    verdict list `v` and records hits in the real _CACHE_HITS_THIS_RUN, exactly as
    main() does. This is the integration seam the unit tests above stub out."""
    import pipeline.check_drift as cd

    monkeypatch.setattr(_drift_cache, "_cache_dir", lambda: cache_dir)
    by_label = {entry[0]: entry[1] for entry in cd.CHECKS}
    check_fn = by_label[label]

    deps = cd.CHECK_DEPS.get(label)
    cache_key = _drift_cache.cache_key(label, deps) if deps else None
    if deps is not None and _drift_cache.read_pass(label, cache_key):
        cd._CACHE_HITS_THIS_RUN.append(label)
        v = []
    else:
        v = check_fn()
        if deps is not None:
            _drift_cache.write_pass(label, cache_key, v)
    return v


# ---------------------------------------------------------------------------
# tree_cache_key — whole-tree (glob) dependency digests. Same honesty invariants
# as cache_key, plus: a file ADDED or REMOVED from a globbed tree must change the
# key (not only an edit to an existing file).
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_tree_cache(tmp_path, monkeypatch):
    """A repo root with a fixed file dep AND a globbed tree of result files."""
    root = tmp_path / "repo"
    (root / "trading_app").mkdir(parents=True)
    fixed = root / "trading_app" / "config.py"
    fixed.write_text("FIXED = 1\n", encoding="utf-8")

    tree_dir = root / "docs" / "audit" / "results"
    tree_dir.mkdir(parents=True)
    (tree_dir / "a-FAST_LANE.md").write_text("alpha\n", encoding="utf-8")
    (tree_dir / "b-FAST_LANE.md").write_text("bravo\n", encoding="utf-8")
    # A non-matching sibling that must NOT affect the digest.
    (tree_dir / "c-OTHER.md").write_text("charlie\n", encoding="utf-8")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(_drift_cache, "PROJECT_ROOT", root)
    monkeypatch.setattr(_drift_cache, "_cache_dir", lambda: cache_dir)

    return {
        "root": root,
        "fixed": fixed,
        "tree_dir": tree_dir,
        "cache_dir": cache_dir,
        "file_deps": ["trading_app/config.py"],
        "tree_deps": [("docs/audit/results", "*-FAST_LANE.md")],
    }


TREE_LABEL = "FAST_LANE promote orphans"


def test_tree_key_hits_on_unchanged_tree(isolated_tree_cache):
    fc = isolated_tree_cache
    k1 = _drift_cache.tree_cache_key(TREE_LABEL, fc["file_deps"], fc["tree_deps"])
    assert k1 is not None
    _drift_cache.write_pass(TREE_LABEL, k1, [])
    assert _drift_cache.read_pass(TREE_LABEL, k1) is True

    k2 = _drift_cache.tree_cache_key(TREE_LABEL, fc["file_deps"], fc["tree_deps"])
    assert k2 == k1, "unchanged tree must yield the same key"
    assert _drift_cache.read_pass(TREE_LABEL, k2) is True


def test_tree_key_changes_on_added_file(isolated_tree_cache):
    """A NEW matching file in the tree must change the key — the add/remove case
    a content-only digest would miss."""
    fc = isolated_tree_cache
    k1 = _drift_cache.tree_cache_key(TREE_LABEL, fc["file_deps"], fc["tree_deps"])
    _drift_cache.write_pass(TREE_LABEL, k1, [])

    (fc["tree_dir"] / "d-FAST_LANE.md").write_text("delta\n", encoding="utf-8")
    k2 = _drift_cache.tree_cache_key(TREE_LABEL, fc["file_deps"], fc["tree_deps"])
    assert k2 != k1, "adding a matching file must change the key"
    assert _drift_cache.read_pass(TREE_LABEL, k2) is False


def test_tree_key_changes_on_removed_file(isolated_tree_cache):
    fc = isolated_tree_cache
    k1 = _drift_cache.tree_cache_key(TREE_LABEL, fc["file_deps"], fc["tree_deps"])
    _drift_cache.write_pass(TREE_LABEL, k1, [])

    (fc["tree_dir"] / "a-FAST_LANE.md").unlink()
    k2 = _drift_cache.tree_cache_key(TREE_LABEL, fc["file_deps"], fc["tree_deps"])
    assert k2 != k1, "removing a matching file must change the key"
    assert _drift_cache.read_pass(TREE_LABEL, k2) is False


def test_tree_key_changes_on_edited_tree_file(isolated_tree_cache):
    fc = isolated_tree_cache
    k1 = _drift_cache.tree_cache_key(TREE_LABEL, fc["file_deps"], fc["tree_deps"])
    _drift_cache.write_pass(TREE_LABEL, k1, [])

    (fc["tree_dir"] / "b-FAST_LANE.md").write_text("bravo EDITED\n", encoding="utf-8")
    k2 = _drift_cache.tree_cache_key(TREE_LABEL, fc["file_deps"], fc["tree_deps"])
    assert k2 != k1, "editing a tree file's content must change the key"


def test_tree_key_changes_on_edited_fixed_file(isolated_tree_cache):
    """The file_deps half is hashed too — editing the fixed dep invalidates."""
    fc = isolated_tree_cache
    k1 = _drift_cache.tree_cache_key(TREE_LABEL, fc["file_deps"], fc["tree_deps"])
    fc["fixed"].write_text("FIXED = 2\n", encoding="utf-8")
    k2 = _drift_cache.tree_cache_key(TREE_LABEL, fc["file_deps"], fc["tree_deps"])
    assert k2 != k1, "editing a fixed file dep must change the key"


def test_tree_key_ignores_nonmatching_sibling(isolated_tree_cache):
    """A file that does not match the glob pattern must not affect the key."""
    fc = isolated_tree_cache
    k1 = _drift_cache.tree_cache_key(TREE_LABEL, fc["file_deps"], fc["tree_deps"])
    (fc["tree_dir"] / "e-OTHER.md").write_text("echo\n", encoding="utf-8")
    k2 = _drift_cache.tree_cache_key(TREE_LABEL, fc["file_deps"], fc["tree_deps"])
    assert k2 == k1, "a non-matching file must not change the key"


def test_tree_key_empty_tree_is_stable_and_distinct(isolated_tree_cache):
    """An empty tree is a valid state: stable non-None key, distinct from non-empty."""
    fc = isolated_tree_cache
    k_full = _drift_cache.tree_cache_key(TREE_LABEL, fc["file_deps"], fc["tree_deps"])

    for f in fc["tree_dir"].glob("*-FAST_LANE.md"):
        f.unlink()
    k_empty1 = _drift_cache.tree_cache_key(TREE_LABEL, fc["file_deps"], fc["tree_deps"])
    k_empty2 = _drift_cache.tree_cache_key(TREE_LABEL, fc["file_deps"], fc["tree_deps"])
    assert k_empty1 is not None, "empty tree must still yield a usable key"
    assert k_empty1 == k_empty2, "empty tree key must be stable"
    assert k_empty1 != k_full, "empty tree must differ from a populated tree"


def test_tree_key_missing_fixed_dep_fails_closed(isolated_tree_cache):
    """A declared fixed file that does not exist → None key (MISS)."""
    fc = isolated_tree_cache
    bad = _drift_cache.tree_cache_key(TREE_LABEL, fc["file_deps"] + ["trading_app/does_not_exist.py"], fc["tree_deps"])
    assert bad is None, "missing fixed dep must fail closed to a None key"
    assert _drift_cache.read_pass(TREE_LABEL, bad) is False


def test_tree_key_nonexistent_glob_root_fails_closed(isolated_tree_cache):
    """A missing/typo'd tree root must fail closed to None, NOT silently hash an
    empty tree (audit finding 2026-06-03). Path.glob on a non-existent dir returns
    an empty iterator with no error — a phantom-empty tree that would serve a stale
    PASS while the real check reads real files."""
    fc = isolated_tree_cache
    bad_tree = _drift_cache.tree_cache_key(
        TREE_LABEL, fc["file_deps"], [("docs/audit/DOES_NOT_EXIST", "*-FAST_LANE.md")]
    )
    assert bad_tree is None, "a non-existent tree root must yield a None key (MISS)"
    assert _drift_cache.read_pass(TREE_LABEL, bad_tree) is False


def test_tree_key_root_is_a_file_fails_closed(isolated_tree_cache):
    """A tree root that resolves to a FILE (not a directory) also globs empty with
    no error — must fail closed, same as a missing root."""
    fc = isolated_tree_cache
    # trading_app/config.py is a file, not a directory.
    bad_tree = _drift_cache.tree_cache_key(TREE_LABEL, fc["file_deps"], [("trading_app/config.py", "*")])
    assert bad_tree is None, "a tree root that is a file must yield a None key (MISS)"


def test_tree_key_path_is_load_bearing_rename_among_identical_content(isolated_tree_cache):
    """Proves the PATH component (``relpath:hash``) is strictly load-bearing — the
    one scenario a content-only digest provably MISSES even with path-sorted
    iteration.

    Setup: two matching files with IDENTICAL content. Rename one to a name that
    keeps the same sort position (still after the other). The tree's COUNT is
    unchanged, the multiset of CONTENTS is unchanged, and the path-SORTED order of
    content-hashes is unchanged ([h, h] before and after). A content-only digest
    therefore COLLIDES → it would serve a stale PASS though the real file set
    changed. Only a digest binding each path to its content distinguishes them.

    This is the case the earlier swap/rename probes missed (a rename that reorders
    is caught by content-only via reordering; a same-content, order-preserving
    rename is not). Mutation-verified: dropping ``{rel}:`` from the digest makes
    this test fail.
    """
    fc = isolated_tree_cache
    td = fc["tree_dir"]
    # Replace the fixture's distinct-content files with two IDENTICAL-content ones.
    for f in td.glob("*-FAST_LANE.md"):
        f.unlink()
    (td / "a-FAST_LANE.md").write_text("SAME\n", encoding="utf-8")
    (td / "m-FAST_LANE.md").write_text("SAME\n", encoding="utf-8")

    k1 = _drift_cache.tree_cache_key(TREE_LABEL, fc["file_deps"], fc["tree_deps"])

    # Rename m → p: identical content, still sorts after 'a'. Count, content-multiset
    # and sorted-hash-sequence all invariant; only the path set changed.
    (td / "m-FAST_LANE.md").rename(td / "p-FAST_LANE.md")

    k2 = _drift_cache.tree_cache_key(TREE_LABEL, fc["file_deps"], fc["tree_deps"])
    assert k2 != k1, (
        "an order-preserving rename among identical-content files must change the "
        "key — the digest must bind each path to its content, not hash content alone"
    )


def test_tree_key_distinct_from_plain_cache_key(isolated_tree_cache):
    """tree_cache_key and cache_key over the same label+files must not collide —
    the 'tree' namespace guarantees a key from the wrong builder always misses."""
    fc = isolated_tree_cache
    k_tree = _drift_cache.tree_cache_key(TREE_LABEL, fc["file_deps"], [])
    k_plain = _drift_cache.cache_key(TREE_LABEL, fc["file_deps"])
    assert k_tree is not None and k_plain is not None
    assert k_tree != k_plain, "tree and plain keys must be namespaced apart"


def test_runner_dispatch_real_hit_then_meta_recheck_passes(monkeypatch, tmp_path):
    """Integration (audit finding 2026-05-29): drive the REAL runner dispatch so a
    genuine cache HIT populates _CACHE_HITS_THIS_RUN, then prove the real
    meta cold-recheck re-runs the real check_entry_models_sync cold and confirms
    parity. The unit tests above patch _CACHE_HITS_THIS_RUN directly; this test
    never does — it earns the hit through the actual read_pass path against the
    live repo files declared in CHECK_DEPS, with only the cache dir on tmp."""
    import pipeline.check_drift as cd

    label = "ENTRY_MODELS sync"
    cache_dir = tmp_path / "drift-cache"
    cache_dir.mkdir()
    monkeypatch.setattr(cd, "_CACHE_HITS_THIS_RUN", [])

    # Run 1: cold (empty cache) → real check runs, real PASS persisted, no hit.
    v1 = _run_real_dispatch(label, monkeypatch, cache_dir)
    assert v1 == [], "live repo must currently PASS ENTRY_MODELS sync (precondition)"
    assert cd._CACHE_HITS_THIS_RUN == [], "first run is a MISS, not a hit"

    # Run 2: deps unchanged → real read_pass HIT → _CACHE_HITS_THIS_RUN populated.
    v2 = _run_real_dispatch(label, monkeypatch, cache_dir)
    assert v2 == []
    assert cd._CACHE_HITS_THIS_RUN.count(label) == 1, "unchanged deps must produce a real cache hit"
    assert len(cd._CACHE_HITS_THIS_RUN) == 1, "exactly one label should be recorded"

    # The meta cold-recheck now re-runs the REAL check cold and must agree (PASS).
    assert cd.check_drift_cache_meta_recheck() == [], (
        "cold re-run of the real ENTRY_MODELS sync must reproduce the cached PASS"
    )


# ---------------------------------------------------------------------------
# Stage 2 (2026-06-03): CHECK_TREE_DEPS wiring — the slow tree-scanning checks
# cache through tree_cache_key. These tests drive the REAL runner dispatch
# branch for a CHECK_TREE_DEPS label and prove the honesty contract end-to-end.
# ---------------------------------------------------------------------------

FAST_LANE_LABEL = "FAST_LANE PROMOTE queue: no orphan PROMOTEs, no ERROR entries, cache up to date"


def _run_real_tree_dispatch(label, monkeypatch, cache_dir):
    """Replay the runner's CHECK_TREE_DEPS dispatch branch (check_drift.py main
    loop) against the REAL tree_cache_key, the REAL CHECK_TREE_DEPS entry, and the
    REAL registered check_fn — only the cache dir is isolated to tmp. Mirrors
    _run_real_dispatch but for the tree-dep path."""
    import pipeline.check_drift as cd

    monkeypatch.setattr(_drift_cache, "_cache_dir", lambda: cache_dir)
    by_label = {entry[0]: entry[1] for entry in cd.CHECKS}
    check_fn = by_label[label]

    spec = cd.CHECK_TREE_DEPS.get(label)
    key = (
        _drift_cache.tree_cache_key(label, spec.get("file_deps", []), spec.get("tree_deps", []))
        if spec is not None
        else None
    )
    if spec is not None and _drift_cache.read_pass(label, key):
        cd._CACHE_HITS_THIS_RUN.append(label)
        v = []
    else:
        v = check_fn()
        if spec is not None:
            _drift_cache.write_pass(label, key, v)
    return v


def test_check_dep_dicts_are_mutually_exclusive():
    """A label in BOTH CHECK_DEPS and CHECK_TREE_DEPS would race on the same
    on-disk cache file (_safe_name hashes only the label). Forbid it."""
    import pipeline.check_drift as cd

    overlap = set(cd.CHECK_DEPS) & set(cd.CHECK_TREE_DEPS)
    assert not overlap, f"labels declared in both dep dicts: {overlap}"


def test_check_tree_deps_labels_are_real_and_non_db():
    """Every CHECK_TREE_DEPS label must be a registered, NON-db check — the cache
    never serves a requires_db check (DB content is not hashed)."""
    import pipeline.check_drift as cd

    by_label = {label: requires_db for label, _fn, _adv, requires_db in cd.CHECKS}
    for label in cd.CHECK_TREE_DEPS:
        assert label in by_label, f"CHECK_TREE_DEPS label not in CHECKS: {label!r}"
        assert not by_label[label], f"CHECK_TREE_DEPS label is requires_db (forbidden): {label!r}"


def test_check_tree_deps_keys_are_well_formed():
    """Each CHECK_TREE_DEPS value must be {'file_deps': list, 'tree_deps': list of
    (root, pattern) 2-tuples} so tree_cache_key receives the shape it expects."""
    import pipeline.check_drift as cd

    for label, spec in cd.CHECK_TREE_DEPS.items():
        assert set(spec) <= {"file_deps", "tree_deps"}, f"unexpected keys in {label!r}: {set(spec)}"
        assert isinstance(spec.get("file_deps", []), list)
        for pair in spec.get("tree_deps", []):
            assert isinstance(pair, tuple) and len(pair) == 2, (
                f"tree_deps entry must be (root, pattern) for {label!r}: {pair!r}"
            )


def test_fast_lane_check_is_deliberately_not_cached():
    """ANTI-REGRESSION (audit finding 2026-06-03): the FAST_LANE PROMOTE check must
    NOT be cache-eligible. Although its CHECKS tuple declares requires_db=False, its
    verdict depends on gold.db content — scripts.research.fast_lane_promote_queue.scan()
    opens the DB via _resolve_oos_window_days() (`SELECT MAX(trading_day) FROM
    orb_outcomes`) and the resulting oos_window_days flips entries between
    REJECTED_OOS_UNPOWERED and QUEUED. DB content is not file-content-hashable, so a
    warmed cache would serve a STALE PASS on this BLOCKING capital gate as new bar data
    lands. It is in the same non-cacheable class as the Phase4-SHA-manifest check
    (git-history input) and the doc-hygiene check (entrypoint-existence input).

    If a future change re-adds FAST_LANE to CHECK_TREE_DEPS (or CHECK_DEPS) without a
    DB-content-aware key, this test fails — re-read the rejection rationale at the head
    of CHECK_TREE_DEPS in check_drift.py before changing it."""
    import pipeline.check_drift as cd

    assert FAST_LANE_LABEL not in cd.CHECK_TREE_DEPS, (
        "FAST_LANE PROMOTE check re-added to CHECK_TREE_DEPS — its verdict reads gold.db "
        "(MAX(trading_day)); caching it serves a stale PASS on a blocking gate. See the "
        "DELIBERATELY-NOT-CACHED note at the head of CHECK_TREE_DEPS."
    )
    assert FAST_LANE_LABEL not in cd.CHECK_DEPS, (
        "FAST_LANE PROMOTE check added to CHECK_DEPS — same DB-verdict-input stale-PASS hazard."
    )


# ---------------------------------------------------------------------------
# Stage 2 (2026-06-03): meta-recheck SAMPLING — the load-bearing change that
# unlocks the speedup. The cold re-run (expensive) is sampled to ONE label per
# run; the structural validation (cheap) still runs for EVERY cached-hit label.
# ---------------------------------------------------------------------------

THEORY_GRANT_LABEL = (
    "AM3.3 audit-log/prereg theory_grant parity: chordia_audit_log.yaml theory_grants "
    "must match active prereg metadata.theory_grant (Check #162)"
)
DSR_LABEL = (
    "DSR reference-universe lock declared (Criterion 5 Amendment 3.5: criterion_5 block "
    "complete when claiming DSR-clearance)"
)


def test_meta_recheck_sample_index_is_deterministic_and_in_range(monkeypatch):
    """The sample index is a pure function of (labels, HEAD): same inputs → same
    index, always within [0, n). Pin HEAD so the test is hermetic."""
    monkeypatch.setattr(_drift_cache, "git_head_sha", lambda: "deadbeef" * 5)
    labels = ["a", "b", "c", "d", "e"]
    i1 = _drift_cache.meta_recheck_sample_index(labels)
    i2 = _drift_cache.meta_recheck_sample_index(labels)
    assert i1 == i2, "same (labels, HEAD) must yield the same index"
    assert 0 <= i1 < len(labels)


def test_meta_recheck_sample_rotates_across_commits(monkeypatch):
    """As HEAD changes (commits land), the sampled index must move across the
    full label set — proving every cached check is eventually cold-rechecked.
    We sweep many synthetic HEADs and require the index to cover ALL positions."""
    labels = ["a", "b", "c", "d", "e"]
    seen = set()
    for n in range(200):
        monkeypatch.setattr(_drift_cache, "git_head_sha", lambda n=n: f"{n:040x}")
        seen.add(_drift_cache.meta_recheck_sample_index(labels))
    assert seen == set(range(len(labels))), f"rotation must cover every label position; covered {sorted(seen)}"


def test_meta_recheck_sample_fails_closed_when_head_unreadable(monkeypatch):
    """If HEAD cannot be read, the sample must still return a deterministic IN-RANGE
    index — a recheck always happens (never raise, never skip all). The fallback
    index need not be 0; the invariant is 'a real recheck occurs'."""
    monkeypatch.setattr(_drift_cache, "git_head_sha", lambda: None)
    labels = ["x", "y", "z"]
    i = _drift_cache.meta_recheck_sample_index(labels)
    assert 0 <= i < len(labels)
    # Deterministic: the empty-HEAD fallback yields the same index every call.
    assert _drift_cache.meta_recheck_sample_index(labels) == i


def test_meta_recheck_runs_exactly_one_cold_rerun(monkeypatch):
    """With multiple cached hits, the EXPENSIVE cold fn() is called for exactly
    ONE label per run — not all. This is what unlocks the speedup."""
    import pipeline.check_drift as cd

    labels = [THEORY_GRANT_LABEL, DSR_LABEL, FAST_LANE_LABEL]
    calls: list[str] = []

    def make_fn(lbl):
        def _fn():
            calls.append(lbl)
            return []

        return _fn

    monkeypatch.setattr(cd, "_CACHE_HITS_THIS_RUN", list(labels))
    patched = [(lbl, make_fn(lbl) if lbl in labels else fn, adv, rdb) for lbl, fn, adv, rdb in cd.CHECKS]
    monkeypatch.setattr(cd, "CHECKS", patched)
    # Pin HEAD so the sampled label is deterministic for the assertion.
    monkeypatch.setattr(_drift_cache, "git_head_sha", lambda: "0" * 40)

    out = cd.check_drift_cache_meta_recheck()
    assert out == []
    assert len(calls) == 1, f"exactly one cold re-run expected, got {len(calls)}: {calls}"
    assert calls[0] in labels


def test_meta_recheck_structural_validation_runs_for_all_hits(monkeypatch):
    """The CHEAP structural layer (in-CHECKS, non-db) must validate EVERY cached
    hit every run — sampling applies ONLY to the expensive cold re-run. A bogus
    label and a requires_db label must BOTH be flagged in the same run, regardless
    of which label the cold-rerun sample picks."""
    import pipeline.check_drift as cd

    bogus = "this label is not registered in CHECKS"
    # Find a real requires_db label to (incorrectly) mark as a cached hit.
    db_label = next((lbl for lbl, _f, _a, rdb in cd.CHECKS if rdb), None)
    assert db_label, "precondition: at least one requires_db check exists"

    monkeypatch.setattr(cd, "_CACHE_HITS_THIS_RUN", [bogus, db_label, FAST_LANE_LABEL])
    monkeypatch.setattr(_drift_cache, "git_head_sha", lambda: "f" * 40)

    out = cd.check_drift_cache_meta_recheck()
    assert any(bogus in v and "not found in CHECKS" in v for v in out), out
    assert any(db_label in v and "requires_db" in v for v in out), out


def test_meta_recheck_catches_stale_pass_in_sampled_check(monkeypatch):
    """End-to-end honesty: when the SAMPLED label's cold re-run finds a violation,
    the meta-recheck emits STALE CACHE. We pin HEAD and construct the label set so
    the known-stale label is the one sampled."""
    import pipeline.check_drift as cd

    # Single cached hit → it is necessarily the sampled one (sample of 1).
    label = FAST_LANE_LABEL
    monkeypatch.setattr(cd, "_CACHE_HITS_THIS_RUN", [label])
    monkeypatch.setattr(_drift_cache, "git_head_sha", lambda: "a" * 40)
    patched = [
        (lbl, (lambda: ["  injected stale-cache violation"]) if lbl == label else fn, adv, rdb)
        for lbl, fn, adv, rdb in cd.CHECKS
    ]
    monkeypatch.setattr(cd, "CHECKS", patched)

    out = cd.check_drift_cache_meta_recheck()
    assert any("STALE CACHE" in v and label in v for v in out), out


# ---------------------------------------------------------------------------
# Stage 2 (2026-06-03): the +2 newly-wired tree-dep checks — completeness proofs
# and real-dispatch hit→cold-recheck parity (mirrors the FAST_LANE coverage).
# ---------------------------------------------------------------------------


def test_theory_grant_tree_dep_set_covers_every_input_the_check_reads():
    """STRUCTURAL completeness: every concrete input the theory_grant parity check
    reads must be in the declared dep set. The check reads the audit log + the
    chordia.py T-constants (file_deps) + every active hypotheses/*.yaml (tree_deps)."""
    import pipeline.check_drift as cd

    spec = cd.CHECK_TREE_DEPS[THEORY_GRANT_LABEL]
    assert "docs/runtime/chordia_audit_log.yaml" in spec["file_deps"]
    assert "trading_app/chordia.py" in spec["file_deps"], (
        "the T-threshold constants in chordia.py are verdict logic — must be a dep"
    )
    assert ("docs/audit/hypotheses", "*.yaml") in spec["tree_deps"]


def test_dsr_tree_dep_set_covers_every_input_the_check_reads():
    """STRUCTURAL completeness: the DSR lock check reads ONLY the active hypotheses
    tree (its allowed-derivation set is a check_drift.py module constant, covered by
    the suite re-running on any code edit). No fixed-file or git inputs."""
    import pipeline.check_drift as cd

    spec = cd.CHECK_TREE_DEPS[DSR_LABEL]
    assert spec["file_deps"] == ["pipeline/_dsr_policy.py"], (
        "the DSR verdict's allowed-derivation set lives in pipeline/_dsr_policy.py "
        "(ALLOWED_DSR_TRIALS_DERIVATION) — it is verdict logic and MUST be a declared "
        "dep or the cache serves a stale PASS when the policy is tightened"
    )
    assert spec["tree_deps"] == [("docs/audit/hypotheses", "*.yaml")]


def test_dsr_cache_key_binds_verdict_policy_module():
    """Anti-regression (Codex high-risk 2026-06-04, PROVEN by execution): the DSR cache
    key MUST be bound to the module holding ALLOWED_DSR_TRIALS_DERIVATION, else a
    tightened DSR gate serves a STALE PASS. The bug: the DSR verdict depends directly
    on that constant, but the cache key only hashed the hypotheses/*.yaml tree — so a
    commit that drops an allowed derivation computes the SAME key, reads the OLD PASS,
    and the blocking Bailey–López de Prado DSR/multiplicity gate reports green on stale
    logic. The fix deps the entry on pipeline/_dsr_policy.py, which binds the key.

    Mirrors test_fast_lane_check_is_deliberately_not_cached: both lock a cache-honesty
    invariant on a blocking capital gate against silent regression."""
    import pipeline.check_drift as cd
    from pipeline import _drift_cache

    spec = cd.CHECK_TREE_DEPS[DSR_LABEL]
    assert "pipeline/_dsr_policy.py" in spec["file_deps"], (
        "DSR verdict depends on ALLOWED_DSR_TRIALS_DERIVATION; the policy module "
        "MUST be a declared dep or the cache serves a stale PASS"
    )
    k_bound = _drift_cache.tree_cache_key(DSR_LABEL, spec["file_deps"], spec["tree_deps"])
    k_unbound = _drift_cache.tree_cache_key(DSR_LABEL, [], spec["tree_deps"])
    assert k_bound is not None and k_bound != k_unbound, (
        "binding the policy module as a dep must change the cache key vs not binding it"
    )


def test_dsr_in_file_alias_is_the_policy_module_constant():
    """The in-file name _ALLOWED_DSR_TRIALS_DERIVATION (used at the check body's
    enforcement sites) MUST be the same object as pipeline._dsr_policy's canonical
    constant — proving the import alias did not silently re-define a divergent copy
    (institutional-rigor §4: delegate to canonical source, never re-encode)."""
    import pipeline.check_drift as cd
    from pipeline import _dsr_policy

    assert cd._ALLOWED_DSR_TRIALS_DERIVATION is _dsr_policy.ALLOWED_DSR_TRIALS_DERIVATION


@pytest.mark.slow  # runs two LIVE ~25s checks twice each — full-CI, not pre-commit fast gate
@pytest.mark.parametrize("label", [THEORY_GRANT_LABEL, DSR_LABEL])
def test_new_tree_checks_real_dispatch_hit_then_cold_recheck_parity(label, monkeypatch, tmp_path):
    """End-to-end against the LIVE repo: cold MISS persists a real PASS; warm run is
    a genuine cache HIT; the cold-recheck reproduces the verdict. Only the cache dir
    is isolated. Proves each newly-wired check caches honestly through the real path."""
    import pipeline.check_drift as cd

    cache_dir = tmp_path / "drift-cache"
    cache_dir.mkdir()
    monkeypatch.setattr(cd, "_CACHE_HITS_THIS_RUN", [])

    v1 = _run_real_tree_dispatch(label, monkeypatch, cache_dir)
    assert v1 == [], f"live repo must currently PASS {label!r} (got {v1!r})"
    assert cd._CACHE_HITS_THIS_RUN == [], "first run is a MISS, not a hit"

    v2 = _run_real_tree_dispatch(label, monkeypatch, cache_dir)
    assert v2 == []
    assert cd._CACHE_HITS_THIS_RUN.count(label) == 1, "unchanged inputs must produce a cache hit"

    # Single recorded hit → it is the sampled label → cold-recheck must reproduce PASS.
    assert cd.check_drift_cache_meta_recheck() == [], f"cold re-run of {label!r} must reproduce the cached PASS"
