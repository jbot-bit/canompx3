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
