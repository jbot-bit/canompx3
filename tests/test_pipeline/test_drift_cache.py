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
