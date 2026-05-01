# Fix Drift Check 117 CRLF Perf Hang

task: Make `check_no_crlf_in_tracked_text_blobs` finish in <5s instead of 70s on Windows. Currently spawns ~20k subprocesses (git check-attr + git show per tracked file). Replace with batched `git check-attr --stdin -z` + a single `git grep -lI --cached -P '\r$'` style scan against HEAD's tree.
mode: IMPLEMENTATION
scope_lock:
  - pipeline/check_drift.py
blast_radius:
  - pipeline/check_drift.py — only `check_no_crlf_in_tracked_text_blobs()` body changed; verdict semantics preserved (HEAD-blob CRLF detection on text-attribute files). Returns identical violation list shape. No callers' logic changed.
  - Reads: git ls-files / check-attr / grep against HEAD tree (read-only). Writes: none.
  - Trading logic, allocator, strategy results, preregs, audit logs UNTOUCHED.
  - Validation: replay drift check (118 PASS expected, identical output). No new advisories. Replay pre-commit hook end-to-end <30s.
