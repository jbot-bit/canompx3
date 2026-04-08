"""Phase 4 Stage 4.1 write-side discovery gates — git cleanliness + single-use.

This module hosts the write-side Phase 4 discipline gates that MUST fire
before discovery begins enumerating experimental strategies against a
pre-registered hypothesis file. Both gates have side effects
(``check_git_cleanliness`` runs ``git`` subprocess; ``check_single_use``
queries ``experimental_strategies``) which would violate the pure-YAML
invariant of ``trading_app.hypothesis_loader`` — hence their placement in
a separate module.

Authority chain
---------------

- Registry workflow: ``docs/audit/hypotheses/README.md`` § "Workflow"
- Locked criteria: ``docs/institutional/pre_registered_criteria.md``
  Criterion 1 (pre-registration), Amendment 2.7 (Mode A holdout)
- Canonical stage doc: ``docs/plans/2026-04-08-phase-4-clean-rediscovery-design.md``
  § Stage 4.1 — "single-use enforcement" and "git cleanliness" rules
- Institutional rigor: ``.claude/rules/institutional-rigor.md`` rule 4
  (delegate to canonical sources), rule 6 (no silent failures)

Why these gates exist
---------------------

**Git cleanliness.** The SHA stamped onto every ``experimental_strategies``
row by discovery is a content hash of the hypothesis file at the moment of
the run. If the file is edited AFTER the SHA is computed but BEFORE the
first row is written, the stamp is stale and the audit trail is corrupted.
If the file is not committed at all, there is no lock point against future
edits — the "pre-registered" discipline is meaningless. Both failure modes
are caught here by verifying the file is tracked by git AND clean relative
to HEAD before the SHA is used.

**Single-use.** A pre-registered hypothesis file represents a committed
trial budget under Criterion 2 (MinBTL). Re-running the same file against
the same DB silently doubles the multiple-testing family without amending
the pre-registration. This is a form of data snooping prohibited by
``.claude/rules/quant-agent-identity.md`` § Seven Sins. The gate queries
``experimental_strategies`` for prior usage of the SHA; any prior use is a
hard fail with a clear superseding instruction.

Callers
-------

Both functions are called exclusively from
``trading_app.strategy_discovery.run_discovery`` (Phase 4 Stage 4.1d,
landing after this module). They are NOT called from the validator —
Stage 4.0's validator reads stamped rows after the fact and has no
opinion on when or how they were written. The validator only enforces
that a SHA points at a real file (that check uses
``hypothesis_loader.find_hypothesis_file_by_sha`` which is pure-YAML).

Exception semantics
-------------------

Both gates raise ``HypothesisLoaderError`` (imported from the loader) to
give callers a single exception type to catch for both load-time and
write-time discipline violations. Callers (the discovery CLI main()) are
expected to wrap the gate call chain in a try/except and translate any
``HypothesisLoaderError`` into a clean ``parser.error`` exit.

@research-source: docs/institutional/pre_registered_criteria.md
@research-source: docs/audit/hypotheses/README.md
@canonical-source: trading_app/phase_4_discovery_gates.py
@revalidated-for: Phase 4 Stage 4.1c (2026-04-08)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from trading_app.hypothesis_loader import HypothesisLoaderError


def check_git_cleanliness(path: Path) -> None:
    """Verify a hypothesis file is tracked AND clean relative to HEAD.

    Runs two git subprocess calls in sequence:

    1. ``git ls-files --error-unmatch <path>`` — exits non-zero if the file
       is not tracked by git. Catches "the user wrote a hypothesis file but
       forgot to ``git add`` + commit it" — the file exists on disk but has
       no lock point, so the SHA would drift the first time the file is
       edited after discovery runs.
    2. ``git diff --quiet HEAD -- <path>`` — exits non-zero if the tracked
       file has uncommitted modifications relative to HEAD. Catches "the
       user edited a previously-committed hypothesis file and forgot to
       commit the edits" — the SHA computed from disk differs from the SHA
       of the last committed version.

    Both must pass. A dirty-or-untracked file is a DISCIPLINE VIOLATION,
    not a soft warning: discovery would stamp rows with a SHA that cannot
    be reproduced from any committed version of the file, corrupting the
    audit trail.

    Parameters
    ----------
    path
        Filesystem path to the hypothesis YAML file. Must be an absolute
        or repo-relative path. The subprocess calls use the caller's
        current working directory — callers are expected to run discovery
        from the repo root.

    Raises
    ------
    HypothesisLoaderError
        If the file is not tracked, is dirty, does not exist, or if the
        git subprocess itself fails (e.g., git not installed). Error
        message cites the specific failure mode and the registry README.
    FileNotFoundError
        Propagated from subprocess if the ``git`` executable is not
        on PATH — that is an environment problem, not a discipline
        violation, and should surface as an operational failure.

    Notes
    -----
    Uses ``subprocess.run(..., check=False, capture_output=True)`` so that
    non-zero exit codes are handled explicitly rather than raising
    ``CalledProcessError``. This gives clearer error messages tied to the
    Phase 4 discipline rule rather than generic subprocess traceback.
    """
    if not path.is_file():
        raise HypothesisLoaderError(
            f"Hypothesis file not found at {path}. Cannot run git cleanliness "
            f"check. See docs/audit/hypotheses/README.md § 'Workflow'."
        )

    # Gate 1: is the file tracked by git?
    tracked_result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked_result.returncode != 0:
        raise HypothesisLoaderError(
            f"Hypothesis file {path} is not tracked by git. Pre-registration "
            f"requires a commit as the lock point — the SHA is only meaningful "
            f"against a committed version of the file. Run 'git add {path} && "
            f"git commit -m \"pre-register <slug>\"' before running discovery. "
            f"See docs/audit/hypotheses/README.md § 'Workflow' step 6."
        )

    # Gate 2: is the tracked file clean relative to HEAD?
    diff_result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if diff_result.returncode != 0:
        raise HypothesisLoaderError(
            f"Hypothesis file {path} has uncommitted changes relative to HEAD. "
            f"The file was edited after its last commit, which would cause the "
            f"SHA stamped on experimental_strategies rows to drift from any "
            f"committed version of the file. Commit the changes OR revert them "
            f"before running discovery. If you intended to supersede the prior "
            f"registration, create a NEW dated hypothesis file per the registry "
            f"README's 'no amending locked files — supersede only' rule. "
            f"See docs/audit/hypotheses/README.md § 'Rules'."
        )


def check_single_use(sha: str, con: Any) -> None:
    """Verify a hypothesis file's SHA has never been used before.

    Queries ``experimental_strategies`` for any row carrying the given SHA
    in the ``hypothesis_file_sha`` column. If any exist, the file has
    already been used in a discovery run — re-running it without
    amendment silently doubles the multiple-testing family (a form of
    data snooping). Registry README's "no amending — supersede only"
    rule is enforced here at runtime.

    Parameters
    ----------
    sha
        The content SHA of the hypothesis file, as produced by
        ``hypothesis_loader.compute_file_sha``. Lowercase hex digest.
    con
        A DuckDB connection (or compatible) that can query
        ``experimental_strategies``. The connection is NOT closed by
        this function — caller's responsibility. Read-only query; the
        connection does not need to be in write mode.

    Raises
    ------
    HypothesisLoaderError
        If at least one row exists with ``hypothesis_file_sha = sha``.
        Error message includes the prior usage count, the earliest
        prior ``created_at`` timestamp, and a clear instruction to
        supersede with a new dated hypothesis file.

    Notes
    -----
    The connection type is annotated as ``Any`` rather than a concrete
    DuckDB type so the gate can be tested with in-memory connections,
    mocks, or any compatible cursor protocol without importing
    ``duckdb`` at module load time (avoids a circular import risk with
    any future module that needs to call this gate).

    The query itself is parameterized via a positional placeholder to
    prevent SQL injection from a malformed SHA string. DuckDB's
    ``execute(sql, [params])`` is the canonical safe pattern.
    """
    query = """
        SELECT COUNT(*) AS n, MIN(created_at) AS first_used
        FROM experimental_strategies
        WHERE hypothesis_file_sha = ?
    """
    row = con.execute(query, [sha]).fetchone()
    if row is None:
        # No rows returned — should not happen for a COUNT query, but
        # treat as "zero prior uses" for safety.
        return
    count, first_used = row[0], row[1]
    if count > 0:
        raise HypothesisLoaderError(
            f"Hypothesis file SHA {sha[:12]}... has already been used by "
            f"{count} experimental_strategies row(s) (first at {first_used}). "
            f"A pre-registered hypothesis file is single-use by Criterion 2 / "
            f"registry README discipline — re-running it silently doubles the "
            f"multiple-testing family. To run a new discovery with the same "
            f"hypothesis family, supersede with a NEW dated file per "
            f"docs/audit/hypotheses/README.md § 'Naming convention'. If you "
            f"intended to clear the prior rows (e.g., rollback of a failed "
            f"run), delete them explicitly before re-running."
        )


__all__ = [
    "check_git_cleanliness",
    "check_single_use",
]
