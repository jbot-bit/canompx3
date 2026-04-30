# Doctrine-Conflict Verdict — `sequential_thinking` MCP

**Date:** 2026-05-01
**Branch:** `tooling/seq-thinking-mcp-eval`
**Subject:** Does the official `@modelcontextprotocol/server-sequential-thinking` server (single tool: `sequential_thinking`, supporting `thought` / `nextThoughtNeeded` / `isRevision` / `branchFromThought`) conflict with project doctrine?

**Verdict:** **COMPLEMENT** (with two specific tensions, both resolvable by procedure rather than tool change).

---

## 1. Source quotation — `.claude/rules/institutional-rigor.md`

The rule is gated by frontmatter:
```
paths:
  - "pipeline/**"
  - "trading_app/**"
  - "scripts/**"
  - "research/**"
  - "docs/institutional/**"
  - "docs/specs/**"
```

Lines 11-17 (header + supremacy clause):
> # Institutional Rigor — Working-Style Hard Rule
>
> **Non-negotiable.** The user has been explicit: we do the proper long-term institutional-grounded fix. We do not band-aid. We do not skip. We review our own work before claiming done.
>
> This rule supersedes "just ship it" when they conflict.

The 8 non-skip rules quoted literally (headings only; full text in source):

1. **Self-review before claim-of-done — MANDATORY** (l.19). "Before marking any stage complete, run the code-review skill (or a structured self-review) on the work just done. Produce line citations, not narrative. Execute the code to verify claims, do not rely on reading." (l.21)
2. **After any fix, review the fix** (l.26). "Fixes introduce new bugs … Do not declare 'done' after a fix without a second review pass." (l.27)
3. **Refactor when you see a pattern of bugs** (l.32). "If review cycles keep finding new divergences, the architecture is wrong — stop patching." (l.33)
4. **Delegate to canonical sources — never re-encode** (l.36). "If `trading_app/config.py` has filter logic, new code must CALL the canonical implementation, not re-implement it. Parallel models drift." (l.37)
5. **No dead code — remove or populate** (l.46).
6. **No silent failures** (l.52).
7. **Ground in local resources before training memory** (l.59). "`resources/` has 15+ institutional PDFs … `docs/institutional/literature/` is the CANONICAL CITATION SOURCE for research methodology claims … Cite from there, not from training memory." (l.61-62)
8. **Verify before claiming** (l.69). "'Done' means: tests pass (show output) + dead code swept (`grep -r`) + drift check passes + self-review passed. All four required. 'It should work' is not acceptable. Run it." (l.71-72)

Plus the "treadmill signal" (l.75-77):
> If you find yourself saying "oh and also fix X" more than twice in a session, stop. The architecture is wrong. Propose a refactor. Do not keep patching.

And the forbidden list (l.85-94):
> - "I'll just fix this one thing real quick"
> - "Close enough" / "TODO later" / "we can revisit"
> - "It probably works"
> - Re-encoding logic that already exists in a canonical source
> - …

## 2. Source quotation — `CLAUDE.md` § 2-Pass Implementation Method

Lines 125-129 of `C:/Users/joshd/canompx3/CLAUDE.md`:

> ### 2-Pass Implementation Method (MANDATORY)
> 1. **Discovery:** Read affected files, understand blast radius, articulate PURPOSE before writing code.
> 2. **Implementation:** Write → verify (drift + tests + behavioral audit) → fix regressions → **self-review** → fix new findings.
>
> One task at a time. Never batch without verification. **"Done" = tests pass (show output) + dead code swept (`grep -r`) + `check_drift.py` passes + self-review passed.** All four required.

Adjacent gate at l.131-134 (Design Proposal Gate):

> Before writing ANY code on a non-trivial change, present: (1) **What** and why, (2) **Files** to touch, (3) **Blast radius**, (4) **Approach**.
>
> **Self-check (DO NOT SKIP):** Simulate happy path, edge case (NULL/empty/sparse), and failure mode internally. SHOW what you tested and found — don't just claim you checked. Performative self-review (claiming you checked without showing evidence) is worse than no self-review.

## 3. What `sequential_thinking` actually does

Per upstream README (verified via WebFetch 2026-05-01): a single tool with this schema —
- `thought` (string)
- `nextThoughtNeeded` (boolean)
- `thoughtNumber` (int) / `totalThoughts` (int)
- `isRevision` (bool, optional) / `revisesThought` (int, optional)
- `branchFromThought` (int, optional) / `branchId` (string, optional)
- `needsMoreThoughts` (bool, optional)

It is a structured scratchpad. It does not execute code, query data, write files, or bypass any project hook. It produces additional model output framed as a chain of revisable thoughts.

## 4. Tension analysis (the part that could conflict)

### Tension A — "Self-review before claim-of-done" (rule 1) vs. "revise prior conclusions" (seq-thinking's `isRevision` / `revisesThought`)

**Possible conflict:** seq-thinking's revision affordance might be construed as the self-review pass, allowing the agent to skip the *external* code-review skill / drift / tests gate (rule 8) by claiming "I already revised my own thinking."

**Resolution:** rule 1 says "run the code-review skill (or a structured self-review)" but rule 8 binds "done" to **four concrete artefacts**: "tests pass (show output) + dead code swept (`grep -r`) + drift check passes + self-review passed." Three of those four are external to the model (tests, grep, drift check). Seq-thinking cannot satisfy them by construction — it has no shell, no filesystem, no DB. Therefore mid-flow revision via `isRevision` substitutes for none of the four `done` artefacts. **No conflict if the discipline checklist enforces the four-artefact gate.**

### Tension B — Design Proposal Gate's "Self-check (DO NOT SKIP) … SHOW what you tested" vs. seq-thinking's internal-only revision trace

**Possible conflict:** the Gate explicitly forbids "performative self-review (claiming you checked without showing evidence)." A long seq-thinking trace LOOKS like rigour without actually exercising any code path. An agent could substitute thought-count for execution-evidence.

**Resolution:** the Gate's "SHOW what you tested" already names the failure mode. Seq-thinking trace ≠ "evidence." The discipline checklist's **doctrine-no-regression gate** addresses this directly: the eval transcript must show the agent still ran Discovery (file reads, grep, blast-radius mapping) and Implementation (verify gates) — seq-thinking is permitted to *augment* the Discovery articulation, not *replace* it.

### Non-tensions

- Rule 4 (delegate to canonical sources) — orthogonal; seq-thinking doesn't write code.
- Rule 7 (ground in local resources) — orthogonal; the agent must still read `resources/` PDFs and `docs/institutional/literature/`. Seq-thinking can structure the citation reasoning but is not itself a source.
- Rule 3 (refactor on pattern of bugs) — *complemented*. Seq-thinking's `branchFromThought` is well-suited to surfacing "this is the third patch — should we refactor?" mid-analysis.
- Treadmill signal — *complemented*. Branching is exactly the cognitive operation the treadmill rule asks for ("propose options").

## 5. Verdict

**COMPLEMENT.** The tool is purpose-built for the kind of multi-frame branching the project's doctrine repeatedly demands (treadmill rule, Design Gate self-check, refactor rule, Two-Track decision rule, audit-first default). The two real tensions (A and B above) are about **misuse**, not about tool semantics — they are addressable by the discipline-checklist gates in `discipline-checklist.md` and do not require modifying the upstream server.

**Operational guardrail (binding for the eval):** seq-thinking output MUST NOT be cited as evidence for any of the four `done` artefacts in rule 8. Tests, `grep -r`, `check_drift.py`, and the self-review pass remain external and unmediated.
