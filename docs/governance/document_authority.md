# Document Authority Registry

Date: 2026-04-11

## Purpose

This file is the registry for document roles in this repo.

The problem it fixes is simple: authority had been spread across multiple docs,
plans, and habits. That makes it too easy for stale prose to survive after code
or policy changes. This registry states what each top-level document is allowed
to mean, and what it is not allowed to mean.

## Authority map

| Surface | Role | Binding for | Not binding for |
|---|---|---|---|
| `CLAUDE.md` | repo operating contract | repo workflow, routing, architectural guardrails, verification expectations | trading logic details, research truth, planned-but-unbuilt features |
| `TRADING_RULES.md` | live trading doctrine | live trading semantics, session/entry/filter interpretation, what is currently tradeable | research methodology, future feature plans |
| `RESEARCH_RULES.md` | research doctrine | methodology, statistical standards, discovery discipline, interpretation discipline | live deployment state, portfolio routing |
| `docs/institutional/pre_registered_criteria.md` | locked promotion/validation policy | the criteria validated strategies must satisfy | implementation details of how code is written |
| `docs/governance/system_authority_map.md` | whole-project authority/context map | where major truth surfaces live, how categories relate, what is linked vs derived | live runtime values by itself |
| `ROADMAP.md` | planning inventory | planned or not-yet-built work only | current implementation truth, live behavior, policy |
| `HANDOFF.md` | cross-tool baton | current session baton, recent changes, local warnings | durable truth when code or canonical docs disagree |
| `docs/plans/` | design history and active decisions | durable design decisions and rationale while active and not archived | live runtime truth when code or DB disagree |
| `docs/ARCHITECTURE.md` | operational reference guide | command reference and orientation when consistent with code | current runtime truth when code/DB disagree |
| `docs/MONOREPO_ARCHITECTURE.md` | monorepo orientation reference | service inventory and repository navigation | current DB location, canonical runtime policy, live deployment semantics |
| `docs/context/*.md` | generated task-routing orientation | deterministic task-context views when rendered from `context/registry.py` | live runtime truth by itself; policy beyond the canonical doctrine docs |
| `REPO_MAP.md` | generated inventory | file/directory layout snapshot generated from the repo | behavior, policy, runtime state |

## Document classes

Use these classes instead of mixing live facts, decisions, and snapshots in the
same prose.

| Class | Allowed content | Required marker |
|---|---|---|
| `generated` | Dynamic facts rendered from code, registries, or published read models | Source path plus "do not edit by hand" |
| `snapshot` | Dated/commit-stamped state captured for audit history | Date/commit plus "not live truth" |
| `decision` | Rationale, verdicts, and ownership decisions | Evidence links and scope |
| `contract` | Active interface expectations between repo surfaces | Owning source and verification gate |
| `result` | Research/audit output | Verdict, provenance, reproduction, caveats |
| `handoff` | Current cross-tool baton | Compact current-state warning only |

Dynamic facts include counts, active lanes, strategy IDs, live profile contents,
session lists, schemas, and current git/runtime state. Those belong in code,
canonical data, generated docs, or stamped snapshots — not unqualified prose.

## Conflict rules

1. If live code or DB behavior disagrees with a document, code/DB wins for
   current behavior. The document must then be updated or explicitly marked
   stale.
2. For trading decisions, `TRADING_RULES.md` wins over general operating docs.
3. For methodology and evidence standards, `RESEARCH_RULES.md` wins over
   trading summaries and plans.
4. `ROADMAP.md` is planning-only. It must not be cited as proof that something
   already exists or works.
5. `HANDOFF.md` and `docs/plans/` are context and design surfaces, not
   canonical research truth. They can explain decisions; they do not override
   canonical layers or live code.
6. `docs/ARCHITECTURE.md`, `docs/MONOREPO_ARCHITECTURE.md`, and `REPO_MAP.md`
   are reference surfaces. They must point back to canonical code or generated
   sources and must not drift into pseudo-authority.
7. Generated docs win over hand-edited copies of the same facts only when they
   are in sync with their generator. If generated output drifts from the
   generator, the generator wins and the doc must be re-rendered.

## Maintenance rules

1. Any new canonical surface should be added here in the same change that
   introduces it.
2. If a code change materially changes live behavior, update the relevant
   authority doc in the same workstream or record the gap explicitly in
   `HANDOFF.md`.
3. Archived plans are historical only. They do not carry current authority.
4. Do not commit preregs/results with placeholder provenance markers such as
   `UNSTAMPED` or `TO_BE_STAMPED`.
5. Do not describe a `design_only` prereg as executable. It must remain blocked
   until its data tables and bounded runner exist.
6. If a doc says "current", "live", or "deployed" about dynamic state, it must
   either link to the canonical source or declare itself as a dated snapshot.
