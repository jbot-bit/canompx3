# Post-Result Sanity Pass

Use this prompt after any claimed result, review verdict, closure decision,
implementation summary, or plan conclusion that is about to affect next steps.

The purpose is simple:

- stop self-delusion about whether the thing is actually true
- stop tunnel vision about whether it actually matters
- stop closeness bias from missing obvious gaps on a second pass

This is one checker with three mandatory passes in fixed order.

---

## The prompt

```text
Run a POST-RESULT SANITY PASS on this work.

Object under review:
- <commit / diff / result / review / conclusion / plan>

Scope:
- <exact files / exact claim / exact artifact>

Proof hierarchy:
Use the highest-authority source available in this order:
1. current code / current files
2. git history / git state
3. direct test output / command output / generated artifact
4. canonical data tables if the claim is data-driven
5. docs only for intent or stated design, not for proof unless the claim is about documentation itself

Do not trust summaries when direct proof exists.

Pass 1: Reality check
- Prove whether the specific claim is actually true.
- Use direct evidence from the proof hierarchy.
- Do not trust summaries.
- If unproven, say unproven.
- If false, say false.
- If partially true, state exactly what is true vs not proved.
- If the claim is data / research driven, explicitly verify:
  - source-of-truth chain
  - no lookahead / leakage / holdout contamination
  - no bad joins / bad denominators / bad filters
  - no execution-cost optimism
  - no multiple-testing distortion hidden behind a single headline result

Pass 2: Tunnel-vision check
- Assume the local claim may be true.
- Check whether this is actually the main bottleneck or just a local improvement.
- State whether this work is:
  - the real bottleneck
  - a valid but lower-priority win
  - a distraction / local optimization
- Name the real blocker or higher-EV next step if different.

Pass 3: Fresh-eyes check
- Re-read it skeptically as if you were not invested in the thread.
- Look for:
  - blind spots
  - hidden assumptions
  - naming drift
  - proof gaps
  - weak tests
  - scope creep
  - technically true but operationally misleading conclusions
- Prefer concrete findings only.

Output format:
- `reality:` true | false | partial | unproven
- `focus:` correct | somewhat_tunnel_visioned | wrong_focus
- `fresh_eyes_findings:`
  - <finding or none>
- `bottom_line:`
  - <one blunt sentence>
- `next_action:`
  - <single best next move>

Decision rule:
- If `reality != true`, do not recommend rollout, closure, or promotion.
- If `reality = true` but `focus != correct`, say the work is real but not the main thing.
- If `reality = true` and `focus = correct` but findings remain, say done for now, but watch the named risk.
- If all three passes are clean, say so plainly.

Rules:
- No cheerleading.
- No trust in prior summaries without proof.
- No "looks good" unless the proof chain is explicit.
- No collapsing Pass 2 or Pass 3 into vague prose.
```

---

## When to use it

Use this after:

- a code review that says "ready"
- a closure claim like "this lane is done"
- a result summary like "candidate X won"
- a planning conclusion like "next step is Y"
- a verification pass that says "all good"

Do not use it as a substitute for:

- a full system audit
- a hypothesis T0-T8 audit
- a pre-registration writer
- deployment readiness review

This prompt is a post-result checker, not a discovery engine and not a full
governance workflow.

---

## What it should prevent

| Failure mode | How this prompt blocks it |
|---|---|
| Claim accepted from summary instead of proof | Proof hierarchy + Pass 1 |
| Technically true local win mistaken for the main task | Pass 2 |
| Familiarity bias after a long thread | Pass 3 |
| "Looks good" with weak evidence | explicit proof-chain requirement |
| Closure / promotion despite partial or unproven state | decision rule |

---

## Notes

- For code/repo claims, prefer current files and git state over memory.
- For data-driven claims, docs and prior result notes are orientation only.
- For data-driven claims in this repo, canonical proof means the canonical truth
  layers and direct outputs rebuilt from them, not derived tables or summaries.
- If the result depends on derived layers, say so explicitly and step back to
  canonical layers where feasible.
- If Pass 1 fails, stop there in substance. Passes 2 and 3 can still note
  context, but the result is not confirmed.
