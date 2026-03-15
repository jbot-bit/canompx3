---
name: audit
description: Run the full phased system audit (all 11 phases). No arguments needed.
disable-model-invocation: true
---
Run the full phased system audit (all 11 phases). No arguments needed.

Use when: "full audit", "run all audit phases", "system audit", "audit everything"

```bash
.venv/bin/python scripts/audits/run_all.py
```

This runs phases 0-10 in order, stopping on CRITICAL findings. See `docs/prompts/SYSTEM_AUDIT.md` for phase definitions.
