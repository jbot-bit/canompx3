---
name: verify-complete
description: Verify completeness of recent changes
---
Verify completeness of recent changes: $ARGUMENTS

Use when: "verify", "did I break anything", "check my work", "run checks", "is it complete", "post-edit check"

Dispatch the verify-complete agent on recent changes. If no arguments given, run against all uncommitted changes from `git status --short`.
