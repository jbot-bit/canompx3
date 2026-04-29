# MCP venv drift — cryptography<47 pin durability

mode: IMPLEMENTATION
task: Add constraints.txt + CLAUDE.md install note + drift check to harden the cryptography<47 pin against silent regression

## Scope Lock

- constraints.txt (NEW — done)
- CLAUDE.md (one-paragraph note added — done)
- pipeline/check_drift.py (add `check_cryptography_pin_holds` + register in CHECKS)

## Blast Radius

- `constraints.txt` — new advisory file. Not imported by any code. Read by humans/agents installing sidecar deps.
- `CLAUDE.md` — doctrine doc. Read on session start.
- `pipeline/check_drift.py` — new check function. Pattern matches existing `check_all_imports_resolve`: reads package metadata, returns violations list. Fail-closed only when BOTH (a) cryptography>=47 installed AND (b) fastmcp importable. Otherwise PASS (irrelevant). No schema, canonical-source, or trading-execution touch. Advisory at the venv-drift class.

## Why fail-closed

The pin protects MCP server startup. If a future `pip install -U cryptography` lands cryptography 47 while fastmcp is still in the venv, every FastMCP server (CRG, gold-db) crashes silently at next launch. Catching this at pre-commit is strictly better than catching it at next session start.

## Verification

- Run `python pipeline/check_drift.py` — passes (cryptography 46.0.7 currently installed).
- Spot-check the new function inline by simulating cryptography==47 (read-only audit, not a real install).
- Check count = previous + 1.

## Stage status

Stage 1 of 1 — single drift check addition, isolated function, no inter-file dependency.
