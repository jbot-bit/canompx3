"""LLM Hypothesis Proposer (LHP) — Track A of v2 plan.

Produces draft pre-registration YAMLs at
``docs/audit/hypotheses/YYYY-MM-DD-llm-<slug>.draft.yaml`` from an LLM grounded
in the local literature corpus and adjacency context against
``validated_setups``.

The ``.draft.yaml`` suffix is intentional: ``trading_app.hypothesis_loader``
discovers files via ``directory.glob("*.yaml")``, so drafts are invisible to
the pipeline until a human reviews and renames them to ``.yaml``.

Modules
-------
- ``literature_index`` — load ``docs/institutional/literature/*.md``
- ``adjacency`` — read-only ``validated_setups`` queries
- ``llm_client`` — OpenRouter / Anthropic wrapper with cost ceiling
- ``yaml_emitter`` — write draft file
- ``static_checks`` — pre-write validation, delegating to canonical sources

This package is leaf: nothing in ``pipeline/``, ``trading_app/``, or
``research/`` imports from here.
"""

from __future__ import annotations
