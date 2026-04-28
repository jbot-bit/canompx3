"""Evidence-pack generator — derivative compiler over canonical artefacts.

Authority: docs/plans/2026-04-28-evidence-pack-generator-design.md
Stage:     docs/runtime/stages/evidence-pack-generator-stage1.md

This package is read-only over canonical sources and writes ONLY to
``reports/evidence_packs/<slug>/<run-iso8601>/``. It is derivative; not
authoritative for any decision-ledger entry.
"""

from __future__ import annotations

PACK_VERSION = "0.1.0"
