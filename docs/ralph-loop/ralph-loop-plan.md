## Iteration: 209
## Target: trading_app/derived_state.py:36-46
## Finding: build_profile_fingerprint per-lane dict omits explicit orb_minutes; strategy_id encodes it implicitly but fingerprint consumers cannot inspect aperture without re-parsing
## Classification: [mechanical]
## Blast Radius: 1 production file (derived_state.py). 5 callers store the hash; fingerprint change will invalidate cached state envelopes (correct behavior). No logic change.
## Invariants: [1] strategy_id remains in fingerprint; [2] parse_strategy_id is the only source of orb_minutes; [3] fingerprint function signature unchanged
## Diff estimate: 5 lines
## Doctrine cited: integrity-guardian.md § 5 (Evidence over assertion — explicit encoding better than implicit); institutional-rigor.md § 4 (delegate to canonical sources — parse_strategy_id is canonical for orb_minutes extraction)
