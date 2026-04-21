# G3 — DSR + N̂ Certificate

**Candidate:** `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`
**Hypothesis family:** `live deployed lane retrospective`
**Pre-reg:** `N/A — deployed lane retrospective verdict`

---

## Purpose

This certificate follows the Terminal 3 template but uses the Phase B corrective rule: bracket `rho_hat` at `0.3 / 0.5 / 0.7`, then fail closed for `KEEP` unless `DSR > 0.95` at the directive's conservative `rho_hat=0.7` bound.

## Live computation

| Input | Value | Source / query |
|---|---|---|
| `T` | `1508` | `active_validated_setups.sample_size` |
| `SR_hat` | `0.090400` | `active_validated_setups.sharpe_ratio` |
| `V[SR_n]` | `0.006712` | `trading_app.dsr.estimate_var_sr_from_db(GOLD_DB_PATH)` |
| `M` raw trials | `35616` | `active_validated_setups.n_trials_at_discovery` |

### Bracketed `N_hat` / DSR

| `rho_hat` | `N_hat = rho + (1-rho) * M` | `SR_0` | `DSR` |
|---|---:|---:|---:|
| `0.3` | `24931.5` | `0.334141` | `0.000000` |
| `0.5` | `17808.5` | `0.327673` | `0.000000` |
| `0.7` | `10685.5` | `0.317625` | `0.000000` |

## Verdict

- [x] `FAIL` at the directive's conservative `rho_hat=0.7` bound.
- [x] `rho_hat=0.7` result used for the Phase B keep/degrade decision.
- [x] gold-db MCP unavailable in-session; canonical substitute was read-only Python against `pipeline.paths.GOLD_DB_PATH`.

## Literature citation

- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` Eq. 2 + Appendix A.3 Eq. 9
- `docs/institutional/pre_registered_criteria.md` Amendment 2.1

## Authored by / committed

- Author: `Codex`
- Commit SHA of candidate script: `5e768af8`
- Commit SHA of DSR helper: `5e768af8609c`
- Pinned `pre_registered_criteria.md` commit SHA at eval time: `126ed6b883fb`
