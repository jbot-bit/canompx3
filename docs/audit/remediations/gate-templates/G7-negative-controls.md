# G7 — Negative-Controls Certificate

**Candidate:** ________________________
**Test date:** ____________

---

## Purpose

Per `docs/institutional/edge-finding-playbook.md` §10: every candidate advancing to shadow or deploy must pass three control tests. Controls prevent the framework from silently passing candidates that would pass on pure noise.

## Three required controls

### Control 1 — Destruction shuffle (must FAIL)

Randomly shuffle the feature column (or trade labels) and rerun the candidate's statistic. The candidate should NOT pass its own gate on shuffled data.

```
$ python -c "
import numpy as np
# pseudocode: replace <compute_candidate_stat> with candidate's native metric
true_stat = <compute_candidate_stat>(real_data)
shuffled_stats = []
for _ in range(100):
    shuffled = real_data.copy()
    shuffled['<feature_col>'] = np.random.permutation(shuffled['<feature_col>'].values)
    shuffled_stats.append(<compute_candidate_stat>(shuffled))
print(f'true: {true_stat:.4f}, shuffled mean: {np.mean(shuffled_stats):.4f}, shuffled 95%ile: {np.percentile(shuffled_stats, 95):.4f}')
"
<paste output>
```

- [ ] Shuffled statistic FAILS the candidate's gate (95th percentile of shuffled stat < candidate's gate threshold)

### Control 2 — Known-null RNG (must FAIL)

Replace the feature column with pure Gaussian noise of matched mean/std. Candidate must NOT pass.

```
$ python -c "
# replace feature with rng-normal same mean/std
real_mean, real_std = real_data['<feature_col>'].mean(), real_data['<feature_col>'].std()
fake = real_data.copy()
fake['<feature_col>'] = np.random.normal(real_mean, real_std, len(fake))
fake_stat = <compute_candidate_stat>(fake)
print(f'fake_stat: {fake_stat:.4f} vs gate threshold: {GATE_VALUE:.4f}')
"
<paste output>
```

- [ ] Fake-feature statistic FAILS the candidate's gate

### Control 3 — Positive control (must PASS)

Replace the feature with a KNOWN-good predictor (e.g., `pnl_r` itself — the outcome). Candidate's framework must pass trivially, proving the framework is not broken.

```
$ python -c "
pos = real_data.copy()
pos['<feature_col>'] = pos['pnl_r']  # perfect predictor (trivially)
pos_stat = <compute_candidate_stat>(pos)
print(f'positive_control_stat: {pos_stat:.4f} vs gate threshold: {GATE_VALUE:.4f}')
"
<paste output>
```

- [ ] Positive control PASSES the candidate's gate (if not, framework is broken)

## Verdict

- [ ] CLEAR — all 3 controls behave as expected (Control 1 FAIL, Control 2 FAIL, Control 3 PASS)
- [ ] FAIL — at least one control violates expectations

## Failure disposition

- Control 1 or 2 PASS unexpectedly → framework is too permissive, or candidate's metric is not actually testing the feature. Investigate before any shadow deploy.
- Control 3 FAIL → framework is broken. All prior results using this framework are suspect.

## Literature citation

- `docs/institutional/edge-finding-playbook.md` §10

## Authored by / committed

- Author: ____________________________
- Commit SHA of candidate's script: ________________
- Commit SHA of control script: ________________
- Commit SHA of this certificate: ________________
