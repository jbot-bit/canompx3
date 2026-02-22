# Walk-Forward + Stress Validation Report

Strict verdict gates: PROMOTE only if all anti-overfit walk-forward and stress checks pass.

- B1 [KILL] base_plus_both: avg=+0.1832, uplift=+0.1549, sig/yr=51.2, wfYears=4, wf+avg=0.75, wf+uplift=0.75, slip0.05=+0.1332, trim5=+0.0816, bootP=0.99
- A2 [KILL] base_plus_both: avg=+0.1603, uplift=+0.1193, sig/yr=36.5, wfYears=3, wf+avg=1.00, wf+uplift=0.67, slip0.05=+0.1103, trim5=+0.0995, bootP=0.99
- B2 [KILL] base_plus_vol60: avg=+0.0885, uplift=+0.0774, sig/yr=75.0, wfYears=5, wf+avg=0.80, wf+uplift=0.80, slip0.05=+0.0385, trim5=-0.0198, bootP=0.94
- A1 [KILL] base: avg=+0.0814, uplift=+0.0000, sig/yr=82.2, wfYears=4, wf+avg=0.75, wf+uplift=0.00, slip0.05=+0.0314, trim5=+0.0111, bootP=0.94
- A3 [KILL] base_plus_both: avg=+0.0445, uplift=+0.0100, sig/yr=32.8, wfYears=3, wf+avg=0.67, wf+uplift=1.00, slip0.05=-0.0055, trim5=-0.0278, bootP=0.73
- A0 [PROMOTE] base_plus_both: avg=+0.5542, uplift=+0.2775, sig/yr=23.7, wfYears=3, wf+avg=1.00, wf+uplift=1.00, slip0.05=+0.5042, trim5=+0.4355, bootP=1.00