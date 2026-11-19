# PRISM-U Comparative Auditing Summary

## Fixed-Budget Baseline Comparisons

| Policy | Budget (B_Q) | Audit AUROC | AUROC 95% CI | Brier Score | Mean Leakage (L_hat) | Unique Trajectories |
| --- | --- | --- | --- | --- | --- | --- |
| random | 5 | 0.9062 | [0.6875, 1.0000] | 0.176200 | 0.7066 | 1 |
| random | 10 | 0.9531 | [0.8125, 1.0000] | 0.127321 | 0.8247 | 1 |
| fixed_topk | 5 | 1.0000 | [1.0000, 1.0000] | 0.070535 | 0.9323 | 1 |
| fixed_topk | 10 | 1.0000 | [1.0000, 1.0000] | 0.022195 | 0.9323 | 1 |
| privacy_blind_active | 5 | 1.0000 | [1.0000, 1.0000] | 0.123437 | 0.9479 | 9 |
| privacy_blind_active | 10 | 1.0000 | [1.0000, 1.0000] | 0.100723 | 0.9479 | 10 |
| static_privacy_aware | 5 | 0.7500 | [0.5074, 0.9531] | 0.159328 | 0.5451 | 1 |
| static_privacy_aware | 10 | 0.8750 | [0.6398, 1.0000] | 0.115330 | 0.5556 | 1 |
| adaptive_gain_static_privacy | 5 | 0.9062 | [0.7188, 1.0000] | 0.106632 | 0.5516 | 12 |
| adaptive_gain_static_privacy | 10 | 0.8906 | [0.7500, 1.0000] | 0.117306 | 0.5689 | 13 |
| full_prism_u | 5 | 0.9375 | [0.8125, 1.0000] | 0.110397 | 0.5566 | 10 |
| full_prism_u | 10 | 0.8438 | [0.6406, 1.0000] | 0.135994 | 0.5594 | 14 |


## Sequential Stopping Decisions (Tri-State Verification)

| Policy | Coverage | Inconclusive | False-PASS | False-FAIL | Cond. Accuracy | Audit AUROC | Mean Queries E[T] | Median Queries | Mean Leakage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| privacy_blind_active | 60.0% | 40.0% | 0.0% | 0.0% | 100.0% | 1.0000 | 7.5 | 6.5 | 0.9479 |
| static_privacy_aware | 5.0% | 95.0% | 0.0% | 0.0% | 100.0% | 0.8750 | 9.9 | 10.0 | 0.5556 |
| full_prism_u | 75.0% | 25.0% | 0.0% | 50.0% | 86.7% | 0.5000 | 7.8 | 8.0 | 0.5594 |

