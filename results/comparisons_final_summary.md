# PRISM-U Comparative Auditing Summary

## Fixed-Budget Baseline Comparisons

| Policy | Budget (B_Q) | Audit AUROC | AUROC 95% CI | Brier Score | Mean Leakage (L_hat) | Unique Trajectories |
| --- | --- | --- | --- | --- | --- | --- |
| random | 5 | 1.0000 | [1.0000, 1.0000] | 0.201847 | 0.7066 | 1 |
| random | 10 | 1.0000 | [1.0000, 1.0000] | 0.118323 | 0.8247 | 1 |
| random | 20 | 1.0000 | [1.0000, 1.0000] | 0.092493 | 0.8247 | 1 |
| fixed_topk | 5 | 1.0000 | [1.0000, 1.0000] | 0.046543 | 0.9323 | 1 |
| fixed_topk | 10 | 1.0000 | [1.0000, 1.0000] | 0.009274 | 0.9323 | 1 |
| fixed_topk | 20 | 1.0000 | [1.0000, 1.0000] | 0.003118 | 0.9323 | 1 |
| privacy_blind_active | 5 | 1.0000 | [1.0000, 1.0000] | 0.136590 | 0.9479 | 3 |
| privacy_blind_active | 10 | 1.0000 | [1.0000, 1.0000] | 0.000295 | 0.9479 | 3 |
| privacy_blind_active | 20 | 1.0000 | [1.0000, 1.0000] | 0.000000 | 0.9479 | 5 |
| static_privacy_aware | 5 | 1.0000 | [1.0000, 1.0000] | 0.139276 | 0.5451 | 1 |
| static_privacy_aware | 10 | 1.0000 | [1.0000, 1.0000] | 0.083472 | 0.5556 | 1 |
| static_privacy_aware | 20 | 1.0000 | [1.0000, 1.0000] | 0.036077 | 0.6458 | 1 |
| adaptive_gain_static_privacy | 5 | 1.0000 | [1.0000, 1.0000] | 0.044541 | 0.5951 | 2 |
| adaptive_gain_static_privacy | 10 | 1.0000 | [1.0000, 1.0000] | 0.033475 | 0.5951 | 2 |
| adaptive_gain_static_privacy | 20 | 1.0000 | [1.0000, 1.0000] | 0.000063 | 0.5972 | 5 |
| full_prism_u | 5 | 1.0000 | [1.0000, 1.0000] | 0.038670 | 0.5576 | 2 |
| full_prism_u | 10 | 1.0000 | [1.0000, 1.0000] | 0.042521 | 0.5795 | 3 |
| full_prism_u | 20 | 1.0000 | [1.0000, 1.0000] | 0.000018 | 0.5993 | 5 |


## Sequential Stopping Decisions (Tri-State Verification)

| Policy | Coverage | Inconclusive | False-PASS | False-FAIL | Cond. Accuracy | Audit AUROC | Mean Queries E[T] | Median Queries | Mean Leakage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| privacy_blind_active | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 1.0000 | 8.2 | 9.0 | 0.9479 |
| static_privacy_aware | 0.0% | 100.0% | 0.0% | 0.0% | nan% | 1.0000 | 20.0 | 20.0 | 0.6458 |
| full_prism_u | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 1.0000 | 12.2 | 12.0 | 0.5885 |

