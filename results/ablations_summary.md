# PRISM-U Component Ablation Suite (A1–A6)

| Ablation / Setting | Coverage | Inconclusive | False-PASS | False-FAIL | Cond. Accuracy | Audit AUROC | Mean Queries | Mean Leakage | Max Leakage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full PRISM-U (Default Reference) | 100.0% | 0.0% | 0.0% | 75.0% | 85.0% | 0.5000 | 8.7 | 0.5596 | 0.6076 |
| A1: No Privacy Penalty (lambda=0, B_P=inf) | 100.0% | 0.0% | 25.0% | 0.0% | 80.0% | 0.5000 | 9.9 | 0.9479 | 0.9479 |
| A2: No Early Stopping (force to B_Q) | 100.0% | 0.0% | 0.0% | 75.0% | 85.0% | 0.5000 | 20.0 | 0.5702 | 0.6146 |
| A3: Static Risk R(x) (No Transcript Conditioning) | 65.0% | 35.0% | 0.0% | 0.0% | 100.0% | 0.5000 | 15.4 | 0.6191 | 0.6458 |
| A4: Static Gain G(x) (No Posterior Mixture Gain) | 90.0% | 10.0% | 0.0% | 0.0% | 100.0% | 0.5000 | 12.5 | 0.6283 | 0.6493 |
| A5: No Feasibility Filter (Soft Penalty Only) | 100.0% | 0.0% | 0.0% | 75.0% | 85.0% | 0.5000 | 8.7 | 0.5596 | 0.6076 |
| A6: Ratio Acquisition G / (Delta L + eps) | 100.0% | 0.0% | 0.0% | 50.0% | 90.0% | 0.5000 | 8.2 | 0.5375 | 0.5451 |

