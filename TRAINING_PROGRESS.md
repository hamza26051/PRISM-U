# PRISM-U — Live Training & Execution Progress Tracker

**Document Purpose:** Real-time operational dashboard auto-updated directly from verified disk checkpoints, JSON training logs, and per-epoch telemetry.  
**Last Updated:** 2026-11-11 13:31:56 (Local Time)  
**Host Hardware:** NVIDIA RTX 4000 Ada Generation (21.47 GB VRAM) | Intel Core i7-14700 | 32 GB RAM  

---

## 1. Executive Status Dashboard

| Metric | Current Status / Value | Details |
| :--- | :--- | :--- |
| **Pipeline State** | ✅ **MILESTONE 6 COMPLETE** | Milestone 6 baselines verified on validation bank (10/10 pytest passing) |
| **Active Process** | `None (Idle)` | Auto-logged from disk artifacts |
| **Current Target** | `Ready for Milestone 8 (MIA Evaluator) & Milestone 7/9` | Phase 2 Reference Bank & Query Infrastructure |
| **Bank Checkpoints** | **102 / 102** models | Across 12 surrogate + 4 validation + 1 final families |
| **Completed Families** | **17 / 17** families | Includes all 6 conditions + full 55k evaluation |
| **Stage 2 Validation Gate** | ✅ PASS (`stage2_complete: true`) | `stage2_artifacts/validation_report.json` |

---

## 2. Live Epoch-by-Epoch Training Log

*No active epoch loop running at this instant — showing latest verified family status below.*

---

## 3. Milestone Progress Tracker

```text
[X] Milestone 0: Workspace Audit & Experiment Specification
[X] Milestone 1: Deterministic CIFAR-10 Partitioning (40k train, 5k aux, 5k val, 10k test)
[X] Milestone 2: Stage 1 Base Models (original_seed42.pt, retrain_seed42.pt)
[X] Milestone 3 (Seed 42): Controlled Gradient Ascent & LR Calibration Sweep
[X] Phase 2 Preflight & Option 1: Seed-42 Query Separation Check + DataLoader Benchmark
[X] Milestone 5: Full Stage 2 Model Bank (102 conditions) & Auxiliary Cache
[X] Milestone 4: Black-Box Target API (models/black_box_target.py)
[X] Milestone 6: Baseline Black-Box Auditors (Random, Fixed, Privacy-Blind Active)
[ ] Milestone 7: Response-Adaptive Audit Gain Estimator
[ ] Milestone 8: Retained-Membership Privacy Evaluator (MIA AUC & Advantage)
[ ] Milestone 9: Transcript-Level Empirical Privacy Estimator
[ ] Milestone 10: PRISM-U Acquisition Policy (Ratio & Lagrangian)
[ ] Milestone 11: Sequential Evidence Model & Calibrated Stopping Rules
[ ] Milestone 12-16: Benchmarks, Sweeps, Ablations & Dashboard
```

---

## 4. Model Bank Family Progress (Auto-Verified from Disk)

| Role | Seed | Status | Checkpoints | Original (Test / Forget) | Retrain (Test / Forget) | Orig Time | Retrain Time |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `surrogate` | `101` | ✅ Complete | 6/6 | 92.43% / 99.22% | 91.38% / 92.27% | 738.1s (12.3m) | 864.5s (14.4m) |
| `surrogate` | `102` | ✅ Complete | 6/6 | 92.22% / 99.28% | 91.84% / 92.67% | 968.5s (16.1m) | 876.2s (14.6m) |
| `surrogate` | `103` | ✅ Complete | 6/6 | 92.19% / 98.88% | 91.44% / 91.90% | 965.6s (16.1m) | 877.2s (14.6m) |
| `surrogate` | `104` | ✅ Complete | 6/6 | 92.55% / 99.45% | 91.83% / 92.20% | 966.6s (16.1m) | 876.9s (14.6m) |
| `surrogate` | `105` | ✅ Complete | 6/6 | 91.80% / 99.10% | 91.73% / 91.55% | 968.6s (16.1m) | 876.0s (14.6m) |
| `surrogate` | `106` | ✅ Complete | 6/6 | 92.08% / 99.15% | 92.04% / 91.95% | 966.9s (16.1m) | 807.9s (13.5m) |
| `surrogate` | `107` | ✅ Complete | 6/6 | 92.10% / 99.25% | 91.44% / 92.47% | 688.8s (11.5m) | 624.0s (10.4m) |
| `surrogate` | `108` | ✅ Complete | 6/6 | 91.49% / 99.02% | 91.49% / 92.17% | 689.6s (11.5m) | 623.3s (10.4m) |
| `surrogate` | `109` | ✅ Complete | 6/6 | 91.86% / 98.95% | 91.03% / 91.70% | 690.0s (11.5m) | 624.6s (10.4m) |
| `surrogate` | `110` | ✅ Complete | 6/6 | 92.24% / 99.15% | 92.09% / 91.83% | 689.3s (11.5m) | 625.1s (10.4m) |
| `surrogate` | `111` | ✅ Complete | 6/6 | 92.18% / 99.20% | 91.17% / 91.47% | 688.8s (11.5m) | 621.9s (10.4m) |
| `surrogate` | `112` | ✅ Complete | 6/6 | 92.29% / 99.08% | 92.28% / 92.33% | 687.1s (11.5m) | 623.1s (10.4m) |
| `validation` | `201` | ✅ Complete | 6/6 | 92.35% / 99.05% | 91.63% / 92.58% | 690.1s (11.5m) | 592.0s (9.9m) |
| `validation` | `202` | ✅ Complete | 6/6 | 92.34% / 99.35% | 91.85% / 92.15% | 742.6s (12.4m) | 907.9s (15.1m) |
| `validation` | `203` | ✅ Complete | 6/6 | 92.45% / 99.00% | 91.88% / 92.15% | 996.7s (16.6m) | 902.3s (15.0m) |
| `validation` | `204` | ✅ Complete | 6/6 | 92.12% / 99.12% | 91.55% / 92.00% | 977.2s (16.3m) | 883.8s (14.7m) |
| `final` | `42` | ✅ Complete | 6/6 | 92.72% / 99.28% | 91.73% / 91.97% | 1834.5s (30.6m) | 1665.3s (27.8m) |

---

## 5. Documented Bottlenecks & Mitigations

1. **Windows DataLoader Process-Spawning Bottleneck (Resolved):**
   - *Measured Before Fix:* `num_workers=2` inside the 40-epoch loop took `16.26s/epoch` (`~5.36s/epoch` pure spawn waste $\approx +1.91$ hrs across 32 models).
   - *Measured After Fix:* Moving `DataLoader` outside the epoch loop in `models/train.py` + setting `runtime.num_workers: 0` in `configs/stage2.yaml` yields **`10.90s/epoch`**, matching the `10.95s/epoch` pure GPU compute floor with identical FP32 deterministic training.
2. **Pure GA Learning Rate Sensitivity (`lr=0.0001` vs `>=0.0005`):**
   - *Verified via Option 1 (`final_responses.npz`):* `retrain` vs `original` has strong separation on $Q$ (mean JS `0.027925`, 346/5000 top-1 flips), while `ga25..ga100` at `lr=0.0001` stay close to `original` (JS `0.000008`–`0.000182`).
   - *Why Phase 2 Training Proceeds Cleanly:* `original.pt` and `retrain.pt` across all 16 seeds are 100% independent of the unlearning method; any improved unlearning trajectory (e.g., NegGrad+ / SalUn) can be generated in seconds per seed on top of these same checkpoints later.
