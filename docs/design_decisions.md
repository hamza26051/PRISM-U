# PRISM-U Design Decisions Log

This document records confirmed and provisional research and architectural decisions for PRISM-U.
Format: Date | Status (Frozen / Provisional) | Decision | Rationale | Owner

---

## 1. Hardware & Execution Strategy
- **Date:** 2026-11-05
- **Status:** Frozen
- **Decision:** Primary training and auditing will execute locally on host workstation (NVIDIA RTX 4000 Ada Generation, 20 GB VRAM, Intel i7-14700, 32 GB RAM). Google Colab serves as secondary/backup compute if multi-seed scaling demands it.
- **Rationale:** The 20 GB VRAM and 20-core CPU easily accommodate CIFAR-10 ResNet-18 training, eliminating Colab disconnect and timeout risks during development.
- **Owner:** Research & Engineering Team

## 2. Benchmark & Threat Model
- **Date:** 2026-11-05
- **Status:** Frozen
- **Primary Benchmark:** CIFAR-10 classification with adapted ResNet-18 ($3\times3$ conv1, no maxpool).
- **Forget Split:** Deterministic 10% forget set ($D_u = 4,000$ samples), 90% retain set ($D_r = 36,000$ samples), 5,000 auxiliary query pool ($Q$), 5,000 validation samples, 10,000 official test samples.
- **Black-Box Output Representation:** Full softmax class probability vectors ($\Delta^{K-1}$, continuous simplex).
- **Access Boundary:** Auditor submits query $x \in Q$ and observes $P(Y \mid x)$. Auditor has zero access to target weights, gradients, optimizer state, or training data.
- **Owner:** Research & Engineering Team

## 3. Reference Populations & Ground Truth Definitions
- **Date:** 2026-11-05
- **Status:** Frozen
- **Compliant Hypothesis ($H_C$):** Defined empirically by honest retraining from scratch on $D_r$ ($M_{\text{retrain}}$).
- **Non-Compliant Hypothesis ($H_N$):** Defined by unlearned models exhibiting incomplete forgetting: $M_{\text{original}}$ (0% unlearned) and partial unlearning checkpoints (e.g., GA-25, GA-50, GA-75).
- **Experimental Status of Full Execution:** 100% execution of an unlearning procedure (e.g. GA-100) is **not** assumed compliant; it is treated as an experimental target to be evaluated by the auditor.
- **Owner:** Research & Engineering Team

## 4. Privacy Budget & Accounting Rigor
- **Date:** 2026-11-05
- **Status:** Frozen
- **Constraint Nature:** Empirical retained-membership leakage evaluation ($\hat{L}(\tau_t)$), NOT formal Differential Privacy ($(\epsilon, \delta)$-DP).
- **Accounting Principle:** MIA AUC / advantage is an attack performance metric, not an additive unit. Cumulative transcript risk will be modeled joint-sequentially rather than as a naive sum of per-query AUCs.
- **Owner:** Research & Engineering Team

## 5. Repository Layout
- **Date:** 2026-11-05
- **Status:** Frozen
- **Structure:** Unified root layout adhering to Section 15 of `PRISM_U_Antigravity_Handoff.md`:
  - `data/`: Datasets, transforms, deterministic splitting.
  - `models/`: ResNet architecture, from-scratch training, checkpoint loading.
  - `unlearning/`: Unlearning procedures (Gradient Ascent, SalUn).
  - `surrogates/`: Reference bank layouts, response caching, diagnostics.
  - `audit/`: Black-box interface, query policies, evidence updating, stopping rules.
  - `attacks/`: Membership inference attacks and evaluation.
  - `evaluation/`: Metrics, calibration, confidence intervals, plotting.
  - `configs/`: YAML experiment configurations.
  - `scripts/`: Top-level CLI entrypoints.
  - `tests/`: Automated unit and integration tests.
- **Owner:** Research & Engineering Team

