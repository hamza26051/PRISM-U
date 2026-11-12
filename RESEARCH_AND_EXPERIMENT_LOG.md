# PRISM-U Research, Development & Experimentation Log

**Project:** PRISM-U — Privacy-Budgeted Active Auditing for Machine Unlearning  
**Team:** Final-Year Project (FYP) Research Team  
**Institution:** Computer Science & AI Research  
**Log Maintained By:** Antigravity AI Research & Engineering Agent  
**First Entry Date:** November 5, 2026  
**Last Updated:** November 5, 2026  
**Scientific Integrity Standard:** Zero hallucination policy. Every numerical value recorded in this document is traced directly to raw source JSON files, YAML configs, checkpoint binary headers, or direct system telemetry. Any value not yet computed or verified is explicitly marked as `[Pending Measurement]`.

---

## 1. System & Hardware Environment Specification

### A. Host Workstation Specifications (Physical Hardware)
*Source: Verified via Windows System Telemetry and User Hardware Telemetry (November 5, 2026)*

| Component | Hardware Specification | Verification Source |
| :--- | :--- | :--- |
| **Primary GPU** | **NVIDIA RTX 4000 Ada Generation** (20 GB GDDR6 VRAM) | Verified via `Win32_VideoController` (Driver: 32.0.15.9671) and User Telemetry |
| **Integrated GPU** | Intel(R) UHD Graphics 770 | Verified via `Win32_VideoController` (Driver: 32.0.101.7088) |
| **Processor (CPU)** | **Intel(R) Core(TM) i7-14700** (20 Cores: 8 P-cores, 12 E-cores; 28 Threads; Base: 2.10 GHz) | Verified via User System Telemetry |
| **Installed RAM** | **32.0 GB** (Speed: 4400 MT/s) | Verified via User System Telemetry |
| **Storage** | **2.29 TB NVMe SSD** (310 GB used, 1.98 TB available) | Verified via User System Telemetry |
| **Operating System** | Windows 11 Enterprise (64-bit) | Verified via PowerShell Environment |
| **Host Python** | Python 3.11.9 (`C:\Program Files\Python311\python.exe`) | Verified via `py -0p` command |
| **Project Virtualenv** | `.venv` (Python 3.11.9 + PyTorch `2.6.0+cu124` + torchvision `0.21.0+cu124`) | **Provisioned & Fully Verified** |
| **CUDA Verification** | CUDA 12.4 runtime active; Recognized GPU: `NVIDIA RTX 4000 Ada Generation` (`21.47 GB` total VRAM) | Verified via `torch.cuda.is_available()` and `get_device_name(0)` |
| **Test Suite Status** | `pytest tests/` (3/3 unit tests passing); `scripts/verify_inputs.py` (PASS); `experiments/smoke_test.py` (CUDA forward/backward PASS) | All static and GPU plumbing tests verified |

---

## 2. Dataset Partitioning Specification (Milestone 1 — Verified)

*Source: [`splits/cifar10_seed42.json`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/splits/cifar10_seed42.json) and [`configs/cifar10_stage1.yaml`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/configs/cifar10_stage1.yaml)*

- **Dataset:** CIFAR-10 (60,000 $32\times32$ color images across 10 balanced object classes).
- **RNG Algorithm:** NumPy `default_rng(seed=42)`
- **Mathematical Partition Constraints (All Disjointness Checks Passed):**

$$\begin{aligned}
D_{\text{total}} &= D_{\text{target}} \cup D_{\text{aux}} \cup D_{\text{val}} = 50,000 \\
D_{\text{target}} &= D_r \cup D_u = 40,000 \\
D_r \cap D_u &= \emptyset \quad (|D_r| = 36,000, \, |D_u| = 4,000) \\
D_{\text{target}} \cap D_{\text{aux}} &= \emptyset \quad (|D_{\text{aux}}| = 5,000) \\
D_{\text{target}} \cap D_{\text{val}} &= \emptyset \quad (|D_{\text{val}}| = 5,000) \\
D_{\text{aux}} \cap D_{\text{val}} &= \emptyset \\
D_{\text{test}} &= \text{Official CIFAR-10 Test Set} \quad (|D_{\text{test}}| = 10,000)
\end{aligned}$$

### Exact Split Partition Sizes:
1. **Target Training Pool ($D_{\text{target}}$):** `40,000` samples (Indices: `target_train`)
   - **Retain Set ($D_r$):** `36,000` samples (90.0% of target train; indices: `retain`)
   - **Forget Set ($D_u$):** `4,000` samples (10.0% of target train; indices: `forget`)
2. **Auxiliary Query Pool ($Q$):** `5,000` samples (indices: `auxiliary`)
3. **Validation Calibration Set ($D_{\text{val}}$):** `5,000` samples (indices: `validation`)
4. **Official Test Set ($D_{\text{test}}$):** `10,000` samples (indices: `official_test`, `0..9999`)

---

## 3. Stage 1 Baseline Target Training & Evaluation Log (Milestone 2 — Verified)

### A. Architectural Hyperparameters (ResNet-18 Adapted for CIFAR)
*Source: [`models/resnet.py`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/models/resnet.py) and [`configs/cifar10_stage1.yaml`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/configs/cifar10_stage1.yaml)*

- **Base Architecture:** `torchvision.models.resnet18(weights=None, num_classes=10)`
- **Modifications for CIFAR-10 ($32\times32$):**
  - `conv1`: `nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)` (replaces standard ImageNet $7\times7, \text{stride}=2$).
  - `maxpool`: `nn.Identity()` (bypasses downsampling to retain spatial feature maps).
  - `fc`: `nn.Linear(512, 10)`
- **Total Parameter Count:** `11,173,962` (Verified via `torch.numel()` on CIFAR ResNet-18)
- **Trainable Parameter Count:** `11,173,962` (100% trainable; no frozen layers)
- **Runtime Environment:** Verified on host NVIDIA RTX 4000 Ada Generation (21.47 GB VRAM, PyTorch 2.6.0+cu124)

### B. Training Hyperparameters
*Source: [`configs/cifar10_stage1.yaml`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/configs/cifar10_stage1.yaml) and training run logs*

- **Optimizer:** `torch.optim.SGD`
  - Base Learning Rate ($\eta$): `0.1`
  - Momentum ($\mu$): `0.9`
  - Weight Decay ($\lambda$): `0.0005` ($5\times 10^{-4}$)
- **Learning Rate Scheduler:** `torch.optim.lr_scheduler.MultiStepLR`
  - Milestones: `[20, 30]`
  - Gamma ($\gamma$): `0.1`
  - Schedule:
    - Epochs 1–20: $\eta = 0.1$
    - Epochs 21–30: $\eta = 0.01$
    - Epochs 31–40: $\eta = 0.001$
- **Batch Size:** `128`
- **Total Epochs:** `40`
- **Loss Function:** `nn.CrossEntropyLoss(label_smoothing=0.0)`
- **Data Augmentation (Training):**
  - `RandomCrop(32, padding=4)`
  - `RandomHorizontalFlip()`
  - `ToTensor()`
  - `Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])`
- **Evaluation Transform:** `ToTensor()`, `Normalize(...)`

---

### C. Exact Training Run Records (Stage 1 Artifacts)

#### 1. Target Model: Original Model ($M_{\text{original}}$, Seed 42)
*Source: [`results/train_original_seed42.json`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/results/train_original_seed42.json) and [`results/eval_original_seed42.json`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/results/eval_original_seed42.json)*

- **Training Hardware Environment:** Google Colab GPU (Tesla T4, 16 GB VRAM; PyTorch 2.11.0+cu130, Python 3.13.15)
- **Training Wall-Clock Time:** `1834.544555902481` seconds (~30 minutes 35 seconds)
- **Best Validation Accuracy achieved during training:** `0.9268` (Epoch 38)
- **Checkpoint Location:** `checkpoints/original_seed42.pt`
- **Checkpoint Exact File Size:** `44,777,035` bytes
- **Exact Evaluation Metrics (from `eval_original_seed42.json`):**

| Data Partition | $N$ | Classification Accuracy | Cross-Entropy Loss | Mean Confidence | Mean Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Retain Set ($D_r$)** | 36,000 | `0.9927500000000000` (99.275%) | `0.0284729767204428` | `0.9832237362861633` | `0.0562835074961185` |
| **Forget Set ($D_u$)** | 4,000 | `0.9927500000000000` (99.275%) | `0.0253981685783071` | `0.9842795133590698` | `0.0539211668074131` |
| **Auxiliary Set ($Q$)** | 5,000 | `0.9292000000000000` (92.920%) | `0.2559773013759124` | `0.9604726433753967` | `0.1134804189205170` |
| **Validation Set ($D_{\text{val}}$)** | 5,000 | `0.9268000000000000` (92.680%) | `0.2477209996065405` | `0.9586779475212097` | `0.1165057122707367` |
| **Official Test Set ($D_{\text{test}}$)** | 10,000 | `0.9272000000000000` (92.720%) | `0.2511528227212712` | `0.9592579007148743` | `0.1158430278301239` |

---

#### 2. Honest Retrained Reference Model ($M_{\text{retrain}}$, Seed 42)
*Source: [`results/train_retrain_seed42.json`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/results/train_retrain_seed42.json) and [`results/eval_retrain_seed42.json`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/results/eval_retrain_seed42.json)*

- **Training Hardware Environment:** Google Colab GPU (Tesla T4, 16 GB VRAM; PyTorch 2.11.0+cu130, Python 3.13.15)
- **Training Wall-Clock Time:** `1665.28555893898` seconds (~27 minutes 45 seconds)
- **Best Validation Accuracy achieved during training:** `0.9248` (Epoch 39)
- **Checkpoint Location:** `checkpoints/retrain_seed42.pt`
- **Checkpoint Exact File Size:** `44,776,907` bytes
- **Exact Evaluation Metrics (from `eval_retrain_seed42.json`):**

| Data Partition | $N$ | Classification Accuracy | Cross-Entropy Loss | Mean Confidence | Mean Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Retain Set ($D_r$)** | 36,000 | `0.9895277777777778` (98.953%) | `0.0370521974196835` | `0.9790898561477661` | `0.0682484433054924` |
| **Forget Set ($D_u$)** | 4,000 | `0.9197500000000000` (91.975%) | `0.2709346471092408` | `0.9553377628326416` | `0.1288582980632782` |
| **Auxiliary Set ($Q$)** | 5,000 | `0.9174000000000000` (91.740%) | `0.2786258330354315` | `0.9519886970520020` | `0.1329746842384338` |
| **Validation Set ($D_{\text{val}}$)** | 5,000 | `0.9248000000000000` (92.480%) | `0.2534486047201610` | `0.9529903531074524` | `0.1327217817306519` |
| **Official Test Set ($D_{\text{test}}$)** | 10,000 | `0.9173000000000000` (91.730%) | `0.2827233930719945` | `0.9522013664245605` | `0.1335160583257675` |

---

## 4. Scientific Baseline Comparison: $M_{\text{original}}$ vs. $M_{\text{retrain}}$

*Mathematical difference observed on held-out forget set $D_u$:*
- **Accuracy Drop on $D_u$:** $99.275\% \to 91.975\%$ ($\Delta = -7.300\%$)
- **Loss Increase on $D_u$:** $0.025398 \to 0.270935$ ($\Delta = +0.245537$)
- **Entropy Increase on $D_u$:** $0.053921 \to 0.128858$ ($\Delta = +0.074937$)
- **Retain Utility Preservation ($D_r$):** Retain accuracy remains $98.953\%$ vs. $99.275\%$ ($\Delta = -0.322\%$), showing honest retraining preserves retained capability while removing memorization of $D_u$.

---

## 5. Next Planned Experiment: Seed-42 Gradient Ascent Unlearning Generation (Milestone 3)

### A. Pre-Declared Protocol & Hyperparameters
*Source: [`configs/stage2.yaml`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/configs/stage2.yaml) and [`unlearning/gradient_ascent.py`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/unlearning/gradient_ascent.py)*

- **Source Model:** `checkpoints/original_seed42.pt`
- **Target Unlearning Algorithm:** Gradient Ascent (GA)
- **Objective:** $\theta_{t+1} = \theta_t + \eta_{\text{GA}} \nabla_\theta \mathcal{L}_{\text{CE}}(f(X_u; \theta_t), Y_u)$ for $(X_u, Y_u) \sim D_u$.
- **Learning Rate ($\eta_{\text{GA}}$):** `0.0001` ($1\times 10^{-4}$)
- **Batch Size:** `128`
- **Total Steps:** `100` steps
- **Execution Truncation Checkpoints Planned:**
  - `ga25.pt`: Step 25 (25% execution)
  - `ga50.pt`: Step 50 (50% execution)
  - `ga75.pt`: Step 75 (75% execution)
  - `ga100.pt`: Step 100 (100% execution — *Experimental Target*)
- **Target Destination:** `checkpoints/final/seed42/ga/`
- **Status:** `[Pending Local GPU Execution upon virtualenv installation]`

---

## 6. Audit Trail of Every Data File Referenced

| Artifact | Type | File Location | Verification Status |
| :--- | :--- | :--- | :--- |
| `cifar10_stage1.yaml` | YAML Config | `configs/cifar10_stage1.yaml` | Verified on disk |
| `stage2.yaml` | YAML Config | `configs/stage2.yaml` | Verified on disk |
| `cifar10_seed42.json` | JSON Split | `splits/cifar10_seed42.json` | Verified on disk (1,169,299 bytes) |
| `original_seed42.pt` | PyTorch Weights | `checkpoints/original_seed42.pt` | Verified on disk (44,777,035 bytes) |
| `retrain_seed42.pt` | PyTorch Weights | `checkpoints/retrain_seed42.pt` | Verified on disk (44,776,907 bytes) |
| `eval_original_seed42.json` | JSON Evaluation | `results/eval_original_seed42.json` | Verified on disk (1,157 bytes) |
| `eval_retrain_seed42.json` | JSON Evaluation | `results/eval_retrain_seed42.json` | Verified on disk (1,163 bytes) |
| `train_original_seed42.json`| JSON Train Log | `results/train_original_seed42.json` | Verified on disk (8,192 bytes) |
| `train_retrain_seed42.json` | JSON Train Log | `results/train_retrain_seed42.json` | Verified on disk (8,558 bytes) |
| `probs_original_seed42.npz` | NPZ Array Cache | `results/probs_original_seed42.npz` | Verified on disk (2,309,547 bytes) |
| `probs_retrain_seed42.npz` | NPZ Array Cache | `results/probs_retrain_seed42.npz` | Verified on disk (2,313,409 bytes) |
| `cifar-10-python.tar.gz`   | Dataset Archive | `data/cifar10/cifar-10-python.tar.gz` | Verified on disk (170,498,071 bytes, MD5: `C58F30108F718F92721AF3B95E74349A`) |
| `cifar-10-batches-py/`     | Extracted Data  | `data/cifar10/cifar-10-batches-py/`  | Verified: 5 train batches, 1 test batch, metadata |


---

## 7. Raw Dataset Download & Storage Log (November 5, 2026 — Verified & Extracted)

- **Target Destination:** `./data/cifar10/`
- **Compressed Archive:** `data/cifar10/cifar-10-python.tar.gz`
- **Exact File Size:** `170,498,071` bytes (~170.50 MB)
- **Archive MD5 Checksum:** `C58F30108F718F92721AF3B95E74349A`
- **Official Origin:** University of Toronto Department of Computer Science (`https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`)
- **Extraction Directory:** `data/cifar10/cifar-10-batches-py/`
- **Verified Extracted Files:**

| File Name | Exact Disk Size | Description | Integrity Verification |
| :--- | :---: | :--- | :---: |
| `data_batch_1` | `31,035,704` bytes | Training partition 1 (10,000 images, labels) | Verified via `data.datasets.make_datasets` |
| `data_batch_2` | `31,035,320` bytes | Training partition 2 (10,000 images, labels) | Verified via `data.datasets.make_datasets` |
| `data_batch_3` | `31,035,999` bytes | Training partition 3 (10,000 images, labels) | Verified via `data.datasets.make_datasets` |
| `data_batch_4` | `31,035,696` bytes | Training partition 4 (10,000 images, labels) | Verified via `data.datasets.make_datasets` |
| `data_batch_5` | `31,035,623` bytes | Training partition 5 (10,000 images, labels) | Verified via `data.datasets.make_datasets` |
| `test_batch` | `31,035,526` bytes | Official test set (10,000 images, labels) | Verified via `data.datasets.make_datasets` |
| `batches.meta` | `158` bytes | Label dictionary (10 classes) | Verified via `data.datasets.make_datasets` |
| `readme.html` | `88` bytes | Dataset metadata HTML | Verified on disk |

- **Dataset Loader Test:** Executed `from data.datasets import make_datasets; train_aug, train_eval, test = make_datasets('./data/cifar10')`. Result: `Train size: 50,000`, `Test size: 10,000`. All batches loaded and verified on host.
- **Current Execution Status:** **Dataset download and extraction 100% complete. All training is paused until tomorrow per team instruction.**

---

## 8. Milestone 3 Execution: Seed-42 Controlled Gradient Ascent Unlearning (November 10, 2026 — Completed & Evaluated)

### A. Execution Environment & Hyperparameter Record
- **Date & Time of Run:** November 10, 2026 (~09:00 AM local time)
- **Execution Hardware:** NVIDIA RTX 4000 Ada Generation (21.47 GB VRAM), Intel Core i7-14700, PyTorch `2.6.0+cu124`
- **Source Model:** `checkpoints/original_seed42.pt` (ResNet-18 trained on $D_r \cup D_u$)
- **Algorithm:** Gradient Ascent (GA) on forget set $D_u$ (`unlearning/gradient_ascent.py`)
- **Loss Formulation:** $\mathcal{L}_{\text{GA}} = -\mathcal{L}_{\text{CE}}(f(X_u; \theta), Y_u)$
- **Learning Rate ($\eta_{\text{GA}}$):** `0.0001` ($1\times 10^{-4}$)
- **Optimizer:** `torch.optim.SGD` (momentum=0.0, weight_decay=0.0)
- **Batch Size:** `128`
- **Total Steps:** `100` steps (~3.2 epochs over the 4,000 forget samples)
- **BatchNorm Mode:** Kept in `.eval()` mode to preserve running statistics of the underlying representation
- **Milestones:** Checkpoints saved at 25%, 50%, 75%, and 100% of the 100 steps:
  - Step 25: `ga25.pt` (`44,768,132` bytes)
  - Step 50: `ga50.pt` (`44,768,132` bytes)
  - Step 75: `ga75.pt` (`44,768,132` bytes)
  - Step 100: `ga100.pt` (`44,768,258` bytes)
- **Checkpoint Directory:** `stage2_artifacts/checkpoints/final/seed42/ga/`
- **Evaluation Record:** `stage2_artifacts/logs/final/seed42/family_evaluation.json`

---

### B. Empirical Evaluation Results Across All 6 Conditions (Seed 42)

*Evaluated on NVIDIA RTX 4000 Ada GPU using batch size 256 across all 55,000 partition samples:*

#### 1. Performance on Forget Set ($D_u$, $N = 4,000$) — The Primary Unlearning Target

| Condition | Fraction | Step | Forget Accuracy | Forget Cross-Entropy | Mean Confidence | Mean Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`original`** | 0% | 0 | `0.9927500000000000` (99.275%) | `0.0253952006697655` | `0.9842800102233886` | `0.0539219594001770` |
| **`ga25`** | 25% | 25 | `0.9927500000000000` (99.275%) | `0.0258453186154366` | `0.9842028236389160` | `0.0540881676673889` |
| **`ga50`** | 50% | 50 | `0.9927500000000000` (99.275%) | `0.0264604145884514` | `0.9840747451782227` | `0.0543371007442474` |
| **`ga75`** | 75% | 75 | `0.9925000000000000` (99.250%) | `0.0270508854389191` | `0.9839636840820313` | `0.0545891802310944` |
| **`ga100`** | 100% | 100 | `0.9917500000000000` (99.175%) | `0.0278052688837051` | `0.9838098983764648` | `0.0549297313690186` |
| **`retrain`** (Ideal) | — | — | `0.9197500000000000` (91.975%) | `0.2709327020645141` | `0.9553327026367188` | `0.1288651494979858` |

#### 2. Performance on Retain Set ($D_r$, $N = 36,000$) — Utility & Preservation

| Condition | Fraction | Retain Accuracy | Retain Cross-Entropy | Mean Confidence | Mean Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`original`** | 0% | `0.9927500000000000` (99.275%) | `0.0284727379414770` | `0.9832233378092448` | `0.0562844306627909` |
| **`ga25`** | 25% | `0.9926944444444444` (99.269%) | `0.0285165654222171` | `0.9832026477389866` | `0.0563294445408715` |
| **`ga50`** | 50% | `0.9925555555555555` (99.256%) | `0.0286228262450960` | `0.9831626658969456` | `0.0564114031526778` |
| **`ga75`** | 75% | `0.9924444444444445` (99.244%) | `0.0287582782308261` | `0.9831125314500597` | `0.0565230766932170` |
| **`ga100`** | 100% | `0.9922222222222222` (99.222%) | `0.0289703618023131` | `0.9830566283331977` | `0.0566631191836463` |
| **`retrain`** (Ideal) | — | `0.9895277777777778` (98.953%) | `0.0370529515743256` | `0.9790888701544868` | `0.0682506650553809` |

#### 3. Generalization Performance on Official Test Set ($D_{\text{test}}$, $N = 10,000$)

| Condition | Test Accuracy | Test Cross-Entropy | Mean Confidence | Mean Entropy |
| :--- | :---: | :---: | :---: | :---: |
| **`original`** | `0.9272` (92.72%) | `0.2511424067974091` | `0.9592566488265991` | `0.1158454469561577` |
| **`ga25`** | `0.9273` (92.73%) | `0.2512666563510895` | `0.9592872858047485` | `0.1158438874244690` |
| **`ga50`** | `0.9271` (92.71%) | `0.2514818139076233` | `0.9593025075912476` | `0.1158521036624908` |
| **`ga75`** | `0.9268` (92.68%) | `0.2518110302448273` | `0.9593010255813599` | `0.1158889301061630` |
| **`ga100`** | `0.9264` (92.64%) | `0.2521989478588104` | `0.9593002687454224` | `0.1159251077771187` |
| **`retrain`** (Ideal) | `0.9173` (91.73%) | `0.2827169815063477` | `0.9521996760368348` | `0.1335192612290382` |

---

### C. Scientific & Experimental Issue Identified: Unlearning Severity Under Default Hyperparameters

> [!WARNING] Critical Research Finding / Issue Identified
> Under the repository's inherited default hyperparameters (`learning_rate: 0.0001`, `total_steps: 100`):
> 1. **Minimal Forget Loss Movement:** Forget cross-entropy moved only slightly from $0.025395$ to $0.027805$ after 100 steps ($\Delta = +0.002410$).
> 2. **Near-Zero Accuracy Reduction on Forget Set:** Forget accuracy only dropped from $99.275\%$ to $99.175\%$ (a decrease of only 4 misclassified examples out of 4,000!).
> 3. **Contrast with Honest Retraining:** Honest retraining drops forget accuracy to $91.975\%$ (a $7.3\%$ drop) and increases forget loss to $0.270933$ ($\Delta = +0.245537$, which is **100x larger** than GA-100's movement).
> 4. **Implication for Auditing:** Under these settings, `ga25`, `ga50`, `ga75`, and `ga100` are behaviorally indistinguishable from `original` (all clustered between 99.18% and 99.28% forget accuracy). The unlearning algorithm did not effectively unlearn the forget set within 100 steps at $\eta = 10^{-4}$.

---

## 9. Updated Audit Trail of All Artifacts Generated

| Artifact | Type | File Location | Exact Disk Size | Verification Status |
| :--- | :--- | :--- | :---: | :--- |
| `ga25.pt` | Checkpoint | `stage2_artifacts/checkpoints/final/seed42/ga/ga25.pt` | `44,768,132` bytes | Verified on GPU |
| `ga50.pt` | Checkpoint | `stage2_artifacts/checkpoints/final/seed42/ga/ga50.pt` | `44,768,132` bytes | Verified on GPU |
| `ga75.pt` | Checkpoint | `stage2_artifacts/checkpoints/final/seed42/ga/ga75.pt` | `44,768,132` bytes | Verified on GPU |
| `ga100.pt` | Checkpoint | `stage2_artifacts/checkpoints/final/seed42/ga/ga100.pt` | `44,768,258` bytes | Verified on GPU |
| `family_evaluation.json` | Evaluation JSON | `stage2_artifacts/logs/final/seed42/family_evaluation.json` | `5,633` bytes | Verified on GPU |
| `model_manifest.csv` | Bank Manifest | `stage2_artifacts/manifests/model_manifest.csv` | `10,480` bytes | Verified on disk |



---

## 10. Gradient Ascent Learning Rate Calibration Sweep (November 10, 2026 — Verified)

*Source: [`stage2_artifacts/diagnostics/ga_calibration_sweep.json`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/stage2_artifacts/diagnostics/ga_calibration_sweep.json)*

- **Objective:** Determine the boundary where unconstrained Gradient Ascent exhibits measurable unlearning without triggering catastrophic parameter divergence or retain utility collapse.
- **Hardware:** NVIDIA RTX 4000 Ada Generation (21.47 GB VRAM)
- **Candidate LRs Evaluated:** `[0.0002, 0.0005, 0.001, 0.002, 0.005]`
- **Duration:** 39.8 seconds total for all 5 sweeps (100 steps each + evaluations)

### Empirical Findings:

1. **$\eta = 0.0002$:**
   - Step 25: Forget Acc = `0.99275` (Loss=`0.026367`), Retain Acc = `0.99258` (Loss=`0.028604`), Test Acc = `0.9272`
   - Step 50: Forget Acc = `0.99175` (Loss=`0.027955`), Retain Acc = `0.99219` (Loss=`0.029045`), Test Acc = `0.9264`
   - Step 75: Forget Acc = `0.99150` (Loss=`0.029927`), Retain Acc = `0.99172` (Loss=`0.029830`), Test Acc = `0.9265`
   - Step 100: Forget Acc = `0.99050` (Loss=`0.033100`), Retain Acc = `0.99083` (Loss=`0.031400`), Test Acc = `0.9267`
   - *Behavior:* Stable and monotonic, but unlearning remains mild ($0.225\%$ drop).

2. **$\eta = 0.0005$ (Instability Threshold):**
   - Step 25: Forget Acc = `0.99150` (Loss=`0.028475`), Retain Acc = `0.99208` (Loss=`0.029323`), Test Acc = `0.9269`
   - Step 50: Forget Acc = `0.98775` (Loss=`0.039292`), Retain Acc = `0.98933` (Loss=`0.035287`), Test Acc = `0.9243`
   - Step 75: Forget Acc = `0.96875` (Loss=`0.090870`), Retain Acc = `0.97289` (Loss=`0.074210`), Test Acc = `0.9096`
   - Step 100: Forget Acc = `0.64075` (Loss=`2.307339`), Retain Acc = `0.65378` (Loss=`2.176505`), Test Acc = `0.6286`
   - *Behavior:* Shows smooth gradual unlearning through step 75 ($99.28\% \to 96.88\%$), but catastrophic collapse occurs between step 75 and 100 as gradients compound.

3. **$\eta \ge 0.001$:**
   - Rapid parameter divergence occurs ($\text{Loss} \to \text{NaN}$, Accuracy $\to 10.0\%$ random guess) by step 50–75.

### Research Conclusion:
Unconstrained Gradient Ascent on a forget set without a retain preservation penalty inevitably couples forget loss growth with retain degradation. In PRISM-U:
- The default $\eta = 0.0001$ models a **mildly non-compliant / incompletely unlearned model** ($H_N$).
- Honest retraining $M_{\text{retrain}}$ provides the **true compliant anchor** ($H_C$).
- This provides the exact behavioral contrast needed to build and validate the black-box auditor.


---

## 11. Option 1 Diagnostic: Seed-42 Black-Box Auxiliary Query Separation (November 10, 2026 — Verified)

*Source: [`stage2_artifacts/responses/final_responses.npz`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/stage2_artifacts/responses/final_responses.npz) & [`stage2_artifacts/query_pool/query_indices.npy`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/stage2_artifacts/query_pool/query_indices.npy)*

- **Command Executed:** `.venv\Scripts\python.exe scripts/cache_responses.py --config configs/stage2.yaml --role final --device cuda`
- **Query Pool Size ($|Q|$):** `5,000` auxiliary CIFAR-10 images (`split="auxiliary"`, verified disjoint from `target_train` and `validation`).
- **Response Tensor Shape:** `(6, 5000, 10)` float32 softmax probabilities across `['retrain', 'original', 'ga25', 'ga50', 'ga75', 'ga100']`.

### A. Pairwise Divergence vs. Honest Reference (`retrain`, $H_C$) on $Q$ ($N = 5,000$)

| Model Condition | Mean JS Divergence | Max JS Divergence | Mean $L_1$ Distance | Top-1 Prediction Flips |
| :--- | :---: | :---: | :---: | :---: |
| **`retrain`** ($H_C$) | `0.000000` | `0.000000` | `0.000000` | `0.00%` (`0 / 5,000`) |
| **`original`** ($H_N$, 0% GA) | `0.027925` | `0.669685` | `0.147602` | `6.92%` (`346 / 5,000`) |
| **`ga25`** ($H_N$, 25% GA) | `0.027937` | `0.669367` | `0.147679` | `6.90%` (`345 / 5,000`) |
| **`ga50`** ($H_N$, 50% GA) | `0.027950` | `0.668848` | `0.147791` | `6.96%` (`348 / 5,000`) |
| **`ga75`** ($H_N$, 75% GA) | `0.027994` | `0.668865` | `0.147998` | `6.98%` (`349 / 5,000`) |
| **`ga100`** (Experimental) | `0.028025` | `0.668293` | `0.148143` | `6.98%` (`349 / 5,000`) |

### B. Pairwise Divergence vs. Original Model (`original`, 0% GA) on $Q$ ($N = 5,000$)

| Model Condition | Mean JS Divergence | Max JS Divergence | Mean $L_1$ Distance | Top-1 Prediction Flips |
| :--- | :---: | :---: | :---: | :---: |
| **`retrain`** ($H_C$) | `0.027925` | `0.669685` | `0.147602` | `6.92%` (`346 / 5,000`) |
| **`original`** (0% GA) | `0.000000` | `0.000000` | `0.000000` | `0.00%` (`0 / 5,000`) |
| **`ga25`** (25% GA) | `0.000008` | `0.000639` | `0.001930` | `0.04%` (`2 / 5,000`) |
| **`ga50`** (50% GA) | `0.000041` | `0.003056` | `0.004464` | `0.18%` (`9 / 5,000`) |
| **`ga75`** (75% GA) | `0.000092` | `0.007540` | `0.006565` | `0.32%` (`16 / 5,000`) |
| **`ga100`** (100% GA) | `0.000182` | `0.014389` | `0.009383` | `0.48%` (`24 / 5,000`) |

### C. Research Bottleneck Identified from Diagnostic
1. **Signal Exists Between $H_C$ and $H_N$:** `retrain` and `original`/GA differ strongly on $Q$ (mean JS `0.0279`, 346 top-1 flips), confirming the 5,000 auxiliary queries carry genuine compliance signal (once averaged across the 12-seed surrogate bank to separate training-seed noise from unlearning signal).
2. **Low Inter-Fraction Diversity in Pure GA ($\eta = 10^{-4}$):** `ga25`, `ga50`, `ga75`, and `ga100` differ from `original` by only `0.000008`–`0.000182` mean JS (>150x smaller than the `retrain` vs `original` gap). Thus, `ga25..ga75` act as near-clones of `original` rather than spanning graduated unlearning severities.
3. **Strategic Plan:** Proceeding with Phase 2 training (`original` and `retrain` across seeds `101–112` and `201–204`) is **100% reusable and unblocked** because `original.pt` and `retrain.pt` are shared across all future unlearning algorithms (including SalUn or retain-regularized GA, which can be generated in seconds per seed on top of those checkpoints later).

---

## 12. Phase 2 Engineering Benchmark & Bottleneck Resolution (November 10, 2026)

### A. Measured Training Throughput on RTX 4000 Ada (Single Epoch, 40k Images, Batch Size 128, FP32 Deterministic)

| Configuration | Measured Epoch Time | Root Cause / Observation |
| :--- | :---: | :--- |
| **`num_workers=2` (Old `train.py` inside epoch loop)** | `16.26 s / epoch` | Windows `spawn()` recreated 2 Python worker processes every epoch (+5.36s overhead/epoch $\approx$ +1.91 hours wasted across 32 models). |
| **`num_workers=4` (1st epoch spawn)** | `21.34 s / epoch` | Heavy Windows multiprocessing startup overhead on first epoch. |
| **`num_workers=4` (2nd epoch persistent)** | `10.90 s / epoch` | Zero spawn overhead once persistent workers are warm. |
| **`num_workers=0` (In-process DataLoader)** | `10.90 s / epoch` | Matches warm persistent-worker throughput with zero spawn startup penalty. |
| **Pure GPU Compute Baseline (313 steps, synthetic GPU tensor)** | `10.95 s / epoch` | Proves FP32 deterministic GPU math is saturated at ~10.9s/epoch; CPU data loading with `num_workers=0` adds ~0s bottleneck. |

### B. Engineering Fixes Applied (Verified via `pytest` 3/3 Pass & `smoke_test.py` CUDA Pass)
1. [`data/datasets.py`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/data/datasets.py): Added optional `generator` parameter to `make_loader()`.
2. [`models/train.py`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/models/train.py): Moved `train_loader` instantiation **outside** the 40-epoch loop and reseeded `train_generator.manual_seed(seed + epoch)` per epoch.
3. [`configs/stage2.yaml`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/configs/stage2.yaml): Set `runtime.num_workers: 0` for local Windows execution to eliminate multiprocessing spawn delays while preserving exact FP32 `cudnn.deterministic=True` training behavior.
4. **Estimated Total Phase 2 Compute Budget (Remaining 16 Families):** ~10.9s $\times$ 40 epochs $\times$ 2 models $\approx$ ~14.5–15.5 minutes per family (~4.0–4.2 hours total for all 16 families, saving ~2+ hours vs unoptimized code).


---

## 13. Phase 2 Pilot Family Results: Surrogate Seed `101` (November 10, 2026 — Completed & Verified)

*Sources:*
- [`stage2_artifacts/logs/surrogate/seed101/train_original.json`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/stage2_artifacts/logs/surrogate/seed101/train_original.json)
- [`stage2_artifacts/logs/surrogate/seed101/train_retrain.json`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/stage2_artifacts/logs/surrogate/seed101/train_retrain.json)
- [`stage2_artifacts/logs/surrogate/seed101/family_evaluation.json`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/stage2_artifacts/logs/surrogate/seed101/family_evaluation.json)

### A. Training Execution Summary (NVIDIA RTX 4000 Ada, FP32 Deterministic)

| Model | Training Set Size | Epochs | Best Val Accuracy | Training Seconds | Speedup vs. Stage 1 Colab T4 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`surrogate_seed101_original`** | 40,000 ($D_r \cup D_u$) | 40 | `0.9306` (93.06%) | `738.10 s` (12.30 min) | **2.49$\times$ faster** (vs `1834.54s` T4) |
| **`surrogate_seed101_retrain`** | 36,000 ($D_r$ only) | 40 | `0.9188` (91.88%) | `864.46 s` (14.41 min) | **1.93$\times$ faster** (vs `1665.29s` T4) |

### B. Full 6-Condition Family Evaluation (`family_evaluation.json`)

| Condition | Hypothesis | Retain Acc ($D_r$, 36k) | Forget Acc ($D_u$, 4k) | Forget Loss | Validation Acc (5k) | Test Acc (10k) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`retrain`** | `HC` | `0.990278` (99.03%) | `0.922750` (92.28%) | `0.278604` | `0.9188` (91.88%) | `0.9138` (91.38%) |
| **`original`** | `HN` | `0.992194` (99.22%) | `0.992250` (99.23%) | `0.028497` | `0.9306` (93.06%) | `0.9243` (92.43%) |
| **`ga25`** | `HN` | `0.992167` (99.22%) | `0.991750` (99.18%) | `0.028696` | `0.9306` (93.06%) | `0.9244` (92.44%) |
| **`ga50`** | `HN` | `0.992056` (99.21%) | `0.991500` (99.15%) | `0.028945` | `0.9308` (93.08%) | `0.9247` (92.47%) |
| **`ga75`** | `HN` | `0.992111` (99.21%) | `0.991250` (99.13%) | `0.029190` | `0.9306` (93.06%) | `0.9246` (92.46%) |
| **`ga100`** | `EXPERIMENTAL` | `0.992222` (99.22%) | `0.991000` (99.10%) | `0.029523` | `0.9306` (93.06%) | `0.9242` (92.42%) |


---

## 14. Phase 2 Surrogate Bank Completion & Clean Pause Checkpoint (November 10, 2026)

*Verified directly from `stage2_artifacts/logs/{role}/seed{seed}/family_evaluation.json`, `train_*.json`, and `retrain.resume.pt` on disk.*

### A. Overall Phase 2 Progress Snapshot (Paused Cleanly at User Request)
- **Checkpoints Saved on Disk:** **`80 / 102`** models (`72/72` Surrogate + `6/6` Final + `2/24` Validation)
- **Completed Families:** **`13 / 17`** (`surrogate 101–112`, `final 42`)
- **In-Progress Family (`validation seed 201`):**
  - `original_seed201`: **Completed** (`Test Acc = 0.9235`, `Val Acc = 0.9262`, `Retain Acc = 0.9914`, `Forget Acc = 0.9905`, `training_seconds = 690.1s`)
  - `retrain_seed201`: **Paused Cleanly & Verified** at **Epoch 12 / 40** (`best_val = 0.8018`, `history_len = 12`, `retrain.resume.pt` = `89,492,134` bytes, verified via `torch.load`). Next run resumes automatically at **Epoch 13 / 40**.
- **Remaining Families:** `validation` seeds `201` (epochs 13–40 + GA), `202`, `203`, `204`, followed by `cache_responses`, `diagnostics`, and `validate_stage2` (~1.2 hours estimated on RTX 4000 Ada).

### B. Complete Verified Surrogate Bank Results (`seeds 101–112`, 72 Models)

| Seed | Orig Test Acc | Orig Val Acc | Orig Retain Acc | Orig Forget Acc | Orig Time | Retrain Test Acc | Retrain Val Acc | Retrain Retain Acc | Retrain Forget Acc | Retrain Time | GA-100 Forget Acc (Loss) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`101`** | `0.9243` | `0.9306` | `0.9922` | `0.9922` | `738.1s` | `0.9138` | `0.9188` | `0.9903` | `0.9227` | `864.5s` | `0.9910` (`0.02952`) |
| **`102`** | `0.9222` | `0.9258` | `0.9928` | `0.9928` | `968.5s` | `0.9184` | `0.9212` | `0.9925` | `0.9267` | `876.2s` | `0.9920` (`0.02741`) |
| **`103`** | `0.9219` | `0.9280` | `0.9894` | `0.9888` | `965.6s` | `0.9144` | `0.9230` | `0.9876` | `0.9190` | `877.2s` | `0.9878` (`0.04137`) |
| **`104`** | `0.9255` | `0.9272` | `0.9939` | `0.9945` | `966.6s` | `0.9183` | `0.9198` | `0.9926` | `0.9220` | `876.9s` | `0.9935` (`0.02463`) |
| **`105`** | `0.9180` | `0.9244` | `0.9917` | `0.9910` | `968.6s` | `0.9173` | `0.9196` | `0.9910` | `0.9155` | `876.0s` | `0.9895` (`0.03374`) |
| **`106`** | `0.9208` | `0.9208` | `0.9913` | `0.9915` | `966.9s` | `0.9204` | `0.9238` | `0.9924` | `0.9195` | `807.9s` | `0.9898` (`0.03122`) |
| **`107`** | `0.9210` | `0.9244` | `0.9917` | `0.9925` | `688.8s` | `0.9144` | `0.9224` | `0.9914` | `0.9247` | `624.0s` | `0.9922` (`0.03007`) |
| **`108`** | `0.9149` | `0.9226` | `0.9902` | `0.9902` | `689.6s` | `0.9149` | `0.9212` | `0.9908` | `0.9217` | `623.3s` | `0.9905` (`0.03447`) |
| **`109`** | `0.9186` | `0.9236` | `0.9901` | `0.9895` | `690.0s` | `0.9103` | `0.9182` | `0.9827` | `0.9170` | `624.6s` | `0.9892` (`0.03469`) |
| **`110`** | `0.9224` | `0.9244` | `0.9902` | `0.9915` | `689.3s` | `0.9209` | `0.9216` | `0.9914` | `0.9183` | `625.1s` | `0.9912` (`0.02976`) |
| **`111`** | `0.9218` | `0.9238` | `0.9908` | `0.9920` | `688.8s` | `0.9117` | `0.9126` | `0.9873` | `0.9147` | `621.9s` | `0.9915` (`0.03572`) |
| **`112`** | `0.9229` | `0.9240` | `0.9910` | `0.9908` | `687.1s` | `0.9228` | `0.9248` | `0.9919` | `0.9233` | `623.1s` | `0.9900` (`0.03487`) |

### C. How to Resume Phase 2 Seamlessly on Your Next Session
Run the single resumable command from the repository root:
```powershell
.venv\Scripts\python.exe -u run_stage2.py --config configs/stage2.yaml --phase all --device cuda
```
This will automatically:
1. Skip all 12 completed surrogate seeds (`101–112`) and `validation seed 201 original`.
2. Resume `validation seed 201 retrain` directly from **Epoch 13/40**.
3. Complete `validation` seeds `202–204` and `final` seed `42` checks.
4. Cache black-box probabilities over the 5,000-query pool (`surrogate_responses.npz`, `validation_responses.npz`, `final_responses.npz`), compute separability diagnostics, and pass the Stage 2 validation gate (`validation_report.json`).


---

## 15. Phase 2 Completion: Validation Bank, Response Caches, Diagnostics & Hard Gate (November 11, 2026 — Verified)

*Sources:*
- [`stage2_artifacts/validation_report.json`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/stage2_artifacts/validation_report.json)
- [`stage2_artifacts/diagnostics/surrogate_diagnostics.json`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/stage2_artifacts/diagnostics/surrogate_diagnostics.json)
- [`stage2_artifacts/diagnostics/validation_diagnostics.json`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/stage2_artifacts/diagnostics/validation_diagnostics.json)
- [`stage2_artifacts/logs/validation/seed201..204/family_evaluation.json`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/stage2_artifacts/logs/validation/)

### A. Resumption Optimization Bottleneck Resolved Before Resuming
- **Bottleneck Found:** When `scripts/build_bank.py` resumed across already-completed surrogate seeds `101–112`, `evaluate_family()` lacked a skip check and unconditionally reloaded all 6 checkpoints per seed across 55,000 samples (~45s/seed $\approx 9\text{ minutes}$ of redundant evaluation).
- **Mitigation Applied:** Added a skip check for existing `family_evaluation.json` (unless `--force`) in [`scripts/build_bank.py`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/scripts/build_bank.py) and accurate `session_epochs` ETA tracking in [`models/train.py`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/models/train.py).
- **Result:** Resuming skipped all 12 completed surrogate families in $<5\text{ seconds}$ and picked up `retrain_seed201` cleanly at Epoch 13/40.

### B. Complete Validation Bank Results (`seeds 201–204`, 24 Models)

| Seed | Orig Test Acc | Orig Val Acc | Orig Retain Acc | Orig Forget Acc | Orig Time | Retrain Test Acc | Retrain Val Acc | Retrain Retain Acc | Retrain Forget Acc | Retrain Time | GA-100 Forget Acc (Loss) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`201`** | `0.9235` | `0.9262` | `0.9914` | `0.9905` | `690.1s` | `0.9163` | `0.9240` | `0.9922` | `0.9258` | `592.0s` *(resumed Ep 13–40)* | `0.9873` (`0.03649`) |
| **`202`** | `0.9234` | `0.9292` | `0.9942` | `0.9935` | `742.6s` | `0.9185` | `0.9220` | `0.9921` | `0.9215` | `907.9s` | `0.9930` (`0.02473`) |
| **`203`** | `0.9245` | `0.9288` | `0.9929` | `0.9900` | `996.7s` | `0.9188` | `0.9212` | `0.9879` | `0.9215` | `902.3s` | `0.9893` (`0.03308`) |
| **`204`** | `0.9212` | `0.9218` | `0.9912` | `0.9913` | `977.2s` | `0.9155` | `0.9168` | `0.9901` | `0.9200` | `883.8s` | `0.9910` (`0.03355`) |

### C. Black-Box Response Caching (`stage2_artifacts/responses/`)
All 102 model conditions were queried strictly through the black-box softmax output interface over the frozen 5,000-image auxiliary pool (`query_indices.npy`):
- `surrogate_responses.npz`: shape `(72, 5000, 10)` (`12 HC`, `48 HN`, `12 EXPERIMENTAL`)
- `validation_responses.npz`: shape `(24, 5000, 10)` (`4 HC`, `16 HN`, `4 EXPERIMENTAL`)
- `final_responses.npz`: shape `(6, 5000, 10)` (`1 HC`, `4 HN`, `1 EXPERIMENTAL`)
- Every tensor passed `finite: true` and `probability_simplex: true` checks.

### D. Surrogate & Validation Separability Diagnostics (`EXPERIMENTAL` Excluded)

| Metric | Surrogate Bank (`101–112`, 12 HC vs 48 HN) | Validation Bank (`201–204`, 4 HC vs 16 HN) |
| :--- | :---: | :---: |
| **JS Divergence Mean** | `0.004658` | `0.010355` |
| **JS Divergence Median** | `0.000132` | `0.000153` |
| **JS Divergence Max** | `0.130292` | `0.303021` |
| **Mean $L_1$ Separation** | `0.058921` | `0.089108` |
| **Median Fisher Score** | `0.077095` | `0.288489` |
| **Mean Top-1 Disagreement** | `0.067908` (6.79%) | `0.067481` (6.75%) |

#### Key Scientific Result — Surrogate Query Ranking Generalization to Unseen Validation Models:
Queries ranked **exclusively on the Surrogate Bank (`101–112`)** transfer strongly to the **unseen Validation Bank (`201–204`)**:

| Top-$k$ Queries (Ranked by Surrogate JS) | Validation Mean JS | Validation Mean $L_1$ | Validation Top-1 Disagreement | Gain vs Pool Mean JS (`0.010355`) |
| :---: | :---: | :---: | :---: | :---: |
| **Top 10** | `0.082913` | `0.565685` | `49.38%` (`0.49375`) | **8.01$\times$** |
| **Top 25** | `0.072709` | `0.541114` | `45.75%` (`0.45750`) | **7.02$\times$** |
| **Top 50** | `0.061647` | `0.493368` | `43.50%` (`0.43500`) | **5.95$\times$** |
| **Top 100** | `0.051655` | `0.429143` | `37.70%` (`0.37703`) | **4.99$\times$** |
| **Top 250** | `0.048551` | `0.409156` | `36.73%` (`0.36725`) | **4.69$\times$** |
| **Top 500** | `0.042214` | `0.357148` | `31.22%` (`0.31219`) | **4.08$\times$** |
| **Top 1000** | `0.033993` | `0.290295` | `23.77%` (`0.23766`) | **3.28$\times$** |

### E. Stage 2 Hard Validation Gate (`validation_report.json`)
- **`stage2_complete`: `true`**
- All checks passed: `split_valid: true`, `model_bank` (`72/72`, `24/24`, `6/6`), `seed_isolation: true`, `labels.ok: true`, `family_evaluations` (`17/17`), `query_pool.ok: true` (`5000`), `response_caches` (`ok: true`), `diagnostics_present: true`, `final_seed_not_used_in_reference_banks: true`.
- *Note on minor matplotlib cache warning:* `matplotlib` logged a Windows permission notice when accessing `AppData\Local\matplotlib` inside sandbox and automatically wrote to temp directory; all plots (`separability_histogram.png`, `top100_queries.png`) saved cleanly.

---

## 16. Next Recommended Development Phase (Phase 3 / Milestones 4, 6–11)
With Phase 2 (`Milestone 5`) 100% validated (`102 / 102` checkpoints and response caches ready), the scientific ML & auditing pipeline begins:
1. **Milestone 4 — Strict Black-Box Target Interface (`models/black_box_target.py`):** Query-budget enforcement, probability-only access, immutable transcript $\tau_t$, and weight-isolation tests.
2. **Milestone 6 — Basic Auditing Baselines (`audit/`):** Random query policy, Fixed top-$k$ policy (using surrogate ranking), and Privacy-blind active policy to confirm baseline compliance discrimination on validation (`201–204`).
3. **Milestone 8 — Retained-Membership Privacy Evaluator (`attacks/`):** Independent MIA evaluator (AUC, advantage, confidence intervals) measuring actual retained-set privacy leakage from transcripts.
4. **Milestones 7, 9, 10, 11 — Core PRISM-U Novelty:** Response-adaptive $G_t(x \mid \tau_t)$, transcript-dependent empirical leakage $\hat{L}(\tau_t) / \Delta L_t(x \mid \tau_t)$, ratio/Lagrangian acquisition, and sequential LLR with **PASS / FAIL / INCONCLUSIVE** abstention.


---

## 17. Milestone 4 Completion: Strict Black-Box Target Interface & Isolation Tests (November 11, 2026 — Verified)

### A. Implemented Modules
1. [`audit/transcript.py`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/audit/transcript.py):
   - Immutable `TranscriptEntry` (frozen dataclass with read-only NumPy response copy and UTC timestamp).
   - `AuditTranscript` append-only transcript container $\tau_t = \{(x_1, r_1), \dots, (x_t, r_t)\}$ with defensive response array copies and JSON serialization.
2. [`models/black_box_target.py`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/models/black_box_target.py):
   - `BlackBoxTarget` abstract interface exposing only `query()`, `query_batch()`, `query_budget`, `queries_used`, `queries_remaining`, `access_mode`, `pool_size`, and `transcript`.
   - `PyTorchBlackBoxTarget`: Wraps live `nn.Module` / `.pt` checkpoints inside a non-inspectable closure (disabling gradients, never storing model/weights/checkpoint/labels on instance attributes).
   - `CachedBlackBoxTarget`: Fast deterministic wrapper backed by Stage 2 `.npz` probability caches for budget sweeps and ablations.
   - `load_cached_target_bank()`: Separates anonymized `BlackBoxTarget` instances (`target_000`, ...) from `ground_truth` metadata held strictly by the evaluation harness.
   - Access modes supported: `probabilities` (frozen primary setting), `logits` (`PyTorchBlackBoxTarget` only), and `top1`.
   - Enforcement guarantees: atomic `QueryBudgetExhaustedError`, `InvalidQueryError` (blocks probing outside permitted auxiliary pool $Q$), and `BlackBoxIsolationError` (blocks forbidden attribute reads and arbitrary tensor queries when pool-bound).

### B. Automated Verification Suite (`tests/test_black_box_target.py` & `tests/test_static.py`)
- **Command Run:** `.venv\Scripts\pytest.exe -v`
- **Outcome:** **`7 passed in 4.92s`**
  - `test_audit_modules_cannot_import_model_or_checkpoint_internals`: AST static verification confirming zero imports of `torch`, `models.resnet`, `load_model`, `state_dict`, or `__closure__` inside `audit/`.
  - `test_pytorch_target_isolation_and_access_modes`: Verifies isolation exceptions, probabilities/logits/top1 outputs, defensive transcript immutability, and atomic budget checks.
  - `test_cached_target_budget_pool_and_transcript`: Verifies pool range validation, budget exhaustion, JSON transcript export, and `top1`/rejection of `logits`.
  - `test_live_pytorch_vs_cached_target_equivalence_on_real_validation_bank`: Verified on `validation_seed201_retrain`, `validation_seed201_original`, and `validation_seed201_ga50` against `validation_responses.npz` (`atol=1e-3`, measured max float32 CPU vs CUDA kernel reduction diff `~3.4e-4`, 100% top-1 `argmax` match).


---

## 18. Milestone 6 Completion: Basic Auditing Baselines & Validation Discrimination (November 11, 2026 — Verified)

*Sources:*
- [`results/milestone6_validation_baselines.json`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/results/milestone6_validation_baselines.json)
- [`results/milestone6_auroc_vs_budget.png`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/results/milestone6_auroc_vs_budget.png)

### A. Implemented Modules & Experiment Hygiene
- [`audit/evidence.py`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/audit/evidence.py): `SurrogateEvidenceModel` using Centered Log-Ratio (CLR) embeddings of reference softmax probabilities, calibrated strictly via Leave-One-Seed-Out (LOSO) CV across the 12 surrogate seeds (`101–112`, selecting $\sigma = 1.0$). Positive `LLR` denotes compliance ($H_C$); negative `LLR` denotes non-compliance ($H_N$).
- [`audit/random_policy.py`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/audit/random_policy.py): `RandomQueryPolicy` (uniform sampling without replacement).
- [`audit/fixed_policy.py`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/audit/fixed_policy.py): `FixedTopKPolicy` (static surrogate JS-divergence ranking).
- [`audit/audit_gain.py`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/audit/audit_gain.py): `StaticJSGainEstimator`, `AdaptivePosteriorGainEstimator` ($G_t(x \mid \tau_t)$ via expected binary posterior entropy reduction), and `PrivacyBlindActivePolicy`.
- [`audit/baseline_auditors.py`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/audit/baseline_auditors.py): Black-box audit driver using solely `target.query()`.
- [`evaluation/metrics.py`](file:///c:/Users/23K-0068/Desktop/New%20folder%20(2)/evaluation/metrics.py): AUROC, stratified bootstrap 95% CIs, Brier calibration, coverage, false-PASS/false-FAIL rates.
- **Zero Leakage Verified:** All parameters fitted solely on `surrogate_responses.npz`; evaluated on held-out `validation_responses.npz` (`seeds 201–204`, 4 $H_C$ vs 16 $H_N$); `final` seed `42` remains completely untouched.

### B. Empirical Baseline Results on Held-Out Validation Bank (`4 HC` vs `16 HN`)

| Budget ($B_Q$) | Random (10 Seeds) AUROC [95% CI] | Fixed Top-$k$ (JS) AUROC [95% CI] | Privacy-Blind Active ($G_t$) AUROC [95% CI] | Random Brier | Fixed Brier | Active Brier |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`5`** | `0.7178` `[0.625, 0.806]` | `1.0000` `[1.000, 1.000]` | `1.0000` `[1.000, 1.000]` | `0.2121` | `0.0705` | `0.1234` |
| **`10`** | `0.8122` `[0.734, 0.877]` | `1.0000` `[1.000, 1.000]` | `1.0000` `[1.000, 1.000]` | `0.1929` | `0.0222` | `0.1007` |
| **`20`** | `0.8373` `[0.754, 0.905]` | `1.0000` `[1.000, 1.000]` | `1.0000` `[1.000, 1.000]` | `0.1650` | `0.0296` | `0.1840` |
| **`40`** | `0.8628` `[0.794, 0.922]` | `1.0000` `[1.000, 1.000]` | `1.0000` `[1.000, 1.000]` | `0.1396` | `0.0102` | `3.66e-6` |
| **`80`** | `0.9158` `[0.859, 0.963]` | `1.0000` `[1.000, 1.000]` | `1.0000` `[1.000, 1.000]` | `0.0740` | `4.31e-4` | `2.26e-12` |
| **`160`** | `0.9245` `[0.879, 0.964]` | `1.0000` `[1.000, 1.000]` | `1.0000` `[1.000, 1.000]` | `0.0869` | `8.47e-10` | `4.23e-20` |

### C. Evidence Margin, Response-Adaptivity & Experimental `ga100` Findings

1. **Query Efficiency (RQ1 Validation):**
   - Random querying requires `80–160` queries to reach `0.916–0.925` AUROC.
   - Surrogate-guided policies (`Fixed Top-k` and `Privacy-Blind Active`) achieve **`1.0000` AUROC within just `5` queries**.
2. **Evidence Separation Strength (`mean LLR(H_C) - mean LLR(H_N)`):**
   - At $B_Q = 5$: Active margin = `3.90` (`+2.434` vs `-1.469`) vs Fixed margin = `2.63` (`+0.633` vs `-1.994`).
   - At $B_Q = 20$: Active margin = `11.79` (`+7.269` vs `-4.518`) vs Fixed margin = `5.32` (`+2.529` vs `-2.787`) — **2.22$\times$ wider separation**.
   - At $B_Q = 40$: Active margin = `21.17` vs Fixed margin = `10.59`.
3. **Empirical Verification of Response-Adaptivity:**
   - `Fixed Top-k` uses `1` identical query trajectory across all `24` targets.
   - `Privacy-Blind Active` branches into **`9` unique trajectories at $B_Q=5$, `20` at $B_Q=20$, and `24 / 24` unique trajectories by $B_Q=80$**.
4. **Scientific Caveat Discovered (Transient Miscalibration Under Adaptive Sampling on `validation_seed203`):**
   - While `Privacy-Blind Active` achieves `1.0000` AUROC at $B_Q=20$, its Brier score temporarily spikes (`0.1840`) because `validation_seed203`'s $H_N$ models (`original`, `ga25/50/75`) transiently drift to `LLR = +3.04 to +3.51` at $B_Q=20$ before collapsing strongly negative (`-6.49 to -6.80`) by $B_Q=40$.
   - *Implication for Milestone 11:* Validates the handoff caution (Section 8.5) that raw likelihood ratios under adaptive querying must NOT use uncalibrated fixed-sample stopping thresholds; validation-calibrated stopping boundaries (or uncertainty-damped likelihood temperatures) are essential to prevent premature false-PASS on out-of-distribution seeds.
5. **Experimental `ga100` Verdict:**
   - Under `Privacy-Blind Active`, mean LLR for `EXPERIMENTAL ga100` stays strongly negative (`-1.07` at $B_Q=5$, `-15.30` at $B_Q=40$, `-30.97` at $B_Q=160$), confirming it aligns with non-compliance ($H_N$) rather than `retrain` ($H_C$).
