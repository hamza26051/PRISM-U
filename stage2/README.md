# PRISM-U Stage 2 — Complete Reference/Surrogate Bank and Query-Response Infrastructure

This directory implements the complete Stage 2 defined for PRISM-U. It consumes the completed CIFAR-10 / ResNet-18 Stage 1 experiment and produces the reference behaviour required by the later auditor stages.

## Scientific boundary

Stage 2 does **not** implement the final privacy-aware acquisition function, privacy budget, sequential LLR stopping, or PASS / FAIL / INCONCLUSIVE decision rule. Those begin after this stage.

Stage 2 is complete when it has built:

1. independent compliant and non-compliant reference model populations;
2. a separate validation population;
3. the held-out seed-42 target family;
4. a frozen 5,000-image auxiliary query pool;
5. black-box probability responses for every model/query pair;
6. surrogate and validation separability diagnostics; and
7. a passing `validation_report.json` with `stage2_complete: true`.

## Model population

The default configuration uses:

- **12 surrogate seeds:** 101–112
- **4 validation seeds:** 201–204
- **1 final held-out family:** seed 42 from Stage 1

Each family contains six conditions:

| Condition | Hypothesis label | Meaning |
| --- | --- | --- |
| `retrain` | `HC` | honest retraining without the forget set |
| `original` | `HN` | no unlearning / 0% |
| `ga25` | `HN` | 25% of the gradient-ascent trajectory |
| `ga50` | `HN` | 50% |
| `ga75` | `HN` | 75% |
| `ga100` | `EXPERIMENTAL` | complete GA procedure; deliberately **not** assumed compliant |

This gives:

- 72 surrogate conditions = 12 HC + 48 HN + 12 experimental;
- 24 validation conditions = 4 HC + 16 HN + 4 experimental;
- 6 final conditions for seed 42;
- **102 total model conditions**.

Only the surrogate HC/HN population defines Stage 2 query separability. The validation bank measures generalisation. Seed 42 is not used to rank queries or tune anything.

## Connection to Stage 1

The default config points to the Stage 1 files already present in this repository:

```text
PRISM-U-stage1/PRISM-U-stage1/splits/cifar10_seed42.json
prism_u_stage1_results/results/original_seed42.pt
prism_u_stage1_results/results/retrain_seed42.pt
```

The `.pt` files are Git LFS objects. After cloning, run:

```bash
git lfs install
git lfs pull
```

If those commands are skipped, the `.pt` files may be tiny pointer files. `verify_inputs.py` detects this and stops before expensive training begins.

Stage 2 reuses the exact Stage 1 architecture and training recipe: CIFAR-10 ResNet-18, 40 epochs, SGD, LR 0.1, momentum 0.9, weight decay 5e-4, milestones 20/30, gamma 0.1.

## Repository layout

```text
stage2/
├── configs/
│   └── stage2.yaml
├── prismu_stage2/
│   ├── __init__.py
│   ├── core.py
│   └── layout.py
├── scripts/
│   ├── verify_inputs.py
│   ├── build_bank.py
│   ├── cache_responses.py
│   ├── diagnostics.py
│   └── validate_stage2.py
├── tests/
│   └── test_static.py
├── PRISM_U_Stage2_Complete_Colab.ipynb
├── requirements.txt
├── run_stage2.py
└── README.md
```

Generated outputs are written to `stage2_artifacts/` by default and are intentionally gitignored because the checkpoint bank is several GB.

## Generated artifact layout

```text
stage2_artifacts/
├── checkpoints/
│   ├── surrogate/seed101/...seed112/
│   ├── validation/seed201/...seed204/
│   └── final/seed42/ga/
├── logs/
├── manifests/
│   └── model_manifest.csv
├── query_pool/
│   ├── query_indices.npy
│   └── query_manifest.csv
├── responses/
│   ├── surrogate_responses.npz
│   ├── validation_responses.npz
│   └── final_responses.npz
├── diagnostics/
│   ├── query_separability.csv
│   ├── validation_query_separability.csv
│   ├── surrogate_diagnostics.json
│   ├── validation_diagnostics.json
│   ├── separability_histogram.png
│   └── top100_queries.png
├── reports/
│   └── split_validation.json
└── validation_report.json
```

## First-time setup

Run from the repository root:

```bash
git lfs install
git lfs pull
python -m pip install -r stage2/requirements.txt
python stage2/scripts/verify_inputs.py
python -m pytest stage2/tests -q
```

Do not begin the full bank until `verify_inputs.py` reports `PASS`.

## Recommended GPU workflow

Training 16 independent original/retrain pairs is intentionally substantial. The pipeline is resumable and should normally be run seed-by-seed.

### 1. Build the 12 surrogate families

```bash
python stage2/run_stage2.py --phase bank --role surrogate --seed 101
python stage2/run_stage2.py --phase bank --role surrogate --seed 102
python stage2/run_stage2.py --phase bank --role surrogate --seed 103
python stage2/run_stage2.py --phase bank --role surrogate --seed 104
python stage2/run_stage2.py --phase bank --role surrogate --seed 105
python stage2/run_stage2.py --phase bank --role surrogate --seed 106
python stage2/run_stage2.py --phase bank --role surrogate --seed 107
python stage2/run_stage2.py --phase bank --role surrogate --seed 108
python stage2/run_stage2.py --phase bank --role surrogate --seed 109
python stage2/run_stage2.py --phase bank --role surrogate --seed 110
python stage2/run_stage2.py --phase bank --role surrogate --seed 111
python stage2/run_stage2.py --phase bank --role surrogate --seed 112
```

Each command trains the independent original and honest-retrain models and then generates GA-25/50/75/100 from that seed's original checkpoint. An epoch-level `.resume.pt` file is maintained until training finishes.

### 2. Build the 4 validation families

```bash
python stage2/run_stage2.py --phase bank --role validation --seed 201
python stage2/run_stage2.py --phase bank --role validation --seed 202
python stage2/run_stage2.py --phase bank --role validation --seed 203
python stage2/run_stage2.py --phase bank --role validation --seed 204
```

### 3. Build the final seed-42 GA family

The seed-42 original and honest retrain already come from Stage 1. Stage 2 only creates the partial GA trajectory:

```bash
python stage2/run_stage2.py --phase bank --role final --seed 42
```

### 4. Freeze the Stage 1 auxiliary set and cache every black-box response

```bash
python stage2/run_stage2.py --phase responses
```

This uses the exact 5,000 Stage 1 `auxiliary` indices. The query pool is checked to be disjoint from target-train and validation data.

Every model is queried only through softmax probability vectors. The resulting tensors have shapes:

```text
surrogate: [72, 5000, 10]
validation: [24, 5000, 10]
final:      [ 6, 5000, 10]
```

### 5. Compute Stage 2 diagnostics

```bash
python stage2/run_stage2.py --phase diagnostics
```

For each query, the surrogate bank records:

- Jensen-Shannon divergence between HC/HN mean responses;
- L1 separation;
- Fisher-style between/within separation;
- top-1 HC/HN disagreement;
- within-HC and within-HN variance.

The surrogate bank defines the query ranking. The validation bank then measures the same queries without redefining that ranking. GA-100 is excluded from HC/HN reference estimates.

### 6. Run the hard Stage 2 completion gate

```bash
python stage2/run_stage2.py --phase validate
```

Success ends with:

```text
Stage 2 complete: True
```

and creates:

```text
stage2_artifacts/validation_report.json
```

with:

```json
{
  "stage": 2,
  "stage2_complete": true
}
```

At that point Stage 2 is complete and its cached reference distributions are ready for the next PRISM-U auditing stage.

## One-command mode

If running on a machine where you can leave the GPU job running until all families finish:

```bash
python stage2/run_stage2.py --phase all
```

The command is resumable. Existing completed checkpoints/logs/caches are skipped.

## Google Colab

Use `PRISM_U_Stage2_Complete_Colab.ipynb`. The recommended setup is to mount Google Drive and change `paths.artifact_root` in `stage2/configs/stage2.yaml` to a Drive directory, for example:

```yaml
artifact_root: /content/drive/MyDrive/PRISM-U-stage2-artifacts
```

This is strongly recommended: Colab's `/content` is temporary, while the full model bank takes many separate training runs.

The notebook intentionally does not store a GitHub token. Authenticate/clone the private repository using your preferred GitHub method, then execute the notebook cells from top to bottom.

## Changing artifact locations without editing code

All paths live in `stage2/configs/stage2.yaml`. In particular, you can point the Stage 1 checkpoints to an extracted local/Drive folder instead of the repository LFS files:

```yaml
paths:
  stage1_split: /content/drive/MyDrive/PRISM-U-stage1/cifar10_seed42.json
  stage1_original: /content/drive/MyDrive/PRISM-U-stage1/original_seed42.pt
  stage1_retrain: /content/drive/MyDrive/PRISM-U-stage1/retrain_seed42.pt
  artifact_root: /content/drive/MyDrive/PRISM-U-stage2-artifacts
```

## Why GA-100 is not HC

`ga100` means the configured gradient-ascent procedure completed all 100 steps. That does not logically prove that the resulting model is equivalent to honest retraining. For that reason:

- honest retraining is the HC anchor;
- original / GA-25 / GA-50 / GA-75 are controlled HN references;
- GA-100 remains an experimental target until the later auditor evaluates it.

## What comes after Stage 2

The next stage can load `surrogate_responses.npz` and estimate query usefulness from the behavioural difference between `P(r | x, HC)` and `P(r | x, HN)`. Privacy risk, budget-aware active acquisition, sequential evidence, stopping thresholds, and PASS/FAIL/INCONCLUSIVE decisions should be implemented there rather than leaked into Stage 2.
