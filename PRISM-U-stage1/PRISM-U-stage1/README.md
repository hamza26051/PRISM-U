# PRISM-U — Stage 1 Controlled Benchmark

This repository implements the first PRISM-U milestone:

1. create a deterministic CIFAR-10 experimental split,
2. train an **original** CIFAR-10 ResNet-18 on the target training pool,
3. define forget and retained subsets,
4. train a fresh **honest retrained reference** only on the retained subset,
5. evaluate both models on retain / forget / auxiliary / validation / official test sets,
6. save checkpoints, metrics, and per-example probabilities for later auditing work.

## Why the split is structured this way

The 50,000 CIFAR-10 training examples are deterministically split into:

- 40,000 target-training examples
  - 4,000 forget examples (10%)
  - 36,000 retained examples
- 5,000 auxiliary examples for future surrogate/query-pool work
- 5,000 validation examples for calibration
- CIFAR-10's official 10,000 test examples stay untouched for evaluation

The percentages and sizes are configuration choices, not universal PRISM-U constants. They are stored in `configs/cifar10_stage1.yaml` and can be changed later.

## Quick start

### 1. Create an environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Optional: run a quick offline smoke test

```bash
python -m experiments.smoke_test
```

You should see `SMOKE TEST PASSED`.

### 4. Generate the deterministic split

```bash
python -m data.splits --config configs/cifar10_stage1.yaml
```

Expected sizes:

```text
target_train: 40000
forget: 4000
retain: 36000
auxiliary: 5000
validation: 5000
official_test: 10000
```

### 5. Run the complete Stage 1 pipeline

On a GPU machine / Google Colab:

```bash
python -m experiments.run_stage1 --config configs/cifar10_stage1.yaml
```

For a very short plumbing test first:

```bash
python -m experiments.run_stage1 --config configs/cifar10_stage1.yaml --epochs 1
```

The full default run uses 40 epochs for each model.

### 6. Or train each model separately

Original model:

```bash
python -m models.train_target --config configs/cifar10_stage1.yaml --mode original
```

Honest retrained reference:

```bash
python -m models.train_target --config configs/cifar10_stage1.yaml --mode retrain
```

### 7. Evaluate a checkpoint manually

```bash
python -m evaluation.evaluate --config configs/cifar10_stage1.yaml --checkpoint checkpoints/original_seed42.pt
```

```bash
python -m evaluation.evaluate --config configs/cifar10_stage1.yaml --checkpoint checkpoints/retrain_seed42.pt
```

## What gets created

```text
splits/
  cifar10_seed42.json

checkpoints/
  original_seed42.pt
  retrain_seed42.pt

results/
  train_original_seed42.json
  train_retrain_seed42.json
  eval_original_seed42.json
  eval_retrain_seed42.json
  probs_original_seed42.npz
  probs_retrain_seed42.npz
```

## What "original" and "retrain" mean

`original` is trained on all 40,000 target-training examples, including both retained and forget records.

`retrain` is initialized from scratch and trained only on the 36,000 retained records. It is the Stage-1 honest reference for successful removal.

## When Stage 1 is complete

Do not move on until all of these are true:

- the split file is reproducible,
- both checkpoints exist,
- both models evaluate successfully,
- results JSON files are saved,
- probabilities are saved for later auditing,
- another teammate can clone/copy the project and reproduce the same split from seed 42.

## Next PRISM-U milestone

After Stage 1, implement one existing unlearning method and create controlled incomplete-unlearning targets (for example early-stopped 25%, 50%, 75%, and full execution). Only after those targets exist should the active auditor be built.
