# PRISM-U — Privacy-Budgeted Active Auditing for Machine Unlearning

PRISM-U is a research framework for **black-box, privacy-aware, active auditing of machine unlearning**.

Given a model that has undergone an unlearning procedure to remove a designated forget set $D_u$, an external auditor queries the model through a black-box prediction interface. The auditor sequentially chooses queries to distinguish compliant from non-compliant models, subject to cumulative query limits ($B_Q$) and empirical retained-membership privacy constraints ($B_P$).

## Repository Structure

```text
├── configs/               # YAML experiment configurations
│   ├── cifar10_stage1.yaml
│   └── stage2.yaml
├── data/                  # Data loading, augmentations, and deterministic splits
│   ├── datasets.py
│   └── splits.py
├── models/                # Model architectures and training routines
│   ├── resnet.py
│   └── train.py
├── unlearning/            # Machine unlearning algorithms
│   └── gradient_ascent.py
├── surrogates/            # Surrogate model banks and response caching
│   ├── bank.py
│   └── response_store.py
├── audit/                 # Black-box auditing engine, policies, and stopping rules
├── attacks/               # Retained-membership privacy evaluators (MIA)
├── evaluation/            # Metrics, calibration, confidence intervals
│   └── evaluate.py
├── experiments/           # Experiment drivers and smoke tests
│   └── smoke_test.py
├── scripts/               # CLI entry points
│   ├── run_stage1.py
│   ├── run_stage2.py
│   ├── build_bank.py
│   ├── cache_responses.py
│   ├── diagnostics.py
│   ├── validate_stage2.py
│   └── verify_inputs.py
├── tests/                 # Unit and static tests
│   └── test_static.py
├── splits/                # Saved dataset partition files
│   └── cifar10_seed42.json
├── checkpoints/           # Model checkpoints (.pt)
│   ├── original_seed42.pt
│   └── retrain_seed42.pt
├── results/               # Experiment logs and evaluation metrics
├── notebooks/             # Google Colab notebooks for remote GPU execution
└── docs/                  # Documentation and design decisions
    └── design_decisions.md
```

## Quick Start

### 1. Requirements
```bash
pip install -r requirements.txt
```

### 2. Verify Preflight Inputs
```bash
python scripts/verify_inputs.py --config configs/stage2.yaml
```

### 3. Run Static Tests
```bash
pytest tests/
```

### 4. Build Model Bank (or run targeted seed)
```bash
python run_stage2.py --phase bank --role final --seed 42
```

