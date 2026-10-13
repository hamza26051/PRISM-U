# PRISM-U Phase 2 Code Package

This source package is designed to be uploaded to `PRISM_U_Phase2_Complete_Colab.ipynb`.

Do not add Stage 1 checkpoints to this ZIP. The Colab notebook asks for them separately and places them in `stage1_inputs/`.

Required Stage 1 inputs:
- `original_seed42.pt`
- `retrain_seed42.pt`
- `cifar10_seed42.json`
- optional `cifar10_stage1.yaml` for configuration cross-checking

The default Phase 2 configuration builds:
- surrogate seeds 101-112
- validation seeds 201-204
- held-out final seed 42
- conditions: retrain, original, GA25, GA50, GA75, GA100

It uses the Stage 1 auxiliary split as the fixed 5,000-query pool and saves all generated artifacts under Google Drive at `MyDrive/PRISM-U/phase2/` when run through the supplied notebook.

Generated model conditions: 102 total (100 new Phase 2 checkpoints plus the 2 Stage 1 seed-42 checkpoints reused as original/retrain).

`GA100` is experimental and is not labeled compliant. Honest retraining is the compliant reference.
