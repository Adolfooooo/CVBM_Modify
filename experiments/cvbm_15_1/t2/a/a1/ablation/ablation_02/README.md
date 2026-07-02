## Ablation 03: Remove Contrastive Learning, Keep SKC

This ablation keeps the Cross-SKC semantic interaction backbone and removes the active contrastive learning branch used in the baseline self-training stage.

### Baseline Reference

- `experiments/cvbm_15_1/t2/a/pancreas_train.py`

### What Is Removed

- `BranchBatchPrototypeLoss`
- Prototype-related arguments and logging
- Prototype feature optimization during self-training

### What Is Kept

- Cross-SKC backbone: `CVBMArgumentWithCrossSKC3DProto`
- Pre-train stage
- Mean-teacher self-training pipeline
- Foreground/background mixed supervision losses

### Note

The legacy patch-level InfoNCE code in the baseline file is already commented out and is not an active training component. Therefore, this ablation removes the prototype-based contrastive branch, which is the effective contrastive module in the current baseline.
