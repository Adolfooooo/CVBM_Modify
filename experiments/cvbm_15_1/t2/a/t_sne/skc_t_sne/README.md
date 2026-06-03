# t-SNE Visualization of Bottleneck Features Before and After SKC

This experiment visualizes LA validation/test labeled samples with GT labels downsampled to the bottleneck feature size.

It extracts four bottleneck feature sets:

- `fg_before_skc`
- `fg_after_skc`
- `bg_before_skc`
- `bg_after_skc`

Foreground and background tokens are sampled with a balanced reservoir: 1000 foreground tokens and 1000 background tokens by default. The same token positions are used for all four feature sets, then a joint t-SNE is fit and plotted as a 2x2 figure.

## Run

From the repository root:

```bash
/home/xuminghao/anaconda3/envs/cvbm/bin/python -m experiments.cvbm_15_1.t2.a.t_sne.run_la_tsne_skc \
  --root-path ~/Datasets/LA/UA_MT \
  --checkpoint ./results/cvbm_15_1_t2_a/1/CVBM_LA_CrossSKC_16_labeled/self_train/CVBM_Argument_best_model.pth \
  --output-dir ./experiments/cvbm_15_1/t2/a/t_sne/outputs/la_cross_skc_16 \
  --tokens-per-class 1000 \
  --seed 1337
```

The script requires PyTorch, h5py, matplotlib, tqdm, and scikit-learn. If the `cvbm` environment lacks scikit-learn, install it before running:

```bash
/home/xuminghao/anaconda3/envs/cvbm/bin/python -m pip install scikit-learn
```

The default split is `test.list`, matching the existing LA validation logic in `utils/test_3d_patch.py`.

## Outputs

The script writes:

- `tsne_skc_2x2.png`
- `tsne_skc_2x2.pdf`
- `tsne_skc_paper_1x2.png`
- `tsne_skc_paper_1x2.pdf`
- `features_seed1337.npz`
- `metrics.json`
- `config.json`

`metrics.json` includes feature-space silhouette score and foreground/background centroid distance for each panel.

## Interpretation

The expected visual conclusion is that foreground and background token distributions become more separated after SKC, supporting that the semantic knowledge interaction module improves bottleneck feature discriminability.

For a paper-style before/after figure closer to a compact 1x2 visualization, use stricter GT-purity sampling:

```bash
/home/xuminghao/anaconda3/envs/cvbm/bin/python -m experiments.cvbm_15_1.t2.a.t_sne.run_la_tsne_skc \
  --root-path ~/Datasets/LA/UA_MT \
  --checkpoint ./results/cvbm_15_1_t2_a/1/CVBM_LA_CrossSKC_16_labeled/self_train/CVBM_Argument_best_model.pth \
  --output-dir ./experiments/cvbm_15_1/t2/a/t_sne/pic_paper \
  --tokens-per-class 1000 \
  --label-sampling pure \
  --fg-min-ratio 0.80 \
  --bg-max-ratio 0.03 \
  --feature-normalization standard_l2 \
  --tsne-perplexity 18 \
  --tsne-early-exaggeration 48 \
  --tsne-iterations 2000 \
  --paper-after-feature bg_after_skc \
  --seed 1337
```
