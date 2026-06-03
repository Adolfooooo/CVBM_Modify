from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cvbm_15_1.t2.a.modules import CVBMArgumentWithCrossSKC3DProto


PANEL_ORDER = (
    ("fg_before_skc", "FG before SKC"),
    ("fg_after_skc", "FG after SKC"),
    ("bg_before_skc", "BG before SKC"),
    ("bg_after_skc", "BG after SKC"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="t-SNE visualization of LA bottleneck features before and after SKC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root-path", type=Path, default=Path("~/Datasets/LA/UA_MT").expanduser())
    parser.add_argument("--split-list", type=str, default="test.list")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "./results/cvbm_15_1_t2_a/1/CVBM_LA_CrossSKC_16_labeled/"
            "self_train/CVBM_Argument_best_model.pth"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./experiments/cvbm_15_1/t2/a/t_sne/outputs/la_cross_skc_16"),
    )
    parser.add_argument("--patch-size", type=int, nargs=3, default=(112, 112, 80))
    parser.add_argument("--tokens-per-class", type=int, default=1000)
    parser.add_argument("--patches-per-case", type=int, default=8)
    parser.add_argument(
        "--label-sampling",
        choices=("nearest", "pure"),
        default="pure",
        help=(
            "nearest uses nearest-neighbor GT downsampling; pure keeps only bottleneck "
            "tokens with high foreground/background GT purity."
        ),
    )
    parser.add_argument("--fg-min-ratio", type=float, default=0.70)
    parser.add_argument("--bg-max-ratio", type=float, default=0.05)
    parser.add_argument(
        "--fg-patch-ratio",
        type=float,
        default=0.75,
        help="Fraction of sampled patches whose origin is drawn around a foreground voxel.",
    )
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all cases in split list.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-filters", type=int, default=16)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--feature-normalization", choices=("standard", "standard_l2"), default="standard_l2")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-early-exaggeration", type=float, default=24.0)
    parser.add_argument("--tsne-iterations", type=int, default=1000)
    parser.add_argument("--tsne-metric", type=str, default="euclidean")
    parser.add_argument(
        "--paper-after-feature",
        choices=("fg_after_skc", "bg_after_skc"),
        default="bg_after_skc",
        help="After-SKC feature used by the paper-style 1x2 figure.",
    )
    parser.add_argument("--strict-load", action="store_true")
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def load_case_names(root_path: Path, split_list: str, max_cases: int) -> List[str]:
    list_path = root_path / split_list
    if not list_path.exists():
        raise FileNotFoundError(f"split list not found: {list_path}")
    cases = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    if max_cases > 0:
        cases = cases[:max_cases]
    if not cases:
        raise RuntimeError(f"no cases found in {list_path}")
    return cases


def case_h5_path(root_path: Path, case_name: str) -> Path:
    return root_path / "2018LA_Seg_Training Set" / case_name / "mri_norm2.h5"


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = CVBMArgumentWithCrossSKC3DProto(
        n_channels=1,
        n_classes=2,
        n_filters=args.n_filters,
        normalization="instancenorm",
        has_dropout=True,
    ).to(device)

    checkpoint = normalize_path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    state = torch.load(str(checkpoint), map_location=device)
    if isinstance(state, dict) and "net" in state and isinstance(state["net"], dict):
        state = state["net"]
    if not isinstance(state, dict):
        raise RuntimeError(f"unsupported checkpoint format: {checkpoint}")

    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}

    incompatible = model.load_state_dict(state, strict=args.strict_load)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "checkpoint loaded with non-strict key differences: "
            f"missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )

    model.eval()
    return model


def pad_to_patch(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    pads = []
    for size, target in zip(image.shape, patch_size):
        total = max(target - size, 0)
        before = total // 2
        after = total - before
        pads.append((before, after))
    if any(before or after for before, after in pads):
        image = np.pad(image, pads, mode="constant", constant_values=0)
        label = np.pad(label, pads, mode="constant", constant_values=0)
    return image, label


def clamp_origin(origin: np.ndarray, shape: Tuple[int, int, int], patch_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
    max_origin = np.maximum(np.asarray(shape) - np.asarray(patch_size), 0)
    origin = np.minimum(np.maximum(origin, 0), max_origin)
    return tuple(int(value) for value in origin)


def sample_patch_origins(
    label: np.ndarray,
    patch_size: Tuple[int, int, int],
    patches_per_case: int,
    fg_patch_ratio: float,
    rng: np.random.Generator,
) -> List[Tuple[int, int, int]]:
    shape = label.shape
    origins: List[Tuple[int, int, int]] = [
        clamp_origin((np.asarray(shape) - np.asarray(patch_size)) // 2, shape, patch_size)
    ]
    fg_count = int(round(max(patches_per_case - 1, 0) * fg_patch_ratio))
    random_count = max(patches_per_case - 1 - fg_count, 0)

    fg_voxels = np.argwhere(label > 0)
    if fg_voxels.size > 0:
        chosen = fg_voxels[rng.integers(0, len(fg_voxels), size=fg_count)]
        for voxel in chosen:
            offset = np.asarray([rng.integers(0, size) for size in patch_size])
            origins.append(clamp_origin(voxel - offset, shape, patch_size))
    else:
        random_count += fg_count

    max_origin = np.maximum(np.asarray(shape) - np.asarray(patch_size), 0)
    for _ in range(random_count):
        origin = np.asarray([rng.integers(0, high + 1) if high > 0 else 0 for high in max_origin])
        origins.append(tuple(int(value) for value in origin))

    return origins[:patches_per_case]


def crop_patch(array: np.ndarray, origin: Tuple[int, int, int], patch_size: Tuple[int, int, int]) -> np.ndarray:
    x, y, z = origin
    sx, sy, sz = patch_size
    return array[x : x + sx, y : y + sy, z : z + sz]


@torch.no_grad()
def extract_bottleneck_features(
    model: torch.nn.Module,
    image_patch: np.ndarray,
    label_patch: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    image_tensor = torch.from_numpy(image_patch[None, None].astype(np.float32)).to(device)
    label_tensor = torch.from_numpy(label_patch[None, None].astype(np.float32)).to(device)

    fg_feats = list(model.encoder(image_tensor))
    bg_feats = list(model.encoder(image_tensor))
    fg_before = fg_feats[-1]
    bg_before = bg_feats[-1]
    fg_after, bg_after = model.skc(fg_before, bg_before)

    if args.label_sampling == "pure":
        label_ratio = F.adaptive_avg_pool3d(label_tensor, output_size=fg_before.shape[-3:])
        token_ratio = label_ratio.reshape(-1).cpu().numpy()
        token_labels = np.full(token_ratio.shape, -1, dtype=np.int64)
        token_labels[token_ratio <= args.bg_max_ratio] = 0
        token_labels[token_ratio >= args.fg_min_ratio] = 1
    else:
        label_bn = F.interpolate(label_tensor, size=fg_before.shape[-3:], mode="nearest")
        token_labels = (label_bn.reshape(-1) > 0).long().cpu().numpy()

    features = {
        "fg_before_skc": flatten_tokens(fg_before),
        "fg_after_skc": flatten_tokens(fg_after),
        "bg_before_skc": flatten_tokens(bg_before),
        "bg_after_skc": flatten_tokens(bg_after),
    }
    return features, token_labels


def flatten_tokens(feature: torch.Tensor) -> np.ndarray:
    return feature[0].permute(1, 2, 3, 0).reshape(-1, feature.shape[1]).detach().cpu().numpy().astype(np.float32)


class BalancedReservoir:
    def __init__(self, feature_names: Iterable[str], tokens_per_class: int, rng: np.random.Generator) -> None:
        self.feature_names = tuple(feature_names)
        self.tokens_per_class = tokens_per_class
        self.rng = rng
        self.storage: Dict[int, Dict[str, np.ndarray]] = {0: {}, 1: {}}
        self.filled = {0: 0, 1: 0}
        self.seen = {0: 0, 1: 0}

    def update(self, features: Dict[str, np.ndarray], labels: np.ndarray) -> None:
        for class_id in (0, 1):
            token_indices = np.flatnonzero(labels == class_id)
            for token_idx in token_indices:
                self._update_one(class_id, token_idx, features)

    def _update_one(self, class_id: int, token_idx: int, features: Dict[str, np.ndarray]) -> None:
        self.seen[class_id] += 1
        if self.filled[class_id] < self.tokens_per_class:
            slot = self.filled[class_id]
            self.filled[class_id] += 1
            if not self.storage[class_id]:
                for name in self.feature_names:
                    dim = features[name].shape[1]
                    self.storage[class_id][name] = np.empty((self.tokens_per_class, dim), dtype=np.float32)
        else:
            replacement = int(self.rng.integers(0, self.seen[class_id]))
            if replacement >= self.tokens_per_class:
                return
            slot = replacement

        for name in self.feature_names:
            self.storage[class_id][name][slot] = features[name][token_idx]

    def is_full(self) -> bool:
        return all(self.filled[class_id] >= self.tokens_per_class for class_id in (0, 1))

    def as_panel_arrays(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        if not self.is_full():
            raise RuntimeError(
                "not enough tokens collected: "
                f"background={self.filled[0]}/{self.tokens_per_class}, "
                f"foreground={self.filled[1]}/{self.tokens_per_class}; "
                f"seen background={self.seen[0]}, foreground={self.seen[1]}"
            )
        labels = np.concatenate(
            [
                np.zeros(self.tokens_per_class, dtype=np.int64),
                np.ones(self.tokens_per_class, dtype=np.int64),
            ]
        )
        panel_arrays = {}
        for name in self.feature_names:
            panel_arrays[name] = np.concatenate(
                [self.storage[0][name], self.storage[1][name]],
                axis=0,
            )
        return panel_arrays, labels


def collect_features(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> Tuple[Dict[str, np.ndarray], np.ndarray, dict]:
    rng = np.random.default_rng(args.seed)
    root_path = normalize_path(args.root_path)
    cases = load_case_names(root_path, args.split_list, args.max_cases)
    patch_size = tuple(args.patch_size)
    reservoir = BalancedReservoir((name for name, _ in PANEL_ORDER), args.tokens_per_class, rng)

    processed_cases = 0
    processed_patches = 0
    for case_name in tqdm(cases, desc="Collecting bottleneck tokens"):
        path = case_h5_path(root_path, case_name)
        if not path.exists():
            raise FileNotFoundError(f"case h5 not found: {path}")
        with h5py.File(path, "r") as h5f:
            image = h5f["image"][:]
            label = h5f["label"][:]

        image, label = pad_to_patch(image, label, patch_size)
        origins = sample_patch_origins(
            label=label,
            patch_size=patch_size,
            patches_per_case=args.patches_per_case,
            fg_patch_ratio=args.fg_patch_ratio,
            rng=rng,
        )

        for origin in origins:
            image_patch = crop_patch(image, origin, patch_size)
            label_patch = crop_patch(label, origin, patch_size)
            features, token_labels = extract_bottleneck_features(model, image_patch, label_patch, device, args)
            reservoir.update(features, token_labels)
            processed_patches += 1

        processed_cases += 1

    panel_arrays, token_labels = reservoir.as_panel_arrays()
    stats = {
        "processed_cases": processed_cases,
        "processed_patches": processed_patches,
        "tokens_per_class": args.tokens_per_class,
        "seen_background_tokens": int(reservoir.seen[0]),
        "seen_foreground_tokens": int(reservoir.seen[1]),
        "filled_background_tokens": int(reservoir.filled[0]),
        "filled_foreground_tokens": int(reservoir.filled[1]),
    }
    return panel_arrays, token_labels, stats


def import_sklearn_tools():
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler, normalize
    except ImportError as exc:
        raise RuntimeError(
            "This experiment requires scikit-learn for PCA, t-SNE, and silhouette metrics. "
            "Install it in the active environment, for example: pip install scikit-learn"
        ) from exc
    return PCA, TSNE, silhouette_score, StandardScaler, normalize


def preprocess_for_tsne(features: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, dict]:
    PCA, _, _, StandardScaler, normalize = import_sklearn_tools()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    if args.feature_normalization == "standard_l2":
        scaled = normalize(scaled, norm="l2", axis=1)

    pca_components = min(args.pca_components, scaled.shape[0] - 1, scaled.shape[1])
    if pca_components > 0 and pca_components < scaled.shape[1]:
        pca = PCA(n_components=pca_components, random_state=args.seed)
        tsne_input = pca.fit_transform(scaled)
        explained_variance = float(np.sum(pca.explained_variance_ratio_))
    else:
        tsne_input = scaled
        explained_variance = None

    metadata = {
        "pca_components": int(pca_components),
        "pca_explained_variance": explained_variance,
    }
    return scaled, tsne_input, metadata


def run_tsne(tsne_input: np.ndarray, args: argparse.Namespace, seed_offset: int = 0) -> np.ndarray:
    _, TSNE, _, _, _ = import_sklearn_tools()
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": args.tsne_perplexity,
        "learning_rate": "auto",
        "init": "pca",
        "early_exaggeration": args.tsne_early_exaggeration,
        "metric": args.tsne_metric,
        "random_state": args.seed + seed_offset,
    }
    if "max_iter" in inspect.signature(TSNE).parameters:
        tsne_kwargs["max_iter"] = args.tsne_iterations
    else:
        tsne_kwargs["n_iter"] = args.tsne_iterations

    return TSNE(**tsne_kwargs).fit_transform(tsne_input)


def make_tsne(panel_arrays: Dict[str, np.ndarray], labels: np.ndarray, args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], dict]:
    _, _, silhouette_score, _, _ = import_sklearn_tools()

    ordered_features = [panel_arrays[name] for name, _ in PANEL_ORDER]
    all_features = np.concatenate(ordered_features, axis=0)

    scaled, tsne_input, pca_metadata = preprocess_for_tsne(all_features, args)
    embedding_all = run_tsne(tsne_input, args)
    split_size = len(labels)
    embeddings = {
        name: embedding_all[idx * split_size : (idx + 1) * split_size]
        for idx, (name, _) in enumerate(PANEL_ORDER)
    }

    metrics = {}
    for idx, (name, _) in enumerate(PANEL_ORDER):
        start = idx * split_size
        end = (idx + 1) * split_size
        scaled_panel = scaled[start:end]
        bg_center = scaled_panel[labels == 0].mean(axis=0)
        fg_center = scaled_panel[labels == 1].mean(axis=0)
        metrics[name] = {
            "feature_silhouette": float(silhouette_score(scaled_panel, labels)),
            "feature_centroid_distance": float(np.linalg.norm(fg_center - bg_center)),
        }

    metadata = {
        **pca_metadata,
        "tsne_perplexity": args.tsne_perplexity,
        "tsne_early_exaggeration": args.tsne_early_exaggeration,
        "tsne_iterations": args.tsne_iterations,
        "tsne_metric": args.tsne_metric,
        "feature_normalization": args.feature_normalization,
        "metrics": metrics,
    }
    return embeddings, metadata


def make_paper_tsne(panel_arrays: Dict[str, np.ndarray], labels: np.ndarray, args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], dict]:
    _, _, silhouette_score, _, _ = import_sklearn_tools()
    feature_names = ("fg_before_skc", args.paper_after_feature)
    titles = ("Before SKC", "After SKC")
    embeddings = {}
    metrics = {}
    pca_metadata = {}

    for idx, (name, title) in enumerate(zip(feature_names, titles)):
        scaled, tsne_input, metadata = preprocess_for_tsne(panel_arrays[name], args)
        embeddings[title] = run_tsne(tsne_input, args, seed_offset=idx + 101)
        bg_center = scaled[labels == 0].mean(axis=0)
        fg_center = scaled[labels == 1].mean(axis=0)
        metrics[title] = {
            "source_feature": name,
            "feature_silhouette": float(silhouette_score(scaled, labels)),
            "feature_centroid_distance": float(np.linalg.norm(fg_center - bg_center)),
        }
        pca_metadata[title] = metadata

    return embeddings, {
        "before_feature": "fg_before_skc",
        "after_feature": args.paper_after_feature,
        "pca": pca_metadata,
        "metrics": metrics,
    }


def plot_embeddings(embeddings: Dict[str, np.ndarray], labels: np.ndarray, metrics: dict, output_dir: Path) -> None:
    colors = {0: "#4c78a8", 1: "#e45756"}
    class_names = {0: "background", 1: "foreground"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=160)
    axes = axes.ravel()

    for axis, (name, title) in zip(axes, PANEL_ORDER):
        embedding = embeddings[name]
        for class_id in (0, 1):
            mask = labels == class_id
            axis.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=7,
                c=colors[class_id],
                label=class_names[class_id],
                alpha=0.72,
                linewidths=0,
            )
        metric = metrics[name]
        axis.set_title(
            f"{title}\n"
            f"silhouette={metric['feature_silhouette']:.3f}, "
            f"centroid={metric['feature_centroid_distance']:.2f}",
            fontsize=11,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("t-SNE Visualization of Bottleneck Features Before and After SKC", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output_dir / "tsne_skc_2x2.png", bbox_inches="tight")
    fig.savefig(output_dir / "tsne_skc_2x2.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_paper_embeddings(embeddings: Dict[str, np.ndarray], labels: np.ndarray, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.linewidth": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )
    colors = {0: "#1f4e79", 1: "#f0e442"}
    edgecolors = {0: "#1f4e79", 1: "#1f4e79"}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), dpi=180)

    for axis, key, panel_label in zip(axes, ("Before SKC", "After SKC"), ("(a)", "(b)")):
        embedding = embeddings[key]
        for class_id in (0, 1):
            mask = labels == class_id
            axis.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=7,
                c=colors[class_id],
                edgecolors=edgecolors[class_id],
                linewidths=0.25,
                alpha=0.95,
            )
        axis.set_xlabel(panel_label, fontsize=12)
        axis.set_ylabel("")
        axis.tick_params(labelsize=10, width=1.0)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        axis.set_box_aspect(0.85)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(output_dir / "tsne_skc_paper_1x2.png", bbox_inches="tight")
    fig.savefig(output_dir / "tsne_skc_paper_1x2.pdf", bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    output_dir: Path,
    panel_arrays: Dict[str, np.ndarray],
    labels: np.ndarray,
    embeddings: Dict[str, np.ndarray],
    args: argparse.Namespace,
    collection_stats: dict,
    tsne_metadata: dict,
    paper_embeddings: Dict[str, np.ndarray] | None = None,
    paper_metadata: dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"features_seed{args.seed}.npz",
        labels=labels,
        **{f"features_{name}": values for name, values in panel_arrays.items()},
        **{f"embedding_{name}": values for name, values in embeddings.items()},
        **({f"paper_embedding_{name.replace(' ', '_').lower()}": values for name, values in paper_embeddings.items()} if paper_embeddings else {}),
    )

    config = vars(args).copy()
    config["root_path"] = str(normalize_path(args.root_path))
    config["checkpoint"] = str(normalize_path(args.checkpoint))
    config["output_dir"] = str(output_dir.resolve())
    config["patch_size"] = list(args.patch_size)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    summary = {
        "collection": collection_stats,
        "tsne": tsne_metadata,
    }
    if paper_metadata is not None:
        summary["paper_tsne"] = paper_metadata
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.root_path = normalize_path(args.root_path)
    args.checkpoint = normalize_path(args.checkpoint)
    args.output_dir = normalize_path(args.output_dir)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu to run on CPU.")
    device = torch.device(args.device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args, device)
    panel_arrays, labels, collection_stats = collect_features(args, model, device)
    embeddings, tsne_metadata = make_tsne(panel_arrays, labels, args)
    paper_embeddings, paper_metadata = make_paper_tsne(panel_arrays, labels, args)
    plot_embeddings(embeddings, labels, tsne_metadata["metrics"], args.output_dir)
    plot_paper_embeddings(paper_embeddings, labels, args.output_dir)
    save_outputs(
        args.output_dir,
        panel_arrays,
        labels,
        embeddings,
        args,
        collection_stats,
        tsne_metadata,
        paper_embeddings,
        paper_metadata,
    )

    print(f"saved t-SNE figure to: {args.output_dir / 'tsne_skc_2x2.png'}")
    print(f"saved paper-style figure to: {args.output_dir / 'tsne_skc_paper_1x2.png'}")
    print(f"saved metrics to: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
