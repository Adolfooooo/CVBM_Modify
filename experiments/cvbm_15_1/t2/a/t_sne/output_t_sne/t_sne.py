from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cvbm_15_1.t2.a.modules import CVBMArgumentWithCrossSKC3DProto


DEFAULT_ROOT = Path("/home/xuminghao/Datasets/LA/UA_MT")
DEFAULT_CKPT = Path(
    "/home/xuminghao/Projects/CVBM/CVBM-ABD/results/cvbm_15_1_t2_a/1/"
    "CVBM_LA_CrossSKC_16_labeled/self_train/CVBM_Argument_best_model.pth"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    return cleaned


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = CVBMArgumentWithCrossSKC3DProto(
        n_channels=1,
        n_classes=args.num_classes,
        n_filters=args.n_filters,
        normalization="instancenorm",
        has_dropout=True,
    ).to(device)

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    if isinstance(checkpoint, dict) and "net" in checkpoint and isinstance(checkpoint["net"], dict):
        checkpoint = checkpoint["net"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {args.ckpt}")

    checkpoint = clean_state_dict(checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=args.strict_load)

    print("=" * 80)
    print("Loaded checkpoint:", args.ckpt)
    print("Missing keys:", len(missing_keys))
    print("Unexpected keys:", len(unexpected_keys))
    print("=" * 80)

    model.eval()
    return model


def load_case_names(root_path: Path, split: str, max_cases: int) -> List[str]:
    list_path = root_path / f"{split}.list"
    if not list_path.exists():
        raise FileNotFoundError(f"Cannot find split list: {list_path}")

    cases = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    if max_cases > 0:
        cases = cases[:max_cases]
    if not cases:
        raise RuntimeError(f"No cases found in {list_path}")
    return cases


def case_h5_path(root_path: Path, case_name: str) -> Path:
    return root_path / "2018LA_Seg_Training Set" / case_name / "mri_norm2.h5"


def pad_to_patch(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    pads = []
    for current, target in zip(image.shape, patch_size):
        total = max(target - current, 0)
        before = total // 2
        after = total - before
        pads.append((before, after))

    if any(before or after for before, after in pads):
        image = np.pad(image, pads, mode="constant", constant_values=0)
        label = np.pad(label, pads, mode="constant", constant_values=0)

    return image, label


def clamp_origin(
    origin: np.ndarray,
    shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
) -> Tuple[int, int, int]:
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
    origins = [
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


def resize_label_to_feature_size(label: torch.Tensor, feature_size: Tuple[int, int, int]) -> torch.Tensor:
    if label.ndim == 4:
        label = label.unsqueeze(1)
    elif label.ndim != 5:
        raise ValueError(f"Unsupported label shape: {tuple(label.shape)}")

    label = F.interpolate(label.float(), size=feature_size, mode="nearest")
    return label.squeeze(1).long()


def resize_prediction_to_feature_size(logits: torch.Tensor, feature_size: Tuple[int, int, int]) -> torch.Tensor:
    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
    pred = F.interpolate(pred.unsqueeze(1).float(), size=feature_size, mode="nearest")
    return pred.squeeze(1).long()


def compute_boundary_band_per_class(
    labels: torch.Tensor,
    num_classes: int,
    boundary_width: int,
    include_background: bool,
) -> Dict[int, torch.Tensor]:
    boundary_dict = {}
    radius = boundary_width
    kernel_size = 2 * radius + 1
    start_class = 0 if include_background else 1

    for cls_id in range(start_class, num_classes):
        cls_mask = (labels == cls_id).float().unsqueeze(1)
        if cls_mask.sum() == 0:
            continue

        dilated = F.max_pool3d(cls_mask, kernel_size=kernel_size, stride=1, padding=radius)
        eroded = 1.0 - F.max_pool3d(1.0 - cls_mask, kernel_size=kernel_size, stride=1, padding=radius)
        boundary_band = (dilated - eroded) > 0
        boundary_dict[cls_id] = boundary_band.squeeze(1) & (labels == cls_id)

    return boundary_dict


def select_feature_tensor(
    outputs: Tuple[torch.Tensor, ...],
    feature_source: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out_fg, fused_logits, out_bg, _, _, feat_fg, feat_bg = outputs

    if feature_source == "decoder_fg":
        features = feat_fg
    elif feature_source == "decoder_bg":
        features = feat_bg
    elif feature_source == "decoder_cat":
        features = torch.cat([feat_fg, feat_bg], dim=1)
    elif feature_source == "logits":
        features = fused_logits
    elif feature_source == "prob":
        features = torch.softmax(fused_logits, dim=1)
    elif feature_source == "fg_logits":
        features = out_fg
    elif feature_source == "bg_logits":
        features = out_bg
    else:
        raise ValueError(f"Unsupported feature_source: {feature_source}")

    return features, fused_logits


def sample_features_3d(
    features_3d: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[int, List[torch.Tensor]]:
    _, channels, h, w, d = features_3d.shape
    feature_size = (h, w, d)

    if args.label_source == "gt":
        used_labels = resize_label_to_feature_size(labels, feature_size)
    elif args.label_source == "pred":
        used_labels = resize_prediction_to_feature_size(logits, feature_size)
    else:
        raise ValueError("label_source must be 'gt' or 'pred'.")

    features_flat = features_3d.permute(0, 2, 3, 4, 1).contiguous().view(-1, channels)
    sampled = defaultdict(list)

    boundary_dict = None
    if args.sample_region in ("boundary", "non_boundary"):
        boundary_dict = compute_boundary_band_per_class(
            labels=used_labels,
            num_classes=args.num_classes,
            boundary_width=args.boundary_width,
            include_background=args.include_background,
        )

    start_class = 0 if args.include_background else 1
    for cls_id in range(start_class, args.num_classes):
        if cls_id == args.ignore_label:
            continue

        cls_mask = used_labels == cls_id
        if args.sample_region == "boundary":
            if boundary_dict is None or cls_id not in boundary_dict:
                continue
            selected_mask = cls_mask & boundary_dict[cls_id]
        elif args.sample_region == "non_boundary":
            if boundary_dict is None or cls_id not in boundary_dict:
                continue
            selected_mask = cls_mask & (~boundary_dict[cls_id])
        elif args.sample_region == "all":
            selected_mask = cls_mask
        else:
            raise ValueError("sample_region must be one of: all, boundary, non_boundary.")

        selected_indices = torch.where(selected_mask.contiguous().view(-1))[0]
        if selected_indices.numel() == 0:
            continue

        sample_num = min(args.samples_per_class_per_patch, selected_indices.numel())
        selected_order = torch.randperm(selected_indices.numel(), device=features_flat.device)[:sample_num]
        sampled_indices = selected_indices[selected_order]
        sampled[cls_id].append(features_flat[sampled_indices].detach().cpu())

    return sampled


def current_bank_count(class_feature_bank: Dict[int, List[torch.Tensor]], cls_id: int) -> int:
    return sum(item.shape[0] for item in class_feature_bank.get(cls_id, []))


def bank_is_full(class_feature_bank: Dict[int, List[torch.Tensor]], args: argparse.Namespace) -> bool:
    start_class = 0 if args.include_background else 1
    return all(current_bank_count(class_feature_bank, cls_id) >= args.samples_per_class for cls_id in range(start_class, args.num_classes))


@torch.no_grad()
def collect_features(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> Tuple[np.ndarray, np.ndarray, dict]:
    root_path = Path(args.root_path)
    cases = load_case_names(root_path, args.split, args.max_cases)
    rng = np.random.default_rng(args.seed)
    patch_size = tuple(args.patch_size)
    class_feature_bank = defaultdict(list)

    processed_cases = 0
    processed_patches = 0
    first_shape_printed = False

    for case_idx, case_name in enumerate(cases):
        path = case_h5_path(root_path, case_name)
        if not path.exists():
            raise FileNotFoundError(f"Cannot find LA h5 file: {path}")

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

            image_tensor = torch.from_numpy(image_patch[None, None].astype(np.float32)).to(device)
            label_tensor = torch.from_numpy(label_patch[None].astype(np.int64)).to(device)

            outputs = model(image_tensor, image_tensor)
            if not isinstance(outputs, (tuple, list)) or len(outputs) < 7:
                raise RuntimeError("Expected CVBMArgumentWithCrossSKC3DProto to return 7 outputs.")

            features_3d, logits = select_feature_tensor(tuple(outputs), args.feature_source)

            if not first_shape_printed:
                print("=" * 80)
                print("Logits shape:", tuple(logits.shape))
                print(f"Feature source: {args.feature_source}")
                print("Feature shape used for t-SNE:", tuple(features_3d.shape))
                print("=" * 80)
                first_shape_printed = True

            sampled = sample_features_3d(
                features_3d=features_3d,
                logits=logits,
                labels=label_tensor,
                args=args,
            )
            for cls_id, feat_list in sampled.items():
                class_feature_bank[cls_id].extend(feat_list)

            processed_patches += 1
            if bank_is_full(class_feature_bank, args):
                break

        processed_cases += 1
        counts = {
            cls_id: current_bank_count(class_feature_bank, cls_id)
            for cls_id in range(0 if args.include_background else 1, args.num_classes)
        }
        print(f"Processed case {case_idx + 1}/{len(cases)}: {case_name}, sampled counts: {counts}")

        if bank_is_full(class_feature_bank, args):
            break

    final_features = []
    final_labels = []
    stats = {
        "processed_cases": processed_cases,
        "processed_patches": processed_patches,
        "feature_source": args.feature_source,
        "label_source": args.label_source,
        "sample_region": args.sample_region,
    }

    print("=" * 80)
    print("Collected feature statistics:")
    for cls_id in sorted(class_feature_bank.keys()):
        cls_features = torch.cat(class_feature_bank[cls_id], dim=0)
        if cls_features.shape[0] > args.samples_per_class:
            selected = torch.randperm(cls_features.shape[0])[: args.samples_per_class]
            cls_features = cls_features[selected]

        final_features.append(cls_features.numpy())
        final_labels.append(np.full((cls_features.shape[0],), cls_id, dtype=np.int64))
        stats[f"class_{cls_id}_points"] = int(cls_features.shape[0])
        print(f"Class {cls_id}: {cls_features.shape[0]} points")
    print("=" * 80)

    if len(final_features) == 0:
        raise RuntimeError("No features were collected. Check labels, predictions, or class ids.")

    return np.concatenate(final_features, axis=0), np.concatenate(final_labels, axis=0), stats


def run_tsne(features: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    print("Running t-SNE...")
    print("Original feature shape:", features.shape)

    features = StandardScaler().fit_transform(features)
    n_samples = features.shape[0]
    if n_samples <= 3:
        raise RuntimeError(f"Too few samples for t-SNE: {n_samples}")

    perplexity = min(args.perplexity, max(2, (n_samples - 1) // 3))
    print("Number of samples:", n_samples)
    print("Perplexity:", perplexity)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=args.seed,
    )
    return tsne.fit_transform(features)


def get_class_name(cls_id: int) -> str:
    class_names = {
        0: "Background",
        1: "LA",
    }
    return class_names.get(int(cls_id), f"Class {cls_id}")


def get_paper_like_color(cls_id: int) -> str | None:
    color_map = {
        0: "#4c78a8",
        1: "#e45756",
    }
    return color_map.get(int(cls_id), None)


def plot_tsne_scatter(tsne_result: np.ndarray, labels: np.ndarray, args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir, exist_ok=True)
    plt.figure(figsize=(7, 6))

    for cls_id in np.unique(labels):
        mask = labels == cls_id
        color = get_paper_like_color(cls_id) if args.paper_colors else None
        plt.scatter(
            tsne_result[mask, 0],
            tsne_result[mask, 1],
            s=args.point_size,
            alpha=args.alpha,
            c=color,
            label=get_class_name(cls_id),
            linewidths=0,
        )

    plt.title(f"t-SNE of LA 3D Features ({args.feature_source})")
    plt.xticks([])
    plt.yticks([])
    plt.legend(markerscale=3, fontsize=9)
    plt.tight_layout()

    save_path = os.path.join(args.save_dir, "tsne_la3d_scatter.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved:", save_path)


def plot_tsne_density(tsne_result: np.ndarray, args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir, exist_ok=True)
    x = tsne_result[:, 0]
    y = tsne_result[:, 1]

    if len(x) < 10:
        print("Too few points for density estimation.")
        return

    try:
        xy = np.vstack([x, y])
        density = gaussian_kde(xy)(xy)
    except Exception as exc:
        print("Density estimation failed:", exc)
        return

    order = density.argsort()
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(x[order], y[order], c=density[order], s=args.point_size, cmap="jet", linewidths=0)
    plt.title(f"Density of LA 3D Features ({args.feature_source})")
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar(sc)
    cbar.set_ticks([density.min(), density.max()])
    cbar.set_ticklabels(["Low", "High"])
    plt.tight_layout()

    save_path = os.path.join(args.save_dir, "tsne_la3d_density.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved:", save_path)


def plot_tsne_combined(tsne_result: np.ndarray, labels: np.ndarray, args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for cls_id in np.unique(labels):
        mask = labels == cls_id
        color = get_paper_like_color(cls_id) if args.paper_colors else None
        axes[0].scatter(
            tsne_result[mask, 0],
            tsne_result[mask, 1],
            s=args.point_size,
            alpha=args.alpha,
            c=color,
            label=get_class_name(cls_id),
            linewidths=0,
        )

    axes[0].set_title("Feature Clusters")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].legend(markerscale=3, fontsize=9)

    x = tsne_result[:, 0]
    y = tsne_result[:, 1]
    try:
        xy = np.vstack([x, y])
        density = gaussian_kde(xy)(xy)
        order = density.argsort()
        sc = axes[1].scatter(x[order], y[order], c=density[order], s=args.point_size, cmap="jet", linewidths=0)

        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        axes[1].set_xlim(x.min() - x_range * 0.04, x.max() + x_range * 0.22)
        axes[1].set_ylim(y.min() - y_range * 0.04, y.max() + y_range * 0.04)
        axes[1].set_title("Feature Density")

        cax = axes[1].inset_axes([0.92, 0.70, 0.015, 0.25])
        cbar = fig.colorbar(sc, cax=cax, orientation="vertical")
        cbar.set_ticks([density.min(), density.max()])
        cbar.set_ticklabels(["Low", "High"])
        cbar.ax.tick_params(labelsize=8, length=0, pad=2)
    except Exception:
        axes[1].scatter(x, y, s=args.point_size, alpha=args.alpha, linewidths=0)
        axes[1].set_title("Density Estimation Failed")

    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.tight_layout()

    save_path = os.path.join(args.save_dir, "tsne_la3d_combined.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved:", save_path)


def save_npz(tsne_result: np.ndarray, labels: np.ndarray, features: np.ndarray, stats: dict, args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir, exist_ok=True)
    raw_path = os.path.join(args.save_dir, "la3d_features_raw.npz")
    tsne_path = os.path.join(args.save_dir, "la3d_features_tsne.npz")
    config_path = os.path.join(args.save_dir, "la3d_tsne_config.json")

    np.savez_compressed(raw_path, features=features, labels=labels)
    np.savez_compressed(tsne_path, tsne=tsne_result, labels=labels)

    config = vars(args).copy()
    config["root_path"] = str(args.root_path)
    config["ckpt"] = str(args.ckpt)
    config["save_dir"] = str(args.save_dir)
    config["stats"] = stats
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Saved:", raw_path)
    print("Saved:", tsne_path)
    print("Saved:", config_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--root_path", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--save_dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--n_filters", type=int, default=16)
    parser.add_argument("--strict_load", action="store_true")

    parser.add_argument(
        "--feature_source",
        type=str,
        default="decoder_cat",
        choices=["decoder_fg", "decoder_bg", "decoder_cat", "logits", "prob", "fg_logits", "bg_logits"],
        help="Feature tensor to reduce with t-SNE.",
    )
    parser.add_argument("--label_source", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--include_background", action="store_true", default=True)
    parser.add_argument("--ignore_label", type=int, default=255)

    parser.add_argument("--patch_size", type=int, nargs=3, default=[112, 112, 80])
    parser.add_argument("--patches_per_case", type=int, default=4)
    parser.add_argument("--fg_patch_ratio", type=float, default=0.75)
    parser.add_argument("--max_cases", type=int, default=0)

    parser.add_argument("--samples_per_class_per_patch", type=int, default=600)
    parser.add_argument("--samples_per_class", type=int, default=2500)
    parser.add_argument("--sample_region", type=str, default="all", choices=["all", "boundary", "non_boundary"])
    parser.add_argument("--boundary_width", type=int, default=3)

    parser.add_argument("--perplexity", type=float, default=32.0)
    parser.add_argument("--point_size", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--paper_colors", action="store_true", default=True)

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--gpu_id", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.root_path = Path(os.path.expanduser(str(args.root_path))).resolve()
    args.ckpt = Path(os.path.expanduser(str(args.ckpt))).resolve()
    args.save_dir = Path(os.path.expanduser(str(args.save_dir))).resolve()

    if not args.root_path.exists():
        raise FileNotFoundError(f"LA root path does not exist: {args.root_path}")
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.ckpt}")

    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = load_model(args, device)
    features, labels, stats = collect_features(args, model, device)
    tsne_result = run_tsne(features, args)

    save_npz(tsne_result, labels, features, stats, args)
    plot_tsne_scatter(tsne_result, labels, args)
    plot_tsne_density(tsne_result, args)
    plot_tsne_combined(tsne_result, labels, args)


if __name__ == "__main__":
    main()
