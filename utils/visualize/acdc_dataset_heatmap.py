#!/usr/bin/env python3
"""
ACDC dual-branch prediction visualizer.

This standalone script mirrors the data pipeline used in ``just_try/ACDC/ACDC_train_6_1.py``:

1. The dataset is enumerated via ``train_slices.list`` or ``val.list`` exactly as in training.
2. Each slice is resized to the network patch size (default 256x256) and fed twice into the
   CVBM2d Argument model so that we can grab the two prediction heads (foreground / background).
3. For every slice we draw a 3-panel figure: original image + two prediction heatmaps rendered as
   per-pixel grids whose values are normalized to [0, 1].

Run example
-----------

.. code-block:: bash

    python utils/visualize/acdc_dataset_heatmap.py \
        --root_path /root/ACDC \
        --checkpoint results/CVBM_6_1/1/iter_30000_dice_0.92.pth \
        --split val \
        --output_dir fig/heatmaps_val
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from networks.CVBM import CVBM, CVBM_Argument
from networks.unet import CVBM2d, CVBM2d_Argument


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------


class ACDCSliceDataset(Dataset):
    """
    Minimal dataset loader for the ACDC HDF5 slices used in ``ACDC_train_6_1.py``.

    Each item contains:
        - ``image`` : torch.FloatTensor shaped [1, H, W] ready to be batched.
        - ``label`` : torch.LongTensor shaped [H, W] (kept for reference / potential overlays).
        - ``case``  : str identifier read from the manifest file.

    Parameters
    ----------
    root_path : str or Path
        Root directory that stores ``train_slices.list``/``val.list`` and their ``data`` folders.
    split : {"train", "val", "test"}
        Dataset split to load. ``train`` reads from ``data/slices/*.h5`` while the rest read
        from ``data/*.h5`` just like the training script.
    patch_size : Iterable[int, int]
        Target spatial size fed to the network. Slices are resized with bilinear (images) / nearest
        (labels) interpolation so the tensor shape matches what ``WeakStrongAugment`` produces.
    """

    def __init__(
        self,
        root_path: str | Path,
        split: str = "val",
        patch_size: Iterable[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.root_path = Path(root_path)
        self.split = split
        self.patch_size = tuple(int(x) for x in patch_size)

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}'. Expected train/val/test.")

        manifest_name = "train_slices.list" if split == "train" else f"{split}.list"
        manifest_path = self.root_path / manifest_name
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest {manifest_path} not found.")

        with open(manifest_path, "r") as handle:
            self.cases = [line.strip() for line in handle if line.strip()]
        if not self.cases:
            raise RuntimeError(f"No cases found in {manifest_path}.")

        self.data_dir = (
            self.root_path / "data" / "slices"
            if split == "train"
            else self.root_path / "data"
        )

        self.samples = self._enumerate_samples()
        logging.info("Loaded %d %s slices from %s", len(self.samples), split, manifest_path)

    def _enumerate_samples(self) -> list[dict[str, Optional[int] | str]]:
        """
        Build a per-slice index so evaluation splits that store whole volumes can still be
        visualised slice by slice.
        """
        samples = []
        if self.split == "train":
            for case in self.cases:
                samples.append({"case": case, "slice_idx": None, "sample_id": case})
            return samples

        # val/test: each entry is a full 3D volume; enumerate slices for visualisation
        for case in self.cases:
            h5_path = self.data_dir / f"{case}.h5"
            if not h5_path.exists():
                raise FileNotFoundError(f"HDF5 file {h5_path} not found for case '{case}'.")
            with h5py.File(h5_path, "r") as handle:
                num_slices = handle["image"].shape[0]
            for slice_idx in range(num_slices):
                samples.append(
                    {
                        "case": case,
                        "slice_idx": slice_idx,
                        "sample_id": f"{case}_slice{slice_idx:03d}",
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        meta = self.samples[index]
        case = meta["case"]
        slice_idx = meta["slice_idx"]

        if self.split == "train":
            h5_path = self.data_dir / f"{case}.h5"
        else:
            h5_path = self.data_dir / f"{case}.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 file {h5_path} not found for case '{case}'.")

        with h5py.File(h5_path, "r") as handle:
            if slice_idx is None:
                image_np = np.asarray(handle["image"])  # (H, W)
                label_np = np.asarray(handle["label"])  # (H, W)
            else:
                image_np = np.asarray(handle["image"][slice_idx])
                label_np = np.asarray(handle["label"][slice_idx])

        image_tensor = self._prepare_image(image_np)
        label_tensor = self._prepare_label(label_np)

        return {
            "image": image_tensor,  # torch.FloatTensor [1, H, W]
            "label": label_tensor,  # torch.LongTensor [H, W]
            "case": case,
            "slice_idx": slice_idx,
            "sample_id": meta["sample_id"],
        }

    def _prepare_image(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
        tensor = F.interpolate(tensor, size=self.patch_size, mode="bilinear", align_corners=False)
        return tensor.squeeze(0)  # -> [1, H, W]

    def _prepare_label(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).float()
        tensor = F.interpolate(tensor, size=self.patch_size, mode="nearest")
        return tensor.squeeze().long()  # -> [H, W]


# --------------------------------------------------------------------------------------
# Model helpers
# --------------------------------------------------------------------------------------


def build_model(model_name: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    """
    Instantiate the requested segmentation model and move it to ``device``.

    The available names follow ``net_factory`` in the training code. The manual instantiation
    avoids the unconditional ``.cuda()`` call in ``net_factory`` so the script also works on CPU.
    """
    in_channels = 1
    if model_name == "CVBM2d":
        model = CVBM2d(in_chns=in_channels, class_num=num_classes)
    elif model_name == "CVBM2d_Argument":
        model = CVBM2d_Argument(in_chns=in_channels, class_num=num_classes)
    elif model_name == "CVBM_Argument":
        # Disable dropout for deterministic visualisation.
        model = CVBM_Argument(
            n_channels=in_channels,
            n_classes=num_classes,
            normalization="instancenorm",
            has_dropout=False,
        )
    elif model_name == "CVBM":
        model = CVBM(
            n_channels=in_channels,
            n_classes=num_classes,
            normalization="instancenorm",
            has_dropout=False,
        )
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            "Expected one of {CVBM2d, CVBM2d_Argument, CVBM, CVBM_Argument}."
        )

    return model.to(device)


def load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    """
    Load weights regardless of whether they were saved via ``torch.save(model.state_dict())`` or the
    ``save_net_opt`` helper (dict with ``net`` / ``opt`` entries).
    """
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "net" in state:
        state = state["net"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Strip possible "module." prefixes from DataParallel checkpoints.
    cleaned_state = {
        (key.replace("module.", "", 1) if key.startswith("module.") else key): value
        for key, value in state.items()
    }

    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if missing:
        logging.warning("Missing keys in checkpoint: %s", missing)
    if unexpected:
        logging.warning("Unexpected keys in checkpoint: %s", unexpected)


# --------------------------------------------------------------------------------------
# Visualisation helpers
# --------------------------------------------------------------------------------------


def _normalize_zero_one(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v - min_v < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_v) / (max_v - min_v)


def _prediction_to_heatmap(
    prediction: torch.Tensor,
    *,
    class_index: Optional[int] = None,
) -> np.ndarray:
    """
    Convert a raw logits tensor shaped [C, H, W] into an HxW probability map.
    """
    probs = torch.softmax(prediction, dim=0)  # [C, H, W]
    if class_index is None:
        heatmap = probs.max(dim=0).values
    else:
        if class_index < 0 or class_index >= probs.shape[0]:
            raise ValueError(f"class_index {class_index} outside [0, {probs.shape[0] - 1}]")
        heatmap = probs[class_index]
    return _normalize_zero_one(heatmap.cpu().numpy())


def _add_pixel_grid(ax: plt.Axes, shape: tuple[int, int]) -> None:
    h, w = shape
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3, alpha=0.4)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)


def plot_triplet(
    image_2d: np.ndarray,
    heatmap_main: np.ndarray,
    heatmap_aux: np.ndarray,
    *,
    title: str,
    save_path: Path,
    show: bool = False,
) -> None:
    """
    Render the 3-panel figure and optionally show it on screen.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150, constrained_layout=True)

    axes[0].imshow(image_2d, cmap="gray", interpolation="nearest")
    axes[0].set_title(f"{title}\nOriginal")
    axes[0].axis("off")
    _add_pixel_grid(axes[0], image_2d.shape)

    for ax, heatmap, subtitle in zip(
        axes[1:], (heatmap_main, heatmap_aux), ("Prediction #1", "Prediction #2")
    ):
        im = ax.imshow(heatmap, cmap="turbo", interpolation="nearest", vmin=0.0, vmax=1.0)
        ax.set_title(subtitle)
        ax.axis("off")
        _add_pixel_grid(ax, heatmap.shape)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("Probability")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    logging.info("Saved heatmap to %s", save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


def make_augmented_view(image: torch.Tensor, noise_std: float) -> torch.Tensor:
    """
    Create the second model input.

    Args
    ----
    image : torch.Tensor [B, 1, H, W]
        Weakly augmented slice (already resized).
    noise_std : float
        Standard deviation of the Gaussian noise applied to obtain a stronger perturbation.
        Set to 0.0 to simply clone ``image``.
    """
    if noise_std <= 0.0:
        return image.clone()
    noise = torch.randn_like(image) * noise_std
    return torch.clamp(image + noise, min=float(image.min()), max=float(image.max()))


# --------------------------------------------------------------------------------------
# Main routine
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 3-panel heatmaps (original + dual predictions) for the ACDC dataset."
    )
    parser.add_argument("--root_path", type=Path, default="/root/ACDC", help="Root directory of the ACDC dataset.")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test", help="Dataset split.")
    parser.add_argument("--patch_size", type=int, nargs=2, default=[256, 256], help="Model input size (H W).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=2, help="Dataloader worker count.")
    parser.add_argument("--model", type=str, default="CVBM2d_Argument", help="Model name used during training.")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of output classes.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained checkpoint.")
    parser.add_argument("--output_dir", type=Path, default=Path("fig/acdc_heatmaps"), help="Directory to store figures.")
    parser.add_argument("--device", type=str, default=None, help="Torch device string (default: auto-detect).")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on the number of slices to plot.")
    parser.add_argument("--main_class", type=int, default=None, help="Specific class to visualise for prediction #1.")
    parser.add_argument("--aux_class", type=int, default=None, help="Specific class to visualise for prediction #2.")
    parser.add_argument("--strong_noise_std", type=float, default=0.05, help="Std of Gaussian noise for the second view.")
    parser.add_argument("--show", action="store_true", help="Display figures interactively in addition to saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logging.info("Using device: %s", device)

    dataset = ACDCSliceDataset(
        root_path=args.root_path,
        split=args.split,
        patch_size=args.patch_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(args.model, args.num_classes, device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    rendered = 0
    progress = tqdm(dataloader, desc="Visualizing", unit="slice")
    with torch.no_grad():
        for batch in progress:
            if args.max_samples is not None and rendered >= args.max_samples:
                break

            image = batch["image"].to(device)  # [B, 1, H, W]
            strong_view = make_augmented_view(image, args.strong_noise_std)

            preds_fg, _, preds_bg, _, _ = model(image, strong_view)

            for b in range(image.shape[0]):
                sample_id = batch["sample_id"][b]
                case_id = batch["case"][b]
                image_np = _normalize_zero_one(image[b, 0].cpu().numpy())
                heatmap_main = _prediction_to_heatmap(
                    preds_fg[b].cpu(),
                    class_index=args.main_class,
                )
                heatmap_aux = _prediction_to_heatmap(
                    preds_bg[b].cpu(),
                    class_index=args.aux_class,
                )

                save_path = args.output_dir / f"{sample_id}.png"
                plot_triplet(
                    image_np,
                    heatmap_main,
                    heatmap_aux,
                    title=f"{case_id} | {sample_id}",
                    save_path=save_path,
                    show=args.show,
                )

                rendered += 1
                if args.max_samples is not None and rendered >= args.max_samples:
                    break

            progress.set_postfix({"rendered": rendered})

    logging.info("Finished rendering %d samples.", rendered)


if __name__ == "__main__":
    main()
