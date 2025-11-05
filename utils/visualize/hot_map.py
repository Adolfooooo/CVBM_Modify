import numpy as np
import matplotlib.pyplot as plt

def plot_probability_heatmap(prob_matrix: np.ndarray, save_path: str | None = None):
    """
    将二维概率矩阵可视化为热力图。
    参数
    ----
    prob_matrix : 2D numpy array，元素建议在 [0, 1]（概率）
    save_path   : 可选，保存图片的路径（例如 "probability_heatmap.png"）
    """
    if prob_matrix.ndim != 2:
        raise ValueError("prob_matrix must be a 2D array.")

    # 可选：将数据裁剪到 [0, 1] 区间，避免异常值影响配色范围
    prob_matrix = prob_matrix.detach().cpu().numpy()
    data = np.clip(prob_matrix, 0.0, 1.0)

    plt.figure(figsize=(6, 5), dpi=120)
    im = plt.imshow(
        data,
        cmap="viridis",          # 常用感知均匀配色
        origin="lower",          # (0,0) 放左下角，符合直觉
        interpolation="nearest", # 保持像素格不插值
        vmin=0.0, vmax=1.0       # 概率范围固定为 [0,1]
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Probability", rotation=90)

    plt.title("Probability Heatmap")
    plt.xlabel("X (column)")
    plt.ylabel("Y (row)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


    import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _to_numpy(x):
    """Convert torch.Tensor or numpy array to numpy array (on CPU)."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_prob_map(soft, target_hw):
    """
    Convert soft label tensor/array to an HxW probability map in [0,1].
    Accepts:
      - HxW (already single-class prob map)
      - CxHxW (channel-first), takes max across channels by default
      - HxWxC (channel-last), takes max across channels by default
    """
    soft = _to_numpy(soft).astype(np.float32)
    Ht, Wt = target_hw

    if soft.ndim == 2:
        if soft.shape != (Ht, Wt):
            raise ValueError(f"Soft prob shape {soft.shape} does not match pseudo label shape {(Ht, Wt)}.")
        prob = soft

    elif soft.ndim == 3:
        # Try channel-first: (C,H,W)
        if soft.shape[1:] == (Ht, Wt):
            prob = soft.max(axis=0)
        # Try channel-last: (H,W,C)
        elif soft.shape[:2] == (Ht, Wt):
            prob = soft.max(axis=2)
        else:
            raise ValueError(
                f"Cannot align soft label shape {soft.shape} to target {(Ht, Wt)}. "
                "Expected (C,H,W) or (H,W,C) with matching H,W."
            )
    else:
        raise ValueError(f"Unsupported soft label ndim={soft.ndim}. Provide HxW or CxHxW or HxWxC.")

    # Optional clamp to [0,1]
    prob = np.clip(prob, 0.0, 1.0)
    return prob


def plot_pseudo_and_heatmap(pseudo_labels, soft_labels, class_names=None, save_path=None):
    """
    Plot side-by-side:
      - Left: pseudo-label segmentation (discrete colormap)
      - Right: soft-label probability heatmap (continuous 0-1)
    Args:
      pseudo_labels: HxW (int) pseudo label map (torch.Tensor or np.ndarray)
      soft_labels:   HxW or CxHxW or HxWxC soft probabilities (same H,W)
      class_names:   optional list/tuple of class names aligned to class ids found in pseudo_labels
      save_path:     optional path to save (e.g., 'vis_pair.png' or 'vis_pair.jpg')
    """
    # --- Prepare data ---
    pseudo = _to_numpy(pseudo_labels)
    if pseudo.ndim != 2:
        raise ValueError("pseudo_labels must be a 2D map (HxW).")
    if not np.issubdtype(pseudo.dtype, np.integer):
        # Ensure integers for discrete classes
        pseudo = pseudo.astype(np.int32)

    H, W = pseudo.shape
    prob = _as_prob_map(soft_labels, (H, W))

    # --- Pseudo-label colormap (discrete) ---
    unique_classes = np.unique(pseudo)
    num_classes = int(unique_classes.max()) + 1 if unique_classes.size else 1

    # Build a ListedColormap using tab20 (cycled if classes > 20)
    base_cmap = plt.get_cmap("tab20")
    colors_list = [base_cmap(i % base_cmap.N) for i in range(num_classes)]
    cmap_discrete = colors.ListedColormap(colors_list)
    boundaries = np.arange(num_classes + 1) - 0.5
    norm_discrete = colors.BoundaryNorm(boundaries, cmap_discrete.N)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)

    # Left: Pseudo labels
    ax0 = axes[0]
    im0 = ax0.imshow(pseudo, cmap=cmap_discrete, norm=norm_discrete,
                     interpolation="nearest", origin="lower")
    ax0.set_title("Pseudo-label Segmentation")
    ax0.set_xlabel("X (column)")
    ax0.set_ylabel("Y (row)")

    # Build ticks for colorbar using only classes present
    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04,
                         ticks=unique_classes)
    if class_names:
        # Map tick labels via class_names if index exists, else fallback to id
        tick_labels = []
        for cid in unique_classes:
            if 0 <= cid < len(class_names):
                tick_labels.append(str(class_names[cid]))
            else:
                tick_labels.append(str(cid))
        cbar0.ax.set_yticklabels(tick_labels)

    # Right: Probability heatmap
    ax1 = axes[1]
    im1 = ax1.imshow(prob, cmap="viridis", vmin=0.0, vmax=1.0,
                     interpolation="nearest", origin="lower")
    ax1.set_title("Soft-label Probability")
    ax1.set_xlabel("X (column)")
    ax1.set_ylabel("Y (row)")
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Probability")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
