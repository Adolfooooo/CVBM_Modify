#!/usr/bin/env python3
"""
可视化脚本: 与 ACDC_train_4_6_3.py 相同的数据管线, 读取 .pth 权重, 绘制四联图.

- 输入: 训练阶段生成的 .pth 模型权重、ACDC 数据根目录、输出目录等命令行参数。
- 输出: 每个样本一张图像 (4 个子图), 依次展示原始图、标签叠加、弱增强预测叠加、强增强预测叠加。
- 功能点:
    * 复用 BaseDataSets + WeakStrongAugment + CreateOnehotLabel 的采样格式, 无需重新写数据集读取逻辑。
    * 模型推理采用 net_factory, 以便替换其它网络。
    * 可视化模块 (VisualizationModule) 可独立替换, 方便后续拼接其它展示逻辑。
    * 支持全量遍历或抽样遍历 (通过 --visualize_all / --max_samples 控制)。
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List

import matplotlib

# 使用非交互式后端, 方便在服务器/终端中保存图片
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders.dataset import BaseDataSets, CreateOnehotLabel, WeakStrongAugment
from networks.net_factory import net_factory


@dataclass
class VisualizationModule:
    """
    单个可视化模块。

    Args:
        title: str, 子图标题。
        renderer: Callable, 签名为 renderer(ax, context)。ax 为 matplotlib.Axes,
                  context 为包含图像/掩码数据的上下文字典。
    """

    title: str
    renderer: Callable[[plt.Axes, Dict[str, np.ndarray]], None]

    def __call__(self, ax: plt.Axes, context: Dict[str, np.ndarray]) -> None:
        """执行渲染逻辑。"""
        self.renderer(ax, context)
        ax.set_title(self.title, fontsize=11)
        ax.axis("off")


def get_parser() -> argparse.ArgumentParser:
    """配置命令行参数。"""
    parser = argparse.ArgumentParser(description="ACDC 可视化脚本")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="训练得到的 .pth 文件路径, 例如 './results/.../best_model.pth'")
    parser.add_argument("--data_root", type=str, required=True,
                        help="ACDC 数据集根目录, 与训练脚本的 --root_path 保持一致")
    parser.add_argument("--save_dir", type=str, default="./visual_outputs",
                        help="保存可视化图像的文件夹")
    parser.add_argument("--model", type=str, default="CVBM2d_Argument",
                        help="net_factory 中的模型名, 默认与 ACDC_train_4_6_3.py 相同")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="分割类别数 (含背景)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="DataLoader 的 batch size, 仅影响可视化速度")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader 的 worker 数量")
    parser.add_argument("--visualize_all", action="store_true",
                        help="若设置则遍历所有样本, 否则只处理 --max_samples 个样本")
    parser.add_argument("--max_samples", type=int, default=16,
                        help="当未设置 --visualize_all 时, 要可视化的样本数量")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="掩码叠加到图像上的透明度系数 (0~1)")
    parser.add_argument("--seed", type=int, default=1337,
                        help="随机种子, 影响数据增强及可视化顺序")
    return parser


def build_color_palette(num_classes: int) -> np.ndarray:
    """
    生成颜色查找表, shape=[num_classes, 3], RGB 范围 0~255。
    Returns:
        palette: np.ndarray, uint8。
    """
    base_colors = np.array(
        [
            (0, 0, 0),        # 背景
            (255, 92, 92),    # 类1
            (79, 193, 255),   # 类2
            (120, 255, 184),  # 类3
            (252, 255, 164),  # 类4 (若有)
            (255, 180, 255),  # 额外色
            (164, 140, 255),
        ],
        dtype=np.uint8,
    )
    if num_classes <= base_colors.shape[0]:
        return base_colors[:num_classes]
    rng = np.random.default_rng(42)
    extra = rng.integers(0, 255, size=(num_classes - base_colors.shape[0], 3), dtype=np.uint8)
    return np.concatenate([base_colors, extra], axis=0)


def tensor_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    """
    将 [C, H, W] 或 [H, W] 形状的 Tensor 转换为 numpy 单通道图像。

    Args:
        image_tensor: torch.Tensor, shape=[C,H,W] 或 [H,W], 数据类型 float。
    Returns:
        image_np: np.ndarray, shape=[H,W], dtype=float32。
    """
    if image_tensor.ndim == 3:
        # 只取第一个通道即可满足单通道展示需求。
        image_np = image_tensor[0].detach().cpu().numpy()
    else:
        image_np = image_tensor.detach().cpu().numpy()
    return image_np.astype(np.float32)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    将任意范围的灰度图缩放到 [0, 255]。
    Args:
        image: np.ndarray, shape=[H,W], 任意值域。
    Returns:
        np.ndarray, uint8, shape=[H,W]。
    """
    image = image - np.min(image)
    denom = np.ptp(image) + 1e-8
    scaled = image / denom
    return (scaled * 255).clip(0, 255).astype(np.uint8)


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    palette: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    将掩码叠加到灰度图上。

    Args:
        image: np.ndarray, shape=[H,W], 输入灰度图。
        mask: np.ndarray, shape=[H,W], 整型标签, 值域[0, num_classes-1]。
        palette: np.ndarray, shape=[num_classes, 3], 每个类别的 RGB 颜色。
        alpha: float, 叠加透明度。
    Returns:
        overlay: np.ndarray, shape=[H,W,3], uint8, 用于可视化的彩色图。
    """
    base = normalize_to_uint8(image)
    # 拓展到三通道, 便于上色
    overlay = np.stack([base, base, base], axis=-1).astype(np.float32)
    clipped_mask = np.clip(mask, 0, palette.shape[0] - 1)
    color_mask = palette[clipped_mask]
    fg = clipped_mask > 0  # 仅在前景区域上色
    overlay[fg] = (overlay[fg] * (1 - alpha) + color_mask[fg] * alpha)
    return overlay.astype(np.uint8)


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """
    构造并加载模型。

    Returns:
        model: torch.nn.Module, 已加载权重并处于 eval 模式。
    """
    model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes, mode="test")
    model = model.to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict):
        if "net" in checkpoint:
            state_dict = checkpoint["net"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    logging.info("Loaded checkpoint from %s", args.checkpoint)
    return model


def build_dataloader(args: argparse.Namespace) -> DataLoader:
    """
    创建数据加载器, 复用训练阶段的数据增强。
    Returns:
        torch.utils.data.DataLoader。
    """
    transform = transforms.Compose([
        WeakStrongAugment(args.patch_size),
        CreateOnehotLabel(args.num_classes),
    ])
    dataset = BaseDataSets(base_dir=args.data_root, split="train", transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    return loader


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    weak_image: torch.Tensor,
    strong_image: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """
    执行一次前向推理。

    Args:
        model: torch.nn.Module, 已加载好权重。
        weak_image: torch.Tensor, shape=[1,H,W], 弱增强输入。
        strong_image: torch.Tensor, shape=[1,H,W], 强增强输入。
    Returns:
        dict: 包含 'weak_pred'、'strong_pred' 两个 numpy 掩码, shape=[H,W]。
    """
    outputs_fg, _, outputs_bg, _, _ = model(weak_image.unsqueeze(0), strong_image.unsqueeze(0))
    weak_pred = torch.argmax(outputs_fg, dim=1).squeeze(0).cpu().numpy()
    strong_pred = torch.argmax(outputs_bg, dim=1).squeeze(0).cpu().numpy()
    return {"weak_pred": weak_pred.astype(np.int32), "strong_pred": strong_pred.astype(np.int32)}


def create_visual_modules(alpha: float) -> List[VisualizationModule]:
    """定义四个默认的可视化模块."""

    def render_original(ax: plt.Axes, context: Dict[str, np.ndarray]) -> None:
        ax.imshow(context["original_image"], cmap="gray")

    def make_overlay_module(title: str, mask_key: str, image_key: str) -> VisualizationModule:
        def renderer(ax: plt.Axes, context: Dict[str, np.ndarray]) -> None:
            overlay = overlay_mask_on_image(
                context[image_key],
                context[mask_key],
                context["palette"],
                context["alpha"],
            )
            ax.imshow(overlay)
        return VisualizationModule(title, renderer)

    modules = [
        VisualizationModule("Original Image", render_original),
        make_overlay_module("Label Overlay", "gt_mask", "original_image"),
        make_overlay_module("Weak Aug. Prediction", "weak_pred", "weak_image"),
        make_overlay_module("Strong Aug. Prediction", "strong_pred", "strong_image"),
    ]
    return modules


def prepare_context(
    sample: Dict[str, torch.Tensor],
    index: int,
    preds: Dict[str, np.ndarray],
    palette: np.ndarray,
    alpha: float,
) -> Dict[str, np.ndarray]:
    """
    整理单个样本绘图所需的数据。

    Args:
        sample: DataLoader 返回的 batch, 包含图像/标签等字段。
        index: int, 取 batch 内第 index 个样本。
        preds: dict, run_inference 输出的预测结果。
        palette: np.ndarray, 颜色查找表。
        alpha: float, 叠加透明度。
    Returns:
        context: dict, 供可视化模块使用的字典。
    """
    weak_image = sample["image"][index]
    strong_image = sample["image_strong"][index]
    label_mask = sample["label"][index].cpu().numpy()
    context = {
        "case_id": sample["case"][index],
        "original_image": tensor_to_numpy(weak_image),
        "weak_image": tensor_to_numpy(weak_image),
        "strong_image": tensor_to_numpy(strong_image),
        "gt_mask": label_mask,
        "weak_pred": preds["weak_pred"],
        "strong_pred": preds["strong_pred"],
        "palette": palette,
        "alpha": alpha,
    }
    return context


def save_figure(modules: List[VisualizationModule], context: Dict[str, np.ndarray], save_path: str) -> None:
    """
    根据模块列表绘制并保存图像。

    Args:
        modules: List[VisualizationModule], 定义子图内容。
        context: dict, prepare_context 的输出, 包含绘图数据。
        save_path: str, 输出路径。
    """
    fig, axes = plt.subplots(1, len(modules), figsize=(5 * len(modules), 4))
    if len(modules) == 1:
        axes = [axes]
    for ax, module in zip(axes, modules):
        module(ax, context)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def determine_device() -> torch.device:
    """根据硬件情况返回 cuda/cpu 设备。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = get_parser()
    parser.add_argument("--patch_size", type=int, nargs=2, metavar=("H", "W"), default=[256, 256],
                        help="WeakStrongAugment 输出的裁剪尺寸, 例如 --patch_size 256 256")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = determine_device()
    palette = build_color_palette(args.num_classes)
    modules = create_visual_modules(args.alpha)
    model = load_model(args, device)
    dataloader = build_dataloader(args)

    sample_limit = len(dataloader.dataset) if args.visualize_all else min(args.max_samples, len(dataloader.dataset))
    processed = 0

    logging.info("Start visualization: %d samples", sample_limit)
    for batch in dataloader:
        batch_size = batch["image"].shape[0]
        for idx in range(batch_size):
            if processed >= sample_limit:
                break
            weak_image = batch["image"][idx].to(device)
            strong_image = batch["image_strong"][idx].to(device)
            preds = run_inference(model, weak_image, strong_image)
            context = prepare_context(batch, idx, preds, palette, args.alpha)
            case_id = context["case_id"]
            save_path = os.path.join(args.save_dir, f"{case_id}.png")
            save_figure(modules, context, save_path)
            processed += 1
            logging.info("Saved visualization for case %s (%d/%d)", case_id, processed, sample_limit)
        if processed >= sample_limit:
            break
    logging.info("Visualization finished: %d figures", processed)


if __name__ == "__main__":
    main()
