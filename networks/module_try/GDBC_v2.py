"""
Improved Global Dual-Branch Consistency (GDBC) loss.

Key differences vs. the original implementation:
1. Accepts weak/strong foreground features separately so L_cv acts on cross-view descriptors.
2. Builds probability weights from (teacher) softmax outputs for both fg/bg, ensuring valid pooling
   weights and making it easy to plug in EMA predictions.
3. Keeps the separation term focused on foreground vs. background descriptors, so L_cv and L_sep
   no longer fight the exact same vector pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _prob_weighted_global_pool(feat: torch.Tensor, prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Args:
        feat: (B, C, H, W) feature map
        prob: (B, 1, H, W) weight map in [0, 1]
    Returns:
        pooled: (B, C)
    """
    b, c, _, _ = feat.shape
    assert prob.shape[0] == b and prob.shape[1] == 1, "Probability map must be (B,1,H,W)"
    num = (feat * prob).sum(dim=(2, 3))
    den = prob.sum(dim=(2, 3)).clamp_min(eps)
    return num / den


@dataclass
class GDBCWeights:
    fg: torch.Tensor  # (B,1,H,W)
    bg: torch.Tensor  # (B,1,H,W)


class GDBCLossV2(nn.Module):
    """
    Cross-view consistency + foreground/background separation on global descriptors.

    Expected call:
        loss_dict = criterion(
            feats_fg_weak=features from weak/standard view (B,C,H,W),
            feats_bg_strong=features from strong/aug view (B,C,H,W),
            feats_bg=background branch features (B,C,H,W),
            probs_fg=student logits or probabilities for the foreground branch,
            probs_bg=student logits/probabilities for the background branch (optional),
            teacher_probs_fg=EMA logits/probabilities (optional),
            teacher_probs_bg=EMA logits/probabilities (optional),
            global_step=current_iteration
        )
    """

    def __init__(
        self,
        lambda_cv: float = 0.5,
        lambda_sep: float = 0.5,
        cos_margin: float = 0.5,
        warmup_steps: int = 1500,
        use_teacher_probs: bool = True,
        expect_logits: bool = True,
        fg_classes: Tuple[int, ...] = (1, 2, 3),
        bg_class: int = 0,
        conf_threshold: Optional[float] = 0.2,
        downsample_mode: str = "area",
    ) -> None:
        super().__init__()
        self.lambda_cv = float(lambda_cv)
        self.lambda_sep = float(lambda_sep)
        self.cos_margin = float(cos_margin)
        self.warmup_steps = int(warmup_steps)
        self.use_teacher_probs = bool(use_teacher_probs)
        self.expect_logits = bool(expect_logits)
        self.fg_classes = tuple(fg_classes)
        self.bg_class = int(bg_class)
        self.conf_threshold = conf_threshold
        self.downsample_mode = downsample_mode
        self.bg_adapter: Optional[nn.Conv2d] = None

    def forward(
        self,
        *,
        feats_fg_weak: torch.Tensor,
        feats_bg_strong: torch.Tensor,
        feats_bg: torch.Tensor,
        probs_fg: torch.Tensor,
        probs_bg: Optional[torch.Tensor] = None,
        teacher_probs_fg: Optional[torch.Tensor] = None,
        teacher_probs_bg: Optional[torch.Tensor] = None,
        global_step: Optional[int] = None,
    ) -> dict:
        target_hw = feats_fg_weak.shape[2:]
        feats_bg_strong = self._resize_feat(feats_bg_strong, target_hw)
        feats_bg = self._resize_feat(feats_bg, target_hw)
        feats_bg = self._align_bg_channels(feats_bg, feats_fg_weak.shape[1])
        self._validate_shapes(feats_fg_weak, feats_bg_strong, feats_bg, probs_fg)

        fg_weight_source = (
            teacher_probs_fg if (self.use_teacher_probs and teacher_probs_fg is not None) else probs_fg
        )
        bg_weight_source = (
            teacher_probs_bg if (self.use_teacher_probs and teacher_probs_bg is not None) else probs_bg
        )

        fg_weight_source = self._maybe_softmax(fg_weight_source)
        if bg_weight_source is not None:
            bg_weight_source = self._maybe_softmax(bg_weight_source)

        weights = self._build_weights(
            fg_weight_source.detach(),
            bg_weight_source.detach() if bg_weight_source is not None else None,
            target_hw=feats_fg_weak.shape[2:],
        )

        g_fg_w = _prob_weighted_global_pool(feats_fg_weak, weights.fg)
        g_bg_s = _prob_weighted_global_pool(feats_bg_strong, weights.fg)
        g_bg = _prob_weighted_global_pool(feats_bg, weights.bg)

        g_fg_w = F.normalize(g_fg_w, dim=1)
        g_bg_s = F.normalize(g_bg_s, dim=1)
        g_bg = F.normalize(g_bg, dim=1)

        l_cv = F.mse_loss(g_fg_w, g_bg_s)
        cos_fg_bg = (g_fg_w * g_bg).sum(dim=1)
        l_sep = F.relu(cos_fg_bg - self.cos_margin).mean()

        sep_scale = self._warmup_scale(global_step)
        total = self.lambda_cv * l_cv + (self.lambda_sep * sep_scale) * l_sep

        return {
            "loss": total,
            "l_cv": l_cv.detach(),
            "l_sep": l_sep.detach(),
            "sep_scale": torch.tensor(sep_scale, device=feats_fg_weak.device),
            "cos_sim_fg_bg": cos_fg_bg.detach().mean(),
        }

    # ------------------------------------------------------------------ helpers
    def _validate_shapes(self, fg_w, fg_s, bg, probs_fg):
        assert fg_w.shape == fg_s.shape == bg.shape, "feature tensors must share shape"
        assert fg_w.dim() == 4, ...
        assert probs_fg.shape[0] == fg_w.shape[0], "batch mismatch"

    def _resize_feat(self, feat: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        if feat.shape[2:] == target_hw:
            return feat
        return F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)

    def _maybe_softmax(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.softmax(tensor, dim=1) if self.expect_logits else tensor

    def _build_weights(
        self,
        probs_fg: torch.Tensor,
        probs_bg: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
    ) -> GDBCWeights:
        # Sum foreground classes; if empty, assume all except bg_class
        if len(self.fg_classes) == 0:
            fg_prob = 1.0 - probs_fg[:, self.bg_class:self.bg_class + 1]
        else:
            fg_prob = probs_fg[:, self.fg_classes, :, :].sum(dim=1, keepdim=True)

        if probs_bg is not None:
            bg_prob = probs_bg[:, self.bg_class:self.bg_class + 1, :, :]
        else:
            bg_prob = (1.0 - fg_prob).clamp_min(0.0)

        fg_weight = self._downsample(fg_prob, target_hw)
        bg_weight = self._downsample(bg_prob, target_hw)

        fg_weight, bg_weight = self._normalize_pair(fg_weight, bg_weight)
        if self.conf_threshold is not None:
            thresh = float(self.conf_threshold)
            fg_weight = fg_weight * (fg_weight >= thresh).float()
            bg_weight = bg_weight * (bg_weight >= thresh).float()
        return GDBCWeights(fg=fg_weight.detach(), bg=bg_weight.detach())

    def _downsample(self, tensor: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        if tensor.shape[2:] == target_hw:
            return tensor
        if self.downsample_mode == "area":
            return F.interpolate(tensor, size=target_hw, mode="area")
        return F.interpolate(tensor, size=target_hw, mode="bilinear", align_corners=False)

    def _normalize_pair(self, fg: torch.Tensor, bg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_sum = (fg + bg).clamp_min(1e-6)
        fg = fg / pair_sum
        bg = bg / pair_sum
        return fg, bg

    def _align_bg_channels(self, tensor: torch.Tensor, target_c: int) -> torch.Tensor:
        current_c = tensor.shape[1]
        if current_c == target_c:
            return tensor
        if (
            self.bg_adapter is None
            or self.bg_adapter.in_channels != current_c
            or self.bg_adapter.out_channels != target_c
        ):
            self.bg_adapter = nn.Conv2d(current_c, target_c, kernel_size=1, bias=False)
        device, dtype = tensor.device, tensor.dtype
        self.bg_adapter = self.bg_adapter.to(device=device, dtype=dtype)
        return self.bg_adapter(tensor)

    def _warmup_scale(self, global_step: Optional[int]) -> float:
        if global_step is None or self.warmup_steps <= 0:
            return 1.0
        return min(1.0, max(0.0, global_step / float(self.warmup_steps)))


__all__ = ["GDBCLossV2"]
