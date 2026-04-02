"""3D semantic knowledge complementarity models for experiment cvbm_14_1.

This version ports the 2D SKC idea to 3D by concatenating foreground and
background bottleneck features along the channel dimension, mixing them with
lightweight 1x1x1 convolutions, and then projecting branch-specific residuals
back to the original bottleneck resolution.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from networks.CVBM import Decoder, Encoder


class SemanticKnowledgeComplementarity3D(nn.Module):
    """3D channel-concatenation SKC block.

    The block first pools the concatenated ``fg/bg`` bottleneck features to a
    compact 3D window. Both branch-specific residuals are then generated from
    the same mixed representation, which creates an explicit feature-level
    interaction path between the two branches.
    """

    def __init__(
        self,
        channels: int,
        attn_shape: Tuple[int, int, int] = (4, 8, 8),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_shape = attn_shape
        self.resize_mode = "trilinear"
        hidden = max(channels // 2, 32)

        self.fg_mlp = nn.Sequential(
            nn.Conv3d(channels * 2, hidden, kernel_size=1),
            nn.InstanceNorm3d(hidden, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(hidden, channels, kernel_size=1),
        )
        self.bg_mlp = nn.Sequential(
            nn.Conv3d(channels * 2, hidden, kernel_size=1),
            nn.InstanceNorm3d(hidden, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(hidden, channels, kernel_size=1),
        )

    def forward(
        self,
        fg_feature: torch.Tensor,
        bg_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        original_size = fg_feature.shape[-3:]
        combined = torch.cat([fg_feature, bg_feature], dim=1)

        pooled = F.interpolate(
            combined,
            size=self.attn_shape,
            mode=self.resize_mode,
            align_corners=False,
        )

        fg_delta = self.fg_mlp(pooled)
        bg_delta = self.bg_mlp(pooled)

        fg_delta = F.interpolate(
            fg_delta,
            size=original_size,
            mode=self.resize_mode,
            align_corners=False,
        )
        bg_delta = F.interpolate(
            bg_delta,
            size=original_size,
            mode=self.resize_mode,
            align_corners=False,
        )

        return fg_feature + fg_delta, bg_feature + bg_delta


class CVBMArgumentWithSKC3D(nn.Module):
    """CVBM_Argument backbone with 3D channel-concatenation SKC."""

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 2,
        n_filters: int = 16,
        normalization: str = "instancenorm",
        has_dropout: bool = False,
        has_residual: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            n_channels,
            n_classes,
            n_filters,
            normalization,
            has_dropout,
            has_residual,
        )
        self.decoder_fg = Decoder(
            n_channels,
            n_classes,
            n_filters,
            normalization,
            has_dropout,
            has_residual,
            up_type=0,
        )
        self.decoder_bg = Decoder(
            n_channels,
            n_classes,
            n_filters,
            normalization,
            has_dropout,
            has_residual,
            up_type=2,
        )
        self.skc = SemanticKnowledgeComplementarity3D(channels=n_filters * 16)
        self.final_seg = nn.Conv3d(n_classes * 2, n_classes, kernel_size=1)

    def forward(
        self,
        input_fg: torch.Tensor,
        input_bg: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fg_feats = list(self.encoder(input_fg))
        bg_feats = list(self.encoder(input_bg))

        fg_feats[-1], bg_feats[-1] = self.skc(fg_feats[-1], bg_feats[-1])

        out_fg, attn_fg = self.decoder_fg(fg_feats)
        out_bg, attn_bg = self.decoder_bg(bg_feats)
        fused_logits = self.final_seg(torch.cat([out_fg, out_bg], dim=1))
        return out_fg, fused_logits, out_bg, attn_fg, attn_bg
