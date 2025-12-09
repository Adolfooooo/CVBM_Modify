"""2D semantic knowledge complementarity components."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from networks.unet import Encoder, Decoder


class SemanticKnowledgeComplementarity2D(nn.Module):
    """Lightweight cross-attention-style refinement for 2D feature maps.

    For 2D inputs we approximate the SKC idea with shared 1x1 convolutions on
    pooled foreground/background features, then project residuals back to the
    original resolution.
    """

    def __init__(
        self,
        channels: int,
        attn_shape: Tuple[int, int] = (8, 8),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_shape = attn_shape
        self.resize_mode = "bilinear"
        hidden = max(channels // 2, 8)

        self.fg_mlp = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1),
            nn.InstanceNorm2d(hidden, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.bg_mlp = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1),
            nn.InstanceNorm2d(hidden, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(
        self,
        fg_feature: torch.Tensor,
        bg_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        original_size = fg_feature.shape[-2:]
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


class CVBMArgumentWithSKC2D(nn.Module):
    """2D CVBM_Argument backbone with the SKC module between encoder and decoders."""

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 2,
        n_filters: Tuple[int, ...] = (16, 32, 64, 128, 256),
        has_dropout: bool = False,
    ) -> None:
        super().__init__()
        params = {
            "in_chns": n_channels,
            "feature_chns": list(n_filters),
            "dropout": [0.05, 0.1, 0.2, 0.3, 0.5] if has_dropout else [0.0] * 5,
            "class_num": n_classes,
            "up_type": 1,
            "acti_func": "relu",
        }
        params_bg = params.copy()
        params_bg["up_type"] = 2

        self.encoder = Encoder(params)
        self.decoder_fg = Decoder(params)
        self.decoder_bg = Decoder(params_bg)
        self.skc = SemanticKnowledgeComplementarity2D(channels=n_filters[-1])
        self.final_seg = nn.Conv2d(n_classes * 2, n_classes, kernel_size=1)

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
