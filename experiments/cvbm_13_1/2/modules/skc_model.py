from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from networks.CVBM import Decoder, Encoder
from .self_attention import LocalCrossAttentionBlock3D, Self_Attention_Global_Block


class SemanticKnowledgeLocalCrossInteraction3D(nn.Module):
    """Local cross-attention followed by original shared global refinement."""

    def __init__(
        self,
        channels: int,
        attn_shape: Tuple[int, int, int] = (4, 8, 8),
        local_window: Tuple[int, int, int] = (2, 2, 2),
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_shape = attn_shape
        self.resize_mode = "trilinear"

        self.local_block = LocalCrossAttentionBlock3D(
            channels=channels,
            window_size=local_window,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.global_block = Self_Attention_Global_Block(
            num_heads=num_heads,
            embedding_channels=channels,
            attention_dropout_rate=dropout,
        )
        self.global_norm = nn.InstanceNorm3d(channels, affine=True)
        self.global_conv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(
        self,
        fg_feature: torch.Tensor,
        bg_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        original_size = fg_feature.shape[-3:]

        fg_resized = F.interpolate(
            fg_feature,
            size=self.attn_shape,
            mode=self.resize_mode,
            align_corners=False,
        )
        bg_resized = F.interpolate(
            bg_feature,
            size=self.attn_shape,
            mode=self.resize_mode,
            align_corners=False,
        )

        fg_local, bg_local = self.local_block(fg_resized, bg_resized)
        combined = torch.cat([fg_local, bg_local], dim=0)

        global_refined = self.global_block(combined)
        global_refined = self.global_conv(self.global_norm(global_refined))
        fg_refined, bg_refined = torch.chunk(global_refined, chunks=2, dim=0)

        fg_delta = F.interpolate(
            fg_refined,
            size=original_size,
            mode=self.resize_mode,
            align_corners=False,
        )
        bg_delta = F.interpolate(
            bg_refined,
            size=original_size,
            mode=self.resize_mode,
            align_corners=False,
        )

        return fg_feature + fg_delta, bg_feature + bg_delta


class CVBMArgumentWithLocalCrossSKC3D(nn.Module):
    """CVBM_Argument with local cross-attention and original global refinement."""

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
        self.skc = SemanticKnowledgeLocalCrossInteraction3D(
            channels=n_filters * 16,
            attn_shape=(4, 8, 8),
            local_window=(2, 2, 2),
            num_heads=4,
            dropout=0.0,
        )
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
