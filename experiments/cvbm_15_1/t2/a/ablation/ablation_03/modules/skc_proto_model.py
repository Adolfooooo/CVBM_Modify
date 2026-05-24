from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from networks.CVBM import (
    ConvBlock,
    Decoder,
    DownsamplingConvBlock,
    Encoder,
    ResidualConvBlock,
    Upsampling_function,
)
from .self_attention import LocalCrossAttentionBlock3D, GlobalCrossAttentionBlock3D


class DecoderWithFeature(Decoder):
    def forward(self, features):
        x1, x2, x3, x4, x5 = features

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up)
        out_seg2 = self.out_conv2(x9)
        out_tanh = self.tanh(out_seg2)
        proto_feature = x9
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        return out_seg, out_tanh, proto_feature


class SemanticKnowledgeCrossInteraction3D(nn.Module):
    """Dual-branch self-attention refinement module at the bottleneck feature level."""

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
        self.global_block = GlobalCrossAttentionBlock3D(
            channels=channels,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.fg_out = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.InstanceNorm3d(channels, affine=True),
            nn.GELU(),
        )
        self.bg_out = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.InstanceNorm3d(channels, affine=True),
            nn.GELU(),
        )

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

        fg_refined, bg_refined = self.local_block(fg_resized, bg_resized)
        fg_refined, bg_refined = self.global_block(fg_refined, bg_refined)

        fg_refined = self.fg_out(fg_refined)
        bg_refined = self.bg_out(bg_refined)

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


class CVBMArgumentWithCrossSKC3DProto(nn.Module):
    """CVBM_Argument with cross-SKC and decoder pre-classifier features."""

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
        self.decoder_fg = DecoderWithFeature(
            n_channels,
            n_classes,
            n_filters,
            normalization,
            has_dropout,
            has_residual,
            up_type=0,
        )
        self.decoder_bg = DecoderWithFeature(
            n_channels,
            n_classes,
            n_filters,
            normalization,
            has_dropout,
            has_residual,
            up_type=2,
        )
        self.skc = SemanticKnowledgeCrossInteraction3D(
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fg_feats = list(self.encoder(input_fg))
        bg_feats = list(self.encoder(input_bg))

        fg_feats[-1], bg_feats[-1] = self.skc(fg_feats[-1], bg_feats[-1])

        out_fg, attn_fg, feat_fg = self.decoder_fg(fg_feats)
        out_bg, attn_bg, feat_bg = self.decoder_bg(bg_feats)

        fused_logits = self.final_seg(torch.cat([out_fg, out_bg], dim=1))
        return out_fg, fused_logits, out_bg, attn_fg, attn_bg, feat_fg, feat_bg
