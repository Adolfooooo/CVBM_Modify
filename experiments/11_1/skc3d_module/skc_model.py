"""3D semantic knowledge complementarity models.

The module defined here decorates the existing CVBM_Argument architecture with
semantic knowledge complementarity (SKC) blocks that refine the shared encoder
features before they reach the task-specific decoders.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .self_attention import (
    Self_Attention_Global_Block,
    Self_Attention_Local_Block,
)
from networks.CVBM import Decoder, Encoder


class SemanticKnowledgeComplementarity3D(nn.Module):
    """Cross-attention refinement of encoder features for 3D volumes.

    Args:
        channels: Number of channels of the bottleneck feature map (C).
        attn_shape: Spatial size (D, H, W) required by the attention blocks.
        num_heads: Number of attention heads used in both local and global blocks.
        dropout: Dropout rate passed to the attention layers.

    Inputs:
        fg_feature: Tensor of shape ``[B, C, D, H, W]`` containing the weak/foreground
            branch features coming out of the shared encoder.
        bg_feature: Tensor of shape ``[B, C, D, H, W]`` containing the strong/background
            branch features (1 - target label) coming out of the encoder.

    Returns:
        Tuple[Tensor, Tensor]: A pair of tensors with the same shape as the inputs
        that correspond to the SKC-refined features for the foreground and background
        decoders respectively.
    """

    def __init__(
        self,
        channels: int,
        attn_shape: Tuple[int, int, int] = (4, 8, 8),
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_shape = attn_shape
        self.resize_mode = "trilinear"

        # Local channel-wise cross-attention + normalization + 1x1x1 convolution.
        self.local_block = Self_Attention_Local_Block(
            num_heads=num_heads,
            embedding_channels=channels,
            attention_dropout_rate=dropout,
        )
        self.local_norm = nn.InstanceNorm3d(channels, affine=True)
        self.local_conv = nn.Conv3d(channels, channels, kernel_size=1)

        # Global channel-wise cross-attention + normalization + 1x1x1 convolution.
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
        # ``fg_feature`` and ``bg_feature`` already share the same shape since they
        # stem from the same encoder; keep the original resolution for later.
        original_size = fg_feature.shape[-3:]
        combined = torch.cat([fg_feature, bg_feature], dim=0)

        # The attention blocks expect a fixed spatial resolution (4x8x8). We resize
        # the latent features to that window, run the SKC stack and project them
        # back to the original resolution afterwards.
        resized = F.interpolate(
            combined,
            size=self.attn_shape,
            mode=self.resize_mode,
            align_corners=False,
        )

        local_refined = self.local_block(resized)
        local_refined = self.local_conv(self.local_norm(local_refined))

        global_refined = self.global_block(local_refined)
        global_refined = self.global_conv(self.global_norm(global_refined))

        restored = F.interpolate(
            global_refined,
            size=original_size,
            mode=self.resize_mode,
            align_corners=False,
        )
        fg_delta, bg_delta = torch.chunk(restored, chunks=2, dim=0)

        # Residual connection ensures that the SKC block complements instead of
        # overwriting the encoder responses.
        return fg_feature + fg_delta, bg_feature + bg_delta


class CVBMArgumentWithSKC3D(nn.Module):
    """CVBM_Argument backbone with the SKC module between encoder and decoders.

    The architecture keeps the same shared encoder as :class:`CVBM_Argument`, but
    the highest-level features of both streams are refined through
    :class:`SemanticKnowledgeComplementarity3D` before being fed into the
    foreground (weak augmentation) and background (strong augmentation) decoders.

    Args:
        n_channels: Number of input channels (1 for the LA dataset).
        n_classes: Number of segmentation classes.
        n_filters: Base channel width used by the encoder/decoder pyramid.
        normalization: Normalization type passed to the building blocks.
        has_dropout: Whether to enable the decoder dropout that exists in CVBM.
        has_residual: Whether to use residual convolutions inside the blocks.
    """

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
        """Runs the two-stream segmentation backbone with SKC in between.

        Args:
            input_fg: Weak/foreground-augmented tensor ``[B, 1, D, H, W]``.
            input_bg: Strong/background-augmented tensor ``[B, 1, D, H, W]``.

        Returns:
            Tuple containing (foreground logits, fused logits, background logits,
            foreground attention logits, background attention logits).
        """

        fg_feats = list(self.encoder(input_fg))
        bg_feats = list(self.encoder(input_bg))

        # Insert SKC right before the decoder stage so both heads receive
        # complementary semantics.
        fg_feats[-1], bg_feats[-1] = self.skc(fg_feats[-1], bg_feats[-1])

        out_fg, attn_fg = self.decoder_fg(fg_feats)
        out_bg, attn_bg = self.decoder_bg(bg_feats)

        fused_logits = self.final_seg(torch.cat([out_fg, out_bg], dim=1))
        return out_fg, fused_logits, out_bg, attn_fg, attn_bg
