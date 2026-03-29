"""Foreground-aware refinement model for 3D CVBM experiments.

This module keeps the original foreground/background dual-view design and
adds a lightweight ROI proposal plus local refinement head on top of the
fused prediction. The second returned tensor is the refined segmentation
logits so the existing validation path can stay unchanged.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from networks.CVBM import Decoder, Encoder
from ..modules import SemanticKnowledgeComplementarity3D


class ConvNormAct3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RegionProposalHead3D(nn.Module):
    """Predicts a soft ROI mask from foreground/background/fused cues."""

    def __init__(self, in_channels: int = 6, hidden_channels: int = 16) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct3D(in_channels, hidden_channels),
            ConvNormAct3D(hidden_channels, hidden_channels),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, cues: torch.Tensor) -> torch.Tensor:
        return self.block(cues)


class RegionRefineHead3D(nn.Module):
    """Generates a residual refinement inside the proposed ROI."""

    def __init__(self, in_channels: int, hidden_channels: int = 16, n_classes: int = 2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct3D(in_channels, hidden_channels),
            ConvNormAct3D(hidden_channels, hidden_channels),
            nn.Conv3d(hidden_channels, n_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CVBMArgumentWithSKCForegroundRefine3D(nn.Module):
    """CVBM dual-view backbone with SKC and foreground-aware ROI refinement."""

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
        self.proposal_head = RegionProposalHead3D(in_channels=6, hidden_channels=n_filters)
        self.refine_head = RegionRefineHead3D(
            in_channels=n_classes * 3 + 1,
            hidden_channels=n_filters,
            n_classes=n_classes,
        )

    def _build_roi_cues(
        self,
        out_fg: torch.Tensor,
        fused_logits: torch.Tensor,
        out_bg: torch.Tensor,
        attn_fg: torch.Tensor,
        attn_bg: torch.Tensor,
    ) -> torch.Tensor:
        fg_prob = F.softmax(out_fg, dim=1)[:, 1:2]
        fused_prob = F.softmax(fused_logits, dim=1)[:, 1:2]
        bg_fg_prob = F.softmax(out_bg, dim=1)[:, 0:1]
        agreement = 0.5 * (fg_prob + bg_fg_prob)
        fg_bg_disagreement = torch.abs(fg_prob - bg_fg_prob)
        fg_fuse_disagreement = torch.abs(fg_prob - fused_prob)
        boundary_cue = torch.abs(torch.tanh(attn_fg[:, 1:2]) - torch.tanh(attn_bg[:, 0:1]))
        return torch.cat(
            [
                fg_prob,
                fused_prob,
                bg_fg_prob,
                agreement,
                fg_bg_disagreement,
                boundary_cue + fg_fuse_disagreement,
            ],
            dim=1,
        )

    def forward(
        self,
        input_fg: torch.Tensor,
        input_bg: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        fg_feats = list(self.encoder(input_fg))
        bg_feats = list(self.encoder(input_bg))
        fg_feats[-1], bg_feats[-1] = self.skc(fg_feats[-1], bg_feats[-1])

        out_fg, attn_fg = self.decoder_fg(fg_feats)
        out_bg, attn_bg = self.decoder_bg(bg_feats)
        fused_logits = self.final_seg(torch.cat([out_fg, out_bg], dim=1))

        roi_cues = self._build_roi_cues(out_fg, fused_logits, out_bg, attn_fg, attn_bg)
        roi_logits = self.proposal_head(roi_cues)
        roi_mask = torch.sigmoid(roi_logits)

        refine_input = torch.cat([out_fg, fused_logits, out_bg, roi_mask], dim=1)
        refine_delta = self.refine_head(refine_input)
        refined_logits = fused_logits + roi_mask * refine_delta

        return (
            out_fg,
            refined_logits,
            out_bg,
            attn_fg,
            attn_bg,
            fused_logits,
            roi_logits,
            roi_mask,
            refine_delta,
        )
