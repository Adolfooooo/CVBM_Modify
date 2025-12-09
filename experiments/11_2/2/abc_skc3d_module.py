"""SKC-enhanced CVBM model with ABC auxiliary heads for fg/bg decoders."""

import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

from networks.CVBM import Encoder, Upsampling_function, ConvBlock, ResidualConvBlock

_SKC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "11_1"))
if _SKC_PATH not in sys.path:
    sys.path.append(_SKC_PATH)
from skc3d_module.skc_model import SemanticKnowledgeComplementarity3D


class ABCDecoder(nn.Module):
    """CVBM decoder augmented with an auxiliary ABC head."""

    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none',
                 has_dropout=False, has_residual=False, up_type=0):
        super().__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)
        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)
        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)
        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)
        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)

        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.out_conv_abc = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.out_conv_tanh = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

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

        out_tanh_logits = self.out_conv_tanh(x9)
        out_tanh = self.tanh(out_tanh_logits)

        if self.has_dropout:
            x9 = self.dropout(x9)

        out_main = self.out_conv(x9)
        out_abc = self.out_conv_abc(x9)
        return out_main, out_abc, out_tanh


class CVBMArgumentWithSKC3DABC(nn.Module):
    """Two-stream CVBM backbone with SKC and ABC auxiliary heads."""

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
        self.decoder_fg = ABCDecoder(
            n_channels,
            n_classes,
            n_filters,
            normalization,
            has_dropout,
            has_residual,
            up_type=0,
        )
        self.decoder_bg = ABCDecoder(
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

    def forward(self, input_fg, input_bg):
        fg_feats = list(self.encoder(input_fg))
        bg_feats = list(self.encoder(input_bg))

        fg_feats[-1], bg_feats[-1] = self.skc(fg_feats[-1], bg_feats[-1])

        out_fg, out_fg_abc, tanh_fg = self.decoder_fg(fg_feats)
        out_bg, out_bg_abc, tanh_bg = self.decoder_bg(bg_feats)
        fused_logits = self.final_seg(torch.cat([out_fg, out_bg], dim=1))

        return out_fg, out_fg_abc, fused_logits, out_bg, out_bg_abc, tanh_fg, tanh_bg
