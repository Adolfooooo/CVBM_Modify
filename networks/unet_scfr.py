import torch
import torch.nn as nn

from networks.unet import Encoder, Decoder


class SCFRModule(nn.Module):
    """
    Semantic-Guided Cross-View Feature Rectification (SCFR).
    Uses the foreground branch to guide background refinement before fusion.
    """

    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction)
        self.channel_squeeze = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.fg_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff_conv = nn.Sequential(
            nn.Conv2d(reduced_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.rectify_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_fg: torch.Tensor, x_bg: torch.Tensor) -> torch.Tensor:
        attention = self.fg_attention(x_fg)
        x_bg_filtered = x_bg * attention
        fg_compact = self.channel_squeeze(x_fg)
        bg_compact = self.channel_squeeze(x_bg_filtered)
        combined = torch.cat([fg_compact, bg_compact], dim=1)
        rectified = self.diff_conv(combined)
        rectified = self.rectify_conv(rectified)
        return x_bg + self.gamma * rectified


class CVBM2d_SCFRArgument(nn.Module):
    """
    CVBM2d variant that inserts the SCFR module before fusing FG and BG streams.
    """

    def __init__(self, in_chns: int, class_num: int):
        super().__init__()
        params = {
            'in_chns': in_chns,
            'feature_chns': [16, 32, 64, 128, 256],
            'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
            'class_num': class_num,
            'up_type': 1,
            'acti_func': 'relu'
        }
        params_bg = {
            **params,
            'up_type': 2
        }
        self.encoder = Encoder(params)
        self.decoder_fg = Decoder(params)
        self.decoder_bg = Decoder(params_bg)

        feat_dim = params['feature_chns'][0]
        self.scfr = SCFRModule(in_channels=feat_dim)
        self.final_seg = nn.Conv2d(feat_dim * 2, class_num, kernel_size=1)
        self.bg_head = nn.Conv2d(feat_dim, class_num, kernel_size=1)

    def forward(self, input_fg: torch.Tensor, input_bg: torch.Tensor):
        features_fg = self.encoder(input_fg)
        features_bg = self.encoder(input_bg)

        logits_fg, feat_fg = self.decoder_fg(features_fg)
        _, feat_bg = self.decoder_bg(features_bg)

        feat_bg_rectified = self.scfr(feat_fg, feat_bg)
        logits_bg = self.bg_head(feat_bg_rectified)
        fused_logits = self.final_seg(torch.cat((feat_fg, feat_bg_rectified), dim=1))

        return logits_fg, fused_logits, logits_bg, feat_fg, feat_bg_rectified
