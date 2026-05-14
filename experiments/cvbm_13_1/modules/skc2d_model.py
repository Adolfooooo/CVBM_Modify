from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from networks.unet import Encoder, Decoder


class BidirectionalCrossAttention2D(nn.Module):
    """Bidirectional cross-attention between foreground and background tokens."""

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.fg_q_norm = nn.LayerNorm(channels)
        self.bg_q_norm = nn.LayerNorm(channels)
        self.fg_kv_norm = nn.LayerNorm(channels)
        self.bg_kv_norm = nn.LayerNorm(channels)

        self.fg_to_bg_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.bg_to_fg_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        hidden_dim = channels * 4
        self.fg_ffn_norm = nn.LayerNorm(channels)
        self.bg_ffn_norm = nn.LayerNorm(channels)
        self.fg_ffn = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels),
            nn.Dropout(dropout),
        )
        self.bg_ffn = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        fg_tokens: torch.Tensor,
        bg_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fg_query = self.fg_q_norm(fg_tokens)
        bg_query = self.bg_q_norm(bg_tokens)
        fg_context = self.fg_kv_norm(fg_tokens)
        bg_context = self.bg_kv_norm(bg_tokens)

        fg_delta, _ = self.fg_to_bg_attn(
            query=fg_query,
            key=bg_context,
            value=bg_context,
            need_weights=False,
        )
        bg_delta, _ = self.bg_to_fg_attn(
            query=bg_query,
            key=fg_context,
            value=fg_context,
            need_weights=False,
        )

        fg_tokens = fg_tokens + fg_delta
        bg_tokens = bg_tokens + bg_delta

        fg_tokens = fg_tokens + self.fg_ffn(self.fg_ffn_norm(fg_tokens))
        bg_tokens = bg_tokens + self.bg_ffn(self.bg_ffn_norm(bg_tokens))
        return fg_tokens, bg_tokens


class LocalCrossAttentionBlock2D(nn.Module):
    """Cross-attention inside local non-overlapping 2D windows."""

    def __init__(
        self,
        channels: int,
        window_size: Tuple[int, int] = (2, 2),
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.cross_attn = BidirectionalCrossAttention2D(
            channels=channels,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.fg_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.GELU(),
        )
        self.bg_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.GELU(),
        )

    def _window_partition(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        b, c, h, w = x.shape
        wh, ww = self.window_size
        assert h % wh == 0 and w % ww == 0, "Feature map must be divisible by window_size"

        x = x.view(b, c, h // wh, wh, w // ww, ww)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        tokens = x.view(-1, wh * ww, c)
        return tokens, (b, c, h, w)

    def _window_reverse(self, tokens: torch.Tensor, shape_meta: Tuple[int, int, int, int]) -> torch.Tensor:
        b, c, h, w = shape_meta
        wh, ww = self.window_size
        num_h, num_w = h // wh, w // ww

        x = tokens.view(b, num_h, num_w, wh, ww, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, c, h, w)

    def forward(
        self,
        fg_feature: torch.Tensor,
        bg_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fg_tokens, meta = self._window_partition(fg_feature)
        bg_tokens, _ = self._window_partition(bg_feature)

        fg_tokens, bg_tokens = self.cross_attn(fg_tokens, bg_tokens)

        fg_feature = self._window_reverse(fg_tokens, meta)
        bg_feature = self._window_reverse(bg_tokens, meta)

        fg_feature = self.fg_proj(fg_feature)
        bg_feature = self.bg_proj(bg_feature)
        return fg_feature, bg_feature


class GlobalCrossAttentionBlock2D(nn.Module):
    """Cross-attention over the full 2D bottleneck token sequence."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cross_attn = BidirectionalCrossAttention2D(
            channels=channels,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.fg_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.GELU(),
        )
        self.bg_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.GELU(),
        )

    def forward(
        self,
        fg_feature: torch.Tensor,
        bg_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = fg_feature.shape

        fg_tokens = fg_feature.flatten(2).transpose(1, 2).contiguous()
        bg_tokens = bg_feature.flatten(2).transpose(1, 2).contiguous()

        fg_tokens, bg_tokens = self.cross_attn(fg_tokens, bg_tokens)

        fg_feature = fg_tokens.transpose(1, 2).contiguous().view(b, c, h, w)
        bg_feature = bg_tokens.transpose(1, 2).contiguous().view(b, c, h, w)

        fg_feature = self.fg_proj(fg_feature)
        bg_feature = self.bg_proj(bg_feature)
        return fg_feature, bg_feature


class SemanticKnowledgeCrossInteraction2D(nn.Module):
    """Bidirectional fg-bg interaction module at the bottleneck feature level."""

    def __init__(
        self,
        channels: int,
        attn_shape: Tuple[int, int] = (8, 8),
        local_window: Tuple[int, int] = (2, 2),
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_shape = attn_shape
        self.resize_mode = "bilinear"

        self.local_block = LocalCrossAttentionBlock2D(
            channels=channels,
            window_size=local_window,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.global_block = GlobalCrossAttentionBlock2D(
            channels=channels,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.fg_out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.GELU(),
        )
        self.bg_out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.GELU(),
        )

    def forward(
        self,
        fg_feature: torch.Tensor,
        bg_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        original_size = fg_feature.shape[-2:]

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


class CVBMArgumentWithCrossSKC2D(nn.Module):
    """2D CVBM_Argument with explicit bidirectional fg-bg interaction at bottleneck."""

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 2,
        n_filters: Tuple[int, ...] = (16, 32, 64, 128, 256),
        has_dropout: bool = False,
        attn_shape: Tuple[int, int] = (8, 8),
        local_window: Tuple[int, int] = (2, 2),
        num_heads: int = 4,
        dropout: float = 0.0,
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
        self.skc = SemanticKnowledgeCrossInteraction2D(
            channels=n_filters[-1],
            attn_shape=attn_shape,
            local_window=local_window,
            num_heads=num_heads,
            dropout=dropout,
        )
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
