from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class BidirectionalCrossAttention(nn.Module):
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


class LocalCrossAttentionBlock3D(nn.Module):
    """Cross-attention inside local non-overlapping 3D windows."""

    def __init__(
        self,
        channels: int,
        window_size: Tuple[int, int, int] = (2, 2, 2),
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.cross_attn = BidirectionalCrossAttention(
            channels=channels,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.fg_proj = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.InstanceNorm3d(channels, affine=True),
            nn.GELU(),
        )
        self.bg_proj = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.InstanceNorm3d(channels, affine=True),
            nn.GELU(),
        )

    def _window_partition(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int]]:
        b, c, d, h, w = x.shape
        wd, wh, ww = self.window_size
        assert d % wd == 0 and h % wh == 0 and w % ww == 0, "Feature map must be divisible by window_size"

        x = x.view(b, c, d // wd, wd, h // wh, wh, w // ww, ww)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        tokens = x.view(-1, wd * wh * ww, c)
        return tokens, (b, c, d, h, w)

    def _window_reverse(self, tokens: torch.Tensor, shape_meta: Tuple[int, int, int, int, int]) -> torch.Tensor:
        b, c, d, h, w = shape_meta
        wd, wh, ww = self.window_size
        num_d, num_h, num_w = d // wd, h // wh, w // ww

        x = tokens.view(b, num_d, num_h, num_w, wd, wh, ww, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        return x.view(b, c, d, h, w)

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


class GlobalCrossAttentionBlock3D(nn.Module):
    """Cross-attention over the full bottleneck token sequence."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cross_attn = BidirectionalCrossAttention(
            channels=channels,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.fg_proj = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.InstanceNorm3d(channels, affine=True),
            nn.GELU(),
        )
        self.bg_proj = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.InstanceNorm3d(channels, affine=True),
            nn.GELU(),
        )

    def forward(
        self,
        fg_feature: torch.Tensor,
        bg_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, d, h, w = fg_feature.shape

        fg_tokens = fg_feature.flatten(2).transpose(1, 2).contiguous()
        bg_tokens = bg_feature.flatten(2).transpose(1, 2).contiguous()

        fg_tokens, bg_tokens = self.cross_attn(fg_tokens, bg_tokens)

        fg_feature = fg_tokens.transpose(1, 2).contiguous().view(b, c, d, h, w)
        bg_feature = bg_tokens.transpose(1, 2).contiguous().view(b, c, d, h, w)

        fg_feature = self.fg_proj(fg_feature)
        bg_feature = self.bg_proj(bg_feature)
        return fg_feature, bg_feature
