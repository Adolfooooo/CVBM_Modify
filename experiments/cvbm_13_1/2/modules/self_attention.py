from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import Dropout, Softmax, LayerNorm
from einops import rearrange


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


class TokenLearner_Global(nn.Module):
    def __init__(self, img_size=(4, 8, 8), patch_size=(4, 8, 8), in_chans=1, embed_dim=256):
        super().__init__()
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1] * (img_size[2] // patch_size[2]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.map_in = nn.Sequential(nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size), nn.GELU())

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class Self_Attention_Global(nn.Module):
    def __init__(self, num_heads, embedding_channels, attention_dropout_rate):
        super().__init__()
        self.KV_size = embedding_channels * num_heads

        self.num_heads = num_heads
        self.attention_head_size = embedding_channels
        self.q = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.k = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.v = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.softmax = Softmax(dim=3)
        self.psi = nn.InstanceNorm2d(self.num_heads)
        self.out = nn.Linear(embedding_channels * self.num_heads, embedding_channels, bias=False)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)

    def multi_head_rep(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, emb):
        emb_l, emb_u = torch.split(emb, emb.size(0) // 2, dim=0)
        _, n, c = emb_u.size()

        q = self.q(emb)
        k = self.k(emb)
        v = self.v(emb)

        mh_q = self.multi_head_rep(q).transpose(-1, -2)
        mh_k = self.multi_head_rep(k)
        mh_v = self.multi_head_rep(v).transpose(-1, -2)

        self_attn = torch.matmul(mh_q, mh_k)
        self_attn = self.attn_dropout(self.softmax(self.psi(self_attn)))
        self_attn = torch.matmul(self_attn, mh_v)

        self_attn = self_attn.permute(0, 3, 2, 1).contiguous()
        new_shape = self_attn.size()[:-2] + (self.KV_size,)
        self_attn = self_attn.view(*new_shape)

        out = self.out(self_attn)
        out = self.proj_dropout(out)
        return out


class Self_Attention_Global_Block(nn.Module):
    def __init__(self, num_heads, embedding_channels, attention_dropout_rate):
        super().__init__()
        self.token_learner = TokenLearner_Global(img_size=(4, 8, 8), patch_size=(4, 8, 8), in_chans=1, embed_dim=256)
        self.attn_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.attn = Self_Attention_Global(num_heads, embedding_channels, attention_dropout_rate)
        self.ffn_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.map_out = nn.Sequential(
            nn.Conv3d(embedding_channels, embedding_channels, kernel_size=1, padding=0),
            nn.GELU(),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        if not self.training:
            x = torch.cat((x, x))

        b, c, d, h, w = x.shape
        x = x.contiguous().view(b * c, d, h, w).unsqueeze(1)
        x = self.token_learner(x)
        x = rearrange(x, '(b c) 1 (d h w) -> b (d h w) c', b=b, c=c, d=d, h=h, w=w)

        res = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = x + res
        x = self.ffn_norm(x)

        b_tokens, n_patch, hidden = x.size()
        x = x.permute(0, 2, 1).contiguous().view(b_tokens, hidden, d, h, w)
        x = self.map_out(x)

        if not self.training:
            x = torch.split(x, x.size(0) // 2, dim=0)[0]

        return x
