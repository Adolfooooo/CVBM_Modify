from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import Dropout, Softmax, LayerNorm
from einops import rearrange


class TokenLearner_Local(nn.Module):
    def __init__(self, img_size=(4, 8, 8), patch_size=(2, 2, 2), in_chans=1, embed_dim=8):
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


class Self_Attention_Local(nn.Module):
    def __init__(self, num_heads, embedding_channels, attention_dropout_rate, patch_size=(2, 2, 2)):
        super().__init__()
        self.KV_size = embedding_channels * num_heads
        self.patch_size = patch_size
        self.embedding_channels = embedding_channels
        self.num_heads = num_heads
        self.attention_head_size = embedding_channels
        self.q = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.k = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.v = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)

        self.psi = nn.InstanceNorm2d(self.num_heads)
        self.softmax = Softmax(dim=3)
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

        mh_q = rearrange(
            q,
            '(b d h w) (p1 p2 p3) (c heads) -> (b d h w) heads c (p1 p2 p3)',
            p1=2, p2=2, p3=2, d=2, h=4, w=4, c=self.embedding_channels, heads=self.num_heads,
        )
        mh_k = rearrange(
            k,
            '(b d h w) (p1 p2 p3) (c heads) -> (b d h w) heads (p1 p2 p3) c',
            p1=2, p2=2, p3=2, d=2, h=4, w=4, c=self.embedding_channels, heads=self.num_heads,
        )
        mh_v = rearrange(
            v,
            '(b d h w) (p1 p2 p3) (c heads) -> (b d h w) heads c (p1 p2 p3)',
            p1=2, p2=2, p3=2, d=2, h=4, w=4, c=self.embedding_channels, heads=self.num_heads,
        )

        self_attn = torch.matmul(mh_q, mh_k)
        self_attn = self.attn_dropout(self.softmax(self.psi(self_attn)))
        self_attn = torch.matmul(self_attn, mh_v)

        self_attn = rearrange(
            self_attn.squeeze(1),
            '(b d h w) heads c (p1 p2 p3) -> b (d p1 h p2 w p3) (c heads)',
            p1=2, p2=2, p3=2, d=2, h=4, w=4, c=self.embedding_channels, heads=self.num_heads,
        )
        out = self.out(self_attn)
        return out


class Self_Attention_Local_Block(nn.Module):
    def __init__(self, num_heads, embedding_channels, attention_dropout_rate):
        super().__init__()
        self.token_learner = TokenLearner_Local(img_size=(4, 8, 8), patch_size=(2, 2, 2), in_chans=1, embed_dim=8)
        self.attn_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.attn = Self_Attention_Local(num_heads, embedding_channels, attention_dropout_rate)
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

        res = rearrange(x, '(b c) (d h w) (p1 p2 p3) -> b (d p1 h p2 w p3) c', b=b, c=c, d=2, h=4, w=4, p1=2, p2=2, p3=2)
        x = rearrange(x, '(b c) (d h w) (p1 p2 p3) -> (b d h w) (p1 p2 p3) c', b=b, c=c, d=2, h=4, w=4, p1=2, p2=2, p3=2)

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
