# # -*- coding: utf-8 -*-
# # bhc2d.py
# # 2D Background-aware Hard-negative Contrast (BHC)
# # 设定：两视图 = 同一 cutmix 样本的 弱增强(前景分支输出) vs 强增强(背景分支输出)
# # 锚(正) = output_mix 的低置信 patch；负 = 弱/强两视图在“背景通道”上均高置信的 patch
# # 损失 = 仅推远（cos margin hinge），不含正向吸引项、不做边界剔除

# from typing import Tuple, Optional, Dict, List
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # ---------- 基础工具 ----------

# def softmax_confidence(logits: torch.Tensor, dim: int = 1) -> torch.Tensor:
#     """
#     根据 logits 计算像素级不确定度 u = 1 - max_prob
#     Args:
#         logits: [B, C, H, W]
#     Returns:
#         u: [B, 1, H, W], 值域 [0,1]
#     """
#     prob = F.softmax(logits, dim=dim)
#     max_prob, _ = prob.max(dim=dim, keepdim=True)
#     u = 1.0 - max_prob
#     return u


# def bg_prob_from_logits(
#     logits: torch.Tensor,
#     bg_index: int = 0,
#     assume_multiclass: Optional[bool] = None
# ) -> torch.Tensor:
#     """
#     从 logits 提取“背景通道”的概率图 p_bg
#     支持两种输出形态：
#       - 多类 softmax（常见）：C>=2，直接取 softmax(...)[bg_index]
#       - 二类但只有一个前景通道（sigmoid）：C==1，则 p_bg = 1 - sigmoid(logits)
#     Args:
#         logits: [B, C, H, W]
#         bg_index: 背景通道索引（多类时使用）
#         assume_multiclass: 显式指定是否多类；None 时根据 C 自动判断
#     Returns:
#         p_bg: [B, 1, H, W] 背景概率
#     """
#     B, C, H, W = logits.shape
#     if assume_multiclass is None:
#         assume_multiclass = (C >= 2)
#     if assume_multiclass:
#         prob = F.softmax(logits, dim=1)
#         p_bg = prob[:, bg_index:bg_index+1]
#     else:
#         # 单通道前景：背景 = 1 - 前景
#         p_fg = torch.sigmoid(logits)
#         p_bg = 1.0 - p_fg
#     return p_bg


# def patchify_mean(
#     x: torch.Tensor,
#     patch_size: Tuple[int, int] = (8, 8),
#     stride: Tuple[int, int] = (8, 8)
# ) -> Tuple[torch.Tensor, int]:
#     """
#     将 [B, D, H, W] 切成 patch，并对每个 patch 做均值池化，返回向量。
#     Returns:
#         vecs: [B, N, D],  N=patch 数
#         N: patch 数
#     """
#     B, D, H, W = x.shape
#     ph, pw = patch_size
#     sh, sw = stride
#     unf = F.unfold(x, kernel_size=(ph, pw), stride=(sh, sw))  # [B, D*ph*pw, N]
#     N = unf.shape[-1]
#     vecs = unf.view(B, D, ph*pw, N).mean(dim=2).permute(0, 2, 1).contiguous()  # [B, N, D]
#     return vecs, N


# # ---------- 投影头 ----------

# class ProjectionHead(nn.Module):
#     """
#     将 mix logits（或 mix 中间特征）投影到对比空间（维度 D）
#     通常 in_ch = num_classes（直接对 logits 做 1x1）
#     """
#     def __init__(self, in_ch: int, out_dim: int = 128, gn_groups: int = 8):
#         super().__init__()
#         self.proj = nn.Sequential(
#             nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
#             nn.GroupNorm(max(1, min(gn_groups, in_ch)), in_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_ch, out_dim, kernel_size=1, bias=False),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B, C, H, W] -> [B, D, H, W]
#         return self.proj(x)


# # ---------- 负样本掩膜（弱视图 vs 强视图 的“背景高置信交集”） ----------

# def build_bg_consensus_mask_from_pair(
#     weak_fg_logits: torch.Tensor,   # 弱增强：前景分支输出（同一 cutmix 样本）
#     strong_bg_logits: torch.Tensor, # 强增强：背景分支输出（同一 cutmix 样本）
#     tau_weak: float = 0.75,         # 弱视图背景概率阈值
#     tau_strong: float = 0.80,       # 强视图背景概率阈值
#     bg_index: int = 0,
#     assume_multiclass: Optional[bool] = None
# ) -> torch.Tensor:
#     """
#     对“同一 cutmix 样本”的弱/强两路输出，构造像素级背景负样本掩膜：
#       neg_mask_pair = [p_bg(weak) > tau_weak] AND [p_bg(strong) > tau_strong]
#     Args:
#         weak_fg_logits:  [B,C,H,W] （来自 out_*_fg）
#         strong_bg_logits:[B,C,H,W] （来自 out_*_bg）
#     Returns:
#         neg_mask_pair: [B,1,H,W]，float{0,1}
#     """
#     p_bg_weak   = bg_prob_from_logits(weak_fg_logits,   bg_index, assume_multiclass)
#     p_bg_strong = bg_prob_from_logits(strong_bg_logits, bg_index, assume_multiclass)
#     m_weak   = (p_bg_weak   > tau_weak).float()
#     m_strong = (p_bg_strong > tau_strong).float()
#     neg_mask_pair = m_weak * (1-m_strong)
#     return neg_mask_pair


# def build_bg_consensus_mask_batch(
#     weak_list: List[torch.Tensor],     # [out_unl_fg, out_l_fg, ...]
#     strong_list: List[torch.Tensor],   # [out_unl_bg, out_l_bg, ...]，与 weak_list 一一对应
#     tau_weak: float = 0.75,
#     tau_strong: float = 0.80,
#     bg_index: int = 0,
#     assume_multiclass: Optional[bool] = None
# ) -> torch.Tensor:
#     """
#     针对一个 batch 内的多个 cutmix 样本对，构造并拼接像素级背景负样本掩膜。
#     Returns:
#         neg_mask: [B_total, 1, H, W]，按样本对顺序 cat 到一起
#     """
#     assert len(weak_list) == len(strong_list), "weak_list 与 strong_list 数量不一致"
#     masks = []
#     for weak_logits, strong_logits in zip(weak_list, strong_list):
#         masks.append(
#             build_bg_consensus_mask_from_pair(
#                 weak_logits, strong_logits,
#                 tau_weak=tau_weak, tau_strong=tau_strong,
#                 bg_index=bg_index, assume_multiclass=assume_multiclass
#             )
#         )
#     neg_mask = torch.cat(masks, dim=0)
#     return neg_mask  # [B_total,1,H,W]


# # ---------- patch 级锚/负选择 ----------

# def select_bhc_patches(
#     q_m: torch.Tensor,                 # [B,C,H,W]（mix logits）
#     neg_mask: torch.Tensor,            # [B,1,H,W]（像素级背景负样本掩膜）
#     topk: int = 4,
#     patch_size: Tuple[int, int] = (8, 8),
#     stride: Tuple[int, int] = (8, 8),
#     tau_patch: float = 0.70            # patch 级“背景纯度”阈值
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     从 q_m 选 Top-k 低置信 patch 作为锚；从 neg_mask 选“背景纯度>=tau_patch”的 patch 作为负样本。
#     Returns:
#         idx_anchor: [B,k]（每图锚 patch 索引）
#         idx_neg:    [B,Nn]（每图负 patch 索引；按最长补齐，不足用 -1）
#         u_patch:    [B,N]（每 patch 不确定度，供锚权重等使用）
#     """
#     # 不确定度 u_patch
#     u = softmax_confidence(q_m)                       # [B,1,H,W]
#     u_patch, N = patchify_mean(u, patch_size, stride) # [B,N,1]
#     u_patch = u_patch[..., 0]                         # [B,N]

#     # 背景纯度（patch 级）
#     neg_patch, _ = patchify_mean(neg_mask, patch_size, stride)  # [B,N,1]
#     neg_patch = neg_patch[..., 0]                                # [B,N]
#     neg_valid = (neg_patch >= tau_patch)                         # [B,N] bool

#     # 每图 Top-k 低置信（=高不确定）为锚
#     k = max(1, min(topk, u_patch.shape[1]))
#     idx_anchor = torch.topk(u_patch, k=k, dim=1, largest=True).indices  # [B,k]

#     # 每图负样本索引（数量按最长补齐）
#     B = neg_valid.shape[0]
#     max_neg = neg_valid.sum(dim=1).max().item() if B > 0 else 0
#     if max_neg == 0:
#         # 返回空负样本；后续损失需处理“无负”的情况
#         empty = torch.full((B, 0), -1, device=q_m.device, dtype=torch.long)
#         return idx_anchor, empty, u_patch

#     idx_neg_list = []
#     for b in range(B):
#         idx_b = torch.nonzero(neg_valid[b], as_tuple=False).squeeze(1)  # [Nb]
#         if idx_b.numel() == 0:
#             pad = torch.full((max_neg,), -1, device=q_m.device, dtype=torch.long)
#             idx_neg_list.append(pad)
#         elif idx_b.numel() < max_neg:
#             pad = torch.full((max_neg - idx_b.numel(),), -1, device=q_m.device, dtype=torch.long)
#             idx_neg_list.append(torch.cat([idx_b, pad], dim=0))
#         else:
#             idx_neg_list.append(idx_b[:max_neg])
#     idx_neg = torch.stack(idx_neg_list, dim=0)  # [B,max_neg]
#     return idx_anchor, idx_neg, u_patch


# def gather_patch_vectors(
#     feat_mix: torch.Tensor,            # [B,D,H,W]（投影后的特征）
#     idx_anchor: torch.Tensor,          # [B,k]
#     idx_neg: torch.Tensor,             # [B,Nn]（可含 -1）
#     patch_size: Tuple[int, int] = (8, 8),
#     stride: Tuple[int, int] = (8, 8)
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     从 feat_mix 中按索引取出锚/负 patch 的向量（均值池化 + L2 归一化）。
#     Returns:
#         Z_a: [B,k,D]
#         Z_n: [B,Nn,D]
#         mask_n_valid: [B,Nn,1]（负样本有效位掩膜）
#     """
#     vecs, N = patchify_mean(feat_mix, patch_size, stride)  # [B,N,D]
#     B, _, D = vecs.shape

#     # 锚
#     Z_a = torch.gather(vecs, dim=1, index=idx_anchor.unsqueeze(-1).expand(-1, -1, D))  # [B,k,D]

#     # 负（处理 -1 索引）
#     mask_n_valid = (idx_neg >= 0).float().unsqueeze(-1)  # [B,Nn,1]
#     idx_safe = idx_neg.clamp(min=0)
#     Z_n = torch.gather(vecs, dim=1, index=idx_safe.unsqueeze(-1).expand(-1, -1, D))
#     Z_n = Z_n * mask_n_valid  # 无效位置零

#     # 归一化
#     Z_a = F.normalize(Z_a, dim=-1)
#     Z_n = F.normalize(Z_n, dim=-1)
#     return Z_a, Z_n, mask_n_valid


# # ---------- 仅推远（margin hinge） ----------

# def bhc_repulsive_loss(
#     Z_a: torch.Tensor,                 # [B,k,D]
#     Z_n: torch.Tensor,                 # [B,Nn,D]
#     mask_n_valid: torch.Tensor,        # [B,Nn,1]
#     anchor_weights: Optional[torch.Tensor] = None,  # [B,k] or None（可用锚的不确定度）
#     margin: float = 0.2,
#     temperature: float = 0.07
# ) -> Tuple[torch.Tensor, Dict[str, float]]:
#     """
#     对每个锚 a，最小化 mean_n ReLU( cos(a,n)/T - margin )，仅推远、不含吸引项
#     Returns:
#         loss: 标量
#         stats: {'num_anchors':..., 'num_neg':...}
#     """
#     B, k, D = Z_a.shape
#     _, Nn, _ = Z_n.shape

#     # 有效负样本检测
#     valid_n_count = mask_n_valid.sum().item()
#     if valid_n_count == 0 or k == 0:
#         zero = Z_a.mean() * 0.0
#         return zero, {"num_anchors": 0, "num_neg": 0}

#     losses = []
#     valid_anchor_cnt = 0
#     for b in range(B):
#         # [k,D] x [D,Nn] -> [k,Nn]
#         sim = (Z_a[b] @ Z_n[b].transpose(0, 1)) / temperature
#         # 只对有效负样本计算
#         m_b = mask_n_valid[b].squeeze(-1)  # [Nn]
#         if m_b.sum() == 0:
#             continue
#         sim = sim[:, m_b.bool()]  # [k, Nb_valid]
#         l = F.relu(sim - margin).mean(dim=1)  # [k]
#         if anchor_weights is not None:
#             l = l * anchor_weights[b]
#         losses.append(l.mean())
#         valid_anchor_cnt += k

#     if len(losses) == 0:
#         zero = Z_a.mean() * 0.0
#         return zero, {"num_anchors": 0, "num_neg": 0}

#     loss = torch.stack(losses).mean()
#     return loss, {"num_anchors": valid_anchor_cnt, "num_neg": int(valid_n_count)}


# # ---------- 锚权重（用不确定度） ----------

# def anchor_weights_from_uncertainty(
#     u_patch: torch.Tensor,             # [B,N]
#     idx_anchor: torch.Tensor,          # [B,k]
#     gamma: float = 1.0
# ) -> torch.Tensor:
#     """
#     取出锚对应的 patch 不确定度并做幂次放大，作为锚权重（聚焦困难样本）。
#     Returns:
#         w: [B,k]
#     """
#     w = torch.gather(u_patch, dim=1, index=idx_anchor)  # [B,k]
#     if gamma != 1.0:
#         w = w.pow(gamma)
#     return w



# -*- coding: utf-8 -*-
# bhc2d_realign.py
# 2D BHC（语义对齐版）：弱=前景分支（用“非前景高置信”），强=背景分支（用“背景高置信”）
# 锚：q_m 的低置信 patch； 对比特征：融合前 [fg; bg] 的 1x1 投影； 仅推远（无正向/无边界剔除）

from typing import Tuple, Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 基础 ----------

def softmax_confidence(logits: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """像素级不确定度 u = 1 - max_prob；返回 [B,1,H,W]"""
    prob = F.softmax(logits, dim=dim)
    max_prob, _ = prob.max(dim=dim, keepdim=True)
    return 1.0 - max_prob


def prob_not_foreground_weak(
    weak_logits: torch.Tensor,
    bg_index: Optional[int] = None,
    fg_indices: Optional[List[int]] = None,
    assume_multiclass: Optional[bool] = None
) -> torch.Tensor:
    """
    弱视图（前景分支）的“非前景概率”：
      - 多类(softmax)：1 - max_{c in fg_indices} p[c]；若未显式给 fg_indices，则默认非 bg_index 的全部通道为前景类
      - 单通道(sigmoid 前景)：1 - sigmoid(fg_logit)
    返回 [B,1,H,W]
    """
    B, C, H, W = weak_logits.shape
    if assume_multiclass is None:
        assume_multiclass = (C >= 2)
    if assume_multiclass:
        prob = F.softmax(weak_logits, dim=1)  # [B,C,H,W]
        if fg_indices is None:
            assert bg_index is not None, "多类情况下未提供 fg_indices，请至少提供 bg_index。"
            fg_indices = [i for i in range(C) if i != bg_index]
        if len(fg_indices) == 0:
            # 极端：没有前景类，非前景=1
            return torch.ones((B,1,H,W), device=weak_logits.device, dtype=weak_logits.dtype)
        p_fg_max, _ = prob[:, fg_indices].max(dim=1, keepdim=True)  # [B,1,H,W]
        return 1.0 - p_fg_max
    else:
        # 单前景通道
        p_fg = torch.sigmoid(weak_logits)
        return 1.0 - p_fg


def bg_prob_from_strong(
    strong_logits: torch.Tensor,
    bg_index: int = 0,
    assume_multiclass: Optional[bool] = None
) -> torch.Tensor:
    """
    强视图（背景分支）的“背景概率”：
      - 多类(softmax)：p[bg_index]
      - 单通道(sigmoid 前景)：1 - sigmoid(fg)
    返回 [B,1,H,W]
    """
    # B, C, H, W = strong_logits.shape
    # if assume_multiclass is None:
    #     assume_multiclass = (C >= 2)
    # if assume_multiclass:
    #     prob = F.softmax(strong_logits, dim=1)
    #     return prob[:, bg_index:bg_index+1]
    # else:
    #     p_fg = torch.sigmoid(strong_logits)
    #     return 1.0 - p_fg
    B, C, H, W = strong_logits.shape
    if assume_multiclass is None:
        assume_multiclass = (C >= 2)
    bg_indices=[]
    if assume_multiclass:
        prob = F.softmax(strong_logits, dim=1)  # [B,C,H,W]
        if bg_indices is None:
            assert bg_index is not None, "多类情况下未提供 fg_indices，请至少提供 bg_index。"
            bg_indices = [i for i in range(C) if i != bg_index]
        if len(bg_indices) == 0:
            # 极端：没有前景类，非前景=1
            return torch.ones((B,1,H,W), device=strong_logits.device, dtype=strong_logits.dtype)
        p_bg_max, _ = prob[:, bg_indices].max(dim=1, keepdim=True)  # [B,1,H,W]
        return p_bg_max
    else:
        # 单前景通道
        p_bg = torch.sigmoid(strong_logits)
        return 1.0 - p_bg


def patchify_mean(
    x: torch.Tensor,
    patch_size: Tuple[int,int] = (8,8),
    stride: Tuple[int,int] = (8,8)
) -> Tuple[torch.Tensor, int]:
    """[B,D,H,W] -> [B,N,D]（每 patch 平均），返回向量与 N=patch 数"""
    B, D, H, W = x.shape
    ph, pw = patch_size; sh, sw = stride
    unf = F.unfold(x, kernel_size=(ph,pw), stride=(sh,sw))  # [B, D*ph*pw, N]
    N = unf.shape[-1]
    vecs = unf.view(B, D, ph*pw, N).mean(dim=2).permute(0,2,1).contiguous()  # [B,N,D]
    return vecs, N


# ---------- 投影头（对比特征用：融合前 [fg; bg] -> D） ----------
class ProjectionHead(nn.Module):
    """对 concat([out_fg, out_bg], dim=1) 做 1x1 投影到对比空间"""
    def __init__(self, in_ch: int, out_dim: int = 128, gn_groups: int = 8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            nn.GroupNorm(max(1, min(gn_groups, in_ch)), in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_dim, 1, bias=False)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)  # [B,D,H,W]


# ---------- 负样本像素掩膜（弱=非前景高置信 ∧ 强=背景高置信） ----------
def build_neg_mask_weak_strong_pair(
    weak_fg_logits: torch.Tensor,     # out_*_fg（弱：前景分支）
    strong_bg_logits: torch.Tensor,   # out_*_bg（强：背景分支）
    tau_notfg: float = 0.7,
    tau_bg: float = 0.8,
    bg_index: int = 0,
    fg_indices: Optional[List[int]] = None,
    weak_is_multiclass: Optional[bool] = None,
    strong_is_multiclass: Optional[bool] = None
) -> torch.Tensor:
    """
    对单对 cutmix 样本构造负样本像素掩膜：
      mask = [p_not_fg(弱) > tau_notfg] ∧ [p_bg(强) > tau_bg]
    返回 [B,1,H,W]
    """
    p_not_fg = prob_not_foreground_weak(
        weak_fg_logits, bg_index=bg_index, fg_indices=fg_indices,
        assume_multiclass=weak_is_multiclass
    )
    p_bg_strong = bg_prob_from_strong(
        strong_bg_logits, bg_index=bg_index, assume_multiclass=strong_is_multiclass
    )
    m_weak = (p_not_fg > tau_notfg).float()
    m_strong = (p_bg_strong > tau_bg).float()
    print()
    print(f"m_weak:{m_weak.sum()}")
    print(f"m_strong:{m_strong.sum()}")
    return m_weak * m_strong


def build_neg_mask_batch(
    weak_list: List[torch.Tensor],
    strong_list: List[torch.Tensor],
    tau_notfg: float = 0.7,
    tau_bg: float = 0.8,
    bg_index: int = 0,
    fg_indices: Optional[List[int]] = None,
    weak_is_multiclass: Optional[bool] = None,
    strong_is_multiclass: Optional[bool] = None
) -> torch.Tensor:
    """对 batch 内多对样本逐对构造并 cat：返回 [B_total,1,H,W]"""
    assert len(weak_list) == len(strong_list)
    masks = []
    for w, s in zip(weak_list, strong_list):
        masks.append(build_neg_mask_weak_strong_pair(
            w, s, tau_notfg, tau_bg, bg_index, fg_indices,
            weak_is_multiclass, strong_is_multiclass
        ))
    return torch.cat(masks, dim=0)


# ---------- patch 级锚/负选择、向量提取、损失 ----------
def select_bhc_patches(
    q_m: torch.Tensor,          # [B,C,H,W]（最终融合 logits）
    neg_mask: torch.Tensor,     # [B,1,H,W]（负像素掩膜）
    topk: int = 4,
    patch_size: Tuple[int,int] = (8,8),
    stride: Tuple[int,int] = (8,8),
    tau_patch: float = 0.7
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    锚：q_m 的 Top-k 低置信 patch；负：neg_mask patch 纯度 >= tau_patch
    返回：idx_anchor [B,k]，idx_neg [B,Nn]（不足用-1），u_patch [B,N]
    """
    u = softmax_confidence(q_m)                          # [B,1,H,W]
    u_patch, N = patchify_mean(u, patch_size, stride)    # [B,N,1]
    u_patch = u_patch[...,0]                             # [B,N]

    neg_patch, _ = patchify_mean(neg_mask, patch_size, stride)  # [B,N,1]
    neg_patch = neg_patch[...,0]                                  # [B,N]
    neg_valid = (neg_patch >= tau_patch)                          # [B,N]

    k = max(1, min(topk, u_patch.shape[1]))
    idx_anchor = torch.topk(u_patch, k=k, dim=1, largest=True).indices  # [B,k]

    B = neg_valid.shape[0]
    max_neg = neg_valid.sum(dim=1).max().item() if B > 0 else 0
    if max_neg == 0:
        empty = torch.full((B,0), -1, dtype=torch.long, device=q_m.device)
        return idx_anchor, empty, u_patch

    idx_list = []
    for b in range(B):
        idx_b = torch.nonzero(neg_valid[b], as_tuple=False).squeeze(1)
        if idx_b.numel() == 0:
            pad = torch.full((max_neg,), -1, dtype=torch.long, device=q_m.device)
            idx_list.append(pad)
        elif idx_b.numel() < max_neg:
            pad = torch.full((max_neg - idx_b.numel(),), -1, dtype=torch.long, device=q_m.device)
            idx_list.append(torch.cat([idx_b, pad], dim=0))
        else:
            idx_list.append(idx_b[:max_neg])
    idx_neg = torch.stack(idx_list, dim=0)  # [B,max_neg]
    return idx_anchor, idx_neg, u_patch


def gather_patch_vectors(
    feat: torch.Tensor,         # [B,D,H,W]（对比特征，来自融合前 [fg;bg] 的投影）
    idx_anchor: torch.Tensor,   # [B,k]
    idx_neg: torch.Tensor,      # [B,Nn]（可含 -1）
    patch_size: Tuple[int,int] = (8,8),
    stride: Tuple[int,int] = (8,8)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 Z_a [B,k,D]、Z_n [B,Nn,D]、mask_n_valid [B,Nn,1]"""
    vecs, N = patchify_mean(feat, patch_size, stride)  # [B,N,D]
    B, _, D = vecs.shape
    Z_a = torch.gather(vecs, dim=1, index=idx_anchor.unsqueeze(-1).expand(-1,-1,D))  # [B,k,D]
    mask_n_valid = (idx_neg >= 0).float().unsqueeze(-1)
    idx_safe = idx_neg.clamp(min=0)
    Z_n = torch.gather(vecs, dim=1, index=idx_safe.unsqueeze(-1).expand(-1,-1,D))
    Z_n = Z_n * mask_n_valid
    # L2
    Z_a = F.normalize(Z_a, dim=-1)
    Z_n = F.normalize(Z_n, dim=-1)
    return Z_a, Z_n, mask_n_valid


def bhc_repulsive_loss(
    Z_a: torch.Tensor,                 # [B,k,D]
    Z_n: torch.Tensor,                 # [B,Nn,D]
    mask_n_valid: torch.Tensor,        # [B,Nn,1]
    anchor_weights: Optional[torch.Tensor] = None,  # [B,k]（通常用锚的不确定度）
    margin: float = 0.2,
    temperature: float = 0.07
) -> Tuple[torch.Tensor, Dict[str,float]]:
    """仅推远：mean_n ReLU( cos/T - margin )；返回（loss, stats）"""
    B, k, D = Z_a.shape
    _, Nn, _ = Z_n.shape
    if mask_n_valid.sum().item() == 0 or k == 0:
        zero = Z_a.mean() * 0.0
        return zero, {"num_anchors": 0, "num_neg": 0}

    losses = []
    valid_anchor = 0
    for b in range(B):
        m = mask_n_valid[b].squeeze(-1)  # [Nn]
        if m.sum() == 0:
            continue
        sim = (Z_a[b] @ Z_n[b].transpose(0,1)) / temperature  # [k,Nn]
        sim = sim[:, m.bool()]                                 # [k, Nn_valid]
        l = F.relu(sim - margin).mean(dim=1)                   # [k]
        if anchor_weights is not None:
            l = l * anchor_weights[b]
        losses.append(l.mean())
        valid_anchor += k

    if not losses:
        zero = Z_a.mean() * 0.0
        return zero, {"num_anchors": 0, "num_neg": 0}

    loss = torch.stack(losses).mean()
    return loss, {"num_anchors": valid_anchor, "num_neg": int(mask_n_valid.sum().item())}


def anchor_weights_from_uncertainty(u_patch: torch.Tensor, idx_anchor: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """锚权重 w = u^gamma；u_patch [B,N] -> w [B,k]"""
    w = torch.gather(u_patch, dim=1, index=idx_anchor)
    return w.pow(gamma) if gamma != 1.0 else w
