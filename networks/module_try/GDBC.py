'''
Global Dual-Branch Consistency Regularization (GDBC)

- Probability-weighted Global Pooling
- Consistency & Separation Regularization
- Strategy A (optional): build weights by downsampling class probs to feature size
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

def prob_weighted_global_pool(feat, prob, eps=1e-6):
    """
    feat: (B, C, H, W)
    prob: (B, 1, H, W)  in [0,1]
    return: (B, C)
    """
    B, C, H, W = feat.shape
    wsum = prob.sum(dim=(2, 3)) + eps            # (B, 1)
    num  = (feat * prob).sum(dim=(2, 3))         # (B, C)
    return num / wsum.clamp_min(eps)             # (B, C)

class GDBCLoss(nn.Module):
    """
    Global Dual-Branch Consistency Regularization
    - L_cv  : pull together (consistency)
    - L_sep : push apart (separation) with cosine margin hinge

    Strategy A (single-scale, optional):
      If use_strategyA=True, we assume p_fg (or teacher_p_fg when use_teacher_probs=True)
      is a high-res class probability map with shape (B, C_cls, H_hr, W_hr).
      We downsample it to feats' spatial size to build W_fg/W_bg, then reuse the
      original pipeline (probability-weighted pooling + L_cv/L_sep).
    """
    def __init__(
        self,
        lambda_cv=0.1,
        lambda_sep=0.1,
        cos_margin=0.0,         # target upper bound for cosine similarity (<= margin)
        warmup_steps=1000,
        use_projection=False,
        proj_dim=128,
        eps=1e-6,
        use_teacher_probs=False,
        conf_threshold=None,     # e.g., 0.2 to ignore ultra-low-confidence pixels (optional)

        # ---- Strategy A switches & params (new) ----
        use_strategyA=False,     # turn on single-scale downsampled-prob weighting
        fg_classes=(1, 2, 3),         # indices of foreground classes (semantic mode)
        bg_class=0,              # background class index; if None => 1 - sum(fg)
        downsample_mode="area"   # 'area' (recommended) or 'bilinear'
    ):
        super().__init__()
        self.lambda_cv = float(lambda_cv)
        self.lambda_sep = float(lambda_sep)
        self.cos_margin = float(cos_margin)
        self.warmup_steps = int(warmup_steps)
        self.eps = float(eps)
        self.use_teacher_probs = bool(use_teacher_probs)
        self.conf_threshold = conf_threshold

        self.use_strategyA = bool(use_strategyA)
        self.fg_classes = tuple(fg_classes) if fg_classes is not None else tuple()
        self.bg_class = None if bg_class is None else int(bg_class)
        self.downsample_mode = str(downsample_mode)

        self.proj = None
        if use_projection:
            # simple projection head for stability
            self.proj = nn.Sequential(
                nn.Linear(in_features=None, out_features=None)  # placeholder, set at runtime
            )

        # 延迟设置：等第一次前向拿到 C 后再构建 MLP
        self._proj_inited = False
        self._proj_dim = int(proj_dim) if use_projection else None
        self.use_projection = use_projection

    def _maybe_init_proj(self, C, device):
        if not self.use_projection or self._proj_inited:
            return
        self.proj = nn.Sequential(
            nn.Linear(C, self._proj_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self._proj_dim, self._proj_dim, bias=True)
        ).to(device)
        self._proj_inited = True

    def _gate_probs(self, p):
        if self.conf_threshold is None:
            return p
        return p * (p >= self.conf_threshold).float()
    
    def _process_soft_label(self, prediction:torch.Tensor, is_soft_label):
        """
        输入 prediction: (B, C, H, W)
        - is_soft_label=True  : 使用每像素最大类概率作为权重 (B,1,H,W)
        - is_soft_label=False : 仅在前景通道内取最大 (忽略显式背景), 作为权重 (B,1,H,W)
        """
        if is_soft_label:
            probs, _ = torch.max(prediction, dim=1, keepdim=True)   # (B,1,H,W)
        else:
            fore_prediction = prediction[:, 1:-1, ...] if prediction.size(1) > 2 else prediction[:, 1:2, ...]
            probs, _ = torch.max(fore_prediction, dim=1, keepdim=True)  # (B,1,H,W) within FG channels
        return probs  # (B,1,H,W)

    # ---- Strategy A helper ----
    def _downsample_to(self, tensor, size_hw, mode="area"):
        if mode == "area":
            return F.interpolate(tensor, size=size_hw, mode="area")
        else:
            return F.interpolate(tensor, size=size_hw, mode="bilinear", align_corners=False)

    def _build_weights_single_scale(self, probs_hr, target_hw):
        """
        probs_hr: (B, C_cls, H_hr, W_hr)  -- softmax probs over classes (detached at callsite)
        return:   W_fg, W_bg with shape (B,1,H_t,W_t)
        """
        assert probs_hr.dim() == 4, "Strategy A expects class probabilities of shape (B,C_cls,H,W)."
        # 语义导向: 前景类求和作为 W_fg；背景类用 bg_class 或 1 - W_fg
        W_fg_hr = probs_hr[:, self.fg_classes, :, :].sum(dim=1, keepdim=True) if len(self.fg_classes) > 0 else probs_hr[:, :0, :, :].sum(dim=1, keepdim=True)
        if self.bg_class is not None and self.bg_class < probs_hr.size(1):
            W_bg_hr = probs_hr[:, self.bg_class:self.bg_class+1, :, :]
        else:
            W_bg_hr = (1.0 - W_fg_hr).clamp_min(0.0)

        # 下采样对齐到特征分辨率
        W_fg = self._downsample_to(W_fg_hr, target_hw, mode=self.downsample_mode)
        W_bg = self._downsample_to(W_bg_hr, target_hw, mode=self.downsample_mode)

        # 归一化 & 可选阈值
        pair_sum = (W_fg + W_bg).clamp_min(self.eps)
        W_fg = W_fg / pair_sum
        W_bg = W_bg / pair_sum

        if self.conf_threshold is not None:
            thr = float(self.conf_threshold)
            W_fg = W_fg * (W_fg >= thr).float()
            W_bg = W_bg * (W_bg >= thr).float()

        return W_fg.detach(), W_bg.detach()

    def forward(
        self,
        feats_fg, feats_bg,      # (B, C, H, W) encoder feature (same spatial size)
        p_fg, p_bg,              # 默认: (B, C, H, W) 作为预测；当 use_strategyA=True 时，p_fg 可传 (B, C_cls, H_hr, W_hr)
        global_step:int=None,
        teacher_p_fg=None,       # use_strategyA=True 时可传 teacher 的类概率 (B, C_cls, H_hr, W_hr)
        teacher_p_bg=None,
        is_soft_label=False,     # 默认分支的权重构造开关（非策略A）
    ):
        """
        返回:
          dict(loss=..., l_cv=..., l_sep=..., g_fg=..., g_bg=...)
        使用方法: 将该 loss 加到总损失中即可。
        """
        device = feats_fg.device
        B, C, H, W = feats_fg.shape
        self._maybe_init_proj(C, device)

        # ------------------ Strategy A branch ------------------
        # 选择使用 teacher 或 student 的“高分辨率类概率”
        probs_hr = teacher_p_fg if (self.use_teacher_probs and (teacher_p_fg is not None)) else p_fg
        # 若误传了 logits，这里请在调用处先 softmax；此处默认已是概率
        # probs_hr = F.softmax(probs_hr, dim=1)

        # 将高分辨率类概率下采样并构造 (W_fg, W_bg) 到 feats 的空间尺寸
        W_fg, W_bg = self._build_weights_single_scale(probs_hr.detach(), target_hw=(H, W))

        # 使用下采样得到的权重做全局池化
        g_fg = prob_weighted_global_pool(feats_fg, W_fg, eps=self.eps)   # (B, C)
        g_bg = prob_weighted_global_pool(feats_bg, W_bg, eps=self.eps)   # (B, C)

        # optional projection
        if self.use_projection:
            g_fg = self.proj(g_fg)
            g_bg = self.proj(g_bg)

        # unit-norm for stability
        g_fg = F.normalize(g_fg, dim=1, eps=self.eps)
        g_bg = F.normalize(g_bg, dim=1, eps=self.eps)

        # consistency loss: pull together
        l_cv = F.mse_loss(g_fg, g_bg)

        # separation loss: push apart (cosine should be <= margin)
        cos_sim = (g_fg * g_bg).sum(dim=1)                  # (B,)
        hinge = F.relu(cos_sim - self.cos_margin)           # penalize when too similar
        l_sep = hinge.mean()

        # warmup: gradually enable L_sep
        if global_step is None or self.warmup_steps <= 0:
            sep_scale = 1.0
        else:
            sep_scale = float(min(1.0, max(0.0, (global_step - 0) / max(1, self.warmup_steps))))

        loss = self.lambda_cv * l_cv + (self.lambda_sep * sep_scale) * l_sep

        return {
            "loss": loss,
            "l_cv": l_cv.detach(),
            "l_sep": l_sep.detach(),
            "cos_sim_mean": cos_sim.mean().detach(),
            "g_fg": g_fg.detach(),
            "g_bg": g_bg.detach(),
            "sep_scale": torch.tensor(sep_scale, device=device)
        }
