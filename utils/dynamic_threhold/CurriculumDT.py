import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ CDT (dimension-agnostic) ------------------
class CurriculumDynamicThresholdingND(nn.Module):
    """
    Works for logits of shape [B, C, ...spatial...].
    Implements Eq.(2)->(4)->(5) from the paper.
    Returns:
      delta: [B, ...]  bool mask of confident predictions
      T_c:   [C]       per-class dynamic thresholds
    """
    def __init__(self, tau: float = 0.6, eps: float = 1e-6, ignore_index: int = 255):
        super().__init__()
        self.tau = float(tau)
        self.eps = eps
        self.ignore_index = ignore_index

    @torch.no_grad()
    def _class_status(self, probs: torch.Tensor):
        C = probs.shape[1]
        conf, y_hat = probs.max(dim=1)                 # conf:[B,*], y_hat:[B,*]
        high = conf > self.tau                         # 1(p̂ > τ)
        idx = y_hat[high].reshape(-1)
        sigma = torch.bincount(idx, minlength=C).float() if idx.numel() else probs.new_zeros(C)
        sigma_hat = sigma / torch.clamp(sigma.max(), min=self.eps)
        return conf, y_hat, sigma_hat

    def forward(self, logits: torch.Tensor):
        probs = F.softmax(logits, dim=1)
        conf, y_hat, sigma_hat = self._class_status(probs)
        # Eq.(4): T_c = (σ̂_c / (2 - σ̂_c)) * τ
        T_c = (sigma_hat / (2.0 - sigma_hat.clamp(max=1.0))) * self.tau   # [C]
        T_map = T_c[y_hat]                                                # [B,*]
        delta = conf > T_map                                              # Eq.(5)
        return delta, T_c, y_hat

    @torch.no_grad()
    def make_pseudo(self, teacher_logits: torch.Tensor):
        """From teacher logits -> (pseudo, mask, T_c)."""
        mask, T_c, y_hat = self.forward(teacher_logits)
        pseudo = y_hat.clone()
        pseudo[~mask] = self.ignore_index
        return pseudo, mask, T_c