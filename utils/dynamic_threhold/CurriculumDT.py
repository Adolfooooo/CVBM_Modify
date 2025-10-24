import torch
import torch.nn as nn
import torch.nn.functional as F

class CurriculumDynamicThresholding(nn.Module):
    """
    Dimension-agnostic CDT for semi-supervised segmentation.
    Works for logits of shape [B, C, ...spatial...] (2D/3D).
    Returns:
      delta: [B, ...spatial...]  bool mask of confident predictions
      T_c:   [C]                 per-class dynamic thresholds
    """
    def __init__(self, tau: float = 0.6, eps: float = 1e-6):
        super().__init__()
        self.tau = float(tau)
        self.eps = eps

    @torch.no_grad()
    def _learning_status(self, probs: torch.Tensor):
        # probs: [B, C, *S]
        C = probs.shape[1]
        conf, y_hat = probs.max(dim=1)           # conf:[B,*S], y_hat:[B,*S] in {0..C-1}
        high = conf > self.tau                   # high-confidence mask

        idx = y_hat[high].reshape(-1)            # predicted classes among high-conf pixels/voxels
        if idx.numel():
            sigma = torch.bincount(idx, minlength=C).float()
        else:
            sigma = probs.new_zeros(C)

        sigma_hat = sigma / torch.clamp(sigma.max(), min=self.eps)  # in [0,1]
        return conf, y_hat, sigma_hat

    def forward(self, logits: torch.Tensor):
        """
        logits: [B, C, ...spatial...]
        """
        probs = F.softmax(logits, dim=1)         # class probs
        conf, y_hat, sigma_hat = self._learning_status(probs)

        # Non-linear convex mapping (Eq.4): T_c = [σ̂_c / (2 - σ̂_c)] * τ
        T_c = (sigma_hat / (2.0 - sigma_hat.clamp(max=1.0))) * self.tau  # [C]

        # Per-location threshold by predicted class
        T_map = T_c[y_hat]                       # broadcast to [B, *S]
        delta = conf > T_map                     # Eq.(5)

        return delta, T_c
