import torch
from typing import Dict, Iterable, List, Tuple


class ConfidenceTracker:
    """
    Maintains per-case confidence heatmaps via exponential moving average.

    The tracker stores tensors on CPU in float16 to keep memory low, and provides
    smoothed confidence maps for the current mini-batch.
    """

    def __init__(self, momentum: float = 0.9, device: torch.device | None = None):
        """
        Args:
            momentum (float): EMA coefficient. New values contribute (1 - m).
            device (torch.device): Optional default device used when returning tensors.
        """
        self.momentum = momentum
        self.device = device
        self._storage: Dict[str, torch.Tensor] = {}

    def _to_cpu_half(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to("cpu", dtype=torch.float16)

    def _to_device(self, tensor: torch.Tensor, device: torch.device | None) -> torch.Tensor:
        out_device = device or self.device or tensor.device
        return tensor.to(out_device, dtype=torch.float32)

    def update(
        self,
        case_ids: Iterable[str],
        conf_maps: torch.Tensor,
        device: torch.device | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update EMA buffers and return smoothed confidence maps plus their per-sample mean.

        Args:
            case_ids (Iterable[str]): Case identifier for each element in the batch.
            conf_maps (Tensor): Shape [B, 1, H, W], current confidence heatmaps.
            device (torch.device, optional): Target device for returned tensors.

        Returns:
            smooth_maps (Tensor): Shape [B, 1, H, W], EMA confidence on requested device.
            mean_scores (Tensor): Shape [B], average confidence per sample.
        """
        assert conf_maps.ndim == 4, "conf_maps must be [B, 1, H, W]"

        smooth_list: List[torch.Tensor] = []
        mean_scores: List[float] = []

        for case_id, conf in zip(case_ids, conf_maps):
            stored = self._storage.get(case_id)
            if stored is None:
                updated = self._to_cpu_half(conf)
            else:
                stored = stored.to(dtype=torch.float32)
                updated = self.momentum * stored + (1.0 - self.momentum) * conf.detach().cpu()
                updated = updated.to(dtype=torch.float16)
            self._storage[case_id] = updated
            restored = self._to_device(updated, device).unsqueeze(0)
            smooth_list.append(restored)
            mean_scores.append(restored.mean().item())

        smooth_maps = torch.cat(smooth_list, dim=0)
        mean_tensor = torch.tensor(mean_scores, device=smooth_maps.device, dtype=torch.float32)
        return smooth_maps, mean_tensor
