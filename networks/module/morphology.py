import torch
import torch.nn.functional as F


def _binary_pool(input_mask: torch.Tensor, kernel_size: int, mode: str) -> torch.Tensor:
    """
    Internal helper performing dilation/erosion with max pooling.
    """
    padding = kernel_size // 2
    if mode == "dilate":
        return F.max_pool2d(input_mask, kernel_size, stride=1, padding=padding)
    if mode == "erode":
        neg = -input_mask
        eroded = -F.max_pool2d(neg, kernel_size, stride=1, padding=padding)
        return torch.clamp(eroded, min=0.0)
    raise ValueError(f"Unsupported mode {mode}")


def create_narrow_band(
    mask: torch.Tensor,
    inner_kernel: int = 3,
    outer_kernel: int = 7,
) -> torch.Tensor:
    """
    Generates a narrow band mask around the provided binary mask.

    Args:
        mask (Tensor): Shape [B, 1, H, W], binary front (values {0,1}).
        inner_kernel (int): Kernel for erosion (controls inner boundary).
        outer_kernel (int): Kernel for dilation (controls outer boundary).

    Returns:
        Tensor: Shape [B, 1, H, W] with values in {0,1} representing the band.
    """
    assert mask.ndim == 4, "mask needs to be [B, 1, H, W]"
    if mask.dtype != torch.float32:
        mask = mask.float()
    dilated = _binary_pool(mask, outer_kernel, mode="dilate")
    eroded = _binary_pool(mask, inner_kernel, mode="erode")
    band = torch.clamp(dilated - eroded, min=0.0, max=1.0)
    return band
