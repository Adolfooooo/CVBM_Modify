from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class BranchBatchPrototypeLoss(nn.Module):
    """Shared-space prototype contrast for foreground/background branches.

    Foreground and background decoder features are projected by the same
    projector and build one batch-local prototype set. Background labels are
    complemented before entering the shared semantic space, so both branches use
    the same foreground-object label convention.
    """

    def __init__(
        self,
        in_channels: int,
        proj_dim: int = 32,
        num_classes: int = 2,
        temperature: float = 0.2,
        confidence_threshold: float = 0.8,
        query_threshold: float = 0.0,
        patch_size: Tuple[int, int, int] = (8, 8, 8),
        max_queries: int = 4096,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.num_classes = num_classes
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.query_threshold = query_threshold
        self.patch_size = patch_size
        self.max_queries = max_queries
        self.shared_projector = nn.Conv3d(in_channels, proj_dim, kernel_size=1, bias=False)

    def forward(
        self,
        feat_fg: torch.Tensor,
        feat_bg: torch.Tensor,
        labels_fg: torch.Tensor,
        labels_bg: torch.Tensor,
        confidence_fg: torch.Tensor,
        confidence_bg: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        feat_fg, labels_fg, confidence_fg = self._pool_inputs(feat_fg, labels_fg, confidence_fg)
        feat_bg, labels_bg, confidence_bg = self._pool_inputs(feat_bg, labels_bg, confidence_bg)
        z_fg = F.normalize(self.shared_projector(feat_fg), p=2, dim=1)
        z_bg = F.normalize(self.shared_projector(feat_bg), p=2, dim=1)
        labels_bg_shared = (self.num_classes - 1 - labels_bg).long()

        shared_features = torch.cat([z_fg, z_bg], dim=0)
        shared_labels = torch.cat([labels_fg, labels_bg_shared], dim=0)
        shared_confidence = torch.cat([confidence_fg, confidence_bg], dim=0)
        loss, query_count, query_branch_counts = self._shared_loss(
            shared_features,
            shared_labels,
            shared_confidence,
            fg_batch_size=z_fg.shape[0],
        )
        if query_count == 0:
            loss = (z_fg.mean() + z_bg.mean()) * 0.0

        stats = {
            "proto_fg_queries": float(query_branch_counts[0]),
            "proto_bg_queries": float(query_branch_counts[1]),
        }
        return loss, stats

    def _pool_inputs(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled_features = F.avg_pool3d(features, kernel_size=self.patch_size, stride=self.patch_size)
        pooled_labels = F.avg_pool3d(
            labels.float().unsqueeze(1),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).squeeze(1)
        pooled_confidence = F.avg_pool3d(
            confidence.float().unsqueeze(1),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).squeeze(1)
        return pooled_features, (pooled_labels >= 0.5).long(), pooled_confidence

    def _shared_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        confidence: torch.Tensor,
        fg_batch_size: int,
    ) -> Tuple[torch.Tensor, int, Tuple[int, int]]:
        prototypes, valid_classes = self._build_batch_prototypes(features, labels, confidence)
        if valid_classes.sum().item() < 2:
            return features.mean() * 0.0, 0, (0, 0)

        flat_features = features.permute(0, 2, 3, 4, 1).reshape(-1, features.shape[1])
        flat_labels = labels.reshape(-1).long()
        flat_conf = confidence.reshape(-1)
        num_patches_per_sample = labels[0].numel()
        flat_sample_ids = torch.arange(
            labels.shape[0],
            device=labels.device,
        ).repeat_interleave(num_patches_per_sample)

        query_mask = valid_classes[flat_labels] & (flat_conf >= self.query_threshold)
        query_idx = query_mask.nonzero(as_tuple=False).squeeze(1)
        if query_idx.numel() == 0:
            return features.mean() * 0.0, 0, (0, 0)
        if self.max_queries > 0 and query_idx.numel() > self.max_queries:
            perm = torch.randperm(query_idx.numel(), device=query_idx.device)[:self.max_queries]
            query_idx = query_idx[perm]

        queries = flat_features[query_idx]
        targets = flat_labels[query_idx]
        logits = torch.mm(queries, prototypes.t()) / self.temperature
        logits[:, ~valid_classes] = -1e4
        query_samples = flat_sample_ids[query_idx]
        fg_count = int((query_samples < fg_batch_size).sum().item())
        bg_count = int((query_samples >= fg_batch_size).sum().item())
        return F.cross_entropy(logits, targets), int(query_idx.numel()), (fg_count, bg_count)

    def _build_batch_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        channels = features.shape[1]
        flat_features = features.permute(0, 2, 3, 4, 1).reshape(-1, channels)
        flat_labels = labels.reshape(-1).long()
        flat_conf = confidence.reshape(-1)

        prototypes = features.new_zeros(self.num_classes, channels)
        valid_classes = torch.zeros(self.num_classes, device=features.device, dtype=torch.bool)
        for class_idx in range(self.num_classes):
            class_mask = (flat_labels == class_idx) & (flat_conf >= self.confidence_threshold)
            if class_mask.any():
                prototypes[class_idx] = flat_features[class_mask].mean(dim=0)
                valid_classes[class_idx] = True

        prototypes = F.normalize(prototypes, p=2, dim=-1)
        return prototypes, valid_classes
