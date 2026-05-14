from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class MomentumBranchPrototypeLoss(nn.Module):
    """Branch-aware prototype contrast with EMA-updated prototype memory."""

    def __init__(
        self,
        in_channels: int,
        proj_dim: int = 32,
        num_classes: int = 2,
        temperature: float = 0.2,
        confidence_threshold: float = 0.8,
        query_threshold: float = 0.0,
        momentum: float = 0.99,
        patch_size: Tuple[int, int, int] = (8, 8, 8),
        max_queries: int = 4096,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not 0 <= momentum < 1:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        self.num_classes = num_classes
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.query_threshold = query_threshold
        self.momentum = momentum
        self.patch_size = patch_size
        self.max_queries = max_queries
        self.fg_projector = nn.Conv3d(in_channels, proj_dim, kernel_size=1, bias=False)
        self.bg_projector = nn.Conv3d(in_channels, proj_dim, kernel_size=1, bias=False)

        self.register_buffer("fg_prototypes", torch.zeros(num_classes, proj_dim))
        self.register_buffer("bg_prototypes", torch.zeros(num_classes, proj_dim))
        self.register_buffer("fg_initialized", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("bg_initialized", torch.zeros(num_classes, dtype=torch.bool))

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
        z_fg = F.normalize(self.fg_projector(feat_fg), p=2, dim=1)
        z_bg = F.normalize(self.bg_projector(feat_bg), p=2, dim=1)

        with torch.no_grad():
            fg_updates, fg_update_mask = self._build_batch_prototypes(z_fg.detach(), labels_fg, confidence_fg)
            bg_updates, bg_update_mask = self._build_batch_prototypes(z_bg.detach(), labels_bg, confidence_bg)
            self._momentum_update(self.fg_prototypes, self.fg_initialized, fg_updates, fg_update_mask)
            self._momentum_update(self.bg_prototypes, self.bg_initialized, bg_updates, bg_update_mask)

        fg_loss, fg_count = self._branch_loss(z_fg, labels_fg, confidence_fg, self.fg_prototypes, self.fg_initialized)
        bg_loss, bg_count = self._branch_loss(z_bg, labels_bg, confidence_bg, self.bg_prototypes, self.bg_initialized)

        valid = int(fg_count > 0) + int(bg_count > 0)
        if valid == 0:
            loss = (z_fg.mean() + z_bg.mean()) * 0.0
        else:
            loss = (fg_loss + bg_loss) / valid

        stats = {
            "proto_fg_queries": float(fg_count),
            "proto_bg_queries": float(bg_count),
            "proto_fg_ready": float(self.fg_initialized.sum().item()),
            "proto_bg_ready": float(self.bg_initialized.sum().item()),
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

    def _branch_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        confidence: torch.Tensor,
        prototypes: torch.Tensor,
        initialized: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        if initialized.sum().item() < 2:
            return features.mean() * 0.0, 0

        flat_features = features.permute(0, 2, 3, 4, 1).reshape(-1, features.shape[1])
        flat_labels = labels.reshape(-1).long()
        flat_conf = confidence.reshape(-1)

        query_mask = initialized[flat_labels] & (flat_conf >= self.query_threshold)
        query_idx = query_mask.nonzero(as_tuple=False).squeeze(1)
        if query_idx.numel() == 0:
            return features.mean() * 0.0, 0
        if self.max_queries > 0 and query_idx.numel() > self.max_queries:
            perm = torch.randperm(query_idx.numel(), device=query_idx.device)[:self.max_queries]
            query_idx = query_idx[perm]

        queries = flat_features[query_idx]
        targets = flat_labels[query_idx]
        logits = torch.mm(queries, prototypes.t()) / self.temperature
        logits[:, ~initialized] = -1e4
        return F.cross_entropy(logits, targets), int(query_idx.numel())

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
        update_mask = torch.zeros(self.num_classes, device=features.device, dtype=torch.bool)
        for class_idx in range(self.num_classes):
            class_mask = (flat_labels == class_idx) & (flat_conf >= self.confidence_threshold)
            if class_mask.any():
                prototypes[class_idx] = flat_features[class_mask].mean(dim=0)
                update_mask[class_idx] = True

        prototypes = F.normalize(prototypes, p=2, dim=-1)
        return prototypes, update_mask

    def _momentum_update(
        self,
        prototypes: torch.Tensor,
        initialized: torch.Tensor,
        updates: torch.Tensor,
        update_mask: torch.Tensor,
    ) -> None:
        for class_idx in range(self.num_classes):
            if not update_mask[class_idx]:
                continue
            if initialized[class_idx]:
                prototypes[class_idx].mul_(self.momentum).add_(updates[class_idx], alpha=1.0 - self.momentum)
            else:
                prototypes[class_idx].copy_(updates[class_idx])
                initialized[class_idx] = True
            prototypes[class_idx].copy_(F.normalize(prototypes[class_idx], p=2, dim=0))
