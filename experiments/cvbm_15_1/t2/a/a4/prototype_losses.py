from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class BranchBatchPrototypeLoss(nn.Module):
    """Sample-positive, batch-negative prototype contrast over decoder features."""

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
        self.fg_projector = nn.Conv3d(in_channels, proj_dim, kernel_size=1, bias=False)
        self.bg_projector = nn.Conv3d(in_channels, proj_dim, kernel_size=1, bias=False)

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

        fg_loss, fg_count, fg_proto_count = self._branch_loss(z_fg, labels_fg, confidence_fg)
        bg_loss, bg_count, bg_proto_count = self._branch_loss(z_bg, labels_bg, confidence_bg)

        valid = int(fg_count > 0) + int(bg_count > 0)
        if valid == 0:
            loss = (z_fg.mean() + z_bg.mean()) * 0.0
        else:
            loss = (fg_loss + bg_loss) / valid

        stats = {
            "proto_fg_queries": float(fg_count),
            "proto_bg_queries": float(bg_count),
            "proto_fg_count": float(fg_proto_count),
            "proto_bg_count": float(bg_proto_count),
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
    ) -> Tuple[torch.Tensor, int, int]:
        prototypes, prototype_labels, prototype_sample_ids = self._build_sample_prototypes(features, labels, confidence)
        prototype_count = prototypes.shape[0]
        if prototype_count < 2 or prototype_labels.unique().numel() < 2:
            return features.mean() * 0.0, 0, int(prototype_count)

        batch_size = features.shape[0]
        flat_features = features.permute(0, 2, 3, 4, 1).reshape(-1, features.shape[1])
        flat_labels = labels.reshape(-1).long()
        flat_conf = confidence.reshape(-1)
        flat_sample_ids = torch.arange(batch_size, device=features.device).view(
            batch_size, 1, 1, 1
        ).expand_as(labels).reshape(-1)

        valid_query_class = torch.zeros_like(flat_labels, dtype=torch.bool)
        for class_idx in prototype_labels.unique():
            valid_query_class |= flat_labels == class_idx

        query_mask = valid_query_class & (flat_conf >= self.query_threshold)
        query_idx = query_mask.nonzero(as_tuple=False).squeeze(1)
        if query_idx.numel() == 0:
            return features.mean() * 0.0, 0, int(prototype_count)
        if self.max_queries > 0 and query_idx.numel() > self.max_queries:
            perm = torch.randperm(query_idx.numel(), device=query_idx.device)[:self.max_queries]
            query_idx = query_idx[perm]

        queries = flat_features[query_idx]
        targets = flat_labels[query_idx]
        query_sample_ids = flat_sample_ids[query_idx]
        logits = torch.mm(queries, prototypes.t()) / self.temperature
        same_class_mask = targets.unsqueeze(1).eq(prototype_labels.unsqueeze(0))
        same_sample_mask = query_sample_ids.unsqueeze(1).eq(prototype_sample_ids.unsqueeze(0))
        positive_mask = same_class_mask & same_sample_mask
        valid_queries = positive_mask.any(dim=1)
        if not valid_queries.any():
            return features.mean() * 0.0, 0, int(prototype_count)

        logits = logits[valid_queries]
        same_class_mask = same_class_mask[valid_queries]
        positive_mask = positive_mask[valid_queries]
        denominator_mask = positive_mask | ~same_class_mask
        log_pos = torch.logsumexp(logits.masked_fill(~positive_mask, -1e4), dim=1)
        log_all = torch.logsumexp(logits.masked_fill(~denominator_mask, -1e4), dim=1)
        loss = -(log_pos - log_all).mean()
        return loss, int(valid_queries.sum().item()), int(prototype_count)

    def _build_sample_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, channels = features.shape[:2]
        sample_features = features.permute(0, 2, 3, 4, 1).contiguous()

        prototypes = []
        prototype_labels = []
        prototype_sample_ids = []
        for batch_idx in range(batch_size):
            flat_features = sample_features[batch_idx].reshape(-1, channels)
            flat_labels = labels[batch_idx].reshape(-1).long()
            flat_conf = confidence[batch_idx].reshape(-1)

            for class_idx in range(self.num_classes):
                class_mask = (flat_labels == class_idx) & (flat_conf >= self.confidence_threshold)
                if class_mask.any():
                    prototypes.append(flat_features[class_mask].mean(dim=0))
                    prototype_labels.append(class_idx)
                    prototype_sample_ids.append(batch_idx)

        if not prototypes:
            return (
                features.new_zeros(0, channels),
                labels.new_zeros(0, dtype=torch.long),
                labels.new_zeros(0, dtype=torch.long),
            )

        prototypes = F.normalize(torch.stack(prototypes, dim=0), p=2, dim=-1)
        prototype_labels = torch.tensor(prototype_labels, device=features.device, dtype=torch.long)
        prototype_sample_ids = torch.tensor(prototype_sample_ids, device=features.device, dtype=torch.long)
        return prototypes, prototype_labels, prototype_sample_ids
