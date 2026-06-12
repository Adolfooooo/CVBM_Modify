from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class BranchBatchPrototypeLoss(nn.Module):
    """Branch prototype contrast with explicit fg-bg complementary alignment.

    This variant keeps the original branch-local prototype losses, then aligns
    foreground-branch prototypes with their complementary background-branch
    prototypes: fg class 0 <-> bg class 1 and fg class 1 <-> bg class 0.
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
        relation_weight: float = 0.2,
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
        self.relation_weight = relation_weight
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

        fg_loss, fg_count, fg_prototypes, fg_valid = self._branch_loss(z_fg, labels_fg, confidence_fg)
        bg_loss, bg_count, bg_prototypes, bg_valid = self._branch_loss(z_bg, labels_bg, confidence_bg)
        relation_loss, relation_pairs = self._prototype_relation_loss(
            fg_prototypes,
            bg_prototypes,
            fg_valid,
            bg_valid,
        )

        valid = int(fg_count > 0) + int(bg_count > 0)
        if valid == 0:
            loss = (z_fg.mean() + z_bg.mean()) * 0.0
        else:
            loss = (fg_loss + bg_loss) / valid
        if relation_pairs > 0:
            loss = loss + self.relation_weight * relation_loss

        stats = {
            "proto_fg_queries": float(fg_count),
            "proto_bg_queries": float(bg_count),
            "proto_relation_pairs": float(relation_pairs),
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
    ) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        prototypes, valid_classes = self._build_batch_prototypes(features, labels, confidence)
        if valid_classes.sum().item() < 2:
            return features.mean() * 0.0, 0, prototypes, valid_classes

        flat_features = features.permute(0, 2, 3, 4, 1).reshape(-1, features.shape[1])
        flat_labels = labels.reshape(-1).long()
        flat_conf = confidence.reshape(-1)

        query_mask = valid_classes[flat_labels] & (flat_conf >= self.query_threshold)
        query_idx = query_mask.nonzero(as_tuple=False).squeeze(1)
        if query_idx.numel() == 0:
            return features.mean() * 0.0, 0, prototypes, valid_classes
        if self.max_queries > 0 and query_idx.numel() > self.max_queries:
            perm = torch.randperm(query_idx.numel(), device=query_idx.device)[:self.max_queries]
            query_idx = query_idx[perm]

        queries = flat_features[query_idx]
        targets = flat_labels[query_idx]
        logits = torch.mm(queries, prototypes.t()) / self.temperature
        logits[:, ~valid_classes] = -1e4
        return F.cross_entropy(logits, targets), int(query_idx.numel()), prototypes, valid_classes

    def _prototype_relation_loss(
        self,
        fg_prototypes: torch.Tensor,
        bg_prototypes: torch.Tensor,
        fg_valid: torch.Tensor,
        bg_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        losses = []
        for fg_class in range(self.num_classes):
            bg_class = self.num_classes - 1 - fg_class
            if fg_valid[fg_class] and bg_valid[bg_class]:
                fg_logits = torch.mm(
                    fg_prototypes[fg_class:fg_class + 1],
                    bg_prototypes.t(),
                ) / self.temperature
                fg_logits[:, ~bg_valid] = -1e4
                losses.append(F.cross_entropy(
                    fg_logits,
                    torch.tensor([bg_class], device=fg_logits.device),
                ))

                bg_logits = torch.mm(
                    bg_prototypes[bg_class:bg_class + 1],
                    fg_prototypes.t(),
                ) / self.temperature
                bg_logits[:, ~fg_valid] = -1e4
                losses.append(F.cross_entropy(
                    bg_logits,
                    torch.tensor([fg_class], device=bg_logits.device),
                ))

        if not losses:
            return (fg_prototypes.mean() + bg_prototypes.mean()) * 0.0, 0
        return torch.stack(losses).mean(), len(losses)

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
