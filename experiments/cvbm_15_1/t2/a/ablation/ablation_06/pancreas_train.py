import logging
import os
import random
import shutil
import sys
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.datasets_3d import Pancreas, TwoStreamBatchSampler, WeakStrongAugment3d

proto_branch_parser = argparse.ArgumentParser(add_help=False)
proto_branch_parser.add_argument("--proto_branch", type=str, default="fg", choices=["fg", "bg"])
proto_branch_args, remaining_argv = proto_branch_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv

from experiments.cvbm_15_1.t2.a import pancreas_train as pancreas_base
from networks.net_factory import net_factory
from utils import losses, test_3d_patch
from utils.BCP_utils import context_mask_pancreas, mix_loss, update_ema_variables


class SingleBranchPrototypeLoss(torch.nn.Module):
    """Single-branch prototype contrast for either foreground or background."""

    def __init__(
        self,
        in_channels,
        branch="fg",
        proj_dim=32,
        num_classes=2,
        temperature=0.2,
        confidence_threshold=0.8,
        query_threshold=0.0,
        patch_size=(8, 8, 8),
        max_queries=4096,
    ):
        super().__init__()
        if branch not in ("fg", "bg"):
            raise ValueError(f"branch must be 'fg' or 'bg', got {branch}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.branch = branch
        self.num_classes = num_classes
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.query_threshold = query_threshold
        self.patch_size = patch_size
        self.max_queries = max_queries
        self.projector = torch.nn.Conv3d(in_channels, proj_dim, kernel_size=1, bias=False)

    def forward(self, features, labels, confidence):
        features, labels, confidence = self._pool_inputs(features, labels, confidence)
        z = F.normalize(self.projector(features), p=2, dim=1)

        branch_loss, branch_count = self._branch_loss(z, labels, confidence)
        if branch_count == 0:
            loss = z.mean() * 0.0
        else:
            loss = branch_loss

        stats = {
            "proto_fg_queries": float(branch_count) if self.branch == "fg" else 0.0,
            "proto_bg_queries": float(branch_count) if self.branch == "bg" else 0.0,
        }
        return loss, stats

    def _pool_inputs(self, features, labels, confidence):
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

    def _branch_loss(self, features, labels, confidence):
        prototypes, valid_classes = self._build_batch_prototypes(features, labels, confidence)
        if valid_classes.sum().item() < 2:
            return features.mean() * 0.0, 0

        flat_features = features.permute(0, 2, 3, 4, 1).reshape(-1, features.shape[1])
        flat_labels = labels.reshape(-1).long()
        flat_conf = confidence.reshape(-1)

        query_mask = valid_classes[flat_labels] & (flat_conf >= self.query_threshold)
        query_idx = query_mask.nonzero(as_tuple=False).squeeze(1)
        if query_idx.numel() == 0:
            return features.mean() * 0.0, 0
        if self.max_queries > 0 and query_idx.numel() > self.max_queries:
            perm = torch.randperm(query_idx.numel(), device=query_idx.device)[:self.max_queries]
            query_idx = query_idx[perm]

        queries = flat_features[query_idx]
        targets = flat_labels[query_idx]
        logits = torch.mm(queries, prototypes.t()) / self.temperature
        logits[:, ~valid_classes] = -1e4
        return F.cross_entropy(logits, targets), int(query_idx.numel())

    def _build_batch_prototypes(self, features, labels, confidence):
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


args = pancreas_base.args
args.proto_branch = proto_branch_args.proto_branch
if args.exp == "CVBM_Pancreas":
    args.exp = "CVBM_Pancreas_Ablation_06_SingleProtoBranch"
if args.snapshot_path == "./results/CVBM_15_1_t2_a/1/":
    args.snapshot_path = "./results/CVBM_15_1_t2_a/ablation_06/1/"

train_data_path = args.root_path
patch_size = args.patch_size
num_classes = 2
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr


def pre_train(local_args, snapshot_path):
    model = net_factory(net_type=local_args.model, in_chns=1, class_num=num_classes, mode="train")
    db_train = Pancreas(
        base_dir=train_data_path,
        split="train",
        transform=transforms.Compose([
            WeakStrongAugment3d(local_args.patch_size, flag_rot=True)
        ]))
    labelnum = local_args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, local_args.max_samples))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs,
        unlabeled_idxs,
        local_args.batch_size,
        local_args.batch_size - local_args.labeled_bs,
    )
    sub_bs = int(local_args.labeled_bs / 2)

    def worker_init_fn(worker_id):
        random.seed(local_args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=local_args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        pin_memory_device="cuda",
        persistent_workers=True,
    )
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    dice_loss = losses.mask_DiceLoss(nclass=2)

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("%d itertations per epoch", len(trainloader))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            model.train()
            volume_batch = sampled_batch["image"][:local_args.labeled_bs].cuda()
            label_batch = sampled_batch["label"][:local_args.labeled_bs].cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:]

            volume_batch_strong = sampled_batch["image_strong"].cuda()
            label_batch_strong = sampled_batch["label_strong"].cuda()
            img_a_s, img_b_s = volume_batch_strong[:sub_bs], volume_batch_strong[sub_bs:local_args.labeled_bs]
            lab_a_s, lab_b_s = label_batch_strong[:sub_bs], label_batch_strong[sub_bs:local_args.labeled_bs]
            with torch.no_grad():
                img_mask, _ = context_mask_pancreas(img_a, local_args.mask_ratio)

            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)

            volume_batch_strong = img_a_s * img_mask + img_b_s * (1 - img_mask)
            label_batch_strong = lab_a_s * img_mask + lab_b_s * (1 - img_mask)

            outputs_fg, _, outputs_bg, _, _ = model(volume_batch, volume_batch_strong)
            loss_seg = 0
            loss_seg_dice = 0

            y2 = outputs_fg[:local_args.labeled_bs, ...]
            y_prob2 = F.softmax(y2, dim=1)
            loss_seg += F.cross_entropy(y2[:local_args.labeled_bs], (label_batch[:local_args.labeled_bs, ...] == 1).long())
            loss_seg_dice += dice_loss(y2, label_batch[:local_args.labeled_bs, ...] == 1)

            y_bg = outputs_bg[:local_args.labeled_bs, ...]
            loss_seg += F.cross_entropy(y_bg[:local_args.labeled_bs], (label_batch_strong[:local_args.labeled_bs, ...] == 0).long())
            loss_seg_dice += dice_loss(y_bg, label_batch_strong[:local_args.labeled_bs, ...] == 0)

            loss = (loss_seg + loss_seg_dice) / 2

            iter_num += 1
            writer.add_scalar("pre/loss_seg_dice", loss_seg_dice, iter_num)
            writer.add_scalar("pre/loss_seg", loss_seg, iter_num)
            writer.add_scalar("pre/loss_all", loss, iter_num)
            logging.info("y_prob2: %s, label_batch: %s", torch.argmax(y_prob2, dim=1).sum(), label_batch.sum())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info(
                "iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f",
                iter_num,
                loss,
                loss_seg_dice,
                loss_seg,
            )

            if iter_num % 200 == 0 and torch.argmax(y_prob2, dim=1).sum() != 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_Pancreas_argument(
                    model,
                    num_classes=num_classes,
                    patch_size=patch_size,
                    stride_xy=16,
                    stride_z=16,
                    dataset_path=local_args.root_path,
                )
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path, f"iter_{iter_num}_dice_{best_dice}.pth")
                    save_best_path = os.path.join(snapshot_path, f"{local_args.model}_best_model.pth")
                    pancreas_base.save_net_opt(model, optimizer, save_mode_path)
                    pancreas_base.save_net_opt(model, optimizer, save_best_path)
                    logging.info("save best model to %s", save_mode_path)
                writer.add_scalar("4_Var_dice/Dice", dice_sample, iter_num)
                writer.add_scalar("4_Var_dice/Best_dice", best_dice, iter_num)

            if iter_num >= pre_max_iterations:
                break

        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(local_args, pre_snapshot_path, self_snapshot_path):
    model = pancreas_base.CVBMArgumentWithCrossSKC3DProto(
        n_channels=1,
        n_classes=num_classes,
        normalization="instancenorm",
        has_dropout=True,
    ).cuda()
    ema_model = pancreas_base.CVBMArgumentWithCrossSKC3DProto(
        n_channels=1,
        n_classes=num_classes,
        normalization="instancenorm",
        has_dropout=True,
    ).cuda()
    for param in ema_model.parameters():
        param.detach_()

    db_train = Pancreas(
        base_dir=train_data_path,
        split="train",
        transform=transforms.Compose([
            WeakStrongAugment3d(local_args.patch_size, flag_rot=True)
        ]))
    labelnum = local_args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, local_args.max_samples))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs,
        unlabeled_idxs,
        local_args.batch_size,
        local_args.batch_size - local_args.labeled_bs,
    )
    sub_bs = int(local_args.labeled_bs / 2)

    def worker_init_fn(worker_id):
        random.seed(local_args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=local_args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        pin_memory_device="cuda",
        persistent_workers=True,
    )
    proto_criterion = SingleBranchPrototypeLoss(
        in_channels=16,
        branch=local_args.proto_branch,
        proj_dim=local_args.proto_dim,
        num_classes=num_classes,
        temperature=local_args.proto_temperature,
        confidence_threshold=local_args.proto_conf_threshold,
        query_threshold=local_args.proto_query_threshold,
        patch_size=local_args.proto_patch,
        max_queries=local_args.proto_max_queries,
    ).cuda()
    optimizer = optim.SGD(
        list(model.parameters()) + list(proto_criterion.parameters()),
        lr=base_lr,
        momentum=0.9,
        weight_decay=0.0001,
    )

    pretrained_model = os.path.join(pre_snapshot_path, f"{local_args.model}_best_model.pth")
    pancreas_base.load_pretrained_backbone(model, pretrained_model)
    pancreas_base.load_pretrained_backbone(ema_model, pretrained_model)

    writer = SummaryWriter(self_snapshot_path + "/log")
    logging.info("%d itertations per epoch", len(trainloader))
    iter_num = 0
    max_epoch = self_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    best_dice = 0
    ema_best_dice = 0

    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            model.train()
            ema_model.train()

            volume_batch = sampled_batch["image"].cuda()
            label_batch = sampled_batch["label"].cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:local_args.labeled_bs]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:local_args.labeled_bs]
            unimg_a = volume_batch[local_args.labeled_bs:local_args.labeled_bs + sub_bs]
            unimg_b = volume_batch[local_args.labeled_bs + sub_bs:]

            volume_batch_strong = sampled_batch["image_strong"].cuda()
            label_batch_strong = sampled_batch["label_strong"].cuda()
            img_a_s, img_b_s = volume_batch_strong[:sub_bs], volume_batch_strong[sub_bs:local_args.labeled_bs]
            lab_a_s, lab_b_s = label_batch_strong[:sub_bs], label_batch_strong[sub_bs:local_args.labeled_bs]
            lab_a_s_bg = label_batch_strong[:sub_bs] == 0
            lab_b_s_bg = label_batch_strong[sub_bs:local_args.labeled_bs] == 0
            unimg_a_s = volume_batch_strong[local_args.labeled_bs:local_args.labeled_bs + sub_bs]
            unimg_b_s = volume_batch_strong[local_args.labeled_bs + sub_bs:]

            with torch.no_grad():
                unoutput_a_fg, unoutput_a, unoutput_a_bg, _, _, _, _ = ema_model(unimg_a, unimg_a_s)
                unoutput_b_fg, unoutput_b, unoutput_b_bg, _, _, _, _ = ema_model(unimg_b, unimg_b_s)
                plab_a = pancreas_base.get_cut_mask(unoutput_a, nms=1)
                plab_b = pancreas_base.get_cut_mask(unoutput_b, nms=1)
                plab_a_fg = pancreas_base.get_cut_mask(unoutput_a_fg, nms=1)
                plab_b_fg = pancreas_base.get_cut_mask(unoutput_b_fg, nms=1)
                plab_a_s_bg = pancreas_base.get_cut_mask(unoutput_a_bg, nms=1)
                plab_b_s_bg = pancreas_base.get_cut_mask(unoutput_b_bg, nms=1)
                img_mask, loss_mask = context_mask_pancreas(img_a, local_args.mask_ratio)

            mixl_img = img_a * img_mask + unimg_a * (1 - img_mask)
            mixu_img = unimg_b * img_mask + img_b * (1 - img_mask)
            mixl_img_s = img_a_s * img_mask + unimg_a_s * (1 - img_mask)
            mixu_img_s = unimg_b_s * img_mask + img_b_s * (1 - img_mask)

            mixl_lab = lab_a * img_mask + plab_a * (1 - img_mask)
            mixu_lab = plab_b * img_mask + lab_b * (1 - img_mask)

            outputs_l_fg, outputs_l, outputs_l_bg, _, _, feat_l_fg, feat_l_bg = model(mixl_img, mixl_img_s)
            outputs_u_fg, outputs_u, outputs_u_bg, _, _, feat_u_fg, feat_u_bg = model(mixu_img, mixu_img_s)

            consistency_weight = pancreas_base.get_current_consistency_weight(iter_num // 150)

            loss_l = mix_loss(outputs_l_fg, lab_a, plab_a_fg, loss_mask, u_weight=local_args.u_weight)
            loss_u = mix_loss(outputs_u_fg, plab_b_fg, lab_b, loss_mask, u_weight=local_args.u_weight, unlab=True)
            loss_l_bg = mix_loss(outputs_l_bg, lab_a_s_bg, plab_a_s_bg, loss_mask, u_weight=local_args.u_weight)
            loss_u_bg = mix_loss(outputs_u_bg, plab_b_s_bg, lab_b_s_bg, loss_mask, u_weight=local_args.u_weight, unlab=True)

            with torch.no_grad():
                if local_args.proto_branch == "fg":
                    proto_labels_l = lab_a * img_mask + plab_a_fg * (1 - img_mask)
                    proto_labels_u = plab_b_fg * img_mask + lab_b * (1 - img_mask)
                    conf_a = F.softmax(unoutput_a_fg, dim=1).max(dim=1).values
                    conf_b = F.softmax(unoutput_b_fg, dim=1).max(dim=1).values
                    proto_conf_l = torch.ones_like(lab_a, dtype=outputs_l.dtype) * img_mask + conf_a * (1 - img_mask)
                    proto_conf_u = conf_b * img_mask + torch.ones_like(lab_b, dtype=outputs_u.dtype) * (1 - img_mask)
                else:
                    proto_labels_l = lab_a_s_bg * img_mask + plab_a_s_bg * (1 - img_mask)
                    proto_labels_u = plab_b_s_bg * img_mask + lab_b_s_bg * (1 - img_mask)
                    conf_a = F.softmax(unoutput_a_bg, dim=1).max(dim=1).values
                    conf_b = F.softmax(unoutput_b_bg, dim=1).max(dim=1).values
                    proto_conf_l = torch.ones_like(lab_a_s_bg, dtype=outputs_l.dtype) * img_mask + conf_a * (1 - img_mask)
                    proto_conf_u = conf_b * img_mask + torch.ones_like(lab_b_s_bg, dtype=outputs_u.dtype) * (1 - img_mask)

                proto_labels = torch.cat([proto_labels_l, proto_labels_u], dim=0).long()
                proto_conf = torch.cat([proto_conf_l, proto_conf_u], dim=0)

            proto_features = torch.cat([feat_l_fg, feat_u_fg], dim=0)
            if local_args.proto_branch == "bg":
                proto_features = torch.cat([feat_l_bg, feat_u_bg], dim=0)

            proto_loss, proto_stats = proto_criterion(
                features=proto_features,
                labels=proto_labels,
                confidence=proto_conf,
            )

            loss = loss_l + loss_u + loss_l_bg + loss_u_bg + local_args.proto_weight * proto_loss

            iter_num += 1
            writer.add_scalar("Self/consistency", consistency_weight, iter_num)
            writer.add_scalar("Self/loss_l", loss_l, iter_num)
            writer.add_scalar("Self/loss_u", loss_u, iter_num)
            writer.add_scalar("Self/loss_l_bg", loss_l_bg, iter_num)
            writer.add_scalar("Self/loss_u_bg", loss_u_bg, iter_num)
            writer.add_scalar("Self/proto_loss", proto_loss, iter_num)
            writer.add_scalar("Self/proto_fg_queries", proto_stats["proto_fg_queries"], iter_num)
            writer.add_scalar("Self/proto_bg_queries", proto_stats["proto_bg_queries"], iter_num)
            writer.add_scalar("Self/loss_all", loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info(
                "iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f, loss_proto: %03f",
                iter_num,
                loss,
                loss_l,
                loss_u,
                proto_loss,
            )

            update_ema_variables(model, ema_model, 0.99)

            if iter_num % 200 == 0:
                model.eval()
                ema_model.eval()
                dice_sample = test_3d_patch.var_all_case_Pancreas_argument(
                    model,
                    num_classes=num_classes,
                    patch_size=patch_size,
                    stride_xy=16,
                    stride_z=16,
                    dataset_path=local_args.root_path,
                )
                ema_dice_sample = test_3d_patch.var_all_case_Pancreas_argument(
                    ema_model,
                    num_classes=num_classes,
                    patch_size=patch_size,
                    stride_xy=16,
                    stride_z=16,
                    dataset_path=local_args.root_path,
                )
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(self_snapshot_path, f"iter_{iter_num}_dice_{best_dice}.pth")
                    save_best_path = os.path.join(self_snapshot_path, f"{local_args.model}_best_model.pth")
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to %s", save_mode_path)
                if ema_dice_sample > ema_best_dice:
                    ema_best_dice = round(ema_dice_sample, 4)
                    save_mode_path = os.path.join(self_snapshot_path, f"iter_{iter_num}_ema_dice_{ema_best_dice}.pth")
                    save_ema_best_path = os.path.join(self_snapshot_path, f"{local_args.model}_ema_best_model.pth")
                    torch.save(ema_model.state_dict(), save_mode_path)
                    torch.save(ema_model.state_dict(), save_ema_best_path)
                    logging.info("save best ema model to %s", save_mode_path)
                writer.add_scalar("4_Var_dice/Dice", ema_dice_sample, iter_num)
                writer.add_scalar("4_Var_dice/Best_dice", ema_best_dice, iter_num)
                model.train()

            if iter_num % 200 == 1:
                ins_width = 2
                _, _, h_size, w_size, d_size = outputs_l.size()
                snapshot_img = torch.zeros(size=(d_size, 3, 3 * h_size + 3 * ins_width, w_size + ins_width), dtype=torch.float32)

                snapshot_img[:, :, h_size:h_size + ins_width, :] = 1
                snapshot_img[:, :, 2 * h_size + ins_width:2 * h_size + 2 * ins_width, :] = 1
                snapshot_img[:, :, 3 * h_size + 2 * ins_width:3 * h_size + 3 * ins_width, :] = 1
                snapshot_img[:, :, :, w_size:w_size + ins_width] = 1

                outputs_l_soft = F.softmax(outputs_l, dim=1)
                seg_out = outputs_l_soft[0, 1, ...].permute(2, 0, 1)
                target = mixl_lab[0, ...].permute(2, 0, 1)
                train_img = mixl_img[0, 0, ...].permute(2, 0, 1)

                normalized_img = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 0, :h_size, :w_size] = normalized_img
                snapshot_img[:, 1, :h_size, :w_size] = normalized_img
                snapshot_img[:, 2, :h_size, :w_size] = normalized_img

                snapshot_img[:, 0, h_size + ins_width:2 * h_size + ins_width, :w_size] = target
                snapshot_img[:, 1, h_size + ins_width:2 * h_size + ins_width, :w_size] = target
                snapshot_img[:, 2, h_size + ins_width:2 * h_size + ins_width, :w_size] = target

                snapshot_img[:, 0, 2 * h_size + 2 * ins_width:3 * h_size + 2 * ins_width, :w_size] = seg_out
                snapshot_img[:, 1, 2 * h_size + 2 * ins_width:3 * h_size + 2 * ins_width, :w_size] = seg_out
                snapshot_img[:, 2, 2 * h_size + 2 * ins_width:3 * h_size + 2 * ins_width, :w_size] = seg_out

                writer.add_images(f"Epoch_{epoch}_Iter_{iter_num}_labeled", snapshot_img)

                outputs_u_soft = F.softmax(outputs_u, dim=1)
                seg_out = outputs_u_soft[0, 1, ...].permute(2, 0, 1)
                target = mixu_lab[0, ...].permute(2, 0, 1)
                train_img = mixu_img[0, 0, ...].permute(2, 0, 1)

                normalized_img = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 0, :h_size, :w_size] = normalized_img
                snapshot_img[:, 1, :h_size, :w_size] = normalized_img
                snapshot_img[:, 2, :h_size, :w_size] = normalized_img

                snapshot_img[:, 0, h_size + ins_width:2 * h_size + ins_width, :w_size] = target
                snapshot_img[:, 1, h_size + ins_width:2 * h_size + ins_width, :w_size] = target
                snapshot_img[:, 2, h_size + ins_width:2 * h_size + ins_width, :w_size] = target

                snapshot_img[:, 0, 2 * h_size + 2 * ins_width:3 * h_size + 2 * ins_width, :w_size] = seg_out
                snapshot_img[:, 1, 2 * h_size + 2 * ins_width:3 * h_size + 2 * ins_width, :w_size] = seg_out
                snapshot_img[:, 2, 2 * h_size + 2 * ins_width:3 * h_size + 2 * ins_width, :w_size] = seg_out

                writer.add_images(f"Epoch_{epoch}_Iter_{iter_num}_unlabel", snapshot_img)

            if iter_num >= self_max_iterations:
                break

        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    pre_snapshot_path = "{}/{}_{}_labeled/pre_train".format(args.snapshot_path, args.exp, args.labelnum)
    self_snapshot_path = "{}/{}_{}_labeled/self_train".format(args.snapshot_path, args.exp, args.labelnum)
    print("Starting Pancreas single-branch prototype ablation training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + "/code"):
            shutil.rmtree(snapshot_path + "/code")

    logging.basicConfig(
        filename=pre_snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    logging.basicConfig(
        filename=self_snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)
