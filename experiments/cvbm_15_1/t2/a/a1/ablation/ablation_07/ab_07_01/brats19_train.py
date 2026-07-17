import sys
import os
import argparse
import ast
import logging
import shutil
import random

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from tensorboardX import SummaryWriter
from skimage.measure import label
import numpy as np

from utils import losses, ramps, test_3d_patch
from dataloaders.brats19.brats19_dataset import BRATSDataset
from dataloaders.datasets_3d import WeakStrongAugment3d, TwoStreamBatchSampler
from experiments.cvbm_15_1.t2.a.a1.modules import CVBMArgumentWithCrossSKC3DProto
from experiments.cvbm_15_1.t2.a.prototype_losses import BranchBatchPrototypeLoss
from networks.net_factory import net_factory
from utils.BCP_utils import context_mask_pancreas, mix_loss, update_ema_variables


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/BRATS19', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='CVBM_BRATS19_FgStrong_BgWeak_Ablation_07_01', help='exp_name')
parser.add_argument('--model', type=str, default='CVBM_Argument', help='model_name')
parser.add_argument('--pre_max_iteration', type=int, default=2000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int, default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int, default=250, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--patch_size', type=ast.literal_eval, default=(96, 96, 96), help='patch_size of loading image')
parser.add_argument('--num_classes', type=int, default=2, help="number of dataset's class")
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=0, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=25, help='trained samples')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_workers', type=int, default=8, help='cpu core num_workers')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default='10.0', help='magnitude')
parser.add_argument('--proto_weight', type=float, default=0.1, help='weight for branch prototype loss')
parser.add_argument('--proto_dim', type=int, default=32, help='projection dimension for prototype loss')
parser.add_argument('--proto_temperature', type=float, default=0.2, help='temperature for prototype contrast')
parser.add_argument('--proto_conf_threshold', type=float, default=0.8, help='confidence threshold for prototype construction')
parser.add_argument('--proto_query_threshold', type=float, default=0.0, help='confidence threshold for prototype queries')
parser.add_argument('--proto_patch', type=ast.literal_eval, default=(8, 8, 8), help='patch size for prototype pooling')
parser.add_argument('--proto_max_queries', type=int, default=4096, help='max patch queries per prototype branch')
parser.add_argument('--train_num', type=int, default=1, help='the count of train')
# -- setting of BANET
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2 / 3, help='ratio of mask/image')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
parser.add_argument('--beta', type=float, default=0.3, help='balance factor to control regional and sdm loss')
parser.add_argument('--snapshot_path', type=str, default='./results/CVBM_15_1_t2_a/a1/ablation_07/ab_07_01/1/', help='snapshot path to save model')
args = parser.parse_args()
torch.backends.cudnn.benchmark = True


def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC(masks)
    return masks


def LargestCC(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)

    return torch.Tensor(batch_list).cuda()


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])


def load_pretrained_backbone(model, ckpt_path):
    """
    Loads encoder/decoder weights from a vanilla CVBM_Argument checkpoint into
    the SKC-augmented backbone.
    """
    state = torch.load(str(ckpt_path))
    state_dict = state['net'] if 'net' in state else state
    model_dict = model.state_dict()
    compatible = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(compatible)
    model.load_state_dict(model_dict)
    logging.info("Loaded %d/%d tensors from %s into %s",
                 len(compatible), len(model_dict), ckpt_path, model.__class__.__name__)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = args.patch_size
num_classes = args.num_classes


def pre_train(args, snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    db_train = BRATSDataset(
        base_dir=train_data_path,
        split='train',
        transform=transforms.Compose([
            WeakStrongAugment3d(args.patch_size, flag_rot=False)
        ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, len(db_train)))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    sub_bs = int(args.labeled_bs / 2)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        pin_memory_device="cuda",
        persistent_workers=True,
    )
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("%d itertations per epoch", len(trainloader))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            model.train()
            volume_batch, label_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][:args.labeled_bs]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:]

            volume_batch_strong, label_batch_strong = sampled_batch['image_strong'], sampled_batch['label_strong']
            volume_batch_strong, label_batch_strong = volume_batch_strong.cuda(), label_batch_strong.cuda()
            img_a_s, img_b_s = volume_batch_strong[:sub_bs], volume_batch_strong[sub_bs:args.labeled_bs]
            lab_a_s, lab_b_s = label_batch_strong[:sub_bs], label_batch_strong[sub_bs:args.labeled_bs]
            with torch.no_grad():
                img_mask, loss_mask = context_mask_pancreas(img_a, args.mask_ratio)

            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)

            volume_batch_strong = img_a_s * img_mask + img_b_s * (1 - img_mask)
            label_batch_strong = lab_a_s * img_mask + lab_b_s * (1 - img_mask)

            outputs_fg, outputs, outputs_bg, out_tanh, out_tanh_bg = model(volume_batch_strong, volume_batch)
            loss_seg = 0
            loss_seg_dice = 0

            y2 = outputs_fg[:args.labeled_bs, ...]
            y_prob2 = F.softmax(y2, dim=1)
            loss_seg += F.cross_entropy(y2[:args.labeled_bs], (label_batch_strong[:args.labeled_bs, ...] == 1).long())
            loss_seg_dice += DICE(y2, label_batch_strong[:args.labeled_bs, ...] == 1)

            y_bg = outputs_bg[:args.labeled_bs, ...]
            y_prob_bg = F.softmax(y_bg, dim=1)
            loss_seg += F.cross_entropy(y_bg[:args.labeled_bs], (label_batch[:args.labeled_bs, ...] == 0).long())
            loss_seg_dice += DICE(y_bg, label_batch[:args.labeled_bs, ...] == 0)

            loss = (loss_seg + loss_seg_dice) / 2

            iter_num += 1

            writer.add_scalar('pre/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('pre/loss_seg', loss_seg, iter_num)
            # writer.add_scalar('pre/loss_sdf', loss_seg, iter_num)
            writer.add_scalar('pre/loss_all', loss, iter_num)
            writer.add_scalar('pre/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('pre/loss_seg', loss_seg, iter_num)
            writer.add_scalar('pre/loss_all', loss, iter_num)
            logging.info("y_prob2: %s, label_batch: %s", torch.argmax(y_prob2, dim=1).sum(), label_batch.sum())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f', iter_num, loss, loss_seg_dice, loss_seg)

            if iter_num % 200 == 0 and torch.argmax(y_prob2, dim=1).sum() != 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_BRATS19_argument(
                    model,
                    num_classes=num_classes,
                    patch_size=patch_size,
                    stride_xy=64,
                    stride_z=64,
                    dataset_path=args.root_path)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path, f'iter_{iter_num}_dice_{best_dice}.pth')
                    save_best_path = os.path.join(snapshot_path, f'{args.model}_best_model.pth')
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                    logging.info("save best model to %s", save_mode_path)
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)

            if iter_num >= pre_max_iterations:
                break

        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(args, pre_snapshot_path, self_snapshot_path):
    # Cross-attention semantic interaction backbone:
    # foreground bottleneck features query background K/V through local and global
    # cross-attention before task-specific decoding.
    model = CVBMArgumentWithCrossSKC3DProto(
        n_channels=1,
        n_classes=num_classes,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()
    ema_model = CVBMArgumentWithCrossSKC3DProto(
        n_channels=1,
        n_classes=num_classes,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()
    for param in ema_model.parameters():
        param.detach_()
    db_train = BRATSDataset(
        base_dir=train_data_path,
        split='train',
        transform=transforms.Compose([
            WeakStrongAugment3d(args.patch_size, flag_rot=False)
        ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, len(db_train)))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    sub_bs = int(args.labeled_bs / 2)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        pin_memory_device="cuda",
        persistent_workers=True,
    )
    proto_criterion = BranchBatchPrototypeLoss(
        in_channels=16,
        proj_dim=args.proto_dim,
        num_classes=num_classes,
        temperature=args.proto_temperature,
        confidence_threshold=args.proto_conf_threshold,
        query_threshold=args.proto_query_threshold,
        patch_size=args.proto_patch,
        max_queries=args.proto_max_queries,
    ).cuda()
    optimizer = optim.SGD(
        list(model.parameters()) + list(proto_criterion.parameters()),
        lr=base_lr,
        momentum=0.9,
        weight_decay=0.0001,
    )

    pretrained_model = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')
    load_pretrained_backbone(model, pretrained_model)
    load_pretrained_backbone(ema_model, pretrained_model)

    writer = SummaryWriter(self_snapshot_path + '/log')
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

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:args.labeled_bs]
            lab_a_bg, lab_b_bg = label_batch[:sub_bs] == 0, label_batch[sub_bs:args.labeled_bs] == 0
            unimg_a, unimg_b = volume_batch[args.labeled_bs:args.labeled_bs + sub_bs], volume_batch[args.labeled_bs + sub_bs:]

            volume_batch_strong, label_batch_strong = sampled_batch['image_strong'], sampled_batch['label_strong']
            volume_batch_strong, label_batch_strong = volume_batch_strong.cuda(), label_batch_strong.cuda()
            img_a_s, img_b_s = volume_batch_strong[:sub_bs], volume_batch_strong[sub_bs:args.labeled_bs]
            lab_a_s, lab_b_s = label_batch_strong[:sub_bs], label_batch_strong[sub_bs:args.labeled_bs]
            unimg_a_s, unimg_b_s = volume_batch_strong[args.labeled_bs:args.labeled_bs + sub_bs], volume_batch_strong[args.labeled_bs + sub_bs:]

            with torch.no_grad():
                unoutput_a_fg, unoutput_a, unoutput_a_bg, unoutput_a_sdm, unoutput_a_sdm_bg, _, _ = ema_model(unimg_a_s, unimg_a)
                unoutput_b_fg, unoutput_b, unoutput_b_bg, unoutput_b_sdm, unoutput_b_sdm_bg, _, _ = ema_model(unimg_b_s, unimg_b)
                plab_a = get_cut_mask(unoutput_a, nms=1)
                plab_b = get_cut_mask(unoutput_b, nms=1)
                plab_a_fg = get_cut_mask(unoutput_a_fg, nms=1)
                plab_b_fg = get_cut_mask(unoutput_b_fg, nms=1)
                plab_a_bg = get_cut_mask(unoutput_a_bg, nms=1)
                plab_b_bg = get_cut_mask(unoutput_b_bg, nms=1)
                img_mask, loss_mask = context_mask_pancreas(img_a, args.mask_ratio)

            mixl_img_bg = img_a * img_mask + unimg_a * (1 - img_mask)
            mixu_img_bg = unimg_b * img_mask + img_b * (1 - img_mask)
            mixl_img_fg = img_a_s * img_mask + unimg_a_s * (1 - img_mask)
            mixu_img_fg = unimg_b_s * img_mask + img_b_s * (1 - img_mask)

            mixl_lab = lab_a * img_mask + plab_a * (1 - img_mask)
            mixu_lab = plab_b * img_mask + lab_b * (1 - img_mask)

            outputs_l_fg, outputs_l, outputs_l_bg, sdm_outputs_l, sdm_outputs_l_bg, feat_l_fg, feat_l_bg = model(mixl_img_fg, mixl_img_bg)
            outputs_u_fg, outputs_u, outputs_u_bg, sdm_outputs_u, sdm_outputs_u_bg, feat_u_fg, feat_u_bg = model(mixu_img_fg, mixu_img_bg)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss_l = mix_loss(outputs_l_fg, lab_a_s, plab_a_fg, loss_mask, u_weight=args.u_weight)
            loss_u = mix_loss(outputs_u_fg, plab_b_fg, lab_b_s, loss_mask, u_weight=args.u_weight, unlab=True)
            loss_l_bg = mix_loss(outputs_l_bg, lab_a_bg, plab_a_bg, loss_mask, u_weight=args.u_weight)
            loss_u_bg = mix_loss(outputs_u_bg, plab_b_bg, lab_b_bg, loss_mask, u_weight=args.u_weight, unlab=True)

            with torch.no_grad():
                proto_labels_l_fg = lab_a_s * img_mask + plab_a_fg * (1 - img_mask)
                proto_labels_u_fg = plab_b_fg * img_mask + lab_b_s * (1 - img_mask)
                proto_labels_l_bg = lab_a_bg * img_mask + plab_a_bg * (1 - img_mask)
                proto_labels_u_bg = plab_b_bg * img_mask + lab_b_bg * (1 - img_mask)
                proto_labels_fg = torch.cat([proto_labels_l_fg, proto_labels_u_fg], dim=0).long()
                proto_labels_bg = torch.cat([proto_labels_l_bg, proto_labels_u_bg], dim=0).long()
                conf_a_fg = F.softmax(unoutput_a_fg, dim=1).max(dim=1).values
                conf_b_fg = F.softmax(unoutput_b_fg, dim=1).max(dim=1).values
                conf_a_bg = F.softmax(unoutput_a_bg, dim=1).max(dim=1).values
                conf_b_bg = F.softmax(unoutput_b_bg, dim=1).max(dim=1).values
                proto_conf_l_fg = torch.ones_like(lab_a_s, dtype=outputs_l.dtype) * img_mask + conf_a_fg * (1 - img_mask)
                proto_conf_u_fg = conf_b_fg * img_mask + torch.ones_like(lab_b_s, dtype=outputs_u.dtype) * (1 - img_mask)
                proto_conf_l_bg = torch.ones_like(lab_a_bg, dtype=outputs_l.dtype) * img_mask + conf_a_bg * (1 - img_mask)
                proto_conf_u_bg = conf_b_bg * img_mask + torch.ones_like(lab_b_bg, dtype=outputs_u.dtype) * (1 - img_mask)
                proto_conf_fg = torch.cat([proto_conf_l_fg, proto_conf_u_fg], dim=0)
                proto_conf_bg = torch.cat([proto_conf_l_bg, proto_conf_u_bg], dim=0)

            proto_loss, proto_stats = proto_criterion(
                feat_fg=torch.cat([feat_l_fg, feat_u_fg], dim=0),
                feat_bg=torch.cat([feat_l_bg, feat_u_bg], dim=0),
                labels_fg=proto_labels_fg,
                labels_bg=proto_labels_bg,
                confidence_fg=proto_conf_fg,
                confidence_bg=proto_conf_bg,
            )

            loss = loss_l + loss_u + loss_l_bg + loss_u_bg + args.proto_weight * proto_loss

            iter_num += 1
            writer.add_scalar('Self/consistency', consistency_weight, iter_num)
            writer.add_scalar('Self/loss_l', loss_l, iter_num)
            writer.add_scalar('Self/loss_u', loss_u, iter_num)
            writer.add_scalar('Self/loss_l_bg', loss_l_bg, iter_num)
            writer.add_scalar('Self/loss_u_bg', loss_u_bg, iter_num)
            writer.add_scalar('Self/proto_loss', proto_loss, iter_num)
            writer.add_scalar('Self/proto_fg_queries', proto_stats["proto_fg_queries"], iter_num)
            writer.add_scalar('Self/proto_bg_queries', proto_stats["proto_bg_queries"], iter_num)
            writer.add_scalar('Self/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f, loss_proto: %03f',
                         iter_num, loss, loss_l, loss_u, proto_loss)

            update_ema_variables(model, ema_model, 0.99)

            if iter_num % 200 == 0:
                model.eval()
                ema_model.eval()
                dice_sample = test_3d_patch.var_all_case_BRATS19_argument(
                    model,
                    num_classes=num_classes,
                    patch_size=patch_size,
                    stride_xy=64,
                    stride_z=64,
                    dataset_path=args.root_path)
                ema_dice_sample = test_3d_patch.var_all_case_BRATS19_argument(
                    ema_model,
                    num_classes=num_classes,
                    patch_size=patch_size,
                    stride_xy=64,
                    stride_z=64,
                    dataset_path=args.root_path)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(self_snapshot_path, f'iter_{iter_num}_dice_{best_dice}.pth')
                    save_best_path = os.path.join(self_snapshot_path, f'{args.model}_best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to %s", save_mode_path)
                if ema_dice_sample > ema_best_dice:
                    ema_best_dice = round(ema_dice_sample, 4)
                    save_mode_path = os.path.join(self_snapshot_path, f'iter_{iter_num}_ema_dice_{ema_best_dice}.pth')
                    save_ema_best_path = os.path.join(self_snapshot_path, f'{args.model}_ema_best_model.pth')
                    torch.save(ema_model.state_dict(), save_mode_path)
                    torch.save(ema_model.state_dict(), save_ema_best_path)
                    logging.info("save best ema model to %s", save_mode_path)
                writer.add_scalar('4_Var_dice/Dice', ema_dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', ema_best_dice, iter_num)
                model.train()

            if iter_num % 200 == 1:
                ins_width = 2
                B, C, H, W, D = outputs_l.size()
                snapshot_img = torch.zeros(size=(D, 3, 3 * H + 3 * ins_width, W + ins_width), dtype=torch.float32)

                snapshot_img[:, :, H:H + ins_width, :] = 1
                snapshot_img[:, :, 2 * H + ins_width:2 * H + 2 * ins_width, :] = 1
                snapshot_img[:, :, 3 * H + 2 * ins_width:3 * H + 3 * ins_width, :] = 1
                snapshot_img[:, :, :, W:W + ins_width] = 1

                outputs_l_soft = F.softmax(outputs_l, dim=1)
                seg_out = outputs_l_soft[0, 1, ...].permute(2, 0, 1)
                target = mixl_lab[0, ...].permute(2, 0, 1)
                train_img = mixl_img_fg[0, 0, ...].permute(2, 0, 1)

                snapshot_img[:, 0, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 1, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 2, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))

                snapshot_img[:, 0, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 1, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 2, H + ins_width:2 * H + ins_width, :W] = target

                snapshot_img[:, 0, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 1, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 2, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out

                writer.add_images(f'Epoch_{epoch}_Iter_{iter_num}_labeled', snapshot_img)

                outputs_u_soft = F.softmax(outputs_u, dim=1)
                seg_out = outputs_u_soft[0, 1, ...].permute(2, 0, 1)
                target = mixu_lab[0, ...].permute(2, 0, 1)
                train_img = mixu_img_fg[0, 0, ...].permute(2, 0, 1)

                snapshot_img[:, 0, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 1, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 2, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))

                snapshot_img[:, 0, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 1, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 2, H + ins_width:2 * H + ins_width, :W] = target

                snapshot_img[:, 0, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 1, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 2, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out

                writer.add_images(f'Epoch_{epoch}_Iter_{iter_num}_unlabel', snapshot_img)

            if iter_num >= self_max_iterations:
                break

        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    pre_snapshot_path = "{}/{}/brats19/label{}/proto_w{}_patch{}/pre_train".format(
        args.snapshot_path,
        args.exp,
        args.labelnum,
        args.proto_weight,
        args.proto_patch,
    )
    self_snapshot_path = "{}/{}/brats19/label{}/proto_w{}_patch{}/self_train".format(
        args.snapshot_path,
        args.exp,
        args.labelnum,
        args.proto_weight,
        args.proto_patch,
    )
    print("Starting BRATS19 FG-query-BG Cross-SKC training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')

    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)
