import sys
import os
import argparse
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
from dataloaders.datasets_3d import WeakStrongAugment3d, Pancreas, TwoStreamBatchSampler
from utils.BCP_utils import context_mask_pancreas, mix_loss, update_ema_variables
from .foreground_refine_model import CVBMArgumentWithSKCForegroundRefine3D


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/Pancreas', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='CVBM_Pancreas', help='exp_name')
parser.add_argument('--model', type=str, default='CVBM_Argument_FGR', help='model_name')
parser.add_argument('--pre_max_iteration', type=int, default=5000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int, default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int, default=62, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--patch_size', type=tuple, default=(96, 96, 96), help='patch_size of loading image')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=0, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=12, help='trained samples')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_workers', type=int, default=8, help='cpu core num_workers')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default='10.0', help='magnitude')
parser.add_argument('--topnum', type=int, default=32, help='negative sample contrast learning')
parser.add_argument('--contrast_patch', type=tuple, default=(8, 8, 8))
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2 / 3, help='ratio of mask/image')
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
parser.add_argument('--beta', type=float, default=0.3, help='balance factor to control regional and sdm loss')
parser.add_argument('--snapshot_path', type=str, default='./results/CVBM_11_1_6/1/', help='snapshot path to save model')
parser.add_argument('--roi_warmup', type=int, default=400, help='iteration to start ROI supervision')
parser.add_argument('--roi_kernel', type=int, default=7, help='odd kernel size used to dilate foreground ROI')
parser.add_argument('--roi_threshold', type=float, default=0.55, help='threshold for pseudo ROI generation')
parser.add_argument('--proposal_loss_weight', type=float, default=0.2, help='weight of ROI proposal loss')
parser.add_argument('--roi_loss_weight', type=float, default=0.5, help='weight of ROI refinement segmentation loss')
parser.add_argument('--fuse_loss_weight', type=float, default=0.5, help='weight of fused/global refinement losses')
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
    batch_list = []
    for n in range(segmentation.shape[0]):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largest_cc = n_prob
        batch_list.append(largest_cc)
    return torch.Tensor(batch_list).cuda()


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def select_patches_for_contrast_3d(output_mix, topnum=16, patch_size=(4, 4, 4), choose_largest=False):
    bsz, channels, height, width, depth = output_mix.shape
    ph, pw, pd = patch_size
    assert height % ph == 0 and width % pw == 0 and depth % pd == 0, "H/W/D must be divisible by patch_size"

    probs = F.softmax(output_mix, dim=1)
    score = probs.max(dim=1, keepdim=True).values

    patch_conf = F.avg_pool3d(score, kernel_size=(ph, pw, pd), stride=(ph, pw, pd)).flatten(1)
    _, top_idx = patch_conf.topk(topnum, dim=1, largest=choose_largest)
    _, num_locations = patch_conf.shape
    all_idx = torch.arange(num_locations, device=output_mix.device).unsqueeze(0).expand(bsz, -1)
    neg_mask = torch.ones_like(all_idx, dtype=torch.bool)
    neg_mask.scatter_(1, top_idx, False)
    num_neg = num_locations - topnum

    feat_patches = F.avg_pool3d(probs, kernel_size=(ph, pw, pd), stride=(ph, pw, pd))
    feat_patches = feat_patches.flatten(2).permute(0, 2, 1).contiguous()

    pos_patches = torch.gather(feat_patches, dim=1, index=top_idx.unsqueeze(-1).expand(-1, -1, channels))
    neg_patches = feat_patches[neg_mask].view(bsz, num_neg, channels)
    return pos_patches, neg_patches


def build_roi_target(mask, kernel_size):
    if mask.dim() == 4:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size == 1:
        return mask
    padding = kernel_size // 2
    dilated = F.max_pool3d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = 1.0 - F.max_pool3d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=padding)
    boundary = (dilated - eroded).clamp_(0.0, 1.0)
    return torch.clamp(mask + boundary + dilated, max=1.0)


def build_pseudo_roi_target(fg_logits, fused_logits, bg_logits, roi_mask=None, roi_threshold=0.55):
    fg_prob = F.softmax(fg_logits, dim=1)[:, 1:2]
    fused_prob = F.softmax(fused_logits, dim=1)[:, 1:2]
    bg_fg_prob = F.softmax(bg_logits, dim=1)[:, 0:1]
    agreement = 0.5 * (fg_prob + bg_fg_prob)
    disagreement = torch.abs(fg_prob - bg_fg_prob)
    refine_region = torch.abs(fg_prob - fused_prob)

    confident_fg = (agreement > roi_threshold).float()
    conflict_band = (disagreement > max(0.15, roi_threshold * 0.5)).float()
    refine_band = (refine_region > max(0.10, roi_threshold * 0.4)).float()

    pseudo_roi = torch.clamp(confident_fg + conflict_band + refine_band, max=1.0)
    if roi_mask is not None:
        pseudo_roi = torch.max(pseudo_roi, (roi_mask.detach() > roi_threshold).float())
    return pseudo_roi


def supervised_seg_loss(logits, target, dice_loss):
    target = target.long()
    loss_ce = F.cross_entropy(logits, target)
    loss_dice = dice_loss(logits, target)
    return (loss_ce + loss_dice) / 2, loss_ce, loss_dice


def roi_seg_loss(logits, target, roi_mask, dice_loss):
    target = target.long()
    if roi_mask.dim() == 5:
        flat_mask = roi_mask[:, 0].float()
    else:
        flat_mask = roi_mask.float()
    ce_map = CE(logits, target)
    loss_ce = (ce_map * flat_mask).sum() / (flat_mask.sum() + 1e-6)
    loss_dice = dice_loss(logits, target, flat_mask)
    return (loss_ce + loss_dice) / 2, loss_ce, loss_dice


def configure_logging(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


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
num_classes = 2


def build_trainloader(args, db_train, labeled_idxs, unlabeled_idxs):
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs,
        unlabeled_idxs,
        args.batch_size,
        args.batch_size - args.labeled_bs,
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    return DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        pin_memory_device="cuda",
        persistent_workers=args.num_workers > 0,
    )


def pre_train(args, snapshot_path):
    model = CVBMArgumentWithSKCForegroundRefine3D(
        n_channels=1,
        n_classes=num_classes,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()

    db_train = Pancreas(
        base_dir=train_data_path,
        split='train',
        transform=transforms.Compose([WeakStrongAugment3d(args.patch_size, flag_rot=True)]),
    )
    labeled_idxs = list(range(args.labelnum))
    unlabeled_idxs = list(range(args.labelnum, args.max_samples))
    trainloader = build_trainloader(args, db_train, labeled_idxs, unlabeled_idxs)
    sub_bs = int(args.labeled_bs / 2)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    dice_loss = losses.mask_DiceLoss(nclass=2)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("%d itertations per epoch", len(trainloader))

    iter_num = 0
    best_dice = -1.0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            model.train()
            volume_batch = sampled_batch['image'][:args.labeled_bs].cuda()
            label_batch = sampled_batch['label'][:args.labeled_bs].cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:]

            volume_batch_strong = sampled_batch['image_strong'][:args.labeled_bs].cuda()
            label_batch_strong = sampled_batch['label_strong'][:args.labeled_bs].cuda()
            img_a_s, img_b_s = volume_batch_strong[:sub_bs], volume_batch_strong[sub_bs:]
            lab_a_s, lab_b_s = label_batch_strong[:sub_bs], label_batch_strong[sub_bs:]

            with torch.no_grad():
                img_mask, _ = context_mask_pancreas(img_a, args.mask_ratio)

            mix_mask = img_mask.unsqueeze(0).unsqueeze(0).float()
            volume_batch = img_a * mix_mask + img_b * (1.0 - mix_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)
            volume_batch_strong = img_a_s * mix_mask + img_b_s * (1.0 - mix_mask)
            label_batch_strong = lab_a_s * img_mask + lab_b_s * (1 - img_mask)

            (
                outputs_fg,
                outputs_refined,
                outputs_bg,
                out_tanh,
                out_tanh_bg,
                outputs_fused,
                roi_logits,
                roi_mask,
                refine_delta,
            ) = model(volume_batch, volume_batch_strong)

            target_fg = (label_batch == 1).long()
            target_bg = (label_batch_strong == 0).long()
            roi_target = build_roi_target(target_fg, args.roi_kernel)

            loss_fg, loss_fg_ce, loss_fg_dice = supervised_seg_loss(outputs_fg, target_fg, dice_loss)
            loss_bg, loss_bg_ce, loss_bg_dice = supervised_seg_loss(outputs_bg, target_bg, dice_loss)
            loss_fuse, loss_fuse_ce, loss_fuse_dice = supervised_seg_loss(outputs_fused, target_fg, dice_loss)
            loss_refined, loss_refined_ce, loss_refined_dice = supervised_seg_loss(outputs_refined, target_fg, dice_loss)

            if iter_num >= args.roi_warmup:
                loss_roi_prop = F.binary_cross_entropy_with_logits(roi_logits, roi_target)
                loss_roi_ref, _, _ = roi_seg_loss(outputs_refined, target_fg, roi_target, dice_loss)
            else:
                loss_roi_prop = outputs_refined.new_tensor(0.0)
                loss_roi_ref = outputs_refined.new_tensor(0.0)

            loss = (
                loss_fg
                + loss_bg
                + args.fuse_loss_weight * (loss_fuse + loss_refined)
                + args.proposal_loss_weight * loss_roi_prop
                + args.roi_loss_weight * loss_roi_ref
            )

            iter_num += 1
            writer.add_scalar('pre/loss_fg', loss_fg, iter_num)
            writer.add_scalar('pre/loss_bg', loss_bg, iter_num)
            writer.add_scalar('pre/loss_fuse', loss_fuse, iter_num)
            writer.add_scalar('pre/loss_refined', loss_refined, iter_num)
            writer.add_scalar('pre/loss_roi_prop', loss_roi_prop, iter_num)
            writer.add_scalar('pre/loss_roi_ref', loss_roi_ref, iter_num)
            writer.add_scalar('pre/roi_mask_mean', roi_mask.mean(), iter_num)
            writer.add_scalar('pre/refine_delta_norm', refine_delta.abs().mean(), iter_num)
            writer.add_scalar('pre/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_prob2 = F.softmax(outputs_fg, dim=1)
            logging.info(
                'iteration %d : loss=%03f, fg_ce=%03f, fg_dice=%03f, bg_ce=%03f, bg_dice=%03f, roi_mean=%03f, fg_pixels=%s',
                iter_num,
                loss,
                loss_fg_ce,
                loss_fg_dice,
                loss_bg_ce,
                loss_bg_dice,
                roi_mask.mean(),
                torch.argmax(y_prob2, dim=1).sum(),
            )

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_Pancreas_argument(
                    model,
                    num_classes=num_classes,
                    patch_size=patch_size,
                    stride_xy=16,
                    stride_z=16,
                    dataset_path=args.root_path,
                )
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path, f'iter_{iter_num}_dice_{best_dice}.pth')
                    save_best_path = os.path.join(snapshot_path, f'{args.model}_best_model.pth')
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                    logging.info("save best model to %s", save_mode_path)
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= pre_max_iterations:
                break
        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(args, pre_snapshot_path, self_snapshot_path):
    model = CVBMArgumentWithSKCForegroundRefine3D(
        n_channels=1,
        n_classes=num_classes,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()
    ema_model = CVBMArgumentWithSKCForegroundRefine3D(
        n_channels=1,
        n_classes=num_classes,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()
    for param in ema_model.parameters():
        param.detach_()

    db_train = Pancreas(
        base_dir=train_data_path,
        split='train',
        transform=transforms.Compose([WeakStrongAugment3d(args.patch_size, flag_rot=True)]),
    )
    labeled_idxs = list(range(args.labelnum))
    unlabeled_idxs = list(range(args.labelnum, args.max_samples))
    trainloader = build_trainloader(args, db_train, labeled_idxs, unlabeled_idxs)
    sub_bs = int(args.labeled_bs / 2)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    bcl_loss = losses.BlockContrastiveLoss()
    dice_loss = losses.mask_DiceLoss(nclass=2)

    pretrained_model = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')
    load_net(model, pretrained_model)
    load_net(ema_model, pretrained_model)

    writer = SummaryWriter(self_snapshot_path + '/log')
    logging.info("%d itertations per epoch", len(trainloader))
    iter_num = 0
    max_epoch = self_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    best_dice = -1.0
    ema_best_dice = -1.0

    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            model.train()
            ema_model.train()

            volume_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:args.labeled_bs]
            lab_a_bg, lab_b_bg = label_batch[:sub_bs] == 0, label_batch[sub_bs:args.labeled_bs] == 0
            unimg_a = volume_batch[args.labeled_bs:args.labeled_bs + sub_bs]
            unimg_b = volume_batch[args.labeled_bs + sub_bs:]

            volume_batch_strong = sampled_batch['image_strong'].cuda()
            label_batch_strong = sampled_batch['label_strong'].cuda()
            img_a_s, img_b_s = volume_batch_strong[:sub_bs], volume_batch_strong[sub_bs:args.labeled_bs]
            lab_a_s, lab_b_s = label_batch_strong[:sub_bs], label_batch_strong[sub_bs:args.labeled_bs]
            lab_a_s_bg = label_batch_strong[:sub_bs] == 0
            lab_b_s_bg = label_batch_strong[sub_bs:args.labeled_bs] == 0
            unimg_a_s = volume_batch_strong[args.labeled_bs:args.labeled_bs + sub_bs]
            unimg_b_s = volume_batch_strong[args.labeled_bs + sub_bs:]

            with torch.no_grad():
                (
                    unoutput_a_fg,
                    unoutput_a,
                    unoutput_a_bg,
                    _,
                    _,
                    unoutput_a_global,
                    _,
                    unroi_a_mask,
                    _,
                ) = ema_model(unimg_a, unimg_a_s)
                (
                    unoutput_b_fg,
                    unoutput_b,
                    unoutput_b_bg,
                    _,
                    _,
                    unoutput_b_global,
                    _,
                    unroi_b_mask,
                    _,
                ) = ema_model(unimg_b, unimg_b_s)

                plab_a = get_cut_mask(unoutput_a, nms=1)
                plab_b = get_cut_mask(unoutput_b, nms=1)
                plab_a_fg = get_cut_mask(unoutput_a_fg, nms=1)
                plab_b_fg = get_cut_mask(unoutput_b_fg, nms=1)
                plab_a_s_bg = get_cut_mask(unoutput_a_bg, nms=1)
                plab_b_s_bg = get_cut_mask(unoutput_b_bg, nms=1)
                proi_a = build_pseudo_roi_target(
                    unoutput_a_fg,
                    unoutput_a_global,
                    unoutput_a_bg,
                    roi_mask=unroi_a_mask,
                    roi_threshold=args.roi_threshold,
                )
                proi_b = build_pseudo_roi_target(
                    unoutput_b_fg,
                    unoutput_b_global,
                    unoutput_b_bg,
                    roi_mask=unroi_b_mask,
                    roi_threshold=args.roi_threshold,
                )
                img_mask, loss_mask = context_mask_pancreas(img_a, args.mask_ratio)

            mix_mask = img_mask.unsqueeze(0).unsqueeze(0).float()
            mixl_img = img_a * mix_mask + unimg_a * (1.0 - mix_mask)
            mixu_img = unimg_b * mix_mask + img_b * (1.0 - mix_mask)
            mixl_img_s = img_a_s * mix_mask + unimg_a_s * (1.0 - mix_mask)
            mixu_img_s = unimg_b_s * mix_mask + img_b_s * (1.0 - mix_mask)

            mixl_lab = lab_a * img_mask + plab_a * (1 - img_mask)
            mixu_lab = plab_b * img_mask + lab_b * (1 - img_mask)

            roi_lab_a = build_roi_target((lab_a == 1).long(), args.roi_kernel)
            roi_lab_b = build_roi_target((lab_b == 1).long(), args.roi_kernel)
            mixl_roi = roi_lab_a * mix_mask + proi_a * (1.0 - mix_mask)
            mixu_roi = proi_b * mix_mask + roi_lab_b * (1.0 - mix_mask)

            (
                outputs_l_fg,
                outputs_l,
                outputs_l_bg,
                _,
                _,
                outputs_l_global,
                roi_l_logits,
                roi_l_mask,
                _,
            ) = model(mixl_img, mixl_img_s)
            (
                outputs_u_fg,
                outputs_u,
                outputs_u_bg,
                _,
                _,
                outputs_u_global,
                roi_u_logits,
                roi_u_mask,
                _,
            ) = model(mixu_img, mixu_img_s)

            output_mix_bg_fg = torch.cat([outputs_l, outputs_u], dim=0)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss_l = mix_loss(outputs_l_fg, lab_a, plab_a_fg, loss_mask, u_weight=args.u_weight)
            loss_u = mix_loss(outputs_u_fg, plab_b_fg, lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
            loss_l_bg = mix_loss(outputs_l_bg, lab_a_s_bg, plab_a_s_bg, loss_mask, u_weight=args.u_weight)
            loss_u_bg = mix_loss(outputs_u_bg, plab_b_s_bg, lab_b_s_bg, loss_mask, u_weight=args.u_weight, unlab=True)
            loss_l_fuse = mix_loss(outputs_l, lab_a, plab_a, loss_mask, u_weight=args.u_weight)
            loss_u_fuse = mix_loss(outputs_u, plab_b, lab_b, loss_mask, u_weight=args.u_weight, unlab=True)

            pos_patches, neg_patches = select_patches_for_contrast_3d(
                output_mix_bg_fg,
                topnum=args.topnum,
                patch_size=args.contrast_patch,
                choose_largest=False,
            )
            loss_bcl = bcl_loss(pos_patches, neg_patches)

            if iter_num >= args.roi_warmup:
                loss_prop_l = F.binary_cross_entropy_with_logits(roi_l_logits, mixl_roi)
                loss_prop_u = F.binary_cross_entropy_with_logits(roi_u_logits, mixu_roi)
                loss_roi_l, _, _ = roi_seg_loss(outputs_l, mixl_lab, mixl_roi, dice_loss)
                loss_roi_u, _, _ = roi_seg_loss(outputs_u, mixu_lab, mixu_roi, dice_loss)
            else:
                loss_prop_l = outputs_l.new_tensor(0.0)
                loss_prop_u = outputs_l.new_tensor(0.0)
                loss_roi_l = outputs_l.new_tensor(0.0)
                loss_roi_u = outputs_l.new_tensor(0.0)

            loss = (
                loss_l
                + loss_u
                + loss_l_bg
                + loss_u_bg
                + args.fuse_loss_weight * (loss_l_fuse + loss_u_fuse)
                + consistency_weight * loss_bcl
                + args.proposal_loss_weight * (loss_prop_l + consistency_weight * loss_prop_u)
                + args.roi_loss_weight * (loss_roi_l + consistency_weight * loss_roi_u)
            )

            iter_num += 1
            writer.add_scalar('Self/consistency', consistency_weight, iter_num)
            writer.add_scalar('Self/loss_l', loss_l, iter_num)
            writer.add_scalar('Self/loss_u', loss_u, iter_num)
            writer.add_scalar('Self/loss_l_bg', loss_l_bg, iter_num)
            writer.add_scalar('Self/loss_u_bg', loss_u_bg, iter_num)
            writer.add_scalar('Self/loss_l_fuse', loss_l_fuse, iter_num)
            writer.add_scalar('Self/loss_u_fuse', loss_u_fuse, iter_num)
            writer.add_scalar('Self/loss_roi_l', loss_roi_l, iter_num)
            writer.add_scalar('Self/loss_roi_u', loss_roi_u, iter_num)
            writer.add_scalar('Self/loss_prop_l', loss_prop_l, iter_num)
            writer.add_scalar('Self/loss_prop_u', loss_prop_u, iter_num)
            writer.add_scalar('Self/roi_l_mean', roi_l_mask.mean(), iter_num)
            writer.add_scalar('Self/roi_u_mean', roi_u_mask.mean(), iter_num)
            writer.add_scalar('Self/bclloss', loss_bcl, iter_num)
            writer.add_scalar('Self/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, 0.99)

            logging.info(
                'iteration %d : loss=%03f, loss_l=%03f, loss_u=%03f, loss_fuse=%03f, loss_roi=%03f, loss_bcl=%03f',
                iter_num,
                loss,
                loss_l,
                loss_u,
                loss_l_fuse + loss_u_fuse,
                loss_roi_l + loss_roi_u,
                loss_bcl,
            )

            if iter_num % 200 == 0:
                model.eval()
                ema_model.eval()
                dice_sample = test_3d_patch.var_all_case_Pancreas_argument(
                    model,
                    num_classes=num_classes,
                    patch_size=patch_size,
                    stride_xy=16,
                    stride_z=16,
                    dataset_path=args.root_path,
                )
                ema_dice_sample = test_3d_patch.var_all_case_Pancreas_argument(
                    ema_model,
                    num_classes=num_classes,
                    patch_size=patch_size,
                    stride_xy=16,
                    stride_z=16,
                    dataset_path=args.root_path,
                )
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
                _, _, height, width, depth = outputs_l.size()
                snapshot_img = torch.zeros(size=(depth, 3, 3 * height + 3 * ins_width, width + ins_width), dtype=torch.float32)
                snapshot_img[:, :, height:height + ins_width, :] = 1
                snapshot_img[:, :, 2 * height + ins_width:2 * height + 2 * ins_width, :] = 1
                snapshot_img[:, :, 3 * height + 2 * ins_width:3 * height + 3 * ins_width, :] = 1
                snapshot_img[:, :, :, width:width + ins_width] = 1

                outputs_l_soft = F.softmax(outputs_l, dim=1)
                seg_out = outputs_l_soft[0, 1, ...].permute(2, 0, 1)
                target = mixl_lab[0, ...].permute(2, 0, 1)
                train_img = mixl_img[0, 0, ...].permute(2, 0, 1)

                norm_train_img = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img) + 1e-6)
                snapshot_img[:, 0, :height, :width] = norm_train_img
                snapshot_img[:, 1, :height, :width] = norm_train_img
                snapshot_img[:, 2, :height, :width] = norm_train_img
                snapshot_img[:, 0, height + ins_width:2 * height + ins_width, :width] = target
                snapshot_img[:, 1, height + ins_width:2 * height + ins_width, :width] = target
                snapshot_img[:, 2, height + ins_width:2 * height + ins_width, :width] = target
                snapshot_img[:, 0, 2 * height + 2 * ins_width:3 * height + 2 * ins_width, :width] = seg_out
                snapshot_img[:, 1, 2 * height + 2 * ins_width:3 * height + 2 * ins_width, :width] = seg_out
                snapshot_img[:, 2, 2 * height + 2 * ins_width:3 * height + 2 * ins_width, :width] = seg_out
                writer.add_images(f'Epoch_{epoch}_Iter_{iter_num}_labeled', snapshot_img)

                outputs_u_soft = F.softmax(outputs_u, dim=1)
                seg_out = outputs_u_soft[0, 1, ...].permute(2, 0, 1)
                target = mixu_lab[0, ...].permute(2, 0, 1)
                train_img = mixu_img[0, 0, ...].permute(2, 0, 1)

                norm_train_img = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img) + 1e-6)
                snapshot_img[:, 0, :height, :width] = norm_train_img
                snapshot_img[:, 1, :height, :width] = norm_train_img
                snapshot_img[:, 2, :height, :width] = norm_train_img
                snapshot_img[:, 0, height + ins_width:2 * height + ins_width, :width] = target
                snapshot_img[:, 1, height + ins_width:2 * height + ins_width, :width] = target
                snapshot_img[:, 2, height + ins_width:2 * height + ins_width, :width] = target
                snapshot_img[:, 0, 2 * height + 2 * ins_width:3 * height + 2 * ins_width, :width] = seg_out
                snapshot_img[:, 1, 2 * height + 2 * ins_width:3 * height + 2 * ins_width, :width] = seg_out
                snapshot_img[:, 2, 2 * height + 2 * ins_width:3 * height + 2 * ins_width, :width] = seg_out
                writer.add_images(f'Epoch_{epoch}_Iter_{iter_num}_unlabel', snapshot_img)

            if iter_num >= self_max_iterations:
                break
        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    pre_snapshot_path = "{}/{}_{}_labeled/pre_train".format(args.snapshot_path, args.exp, args.labelnum)
    self_snapshot_path = "{}/{}_{}_labeled/self_train".format(args.snapshot_path, args.exp, args.labelnum)
    print("Starting Pancreas training with SKC3D + foreground refinement.")

    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')

    configure_logging(pre_snapshot_path + "/log.txt")
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    configure_logging(self_snapshot_path + "/log.txt")
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)
