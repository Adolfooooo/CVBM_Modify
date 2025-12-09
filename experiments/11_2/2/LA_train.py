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
from dataloaders.dataset import LAHeart, RandomRotFlip, RandomCrop, ToTensor, TwoStreamBatchSampler, DualAugmentTransform
from dataloaders.datasets_3d import WeakStrongAugment3d
from utils.util import compute_sdf, compute_sdf_bg
from utils.BCP_utils import context_mask, mix_loss, update_ema_variables
from .abc_skc3d_module import CVBMArgumentWithSKC3DABC

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/LA', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='CVBM_LA', help='exp_name')
parser.add_argument('--model', type=str, default='CVBM_Argument', help='model_name')
parser.add_argument('--pre_max_iteration', type=int, default=2000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int, default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--patch_size', type=tuple, default=(112, 112, 80), help='patch_size of loading image')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=0, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=8, help='trained samples')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_workers', type=int, default=8, help='cpu core num_workers')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default='10.0', help='magnitude')
# -- setting of BANET
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2 / 3, help='ratio of mask/image')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
parser.add_argument('--beta', type=float, default=0.3, help='balance factor to control regional and sdm loss')
parser.add_argument('--head_ema', type=float, default=0.99, help='EMA momentum for main head <- ABC head')
parser.add_argument('--snapshot_path', type=str, default='./experiments/11_2/2/results/', help='snapshot path to save model')
args = parser.parse_args()
torch.backends.cudnn.benchmark = True

def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC(masks)
    return masks


def get_cut_mask_bg(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs <= thres).type(torch.int64)
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


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def compute_class_ratio(dataset, num_cls=2, num_sample=4):
    class_size = [0 for _ in range(num_cls)]
    for idx, sample in enumerate(dataset):
        label = sample["label"]
        if torch.is_tensor(label):
            label = label.cpu().numpy()
        for c in range(num_cls):
            class_size[c] += np.sum(label == c)
        if idx + 1 >= num_sample:
            break
    ir2 = min(class_size) / np.array(class_size)
    return torch.tensor(ir2, dtype=torch.float32).cuda()


def make_balance_mask(label_tensor, ratio_vec):
    """Build Bernoulli mask per voxel based on class frequency."""
    prob = label_tensor.to(dtype=torch.float32)
    for i in range(ratio_vec.shape[0]):
        prob = torch.where(prob == i, ratio_vec[i], prob)
    mask = torch.bernoulli(prob)
    return mask


def dice_ce_with_mask(logits, target, mask=None):
    target = target.long()
    dice_loss = DICE_MASK(logits, target, mask)
    ce_raw = CE(logits, target)
    if mask is None:
        ce_loss = ce_raw.mean()
    else:
        ce_loss = (ce_raw * mask).sum() / (mask.sum() + 1e-6)
    return 0.5 * (dice_loss + ce_loss)


@torch.no_grad()
def ema_head_from_abc(decoder, alpha):
    """EMA update: main head parameters track the ABC head."""
    if not hasattr(decoder, "out_conv") or not hasattr(decoder, "out_conv_abc"):
        return

    abc_params = dict(decoder.out_conv_abc.named_parameters())
    for name, param in decoder.out_conv.named_parameters():
        if name in abc_params and abc_params[name].shape == param.shape:
            param.data.mul_(alpha).add_((1 - alpha) * abc_params[name].data)

    abc_buffers = dict(decoder.out_conv_abc.named_buffers())
    for name, buf in decoder.out_conv.named_buffers():
        if name in abc_buffers and abc_buffers[name].shape == buf.shape:
            buf.data.mul_(alpha).add_((1 - alpha) * abc_buffers[name].data)


def select_patches_for_contrast_3d(output_mix, topnum=16, patch_size=(4, 4, 4), choose_largest=False):
    """
    output_mix: [B, C, H, W, D] (e.g. 3D segmentation logits)
    返回:
      pos_patches: [B, topnum, C]
      neg_patches: [B, L-topnum, C]
    """
    B, C, H, W, D = output_mix.shape
    ph, pw, pd = patch_size
    assert H % ph == 0 and W % pw == 0 and D % pd == 0, "H/W/D 必须能被 patch_size 整除"

    # 1) softmax 概率 + 置信度
    probs = F.softmax(output_mix, dim=1)                # [B, C, H, W, D]
    score = probs.max(dim=1, keepdim=True).values       # [B, 1, H, W, D]

    # 2) 用 avg_pool3d 得到每个 patch 的均值置信度
    patch_conf = F.avg_pool3d(score, kernel_size=(ph, pw, pd), stride=(ph, pw, pd))  # [B, 1, H/ph, W/pw, D/pd]
    patch_conf = patch_conf.flatten(1)   # [B, L], L=patch_num

    # 3) 选取 topnum patch index
    top_vals, top_idx = patch_conf.topk(topnum, dim=1, largest=choose_largest)  # [B, topnum]
    B_, L = patch_conf.shape
    all_idx = torch.arange(L, device=output_mix.device).unsqueeze(0).expand(B_, -1)
    mask = torch.ones_like(all_idx, dtype=torch.bool)
    mask.scatter_(1, top_idx, False)   # 正样本位置设为 False
    num_neg = L - topnum

    # 4) 提取每个 patch 的特征（用 avg_pool3d）
    feat_patches = F.avg_pool3d(probs, kernel_size=(ph, pw, pd), stride=(ph, pw, pd))  # [B, C, H/ph, W/pw, D/pd]
    feat_patches = feat_patches.flatten(2).permute(0, 2, 1).contiguous()  # [B, L, C]

    # 5) 正负样本提取
    pos_patches = torch.gather(
        feat_patches, dim=1,
        index=top_idx.unsqueeze(-1).expand(-1, -1, C)   # [B, topnum, C]
    )
    neg_patches = feat_patches[mask].view(B, num_neg, C)

    return pos_patches, neg_patches


train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')
DICE_MASK = losses.mask_DiceLoss(nclass=2)

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2


def pre_train(args, snapshot_path):
    model = CVBMArgumentWithSKC3DABC(
        n_channels=1,
        n_classes=num_classes,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                            WeakStrongAugment3d(args.patch_size, flag_rot=False)
                       ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    sub_bs = int(args.labeled_bs / 2)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train, batch_sampler=batch_sampler, 
        num_workers=args.num_workers, 
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        pin_memory_device="cuda",     # PyTorch 2.0+
        # prefetch_factor=4,            # 可试 2~4
        persistent_workers=True, 
        )
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=2)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            model.train()
            volume_batch, label_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][
                                                                                  :args.labeled_bs]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:]
            with torch.no_grad():
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

            """Mix Input"""
            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)

            outputs_fg, outputs_fg_abc, outputs, outputs_bg, outputs_bg_abc, out_tanh, out_tanh_bg = model(volume_batch, volume_batch)
            loss_seg = 0
            loss_seg_dice = 0
            loss_sdf = 0

            y2 = outputs_fg[:args.labeled_bs, ...]

            y_prob2 = F.softmax(y2, dim=1)
            loss_seg += F.cross_entropy(y2[:args.labeled_bs], (label_batch[:args.labeled_bs, ...] == 1).long())
            loss_seg_dice += DICE(y_prob2, label_batch[:args.labeled_bs, ...] == 1)

            y_bg = outputs_bg[:args.labeled_bs, ...]
            y_prob_bg = F.softmax(y_bg, dim=1)
            loss_seg += F.cross_entropy(y_bg[:args.labeled_bs], (label_batch[:args.labeled_bs, ...] == 0).long())
            loss_seg_dice += DICE(y_prob_bg, label_batch[:args.labeled_bs, ...] == 0)

            loss = (loss_seg + loss_seg_dice) / 2

            iter_num += 1
            writer.add_scalar('pre/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('pre/loss_seg', loss_seg, iter_num)
            writer.add_scalar('pre/loss_sdf', loss_seg, iter_num)
            writer.add_scalar('pre/loss_all', loss, iter_num)
            logging.info("y_prob2: {}, label_batch: {}".format(torch.argmax(y_prob2, dim=1).sum(), label_batch.sum()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f,loss_sdf: %03f' % (
            iter_num, loss, loss_seg_dice, loss_seg, loss_sdf))

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA_argument(model, num_classes=num_classes, patch_size=patch_size,
                                                            stride_xy=18, stride_z=4, dataset_path=args.root_path)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                    # torch.save(model.state_dict(), save_mode_path)
                    # torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)

            if iter_num >= pre_max_iterations:
                break

        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(args, pre_snapshot_path, self_snapshot_path):
    model = CVBMArgumentWithSKC3DABC(
        n_channels=1,
        n_classes=num_classes,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()
    ema_model = CVBMArgumentWithSKC3DABC(
        n_channels=1,
        n_classes=num_classes,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()
    for param in ema_model.parameters():
        param.detach_()

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                            WeakStrongAugment3d(args.patch_size, flag_rot=False)
                       ]))

    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    sub_bs = int(args.labeled_bs / 2)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        pin_memory_device="cuda",
        persistent_workers=True
        )
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    BCLLoss = losses.BlockContrastiveLoss()

    pretrained_model = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')
    load_net(model, pretrained_model)
    load_net(ema_model, pretrained_model)

    class_ratio = compute_class_ratio(db_train, num_cls=num_classes, num_sample=4)

    writer = SummaryWriter(self_snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0

    max_epoch = self_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    best_dice = 0
    ema_best_dice = 0

    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            model.train()
            ema_model.train()

            # head-level EMA (main <- ABC) for both student and teacher
            ema_head_from_abc(model.decoder_fg, args.head_ema)
            ema_head_from_abc(model.decoder_bg, args.head_ema)
            ema_head_from_abc(ema_model.decoder_fg, args.head_ema)
            ema_head_from_abc(ema_model.decoder_bg, args.head_ema)

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:args.labeled_bs]
            lab_a_bg, lab_b_bg = label_batch[:sub_bs] == 0, label_batch[sub_bs:args.labeled_bs] == 0
            unimg_a, unimg_b = volume_batch[args.labeled_bs:args.labeled_bs + sub_bs], volume_batch[
                                                                                       args.labeled_bs + sub_bs:]
            volume_batch_strong, label_batch_strong = sampled_batch['image_strong'], sampled_batch['label_strong']
            volume_batch_strong, label_batch_strong = volume_batch_strong.cuda(), label_batch_strong.cuda()
            img_a_s, img_b_s = volume_batch_strong[:sub_bs], volume_batch_strong[sub_bs:args.labeled_bs]
            lab_a_s, lab_b_s = label_batch_strong[:sub_bs], label_batch_strong[sub_bs:args.labeled_bs]
            lab_a_s_bg, lab_b_s_bg = label_batch_strong[:sub_bs] == 0, label_batch_strong[sub_bs:args.labeled_bs] == 0
            unimg_a_s, unimg_b_s = volume_batch_strong[args.labeled_bs:args.labeled_bs + sub_bs], volume_batch_strong[
                                                                                       args.labeled_bs + sub_bs:]

            with torch.no_grad():
                unoutput_a_fg, unoutput_a_fg_abc, unoutput_a, unoutput_a_bg, unoutput_a_bg_abc, _, _ = ema_model(unimg_a, unimg_a_s)
                unoutput_b_fg, unoutput_b_fg_abc, unoutput_b, unoutput_b_bg, unoutput_b_bg_abc, _, _ = ema_model(unimg_b, unimg_b_s)
                plab_a = get_cut_mask(unoutput_a, nms=1)
                plab_b = get_cut_mask(unoutput_b, nms=1)
                plab_a_fg = get_cut_mask(unoutput_a_fg, nms=1)
                plab_b_fg = get_cut_mask(unoutput_b_fg, nms=1)
                plab_a_s_bg = get_cut_mask(unoutput_a_bg, nms=1)
                plab_b_s_bg = get_cut_mask(unoutput_b_bg, nms=1)
                plab_a_fg_abc = torch.argmax(unoutput_a_fg_abc, dim=1, keepdim=True).long()
                plab_b_fg_abc = torch.argmax(unoutput_b_fg_abc, dim=1, keepdim=True).long()
                plab_a_bg_abc = torch.argmax(unoutput_a_bg_abc, dim=1, keepdim=True).long()
                plab_b_bg_abc = torch.argmax(unoutput_b_bg_abc, dim=1, keepdim=True).long()
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

            mixl_img = img_a * img_mask + unimg_a * (1 - img_mask)
            mixu_img = unimg_b * img_mask + img_b * (1 - img_mask)
            mixl_img_s = img_a_s * img_mask + unimg_a_s * (1 - img_mask)
            mixu_img_s = unimg_b_s * img_mask + img_b_s * (1 - img_mask)
            mixl_lab = lab_a * img_mask + plab_a * (1 - img_mask)
            mixu_lab = plab_b * img_mask + lab_b * (1 - img_mask)
            mixl_lab_bg = lab_a_s_bg * img_mask + plab_a_s_bg * (1 - img_mask)
            mixu_lab_bg = plab_b_s_bg * img_mask + lab_b_s_bg * (1 - img_mask)

            outputs_l_fg, outputs_l_fg_abc, outputs_l, outputs_l_bg, outputs_l_bg_abc, sdm_outputs_l, sdm_outputs_l_bg = model(mixl_img, mixl_img_s)
            outputs_u_fg, outputs_u_fg_abc, outputs_u, outputs_u_bg, outputs_u_bg_abc, sdm_outputs_u, sdm_outputs_u_bg = model(mixu_img, mixu_img_s)

            output_mix_bg_fg = torch.cat([outputs_l, outputs_u], dim=0)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss_l = mix_loss(outputs_l_fg, lab_a, plab_a_fg, loss_mask, u_weight=args.u_weight)
            loss_u = mix_loss(outputs_u_fg, plab_b_fg, lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
            loss_l_bg = mix_loss(outputs_l_bg, lab_a_s_bg, plab_a_s_bg, loss_mask, u_weight=args.u_weight)
            loss_u_bg = mix_loss(outputs_u_bg, plab_b_s_bg, lab_b_s_bg, loss_mask, u_weight=args.u_weight, unlab=True)

            progress = min(1.0, iter_num / self_max_iterations)
            balance_ratio = 1 - progress * (1 - class_ratio)

            mask_fg_l = make_balance_mask(mixl_lab, balance_ratio)
            mask_fg_u = make_balance_mask(mixu_lab, balance_ratio)
            mask_bg_l = make_balance_mask(mixl_lab_bg, balance_ratio)
            mask_bg_u = make_balance_mask(mixu_lab_bg, balance_ratio)

            loss_l_fg_abc = dice_ce_with_mask(outputs_l_fg_abc, mixl_lab, mask_fg_l)
            loss_u_fg_abc = dice_ce_with_mask(outputs_u_fg_abc, mixu_lab, mask_fg_u)
            loss_l_bg_abc = dice_ce_with_mask(outputs_l_bg_abc, mixl_lab_bg, mask_bg_l)
            loss_u_bg_abc = dice_ce_with_mask(outputs_u_bg_abc, mixu_lab_bg, mask_bg_u)

            pos_patches, neg_patches = select_patches_for_contrast_3d(output_mix_bg_fg, topnum=440, patch_size=(4, 4, 20), choose_largest=False)
            bclloss = BCLLoss(pos_patches, neg_patches)

            loss = (loss_l + loss_u + loss_l_bg + loss_u_bg +
                    loss_l_fg_abc + loss_u_fg_abc + loss_l_bg_abc + loss_u_bg_abc +
                    consistency_weight * bclloss)

            iter_num += 1
            writer.add_scalar('Self/consistency', consistency_weight, iter_num)
            writer.add_scalar('Self/loss_l', loss_l, iter_num)
            writer.add_scalar('Self/loss_u', loss_u, iter_num)
            writer.add_scalar('Self/loss_l_bg', loss_l_bg, iter_num)
            writer.add_scalar('Self/loss_u_bg', loss_u_bg, iter_num)
            writer.add_scalar('Self/loss_l_fg_abc', loss_l_fg_abc, iter_num)
            writer.add_scalar('Self/loss_u_fg_abc', loss_u_fg_abc, iter_num)
            writer.add_scalar('Self/loss_l_bg_abc', loss_l_bg_abc, iter_num)
            writer.add_scalar('Self/loss_u_bg_abc', loss_u_bg_abc, iter_num)
            writer.add_scalar('Self/bclloss', bclloss, iter_num)
            writer.add_scalar('Self/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info(
                'iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f, loss_bcl: %03f, loss_fgabc: %03f, loss_bgabc: %03f' % (
                iter_num, loss, loss_l, loss_u, bclloss, loss_l_fg_abc + loss_u_fg_abc, loss_l_bg_abc + loss_u_bg_abc))

            update_ema_variables(model, ema_model, 0.99)

            if iter_num % 200 == 0:
                model.eval()
                ema_model.eval()
                dice_sample = test_3d_patch.var_all_case_LA_argument(model, num_classes=num_classes, patch_size=patch_size,
                                                            stride_xy=18, stride_z=4, dataset_path=args.root_path)
                ema_dice_sample = test_3d_patch.var_all_case_LA_argument(ema_model, num_classes=num_classes, patch_size=patch_size,
                                                            stride_xy=18, stride_z=4, dataset_path=args.root_path)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 7)
                    save_mode_path = os.path.join(self_snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(self_snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                if ema_dice_sample > ema_best_dice:
                    ema_best_dice = round(ema_dice_sample, 7)
                    save_mode_path = os.path.join(self_snapshot_path, 'iter_{}_ema_dice_{}.pth'.format(iter_num, ema_best_dice))
                    save_ema_best_path = os.path.join(self_snapshot_path, '{}_ema_best_model.pth'.format(args.model))
                    torch.save(ema_model.state_dict(), save_mode_path)
                    torch.save(ema_model.state_dict(), save_ema_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
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
                seg_out = outputs_l_soft[0, 1, ...].permute(2, 0, 1)  # y
                target = mixl_lab[0, ...].permute(2, 0, 1)
                train_img = mixl_img[0, 0, ...].permute(2, 0, 1)

                snapshot_img[:, 0, :H, :W] = (train_img - torch.min(train_img)) / (
                            torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 1, :H, :W] = (train_img - torch.min(train_img)) / (
                            torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 2, :H, :W] = (train_img - torch.min(train_img)) / (
                            torch.max(train_img) - torch.min(train_img))

                snapshot_img[:, 0, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 1, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 2, H + ins_width:2 * H + ins_width, :W] = target

                snapshot_img[:, 0, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 1, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 2, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out

                writer.add_images('Epoch_%d_Iter_%d_labeled' % (epoch, iter_num), snapshot_img)

                outputs_u_soft = F.softmax(outputs_u, dim=1)
                seg_out = outputs_u_soft[0, 1, ...].permute(2, 0, 1)  # y
                target = mixu_lab[0, ...].permute(2, 0, 1)
                train_img = mixu_img[0, 0, ...].permute(2, 0, 1)

                snapshot_img[:, 0, :H, :W] = (train_img - torch.min(train_img)) / (
                            torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 1, :H, :W] = (train_img - torch.min(train_img)) / (
                            torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 2, :H, :W] = (train_img - torch.min(train_img)) / (
                            torch.max(train_img) - torch.min(train_img))

                snapshot_img[:, 0, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 1, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 2, H + ins_width:2 * H + ins_width, :W] = target

                snapshot_img[:, 0, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 1, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 2, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out

                writer.add_images('Epoch_%d_Iter_%d_unlabel' % (epoch, iter_num), snapshot_img)

            if iter_num >= self_max_iterations:
                break

        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    ## make logger file
    pre_snapshot_path = "{}/{}_{}_labeled/pre_train".format(args.snapshot_path, args.exp, args.labelnum)
    self_snapshot_path = "{}/{}_{}_labeled/self_train".format(args.snapshot_path, args.exp, args.labelnum)
    print("Starting ABC-SKC training (fg->bg + head EMA).")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy(os.path.abspath(__file__), self_snapshot_path)
    # # -- Pre-Training
    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)
    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)
