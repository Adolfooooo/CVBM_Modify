import argparse
import logging

import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label
from einops import rearrange

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, CreateOnehotLabel, WeakStrongAugment)
from networks.net_factory import net_factory
from utils import losses, ramps, feature_memory, contrastive_losses, val_2d, create_onehot
from utils.dynamic_threhold.DynamicThresholdUpdater import DynamicThresholdUpdater
from utils.dynamic_threhold.plo import PseudoLabelOptimizer
from networks.CVBM import CVBM, CVBM_Argument

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/xuminghao/Datasets/ACDC/ACDC_ABD', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='CVBM2d_ACDC', help='experiment_name')
parser.add_argument('--model', type=str, default='CVBM2d_Argument', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=0, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=3, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default='6.0', help=' bbmagnitude')
parser.add_argument('--s_param', type=int, default=6, help='multinum of random masks')
parser.add_argument('--snapshot_path', type=str, default='./results/CVBM_4_6_4_pre_train/1', help='snapshot_path')

args = parser.parse_args()
pre_max_iterations = args.pre_iterations
self_max_iterations = args.max_iterations
dice_loss = losses.DiceLoss(n_classes=4)
onehot_dice_loss = losses.DiceLoss2d(n_classes=4)
onehot_ce_loss=losses.CrossEntropyLoss(n_classes=4)
alpha = 0.99
num_classes = args.num_classes
dynamic_threshold_bg = [1/num_classes for i in range(4)]
dynamic_threshold_fg = [1/num_classes for i in range(4)]
dynamic_threshold_class = [1/num_classes for i in range(args.num_classes)]
plt_bg, plt_fg = {}, {}



def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        # -- with 'try'
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()


def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i]  # == c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)

        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()


def get_ACDC_2DLargestCC_onehot(segmentation):
    '''
    inputs: segementation: BxHxW, indices
    '''
    batch_list = []
    batch_num = segmentation.shape[0]
    for i in range(0, batch_num):
        class_list = []
        for c in range(0, 4):
            temp_seg = segmentation[i]  # == c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)

        batch_list.append(class_list)

    return torch.Tensor(batch_list).cuda()


def get_ACDC_masks(output, nms=0,onehot=False):
    probs = F.softmax(output, dim=1)
    probs, indices = torch.max(probs, dim=1)

    if nms == 1:
        if onehot:
            probs = get_ACDC_2DLargestCC_onehot(indices)
        else:
            probs = get_ACDC_2DLargestCC(indices)
    return probs


def get_ACDC_masks_with_confidence(output, nms=0,onehot=False):
    probs = F.softmax(output, dim=1)
    probs, indices = torch.max(probs, dim=1)
    confidence_foreground_selection(probs, indices, threshold=0.7)
    if nms == 1:
        if onehot:
            indices = get_ACDC_2DLargestCC_onehot(indices)
        else:
            indices = get_ACDC_2DLargestCC(indices)
    return indices


def get_ACDC_masks_with_confidence_dynamic(output , dynamic_threhold_updater, nms=0, onehot=False):
    probs = F.softmax(output, dim=1)
    probs, indices = torch.max(probs, dim=1)
    dynamic_threshold_class = dynamic_threhold_updater.update_threshold(class_num=4, dynamic_thresholds=dynamic_threshold_class, alpha=0.99)
    indices = generate_pseudo_labels_with_confidence(probs, indices, threshold=dynamic_threshold_class)
    if nms == 1:
        if onehot:
            indices = get_ACDC_2DLargestCC_onehot(indices)
        else:
            indices = get_ACDC_2DLargestCC(indices)
    return indices


def confidence_foreground_selection(segmentation, indices, threshold): 
    """
    基于置信度的前景模块选择
    
    Args:

        threshold: list - 置信度阈值
    
    Returns:
        torch.Tensor, shape (B, H, W) - 过滤后的类别标签（低置信度区域设为0背景）
    """
    # 创建置信度掩码：只有高于阈值的像素被认为是前景
    confidence_mask = segmentation > threshold
    
    # 应用置信度掩码：低置信度区域设为背景标签0
    filtered_indices = indices * confidence_mask
    
    return filtered_indices


def generate_pseudo_labels_with_confidence(predictions, class_thresholds):
    """
    根据类别阈值生成伪标签
    
    Args:
            模型的预测输出（softmax 概率或者 sigmoid 后的置信度）。
        class_thresholds: list or torch.Tensor, shape (C,)
            每个类别对应的阈值。
    
    Returns:
        torch.Tensor, shape (B, H, W, D)
            伪标签（低于对应类别阈值的像素设为0）
    """
    B, C, H, W = predictions.shape
    if not torch.is_tensor(class_thresholds):
        class_thresholds = torch.tensor(class_thresholds, device=predictions.device, dtype=predictions.dtype)

    # (B, H, W) 取每个像素的最大概率类别
    probs, pred_classes = torch.max(predictions, dim=1)  # probs: (B,H,W), pred_classes: (B,H,W)

    # 获取每个像素对应的类别阈值
    thresholds = class_thresholds[pred_classes]  # (B,H,W)

    # 判断是否超过类别阈值
    mask = probs > thresholds  # (B,H,W)

    # 应用阈值过滤，低置信度设为背景（0）
    pseudo_labels = pred_classes * mask

    return pseudo_labels


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 5 * args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


def generate_mask(img, number_class):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    onehot_mask = torch.ones(batch_size, number_class, img_x, img_y).cuda()
    patch_x, patch_y = int(img_x * 2 / 3), int(img_y * 2 / 3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w + patch_x, h:h + patch_y] = 0
    loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    onehot_mask[:, :, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long(), onehot_mask.long()


def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x * 2 / (3 * shrink_param)), int(img_y * 2 / (3 * shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s * x_split, (x_s + 1) * x_split - patch_x)
            h = np.random.randint(y_s * y_split, (y_s + 1) * y_split - patch_y)
            mask[w:w + patch_x, h:h + patch_y] = 0
            loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long()


def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_y = int(img_y * 4 / 9)
    h = np.random.randint(0, img_y - patch_y)
    mask[h:h + patch_y, :] = 0
    loss_mask[:, h:h + patch_y, :] = 0
    return mask.long(), loss_mask.long()


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)  # loss = loss_ce
    return loss_dice, loss_ce


def onehot_mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    # CE = CrossEntropyLoss
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = onehot_dice_loss(output_soft, img_l, mask) * image_weight
    loss_dice += onehot_dice_loss(output_soft, patch_l, patch_mask) * patch_weight
    loss_ce = onehot_ce_loss(output_soft, img_l, mask) * image_weight
    loss_ce += onehot_ce_loss(output_soft, patch_l, patch_mask) * patch_weight  # loss = loss_ce

    return loss_dice, loss_ce

### provide for label data to get sclice num
def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def select_patches_for_contrast(output_mix, topnum=16, patch_size=(4, 4), choose_largest=False):
    """
    output_mix: [B, C, H, W]（模型预测输出，示例中用 softmax 概率作为对比特征）
    返回:
      pos_patches: [B, topnum, C]
      neg_patches: [B, L-topnum, C]
    """
    B, C, H, W = output_mix.shape
    ph, pw = patch_size
    assert H % ph == 0 and W % pw == 0, "H/W 必须能被 patch_size 整除（不重叠 patch）"

    # 1) 概率 & 置信度图
    probs = F.softmax(output_mix, dim=1)          # [B, C, H, W]
    score = probs.max(dim=1, keepdim=True).values # [B, 1, H, W]

    # 2) 用 unfold 把置信度图切 patch，并做均值 -> 每个 patch 的置信度
    # unfold 输出 [B, patch_area, L]，L 为 patch 个数
    score_patches = F.unfold(score, kernel_size=(ph, pw), stride=(ph, pw))  # [B, ph*pw, L]
    patch_conf = score_patches.mean(dim=1)                                   # [B, L]

    # 3) 选取最低置信度的 topnum 作为正样本，其余为负样本
    top_vals, top_idx = patch_conf.topk(topnum, dim=1, largest=choose_largest)        # [B, topnum]
    B_, L = patch_conf.shape
    all_idx = torch.arange(L, device=output_mix.device).unsqueeze(0).expand(B_, -1)  # [B, L]
    mask = torch.ones_like(all_idx, dtype=torch.bool)                         # [B, L]
    mask.scatter_(1, top_idx, False)                                          # 正样本位置设为 False
    # 剩余的就是负样本
    num_neg = L - topnum

    # 4) 切特征图并对齐到 [B, L, C]（每个 patch 一个向量，features_dim=C）
    # 对概率图做 unfold: [B, C*ph*pw, L] -> [B, C, ph*pw, L] -> 在 patch 内做均值 -> [B, C, L] -> [B, L, C]
    feat_patches = F.unfold(probs, kernel_size=(ph, pw), stride=(ph, pw))     # [B, C*ph*pw, L]
    feat_patches = feat_patches.view(B, C, ph*pw, L).mean(dim=2)              # [B, C, L]
    feat_patches = feat_patches.permute(0, 2, 1).contiguous()                 # [B, L, C]

    # 5) 基于索引/掩码取出正负 patch，得到 [B, patchnum, features_dim]
    # 正样本（最低置信度的 topnum 个）
    pos_patches = torch.gather(
        feat_patches, dim=1,
        index=top_idx.unsqueeze(-1).expand(-1, -1, C)                         # [B, topnum, C]
    )                                                                          # -> [B, topnum, C]

    # 负样本（其余 patch）
    # 用布尔掩码批量选择，再 reshape 回 [B, L-topnum, C]
    neg_patches = feat_patches[mask].view(B, num_neg, C)                      # [B, L-topnum, C]

    return pos_patches, neg_patches


def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)
        
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train_4_1")
    # model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([
                                WeakStrongAugment(args.patch_size),
                                CreateOnehotLabel(args.num_classes)
                            ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch, onehot_label_batch = sampled_batch['image'], sampled_batch['label'], \
                                                            sampled_batch['onehot_label']
            volume_batch, label_batch, onehot_label_batch = volume_batch.cuda(), label_batch.cuda(), onehot_label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            onehot_lab_a, onehot_lab_b = onehot_label_batch[:labeled_sub_bs] == 0, onehot_label_batch[
                                                                                   labeled_sub_bs:args.labeled_bs] == 0
            img_mask, loss_mask, onehot_mask = generate_mask(img_a, args.num_classes)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)
            onehot_gt_mixl = onehot_lab_a * img_mask + onehot_lab_b * (1 - img_mask)
            # -- original
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl_fg,out_mixl, outputs_mixl_bg, _, _ = model(net_input, net_input)
            loss_dice, loss_ce = mix_loss(out_mixl_fg, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)
            loss_dice_bg, loss_ce_bg = onehot_mix_loss(outputs_mixl_bg, onehot_lab_a, onehot_lab_b, onehot_mask, u_weight=1.0,
                                                       unlab=True)
            loss = (loss_dice + loss_dice_bg + loss_ce + loss_ce_bg) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' % (iter_num, loss, loss_dice, loss_ce))

            if iter_num % 20 == 0:
                image = net_input[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = gt_mixl[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume_argument(sampled_batch["image"], sampled_batch["label"], model,
                                                         classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(args, pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    ema_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    for param in ema_model.parameters():
        param.detach_()  # ema_model set

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([
                                WeakStrongAugment(args.patch_size),
                                CreateOnehotLabel(args.num_classes)
                            ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))

    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    # load_net(ema_model, pre_trained_model)
    # load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    consistency_criterion = losses.mse_loss
    BCLLoss = losses.BlockContrastiveLoss()
    ce_loss = CrossEntropyLoss()
    pseudo_label_optimizer = PseudoLabelOptimizer()

    iter_num = 0
    max_epoch = self_max_iterations // len(trainloader) + 1
    best_performance = 0.0
    ema_best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)    
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            model.train()
            ema_model.train()
            # volume_batch.shape, label_batch.shape, onehot_label_batch.shape
            # torch.Size([24, 1, 256, 256]) torch.Size([24, 256, 256]) torch.Size([24, 4, 256, 256])
            volume_batch, label_batch, onehot_label_batch = sampled_batch['image'], sampled_batch['label'], \
                                                            sampled_batch['onehot_label']
            volume_batch_strong, label_batch_strong, onehot_label_batch_strong = \
                sampled_batch['image_strong'], sampled_batch['label_strong'], sampled_batch['onehot_label_strong']
            volume_batch, label_batch, onehot_label_batch = volume_batch.cuda(), label_batch.cuda(), onehot_label_batch.cuda()
            volume_batch_strong, label_batch_strong, onehot_label_batch_strong = \
                volume_batch_strong.cuda(), label_batch_strong.cuda(), onehot_label_batch_strong.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[
                                                                                                args.labeled_bs + unlabeled_sub_bs:]
            ulab_a, ulab_b = label_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], label_batch[
                                                                                                args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            lab_a_bg, lab_b_bg = onehot_label_batch[:labeled_sub_bs] == 0, onehot_label_batch[
                                                                            labeled_sub_bs:args.labeled_bs] == 0

            img_a_s, img_b_s = volume_batch_strong[:labeled_sub_bs], volume_batch_strong[labeled_sub_bs:args.labeled_bs]
            uimg_a_s, uimg_b_s = volume_batch_strong[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch_strong[args.labeled_bs + unlabeled_sub_bs:]
            ulab_a_s, ulab_b_s = label_batch_strong[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], label_batch_strong[
                                                                                                args.labeled_bs + unlabeled_sub_bs:]
            lab_a_s, lab_b_s = label_batch_strong[:labeled_sub_bs], label_batch_strong[labeled_sub_bs:args.labeled_bs]
            lab_a_bg_s, lab_b_bg_s = onehot_label_batch_strong[:labeled_sub_bs] == 0, onehot_label_batch_strong[
                                                                            labeled_sub_bs:args.labeled_bs] == 0
            
            with torch.no_grad():
                pre_a_fg,pre_a, pre_a_bg_s, _, _ = ema_model(uimg_a, uimg_a_s)
                pre_b_fg,pre_b, pre_b_bg_s, _, _ = ema_model(uimg_b, uimg_b_s)

                plab_a_fg = get_ACDC_masks_with_confidence(pre_a_fg, nms=1)
                plab_b_fg = get_ACDC_masks_with_confidence(pre_b_fg, nms=1)

                plab_a_bg_s = get_ACDC_masks_with_confidence(pre_a_bg_s, nms=1,onehot=True)
                plab_b_bg_s = get_ACDC_masks_with_confidence(pre_b_bg_s, nms=1,onehot=True)
                
                img_mask, loss_mask, onehot_mask = generate_mask(img_a, args.num_classes)
                unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                l_label = lab_b * img_mask + ulab_b * (1 - img_mask)

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            # input_mix_a, input_mix_b
            # torch.Size([6, 1, 256, 256]) torch.Size([6, 1, 256, 256])
            input_mix_a = uimg_a * img_mask + img_a * (1 - img_mask)
            input_mix_b = img_b * img_mask + uimg_b * (1 - img_mask)
            net_input = torch.cat([input_mix_a, input_mix_b], dim=0)

            input_mix_a_s = uimg_a_s * img_mask + img_a_s * (1 - img_mask)
            input_mix_b_s = img_b_s * img_mask + uimg_b_s * (1 - img_mask)
            net_input_s = torch.cat([input_mix_a_s, input_mix_b_s], dim=0)

            # pseudo label used plo
            # pseudo_label_mix_a = plab_a_fg * img_mask + lab_a * (1 - img_mask)
            # pseudo_label_mix_b = lab_b * img_mask + plab_b_fg * (1 - img_mask)
            # pseudo_label_mix_a_s = plab_a_bg_s * onehot_mask + lab_a_bg_s * (1 - onehot_mask)
            # pseudo_label_mix_b_s = lab_b_bg_s * onehot_mask + plab_b_bg_s * (1 - onehot_mask)
            pseudo_pre_a_fg = torch.argmax(torch.softmax(pre_a_fg, dim=1), dim=1)
            pseudo_pre_b_fg = torch.argmax(torch.softmax(pre_b_fg, dim=1), dim=1)
            pseudo_label_mix_a = pseudo_pre_a_fg * img_mask + lab_a * (1 - img_mask)
            pseudo_label_mix_b = lab_b * img_mask + pseudo_pre_b_fg * (1 - img_mask)
            # pseudo_label_mix_a_s = plab_a_bg_s * onehot_mask + lab_a_bg_s * (1 - onehot_mask)
            # pseudo_label_mix_b_s = lab_b_bg_s * onehot_mask + plab_b_bg_s * (1 - onehot_mask)
            
            # out_mix_a_fg,out_mix_a_fgbg, out_mix_a_s_bg
            # torch.Size([6, 4, 256, 256]) torch.Size([6, 4, 256, 256]) torch.Size([6, 4, 256, 256])
            out_mix_a_fg, out_mix_a_fgbg, out_mix_a_s_bg, _, _ = model(input_mix_a, input_mix_a_s)
            # out_mix_b_fg,out_mix_b_fgbg, out_mix_b_s_bg
            # torch.Size([6, 4, 256, 256]) torch.Size([6, 4, 256, 256]) torch.Size([6, 4, 256, 256])
            out_mix_b_fg, out_mix_b_fgbg, out_mix_b_s_bg, _, _ = model(input_mix_b, input_mix_b_s)

            # conv 3x3 connect
            output_mix = torch.cat([out_mix_a_fgbg, out_mix_b_fgbg], dim=0)

            unl_dice, unl_ce = mix_loss(out_mix_a_fg, plab_a_fg, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice, l_ce = mix_loss(out_mix_b_fg, lab_b, plab_b_fg, loss_mask, u_weight=args.u_weight)

            unl_dice_bg, unl_ce_bg = onehot_mix_loss(out_mix_a_s_bg, plab_a_bg_s, lab_a_bg_s, onehot_mask,
                                                        u_weight=args.u_weight, unlab=True)
            l_dice_bg, l_ce_bg = onehot_mix_loss(out_mix_b_s_bg, lab_b_bg_s, plab_b_bg_s, onehot_mask, u_weight=args.u_weight)

            loss_ce = unl_ce + l_ce + unl_ce_bg+ l_ce_bg
            loss_dice = unl_dice + l_dice + unl_dice_bg + l_dice_bg

            # 
            mask_mix_a_confident, mask_mix_a_hesitant, _ = pseudo_label_optimizer(pseudo_label_mix_a)
            mask_mix_b_confident, mask_mix_b_hesitant, _ = pseudo_label_optimizer(pseudo_label_mix_b)
            # mask_mix_a_s_confident, mask_mix_a_s_hesitant, _ = \
            # pseudo_label_optimizer(create_onehot.OneHotConverter.to_label(pseudo_label_mix_a_s))
            # mask_mix_b_s_confident, mask_mix_b_s_hesitant, _ = \
            # pseudo_label_optimizer(create_onehot.OneHotConverter.to_label(pseudo_label_mix_b_s))

            pse_mix_a = torch.argmax(torch.softmax(out_mix_a_fg, dim=1), dim=1)
            pse_mix_b = torch.argmax(torch.softmax(out_mix_b_fg, dim=1), dim=1)
            loss_mix_a = pseudo_label_optimizer.calculate_loss(pse_mix_a, pseudo_label_mix_a, mask_mix_a_confident, mask_mix_a_hesitant)
            loss_mix_b = pseudo_label_optimizer.calculate_loss(pse_mix_b, pseudo_label_mix_b, mask_mix_b_confident, mask_mix_b_hesitant)
            # loss_mix_a_s = pseudo_label_optimizer.calculate_loss(out_mix_a_s_bg, pseudo_label_mix_a_s, mask_mix_a_s_confident, mask_mix_a_s_hesitant)
            # loss_mix_b_s = pseudo_label_optimizer.calculate_loss(out_mix_b_s_bg, pseudo_label_mix_b_s, mask_mix_b_s_confident, mask_mix_b_s_hesitant)
            loss_hesitant_pse = (loss_mix_a + loss_mix_b) / 2.0

            # contrastive learning of negative patches
            pos_patches, neg_patches = select_patches_for_contrast(output_mix, topnum=64, patch_size=(8, 8))
            # 现在形状满足你的要求：
            #   pos_patches: [B, 16, C] 作为正样本（低置信度）
            #   neg_patches: [B, L-16, C] 作为负样本
            bclloss = BCLLoss(pos_patches, neg_patches)

            loss = loss_dice + loss_ce + consistency_weight * bclloss + loss_hesitant_pse
            # loss =loss_dice + loss_ce + consistency_weight * (loss_consist_l + loss_consist_u)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' % (iter_num, loss, loss_dice, loss_ce))

            if iter_num % 20 == 0:
                image = input_mix_a[1, 0:1, :, :]
                writer.add_image('train/Un_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mix_a_fgbg, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Un_Prediction', outputs[1, ...] * 50, iter_num)
                labs = unl_label[1, ...].unsqueeze(0) * 50
                writer.add_image('train/Un_GroundTruth', labs, iter_num)

                image_l = input_mix_b[1, 0:1, :, :]
                writer.add_image('train/L_Image', image_l, iter_num)
                outputs_l = torch.argmax(torch.softmax(out_mix_b_fgbg, dim=1), dim=1, keepdim=True)
                writer.add_image('train/L_Prediction', outputs_l[1, ...] * 50, iter_num)
                labs_l = l_label[1, ...].unsqueeze(0) * 50
                writer.add_image('train/L_GroundTruth', labs_l, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                ema_model.eval()
                metric_list = 0.0
                ema_metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume_argument(sampled_batch["image"], sampled_batch["label"], model,
                                                         classes=num_classes)
                    ema_metric_i = val_2d.test_single_volume_argument(sampled_batch["image"], sampled_batch["label"], ema_model,
                                                         classes=num_classes)
                    metric_list += np.array(metric_i)
                    ema_metric_list += np.array(ema_metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                ema_performance = np.mean(ema_metric_list, axis=0)[0]
                
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                if ema_performance > ema_best_performance:
                    ema_best_performance = ema_performance
                    ema_save_mode_path = os.path.join(snapshot_path,
                                                  'iter_ema_{}_dice_{}.pth'.format(iter_num, round(ema_best_performance, 4)))
                    ema_save_best_path = os.path.join(snapshot_path, '{}_ema_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), ema_save_mode_path)
                    torch.save(model.state_dict(), ema_save_best_path)
                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                logging.info('iteration %d : mean_dice : %f' % (iter_num, ema_performance))
                

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # -- path to save models
    pre_snapshot_path = "{}/{}_{}_labeled/pre_train".format(args.snapshot_path, args.exp, args.labelnum)
    self_snapshot_path = "{}/{}_{}_labeled/self_train".format(args.snapshot_path, args.exp, args.labelnum)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    # shutil.copy('./just_try/ACDC/ACDC_train_4_6_4_pre_train.py', self_snapshot_path)

    # Pre_train
    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # pre_train(args, pre_snapshot_path)


    # Self_train
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)
