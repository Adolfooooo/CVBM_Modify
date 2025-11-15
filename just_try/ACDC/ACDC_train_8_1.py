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
from networks.CVBM import CVBM, CVBM_Argument
from networks.module import ConfidenceTracker, create_narrow_band

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='CVBM2d_ACDC', help='experiment_name')
parser.add_argument('--model', type=str, default='CVBM2d_Argument', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=0, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=6, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=3, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int, default=6, help='multinum of random masks')
parser.add_argument('--snapshot_path', type=str, default='./results/CVBM_8_1/1', help='snapshot_path')
parser.add_argument('--cgmc_min_ratio', type=float, default=0.3, help='minimum masked area ratio for CGMC')
parser.add_argument('--cgmc_max_ratio', type=float, default=0.5, help='maximum masked area ratio for CGMC')
parser.add_argument('--cgmc_threshold_low', type=float, default=0.55, help='initial confidence threshold')
parser.add_argument('--cgmc_threshold_high', type=float, default=0.75, help='late-stage confidence threshold')
parser.add_argument('--cgmc_threshold_ramp', type=float, default=12000.0, help='iterations to ramp confidence threshold')
parser.add_argument('--cgmc_conf_momentum', type=float, default=0.9, help='EMA momentum for confidence tracker')
parser.add_argument('--cgmc_low_conf_start', type=float, default=0.4, help='initial weight for low-confidence supervision')
parser.add_argument('--cgmc_low_conf_end', type=float, default=0.8, help='late-stage weight for low-confidence supervision')
parser.add_argument('--cgmc_low_conf_ramp', type=float, default=15000.0, help='iterations to ramp low-confidence weight')
parser.add_argument('--band_inner', type=int, default=3, help='inner kernel for boundary band erosion')
parser.add_argument('--band_outer', type=int, default=7, help='outer kernel for boundary band dilation')
parser.add_argument('--band_kl_weight', type=float, default=0.05, help='weight of boundary KL between fg/bg heads')
parser.add_argument('--ema_alpha_high', type=float, default=0.99, help='initial EMA decay')
parser.add_argument('--ema_alpha_low', type=float, default=0.97, help='late-stage EMA decay')
parser.add_argument('--ema_alpha_ramp', type=float, default=15000.0, help='iterations to adjust EMA decay')

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
    confidence_foreground_selection(probs, indices, threshold=0.5)
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


def cgmc_confidence_threshold(iteration, args):
    """
    模块功能: 计算当前迭代下的置信度阈值, 作为CGMC课程的阶段指示。
    输入:
        iteration (int): 当前迭代编号, 标量。
        args (Namespace): 训练参数集合, 内含阈值上下限以及ramp长度。
    输出:
        float: 单值阈值, 范围 [cgmc_threshold_low, cgmc_threshold_high]。
    """
    ramp = min(1.0, iteration / max(1.0, args.cgmc_threshold_ramp))
    return args.cgmc_threshold_low + (args.cgmc_threshold_high - args.cgmc_threshold_low) * ramp


def cgmc_low_conf_scale(iteration, args):
    """
    模块功能: 计算低置信伪标签区域的权重, 逐步从保守到激进。
    输入:
        iteration (int): 当前迭代编号。
        args (Namespace): 训练参数集合, 提供起始/终止权重。
    输出:
        float: 当前迭代下的低置信权重, 落在[start, end]区间。
    """
    ramp = min(1.0, iteration / max(1.0, args.cgmc_low_conf_ramp))
    return args.cgmc_low_conf_start + (args.cgmc_low_conf_end - args.cgmc_low_conf_start) * ramp


def generate_confidence_guided_mask(conf_maps, mean_scores, num_classes, min_ratio, max_ratio):
    """
    模块功能: 根据置信度图生成自适应Mix掩码, 控制遮挡比例与位置。
    输入:
        conf_maps (Tensor): [B, 1, H, W], EMA平滑后的置信度热力图。
        mean_scores (Tensor): [B], 对应样本的平均置信度, 用于调节遮挡面积。
        num_classes (int): 类别数, 生成onehot掩码时需要。
        min_ratio (float): 最小遮挡面积比例。
        max_ratio (float): 最大遮挡面积比例。
    输出:
        img_mask (Tensor): [B, 1, H, W], 值为1表示保留原图, 0表示被混合。
        loss_mask (Tensor): [B, H, W], 与img_mask一致的权重图。
        onehot_mask (Tensor): [B, num_classes, H, W], 供背景onehot损失使用。
    """
    device = conf_maps.device
    batch, _, height, width = conf_maps.shape
    img_masks, loss_masks, onehot_masks = [], [], []
    for idx in range(batch):
        conf = conf_maps[idx, 0]
        mean_conf = mean_scores[idx].clamp(0.0, 1.0).item()
        # 限制调幅：将>0.6的高置信度映射为常数，以避免70%遮挡
        scaled_conf = min(mean_conf, 0.6) / 0.6
        ratio = min_ratio + (max_ratio - min_ratio) * scaled_conf
        ratio = float(np.clip(ratio, min_ratio, max_ratio))
        patch_h = max(1, int(height * np.sqrt(ratio)))
        patch_w = max(1, int(width * np.sqrt(ratio)))

        # 低置信度 -> 从高置信区域采样, 高置信度 -> 从低置信区域采样
        if mean_conf < 0.5:
            preference = conf
        else:
            preference = 1.0 - conf
        flat_pref = preference.flatten()
        topk = min(1024, flat_pref.numel())
        values, indices = torch.topk(flat_pref, topk, largest=True)
        choice = indices[torch.randint(0, topk, (1,))].item()
        cy, cx = divmod(choice, width)
        sy = max(0, cy - patch_h // 2)
        sx = max(0, cx - patch_w // 2)
        ey = min(height, sy + patch_h)
        ex = min(width, sx + patch_w)

        mask = torch.ones((height, width), device=device)
        mask[sy:ey, sx:ex] = 0.0
        img_masks.append(mask.unsqueeze(0))
        loss_masks.append(mask)
        onehot_masks.append(mask.unsqueeze(0).repeat(num_classes, 1, 1))

    img_mask = torch.stack(img_masks, dim=0)
    loss_mask = torch.stack(loss_masks, dim=0)
    onehot_mask = torch.stack(onehot_masks, dim=0)
    return img_mask, loss_mask, onehot_mask


def build_confidence_masks(conf_maps, threshold):
    """
    模块功能: 依据阈值区分高/低置信度区域。
    输入:
        conf_maps (Tensor): [B, 1, H, W], EMA置信度。
        threshold (float): 单值阈值。
    输出:
        high_conf_mask (Tensor): [B, 1, H, W], >= 阈值为1。
        low_conf_mask (Tensor): [B, 1, H, W], < 阈值为1。
    """
    high = (conf_maps >= threshold).float()
    low = 1.0 - high
    return high, low


def confidence_weighted_mix_loss(
    output,
    primary_label,
    secondary_label,
    mask,
    high_conf_mask,
    low_conf_mask,
    unlabeled_region="primary",
    l_weight=1.0,
    u_weight=0.5,
    low_conf_scale=0.5,
):
    """
    模块功能: 基于置信度课程的Mix损失, 区分高/低置信度区域。
    输入:
        output (Tensor): [B, C, H, W], 当前分支输出logits。
        primary_label (Tensor): [B, H, W], mask=1区域对应的标签。
        secondary_label (Tensor): [B, H, W], mask=0区域对应的标签。
        mask (Tensor): [B, 1, H, W], 值1表示primary区域。
        high_conf_mask/low_conf_mask (Tensor): [B, 1, H, W], 区域置信度划分。
        unlabeled_region (str): 'primary' 或 'secondary', 指示伪标签落在哪个区域。
        l_weight (float): 有标签区域的基准权重。
        u_weight (float): 伪标签区域的基准权重。
        low_conf_scale (float): 低置信度伪标签的额外缩放系数。
    输出:
        loss_dice (Tensor): 标量, Dice损失。
        loss_ce (Tensor): 标量, 交叉熵损失。
    """
    ce_fn = nn.CrossEntropyLoss(reduction='none')
    output_soft = F.softmax(output, dim=1)
    mask_primary = mask.squeeze(1)
    mask_secondary = 1.0 - mask_primary
    if unlabeled_region == "primary":
        unlabeled_mask = mask_primary
        labeled_mask = mask_secondary
        unlabeled_label = primary_label
        labeled_label = secondary_label
    else:
        unlabeled_mask = mask_secondary
        labeled_mask = mask_primary
        unlabeled_label = secondary_label
        labeled_label = primary_label

    high_region = unlabeled_mask * high_conf_mask.squeeze(1)
    low_region = unlabeled_mask * low_conf_mask.squeeze(1)
    eps = 1e-6

    def dice_ce(target, region_mask, weight):
        if region_mask.sum() < eps:
            return output.new_tensor(0.0), output.new_tensor(0.0)
        dice_term = dice_loss(output_soft, target.unsqueeze(1), region_mask.unsqueeze(1)) * weight
        ce_map = ce_fn(output, target)
        ce_term = (ce_map * region_mask).sum() / (region_mask.sum() + eps) * weight
        return dice_term, ce_term

    loss_dice, loss_ce_val = 0.0, 0.0

    d_high, c_high = dice_ce(unlabeled_label.long(), high_region, u_weight)
    d_low, c_low = dice_ce(unlabeled_label.long(), low_region, u_weight * low_conf_scale)
    d_lab, c_lab = dice_ce(labeled_label.long(), labeled_mask, l_weight)

    loss_dice = d_high + d_low + d_lab
    loss_ce_val = c_high + c_low + c_lab
    return loss_dice, loss_ce_val


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 5 * args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

# @torch.no_grad()
def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


def get_current_ema_alpha(iteration, args):
    """
    模块功能: 动态调整EMA的指数衰减系数, 让teacher在中后期更快跟随student。
    输入:
        iteration (int): 当前迭代编号。
        args (Namespace): 包含EMA起止参数。
    输出:
        float: 当前EMA alpha, 范围[ema_alpha_low, ema_alpha_high]。
    """
    ramp = min(1.0, iteration / max(1.0, args.ema_alpha_ramp))
    return args.ema_alpha_high - (args.ema_alpha_high - args.ema_alpha_low) * ramp


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


def select_patches_for_contrast(output_mix, topnum=16, patch_size=(4, 4), choose_largest=False, focus_mask=None):
    """
    模块功能: 将特征图切分为patch, 并依据置信度/掩码选取难样本作为正样本。
    输入:
        output_mix (Tensor): [B, C, H, W], mix后的logits。
        topnum (int): 需要的潜在正样本patch数量。
        patch_size (tuple): (ph, pw), patch尺寸。
        choose_largest (bool): True时选择最高置信度patch, False选择最低。
        focus_mask (Tensor or None): [B, 1, H, W], 可选的关注区域, 值越大越容易被采样。
    输出:
        pos_patches (Tensor): [B, topnum, C], 选中的目标patch特征。
        neg_patches (Tensor): [B, L-topnum, C], 其余patch特征。
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
    if focus_mask is not None:
        mask_unfold = F.unfold(focus_mask, kernel_size=(ph, pw), stride=(ph, pw))
        mask_mean = mask_unfold.mean(dim=1)
        # 只在关注区域内优先选择: 关注区域越大, patch_conf越小(更容易被抽取)
        patch_conf = patch_conf + (1.0 - mask_mean)

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
        index=top_idx.unsqueeze(-1).expand(-1, -1, C)
    )

    # 负样本（其余 patch）
    # 用布尔掩码批量选择，再 reshape 回 [B, L-topnum, C]
    neg_patches = feat_patches[mask].view(B, num_neg, C)                      # [B, L-topnum, C]

    return pos_patches, neg_patches


def boundary_kl_loss(fg_logits, bg_logits, band_mask):
    """
    模块功能: 在狭窄边带内约束前景/背景分支的一致性。
    输入:
        fg_logits (Tensor): [B, C, H, W], 前景分支logits。
        bg_logits (Tensor): [B, C, H, W], 背景分支logits。
        band_mask (Tensor): [B, 1, H, W], 仅在窄带内为1。
    输出:
        Tensor: 标量, 表示KL散度损失。
    """
    eps = 1e-6
    if band_mask.sum() < eps:
        return fg_logits.new_tensor(0.0)
    fg_prob = F.softmax(fg_logits.detach(), dim=1)
    bg_logprob = F.log_softmax(bg_logits, dim=1)
    kl_map = F.kl_div(bg_logprob, fg_prob, reduction='none').sum(dim=1, keepdim=True)
    weighted = kl_map * band_mask
    return weighted.sum() / (band_mask.sum() + eps)


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
    labeled_sub_bs = int(args.labeled_bs / 2)
    unlabeled_sub_bs = int((args.batch_size - args.labeled_bs) / 2)

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    ema_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    for param in ema_model.parameters():
        param.detach_()

    confidence_tracker = ConfidenceTracker(momentum=args.cgmc_conf_momentum)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transforms.Compose([
            WeakStrongAugment(args.patch_size),
            CreateOnehotLabel(args.num_classes)
        ])
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs
    )

    trainloader = DataLoader(
        db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # load_net(ema_model, pre_trained_model)
    # load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    BCLLoss = losses.BlockContrastiveLoss()

    iter_num = 0
    max_epoch = self_max_iterations // len(trainloader) + 1
    best_performance = 0.0
    ema_best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            model.train()
            ema_model.train()

            volume_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()
            onehot_label_batch = sampled_batch['onehot_label'].cuda()
            volume_batch_strong = sampled_batch['image_strong'].cuda()
            label_batch_strong = sampled_batch['label_strong'].cuda()
            onehot_label_batch_strong = sampled_batch['onehot_label_strong'].cuda()
            case_batch = list(sampled_batch['case'])

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs]
            uimg_b = volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            ulab_a = label_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs]
            ulab_b = label_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            lab_a_bg = (onehot_label_batch[:labeled_sub_bs] == 0)
            lab_b_bg = (onehot_label_batch[labeled_sub_bs:args.labeled_bs] == 0)

            img_a_s, img_b_s = volume_batch_strong[:labeled_sub_bs], volume_batch_strong[labeled_sub_bs:args.labeled_bs]
            uimg_a_s = volume_batch_strong[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs]
            uimg_b_s = volume_batch_strong[args.labeled_bs + unlabeled_sub_bs:]
            lab_a_bg_s = (onehot_label_batch_strong[:labeled_sub_bs] == 0)
            lab_b_bg_s = (onehot_label_batch_strong[labeled_sub_bs:args.labeled_bs] == 0)

            with torch.no_grad():
                pre_a_fg, pre_a_mix, pre_a_bg, _, _ = ema_model(uimg_a, uimg_a_s)
                pre_b_fg, pre_b_mix, pre_b_bg, _, _ = ema_model(uimg_b, uimg_b_s)

                plab_a_fg = get_ACDC_masks_with_confidence(pre_a_fg, nms=1)
                plab_b_fg = get_ACDC_masks_with_confidence(pre_b_fg, nms=1)
                plab_a_bg = get_ACDC_masks_with_confidence(pre_a_bg, nms=1, onehot=True)
                plab_b_bg = get_ACDC_masks_with_confidence(pre_b_bg, nms=1, onehot=True)

                conf_a_raw = torch.softmax(pre_a_mix, dim=1).max(dim=1, keepdim=True).values
                conf_b_raw = torch.softmax(pre_b_mix, dim=1).max(dim=1, keepdim=True).values
                unlabeled_case_ids = case_batch[args.labeled_bs:]
                conf_cat = torch.cat([conf_a_raw, conf_b_raw], dim=0)
                smooth_conf, smooth_mean = confidence_tracker.update(unlabeled_case_ids, conf_cat, device=conf_cat.device)
                conf_a = smooth_conf[:unlabeled_sub_bs]
                conf_b = smooth_conf[unlabeled_sub_bs:]
                mean_a = smooth_mean[:unlabeled_sub_bs]
                mean_b = smooth_mean[unlabeled_sub_bs:]

                threshold = cgmc_confidence_threshold(iter_num, args)
                low_conf_scale = cgmc_low_conf_scale(iter_num, args)
                high_a, low_a = build_confidence_masks(conf_a, threshold)
                high_b, low_b = build_confidence_masks(conf_b, threshold)

                mask_a, loss_mask_a, onehot_mask_a = generate_confidence_guided_mask(
                    conf_a, mean_a, args.num_classes, args.cgmc_min_ratio, args.cgmc_max_ratio
                )
                mask_b, loss_mask_b, onehot_mask_b = generate_confidence_guided_mask(
                    conf_b, mean_b, args.num_classes, args.cgmc_min_ratio, args.cgmc_max_ratio
                )

                mask_a_flat = mask_a.squeeze(1)
                mask_b_flat = mask_b.squeeze(1)
                unl_label = ulab_a * mask_a_flat + lab_a * (1 - mask_a_flat)
                l_label = lab_b * mask_b_flat + ulab_b * (1 - mask_b_flat)

            net_input_unl = uimg_a * mask_a + img_a * (1 - mask_a)
            net_input_l = img_b * mask_b + uimg_b * (1 - mask_b)
            net_input_unl_s = uimg_a_s * mask_a + img_a_s * (1 - mask_a)
            net_input_l_s = img_b_s * mask_b + uimg_b_s * (1 - mask_b)

            out_unl_fg, out_unl, out_unl_bg, _, _ = model(net_input_unl, net_input_unl_s)
            out_l_fg, out_l, out_l_bg, _, _ = model(net_input_l, net_input_l_s)

            unl_dice, unl_ce = confidence_weighted_mix_loss(
                out_unl_fg, plab_a_fg, lab_a, mask_a, high_a, low_a,
                unlabeled_region="primary", l_weight=1.0, u_weight=args.u_weight,
                low_conf_scale=low_conf_scale
            )
            l_dice, l_ce = confidence_weighted_mix_loss(
                out_l_fg, lab_b, plab_b_fg, mask_b, high_b, low_b,
                unlabeled_region="secondary", l_weight=1.0, u_weight=args.u_weight,
                low_conf_scale=low_conf_scale
            )

            unl_dice_bg, unl_ce_bg = onehot_mix_loss(
                out_unl_bg, plab_a_bg, lab_a_bg_s, onehot_mask_a, u_weight=args.u_weight, unlab=True
            )
            l_dice_bg, l_ce_bg = onehot_mix_loss(
                out_l_bg, lab_b_bg_s, plab_b_bg, onehot_mask_b, u_weight=args.u_weight
            )

            output_mix = torch.cat([out_unl, out_l], dim=0)
            low_focus = torch.cat([low_a, low_b], dim=0)
            pos_patches, neg_patches = select_patches_for_contrast(
                output_mix, topnum=64, patch_size=(8, 8), focus_mask=low_focus
            )
            # 将对比损失权重限制在[0.2,0.6], 防止全图低置信时代价失控
            low_ratio = low_focus.mean().clamp(0.2, 0.6)
            bclloss = BCLLoss(pos_patches, neg_patches) * low_ratio

            fg_mask_a = (plab_a_fg > 0).float().unsqueeze(1) * high_a
            fg_mask_b = (plab_b_fg > 0).float().unsqueeze(1) * high_b
            band_a = create_narrow_band(fg_mask_a, args.band_inner, args.band_outer)
            band_b = create_narrow_band(fg_mask_b, args.band_inner, args.band_outer)
            band_loss = boundary_kl_loss(out_unl_fg, out_unl_bg, band_a) + \
                        boundary_kl_loss(out_l_fg, out_l_bg, band_b)

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            loss_fg = unl_dice + l_dice + unl_ce + l_ce
            loss_bg = unl_dice_bg + l_dice_bg + unl_ce_bg + l_ce_bg
            loss = loss_fg + loss_bg + args.band_kl_weight * band_loss + consistency_weight * bclloss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1
            current_alpha = get_current_ema_alpha(iter_num, args)
            update_model_ema(model, ema_model, current_alpha)

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/fg_loss', loss_fg, iter_num)
            writer.add_scalar('info/bg_loss', loss_bg, iter_num)
            writer.add_scalar('info/band_loss', band_loss, iter_num)
            writer.add_scalar('info/bcl_loss', bclloss, iter_num)
            writer.add_scalar('info/conf_threshold', threshold, iter_num)
            writer.add_scalar('info/low_conf_scale', low_conf_scale, iter_num)
            writer.add_scalar('info/ema_alpha', current_alpha, iter_num)

            if iter_num % 20 == 0:
                writer.add_image('train/Un_Image', net_input_unl[0, 0:1], iter_num)
                writer.add_image('train/L_Image', net_input_l[0, 0:1], iter_num)
                writer.add_image('train/Un_Prediction', torch.argmax(out_unl, dim=1, keepdim=True)[0] * 50, iter_num)
                writer.add_image('train/L_Prediction', torch.argmax(out_l, dim=1, keepdim=True)[0] * 50, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                ema_model.eval()
                metric_list = 0.0
                ema_metric_list = 0.0
                for _, val_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume_argument(
                        val_batch["image"], val_batch["label"], model, classes=num_classes
                    )
                    ema_metric_i = val_2d.test_single_volume_argument(
                        val_batch["image"], val_batch["label"], ema_model, classes=num_classes
                    )
                    metric_list += np.array(metric_i)
                    ema_metric_list += np.array(ema_metric_i)
                metric_list = metric_list / len(db_val)
                ema_metric_list = ema_metric_list / len(db_val)
                performance = np.mean(metric_list, axis=0)[0]
                ema_performance = np.mean(ema_metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(
                        snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4))
                    )
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                if ema_performance > ema_best_performance:
                    ema_best_performance = ema_performance
                    ema_save_mode_path = os.path.join(
                        snapshot_path, 'iter_ema_{}_dice_{}.pth'.format(iter_num, round(ema_best_performance, 4))
                    )
                    ema_save_best_path = os.path.join(snapshot_path, '{}_ema_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), ema_save_mode_path)
                    torch.save(model.state_dict(), ema_save_best_path)

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
    shutil.copy('./just_try/ACDC/ACDC_train_4_6_3.py', self_snapshot_path)

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
