import argparse
import ast
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
from scipy.ndimage import zoom
from skimage.measure import label
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import BaseDataSets, TwoStreamBatchSampler, CreateOnehotLabel, WeakStrongAugment
from networks.net_factory import net_factory
from utils import losses, ramps
from .modules import CVBMArgumentWithCrossSKC2D


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/ACDC', help='Name of dataset root')
parser.add_argument('--exp', type=str, default='CVBM_13_1', help='experiment name')
parser.add_argument('--model', type=str, default='CVBM2d_Argument', help='checkpoint/model tag')
parser.add_argument('--pre_max_iteration', type=int, default=10000, help='maximum pre-train iterations')
parser.add_argument('--self_max_iteration', type=int, default=30000, help='maximum self-train iterations')
parser.add_argument('--batch_size', type=int, default=24, help='batch size per gpu')
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled batch size per gpu')
parser.add_argument('--deterministic', type=int, default=0, help='use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='network input patch size')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4, help='output channels of the network')
parser.add_argument('--labelnum', type=int, default=3, help='number of labeled patients')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency weight')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency rampup')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--topnum', type=int, default=64, help='contrastive topnum')
parser.add_argument('--contrast_patch', type=str, default='(8, 8)', help='contrastive patch size')
parser.add_argument('--train_num', type=int, default=1, help='training run index')
parser.add_argument('--snapshot_path', type=str, default='./results', help='snapshot base path')
args = parser.parse_args()

if args.contrast_patch:
    args.contrast_patch = ast.literal_eval(args.contrast_patch)

torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
dice_loss = losses.DiceLoss(n_classes=args.num_classes)
onehot_dice_loss = losses.DiceLoss2d(n_classes=args.num_classes)
onehot_ce_loss = losses.CrossEntropyLoss(n_classes=args.num_classes)


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


def load_pretrained_backbone(model, ckpt_path):
    state = torch.load(str(ckpt_path))
    state_dict = state['net'] if 'net' in state else state
    model_dict = model.state_dict()
    compatible = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(compatible)
    model.load_state_dict(model_dict)
    logging.info(
        "Loaded %d/%d tensors from %s into %s",
        len(compatible),
        len(model_dict),
        ckpt_path,
        model.__class__.__name__,
    )


def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    batch_size = segmentation.shape[0]
    for i in range(batch_size):
        class_list = []
        for c in range(1, args.num_classes):
            temp_seg = segmentation[i]
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largest_cc * c)
            else:
                class_list.append(temp_prob)

        merged = class_list[0]
        for class_mask in class_list[1:]:
            merged = merged + class_mask
        batch_list.append(merged)

    return torch.as_tensor(np.asarray(batch_list), device=segmentation.device, dtype=segmentation.dtype)


def get_ACDC_2DLargestCC_onehot(segmentation):
    batch_list = []
    batch_size = segmentation.shape[0]
    for i in range(batch_size):
        class_list = []
        for c in range(args.num_classes):
            temp_seg = segmentation[i]
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largest_cc * c)
            else:
                class_list.append(temp_prob)
        batch_list.append(class_list)

    return torch.as_tensor(np.asarray(batch_list), device=segmentation.device, dtype=torch.int64)


def get_ACDC_masks_with_confidence(output, nms=0, onehot=False):
    probs = F.softmax(output, dim=1)
    _, indices = torch.max(probs, dim=1)
    if nms == 1:
        if onehot:
            indices = get_ACDC_2DLargestCC_onehot(indices)
        else:
            indices = get_ACDC_2DLargestCC(indices)
    return indices


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


def generate_mask(img, number_class):
    batch_size, _, img_x, img_y = img.shape
    loss_mask = torch.ones(batch_size, img_x, img_y, device=img.device)
    mask = torch.ones(img_x, img_y, device=img.device)
    onehot_mask = torch.ones(batch_size, number_class, img_x, img_y, device=img.device)
    patch_x, patch_y = int(img_x * 2 / 3), int(img_y * 2 / 3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w + patch_x, h:h + patch_y] = 0
    loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    onehot_mask[:, :, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long(), onehot_mask.long()


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    ce = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (ce(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (ce(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    return loss_dice, loss_ce


def onehot_mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = onehot_dice_loss(output_soft, img_l, mask) * image_weight
    loss_dice += onehot_dice_loss(output_soft, patch_l, patch_mask) * patch_weight
    loss_ce = onehot_ce_loss(output_soft, img_l, mask) * image_weight
    loss_ce += onehot_ce_loss(output_soft, patch_l, patch_mask) * patch_weight
    return loss_dice, loss_ce


def patients_to_slices(dataset, patients_num):
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136, "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    else:
        raise ValueError(f"Unsupported dataset path: {dataset}")
    return ref_dict[str(patients_num)]


def select_patches_for_contrast(output_mix, topnum=16, patch_size=(4, 4), choose_largest=False):
    b, c, h, w = output_mix.shape
    ph, pw = patch_size
    assert h % ph == 0 and w % pw == 0, "H/W must be divisible by patch_size"

    probs = F.softmax(output_mix, dim=1)
    score = probs.max(dim=1, keepdim=True).values

    score_patches = F.unfold(score, kernel_size=(ph, pw), stride=(ph, pw))
    patch_conf = score_patches.mean(dim=1)

    _, top_idx = patch_conf.topk(topnum, dim=1, largest=choose_largest)
    _, num_patches = patch_conf.shape
    all_idx = torch.arange(num_patches, device=output_mix.device).unsqueeze(0).expand(b, -1)
    mask = torch.ones_like(all_idx, dtype=torch.bool)
    mask.scatter_(1, top_idx, False)
    num_neg = num_patches - topnum

    feat_patches = F.unfold(probs, kernel_size=(ph, pw), stride=(ph, pw))
    feat_patches = feat_patches.view(b, c, ph * pw, num_patches).mean(dim=2)
    feat_patches = feat_patches.permute(0, 2, 1).contiguous()

    pos_patches = torch.gather(
        feat_patches,
        dim=1,
        index=top_idx.unsqueeze(-1).expand(-1, -1, c),
    )
    neg_patches = feat_patches[mask].view(b, num_neg, c)
    return pos_patches, neg_patches


def calculate_metric_percase(pred, gt):
    pred = pred.copy()
    gt = gt.copy()
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        from medpy import metric
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    return 0, 0


def test_single_volume_argument(image, label, model, classes, patch_size=None):
    if patch_size is None:
        patch_size = args.patch_size
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    model.eval()
    for ind in range(image.shape[0]):
        slice_img = image[ind, :, :]
        x, y = slice_img.shape[0], slice_img.shape[1]
        slice_img = zoom(slice_img, (patch_size[0] / x, patch_size[1] / y), order=0)
        input_tensor = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0).float().cuda()
        with torch.no_grad():
            output = model(input_tensor, input_tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def build_cross_skc_model():
    return CVBMArgumentWithCrossSKC2D(
        n_channels=1,
        n_classes=args.num_classes,
        has_dropout=True,
        attn_shape=(8, 8),
        local_window=(2, 2),
        num_heads=4,
        dropout=0.0,
    ).cuda()


def pre_train(args, snapshot_path):
    model = net_factory(net_type="CVBM2d_Argument", in_chns=1, class_num=args.num_classes)
    labeled_sub_bs = int(args.labeled_bs / 2)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transforms.Compose([
            WeakStrongAugment(args.patch_size),
            CreateOnehotLabel(args.num_classes),
        ]),
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    logging.info("Total slices: %d, labeled slices: %d", total_slices, labeled_slice)
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs,
        unlabeled_idxs,
        args.batch_size,
        args.batch_size - args.labeled_bs,
    )

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre-training")
    logging.info("%d iterations per epoch", len(trainloader))

    iter_num = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            model.train()
            volume_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()
            onehot_label_batch = sampled_batch['onehot_label'].cuda()

            img_a = volume_batch[:labeled_sub_bs]
            img_b = volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a = label_batch[:labeled_sub_bs]
            lab_b = label_batch[labeled_sub_bs:args.labeled_bs]
            onehot_lab_a = onehot_label_batch[:labeled_sub_bs] == 0
            onehot_lab_b = onehot_label_batch[labeled_sub_bs:args.labeled_bs] == 0

            img_mask, loss_mask, onehot_mask = generate_mask(img_a, args.num_classes)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)
            net_input = img_a * img_mask + img_b * (1 - img_mask)

            out_mixl_fg, _, outputs_mixl_bg, *_ = model(net_input, net_input)
            loss_dice_fg, loss_ce_fg = mix_loss(out_mixl_fg, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)
            loss_dice_bg, loss_ce_bg = onehot_mix_loss(
                outputs_mixl_bg,
                onehot_lab_a,
                onehot_lab_b,
                onehot_mask,
                u_weight=1.0,
                unlab=True,
            )
            loss = (loss_dice_fg + loss_ce_fg + loss_dice_bg + loss_ce_bg) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            writer.add_scalar('pre/total_loss', loss, iter_num)
            writer.add_scalar('pre/mix_dice_fg', loss_dice_fg, iter_num)
            writer.add_scalar('pre/mix_ce_fg', loss_ce_fg, iter_num)
            writer.add_scalar('pre/mix_dice_bg', loss_dice_bg, iter_num)
            writer.add_scalar('pre/mix_ce_bg', loss_ce_bg, iter_num)

            logging.info(
                'pre iteration %d: loss=%f, fg_dice=%f, fg_ce=%f, bg_dice=%f, bg_ce=%f',
                iter_num,
                loss,
                loss_dice_fg,
                loss_ce_fg,
                loss_dice_bg,
                loss_ce_bg,
            )

            if iter_num % 20 == 0 and net_input.shape[0] > 1:
                image = net_input[1, 0:1, :, :]
                writer.add_image('pre/Mixed_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl_fg, dim=1), dim=1, keepdim=True)
                writer.add_image('pre/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = gt_mixl[1, ...].unsqueeze(0) * 50
                writer.add_image('pre/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, val_batch in enumerate(valloader):
                    metric_i = test_single_volume_argument(
                        val_batch["image"],
                        val_batch["label"],
                        model,
                        classes=args.num_classes,
                        patch_size=args.patch_size,
                    )
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('pre/val_mean_dice', performance, iter_num)

                for class_i in range(args.num_classes - 1):
                    writer.add_scalar(f'pre/val_{class_i + 1}_dice', metric_list[class_i, 0], iter_num)
                    writer.add_scalar(f'pre/val_{class_i + 1}_hd95', metric_list[class_i, 1], iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, f'iter_{iter_num}_dice_{round(best_performance, 4)}.pth')
                    save_best_path = os.path.join(snapshot_path, f'{args.model}_best_model.pth')
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('pre iteration %d: mean_dice=%f', iter_num, performance)
                model.train()

            if iter_num >= pre_max_iterations:
                break
        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(args, pre_snapshot_path, self_snapshot_path):
    labeled_sub_bs = int(args.labeled_bs / 2)
    unlabeled_sub_bs = int((args.batch_size - args.labeled_bs) / 2)
    assert args.labeled_bs % 2 == 0, "labeled_bs must be even"
    assert (args.batch_size - args.labeled_bs) % 2 == 0, "unlabeled batch size must be even"

    model = build_cross_skc_model()
    ema_model = build_cross_skc_model()
    for param in ema_model.parameters():
        param.requires_grad_(False)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transforms.Compose([
            WeakStrongAugment(args.patch_size),
            CreateOnehotLabel(args.num_classes),
        ]),
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    logging.info("Total slices: %d, labeled slices: %d", total_slices, labeled_slice)

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs,
        unlabeled_idxs,
        args.batch_size,
        args.batch_size - args.labeled_bs,
    )

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    pretrained_model = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')
    load_pretrained_backbone(model, pretrained_model)
    load_pretrained_backbone(ema_model, pretrained_model)

    bcl_loss_fn = losses.BlockContrastiveLoss()
    writer = SummaryWriter(self_snapshot_path + '/log')
    logging.info("Start self-training")
    logging.info("%d iterations per epoch", len(trainloader))

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

            img_a = volume_batch[:labeled_sub_bs]
            img_b = volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs]
            uimg_b = volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            ulab_a = label_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs]
            ulab_b = label_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a = label_batch[:labeled_sub_bs]
            lab_b = label_batch[labeled_sub_bs:args.labeled_bs]
            lab_a_bg = onehot_label_batch[:labeled_sub_bs] == 0
            lab_b_bg = onehot_label_batch[labeled_sub_bs:args.labeled_bs] == 0

            img_a_s = volume_batch_strong[:labeled_sub_bs]
            img_b_s = volume_batch_strong[labeled_sub_bs:args.labeled_bs]
            uimg_a_s = volume_batch_strong[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs]
            uimg_b_s = volume_batch_strong[args.labeled_bs + unlabeled_sub_bs:]
            lab_a_s = label_batch_strong[:labeled_sub_bs]
            lab_b_s = label_batch_strong[labeled_sub_bs:args.labeled_bs]
            lab_a_bg_s = onehot_label_batch_strong[:labeled_sub_bs] == 0
            lab_b_bg_s = onehot_label_batch_strong[labeled_sub_bs:args.labeled_bs] == 0

            with torch.no_grad():
                pre_a_fg, _, pre_a_bg_s, *_ = ema_model(uimg_a, uimg_a_s)
                pre_b_fg, _, pre_b_bg_s, *_ = ema_model(uimg_b, uimg_b_s)

                plab_a_fg = get_ACDC_masks_with_confidence(pre_a_fg, nms=1)
                plab_b_fg = get_ACDC_masks_with_confidence(pre_b_fg, nms=1)
                plab_a_bg_s = get_ACDC_masks_with_confidence(pre_a_bg_s, nms=1, onehot=True)
                plab_b_bg_s = get_ACDC_masks_with_confidence(pre_b_bg_s, nms=1, onehot=True)

                img_mask, loss_mask, onehot_mask = generate_mask(img_a, args.num_classes)
                unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                l_label = lab_b * img_mask + ulab_b * (1 - img_mask)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l = img_b * img_mask + uimg_b * (1 - img_mask)
            net_input_unl_s = uimg_a_s * img_mask + img_a_s * (1 - img_mask)
            net_input_l_s = img_b_s * img_mask + uimg_b_s * (1 - img_mask)

            out_unl_fg, out_unl, out_unl_bg, *_ = model(net_input_unl, net_input_unl_s)
            out_l_fg, out_l, out_l_bg, *_ = model(net_input_l, net_input_l_s)

            output_mix = torch.cat([out_unl, out_l], dim=0)

            unl_dice, unl_ce = mix_loss(out_unl_fg, plab_a_fg, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice, l_ce = mix_loss(out_l_fg, lab_b, plab_b_fg, loss_mask, u_weight=args.u_weight)

            unl_dice_bg, unl_ce_bg = onehot_mix_loss(
                out_unl_bg,
                plab_a_bg_s,
                lab_a_bg_s,
                onehot_mask,
                u_weight=args.u_weight,
                unlab=True,
            )
            l_dice_bg, l_ce_bg = onehot_mix_loss(
                out_l_bg,
                lab_b_bg_s,
                plab_b_bg_s,
                onehot_mask,
                u_weight=args.u_weight,
            )

            loss_ce = unl_ce + l_ce + unl_ce_bg + l_ce_bg
            loss_dice = unl_dice + l_dice + unl_dice_bg + l_dice_bg

            pos_patches, neg_patches = select_patches_for_contrast(
                output_mix,
                topnum=args.topnum,
                patch_size=args.contrast_patch,
                choose_largest=False,
            )
            bclloss = bcl_loss_fn(pos_patches, neg_patches)

            loss = loss_dice + loss_ce + consistency_weight * bclloss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

            writer.add_scalar('self/total_loss', loss, iter_num)
            writer.add_scalar('self/mix_dice', loss_dice, iter_num)
            writer.add_scalar('self/mix_ce', loss_ce, iter_num)
            writer.add_scalar('self/bclloss', bclloss, iter_num)
            writer.add_scalar('self/consistency_weight', consistency_weight, iter_num)

            logging.info(
                'self iteration %d: loss=%f, mix_dice=%f, mix_ce=%f, bcl=%f',
                iter_num,
                loss,
                loss_dice,
                loss_ce,
                bclloss,
            )

            if iter_num % 20 == 0 and net_input_unl.shape[0] > 1:
                image = net_input_unl[1, 0:1, :, :]
                writer.add_image('self/Un_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_unl_fg, dim=1), dim=1, keepdim=True)
                writer.add_image('self/Un_Prediction', outputs[1, ...] * 50, iter_num)
                labs = unl_label[1, ...].unsqueeze(0) * 50
                writer.add_image('self/Un_GroundTruth', labs, iter_num)

                image_l = net_input_l[1, 0:1, :, :]
                writer.add_image('self/L_Image', image_l, iter_num)
                outputs_l = torch.argmax(torch.softmax(out_l_fg, dim=1), dim=1, keepdim=True)
                writer.add_image('self/L_Prediction', outputs_l[1, ...] * 50, iter_num)
                labs_l = l_label[1, ...].unsqueeze(0) * 50
                writer.add_image('self/L_GroundTruth', labs_l, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                ema_model.eval()
                metric_list = 0.0
                ema_metric_list = 0.0
                for _, val_batch in enumerate(valloader):
                    metric_i = test_single_volume_argument(
                        val_batch["image"],
                        val_batch["label"],
                        model,
                        classes=args.num_classes,
                        patch_size=args.patch_size,
                    )
                    ema_metric_i = test_single_volume_argument(
                        val_batch["image"],
                        val_batch["label"],
                        ema_model,
                        classes=args.num_classes,
                        patch_size=args.patch_size,
                    )
                    metric_list += np.array(metric_i)
                    ema_metric_list += np.array(ema_metric_i)

                metric_list = metric_list / len(db_val)
                ema_metric_list = ema_metric_list / len(db_val)
                performance = np.mean(metric_list, axis=0)[0]
                ema_performance = np.mean(ema_metric_list, axis=0)[0]

                writer.add_scalar('self/val_mean_dice', performance, iter_num)
                writer.add_scalar('self/val_ema_mean_dice', ema_performance, iter_num)

                for class_i in range(args.num_classes - 1):
                    writer.add_scalar(f'self/val_{class_i + 1}_dice', metric_list[class_i, 0], iter_num)
                    writer.add_scalar(f'self/val_{class_i + 1}_hd95', metric_list[class_i, 1], iter_num)
                    writer.add_scalar(f'self/ema_val_{class_i + 1}_dice', ema_metric_list[class_i, 0], iter_num)
                    writer.add_scalar(f'self/ema_val_{class_i + 1}_hd95', ema_metric_list[class_i, 1], iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(self_snapshot_path, f'iter_{iter_num}_dice_{round(best_performance, 4)}.pth')
                    save_best_path = os.path.join(self_snapshot_path, f'{args.model}_best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                if ema_performance > ema_best_performance:
                    ema_best_performance = ema_performance
                    ema_save_mode_path = os.path.join(
                        self_snapshot_path,
                        f'iter_ema_{iter_num}_dice_{round(ema_best_performance, 4)}.pth',
                    )
                    ema_save_best_path = os.path.join(self_snapshot_path, f'{args.model}_ema_best_model.pth')
                    torch.save(ema_model.state_dict(), ema_save_mode_path)
                    torch.save(ema_model.state_dict(), ema_save_best_path)

                logging.info('self iteration %d: mean_dice=%f', iter_num, performance)
                logging.info('self iteration %d: ema_mean_dice=%f', iter_num, ema_performance)

            if iter_num >= self_max_iterations:
                break
        if iter_num >= self_max_iterations:
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

    contrast_tag = f"{args.contrast_patch[0]}x{args.contrast_patch[1]}"
    pre_snapshot_path = (
        f"{args.snapshot_path}/{args.exp}/acdc/label{args.labelnum}/"
        f"topnum{args.topnum}_patch{contrast_tag}/{args.train_num}/pre_train"
    )
    self_snapshot_path = (
        f"{args.snapshot_path}/{args.exp}/acdc/label{args.labelnum}/"
        f"topnum{args.topnum}_patch{contrast_tag}/{args.train_num}/self_train"
    )

    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')

    logging.basicConfig(
        filename=pre_snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()

    logging.basicConfig(
        filename=self_snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)
