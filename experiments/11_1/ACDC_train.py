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

from utils import losses, ramps, val_2d
from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler, WeakStrongAugment
from networks.net_factory import net_factory
from utils.BCP_utils import mix_loss, update_ema_variables
from .skc2d_module import CVBMArgumentWithSKC2D

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/ACDC', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='CVBM2d_ACDC', help='exp_name')
parser.add_argument('--model', type=str, default='CVBM2d_Argument', help='model_name')
parser.add_argument('--pre_max_iteration', type=int, default=10000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int, default=30000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int, default=1400, help='maximum slices to train')
parser.add_argument('--labeled_bs', type=int, default=12, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch_size of loading image')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=0, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=3, help='trained samples')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
parser.add_argument('--num_workers', type=int, default=8, help='cpu core num_workers')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default='6.0', help='magnitude')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2 / 3, help='ratio of mask/image')
parser.add_argument('--snapshot_path', type=str, default='./results/CVBM_ACDC_11_1/1/', help='snapshot path to save model')
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
    state = torch.load(str(ckpt_path))
    state_dict = state['net'] if 'net' in state else state
    model_dict = model.state_dict()
    compatible = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(compatible)
    model.load_state_dict(model_dict)
    logging.info("Loaded %d/%d tensors from %s into %s",
                 len(compatible), len(model_dict), ckpt_path, model.__class__.__name__)


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def select_patches_for_contrast_2d(output_mix, topnum=16, patch_size=(8, 8), choose_largest=False):
    B, C, H, W = output_mix.shape
    ph, pw = patch_size
    assert H % ph == 0 and W % pw == 0, "H/W must be divisible by patch_size"
    probs = F.softmax(output_mix, dim=1)
    score = probs.max(dim=1, keepdim=True).values
    patch_conf = F.avg_pool2d(score, kernel_size=(ph, pw), stride=(ph, pw)).flatten(1)
    _, top_idx = patch_conf.topk(topnum, dim=1, largest=choose_largest)

    B_, L = patch_conf.shape
    all_idx = torch.arange(L, device=output_mix.device).unsqueeze(0).expand(B_, -1)
    mask = torch.ones_like(all_idx, dtype=torch.bool)
    mask.scatter_(1, top_idx, False)
    num_neg = L - topnum

    feat_patches = F.avg_pool2d(probs, kernel_size=(ph, pw), stride=(ph, pw))
    feat_patches = feat_patches.flatten(2).permute(0, 2, 1).contiguous()

    pos_patches = torch.gather(feat_patches, dim=1, index=top_idx.unsqueeze(-1).expand(-1, -1, C))
    neg_patches = feat_patches[mask].view(B, num_neg, C)
    return pos_patches, neg_patches


def context_mask_2d(img, mask_ratio):
    batch_size, channel, img_x, img_y = img.shape
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_pixel_x, patch_pixel_y = int(img_x * mask_ratio), int(img_y * mask_ratio)
    w = np.random.randint(0, img_x - patch_pixel_x)
    h = np.random.randint(0, img_y - patch_pixel_y)
    mask[w:w + patch_pixel_x, h:h + patch_pixel_y] = 0
    loss_mask[:, w:w + patch_pixel_x, h:h + patch_pixel_y] = 0
    return mask.long(), loss_mask.long()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate" in dataset:
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        raise ValueError("Unsupported dataset for patients_to_slices mapping")
    key = str(patiens_num)
    if key not in ref_dict:
        raise ValueError(f"patiens_num={patiens_num} not in ref_dict")
    return ref_dict[key]


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


def val_all_case_ACDC(model, classes, patch_size, dataset_path):
    with open(dataset_path + '/val.list', 'r') as f:
        sample_list = f.readlines()
    sample_list = [item.replace('\n', '') for item in sample_list]
    total_metric = []
    for case in sample_list:
        h5_path = dataset_path + f"/data/{case}.h5"
        import h5py
        with h5py.File(h5_path, 'r') as h5f:
            image, label = h5f['image'][:], h5f['label'][:]
        # image shape [D,H,W]
        metric_case = val_2d.test_single_volume(torch.from_numpy(image).unsqueeze(0),
                                                torch.from_numpy(label).unsqueeze(0),
                                                model, classes, patch_size=patch_size)
        metric_case = np.array(metric_case)  # list of tuples
        total_metric.append(metric_case)
    total_metric = np.array(total_metric)
    avg_metric = total_metric.mean(axis=0)
    dice_mean = avg_metric[:, 0].mean()
    return dice_mean


def pre_train(args, snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes, mode="train")
    db_train = BaseDataSets(base_dir=args.root_path,
                            split='train',
                            transform=transforms.Compose([
                                RandomGenerator(output_size=args.patch_size)
                            ]))
    total_slices = len(db_train)
    labeled_slice = min(patients_to_slices(args.root_path, args.labelnum), total_slices)
    logging.info("Total slices: %d, labeled slices: %d", total_slices, labeled_slice)
    labeled_idxs = list(range(labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
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
    DICE = losses.mask_DiceLoss(nclass=args.num_classes)

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
            with torch.no_grad():
                img_mask, loss_mask = context_mask_2d(img_a, args.mask_ratio)

            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)

            outputs_fg, outputs, outputs_bg, _, _ = model(volume_batch, volume_batch)
            loss_seg = 0
            loss_seg_dice = 0

            y2 = outputs_fg[:args.labeled_bs, ...]
            y_prob2 = F.softmax(y2, dim=1)
            loss_seg += F.cross_entropy(y2[:args.labeled_bs], label_batch[:args.labeled_bs, ...].long())
            loss_seg_dice += DICE(y_prob2, label_batch[:args.labeled_bs, ...])

            y_bg = outputs_bg[:args.labeled_bs, ...]
            y_prob_bg = F.softmax(y_bg, dim=1)
            loss_seg += F.cross_entropy(y_bg[:args.labeled_bs], label_batch[:args.labeled_bs, ...].long())
            loss_seg_dice += DICE(y_prob_bg, (label_batch[:args.labeled_bs, ...] == 0).long())

            loss = (loss_seg + loss_seg_dice) / 2

            iter_num += 1
            writer.add_scalar('pre/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('pre/loss_seg', loss_seg, iter_num)
            writer.add_scalar('pre/loss_all', loss, iter_num)
            logging.info("iter %d pre_loss %f", iter_num, loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_num % 400 == 0:
                model.eval()
                dice_sample = val_all_case_ACDC(model, args.num_classes, args.patch_size, args.root_path)
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
    model = CVBMArgumentWithSKC2D(
        n_channels=1,
        n_classes=args.num_classes,
        has_dropout=True,
    ).cuda()
    ema_model = CVBMArgumentWithSKC2D(
        n_channels=1,
        n_classes=args.num_classes,
        has_dropout=True,
    ).cuda()
    for param in ema_model.parameters():
        param.detach_()
    db_train = BaseDataSets(
        base_dir=args.root_path,
        split='train',
        transform=transforms.Compose([
            WeakStrongAugment(args.patch_size)
        ]))
    total_slices = len(db_train)
    labeled_slice = min(patients_to_slices(args.root_path, args.labelnum), total_slices)
    logging.info("Total slices: %d, labeled slices: %d", total_slices, labeled_slice)
    labeled_idxs = list(range(labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
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
    BCLLoss = losses.BlockContrastiveLoss()

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
            unimg_a, unimg_b = volume_batch[args.labeled_bs:args.labeled_bs + sub_bs], volume_batch[args.labeled_bs + sub_bs:]

            volume_batch_strong, label_batch_strong = sampled_batch['image_strong'], sampled_batch['label_strong']
            volume_batch_strong, label_batch_strong = volume_batch_strong.cuda(), label_batch_strong.cuda()
            img_a_s, img_b_s = volume_batch_strong[:sub_bs], volume_batch_strong[sub_bs:args.labeled_bs]
            lab_a_s, lab_b_s = label_batch_strong[:sub_bs], label_batch_strong[sub_bs:args.labeled_bs]
            unimg_a_s, unimg_b_s = volume_batch_strong[args.labeled_bs:args.labeled_bs + sub_bs], volume_batch_strong[args.labeled_bs + sub_bs:]

            with torch.no_grad():
                unoutput_a_fg, unoutput_a, unoutput_a_bg, _, _ = ema_model(unimg_a, unimg_a_s)
                unoutput_b_fg, unoutput_b, unoutput_b_bg, _, _ = ema_model(unimg_b, unimg_b_s)
                plab_a = get_cut_mask(unoutput_a, nms=1)
                plab_b = get_cut_mask(unoutput_b, nms=1)
                plab_a_fg = get_cut_mask(unoutput_a_fg, nms=1)
                plab_b_fg = get_cut_mask(unoutput_b_fg, nms=1)
                plab_a_bg = get_cut_mask(unoutput_a_bg, nms=1)
                plab_b_bg = get_cut_mask(unoutput_b_bg, nms=1)
                img_mask, loss_mask = context_mask_2d(img_a, args.mask_ratio)

            mixl_img = img_a * img_mask + unimg_a * (1 - img_mask)
            mixu_img = unimg_b * img_mask + img_b * (1 - img_mask)
            mixl_img_s = img_a_s * img_mask + unimg_a_s * (1 - img_mask)
            mixu_img_s = unimg_b_s * img_mask + img_b_s * (1 - img_mask)

            mixl_lab = lab_a * img_mask + plab_a * (1 - img_mask)
            mixu_lab = plab_b * img_mask + lab_b * (1 - img_mask)

            outputs_l_fg, outputs_l, outputs_l_bg, _, _ = model(mixl_img, mixl_img_s)
            outputs_u_fg, outputs_u, outputs_u_bg, _, _ = model(mixu_img, mixu_img_s)

            output_mix_bg_fg = torch.cat([outputs_l, outputs_u], dim=0)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss_l = mix_loss(outputs_l_fg, mixl_lab, plab_a_fg, loss_mask, u_weight=args.u_weight)
            loss_u = mix_loss(outputs_u_fg, mixu_lab, plab_b_fg, loss_mask, u_weight=args.u_weight, unlab=True)
            loss_l_bg = mix_loss(outputs_l_bg, (mixl_lab == 0).long(), plab_a_bg, loss_mask, u_weight=args.u_weight)
            loss_u_bg = mix_loss(outputs_u_bg, (mixu_lab == 0).long(), plab_b_bg, loss_mask, u_weight=args.u_weight, unlab=True)

            pos_patches, neg_patches = select_patches_for_contrast_2d(
                output_mix_bg_fg,
                topnum=32,
                patch_size=(8, 8),
                choose_largest=False)
            bclloss = BCLLoss(pos_patches, neg_patches)

            loss = loss_l + loss_u + loss_l_bg + loss_u_bg + consistency_weight * bclloss

            iter_num += 1
            writer.add_scalar('Self/consistency', consistency_weight, iter_num)
            writer.add_scalar('Self/loss_l', loss_l, iter_num)
            writer.add_scalar('Self/loss_u', loss_u, iter_num)
            writer.add_scalar('Self/loss_l_bg', loss_l_bg, iter_num)
            writer.add_scalar('Self/loss_u_bg', loss_u_bg, iter_num)
            writer.add_scalar('Self/bclloss', bclloss, iter_num)
            writer.add_scalar('Self/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f, loss_bcl: %03f',
                         iter_num, loss, loss_l, loss_u, bclloss)

            update_ema_variables(model, ema_model, 0.99)

            if iter_num % 400 == 0:
                model.eval()
                ema_model.eval()
                dice_sample = val_all_case_ACDC(model, args.num_classes, args.patch_size, args.root_path)
                ema_dice_sample = val_all_case_ACDC(ema_model, args.num_classes, args.patch_size, args.root_path)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 7)
                    save_mode_path = os.path.join(self_snapshot_path, f'iter_{iter_num}_dice_{best_dice}.pth')
                    save_best_path = os.path.join(self_snapshot_path, f'{args.model}_best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to %s", save_mode_path)
                if ema_dice_sample > ema_best_dice:
                    ema_best_dice = round(ema_dice_sample, 7)
                    save_mode_path = os.path.join(self_snapshot_path, f'iter_{iter_num}_ema_dice_{ema_best_dice}.pth')
                    save_ema_best_path = os.path.join(self_snapshot_path, f'{args.model}_ema_best_model.pth')
                    torch.save(ema_model.state_dict(), save_mode_path)
                    torch.save(ema_model.state_dict(), save_ema_best_path)
                    logging.info("save best ema model to %s", save_mode_path)
                writer.add_scalar('4_Var_dice/Dice', ema_dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', ema_best_dice, iter_num)
                model.train()

            if iter_num >= self_max_iterations:
                break

        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    pre_snapshot_path = "{}/{}_{}_labeled/pre_train".format(args.snapshot_path, args.exp, args.labelnum)
    self_snapshot_path = "{}/{}_{}_labeled/self_train".format(args.snapshot_path, args.exp, args.labelnum)
    print("Starting ACDC training with SKC2D.")
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
