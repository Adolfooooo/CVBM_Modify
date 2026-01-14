import sys
import os
import argparse
import logging
import shutil
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter

from dataloaders.dataset import LAHeart
from dataloaders.datasets_3d import WeakStrongAugment3d
from utils import losses, test_3d_patch
from .modules import CVBMArgumentWithSKC3D


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/LA', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='CVBM_LA', help='exp_name')
parser.add_argument('--model', type=str, default='CVBM_Argument', help='model_name')
parser.add_argument('--pre_max_iteration', type=int, default=2000, help='(unused) kept for CLI compatibility')
parser.add_argument('--self_max_iteration', type=int, default=15000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='(unused) kept for CLI compatibility')
parser.add_argument('--labeled_bs', type=int, default=4, help='(unused) kept for CLI compatibility')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--patch_size', type=tuple, default=(112, 112, 80), help='patch_size of loading image')
parser.add_argument('--base_lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--deterministic', type=int, default=0, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=80, help='number of labeled training samples to use')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_workers', type=int, default=8, help='cpu core num_workers')
parser.add_argument('--snapshot_path', type=str, default='./results/CVBM_11_1/1/', help='snapshot path to save model')
args = parser.parse_args()


torch.backends.cudnn.benchmark = True


def _set_deterministic(seed: int) -> None:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _build_train_loader(args) -> DataLoader:
    dataset = LAHeart(
        base_dir=args.root_path,
        split='train',
        num=args.labelnum,
        transform=transforms.Compose([WeakStrongAugment3d(args.patch_size, flag_rot=False)]),
    )

    drop_last = len(dataset) >= args.batch_size

    def worker_init_fn(worker_id):
        worker_seed = args.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
        persistent_workers=args.num_workers > 0,
    )
    try:
        return DataLoader(dataset, pin_memory_device="cuda", **loader_kwargs)
    except TypeError:
        # Older PyTorch versions may not support pin_memory_device/persistent_workers.
        loader_kwargs.pop("persistent_workers", None)
        return DataLoader(dataset, **loader_kwargs)


def train_fully_supervised(args, snapshot_path: str) -> None:
    if args.model != "CVBM_Argument":
        raise ValueError(
            f"LA_train_fully_supervised_copy.py expects --model CVBM_Argument, got {args.model}"
        )

    model = CVBMArgumentWithSKC3D(
        n_channels=1,
        n_classes=2,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()

    trainloader = _build_train_loader(args)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    dice_loss = losses.DiceLoss(n_classes=2)

    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    logging.info("%d iterations per epoch", len(trainloader))

    iter_num = 0
    best_dice = 0.0
    max_iterations = args.self_max_iteration
    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for sampled_batch in trainloader:
            model.train()
            img_fg = sampled_batch.get("image_weak", sampled_batch["image"]).cuda(non_blocking=True)
            img_bg = sampled_batch.get("image_strong", sampled_batch["image"]).cuda(non_blocking=True)
            label_batch = sampled_batch["label"].cuda(non_blocking=True).long()

            logits_fg, logits_fused, logits_bg, *_ = model(img_fg, img_bg)

            loss_ce_fg = F.cross_entropy(logits_fg, label_batch)
            loss_dice_fg = dice_loss(logits_fg, label_batch, softmax=True)

            bg_target = (label_batch == 0).long()
            loss_ce_bg = F.cross_entropy(logits_bg, bg_target)
            loss_dice_bg = dice_loss(logits_bg, bg_target, softmax=True)

            loss_ce_fused = F.cross_entropy(logits_fused, label_batch)
            loss_dice_fused = dice_loss(logits_fused, label_batch, softmax=True)

            loss = (
                loss_ce_fg
                + loss_dice_fg
                + loss_ce_bg
                + loss_dice_bg
                + loss_ce_fused
                + loss_dice_fused
            ) / 6.0

            iter_num += 1
            writer.add_scalar('train/loss_ce_fg', loss_ce_fg.item(), iter_num)
            writer.add_scalar('train/loss_dice_fg', loss_dice_fg.item(), iter_num)
            writer.add_scalar('train/loss_ce_bg', loss_ce_bg.item(), iter_num)
            writer.add_scalar('train/loss_dice_bg', loss_dice_bg.item(), iter_num)
            writer.add_scalar('train/loss_ce_fused', loss_ce_fused.item(), iter_num)
            writer.add_scalar('train/loss_dice_fused', loss_dice_fused.item(), iter_num)
            writer.add_scalar('train/loss_all', loss.item(), iter_num)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            logging.info(
                "iter %d: loss=%.4f (fg_ce=%.4f fg_dice=%.4f fused_ce=%.4f fused_dice=%.4f bg_ce=%.4f bg_dice=%.4f)",
                iter_num,
                loss.item(),
                loss_ce_fg.item(),
                loss_dice_fg.item(),
                loss_ce_fused.item(),
                loss_dice_fused.item(),
                loss_ce_bg.item(),
                loss_dice_bg.item(),
            )

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA_argument(
                    model,
                    num_classes=2,
                    patch_size=args.patch_size,
                    stride_xy=18,
                    stride_z=4,
                    dataset_path=args.root_path,
                )
                writer.add_scalar('val/dice', dice_sample, iter_num)
                if dice_sample > best_dice:
                    best_dice = float(round(dice_sample, 7))
                    save_mode_path = os.path.join(snapshot_path, f'iter_{iter_num}_dice_{best_dice}.pth')
                    save_best_path = os.path.join(snapshot_path, f'{args.model}_best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to %s", save_mode_path)
                writer.add_scalar('val/best_dice', best_dice, iter_num)

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    final_path = os.path.join(snapshot_path, f'{args.model}_final_model.pth')
    torch.save(model.state_dict(), final_path)
    writer.close()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.deterministic:
        _set_deterministic(args.seed)

    snapshot_path = "{}/{}_{}_labeled/fully_supervised".format(args.snapshot_path, args.exp, args.labelnum)
    print("Starting fully-supervised training.")

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(os.path.join(snapshot_path, 'code')):
        shutil.rmtree(os.path.join(snapshot_path, 'code'))

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train_fully_supervised(args, snapshot_path)
