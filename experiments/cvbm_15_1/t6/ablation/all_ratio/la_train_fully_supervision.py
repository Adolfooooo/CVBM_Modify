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
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import LAHeart
from dataloaders.datasets_3d import WeakStrongAugment3d
from experiments.cvbm_15_1.t3.modules import CVBMArgumentWithCrossSKC3DProto
from utils import losses, test_3d_patch


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/xuminghao/Datasets/LA/UA_MT', help='LA dataset root')
parser.add_argument('--exp', type=str, default='CVBM_LA_T3_All_Ratio', help='experiment name')
parser.add_argument('--model', type=str, default='CVBM_Argument', help='checkpoint name prefix')
parser.add_argument('--max_iterations', type=int, default=15000, help='maximum training iterations')
parser.add_argument('--max_samples', type=int, default=80, help='total LA training cases used to compute label ratios')
parser.add_argument('--labelnum', type=int, default=80, help='number of fully supervised labeled training cases')
parser.add_argument('--label_ratio', type=float, default=None,
                    help='optional labeled ratio in percent; overrides labelnum when provided')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--patch_size', type=ast.literal_eval, default=(112, 112, 80),
                    help='training and validation patch size')
parser.add_argument('--base_lr', type=float, default=0.01, help='base learning rate')
parser.add_argument('--deterministic', type=int, default=0, help='use deterministic cudnn and seeds')
parser.add_argument('--gpu', type=str, default='0', help='GPU id')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_workers', type=int, default=8, help='dataloader workers')
parser.add_argument('--val_interval', type=int, default=200, help='validation interval in iterations')
parser.add_argument('--stride_xy', type=int, default=18, help='sliding-window stride for x/y validation')
parser.add_argument('--stride_z', type=int, default=4, help='sliding-window stride for z validation')
parser.add_argument('--eval_head', type=str, default='fg', choices=['fg', 'fused'],
                    help='model output used for online validation')
parser.add_argument('--branch_weight', type=float, default=1.0,
                    help='weight for fg/bg auxiliary branch losses')
parser.add_argument('--snapshot_path', type=str, default='./results/CVBM_15_1/t3/all_ratio',
                    help='snapshot root')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True

num_classes = 2
train_data_path = args.root_path


def resolve_labelnum():
    if args.label_ratio is None:
        labelnum = args.labelnum
    else:
        if args.label_ratio <= 0:
            raise ValueError('--label_ratio must be > 0')
        labelnum = int(round(args.max_samples * args.label_ratio / 100.0))
        labelnum = max(1, labelnum)

    if labelnum < 1:
        raise ValueError('--labelnum must be >= 1')
    if labelnum > args.max_samples:
        raise ValueError('resolved labelnum cannot exceed --max_samples')
    return labelnum


def set_deterministic(seed):
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def create_model():
    return CVBMArgumentWithCrossSKC3DProto(
        n_channels=1,
        n_classes=num_classes,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()


class EvalHeadWrapper(torch.nn.Module):
    def __init__(self, model, head):
        super().__init__()
        self.model = model
        self.head = head

    def forward(self, input_fg, input_bg):
        out_fg, fused_logits, out_bg, *rest = self.model(input_fg, input_bg)
        if self.head == 'fused':
            return fused_logits, out_fg, out_bg, *rest
        return out_fg, fused_logits, out_bg, *rest


def save_checkpoint(model, optimizer, path):
    torch.save({'net': model.state_dict(), 'opt': optimizer.state_dict()}, str(path))


def supervised_loss(logits, target, dice_loss):
    ce = F.cross_entropy(logits, target.long())
    dice = dice_loss(logits, target.long())
    return ce + dice, ce, dice


def build_trainloader():
    labelnum = resolve_labelnum()
    db_train = LAHeart(
        base_dir=train_data_path,
        split='train',
        num=labelnum,
        transform=transforms.Compose([
            WeakStrongAugment3d(args.patch_size, flag_rot=False),
        ]),
    )

    loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': torch.cuda.is_available(),
        'worker_init_fn': worker_init_fn,
        'drop_last': False,
    }
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = True

    return DataLoader(db_train, **loader_kwargs)


def train(snapshot_path):
    model = create_model()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    dice_loss = losses.mask_DiceLoss(nclass=num_classes).cuda()
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))

    trainloader = build_trainloader()
    if len(trainloader) == 0:
        raise RuntimeError('empty trainloader; check --labelnum/--label_ratio and --batch_size')

    labelnum = resolve_labelnum()
    logging.info('%d iterations per epoch', len(trainloader))
    logging.info('label ratio: %.2f%% (%d/%d)', 100.0 * labelnum / args.max_samples, labelnum, args.max_samples)

    iter_num = 0
    best_dice = 0.0
    max_epoch = args.max_iterations // len(trainloader) + 1

    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for sampled_batch in trainloader:
            model.train()
            image = sampled_batch['image'].cuda(non_blocking=True)
            image_strong = sampled_batch['image_strong'].cuda(non_blocking=True)
            label = sampled_batch['label'].cuda(non_blocking=True).long()
            label_strong = sampled_batch['label_strong'].cuda(non_blocking=True).long()

            out_fg, fused_logits, out_bg, *_ = model(image, image_strong)

            loss_main, ce_main, dice_main = supervised_loss(fused_logits, label, dice_loss)
            loss_fg, ce_fg, dice_fg = supervised_loss(out_fg, (label == 1).long(), dice_loss)
            loss_bg, ce_bg, dice_bg = supervised_loss(out_bg, (label_strong == 0).long(), dice_loss)
            loss_aux = 0.5 * (loss_fg + loss_bg)
            loss = loss_main + args.branch_weight * loss_aux

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            writer.add_scalar('train/loss_all', loss.item(), iter_num)
            writer.add_scalar('train/loss_main', loss_main.item(), iter_num)
            writer.add_scalar('train/loss_fg', loss_fg.item(), iter_num)
            writer.add_scalar('train/loss_bg', loss_bg.item(), iter_num)
            writer.add_scalar('train/ce_main', ce_main.item(), iter_num)
            writer.add_scalar('train/dice_main', dice_main.item(), iter_num)
            writer.add_scalar('train/ce_fg', ce_fg.item(), iter_num)
            writer.add_scalar('train/dice_fg', dice_fg.item(), iter_num)
            writer.add_scalar('train/ce_bg', ce_bg.item(), iter_num)
            writer.add_scalar('train/dice_bg', dice_bg.item(), iter_num)

            logging.info(
                'iteration %d : loss: %.6f, main: %.6f, fg: %.6f, bg: %.6f',
                iter_num, loss.item(), loss_main.item(), loss_fg.item(), loss_bg.item(),
            )

            if iter_num % args.val_interval == 0:
                model.eval()
                eval_model = EvalHeadWrapper(model, args.eval_head).cuda()
                dice_sample = test_3d_patch.var_all_case_LA_argument(
                    eval_model,
                    num_classes=num_classes,
                    patch_size=args.patch_size,
                    stride_xy=args.stride_xy,
                    stride_z=args.stride_z,
                    dataset_path=args.root_path,
                )
                writer.add_scalar('val/Dice', dice_sample, iter_num)
                writer.add_scalar('val/Best_dice', best_dice, iter_num)

                if dice_sample > best_dice:
                    best_dice = round(float(dice_sample), 7)
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_checkpoint(model, optimizer, save_mode_path)
                    save_checkpoint(model, optimizer, save_best_path)
                    logging.info('save best model to %s', save_mode_path)

            if iter_num >= args.max_iterations:
                break

        if iter_num >= args.max_iterations:
            iterator.close()
            break

    save_checkpoint(model, optimizer, os.path.join(snapshot_path, '{}_last_model.pth'.format(args.model)))
    writer.close()


if __name__ == '__main__':
    if args.deterministic:
        set_deterministic(args.seed)

    labelnum = resolve_labelnum()
    snapshot_path = os.path.join(
        args.snapshot_path,
        '{}_{}_labeled'.format(args.exp, labelnum),
        'fully_supervised',
    )
    os.makedirs(snapshot_path, exist_ok=True)
    if os.path.exists(os.path.join(snapshot_path, 'code')):
        shutil.rmtree(os.path.join(snapshot_path, 'code'))

    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'log.txt'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(snapshot_path)
