import argparse
import ast
import os

import torch
import torch.nn as nn
from networks.CVBM import Decoder, Encoder


class DecoderWithFeature(Decoder):
    def forward(self, features):
        x1, x2, x3, x4, x5 = features
        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4
        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3
        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2
        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        out_seg2 = self.out_conv2(x9)
        out_tanh = self.tanh(out_seg2)
        proto_feature = x9
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        return out_seg, out_tanh, proto_feature


# a1 ablation: evaluate no-SKC dual-branch checkpoints without importing train-time argparse code.
class CVBMArgumentWithoutSKC3DProto(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization="instancenorm", has_dropout=False, has_residual=False):
        super().__init__()
        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder_fg = DecoderWithFeature(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0)
        self.decoder_bg = DecoderWithFeature(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=2)
        self.final_seg = nn.Conv3d(n_classes * 2, n_classes, kernel_size=1)

    def forward(self, input_fg, input_bg):
        fg_feats = list(self.encoder(input_fg))
        bg_feats = list(self.encoder(input_bg))
        out_fg, attn_fg, feat_fg = self.decoder_fg(fg_feats)
        out_bg, attn_bg, feat_bg = self.decoder_bg(bg_feats)
        fused_logits = self.final_seg(torch.cat([out_fg, out_bg], dim=1))
        return out_fg, fused_logits, out_bg, attn_fg, attn_bg, feat_fg, feat_bg

from utils.test_3d_patch import test_all_case_argument


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/Pancreas', help='Pancreas dataset root')
parser.add_argument('--exp', type=str, default='CVBM_Pancreas_FgQBgKV_Ablation_Without_SKC', help='experiment name')
parser.add_argument('--model', type=str, default='CVBM_Argument', help='checkpoint name prefix')
parser.add_argument('--gpu', type=str, default='0', help='GPU id')
parser.add_argument('--detail', type=int, default=1, help='print metrics for every sample')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-processing')
parser.add_argument('--labelnum', type=int, default=12, help='labeled training cases')
parser.add_argument('--label_ratio', type=float, default=None,
                    help='optional labeled ratio in percent; overrides labelnum when provided')
parser.add_argument('--max_samples', type=int, default=62, help='total Pancreas training cases used to compute label ratios')
parser.add_argument('--patch_size', type=ast.literal_eval, default=(96, 96, 96), help='test patch size')
parser.add_argument('--stride_xy', type=int, default=16, help='sliding-window stride for x/y')
parser.add_argument('--stride_z', type=int, default=16, help='sliding-window stride for z')
parser.add_argument('--eval_head', type=str, default='fg', choices=['fg', 'fused'], help='output head to evaluate')
parser.add_argument('--stage_name', type=str, default='auto',
                    choices=['auto', 'pre_train', 'self_train', 'fully_supervised'],
                    help='checkpoint stage; auto uses fully_supervised for all labels, otherwise self_train')
parser.add_argument('--snapshot_path', type=str, default='./results/CVBM_15_1_t2_a/a1/ablation/without_skc_prototype_only/',
                    help='snapshot root used by pancreas_train.py')
parser.add_argument('--checkpoint_dir', type=str, default=None,
                    help='directory containing checkpoints; overrides snapshot_path/exp/labelnum/stage_name')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='explicit checkpoint path; overrides checkpoint_dir')
parser.add_argument('--ema_checkpoint', type=str, default=None,
                    help='explicit EMA checkpoint path; defaults to <checkpoint_dir>/<model>_ema_best_model.pth')
parser.add_argument('--test_ema', type=int, default=1,
                    help='also test <model>_ema_best_model.pth when available')
parser.add_argument('--save_result', action='store_true', help='save prediction nii.gz files')
parser.add_argument('--test_save_path', type=str, default=None, help='prediction output directory')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
num_classes = 2


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


def resolve_stage_name():
    if args.stage_name != 'auto':
        return args.stage_name
    if resolve_labelnum() >= args.max_samples:
        return 'fully_supervised'
    return 'self_train'


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


def build_checkpoint_dir():
    if args.checkpoint_dir is not None:
        return args.checkpoint_dir
    return os.path.join(
        args.snapshot_path,
        '{}_{}_labeled'.format(args.exp, resolve_labelnum()),
        resolve_stage_name(),
    )


def build_test_save_path(tag=None):
    if args.test_save_path is not None:
        base_path = args.test_save_path
    else:
        base_path = os.path.join(
            args.snapshot_path,
            '{}_{}_labeled'.format(args.exp, resolve_labelnum()),
            '{}_{}_{}_predictions'.format(args.model, resolve_stage_name(), args.eval_head),
        )
    if tag is None or tag == 'model':
        return base_path
    return '{}_{}'.format(base_path, tag)


def create_model():
    return CVBMArgumentWithoutSKC3DProto(
        n_channels=1,
        n_classes=num_classes,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()


def load_checkpoint(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError('checkpoint not found: {}'.format(path))
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'net' in state:
        state = state['net']
    model.load_state_dict(state)


def build_image_list():
    with open(os.path.join(args.root_path, 'test.list'), 'r') as f:
        cases = [item.strip() for item in f.readlines()]
    return [
        os.path.join(args.root_path, 'data', '{}_norm.h5'.format(case))
        for case in cases
    ]


def evaluate_single_checkpoint(checkpoint, tag, test_save_path):
    model = create_model()
    load_checkpoint(model, checkpoint)
    print('init {} weight from {}'.format(tag, checkpoint))

    eval_model = EvalHeadWrapper(model, args.eval_head).cuda()
    eval_model.eval()

    metric = test_all_case_argument(
        eval_model,
        build_image_list(),
        num_classes=num_classes,
        patch_size=args.patch_size,
        stride_xy=args.stride_xy,
        stride_z=args.stride_z,
        save_result=args.save_result,
        test_save_path=test_save_path,
        metric_detail=args.detail,
        nms=args.nms,
    )
    print('{} metric [dice, jaccard, hd95, asd]: {}'.format(tag, metric))
    print('{} metric percent [dsc, jaccard]: {:.4f}, {:.4f}'.format(tag, metric[0] * 100, metric[1] * 100))
    return metric


def evaluate():
    checkpoint_dir = build_checkpoint_dir()
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = os.path.join(checkpoint_dir, '{}_best_model.pth'.format(args.model))

    test_save_path = build_test_save_path()
    os.makedirs(test_save_path, exist_ok=True)
    print(test_save_path)

    metrics = {
        'model': evaluate_single_checkpoint(checkpoint, 'model', test_save_path),
    }

    if args.test_ema:
        ema_checkpoint = args.ema_checkpoint
        if ema_checkpoint is None:
            ema_checkpoint_dir = os.path.dirname(checkpoint) if args.checkpoint is not None else checkpoint_dir
            ema_checkpoint = os.path.join(ema_checkpoint_dir, '{}_ema_best_model.pth'.format(args.model))

        if os.path.exists(ema_checkpoint):
            ema_test_save_path = build_test_save_path('ema_model')
            os.makedirs(ema_test_save_path, exist_ok=True)
            print(ema_test_save_path)
            metrics['ema_model'] = evaluate_single_checkpoint(ema_checkpoint, 'ema model', ema_test_save_path)
        else:
            print('skip ema test; checkpoint not found: {}'.format(ema_checkpoint))

    return metrics


if __name__ == '__main__':
    print(evaluate())
