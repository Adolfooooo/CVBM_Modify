import argparse
import ast
import os

import torch

from experiments.cvbm_15_1.t2.a.a1.modules import CVBMArgumentWithCrossSKC3DProto
from utils.test_3d_patch import test_all_case_argument


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/BRATS19', help='BRATS19 dataset root')
parser.add_argument('--exp', type=str, default='CVBM_BRATS19_FgWeak_BgWeak_Ablation_07_02', help='experiment name')
parser.add_argument('--model', type=str, default='CVBM_Argument', help='checkpoint name prefix')
parser.add_argument('--gpu', type=str, default='0', help='GPU id')
parser.add_argument('--detail', type=int, default=1, help='print metrics for every sample')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-processing')
parser.add_argument('--labelnum', type=int, default=25, help='labeled training cases')
parser.add_argument('--label_ratio', type=float, default=None,
                    help='optional labeled ratio in percent; overrides labelnum when provided')
parser.add_argument('--max_samples', type=int, default=250, help='total BRATS19 training cases used to compute label ratios')
parser.add_argument('--patch_size', type=ast.literal_eval, default=(96, 96, 96), help='test patch size')
parser.add_argument('--stride_xy', type=int, default=16, help='sliding-window stride for x/y')
parser.add_argument('--stride_z', type=int, default=16, help='sliding-window stride for z')
parser.add_argument('--num_classes', type=int, default=2, help="number of dataset's classes")
parser.add_argument('--eval_head', type=str, default='fg', choices=['fg', 'fused'], help='output head to evaluate')
parser.add_argument('--stage_name', type=str, default='auto',
                    choices=['auto', 'pre_train', 'self_train', 'fully_supervised'],
                    help='checkpoint stage; auto uses fully_supervised for all labels, otherwise self_train')
parser.add_argument('--snapshot_path', type=str, default='./results/CVBM_15_1_t2_a/a1/ablation_07/ab_07_02/1/',
                    help='snapshot root used by brats19_train.py')
parser.add_argument('--proto_weight', type=float, default=0.1, help='weight for branch prototype loss')
parser.add_argument('--proto_patch', type=ast.literal_eval, default=(8, 8, 8),
                    help='patch size for prototype pooling')
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
num_classes = args.num_classes


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


def build_proto_dir():
    return 'proto_w{}_patch{}'.format(args.proto_weight, args.proto_patch)


def build_checkpoint_dir():
    if args.checkpoint_dir is not None:
        return args.checkpoint_dir
    return os.path.join(
        args.snapshot_path,
        args.exp,
        'brats19',
        'label{}'.format(resolve_labelnum()),
        build_proto_dir(),
        resolve_stage_name(),
    )


def build_test_save_path(tag=None):
    if args.test_save_path is not None:
        base_path = args.test_save_path
    else:
        base_path = os.path.join(
            args.snapshot_path,
            args.exp,
            'brats19',
            'label{}'.format(resolve_labelnum()),
            build_proto_dir(),
            '{}_{}_{}_predictions'.format(args.model, resolve_stage_name(), args.eval_head),
        )
    if tag is None or tag == 'model':
        return base_path
    return '{}_{}'.format(base_path, tag)


def create_model():
    return CVBMArgumentWithCrossSKC3DProto(
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
        os.path.join(args.root_path, 'data', '{}.h5'.format(case))
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
