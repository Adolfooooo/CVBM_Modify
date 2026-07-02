import argparse
import ast
import os

import torch

# a1 migration: use the Fg-query-Bg-KV Cross-SKC backbone for SKC-preserving ablations.
from experiments.cvbm_15_1.t2.a.a1.modules import CVBMArgumentWithCrossSKC3DProto
from utils.test_3d_patch import test_all_case_argument


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/xuminghao/Datasets/LA/UA_MT', help='LA dataset root')
parser.add_argument('--exp', type=str, default='CVBM_LA_FgQBgKV_All_Ratio', help='experiment name')
parser.add_argument('--model', type=str, default='CVBM_Argument', help='checkpoint name prefix')
parser.add_argument('--gpu', type=str, default='0', help='GPU id')
parser.add_argument('--detail', type=int, default=1, help='print metrics for every sample')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-processing')
parser.add_argument('--labelnum', type=int, default=80, help='labeled training cases')
parser.add_argument('--label_ratio', type=float, default=None,
                    help='optional labeled ratio in percent; overrides labelnum when provided')
parser.add_argument('--max_samples', type=int, default=80, help='total LA training cases used to compute label ratios')
parser.add_argument('--patch_size', type=ast.literal_eval, default=(112, 112, 80), help='test patch size')
parser.add_argument('--stride_xy', type=int, default=18, help='sliding-window stride for x/y')
parser.add_argument('--stride_z', type=int, default=4, help='sliding-window stride for z')
parser.add_argument('--eval_head', type=str, default='fg', choices=['fg', 'fused'], help='output head to evaluate')
parser.add_argument('--snapshot_path', type=str, default='./results/CVBM_15_1/t2/a/a1/all_ratio',
                    help='snapshot root used by la_train.py')
parser.add_argument('--checkpoint_dir', type=str, default=None,
                    help='directory containing checkpoints; overrides snapshot_path/exp/labelnum')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='explicit checkpoint path; overrides checkpoint_dir')
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
        'fully_supervised',
    )


def build_test_save_path():
    if args.test_save_path is not None:
        return args.test_save_path
    return os.path.join(
        args.snapshot_path,
        '{}_{}_labeled'.format(args.exp, resolve_labelnum()),
        '{}_{}_predictions'.format(args.model, args.eval_head),
    )


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
        os.path.join(args.root_path, '2018LA_Seg_Training Set', case, 'mri_norm2.h5')
        for case in cases
    ]


def evaluate():
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = os.path.join(build_checkpoint_dir(), '{}_best_model.pth'.format(args.model))

    test_save_path = build_test_save_path()
    os.makedirs(test_save_path, exist_ok=True)
    print(test_save_path)

    model = create_model()
    load_checkpoint(model, checkpoint)
    print('init weight from {}'.format(checkpoint))

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
    print('metric [dice, jaccard, hd95, asd]: {}'.format(metric))
    print('metric percent [dsc, jaccard]: {:.4f}, {:.4f}'.format(metric[0] * 100, metric[1] * 100))
    return metric


if __name__ == '__main__':
    evaluate()
