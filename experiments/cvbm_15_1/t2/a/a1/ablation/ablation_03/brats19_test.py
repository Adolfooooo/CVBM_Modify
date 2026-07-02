import argparse
import ast
import os

import torch

from utils.test_3d_patch import test_all_case_argument
# a1 migration: ablation_03 keeps the no-interaction SKC idea in local modules.
from .modules import CVBMArgumentWithCrossSKC3DProto


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/BRATS19', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='CVBM_BRATS19_FgQBgKV_Ablation_SKC_Self_Attention', help='exp_name')
parser.add_argument('--model', type=str, default='CVBM_Argument', help='model_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--detail', type=int, default=1, help='print metrics for every sample')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-processing')
parser.add_argument('--labelnum', type=int, default=25, help='labeled data')
parser.add_argument('--patch_size', type=ast.literal_eval, default=(96, 96, 96), help='test patch size')
parser.add_argument('--num_classes', type=int, default=2, help="number of dataset's class")
parser.add_argument('--stride_xy', type=int, default=16, help='sliding-window stride for x/y')
parser.add_argument('--stride_z', type=int, default=16, help='sliding-window stride for z')
parser.add_argument('--stage_name', type=str, default='self_train', help='self_train or pre_train')
parser.add_argument('--snapshot_path', type=str, default='./results/CVBM_15_1_t2_a/a1/ablation_03/1/',
                    help='snapshot root used by brats19_train.py')
parser.add_argument('--proto_weight', type=float, default=0.1, help='weight for branch prototype loss')
parser.add_argument('--proto_patch', type=ast.literal_eval, default=(8, 8, 8),
                    help='patch size for prototype pooling')
parser.add_argument('--datafolder_name', type=str, default='data', help='directory under root_path containing h5 files')
parser.add_argument('--checkpoint_dir', type=str, default=None,
                    help='optional directory containing model checkpoints; overrides snapshot_path-derived path')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='optional explicit checkpoint path; overrides checkpoint_dir and snapshot path resolution')
parser.add_argument('--test_ema', type=int, default=0,
                    help='also test <model>_ema_best_model.pth when stage_name is self_train and the file exists')
parser.add_argument('--save_result', action='store_true', help='save prediction nii.gz files')
parser.add_argument('--test_save_path', type=str, default=None,
                    help='optional prediction output directory; defaults under snapshot_path')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def build_proto_dir():
    return 'proto_w{}_patch{}'.format(args.proto_weight, args.proto_patch)


def build_checkpoint_dir():
    if args.checkpoint_dir is not None:
        return args.checkpoint_dir
    checkpoint_dir = os.path.join(
        args.snapshot_path,
        '{}_{}_labeled'.format(args.exp, args.labelnum),
        args.stage_name,
    )
    legacy_checkpoint_dir = os.path.join(
        args.snapshot_path,
        args.exp,
        'brats19',
        'label{}'.format(args.labelnum),
        build_proto_dir(),
        args.stage_name,
    )
    default_checkpoint = os.path.join(checkpoint_dir, '{}_best_model.pth'.format(args.model))
    legacy_checkpoint = os.path.join(legacy_checkpoint_dir, '{}_best_model.pth'.format(args.model))
    if args.checkpoint is None and not os.path.exists(default_checkpoint) and os.path.exists(legacy_checkpoint):
        print('fallback checkpoint path: {}'.format(legacy_checkpoint_dir))
        return legacy_checkpoint_dir
    return checkpoint_dir


def build_test_save_path():
    if args.test_save_path is not None:
        return args.test_save_path
    return os.path.join(
        args.snapshot_path,
        '{}_{}_labeled'.format(args.exp, args.labelnum),
        '{}_predictions'.format(args.model),
    )


snapshot_path = build_checkpoint_dir()
test_save_path = build_test_save_path()
os.makedirs(test_save_path, exist_ok=True)
print(test_save_path)

with open(os.path.join(args.root_path, 'test.list'), 'r') as f:
    image_list = f.readlines()
image_list = [
    os.path.join(args.root_path, args.datafolder_name, item.strip() + '.h5')
    for item in image_list
]


def load_checkpoint(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError('checkpoint not found: {}'.format(path))
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'net' in state:
        state = state['net']
    model.load_state_dict(state)


def create_model():
    return CVBMArgumentWithCrossSKC3DProto(
        n_channels=1,
        n_classes=args.num_classes,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()


def evaluate_checkpoint(ckpt_path, name):
    model = create_model()
    load_checkpoint(model, ckpt_path)
    print('init {} weight from {}'.format(name, ckpt_path))
    model.eval()

    return test_all_case_argument(
        model,
        image_list,
        num_classes=args.num_classes,
        patch_size=args.patch_size,
        stride_xy=args.stride_xy,
        stride_z=args.stride_z,
        save_result=args.save_result,
        test_save_path=test_save_path,
        metric_detail=args.detail,
        nms=args.nms,
    )


def test_calculate_metric_argument():
    save_model_path = args.checkpoint
    if save_model_path is None:
        save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
    metric = evaluate_checkpoint(save_model_path, 'model')

    ema_metric = None
    save_ema_model_path = os.path.join(snapshot_path, '{}_ema_best_model.pth'.format(args.model))
    if args.test_ema and args.stage_name == 'self_train' and os.path.exists(save_ema_model_path):
        ema_metric = evaluate_checkpoint(save_ema_model_path, 'ema_model')

    return metric, ema_metric


if __name__ == '__main__':
    metric, ema_metric = test_calculate_metric_argument()
    print('model metric: {}'.format(metric))
    if ema_metric is not None:
        print('ema metric: {}'.format(ema_metric))
