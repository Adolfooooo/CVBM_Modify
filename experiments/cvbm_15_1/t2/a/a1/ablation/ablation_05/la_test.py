import argparse
import ast
import os

import torch

# a1 migration: use the Fg-query-Bg-KV Cross-SKC backbone for SKC-preserving ablations.
from experiments.cvbm_15_1.t2.a.a1.modules import CVBMArgumentWithCrossSKC3DProto
from utils.test_3d_patch import test_all_case_argument


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/LA', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='CVBM_LA_FgQBgKV_Ablation_ProtoThreshold', help='exp_name')
parser.add_argument('--model', type=str, default='CVBM_Argument', help='model_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--detail', type=int, default=1, help='print metrics for every sample')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-processing')
parser.add_argument('--labelnum', type=int, default=8, help='labeled data')
parser.add_argument('--patch_size', type=ast.literal_eval, default=(112, 112, 80), help='LA test patch size')
parser.add_argument('--stride_xy', type=int, default=18, help='sliding-window stride for x/y')
parser.add_argument('--stride_z', type=int, default=4, help='sliding-window stride for z')
parser.add_argument('--stage_name', type=str, default='self_train', help='self_train or pre_train')
parser.add_argument(
    '--snapshot_path',
    type=str,
    default='./results/CVBM_15_1_t2_a/a1/ablation_05/1/',
    help='snapshot root used by la_train.py',
)
parser.add_argument(
    '--checkpoint_dir',
    type=str,
    default=None,
    help='optional directory containing checkpoints; overrides snapshot_path/exp/labelnum/stage_name',
)
parser.add_argument(
    '--checkpoint',
    type=str,
    default=None,
    help='optional explicit checkpoint path; overrides checkpoint_dir and snapshot path resolution',
)
parser.add_argument(
    '--test_ema',
    type=int,
    default=1,
    help='also test <model>_ema_best_model.pth when stage_name is self_train and the file exists',
)
parser.add_argument('--save_result', action='store_true', help='save prediction nii.gz files')
parser.add_argument(
    '--test_save_path',
    type=str,
    default=None,
    help='optional prediction output directory; defaults under snapshot_path',
)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
num_classes = 2


def build_checkpoint_dir():
    if args.checkpoint_dir is not None:
        return args.checkpoint_dir
    return os.path.join(
        args.snapshot_path,
        '{}_{}_labeled'.format(args.exp, args.labelnum),
        args.stage_name,
    )


def build_test_save_path():
    if args.test_save_path is not None:
        path = args.test_save_path
    else:
        path = os.path.join(
            args.snapshot_path,
            '{}_{}_labeled'.format(args.exp, args.labelnum),
            '{}_predictions'.format(args.model),
        )
    return os.path.join(path, '')


snapshot_path = build_checkpoint_dir()
test_save_path = build_test_save_path()
os.makedirs(test_save_path, exist_ok=True)
print(test_save_path)

with open(os.path.join(args.root_path, 'test.list'), 'r') as f:
    image_list = f.readlines()
image_list = [
    os.path.join(args.root_path, '2018LA_Seg_Training Set', item.strip(), 'mri_norm2.h5')
    for item in image_list
]


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


def evaluate_checkpoint(path, tag):
    model = create_model()
    load_checkpoint(model, path)
    print('init {} weight from {}'.format(tag, path))
    model.eval()

    return test_all_case_argument(
        model,
        image_list,
        num_classes=num_classes,
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

    metrics = {'model': evaluate_checkpoint(save_model_path, 'model')}

    save_ema_model_path = os.path.join(snapshot_path, '{}_ema_best_model.pth'.format(args.model))
    if args.test_ema and args.stage_name == 'self_train' and os.path.exists(save_ema_model_path):
        metrics['ema_model'] = evaluate_checkpoint(save_ema_model_path, 'ema_model')

    return metrics


if __name__ == '__main__':
    metrics = test_calculate_metric_argument()
    print(metrics)
