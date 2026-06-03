import argparse
import os

import torch

from experiments.cvbm_15_1.t2.a.modules import CVBMArgumentWithCrossSKC3DProto
from networks.net_factory import net_factory
from utils.test_3d_patch import test_all_case, test_all_case_argument


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/LA', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='CVBM_LA_Ablation_ProtoPatch', help='exp_name')
parser.add_argument('--model', type=str, default='CVBM_Argument', help='model_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--detail', type=int, default=1, help='print metrics for every sample')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-processing')
parser.add_argument('--labelnum', type=int, default=8, help='labeled data')
parser.add_argument('--patch_size', type=int, nargs=3, default=(112, 112, 80), help='LA test patch size')
parser.add_argument('--stage_name', type=str, default='self_train', help='self_train or pre_train')
parser.add_argument(
    '--snapshot_path',
    type=str,
    default='./results/CVBM_15_1_t2_a/ablation_04/1/',
    help='snapshot path used by ablation_04/la_train.py',
)
parser.add_argument(
    '--checkpoint',
    type=str,
    default=None,
    help='optional explicit checkpoint path; defaults to <snapshot>/<exp>_<labelnum>_labeled/<stage>/<model>_best_model.pth',
)
parser.add_argument(
    '--test_ema',
    type=int,
    default=1,
    help='also test <model>_ema_best_model.pth when stage_name is self_train',
)
parser.add_argument('--save_result', type=int, default=0, help='save nii.gz predictions')
parser.add_argument(
    '--test_save_path',
    type=str,
    default=None,
    help='optional prediction output directory; defaults under snapshot_path',
)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
num_classes = 2
args.patch_size = tuple(args.patch_size)

snapshot_path = os.path.join(
    args.snapshot_path,
    '{}_{}_labeled'.format(args.exp, args.labelnum),
    args.stage_name,
)
if args.test_save_path is not None:
    test_save_path = args.test_save_path
else:
    test_save_path = os.path.join(
        args.snapshot_path,
        '{}_{}_labeled'.format(args.exp, args.labelnum),
        '{}_predictions'.format(args.model),
    )

os.makedirs(test_save_path, exist_ok=True)
print(test_save_path)

with open(os.path.join(args.root_path, 'test.list'), 'r') as f:
    image_list = f.readlines()
image_list = [
    os.path.join(args.root_path, '2018LA_Seg_Training Set', item.strip(), 'mri_norm2.h5')
    for item in image_list
]


def build_self_train_model():
    return CVBMArgumentWithCrossSKC3DProto(
        n_channels=1,
        n_classes=num_classes,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()


def build_pre_train_model():
    return net_factory(
        net_type=args.model,
        in_chns=1,
        class_num=num_classes,
        mode='test',
    )


def load_checkpoint(model, path):
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'net' in state:
        state = state['net']
    model.load_state_dict(state)


def evaluate_argument_checkpoint(path, tag, model_builder):
    model = model_builder()
    load_checkpoint(model, path)
    print("init {} weight from {}".format(tag, path))
    model.eval()

    return test_all_case_argument(
        model,
        image_list,
        num_classes=num_classes,
        patch_size=args.patch_size,
        stride_xy=18,
        stride_z=4,
        save_result=bool(args.save_result),
        test_save_path=test_save_path,
        metric_detail=args.detail,
        nms=args.nms,
    )


def evaluate_single_input_checkpoint(path, tag):
    model = build_pre_train_model()
    load_checkpoint(model, path)
    print("init {} weight from {}".format(tag, path))
    model.eval()

    return test_all_case(
        model,
        image_list,
        num_classes=num_classes,
        patch_size=args.patch_size,
        stride_xy=18,
        stride_z=4,
        save_result=bool(args.save_result),
        test_save_path=test_save_path,
        metric_detail=args.detail,
        nms=args.nms,
    )


def test_calculate_metric():
    save_model_path = args.checkpoint
    if save_model_path is None:
        save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))

    if args.stage_name == 'pre_train' and args.model == 'CVBM':
        metrics = {'model': evaluate_single_input_checkpoint(save_model_path, 'model')}
        return metrics

    model_builder = build_pre_train_model if args.stage_name == 'pre_train' else build_self_train_model
    metrics = {'model': evaluate_argument_checkpoint(save_model_path, 'model', model_builder)}

    save_ema_model_path = os.path.join(snapshot_path, '{}_ema_best_model.pth'.format(args.model))
    if args.test_ema and args.stage_name == 'self_train' and os.path.exists(save_ema_model_path):
        metrics['ema_model'] = evaluate_argument_checkpoint(save_ema_model_path, 'ema model', build_self_train_model)

    return metrics


if __name__ == '__main__':
    metrics = test_calculate_metric()
    print(metrics)
