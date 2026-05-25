import argparse
import ast
import os

import torch

from experiments.cvbm_15_1.t2.a.modules import CVBMArgumentWithCrossSKC3DProto
from utils.test_3d_patch import test_all_case_argument


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/BRATS19', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='CVBM_BRATS19_Ablation_Remove_Contrast_Keep_SKC',
                    help='exp_name')
parser.add_argument('--model', type=str, default='CVBM_Argument', help='model_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--detail', type=int, default=1, help='print metrics for every samples')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-procssing?')
parser.add_argument('--labelnum', type=int, default=25, help='labeled data')
parser.add_argument('--patch_size', type=ast.literal_eval, default=(96, 96, 96))
parser.add_argument('--num_classes', type=int, default=2, help="number of dataset's class")
parser.add_argument('--stride_xy', type=int, default=16)
parser.add_argument('--stride_z', type=int, default=16)
parser.add_argument('--stage_name', type=str, default='self_train', choices=['pre_train', 'self_train'],
                    help='checkpoint stage to evaluate')
parser.add_argument('--snapshot_path', type=str, default='./results/ablation_03_remove_contrast_keep_skc/',
                    help='snapshot path used by brats19_train.py')
parser.add_argument('--train_num', type=int, default=1, help='the count of train')
parser.add_argument('--save_result', action='store_true', help='save nii.gz predictions')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


snapshot_path = os.path.join(
    args.snapshot_path,
    args.exp,
    'brats19',
    'label{}'.format(args.labelnum),
    str(args.train_num),
    args.stage_name,
)
test_save_path = os.path.join(
    args.snapshot_path,
    args.exp,
    'brats19',
    'label{}'.format(args.labelnum),
    str(args.train_num),
    '{}_predictions'.format(args.model),
)
test_save_path = test_save_path + os.sep

os.makedirs(test_save_path, exist_ok=True)
performance_path = os.path.join(test_save_path, '..', 'performance.txt')
if not os.path.exists(performance_path):
    open(performance_path, 'a').close()

print(test_save_path)
with open(os.path.join(args.root_path, 'test.list'), 'r') as f:
    image_list = f.readlines()
image_list = [
    os.path.join(args.root_path, 'data', item.strip() + '.h5')
    for item in image_list
]


def load_checkpoint(model, path):
    state = torch.load(path)
    if isinstance(state, dict) and 'net' in state:
        state = state['net']
    model.load_state_dict(state)


def build_model():
    return CVBMArgumentWithCrossSKC3DProto(
        n_channels=1,
        n_classes=args.num_classes,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()


def evaluate_checkpoint(checkpoint_path):
    model = build_model()
    load_checkpoint(model, checkpoint_path)
    print("init weight from {}".format(checkpoint_path))
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
    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
    metric = evaluate_checkpoint(save_model_path)

    save_ema_model_path = os.path.join(snapshot_path, '{}_ema_best_model.pth'.format(args.model))
    if args.stage_name == 'self_train' and os.path.exists(save_ema_model_path):
        ema_metric = evaluate_checkpoint(save_ema_model_path)
        return metric, ema_metric

    return metric, None


if __name__ == '__main__':
    metric, ema_metric = test_calculate_metric_argument()
    if ema_metric is None:
        print(metric)
    else:
        print(metric, ema_metric)
