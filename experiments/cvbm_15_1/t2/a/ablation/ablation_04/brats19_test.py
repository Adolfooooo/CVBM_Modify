import argparse
import os

import torch

from experiments.cvbm_15_1.t2.a.modules import CVBMArgumentWithCrossSKC3DProto
from networks.net_factory import net_factory
from utils.test_3d_patch import test_all_case, test_all_case_argument


def parse_3d_tuple(values):
    if isinstance(values, tuple):
        parsed = values
    elif isinstance(values, list):
        parsed = tuple(int(v) for v in values)
    else:
        text = str(values).strip().replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        parts = [item.strip() for item in text.split(",") if item.strip()]
        if len(parts) == 1:
            parts = text.split()
        parsed = tuple(int(v) for v in parts if str(v).strip())

    if len(parsed) != 3:
        raise argparse.ArgumentTypeError(f"expected 3 integers, got {values}")
    if any(v <= 0 for v in parsed):
        raise argparse.ArgumentTypeError(f"all values must be > 0, got {values}")
    return parsed


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/BRATS19', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='CVBM_BRATS19_Ablation_ProtoPatch', help='exp_name')
parser.add_argument('--model', type=str, default='CVBM_Argument', help='model_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--detail', type=int, default=1, help='print metrics for every sample')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-processing')
parser.add_argument('--labelnum', type=int, default=25, help='labeled data')
parser.add_argument('--patch_size', type=int, nargs=3, default=(96, 96, 96), help='BRATS19 test patch size')
parser.add_argument('--num_classes', type=int, default=2, help="number of dataset's classes")
parser.add_argument('--stride_xy', type=int, default=16, help='sliding-window stride in x/y')
parser.add_argument('--stride_z', type=int, default=16, help='sliding-window stride in z')
parser.add_argument('--stage_name', type=str, default='self_train', choices=['pre_train', 'self_train'],
                    help='checkpoint stage to evaluate')
parser.add_argument(
    '--snapshot_path',
    type=str,
    default='./results/CVBM_15_1_t2_a/ablation_04/1/',
    help='snapshot path used by ablation_04/brats19_train.py',
)
parser.add_argument('--proto_weight', type=float, default=0.1, help='weight for branch prototype loss')
parser.add_argument('--proto_patch', type=int, nargs=3, default=(8, 8, 8), help='patch size for prototype pooling')
parser.add_argument(
    '--checkpoint',
    type=str,
    default=None,
    help='optional explicit checkpoint path; defaults to <snapshot>/<exp>/brats19/label<labelnum>/proto_w..._patch.../<stage>/<model>_best_model.pth',
)
parser.add_argument(
    '--test_ema',
    type=int,
    default=1,
    help='also test <model>_ema_best_model.pth when stage_name is self_train',
)
parser.add_argument('--save_result', action='store_true', help='save nii.gz predictions')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.patch_size = parse_3d_tuple(args.patch_size)
args.proto_patch = parse_3d_tuple(args.proto_patch)

proto_dir = "proto_w{}_patch{}".format(args.proto_weight, args.proto_patch)
snapshot_path = os.path.join(
    args.snapshot_path,
    args.exp,
    'brats19',
    'label{}'.format(args.labelnum),
    proto_dir,
    args.stage_name,
)
test_save_path = os.path.join(
    args.snapshot_path,
    args.exp,
    'brats19',
    'label{}'.format(args.labelnum),
    proto_dir,
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


def build_self_train_model():
    return CVBMArgumentWithCrossSKC3DProto(
        n_channels=1,
        n_classes=args.num_classes,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()


def build_pre_train_model():
    return net_factory(
        net_type=args.model,
        in_chns=1,
        class_num=args.num_classes,
        mode='test',
    )


def load_checkpoint(model, path):
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'net' in state:
        state = state['net']
    model.load_state_dict(state)


def evaluate_argument_checkpoint(checkpoint_path, tag, model_builder):
    model = model_builder()
    load_checkpoint(model, checkpoint_path)
    print("init {} weight from {}".format(tag, checkpoint_path))
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


def evaluate_single_input_checkpoint(checkpoint_path, tag):
    model = build_pre_train_model()
    load_checkpoint(model, checkpoint_path)
    print("init {} weight from {}".format(tag, checkpoint_path))
    model.eval()

    return test_all_case(
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


def test_calculate_metric():
    save_model_path = args.checkpoint
    if save_model_path is None:
        save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))

    if args.stage_name == 'pre_train' and args.model == 'CVBM':
        return {'model': evaluate_single_input_checkpoint(save_model_path, 'model')}

    model_builder = build_pre_train_model if args.stage_name == 'pre_train' else build_self_train_model
    metrics = {'model': evaluate_argument_checkpoint(save_model_path, 'model', model_builder)}

    save_ema_model_path = os.path.join(snapshot_path, '{}_ema_best_model.pth'.format(args.model))
    if args.test_ema and args.stage_name == 'self_train' and os.path.exists(save_ema_model_path):
        metrics['ema_model'] = evaluate_argument_checkpoint(save_ema_model_path, 'ema model', build_self_train_model)

    return metrics


if __name__ == '__main__':
    metrics = test_calculate_metric()
    print(metrics)
