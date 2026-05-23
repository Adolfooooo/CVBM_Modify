import os
import argparse
import ast
import torch

from experiments.cvbm_15_1.t2.a.modules import CVBMArgumentWithCrossSKC3DProto
from utils.test_3d_patch import test_all_case_argument

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/LA', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='CVBM_', help='exp_name')
parser.add_argument('--model', type=str,  default='CVBMArgumentWithSKC3D', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-procssing?')
parser.add_argument('--labelnum', type=int, default=25, help='labeled data')
parser.add_argument('--patch_size', type=ast.literal_eval, default=(96, 96, 96))
parser.add_argument('--stage_name',type=str, default='self_train', help='self_train or pre_train')
parser.add_argument('--snapshot_path', type=str, default='./results', help='snapshot path to save model')
parser.add_argument('--train_num', type=int, default=1, help='the count of train')
parser.add_argument('--result_dir', type=str, default='cvbm_15_1_t2_a',
                    help='directory under snapshot_path used by run_brats19.sh')
parser.add_argument('--proto_weight', type=float, default=0.1, help='weight for branch prototype loss')
parser.add_argument('--proto_patch', type=ast.literal_eval, default=(8, 8, 8),
                    help='patch size for prototype pooling')
parser.add_argument('--saved_train_num', type=int, default=1,
                    help='train_num value used by brats19_train.py when writing checkpoints')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
num_classes = 2


def get_run_snapshot_root():
    snapshot_path = os.path.normpath(args.snapshot_path)
    if os.path.basename(os.path.dirname(snapshot_path)) == args.result_dir:
        return snapshot_path
    if os.path.basename(snapshot_path) == args.result_dir:
        return os.path.join(snapshot_path, str(args.train_num))
    return os.path.join(snapshot_path, args.result_dir, str(args.train_num))


run_snapshot_root = get_run_snapshot_root()
proto_dir = "proto_w{}_patch{}".format(args.proto_weight, args.proto_patch)
snapshot_path = os.path.join(
    run_snapshot_root,
    args.exp,
    'brats19',
    'label{}'.format(args.labelnum),
    proto_dir,
    str(args.saved_train_num),
    args.stage_name,
)
test_save_path = os.path.join(
    run_snapshot_root,
    args.exp,
    'brats19',
    'label{}'.format(args.labelnum),
    proto_dir,
    str(args.saved_train_num),
    '{}_predictions'.format(args.model),
)

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
with open(args.root_path + '/test.list', 'r') as f:
    image_list = f.readlines()
image_list = [args.root_path + "/data/" + item.replace('\n', '') + ".h5" for item in image_list]    


def load_checkpoint(model, path):
    state = torch.load(path)
    if isinstance(state, dict) and 'net' in state:
        state = state['net']
    model.load_state_dict(state)


def test_calculate_metric_argument():
    model = CVBMArgumentWithCrossSKC3DProto(
        n_channels=1,
        n_classes=num_classes,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()
    ema_model = CVBMArgumentWithCrossSKC3DProto(
        n_channels=1,
        n_classes=num_classes,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()

    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
    save_ema_model_path = os.path.join(snapshot_path, '{}_ema_best_model.pth'.format(args.model))

    load_checkpoint(model, save_model_path)
    print("init weight from {}".format(save_model_path))

    model.eval()
    ema_model.eval()

    avg_metric = test_all_case_argument(model, image_list, num_classes=num_classes,
                           patch_size=args.patch_size, stride_xy=16, stride_z=16,
                           save_result=False, test_save_path=test_save_path,
                           metric_detail=args.detail, nms=args.nms)
    
    load_checkpoint(ema_model, save_ema_model_path)
    print("init weight from {}".format(save_ema_model_path))
    ema_avg_metric = test_all_case_argument(ema_model, image_list, num_classes=num_classes,
                           patch_size=args.patch_size, stride_xy=16, stride_z=16,
                           save_result=False, test_save_path=test_save_path,
                           metric_detail=args.detail, nms=args.nms)

    return avg_metric, ema_avg_metric


if __name__ == '__main__':

    metric, ema_metric = test_calculate_metric_argument()
    print(metric, ema_metric)
