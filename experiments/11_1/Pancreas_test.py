import os
import argparse
import torch

from utils.test_3d_patch import test_all_case_argument
from .skc3d_module import CVBMArgumentWithSKC3D

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/Pancreas', help='Name of dataset root')
parser.add_argument('--exp', type=str, default='CVBM_Pancreas', help='exp_name')
parser.add_argument('--model', type=str, default='CVBM_Argument', help='model_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--detail', type=int, default=1, help='print metrics for every sample')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-processing')
parser.add_argument('--labelnum', type=int, default=12, help='number of labeled training samples')
parser.add_argument('--stage_name', type=str, default='self_train', help='self_train or pre_train')
parser.add_argument('--snapshot_path', type=str, default='./results/CVBM_11_1/1/', help='snapshot path base')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

snapshot_path = f"{args.snapshot_path}/{args.exp}_{args.labelnum}_labeled/{args.stage_name}"
test_save_path = f"{args.snapshot_path}/{args.exp}_{args.labelnum}_labeled/{args.model}_predictions/"
num_classes = 2

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
with open(args.root_path + '/test.list', 'r') as f:
    image_list = f.readlines()
image_list = [f"{args.root_path}/data/" + item.replace('\n', '') + "_norm.h5" for item in image_list]


def test_calculate_metric_argument():
    model = CVBMArgumentWithSKC3D(
        n_channels=1,
        n_classes=num_classes,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()

    save_model_path = os.path.join(snapshot_path, f'{args.model}_best_model.pth')
    save_ema_model_path = os.path.join(snapshot_path, f'{args.model}_ema_best_model.pth')

    model.load_state_dict(torch.load(save_model_path))
    print(f"init weight from {save_model_path}")

    model.eval()
    avg_metric = test_all_case_argument(
        model,
        image_list,
        num_classes=num_classes,
        patch_size=(96, 96, 96),
        stride_xy=16,
        stride_z=16,
        save_result=False,
        test_save_path=test_save_path,
        metric_detail=args.detail,
        nms=args.nms,
    )

    ema_metric = None
    if os.path.exists(save_ema_model_path):
        model.load_state_dict(torch.load(save_ema_model_path))
        print(f"init weight from {save_ema_model_path}")
        ema_metric = test_all_case_argument(
            model,
            image_list,
            num_classes=num_classes,
            patch_size=(96, 96, 96),
            stride_xy=16,
            stride_z=16,
            save_result=False,
            test_save_path=test_save_path,
            metric_detail=args.detail,
            nms=args.nms,
        )

    return avg_metric, ema_metric


if __name__ == '__main__':
    metric, ema_metric = test_calculate_metric_argument()
    print(metric)
    if ema_metric is not None:
        print(ema_metric)

