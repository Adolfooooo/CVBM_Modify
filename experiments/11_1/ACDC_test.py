import os
import argparse
import torch
import h5py
import numpy as np

from utils import val_2d
from .modules import CVBMArgumentWithSKC2D

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/ACDC', help='Name of dataset root')
parser.add_argument('--exp', type=str, default='CVBM2d_ACDC', help='exp_name')
parser.add_argument('--model', type=str, default='CVBM2d_Argument', help='model_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--labelnum', type=int, default=3, help='number of labeled training samples')
parser.add_argument('--stage_name', type=str, default='self_train', help='self_train or pre_train')
parser.add_argument('--snapshot_path', type=str, default='./results/CVBM_ACDC_11_1/1/', help='snapshot path base')
parser.add_argument('--num_classes', type=int, default=4, help='number of classes')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size for sliding window')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

snapshot_path = f"{args.snapshot_path}/{args.exp}_{args.labelnum}_labeled/{args.stage_name}"
test_save_path = f"{args.snapshot_path}/{args.exp}_{args.labelnum}_labeled/{args.model}_predictions/"

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
with open(args.root_path + '/test.list', 'r') as f:
    sample_list = f.readlines()
sample_list = [item.replace('\n', '') for item in sample_list]
h5_list = [f"{args.root_path}/data/{case}.h5" for case in sample_list]


def test_calculate_metric():
    model = CVBMArgumentWithSKC2D(
        n_channels=1,
        n_classes=args.num_classes,
        has_dropout=True,
    ).cuda()

    save_model_path = os.path.join(snapshot_path, f'{args.model}_best_model.pth')
    save_ema_model_path = os.path.join(snapshot_path, f'{args.model}_ema_best_model.pth')

    model.load_state_dict(torch.load(save_model_path))
    print(f"init weight from {save_model_path}")

    model.eval()
    metric_list = []
    for h5_path in h5_list:
        with h5py.File(h5_path, 'r') as h5f:
            image, label = h5f['image'][:], h5f['label'][:]
        metric_case = val_2d.test_single_volume(torch.from_numpy(image).unsqueeze(0),
                                                torch.from_numpy(label).unsqueeze(0),
                                                model,
                                                args.num_classes,
                                                patch_size=args.patch_size)
        metric_list.append(metric_case)
    metric_arr = np.array(metric_list)
    avg_metric = metric_arr.mean(axis=0)

    ema_metric = None
    if os.path.exists(save_ema_model_path):
        model.load_state_dict(torch.load(save_ema_model_path))
        print(f"init weight from {save_ema_model_path}")
        metric_list = []
        for h5_path in h5_list:
            with h5py.File(h5_path, 'r') as h5f:
                image, label = h5f['image'][:], h5f['label'][:]
            metric_case = val_2d.test_single_volume(torch.from_numpy(image).unsqueeze(0),
                                                    torch.from_numpy(label).unsqueeze(0),
                                                    model,
                                                    args.num_classes,
                                                    patch_size=args.patch_size)
            metric_list.append(metric_case)
        ema_metric = np.array(metric_list).mean(axis=0)

    return avg_metric, ema_metric


if __name__ == '__main__':
    metric, ema_metric = test_calculate_metric()
    print(metric)
    if ema_metric is not None:
        print(ema_metric)
