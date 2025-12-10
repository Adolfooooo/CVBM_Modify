import os
import argparse
import torch
import pdb

from networks.net_factory import net_factory
from utils.test_3d_patch import test_all_case, test_all_case_argument
from .modules import CVBMArgumentWithSKC3D

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/LA', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='CVBM_BRATS19', help='exp_name')
parser.add_argument('--model', type=str,  default='CVBMArgumentWithSKC3D', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-procssing?')
parser.add_argument('--labelnum', type=int, default=25, help='labeled data')
parser.add_argument('--patch_size', type=tuple, default=(96, 96, 96))
parser.add_argument('--stage_name',type=str, default='self_train', help='self_train or pre_train')
parser.add_argument('--snapshot_path', type=str, default='./results/CVBM_11_1/1/', help='snapshot path to save model')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
snapshot_path = "{}/{}_{}_labeled/{}".format(args.snapshot_path, args.exp, args.labelnum, args.stage_name)
test_save_path = "{}/{}_{}_labeled/{}_predictions/".format(args.snapshot_path, args.exp, args.labelnum, args.model)
num_classes = 2

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
with open(args.root_path + '/test.list', 'r') as f:
    image_list = f.readlines()
image_list = [args.root_path + "/data/" + item.replace('\n', '') + ".h5" for item in image_list]    

def test_calculate_metric_argument():
    # model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="test")
    model = CVBMArgumentWithSKC3D(
        n_channels=1,
        n_classes=num_classes,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()
    ema_model = CVBMArgumentWithSKC3D(
        n_channels=1,
        n_classes=num_classes,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()

    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
    save_ema_model_path = os.path.join(snapshot_path, '{}_ema_best_model.pth'.format(args.model))

    model.load_state_dict(torch.load(save_model_path))
    print("init weight from {}".format(save_model_path))

    model.eval()
    ema_model.eval()

    avg_metric = test_all_case_argument(model, image_list, num_classes=num_classes,
                           patch_size=args.patch_size, stride_xy=16, stride_z=16,
                           save_result=False, test_save_path=test_save_path,
                           metric_detail=args.detail, nms=args.nms)
    
    ema_model.load_state_dict(torch.load(save_ema_model_path))
    print("init weight from {}".format(save_ema_model_path))
    ema_avg_metric = test_all_case_argument(ema_model, image_list, num_classes=num_classes,
                           patch_size=args.patch_size, stride_xy=16, stride_z=16,
                           save_result=False, test_save_path=test_save_path,
                           metric_detail=args.detail, nms=args.nms)

    return avg_metric, ema_avg_metric


if __name__ == '__main__':

    metric, ema_metric = test_calculate_metric_argument()
    print(metric, ema_metric)
