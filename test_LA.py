import os
import argparse
import torch
import pdb

from networks.net_factory import net_factory
from utils.test_3d_patch import test_all_case, test_all_case_argument

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/LA', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='CVBM', help='exp_name')
parser.add_argument('--model', type=str,  default='CVBM', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-procssing?')
parser.add_argument('--labelnum', type=int, default=4, help='labeled data')
parser.add_argument('--stage_name',type=str, default='self_train', help='self_train or pre_train')
parser.add_argument('--snapshot_path', type=str, default='./results/CVBM_4_2/1/', help='snapshot path to save model')

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
image_list = [args.root_path + "/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]


def test_calculate_metric():
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="test")

    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))

    model.load_state_dict(torch.load(save_model_path))
    print("init weight from {}".format(save_model_path))

    model.eval()

    avg_metric = test_all_case(model, image_list, num_classes=num_classes,
                           patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                           save_result=False, test_save_path=test_save_path,
                           metric_detail=args.detail, nms=args.nms)

    return avg_metric


def test_calculate_metric_argument():
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="test")

    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
    save_ema_model_path = os.path.join(snapshot_path, '{}_ema_best_model.pth'.format(args.model))

    model.load_state_dict(torch.load(save_model_path))
    print("init weight from {}".format(save_model_path))

    model.eval()

    avg_metric = test_all_case_argument(model, image_list, num_classes=num_classes,
                           patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                           save_result=False, test_save_path=test_save_path,
                           metric_detail=args.detail, nms=args.nms)
    
    model.load_state_dict(torch.load(save_ema_model_path))
    print("init weight from {}".format(save_ema_model_path))
    avg_metric = test_all_case_argument(model, image_list, num_classes=num_classes,
                           patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                           save_result=False, test_save_path=test_save_path,
                           metric_detail=args.detail, nms=args.nms)

    return avg_metric


if __name__ == '__main__':
    if args.model == "CVBM":
        metric = test_calculate_metric()
        print(metric)
    elif args.model == "CVBM_Argument":
        metric = test_calculate_metric_argument()
        print(metric)

# python test_LA.py --model 0214_re01 --gpu 0
