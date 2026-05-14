import argparse
import os

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from skimage.measure import label
from tqdm import tqdm

from dataloaders.dataset import BaseDataSets
from networks.net_factory import net_factory
from .modules import CVBMArgumentWithCrossSKC2D


parser = argparse.ArgumentParser(description='ACDC evaluation for CVBM 13_1 CrossSKC2D')
parser.add_argument('--root_path', type=str, default='/root/ACDC', help='dataset root')
parser.add_argument('--exp', type=str, default='CVBM_13_1', help='experiment name')
parser.add_argument('--model', type=str, default='CVBM2d_Argument', help='checkpoint/model tag')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--labelnum', type=int, default=3, help='number of labeled training samples')
parser.add_argument('--stage_name', type=str, default='self_train', help='self_train or pre_train')
parser.add_argument('--snapshot_path', type=str, default='./results', help='snapshot path base')
parser.add_argument('--num_classes', type=int, default=4, help='number of classes')
parser.add_argument('--topnum', type=int, default=64, help='contrastive topnum used in training path')
parser.add_argument('--contrast_patch', type=str, default='8x8', help='contrastive patch tag used in training path')
parser.add_argument('--train_num', type=int, default=1, help='training run index')
parser.add_argument('--post', action='store_true', help='apply largest-connected-component post process')
args = parser.parse_args()

if args.contrast_patch.startswith("("):
    contrast_values = args.contrast_patch.strip("()").split(",")
    args.contrast_patch = f"{contrast_values[0].strip()}x{contrast_values[1].strip()}"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

snapshot_path = (
    f"{args.snapshot_path}/{args.exp}/acdc/label{args.labelnum}/"
    f"topnum{args.topnum}_patch{args.contrast_patch}/{args.train_num}/{args.stage_name}"
)


def get_acdc_2d_cct(segmentation):
    batch_list = []
    mask = torch.argmax(segmentation, dim=1).detach().cpu().numpy()
    batch_size = segmentation.shape[0]
    for i in range(batch_size):
        class_list = []
        for c in range(1, 4):
            labels = label(mask[i] == c)
            if labels.max() != 0:
                largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                if largest_cc.sum() < 10:
                    class_list.append(largest_cc * 0)
                else:
                    class_list.append(largest_cc * c)
            else:
                class_list.append(labels)
        merged = class_list[0]
        for class_mask in class_list[1:]:
            merged = merged + class_mask
        batch_list.append(merged)
    batch_list = torch.as_tensor(np.asarray(batch_list), device=segmentation.device, dtype=torch.int64)
    if len(torch.unique(batch_list)) == 1:
        return segmentation
    return torch.nn.functional.one_hot(batch_list, num_classes=args.num_classes).permute(0, 3, 1, 2) * segmentation


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, jc, hd, asd


def test_single_case(net, image, cct=False):
    volume_pred = np.zeros_like(image)
    for ind in range(image.shape[0]):
        slice_img = image[ind, :, :]
        x, y = slice_img.shape[0], slice_img.shape[1]
        slice_img = zoom(slice_img, (256 / x, 256 / y), order=0)
        input_tensor = torch.tensor(slice_img, dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(1)
        with torch.no_grad():
            out_main = net(input_tensor, input_tensor)[0]
            out_main = get_acdc_2d_cct(out_main) if cct else out_main
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0).cpu().detach().numpy()
        pred = zoom(out, (x / 256, y / 256), order=0)
        volume_pred[ind] = pred
    return volume_pred


def test_all_case(net, val_set, name="evaluation", use_cct_=False):
    first_total = 0.0
    second_total = 0.0
    third_total = 0.0

    dataloader = iter(val_set)
    tbar = tqdm(range(len(val_set)), ncols=135)

    for _, _ in enumerate(tbar):
        sampled_batch = next(dataloader)
        image = sampled_batch["image"]
        label = sampled_batch["label"]
        y_tilde = test_single_case(net, image, cct=use_cct_)

        if np.sum(y_tilde == 1) == 0:
            first_metric = 0, 0, 0, 0
        else:
            first_metric = calculate_metric_percase(y_tilde == 1, label == 1)
        if np.sum(y_tilde == 2) == 0:
            second_metric = 0, 0, 0, 0
        else:
            second_metric = calculate_metric_percase(y_tilde == 2, label == 2)
        if np.sum(y_tilde == 3) == 0:
            third_metric = 0, 0, 0, 0
        else:
            third_metric = calculate_metric_percase(y_tilde == 3, label == 3)

        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
        tbar.set_description(f"measuring {name} dataset ...")

    metric_record = first_total + second_total + third_total
    avg_metric = metric_record / (3 * len(val_set))
    print(
        "{} dataset |dice={:.4f}|mIoU={:.4f}|ASD={:.4f}|95HD={:.4f}|".format(
            name, avg_metric[0], avg_metric[1], avg_metric[3], avg_metric[2]
        )
    )
    return avg_metric


def build_model():
    if args.stage_name == "pre_train":
        return net_factory(net_type="CVBM2d_Argument", in_chns=1, class_num=args.num_classes)
    return CVBMArgumentWithCrossSKC2D(
        n_channels=1,
        n_classes=args.num_classes,
        has_dropout=True,
        attn_shape=(8, 8),
        local_window=(2, 2),
        num_heads=4,
        dropout=0.0,
    ).cuda()


def evaluate_checkpoint(path, name):
    model = build_model()
    state = torch.load(path)
    state_dict = state['net'] if isinstance(state, dict) and 'net' in state else state
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"validate {path} for ACDC dataset ...")
    test_dataset = BaseDataSets(base_dir=args.root_path, split="test")
    return test_all_case(model, test_dataset, name=name, use_cct_=args.post)


if __name__ == '__main__':
    save_model_path = os.path.join(snapshot_path, f'{args.model}_best_model.pth')
    save_ema_model_path = os.path.join(snapshot_path, f'{args.model}_ema_best_model.pth')

    metric = evaluate_checkpoint(save_model_path, name='test')
    print(metric)

    if os.path.exists(save_ema_model_path):
        ema_metric = evaluate_checkpoint(save_ema_model_path, name='test_ema')
        print(ema_metric)
