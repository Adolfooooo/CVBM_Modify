import argparse
import os
from skimage.measure import label
import numpy as np
import torch
from medpy import metric
from tqdm import tqdm

# from Configs.config import config
# from Dataloader.dataset import ACDCDataset
# from Model.Unet2D import UNet
from .modules import CVBMArgumentWithSKC2D
from dataloaders.dataset import BaseDataSets


def get_acdc_2d_cct(segmentation):
    batch_list = []
    mask = torch.argmax(segmentation, dim=1).detach().cpu().numpy()
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = mask[i]
            labels = label(temp_seg == c)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                if largestCC.sum() < 10:
                    class_list.append(largestCC * 0)
                else:
                    class_list.append(largestCC * c)
            else:
                class_list.append(labels)
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)
    batch_list = torch.tensor(batch_list).cuda()
    if len(torch.unique(batch_list)) == 1: return segmentation
    return torch.nn.functional.one_hot(batch_list, num_classes=4).permute(0, 3, 1, 2) * segmentation


def test_all_case(net, val_set, name="evaluation", use_cct_=False):
    first_total = 0.0
    second_total = 0.0
    third_total = 0.0

    assert val_set.aug is False, ">> no augmentation for test set"
    dataloader = iter(val_set)
    tbar = range(len(val_set))
    tbar = tqdm(tbar, ncols=135)

    for (idx, _) in enumerate(tbar):
        image, label = next(dataloader)
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

        tbar.set_description("measuring {} dataset ...".format(name))

    metric_record = (first_total + second_total + third_total)
    avg_metric = metric_record / (3 * len(val_set))
    print("{} dataset "
          "|dice={:.4f}|mIoU={:.4f}|ASD={:.4f}|95HD={:.4f}|".format(name, avg_metric[0], avg_metric[1],
                                                                    avg_metric[3], avg_metric[2]))

    return avg_metric


def test_single_case(net, image, cct=False):
    from scipy.ndimage.interpolation import zoom
    volume_pred = np.zeros_like(image)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        out_main = net(torch.tensor(slice, dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(1))
        out_main = get_acdc_2d_cct(torch.tensor(out_main)) if cct else out_main
        out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0).cpu().detach().numpy()
        pred = zoom(out, (x / 256, y / 256), order=0)
        volume_pred[ind] = pred
    return volume_pred


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd


def test_calculate_metric(snapshot_path, use_cct=False):
    save_model_path = os.path.join(snapshot_path, f'{args.model}_best_model.pth')
    save_ema_model_path = os.path.join(snapshot_path, f'{args.model}_ema_best_model.pth')

    net = CVBMArgumentWithSKC2D(
        n_channels=1,
        n_classes=args.num_classes,
        has_dropout=True,
    ).cuda()
    net.load_state_dict(torch.load(save_model_path), strict=True)
    net.eval()
    print("validate {} for ACDC dataset ...".format(str(save_model_path)))
    # val_dataset = ACDCDataset(os.path.join(config.code_path, "Dataloader"), config.data_path,
    #                           split="eval", config=config)

    # follows the previous works' setting
    # _ = test_all_case(net, val_dataset, name='val', use_cct_=use_cct)
    # test_dataset = ACDCDataset(os.path.join(config.code_path, "Dataloader"), config.data_path,
    #                            split="test", config=config)
    test_dataset = BaseDataSets(base_dir=args.root_path, split="test")
    avg_metric = test_all_case(net, test_dataset, name='test', use_cct_=use_cct)

    return avg_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Medical Semi-supervised Semantic Segmentation (valid)')
    parser.add_argument('--root_path', type=str, default='/root/ACDC', help='Name of dataset root')
    parser.add_argument('--exp', type=str, default='CVBM2d_ACDC', help='exp_name')
    parser.add_argument('--model', type=str, default='CVBM2d_Argument', help='model_name')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--labelnum', type=int, default=3, help='number of labeled training samples')
    parser.add_argument('--stage_name', type=str, default='self_train', help='self_train or pre_train')
    parser.add_argument('--snapshot_path', type=str, default='./results/CVBM_11_1/1', help='snapshot path base')
    parser.add_argument('--num_classes', type=int, default=4, help='number of classes')
    parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size for sliding window')
    
    parser.add_argument("--post", action="store_true",
                        help="implement post process or not")

    args = parser.parse_args()

    # default_path = os.path.join(config.code_path, "saved", args.env_name)
    snapshot_path = f"{args.snapshot_path}/{args.exp}_{args.labelnum}_labeled/{args.stage_name}"
    test_save_path = f"{args.snapshot_path}/{args.exp}_{args.labelnum}_labeled/{args.model}_predictions/"

    # ckpt = os.listdir(default_path)
    # ckpt = [i for i in ckpt if ".pth" in str(i)][0]
    # print("validate {} for ACDC dataset ...".format(str(ckpt)))
    metric = test_calculate_metric(snapshot_path, use_cct=args.post)