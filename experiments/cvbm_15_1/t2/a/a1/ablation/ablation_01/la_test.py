import argparse
import ast
import os

import torch
import torch.nn as nn

from networks.CVBM import Decoder, Encoder
from utils.test_3d_patch import test_all_case_argument


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/LA', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='CVBM_LA_FgQBgKV_Ablation_Without_SKC', help='exp_name')
parser.add_argument('--model', type=str, default='CVBM_Argument', help='model_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--detail', type=int, default=1, help='print metrics for every sample')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-processing')
parser.add_argument('--labelnum', type=int, default=8, help='labeled data')
parser.add_argument('--patch_size', type=ast.literal_eval, default=(112, 112, 80), help='LA test patch size')
parser.add_argument('--stage_name', type=str, default='self_train', help='self_train or pre_train')
parser.add_argument(
    '--snapshot_path',
    type=str,
    default='./results/CVBM_15_1_t2_a/a1/ablation/without_skc_prototype_only/',
    help='snapshot path used by la_train.py',
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
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
num_classes = 2


class DecoderWithFeature(Decoder):
    def forward(self, features):
        x1, x2, x3, x4, x5 = features

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up)
        out_seg2 = self.out_conv2(x9)
        out_tanh = self.tanh(out_seg2)
        proto_feature = x9
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        return out_seg, out_tanh, proto_feature


# a1 ablation: remove SKC interaction while preserving the dual-branch prototype interface.
# a1 ablation: evaluate no-SKC dual-branch checkpoints.
class CVBMArgumentWithoutSKC3DProto(nn.Module):
    """Dual-branch CVBM prototype model without semantic interaction."""

    def __init__(
        self,
        n_channels=1,
        n_classes=2,
        n_filters=16,
        normalization="instancenorm",
        has_dropout=False,
        has_residual=False,
    ):
        super().__init__()
        self.encoder = Encoder(
            n_channels,
            n_classes,
            n_filters,
            normalization,
            has_dropout,
            has_residual,
        )
        self.decoder_fg = DecoderWithFeature(
            n_channels,
            n_classes,
            n_filters,
            normalization,
            has_dropout,
            has_residual,
            up_type=0,
        )
        self.decoder_bg = DecoderWithFeature(
            n_channels,
            n_classes,
            n_filters,
            normalization,
            has_dropout,
            has_residual,
            up_type=2,
        )
        self.final_seg = nn.Conv3d(n_classes * 2, n_classes, kernel_size=1)

    def forward(self, input_fg, input_bg):
        fg_feats = list(self.encoder(input_fg))
        bg_feats = list(self.encoder(input_bg))

        out_fg, attn_fg, feat_fg = self.decoder_fg(fg_feats)
        out_bg, attn_bg, feat_bg = self.decoder_bg(bg_feats)

        fused_logits = self.final_seg(torch.cat([out_fg, out_bg], dim=1))
        return out_fg, fused_logits, out_bg, attn_fg, attn_bg, feat_fg, feat_bg


snapshot_path = os.path.join(
    args.snapshot_path,
    '{}_{}_labeled'.format(args.exp, args.labelnum),
    args.stage_name,
)
test_save_path = os.path.join(
    args.snapshot_path,
    '{}_{}_labeled'.format(args.exp, args.labelnum),
    '{}_predictions'.format(args.model),
)

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)

with open(os.path.join(args.root_path, 'test.list'), 'r') as f:
    image_list = f.readlines()
image_list = [
    os.path.join(args.root_path, '2018LA_Seg_Training Set', item.strip(), 'mri_norm2.h5')
    for item in image_list
]


def build_model():
    return CVBMArgumentWithoutSKC3DProto(
        n_channels=1,
        n_classes=num_classes,
        normalization='instancenorm',
        has_dropout=True,
    ).cuda()


def load_checkpoint(model, path):
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'net' in state:
        state = state['net']
    model.load_state_dict(state)


def evaluate_checkpoint(path, tag):
    model = build_model()
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
        save_result=False,
        test_save_path=test_save_path,
        metric_detail=args.detail,
        nms=args.nms,
    )


def test_calculate_metric_argument():
    save_model_path = args.checkpoint
    if save_model_path is None:
        save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))

    avg_metric = evaluate_checkpoint(save_model_path, 'model')
    metrics = {'model': avg_metric}

    save_ema_model_path = os.path.join(snapshot_path, '{}_ema_best_model.pth'.format(args.model))
    if args.test_ema and args.stage_name == 'self_train' and os.path.exists(save_ema_model_path):
        metrics['ema_model'] = evaluate_checkpoint(save_ema_model_path, 'ema model')

    return metrics


if __name__ == '__main__':
    metrics = test_calculate_metric_argument()
    print(metrics)
