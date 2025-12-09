class VNet_Decouple_Attention_ABC(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, num_heads=2):
        super(VNet_Decouple_Attention_ABC, self).__init__()

        self.encoder = Encoder(n_channels=n_channels, n_filters=n_filters, normalization=normalization,
                               has_dropout=has_dropout)


        self.decoder_l = Decoder(n_classes=n_classes, n_filters=n_filters, normalization=normalization,
                               has_dropout=has_dropout)

        self.decoder_u = Decoder(n_classes=n_classes, n_filters=n_filters, normalization=normalization,
                               has_dropout=has_dropout)

        self.cross_attention_local = Cross_Attention_Local_Block(num_heads=num_heads,
                                                                 embedding_channels=n_filters * 16,
                                                                 attention_dropout_rate=0.1,
                                                                 )

        self.cross_attention_global = Cross_Attention_Global_Block(num_heads=num_heads,
                                                                   embedding_channels=n_filters * 16,
                                                                   attention_dropout_rate=0.1,
                                                                   )




    def forward(self, image, pred_type = None):



        if pred_type == "labeled":
            features = self.encoder(image)
            x1, x2, x3, x4, x5 = features
            x5 = self.cross_attention_local(x5, pred_type)
            x5 = self.cross_attention_global(x5, pred_type)
            features = [x1, x2, x3, x4, x5]
            out, out_abc = self.decoder_l(features)
            return out ,out_abc

        if pred_type == "unlabeled":
            features = self.encoder(image)
            x1, x2, x3, x4, x5 = features
            x5 = self.cross_attention_local(x5, pred_type)
            x5 = self.cross_attention_global(x5, pred_type)
            features = [x1, x2, x3, x4, x5]
            out,out_abc =self.decoder_u(features)
            return out, out_abc


class Decoder(nn.Module):
    def __init__(self, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)

        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)

        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv9 = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.out_conv9_abc = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)



    def decoder(self, features):
        x1, x2, x3, x4, x5 = features
        # print(x1.size(), x2.size(), x3.size())
        x5_up = self.block_five_up(x5)
        # print("1",x5.size(), x5_up.size(), x4.size())
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
        if self.has_dropout:
            x9 = self.dropout(x9)

        out = self.out_conv9(x9)
        out_abc = self.out_conv9_abc(x9)
        return out,out_abc

    def forward(self, features, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False

        out= self.decoder(features)  # logits [4,32,64,128,128]

        if turnoff_drop:
            self.has_dropout = has_dropout

        return out


import os
import sys
import logging
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--exp', type=str, default='/synapse_20p/skcdf/fold1')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_20p')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_20p')
parser.add_argument('-se', '--split_eval', type=str, default='eval')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True)  # <--
parser.add_argument('-ep', '--max_epoch', type=int, default=1500)
parser.add_argument('--cps_loss', type=str, default='w_ce+dice')
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=2)

parser.add_argument('--base_lr', type=float, default=0.3)

parser.add_argument('-s', '--ema_w', type=float, default=0.99)

parser.add_argument('-g', '--gpu', type=str, default='1')
parser.add_argument('-w', '--cps_w', type=float, default=10)

parser.add_argument('-r', '--cps_rampup', action='store_true', default=True)  # <--
parser.add_argument('-cr', '--consistency_rampup', type=float, default=None)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.SKCDF import VNet_Decouple_Attention_ABC
from utils import maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr
from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss
from data.transforms import RandomCrop, CenterCrop, ToTensor, RandomFlip_X, RandomFlip_Y
from data.data_loaders import Synapse_AMOS
from utils.config import Config

config = Config(args.task)

def sigmoid_rampup(current, rampup_length):
    '''Exponential rampup from https://arxiv.org/abs/1610.02242'''
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch):
    if args.cps_rampup:
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        if args.consistency_rampup is None:
            args.consistency_rampup = args.max_epoch
        return args.cps_w * sigmoid_rampup(epoch, args.consistency_rampup)
    else:
        return args.cps_w

def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def make_loss_function(name, weight=None):
    if name == 'ce':
        return RobustCrossEntropyLoss()
    elif name == 'wce':
        return RobustCrossEntropyLoss(weight=weight)
    elif name == 'ce+dice':
        return DC_and_CE_loss()
    elif name == 'wce+dice':
        return DC_and_CE_loss(w_ce=weight)
    elif name == 'w_ce+dice':
        return DC_and_CE_loss(w_dc=weight, w_ce=weight)
    elif name == 'dice':
        return SoftDiceLoss()
    else:
        raise ValueError(name)

def make_loader(split, dst_cls=Synapse_AMOS, repeat=None, is_training=True, unlabeled=False):
    if is_training:
        dst = dst_cls(
            task=args.task,
            split=split,
            repeat=repeat,
            unlabeled=unlabeled,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                RandomCrop(config.patch_size),
                RandomFlip_X(),
                RandomFlip_Y(),
                ToTensor()
            ])
        )
        return DataLoader(
            dst,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            drop_last=True
        )
    else:
        dst = dst_cls(
            task=args.task,
            split=split,
            is_val=True,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                CenterCrop(config.patch_size),
                ToTensor()
            ])
        )
        return DataLoader(dst, pin_memory=True)

def make_vnet_decouple_attention_abc():
    model = VNet_Decouple_Attention_ABC(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=True
    ).cuda()


    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True
    )
    return model, optimizer

if __name__ == '__main__':
    import random

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # make logger file
    snapshot_path = f'/data/hdd1/ygc/logs/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))

    # make logger
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # make data loader
    unlabeled_loader = make_loader(args.split_unlabeled, unlabeled=True)  # 16
    labeled_loader = make_loader(args.split_labeled, repeat=len(unlabeled_loader.dataset))  # 4
    eval_loader = make_loader(args.split_eval, is_training=False)  # 4

    logging.info(f'{len(labeled_loader)} itertations per epoch (labeled)')
    logging.info(f'{len(unlabeled_loader)} itertations per epoch (unlabeled)')

    model, optimizer = make_vnet_decouple_attention_abc()
    model = xavier_normal_init_weight(model)

    logging.info(optimizer)

    sup_loss_func = make_loss_function(args.sup_loss)
    unsup_loss_func = make_loss_function(args.cps_loss)

    num_cls = config.num_cls
    num_sample = 4
    training_set = Synapse_AMOS(split=args.split_labeled, num_cls=num_cls,task=args.task)
    loop = 0
    class_size = [0 for c in range(num_cls)]  # {0: 0, 1: 0, 2: 0, 3: 0,...}
    for sample in training_set:
        image = sample["image"]
        label = sample["label"]
        for i in range(num_cls):
            num = np.sum(label == i)
            class_size[i] = class_size[i] + num
        loop = loop + 1
        if loop == num_sample:
            break
    # print(class_size)
    ir2 = min(class_size) / np.array(class_size)
    ir2 = torch.tensor(ir2).cuda()

    cps_w = get_current_consistency_weight(0)

    best_eval = 0.0
    best_epoch = 0

    for epoch_num in range(args.max_epoch + 1):
        loss_list = []
        loss_cps_list = []
        loss_sup_list = []

        model.train()

        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader)):
            for out_conv9_name, out_conv9_params in model.decoder_l.out_conv9.named_parameters():
                if out_conv9_name in model.decoder_l.out_conv9_abc.state_dict().keys():
                    out_conv9_abc_params = model.decoder_l.out_conv9_abc.state_dict()[out_conv9_name]
                    if out_conv9_params.shape == out_conv9_abc_params.shape:
                        out_conv9_params.data = args.ema_w * out_conv9_params.data + (1 - args.ema_w) * out_conv9_abc_params.data

            for decoder_u_name, decoder_u_params in model.decoder_u.named_parameters():
                if decoder_u_name in model.decoder_l.state_dict().keys():
                    decoder_l_params = model.decoder_l.state_dict()[decoder_u_name]
                    if decoder_u_params.shape == decoder_l_params.shape:
                        decoder_u_params.data = args.ema_w * decoder_u_params.data + (1 - args.ema_w) * decoder_l_params.data

            for out_conv9_name, out_conv9_params in model.decoder_u.out_conv9.named_parameters():
                if out_conv9_name in model.decoder_u.out_conv9_abc.state_dict().keys():
                    out_conv9_abc_params = model.decoder_u.out_conv9_abc.state_dict()[out_conv9_name]
                    if out_conv9_params.shape == out_conv9_abc_params.shape:
                        out_conv9_params.data = args.ema_w * out_conv9_params.data + (
                                    1 - args.ema_w) * out_conv9_abc_params.data

            optimizer.zero_grad()
            image_l, label_l = fetch_data(batch_l)
            image_u = fetch_data(batch_u, labeled=False)
            image = torch.cat([image_l, image_u], dim=0)
            tmp_bs = image.shape[0] // 2

            if True:
                if True:

                    output_A,output_A_abc = model(image, pred_type = "labeled" )

                    output_l = output_A[:tmp_bs, ...]
                    output_l_abc = output_A_abc[:tmp_bs, ...]

                    label_l_prob = label_l.to(dtype=torch.float64).detach()
                    for i in range(num_cls):
                        label_l_prob = torch.where(label_l_prob == i, ir2[i], label_l_prob)
                    maskforbalance = torch.bernoulli(label_l_prob.detach())

                    L_sup = sup_loss_func(output_l,label_l) + sup_loss_func(output_l_abc * maskforbalance,label_l * maskforbalance)

                    pseudo_label_prob = output_A[tmp_bs:, ...]
                    pseudo_label_prob_abc = output_A_abc[tmp_bs:, ...]

                    pseudo_label = torch.argmax(pseudo_label_prob, dim=1, keepdim=True).long()
                    pseudo_label_abc = torch.argmax(pseudo_label_prob_abc, dim=1, keepdim=True).long()

                    ir22 = 1 - (epoch_num / args.max_epoch) * (1 - ir2)

                    pseudo_label_abc_prob = pseudo_label_abc.to(dtype=torch.float64).detach()
                    for i in range(num_cls):
                        pseudo_label_abc_prob = torch.where(pseudo_label_abc_prob == i, ir22[i], pseudo_label_abc_prob)
                    maskforbalance_u = torch.bernoulli(pseudo_label_abc_prob.detach())

                    output_u, output_u_abc = model(image_u, pred_type = "unlabeled")

                    L_u = unsup_loss_func(output_u, pseudo_label.detach()) + unsup_loss_func(output_u_abc * maskforbalance_u, pseudo_label_abc.detach() * maskforbalance_u)

                    loss = L_sup + cps_w * L_u

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            else:
                raise NotImplementedError

            loss_list.append(loss.item())
            loss_sup_list.append(L_sup.item())
            loss_cps_list.append(L_u.item())

        lr = get_lr(optimizer)

        writer.add_scalar('lr', lr, epoch_num)
        writer.add_scalar('cps_w', cps_w, epoch_num)
        writer.add_scalar('loss/loss', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/sup', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/cps', np.mean(loss_cps_list), epoch_num)

        logging.info(
            f'epoch {epoch_num} : loss : {np.mean(loss_list)}, cpsw:{cps_w} lr: {lr} ')

        optimizer.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        cps_w = get_current_consistency_weight(epoch_num)


        if epoch_num % 1 == 0:
            # ''' ===== evaluation
            dice_list = [[] for _ in range(config.num_cls - 1)]
            model.eval()

            dice_func = SoftDiceLoss(smooth=1e-8)
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, gt = fetch_data(batch)
                    output,_ = model(image, pred_type = "unlabeled")
                    del image

                    shp = output.shape
                    gt = gt.long()
                    y_onehot = torch.zeros(shp).cuda()
                    y_onehot.scatter_(1, gt, 1)

                    x_onehot = torch.zeros(shp).cuda()
                    output = torch.argmax(output, dim=1, keepdim=True).long()
                    x_onehot.scatter_(1, output, 1)

                    dice = dice_func(x_onehot, y_onehot, is_training=False)
                    dice = dice.data.cpu().numpy()
                    for i, d in enumerate(dice):
                        dice_list[i].append(d)

            dice_mean = []
            for dice in dice_list:
                dice_mean.append(np.mean(dice))
            logging.info(f'evaluation epoch {epoch_num}, dice: {np.mean(dice_mean)}, {dice_mean}')
            # '''
            if np.mean(dice_mean) > best_eval:
                best_eval = np.mean(dice_mean)
                best_epoch = epoch_num
                save_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
                torch.save(model.state_dict(),save_path)
                logging.info(f'saving best model to {save_path}')
            logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')

            if epoch_num - best_epoch == config.early_stop_patience:
                logging.info(f'Early stop.')
                break

    writer.close()
