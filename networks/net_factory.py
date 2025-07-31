from networks.CVBM import CVBM, CVBM_Argument
from networks.unet import CVBM2d

def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train"):
    if net_type == "CVBM2d":
        net = CVBM2d(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "CVBM" and mode == "train":
        net = CVBM(n_channels=in_chns, n_classes=class_num, normalization='instancenorm', has_dropout=True).cuda()
    elif net_type == "CVBM" and mode == "train_4_1":
        net = CVBM_Argument(n_channels=in_chns, n_classes=class_num, normalization='instancenorm', has_dropout=True).cuda()
    elif net_type == "CVBM" and mode == "test":
        net = CVBM(n_channels=in_chns, n_classes=class_num, normalization='instancenorm', has_dropout=False).cuda()
    return net
