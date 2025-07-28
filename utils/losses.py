import torch
from torch.nn import functional as F
import torch.nn as nn
import contextlib
import pdb
import numpy as np


class mask_DiceLoss(nn.Module):
    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(mask_DiceLoss, self).__init__()
        self.smooth = smooth
        if class_weights is None:
            # default weight is all 1
            self.class_weights = nn.Parameter(torch.ones((1, nclass)).type(torch.float32), requires_grad=False)
        else:
            class_weights = np.array(class_weights)
            assert nclass == class_weights.shape[0]
            self.class_weights = nn.Parameter(torch.tensor(class_weights, dtype=torch.float32), requires_grad=False)

    def prob_forward(self, pred, target, mask=None):
        size = pred.size()
        N, nclass = size[0], size[1]
        # N x C x H x W
        pred_one_hot = pred.view(N, nclass, -1)
        target = target.view(N, 1, -1)
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, logits, target, mask=None):
        size = logits.size()
        N, nclass = size[0], size[1]

        logits = logits.view(N, nclass, -1)
        target = target.view(N, 1, -1)

        pred, nclass = get_probability(logits)

        # N x C x H x W
        pred_one_hot = pred
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def mse_loss(input1, input2):
    return torch.mean((input1 - input2) ** 2)

class mask_DiceLoss_2d(nn.Module):
    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(mask_DiceLoss_2d, self).__init__()
        self.smooth = smooth
        if class_weights is None:
            # default weight is all 1
            self.class_weights = nn.Parameter(torch.ones((1, nclass)).type(torch.float32), requires_grad=False)
        else:
            class_weights = np.array(class_weights)
            assert nclass == class_weights.shape[0]
            self.class_weights = nn.Parameter(torch.tensor(class_weights, dtype=torch.float32), requires_grad=False)

    def prob_forward(self, pred, target, mask=None):
        size = pred.size()
        N, nclass = size[0], size[1]
        # N x C x H x W
        pred_one_hot = pred.view(N, nclass, -1)
        target = target.view(N, 1, -1)
        target_one_hot = target

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, logits, target, mask=None):
        size = logits.size()
        N, nclass = size[0], size[1]

        logits = logits.view(N, nclass, -1)
        target = target.view(N, 1, -1)

        pred, nclass = get_probability(logits)

        # N x C x H x W
        pred_one_hot = pred
        target_one_hot = target

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _one_hot_mask_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor * i == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, mask=None, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        # print(inputs.size(),target.size())
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        if mask is not None:
            mask = self._one_hot_mask_encoder(mask)
            for i in range(0, self.n_classes):
                dice = self._dice_mask_loss(inputs[:, i], target[:, i], mask[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        else:
            for i in range(0, self.n_classes):
                dice = self._dice_loss(inputs[:, i], target[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        return loss / self.n_classes
class DiceLoss3(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss3, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _one_hot_mask_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor * i == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, mask=None, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        if mask is not None:
            mask = self._one_hot_mask_encoder(mask)
            for i in range(0, self.n_classes):
                dice = self._dice_mask_loss(inputs[:, i], target[:, i], mask[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        else:
            for i in range(0, self.n_classes):
                dice = self._dice_loss(inputs[:, i], target[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        return loss / self.n_classes
class DiceLoss2d(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss2d, self).__init__()
        self.n_classes = n_classes

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def _dice_mask_loss(self, score, target, mask):
        # print(score.shape, target.shape, mask.shape)
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, mask=None, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = target
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        if mask is not None:
            mask = mask
            for i in range(0, self.n_classes):
                dice = self._dice_mask_loss(inputs[:, i], target[:, i], mask[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        else:
            for i in range(0, self.n_classes):
                dice = self._dice_loss(inputs[:, i], target[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        return loss / self.n_classes

class CrossEntropyLoss(nn.Module):
    def __init__(self, n_classes):
        super(CrossEntropyLoss, self).__init__()
        self.class_num = n_classes

    def _ce_loss(self, score, target, mask):
        target = target.float()
        # print(torch.max(score),torch.min(score))
        loss = (-target * torch.log(score) * mask.float()).sum() / (mask.sum() + 1e-16)
        return loss

    def forward(self, inputs, target, mask):
        inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target)
        # mask = self._one_hot_mask_encoder(mask)
        loss = 0.0
        for i in range(0, self.class_num):
            loss += self._ce_loss(inputs[:, i], target[:, i], mask[:, i])
        return loss / self.class_num
class CrossEntropyLossonehot(nn.Module):
    def __init__(self, n_classes):
        super(CrossEntropyLossonehot, self).__init__()
        self.class_num = n_classes

    def _ce_loss(self, score, target, mask):
        target = target.float()
        # print(torch.max(score),torch.min(score))
        loss = (-target * torch.log(score) * mask.float()).sum() / (mask.sum() + 1e-16)
        return loss

    def forward(self, inputs, target, mask):
        inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target)
        # mask = self._one_hot_mask_encoder(mask)
        loss = 0.0
        for i in range(0, self.class_num):
            loss += self._ce_loss(inputs[:, i], target[:, i], mask[:, i])
        return loss / self.class_num

def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot


def get_probability(logits):
    """ Get probability from logits, if the channel of logits is 1 then use sigmoid else use softmax.
    :param logits: [N, C, H, W] or [N, C, D, H, W]
    :return: prediction and class num
    """
    size = logits.size()
    # N x 1 x H x W
    if size[1] > 1:
        pred = F.softmax(logits, dim=1)
        nclass = size[1]
    else:
        pred = F.sigmoid(logits)
        pred = torch.cat([1 - pred, pred], 1)
        nclass = 2
    return pred, nclass


class Dice_Loss(nn.Module):
    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(Dice_Loss, self).__init__()
        self.smooth = smooth
        if class_weights is None:
            # default weight is all 1
            self.class_weights = nn.Parameter(torch.ones((1, nclass)).type(torch.float32), requires_grad=False)
        else:
            class_weights = np.array(class_weights)
            assert nclass == class_weights.shape[0]
            self.class_weights = nn.Parameter(torch.tensor(class_weights, dtype=torch.float32), requires_grad=False)

    def prob_forward(self, pred, target, mask=None):
        size = pred.size()
        N, nclass = size[0], size[1]
        # N x C x H x W
        pred_one_hot = pred.view(N, nclass, -1)
        target = target.view(N, 1, -1)
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, logits, target, mask=None):
        size = logits.size()
        N, nclass = size[0], size[1]

        logits = logits.view(N, nclass, -1)
        target = target.view(N, 1, -1)

        pred, nclass = get_probability(logits)

        # N x C x H x W
        pred_one_hot = pred
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss


class softDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(softDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target):
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice
        return loss / self.n_classes


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    # pdb.set_trace()
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8  ###2-p length of vector
    return d


class VAT2d(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT2d, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = softDiceLoss(4)

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x)[0], dim=1)

        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)[0]
                logp_hat = F.softmax(pred_hat, dim=1)
                adv_distance = self.loss(logp_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            r_adv = d * self.epi
            pred_hat = model(x + r_adv)[0]
            logp_hat = F.softmax(pred_hat, dim=1)
            lds = self.loss(logp_hat, pred)
        return lds


class VAT3d(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT3d, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = Binary_dice_loss

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x)[0], dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)  ### initialize a random tensor between [-0.5, 0.5]
        d = _l2_normalize(d)  ### an unit vector
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)[0]
                p_hat = F.softmax(pred_hat, dim=1)
                adv_distance = self.loss(p_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
            pred_hat = model(x + self.epi * d)[0]
            p_hat = F.softmax(pred_hat, dim=1)
            lds = self.loss(p_hat, pred)
        return lds


@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)


class FocalLoss(nn.Module):
    """
    Focal Loss实现
    
    Args:
        alpha (float or tensor): 类别权重，用于平衡正负样本
        gamma (float): 调制因子，控制难易样本的权重差异
        reduction (str): 损失聚合方式 ('none', 'mean', 'sum')
        ignore_index (int): 忽略的类别索引
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        """
        前向传播
        
        Args:
            inputs: 模型预测logits, shape: (N, C) 或 (N, C, H, W)
            targets: 真实标签, shape: (N,) 或 (N, H, W)
        
        Returns:
            focal loss值
        """
        # 计算交叉熵损失，不进行reduction
        ce_loss = F.cross_entropy(inputs, targets, 
                                reduction='none', 
                                ignore_index=self.ignore_index)
        
        # 计算预测概率
        pt = torch.exp(-ce_loss)  # pt = p_t (预测正确类别的概率)
        
        # 计算alpha权重
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        else:
            # 如果alpha是tensor，需要根据targets选择对应的alpha值
            alpha_t = self.alpha.gather(0, targets)
        
        # 计算focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        # 根据reduction参数聚合损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 多类别版本的Focal Loss
class MultiClassFocalLoss(nn.Module):
    """
    多类别Focal Loss，支持为每个类别设置不同的alpha权重
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        # 设置类别权重
        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        # 计算softmax概率
        log_pt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt)
        
        # 选择目标类别的概率
        log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # 应用alpha权重
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha.gather(0, targets)
            log_pt = alpha_t * log_pt
        
        # 计算focal loss
        focal_loss = -((1 - pt) ** self.gamma) * log_pt
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss实现，主要用于分割任务
    
    Args:
        alpha (float): 控制假正例(FP)的权重
        beta (float): 控制假负例(FN)的权重
        smooth (float): 平滑项，防止除零
        reduction (str): 损失聚合方式
    
    Note:
        - alpha=beta=0.5时退化为Dice Loss
        - alpha>beta时更关注召回率(减少假负例)
        - alpha<beta时更关注精确率(减少假正例)
    """
    
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-7, reduction='mean'):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        前向传播
        
        Args:
            inputs: 预测概率, shape: (N, C, H, W) 或 (N, C)
            targets: 真实标签, shape: (N, H, W) 或 (N,)
        """
        # 如果inputs是logits，转换为概率
        if inputs.dim() > 2:
            # 分割任务：(N, C, H, W)
            inputs = torch.softmax(inputs, dim=1)
            
            # 将targets转换为one-hot编码
            num_classes = inputs.shape[1]
            targets_one_hot = F.one_hot(targets, num_classes=num_classes)
            targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
            
            # 计算每个类别的Tversky系数
            tversky_scores = []
            for class_idx in range(num_classes):
                pred_class = inputs[:, class_idx]
                target_class = targets_one_hot[:, class_idx]
                
                # 计算TP, FP, FN
                tp = (pred_class * target_class).sum(dim=(1, 2))
                fp = (pred_class * (1 - target_class)).sum(dim=(1, 2))
                fn = ((1 - pred_class) * target_class).sum(dim=(1, 2))
                
                # 计算Tversky系数
                tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
                tversky_scores.append(tversky)
            
            # 平均所有类别的Tversky系数
            tversky_score = torch.stack(tversky_scores, dim=1).mean(dim=1)
            
        else:
            # 分类任务：(N, C)
            inputs = torch.softmax(inputs, dim=1)
            targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()
            
            # 计算TP, FP, FN
            tp = (inputs * targets_one_hot).sum(dim=1)
            fp = (inputs * (1 - targets_one_hot)).sum(dim=1)
            fn = ((1 - inputs) * targets_one_hot).sum(dim=1)
            
            # 计算Tversky系数
            tversky_score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Tversky Loss = 1 - Tversky系数
        tversky_loss = 1 - tversky_score
        
        if self.reduction == 'mean':
            return tversky_loss.mean()
        elif self.reduction == 'sum':
            return tversky_loss.sum()
        else:
            return tversky_loss

# 二分类专用的Tversky Loss
class BinaryTverskyLoss(nn.Module):
    """
    二分类Tversky Loss，计算更简单高效
    """
    
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-7):
        super(BinaryTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # 确保inputs是概率值
        if inputs.dim() == 1 or (inputs.dim() == 2 and inputs.shape[1] == 1):
            # 二分类sigmoid输出
            inputs = torch.sigmoid(inputs).flatten()
        else:
            # 多类别输出的正类概率
            inputs = torch.softmax(inputs, dim=1)[:, 1]
        
        targets = targets.float().flatten()
        
        # 计算TP, FP, FN
        tp = (inputs * targets).sum()
        fp = (inputs * (1 - targets)).sum()
        fn = ((1 - inputs) * targets).sum()
        
        # 计算Tversky系数
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1 - tversky

class BinaryTverskyLoss3D(nn.Module):
    """
    专门用于3D二分类分割的Tversky Loss
    支持输入格式: [N, C, D, H, W] 和 [N, D, H, W]
    """
    
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-7):
        super(BinaryTverskyLoss3D, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C, D, H, W] 模型输出logits
            targets: [N, D, H, W] 真实标签 (0或1)
        """
        # 确保inputs是概率值
        if inputs.shape[1] == 1:
            # 单类别输出，使用sigmoid
            inputs = torch.sigmoid(inputs).squeeze(1)  # [N, D, H, W]
        elif inputs.shape[1] == 2:
            # 二类别输出，使用softmax取正类概率
            inputs = torch.softmax(inputs, dim=1)[:, 1]  # [N, D, H, W]
        else:
            raise ValueError(f"Expected 1 or 2 classes, got {inputs.shape[1]}")
        
        # 确保targets是float类型
        targets = targets.float()
        
        # 现在inputs和targets都是 [N, D, H, W]
        # 将空间维度展平进行计算，保持batch维度
        inputs_flat = inputs.view(inputs.shape[0], -1)    # [N, D*H*W]
        targets_flat = targets.view(targets.shape[0], -1)  # [N, D*H*W]
        
        # 计算每个样本的TP, FP, FN
        tp = (inputs_flat * targets_flat).sum(dim=1)                    # [N]
        fp = (inputs_flat * (1 - targets_flat)).sum(dim=1)             # [N]
        fn = ((1 - inputs_flat) * targets_flat).sum(dim=1)             # [N]
        
        # 计算Tversky系数
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # 返回平均loss
        return (1 - tversky).mean()