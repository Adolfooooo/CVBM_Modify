import torch
import torch.nn.functional as F
from typing import Dict, List

class DynamicThresholdUpdater:
    """
    动态阈值更新器 - 支持多分类
    输入：单个特征图 + 对应伪标签
    """
    
    def __init__(self, 
                 class_num: int,
                 dynamic_thresholds: List[float],
                 alpha: float,
                 plt_thresholds: Dict):
        """
        Args:
            class_num: 分类类别数
            dynamic_thresholds: 各类别动态阈值列表 [threshold_class0, threshold_class1, ...]
            alpha: 指数移动平均权重系数
            plt_thresholds: 阈值历史记录字典
        """
        self.class_num = class_num
        self.dynamic_thresholds = dynamic_thresholds
        self.alpha = alpha
        self.plt_thresholds = plt_thresholds
        
        # 验证阈值列表长度
        assert len(dynamic_thresholds) == class_num, f"阈值列表长度({len(dynamic_thresholds)})必须等于类别数({class_num})"
    
    def update_threshold(self, 
                        output_unlabeled: torch.Tensor, 
                        pseudo_label: torch.Tensor,
                        iter_num: int):
        """
        更新动态阈值 - 处理单个特征图，支持多分类
        
        Args:
            output_unlabeled: 单个解码器输出特征图 [B, C, H, W, D] 或 [B, C, H, W]
            pseudo_label: 对应的伪标签 [B, H, W, D] 或 [B, H, W]
            iter_num: 当前迭代次数
        """
        mean_probs = [0] * self.class_num
        
        # 转换为softmax概率
        current_unlabeled_soft = F.softmax(output_unlabeled, dim=1).cpu()
        
        # 计算每个类别的概率均值
        for class_idx in range(self.class_num):
            # 获取当前类别的像素索引
            index_gt_class = (pseudo_label == class_idx).cpu()
            
            # 计算当前类别的概率均值
            if index_gt_class.sum() > 0:
                mean_probs[class_idx] = current_unlabeled_soft[:, class_idx][index_gt_class].mean().item()
            else:
                mean_probs[class_idx] = 0
        
        # 更新每个类别的阈值
        for class_idx in range(self.class_num):
            if mean_probs[class_idx] != 0:
                self.dynamic_thresholds[class_idx] = (self.dynamic_thresholds[class_idx] * self.alpha + 
                                                    (1 - self.alpha) * mean_probs[class_idx])
                self.dynamic_thresholds[class_idx] = max(0.5, self.dynamic_thresholds[class_idx])
                self.dynamic_thresholds[class_idx] = min(0.95, self.dynamic_thresholds[class_idx])
        
        # 记录历史
        self.plt_thresholds[iter_num] = self.dynamic_thresholds.copy()