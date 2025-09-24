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
                #  plt_thresholds: Dict):
    ):
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
        # self.plt_thresholds = plt_thresholds
        
        # 验证阈值列表长度
        assert len(dynamic_thresholds) == class_num, f"阈值列表长度({len(dynamic_thresholds)})必须等于类别数({class_num})"
    
    
    def update_threshold(self, 
                        output_unlabeled: torch.Tensor, 
                        iter_num: int):
        """
        更新动态阈值 - 处理单个特征图，支持多分类
        
        Args:
            output_unlabeled: 单个解码器输出特征图 [B, C, H, W, D] 或 [B, C, H, W]
            iter_num: 当前迭代次数
        """
        mean_probs = [0] * self.class_num
        # get pseudo_label, shape:[B, H, W, D] or [B, H, W]
        pseudo_label = torch.argmax(output_unlabeled, dim=1)
        # 转换为softmax概率
        current_unlabeled_soft = F.softmax(output_unlabeled, dim=1)
        
        # 计算每个类别的概率均值
        for class_idx in range(self.class_num):
            # 获取当前类别的像素索引
            index_gt_class = (pseudo_label == class_idx)
            
            # 计算当前类别的概率均值
            if index_gt_class.sum() > 0:
                mean_probs[class_idx] = current_unlabeled_soft[:, class_idx][index_gt_class].mean().item()
            else:
                mean_probs[class_idx] = 0
        
        # 更新每个类别的阈值
        for class_idx in range(self.class_num):
            if mean_probs[class_idx] != 0:
                self.dynamic_thresholds[class_idx] = (
                    self.dynamic_thresholds[class_idx] * self.alpha + 
                                                    (1 - self.alpha) * mean_probs[class_idx])
                self.dynamic_thresholds[class_idx] = max(0.5, self.dynamic_thresholds[class_idx])
                self.dynamic_thresholds[class_idx] = min(0.95, self.dynamic_thresholds[class_idx])
        
        # 记录历史
        # self.plt_thresholds[iter_num] = self.dynamic_thresholds.copy()


class DynamicThresholdUpdater_Add_adaptive_alpha:
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
        
        # 新增：每类别的统计信息
        self.class_stats = {
            'mean_confidence': [0.0] * class_num,
            'confidence_std': [0.0] * class_num,
            'sample_count': [0] * class_num,
            'reliability_score': [1.0] * class_num  # 可靠性得分
        }
        # 验证阈值列表长度
        assert len(dynamic_thresholds) == class_num, f"阈值列表长度({len(dynamic_thresholds)})必须等于类别数({class_num})"
    
    def _update_class_stats(self, class_idx: int, probs: torch.Tensor, mask: torch.Tensor):
        """更新类别统计信息"""
        if mask.sum() > 0:
            valid_probs = probs[mask]
            
            # 更新均值和标准差
            self.class_stats['mean_confidence'][class_idx] = valid_probs.mean().item()
            self.class_stats['confidence_std'][class_idx] = valid_probs.std().item()
            self.class_stats['sample_count'][class_idx] = mask.sum().item()
            
            # 计算可靠性得分（基于样本数和标准差）
            sample_reliability = min(1.0, mask.sum().item() / 100.0)  # 样本数可靠性
            std_reliability = max(0.1, 1.0 - self.class_stats['confidence_std'][class_idx])  # 标准差可靠性
            self.class_stats['reliability_score'][class_idx] = (sample_reliability + std_reliability) / 2.0
    
    def update_threshold(self, 
                        output_unlabeled: torch.Tensor, 
                        iter_num: int):
        """
        更新动态阈值 - 处理单个特征图，支持多分类
        
        Args:
            output_unlabeled: 单个解码器输出特征图 [B, C, H, W, D] 或 [B, C, H, W]
            iter_num: 当前迭代次数
        """
        mean_probs = [0] * self.class_num
        # get pseudo_label, shape:[B, H, W, D] or [B, H, W]
        pseudo_label = torch.argmax(output_unlabeled, dim=1)
        # 转换为softmax概率
        current_unlabeled_soft = F.softmax(output_unlabeled, dim=1)
        
        
        # 计算每个类别的概率均值
        for class_idx in range(self.class_num):
            # 获取当前类别的像素索引
            index_gt_class = (pseudo_label == class_idx)
            
            # 计算当前类别的概率均值
            if index_gt_class.sum() > 0:
                mean_probs[class_idx] = current_unlabeled_soft[:, class_idx][index_gt_class].mean().item()
            else:
                mean_probs[class_idx] = 0
        
            # 更新类别统计信息
            self._update_class_stats(class_idx, current_unlabeled_soft[:, class_idx], index_gt_class)
            # 根据可靠性得分调整学习率
            reliability = self.class_stats['reliability_score'][class_idx]
            adaptive_alpha = self.alpha * reliability + (1 - reliability) * 0.95  # 不可靠时更保守
            # 预热期间使用更保守的更新
            if iter_num < 500:
                adaptive_alpha = min(adaptive_alpha, 0.95)

            if mean_probs[class_idx] != 0:
                # 指数移动平均更新
                old_threshold = self.dynamic_thresholds[class_idx]
                self.dynamic_thresholds[class_idx] = (old_threshold * adaptive_alpha + (1 - adaptive_alpha) * mean_probs[class_idx])
                self.dynamic_thresholds[class_idx] = max(0.5, self.dynamic_thresholds[class_idx])
                self.dynamic_thresholds[class_idx] = min(0.95, self.dynamic_thresholds[class_idx])
        
        # 记录历史
        # self.plt_thresholds[iter_num] = self.dynamic_thresholds.copy()




import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import deque

class ImprovedDynamicThresholdUpdater:
    """
    改进的动态阈值更新器 - 解决原算法的关键问题
    """
    
    def __init__(self, 
                 class_num: int,
                 initial_thresholds: List[float] = None,
                 alpha: float = 0.9,
                 plt_thresholds: Dict = None,
                 class_frequencies: List[float] = None,
                 uncertainty_weight: float = 0.3,
                 history_window: int = 10):
        """
        Args:
            class_num: 分类类别数
            initial_thresholds: 初始阈值，如果为None则自动初始化
            alpha: 指数移动平均权重系数
            plt_thresholds: 阈值历史记录字典
            class_frequencies: 各类别在训练集中的频率 [freq_class0, freq_class1, ...]
            uncertainty_weight: 不确定性权重系数
            history_window: 历史窗口大小，用于异常检测
        """
        self.class_num = class_num
        self.alpha = alpha
        self.uncertainty_weight = uncertainty_weight
        self.history_window = history_window
        
        # 初始化阈值
        if initial_thresholds is None:
            self.dynamic_thresholds = [0.7] * class_num  # 更保守的初始值
        else:
            assert len(initial_thresholds) == class_num
            self.dynamic_thresholds = initial_thresholds.copy()
        
        # 类别频率（用于自适应约束）
        if class_frequencies is None:
            self.class_frequencies = [1.0 / class_num] * class_num
        else:
            assert len(class_frequencies) == class_num
            self.class_frequencies = class_frequencies
        
        # 历史记录
        self.plt_thresholds = plt_thresholds if plt_thresholds is not None else {}
        
        # 新增：历史置信度队列，用于异常检测
        self.confidence_history = [deque(maxlen=history_window) for _ in range(class_num)]
        
        # 新增：每类别的统计信息
        self.class_stats = {
            'mean_confidence': [0.0] * class_num,
            'confidence_std': [0.0] * class_num,
            'sample_count': [0] * class_num,
            'reliability_score': [1.0] * class_num  # 可靠性得分
        }
        
        # 计算自适应约束范围
        self._compute_adaptive_bounds()
    
    def _compute_adaptive_bounds(self):
        """根据类别频率计算自适应阈值约束范围"""
        self.min_thresholds = []
        self.max_thresholds = []
        
        for freq in self.class_frequencies:
            # 低频类别允许更低的最小阈值，高频类别要求更高的最小阈值
            min_thresh = max(0.3, 0.5 - 0.2 * (1 - freq))
            max_thresh = min(0.95, 0.8 + 0.15 * freq)
            
            self.min_thresholds.append(min_thresh)
            self.max_thresholds.append(max_thresh)
    
    def _compute_uncertainty(self, probs: torch.Tensor) -> float:
        """计算预测的不确定性（熵）"""
        # 避免log(0)
        probs = torch.clamp(probs, min=1e-8)
        entropy = -torch.sum(probs * torch.log(probs), dim=0)
        return entropy.mean().item()
    
    def _detect_anomalies(self, class_idx: int, current_confidence: float) -> bool:
        """检测当前置信度是否异常"""
        if len(self.confidence_history[class_idx]) < 3:
            return False
        
        history = list(self.confidence_history[class_idx])
        hist_mean = np.mean(history)
        hist_std = np.std(history)
        
        # 使用3-sigma规则检测异常
        if hist_std > 0:
            z_score = abs(current_confidence - hist_mean) / hist_std
            return z_score > 3.0
        
        return False
    
    def _update_class_stats(self, class_idx: int, probs: torch.Tensor, mask: torch.Tensor):
        """更新类别统计信息"""
        if mask.sum() > 0:
            valid_probs = probs[mask]
            
            # 更新均值和标准差
            self.class_stats['mean_confidence'][class_idx] = valid_probs.mean().item()
            self.class_stats['confidence_std'][class_idx] = valid_probs.std().item()
            self.class_stats['sample_count'][class_idx] = mask.sum().item()
            
            # 计算可靠性得分（基于样本数和标准差）
            sample_reliability = min(1.0, mask.sum().item() / 100.0)  # 样本数可靠性
            std_reliability = max(0.1, 1.0 - self.class_stats['confidence_std'][class_idx])  # 标准差可靠性
            self.class_stats['reliability_score'][class_idx] = (sample_reliability + std_reliability) / 2.0
    
    def update_threshold(self, 
                        output_unlabeled: torch.Tensor, 
                        pseudo_label: torch.Tensor,
                        labeled_output: Optional[torch.Tensor] = None,
                        labeled_gt: Optional[torch.Tensor] = None,
                        iter_num: int = 0,
                        warmup_iters: int = 500):
        """
        改进的阈值更新方法
        
        Args:
            output_unlabeled: 无标签数据的模型输出 [B, C, H, W, D] 或 [B, C, H, W]
            pseudo_label: 伪标签 [B, H, W, D] 或 [B, H, W]
            labeled_output: 有标签数据的模型输出（用于校准）
            labeled_gt: 真实标签（用于校准）
            iter_num: 当前迭代次数
            warmup_iters: 预热迭代数，在此期间使用更保守的更新
        """
        
        # 转换为softmax概率
        current_unlabeled_soft = F.softmax(output_unlabeled, dim=1).cpu()
        
        # 如果有标签数据，用于校准
        calibration_factor = 1.0
        if labeled_output is not None and labeled_gt is not None:
            calibration_factor = self._compute_calibration_factor(labeled_output, labeled_gt)
        
        # 更新每个类别的阈值
        for class_idx in range(self.class_num):
            # 获取当前类别的像素mask
            class_mask = (pseudo_label == class_idx).cpu()
            
            if class_mask.sum() < 5:  # 样本数过少，跳过更新
                continue
            
            # 获取当前类别的预测概率
            class_probs = current_unlabeled_soft[:, class_idx]
            valid_probs = class_probs[class_mask]
            
            # 更新类别统计信息
            self._update_class_stats(class_idx, class_probs, class_mask)
            
            # 计算基础置信度
            base_confidence = valid_probs.mean().item()
            
            # 计算不确定性调整
            uncertainty = self._compute_uncertainty(current_unlabeled_soft[:, class_idx:class_idx+1][class_mask])
            uncertainty_adjustment = self.uncertainty_weight * uncertainty
            
            # 调整后的置信度
            adjusted_confidence = base_confidence - uncertainty_adjustment
            
            # 应用校准因子
            adjusted_confidence *= calibration_factor
            
            # 异常检测
            is_anomaly = self._detect_anomalies(class_idx, adjusted_confidence)
            
            if not is_anomaly:
                # 更新历史
                self.confidence_history[class_idx].append(adjusted_confidence)
                
                # 根据可靠性得分调整学习率
                reliability = self.class_stats['reliability_score'][class_idx]
                adaptive_alpha = self.alpha * reliability + (1 - reliability) * 0.95  # 不可靠时更保守
                
                # 预热期间使用更保守的更新
                if iter_num < warmup_iters:
                    adaptive_alpha = min(adaptive_alpha, 0.95)
                
                # 指数移动平均更新
                old_threshold = self.dynamic_thresholds[class_idx]
                new_threshold = (old_threshold * adaptive_alpha + 
                               (1 - adaptive_alpha) * adjusted_confidence)
                
                # 应用自适应约束
                new_threshold = max(self.min_thresholds[class_idx], new_threshold)
                new_threshold = min(self.max_thresholds[class_idx], new_threshold)
                
                # 额外的平滑约束：单次更新幅度不超过10%
                max_change = abs(old_threshold * 0.1)
                if abs(new_threshold - old_threshold) > max_change:
                    if new_threshold > old_threshold:
                        new_threshold = old_threshold + max_change
                    else:
                        new_threshold = old_threshold - max_change
                
                self.dynamic_thresholds[class_idx] = new_threshold
        
        # 记录历史（包含统计信息）
        self.plt_thresholds[iter_num] = {
            'thresholds': self.dynamic_thresholds.copy(),
            'stats': {k: v.copy() if isinstance(v, list) else v for k, v in self.class_stats.items()},
            'calibration_factor': calibration_factor
        }
    
    def _compute_calibration_factor(self, labeled_output: torch.Tensor, labeled_gt: torch.Tensor) -> float:
        """基于有标签数据计算校准因子"""
        with torch.no_grad():
            labeled_soft = F.softmax(labeled_output, dim=1).cpu()
            
            # 计算预测准确率
            pred_labels = labeled_soft.argmax(dim=1)
            accuracy = (pred_labels == labeled_gt.cpu()).float().mean().item()
            
            # 校准因子：准确率越低，越应该降低阈值
            calibration_factor = max(0.5, min(1.2, accuracy + 0.2))
            
        return calibration_factor
    
    def get_threshold_mask(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        根据当前阈值生成mask
        
        Args:
            predictions: 模型预测输出 [B, C, H, W, D] 或 [B, C, H, W]
            
        Returns:
            mask: 高置信度像素的mask [B, H, W, D] 或 [B, H, W]
        """
        probs = F.softmax(predictions, dim=1)
        max_probs, pred_labels = probs.max(dim=1)
        
        # 为每个类别应用对应的阈值
        threshold_mask = torch.zeros_like(max_probs, dtype=torch.bool)
        
        for class_idx in range(self.class_num):
            class_mask = (pred_labels == class_idx)
            class_threshold_mask = max_probs > self.dynamic_thresholds[class_idx]
            threshold_mask |= (class_mask & class_threshold_mask)
        
        return threshold_mask
    
    def get_class_weights(self) -> List[float]:
        """基于当前统计信息计算类别权重"""
        weights = []
        for class_idx in range(self.class_num):
            # 基于可靠性和样本数计算权重
            reliability = self.class_stats['reliability_score'][class_idx]
            sample_ratio = max(0.1, self.class_stats['sample_count'][class_idx] / 1000.0)
            weight = reliability * sample_ratio
            weights.append(weight)
        
        # 归一化
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / self.class_num] * self.class_num
            
        return weights
    
    def print_status(self, iter_num: int):
        """打印当前状态"""
        print(f"\n=== Iter {iter_num} Dynamic Thresholds Status ===")
        for class_idx in range(self.class_num):
            thresh = self.dynamic_thresholds[class_idx]
            reliability = self.class_stats['reliability_score'][class_idx]
            sample_count = self.class_stats['sample_count'][class_idx]
            confidence_std = self.class_stats['confidence_std'][class_idx]
            
            print(f"Class {class_idx}: thresh={thresh:.3f}, "
                  f"reliability={reliability:.3f}, "
                  f"samples={sample_count}, "
                  f"std={confidence_std:.3f}")