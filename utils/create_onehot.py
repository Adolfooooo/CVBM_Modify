import torch
import numpy as np

class OneHotConverter(object):
    
    @staticmethod
    def to_onehot(sample, num_classes):
        """
        将标签转换为onehot编码
        
        Args:
            sample: dict, must include 'image' or 'label' keys
            num_classes: 类别数量
            
        Returns:
            dict, with onehot labels added
        """
        result = {}
        
        # 复制image字段
        for key in sample:
            if key.startswith('image'):
                result[key] = sample[key]
        
        # 转换label字段
        for key in sample:
            if key.startswith('label'):
                label = sample[key]
                onehot_key = f'onehot_{key}'
                
                # 判断label的维度并相应创建onehot_label
                if label.dim() == 2:  # 单张图片 [H, W]
                    onehot_label = torch.zeros((num_classes, label.shape[0], label.shape[1]), 
                                             dtype=torch.float32, device=label.device)
                    for i in range(num_classes):
                        onehot_label[i, :, :] = (label == i).type(torch.float32)
                
                elif label.dim() == 3:  # 批量数据 [B, H, W]
                    onehot_label = torch.zeros((num_classes, label.shape[0], label.shape[1], label.shape[2]), 
                                             dtype=torch.float32, device=label.device)
                    for i in range(num_classes):
                        onehot_label[i, :, :, :] = (label == i).type(torch.float32)
                
                else:
                    raise ValueError(f"Unsupported label dimension: {label.dim()}. Expected 2D or 3D tensor.")
                
                # result[key] = label
                result[onehot_key] = onehot_label.permute(1, 0, 2, 3)
        
        return result
    
    @staticmethod
    def to_label(sample, num_classes=None):
        """
        将onehot编码转换为标签
        
        Args:
            sample: dict, must include 'image' or 'onehot_label' keys  
            num_classes: 类别数量 (此函数中不使用，但保持接口一致性)
            
        Returns:
            dict, with labels added
        """
        result = {}
        
        # 复制image字段
        for key in sample:
            if key.startswith('image'):
                result[key] = sample[key]
        
        # 复制普通label字段
        for key in sample:
            if key.startswith('label') and not key.startswith('onehot_'):
                result[key] = sample[key]
        
        # 转换onehot字段
        for key in sample:
            if key.startswith('onehot_'):
                onehot_label = sample[key]
                original_key = key[7:]  # 去掉'onehot_'
                
                if isinstance(onehot_label, torch.Tensor):
                    # 根据onehot_label的维度进行不同处理
                    if onehot_label.dim() == 3:  # [C, H, W] -> [H, W]
                        label = torch.argmax(onehot_label, dim=0)
                    elif onehot_label.dim() == 4:  # [C, B, H, W] -> [B, H, W]  
                        label = torch.argmax(onehot_label, dim=0)
                    else:
                        raise ValueError(f"Unsupported onehot dimension: {onehot_label.dim()}. Expected 3D or 4D tensor.")
                else:
                    # NumPy数组处理
                    if onehot_label.ndim == 3:  # [C, H, W] -> [H, W]
                        label = np.argmax(onehot_label, axis=0)
                    elif onehot_label.ndim == 4:  # [C, B, H, W] -> [B, H, W]
                        label = np.argmax(onehot_label, axis=0)
                    else:
                        raise ValueError(f"Unsupported onehot dimension: {onehot_label.ndim}. Expected 3D or 4D array.")
                
                result[original_key] = label
        
        return result