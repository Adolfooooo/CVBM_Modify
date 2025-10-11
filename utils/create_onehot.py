import torch
import numpy as np

class OneHotConverter:
    @staticmethod
    def to_onehot(label, num_classes, device='cpu'):
        """
        将标签转换为 one-hot 编码
        
        Args:
            label (torch.Tensor | np.ndarray): 
                - 形状: [H, W] 或 [B, H, W]
                - 任意大小的整型标签张量或数组
            num_classes (int): 类别数量
            device (str): 处理设备 ('cpu' 或 'cuda')
                
        Returns:
            torch.Tensor:
                - 若输入为 [H, W] -> 输出形状为 [num_classes, H, W]
                - 若输入为 [B, H, W] -> 输出形状为 [B, num_classes, H, W]
        """
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)

        if not torch.is_tensor(label):
            raise TypeError("label must be a torch.Tensor or np.ndarray")

        # 将 label 移动到指定 device
        device = torch.device(device)
        label = label.to(device)

        if label.dim() == 2:
            # 单张图片 [H, W]
            onehot = torch.zeros((num_classes, *label.shape), dtype=torch.float32, device=device)
            onehot.scatter_(0, label.unsqueeze(0), 1.0)

        elif label.dim() == 3:
            # 批量数据 [B, H, W]
            onehot = torch.zeros((label.shape[0], num_classes, label.shape[1], label.shape[2]),
                                 dtype=torch.float32, device=device)
            onehot.scatter_(1, label.unsqueeze(1), 1.0)

        else:
            raise ValueError(f"Unsupported label dimension: {label.dim()}. Expected 2D or 3D tensor.")
        
        return onehot

    @staticmethod
    def to_label(onehot, device='cpu'):
        """
        将 one-hot 编码还原为标签
        
        Args:
            onehot (torch.Tensor | np.ndarray):
                - 形状: [num_classes, H, W] 或 [B, num_classes, H, W]
                - 任意大小的浮点张量或数组
            device (str): 处理设备 ('cpu' 或 'cuda')
                
        Returns:
            torch.Tensor:
                - 若输入为 [num_classes, H, W] -> 输出 [H, W]
                - 若输入为 [B, num_classes, H, W] -> 输出 [B, H, W]
        """
        if isinstance(onehot, np.ndarray):
            onehot = torch.from_numpy(onehot)

        if not torch.is_tensor(onehot):
            raise TypeError("onehot must be a torch.Tensor or np.ndarray")

        # 将 onehot 移动到指定 device
        device = torch.device(device)
        onehot = onehot.to(device)

        if onehot.dim() == 3:
            # [C, H, W]
            label = torch.argmax(onehot, dim=0)
        elif onehot.dim() == 4:
            # [B, C, H, W]
            label = torch.argmax(onehot, dim=1)
        else:
            raise ValueError(f"Unsupported onehot dimension: {onehot.dim()}. Expected 3D or 4D tensor.")
        
        return label.to(device)
