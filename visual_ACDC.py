import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import h5py

from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from networks.net_factory import net_factory
from dataloaders.dataset import RandomGenerator, CreateOnehotLabel

class ModelVisualizer:
    def __init__(
            self, model_path, 
            transform=transforms.Compose([
                RandomGenerator([256, 256]),
                CreateOnehotLabel(num_classes=4)
            ]),
            num_classes=None
        ):
        self.num_classes = num_classes
        self.model = net_factory(net_type='CVBM2d_Argument', in_chns=1, class_num=num_classes, mode="train").cpu()
        
        # 加载模型权重（如果存在）
        try:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"成功加载模型权重：{model_path}")
        except:
            print("未找到权重文件，使用随机初始化的模型进行demo")
        
        self.model.eval()
        
        # 定义预处理
        self.transform = transform
        
        # 定义类别颜色
        self.colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF'][:num_classes]
        self.class_names = [f'Class {i}' for i in range(num_classes)]
    
    def load_and_preprocess_h5_image(self, image_sample):
        """
        加载和预处理图片, 适用于原始读取为.h5格式的图片
        input: image_sample - h5py.File对象
        return: input_tensor - 预处理后的张量
        """
        # 预处理用于模型输入
        input_tensor = self.transform(image_sample)
        # ACDC .h5 shape: [1, 256, 256]
        image = input_tensor['image']
        label = input_tensor['label']
        
        return input_tensor, image.squeeze(0).cpu().numpy(), label.cpu().numpy()
    
    def predict(self, input_tensor):
        """模型预测"""
        batch_input = torch.cat([input_tensor['image'].unsqueeze(0)]*6, dim=0)
        with torch.no_grad():
            output, *_= self.model(batch_input, batch_input)
            # 获取预测类别
            predicted = torch.argmax(output, dim=1).squeeze(0)
            return output, predicted
    
    def create_colored_mask(self, prediction):
        """创建彩色分割掩码"""
        h, w = prediction.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id in range(self.num_classes):
            mask = prediction == class_id
            color = np.array([int(self.colors[class_id][1:3], 16),
                             int(self.colors[class_id][3:5], 16),
                             int(self.colors[class_id][5:7], 16)])
            
            colored_mask[mask] = color
        return colored_mask
    
    def add_scale_bar(self, ax, image_shape, scale_length_pixels=50, scale_length_real=10):
        """添加标尺"""
        h, w = image_shape[:2]
        # 在右下角添加标尺
        start_x = w - scale_length_pixels - 20
        start_y = h - 30
        
        # 绘制标尺线
        ax.plot([start_x, start_x + scale_length_pixels], 
                [start_y, start_y], 'white', linewidth=3)
        ax.plot([start_x, start_x + scale_length_pixels], 
                [start_y, start_y], 'black', linewidth=1)
        
        # 添加标尺文本
        ax.text(start_x + scale_length_pixels/2, start_y - 15, 
                f'{scale_length_real} units', 
                ha='center', va='top', color='white', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    def visualize_features(self, features, title="Feature Maps"):
        """可视化特征图"""
        # 取第一个样本的特征
        features = features.squeeze(0).cpu().numpy()
        
        # 选择前16个通道进行可视化
        num_channels = min(16, features.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle(title, fontsize=16)
        
        for i in range(num_channels):
            row, col = i // 4, i % 4
            feature_map = features[i]
            
            im = axes[row, col].imshow(feature_map, cmap='viridis')
            axes[row, col].set_title(f'Channel {i}')
            axes[row, col].axis('off')
            
            # 添加颜色条
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # 隐藏多余的子图
        for i in range(num_channels, 16):
            row, col = i // 4, i % 4
            axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig
    

    '''
    完整的可视化流程
    input: image_sample - h5py.File对象; 或dict对象，keys=('images', 'labels'), Tensor对象, Tensor对象需为[H, W]格式
    features: 中间层特征张量, 可选, not included batch dimension, shape: [C, H, W]
    is_h5: 是否为.h5格式图片
    return: fig - matplotlib图形对象
    '''
    def visualize_prediction(self, image_sample, features, num_classes=4, is_h5=False, pic_name='demo_visual_output.png'):
    # def visualize_results(self, image_sample):
        # 加载和预处理图片
        if is_h5:
            input_tensor, image = self.load_and_preprocess_h5_image(image_sample)
        else:
            image = image_sample['image'].squeeze(0).detach().cpu().numpy()
            label = image_sample['label'].squeeze(0).detach().cpu().numpy()

        # Model predict, The number of prediction's Batch is 1
        # output, prediction = self.predict(input_tensor)
        pseudo_label, pseudo_label_indices = torch.max(features, dim=0)
        # 创建彩色掩码
        colored_mask = self.create_colored_mask(pseudo_label_indices.detach().cpu().numpy())
        
        # 创建主要的可视化图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 输入图片
        axes[0].imshow(image)
        axes[0].set_title('Input Image', fontsize=14)
        axes[0].axis('off')
        self.add_scale_bar(axes[0], image.shape)
        
        # 2. True label
        axes[1].imshow(label)
        axes[1].set_title('Label', fontsize=14)
        axes[1].axis('off')
        self.add_scale_bar(axes[1], colored_mask.shape)
        
        # 3. 叠加结果 - 需要处理单通道输入图像
        if len(image.shape) == 2:
            image_rgb = np.stack([image, image, image], axis=-1)
        else:
            image_rgb = image

        alpha = 0.6
        # 正确的数据范围处理
        if image_rgb.max() <= 1.0:  # 如果是0-1范围的浮点数
            image_rgb_uint8 = (image_rgb * 255).astype(np.uint8)
        else:  # 如果已经是0-255范围
            image_rgb_uint8 = image_rgb.astype(np.uint8)

        colored_mask = colored_mask.astype(np.uint8)

        # 保留原图作为初始底图
        overlay = image_rgb_uint8.copy()
        # # 只在前景区域进行融合
        # foreground_mask = pseudo_label_indices.detach().cpu().numpy() > 0
        # overlay[foreground_mask] = (
        #     image_rgb[foreground_mask] * (1 - alpha) +
        #     colored_mask[foreground_mask] * alpha
        # ).astype(np.uint8)


        foreground_mask = pseudo_label_indices.detach().cpu().numpy() > 0
        overlay[foreground_mask] = colored_mask[foreground_mask]

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay Result', fontsize=14)
        axes[2].axis('off')
        self.add_scale_bar(axes[2], overlay.shape)

        # 添加图例
        legend_patches = [mpatches.Patch(color=color, label=name) 
                        for color, name in zip(self.colors, self.class_names)]
        fig.legend(handles=legend_patches, loc='center right', bbox_to_anchor=(1.15, 0.5))
        
        plt.tight_layout()
        plt.savefig(f'images/{pic_name}', dpi=300, bbox_inches='tight')
        # plt.show()
        
        # 可视化中间层特征
        # if hasattr(self.model, 'intermediate_features') and self.model.intermediate_features:
        #     features = self.model.intermediate_features['encoder_output']
        #     feature_fig = self.visualize_features(features, "Encoder Output Feature Maps")
        #     plt.show()
        
        return fig


    def visualiza_patch_in_pic(self, image_sample, features, low_confidence_indices, 
                        grid_size, is_h5=False, pic_name='patch_visual_output.png'):
        """
        使用预计算的低置信度patches索引进行可视化
        
        Args:
            image_sample: 图像样本
            features: 特征图，用于获取分割结果
            low_confidence_indices: 低置信度patches的索引 (来自topk的结果)
            grid_size: 网格大小 (如16表示16x16网格)
            is_h5: 是否为h5格式
            pic_name: 输出图片名称
        """
        # 加载和预处理图片
        if is_h5:
            input_tensor, image, label = self.load_and_preprocess_h5_image(image_sample)
        else:
            image = image_sample['image'].squeeze(0).detach().cpu().numpy()
            label = image_sample['label'].squeeze(0).detach().cpu().numpy()
        
        # 获取预测结果
        pseudo_label, pseudo_label_indices = torch.max(features, dim=0)
        
        # 图像尺寸
        H, W = image.shape[:2]
        patch_h = H // grid_size
        patch_w = W // grid_size
        
        # 将一维索引转换为二维坐标
        def index_to_coords(idx, grid_size):
            """将一维索引转换为二维坐标 (i, j)"""
            i = idx // grid_size  # 行
            j = idx % grid_size   # 列
            return i, j
        
        # 转换索引为坐标和边界框信息
        low_confidence_patches = []
        for idx in low_confidence_indices.cpu().numpy():  # 假设indices是tensor
            i, j = index_to_coords(idx, grid_size)
            
            # 计算patch的边界
            start_h = i * patch_h
            end_h = np.minimum((i + 1) * patch_h, H)
            start_w = j * patch_w
            end_w = np.minimum((j + 1) * patch_w, W)
            
            low_confidence_patches.append({
                'idx': idx, 'i': i, 'j': j,
                'start_h': start_h, 'end_h': end_h,
                'start_w': start_w, 'end_w': end_w
            })
        
        # 创建可视化图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 原始图片
        if len(image.shape) == 2:
            axes[0].imshow(image, cmap='gray')
        else:
            axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        self.add_scale_bar(axes[0], image.shape)
        
        # 2. 显示patch分割和低置信度区域
        if len(image.shape) == 2:
            image_with_patches = np.stack([image, image, image], axis=-1)
        else:
            image_with_patches = image.copy()
        
        # 确保图像数据范围正确
        if image_with_patches.max() <= 1.0:
            image_with_patches = (image_with_patches * 255).astype(np.uint8)
        else:
            image_with_patches = image_with_patches.astype(np.uint8)
        
        overlay_patches = image_with_patches.copy()
        
        # 绘制网格线（可选）
        for i in range(1, grid_size):
            # 垂直线
            x = i * patch_w
            if x < W:
                overlay_patches[:, x-1:x+1] = [200, 200, 200]
            # 水平线
            y = i * patch_h
            if y < H:
                overlay_patches[y-1:y+1, :] = [200, 200, 200]
        
        # 高亮低置信度patches
        for patch_num, patch in enumerate(low_confidence_patches):
            start_h, end_h = patch['start_h'], patch['end_h']
            start_w, end_w = patch['start_w'], patch['end_w']
            
            # 红色边界
            overlay_patches[start_h:start_h+3, start_w:end_w] = [255, 0, 0]
            overlay_patches[end_h-3:end_h, start_w:end_w] = [255, 0, 0]
            overlay_patches[start_h:end_h, start_w:start_w+3] = [255, 0, 0]
            overlay_patches[start_h:end_h, end_w-3:end_w] = [255, 0, 0]
            
            # 半透明红色覆盖
            alpha = 0.3
            overlay_patches[start_h:end_h, start_w:end_w] = (
                overlay_patches[start_h:end_h, start_w:end_w] * (1 - alpha) +
                np.array([255, 0, 0]) * alpha
            ).astype(np.uint8)
            
            # 添加patch编号
            center_h = (start_h + end_h) // 2
            center_w = (start_w + end_w) // 2
            axes[1].text(center_w, center_h, str(patch['idx']), 
                        fontsize=10, color='white', weight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='circle,pad=0.3', facecolor='red', alpha=0.8))
        
        axes[1].imshow(overlay_patches)
        axes[1].set_title(f'Low Confidence Patches (Grid: {grid_size}x{grid_size})', fontsize=14)
        axes[1].axis('off')
        
        # 3. 叠加分割结果和低置信度区域
        colored_mask = self.create_colored_mask(pseudo_label_indices.detach().cpu().numpy())
        
        if len(image.shape) == 2:
            image_rgb = np.stack([image, image, image], axis=-1)
        else:
            image_rgb = image.copy()
        
        if image_rgb.max() <= 1.0:
            image_rgb_uint8 = (image_rgb * 255).astype(np.uint8)
        else:
            image_rgb_uint8 = image_rgb.astype(np.uint8)
        
        final_overlay = image_rgb_uint8.copy()
        
        # 添加分割结果
        foreground_mask = pseudo_label_indices.detach().cpu().numpy() > 0
        final_overlay[foreground_mask] = colored_mask[foreground_mask]
        
        # 标记低置信度patches边界
        for patch in low_confidence_patches:
            start_h, end_h = patch['start_h'], patch['end_h']
            start_w, end_w = patch['start_w'], patch['end_w']
            
            # 白色边界框
            final_overlay[start_h:start_h+2, start_w:end_w] = [255, 255, 255]
            final_overlay[end_h-2:end_h, start_w:end_w] = [255, 255, 255]
            final_overlay[start_h:end_h, start_w:start_w+2] = [255, 255, 255]
            final_overlay[start_h:end_h, end_w-2:end_w] = [255, 255, 255]
        
        axes[2].imshow(final_overlay)
        axes[2].set_title('Segmentation + Low Confidence Patches', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'images/{pic_name}', dpi=300, bbox_inches='tight')
        
        return fig

    def visualiza_patch_in_pic_version1(self, image_sample, features, low_confidence_indices, 
                            grid_size, is_h5=False, pic_name='patch_visual_output.png'):
        """
        使用预计算的低置信度patches索引进行可视化
        """
        # 0) 载入图像与标签
        if is_h5:
            input_tensor, image, label = self.load_and_preprocess_h5_image(image_sample)
        else:
            image = image_sample['image'].squeeze(0).detach().cpu().numpy()
            label = image_sample['label'].squeeze(0).detach().cpu().numpy()

        # 1) 预测（取每像素类别索引）
        # features: [C, H, W]，取 dim=0 的 argmax
        # 如果 features 是 [B, C, H, W]，请在外面或这里先 squeeze 掉 B 维
        pseudo_label, pseudo_label_indices = torch.max(features, dim=0)  # -> [H, W]

        # 2) 图像尺寸与网格
        H, W = int(image.shape[0]), int(image.shape[1])
        patch_h = max(1, H // int(grid_size))
        patch_w = max(1, W // int(grid_size))

        # 3) 工具：一维 -> 二维网格坐标（返回整数）
        def index_to_coords(idx, g):
            idx = int(idx)
            i = idx // g
            j = idx %  g
            return int(i), int(j)

        # 4) 把低置信度索引展平成一维标量列表（防止出现 array 元素）
        #    例如 (K,1)/(1,K) -> (K,)；并转 int
        lci = np.asarray(low_confidence_indices.detach().cpu().numpy()).ravel().tolist()
        lci = [int(x) for x in lci]  # 确保每个是 Python int

        # 5) 计算每个 patch 的边界（全部为整数标量）
        low_confidence_patches = []
        for idx in lci:
            i, j = index_to_coords(idx, grid_size)

            start_h = int(i * patch_h)
            end_h   = int(min((i + 1) * patch_h, H))
            start_w = int(j * patch_w)
            end_w   = int(min((j + 1) * patch_w, W))

            # 保险：裁剪到边界内，避免负数/越界
            start_h = max(0, min(start_h, H))
            end_h   = max(0, min(end_h,   H))
            start_w = max(0, min(start_w, W))
            end_w   = max(0, min(end_w,   W))
            if end_h <= start_h or end_w <= start_w:
                continue  # 空patch跳过

            low_confidence_patches.append({
                'idx': idx, 'i': i, 'j': j,
                'start_h': start_h, 'end_h': end_h,
                'start_w': start_w, 'end_w': end_w
            })

        # 6) 准备三幅图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 6.1 原图
        if image.ndim == 2:
            axes[0].imshow(image, cmap='gray')
        else:
            axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        self.add_scale_bar(axes[0], image.shape)

        # 6.2 网格 + 低置信度高亮
        if image.ndim == 2:
            image_with_patches = np.stack([image, image, image], axis=-1)
        else:
            image_with_patches = image.copy()

        # 统一到 uint8
        if image_with_patches.max() <= 1.0:
            image_with_patches = (image_with_patches * 255).astype(np.uint8)
        else:
            image_with_patches = image_with_patches.astype(np.uint8)

        overlay_patches = image_with_patches.copy()

        # 画网格（灰色线）
        for k in range(1, int(grid_size)):
            x = k * patch_w
            if 0 < x < W:
                overlay_patches[:, max(0, x-1):min(W, x+1)] = [200, 200, 200]
            y = k * patch_h
            if 0 < y < H:
                overlay_patches[max(0, y-1):min(H, y+1), :] = [200, 200, 200]

        # 高亮每个低置信度 patch
        for patch in low_confidence_patches:
            sh, eh = patch['start_h'], patch['end_h']
            sw, ew = patch['start_w'], patch['end_w']

            # 红框（2-3 像素厚）
            overlay_patches[sh:min(H, sh+3), sw:ew] = [255, 0, 0]
            overlay_patches[max(0, eh-3):eh, sw:ew] = [255, 0, 0]
            overlay_patches[sh:eh, sw:min(W, sw+3)] = [255, 0, 0]
            overlay_patches[sh:eh, max(0, ew-3):ew] = [255, 0, 0]

            # 半透明红色覆盖
            alpha = 0.3
            # 转 float 计算再回 uint8
            box = overlay_patches[sh:eh, sw:ew].astype(np.float32)
            box = box * (1 - alpha) + np.array([255, 0, 0], dtype=np.float32) * alpha
            overlay_patches[sh:eh, sw:ew] = np.clip(box, 0, 255).astype(np.uint8)

            # 标注 patch 索引（画在第二个子图上）
            ch = (sh + eh) // 2
            cw = (sw + ew) // 2
            axes[1].text(cw, ch, str(patch['idx']),
                        fontsize=10, color='white', weight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='circle,pad=0.3', facecolor='red', alpha=0.8))

        axes[1].imshow(overlay_patches)
        axes[1].set_title(f'Low Confidence Patches (Grid: {grid_size}x{grid_size})', fontsize=14)
        axes[1].axis('off')

        # 6.3 分割结果 + 低置信度框
        colored_mask = self.create_colored_mask(pseudo_label_indices.detach().cpu().numpy())

        if image.ndim == 2:
            image_rgb = np.stack([image, image, image], axis=-1)
        else:
            image_rgb = image.copy()

        image_rgb_uint8 = (image_rgb * 255).astype(np.uint8) if image_rgb.max() <= 1.0 else image_rgb.astype(np.uint8)
        final_overlay = image_rgb_uint8.copy()

        # 前景着色
        foreground_mask = pseudo_label_indices.detach().cpu().numpy() > 0
        final_overlay[foreground_mask] = colored_mask[foreground_mask]

        # 白边框标记低置信度 patch
        for patch in low_confidence_patches:
            sh, eh = patch['start_h'], patch['end_h']
            sw, ew = patch['start_w'], patch['end_w']
            final_overlay[sh:min(H, sh+2), sw:ew] = [255, 255, 255]
            final_overlay[max(0, eh-2):eh, sw:ew] = [255, 255, 255]
            final_overlay[sh:eh, sw:min(W, sw+2)] = [255, 255, 255]
            final_overlay[sh:eh, max(0, ew-2):ew] = [255, 255, 255]

        axes[2].imshow(final_overlay)
        axes[2].set_title('Segmentation + Low Confidence Patches', fontsize=14)
        axes[2].axis('off')

        # 7) 存图
        os.makedirs('images', exist_ok=True)
        plt.tight_layout()
        plt.savefig(f'images/{pic_name}', dpi=300, bbox_inches='tight')

        return fig



# 使用示例
def run_demo():
    """运行演示"""
    # 创建一个示例图片（如果没有真实图片）

    h5f = h5py.File('/root/ACDC/data/slices/patient001_frame02_slice_9.h5', 'r')
    image = h5f['image'][:]
    
    # 初始化可视化器
    visualizer = ModelVisualizer('./results/CVBM_4_2/1/CVBM2d_ACDC_3_labeled/self_train/CVBM2d_Argument_best_model.pth', num_classes=4)
    
    # 运行可视化
    print(f"开始可视化...")
    visualizer.visualize_results(h5f)

if __name__ == "__main__":
    run_demo()
