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
    def __init__(self, model_path, num_classes=None):
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
        self.transform = transform=transforms.Compose([
            RandomGenerator([256, 256]),
            CreateOnehotLabel(num_classes)
        ])
        
        # 定义类别颜色
        self.colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF'][:num_classes]
        self.class_names = [f'Class {i}' for i in range(num_classes)]
    
    def load_and_preprocess_image(self, image_sample):
        """加载和预处理图片"""
        # 预处理用于模型输入
        input_tensor = self.transform(image_sample)
        image = input_tensor['image']
        
        return input_tensor, image.squeeze(0).cpu().numpy()
    
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
    
    def visualize_results(self, image_sample):
        """完整的可视化流程"""
        # 加载和预处理图片
        input_tensor, display_image = self.load_and_preprocess_image(image_sample)
        
        # 模型预测
        output, prediction = self.predict(input_tensor)
        
        # 创建彩色掩码
        colored_mask = self.create_colored_mask(prediction[0].cpu().numpy())
        
        # 创建主要的可视化图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 输入图片
        axes[0].imshow(display_image)
        axes[0].set_title('Input Image', fontsize=14)
        axes[0].axis('off')
        self.add_scale_bar(axes[0], display_image.shape)
        
        # 2. 分割结果
        axes[1].imshow(colored_mask)
        axes[1].set_title('Segmentation Result', fontsize=14)
        axes[1].axis('off')
        self.add_scale_bar(axes[1], colored_mask.shape)
        
        # 3. 叠加结果 - 需要处理单通道输入图像
        if len(display_image.shape) == 2:
            display_image_rgb = np.stack([display_image, display_image, display_image], axis=-1)
        else:
            display_image_rgb = display_image

        alpha = 0.6
        overlay = cv2.addWeighted(display_image_rgb.astype(np.uint8), 1-alpha, 
                                colored_mask.astype(np.uint8), alpha, 0)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay Result', fontsize=14)
        axes[2].axis('off')
        self.add_scale_bar(axes[2], overlay.shape)
        
        # 添加图例
        legend_patches = [mpatches.Patch(color=color, label=name) 
                        for color, name in zip(self.colors, self.class_names)]
        fig.legend(handles=legend_patches, loc='center right', bbox_to_anchor=(1.15, 0.5))
        
        plt.tight_layout()
        plt.savefig('images/demo_visual_output.png', dpi=300, bbox_inches='tight')
        # plt.show()
        
        # 可视化中间层特征
        # if hasattr(self.model, 'intermediate_features') and self.model.intermediate_features:
        #     features = self.model.intermediate_features['encoder_output']
        #     feature_fig = self.visualize_features(features, "Encoder Output Feature Maps")
        #     plt.show()
        
        return fig

# 使用示例
def run_demo():
    """运行演示"""
    # 创建一个示例图片（如果没有真实图片）

    h5f = h5py.File('/root/ACDC/data/slices/patient001_frame02_slice_9.h5', 'r')
    image = h5f['image'][:]

    # demo_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    # demo_image_pil = Image.fromarray(demo_image)
    image = (image * 255).astype(np.uint8)
    image_rgb = np.stack([image, image, image], axis=-1)
    Image.fromarray(image_rgb.astype(np.uint8)).save('images/demo_input.jpg')
    
    # 初始化可视化器
    visualizer = ModelVisualizer('./results/CVBM_4_3/1/CVBM2d_ACDC_3_labeled/self_train/CVBM2d_Argument_best_model.pth', num_classes=4)
    
    # 运行可视化
    print(f"开始可视化...")
    visualizer.visualize_results(h5f)

if __name__ == "__main__":
    run_demo()
