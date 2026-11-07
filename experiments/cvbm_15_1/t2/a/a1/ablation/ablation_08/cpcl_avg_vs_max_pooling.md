# Ablation 08：CPCL 原有平均池化与最大池化对比

本实验仅将 CPCL 的特征、标签、置信度三处局部平均池化改为最大池化。kernel_size 和 stride 仍为 proto_patch，标签 >=0.5 二值化、阈值、投影、原型构造和损失计算均保持不变。

默认 patch_max：F.max_pool3d；对照 patch_avg：原有 F.avg_pool3d。三个训练入口均支持 --proto_pooling {patch_max,patch_avg}。共享 CPCL 默认仍为 patch_avg，不影响已有实验。

注意：最大池化后，二值标签窗口只要包含前景就为前景；置信度取窗口最大值。这是替换算子的直接结果。两组应指定不同 snapshot_path，以免覆盖结果。未启动训练。
