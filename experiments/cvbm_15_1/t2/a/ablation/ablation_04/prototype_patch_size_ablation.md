# Ablation 04: Prototype Contrast Patch Size

本次消融实验针对 `experiments/cvbm_15_1/t2/a/pancreas_train.py` 中当前实际生效的对比学习分块模块展开。

需要先说明一点：

- 当前基线文件里旧的 `BlockInfoNCELoss` + `contrast_patch` 代码已经被注释，不参与训练。
- 当前真正生效的对比学习模块是 `BranchBatchPrototypeLoss`。
- 因此，本次“不同 patch_size 的影响”应当以 `--proto_patch` 为主变量进行消融。

基线参考：

- `experiments/cvbm_15_1/t2/a/pancreas_train.py`

本次实验脚本：

- `experiments/cvbm_15_1/t2/a/ablation/ablation_04/pancreas_train.py`

## 实验目标

分析 prototype contrast 模块中不同 `proto_patch` 设置对 Pancreas 半监督分割性能的影响，重点观察：

- patch 过小时，prototype query 数量增加，监督更细粒度，但噪声和显存压力也会上升。
- patch 过大时，prototype pooling 更平滑，噪声降低，但局部边界信息可能被过度平均。
- 不同 patch size 下，prototype branch 的查询数量 `proto_fg_queries` / `proto_bg_queries` 是否稳定。
- patch size 改变后是否引入尺寸不整除问题、query 数量过少问题，或导致训练不稳定。

## 实验变量

固定不变的部分：

- 数据集与数据划分
- 预训练和自训练迭代数
- `labelnum`
- 网络结构
- `proto_weight`
- `proto_dim`
- `proto_temperature`
- 其他训练策略

唯一主变量：

- `--proto_patch`

建议对比组：

1. 基线组：`--proto_patch 8 8 8`
2. 小 patch 组：`--proto_patch 4 4 4`
3. 中等偏大组：`--proto_patch 12 12 12`
4. 大 patch 组：`--proto_patch 16 16 16`

如果计算资源允许，可以补充：

5. 极小 patch 组：`--proto_patch 2 2 2`
6. 非立方 patch 组：`--proto_patch 8 8 4`

## 尺寸约束与注意事项

本实验默认输入 patch 为 `--patch_size 96 96 96`。

由于 prototype loss 内部会执行：

```python
F.avg_pool3d(features, kernel_size=self.patch_size, stride=self.patch_size)
```

所以必须保证：

- `patch_size` 的每一维都能被 `proto_patch` 对应维度整除。
- 否则会出现 pooling 尺寸不匹配或边界被截断的问题。

对当前默认配置 `96 x 96 x 96` 来说，以下值是安全的：

- `4 4 4`
- `6 6 6`
- `8 8 8`
- `12 12 12`
- `16 16 16`
- `24 24 24`

以下情况需要特别注意：

- `proto_patch` 太大时，pool 后 patch 数过少，prototype contrast 可能退化。
- `proto_patch` 太小时，query 数大量增加，显存占用和训练时间会明显上升。
- 如果后续重新启用旧的 `contrast_patch` 分支，还需要额外保证 `contrast_patch` 能整除 logit 尺寸，并且 `topnum < patch_num`。

本次 `ablation_04/pancreas_train.py` 已额外加入：

- `--patch_size` / `--proto_patch` / `--contrast_patch` 的稳定命令行解析
- `proto_patch` 与输入尺寸的整除检查
- `contrast_patch` 与输入尺寸的整除检查
- 对 legacy `topnum` 的提示性检查

## 实验安排

建议先按下面顺序执行：

1. 先跑基线 `8 8 8`，确认结果与主实验版本接近。
2. 再跑 `4 4 4`，观察更细粒度 patch 是否提升 Dice，但是否带来更大波动。
3. 再跑 `12 12 12` 和 `16 16 16`，观察 coarse prototype pooling 对稳定性和性能的影响。
4. 统一记录最佳 Dice、EMA 最佳 Dice、训练时长、显存占用、`proto_fg_queries`、`proto_bg_queries`。

建议记录表头：

- experiment_name
- proto_patch
- best_model_dice
- best_ema_dice
- proto_fg_queries_mean
- proto_bg_queries_mean
- gpu_memory
- train_time
- notes

## 运行脚本

下面脚本可直接作为 `run_pancreas.sh` 风格的运行模板使用。

```bash
#!/usr/bin/env bash

set -euo pipefail

ROOT_PATH="/root/Pancreas"
SCRIPT="experiments/cvbm_15_1/t2/a/ablation/ablation_04/pancreas_train.py"
EXP_NAME="CVBM_Pancreas_Ablation_ProtoPatch"
GPU="0"
LABELNUM="12"

run_case () {
  local tag="$1"
  local p1="$2"
  local p2="$3"
  local p3="$4"

  CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT}" \
    --root_path "${ROOT_PATH}" \
    --exp "${EXP_NAME}_${tag}" \
    --gpu "${GPU}" \
    --labelnum "${LABELNUM}" \
    --patch_size 96 96 96 \
    --proto_patch "${p1}" "${p2}" "${p3}" \
    --contrast_patch 8 8 8 \
    --proto_weight 0.1 \
    --proto_dim 32 \
    --proto_temperature 0.2 \
    --proto_conf_threshold 0.8 \
    --proto_query_threshold 0.0 \
    --proto_max_queries 4096 \
    --snapshot_path "./results/CVBM_15_1_t2_a/ablation_04/${tag}"
}

run_case "proto_patch_8_8_8" 8 8 8
run_case "proto_patch_4_4_4" 4 4 4
run_case "proto_patch_12_12_12" 12 12 12
run_case "proto_patch_16_16_16" 16 16 16
```

## 结果分析建议

如果实验结果出现以下现象，可按下面思路解释：

- `4 4 4` 优于 `8 8 8`
  说明更细粒度的局部 prototype 对 pancreas 边界和小区域判别更有帮助。

- `4 4 4` 不稳定或效果下降
  说明 patch 太细导致伪标签噪声被放大，prototype 构造受到低质量 query 干扰。

- `12 12 12` 或 `16 16 16` 更稳定但上限下降
  说明 coarse pooling 降低了噪声，但削弱了局部判别能力。

- 大 patch 明显退化
  说明 prototype token 数不足，branch-level contrast 的监督密度不够。

## 建议最终汇报方式

建议至少汇报以下内容：

- 不同 `proto_patch` 下的 best Dice / best EMA Dice
- 每组训练的 `proto_fg_queries` / `proto_bg_queries` 统计
- 最优组与基线组的可视化对比
- 对 patch granularity 与 prototype contrast 有效性的结论总结
