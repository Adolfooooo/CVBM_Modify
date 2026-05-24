# Ablation 06: Prototype Contrast Single Branch

本次消融针对 `experiments/cvbm_15_1/t2/a/pancreas_train.py` 当前实际生效的对比学习模块进行修改。

基线脚本中的对比学习并不是已注释的 `BlockInfoNCELoss`，而是自训练阶段使用的 `BranchBatchPrototypeLoss`。该模块同时对：

- 前景 decoder feature 分支 `feat_fg`
- 背景 decoder feature 分支 `feat_bg`

分别构建 prototype，并将两条分支的 contrast loss 做平均。

本次 `ablation_06` 的目标是：

- 保留前景分支的 prototype contrast
- 去掉背景分支的 prototype contrast
- 保留原有前景/背景分割监督、SKC 交互、EMA teacher、mix loss 等其余训练流程不变

对应实验脚本：

- `experiments/cvbm_15_1/t2/a/ablation/ablation_06/pancreas_train.py`

## 实验目的

验证当前性能收益是否主要来自前景分支的 prototype contrast。

由于 Pancreas 任务的核心目标类别是胰腺前景，单分支消融可以回答两个问题：

1. 背景分支的 contrast 监督是否真正带来增益。
2. 如果只保留前景分支，性能是否仍能接近双分支基线，同时降低额外约束带来的噪声。

## 变量设置

基线组：

- `experiments/cvbm_15_1/t2/a/pancreas_train.py`
- prototype contrast 使用前景分支 + 背景分支

消融组：

- `experiments/cvbm_15_1/t2/a/ablation/ablation_06/pancreas_train.py`
- prototype contrast 仅保留前景分支

固定不变部分：

- 数据集与划分方式
- `pre_train` 和 `self_train` 迭代数
- 网络骨干与 SKC 模块
- 混合监督方式
- `proto_weight`
- `proto_dim`
- `proto_temperature`
- `proto_patch`
- 其余训练超参数

唯一核心变量：

- prototype contrast 是否保留背景分支

## 实现说明

本次单分支消融的具体改动为：

- 自定义 `SingleBranchPrototypeLoss`
- 只对 `feat_fg`、`labels_fg`、`confidence_fg` 构造前景 prototype contrast
- 不再对 `feat_bg` 构造背景 prototype contrast
- TensorBoard 中保留 `proto_fg_queries` 统计
- `proto_bg_queries` 固定记录为 `0.0`，便于和基线日志直接对照

也就是说，本次消融移除的是“背景分支的对比学习约束”，不是移除背景分支的分割监督。

## 实验安排

建议按以下顺序执行：

1. 先运行基线双分支版本，记录最优 Dice 和 EMA Dice。
2. 再运行 `ablation_06` 单分支版本，保持其余超参数完全一致。
3. 对比两组的：
   - best Dice
   - best EMA Dice
   - `proto_fg_queries`
   - 收敛速度
   - 训练稳定性
4. 如果单分支结果接近或优于双分支，说明前景分支已经承担了主要的对比约束贡献，背景分支 contrast 价值有限。
5. 如果单分支明显下降，说明双分支 prototype contrast 对 fg/bg 表征解耦仍然有效。

建议记录表头：

- experiment_name
- contrast_branch
- best_model_dice
- best_ema_dice
- proto_fg_queries_mean
- proto_bg_queries_mean
- train_time
- gpu_memory
- notes

## 运行脚本

下面给出 `run_pancreas.sh` 风格的运行模板。脚本中同时包含双分支基线和单分支消融，便于直接对照。

```bash
#!/usr/bin/env bash

set -euo pipefail

ROOT_PATH="/root/Pancreas"
GPU="0"
LABELNUM="12"

BASE_SCRIPT="experiments/cvbm_15_1/t2/a/pancreas_train.py"
ABLATION_SCRIPT="experiments/cvbm_15_1/t2/a/ablation/ablation_06/pancreas_train.py"

run_baseline () {
  CUDA_VISIBLE_DEVICES="${GPU}" python "${BASE_SCRIPT}" \
    --root_path "${ROOT_PATH}" \
    --exp "CVBM_Pancreas_Baseline_TwoBranchProto" \
    --gpu "${GPU}" \
    --labelnum "${LABELNUM}" \
    --snapshot_path "./results/CVBM_15_1_t2_a/baseline_two_branch"
}

run_ablation_single_branch () {
  CUDA_VISIBLE_DEVICES="${GPU}" python "${ABLATION_SCRIPT}" \
    --root_path "${ROOT_PATH}" \
    --exp "CVBM_Pancreas_Ablation_06_SingleProtoBranch" \
    --gpu "${GPU}" \
    --labelnum "${LABELNUM}" \
    --snapshot_path "./results/CVBM_15_1_t2_a/ablation_06/single_branch"
}

run_baseline
run_ablation_single_branch
```

如果需要多卡或多组 `labelnum` 对照，可以在这个模板上继续扩展。

## 结果分析建议

如果实验结果出现以下情况，可以按下面方式解释：

- 单分支接近双分支
  说明 prototype contrast 的主要收益来自前景目标表征，背景 contrast 的附加价值有限。

- 单分支优于双分支
  说明背景分支 contrast 可能引入了额外噪声，尤其在伪标签存在误差时，背景 prototype 约束会干扰优化。

- 单分支明显弱于双分支
  说明双分支 contrast 确实帮助模型建立了更稳定的前景/背景分离表征，去掉背景分支会削弱判别能力。

- 单分支更稳定但上限略低
  说明背景 contrast 对冲高性能有帮助，但同时增加了训练耦合复杂度。
