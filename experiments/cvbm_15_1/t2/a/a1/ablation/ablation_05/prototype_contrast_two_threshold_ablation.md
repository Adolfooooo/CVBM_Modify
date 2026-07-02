# Ablation 05: Prototype Contrast Two Thresholds

本次消融实验针对 `experiments/cvbm_15_1/t2/a/pancreas_train.py` 中当前实际生效的 prototype contrast 模块展开，重点分析两个阈值超参数对性能的影响：

- `--proto_conf_threshold`
- `--proto_query_threshold`

对应的消融训练脚本保存在：

- `experiments/cvbm_15_1/t2/a/ablation/ablation_05/pancreas_train.py`

## 实验背景

当前 `t2/a` 版本中，旧的 `BlockInfoNCELoss` 对比学习分支已经被注释，训练时真正参与优化的是：

- `BranchBatchPrototypeLoss`

其中两个阈值的作用分别是：

- `proto_conf_threshold`：用于筛选哪些 patch 可以参与 batch prototype 构建。阈值越高，prototype 更纯净，但样本数量更少。
- `proto_query_threshold`：用于筛选哪些 patch 可以作为 query 参与 prototype classification。阈值越高，query 更可靠，但监督密度会下降。

因此，这次消融的目标就是拆分观察：

- prototype 构建阈值对 prototype 质量和类别覆盖率的影响
- query 阈值对监督密度、噪声控制和最终 Dice 的影响

## 实验目标

重点回答下面几个问题：

1. 提高 `proto_conf_threshold` 是否能通过提升 prototype 纯度来改善性能。
2. 提高 `proto_query_threshold` 是否能减少噪声 query，从而稳定训练。
3. 两个阈值同时提高时，是否会导致可用 patch 数过少，进而削弱对比监督。
4. `proto_fg_queries` 和 `proto_bg_queries` 的数量变化是否与最终性能趋势一致。

## 实验设置

建议以当前主实验默认配置作为基线：

- `proto_conf_threshold = 0.8`
- `proto_query_threshold = 0.0`

固定不变的部分：

- 数据集与数据划分
- 网络结构
- 预训练与自训练迭代数
- `proto_weight`
- `proto_dim`
- `proto_temperature`
- `proto_patch`
- 其他训练超参数

本次只调整上述两个阈值。

## 建议消融方案

### 第一组：单变量消融 `proto_conf_threshold`

固定：

- `proto_query_threshold = 0.0`

对比：

1. `proto_conf_threshold = 0.6`
2. `proto_conf_threshold = 0.7`
3. `proto_conf_threshold = 0.8`  `# baseline`
4. `proto_conf_threshold = 0.9`

预期：

- 较低阈值会引入更多 prototype patch，类别覆盖更充分，但 prototype 纯度可能下降。
- 较高阈值会提升 prototype 可靠性，但可能造成某些类别 patch 不足，尤其影响前景类。

### 第二组：单变量消融 `proto_query_threshold`

固定：

- `proto_conf_threshold = 0.8`

对比：

1. `proto_query_threshold = 0.0`  `# baseline`
2. `proto_query_threshold = 0.3`
3. `proto_query_threshold = 0.5`
4. `proto_query_threshold = 0.7`

预期：

- `0.0` 会保留全部有效 query，监督最密，但噪声最多。
- 中等阈值可能在 query 质量和监督密度之间取得平衡。
- 过高阈值可能导致 query 数过少，prototype contrast 贡献减弱。

### 第三组：双变量组合消融

如果计算资源允许，建议再补一组组合实验：

1. `conf=0.7, query=0.3`
2. `conf=0.8, query=0.3`
3. `conf=0.8, query=0.5`
4. `conf=0.9, query=0.5`

这组实验用于判断：

- prototype 更“纯”与 query 更“严”的组合是否协同
- 是否存在一个兼顾 prototype 稳定性和 query 数量的折中区间

## 实验记录建议

每组实验建议统一记录：

- `best_model_dice`
- `best_ema_dice`
- `proto_fg_queries` 平均值
- `proto_bg_queries` 平均值
- 是否出现 query 数骤降
- 训练稳定性
- 单次训练时长和显存占用

建议汇总表字段：

- `experiment_name`
- `proto_conf_threshold`
- `proto_query_threshold`
- `best_model_dice`
- `best_ema_dice`
- `proto_fg_queries_mean`
- `proto_bg_queries_mean`
- `gpu_memory`
- `train_time`
- `notes`

## 运行脚本

下面脚本按 `run_pancreas.sh` 风格给出，可直接作为批量运行模板使用。

```bash
#!/usr/bin/env bash

set -euo pipefail

ROOT_PATH="/root/Pancreas"
SCRIPT="experiments/cvbm_15_1/t2/a/ablation/ablation_05/pancreas_train.py"
GPU="0"
LABELNUM="12"
EXP_NAME="CVBM_Pancreas_Ablation_ProtoThreshold"

run_case () {
  local tag="$1"
  local conf_thr="$2"
  local query_thr="$3"

  CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT}" \
    --root_path "${ROOT_PATH}" \
    --exp "${EXP_NAME}_${tag}" \
    --gpu "${GPU}" \
    --labelnum "${LABELNUM}" \
    --proto_conf_threshold "${conf_thr}" \
    --proto_query_threshold "${query_thr}" \
    --proto_weight 0.1 \
    --proto_dim 32 \
    --proto_temperature 0.2 \
    --proto_patch 8 8 8 \
    --proto_max_queries 4096 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_05/${tag}"
}

# baseline
run_case "baseline_conf0.8_query0.0" 0.8 0.0

# conf threshold ablation
run_case "conf0.6_query0.0" 0.6 0.0
run_case "conf0.7_query0.0" 0.7 0.0
run_case "conf0.9_query0.0" 0.9 0.0

# query threshold ablation
run_case "conf0.8_query0.3" 0.8 0.3
run_case "conf0.8_query0.5" 0.8 0.5
run_case "conf0.8_query0.7" 0.8 0.7

# optional joint ablation
run_case "conf0.7_query0.3" 0.7 0.3
run_case "conf0.9_query0.5" 0.9 0.5
```

## 结果分析建议

如果结果呈现下面趋势，可以按对应逻辑解释：

- `conf` 提高后性能提升
  说明 prototype 纯度比 patch 数量更关键，低质量 patch 会明显污染类原型。

- `conf` 提高后性能下降
  说明 prototype 构建阶段过度筛选，导致 prototype 覆盖不足，尤其前景类样本不够。

- `query` 从 `0.0` 提高到中等值后性能提升
  说明 query 端去噪是有效的，全部 patch 参与会带来明显伪标签噪声。

- `query` 过高后性能下降
  说明虽然 query 更干净，但监督密度不够，contrast 分支开始退化。

- `conf` 和 `query` 同时提高后 query 数显著减少
  说明 prototype 构建和 query 监督都变得过于保守，可能导致 prototype loss 实际贡献不足。

## 推荐执行顺序

建议按以下顺序跑，便于尽快观察趋势：

1. 跑 baseline：`conf=0.8, query=0.0`
2. 跑 `query` 单变量组：优先观察 query 去噪收益
3. 跑 `conf` 单变量组：观察 prototype 纯度影响
4. 最后补组合组：验证两个阈值是否存在协同区间

## 最终结论建议

最终汇报时建议明确回答：

- 两个阈值中，哪个对 Dice 更敏感
- 提升性能主要来自 prototype 更纯，还是 query 更可靠
- 最优阈值组合是否会显著减少 `proto_fg_queries`
- 对后续主实验是否建议保留默认 `0.8 / 0.0`，还是调整到新的最佳组合
