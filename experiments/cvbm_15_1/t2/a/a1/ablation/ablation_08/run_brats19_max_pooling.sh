#!/usr/bin/env bash
set -euo pipefail
trap '' HUP
cd /home/xuminghao/Projects/CVBM/CVBM-ABD
export PATH="$HOME/anaconda3/envs/cvbm/bin:$PATH"
mkdir -p logs
# Do not run both dataset scripts concurrently: each uses GPU 0 and GPU 1.

# BRATS19 train: num1, labelnum=12, GPU=0; repeats retain original seed 1337.
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.brats19_train \
    --root_path /home/xuminghao/Datasets/BRATS19 \
    --gpu 0 \
    --exp CVBM_BRATS19_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 12 \
    --patch_size 96 96 96 \
    --proto_pooling patch_max \
    --proto_patch 8 8 8 \
    --proto_weight 0.1 \
    --proto_dim 32 \
    --proto_temperature 0.2 \
    --proto_conf_threshold 0.8 \
    --proto_query_threshold 0.0 \
    --proto_max_queries 4096 \
    --deterministic 0 \
    --seed 1337 \
    --snapshot_path ./results/cvbm_15_1_t2_a1_ablation_08/max_pool/BRATS19/num1 \
    > ./logs/ablation08_maxpool_BRATS19_label12_num1_train.log 2>&1 &
pid0=$!

# BRATS19 train: num1, labelnum=25, GPU=1; repeats retain original seed 1337.
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.brats19_train \
    --root_path /home/xuminghao/Datasets/BRATS19 \
    --gpu 1 \
    --exp CVBM_BRATS19_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 25 \
    --patch_size 96 96 96 \
    --proto_pooling patch_max \
    --proto_patch 8 8 8 \
    --proto_weight 0.1 \
    --proto_dim 32 \
    --proto_temperature 0.2 \
    --proto_conf_threshold 0.8 \
    --proto_query_threshold 0.0 \
    --proto_max_queries 4096 \
    --deterministic 0 \
    --seed 1337 \
    --snapshot_path ./results/cvbm_15_1_t2_a1_ablation_08/max_pool/BRATS19/num1 \
    > ./logs/ablation08_maxpool_BRATS19_label25_num1_train.log 2>&1 &
pid1=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if [ "$status" -ne 0 ]; then exit 1; fi

# BRATS19 test: num1, labelnum=12, GPU=0; repeats retain original seed 1337.
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.brats19_test \
    --root_path /home/xuminghao/Datasets/BRATS19 \
    --gpu 0 \
    --exp CVBM_BRATS19_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 12 \
    --patch_size 96 96 96 \
    --stage_name self_train \
    --checkpoint_dir './results/cvbm_15_1_t2_a1_ablation_08/max_pool/BRATS19/num1/CVBM_BRATS19_A1_Ablation08_MaxPool/brats19/label12/proto_w0.1_patch(8, 8, 8)/self_train' \
    --test_save_path './results/cvbm_15_1_t2_a1_ablation_08/max_pool/BRATS19/num1/CVBM_BRATS19_A1_Ablation08_MaxPool/brats19/label12/proto_w0.1_patch(8, 8, 8)/self_train/CVBM_Argument_predictions' \
    --proto_patch 8 8 8 \
    --proto_weight 0.1 \
    --snapshot_path ./results/cvbm_15_1_t2_a1_ablation_08/max_pool/BRATS19/num1 \
    > ./logs/ablation08_maxpool_BRATS19_label12_num1_test.log 2>&1 &
pid0=$!

# BRATS19 test: num1, labelnum=25, GPU=1; repeats retain original seed 1337.
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.brats19_test \
    --root_path /home/xuminghao/Datasets/BRATS19 \
    --gpu 1 \
    --exp CVBM_BRATS19_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 25 \
    --patch_size 96 96 96 \
    --stage_name self_train \
    --checkpoint_dir './results/cvbm_15_1_t2_a1_ablation_08/max_pool/BRATS19/num1/CVBM_BRATS19_A1_Ablation08_MaxPool/brats19/label25/proto_w0.1_patch(8, 8, 8)/self_train' \
    --test_save_path './results/cvbm_15_1_t2_a1_ablation_08/max_pool/BRATS19/num1/CVBM_BRATS19_A1_Ablation08_MaxPool/brats19/label25/proto_w0.1_patch(8, 8, 8)/self_train/CVBM_Argument_predictions' \
    --proto_patch 8 8 8 \
    --proto_weight 0.1 \
    --snapshot_path ./results/cvbm_15_1_t2_a1_ablation_08/max_pool/BRATS19/num1 \
    > ./logs/ablation08_maxpool_BRATS19_label25_num1_test.log 2>&1 &
pid1=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if [ "$status" -ne 0 ]; then exit 1; fi

# BRATS19 train: num2, labelnum=12, GPU=0; repeats retain original seed 1337.
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.brats19_train \
    --root_path /home/xuminghao/Datasets/BRATS19 \
    --gpu 0 \
    --exp CVBM_BRATS19_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 12 \
    --patch_size 96 96 96 \
    --proto_pooling patch_max \
    --proto_patch 8 8 8 \
    --proto_weight 0.1 \
    --proto_dim 32 \
    --proto_temperature 0.2 \
    --proto_conf_threshold 0.8 \
    --proto_query_threshold 0.0 \
    --proto_max_queries 4096 \
    --deterministic 0 \
    --seed 1337 \
    --snapshot_path ./results/cvbm_15_1_t2_a1_ablation_08/max_pool/BRATS19/num2 \
    > ./logs/ablation08_maxpool_BRATS19_label12_num2_train.log 2>&1 &
pid0=$!

# BRATS19 train: num2, labelnum=25, GPU=1; repeats retain original seed 1337.
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.brats19_train \
    --root_path /home/xuminghao/Datasets/BRATS19 \
    --gpu 1 \
    --exp CVBM_BRATS19_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 25 \
    --patch_size 96 96 96 \
    --proto_pooling patch_max \
    --proto_patch 8 8 8 \
    --proto_weight 0.1 \
    --proto_dim 32 \
    --proto_temperature 0.2 \
    --proto_conf_threshold 0.8 \
    --proto_query_threshold 0.0 \
    --proto_max_queries 4096 \
    --deterministic 0 \
    --seed 1337 \
    --snapshot_path ./results/cvbm_15_1_t2_a1_ablation_08/max_pool/BRATS19/num2 \
    > ./logs/ablation08_maxpool_BRATS19_label25_num2_train.log 2>&1 &
pid1=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if [ "$status" -ne 0 ]; then exit 1; fi

# BRATS19 test: num2, labelnum=12, GPU=0; repeats retain original seed 1337.
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.brats19_test \
    --root_path /home/xuminghao/Datasets/BRATS19 \
    --gpu 0 \
    --exp CVBM_BRATS19_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 12 \
    --patch_size 96 96 96 \
    --stage_name self_train \
    --checkpoint_dir './results/cvbm_15_1_t2_a1_ablation_08/max_pool/BRATS19/num2/CVBM_BRATS19_A1_Ablation08_MaxPool/brats19/label12/proto_w0.1_patch(8, 8, 8)/self_train' \
    --test_save_path './results/cvbm_15_1_t2_a1_ablation_08/max_pool/BRATS19/num2/CVBM_BRATS19_A1_Ablation08_MaxPool/brats19/label12/proto_w0.1_patch(8, 8, 8)/self_train/CVBM_Argument_predictions' \
    --proto_patch 8 8 8 \
    --proto_weight 0.1 \
    --snapshot_path ./results/cvbm_15_1_t2_a1_ablation_08/max_pool/BRATS19/num2 \
    > ./logs/ablation08_maxpool_BRATS19_label12_num2_test.log 2>&1 &
pid0=$!

# BRATS19 test: num2, labelnum=25, GPU=1; repeats retain original seed 1337.
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.brats19_test \
    --root_path /home/xuminghao/Datasets/BRATS19 \
    --gpu 1 \
    --exp CVBM_BRATS19_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 25 \
    --patch_size 96 96 96 \
    --stage_name self_train \
    --checkpoint_dir './results/cvbm_15_1_t2_a1_ablation_08/max_pool/BRATS19/num2/CVBM_BRATS19_A1_Ablation08_MaxPool/brats19/label25/proto_w0.1_patch(8, 8, 8)/self_train' \
    --test_save_path './results/cvbm_15_1_t2_a1_ablation_08/max_pool/BRATS19/num2/CVBM_BRATS19_A1_Ablation08_MaxPool/brats19/label25/proto_w0.1_patch(8, 8, 8)/self_train/CVBM_Argument_predictions' \
    --proto_patch 8 8 8 \
    --proto_weight 0.1 \
    --snapshot_path ./results/cvbm_15_1_t2_a1_ablation_08/max_pool/BRATS19/num2 \
    > ./logs/ablation08_maxpool_BRATS19_label25_num2_test.log 2>&1 &
pid1=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if [ "$status" -ne 0 ]; then exit 1; fi

printf "DONE BRATS19 max pooling training and testing\n"
