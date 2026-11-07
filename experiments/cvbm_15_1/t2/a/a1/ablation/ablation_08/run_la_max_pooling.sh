#!/usr/bin/env bash
set -euo pipefail
trap '' HUP
cd /home/xuminghao/Projects/CVBM/CVBM-ABD
export PATH="$HOME/anaconda3/envs/cvbm/bin:$PATH"
mkdir -p logs
# Do not run both dataset scripts concurrently: each uses GPU 0 and GPU 1.

# LA train: num1, labelnum=8, GPU=0; repeats retain original seed 1337.
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.la_train \
    --root_path /home/xuminghao/Datasets/LA/UA_MT \
    --gpu 0 \
    --exp CVBM_LA_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 8 \
    --patch_size 112 112 80 \
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
    --snapshot_path ./results/cvbm_15_1_t2_a1_ablation_08/max_pool/LA/num1 \
    > ./logs/ablation08_maxpool_LA_label8_num1_train.log 2>&1 &
pid0=$!

# LA train: num1, labelnum=16, GPU=1; repeats retain original seed 1337.
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.la_train \
    --root_path /home/xuminghao/Datasets/LA/UA_MT \
    --gpu 1 \
    --exp CVBM_LA_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 16 \
    --patch_size 112 112 80 \
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
    --snapshot_path ./results/cvbm_15_1_t2_a1_ablation_08/max_pool/LA/num1 \
    > ./logs/ablation08_maxpool_LA_label16_num1_train.log 2>&1 &
pid1=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if [ "$status" -ne 0 ]; then exit 1; fi

# LA test: num1, labelnum=8, GPU=0; repeats retain original seed 1337.
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.la_test \
    --root_path /home/xuminghao/Datasets/LA/UA_MT \
    --gpu 0 \
    --exp CVBM_LA_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 8 \
    --patch_size 112 112 80 \
    --stage_name self_train \
    --checkpoint_dir './results/cvbm_15_1_t2_a1_ablation_08/max_pool/LA/num1/CVBM_LA_A1_Ablation08_MaxPool_8_labeled/self_train' \
    --test_save_path './results/cvbm_15_1_t2_a1_ablation_08/max_pool/LA/num1/CVBM_LA_A1_Ablation08_MaxPool_8_labeled/self_train/CVBM_Argument_predictions' \
    --snapshot_path ./results/cvbm_15_1_t2_a1_ablation_08/max_pool/LA/num1 \
    > ./logs/ablation08_maxpool_LA_label8_num1_test.log 2>&1 &
pid0=$!

# LA test: num1, labelnum=16, GPU=1; repeats retain original seed 1337.
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.la_test \
    --root_path /home/xuminghao/Datasets/LA/UA_MT \
    --gpu 1 \
    --exp CVBM_LA_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 16 \
    --patch_size 112 112 80 \
    --stage_name self_train \
    --checkpoint_dir './results/cvbm_15_1_t2_a1_ablation_08/max_pool/LA/num1/CVBM_LA_A1_Ablation08_MaxPool_16_labeled/self_train' \
    --test_save_path './results/cvbm_15_1_t2_a1_ablation_08/max_pool/LA/num1/CVBM_LA_A1_Ablation08_MaxPool_16_labeled/self_train/CVBM_Argument_predictions' \
    --snapshot_path ./results/cvbm_15_1_t2_a1_ablation_08/max_pool/LA/num1 \
    > ./logs/ablation08_maxpool_LA_label16_num1_test.log 2>&1 &
pid1=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if [ "$status" -ne 0 ]; then exit 1; fi

# LA train: num2, labelnum=8, GPU=0; repeats retain original seed 1337.
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.la_train \
    --root_path /home/xuminghao/Datasets/LA/UA_MT \
    --gpu 0 \
    --exp CVBM_LA_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 8 \
    --patch_size 112 112 80 \
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
    --snapshot_path ./results/cvbm_15_1_t2_a1_ablation_08/max_pool/LA/num2 \
    > ./logs/ablation08_maxpool_LA_label8_num2_train.log 2>&1 &
pid0=$!

# LA train: num2, labelnum=16, GPU=1; repeats retain original seed 1337.
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.la_train \
    --root_path /home/xuminghao/Datasets/LA/UA_MT \
    --gpu 1 \
    --exp CVBM_LA_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 16 \
    --patch_size 112 112 80 \
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
    --snapshot_path ./results/cvbm_15_1_t2_a1_ablation_08/max_pool/LA/num2 \
    > ./logs/ablation08_maxpool_LA_label16_num2_train.log 2>&1 &
pid1=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if [ "$status" -ne 0 ]; then exit 1; fi

# LA test: num2, labelnum=8, GPU=0; repeats retain original seed 1337.
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.la_test \
    --root_path /home/xuminghao/Datasets/LA/UA_MT \
    --gpu 0 \
    --exp CVBM_LA_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 8 \
    --patch_size 112 112 80 \
    --stage_name self_train \
    --checkpoint_dir './results/cvbm_15_1_t2_a1_ablation_08/max_pool/LA/num2/CVBM_LA_A1_Ablation08_MaxPool_8_labeled/self_train' \
    --test_save_path './results/cvbm_15_1_t2_a1_ablation_08/max_pool/LA/num2/CVBM_LA_A1_Ablation08_MaxPool_8_labeled/self_train/CVBM_Argument_predictions' \
    --snapshot_path ./results/cvbm_15_1_t2_a1_ablation_08/max_pool/LA/num2 \
    > ./logs/ablation08_maxpool_LA_label8_num2_test.log 2>&1 &
pid0=$!

# LA test: num2, labelnum=16, GPU=1; repeats retain original seed 1337.
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.la_test \
    --root_path /home/xuminghao/Datasets/LA/UA_MT \
    --gpu 1 \
    --exp CVBM_LA_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 16 \
    --patch_size 112 112 80 \
    --stage_name self_train \
    --checkpoint_dir './results/cvbm_15_1_t2_a1_ablation_08/max_pool/LA/num2/CVBM_LA_A1_Ablation08_MaxPool_16_labeled/self_train' \
    --test_save_path './results/cvbm_15_1_t2_a1_ablation_08/max_pool/LA/num2/CVBM_LA_A1_Ablation08_MaxPool_16_labeled/self_train/CVBM_Argument_predictions' \
    --snapshot_path ./results/cvbm_15_1_t2_a1_ablation_08/max_pool/LA/num2 \
    > ./logs/ablation08_maxpool_LA_label16_num2_test.log 2>&1 &
pid1=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if [ "$status" -ne 0 ]; then exit 1; fi

printf "DONE LA max pooling training and testing\n"
