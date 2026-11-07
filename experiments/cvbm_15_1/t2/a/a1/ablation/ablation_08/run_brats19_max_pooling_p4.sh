#!/usr/bin/env bash
# BRATS19 CPCL max-pooling: 4 jobs at once (num1/num2 x 2 label ratios), 2 processes per GPU.
# gpu0 <- num1/label12 + num2/label25 ; gpu1 <- num1/label25 + num2/label12
set -uo pipefail
trap '' HUP
cd /home/xuminghao/Projects/CVBM/CVBM-ABD
export PATH="$HOME/anaconda3/envs/cvbm/bin:$PATH"
mkdir -p logs
R=./results/cvbm_15_1_t2_a1_ablation_08/max_pool/BRATS19
ts() { date '+%F %T'; }
pids=()
fail=0
join() { for p in "${pids[@]}"; do wait "$p" || fail=1; done; }

echo "[$(ts)] ===== TRAIN 4 jobs (2 per GPU) ====="
echo "[$(ts)] TRAIN BRATS19 label12 num1 gpu0"
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
    --num_workers 4 \
    --snapshot_path $R/num1 \
    > ./logs/ablation08_maxpool_BRATS19_label12_num1_train.log 2>&1 &
pids+=($!)
echo "[$(ts)] TRAIN BRATS19 label25 num2 gpu0"
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.brats19_train \
    --root_path /home/xuminghao/Datasets/BRATS19 \
    --gpu 0 \
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
    --num_workers 4 \
    --snapshot_path $R/num2 \
    > ./logs/ablation08_maxpool_BRATS19_label25_num2_train.log 2>&1 &
pids+=($!)
echo "[$(ts)] TRAIN BRATS19 label25 num1 gpu1"
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
    --num_workers 4 \
    --snapshot_path $R/num1 \
    > ./logs/ablation08_maxpool_BRATS19_label25_num1_train.log 2>&1 &
pids+=($!)
echo "[$(ts)] TRAIN BRATS19 label12 num2 gpu1"
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.brats19_train \
    --root_path /home/xuminghao/Datasets/BRATS19 \
    --gpu 1 \
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
    --num_workers 4 \
    --snapshot_path $R/num2 \
    > ./logs/ablation08_maxpool_BRATS19_label12_num2_train.log 2>&1 &
pids+=($!)
join
if [ "$fail" -ne 0 ]; then echo "[$(ts)] TRAIN FAILED"; exit 1; fi

pids=(); fail=0
echo "[$(ts)] ===== TEST 4 jobs (2 per GPU) ====="
echo "[$(ts)] TEST BRATS19 label12 num1 gpu0"
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.brats19_test \
    --root_path /home/xuminghao/Datasets/BRATS19 \
    --gpu 0 \
    --exp CVBM_BRATS19_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 12 \
    --patch_size 96 96 96 \
    --stage_name self_train \
    --checkpoint_dir "$R/num1/CVBM_BRATS19_A1_Ablation08_MaxPool/brats19/label12/proto_w0.1_patch(8, 8, 8)/self_train" \
    --test_save_path "$R/num1/CVBM_BRATS19_A1_Ablation08_MaxPool/brats19/label12/proto_w0.1_patch(8, 8, 8)/self_train/CVBM_Argument_predictions" \
    --proto_patch 8 8 8 \
    --proto_weight 0.1 \
    --snapshot_path $R/num1 \
    > ./logs/ablation08_maxpool_BRATS19_label12_num1_test.log 2>&1 &
pids+=($!)
echo "[$(ts)] TEST BRATS19 label25 num2 gpu0"
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.brats19_test \
    --root_path /home/xuminghao/Datasets/BRATS19 \
    --gpu 0 \
    --exp CVBM_BRATS19_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 25 \
    --patch_size 96 96 96 \
    --stage_name self_train \
    --checkpoint_dir "$R/num2/CVBM_BRATS19_A1_Ablation08_MaxPool/brats19/label25/proto_w0.1_patch(8, 8, 8)/self_train" \
    --test_save_path "$R/num2/CVBM_BRATS19_A1_Ablation08_MaxPool/brats19/label25/proto_w0.1_patch(8, 8, 8)/self_train/CVBM_Argument_predictions" \
    --proto_patch 8 8 8 \
    --proto_weight 0.1 \
    --snapshot_path $R/num2 \
    > ./logs/ablation08_maxpool_BRATS19_label25_num2_test.log 2>&1 &
pids+=($!)
echo "[$(ts)] TEST BRATS19 label25 num1 gpu1"
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.brats19_test \
    --root_path /home/xuminghao/Datasets/BRATS19 \
    --gpu 1 \
    --exp CVBM_BRATS19_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 25 \
    --patch_size 96 96 96 \
    --stage_name self_train \
    --checkpoint_dir "$R/num1/CVBM_BRATS19_A1_Ablation08_MaxPool/brats19/label25/proto_w0.1_patch(8, 8, 8)/self_train" \
    --test_save_path "$R/num1/CVBM_BRATS19_A1_Ablation08_MaxPool/brats19/label25/proto_w0.1_patch(8, 8, 8)/self_train/CVBM_Argument_predictions" \
    --proto_patch 8 8 8 \
    --proto_weight 0.1 \
    --snapshot_path $R/num1 \
    > ./logs/ablation08_maxpool_BRATS19_label25_num1_test.log 2>&1 &
pids+=($!)
echo "[$(ts)] TEST BRATS19 label12 num2 gpu1"
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.brats19_test \
    --root_path /home/xuminghao/Datasets/BRATS19 \
    --gpu 1 \
    --exp CVBM_BRATS19_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 12 \
    --patch_size 96 96 96 \
    --stage_name self_train \
    --checkpoint_dir "$R/num2/CVBM_BRATS19_A1_Ablation08_MaxPool/brats19/label12/proto_w0.1_patch(8, 8, 8)/self_train" \
    --test_save_path "$R/num2/CVBM_BRATS19_A1_Ablation08_MaxPool/brats19/label12/proto_w0.1_patch(8, 8, 8)/self_train/CVBM_Argument_predictions" \
    --proto_patch 8 8 8 \
    --proto_weight 0.1 \
    --snapshot_path $R/num2 \
    > ./logs/ablation08_maxpool_BRATS19_label12_num2_test.log 2>&1 &
pids+=($!)
join
if [ "$fail" -ne 0 ]; then echo "[$(ts)] TEST FAILED"; exit 1; fi
printf "DONE BRATS19 max pooling 4-way\n"
