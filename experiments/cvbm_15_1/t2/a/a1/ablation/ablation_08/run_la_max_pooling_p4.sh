#!/usr/bin/env bash
# LA CPCL max-pooling: 4 jobs at once (num1/num2 x 2 label ratios), 2 processes per GPU.
# gpu0 <- num1/label8 + num2/label16 ; gpu1 <- num1/label16 + num2/label8
set -uo pipefail
trap '' HUP
cd /home/xuminghao/Projects/CVBM/CVBM-ABD
export PATH="$HOME/anaconda3/envs/cvbm/bin:$PATH"
mkdir -p logs
R=./results/cvbm_15_1_t2_a1_ablation_08/max_pool/LA
ts() { date '+%F %T'; }
pids=()
fail=0
join() { for p in "${pids[@]}"; do wait "$p" || fail=1; done; }

echo "[$(ts)] ===== TRAIN 4 jobs (2 per GPU) ====="
echo "[$(ts)] TRAIN LA label8 num1 gpu0"
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
    --num_workers 4 \
    --snapshot_path $R/num1 \
    > ./logs/ablation08_maxpool_LA_label8_num1_train.log 2>&1 &
pids+=($!)
echo "[$(ts)] TRAIN LA label16 num2 gpu0"
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.la_train \
    --root_path /home/xuminghao/Datasets/LA/UA_MT \
    --gpu 0 \
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
    --num_workers 4 \
    --snapshot_path $R/num2 \
    > ./logs/ablation08_maxpool_LA_label16_num2_train.log 2>&1 &
pids+=($!)
echo "[$(ts)] TRAIN LA label16 num1 gpu1"
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
    --num_workers 4 \
    --snapshot_path $R/num1 \
    > ./logs/ablation08_maxpool_LA_label16_num1_train.log 2>&1 &
pids+=($!)
echo "[$(ts)] TRAIN LA label8 num2 gpu1"
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.la_train \
    --root_path /home/xuminghao/Datasets/LA/UA_MT \
    --gpu 1 \
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
    --num_workers 4 \
    --snapshot_path $R/num2 \
    > ./logs/ablation08_maxpool_LA_label8_num2_train.log 2>&1 &
pids+=($!)
join
if [ "$fail" -ne 0 ]; then echo "[$(ts)] TRAIN FAILED"; exit 1; fi

pids=(); fail=0
echo "[$(ts)] ===== TEST 4 jobs (2 per GPU) ====="
echo "[$(ts)] TEST LA label8 num1 gpu0"
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.la_test \
    --root_path /home/xuminghao/Datasets/LA/UA_MT \
    --gpu 0 \
    --exp CVBM_LA_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 8 \
    --patch_size 112 112 80 \
    --stage_name self_train \
    --checkpoint_dir "$R/num1/CVBM_LA_A1_Ablation08_MaxPool_8_labeled/self_train" \
    --test_save_path "$R/num1/CVBM_LA_A1_Ablation08_MaxPool_8_labeled/self_train/CVBM_Argument_predictions" \
    --proto_patch 8 8 8 \
    --proto_weight 0.1 \
    --snapshot_path $R/num1 \
    > ./logs/ablation08_maxpool_LA_label8_num1_test.log 2>&1 &
pids+=($!)
echo "[$(ts)] TEST LA label16 num2 gpu0"
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.la_test \
    --root_path /home/xuminghao/Datasets/LA/UA_MT \
    --gpu 0 \
    --exp CVBM_LA_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 16 \
    --patch_size 112 112 80 \
    --stage_name self_train \
    --checkpoint_dir "$R/num2/CVBM_LA_A1_Ablation08_MaxPool_16_labeled/self_train" \
    --test_save_path "$R/num2/CVBM_LA_A1_Ablation08_MaxPool_16_labeled/self_train/CVBM_Argument_predictions" \
    --proto_patch 8 8 8 \
    --proto_weight 0.1 \
    --snapshot_path $R/num2 \
    > ./logs/ablation08_maxpool_LA_label16_num2_test.log 2>&1 &
pids+=($!)
echo "[$(ts)] TEST LA label16 num1 gpu1"
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.la_test \
    --root_path /home/xuminghao/Datasets/LA/UA_MT \
    --gpu 1 \
    --exp CVBM_LA_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 16 \
    --patch_size 112 112 80 \
    --stage_name self_train \
    --checkpoint_dir "$R/num1/CVBM_LA_A1_Ablation08_MaxPool_16_labeled/self_train" \
    --test_save_path "$R/num1/CVBM_LA_A1_Ablation08_MaxPool_16_labeled/self_train/CVBM_Argument_predictions" \
    --proto_patch 8 8 8 \
    --proto_weight 0.1 \
    --snapshot_path $R/num1 \
    > ./logs/ablation08_maxpool_LA_label16_num1_test.log 2>&1 &
pids+=($!)
echo "[$(ts)] TEST LA label8 num2 gpu1"
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_08.la_test \
    --root_path /home/xuminghao/Datasets/LA/UA_MT \
    --gpu 1 \
    --exp CVBM_LA_A1_Ablation08_MaxPool \
    --model CVBM_Argument \
    --labelnum 8 \
    --patch_size 112 112 80 \
    --stage_name self_train \
    --checkpoint_dir "$R/num2/CVBM_LA_A1_Ablation08_MaxPool_8_labeled/self_train" \
    --test_save_path "$R/num2/CVBM_LA_A1_Ablation08_MaxPool_8_labeled/self_train/CVBM_Argument_predictions" \
    --proto_patch 8 8 8 \
    --proto_weight 0.1 \
    --snapshot_path $R/num2 \
    > ./logs/ablation08_maxpool_LA_label8_num2_test.log 2>&1 &
pids+=($!)
join
if [ "$fail" -ne 0 ]; then echo "[$(ts)] TEST FAILED"; exit 1; fi
printf "DONE LA max pooling 4-way\n"
