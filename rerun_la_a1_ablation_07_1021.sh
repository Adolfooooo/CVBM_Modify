#!/bin/bash

trap "pkill -P $$; exit" SIGINT SIGTERM
wait

# LA a1 ablation 07-01: foreground strong, background weak
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_01.la_train \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 0 \
    --exp 'CVBM_LA_FgStrong_BgWeak_Ablation_07_01' \
    --model 'CVBM_Argument' \
    --labelnum 8 \
    --max_samples 80 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_01/1" \
    > ./logs/ablation_07/rerun1021_LA_train_cvbm_15_1_t2_a1_ablation_07_ab_07_01_label8_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_01.la_train \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 0 \
    --exp 'CVBM_LA_FgStrong_BgWeak_Ablation_07_01' \
    --model 'CVBM_Argument' \
    --labelnum 8 \
    --max_samples 80 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_01/2" \
    > ./logs/ablation_07/rerun1021_LA_train_cvbm_15_1_t2_a1_ablation_07_ab_07_01_label8_num2.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_01.la_train \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 1 \
    --exp 'CVBM_LA_FgStrong_BgWeak_Ablation_07_01' \
    --model 'CVBM_Argument' \
    --labelnum 16 \
    --max_samples 80 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_01/1" \
    > ./logs/ablation_07/rerun1021_LA_train_cvbm_15_1_t2_a1_ablation_07_ab_07_01_label16_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_01.la_train \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 1 \
    --exp 'CVBM_LA_FgStrong_BgWeak_Ablation_07_01' \
    --model 'CVBM_Argument' \
    --labelnum 16 \
    --max_samples 80 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_01/2" \
    > ./logs/ablation_07/rerun1021_LA_train_cvbm_15_1_t2_a1_ablation_07_ab_07_01_label16_num2.log 2>&1 &
wait

python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_01.la_test \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 0 \
    --labelnum 8 \
    --max_samples 80 \
    --exp 'CVBM_LA_FgStrong_BgWeak_Ablation_07_01' \
    --model 'CVBM_Argument' \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_01/1" \
    > ./logs/ablation_07/rerun1021_test_LA_cvbm_15_1_t2_a1_ablation_07_ab_07_01_label8_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_01.la_test \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 0 \
    --labelnum 8 \
    --max_samples 80 \
    --exp 'CVBM_LA_FgStrong_BgWeak_Ablation_07_01' \
    --model 'CVBM_Argument' \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_01/2" \
    > ./logs/ablation_07/rerun1021_test_LA_cvbm_15_1_t2_a1_ablation_07_ab_07_01_label8_num2.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_01.la_test \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 1 \
    --labelnum 16 \
    --max_samples 80 \
    --exp 'CVBM_LA_FgStrong_BgWeak_Ablation_07_01' \
    --model 'CVBM_Argument' \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_01/1" \
    > ./logs/ablation_07/rerun1021_test_LA_cvbm_15_1_t2_a1_ablation_07_ab_07_01_label16_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_01.la_test \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 1 \
    --labelnum 16 \
    --max_samples 80 \
    --exp 'CVBM_LA_FgStrong_BgWeak_Ablation_07_01' \
    --model 'CVBM_Argument' \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_01/2" \
    > ./logs/ablation_07/rerun1021_test_LA_cvbm_15_1_t2_a1_ablation_07_ab_07_01_label16_num2.log 2>&1 &
wait

# LA a1 ablation 07-02: foreground weak, background weak
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_02.la_train \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 0 \
    --exp 'CVBM_LA_FgWeak_BgWeak_Ablation_07_02' \
    --model 'CVBM_Argument' \
    --labelnum 8 \
    --max_samples 80 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_02/1" \
    > ./logs/ablation_07/rerun1021_LA_train_cvbm_15_1_t2_a1_ablation_07_ab_07_02_label8_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_02.la_train \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 0 \
    --exp 'CVBM_LA_FgWeak_BgWeak_Ablation_07_02' \
    --model 'CVBM_Argument' \
    --labelnum 8 \
    --max_samples 80 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_02/2" \
    > ./logs/ablation_07/rerun1021_LA_train_cvbm_15_1_t2_a1_ablation_07_ab_07_02_label8_num2.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_02.la_train \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 1 \
    --exp 'CVBM_LA_FgWeak_BgWeak_Ablation_07_02' \
    --model 'CVBM_Argument' \
    --labelnum 16 \
    --max_samples 80 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_02/1" \
    > ./logs/ablation_07/rerun1021_LA_train_cvbm_15_1_t2_a1_ablation_07_ab_07_02_label16_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_02.la_train \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 1 \
    --exp 'CVBM_LA_FgWeak_BgWeak_Ablation_07_02' \
    --model 'CVBM_Argument' \
    --labelnum 16 \
    --max_samples 80 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_02/2" \
    > ./logs/ablation_07/rerun1021_LA_train_cvbm_15_1_t2_a1_ablation_07_ab_07_02_label16_num2.log 2>&1 &
wait

python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_02.la_test \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 0 \
    --labelnum 8 \
    --max_samples 80 \
    --exp 'CVBM_LA_FgWeak_BgWeak_Ablation_07_02' \
    --model 'CVBM_Argument' \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_02/1" \
    > ./logs/ablation_07/rerun1021_test_LA_cvbm_15_1_t2_a1_ablation_07_ab_07_02_label8_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_02.la_test \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 0 \
    --labelnum 8 \
    --max_samples 80 \
    --exp 'CVBM_LA_FgWeak_BgWeak_Ablation_07_02' \
    --model 'CVBM_Argument' \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_02/2" \
    > ./logs/ablation_07/rerun1021_test_LA_cvbm_15_1_t2_a1_ablation_07_ab_07_02_label8_num2.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_02.la_test \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 1 \
    --labelnum 16 \
    --max_samples 80 \
    --exp 'CVBM_LA_FgWeak_BgWeak_Ablation_07_02' \
    --model 'CVBM_Argument' \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_02/1" \
    > ./logs/ablation_07/rerun1021_test_LA_cvbm_15_1_t2_a1_ablation_07_ab_07_02_label16_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_02.la_test \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 1 \
    --labelnum 16 \
    --max_samples 80 \
    --exp 'CVBM_LA_FgWeak_BgWeak_Ablation_07_02' \
    --model 'CVBM_Argument' \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_02/2" \
    > ./logs/ablation_07/rerun1021_test_LA_cvbm_15_1_t2_a1_ablation_07_ab_07_02_label16_num2.log 2>&1 &
wait

# LA a1 ablation 07-03: foreground strong, background strong
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_03.la_train \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 0 \
    --exp 'CVBM_LA_FgStrong_BgStrong_Ablation_07_03' \
    --model 'CVBM_Argument' \
    --labelnum 8 \
    --max_samples 80 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_03/1" \
    > ./logs/ablation_07/rerun1021_LA_train_cvbm_15_1_t2_a1_ablation_07_ab_07_03_label8_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_03.la_train \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 0 \
    --exp 'CVBM_LA_FgStrong_BgStrong_Ablation_07_03' \
    --model 'CVBM_Argument' \
    --labelnum 8 \
    --max_samples 80 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_03/2" \
    > ./logs/ablation_07/rerun1021_LA_train_cvbm_15_1_t2_a1_ablation_07_ab_07_03_label8_num2.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_03.la_train \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 1 \
    --exp 'CVBM_LA_FgStrong_BgStrong_Ablation_07_03' \
    --model 'CVBM_Argument' \
    --labelnum 16 \
    --max_samples 80 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_03/1" \
    > ./logs/ablation_07/rerun1021_LA_train_cvbm_15_1_t2_a1_ablation_07_ab_07_03_label16_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_03.la_train \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 1 \
    --exp 'CVBM_LA_FgStrong_BgStrong_Ablation_07_03' \
    --model 'CVBM_Argument' \
    --labelnum 16 \
    --max_samples 80 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_03/2" \
    > ./logs/ablation_07/rerun1021_LA_train_cvbm_15_1_t2_a1_ablation_07_ab_07_03_label16_num2.log 2>&1 &
wait

python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_03.la_test \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 0 \
    --labelnum 8 \
    --max_samples 80 \
    --exp 'CVBM_LA_FgStrong_BgStrong_Ablation_07_03' \
    --model 'CVBM_Argument' \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_03/1" \
    > ./logs/ablation_07/rerun1021_test_LA_cvbm_15_1_t2_a1_ablation_07_ab_07_03_label8_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_03.la_test \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 0 \
    --labelnum 8 \
    --max_samples 80 \
    --exp 'CVBM_LA_FgStrong_BgStrong_Ablation_07_03' \
    --model 'CVBM_Argument' \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_03/2" \
    > ./logs/ablation_07/rerun1021_test_LA_cvbm_15_1_t2_a1_ablation_07_ab_07_03_label8_num2.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_03.la_test \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 1 \
    --labelnum 16 \
    --max_samples 80 \
    --exp 'CVBM_LA_FgStrong_BgStrong_Ablation_07_03' \
    --model 'CVBM_Argument' \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_03/1" \
    > ./logs/ablation_07/rerun1021_test_LA_cvbm_15_1_t2_a1_ablation_07_ab_07_03_label16_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_03.la_test \
    --root_path '/home/xuminghao/Datasets/LA/UA_MT' \
    --gpu 1 \
    --labelnum 16 \
    --max_samples 80 \
    --exp 'CVBM_LA_FgStrong_BgStrong_Ablation_07_03' \
    --model 'CVBM_Argument' \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_03/2" \
    > ./logs/ablation_07/rerun1021_test_LA_cvbm_15_1_t2_a1_ablation_07_ab_07_03_label16_num2.log 2>&1 &
wait

