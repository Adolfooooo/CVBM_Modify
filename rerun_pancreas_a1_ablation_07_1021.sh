#!/bin/bash

trap "pkill -P $$; exit" SIGINT SIGTERM
wait

# Pancreas a1 ablation 07-01: foreground strong, background weak
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_01.pancreas_train \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgStrong_BgWeak_Ablation_07_01' \
    --model 'CVBM_Argument' \
    --gpu 0 \
    --labelnum 6 \
    --deterministic 1 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_01/1" \
    > ./logs/ablation_07/rerun1021_train_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_01_label6_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_01.pancreas_train \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgStrong_BgWeak_Ablation_07_01' \
    --model 'CVBM_Argument' \
    --gpu 1 \
    --labelnum 6 \
    --deterministic 1 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_01/2" \
    > ./logs/ablation_07/rerun1021_train_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_01_label6_num2.log 2>&1 &
wait
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_01.pancreas_train \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgStrong_BgWeak_Ablation_07_01' \
    --model 'CVBM_Argument' \
    --gpu 0 \
    --labelnum 12 \
    --deterministic 1 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_01/1" \
    > ./logs/ablation_07/rerun1021_train_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_01_label12_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_01.pancreas_train \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgStrong_BgWeak_Ablation_07_01' \
    --model 'CVBM_Argument' \
    --gpu 1 \
    --labelnum 12 \
    --deterministic 1 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_01/2" \
    > ./logs/ablation_07/rerun1021_train_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_01_label12_num2.log 2>&1 &
wait

python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_01.pancreas_test \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgStrong_BgWeak_Ablation_07_01' \
    --model 'CVBM_Argument' \
    --gpu 0 \
    --labelnum 6 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_01/1" \
    > ./logs/ablation_07/rerun1021_test_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_01_label6_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_01.pancreas_test \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgStrong_BgWeak_Ablation_07_01' \
    --model 'CVBM_Argument' \
    --gpu 1 \
    --labelnum 6 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_01/2" \
    > ./logs/ablation_07/rerun1021_test_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_01_label6_num2.log 2>&1 &
wait
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_01.pancreas_test \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgStrong_BgWeak_Ablation_07_01' \
    --model 'CVBM_Argument' \
    --gpu 0 \
    --labelnum 12 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_01/1" \
    > ./logs/ablation_07/rerun1021_test_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_01_label12_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_01.pancreas_test \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgStrong_BgWeak_Ablation_07_01' \
    --model 'CVBM_Argument' \
    --gpu 1 \
    --labelnum 12 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_01/2" \
    > ./logs/ablation_07/rerun1021_test_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_01_label12_num2.log 2>&1 &
wait

# Pancreas a1 ablation 07-02: foreground weak, background weak
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_02.pancreas_train \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgWeak_BgWeak_Ablation_07_02' \
    --model 'CVBM_Argument' \
    --gpu 0 \
    --labelnum 6 \
    --deterministic 1 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_02/1" \
    > ./logs/ablation_07/rerun1021_train_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_02_label6_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_02.pancreas_train \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgWeak_BgWeak_Ablation_07_02' \
    --model 'CVBM_Argument' \
    --gpu 1 \
    --labelnum 6 \
    --deterministic 1 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_02/2" \
    > ./logs/ablation_07/rerun1021_train_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_02_label6_num2.log 2>&1 &
wait
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_02.pancreas_train \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgWeak_BgWeak_Ablation_07_02' \
    --model 'CVBM_Argument' \
    --gpu 0 \
    --labelnum 12 \
    --deterministic 1 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_02/1" \
    > ./logs/ablation_07/rerun1021_train_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_02_label12_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_02.pancreas_train \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgWeak_BgWeak_Ablation_07_02' \
    --model 'CVBM_Argument' \
    --gpu 1 \
    --labelnum 12 \
    --deterministic 1 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_02/2" \
    > ./logs/ablation_07/rerun1021_train_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_02_label12_num2.log 2>&1 &
wait

python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_02.pancreas_test \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgWeak_BgWeak_Ablation_07_02' \
    --model 'CVBM_Argument' \
    --gpu 0 \
    --labelnum 6 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_02/1" \
    > ./logs/ablation_07/rerun1021_test_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_02_label6_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_02.pancreas_test \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgWeak_BgWeak_Ablation_07_02' \
    --model 'CVBM_Argument' \
    --gpu 1 \
    --labelnum 6 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_02/2" \
    > ./logs/ablation_07/rerun1021_test_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_02_label6_num2.log 2>&1 &
wait
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_02.pancreas_test \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgWeak_BgWeak_Ablation_07_02' \
    --model 'CVBM_Argument' \
    --gpu 0 \
    --labelnum 12 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_02/1" \
    > ./logs/ablation_07/rerun1021_test_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_02_label12_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_02.pancreas_test \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgWeak_BgWeak_Ablation_07_02' \
    --model 'CVBM_Argument' \
    --gpu 1 \
    --labelnum 12 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_02/2" \
    > ./logs/ablation_07/rerun1021_test_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_02_label12_num2.log 2>&1 &
wait

# Pancreas a1 ablation 07-03: foreground strong, background strong
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_03.pancreas_train \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgStrong_BgStrong_Ablation_07_03' \
    --model 'CVBM_Argument' \
    --gpu 0 \
    --labelnum 6 \
    --deterministic 1 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_03/1" \
    > ./logs/ablation_07/rerun1021_train_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_03_label6_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_03.pancreas_train \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgStrong_BgStrong_Ablation_07_03' \
    --model 'CVBM_Argument' \
    --gpu 1 \
    --labelnum 6 \
    --deterministic 1 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_03/2" \
    > ./logs/ablation_07/rerun1021_train_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_03_label6_num2.log 2>&1 &
wait
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_03.pancreas_train \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgStrong_BgStrong_Ablation_07_03' \
    --model 'CVBM_Argument' \
    --gpu 0 \
    --labelnum 12 \
    --deterministic 1 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_03/1" \
    > ./logs/ablation_07/rerun1021_train_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_03_label12_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_03.pancreas_train \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgStrong_BgStrong_Ablation_07_03' \
    --model 'CVBM_Argument' \
    --gpu 1 \
    --labelnum 12 \
    --deterministic 1 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_03/2" \
    > ./logs/ablation_07/rerun1021_train_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_03_label12_num2.log 2>&1 &
wait

python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_03.pancreas_test \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgStrong_BgStrong_Ablation_07_03' \
    --model 'CVBM_Argument' \
    --gpu 0 \
    --labelnum 6 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_03/1" \
    > ./logs/ablation_07/rerun1021_test_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_03_label6_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_03.pancreas_test \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgStrong_BgStrong_Ablation_07_03' \
    --model 'CVBM_Argument' \
    --gpu 1 \
    --labelnum 6 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_03/2" \
    > ./logs/ablation_07/rerun1021_test_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_03_label6_num2.log 2>&1 &
wait
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_03.pancreas_test \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgStrong_BgStrong_Ablation_07_03' \
    --model 'CVBM_Argument' \
    --gpu 0 \
    --labelnum 12 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_03/1" \
    > ./logs/ablation_07/rerun1021_test_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_03_label12_num1.log 2>&1 &
python -m experiments.cvbm_15_1.t2.a.a1.ablation.ablation_07.ab_07_03.pancreas_test \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas_FgStrong_BgStrong_Ablation_07_03' \
    --model 'CVBM_Argument' \
    --gpu 1 \
    --labelnum 12 \
    --snapshot_path "./results/CVBM_15_1_t2_a/a1/ablation_07_rerun_1021/ab_07_03/2" \
    > ./logs/ablation_07/rerun1021_test_pancreas_cvbm_15_1_t2_a1_ablation_07_ab_07_03_label12_num2.log 2>&1 &
wait

