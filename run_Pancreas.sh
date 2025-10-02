#!/bin/bash

trap "pkill -P $$; exit" SIGINT SIGTERM
wait

# Pancreas
python -m just_try.Pancreas.Pancreas_train_4_2_2 \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas' \
    --model 'CVBM_Argument' \
    --gpu 0 \
    --labelnum 6 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_4_2_2/1" \
    > ./logs/output_train_Pancreas_4_2_2_label6_num1.log 2>&1 &
wait
python -m just_try.Pancreas.Pancreas_train_4_2_2 \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --exp 'CVBM_Pancreas' \
    --model 'CVBM_Argument' \
    --gpu 0 \
    --labelnum 12 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_4_2_2/1" \
    > ./logs/output_train_Pancreas_4_2_2_label12_num1.log 2>&1 &
wait
python test_Pancreas.py \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --labelnum 6 \
    --exp 'CVBM_Pancreas' \
    --model 'CVBM_Argument' \
    --snapshot_path "./results/CVBM_4_2_2/1" > ./logs/output_test_Pancreas_4_2_2_label6_num1.log 2>&1 &
python test_Pancreas.py \
    --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
    --labelnum 12 \
    --exp 'CVBM_Pancreas' \
    --model 'CVBM_Argument' \
    --snapshot_path "./results/CVBM_4_2_2/1" > ./logs/output_test_Pancreas_4_2_2_label12_num1.log 2>&1 &
wait


$HOME/send_mail.sh -t xuhaolxy@qq.com -m "Pancreas_4_2_2 label6 and label12 training finish in 43304 4080."
