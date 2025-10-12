#!/bin/bash

trap "pkill -P $$; exit" SIGINT SIGTERM
wait
# ACDC
python -m just_try.ACDC.ACDC_train_4_6_4_pre_train \
    --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
    --exp 'CVBM2d_ACDC' \
    --model 'CVBM2d_Argument' \
    --gpu 0 \
    --labelnum 3 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_4_6_4_pre_train/3" \
    > ./logs/output_ACDC_4_6_4_label3_num3.log 2>&1 &

python -m just_try.ACDC.ACDC_train_4_6_4_pre_train \
    --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
    --exp 'CVBM2d_ACDC' \
    --model 'CVBM2d_Argument' \
    --gpu 0 \
    --labelnum 3 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_4_6_4_pre_train/4" \
    > ./logs/output_ACDC_4_6_4_label3_num4.log 2>&1 &


python -m just_try.ACDC.ACDC_train_4_6_4_pre_train \
    --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
    --exp 'CVBM2d_ACDC' \
    --model 'CVBM2d_Argument' \
    --gpu 1 \
    --labelnum 7 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_4_6_4_pre_train/3" \
    > ./logs/output_ACDC_4_6_4_label7_num3.log 2>&1 &

python -m just_try.ACDC.ACDC_train_4_6_4_pre_train \
    --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
    --exp 'CVBM2d_ACDC' \
    --model 'CVBM2d_Argument' \
    --gpu 1 \
    --labelnum 7 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_4_6_4_pre_train/4" \
    > ./logs/output_ACDC_4_6_4_label7_num4.log 2>&1 &


wait
python test_ACDC.py \
    --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
    --labelnum 3 \
    --exp 'CVBM2d_ACDC' \
    --model 'CVBM2d_Argument' \
    --snapshot_path "./results/CVBM_4_6_4_pre_train/3" > ./logs/output_ACDC_4_6_4_label3_num3_test_result.log 2>&1 &

python test_ACDC.py \
    --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
    --labelnum 3 \
    --exp 'CVBM2d_ACDC' \
    --model 'CVBM2d_Argument' \
    --snapshot_path "./results/CVBM_4_6_4_pre_train/4" > ./logs/output_ACDC_4_6_4_label3_num4_test_result.log 2>&1 &

python test_ACDC.py \
    --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
    --labelnum 7 \
    --exp 'CVBM2d_ACDC' \
    --model 'CVBM2d_Argument' \
    --snapshot_path "./results/CVBM_4_6_4_pre_train/3" > ./logs/output_ACDC_4_6_4_label7_num3_test_result.log 2>&1 &

python test_ACDC.py \
    --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
    --labelnum 7 \
    --exp 'CVBM2d_ACDC' \
    --model 'CVBM2d_Argument' \
    --snapshot_path "./results/CVBM_4_6_4_pre_train/4" > ./logs/output_ACDC_4_6_4_label7_num4_test_result.log 2>&1 &
wait

$HOME/send_mail.sh -t xuhaolxy@gmail.com -m "ACDC_train_4_6_4_lowest label5% label10% num3 and num4 training finish in 43304 3090."






