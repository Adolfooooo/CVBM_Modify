#!/bin/bash

trap "pkill -P $$; exit" SIGINT SIGTERM
wait
# ACDC
python -m just_try.ACDC.ACDC_train_4_2_2_pre_train \
    --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
    --exp 'CVBM2d_ACDC' \
    --model 'CVBM2d_Argument' \
    --gpu 0 \
    --labelnum 3 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_4_2_2/1" \
    > ./logs/output_ACDC_4_2_2_label3_num1.log 2>&1 &

python -m just_try.ACDC.ACDC_train_4_2_2_pre_train \
    --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
    --exp 'CVBM2d_ACDC' \
    --model 'CVBM2d_Argument' \
    --gpu 0 \
    --labelnum 3 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_4_2_2/2" \
    > ./logs/output_ACDC_4_2_2_label3_num2.log 2>&1 &


python -m just_try.ACDC.ACDC_train_4_2_2_pre_train \
    --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
    --exp 'CVBM2d_ACDC' \
    --model 'CVBM2d_Argument' \
    --gpu 1 \
    --labelnum 7 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_4_2_2/1" \
    > ./logs/output_ACDC_4_2_2_label7_num1.log 2>&1 &

python -m just_try.ACDC.ACDC_train_4_2_2_pre_train \
    --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
    --exp 'CVBM2d_ACDC' \
    --model 'CVBM2d_Argument' \
    --gpu 1 \
    --labelnum 7 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_4_2_2/2" \
    > ./logs/output_ACDC_4_2_2_label7_num2.log 2>&1 &


wait
python test_ACDC.py \
    --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
    --labelnum 3 \
    --exp 'CVBM2d_ACDC' \
    --model 'CVBM2d_Argument' \
    --snapshot_path "./results/CVBM_4_2_2/1" > ./logs/output_ACDC_4_2_2_label3_num1_test_result.log 2>&1 &

python test_ACDC.py \
    --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
    --labelnum 3 \
    --exp 'CVBM2d_ACDC' \
    --model 'CVBM2d_Argument' \
    --snapshot_path "./results/CVBM_4_2_2/2" > ./logs/output_ACDC_4_2_2_label3_num2_test_result.log 2>&1 &

python test_ACDC.py \
    --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
    --labelnum 7 \
    --exp 'CVBM2d_ACDC' \
    --model 'CVBM2d_Argument' \
    --snapshot_path "./results/CVBM_4_2_2/1" > ./logs/output_ACDC_4_2_2_label7_num1_test_result.log 2>&1 &

python test_ACDC.py \
    --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
    --labelnum 7 \
    --exp 'CVBM2d_ACDC' \
    --model 'CVBM2d_Argument' \
    --snapshot_path "./results/CVBM_4_2_2/2" > ./logs/output_ACDC_4_2_2_label7_num2_test_result.log 2>&1 &
wait

$HOME/send_mail.sh -t xuhaolxy@gmail.com -m "LA_train_4_2_2_lowest label10% num1 and num2 training finish in 43304 4090."






