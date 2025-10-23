#!/bin/bash

trap "pkill -P $$; exit" SIGINT SIGTERM
wait

python -m just_try.brats19.brats19_train_4_6_3_aug \
    --root_path '/root/BRATS19' \
    --gpu 0 \
    --exp 'CVBM_BRATS19' \
    --model 'CVBM_Argument' \
    --labelnum 25 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_4_6_3/1" \
    > ./logs/output_brats19_train_4_6_3_aug_contrastive20,8,8,24_label10%_num1.log 2>&1 &

python -m just_try.brats19.brats19_train_4_6_3_aug \
    --root_path '/root/BRATS19' \
    --gpu 0 \
    --exp 'CVBM_BRATS19' \
    --model 'CVBM_Argument' \
    --labelnum 50 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_4_6_3/1" \
    > ./logs/output_brats19_train_4_6_3_aug_contrastive20,8,8,24_label20%_num1.log 2>&1 &

wait
# python -m just_try.brats19.brats19_train_4_6_3_aug \
#     --root_path '/root/BRATS19' \
#     --gpu 0 \
#     --exp 'CVBM_LA' \
#     --model 'CVBM_Argument' \
#     --labelnum 25 \
#     --deterministic 0 \
#     --snapshot_path "./results/CVBM_4_6_321" \
#     > ./logs/output_brats19_train_4_6_3_aug_contrastive20,8,8,24_label10%_num2.log 2>&1 &

# python -m just_try.brats19.brats19_train_4_6_3_aug \
#     --root_path '/root/BRATS19' \
#     --gpu 0 \
#     --exp 'CVBM_LA' \
#     --model 'CVBM_Argument' \
#     --labelnum 50 \
#     --deterministic 0 \
#     --snapshot_path "./results/CVBM_4_6_3/2" \
#     > ./logs/output_brats19_train_4_6_3_aug_contrastive20,8,8,24_label20%_num2.log 2>&1 &
# wait


python test_LA.py \
     --root_path '/root/BRATS19' \
     --labelnum 4 \
     --exp 'CVBM_LA' \
     --model 'CVBM_Argument' \
     --snapshot_path "./results/CVBM_4_6_3/1" > ./logs/output_test_brats19_train_4_6_3_aug_contrastive20,8,8,24_label10%_num1.log 2>&1 &

python test_LA.py \
     --root_path '/root/BRATS19' \
     --labelnum 8 \
     --exp 'CVBM_LA' \
     --model 'CVBM_Argument' \
     --snapshot_path "./results/CVBM_4_6_3/1" > ./logs/output_test_brats19_train_4_6_3_aug_contrastive20,8,8,24_label20%_num1.log 2>&1 &



$HOME/send_mail.sh -t xuhaolxy@gmail.com -m "BTATS19_train_4_6_3 label10% label20% num1 and num2 training finish in 43304 3090."



