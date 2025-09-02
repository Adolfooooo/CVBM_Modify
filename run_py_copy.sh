#!/bin/bash




trap "pkill -P $$; exit" SIGINT SIGTERM
train_num=1
train_id="4_2"
wait


python -m just_try.ACDC.ACDC_ablation_train_twice \
    --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
    --exp 'CVBM2d_ACDC' \
    --model 'CVBM2d' \
    --gpu 1 \
    --labelnum 3 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_ablation/1" \
    > output_ACDC_ablation_twice_label3_num1.log 2>&1 &
python -m just_try.ACDC.ACDC_ablation_train_twice \
    --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
    --exp 'CVBM2d_ACDC' \
    --model 'CVBM2d' \
    --gpu 1 \
    --labelnum 7 \
    --deterministic 0 \
    --snapshot_path "./results/CVBM_ablation/1" \
    > output_ACDC_ablation_twice_label7_num1.log 2>&1 &
wait
python test_ACDC.py \
    --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
    --labelnum 3 \
    --exp 'CVBM2d_ACDC' \
    --model 'CVBM2d' \
    --snapshot_path "./results/CVBM_ablation/1" > output_ACDC_ablation_twice_label3_num1_test_result.log 2>&1 &
python test_ACDC.py \
    --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
    --labelnum 7 \
    --exp 'CVBM2d_ACDC' \
    --model 'CVBM2d' \
    --snapshot_path "./results/CVBM_ablation/1" > output_ACDC_ablation_twice_label7_num1_test_result.log 2>&1 &
wait
$HOME/send_mail.sh -t 2708964632@qq.com -m "ACDC ablation_twice label3 and label7 num1 training finish in 43304 3090."







# # ACDC
# python -m just_try.ACDC.ACDC_train_4_3 \
#     --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
#     --exp 'CVBM2d_ACDC' \
#     --model 'CVBM2d_Argument' \
#     --gpu 0 \
#     --labelnum 3 \
#     --deterministic 0 \
#     --snapshot_path "./results/CVBM_4_3/1" \
#     > output_ACDC_4_3_label3_num1.log 2>&1 &
# python -m just_try.ACDC.ACDC_train_4_3 \
#     --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
#     --exp 'CVBM2d_ACDC' \
#     --model 'CVBM2d_Argument' \
#     --gpu 0 \
#     --labelnum 7 \
#     --deterministic 0 \
#     --snapshot_path "./results/CVBM_4_3/1" \
#     > output_ACDC_4_3_label7_num1.log 2>&1 &
# wait
# python test_ACDC.py \
#     --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
#     --labelnum 3 \
#     --exp 'CVBM2d_ACDC' \
#     --model 'CVBM2d_Argument' \
#     --snapshot_path "./results/CVBM_4_3/1" > output_ACDC_4_3_label3_num1_test_result.log 2>&1 &
# python test_ACDC.py \
#     --root_path '/home/xuminghao/Datasets/ACDC/ACDC_ABD' \
#     --labelnum 7 \
#     --exp 'CVBM2d_ACDC' \
#     --model 'CVBM2d_Argument' \
#     --snapshot_path "./results/CVBM_4_3/1" > output_ACDC_4_3_label7_num1_test_result.log 2>&1 &
# wait


# LA
# python -m just_try.LA.LA_train_${train_id} --labelnum 4 --snapshot_path "./results/CVBM_${train_id}/${train_num}" > output_LA_${train_id}_num${train_num}_label4.log 2>&1 &
# python -m just_try.LA.LA_train_${train_id} --labelnum 8 --snapshot_path "./results/CVBM_${train_id}/${train_num}" > output_LA_${train_id}_num${train_num}_label8.log 2>&1 &
# python -m just_try.LA.LA_train_3_1 \
#     --root_path '/home/xuminghao/Datasets/LA/LA_UA-MT_Version' \
#     --exp 'CVBM_LA' \
#     --model 'CVBM' \
#     --gpu 0 \
#     --labelnum 6 \
#     --deterministic 1 \
#     --snapshot_path "./results/CVBM_3_1/1" \
#     > output_LA_3_1_label6.log 2>&1 &
# wait
# python test_LA.py --snapshot_path "./results/CVBM/${train_id}/${train_num}" --labelnum 4 > output_LA_${train_id}_num${train_num}_label4_test_result.log 2>&1 &
# python test_LA.py --snapshot_path "./results/CVBM/${train_id}/${train_num}" --labelnum 8 > output_LA_${train_id}_num${train_num}_label8_test_result.log 2>&1 &

# python -m just_try.LA.LA_train_4_2 \
#     --root_path '/home/xuminghao/Datasets/LA/LA_UA-MT_Version' \
#     --exp 'CVBM_LA' \
#     --model 'CVBM_Argument' \
#     --gpu 0 \
#     --labelnum 4 \
#     --deterministic 0 \
#     --snapshot_path "./results/CVBM_LA/1" \
#     > output_train_LA_4_2_label4_num1.log 2>&1 &
# python -m just_try.LA.LA_train_4_2 \
#     --root_path '/home/xuminghao/Datasets/LA/LA_UA-MT_Version' \
#     --exp 'CVBM_LA' \
#     --model 'CVBM_Argument' \
#     --gpu 0 \
#     --labelnum 8 \
#     --deterministic 0 \
#     --snapshot_path "./results/CVBM_LA/1" \
#     > output_train_LA_4_2_label8_num1.log 2>&1 &
# wait
# python test_LA.py \
#     --root_path '/home/xuminghao/Datasets/LA/LA_UA-MT_Version' \
#     --labelnum 4 \
#     --exp 'CVBM_LA' \
#     --model 'CVBM_Argument' \
#     --snapshot_path "./results/CVBM_LA/1" > output_test_LA_4_2_label4_num1.log 2>&1 &
# python test_LA.py \
#     --root_path '/home/xuminghao/Datasets/LA/LA_UA-MT_Version' \
#     --labelnum 8 \
#     --exp 'CVBM_LA' \
#     --model 'CVBM_Argument' \
#     --snapshot_path "./results/CVBM_LA/1" > output_test_LA_4_2_label8_num1.log 2>&1 &
# wait
# wait


# # Pancreas
# python Pancreas_train.py \
#     --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
#     --exp 'CVBM_Pancreas' \
#     --model 'CVBM' \
#     --gpu 0 \
#     --labelnum 6 \
#     --deterministic 0 \
#     --snapshot_path "./results/CVBM/1" \
#     > output_train_Pancreas_label6_num1.log 2>&1 &
# python Pancreas_train.py \
#     --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
#     --exp 'CVBM_Pancreas' \
#     --model 'CVBM' \
#     --gpu 1 \
#     --labelnum 12 \
#     --deterministic 0 \
#     --snapshot_path "./results/CVBM/1" \
#     > output_train_Pancreas_label12_num1.log 2>&1 &
# wait
# python test_Pancreas.py \
#     --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
#     --labelnum 6 \
#     --exp 'CVBM_Pancreas' \
#     --model 'CVBM' \
#     --snapshot_path "./results/CVBM/1" > output_test_Pancreas_label6_num1.log 2>&1 &
# python test_Pancreas.py \
#     --root_path '/home/xuminghao/Datasets/NIH-Pancreas/Pancreas' \
#     --labelnum 12 \
#     --exp 'CVBM_Pancreas' \
#     --model 'CVBM' \
#     --snapshot_path "./results/CVBM/1" > output_test_Pancreas_label12_num1.log 2>&1 &
# wait
# wait
# $HOME/send_mail.sh -t 2708964632@qq.com -m "Pancreas label6 and label12 training finish in 43304 3090."
