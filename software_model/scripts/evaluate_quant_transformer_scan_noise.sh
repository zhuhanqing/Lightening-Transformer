#!/bin/bash
# @Author: Hanqing Zhu
# @Date:   2023-01-04 22:18:40
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-10 16:52:06
exp='eval_accuracy_scan_noise'
wbits=4
abits=4
id=4bit
headwise=1

# noise settings
input_noise_std=0.03
output_noise_std=0.05
# following setting is added for inference only
phase_noise_std=2 
num_wavelength=12
channel_spacing=0.4
seed=0

resumed_ckpt_path='./resumed_ckpt/best_checkpoint.pth'


for i in {1..3}
do
    # for input_noise_std in 0.03 0.04 0.05 0.06 0.07 0.08 ## uncomment this line when scanning input noise
    # for phase_noise_std in 2 3 4 5 6 7 ## uncomment this line when scanning phase noise
    for num_wavelength in 8 12 16 20 24 ## uncomment this line when scanning # wavelength
    do
        CUDA_VISIBLE_DEVICES=2 python main.py --eval \
        --resume ${resumed_ckpt_path} \
        --model deit_tiny_patch16_224_quant \
        --drop-path 0 \
        --wbits ${wbits} \
        --abits ${abits} \
        --data-path /home/usr1/zixuan/ImageNet/data \
        --headwise \
        --input_noise_std ${input_noise_std} \
        --output_noise_std ${output_noise_std} \
        --phase_noise_std ${phase_noise_std} \
        --num_wavelength ${num_wavelength} \
        --channel_spacing ${channel_spacing} \
        --seed ${seed+$i} \
        --enable_wdm_noise \
        --enable_linear_noise
    done
done

