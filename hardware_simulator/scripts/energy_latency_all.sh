#!/bin/bash

# Define the experiment type
exp='energy_latency_all'

# Define the config_dict with possible values
declare -A config_dict
config_dict=(
    ['dota_b_4bit']='./params/device_params/Dota_B_4bit.yaml'
    ['dota_b_8bit']='./params/device_params/Dota_B_8bit.yaml'
    ['dota_l_4bit']='./params/device_params/Dota_L_4bit.yaml'
    ['dota_l_8bit']='./params/device_params/Dota_L_8bit.yaml'
)

# Define the workload_dict with possible values
declare -A workload_dict
workload_dict=(
    ['deit-t']='197'
    ['deit-s']='197'
    ['deit-b']='197'
    ['bert-b']='128'
    ['bert-l']='320'
)

# Loop through the workload_dict
for model_name in "${!workload_dict[@]}"
do
    # Get the value associated with the key
    tokens="${workload_dict[$model_name]}"
    
    # Loop through the config_dict
    for onn in "${!config_dict[@]}"
    do
        onn_params="${config_dict[$onn]}"
        
        # Call your Python script with the arguments
        python entry_energy_latency_workload.py \
            -e "${exp}" \
            --tokens "${tokens}" \
            --model_name "${model_name}" \
            --config "${onn_params}"
    done
done