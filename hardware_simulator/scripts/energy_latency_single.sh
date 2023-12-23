exp='energy_latency_single_workload'
model_name='deit-t'
tokens=197
onn_params='./params/device_params/Dota_B_4bit.yaml'
# choose onn accelerator params from
# config_dict=(
#     ['dota_b_4bit']='./params/device_params/Dota_B_4bit.yaml'
#     ['dota_b_8bit']='./params/device_params/Dota_B_8bit.yaml'
#     ['dota_l_4bit']='./params/device_params/Dota_L_4bit.yaml'
#     ['dota_l_8bit']='./params/device_params/Dota_L_8bit.yaml'
#     ['mrr_4bit']='./params/device_params/Bs_mrr_bank_4bit.yaml'
#     ['mrr_8bit']='./params/device_params/Bs_mrr_bank_8bit.yaml'
#     ['mzi_4bit']='./params/device_params/Bs_mzi_4bit.yaml'
#     ['mzi_8bit']='./params/device_params/Bs_mzi_8bit.yaml'
# )


python entry_energy_latency_workload.py \
    -e ${exp} \
    --tokens ${tokens} \
    --model_name ${model_name} \
    --config ${onn_params}
