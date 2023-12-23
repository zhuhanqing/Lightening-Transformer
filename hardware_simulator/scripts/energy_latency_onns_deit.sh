exp='energy_latency_onns_deit'
# model_name='deit-t'
tokens=197
optimize_flag='crossbar'
declare -A config_dict
config_dict=(
    ['dota_b_4bit']='./params/device_params/Dota_B_4bit.yaml'
    ['mrr_4bit']='./params/device_params/Bs_mrr_bank_4bit.yaml'
    ['mzi_4bit']='./params/device_params/Bs_mzi_4bit.yaml'
    ['dota_b_8bit']='./params/device_params/Dota_B_8bit.yaml'
    ['mrr_8bit']='./params/device_params/Bs_mrr_bank_8bit.yaml'
    ['mzi_8bit']='./params/device_params/Bs_mzi_8bit.yaml'
)

for model_name in 'deit-t' 'deit-b'
do
    for key in "${!config_dict[@]}"
    do
        # Get the value associated with the key
        onn_params="${config_dict[$key]}"

        python entry_energy_latency_workload.py \
            -e ${exp} \
            --tokens ${tokens} \
            --model_name ${model_name} \
            --config ${onn_params} \
            -o ${optimize_flag}
    done
done