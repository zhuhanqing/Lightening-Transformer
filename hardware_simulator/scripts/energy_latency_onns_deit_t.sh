exp='energy_latency_compare_onns_deit_t'
model_name='deit-t'
tokens=197
declare -A config_dict
config_dict=(
    ['dota_b_4bit']='./params/device_params/Dota_B_4bit.yaml'
    # ['mrr_4bit']='./params/device_params/Bs_mrr_bank_4bit.yaml'
    # ['mzi_4bit']='./params/device_params/Bs_mzi_4bit.yaml'
)

for key in "${!config_dict[@]}"
do
    # Get the value associated with the key
    onn_params="${config_dict[$key]}"

    python entry_energy_latency_workload.py \
        -e ${exp} \
        --tokens ${tokens} \
        --model_name ${model_name} \
        --config ${onn_params} \
        -o 'broadcast'
done