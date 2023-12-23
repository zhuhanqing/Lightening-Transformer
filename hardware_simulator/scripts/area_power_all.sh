## scripts to generate area and power estimation of our optical accelerator system.
## It will save the results to ./results/{exp_name}/
## dota is our circuit
## we will also generate area report for the optical baselines

declare -A config_dict
config_dict=(
    ['dota_b_4bit']='./params/device_params/Dota_B_4bit.yaml'
    ['dota_b_8bit']='./params/device_params/Dota_B_8bit.yaml'
    ['dota_l_4bit']='./params/device_params/Dota_L_4bit.yaml'
    ['dota_l_8bit']='./params/device_params/Dota_L_8bit.yaml'
    ['mrr_4bit']='./params/device_params/Bs_mrr_bank_4bit.yaml'
    ['mzi_4bit']='./params/device_params/Bs_mzi_4bit.yaml'
)


exp='area_power_all'

# Iterate through the keys in the config_dict
for key in "${!config_dict[@]}"
do
    # Get the value associated with the key
    value="${config_dict[$key]}"

    # launch the are and power estimation .py
    python entry_area_power_profile.py \
        -e ${exp} \
        --config "$value"
done