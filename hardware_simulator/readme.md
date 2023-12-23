# Hardware simulator for our photonic Transformer accelerator

This contains the hardware simulator for our photonic Transformer accelerator, DOTA, in our lightning-transformer work.
Our simulator is based on behavior-level simulation.

We support simulates our DOTA-B/L variants with 4-bit/8-bit work mode. And we also support simulating the photonic baselines, MRR bank and MZI.

---

## Code structures

* `./hardwares/`. This directory contains the modeling for photonic tensor cores, including our dynamically-operated crossbar-style PTC and two baselines: MRR bank and MZI.
* `./params/`. 
    * `./params/device_params/` This directory contains the accelerator detailed params as well as all the device parameters.
        * DOTA-B: A 4 tile variant of our DOTA photonic Transformer accelerator.
        * DOTA-L: A 8 tile variant of our DOTA photonic Transformer accelerator.
        * MZI: A 2 tile variant of MZI mesh.
        * MRR bank: A 7 tile variant of MRR bank.
        * *NOTE: we keep DOTA-B, MZI, and MRR bank under the same area budget for fair comparasion.* 

* `entry_area_power_profile.py`. The python file you can launch to profile the area and power of the accelerator.
* `entry_energy_latency_workload.py`. The python file you can launch to profile the energy and latency when running one workload on given accelerator.
* `/results/`. The generated results will be dumpped into this directory.
* `/utils/`. Utility functions.

## AE exp1: Simulate the area and power of our photonic accelerator.

### Single run by run `entry_area_power_profile.py`

To simulate the area and power, run
```
exp='area_power_profile_single' # exp name you give
config='./params/device_params/Dota_B_4bit.yaml' # the param file of the given photonic accelerator

python entry_area_power_profile.py \
        -e ${exp} \
        --config ${config}
```

It will generate the area and power report under `./results/exp_name_you_give/accelerator_name/`. It contains two CSV files for area and power estimation.

For example if you run
```
python entry_area_power_profile.py -e area_power_profile_single --config ./params/device_params/Dota_B_4bit.yaml
```
You will have the area and power report under `./results/area_power_profile_single/dota_4t_2c_4bit/`.
The area report would be 

|dota         |area (mm^2)         |percentage (%)|
|-------------|--------------------|--------------|
|total        |60.329395086        |1             |
|laser        |0.72                |1.19          |
|DAC          |15.84               |26.26         |
|MZM          |7.59416832          |12.59         |
|ADC          |1.6416              |2.72          |
|TIA          |0.0576              |0.1           |
|photonic_core|11.318291999999998  |18.76         |
|adder        |0.051199999999999996|0.08          |
|mem          |14.695398766000002  |24.36         |
|micro_comb   |8.411135999999999   |13.94         |


*Note that we only provide area report for MZI and MRR baselines.*

### Batch run by run `./scripts/area_power_all.sh`

We provide one script to run the area and power estimation for all photonic accelerator variants: DOTA-B-4/8bit, DOTA-L-4/8bit, MRR-4/8bit, MZI-4/8bit. It generated results under `./results/area_power_all/`


## AE exp2: Simulate the energy and latency when running workload on photonic system.

### Single run by run `entry_energy_latency_workload.py`

To simulate the energy and latency for a given Transformer workload (DeiT-T/S/B, BERT-B/L in our work), run the 
```
exp='energy_latency_single_workload' # exp name
model_name='deit-t' # model name
tokens=197 # number of tokens, 197 for deit, you can define number of tokens for bert
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
```

It will generate the energy and latency report under `./results/exp_name_you_give/accelerator_name/`. 

It contains a `total.csv` under this directory for energy and latency estimation, which also has the breakdown based on different layer types, e.g., attention/FFN/QKV/head.

We also privide a more detailed energy breakdown for different layer types under the `modules/` in this directory.
It provides the energy breakdown across different compoents (e.g., laser, DAC, ADC, data-moevemnt, etc.).

You can change the arguments for model_name and the corresponding tokens. We support following arguments for `model_name`
* deit-t
* deit-s
* deit-b
* bert-b
* bert-l

The correct token number for deit on ImageNet dataset is 197. For bert, you can vary different number of tokens.

You can also change the argument to enable/disable architecture-level optimization for our DOTA by settting the argument
```
--optimize_flag arch_opt # set to crossbar to disable arch optimization
```

### Batch run by run `./scripts/energy_latency_all.sh`

We provide one script to run the estimation for all workloads we used in our paper.
* deit-t with tokens=197
* deit-s with tokens=197
* deit-b with tokens=197
* bert-b with tokens=384
* bert-l with tokens=320
for photonic accelerator variants: DOTA-B-4/8bit, DOTA-L-4/8bit, MRR-4/8bit, MZI-4/8bit. It generated results under `./results/energy_latency_all/`

*Note that we only provide reports on linear layers for MZI since it cannot support Attention efficiently due to on-the-fly activation decomposition that is extremely expensive.*
