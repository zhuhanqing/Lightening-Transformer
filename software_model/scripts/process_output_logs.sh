# This is the scripts to process the saved log file from evaluate_quant_transformer_scan_noise.sh
# It will generate a csv file to give you the accurcay mean and std of multiple runs

# set the log file directory

## params when you parse logs for sweep_wavelength
log_file='./logs/deit_t_sweep_input_noise_std.log'
num_iters=3 # number of runs you launch for accurcay test
num_vars=6 # how many variations you sweep

# ## params when you parse logs for sweep input noise std
# log_file='./logs/deit_t_sweep_input_noise_std.log'
# num_iters=3 # number of runs you launch for accurcay test
# num_vars=6 # how many variations you sweep

# ## params when you parse logs for sweep input noise std
# log_file='./logs/deit_t_sweep_phase_noise_std.log'
# num_iters=3 # number of runs you launch for accurcay test
# num_vars=6 # how many variations you sweep

python process_logs.py \
    --file ${log_file} \
    --iters ${num_iters} \
    --num_vars ${num_vars}