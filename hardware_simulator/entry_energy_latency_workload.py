# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu(hqzhu@utexas.edu)
# @Date:   1969-12-31 18:00:00
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-12 21:02:24

import os
import csv
import argparse
from utils.general import ensure_dir
from utils.config import configs
from utils.model import modelParams

from simulator_FFN import FFNPrediction
from simulator_attn import attnPrediction

def main(configs, model_name='deit-s', exp_name='compare_onn', optimize_flag='arch_opt', tokens=197, print_msg=False):
    # extraxt model workload charaterstics
    model_zoo = modelParams()
    ops_list = model_zoo.obtain_ops_list(model_name=model_name, tokens=tokens)
    
    if model_name == 'bert-l':
        factor = 2
    else:
        factor = 1

    sv_path = f"./results/{exp_name}/{model_name}_{tokens}_{configs.core.precision.in_bit}bit/{configs.core.type}_{optimize_flag}_{configs.arch.num_tiles}t_{configs.arch.num_pe_per_tile}c/"
    sv_sub_path = f"./results/{exp_name}/{model_name}_{tokens}_{configs.core.precision.in_bit}bit/{configs.core.type}_{optimize_flag}_{configs.arch.num_tiles}t_{configs.arch.num_pe_per_tile}c/modules/"

    ensure_dir(sv_path)
    ensure_dir(sv_sub_path)

    energy_sum = 0
    latency_sum = 0
    saved_arrays = []
    for item in ops_list:
        idx = item["idx"]
        name = item["name"]
        type = item["type"]
        if type == "fc":
            predictor = FFNPrediction(item, configs)
            predictor.run(print_msg=print_msg)
            predictor.save(sv_name=name, sv_path=sv_sub_path)
            energy_cost = predictor.energy_dict['linear']['comp']['total'][0] + \
                predictor.energy_dict['linear']['datamovement']['total'][0]

            latency_cost = predictor.latency_dict['linear']['total'][1]
            if not 'head' in name and not 'embed' in name:
                energy_cost *= 12 * factor
                latency_cost *= 12 * factor
            saved_arrays.append([name, energy_cost, latency_cost])
        elif type == 'attn':
            if configs.core.type != 'mzi':
                predictor = attnPrediction(item, configs)
                predictor.run(print_msg=print_msg)
                predictor.save(sv_name=name, sv_path=sv_sub_path)
                energy_cost = predictor.energy_dict['Q*K^T']['comp']['total'][0] + predictor.energy_dict['Q*K^T']['datamovement']['total'][0] + \
                    predictor.energy_dict['S*V']['comp']['total'][0] + predictor.energy_dict['S*V']['datamovement']['total'][0]
                # print(predictor.energy_dict['linear']['comp'])
                latency_cost = predictor.latency_dict['Q*K^T']['total'][1] + predictor.latency_dict['S*V']['total'][1]
                energy_cost *= 12 * factor
                latency_cost *= 12 * factor
                saved_arrays.append([name, energy_cost, latency_cost])
        else:
            raise NotImplementedError
        energy_sum += energy_cost
        latency_sum += latency_cost
    
    energy_others, latency_others = model_zoo.obtain_other_costs(model_name=model_name, tokens=tokens)
    saved_arrays.append(["others", energy_others, latency_others])
    energy_sum += energy_others
    latency_sum += latency_others

    def __save_csv(sv_name, total, arrays):
        with open(sv_name, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['', 'energy (mJ)', 'latency (ms)'])
            writer.writerow(total)
            for each in arrays:
                writer.writerow(each)
    __save_csv(os.path.join(sv_path, 'total.csv'), [
               'total', energy_sum, latency_sum], saved_arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default=".params.yaml",
                        metavar="FILE", help="config file")
    parser.add_argument("-m", "--model_name", default="deit-s",
                        help="model")
    parser.add_argument("-t", "--tokens", default=197,
                        help="tokens or sequence length")
    parser.add_argument("-o", "--optimize_flag", default="arch_opt",
                        help="optimize flag for DOTA")
    parser.add_argument("-e", "--exp_name", default="compare_onn",
                        help="experiments name")
    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive=True)
    configs.update(opts)
    
    if configs.core.type == "dota":
        # three different optimize flag
        # broadcast
        # crossbar
        # arch-opt
        assert args.optimize_flag in ["broadcast", "crossbar", "arch_opt"]
        configs.arch.disable_crossbar_topology = 1 if args.optimize_flag == "broadcast" else 0
        if args.optimize_flag == "arch_opt":
            configs.arch.adc_share_flag = 1
            configs.arch.time_accum_factor = 3
            configs.arch.input_mod_sharing_flag = 1
        else:
            configs.arch.adc_share_flag = 0
            configs.arch.time_accum_factor = 1
            configs.arch.input_mod_sharing_flag = 0
    elif configs.core.type == 'mrrbank' or configs.core.type == 'mzi':
        configs.arch.weight_reuse_factor = -1 # fully weight-stationary flow
        args.optimize_flag = 'broadcast'
    else:
        raise ValueError(f"Got unsupportted core type {configs.core.type}")
    print(f"Report energy and latency estimation for {args.model_name}_{args.tokens}_{configs.core.precision.in_bit}bit on {configs.core.type}_{args.optimize_flag}_{configs.arch.num_tiles}t_{configs.arch.num_pe_per_tile}c")

    main(configs=configs, model_name=args.model_name, exp_name=args.exp_name, optimize_flag=args.optimize_flag, tokens=int(args.tokens))

    sv_path = f"./results/{args.exp_name}/{args.model_name}_{args.tokens}_{configs.core.precision.in_bit}bit/{configs.core.type}_{args.optimize_flag}_{configs.arch.num_tiles}t_{configs.arch.num_pe_per_tile}c"

    print(f'Finish and save report to {sv_path}')
    print('-'*20)