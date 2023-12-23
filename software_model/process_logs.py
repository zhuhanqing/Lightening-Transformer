# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu
# @Date:   2023-03-21 15:40:36
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-10 10:43:07
import re
import csv
import statistics
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-f", "--file", default="./robustness/sweep_phase_noise_deit_tiny_4bit.log",
                        help="file")
parser.add_argument("-i", "--iters", default=3,
                        help="iterations")
parser.add_argument("-n", "--num_vars", default=6,
                        help="number of variations you sweep")

args, opts = parser.parse_known_args()

log_file = args.file
num_iters = int(args.iters)
num_variations = int(args.num_vars)


with open(log_file, "r") as file:
    log_data = file.read()

accuracy_pattern = r"\* Acc@1 (\d+\.\d+)"

accuracy_matches = re.findall(accuracy_pattern, log_data)

if accuracy_matches:
    accuracies = [float(match) for match in accuracy_matches]
    print(f"Accuracy: {accuracies}")
else:
    print("Accuracy not found in log file.")
    
indices = [x*num_variations for x in range(num_iters)]
result = []

for i in range(num_variations):
    print("**", indices)
    tmp = [float(accuracy_matches[i]) for i in indices]
    mean = statistics.mean(tmp)
    std = statistics.stdev(tmp)
    tmp.extend([mean, std])
    result.append(tmp)
    indices = [x + 1 for x in indices]

filename = log_file.split("/")[-1].split(".")[0] + '.csv'

def save_arrays_to_file(file_name, arrays):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['test1', 'test2', 'test3', 'mean', 'std'])
        for array in arrays:
            writer.writerow(array)

save_arrays_to_file(filename, result)