# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu(hqzhu@utexas.edu)
# @Date:   1969-12-31 18:00:00
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-09 23:05:25
import argparse
import torch
import torch.utils.benchmark as benchmark
from model import PreDefinedViT

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", default="deit-s",
                        help="model")

args, opts = parser.parse_known_args()

image_size = 224
num_classes = 1000
using_torch_compile = False
device = torch.device('cuda')
# device = torch.device('cpu')

batch_size_list = [1]
num_threads_list = [1]
min_run_time = 100

model_dict = {
    'deit-t': ['Tiny', 16],
    'deit-s': ['Small', 16],
    'deit-b': ['Base', 16]
}

results = []

model_variant = model_dict[args.model_name]
model_name, patch_size = model_variant
for method in ['pre-ln']:
    raw_model = PreDefinedViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, variant=model_name, method=method).to(device)
    model = torch.compile(raw_model) if using_torch_compile else raw_model
    model.eval()

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for batch_size in batch_size_list:
                for num_threads in num_threads_list:
                    x = torch.randn(batch_size, 3, image_size, image_size).to(device)
                    result = benchmark.Timer(stmt='y = model(x)',
                                                setup='from __main__ import model',
                                                globals={'x': x},
                                                num_threads=num_threads,
                                                sub_label=f'batch_size {batch_size} method {method}',
                                                description=model_name,
                                                ).blocked_autorange(min_run_time=min_run_time)
                    results.append(result)
                    print(result)

compare = benchmark.Compare(results)
compare.print()
