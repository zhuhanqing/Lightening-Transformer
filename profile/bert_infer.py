# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu(hqzhu@utexas.edu)
# @Date:   1969-12-31 18:00:00
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-09 23:42:51
import argparse
import torch
import torch.utils.benchmark as benchmark
from model import PreDefinedBERT

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", default="deit-s",
                        help="model")
parser.add_argument("-s", "--seq_length", default=128,
                        help="seq length")

args, opts = parser.parse_known_args()

vocab_size = 30528
max_seq_length = 2048
num_classes = 2
using_torch_compile = False
device = torch.device('cuda')
# device = torch.device('cpu')

batch_size_list = [1]
num_threads_list = [1]
min_run_time = 100

model_dict = {
    'bert-b': ['Base', 768],
    'bert-l': ['Large', 1024]
}

results = []

model_name, emebedding_size = model_dict[args.model_name]
seq_len = int(args.seq_length)
for method in ['pre-ln']:
    raw_model = PreDefinedBERT(vocab_size=vocab_size, max_seq_length=max_seq_length, variant=model_name, method=method, num_classes=num_classes).to(device)
    model = torch.compile(raw_model) if using_torch_compile else raw_model
    model.eval()
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for batch_size in batch_size_list:
                for num_threads in num_threads_list:
                    x = torch.randn(batch_size, seq_len, emebedding_size).to(device)
                    result = benchmark.Timer(stmt='y = model(x)',
                                                setup='from __main__ import model',
                                                globals={'x': x},
                                                num_threads=num_threads,
                                                sub_label=f'batch_size {batch_size} seq_len {seq_len}',
                                                description=model_name + ' ' + method,
                                                ).blocked_autorange(min_run_time=min_run_time)
                    results.append(result)
                    print(result)

compare = benchmark.Compare(results)
compare.print()
