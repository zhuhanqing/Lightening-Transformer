# Profiling Workloads on GPU

We provides profiling scripts on GPU. The reported results are measured on single A100 GPU with automatic mixed precision.

1. `vit_infer.py`, `bert_infer.py` for launching inference on a single device.
2. `model.py` provides a simplidied DeiT and BERT implementation based on [a](https://github.com/zixuanjiang/pre-rmsnorm-transformer) and [b](https://github.com/lucidrains/vit-pytorch).


We provide the benchmark logs of our tested results in `benchmark_logs/`.

---
## Latency Measurement

We use `torch.utils.benchmark` to measure the latency for DeiT and BERT models for SST tasks(BERT for sequential classification).

We set the minimum run to 100 for each measurement.

### How to use

#### DeiT
Launch `python vit_infer.py -m model_name` to obtain latency for DeiT. 
* `-m`: The flag for different models. We can set it to `deit-t`, `deit-s`, `deit-b` to test latency for DeiT-Tiny, DeiT-Small, DeiT-Base.

#### BERT
Launch `python bert_infer.py -m model_name -s seq_length` to obtain latency for BERT for sequence classification. 
* `-m`: The flag for different models. We can set it to `bert-b`, `bert-l` to test latency for DeiT-Tiny, DeiT-Small, DeiT-Base.
* `-s`: The flag for sequence length. You can try 128, 256, 384, 320 for BERT.

### Expected results
If we set model to DeiT-B, and run `python vit_infer.py -m deit-b` the reported results should be like

```
y = model(x): batch_size 1 method pre-ln
Base
setup: from __main__ import model
  Median: 4.37 ms
  IQR:    0.06 ms (4.36 to 4.42)
  226 measurements, 100 runs per measurement, 1 thread
[-------------------  ------------------]
                                  |  Base
1 threads: ------------------------------
      batch_size 1 method pre-ln  |  4.4 

Times are in milliseconds (ms).
```

---

## Power Tracing

We use nvidia-smi to monitor the power usage when running the workloads on GPU.
```
nvidia-smi dmon -s puc -d 1 -i 0 > ./power_results/power_usage.csv
```
* `-s puc`: The `-s`` flag specifies which metrics to monitor. In this case, it's set to `puc``, which stands for "power usage in watts of the GPU."
* `-d 1`: The `-d` flag specifies the update interval in seconds. Here, it's set to 1 second, which means that the power usage will be sampled and recorded every 1 second.
* `-i 0`: The `-i` flag specifies the GPU index to monitor. In this case, it's set to 0, indicating that the monitoring should be done on GPU index 0. You can change this number to monitor a different GPU if you have run jobs on different GPUs in your system. By default, we use GPU with index 0.
* `> ./power_results/power_usage.csv`: Save the monitor power usage result into power_usage.csv.

### How to use

Launch `power_monitor.sh` to mointor the power usage. You can save the power usage into a csv file for further processing.

Please run `power_monitor.sh` before launching inference scripts `vit_infer.py`, `bert_infer.py`.

### Expected results

Take the DeiT-T as an example (see `./benchmark_logs/deit-s-power.csv`).
The monitored power usage shows you the idle power (61W in our case) and work power (72W). 
Then the power during inference is 72-61=11W.

## Energy estimation

Multiply the power with the measured latency for single inference, you can get the energy cost for single inference.

For example, the DeiT-base model has a power of 16W and a latency of 4.37ms, so the energy cost is 113.62 mJ.

---

## AE workflow

Follow the three steps to obtain both latency and power usage
Open two terminals on the same machine.

* First run `./power_monitor.sh > ./power_results/power_usage.csv` to mointor the power usage of GPU 0. `> ./power_results/power_usage.csv`: Save the monitor power usage result into power_usage.csv.
* Then launch the latency measurement file: `python vit_infer.py -m model_name`  or `python bert_infer.py -m model_name -s seq_length` on *another teminal*.
* Kill the power monitor script when the latenyc measurement finishs.

Obtain the power of GPU for running workloads by substracting the power with the idle power.

For example, the DeiT-base model has a power of (87-61=16W) and a latency of 4.37ms, so the energy cost is 113.62 mJ.

---

We refer to the following implementation.
1. [A simplified ViT implementation in PyTorch](https://github.com/lucidrains/vit-pytorch)
2. [BERT implementation from Nvidia](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)
3. [Measurement codes from pre-rmsnorm-transformer](https://github.com/zixuanjiang/pre-rmsnorm-transformer)
