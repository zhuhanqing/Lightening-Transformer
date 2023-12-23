# Train and Inference of DeiT on our lightining-transformer

---

For deit, we built on the official implementation (https://github.com/facebookresearch/deit).

To model the inference accuracy on our photonic accelerator, we explictly inject the analytic transformation of our photonic tensor core during computation. We consider several nonidealties and inject them during inference, including **input encoding magnitude varation**, **input encoding phase variaion**, **output computation variation**, and **WDM dispersion introduced by multiple wavelength**.

Please ensure that you have install the required dependencies following instructions in `../readme.md`, before you run jobs.

---

## Structures
Our code is built upon the offical [DeiT](https://github.com/facebookresearch/deit).
* `./models/quant_vit.py`. The ViT model definition with quantization and analytic transformation of our PTC computation considering different noise resources.
* `./ops/`. Useful utils functions, including the implemented learned-step-size quantization [LSQ](https://github.com/hustzxd/LSQuantization) for transformer quantization.
* `main.py`. The main python file.
* `/scripts/`. This folder contains the scripts for implementing noise-aware training of low-bit DeiT models and testing inference accuracy.



## Data preparation

### Dataset
Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```
### Pretrained checkpoints
Download baseline DeiT models pretrained on ImageNet 2012 and put in the `pretrained` directory.

| name | acc@1 | acc@5 | #params | url |
| --- | --- | --- | --- | --- |
| DeiT-tiny | 72.2 | 91.1 | 5M | [model](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth) |
| DeiT-small | 79.9 | 95.0 | 22M| [model](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) |
| DeiT-base | 81.8 | 95.6 | 86M | [model](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth) |

```
mkdir pretrained
curl -o ./pretrained/deit_tiny_patch16_224-a1311bcf.pth https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth
curl -o ./pretrained/deit_small_patch16_224-cd65a155.pth https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth
curl -o ./pretrained/deit_base_patch16_224-b5f2ef4d.pth https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth
```

### Provided checkpoint for 4-bit DeiT-T
Training with DeiT may takes days given the quantization and the dedicated model of computation based on photonic tensor core.

We provide our checkpoint of DeiT-4bit for you to help you quickly perform evaluation and reproduce our results.
The model is in [google drive link](https://drive.google.com/uc?id=1EZjEnkqyBaBU8pUrYqNLTYMq4Mn0cbKV). 

```
mkdir resumed_ckpt
gdown https://drive.google.com/uc?id=1EZjEnkqyBaBU8pUrYqNLTYMq4Mn0cbKV -O resumed_ckpt/
```

## How to use

### Noise-aware training with a pretrained checkpoint

Train a quantized DeiT model using `./scripts/train_quant_transformer_with_noise.sh` by setting the bit-precision, input noise std. and output noise std and other training settings.

You need to replace the path in `--data-path /path/to/imagenet/data` by the path you put imagenet.

The `--finetune pretrained/deit_tiny_patch16_224-a1311bcf.pth` should be the path to the downloaded pretrained model.


```
wbits=4
abits=4
id=4bit
lr=5e-4
weight_decay=1e-8
batch_size=512
epochs=300
port=47771
headwise=1
input_noise_std=0.03
output_noise_std=0.05

torchrun \
--master_port ${port} \
--nproc_per_node=4 main.py \
--model deit_tiny_patch16_224_quant \
--drop-path 0 \
--batch-size ${batch_size} \
--lr ${lr} \
--min-lr 0 \
--epochs ${epochs} \
--warmup-epochs 0 \
--weight-decay ${weight_decay} \
--wbits ${wbits} \
--abits ${abits} \
--dist-eval \
--output_dir test/deit_tiny_${id}/${wbits}w${abits}a_bs${batch_size}_baselr${lr}_weightdecay${weight_decay}_ft${epochs}_headwise${headwise}_noise_i_${input_noise_std}_o_${output_noise_std}_linear_noise \
--finetune pretrained/deit_tiny_patch16_224-a1311bcf.pth \
--data-path /path/to/imagenet/data \
--headwise \
--input_noise_std ${input_noise_std} \
--output_noise_std ${output_noise_std} \
--enable_linear_noise
```

### Evaluation of a trained model with noise injection

Test the inference accuracy of a trained DeiT model using `./scripts/evaluate_quant_transformer.sh` and setting the corresponding noise levels.

* input_noise_std: Noise std of the input magtitude encoding. Default is 0.03.
* phase_noise_std: Noise std of the input phase encoding. Default is $2^{\circ}$.
* output_noise_std: Noise std of the computed outputs. Default is 0.05.
* num_wavelength: number of wavelength used in the system. We will calculate the wavelength-induced dispersion error. Default is 12.

Set `resumed_ckpt_path='./your/path/to/best_checkpoint.pth'` in the script.

```
exp='eval_accuracy'
wbits=4
abits=4
id=4bit
headwise=1

# noise settings
input_noise_std=0.03
output_noise_std=0.05
# following setting is added for inference only
phase_noise_std=2
num_wavelength=12
channel_spacing=0.4
seed=0

resumed_ckpt_path='./your/path/to/best_checkpoint.pth'


for i in {1..1}
do
    for input_noise_std in 0.03
    do
        CUDA_VISIBLE_DEVICES=0 python main.py --eval \
        --resume ${resumed_ckpt_path} \
        --model deit_tiny_patch16_224_quant \
        --drop-path 0 \
        --wbits ${wbits} \
        --abits ${abits} \
        --data-path /path/to/imagenet/data \
        --headwise \
        --input_noise_std ${input_noise_std} \
        --output_noise_std ${output_noise_std} \
        --phase_noise_std ${phase_noise_std} \
        --num_wavelength ${num_wavelength} \
        --channel_spacing ${channel_spacing} \
        --seed ${seed+$i} \
        --enable_wdm_noise \
        --enable_linear_noise
    done
done
```
It will give you the outputs as follows
```
Test: Total time: 0:29:16 (3.3723 s / it)
* Acc@1 71.052 Acc@5 90.432 loss 1.287
Accuracy of the network on the 50000 test images: 71.1%
```

---

## AE experiments: Reproduce reported results in accuracy and robustness analysis

We test the robustness of model running on our photonic accelerator by sweeping various on-chip noise-resources.
* input_noise_std: Noise std of the input magtitude encoding. Default is 0.03.
* phase_noise_std: Noise std of the input phase encoding. Default is $2^{\circ}$.
* output_noise_std: Noise std of the computed outputs. Default is 0.05.
* num_wavelength: number of wavelength used in the system. We will calculate the wavelength-induced dispersion error. Default is 12.

### Download our checkpoint

One trained DeiT-T-4bit model is provided for quickly reproducing the results.

Download it as follows:
```
mkdir resumed_ckpt
gdown https://drive.google.com/uc?id=1EZjEnkqyBaBU8pUrYqNLTYMq4Mn0cbKV -O resumed_ckpt/
```
It will in the `./resumed_ckpt`.

### Launch jobs with noise sweeping.

You can run `./scripts/evaluate_quant_transformer_scan_noise.sh` to measure the accuracy with varing noise levels.
We will test accurcy by three times.

By uncommenting the corresponding line in the script, you can reproduce the experiments for sweeping input noise std, phase noise std, and number of wavelengths.
```
for input_noise_std in 0.03 0.04 0.05 0.06 0.07 0.08 ## uncomment this line when scanning input noise
    # for phase_noise_std in 2 3 4 5 6 7 ## uncomment this line when scanning phase noise
    # for num_wavelength in 8 12 16 20 24 ## uncomment this line when scanning # wavelength
```


You can redirect the output of running `./scripts/evaluate_quant_transformer_scan_noise.sh` to a log file.
Then use our provided scripts `./scripts/process_output_logs.sh` to process the logs.
You will get the parsed accuracy as well as the mean and std in a CSV file.

```
./scripts/evaluate_quant_transformer.sh &> results.log # redirect results to a log file
./scripts/process_output_logs.sh # set the log_file path and number of iters and how many variations you sweep in the script
```

The expetced results will be like

```
test1,test2,test3,mean,std
71.174,71.014,70.99,71.05933333333333,0.10002666311206546
71.052,71.1,70.972,71.04133333333333,0.06466323014923916
71.034,70.924,70.924,70.96066666666667,0.06350852961085851
70.99,70.952,71.144,71.02866666666667,0.10167267741794891
71.206,70.82,71.184,71.07,0.21678560837842034
```
The first three columns represent the accurcay of 3 different runs with different seeds, followed by two columns being the mean and std of the three data.

Different rows represent the accuracy for different noise values, which is sweeped as in the `./scripts/evaluate_quant_transformer_scan_noise.sh`.

### Scripts `./scripts/evaluate_quant_transformer_scan_noise.sh`.
```
exp='eval_accuracy_scan_noise'
wbits=4
abits=4
id=4bit
headwise=1

# noise settings
input_noise_std=0.03
output_noise_std=0.05
# following setting is added for inference only
phase_noise_std=2
num_wavelength=12
channel_spacing=0.4
seed=0

resumed_ckpt_path='./resumed_ckpt/best_checkpoint.pth'

for i in {1..3}
do
    for input_noise_std in 0.03 0.04 0.05 0.06 0.07 0.08 ## uncomment this line when scanning input noise
    # for phase_noise_std in 2 3 4 5 6 7 ## uncomment this line when scanning phase noise
    # for num_wavelength in 8 12 16 20 24 ## uncomment this line when scanning # wavelength
    do
        CUDA_VISIBLE_DEVICES=0 python main.py --eval \
        --resume ${resumed_ckpt_path} \
        --model deit_tiny_patch16_224_quant \
        --drop-path 0 \
        --wbits ${wbits} \
        --abits ${abits} \
        --data-path /home/usr1/zixuan/ImageNet/data \
        --headwise \
        --input_noise_std ${input_noise_std} \
        --output_noise_std ${output_noise_std} \
        --phase_noise_std ${phase_noise_std} \
        --num_wavelength ${num_wavelength} \
        --channel_spacing ${channel_spacing} \
        --seed ${seed+$i} \
        --enable_wdm_noise \
        --enable_linear_noise
    done
done
```
