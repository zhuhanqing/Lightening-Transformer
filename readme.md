# Lightening-Transformer
This contains the main codebases of the paper [Lightening-Transformer: A Dynamically-operated Optically-interconnected Photonic Transformer Accelerator](https://arxiv.org/abs/2305.19533). 
Accepted by [**HPCA 2024**](https://hpca-conf.org/2024/).


**We provide the first arch-level simulator for the photonic-based AI accelerator/system**, which has been used as the backbone in
* [Hybrid Photonic-digital Accelerator for Attention Mechanism](https://arxiv.org/abs/2501.11286), arxiv 2025
* [SimPhony](https://github.com/ScopeX-ASU/SimPhony), arxiv 2024
* [Scatter](https://github.com/ScopeX-ASU/SCATTER), ICCAD 2024




---

## Usage of the Provided Codebase

We provides three kinds of codebases:

* (1) algorithm codes for training/running models on our photonic accelerator, with the analytic transformation of our unique photonic tensor core embedded in the computation process. See `./software_model` for detailed implementation and usages, including the [DeiT](https://arxiv.org/abs/2012.12877) case.

* (2) hardware simulator for estimating the energy and latency running Transformers on our photonic accelerator. See `./hardware_simulator` for detailed implementation and usages.

* (3) profile codes for profiling latency and power usage of running Transformers on GPU. See `./profile` for detailed implementation and usages. The implementation refers to [Neurips'23, Pre-RMSNorm and Pre-CRMSNorm Transformers: Equivalent and Efficient Pre-LN Transformers](https://github.com/zixuanjiang/pre-rmsnorm-transformer).

---

## Required Dependencies

The DeiT requires to install PyTorch and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models).


```
conda create -n test # create a virtual env
conda install pytorch torchvision torchaudio pytorch-cuda=your_cuda_version -c pytorch -c nvidia # install pytorch
pip install timm==0.3.2 torchpack packaging einops gdown
conda activate test # activate the test env
```

For torch.2.0+, you will encounter the ModuleNotFoundError: No module named 'torch._six' in '/path_to_your_conda_envs/your_env_name/lib/python_version/site-packages/timm/models/layers/helpers.py". This is because torch2.0 doesn't have torch._six. Please replace the helper.py file with the following one.

```
from itertools import repeat
# from torch._six import container_abcs


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
```

## Reference

[1] Hanqing Zhu, Jiaqi Gu, Hanrui Wang, Zixuan Jiang, Rongxing Tang, Zhekai Zhang, Chenghao Feng, Song Han, Ray T. Chen and David Z. Pan. "Lightening-Transformer: A Dynamically-operated Optically-interconnected Photonic Transformer Accelerator", IEEE International Symposium on High-Performance Computer Architecture (HPCA'24). 
