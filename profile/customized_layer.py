import torch
from torch import Tensor

"""
three normalization variants without elementwise_affine transformation
normlize along the last dimension
"""

decorator = torch.compile
# decorator = torch.jit.script


@decorator
def layer_norm(x: Tensor, eps: float):
    x_mean = x.mean(dim=-1, keepdim=True)
    x_var = x.var(dim=-1, keepdim=True, correction=0)
    return (x - x_mean) * torch.rsqrt(x_var + eps)


@decorator
def rms_norm(x, eps: float):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


@decorator
def crms_norm(x, eps: float):
    discarded_element = x.sum(dim=-1, keepdim=True)
    return x * torch.rsqrt((x.square().sum(dim=-1, keepdim=True) + discarded_element.square()) / (x.shape[-1] + 1) + eps)


class CustomizedLayerNorm(torch.nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return layer_norm(x.float(), self.eps).type_as(x)


class RMSNorm(torch.nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x.float(), self.eps).type_as(x)


class CRMSNorm(torch.nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return crms_norm(x.float(), self.eps).type_as(x)


class LinearZeroMeanOutput(torch.nn.Linear):
    def forward(self, x):
        zero_mean_weight = self.weight - self.weight.mean(dim=0, keepdim=True)
        zero_mean_bias = self.bias - self.bias.mean()
        return torch.nn.functional.linear(x, zero_mean_weight, zero_mean_bias)
