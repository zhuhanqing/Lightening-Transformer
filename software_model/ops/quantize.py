# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu
# @Date:   2023-01-02 21:11:56
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-09 21:57:41
import torch
import torch.nn.functional as F
import math
from ._quant_base import _Conv2dQ, Qmodes, _LinearQ, _ActQ
from .simulator import cal_coupler_wdm_error_list
import numpy as np

"""
@inproceedings{
    esser2020learned,
    title={LEARNED STEP SIZE QUANTIZATION},
    author={Steven K. Esser and Jeffrey L. McKinstry and Deepika Bablani and Rathinakumar Appuswamy and Dharmendra S. Modha},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=rkgO66VKDS}
}
    https://quanoview.readthedocs.io/en/latest/_raw/LSQ.html
"""

__all__ = ["QuantLinear", "QuantAct", "QuantConv2d"]


class FunLSQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp):
        assert alpha > 0, 'alpha = {}'.format(alpha)
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp
        q_w = (weight / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = weight / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        # indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
        indicate_middle = 1.0 - indicate_small - indicate_big  # Thanks to @haolibai
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle *
                      (-q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
        # grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * 0) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = indicate_middle * grad_weight
        return grad_weight, grad_alpha, None, None, None


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


def clamp(x, minv, maxv):
    print(minv.dtype)
    x = torch.minimum(x, maxv)
    x = torch.maximum(x, minv)
    return x


class QuantConv2d(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits=-1, nbits_a=-1, mode=Qmodes.layer_wise, offset=False, 
                 input_noise_std=0, output_noise_std=0, phase_noise_std=0, enable_wdm_noise=False, 
                 num_wavelength=9, channel_spacing=0.4, enable_linear_noise=False, **kwargs):
        super(QuantConv2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits, mode=mode,)
        self.enable_linear_noise = enable_linear_noise
        self.input_noise_std = input_noise_std if self.enable_linear_noise else 0
        self.output_noise_std = output_noise_std if self.enable_linear_noise else 0
        self.act = QuantAct(in_features=in_channels, nbits=nbits_a,
                            mode=Qmodes.layer_wise, offset=offset, input_noise_std=self.input_noise_std)

    def add_output_noise(self, x):
        # the noise std is 2sigma not 1sigma, so should be devided by 2
        if self.output_noise_std > 1e-5:
            noise = torch.randn_like(x).mul(
                (self.output_noise_std)).mul(x.data.abs())
            x = x + noise
        return x

    def add_input_noise(self, x):
        # the noise std is 2sigma not 1sigma, so should be devided by 2
        if self.input_noise_std > 1e-5:
            noise = torch.randn_like(x).mul(
                (self.input_noise_std)).mul(x.data.abs())
            x = x + noise
        return x

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        # full range quantization -> -2**(k-1) -> 2**(k-1)-1
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.init_state == 0:
            print(
                f"Conv layer (mode: {self.q_mode}): initialize weight scale for int{self.nbits} quantization")
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(quantize_by_mse(self.weight, Qn, Qp))
            self.init_state.fill_(1)

        with torch.no_grad():
            g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        self.alpha.data.clamp_(min=1e-4)
        # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048) 17min/epoch
        alpha = grad_scale(self.alpha, g)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        # Method2: 25GB GPU memory (AlexNet w4a4 bs 2048) 32min/epoch
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)

        x = self.act(x)
        
        # add noise at w_q
        if self.enable_linear_noise and self.input_noise_std > 1e-5:
            w_q = self.add_input_noise(w_q)
            
        out = F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
        if self.enable_linear_noise and self.output_noise_std > 1e-5:
            out = self.add_output_noise(out)
        
        return out


class QuantLinear(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits=-1, nbits_a=-1, mode=Qmodes.layer_wise, offset=False, 
                 input_noise_std=0, output_noise_std=0, phase_noise_std=0, enable_wdm_noise=False, 
                 num_wavelength=9, channel_spacing=0.4, enable_linear_noise=False, **kwargs):
        super(QuantLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias,
                                          nbits=nbits, mode=mode)
        print(
                f"Linear layer (mode: {self.q_mode}): initialize weight scale for int{self.nbits} quantization")
        self.enable_linear_noise = enable_linear_noise
        self.input_noise_std = input_noise_std if self.enable_linear_noise else 0
        self.output_noise_std = output_noise_std if self.enable_linear_noise else 0
        self.phase_noise_std = phase_noise_std if self.enable_linear_noise else 0
        self.kappa_noise = None if not (enable_wdm_noise and enable_linear_noise) else cal_coupler_wdm_error_list(
            num_wavelength=num_wavelength, channel_spacing=channel_spacing)
        self.num_wavelength = num_wavelength
        self.out_features = out_features
        self.in_features = in_features
        
        if self.kappa_noise is not None:
            self.kappa_noise_term = torch.tensor(self.kappa_noise).unsqueeze(0).expand((in_features // self.num_wavelength) + 1, -1).reshape(-1).contiguous()[:in_features]
        else:
            self.kappa_noise_term = None
        
        self.act = QuantAct(in_features=in_features, nbits=nbits_a,
                            mode=Qmodes.layer_wise, offset=offset, input_noise_std=self.input_noise_std)

    def add_input_noise(self, x):
        # the noise std is 2sigma not 1sigma, so should be devided by 2
        if self.input_noise_std > 1e-5:
            # add mul noise here
            noise = torch.randn_like(x).mul(
                (self.input_noise_std)).mul(x.data.abs())
            x = x + noise
        return x
    
    def add_output_noise(self, x):
        # the noise std is 2sigma not 1sigma, so should be devided by 2
        if self.output_noise_std > 1e-5:
            noise = torch.randn_like(x).mul(
                (self.output_noise_std)).mul(x.data.abs())
            x = x + noise
        return x
    
    def add_phase_noise(self, x, noise_std=2):
        # the noise std is 2sigma not 1sigma, so should be devided by 2
        # DATE O2NN use 0.04 -> 0.04 * 360 / 2pi = 2.29
        if noise_std > 1e-5:
            noise = (torch.randn_like(x).mul_((noise_std) / 180 * np.pi)).cos_()

        x = x * noise

        return x

    def forward(self, x):
        kappa_noise_scale_factor = 2
        
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.init_state == 0:
            print(
                f"Linear layer (mode: {self.q_mode}): initialize weight scale for int{self.nbits} quantization")
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
            # lsq+ init
            # m, v = self.weight.abs().mean(), self.weight.abs().std()
            # self.alpha.data.copy_(torch.max(torch.abs(m - 3*v), torch.abs(m + 3*v)) / 2 ** (self.nbits - 1) )
        assert self.init_state == 1
        with torch.no_grad():
            g = 1.0 / math.sqrt(self.weight.numel() * Qp)
            # g = 1.0 / math.sqrt(self.weight.numel()) / Qp
        # g = 1.0 / math.sqrt(self.weight.numel()) / 4
        self.alpha.data.clamp_(min=1e-4)
        # Method1:
        alpha = grad_scale(self.alpha, g)
        # w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        # w_q = clamp(round_pass(self.weight / alpha), Qn, Qp) * alpha
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        # Method2:
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        x = self.act(x)
        
        # add noise @ w_q
        if self.enable_linear_noise and self.input_noise_std > 1e-5:
            w_q = self.add_input_noise(w_q)
        
        if not self.training and self.phase_noise_std > 1e-5 and self.enable_linear_noise:
            noise_w_q_2 = 0
            noise_x_2 = 0
            if self.kappa_noise is not None:
                if self.kappa_noise_term.device != x.device:
                    self.kappa_noise_term = self.kappa_noise_term.to(x.device)
                # obtain the scaling number
                alpha_x_to_w = self.act.alpha / alpha
                noise_x_2 = torch.matmul(x.square(), self.kappa_noise_term.unsqueeze(-1)) /(alpha_x_to_w * kappa_noise_scale_factor) # [bs, seq, 1]
                noise_w_q_2 = torch.matmul(w_q.square(), -self.kappa_noise_term.unsqueeze(-1))* (alpha_x_to_w / kappa_noise_scale_factor) # [output_features, 1]
            dim_3_flag = False
            if x.dim() == 3:
                dim_3_flag = True
                bs, N, D = x.shape
                bs = bs * N
                x = x.reshape(-1, D)
            else:
                bs, D = x.shape
            
            out = []
            k = 2
            num_chunks = self.out_features//k
            for i in range(k):
                if self.out_features%k != 0: raise RuntimeError
                noisy_x = self.add_phase_noise(x.unsqueeze(-2).expand(-1, num_chunks, -1))
                out.append(torch.einsum('ibk, bk->ib', noisy_x, w_q[i * num_chunks: (i+1) * num_chunks, :]))
                
            out = torch.cat(out, 1)
            
            if self.bias is not None:
                out += self.bias
            if dim_3_flag:
                out = out.reshape(-1, N, self.out_features)
            out = out + (noise_x_2 + noise_w_q_2.squeeze(-1)) # add [bs, seq, 1] and [1, output_features]
        else:
            out = F.linear(x, w_q, self.bias)
        
        # add output noise
        if self.enable_linear_noise and self.output_noise_std > 1e-5:
            out = self.add_output_noise(out)
        
        return out

class QuantAct(_ActQ):
    def __init__(self, in_features, nbits=-1, signed=True, mode=Qmodes.layer_wise, input_noise_std=0, offset=False, **kwargs):
        super(QuantAct, self).__init__(in_features=in_features,
                                       nbits=nbits, mode=mode, offset=offset)
        self.input_noise_std = input_noise_std
        self.offset = offset

    def add_input_noise(self, x):
        # the noise std is 2sigma not 1sigma, so should be devided by 2
        if self.input_noise_std > 1e-5:
            noise = torch.randn_like(x).mul(
                (self.input_noise_std)).mul(x.data.abs())
            x = x + noise
        return x

    def forward(self, x):
        if self.alpha is None:
            return x

        if self.training and self.init_state == 0:
            print(
                f"Act layer: (mode: {self.q_mode}): initialize weight scale for int{self.nbits} quantization with offset: {self.offset}")
            if self.q_mode == Qmodes.kernel_wise:
                print(f'Scale dimension: {self.alpha.shape}')
            # choose implementation from https://github.com/YanjingLi0202/Q-ViT/blob/main/Quant.py
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            if self.signed == 1:
                Qn = -2 ** (self.nbits - 1)
                Qp = 2 ** (self.nbits - 1) - 1
            else:
                Qn = 0
                Qp = 2 ** self.nbits - 1
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            if self.offset:
                self.zero_point.data.copy_(
                    self.zero_point.data * 0.9 + 0.1 * (torch.min(x.detach()) - self.alpha.data * Qn))
            self.init_state.fill_(1)

        assert self.init_state == 1
        if self.signed:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1
        with torch.no_grad():
            g = 1.0 / math.sqrt(x.numel() * Qp)

        self.alpha.data.clamp_(min=1e-4)
        # Method1:
        alpha = grad_scale(self.alpha, g)

        if self.offset:
            zero_point = (self.zero_point.round() -
                          self.zero_point).detach() + self.zero_point
            zero_point = grad_scale(zero_point, g)
            zero_point = zero_point.unsqueeze(0) if len(
                x.shape) == 2 else zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            zero_point = 0

        if len(x.shape) == 2:
            alpha = alpha.unsqueeze(0)
            # zero_point = zero_point.unsqueeze(0)
        elif len(x.shape) == 4:
            alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        x = round_pass((x / alpha + zero_point).clamp(Qn, Qp))
        x = (x - zero_point) * alpha

        x = self.add_input_noise(x)

        # Method2:
        # x = FunLSQ.apply(x, self.alpha, g, Qn, Qp)
        return x
