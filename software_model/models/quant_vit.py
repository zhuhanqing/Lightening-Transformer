# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu
# @Date:   2023-01-02 20:49:40
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-09 17:12:44

""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in:
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929
`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270
`FlexiViT: One Model for All Patch Sizes`
    - https://arxiv.org/abs/2212.08013
The official jax code is released and available at
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision
Acknowledgments:
  * The paper authors for releasing code and weights, thanks!
  * I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch
  * Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
  * Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2020, Ross Wightman
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from functools import partial
from collections import OrderedDict

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model
from ops.quantize import QuantAct, QuantLinear, QuantConv2d
from ops._quant_base import Qmodes
from ops.simulator import cal_coupler_wdm_error_list


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}


class QuantMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks.
    By defualt, bias is enabled.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 wbits=-1, abits=-1, offset=False,
                 input_noise_std=0, output_noise_std=0, phase_noise_std=0,
                 enable_wdm_noise=False, num_wavelength=9, channel_spacing=0.4,
                 enable_linear_noise=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)
        self.wbits = wbits
        self.abits = abits

        self.fc1 = QuantLinear(in_features, hidden_features, nbits=wbits, nbits_a=abits, mode=Qmodes.layer_wise,
                               offset=offset, input_noise_std=input_noise_std, output_noise_std=output_noise_std, phase_noise_std=phase_noise_std,
                               enable_wdm_noise=enable_wdm_noise, num_wavelength=num_wavelength, channel_spacing=channel_spacing,
                               enable_linear_noise=enable_linear_noise) if wbits <= 16 else nn.Linear(in_features, hidden_features)
        self.act = act_layer(inplace=True) if isinstance(
            act_layer, nn.ReLU) else act_layer()
        
        self.fc2 = QuantLinear(hidden_features, out_features, nbits=wbits, nbits_a=abits, mode=Qmodes.layer_wise,
                               offset=offset, input_noise_std=input_noise_std, output_noise_std=output_noise_std, phase_noise_std=phase_noise_std,
                               enable_wdm_noise=enable_wdm_noise, num_wavelength=num_wavelength, channel_spacing=channel_spacing,
                               enable_linear_noise=enable_linear_noise) if wbits <= 16 else nn.Linear(hidden_features, out_features)
        self.drop1 = nn.Dropout(drop_probs[0])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


class QuantAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 wbits=32, abits=32, headwise=False, offset=False, input_noise_std=0, output_noise_std=0, phase_noise_std=0,
                 enable_wdm_noise=False, num_wavelength=9, channel_spacing=0.4, enable_linear_noise=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.input_noise_std = input_noise_std
        self.output_noise_std = output_noise_std
        self.phase_noise_std = phase_noise_std
        self.kappa_noise = None if not enable_wdm_noise else cal_coupler_wdm_error_list(
            num_wavelength=num_wavelength, channel_spacing=channel_spacing)
        self.num_wavelength = num_wavelength
        # original version
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.q_act = ActQ(nbits_a=nbits, in_features=self.num_heads)
        # self.k_act = ActQ(nbits_a=nbits, in_features=self.num_heads)
        # self.v_act = ActQ(nbits_a=nbits, in_features=self.num_heads)
        # self.attn_act = ActQ(nbits_a=nbits, in_features=self.num_heads)

        self.qkv = QuantLinear(dim, dim * 3, bias=qkv_bias, nbits=wbits, nbits_a=abits, mode=Qmodes.layer_wise,
                               offset=offset, input_noise_std=input_noise_std, output_noise_std=output_noise_std, phase_noise_std=phase_noise_std,
                               enable_wdm_noise=enable_wdm_noise, num_wavelength=num_wavelength, channel_spacing=channel_spacing,
                               enable_linear_noise=enable_linear_noise) if wbits <= 16 else nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # head wise quantization for q, k and v
        attn_qmode = Qmodes.layer_wise if not headwise else Qmodes.kernel_wise
        self.quant_q = QuantAct(in_features=num_heads, nbits=abits, mode=attn_qmode,
                                input_noise_std=input_noise_std, offset=offset) if abits <= 16 else None
        self.quant_k = QuantAct(in_features=num_heads, nbits=abits, mode=attn_qmode,
                                input_noise_std=input_noise_std, offset=offset) if abits <= 16 else None
        self.quant_v = QuantAct(in_features=num_heads, nbits=abits, mode=attn_qmode,
                                input_noise_std=input_noise_std, offset=offset) if abits <= 16 else None
        
        # add quantization layer before and after softmax
        self.quant_attn = QuantAct(in_features=num_heads, nbits=abits, mode=attn_qmode,
                                   input_noise_std=0, offset=offset) if abits <= 16 else None
        self.quant_attn2 = QuantAct(in_features=num_heads, nbits=abits, mode=attn_qmode,
                                    input_noise_std=input_noise_std, offset=offset) if abits <= 16 else None

        self.proj = QuantLinear(dim, dim, nbits=wbits, nbits_a=abits, mode=Qmodes.layer_wise,
                                offset=offset,  input_noise_std=input_noise_std, output_noise_std=output_noise_std, phase_noise_std=phase_noise_std,
                                enable_wdm_noise=enable_wdm_noise, num_wavelength=num_wavelength, channel_spacing=channel_spacing,
                                enable_linear_noise=enable_linear_noise) if wbits <= 16 else nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def add_output_noise(self, x):
        # std is 2sigma not 1sigma, should be devided by 2
        if self.output_noise_std > 1e-5:
            noise = torch.randn_like(x).mul(
                (self.output_noise_std)).mul(x.data.abs())
            x = x + noise
        return x

    def add_phase_noise(self, x, noise_std=2):
        # DATE O2NN use 0.04 -> 0.04 * 360 / 2pi = 2.29
        # std is 2sigma not 1sigma, should be devided by 2
        if noise_std > 1e-5:
            noise = (torch.randn_like(x).mul_((noise_std) / 180 * np.pi)).cos_()
        x = x * noise

        return x

    def forward(self, x):
        kappa_noise_scale_factor = 2
        B, N, C = x.shape
        D = C // self.num_heads
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        # quant@q,k,v -> attn computation (no quant) -> quant@attn
        # quant(softmax(quant_q * quant_k))* quant(v)
        if self.quant_q is not None:
            q, k, v = self.quant_q(q), self.quant_k(k), self.quant_v(v)
        
        # enable WDM dispersion noise only during inference
        if not self.training and self.phase_noise_std > 1e-5:
            noise_q_2 = 0
            noise_k_2 = 0
            if self.kappa_noise is not None:
                kappa_noise_term = torch.tensor(self.kappa_noise).unsqueeze(0).to(
                    q.device).expand((D // self.num_wavelength) + 1, -1).reshape(-1).contiguous()[:D]
                # the result each row need to add the noise_q_2 in a element-wise way
                # the result each col need to add the noise_k_2 in a element-wise way
                alpha_q_to_k = torch.div(
                    self.quant_q.alpha, self.quant_k.alpha).reshape(self.num_heads, 1, 1)
                noise_q_2 = torch.matmul(q.square(), kappa_noise_term.unsqueeze(-1)) / (
                    alpha_q_to_k * kappa_noise_scale_factor)  # bs, head, token, 1
                noise_k_2 = (torch.matmul(k.square(), -kappa_noise_term.unsqueeze(-1)) * (
                    alpha_q_to_k / kappa_noise_scale_factor)).transpose(-2, -1)  # bs, head, 1, token
            # phase drfit
            attn = torch.empty(B, self.num_heads, N, N,
                               dtype=torch.float32).to(q.device)
            k = k.transpose(-2, -1)

            for i in range(N):
                attn[:, :, :, i] = torch.matmul(self.add_phase_noise(
                    q, noise_std=self.phase_noise_std), k[:, :, :, i].unsqueeze(-1)).squeeze(-1)
            attn = (attn + (noise_q_2 + noise_k_2)) * self.scale
        else:
            attn = ((q @ k.transpose(-2, -1))) * self.scale

        # no matter where to put scale, since we have a multiplicative noise wrt. value
        if self.output_noise_std > 1e-5:
            attn = self.add_output_noise(attn)
        attn = self.quant_attn(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = self.quant_attn2(attn)

        if not self.training and self.phase_noise_std > 1e-5:
            # dispersion noise
            noise_v_2 = 0
            noise_attn_2 = 0
            if self.kappa_noise is not None:
                kappa_noise_term = torch.tensor(self.kappa_noise).unsqueeze(0).to(
                    q.device).expand((N // self.num_wavelength) + 1, -1).reshape(-1).contiguous()[:N]
                # the result each row need to add the noise_q_2 in a element-wise way
                # the result each col need to add the noise_k_2 in a element-wise way
                alpha_attn_to_v = torch.div(
                    self.quant_attn2.alpha, self.quant_v.alpha).reshape(self.num_heads, 1, 1)
                noise_attn_2 = torch.matmul(attn.square(
                ), kappa_noise_term.unsqueeze(-1)) / (alpha_attn_to_v * kappa_noise_scale_factor)  # bs, head, token, 1
                noise_v_2 = torch.matmul(v.square().transpose(-2, -1), -kappa_noise_term.unsqueeze(-1)).transpose(
                    -2, -1) * (alpha_attn_to_v / kappa_noise_scale_factor)  # bs, head, 1, dim
            x = torch.empty(B, self.num_heads, N, C //
                            self.num_heads, dtype=torch.float32, device=q.device)
            for i in range(C // self.num_heads):
                x[:, :, :, i] = torch.matmul(self.add_phase_noise(
                    attn, noise_std=self.phase_noise_std), v[:, :, :, i].unsqueeze(-1)).squeeze(-1)
            x = x + (noise_attn_2 + noise_v_2)  # add dispersion noise
            x = x.transpose(1, 2).reshape(B, N, C)
        else:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.output_noise_std > 1e-5:
            x = self.add_output_noise(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class QuantBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 wbits=-1, abits=-1, headwise=False, input_noise_std=0, output_noise_std=0, phase_noise_std=0,
                 enable_wdm_noise=False, num_wavelength=9, channel_spacing=0.4, offset=False, enable_linear_noise=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = QuantAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                   proj_drop=drop, wbits=wbits, abits=abits,
                                   headwise=headwise, offset=offset,
                                   input_noise_std=input_noise_std, output_noise_std=output_noise_std, phase_noise_std=phase_noise_std,
                                   enable_wdm_noise=enable_wdm_noise, num_wavelength=num_wavelength, channel_spacing=channel_spacing,
                                   enable_linear_noise=enable_linear_noise)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = QuantMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                            drop=drop, wbits=wbits, abits=abits, offset=offset,
                            input_noise_std=input_noise_std, output_noise_std=output_noise_std, phase_noise_std=phase_noise_std,
                            enable_wdm_noise=enable_wdm_noise, num_wavelength=num_wavelength, channel_spacing=channel_spacing,
                            enable_linear_noise=enable_linear_noise)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class QuantPatchEmbed(nn.Module):
    """ Image to Patch Embedding. Fixed to 8bits.
    """

    def __init__(self, nbits=8, img_size=224, patch_size=16, in_chans=3, embed_dim=768, offset=False,
                 input_noise_std=0, output_noise_std=0, phase_noise_std=0, enable_wdm_noise=False, num_wavelength=9, channel_spacing=0.4, enable_linear_noise=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = QuantConv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, nbits=nbits, nbits_a=nbits, mode=Qmodes.layer_wise, offset=offset,
                                input_noise_std=input_noise_std, output_noise_std=output_noise_std, phase_noise_std=phase_noise_std,
                                enable_wdm_noise=enable_wdm_noise, num_wavelength=num_wavelength, channel_spacing=channel_spacing,
                                enable_linear_noise=enable_linear_noise)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(
                    1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class QuantVisionTransformer(nn.Module):
    """ Quantized Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            # init_values=None,
            # class_token=True,
            # no_embed_class=False,
            # pre_norm=False,
            # fc_norm=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            distilled=False,
            representation_size=None,
            weight_init='',
            block_layers=QuantBlock,
            embed_layer=QuantPatchEmbed,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            # quantization params
            wbits=32,
            abits=32,
            offset=False,
            headwise=False,
            input_noise_std=0,
            output_noise_std=0,
            phase_noise_std=0,
            enable_wdm_noise=False,
            enable_linear_noise=False,
            num_wavelength=9,
            channel_spacing=0.4):
        super().__init__()
        if wbits > 16:
            print("Use float weights.")
        else:
            print(f"Use {wbits} bit weights.")
        if abits > 16:
            print("Use float activations.")
        else:
            print(f"Use {abits} bit activations.")

        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # fixed patch embedding to 8 bits
        self.patch_embed = embed_layer(
            nbits=8, img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, offset=offset, input_noise_std=input_noise_std, output_noise_std=output_noise_std, phase_noise_std=phase_noise_std,
            enable_wdm_noise=enable_wdm_noise, num_wavelength=num_wavelength, channel_spacing=channel_spacing, enable_linear_noise=enable_linear_noise)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(
            1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        if act_layer == nn.ReLU:
            print('using relu nonlinearity')
        self.blocks = nn.ModuleList([
            QuantBlock(
                wbits=wbits, abits=abits,
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[
                    i], norm_layer=norm_layer, act_layer=act_layer,
                headwise=headwise, input_noise_std=input_noise_std, output_noise_std=output_noise_std, phase_noise_std=phase_noise_std,
                enable_wdm_noise=enable_wdm_noise, num_wavelength=num_wavelength, channel_spacing=channel_spacing, offset=offset, enable_linear_noise=enable_linear_noise)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = QuantLinear(embed_dim, num_classes, nbits=8, nbits_a=8, mode=Qmodes.layer_wise, offset=offset,
                                input_noise_std=input_noise_std, output_noise_std=output_noise_std, phase_noise_std=phase_noise_std,
                                enable_wdm_noise=enable_wdm_noise, num_wavelength=num_wavelength, channel_spacing=channel_spacing,
                                enable_linear_noise=enable_linear_noise) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = QuantLinear(self.embed_dim, self.num_classes, nbits=8,
                                         nbits_a=8, offset=offset) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            partial(_init_vit_weights,
                        head_bias=head_bias, jax_impl=True)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(
                self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_token = self.cls_token.expand(B, -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(
                x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        x = self.norm(x)

        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(
                x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

# def resize_pos_embed(posemb, posemb_new):
#     # Rescale the grid of position embeddings when loading from state_dict. Adapted from
#     # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
#     _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
#     ntok_new = posemb_new.shape[1]
#     if True:
#         posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
#         ntok_new -= 1
#     else:
#         posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
#     gs_old = int(math.sqrt(len(posemb_grid)))
#     gs_new = int(math.sqrt(ntok_new))
#     _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
#     posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
#     posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
#     posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
#     posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
#     return posemb


# def checkpoint_filter_fn(state_dict, model):
#     """ convert patch embedding weight from manual patchify + linear proj to conv"""
#     out_dict = {}
#     if 'model' in state_dict:
#         # For deit models
#         state_dict = state_dict['model']
#     for k, v in state_dict.items():
#         if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
#             # For old models that I trained prior to conv based patchification
#             O, I, H, W = model.patch_embed.proj.weight.shape
#             v = v.reshape(O, -1, H, W)
#         elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
#             # To resize pos embedding when using model at different size from pretrained weights
#             v = resize_pos_embed(v, model.pos_embed)
#         out_dict[k] = v
#     return out_dict


@register_model
def deit_tiny_patch16_224_quant(pretrained=False, **kwargs):
    model = QuantVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, distilled=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224_quant(pretrained=False, **kwargs):
    model = QuantVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, distilled=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224_quant(pretrained=False, **kwargs):
    model = QuantVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, distilled=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
