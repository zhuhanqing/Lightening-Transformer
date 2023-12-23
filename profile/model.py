# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu(hqzhu@utexas.edu)
# @Date:   1969-12-31 18:00:00
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-10 12:44:06
import torch
from torch import nn

from customized_layer import CustomizedLayerNorm, RMSNorm, CRMSNorm, LinearZeroMeanOutput

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, zero_mean_output=False):
        super().__init__()
        output_linear_layer = LinearZeroMeanOutput if zero_mean_output else nn.Linear
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            output_linear_layer(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, is_causal=False, zero_mean_output=False):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.is_causal = is_causal
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.to_out = LinearZeroMeanOutput(inner_dim, dim) if zero_mean_output else nn.Linear(inner_dim, dim)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, is_causal=False, norm_layer=nn.LayerNorm, zero_mean_output=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                norm_layer(dim, elementwise_affine=False),
                Attention(dim, heads=heads, dim_head=dim_head, is_causal=is_causal, zero_mean_output=zero_mean_output),
                norm_layer(dim, elementwise_affine=False),
                FeedForward(dim, mlp_dim, zero_mean_output=zero_mean_output)
            ]))

    def forward(self, x):
        for norm1, attn, norm2, ff in self.layers:
            x = x + attn(norm1(x))
            x = x + ff(norm2(x))
        return x


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class BERTForSequenceClassification(nn.Module):
    """No embedding is added for BERT."""
    def __init__(self, vocab_size, max_seq_length, dim, depth, heads, mlp_dim, dim_head=None, method='pre-ln', pre_rms_training=False, num_classes=2):
        super().__init__()

        if dim_head is None:
            dim_head = dim // heads
            assert dim_head * heads == dim, 'dimension must be divisible by number of heads'
        if mlp_dim is None:
            mlp_dim = dim * 4

        self.pre_rms_training = pre_rms_training and method in ['pre-apex-rms', 'pre-rms']
        if method == 'pre-ln':
            norm_layer = nn.LayerNorm
        elif method == 'pre-customized-ln':
            norm_layer = CustomizedLayerNorm
        elif method == 'no-normalization':
            norm_layer = nn.Identity
        elif method == 'pre-rms':
            norm_layer = RMSNorm
        elif method == 'pre-crms':
            norm_layer = CRMSNorm
            dim -= 1
        elif method == 'pre-crms-same-dim':
            norm_layer = CRMSNorm
        else:
            raise NotImplementedError
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, norm_layer=norm_layer, zero_mean_output=self.pre_rms_training)

        self.mlp_head = nn.Sequential(
            norm_layer(dim, elementwise_affine=False),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.transformer(x)
        
        return self.mlp_head(x)

bert_variants = {  # (dim, heads, depth, mlp_dim), if mlp_dim is Nont, then mlp_dim = dim * 4
    'Base':  (768,  12, 12, None),
    'Large': (1024, 16, 24, None),
}


class PreDefinedBERT(BERTForSequenceClassification):
    def __init__(self, vocab_size, max_seq_length, variant='Small', method='pre-ln', pre_rms_training=False, num_classes=2):
        assert variant in bert_variants, f'BERT variant {variant} not supported'
        dim, heads, depth, mlp_dim = bert_variants[variant]
        dim_head = dim // heads
        mlp_dim = dim * 4
        super().__init__(vocab_size, max_seq_length, dim, depth, heads, mlp_dim, dim_head, method, pre_rms_training, num_classes)



class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=None, method='pre-ln', pre_rms_training=False):
        super().__init__()

        if dim_head is None:
            dim_head = dim // heads
            assert dim_head * heads == dim, 'dimension must be divisible by number of heads'
        if mlp_dim is None:
            mlp_dim = dim * 4

        self.pre_rms_training = pre_rms_training and method in ['pre-apex-rms', 'pre-rms']
        if method == 'pre-ln':
            norm_layer = nn.LayerNorm
        elif method == 'pre-customized-ln':
            norm_layer = CustomizedLayerNorm
        elif method == 'no-normalization':
            norm_layer = nn.Identity
        elif method == 'pre-rms':
            norm_layer = RMSNorm
        elif method == 'pre-crms':
            norm_layer = CRMSNorm
            dim -= 1
        elif method == 'pre-crms-same-dim':
            norm_layer = CRMSNorm
        else:
            raise NotImplementedError

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        to_patch_linear = LinearZeroMeanOutput if self.pre_rms_training else nn.Linear
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            to_patch_linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, norm_layer=norm_layer, zero_mean_output=self.pre_rms_training)

        self.pool = pool

        self.mlp_head = nn.Sequential(
            norm_layer(dim, elementwise_affine=False),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        pos_embedding = self.pos_embedding - self.pos_embedding.mean(dim=-1, keepdim=True) if self.pre_rms_training else self.pos_embedding
        x = x + pos_embedding

        cls_token = self.cls_token - self.cls_token.mean(dim=-1, keepdim=True) if self.pre_rms_training else self.cls_token
        cls_tokens = repeat(cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return self.mlp_head(x)


vit_variants = {  # (dim, heads, depth, mlp_dim), if mlp_dim is Nont, then mlp_dim = dim * 4
    'Tiny':  (192,  3,  12, None),
    'Small': (384,  6,  12, None),
    'Base':  (768,  12, 12, None),
    'Large': (1024, 16, 24, None),
    'Huge':  (1280, 16, 32, None),
    'Giant': (1664, 16, 48, 8192),
    '22B':   (6144, 48, 48, None),
}


class PreDefinedViT(ViT):
    def __init__(self, image_size, patch_size, num_classes, variant='Base', pool='cls', channels=3, method='pre-ln', pre_rms_training=False):
        assert variant in vit_variants, f'ViT variant {variant} not supported'
        dim, heads, depth, mlp_dim = vit_variants[variant]
        dim_head = dim // heads
        super().__init__(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool, channels, dim_head, method, pre_rms_training)


class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_length, dim, depth, heads, mlp_dim, dim_head=None, method='pre-ln', pre_rms_training=False):
        super().__init__()

        if dim_head is None:
            dim_head = dim // heads
            assert dim_head * heads == dim, 'dimension must be divisible by number of heads'
        if mlp_dim is None:
            mlp_dim = dim * 4

        self.pre_rms_training = pre_rms_training and method in ['pre-apex-rms', 'pre-rms']
        if method == 'pre-ln':
            norm_layer = nn.LayerNorm
        elif method == 'pre-customized-ln':
            norm_layer = CustomizedLayerNorm
        elif method == 'no-normalization':
            norm_layer = nn.Identity
        elif method == 'pre-rms':
            norm_layer = RMSNorm
        elif method == 'pre-crms':
            norm_layer = CRMSNorm
            dim -= 1
        elif method == 'pre-crms-same-dim':
            norm_layer = CRMSNorm
        else:
            raise NotImplementedError

        self.wte = nn.Embedding(vocab_size, dim)
        self.wpe = nn.Embedding(max_seq_length, dim)

        self.transformer = Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, is_causal=True, norm_layer=norm_layer, zero_mean_output=self.pre_rms_training)
        self.norm = norm_layer(dim, elementwise_affine=False)
        self.final_linear = nn.Linear(dim, vocab_size)

    def forward(self, x):
        b, seq_len = x.size()

        tok_emb = self.wte(x)
        pos_id = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.wpe(pos_id)

        h = tok_emb + pos_emb
        if self.pre_rms_training:
            h = h - h.mean(dim=-1, keepdim=True)
        h = self.transformer(h)
        h = self.norm(h)
        h = self.final_linear(h)
        return h


gpt3_variants = {  # (depth, dim, heads, dim_head)
    'Small':  (12, 768,   12, 64),
    'Medium': (24, 1024,  16, 64),
    'Large':  (24, 1536,  16, 96),
    'XL':     (24, 2048,  24, 128),
    '2.7B':   (32, 2560,  32, 80),
    '6.7B':   (32, 4096,  32, 128),
    '13B':    (40, 5140,  40, 128),
    '175B':   (96, 12288, 96, 128),
}


class PreDefinedGPT3(GPT):
    def __init__(self, vocab_size, max_seq_length, variant='Small', method='pre-ln', pre_rms_training=False):
        assert variant in gpt3_variants, f'GPT3 variant {variant} not supported'
        depth, dim, heads, dim_head = gpt3_variants[variant]
        mlp_dim = dim * 4
        super().__init__(vocab_size, max_seq_length, dim, depth, heads, mlp_dim, dim_head, method, pre_rms_training)
