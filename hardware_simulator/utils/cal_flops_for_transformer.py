# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu(hqzhu@utexas.edu)
# @Date:   1969-12-31 18:00:00
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-09 01:18:58
"""Computes the flops needed for training/running transformer networks.
https://github.com/google-research/electra/blob/master/flops_computation.py
"""

# We checked this code with TensorFlow"s FLOPs counting, although we had to
# correct for this issue: https://github.com/tensorflow/tensorflow/issues/22071
# Assumptions going into the FLOPs counting
#   - An "operation" is a mathematical operation, not a machine instruction. So
#     an "exp" takes one opp like and add, even though in practice an exp
#     might be slower. This is not too bad an assumption because
#     matrix-multiplies dominate the compute for most models, so minor details
#     about activation functions don"t matter too much. Similarly, we count
#     matrix-multiplies as 2*m*n flops instead of m*n, as one might if
#     if considering fused multiply-add ops.
#   - Backward pass takes the same number of FLOPs as forward pass. No exactly
#     right (e.g., for softmax cross entropy loss the backward pass is faster).
#     Importantly, it really is the same for matrix-multiplies, which is most of
#     the compute anyway.
#   - We assume "dense" embedding lookups (i.e., multiplication by a one-hot
#     vector). On some hardware accelerators, these dense operations are
#     actually faster than sparse lookups.
# Please open a github issue if you spot a problem with this code!

# I am not sure if the below constants are 100% right, but they are only applied
# to O(hidden_size) activations, which is generally a lot less compute than the
# matrix-multiplies, which are O(hidden_size^2), so they don't affect the total
# number of FLOPs much.

# random number, >=, multiply activations by dropout mask, multiply activations
# by correction (1 / (1 - dropout_rate))
DROPOUT_FLOPS = 4

# compute mean activation (sum), computate variance of activation
# (square and sum), bias (add), scale (multiply)
LAYER_NORM_FLOPS = 5

# GELU: 0.5 * x * (1 + tanh(sqrt(2 / np.pi) * (x + 0.044715 * pow(x, 3))))
ACTIVATION_FLOPS = 8

# max/substract (for stability), exp, sum, divide
SOFTMAX_FLOPS = 5

__all__ = [
    "get_infer_ops"
]

class TransformerHparams(object):
    """Computes the train/inference FLOPs for transformers."""

    def __init__(self, h, l, s=512, v=30522, e=None, i=None, heads=None,
                 head_size=None, output_frac=0.15625, sparse_embed_lookup=False,
                 decoder=False):
        self.h = h  # hidden size
        self.l = l  # number of layers
        self.s = s  # sequence length
        self.v = v  # vocab size
        self.e = h if e is None else e  # embedding size
        self.i = h * 4 if i is None else i  # intermediate size
        self.kqv = h if head_size is None else head_size * heads  # attn proj sizes
        # attention heads
        self.heads = max(h // 64, 1) if heads is None else heads
        self.output_frac = output_frac  # percent of tokens using an output softmax
        self.sparse_embed_lookup = sparse_embed_lookup  # sparse embedding lookups
        self.decoder = decoder  # decoder has extra attn to encoder states

        self.residual_flops = 0
        self.activation_flops = 0
        self.layer_norm_flops = 0
        self.softmax_flops = 0

    def get_block_flops(self):
        """Get the forward-pass FLOPs for a single transformer block."""
        attn_mul = 2 if self.decoder else 1
        block_flops = dict(
            kqv=3 * 2 * self.h * self.kqv * attn_mul,
            kqv_bias=3 * self.kqv * attn_mul,
            attention_scores=2 * self.kqv * self.s * attn_mul,
            attn_softmax=SOFTMAX_FLOPS * self.s * self.heads * attn_mul,
            attention_dropout=DROPOUT_FLOPS * self.s * self.heads * attn_mul,
            attention_scale=self.s * self.heads * attn_mul,
            attention_weighted_avg_values=2 * self.h * self.s * attn_mul,
            attn_output=2 * self.h * self.h * attn_mul,
            attn_output_bias=self.h * attn_mul,
            attn_output_dropout=DROPOUT_FLOPS * self.h * attn_mul,
            attn_output_residual=self.h * attn_mul,
            attn_output_layer_norm=LAYER_NORM_FLOPS * attn_mul,
            intermediate=2 * self.h * self.i,
            intermediate_act=ACTIVATION_FLOPS * self.i,
            intermediate_bias=self.i,
            output=2 * self.h * self.i,
            output_bias=self.h,
            output_dropout=DROPOUT_FLOPS * self.h,
            output_residual=self.h,
            output_layer_norm=LAYER_NORM_FLOPS * self.h,
        )

        self.softmax_flops += self.s * self.s * self.heads * attn_mul # tokens * tokens * head
        self.residual_flops += self.s * (self.h + self.h) # tokens * hidden size
        self.layer_norm_flops += self.s * (self.h + 1) # tokens * hidden_size 
        self.activation_flops += self.s * self.i # GELU tokens * hidden_size * 4

        return sum(block_flops.values()) * self.s

    def get_embedding_flops(self, output=False):
        """Get the forward-pass FLOPs the transformer inputs or output softmax."""
        embedding_flops = {}
        if output or (not self.sparse_embed_lookup):
            embedding_flops["main_multiply"] = 2 * self.e * self.v
        # input embedding post-processing
        if not output:
            embedding_flops.update(dict(
                tok_type_and_position=2 * self.e * (self.s + 2),
                add_tok_type_and_position=2 * self.e,
                emb_layer_norm=LAYER_NORM_FLOPS * self.e,
                emb_dropout=DROPOUT_FLOPS * self.e
            ))
        # projection layer if e != h
        if self.e != self.h or output:
            embedding_flops.update(dict(
                hidden_kernel=2 * self.h * self.e,
                hidden_bias=self.e if output else self.h
            ))
            # extra hidden layer and output softmax
            if output:
                embedding_flops.update(dict(
                    hidden_activation=ACTIVATION_FLOPS * self.e,
                    hidden_layernorm=LAYER_NORM_FLOPS * self.e,
                    output_softmax=SOFTMAX_FLOPS * self.v,
                    output_target_word=2 * self.v
                ))
                return self.output_frac * sum(embedding_flops.values()) * self.s
        return sum(embedding_flops.values()) * self.s

    def get_binary_classification_flops(self):
        classification_flops = dict(
            hidden=2 * self.h * self.h,
            hidden_bias=self.h,
            hidden_act=ACTIVATION_FLOPS * self.h,
            logits=2 * self.h
        )
        return sum(classification_flops.values()) * self.s

    def get_train_flops(self, batch_size, train_steps, discriminator=False):
        """Get the FLOPs for pre-training the transformer."""
        # 2* for forward/backward pass
        return 2 * batch_size * train_steps * (
            (self.l * self.get_block_flops()) +
            self.get_embedding_flops(output=False) +
            (self.get_binary_classification_flops() if discriminator else
             self.get_embedding_flops(output=True))
        )

    def get_infer_flops(self):
        """Get the FLOPs for running inference with the transformer on a
        classification task."""
        (self.l * self.get_block_flops()) + self.get_embedding_flops(output=False) + self.get_binary_classification_flops()


def get_electra_train_flops(
        h_d, l_d, h_g, l_g, batch_size, train_steps, tied_embeddings,
        e=None, s=512, output_frac=0.15625):
    """Get the FLOPs needed for  pre-training ELECTRA."""
    if e is None:
        e = h_d
    disc = TransformerHparams(
        h_d, l_d, s=s, e=e,
        output_frac=output_frac).get_train_flops(batch_size, train_steps, True)
    gen = TransformerHparams(
        h_g, l_g, s=s, e=e if tied_embeddings else None,
        output_frac=output_frac).get_train_flops(batch_size, train_steps)
    return disc + gen

def get_infer_ops(
    h_d, l_s, seq, heads, head_size=64
):
    """Get the ops needed for Transformer inference. Softmax, layernorm, residual add, activation"""
    estimator = TransformerHparams(h=h_d, l=l_s, s=seq, heads=heads, head_size=head_size)
    estimator.get_infer_flops()
    
    return estimator.softmax_flops, estimator.layer_norm_flops, estimator.residual_flops, estimator.activation_flops
