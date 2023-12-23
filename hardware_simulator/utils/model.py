# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu(hqzhu@utexas.edu)
# @Date:   1969-12-31 18:00:00
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-11 02:41:52
from utils.config import configs
from utils.cal_flops_for_transformer import get_infer_ops

model_zoo ={
    'deit-t': {'patch': 16, 'depth': 12, 'embed_dim': 192, 'num_heads': 3, 'mlp_ratio': 4, 'tokens': 197},
    'deit-s': {'patch': 16, 'depth': 12, 'embed_dim': 384, 'num_heads': 6, 'mlp_ratio': 4, 'tokens': 197},
    'deit-b': {'patch': 16, 'depth': 12, 'embed_dim': 768, 'num_heads': 12, 'mlp_ratio': 4, 'tokens': 197},
    'bert-b': {'depth': 12, 'embed_dim': 768, 'num_heads': 12, 'mlp_ratio': 4, 'tokens': 384},
    'bert-l': {'depth': 24, 'embed_dim': 1024, 'num_heads': 16, 'mlp_ratio': 4, 'tokens': 320},
}


class modelParams():
    # generate op list based on model param
    def __init__(self) -> None:
        super().__init__()
        
    def obtain_other_costs(self, model_name='deit-t', tokens=None):
        """Function to return estimated energy and latency for non-GEMM ops"""
        energy, latency = 0, 0
        
        tokens = model_zoo[model_name]["tokens"] if tokens is None else tokens
        softmax_ops, layer_norm_ops, residual_ops, activation_ops = get_infer_ops(
            h_d=model_zoo[model_name]["embed_dim"], 
            l_s=model_zoo[model_name]["depth"], 
            seq= tokens,
            heads=model_zoo[model_name]["num_heads"],
            head_size=model_zoo[model_name]["embed_dim"] //model_zoo[model_name]["num_heads"]
        )
        bits = 4 # default is 4 bits
        
        # softmax estimation use 
        # "high-speed and low-complexity architecture for softmax function in deep learning,” in 2018 IEEE asia pacific conference on circuits and systems (APCCAS
        softmax_energy_byte = 51.6 / 44.8 * 1e-9 # mJ/Byte
        # other uses mac * ops
        LAYER_NORM_FLOPS = 5
        # GELU: 0.5 * x * (1 + tanh(sqrt(2 / np.pi) * (x + 0.044715 * pow(x, 3))))
        ACTIVATION_FLOPS = 8

        comp_energy = (activation_ops*ACTIVATION_FLOPS + layer_norm_ops * LAYER_NORM_FLOPS + residual_ops) * 100 * 1e-12 + softmax_energy_byte * softmax_ops * bits /8
        datamovement_energy = (activation_ops + residual_ops + layer_norm_ops + softmax_ops) * 1.655e-9 * bits / 16 * 2
        energy = comp_energy + datamovement_energy
        
        # latency: 
        # estimated as memory access latency since all activations are stored on-chip
        bandwidth_sram = 1 / 0.604347 * 64 * 64* 1024 * 1024 * 1024 * 8
        clock_frequency = 500 * 1e6
        latency = (softmax_ops + layer_norm_ops + residual_ops + activation_ops ) * bits / bandwidth_sram

        return energy, latency
        
    def obtain_ops_list(self, model_name='deit-t', tokens=None):
        """Function to return the GEMM workloads dict"""
        ops_list = []
        if 'deit' in model_name:
            model_params = model_zoo[model_name]
            patch = model_params['patch']
            depth = model_params['depth']
            embed_dim = model_params['embed_dim']
            num_heads = model_params['num_heads']
            mlp_ratio = model_params['mlp_ratio']
            num_classes = 1000
            tokens = tokens if tokens is not None else model_params['tokens']
            idx = 0
            # deit family
            # first a 3 by 3 conv
            ops_list.append(
                {"idx": idx, "name": 'embed', "type": "fc", "in_features": 3*patch*patch, "out_features": embed_dim, "bs": 196}
            )
            idx += 1
            # atten block
            ops_list.append(
                {"idx": idx, "name": 'qkv', "type": "fc", "in_features": embed_dim, "out_features": embed_dim*3, "bs": tokens}
            )
            idx += 1
            ops_list.append(
                {"idx": idx, "name": 'attn', "type": "attn", "num_heads": num_heads, "embed_dim": embed_dim, "num_tokens": tokens}
            )
            idx += 1
            ops_list.append(
                {"idx": idx, "name": 'proj', "type": "fc", "in_features": embed_dim, "out_features": embed_dim, "bs": tokens}
            )
            idx += 1
            ops_list.append(
                {"idx": idx, "name": 'FFN1', "type": "fc", "in_features": embed_dim, "out_features": embed_dim*mlp_ratio, "bs": tokens}
            )
            idx += 1
            ops_list.append(
                {"idx": idx, "name": 'FFN2', "type": "fc", "in_features": embed_dim*mlp_ratio, "out_features": embed_dim, "bs": tokens}
            )
            idx += 1
            ops_list.append(
                {"idx": idx, "name": 'head', "type": "fc", "in_features": embed_dim, "out_features": num_classes, "bs": 1}
            )
        elif 'bert' in model_name:
            model_params = model_zoo[model_name]
            depth = model_params['depth']
            embed_dim = model_params['embed_dim']
            num_heads = model_params['num_heads']
            mlp_ratio = model_params['mlp_ratio']
            num_classes = 2
            tokens = tokens if tokens is not None else model_params['tokens']
            idx = 0
            ops_list.append(
                {"idx": idx, "name": 'qkv', "type": "fc", "in_features": embed_dim, "out_features": embed_dim*3, "bs": tokens}
            )
            idx += 1
            ops_list.append(
                {"idx": idx, "name": 'attn', "type": "attn", "num_heads": num_heads, "embed_dim": embed_dim, "num_tokens": tokens}
            )
            idx += 1
            ops_list.append(
                {"idx": idx, "name": 'proj', "type": "fc", "in_features": embed_dim, "out_features": embed_dim, "bs": tokens}
            )
            idx += 1
            ops_list.append(
                {"idx": idx, "name": 'FFN1', "type": "fc", "in_features": embed_dim, "out_features": embed_dim*mlp_ratio, "bs": tokens}
            )
            idx += 1
            ops_list.append(
                {"idx": idx, "name": 'FFN2', "type": "fc", "in_features": embed_dim*mlp_ratio, "out_features": embed_dim, "bs": tokens}
            )
            idx += 1
            ops_list.append(
                {"idx": idx, "name": 'head', "type": "fc", "in_features": embed_dim, "out_features": num_classes, "bs": 1}
            )
            
        return ops_list

if __name__ == "__main__":
    test = modelParams()
    ops_list = test.obtain_ops_list('bert-l', tokens=384)
    print(ops_list)
    
    test.obtain_other_costs('bert-l', tokens=384)