# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu
# @Date:   2023-03-28 15:37:29
# @Last Modified by:   Hanqing Zhu
# @Last Modified time: 2023-03-28 18:58:41
import math

__all__ = ["cal_coupler_wdm_error_list"]

def cal_coupler_wdm_error_list(num_wavelength, channel_spacing):
    channel_spacing = channel_spacing *1e-3
    error_list = [] # 2 * kappa - 1
    
    def coupling_length(w, g=100):
        a = -5.44
        b = 3.53
        c = 0.185
        d = 0.15
        
        L_c = (a * (w - 1.55) + b) * math.exp(g / 1000 / (c * (w - 1.55) + d))
        
        return L_c
    odd_num_wavelength = True if num_wavelength % 2 == 1 else False
    
    for wave_length in range(num_wavelength):
        if odd_num_wavelength:
            wave_length = 1.55 + channel_spacing * (wave_length - (num_wavelength // 2))
        else:
            if wave_length < num_wavelength // 2:
                wave_length = 1.55 + channel_spacing * (wave_length - (num_wavelength // 2))
            else:
                wave_length = 1.55 + channel_spacing * (wave_length - (num_wavelength // 2) + 1)
        kappa = math.sin(math.pi / 4 * coupling_length(1.55) / coupling_length(wave_length)) ** 2
        error_list.append(2 * kappa - 1)
        
    return error_list
