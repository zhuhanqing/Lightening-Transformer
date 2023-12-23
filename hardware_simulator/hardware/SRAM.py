# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu
# @Date:   2023-03-05 19:39:10
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-08 17:16:44
import math

class SRAM:
    def __init__(self, size=2048) -> None:
        
        # the largest SRAM -> 2MB
        self.max_data = size * 1024 * 8
        
        # HBM to SRAM
        self.bandwidth_dram_to_sram = 1024 * 1024 * 1024 * 1024 * 8 # 1TB/s
        self.bandwidth_sram = 1 / 0.604347* 64 * 64 * 1024 * 1024 * 1024 * 8 # based on cacti simulation
        self.bandwidth_sram_to_rf = 1024 * 1024 * 1024 * 1024 * 8 * 100 # set to inifnity
        self.clock_frequency = 500 * 1e6 # 500MHz
 
    def preload_DRAM_SRAM(self, nums=0, bits=32, bandwidth_ratio=1):
        cycle = 0
        latency = nums * bits / (self.bandwidth_dram_to_sram * bandwidth_ratio)
        cycle = math.ceil(latency * self.clock_frequency)
        if nums * bits > self.max_data:
            print('Error: loading DRAM to SRAM exceeds SRAM size')
        else:
            latency = nums * bits / (self.bandwidth_dram_to_sram * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)
        
        return cycle
    
    def load_SRAM_RF(self, nums=0, bits=32, bandwidth_ratio=1):
        cycle = 0
        latency = nums * bits / (self.bandwidth_sram_to_rf * bandwidth_ratio)
        cycle = math.ceil(latency * self.clock_frequency)
        return cycle
    
    def load_GB_SRAM(self, nums=0, bits=32, bandwidth_ratio=1):
        cycle = 0
        latency = nums * bits / (self.bandwidth_sram * bandwidth_ratio)
        cycle = math.ceil(latency * self.clock_frequency)
        return cycle