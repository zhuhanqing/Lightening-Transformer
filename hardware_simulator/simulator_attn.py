# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu
# @Date:   2023-02-12 22:43:23
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-13 19:31:59

import os
import argparse
import math
import logging
import csv
from collections import OrderedDict
from utils.config import configs
from utils.general import ensure_dir

from hardware.photonic_mrr_bank import PhotonicMRRBank
from hardware.photonic_crossbar import PhotonicCrossbar
from hardware.SRAM import SRAM
logging.basicConfig(level=logging.INFO)

class attnPrediction():
    """Simulator for Multi-head attention. QK^T and AV.
    Q* K^T -> [num_heads, N, head_dim] * [num_heads, N, head_dim] -> [num_heads, N, N]
    S * V -> [num_heads, N, N] * [num_heads, N, head_dim] -> [num_heads, N, head_dim]
    """
    def __init__(self, op_info, configs=None) -> None:
        super().__init__()
        
        self.num_heads = op_info['num_heads']
        self.dim = op_info['embed_dim']
        self.num_tokens = op_info['num_tokens']
        self.head_dim = self.dim // self.num_heads

        # bits
        self.in_bit = configs.core.precision.in_bit
        self.w_bit = configs.core.precision.w_bit
        self.act_bit = configs.core.precision.act_bit

        self.core_type = configs.core.type
        self.num_tiles = configs.arch.num_tiles
        self.num_pe_per_tile = configs.arch.num_pe_per_tile

        # arch-level params
        # full_range_support_factor
        # weight_reuse_factor
        # time_accum_factor
        self.full_range_support_factor = 1 if self.core_type == "dota" else configs.arch.full_range_support_factor
        self.weight_reuse_factor = 1 if self.core_type == "dota" else configs.arch.weight_reuse_factor
        self.time_accum_factor = configs.arch.time_accum_factor if self.core_type == "dota" else 1

        # input modulation sharing and ADC sharing flag
        if self.core_type == "dota":
            self.input_mod_sharing_flag = True if configs.arch.input_mod_sharing_flag == 1 else False
            self.adc_share_flag = True if configs.arch.adc_share_flag == 1 else False
            self.disable_crossbar_topology = True if configs.arch.disable_crossbar_topology == 1 else False
        else:
            self.adc_share_flag = False
            self.input_mod_sharing_flag = False

        logging.info(
            f"Use {self.core_type} to build accelerator with {self.num_tiles} tiles ({self.num_pe_per_tile } PEs)...")

        if self.core_type == "dota":
            self.hw = PhotonicCrossbar(
                core_width=configs.core.width,
                core_height=configs.core.height,
                num_wavelength=configs.core.num_wavelength,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                act_bit=self.act_bit,
                config=configs
            )
        elif self.core_type == "mrrbank":
            self.hw = PhotonicMRRBank(
                core_width=configs.core.width,
                core_height=configs.core.height,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                act_bit=self.act_bit,
                config=configs
            )
        else:
            raise NotImplementedError

        # set the work frequency of photonic core
        self.work_freq = configs.core.work_freq
        self.hw.set_work_frequency(self.work_freq)
        self.adder_power = 0.2 / 4.39  # follow tech node scaling law to 14nm # mW @ ISAAC
        # self.adder_power = 0.2
        self.data_movement_DRAM = configs.arch.datamovement.DRAM
        self.data_movement_DRAM_GB = configs.arch.datamovement.DRAM_GB
        self.data_movement_GB2 = configs.arch.datamovement.GB2
        self.data_movement_GB1 = configs.arch.datamovement.GB1
        self.data_movement_NoC = configs.arch.datamovement.NoC
        self.data_movement_RF = configs.arch.datamovement.RF
        
        self.local_buffer_size = configs.arch.memory_size.M2_buffer_size # 4KB

        self.SRAM = SRAM()
        
        # save results to dict
        self.energy_dict = OrderedDict()
        self.latency_dict = OrderedDict()

    def get_latency_crossbar(self, matrix_dim1, matrix_dim2, print_msg=False):
        """
        Get the latency of running attn on dota array. The memory system is carefully designed to avoid being memory-bounded.
        Latency simulation follows https://github.com/GATECH-EIC/ViTCoD.
        """
        latency_dict = {}
        latency_dict['total'] = [0, 0]
        latency_dict['comp'] = [0, 0]
        latency_dict['datamovement'] = [0, 0]
        # latency path: mem -> DRAM -> SRAM1 -> SRAM2 -> RF -> comp
        h1, N1, D1 = matrix_dim1
        h2, D2, N2 = matrix_dim2
        assert (h1 == h2) and (D1 == D2), f"Got incorrect matrix dimension."
        h = h1
        D = D1
        # mat1 is on the height side
        # mat2 is on the width side
        iter_N1 = math.ceil(N1 / self.hw.core_height)
        iter_N2 = math.ceil(N2 / self.hw.core_width)
        iter_D = math.ceil(D / self.hw.num_wavelength)

        # computation cycles
        # update all cycles to ceil
        cycles_computations = math.ceil((
            iter_D * iter_N1 * iter_N2) * h / (self.num_tiles * self.num_pe_per_tile))
        # GHZ * 1e-6 -> ms
        latency_comp = cycles_computations * 1 / self.work_freq * 1e-6
        
        latency_dict['comp'][0] = cycles_computations
        latency_dict['comp'][1] = latency_comp

        # memory cycles:
        # key bottleneck is the off-chip DRAM, only consider this part
        # and our activation is fully on-chip
        # follow VitCod for the estimation: https://github.com/GATECH-EIC/ViTCoD/blob/main/Hardware/Simulator/ViT_FFN.py
        cycles_preload_data_dram_sram = 0
        cycles_preload_data_GB_SRAM = 0
        cylces_load_data_sram_rf = 0
        head_parallel_factor = 1  # parallel
        # implement the tiling algorithm in our paper
        for _head in range(h1 // head_parallel_factor):
            for _iter_N1 in range(math.ceil(N1 / (self.num_tiles * self.hw.core_height))):
                # pre-load a chunk of data from dram to the largest sram
                # other data transfer latency is neligble
                # data should be core_height * D + core_width * D (M1 and M2)
                num_bytes_dram_sram = (
                    self.hw.core_height * D + self.hw.core_width * D)
                cycles_preload_data_dram_sram += self.SRAM.preload_DRAM_SRAM(
                    num_bytes_dram_sram, bits=self.in_bit, bandwidth_ratio=1/self.num_tiles) # mult-tile shares the same one
                num_bytes_GB_sram = self.hw.core_height * D * self.num_tiles + D * N2
                cycles_preload_data_GB_SRAM += self.SRAM.load_GB_SRAM(num_bytes_GB_sram, bits=self.in_bit, bandwidth_ratio=1)
                for _iter_N2 in range(iter_N2):
                    for _iter_D in range(iter_D // self.num_pe_per_tile):
                        pass

        cycles_preload_data_dram_sram = 0
        cylces_load_data_sram_rf = 0
        cycles_memory = max(cycles_preload_data_dram_sram,
                            cylces_load_data_sram_rf,
                            cycles_preload_data_GB_SRAM,)
        latency_memory = cycles_memory * 1 / self.SRAM.clock_frequency * 1e3

        latency_dict['datamovement'][0] = cycles_memory
        latency_dict['datamovement'][1] = latency_memory
        
        
        latency = max(latency_comp, latency_memory)
        latency_dict['total'][0] = -1
        latency_dict['total'][1] = latency
        # latency = latency_comp
        
        if print_msg:
            print("**" * 10)
            print("Latency estimation for ATTN")
            print(f"M1 size {h1} * {N1} * {D1}")
            print(f"M1 size {h2} * {D2} * {N2}")
            print(f"The loop number is {iter_N1}, {iter_N2}, {iter_D}")
            print(
                f"computation cycles: {cycles_computations} at freq {self.work_freq} GHz")
            print(
                f"memory cycles: {cycles_memory} at freq {self.SRAM.clock_frequency*1e-9} GHz")

        return latency, latency_dict

    def get_latency_mrr_bank(self, matrix_dim1, matrix_dim2, print_msg=False):
        """Latency estimation for mrr bank baselines, assuming a computation-bounded system"""
        latency_dict = {}
        latency_dict['total'] = [0, 0]
        latency_dict['comp'] = [0, 0]
        latency_dict['datamovement'] = [0, 0]
        h1, N1, D1 = matrix_dim1
        h2, D2, N2 = matrix_dim2
        assert (h1 == h2) and (D1 == D2), f"Got incorrect matrix dimension."
        h = h1
        D = D1

        iter_N1 = math.ceil(N1 / self.hw.core_height)
        iter_N2 = N2
        iter_D = math.ceil(D / self.hw.core_width)

        if self.full_range_support_factor == 2:
            full_range_support_factor_x = 2  # -> M2
            full_range_support_factor_w = 1  # assume can be full-range -> M1
        elif self.full_range_support_factor == 4:
            full_range_support_factor_x = 2
            full_range_support_factor_w = 2  # assume can be full-range
        else:
            full_range_support_factor_w, full_range_support_factor_x = 1, 1

        # computation cycles:
        cycles_computations = math.ceil(iter_D * iter_N1 * iter_N2 * h / (self.num_tiles * self.num_pe_per_tile)) * self.full_range_support_factor
        # print(
        #     f"computation cycles: {cycles_computations} at freq {self.work_freq} GHz")
        latency_comp = cycles_computations * 1 / self.work_freq * 1e-6
        
        latency_dict['comp'][0] = cycles_computations
        latency_dict['comp'][1] = latency_comp

        # assume a ideal baseline without being bounded by memory
        cycles_preload_data_dram_sram = 0
        cylces_load_data_sram_rf = 0
        cycles_preload_data_GB_SRAM = 0
        cycles_memory = cycles_preload_data_GB_SRAM
        latency_memory = cycles_memory * 1 / self.SRAM.clock_frequency * 1e3
        
        latency_dict['datamovement'][0] = cycles_memory
        latency_dict['datamovement'][1] = latency_memory

        latency = max(latency_comp, latency_memory)
        latency_dict['total'][0] = -1
        latency_dict['total'][1] = latency

        if print_msg:
            print("**" * 10)
            print("Latency estimation for ATTN")
            print(f"M1 size {h1} * {N1} * {D1}")
            print(f"M1 size {h2} * {D2} * {N2}")
            print(f"The loop number is {iter_N1}, {iter_N2}, {iter_D}")
            print(
                f"computation cycles: {cycles_computations} at freq {self.work_freq} GHz")
            print(
                f"memory cycles: {cycles_memory} at freq {self.SRAM.clock_frequency*1e-9} GHz")
        
        print(
            f"memory cycles: {cycles_memory} at at freq {self.SRAM.clock_frequency*1e-9} GHz")
        
        return latency, latency_dict
        

    def get_energy_crossbar(self, matrix_dim1, matrix_dim2, print_msg=False):
        """Energy estimation of our DPTC with mat1 * mat2 in OS dataflow"""
        energy_dict = {}
        energy_dict['comp'] = {i: [0, 0] for i in ["total", "laser", "ADC", "adder"]}
        energy_dict['datamovement'] = { i:[0, 0] for i in ["total", "RF", "LB", "GB", "DRAM"] }
        h1, N1, D1 = matrix_dim1
        h2, D2, N2 = matrix_dim2
        assert (h1 == h2) and (D1 == D2), f"Got incorrect matrix dimension."
        h = h1
        D = D1

        num_computation = h * N1 * N2 * D
        num_ifmap1 = h * N1 * D
        num_ifmap2 = h * D * N2
        num_ofmap = h * N1 * N2

        iter_N1 = math.ceil(N1 / self.hw.core_height)
        iter_N2 = math.ceil(N2 / self.hw.core_width)
        iter_D = math.ceil(D / self.hw.num_wavelength)
        
        if print_msg:
            print("**" * 10)
            print("Energy estimation for ATTN")
            print(f"M1 matrix dims:{h1} * {N1} * {D1}")
            print(f"M2 matrix dims:{h2} * {N2} * {D2}")
            print(
                f"The loop number of N1, N2, D is {iter_N1}, {iter_N2}, {iter_D}")

        ##### energy model for computation part
        # cal hw methods to calculate energy of each component
        D2A_energy = self.hw.cal_D2A_energy()
        TX_energy = self.hw.cal_TX_energy()
        laser_energy = self.hw.cal_laser_energy()
        comp_energy = self.hw.cal_comp_energy()
        A2D_energy = self.hw.cal_A2D_energy()
        RX_energy = self.hw.cal_RX_energy()
        Adder_energy = self.adder_power / self.work_freq # mW/5G -> pJ # mJ = pJ * 1e-9

        # laser energy
        energy_comp_laser = laser_energy * h * \
            (iter_N1 * iter_N2 * iter_D) * 1e-9
            
        # modulation energy (DAC + TX)
        # The block matrix is changed over time scale for both inputs
        if self.disable_crossbar_topology:
            # only broadcast operand2: mat2
            energy_comp_D2A_1 = D2A_energy * h * N1 * N2 * D * 1e-9
            energy_comp_D2A_2 = h * (D2A_energy * iter_N1 * N2 * D * 1e-9) / \
                self.num_tiles if self.input_mod_sharing_flag else h * (
                    D2A_energy * iter_N1 * N2 * D * 1e-9)
            energy_comp_TX_1 = TX_energy * h * N1 * N2 * D * 1e-9
            energy_comp_TX_2 = h * (TX_energy * N2 * iter_N1 * D * 1e-9) / \
                self.num_tiles if self.input_mod_sharing_flag else h * (
                    TX_energy * N2 * iter_N1 * D * 1e-9)
        else:
            # both matrix 1 and matrix 2 is shared enabled by dota topology
            # the shared times is the core_height and core_width
            energy_comp_D2A_1 = D2A_energy * h * N1 * D * iter_N2 * 1e-9
            energy_comp_D2A_2 = h * (D2A_energy * iter_N1 * N2 * D * 1e-9) / \
                self.num_tiles if self.input_mod_sharing_flag else h * (
                    D2A_energy * iter_N1 * N2 * D * 1e-9)
            energy_comp_TX_1 = TX_energy * h * N1 * iter_N2 * D * 1e-9
            energy_comp_TX_2 = h * (TX_energy * N2 * iter_N1 * D * 1e-9) / \
                self.num_tiles if self.input_mod_sharing_flag else h * (
                    TX_energy * N2 * iter_N1 * D * 1e-9)

        energy_comp_D2A = energy_comp_D2A_1 + energy_comp_D2A_2
        energy_comp_TX = energy_comp_TX_1 + energy_comp_TX_2

        # computation energy (comp)
        energy_comp_comp = comp_energy * num_computation * 1e-9

        # output (RX + A2D + adder: time integral & adc sharing (num_pe))
        time_accum_factor = min(
            self.time_accum_factor, math.ceil(D / (self.num_pe_per_tile * self.hw.num_wavelength)))
        
        # consider time integral: TIA + ADC energy should be reduced at the same time since the times we call ADC being reduced by #time_accum_factor
        if self.adc_share_flag:
            # / self.num_pe_per_tile
            ps_size = N1 * N2 * math.ceil(math.ceil(iter_D / time_accum_factor) / self.num_pe_per_tile)
            energy_comp_ADC = h * A2D_energy * \
                (ps_size) * 1e-9 
            energy_comp_TIA = h * self.hw.TIA_energy * \
                (ps_size) * 1e-9 
            energy_comp_adder = h * Adder_energy * \
                (ps_size) * 1e-9 
        else:
            # / time_accum_factor
            ps_size = N1 * N2 * math.ceil(iter_D / time_accum_factor)
            energy_comp_ADC = h * A2D_energy * \
                (ps_size) * 1e-9 
            energy_comp_TIA = h * self.hw.TIA_energy * \
                (ps_size) * 1e-9 
            energy_comp_adder = h * Adder_energy * \
                (ps_size) * 1e-9

        energy_comp_detection = h * self.hw.photo_detector_energy * (N1 * N2 * iter_D) * 1e-9
        
        energy_comp_output = energy_comp_ADC + energy_comp_TIA + energy_comp_adder + energy_comp_detection
        energy_comp = energy_comp_laser + energy_comp_D2A + \
            energy_comp_TX + energy_comp_comp + energy_comp_output

        ##### energy for datamovement
        self.num_byte = self.in_bit / 16
        
        ## RF part, consider both read and write
        if self.disable_crossbar_topology:
            # inputs: mat1
            energy_dm_RF = h * self.num_byte * \
                self.data_movement_RF * (N2 * N1 * D) * 2
            # inputs: mat2
            energy_dm_RF += h * self.num_byte * self.data_movement_RF * \
                (iter_N1 * N2 * D / self.num_tiles) * 2 if self.input_mod_sharing_flag else h * \
                self.num_byte * self.data_movement_RF * (iter_N1 * N2 * D) * 2
        else:
            # inputs: mat1
            energy_dm_RF = h * self.num_byte * \
                self.data_movement_RF * (iter_N2 * N1 * D) * 2
            # inputs: mat2
            energy_dm_RF += h * self.num_byte * self.data_movement_RF * \
                (iter_N1 * N2 * D / self.num_tiles) * 2 if self.input_mod_sharing_flag else h * \
                self.num_byte * self.data_movement_RF * (iter_N1 * N2 * D) * 2
        # partial-sums: assumes the results in the tile are accumulated first and then write back to buffer: N1 * N2 * iter_D / accum / num_pe_per_tile
        energy_dm_RF += h * self.num_byte * self.data_movement_RF * (N1 * N2 * math.ceil(math.ceil(iter_D / time_accum_factor) / self.num_pe_per_tile)) * 2
        
        ## Noc due to adder part
        energy_dm_Noc =  h * self.num_byte * ps_size * self.data_movement_NoC
        energy_dm_RF += energy_dm_Noc
        
        # GLB1: read from GLB1 (energy_dm_GLB1_input) and write from GLB2 to GLB1 (energy_dm_GLB1_input_from_GLB2)
        # fill a tiled of data into local buffer and complete all computations with it to avoid frequent reload
        output_load_time = max(self.hw.core_height * D * self.in_bit / 8 / self.local_buffer_size, 1)

        if self.input_mod_sharing_flag:
            energy_dm_GLB1_input = h * self.num_byte * \
                self.data_movement_GB1 * \
                (N1 * D * iter_N2 + iter_N1 * D * N2 / self.num_tiles)
            energy_dm_GLB1_input_from_GLB2 = h * self.num_byte * \
                self.data_movement_GB1 * \
                 (N1 * D  + N2 * D * iter_N1 / self.num_tiles)
        else:
            energy_dm_GLB1_input = h * self.num_byte * \
                self.data_movement_GB1 * \
                (N1 * D * iter_N2 + iter_N1 * D * N2)
            energy_dm_GLB1_input_from_GLB2 = h * self.num_byte * \
                self.data_movement_GB1 * \
                (N1 * D + + N2 * D * iter_N1)
        # data will be write to GLB
        # NOTE(hqzhu): since we assume we don't spatially distribute psum to different tile -> psum don't have access to GLB
        energy_dm_GLB1_output = h * self.num_byte * \
            self.data_movement_GB1 * (N1 * N2) * (2 * output_load_time - 1)
        energy_dm_GLB1 = energy_dm_GLB1_input + energy_dm_GLB1_input_from_GLB2 + energy_dm_GLB1_output
        # GLB2: play the role of privious DRAM to broadcast data: only read cost for activation, read and write cost for weights
        # input: M1 -> not reloaded, M2 -> roload times: iter_N1 * iter_N2 * iter_D / num_tiles * num_tiles (broadcast)
        # output: the same as DRAM
        if self.input_mod_sharing_flag:
            energy_dm_GLB2 = h * self.num_byte * self.data_movement_GB2 * ((
                N1 * N2 * (2 * output_load_time - 1)) + self.data_movement_GB2 * (N1 * D + N2 * D * iter_N1 / self.num_tiles))
        else:
            energy_dm_GLB2 = h * self.num_byte * self.data_movement_GB2 * ((
                N1 * N2 * (2 * output_load_time - 1)) + self.data_movement_GB2 * (N1 * D + N2 * D * iter_N1))

        # DRAM
        # assume activations is in SRAM, only need to load weights from DRAM for linear layer part
        # energy_dm_DRAM = h * self.num_byte * (self.data_movement_DRAM * (
        #     N1 * N2) + self.data_movement_DRAM_GB * (N1*D + N2 * D * iter_N1 / self.num_tiles))
        energy_dm_DRAM = 0

        energy_dm = energy_dm_DRAM + energy_dm_GLB1 + energy_dm_GLB2 + energy_dm_RF

        energy = energy_dm + energy_comp
        
        if print_msg:
            print(f"Overall estimated energy cost {energy} mJ")
            print(
                f"--Computation energy cost is {energy_comp} mJ  {energy_comp / energy * 100 :.2f} %")
            print(
                f"----Laser energy cost is {energy_comp_laser} mJ  {energy_comp_laser / energy * 100 :.2f} %")
            print(
                f"----Ouput energy cost is {energy_comp_output} mJ  {energy_comp_output / energy * 100 :.2f} %")
            print(
                f"----D2A energy cost is {energy_comp_D2A} mJ  {energy_comp_D2A / energy * 100 :.2f} %")
            print(
                f"----TX energy cost is {energy_comp_TX} mJ  {energy_comp_TX / energy * 100 :.2f} %")
            print(
                f"--Datamovement energy cost is {energy_dm} mJ {energy_dm / energy * 100 :.2f} %")
            print(
                f"----RF energy cost is {energy_dm_RF} mJ {energy_dm_RF / energy * 100 :.2f} %")
            print(
                f"----GLB1 energy cost is {energy_dm_GLB1} mJ {energy_dm_GLB1 / energy * 100 :.2f} %")
            print(
                f"----GLB2 energy cost is {energy_dm_GLB2} mJ {energy_dm_GLB2 / energy * 100 :.2f} %")
            print(
                f"----DRAM energy cost is {energy_dm_DRAM} mJ {energy_dm_DRAM / energy * 100 :.2f} %")

        # save to dict
        # comp
        energy_dict['comp']['total'] = [energy_comp, round(energy_comp / energy * 100, 2)]
        energy_dict['comp']['laser'] = [energy_comp_laser, round(energy_comp_laser / energy * 100, 2)]
        energy_dict['comp']['TX_1'] = [energy_comp_TX_1, round(energy_comp_TX_1 / energy * 100, 2)]
        energy_dict['comp']['DAC_1'] = [energy_comp_D2A_1, round(energy_comp_D2A_1 / energy * 100, 2)]
        energy_dict['comp']['TX_2'] = [energy_comp_TX_2, round(energy_comp_TX_2 / energy * 100, 2)]
        energy_dict['comp']['DAC_2'] = [energy_comp_D2A_2, round(energy_comp_D2A_2 / energy * 100, 2)]
        energy_dict['comp']['RX-Detector'] = [energy_comp_detection, round(energy_comp_detection / energy * 100, 2)]
        energy_dict['comp']['RX-TIA'] = [energy_comp_TIA, round(energy_comp_TIA / energy * 100, 2)]
        energy_dict['comp']['ADC'] = [energy_comp_ADC, round(energy_comp_ADC / energy * 100, 2)]
        energy_dict['comp']['adder'] = [energy_comp_adder, round(energy_comp_adder / energy * 100, 2)]
        # memory
        energy_dict['datamovement']['total'] = [energy_dm, round(energy_dm / energy * 100, 2)]
        energy_dict['datamovement']['RF'] = [energy_dm_RF, round(energy_dm_RF / energy * 100, 2)]
        energy_dict['datamovement']['LB'] = [energy_dm_GLB1, round(energy_dm_GLB1 / energy * 100, 2)]
        energy_dict['datamovement']['GB'] = [energy_dm_GLB2, round(energy_dm_GLB2 / energy * 100, 2)]
        energy_dict['datamovement']['DRAM'] = [energy_dm_DRAM, round(energy_dm_DRAM / energy * 100, 2)]

        return energy, energy_dict

    def get_energy_mrrbank(self, matrix_dim1, matrix_dim2, print_msg=False):
        """Energy estimation for mrrbank. Set to weight-stationary flow for matrix 1"""
        energy_dict = {}
        energy_dict['comp'] = {i: [0, 0] for i in ["total", "laser", "ADC", "RX", "adder"]}
        energy_dict['datamovement'] = { i:[0, 0] for i in ["total", "RF", "LB", "GB", "DRAM"] }
        # weight_reuse_factor = 16 # the weight matrix will be reused for how many cols of input
        h1, N1, D1 = matrix_dim1
        h2, D2, N2 = matrix_dim2
        assert (h1 == h2) and (D1 == D2), f"Got incorrect matrix dimension."
        h = h1
        D = D1
        
        # Set weight reuse for all dimensions
        if self.weight_reuse_factor == -1:
            self.weight_reuse_factor = N2

        num_computation = h * N1 * N2 * D

        # map a N * N weight matrix, then for loop N2
        iter_N1 = math.ceil(N1 / self.hw.core_height)
        iter_N2 = N2
        iter_D = math.ceil(D / self.hw.core_width)

        if print_msg:
            print("**" * 10)
            print("Energy estimation for ATTN")
            print(f"M1 matrix dims:{h1} * {N1} * {D1}")
            print(f"M2 matrix dims:{h2} * {N2} * {D2}")
            print(
                f"The loop number of N1, N2, D is {iter_N1}, {iter_N2}, {iter_D}")

        ##### energy model for computation part
        D2A_energy = self.hw.cal_D2A_energy()
        TX_energy = self.hw.cal_TX_energy()
        laser_energy = self.hw.cal_laser_energy()
        comp_energy_static, comp_energy_dynamic = self.hw.cal_comp_energy()
        A2D_energy = self.hw.cal_A2D_energy()
        RX_energy = self.hw.cal_RX_energy()
        Adder_energy = self.adder_power / self.work_freq

        # whether weight and input can be full-range
        if self.full_range_support_factor == 2:
            full_range_support_factor_x = 2
            full_range_support_factor_w = 1  # assume can be full-range
        elif self.full_range_support_factor == 4:
            full_range_support_factor_x = 2
            full_range_support_factor_w = 2
        else:
            full_range_support_factor_w, full_range_support_factor_x = 1, 1

        energy_comp_laser = laser_energy * h * \
            (iter_N1 * iter_N2 * iter_D) * 1e-9 * self.full_range_support_factor

        ## modulation energy (DAC + TX)
        # mat1 is the static operand
        # weight is shared with multiple data
        ## DAC
        energy_comp_D2A_1= D2A_energy * \
            (num_computation // self.weight_reuse_factor) * \
            1e-9 * full_range_support_factor_w
        energy_comp_D2A_2 = D2A_energy * h * \
            (iter_N1 * N2 * D) * 1e-9 * full_range_support_factor_x
        energy_comp_D2A = energy_comp_D2A_1 + energy_comp_D2A_2
        ## TX
        energy_comp_TX = TX_energy * h * \
            (N2 * iter_N1 * D) * 1e-9 * self.full_range_support_factor

        # computation: mainly the weight encoding energy
        # static enrgy can not be shared with time-sharing
        # is the domaint one, so we don't divide
        energy_comp_comp = num_computation * comp_energy_static * \
            1e-9 * self.full_range_support_factor # should multiply with full_range_support_factor
        energy_comp_comp += comp_energy_dynamic * \
            (num_computation // self.weight_reuse_factor) * \
            1e-9 * full_range_support_factor_w
        
        energy_comp_ADC = h * A2D_energy * (N1 * N2 * iter_D) * 1e-9 * self.full_range_support_factor
        energy_comp_RX = h * RX_energy * (N1 * N2 * iter_D) * 1e-9 * self.full_range_support_factor
        energy_comp_adder = h * Adder_energy * (N1 * N2 * iter_D) * 1e-9 * self.full_range_support_factor
        energy_comp_output = energy_comp_ADC + energy_comp_RX + energy_comp_adder
        
        energy_comp = energy_comp_laser + energy_comp_D2A + \
                       energy_comp_TX + energy_comp_comp + energy_comp_output

        # datamovement related energy
        # NOTE(hqzhu): input's data movement energy should be mutiplied by 2 considering only positive support
        self.num_byte = self.in_bit / 16

        # RF
        energy_dm_RF = h * self.num_byte * self.data_movement_RF * (N1 * D * (
            N2 // self.weight_reuse_factor) * full_range_support_factor_w + iter_N1 * N2 * D * full_range_support_factor_x) * 2 
        # print('atten')
        # output to RF, accumlated first in the same tile and then send back to buffer
        energy_dm_RF += h * self.num_byte * self.data_movement_RF * (N1 * N2 * iter_D * self.full_range_support_factor) / self.num_pe_per_tile * 2
        # NoC send to adder
        energy_dm_RF += h * self.num_byte * (N1 * N2 * iter_D * self.full_range_support_factor) * self.data_movement_NoC

        # GLB-> read

        ## GLB1: read from GLB1 to RF (energy_dm_GLB1_input) and write from GLB2 to GLB1 (energy_dm_GLB1_input_from_GLB2)
        energy_dm_GLB1_input = h * self.num_byte * self.data_movement_GB1 * \
            (N1 * D * (N2 // self.weight_reuse_factor) * full_range_support_factor_w +
             iter_N1 * D * N2 * full_range_support_factor_x)
        energy_dm_GLB1_input_from_GLB2 = h * self.num_byte * self.data_movement_GB1 * \
            (N1*D * full_range_support_factor_w + N2 * D * iter_N1 * full_range_support_factor_x)
        
        # each weight core generates self.hw.core_height * N2 * self.act_bit, need send back to GLB2 if exceeds GLB1 size
        output_load_time = max(math.ceil(self.hw.core_height * N2 * self.act_bit / 8 / self.local_buffer_size), 1)
        
        # output from RF TO GLB1, GLB1 to GLB2, GLB2 to GLB1
        if output_load_time > 1:
            # need reload
            energy_dm_GLB1_output = h * self.num_byte * \
                self.data_movement_GB1 * (N1 * N2) * (iter_D / self.num_pe_per_tile * self.full_range_support_factor * 2 - 1)
        else:
            energy_dm_GLB1_output = h * self.num_byte * \
                self.data_movement_GB1 * (N1 * N2) * iter_D / self.num_pe_per_tile * self.full_range_support_factor # only writes those data to GLB1

        energy_dm_GLB1 = energy_dm_GLB1_input + energy_dm_GLB1_output + energy_dm_GLB1_input_from_GLB2
        
        ## GLB2: write to GLB1
        energy_dm_GLB2 = h * self.num_byte * self.data_movement_GB2 * (N1*D * full_range_support_factor_w + N2 * D * iter_N1 * full_range_support_factor_x)
        if output_load_time > 1:
            # we write partials sums back to GLB2
            # it will generates full_range_support factor that much due to inability of full-range support
            energy_dm_GLB2 += h * self.num_byte * self.data_movement_GB2 * N1 * N2 * (iter_D / self.num_pe_per_tile * self.full_range_support_factor * 2 -2)
        energy_dm_GLB2 += h * self.num_byte * self.data_movement_GB2 * N1 * N2 # assume that output is fully added at the GLB1 level

        energy_dm_DRAM = 0

        energy_dm = energy_dm_DRAM + energy_dm_GLB1 + energy_dm_GLB2 + energy_dm_RF

        # NOTE(hqzhu): multiply energy comp by 2 since it need to handle negative input
        energy = energy_dm + energy_comp
        
        if print_msg:
            print(f"Overall estimated energy cost {energy} mJ")
            print(
                f"--Computation energy cost is {energy_comp} mJ  {energy_comp / energy * 100 :.2f} %")
            print(
                f"----comp energy cost is {energy_comp_comp} mJ  {energy_comp_comp / energy * 100 :.2f} %")
            print(
                f"----Laser energy cost is {energy_comp_laser} mJ  {energy_comp_laser / energy * 100 :.2f} %")
            print(
                f"----Ouput energy cost is {energy_comp_output} mJ  {energy_comp_output / energy * 100 :.2f} %")
            print(
                f"----D2A energy cost is {energy_comp_D2A} mJ  {energy_comp_D2A / energy * 100 :.2f} %")
            print(
                f"----TX energy cost is {energy_comp_TX} mJ  {energy_comp_TX / energy * 100 :.2f} %")
            print(
                f"--Datamovement energy cost is {energy_dm} mJ {energy_dm / energy * 100 :.2f} %")
            print(
                f"----RF energy cost is {energy_dm_RF} mJ {energy_dm_RF / energy * 100 :.2f} %")
            print(
                f"----GLB1 energy cost is {energy_dm_GLB1} mJ {energy_dm_GLB1 / energy * 100 :.2f} %")
            print(
                f"----GLB2 energy cost is {energy_dm_GLB2} mJ {energy_dm_GLB2 / energy * 100 :.2f} %")
            print(
                f"----GLB energy cost is {energy_dm_DRAM} mJ {energy_dm_DRAM / energy * 100 :.2f} %")
        
        # save to dict
        # comp
        energy_dict['comp']['total'] = [energy_comp, round(energy_comp / energy * 100, 2)]
        energy_dict['comp']['laser'] = [energy_comp_laser, round(energy_comp_laser / energy * 100, 2)]
        energy_dict['comp']['TX_1'] = [energy_comp_comp, round(energy_comp_comp / energy * 100, 2)]
        energy_dict['comp']['DAC_1'] = [energy_comp_D2A_1, round(energy_comp_D2A_1 / energy * 100, 2)]
        energy_dict['comp']['TX_2'] = [energy_comp_TX, round(energy_comp_TX / energy * 100, 2)]
        energy_dict['comp']['DAC_2'] = [energy_comp_D2A_2, round(energy_comp_D2A_2 / energy * 100, 2)]
        energy_dict['comp']['RX'] = [energy_comp_RX, round(energy_comp_RX / energy * 100, 2)]
        energy_dict['comp']['ADC'] = [energy_comp_ADC, round(energy_comp_ADC / energy * 100, 2)]
        energy_dict['comp']['adder'] = [energy_comp_adder, round(energy_comp_adder / energy * 100, 2)]
        # memory
        energy_dict['datamovement']['total'] = [energy_dm, round(energy_dm / energy * 100, 2)]
        energy_dict['datamovement']['RF'] = [energy_dm_RF, round(energy_dm_RF / energy * 100, 2)]
        energy_dict['datamovement']['LB'] = [energy_dm_GLB1, round(energy_dm_GLB1 / energy * 100, 2)]
        energy_dict['datamovement']['GB'] = [energy_dm_GLB2, round(energy_dm_GLB2 / energy * 100, 2)]
        energy_dict['datamovement']['DRAM'] = [energy_dm_DRAM, round(energy_dm_DRAM / energy * 100, 2)]

        return energy, energy_dict

    def __save_csv(self, sv_name, dic2d, topic):
        with open(sv_name, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([topic,'energy (mJ)', 'percentage (%)'])
            for each in dic2d:
                writer.writerow([each, '', ''])
                for each_part in dic2d[each]:
                    # comp or dm
                    writer.writerow([each_part, '', ''])
                    for each_compoent in dic2d[each][each_part]:
                        data = [each_compoent]
                        # print(data)
                        data.extend(dic2d[each][each_part][each_compoent])
                        writer.writerow(data)

    def __save_latency_csv(self, sv_name, dic2d, topic):
        with open(sv_name, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([topic,'cycles', 'time (ms)'])
            for each in dic2d:
                writer.writerow([each, '', ''])
                for each_part in dic2d[each]:
                    # comp or dm
                    data = [each_part]
                    data.extend(dic2d[each][each_part])
                    writer.writerow(data)

    def save(self, sv_name, sv_path='./simulate_res/'):
        ensure_dir(sv_path)
        energy_file_name = os.path.join(sv_path, f'{sv_name}_energy.csv')
        self.__save_csv(energy_file_name, self.energy_dict, 'Attn')
    
    def get_energy(self, matrix_dim1, matrix_dim2):
        if self.core_type == "dota":
            energy = self.get_energy_crossbar(matrix_dim1, matrix_dim2)
            latency = self.get_latency_crossbar(matrix_dim1, matrix_dim2)
        elif self.core_type == "mrrbank":
            energy = self.get_energy_mrrbank(matrix_dim1, matrix_dim2)
            latency = self.get_latency_mrr_bank(matrix_dim1, matrix_dim2)
        else:
            raise NotImplementedError

        return energy
    
    def run(self, print_msg=False):
        
        # QK^T
        if print_msg:
            print("Start simulation for ATTN")
            print("--" * 10)
            print("Q*K^T")
        matrix_dim1 = (self.num_heads, self.num_tokens, self.head_dim)
        matrix_dim2 = (self.num_heads, self.head_dim, self.num_tokens)
        if self.core_type == "dota":
            energy, energy_dict = self.get_energy_crossbar(matrix_dim1, matrix_dim2, print_msg=print_msg)
            latency, latency_dict = self.get_latency_crossbar(matrix_dim1, matrix_dim2, print_msg=print_msg)
        elif self.core_type == "mrrbank":
            energy, energy_dict = self.get_energy_mrrbank(matrix_dim1, matrix_dim2, print_msg=print_msg)
            latency, latency_dict = self.get_latency_mrr_bank(matrix_dim1, matrix_dim2, print_msg=print_msg)
        else:
            raise NotImplementedError
        
        self.energy_dict['Q*K^T'] = energy_dict
        self.latency_dict['Q*K^T'] = latency_dict
        
        # SV
        if print_msg:
            print("--" * 10)
            print("S*V")
        matrix_dim1 = (self.num_heads, self.num_tokens, self.num_tokens)
        matrix_dim2 = (self.num_heads, self.num_tokens, self.head_dim)
        # we reset full-range support factor to the half since the S is postive
        self.full_range_support_factor = self.full_range_support_factor // 2 if self.full_range_support_factor > 1 else self.full_range_support_factor
        if self.core_type == "dota":
            energy, energy_dict = self.get_energy_crossbar(matrix_dim1, matrix_dim2, print_msg=print_msg)
            latency, latency_dict = self.get_latency_crossbar(matrix_dim1, matrix_dim2, print_msg=print_msg)
        elif self.core_type == "mrrbank":
            # matrix 1 is positive, then let it to be input
            # swap matrix 1 and matrix 2
            matrix_dim1 = (self.num_heads, self.head_dim, self.num_tokens)
            matrix_dim2 = (self.num_heads, self.num_tokens, self.num_tokens)
            energy, energy_dict = self.get_energy_mrrbank(matrix_dim1, matrix_dim2, print_msg=print_msg)
            latency, latency_dict = self.get_latency_mrr_bank(matrix_dim1, matrix_dim2, print_msg=print_msg)
        else:
            raise NotImplementedError
        
        self.energy_dict['S*V'] = energy_dict
        self.latency_dict['S*V'] = latency_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default=".params.yaml",
                        metavar="FILE", help="config file")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    # deit-base
    OPs_list = []
    OPs_list.append({"idx": 0, "type": "attn", "num_heads": 6, "embed_dim": 384,
                "num_tokens": 197})

    for item in OPs_list:
        idx = item["idx"]
        if item["type"] == "attn":
            
            for i in range(2):
                configs.arch.input_mod_sharing_flag = i%2
                predictor = attnPrediction(item, configs)
                predictor.run()
                predictor.save(f'Attn_deit_small_{configs.core.type}.{configs.arch.input_mod_sharing_flag}')
