# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu
# @Date:   2023-02-12 22:43:23
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-13 19:56:31
# FNN contains q, k, v, porjection and FFN
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
from hardware.photonic_MZI import PhotonicMZI
from hardware.SRAM import SRAM

logging.basicConfig(level=logging.INFO)

class FFNPrediction():
    """Simulator for FFN layer.
    weight matrix [N1, N2] * [N2, N3], where N3 is batch * tokens in FFN module.
    """
    def __init__(self, op_info, configs=None) -> None:
        super().__init__()
        
        self.in_features = op_info['in_features']
        self.out_features = op_info['out_features']
        self.bs = op_info['bs']

        # bits
        self.in_bit = configs.core.precision.in_bit
        self.w_bit = configs.core.precision.w_bit
        self.act_bit = configs.core.precision.act_bit

        # accelerator info
        self.core_type = configs.core.type
        self.num_tiles = configs.arch.num_tiles
        self.num_pe_per_tile = configs.arch.num_pe_per_tile
        
        # arch-level design choices
        self.full_range_support_factor = 1 if self.core_type == "dota" else configs.arch.full_range_support_factor
        self.weight_reuse_factor = 1 if self.core_type == "dota" else configs.arch.weight_reuse_factor
        self.time_accum_factor = configs.arch.time_accum_factor if self.core_type == "dota" else 1

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
        elif self.core_type == "mzi":
            self.hw = PhotonicMZI(
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
        self.adder_power = 0.2 / 4.39  # follow tech node scaling law to 14 nm # mW @ ISAAC

        # get memory movement cost
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
        N1, D1 = matrix_dim1
        D2, N2 = matrix_dim2
        assert (D1 == D2), f"Got incorrect matrix dimension."
        D = D1

        iter_N1 = math.ceil(N1 / self.hw.core_height)
        iter_N2 = math.ceil(N2 / self.hw.core_width)
        iter_D = math.ceil(D / self.hw.num_wavelength)

        # computation cycles:
        cycles_computations = math.ceil(iter_D * iter_N1 * iter_N2 /( self.num_tiles * self.num_pe_per_tile))

        latency_comp = cycles_computations * 1 / self.work_freq * 1e-6

        latency_dict['comp'][0] = cycles_computations
        latency_dict['comp'][1] = latency_comp
        
        # load data cycles:
        cycles_preload_data_dram_sram = 0
        cycles_preload_data_GB_SRAM = 0
        cylces_load_data_sram_rf = 0

        for _iter_N1 in range(math.ceil(N1 / (self.num_tiles * self.hw.core_height))):
            # pre-load a chunk of data, the other loading would be hide in computation cylces
            num_bytes_dram_sram = (
                self.hw.core_height * D)
            cycles_preload_data_dram_sram += self.SRAM.preload_DRAM_SRAM(
                num_bytes_dram_sram, bits=self.in_bit, bandwidth_ratio=1/self.num_tiles)
            num_bytes_GB_sram = self.hw.core_height * D * self.num_tiles + D * N2
            cycles_preload_data_GB_SRAM += self.SRAM.load_GB_SRAM(num_bytes_GB_sram, bits=self.in_bit, bandwidth_ratio=1)

            for _iter_N2 in range(iter_N2):
                for _iter_D in range(math.ceil(iter_D / self.num_pe_per_tile)):
                    # load data from sram to rf
                    # have sram Q and sram K
                    num_bytes_sram_rf = max(
                        self.hw.core_height * self.hw.num_wavelength, self.hw.core_height * self.hw.num_wavelength)
                    # bandwidth is 1 / self.num_pe_per_tile since all pes in the same tile share 1 sram
                    cylces_load_data_sram_rf += self.SRAM.load_SRAM_RF(
                        num_bytes_sram_rf, bits=self.in_bit, bandwidth_ratio=1/self.num_pe_per_tile)

        # bottleneck is DRAM to GB
        cylces_load_data_sram_rf = 0
        cycles_preload_data_GB_SRAM = 0 # set this to 0 as we engineer the memory part to avoid this part being bottleneck
        
        cycles_memory = max(cycles_preload_data_dram_sram,
                            cylces_load_data_sram_rf,
                            cycles_preload_data_GB_SRAM)

        latency_memory = cycles_memory * 1 / self.SRAM.clock_frequency * 1e3

        latency_dict['datamovement'][0] = cycles_memory
        latency_dict['datamovement'][1] = latency_memory

        if print_msg:
            print("**" * 10)
            print("Latency estimation for Linear")
            print(f"M1 size {N1} * {D1}")
            print(f"M1 size {D2} * {N2}")
            print(f"The loop number is {iter_N1}, {iter_N2}, {iter_D}")
            print(
                f"computation cycles: {cycles_computations} at freq {self.work_freq} GHz")
            print(
                f"memory cycles: {cycles_memory} at freq {self.SRAM.clock_frequency*1e-9} GHz")

        latency = max(latency_comp, latency_memory)
        latency_dict['total'][0] = -1
        latency_dict['total'][1] = latency

        return latency, latency_dict

    def get_latency_mzi(self, matrix_dim1, matrix_dim2, print_msg=False):
        """Get the latency of running attn on MZI, , assuming a computation-bounded system"""
        latency_dict = {}
        latency_dict['total'] = [0, 0]
        latency_dict['comp'] = [0, 0]
        latency_dict['datamovement'] = [0, 0]
        N1, D1 = matrix_dim1
        D2, N2 = matrix_dim2
        assert (D1 == D2), f"Got incorrect matrix dimension."
        D = D1

        iter_N1 = math.ceil(N1 / self.hw.core_height)
        iter_N2 = N2
        iter_D = math.ceil(D / self.hw.core_width)

        # computation cycles:
        cycles_computations = math.ceil(iter_D * iter_N1 * iter_N2 / (self.num_tiles * self.num_pe_per_tile))
        latency_comp = cycles_computations * 1 / self.work_freq * 1e-6

        # computation latency have to add the programming MZI latency
        # consider parallel programming for all PTC at one time
        latency_comp_program_mzi = (math.ceil((iter_N1 * iter_D) / (self.num_tiles * self.num_pe_per_tile)) * self.hw.mzi_response_time)
        latency_comp += latency_comp_program_mzi
        
        latency_dict['comp'][0] = cycles_computations
        latency_dict['comp'][1] = latency_comp
        latency_dict['program_mzi'] = [(latency_comp_program_mzi * self.work_freq * 1e6), latency_comp_program_mzi]

        # load data cycles
        # assume a ideal baseline without being bounded by memory
        cycles_preload_data_dram_sram = 0
        cylces_load_data_sram_rf = 0
        cycles_preload_data_GB_SRAM = 0
        cycles_memory = max(cycles_preload_data_dram_sram, 0)
        latency_memory = cycles_memory * 1 / self.SRAM.clock_frequency * 1e3
        
        latency_dict['datamovement'][0] = cycles_memory
        latency_dict['datamovement'][1] = latency_memory
        
        latency = max(latency_comp, latency_memory)
        
        latency_dict['total'][0] = -1
        latency_dict['total'][1] = latency

        if print_msg:
            print("**" * 10)
            print("Latency estimation for Linear")
            print(f"M1 size {N1} * {D1}")
            print(f"M1 size {D2} * {N2}")
            print(f"The loop number is {iter_N1}, {iter_N2}, {iter_D}")
            print(
                f"memory cycles: {cycles_memory} at at freq {self.SRAM.clock_frequency*1e-9} GHz")
            print(
                f"computation cycles: {cycles_computations} at freq {self.work_freq} GHz")

        return latency, latency_dict

    def get_latency_mrr_bank(self, matrix_dim1, matrix_dim2, print_msg=False):
        """Latency estimation for mrr bank baselines, assuming a computation-bounded system"""
        latency_dict = {}
        latency_dict['total'] = [0, 0]
        latency_dict['comp'] = [0, 0]
        latency_dict['datamovement'] = [0, 0]
        N1, D1 = matrix_dim1
        D2, N2 = matrix_dim2
        assert (D1 == D2), f"Got incorrect matrix dimension."
        D = D1

        iter_N1 = math.ceil(N1 / self.hw.core_height)
        iter_N2 = N2
        iter_D = math.ceil(D / self.hw.core_width)

        # MRR only support non-negative operands
        if self.full_range_support_factor == 2:
            full_range_support_factor_x = 2
            full_range_support_factor_w = 1
        elif self.full_range_support_factor == 4:
            full_range_support_factor_x = 2
            full_range_support_factor_w = 2
        else:
            full_range_support_factor_w, full_range_support_factor_x = 1, 1

        # computation cycles:
        # cycles_computations = math.ceil((iter_D * iter_N1 * iter_N2) / (self.num_tiles * self.num_pe_per_tile)) * self.full_range_support_factor
        cycles_computations = math.ceil(iter_D * iter_N1 * iter_N2 / (self.num_tiles * self.num_pe_per_tile)) * self.full_range_support_factor
        
        latency_comp = cycles_computations * 1 / self.work_freq * 1e-6
        
        latency_dict['comp'][0] = cycles_computations
        latency_dict['comp'][1] = latency_comp

        # load data cycles
        # assume a ideal baseline without being bounded by memory
        cycles_preload_data_dram_sram = 0
        cylces_load_data_sram_rf = 0
        cylces_load_data_GB_SRAM = 0
        cylces_load_data_sram_rf = 0
        cycles_memory = max(cycles_preload_data_dram_sram,
                            cylces_load_data_sram_rf,
                            cylces_load_data_GB_SRAM)
        # cycles_memory = max(cycles_preload_data_GB_SRAM, 0)
        latency_memory = cycles_memory * 1 / self.SRAM.clock_frequency * 1e3
        latency_dict['datamovement'][0] = cycles_memory
        latency_dict['datamovement'][1] = latency_memory
        
        latency = max(latency_comp, latency_memory)
        latency_dict['total'][0] = -1
        latency_dict['total'][1] = latency
        
        if print_msg:
            print("**" * 10)
            print("Latency estimation for Linear")
            print(f"M1 size {N1} * {D1}")
            print(f"M1 size {D2} * {N2}")
            print(f"The loop number is {iter_N1}, {iter_N2}, {iter_D}")
            print(
                f"memory cycles: {cycles_memory} at at freq {self.SRAM.clock_frequency*1e-9} GHz")
            print(
                f"computation cycles: {cycles_computations} at freq {self.work_freq} GHz")

        return latency, latency_dict

    def get_energy_crossbar(self, matrix_dim1, matrix_dim2, print_msg=False):
        """Energy estimation for our DPTC.
        Matrix_dim1: weight dimenstion.
        Matrix_dim2: input dimenstion.
        """
        energy_dict = {}
        energy_dict['comp'] = {i: [0, 0] for i in [
            "total", "laser", "ADC", "adder"]}
        energy_dict['datamovement'] = {i: [0, 0]
                                       for i in ["total", "RF", "LB", "GB", "DRAM"]}
        N1, D1 = matrix_dim1
        D2, N2 = matrix_dim2
        D = D1
        assert (D1 == D2), f"Got incorrect matrix dimension."

        num_computation = N1 * N2 * D
        num_ifmap1 = N1 * D
        num_ifmap2 = D * N2
        num_ofmap = N1 * N2

        iter_N1 = math.ceil(N1 / self.hw.core_height)
        iter_N2 = math.ceil(N2 / self.hw.core_width)
        iter_D = math.ceil(D / self.hw.num_wavelength)

        if print_msg:
            print("**" * 10)
            print("Energy estimation for Linear")
            print(f"M1 matrix dims: {N1} * {D1}")
            print(f"M2 matrix dims: {N2} * {D2}")
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
        Adder_energy = self.adder_power / self.work_freq

        ## laser energy
        energy_comp_laser = laser_energy * \
            (iter_N1 * iter_N2 * iter_D) * 1e-9

        ## DAC and TX ernergy for matrix 1 and matrix 2
        if self.disable_crossbar_topology:
            # only matrix 2 is broadcasted and shared
            energy_comp_D2A_1 = D2A_energy * N1 * D * N2 * 1e-9
            energy_comp_D2A_2 = (D2A_energy * iter_N1 * N2 * D * 1e-9) / \
                self.num_tiles if self.input_mod_sharing_flag else (
                    D2A_energy * iter_N1 * N2 * D * 1e-9)
            energy_comp_TX_1 = TX_energy * N1 * N2 * D * 1e-9
            energy_comp_TX_2 = (TX_energy * N2 * iter_N1 * D * 1e-9) / \
                self.num_tiles if self.input_mod_sharing_flag else (
                    TX_energy * N2 * iter_N1 * D * 1e-9)
        else:
            # both matrix 1 and matrix 2 is shared enabled by dota topology
            # the shared times is the core_height and core_width
            energy_comp_D2A_1 = D2A_energy * N1 * D * iter_N2 * 1e-9
            energy_comp_D2A_2 = (D2A_energy * iter_N1 * N2 * D * 1e-9) / \
                self.num_tiles if self.input_mod_sharing_flag else (
                    D2A_energy * iter_N1 * N2 * D * 1e-9)
            energy_comp_TX_1 = TX_energy * N1 * iter_N2 * D * 1e-9
            energy_comp_TX_2 = (TX_energy * N2 * iter_N1 * D * 1e-9) / \
                self.num_tiles if self.input_mod_sharing_flag else (
                    TX_energy * N2 * iter_N1 * D * 1e-9)
        
        energy_comp_D2A = energy_comp_D2A_1 + energy_comp_D2A_2
        energy_comp_TX = energy_comp_TX_1 + energy_comp_TX_2

        # computation energy (comp)
        energy_comp_comp = comp_energy * num_computation * 1e-9

        # output
        # time_accum_factor is the min between our factor and the max possible time-domain reuse factor
        time_accum_factor = min(
            self.time_accum_factor, math.ceil(D / (self.num_pe_per_tile * self.hw.num_wavelength)))

        # RX: ADC, TIA, detector
        if self.adc_share_flag:
            # / self.num_pe_per_tile
            ps_size = N1 * N2 * math.ceil(math.ceil(iter_D / time_accum_factor) / self.num_pe_per_tile)
            energy_comp_ADC = A2D_energy * \
                (ps_size) * 1e-9 
            energy_comp_TIA = self.hw.TIA_energy * \
                (ps_size) * 1e-9
            energy_comp_adder = Adder_energy * \
                (ps_size) * 1e-9
        else:
            # / time_accum_factor
            ps_size = N1 * N2 * math.ceil(iter_D / time_accum_factor)
            energy_comp_ADC = A2D_energy * \
                (ps_size) * 1e-9 
            energy_comp_TIA = self.hw.TIA_energy * \
                (ps_size) * 1e-9 
            energy_comp_adder = Adder_energy * \
                (ps_size) * 1e-9

        energy_comp_detection = self.hw.photo_detector_energy * \
            (N1 * N2 * iter_D) * 1e-9

        energy_comp_RX = energy_comp_detection + energy_comp_TIA

        energy_comp_output = energy_comp_ADC + energy_comp_TIA + \
            energy_comp_adder + energy_comp_detection
        energy_comp = energy_comp_laser + energy_comp_D2A + \
            energy_comp_TX + energy_comp_comp + energy_comp_output

        # datamovement related energy
        self.num_byte = self.in_bit / 16
        ## RF
        # matrix 1
        if self.disable_crossbar_topology:
            energy_dm_RF = self.num_byte * \
                self.data_movement_RF * (N2 * N1 * D) * 2
        else:
            energy_dm_RF = self.num_byte * \
                self.data_movement_RF * (iter_N2 * N1 * D) * 2
        # matrix 2
        energy_dm_RF += self.num_byte * self.data_movement_RF * \
            (iter_N1 * N2 * D / self.num_tiles) * 2 if self.input_mod_sharing_flag else self.num_byte * \
            self.data_movement_RF * (iter_N1 * N2 * D) * 2
        # output: ps is first accumulated(pe in the same tile), then back to buffer
        energy_dm_RF += self.num_byte * self.data_movement_RF * ((N1 * N2 * math.ceil(math.ceil(iter_D / time_accum_factor) / self.num_pe_per_tile))) * 2
        
        # send ps to adder: dependednt on time_acc, adc_share
        energy_dm_Noc = self.num_byte * ps_size * self.data_movement_NoC
        energy_dm_RF += energy_dm_Noc

        # we need to load and unload output from GB
        output_load_time = max(math.ceil(self.hw.core_height * D * self.in_bit / 8 / self.local_buffer_size), 1)
        # LB -> GB: output_load_time
        # GB -> LB: output_load_time - 1
        
        # GLB1
        # input: GLB1 -> RF and GLB2 -> GLB1
        if self.input_mod_sharing_flag:
            energy_dm_GLB1_input = self.num_byte * \
                self.data_movement_GB1 * \
                (N1 * D * iter_N2 + iter_N1 * D * N2 / self.num_tiles)
            energy_dm_GLB1_input_from_GLB2 = self.num_byte * \
                self.data_movement_GB1 * \
                (N1 * D + + N2 * D * iter_N1 / self.num_tiles)
        else:
            energy_dm_GLB1_input = self.num_byte * \
                self.data_movement_GB1 * \
                (N1 * D * iter_N2 + iter_N1 * D * N2)
            energy_dm_GLB1_input_from_GLB2 =self.num_byte * \
                self.data_movement_GB1 * \
                (N1 * D + + N2 * D * iter_N1)
        # output
        # output write to GLB1 since the input buffer size is limited, we cannot finish all output-stationary
        energy_dm_GLB1_output = self.num_byte * \
            self.data_movement_GB1 * (N1 * N2) * (2 * output_load_time - 1)
        energy_dm_GLB1 = energy_dm_GLB1_input + energy_dm_GLB1_input_from_GLB2 + energy_dm_GLB1_output
        
        # GLB2
        # input: M1 -> not reloaded, M2 -> roload times: iter_N1 * iter_N2 * iter_D / num_tiles * num_tiles (broadcast)
        # output: the same as DRAM
        if self.input_mod_sharing_flag:
            energy_dm_GLB2 = self.num_byte * self.data_movement_GB2 * ((
                N1 * N2 * (2 * output_load_time - 1)) + (N1 * D + N2 * D * iter_N1 / self.num_tiles))
        else:
            energy_dm_GLB2 = self.num_byte * self.data_movement_GB2 * ((
                N1 * N2 * (2 * output_load_time - 1)) + (N1 * D + N2 * D * iter_N1))

        # assume only load weight from DRAM
        energy_dm_GLB2 += self.num_byte * self.data_movement_GB2 * (N1 * D)

        # load weight from DRAM
        energy_dm_DRAM = self.num_byte * self.data_movement_DRAM_GB * (N1 * D)
        
        energy_dm = energy_dm_DRAM + energy_dm_GLB1 + energy_dm_GLB2 + energy_dm_RF

        energy = energy_dm + energy_comp
        
        if print_msg:
            print(f"Overall estimated energy cost {energy} mJ")
            print(
                f"--Computation energy cost is {energy_comp} mJ  {energy_comp / energy * 100 :.2f} %")
            print(
                f"----Comp energy cost is {energy_comp_comp} mJ  {energy_comp_comp / energy * 100 :.2f} %")
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

    def get_energy_mzi(self, matrix_dim1, matrix_dim2, print_msg=False):
        """Energy estimation for MZI.
        Matrix_dim1: weight dimenstion. It is the static operand.
        Matrix_dim2: input dimenstion.
        """
        energy_dict = {}
        energy_dict['comp'] = {i: [0, 0] for i in [
            "total", "laser", "ADC", "RX", "adder"]}
        energy_dict['datamovement'] = {i: [0, 0]
                                       for i in ["total", "RF", "LB", "GB", "DRAM"]}

        N1, D1 = matrix_dim1
        D2, N2 = matrix_dim2
        D = D1
        assert (D1 == D2), f"Got incorrect matrix dimension."
        if self.weight_reuse_factor == -1:
            self.weight_reuse_factor = N2
        num_computation = N1 * N2 * D
        num_ifmap1 = N1 * D
        num_ifmap2 = D * N2
        num_ofmap = N1 * N2

        iter_N1 = math.ceil(N1 / self.hw.core_height)
        iter_N2 = N2
        iter_D = math.ceil(D / self.hw.core_width)
        
        if print_msg:
            print("**" * 10)
            print("Energy estimation for Linear")
            print(f"M1 matrix dims: {N1} * {D1}")
            print(f"M2 matrix dims: {N2} * {D2}")
            print(
                f"The loop number of N1, N2, D is {iter_N1}, {iter_N2}, {iter_D}")
        
        D2A_energy = self.hw.cal_D2A_energy()
        TX_energy = self.hw.cal_TX_energy()
        laser_energy = self.hw.cal_laser_energy()
        comp_energy = self.hw.cal_comp_energy()
        A2D_energy = self.hw.cal_A2D_energy()
        RX_energy = self.hw.cal_RX_energy()
        Adder_energy = self.adder_power / self.work_freq

        energy_comp_laser = laser_energy * \
            (iter_N1 * iter_N2 * iter_D) * 1e-9

        # energy for modulation
        # matrix 1 is the static operand
        # matrix 2 is the dynamic input
        
        energy_comp_D2A_1= D2A_energy * \
            ((iter_N1 * iter_N2 * iter_D) // self.weight_reuse_factor) * \
            (self.hw.num_mzis + self.hw.mzi_sigma_dim) * 1e-9

        energy_comp_D2A_2 = D2A_energy * \
            (iter_N1 * N2 * D) * 1e-9
            
        energy_comp_D2A = energy_comp_D2A_1 + energy_comp_D2A_2
        
        # TX for input
        energy_comp_TX = TX_energy * \
            (N2 * iter_N1 * D) * 1e-9

        # comp: TX for weight operand
        energy_comp_comp = ((iter_N1 * iter_N2 * iter_D) // self.weight_reuse_factor) * (self.hw.comp_energy_dynamic *
                                                                                       self.hw.num_mzis + self.hw.mzi_sigma_dim * TX_energy) * 1e-9
        energy_comp_comp += (iter_N1 * iter_N2 * iter_D) * \
            (self.hw.comp_energy_static * self.hw.num_mzis) * 1e-9

        energy_comp_ADC = A2D_energy * (N1 * N2 * iter_D) * 1e-9
        energy_comp_RX = RX_energy * (N1 * N2 * iter_D) * 1e-9
        energy_comp_adder = Adder_energy * (N1 * N2 * iter_D) * 1e-9
        energy_comp_output = energy_comp_ADC + energy_comp_RX

        energy_comp = energy_comp_laser + energy_comp_D2A + \
            energy_comp_TX + energy_comp_comp + energy_comp_output + energy_comp_adder

        # datamovement related energy
        self.num_byte = self.in_bit / 16
        energy_dm_RF = self.num_byte * self.data_movement_RF * (
            ((iter_N1 * iter_N2 * iter_D) // self.weight_reuse_factor) * (self.hw.num_mzis + self.hw.mzi_sigma_dim)
            + iter_N1 * N2 * D + N1 * N2 * iter_D / self.num_pe_per_tile) * 2
        energy_dm_RF += (N1 * N2 * iter_D) * self.data_movement_NoC * self.num_byte

        # GLB1
        # input: GLB -> RF
        energy_dm_GLB1_input = self.num_byte * self.data_movement_GB1 * \
            (iter_N1 * iter_D * (N2 // self.weight_reuse_factor) *
             (self.hw.num_mzis + self.hw.mzi_sigma_dim) + iter_N1 * D * N2)

        # only load weight matrices once
        energy_dm_GLB1_input_from_GLB2 = self.num_byte * self.data_movement_GB1 * (iter_N1 * iter_D * (self.hw.num_mzis + self.hw.mzi_sigma_dim) + N2 * D * iter_N1)

        # output: weight-stationary will generate a large partial sum, you have to send it back to GLB1 at least
        # we need to compute based on output buffer size and the required stored activation
        # if it cannot satisify, we need send it back to GLB2
        # e.g., each tile is generating n1 * N2 output activation and we will generate iter_D / self.num_pe_per_tile
        # (assumed multi-pe results are first summed)
        # if we can only hold n1 * N2 / 2 size activation on on-chip buffer
        # then we need # load the half from GLB1 to GLB2; write to GLB2; then we need fentch it back to GLB1
        # in total, it will takes 3 times
        # cycle 1; half 1 -> 
        # cycle 1: half 2 ->
        # cycle 2: half 1 -> half_1(c1) back to GLB1
        # cycle 2: half 2 -> half_2(c1) back to GLB1
        # totals writes should be N1 * N2 * iter_D /self.num_pe_per_tile -> (iter_D/self.num_pe_per_tile * 2 -1)
        # every time, we have n1 * N2 write to glb2 and write back to glb1 
        # n1 * N2 * (iter_D/self.num_pe_per_tile -1)  * 2 * iter_N1
        output_load_time = max(math.ceil(self.hw.core_height * N2 * self.act_bit / 8 / self.local_buffer_size), 1)
        if output_load_time > 1:
            # need reload
            energy_dm_GLB1_output = self.num_byte * \
                self.data_movement_GB1 * (N1 * N2) * (iter_D / self.num_pe_per_tile * 2 -1)
        else:
            energy_dm_GLB1_output = self.num_byte * \
                self.data_movement_GB1 * (N1 * N2) * iter_D / self.num_pe_per_tile # only writes those data to GLB1

        energy_dm_GLB1 = energy_dm_GLB1_input + energy_dm_GLB1_output + energy_dm_GLB1_input_from_GLB2
        
        # GLB2:
        # GLB2 -> GLB1: input
        energy_dm_GLB2 = self.num_byte * self.data_movement_GB2 * (iter_N1 * iter_D * (self.hw.num_mzis + self.hw.mzi_sigma_dim) + N2 * D * iter_N1)
        # GLB2<->GLB1: output
        if output_load_time > 1:
            # we write partials sums back to GLB2
            energy_dm_GLB2 += self.num_byte * self.data_movement_GB2 * N1 * N2 * (iter_D / self.num_pe_per_tile * 2 -2)
        energy_dm_GLB2 += self.num_byte * self.data_movement_GB2 * N1 * N2
        # DRAM->GLB2
        energy_dm_GLB2 += self.num_byte * self.data_movement_GB2 * (N1 * D)
        # DRAM
        # energy_dm_DRAM = self.num_byte * (self.data_movement_DRAM * (
        #     N1 * N2) + self.data_movement_DRAM_GB * (iter_N1 * iter_D * (self.hw.num_mzis + self.hw.mzi_sigma_dim) + N2 * D * iter_N1 / self.num_tiles))
        energy_dm_DRAM = self.num_byte * self.data_movement_DRAM * (iter_N1 * iter_D * (self.hw.num_mzis + self.hw.mzi_sigma_dim))

        energy_dm = energy_dm_DRAM + energy_dm_GLB1 + energy_dm_GLB2 + energy_dm_RF

        energy = energy_dm + energy_comp
        
        if print_msg:
            print(f"Overall estimated energy cost {energy} mJ")
            print(
                f"--Computation energy cost is {energy_comp} mJ  {energy_comp / energy * 100 :.2f} %")
            print(
                f"----Comp energy cost is {energy_comp_comp} mJ  {energy_comp_comp / energy * 100 :.2f} %")
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

    def get_energy_mrrbank(self, matrix_dim1, matrix_dim2, print_msg=False):
        """Energy estimation for MRR.
        Matrix_dim1: weight dimenstion. It is the static operand.
        Matrix_dim2: input dimenstion.
        """
        # print_msg=True
        energy_dict = {}
        energy_dict['comp'] = {i: [0, 0] for i in ["total", "laser", "ADC", "RX", "adder"]}
        energy_dict['datamovement'] = { i:[0, 0] for i in ["total", "RF", "LB", "GB", "DRAM"] }
        # weight_reuse_factor = 16 # the weight matrix will be reused for how many cols of input
        
        N1, D1 = matrix_dim1
        D2, N2 = matrix_dim2
        D = D1
        assert (D1 == D2), f"Got incorrect matrix dimension."
        
        # Set weight reuse for all dimensions of N2
        if self.weight_reuse_factor == -1:
            self.weight_reuse_factor = N2

        num_computation = N1 * N2 * D

        # map a N * N weight matrix, then for loop N2
        iter_N1 = math.ceil(N1 / self.hw.core_height)
        iter_D = math.ceil(D / self.hw.core_width)
        iter_N2 = N2

        if print_msg:
            print("**" * 10)
            print("Energy estimation for Linear")
            print(f"M1 matrix dims: {N1} * {D1}")
            print(f"M2 matrix dims: {N2} * {D2}")
            print(
                f"The loop number of N1, N2, D is {iter_N1}, {iter_N2}, {iter_D}")

        # computation related energy
        D2A_energy = self.hw.cal_D2A_energy()
        TX_energy = self.hw.cal_TX_energy()
        laser_energy = self.hw.cal_laser_energy()
        comp_energy_static, comp_energy_dynamic = self.hw.cal_comp_energy()
        A2D_energy = self.hw.cal_A2D_energy()
        RX_energy = self.hw.cal_RX_energy()
        Adder_energy = self.adder_power / self.work_freq

        # calculate the full-range support factor
        # add-drop: 2
        # mrr-bank: 4
        if self.full_range_support_factor == 2:
            full_range_support_factor_x = 2
            full_range_support_factor_w = 1  # assume can be full-range
        elif self.full_range_support_factor == 4:
            full_range_support_factor_x = 2
            full_range_support_factor_w = 2
        else:
            full_range_support_factor_w, full_range_support_factor_x = 1, 1

        # computation
        energy_comp_laser = laser_energy * \
            (iter_N1 * iter_N2 * iter_D) * 1e-9 * self.full_range_support_factor
        # assuming weight can be reused
        # weight: N1 * D
        # input: N2 * D
        # weight is shared with multiple data, thus DAC cost is reduced by sharing factor
        energy_comp_D2A_1 = D2A_energy * \
            (num_computation // self.weight_reuse_factor) * \
            1e-9 * full_range_support_factor_w

        energy_comp_D2A_2 = D2A_energy * \
            (iter_N1 * N2 * D) * 1e-9 * full_range_support_factor_x
        
        energy_comp_D2A = energy_comp_D2A_1 + energy_comp_D2A_2

        # only input, weight part is considered in comp
        energy_comp_TX = TX_energy * \
            (iter_N1 * N2 * D) * 1e-9 * self.full_range_support_factor

        # computation: MRR weight encoding: dynamic and static
        energy_comp_comp = num_computation * comp_energy_static * \
            1e-9 * self.full_range_support_factor
        energy_comp_comp += (num_computation // self.weight_reuse_factor) * comp_energy_dynamic * \
            1e-9 * self.full_range_support_factor
        
        energy_comp_ADC = A2D_energy * (N1 * N2 * iter_D) * 1e-9 * self.full_range_support_factor
        energy_comp_RX = RX_energy * (N1 * N2 * iter_D) * 1e-9 * self.full_range_support_factor
        energy_comp_adder = Adder_energy * (N1 * N2 * iter_D) * 1e-9 * self.full_range_support_factor
        energy_comp_output = energy_comp_ADC + energy_comp_RX + energy_comp_adder
        
        energy_comp = energy_comp_laser + energy_comp_D2A + energy_comp_TX + energy_comp_comp + energy_comp_output

        # datamovement related energy
        self.num_byte = self.in_bit / 16
        # RF
        energy_dm_RF = self.num_byte * self.data_movement_RF * (N1 * D * (
            N2 // self.weight_reuse_factor) * full_range_support_factor_w + iter_N1 * N2 * D * full_range_support_factor_x) * 2
        energy_dm_RF += self.num_byte * self.data_movement_RF * (N1 * N2 * iter_D * self.full_range_support_factor) / self.num_pe_per_tile * 2
        energy_dm_RF += (N1 * N2 * iter_D * self.full_range_support_factor) * self.data_movement_NoC * self.num_byte

        # GLB
        energy_dm_GLB1_input = self.num_byte * self.data_movement_GB1 * \
            (N1 * D * (N2 // self.weight_reuse_factor) * full_range_support_factor_w +
             iter_N1 * D * N2 * full_range_support_factor_x)
        
        # each core generates self.hw.core_height * N2 * self.act_bit / 8 data
        # we have iter_D / self.num_pe_per_tile * self.full_range_support_factor copies of data
        output_load_time = max(math.ceil(self.hw.core_height * N2 * self.act_bit / 8 / self.local_buffer_size), 1)
        if output_load_time > 1:
            # need reload
            energy_dm_GLB1_output = self.num_byte * \
                self.data_movement_GB1 * (N1 * N2) * (iter_D / self.num_pe_per_tile * self.full_range_support_factor * 2 - 1)
        else:
            energy_dm_GLB1_output = self.num_byte * \
                self.data_movement_GB1 * (N1 * N2) * iter_D / self.num_pe_per_tile * self.full_range_support_factor # only writes those data to GLB1

        energy_dm_GLB1_input_from_GLB2 = self.num_byte * \
            self.data_movement_GB1 * (N1*D * full_range_support_factor_w + N2 * D * iter_N1 * full_range_support_factor_x)

        energy_dm_GLB1 = energy_dm_GLB1_input + energy_dm_GLB1_output + energy_dm_GLB1_input_from_GLB2

        # GLB2
        # GLB2 -> GLB1: input
        energy_dm_GLB2 = self.num_byte * self.data_movement_GB2 * (N1*D * full_range_support_factor_w + N2 * D * iter_N1 * full_range_support_factor_x)
        # GLB2 <-> GLB1: output
        if output_load_time > 1:
            # we write partials sums back to GLB2
            # it will generates full_range_support factor that much due to inability of full-range support
            energy_dm_GLB2 += self.num_byte * self.data_movement_GB2 * N1 * N2 * (iter_D / self.num_pe_per_tile * self.full_range_support_factor * 2 -2)
        energy_dm_GLB2 += self.num_byte * self.data_movement_GB2 * N1 * N2 # assume that output is fully added at the GLB1 level
        # DRAM -> GLB2
        energy_dm_GLB2 += self.num_byte * self.data_movement_GB2 * (N1 * D * full_range_support_factor_w)

        # DRAM
        # energy_dm_DRAM = self.num_byte * (self.data_movement_DRAM * (
        #     N1 * N2) + self.data_movement_DRAM_GB * (N1 * D * full_range_support_factor_w + N2 * D * full_range_support_factor_x))
        energy_dm_DRAM = self.num_byte * self.data_movement_DRAM * (N1 * D * full_range_support_factor_w)

        energy_dm = energy_dm_DRAM + energy_dm_GLB1 + energy_dm_GLB2 + energy_dm_RF

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

    def save(self, sv_name, sv_path='./simulate_res/'):
        ensure_dir(sv_path)
        energy_file_name = os.path.join(sv_path, f'{sv_name}_energy.csv')
        self.__save_csv(energy_file_name, self.energy_dict, 'FFN')

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

    def get_energy(self, matrix_dim1, matrix_dim2):
        if self.core_type == "dota":
            energy = self.get_energy_crossbar(matrix_dim1, matrix_dim2)
            latency = self.get_latency_crossbar(matrix_dim1, matrix_dim2)
        elif self.core_type == "mrrbank":
            energy = self.get_energy_mrrbank(matrix_dim1, matrix_dim2)
            latency = self.get_latency_mrr_bank(matrix_dim1, matrix_dim2)
        elif self.core_type == "mzi":
            energy = self.get_energy_mzi(matrix_dim1, matrix_dim2)
            latency = self.get_latency_mzi(matrix_dim1, matrix_dim2)
        else:
            raise NotImplementedError

        return energy

    def run(self, print_msg=False):
        if print_msg:
            print("Report linear layer energy estimation")
            print("--" * 10)
        matrix_dim1 = (self.out_features, self.in_features) # weights
        matrix_dim2 = (self.in_features, self.bs) # inputs
        if self.core_type == "dota":
            energy, energy_dict = self.get_energy_crossbar(matrix_dim1, matrix_dim2, print_msg=print_msg)
            latency, latency_dict = self.get_latency_crossbar(matrix_dim1, matrix_dim2, print_msg=print_msg)
        elif self.core_type == "mrrbank":
            energy, energy_dict = self.get_energy_mrrbank(matrix_dim1, matrix_dim2, print_msg=print_msg)
            latency, latency_dict = self.get_latency_mrr_bank(matrix_dim1, matrix_dim2, print_msg=print_msg)
        elif self.core_type == "mzi":
            energy, energy_dict = self.get_energy_mzi(matrix_dim1, matrix_dim2, print_msg=print_msg)
            latency, latency_dict = self.get_latency_mzi(matrix_dim1, matrix_dim2, print_msg=print_msg)
        else:
            raise NotImplementedError
        
        self.energy_dict['linear'] = energy_dict
        self.latency_dict['linear'] = latency_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default=".params.yaml",
                        metavar="FILE", help="config file")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    
    OPs_list = []
    # deit-base
    OPs_list.append({"idx": 0, "type": "fc", "in_features": 384,
                    "out_features": 1152, "bs": 197})

    for item in OPs_list:
        idx = item["idx"]
        if item["type"] == "fc":
            for i in range(2):
                configs.arch.input_mod_sharing_flag = i%2
                predictor = FFNPrediction(item, configs)
                predictor.run()
                predictor.save(f'FFN_deit_small_{configs.arch.input_mod_sharing_flag}')
