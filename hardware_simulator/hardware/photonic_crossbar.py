# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu
# @Date:   2023-02-06 15:30:48
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-11 02:12:24
# basic simulator for photonic dota
import argparse
import math
import logging
from .photonic_core_base import PhotonicCore
from utils.config import configs

__all__ = ["PhotonicCrossbar"]


class PhotonicCrossbar(PhotonicCore):
    """Simulation of our DPTC core implements a [Nh, Nx] * [Nw, Nw] matrix multiplication
    Start with a laser, then multi-wavelength (wavelength num: D) comb, then high-speed modulator for different wavelength.
    At each cross-point, coupler will couple 1/N (N = max(Nw, Nh)) portion of light into individual len-D vector product following one PS + one coupler.
    Use differential photodetector to do detection.
    """
    def __init__(self, 
                 core_type="dota", 
                 core_width=32, 
                 core_height=32, 
                 num_wavelength=32, 
                 in_bit=4, 
                 w_bit=4, 
                 act_bit=4, 
                 config=None) -> None:
        '''
        Args:
        core_type : the type of tensor core, support "dota", "mzi", "mrr"
        core_width: the width of the tensor core, Nw
        core_height: the height of the tensor core, Nh
        num_wavelength: number of wavelengths used for single core, Nx
        in_bit: input precision
        w_bit: weight precision
        act_bit: activation precision
        '''
        super().__init__()
        self.core_type = core_type
        assert self.core_type == "dota", f"The photonic core should be dota, but got {self.core_type}"
        # basic photonic dota params: Nw, Nh, Nx
        self.core_width = core_width  # core_width of photonic dota
        self.core_height = core_height  # core_height of photonic dota
        # num of wavelength used in WDM for len-k vec computation
        self.num_wavelength = num_wavelength

        # precision params
        self.in_bit = in_bit
        self.w_bit = w_bit
        self.act_bit = act_bit

        self.core_ADC_sharing_factor = 1 # whether we share ADC across outputs, multi-channel ADC

        # set work freq
        self.work_freq = config.core.work_freq if config is not None else 1  # GHz
        # self.set_work_frequency(config.core.work_freq)

        # obtain device params
        self._initialize_params(config=config)

        # cal params
        self.insertion_loss = None
        self.insertion_loss_computation = None
        self.insertion_loss_modulation = None
        self.cal_insertion_loss(print_msg=False)
        self.cal_laser_power(print_msg=False)
        self.cal_modulator_param(print_msg=False)
        self.cal_ADC_param(print_msg=False)
        self.cal_DAC_param(print_msg=False)

    def _initialize_params(self, config=None):
        """Initializes required devices params for Photonic dota.
            Input: laser, micro comb, input modulator.
            Computation: ps, dc.
            Output: photodetector.
            Digital: ADC, DAC, TIA.
        """
        self._obtain_laser_param(config.device.laser)
        self._obatin_modulator_param(config.device.mzi_modulator)
        self._obtain_mrr_router_param(config.device.mrr_router)
        self._obtain_phase_shifter_param(config.device.phase_shifter)
        self._obtain_direction_coupler_param(config.device.direction_coupler)
        self._obtain_photodetector_param(config.device.photo_detector)
        self._obtain_y_branch_param(config.device.y_branch)
        self._obtain_micro_comb_param(config.device.micro_comb)

        self._obtain_ADC_param(config.core.interface.ADC)
        self._obtain_DAC_param(config.core.interface.DAC)
        self._obtain_TIA_param(config.core.interface.TIA)
    
    def set_ADC_sharing_factor(self, sharing_factor):
        self.core_ADC_sharing_factor = sharing_factor
    
    def set_precison(self, in_bit, w_bit, act_bit):
        # set input, weight and activation bit-core_width to scale AD/DA energy consumption.
        if (in_bit != self.in_bit) or (w_bit != self.w_bit) or (act_bit != self.act_bit):
            logging.info(
                "core precision change from in: %d-bit w: %d-bit act: %d-bit to %d-bit w: %d-bit act: %d-bit",
                self.in_bit, self.w_bit, self.act_bit, in_bit, w_bit, act_bit)
            self.in_bit = in_bit
            self.w_bit = w_bit
            self.a_bit = act_bit
            self.cal_ADC_param()
            self.cal_DAC_param()

    def set_work_frequency(self, work_freq):
        # set work frequency -> energy should be power * frequency
        if self.work_freq != work_freq:
            # recalculate freq-realted params
            # modulator dynamic power, adc and dac power
            print(
                f"Work frequency of the photonic core change from {self.work_freq} GHz to {work_freq} GHz")
            self.work_freq = work_freq
            self.cal_ADC_param()
            self.cal_DAC_param()
            self.cal_modulator_param()

    def cal_insertion_loss(self, print_msg=False):
        "Function to compute insertion loss the most lossless path"
        # ignor grating coupler since we assume this is a on-chip laser
        # modulation insertion loss: modulator, WDM Mux and demux, splitter
        # we have a 1: N splitter based on y_branch -> log2N levels
        self.insertion_loss_modulation = self.modulator_insertion_loss + \
            self.mrr_router_insertion_loss * 2 + self.y_branch_insertion_loss * \
            math.ceil(math.log2(max(self.core_height, self.core_width)))
        # computation insertion loss: DDOT units (1 splitter + 1 ps + 1 dc)
        self.insertion_loss_computation = self.y_branch_insertion_loss + \
            self.phase_shifter_insertion_loss + self.direction_coupler_insertion_loss
        self.insertion_loss = self.insertion_loss_computation \
            + self.insertion_loss_modulation
        if print_msg:
            print(
                f"insertion loss {self.insertion_loss: .2f} db")
            print(
                f"--insertion loss for modulation {self.insertion_loss_modulation: .2f} db")
            print(
                f"--insertion loss for computation {self.insertion_loss_computation: .2f} db")

    def cal_laser_power(self, print_msg=False):
        """Function to calculate the required laser power follow DAC2021 CrossLight
            P_laser - Sensitivity_detector >= P_inserionloss (dbm)
            We should scale it based on the required activation precision and laser wall plug efficiency.
            We should scale it based on whether we split lights
        """
        if self.insertion_loss_modulation is None or self.insertion_loss_computation is None:
            self.cal_insertion_loss()
        # the laser power is distributed to Nw, Nh waveguides
        P_laser_dbm = self.photo_detector_sensitivity + self.insertion_loss_modulation + self.insertion_loss_computation + \
            10 * math.log10(self.core_width * self.core_height) 
        self.laser_power = 10 ** (P_laser_dbm / 10) / self.laser_wall_plug_eff * 2 ** self.act_bit

        if print_msg:
            print(f'required laser power is {self.laser_power} mW with {P_laser_dbm} db requirement')

    def _obatin_modulator_param(self, config=None):
        if config is not None:
            self.modulator_type = config.type
            assert self.modulator_type == 'mzi'
            self.modulator_energy_per_bit = config.energy_per_bit
            self.modulator_power_static = config.static_power
            self.modulator_length = config.length
            self.modulator_width = config.width
            self.modulator_insertion_loss = config.insertion_loss
        else:
            self.modulator_energy_per_bit = 400
            self.modulator_static_power = 0
            self.modulator_length = 300
            self.modulator_width = 50
            self.modulator_insertion_loss = 0.8

    def cal_modulator_param(self, print_msg=False):
        # indepent to bit width
        self.modulator_power_dynamic = self.modulator_energy_per_bit * \
            self.work_freq * 1e-3  # mW

        if print_msg:
            print(
                f"modulator static power: {self.modulator_power_static: .2f} mW")
            print(
                f"modulator dynamic power: {self.modulator_power_dynamic: .2f} mW")

    def cal_ADC_param(self, print_msg=False):
        self.ADC.set_ADC_work_freq(self.work_freq)
        self.ADC.set_ADC_work_prec(self.act_bit)
        self.ADC.cal_ADC_param(print_msg=print_msg)
        self.core_ADC_power = self.ADC.ADC_power
        self.core_ADC_area = self.ADC.ADC_area

    def cal_DAC_param(self, print_msg=False):
        self.cal_modulator_param(print_msg=False)
        self.DAC.set_DAC_work_freq(self.work_freq)
        self.DAC.set_DAC_work_prec(self.in_bit)
        self.DAC.cal_DAC_param(print_msg=print_msg)
        self.core_DAC_power = self.DAC.DAC_power
        self.core_DAC_area = self.DAC.DAC_area

    def cal_core_area(self):
        """Function to calculate basic area"""
        area = 0
        # compute single DDOT area considering layout and spacing
        # width: y-branch width + ps width + dc width + pd width (5um spacing each -> 20 um)
        self.node_width = self.y_branch_length + self.phase_shifter_length + self.direction_coupler_length + self.photo_detector_width + 30
        self.node_height = self.y_branch_length+ max(max(self.phase_shifter_width, self.direction_coupler_width), self.photo_detector_length * 2) + 20

        self.photonic_core_node_area = self.core_height * self.core_width * self.node_height * self.node_width

        self.photonic_core_adc_area = self.core_height * self.core_width / \
            self.core_ADC_sharing_factor * self.core_ADC_area
        self.photonic_core_dac_area = (self.core_height + self.core_width) * \
            self.num_wavelength * self.core_DAC_area
        self.laser_area = self.laser_area
        self.mzi_modulator_area = (self.core_height + self.core_width) * self.num_wavelength * \
            self.modulator_length * self.modulator_width
        self.mrr_area = (self.core_height + self.core_width) * \
            self.num_wavelength * self.mrr_router_length * self.mrr_router_width * 2

        # splitter area
        self.y_branch_area = self.y_branch_length * self.y_branch_width + (self.y_branch_length * math.ceil(math.log2(self.core_height) + 1)) * (self.y_branch_width * self.core_height) + \
            (self.y_branch_length * math.ceil(math.log2(self.core_width) + 1)
             ) * (self.y_branch_width * self.core_width)

        area = self.photonic_core_node_area + self.photonic_core_adc_area+ self.photonic_core_dac_area + \
            self.laser_area + self.mzi_modulator_area + self.mrr_area + self.y_branch_area

        return area

    def cal_core_power(self):
        """Function to calculate basic power"""
        # add up the power of all device
        # input: laser, dac + modulator
        self.input_power_laser = self.laser_power
        self.input_power_dac = (self.core_height + self.core_width) * \
            self.num_wavelength * self.core_DAC_power
        self.input_power_modulation = (self.core_height + self.core_width) * self.num_wavelength * (
            self.modulator_power_static + self.modulator_power_dynamic + self.mrr_router_power * 2)

        self.input_power = (
            self.input_power_laser +
            self.input_power_dac +
            self.input_power_modulation
        )

        self.computation_power = self.core_height * self.core_width * (self.phase_shifter_power_dynamic + self.phase_shifter_power_static) * 2

        self.output_power_adc = self.core_height * \
            self.core_width * self.core_ADC_power / self.core_ADC_sharing_factor
        self.output_power_detection = self.core_height * \
            self.core_width * (self.photo_detector_power + self.TIA_power)
        self.output_power = (
            self.output_power_adc +
            self.output_power_detection
        )
        self.core_power = self.input_power + self.computation_power + self.output_power

        return self.core_power

    ########################################################################
    def cal_D2A_energy(self):
        self.D2A_energy = self.core_DAC_power / (self.work_freq)  # pJ
        return self.D2A_energy

    def cal_TX_energy(self):
        # mzi modulator
        self.TX_energy = ((self.modulator_power_dynamic +
                          self.modulator_power_static + self.mrr_router_power * 2)) / self. work_freq
        return self.TX_energy

    def cal_A2D_energy(self):
        self.A2D_energy = self.core_ADC_power / (self.work_freq)  # pJ
        return self.A2D_energy

    def cal_RX_energy(self):
        # RX: two detector + TIA
        self.RX_energy = self.photo_detector_power / (self.work_freq) * 2 + self.TIA_power / self.work_freq
        self.photo_detector_energy = self.photo_detector_power / (self.work_freq) * 2
        self.TIA_energy = self.TIA_power / self.work_freq
        return self.RX_energy

    def cal_comp_energy(self):
        self.comp_energy = (self.phase_shifter_power_dynamic + self.phase_shifter_power_static) / (self.work_freq)
        return self.comp_energy

    def cal_laser_energy(self):
        self.laser_energy = self.laser_power / (self.work_freq)
        return self.laser_energy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default=".params.yaml",
                        metavar="FILE", help="config file")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    
    width = configs.core.width
    height = configs.core.height
    wavelength = configs.core.num_wavelength
    print(configs)
    test_pc = PhotonicCrossbar(core_width=configs.core.width,
                               core_height=configs.core.height, 
                               num_wavelength=configs.core.num_wavelength, 
                               in_bit=configs.core.precision.in_bit,
                               w_bit=configs.core.precision.w_bit,
                               act_bit=configs.core.precision.act_bit,
                               config=configs)

    work_freq = 5
    test_pc.set_work_frequency(work_freq=work_freq)
    crossbar_power = test_pc.cal_core_power()
    area = test_pc.cal_core_area()
    print(
        f"Energy efficiency is {2 * width*height*wavelength * work_freq/ (crossbar_power * 1e-3) * 1e-3} TOPS/W")
    print(
        f"Computation density is {2 * width*height*wavelength * work_freq/ (area * 1e-6) * 1e-3} TOPS/mm^2")
    print(f"MAC energy efficiency is {crossbar_power / work_freq / (width * height*wavelength)* 1e3} fJ/MAC")