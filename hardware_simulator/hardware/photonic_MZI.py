# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu
# @Date:   2023-02-06 15:30:48
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-11 02:12:17
# basic simulator for photonic dota
import argparse
import math
from .photonic_core_base import PhotonicCore
from utils.config import configs

__all__ = ["PhotonicMZI"]

class PhotonicMZI(PhotonicCore):
    """Simulation of MZI-style. Default of clement-style to reduce Insertion loss. Implement MVM between [Nw, Nh] * [Nh, 1]
    """

    def __init__(self, 
                 core_type="mzi", 
                 core_width=32, 
                 core_height=32, 
                 in_bit=4, 
                 w_bit=4, 
                 act_bit=4, 
                 config=None) -> None:
        super().__init__()
        self.core_type = core_type
        assert self.core_type == "mzi", f"The photonic core should be dota, but got {self.core_type}"
        self.core_width = core_width  # core_width of photonic dota
        self.core_height = core_height  # core_height of photonic dota
        
        # precision params
        self.in_bit = in_bit
        self.w_bit = w_bit
        self.act_bit = act_bit
        
        # obtain params
        self._initialize_params(config=config)
        
        self.num_mzis = self.core_height * (self.core_height - 1) / 2 + self.core_width * (self.core_width - 1) / 2
        self.mzi_sigma_dim = max(self.core_height, self.core_width)
        

        # set work freq
        self.work_freq = config.core.work_freq if config is not None else 1  # GHz
        # self.set_work_frequency(config.core.work_freq)

        # cal params
        # insertion loss and dynamic modulator power(@freq and precision)
        self.insertion_loss = None
        self.insertion_loss_computation = None
        self.insertion_loss_modulation = None
        self.cal_insertion_loss(print_msg=False)
        self.cal_laser_power(print_msg=False)
        self.cal_modulator_param(print_msg=False)
        self.cal_mzi_param(print_msg=False) # need mzi params
        self.cal_ADC_param(print_msg=False)
        self.cal_DAC_param(print_msg=False)
        
        self.input_modulator_power_dynamic = self.modulator_power_dynamic * \
            self.in_bit / self.w_bit

    def _initialize_params(self, config=None):
        """Initializes required devices params for MZI.
            Input: laser, MZM.
            Computation: MZI.
            Output: photodetector.
            Digital: ADC, DAC, TIA.
        """
        # common params
        self._obtain_laser_param(config.device.laser)
        self._obtain_y_branch_param(config.device.y_branch)
        self._obtain_photodetector_param(config.device.photo_detector)

        self._obtain_mzi_param(config.device.mzi)
        self._obtain_modulator_param(config.device.mzi_modulator)
        self._obtain_phase_shifter_param(config.device.phase_shifter)
        self._obtain_direction_coupler_param(config.device.direction_coupler)

        # DAC and ADC
        self._obtain_ADC_param(config.core.interface.ADC)
        self._obtain_DAC_param(config.core.interface.DAC)
        self._obtain_TIA_param(config.core.interface.TIA)
    
    def set_ADC_sharing_factor(self, sharing_factor):
        self.core_ADC_sharing_factor = sharing_factor

    def set_precison(self, in_bit, w_bit, act_bit):
        # set input, weight and activation bit-core_width to scale AD/DA energy consumption.
        if (in_bit != self.in_bit) or (w_bit != self.w_bit) or (act_bit != self.act_bit):
            print(
                f"core precision change from in-{self.in_bit} w-{self.w_bit} act-{self.act_bit} to in-{in_bit} w-{w_bit} act-{act_bit}")
            self.in_bit = in_bit
            self.w_bit = w_bit
            self.a_bit = act_bit
            self.cal_ADC_param()
            self.cal_DAC_param() # will update midulator

    def set_work_frequency(self, work_freq):
        # set work frequency -> energy should be power * frequency
        if self.work_freq != work_freq:
            # recalculate fre-realted params
            # modulator dynamic power, adc and dac power
            print(
                f"Work frequency of the photonic core change from {self.work_freq} GHz to {work_freq} GHz")
            self.work_freq = work_freq
            self.cal_ADC_param()
            self.cal_DAC_param()
            self.cal_modulator_param()

    def cal_insertion_loss(self, print_msg=False):
        "Function to compute insertion loss"
        # mzi implement weight matrix M * N -> N^2 -> max(M, N) -> M^2
        # for modulation, we only consider single modulator
        self.insertion_loss_modulation = self.modulator_insertion_loss
        # clement design: M + N IL + 1
        self.insertion_loss_computation = self.mzi_insertion_loss * \
            (self.core_height + self.core_width) + self.mzi_insertion_loss
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
        if self.insertion_loss_modulation is None or self.insertion_loss_computation is None:
            self.cal_insertion_loss()
            
        P_laser_dbm = self.photo_detector_sensitivity + self.insertion_loss_modulation + self.insertion_loss_computation + \
            10 * math.log10(self.core_width)  # * self.core_height
        self.laser_power = 10 ** (P_laser_dbm / 10) / \
            self.laser_wall_plug_eff * 2 ** self.act_bit

        if print_msg:
            print(
                f'required laser power is {self.laser_power} mW with {P_laser_dbm} db requirement')

    def _obtain_mzi_param(self, config=None):
        if config is not None:
            self.modulator_type = config.type
            assert self.modulator_type == 'mzi'
            self.mzi_energy_per_bit = config.energy_per_bit
            self.mzi_power_static = config.static_power
            self.mzi_length = config.length
            self.mzi_width = config.width
            self.mzi_insertion_loss = config.insertion_loss
            self.mzi_response_time = config.response_time
        else:
            raise NotImplementedError
            self.mzi_energy_per_bit = 400
            self.mzi_power_static = 0
            self.mzi_length = 300
            self.mzi_width = 50
            self.mzi_insertion_loss = 0.8
            self.mzi_response_time = 2e-3

    def cal_modulator_param(self, print_msg=False):
        # indepent to bit width
        self.modulator_power_dynamic = self.modulator_energy_per_bit * \
            self.work_freq * 1e-3  # mW
        if print_msg:
            print(
                f"modulator static power: {self.modulator_power_static: .2f} mW")
            print(
                f"modulator dynamic power: {self.modulator_power_dynamic: .2f} mW")

    def cal_mzi_param(self, print_msg=False):
        # indepent to bit width
        self.mzi_power_dynamic = self.mzi_energy_per_bit * \
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
        self.cal_mzi_param(print_msg=False) # need updata mzi as well
        self.DAC.set_DAC_work_freq(self.work_freq)
        self.DAC.set_DAC_work_prec(self.in_bit)
        self.DAC.cal_DAC_param(print_msg=print_msg)
        self.core_DAC_power = self.DAC.DAC_power
        self.core_DAC_area = self.DAC.DAC_area

    def cal_core_area(self):
        area = 0
        # compute single node area
        # use mzi number to approximate
        self.photonic_core_area = (self.core_width*(self.core_width - 1) + self.core_height*(self.core_height - 1)) / 2 * self.mzi_length * self.mzi_width # UV
        self.photonic_core_area += max(self.core_height, self.core_width) * self.modulator_length * self.modulator_width # Sigma using mzi modulator
        self.photonic_core_area += self.core_height * self.photo_detector_length * self.photo_detector_width
        self.y_branch_area =  (self.y_branch_length * math.ceil(math.log2(self.core_height) + 1)) * (self.y_branch_width * self.core_height) # splitter
        self.mzm_area = self.core_width * self.modulator_length * self.modulator_width
        self.photonic_core_dac_area = (self.core_width*(self.core_width - 1) + self.core_height*(self.core_height - 1) + max(self.core_height, self.core_width)) * self.core_DAC_area
        self.photonic_core_dac_area += self.core_height * self.core_DAC_area
        self.photonic_core_adc_area = self.core_width * self.core_ADC_area
        return area

    def cal_core_power(self):
        
        # add up the power of all device
        # input: laser, dac + modulator
        self.input_power_laser = self.laser_power
        # N * self.num_wavelength
        self.input_power_dac = self.core_width * \
            self.core_DAC_power + self.num_mzis * self.core_DAC_power
        # N * self.num_wavelength
        self.input_power_modulation = self.core_width * (
            self.modulator_power_static + self.modulator_power_dynamic)

        self.input_power = (
            self.input_power_laser +
            self.input_power_dac +
            self.input_power_modulation
        )

        # M * (M - 1) / 2 MZIs + max(M, N) MZI-based modulator
        self.computation_power = self.num_mzis * (self.mzi_power_static + self.mzi_power_dynamic) + max(self.core_height, self.core_width) * (self.modulator_power_dynamic + self.modulator_power_static)

        self.output_power_adc = self.core_height * self.core_ADC_power / self.core_ADC_sharing_factor
        self.output_power_detection = self.core_height * (self.photo_detector_power + self.TIA_power)
        self.output_power = (
            self.output_power_adc +
            self.output_power_detection
        )
        self.core_power = self.input_power + self.computation_power + self.output_power
        print('***' * 10)
        print(f'overall static power: {self.core_power} mW')
        print(
            f'--input part: {self.input_power} mW   -->  {self.input_power / self.core_power * 100: .2f} %')
        print(
            f'----laser part: {self.input_power_laser} mW   -->  {self.input_power_laser / self.core_power * 100: .2f} %')
        print(
            f'----dac part: {self.input_power_dac} mW   -->  {self.input_power_dac / self.core_power * 100: .2f} %')
        print(
            f'----modulator part: {self.input_power_modulation} mW   -->  {self.input_power_modulation / self.core_power * 100: .2f} %')
        print(
            f'--comp part: {self.computation_power} mW   -->  {self.computation_power / self.core_power * 100: .2f} %')
        print(
            f'--ouput part: {self.output_power} mW   -->  {self.output_power / self.core_power * 100: .2f} %')
        print(
            f'----adc part: {self.output_power_adc} mW   -->  {self.output_power_adc / self.core_power * 100: .2f} %')
        print(
            f'----detection part: {self.output_power_detection} mW   -->  {self.output_power_detection / self.core_power * 100: .2f} %')

        return self.core_power

    ########################################################################
    def cal_D2A_energy(self):
        self.D2A_energy = self.core_DAC_power / (self.work_freq)  # pJ
        return self.D2A_energy

    def cal_TX_energy(self):
        # mzi modulator
        self.TX_energy = ((self.modulator_power_dynamic +
                          self.modulator_power_static)) / self. work_freq
        return self.TX_energy

    def cal_A2D_energy(self):
        self.A2D_energy = self.core_ADC_power / (self.work_freq)  # pJ
        return self.A2D_energy

    def cal_RX_energy(self):
        # We add TIA here since we must add this for sigma
        self.RX_energy = (self.photo_detector_power * 2 + self.TIA_power) / (self.work_freq)
        return self.RX_energy

    def cal_comp_energy(self):
        # we have static power and dynamic power which may can be amortized
        # self.comp_energy = (self.phase_shifter_power_dynamic + self.phase_shifter_power_static) / (self.work_freq)
        self.comp_energy_static = self.mzi_power_static / self.work_freq
        self.comp_energy_dynamic = self.mzi_power_dynamic / self.work_freq
        self.comp_energy = self.comp_energy_dynamic + self.comp_energy_dynamic
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
    test_pc = PhotonicMZI(core_width=configs.core.width,
                          core_height=configs.core.height,
                          in_bit=configs.core.precision.in_bit,
                          w_bit=configs.core.precision.w_bit,
                          act_bit=configs.core.precision.act_bit,
                          config=configs)

    work_freq = 5
    test_pc.set_work_frequency(work_freq=work_freq)
    crossbar_power = test_pc.cal_core_power()
    # area = test_pc.cal_core_area()
    print(
        f"Energy efficiency is {2 * width*height*wavelength * work_freq/ (crossbar_power * 1e-3) * 1e-3} TOPS/W")
    # print(
    #     # f"Computation density is {2 * width*height*wavelength * work_freq/ (area * 1e-6) * 1e-3} TOPS/mm^2")
    print(
        f"MAC energy efficiency is {crossbar_power / work_freq / (width * height*wavelength)* 1e3} fJ/MAC")
    # print(test_pc.cal_TX_energy())
    # print(test_pc.cal_D2A_energy())
    # print(test_pc.cal_comp_energy())
    # print(test_pc.cal_A2D_energy())
    # print(test_pc.cal_RX_energy())
