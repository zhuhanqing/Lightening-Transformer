# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu
# @Date:   2023-02-06 15:30:48
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-11 02:12:05
import math
import argparse
from .photonic_core_base import PhotonicCore
from utils.config import configs

__all__ = ["PhotonicMRRBank"]


class PhotonicMRRBank(PhotonicCore):
    """Simulation of photonic mrr bank arch
    Implement of [Nw, Nh] * [Nh, Nw] matrix vector multiplication
    """

    def __init__(self, 
                 core_type="mrr_bank", 
                 core_width=32, 
                 core_height=32, 
                 in_bit=4, 
                 w_bit=4, 
                 act_bit=4, 
                 config=None) -> None:
        super().__init__()
        self.core_type = core_type
        assert self.core_type == "mrr_bank", f"The photonic core should be dota, but got {self.core_type}"
        # capble of core_width * core_height matrix vector multiplication
        self.core_width = core_width
        self.core_height = core_height

        # set work freq
        self.work_freq = config.core.work_freq if config is not None else 1  # GHz
        
        # precision params
        self.in_bit = in_bit
        self.w_bit = w_bit
        self.act_bit = act_bit

        # obtain params
        self._initialize_params(config=config)

        # cal params: insertion loss and modulator
        self.insertion_loss = None
        self.insertion_loss_computation = None
        self.insertion_loss_modulation = None
        self.cal_insertion_loss(print_msg=False)
        self.cal_laser_power(print_msg=False)
        self.cal_modulator_param(print_msg=False)
        self.cal_ADC_param(print_msg=False)
        self.cal_DAC_param(print_msg=False)

        self.input_modulator_power_dynamic = self.modulator_power_dynamic * \
            self.in_bit / self.w_bit

    def _initialize_params(self, config=None):
        """Initializes required devices params for MRR bank.
            Input: laser, micro comb, array of input modulator.
            Computation: splitter, mrr bank.
            Output: photodetector.
            Digital: ADC, DAC, TIA.
        """
        # common params
        self._obtain_laser_param(config.device.laser)
        self._obtain_micro_comb_param(config.device.micro_comb)
        self._obtain_y_branch_param(config.device.y_branch)
        self._obtain_photodetector_param(config.device.photo_detector)

        self._obtain_modulator_param(config.device.mrr_modulator) # weight
        self._obtain_input_modulator_param(config.device.mrr_modulator) # input

        # DAC and ADC
        self._obtain_ADC_param(config.core.interface.ADC)
        self._obtain_DAC_param(config.core.interface.DAC)
        self._obtain_TIA_param(config.core.interface.TIA)

    def _obtain_input_modulator_param(self, config=None):
        # "override the modulator function"
        if config is not None:
            self.input_modulator_type = config.type
            assert self.input_modulator_type == 'ring'
            self.input_modulator_energy_per_bit = config.energy_per_bit
            self.input_modulator_power_static = config.static_power
            self.input_modulator_length = config.length
            self.input_modulator_width = config.width
            self.input_modulator_insertion_loss = config.insertion_loss
            self.input_modulator_insertion_loss_uc = config.insertion_loss_uc
        else:
            self.input_modulator_energy_per_bit = 400
            self.input_modulator_static_power = 0
            self.input_modulator_length = 300
            self.input_modulator_width = 50
            self.input_modulator_insertion_loss = 0.8
            self.input_modulator_insertion_loss_uc = 0.1
            
    def _obtain_modulator_param(self, config=None):
        if config is not None:
            self.modulator_type = config.type
            assert self.modulator_type == 'ring'
            self.modulator_energy_per_bit = config.energy_per_bit
            self.modulator_power_static = config.static_power
            self.modulator_length = config.length
            self.modulator_width = config.width
            self.modulator_insertion_loss = config.insertion_loss
            self.modulator_insertion_loss_uc = config.insertion_loss_uc
        else:
            self.modulator_energy_per_bit = 400
            self.modulator_static_power = 0
            self.modulator_length = 300
            self.modulator_width = 50
            self.modulator_insertion_loss = 0.8
            self.modulator_insertion_loss_uc = 0.1

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
            self.cal_DAC_param()
            # self.cal_modulator_param()

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
        # compute the insertion loss for modulation part
        # modulation: ring-based modulator (#\lambda * md only one is coupled) + splitter(log2(k) stage * y branch)
        # computaion: ring-based modulator (#\lambda * md)
        self.insertion_loss_modulation = self.modulator_insertion_loss_uc * (self.core_width - 1) + self.modulator_insertion_loss \
            + self.y_branch_insertion_loss * math.ceil(math.log2(self.core_height))
        self.insertion_loss_computation = self.modulator_insertion_loss_uc * \
            (self.core_width - 1) + self.modulator_insertion_loss
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
        # light is splited into core hight copy: input broadcast
        P_laser_dbm = self.photo_detector_sensitivity + self.insertion_loss_computation \
            + self.insertion_loss_modulation + 10 * math.log10(self.core_height)
        self.laser_power = 10 ** (P_laser_dbm / 10) / self.laser_wall_plug_eff * 2 ** self.act_bit

        if print_msg:
            print(f'required laser power is {self.laser_power} mW with {P_laser_dbm} db requirement')

    def cal_modulator_param(self, print_msg=False):
        # indepent to bit width
        self.modulator_power_dynamic = self.modulator_energy_per_bit * \
            self.work_freq * 1e-3  # mW
        self.input_modulator_power_dynamic = self.modulator_power_dynamic * \
            self.in_bit / self.w_bit
        self.weight_modulator_power_dynamic = self.modulator_power_dynamic
        if print_msg:
            print(
                f"input modulator dynamic power: {self.input_modulator_power_dynamic: .2f} mW")
            print(
                f"weight modulator dynamic power: {self.weight_modulator_power_dynamic: .2f} mW")
            print(
                f"modulator static power: {self.modulator_power_static: .2f} mW")

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

    def cal_core_power(self):
        # add up the power of all device
        # input: laser, dac + modulator
        self.input_power_laser = self.laser_power
        self.input_power_dac = (
            self.core_width + self.core_width * self.core_height) * self.core_DAC_power
        self.input_power_modulation = (
            self.modulator_power_static + self.input_modulator_power_dynamic) * self.core_width

        self.input_power = (
            self.input_power_laser +
            self.input_power_dac +
            self.input_power_modulation
        )

        self.computation_power = (self.modulator_power_static +
                                  self.weight_modulator_power_dynamic) * self.core_height * self.core_width

        self.output_power_adc = self.core_height * \
            self.core_ADC_power / self.core_ADC_sharing_factor
        # print(self.core_ADC_power)
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

    def cal_core_area(self):
        """Calculate the PTC area"""
        area = 0
        # 5 um spacing
        self.mrr_bank_width = self.core_width * (self.modulator_length + 5) + self.photo_detector_width + 5
        self.mrr_bank_height = max(self.modulator_width, 2 * self.photo_detector_length)
        
        # mrr banks + modulation part
        self.mrr_bank_area = self.mrr_bank_height * self.mrr_bank_width * self.core_height + self.core_width * (self.modulator_length + 5) * self.modulator_width
        
        # 1 to self.core_height splitter
        self.y_branch_area = (self.y_branch_length * math.ceil(math.log2(self.core_height) + 1)) * (self.y_branch_width * self.core_height)
        
        area = self.mrr_bank_area * self.y_branch_area

        return area

    ########################################################################
    def cal_D2A_energy(self):
        self.D2A_energy = self.core_DAC_power / (self.work_freq)  # pJ
        return self.D2A_energy

    def cal_TX_energy(self):
        TX_energy = ((self.modulator_power_static +
                     self.input_modulator_power_dynamic)) / self. work_freq
        self.TX_energy = TX_energy
        return TX_energy

    def cal_A2D_energy(self):
        self.A2D_energy = self.core_ADC_power / (self.work_freq)  # pJ
        return self.A2D_energy

    def cal_RX_energy(self):
        self.RX_energy = (self.photo_detector_power * 2 + self.TIA_power) / (self.work_freq)
        return self.RX_energy

    def cal_comp_energy(self):
        """weight part may be weight-stationary, the dynamic modulation energy is amortized"""
        self.comp_energy = self.modulator_power_static / (self.work_freq)
        self.comp_energy_dynamic = self.weight_modulator_power_dynamic / self.work_freq
        return self.comp_energy, self.comp_energy_dynamic

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

    width, height = 16, 16
    test_pc = PhotonicMRRBank(
        core_width=width, core_height=height, config=configs)

    # test_pc.set_work_frequency(work_freq=work_freq)
    # # test_pc.cal_modulator_power()
    # core_power = test_pc.cal_core_power()
    # print(
    #     f"Energy efficiency is {2 * 8**3 * work_freq/ (core_power * 1e-3) * 1e-3} TOPS/W")

    work_freq = 5
    test_pc.set_work_frequency(work_freq=work_freq)
    pc_power = test_pc.cal_core_power()
    # energy efficiency is half since it cannot do general matrix multiplication
    print(
        f"Energy efficiency is {2* width * height * work_freq/ (pc_power * 1e-3) * 1e-3} TOPS/W")
    print(f"MAC energy efficiency is {pc_power / work_freq / (width * height)* 1e3} fJ/MAC")
    
    print(test_pc.cal_TX_energy())
    print(test_pc.cal_D2A_energy())
    print(test_pc.cal_comp_energy())
    print(test_pc.cal_A2D_energy())
    print(test_pc.cal_RX_energy())
