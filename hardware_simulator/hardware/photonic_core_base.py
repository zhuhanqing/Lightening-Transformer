# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu(hqzhu@utexas.edu)
# @Date:   1969-12-31 18:00:00
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-08 17:23:00
# Basci class for photonic core
from .ADC import ADC
from .DAC import DAC

__all__ = ["PhotoniceCore"]

class PhotonicCore():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.photonic_core_type = None
        self.width = None
        self.height = None

    ## obtain params for photonic devices
    def _obtain_laser_param(self, config=None):
        if config is not None:
            self.laser_power = config.power
            self.laser_length = config.length
            self.laser_width = config.width
            self.laser_area = self.laser_length * self.laser_width
            self.laser_wall_plug_eff = config.wall_plug_eff
        else:
            self.laser_power = 0.5
            self.laser_length = 400
            self.laser_width = 300
            self.laser_area = self.laser_length * self.laser_width
            self.laser_wall_plug_eff = 0.25

    def _obtain_micro_comb_param(self, config=None):
        if config is not None:
            self.micro_comb_length = config.length
            self.micro_comb_width = config.width
        else:
            self.micro_comb_length = 1184
            self.micro_comb_width = 1184
        self.micro_comb_area = self.micro_comb_length * self.micro_comb_width

    # modulator
    def _obtain_modulator_param(self, config=None):
        if config is not None:
            self.modulator_type = config.type
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

    # basic devices
    def _obtain_y_branch_param(self, config=None):
        if config is not None:
            self.y_branch_length = config.length
            self.y_branch_width = config.width
            self.y_branch_insertion_loss = config.insertion_loss
        else:
            self.y_branch_length = 75
            self.y_branch_width = 3.9
            self.y_branch_insertion_loss = 0.1

    def _obtain_photodetector_param(self, config=None):
        if config is not None:
            self.photo_detector_power = config.power
            self.photo_detector_length = config.length
            self.photo_detector_width = config.width
            self.photo_detector_sensitivity = config.sensitivity
        else:
            self.photo_detector_power = 2.8
            self.photo_detector_length = 40
            self.photo_detector_width = 40
            self.photo_detector_sensitivity = -25

    def _obtain_direction_coupler_param(self, config=None):
        if config is not None:
            self.direction_coupler_length = config.length
            self.direction_coupler_width = config.width
            self.direction_coupler_insertion_loss = config.insertion_loss
        else:
            self.direction_coupler_length = 75
            self.direction_coupler_width = 10
            self.direction_coupler_insertion_loss = 0.3

    def _obtain_phase_shifter_param(self, config=None):
        if config is not None:
            self.phase_shifter_power_dynamic = config.dynamic_power
            self.phase_shifter_power_static = config.static_power
            self.phase_shifter_length = config.length
            self.phase_shifter_width = config.width
            self.phase_shifter_insertion_loss = config.insertion_loss
            # self.phase_shifter_programming_time = config.programming_time
        else:
            self.phase_shifter_power_dynamic = 0
            self.phase_shifter_power_static = 0
            self.phase_shifter_length = 200
            self.phase_shifter_width = 34
            self.phase_shifter_insertion_loss = 0.2
            # self.phase_shifter_programming_time = 10 # ns based on NEOM
            

    def _obtain_mrr_router_param(self, config=None):
        if config is not None:
            self.mrr_router_power = config.static_power
            self.mrr_router_length = config.length
            self.mrr_router_width = config.width
            self.mrr_router_insertion_loss = config.insertion_loss
        else:
            self.mrr_router_power = 2.4
            self.mrr_router_length = 20
            self.mrr_router_width = 20
            self.mrr_router_insertion_loss = 0.25   

    def _obtain_TIA_param(self, config=None):
        if config is not None:
            self.TIA_power = config.power
            self.TIA_area = config.area
        else:
            raise NotImplementedError
            self.TIA_power = 3
            self.TIA_area = 5200
            
    def _obtain_ADC_param(self, config=None):
        if config is not None:
            ADC_choice = config.choice
            self.core_ADC_sharing_factor = config.sharing_factor
            self.ADC = ADC(ADC_choice)
        else:
            raise NotImplementedError
    
    def _obtain_DAC_param(self, config=None):
        if config is not None:
            DAC_choice = config.choice
            self.DAC = DAC(DAC_choice)
        else:
            raise NotImplementedError
    
    
    ## calculate area, insertion loss and energy cost
    def cal_insertion_loss(self):
        raise NotImplementedError

    def cal_TX_energy(self):
        raise NotImplementedError

    def cal_D2A_energy(self):
        raise NotImplementedError

    def cal_RX_energy(self):
        raise NotImplementedError

    def cal_A2D_energy(self):
        raise NotImplementedError

    def cal_comp_energy(self):
        raise NotImplementedError

    def cal_laser_energy(self):
        raise NotImplementedError

    def cal_core_area(self):
        raise NotImplementedError

    def cal_core_power(self):
        raise NotImplementedError
    