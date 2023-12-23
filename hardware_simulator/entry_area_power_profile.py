# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu(hqzhu@utexas.edu)
# @Date:   1969-12-31 18:00:00
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-13 20:19:40
import os
import csv
import logging
import argparse

from collections import OrderedDict
from utils.config import configs
from utils.general import ensure_dir

from hardware.photonic_crossbar import PhotonicCrossbar
from hardware.photonic_mrr_bank import PhotonicMRRBank
from hardware.photonic_MZI import PhotonicMZI

logging.basicConfig(level=logging.INFO)


class areaPrediction():
    def __init__(self, configs=None) -> None:
        super().__init__()

        self.core_type = configs.core.type
        self.num_tiles = configs.arch.num_tiles
        self.num_pe_per_tile = configs.arch.num_pe_per_tile

        # bits
        self.in_bit = configs.core.precision.in_bit
        self.w_bit = configs.core.precision.w_bit
        self.act_bit = configs.core.precision.act_bit

        if self.core_type == "dota":
            self.input_mod_sharing_flag = True if configs.arch.input_mod_sharing_flag == 1 else False
            self.adc_share_flag = True if configs.arch.adc_share_flag == 1 else False
        else:
            self.adc_share_flag = False
            self.input_mod_sharing_flag = False

        # build tensor core
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

        self.power_dict = OrderedDict
        self.area_dict = OrderedDict

    def predict_power_crossbar(self):
        power_dict = {i: [0, 0] for i in ["total", "laser", "DAC",
                                          "MZM", "ADC", "TIA", "Photodetector", "adder", "mem"]}

        self.hw.cal_core_power()
        ## laser power
        power_laser = self.num_pe_per_tile * self.num_tiles * self.hw.laser_power
        
        # MZM
        power_MZM = self.hw.core_height * self.hw.num_wavelength * self.num_tiles * self.num_pe_per_tile * \
            (self.hw.modulator_power_static + self.hw.modulator_power_dynamic + self.hw.mrr_router_power * 2)

        # DAC
        power_DAC = self.hw.core_DAC_power * self.hw.core_height * \
            self.hw.num_wavelength * self.num_tiles * self.num_pe_per_tile

        if self.input_mod_sharing_flag:
            power_MZM += (self.hw.modulator_power_static + self.hw.modulator_power_dynamic + self.hw.mrr_router_power * 2) * \
                self.hw.core_width * self.hw.num_wavelength * self.num_pe_per_tile
            power_DAC += self.hw.core_DAC_power * self.hw.core_width * \
                self.hw.num_wavelength * self.num_pe_per_tile
        else:
            power_MZM += (self.hw.modulator_power_static + self.hw.modulator_power_dynamic + self.hw.mrr_router_power * 2) * \
                self.hw.core_width * self.hw.num_wavelength * self.num_pe_per_tile * self.num_tiles
            power_DAC += self.hw.core_DAC_power * self.hw.core_width * \
                self.hw.num_wavelength * self.num_pe_per_tile * self.num_tiles

        # ADC
        power_ADC = self.hw.core_ADC_power * self.hw.core_height * \
            self.hw.core_width * self.num_tiles
        if not self.adc_share_flag:
            power_ADC *= self.num_pe_per_tile

        # detection (TIA + photodetctor)
        power_photodetector = self.num_pe_per_tile * self.num_tiles * \
            self.hw.core_height * self.hw.core_width * self.hw.photo_detector_power * 2
        if not self.adc_share_flag:
            power_TIA = self.num_pe_per_tile * self.num_tiles * \
                self.hw.core_height * self.hw.core_width * self.hw.TIA_power
        else:
            power_TIA = self.num_tiles * self.hw.core_height * \
                self.hw.core_width * self.hw.TIA_power

        # adder
        power_adder = 0.2 / 4.39  # follow tech node scaling law
        power_adder *= self.num_tiles * self.hw.core_height * self.hw.core_width

        # mem powe from CACTI
        GB_power = 315.2512 * self.num_tiles / 4 # scale when move to 8 tiles
        LB_power = 0.172725 # one 4KB LB
        buffer_power = 0.0154 # activaion buffer

        # GB: 1
        # LB: self.num_tiles * (LB_buffer@input, buffer@output)+ one LB_buffer (input mod sharing)
        # buffer power:  each PE's input buffer, output buffer is shared across PEs in the same tile as we do summation first
        power_memory = GB_power + self.num_tiles * \
            (LB_power + buffer_power * 2) + buffer_power * \
            self.num_tiles * self.num_pe_per_tile + LB_power
            
        if not self.input_mod_sharing_flag:
            power_memory += buffer_power * self.num_tiles * self.num_pe_per_tile
        else:
            power_memory += buffer_power * self.num_pe_per_tile

        power = power_laser + power_ADC + power_DAC + power_MZM + \
            power_photodetector + power_TIA + power_memory + power_adder

        power_dict['total'] = [power, 1]
        power_dict['laser'] = [power_laser,
                               round(power_laser / power * 100, 2)]
        power_dict['DAC'] = [power_DAC, round(power_DAC / power * 100, 2)]
        power_dict['MZM'] = [power_MZM, round(power_MZM / power * 100, 2)]
        power_dict['ADC'] = [power_ADC, round(power_ADC / power * 100, 2)]
        power_dict['TIA'] = [power_TIA, round(power_TIA / power * 100, 2)]
        power_dict['Photodetector'] = [power_photodetector,
                                       round(power_photodetector / power * 100, 2)]
        power_dict['adder'] = [power_adder,
                               round(power_adder / power * 100, 2)]
        power_dict['mem'] = [power_memory, round(
            power_memory / power * 100, 2)]

        self.power_dict = power_dict

    def predict_area_crossbar(self):
        area_dict = {i: [0, 0] for i in ["total", "laser", "DAC",
                                         "MZM", "ADC", "TIA", "photonic_core", "adder", "mem"]}

        self.hw.cal_core_area()

        # photonic part
        # we consider spacing and real layout
        area_photonic_core = self.hw.photonic_core_node_area * \
            self.num_tiles * self.num_pe_per_tile
        area_modulator = self.hw.mrr_area * self.num_tiles * self.num_pe_per_tile
        area_y_branch = self.hw.y_branch_area * self.num_tiles * self.num_pe_per_tile
        # ADC part
        area_ADC = self.hw.core_ADC_area * self.hw.core_height * \
            self.hw.core_width * self.num_tiles
        if not self.adc_share_flag:
            area_ADC *= self.num_pe_per_tile
        # DAC part
        area_DAC = self.hw.core_DAC_area * self.hw.core_height * \
            self.hw.num_wavelength * self.num_tiles * self.num_pe_per_tile
        area_MZM = self.hw.modulator_length * self.hw.modulator_width * \
            self.hw.core_height * self.hw.num_wavelength * \
            self.num_tiles * self.num_pe_per_tile
        if self.input_mod_sharing_flag:
            area_DAC += self.hw.core_DAC_area * self.hw.core_width * \
                self.hw.num_wavelength * self.num_pe_per_tile
            area_MZM += self.hw.modulator_length * self.hw.modulator_width * \
                self.hw.core_width * self.hw.num_wavelength * self.num_pe_per_tile
        else:
            area_DAC += self.hw.core_DAC_area * self.hw.core_width * \
                self.hw.num_wavelength * self.num_tiles * self.num_pe_per_tile
            area_MZM += self.hw.modulator_length * self.hw.modulator_width * \
                self.hw.core_width * self.hw.num_wavelength * \
                self.num_pe_per_tile * self.num_tiles
        
        # TIA part
        if not self.adc_share_flag:
            area_tia = self.hw.core_height * self.hw.core_width * \
            self.hw.TIA_area * self.num_tiles * self.num_pe_per_tile
        else:
            area_tia = self.hw.core_height * self.hw.core_width * \
            self.hw.TIA_area * self.num_tiles

        # laser part: modulation for M1 and M2
        area_laser = self.hw.laser_area * \
            (self.num_tiles + self.num_pe_per_tile)
        area_micro_comb = self.hw.micro_comb_area * \
            (self.num_tiles + self.num_pe_per_tile)

        # memory part
        GB_memory = 14.348352  # 2MB
        LB_memory = 0.0683105  # 4kB for each tile
        buffer_memory = 0.000305237  # 256 B for each pe

        area_memory = 1e6 * (GB_memory * self.num_tiles / 4 + LB_memory * self.num_tiles + LB_memory + buffer_memory * 2 *
            self.num_tiles + buffer_memory * (self.num_tiles * self.num_pe_per_tile + self.num_pe_per_tile))

        area_adder = 0.00024 * 1e6 / 2.7  # follow scaling law
        area_adder *= self.num_tiles * self.hw.core_height * self.hw.core_width

        area = area_photonic_core + area_modulator + area_y_branch + area_ADC + \
            area_DAC + area_MZM + area_tia + area_laser + \
                area_memory + area_adder + area_micro_comb

        area_dict['total'] = [area*1e-6, 1]
        area_dict['laser'] = [area_laser * 1e-6,
                              round(area_laser / area * 100, 2)]
        area_dict['micro_comb'] = [area_micro_comb * 1e-6,
                              round(area_micro_comb / area * 100, 2)]
        area_dict['DAC'] = [area_DAC * 1e-6, round(area_DAC / area * 100, 2)]
        # add mzm and mrr router
        area_dict['MZM'] = [(area_MZM+area_modulator) * 1e-6,
                            round((area_MZM+area_modulator) / area * 100, 2)]
        area_dict['ADC'] = [area_ADC * 1e-6, round(area_ADC / area * 100, 2)]
        area_dict['TIA'] = [area_tia * 1e-6, round(area_tia / area * 100, 2)]
        # add y branch to photonic_core
        area_dict['photonic_core'] = [
            (area_photonic_core+area_y_branch) * 1e-6, round((area_photonic_core + area_y_branch) / area * 100, 2)]
        area_dict['adder'] = [area_adder * 1e-6,
                              round(area_adder / area * 100, 2)]
        area_dict['mem'] = [area_memory * 1e-6,
                            round(area_memory / area * 100, 2)]

        self.area_dict = area_dict

    def predict_area_mrrbank(self):
        area_dict = {i: [0, 0] for i in ["total", "laser", "DAC", "ADC", "TIA", "photonic_core", "adder", "mem"]}

        self.hw.cal_core_area()

        # photonic part
        # we consider spacing and real layout
        area_photonic_core = self.hw.mrr_bank_area * self.num_tiles * self.num_pe_per_tile

        area_y_branch = self.hw.y_branch_area * self.num_tiles * self.num_pe_per_tile
        
        # ADC part
        area_ADC = self.hw.core_ADC_area * self.hw.core_height * self.num_tiles * self.num_pe_per_tile
        
        # DAC part
        area_DAC = self.hw.core_DAC_area * self.hw.core_height * self.hw.core_width * self.num_tiles * self.num_pe_per_tile
        
        area_DAC += self.hw.core_DAC_area * self.hw.core_width * self.num_tiles * self.num_pe_per_tile

        # TIA part
        area_tia = self.hw.core_width * self.hw.TIA_area * self.num_tiles * self.num_pe_per_tile

        # laser part
        area_laser = self.hw.laser_area * self.num_tiles
        area_micro_comb = self.hw.micro_comb_area * self.num_tiles

        # memory part
        GB_memory = 14.348352  # 2MB
        LB_memory = 0.0683105  # 4kB for each tile
        buffer_memory = 0.000305237  # 256 B for each pe
        
        #  it is weight stationary
        # so for LB part: input still a 4KB one, weight is a small one, activation need a extra 4KB one per tile
        # buffer for PE: inputs: buffer_memory * (self.num_tiles * self.num_pe_per_tile*2), output buffer still one per tile
        area_memory = 1e6 * (GB_memory * self.num_tiles / 4 + LB_memory *2 * self.num_tiles + buffer_memory * 2 *
            self.num_tiles + buffer_memory * (self.num_tiles * self.num_pe_per_tile * 2))

        area_adder = 0.00024 * 1e6 / 2.7  # follow scaling law
        area_adder *= self.num_tiles * self.hw.core_height * self.hw.core_width

        area = area_photonic_core + area_y_branch + area_ADC + \
            area_DAC + area_tia + area_laser + \
                area_memory + area_adder + area_micro_comb

        area_dict['total'] = [area*1e-6, 1]
        area_dict['laser'] = [area_laser * 1e-6,
                              round(area_laser / area * 100, 2)]
        area_dict['micro_comb'] = [area_micro_comb * 1e-6,
                              round(area_micro_comb / area * 100, 2)]
        area_dict['DAC'] = [area_DAC * 1e-6, round(area_DAC / area * 100, 2)]
        area_dict['ADC'] = [area_ADC * 1e-6, round(area_ADC / area * 100, 2)]
        area_dict['TIA'] = [area_tia * 1e-6, round(area_tia / area * 100, 2)]
        # add y branch to photonic_core
        area_dict['photonic_core'] = [
            (area_photonic_core+area_y_branch) * 1e-6, round((area_photonic_core + area_y_branch) / area * 100, 2)]
        area_dict['adder'] = [area_adder * 1e-6,
                              round(area_adder / area * 100, 2)]
        area_dict['mem'] = [area_memory * 1e-6,
                            round(area_memory / area * 100, 2)]

        self.area_dict = area_dict
        
    def predict_area_mzi(self):
        area_dict = {i: [0, 0] for i in ["total", "laser", "DAC", "ADC", "TIA", "photonic_core", "adder", "mem"]}

        self.hw.cal_core_area()

        # photonic part
        # we consider spacing and real layout
        area_photonic_core = self.hw.photonic_core_area * self.num_tiles * self.num_pe_per_tile

        area_y_branch = self.hw.y_branch_area * self.num_tiles * self.num_pe_per_tile
        
        area_modulator = self.hw.mzm_area * self.num_tiles * self.num_pe_per_tile
        
        # ADC part
        area_ADC = self.hw.photonic_core_adc_area * self.num_tiles * self.num_pe_per_tile
        # DAC part
        area_DAC = self.hw.photonic_core_dac_area * self.num_tiles * self.num_pe_per_tile

        # TIA part  
        area_tia = self.hw.core_width * self.hw.TIA_area * self.num_tiles * self.num_pe_per_tile

        # laser part
        area_laser = self.hw.laser_area * self.num_tiles

        # memory part
        GB_memory = 14.348352  # 2MB
        LB_memory = 0.0683105  # 4kB for each tile
        buffer_memory = 0.000305237  # 256 B for each pe

        # it is weight stationary
        # so for LB part: input still a 4KB one, weight is a small one, activation need a extra 4KB one per tile
        # buffer for PE: inputs: buffer_memory * (self.num_tiles * self.num_pe_per_tile*2), output buffer still one per tile
        area_memory = 1e6 * (GB_memory * self.num_tiles / 4 + LB_memory *2 * self.num_tiles + buffer_memory *
            self.num_tiles + buffer_memory * (self.num_tiles * self.num_pe_per_tile*2) + buffer_memory * self.num_tiles)


        area_adder = 0.00024 * 1e6 / 2.7  # follow scaling law
        area_adder *= self.num_tiles * self.hw.core_height * self.hw.core_width

        area = area_photonic_core + area_y_branch + area_ADC + \
            area_DAC + area_tia + area_laser + \
                area_memory + area_adder
        area_dict['total'] = [area*1e-6, 1]
        area_dict['laser'] = [area_laser * 1e-6,
                              round(area_laser / area * 100, 2)]
        area_dict['DAC'] = [area_DAC * 1e-6, round(area_DAC / area * 100, 2)]
        area_dict['ADC'] = [area_ADC * 1e-6, round(area_ADC / area * 100, 2)]
        area_dict['TIA'] = [area_tia * 1e-6, round(area_tia / area * 100, 2)]
        # add y branch to photonic_core
        area_dict['photonic_core'] = [
            (area_photonic_core+area_y_branch+area_modulator) * 1e-6, round((area_photonic_core + area_y_branch+area_modulator) / area * 100, 2)]
        area_dict['adder'] = [area_adder * 1e-6,
                              round(area_adder / area * 100, 2)]
        area_dict['mem'] = [area_memory * 1e-6,
                            round(area_memory / area * 100, 2)]

        self.area_dict = area_dict

    def __save_area_csv(self, sv_name, dic2d, topic):
        with open(sv_name, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([topic, 'area (mm^2)', 'percentage (%)'])
            for each in dic2d:
                data = [each]
                data.extend(dic2d[each])
                writer.writerow(data)

    def __save_power_csv(self, sv_name, dic2d, topic):
        with open(sv_name, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([topic, 'power (mW)', 'percentage (%)'])
            for each in dic2d:
                data = [each]
                data.extend(dic2d[each])
                writer.writerow(data)

    def save(self, sv_name, sv_path):
        self.__save_area_csv(os.path.join(
            sv_path, sv_name+'_area.csv'), self.area_dict, self.core_type)
        if self.core_type == 'dota':
            self.__save_power_csv(os.path.join(
                sv_path, sv_name+'_power.csv'), self.power_dict, 'dota')

    def run(self):
        if self.core_type == "dota":
            self.predict_area_crossbar()
            self.predict_power_crossbar()
        elif self.core_type == "mzi":
            self.predict_area_mzi()
        elif self.core_type == "mrrbank":
            self.predict_area_mrrbank()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default=".params.yaml",
                        metavar="FILE", help="config file")
    parser.add_argument("-e", "--exp", default='area_power',
                        help="exp name")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    sv_path = f"./results/{args.exp}/{configs.core.type}_{configs.arch.num_tiles}t_{configs.arch.num_pe_per_tile}c_{configs.core.precision.in_bit}bit/"
    ensure_dir(sv_path)
    area_predictor = areaPrediction(configs=configs)

    print(f"Report energy and latency estimation for {configs.core.type}_{configs.arch.num_tiles}t_{configs.arch.num_pe_per_tile}c_{configs.core.precision.in_bit}bit")
    area_predictor.run()
    area_predictor.save(sv_name=configs.core.type, sv_path=sv_path)
    print(f'Finish and save report to {sv_path}')
    print('-'*20)
