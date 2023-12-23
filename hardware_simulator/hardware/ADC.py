# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu
# @Date:   1969-12-31 18:00:00
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-10 23:43:45
import logging


ADC_list = {
    1: {'area': 2850, 'prec': 8, 'power': 14.8, 'sample_rate': 10, 'type': 'sar'},
}

class ADC():
    def __init__(self, choice=1) -> None:
        self.ADC_choice = choice
        
        assert choice == 1

        # loaded ADC params
        # make it private
        self.__ADC_area = 0
        self.__ADC_prec = 0
        self.__ADC_power = 0
        self.__ADC_sample_rate = 0
        self.__ADC_type = None
        
        # obtain ADC param
        self._obatin_ADC_param()
        self.ADC_freq = self.__ADC_sample_rate # set to sample rate by default
        self.ADC_prec = self.__ADC_prec # set to sample rate by default
    
    def _obatin_ADC_param(self):
        if self.ADC_choice is not None:
            self.__chosen_ADC_list = ADC_list[self.ADC_choice]
            self.__ADC_area = self.__chosen_ADC_list['area']
            self.__ADC_prec = self.__chosen_ADC_list['prec']
            self.__ADC_power = self.__chosen_ADC_list['power']
            self.__ADC_sample_rate = self.__chosen_ADC_list['sample_rate']
            self.__ADC_type = self.__chosen_ADC_list['type']
        else:
            raise NotImplementedError

    def set_ADC_work_freq(self, work_freq):
        if work_freq > self.__ADC_sample_rate:
            raise ValueError(f"Got required ADC work frequency {work_freq} exceeds the ADC frequency limit")
        self.ADC_freq = work_freq
    
    def set_ADC_work_prec(self, work_prec):
        if work_prec > self.__ADC_prec:
            raise ValueError(f"Got required ADC work precision {work_prec} exceeds the ADC precision limit")
        self.ADC_prec = work_prec

    def cal_ADC_param(self, print_msg=False):
        # convert power to desired freq and bit width
        if self.__ADC_type == "sar":
            # P \propto N
            self.ADC_power = self.__ADC_power * self.ADC_freq / \
                self.__ADC_sample_rate * (self.ADC_prec / self.__ADC_prec)
        elif self.__ADC_type == "flash":
            # P \propto (2**N - 1)
            self.ADC_power = self.__ADC_power * self.ADC_freq / \
                self.__ADC_sample_rate * \
                ((2**self.ADC_prec - 1) / (2**self.__ADC_prec - 1))

        self.ADC_area = self.__ADC_area
        
        if print_msg:
            logging.info('The %s-bit ADC power @%sGHz is %.2f mW', self.ADC_prec, self.ADC_freq, self.ADC_power)
            logging.info('The %s-bit ADC area is %.4f um^2', self.ADC_prec, self.ADC_area)
        
        
if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    test = ADC(choice=1)
    test.set_ADC_work_freq(4)
    test.set_ADC_work_prec(6)
    test.cal_ADC_param(print_msg=True)