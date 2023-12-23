# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu
# @Date:   1969-12-31 18:00:00
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-10 23:43:37
import logging


#  area: um^2, prec: bit, power: mw, sample_rate: GSample/s
DAC_list = {
    1: {'area': 11000, 'prec': 8, 'power': 50, 'sample_rate': 14, 'FoM': None, 'type': 'cap'}
}

class DAC():
    def __init__(self, choice=1) -> None:
        self.DAC_choice = choice
        assert choice == 1
        # loaded DAC params
        # make it private
        self.__DAC_area = 0
        self.__DAC_prec = 0
        self.__DAC_power = 0
        self.__DAC_sample_rate = 0
        self.__DAC_type = None
        self.__DAC_FoM = 0
        
        # obtain DAC param
        self._obatin_DAC_param()
        self.DAC_freq = self.__DAC_sample_rate # set to sample rate by default
        self.DAC_prec = self.__DAC_prec # set to sample rate by default
    
    def _obatin_DAC_param(self):
        if self.DAC_choice is not None:
            self.__chosen_DAC_list = DAC_list[self.DAC_choice]
            self.__DAC_area = self.__chosen_DAC_list['area']
            self.__DAC_prec = self.__chosen_DAC_list['prec']
            self.__DAC_power = self.__chosen_DAC_list['power']
            self.__DAC_sample_rate = self.__chosen_DAC_list['sample_rate']
            self.__DAC_type = self.__chosen_DAC_list['type']
            self.__DAC_FoM = self.__chosen_DAC_list['FoM']
        else:
            raise NotImplementedError

    def set_DAC_work_freq(self, work_freq):
        if work_freq > self.__DAC_sample_rate:
            raise ValueError(f"Got required DAC work frequency {work_freq} exceeds the DAC frequency limit")
        self.DAC_freq = work_freq
    
    def set_DAC_work_prec(self, work_prec):
        if work_prec > self.__DAC_prec:
            raise ValueError(f"Got required DAC work precision {work_prec} exceeds the DAC precision limit")
        self.DAC_prec = work_prec

    def cal_DAC_param(self, print_msg=False):
        # convert power to desired freq and bit width
        if self.__DAC_FoM is not None:
            # following 2 * FoM * nb * Fs / Br (assuming Fs=Br)
            self.DAC_power = 2 * self.__DAC_FoM * \
                self.DAC_prec * self.DAC_freq * 1e-3
        else:
            # P \propto 2**N/(N+1) * f_clk
            self.DAC_power = self.__DAC_power * (2**self.DAC_prec / (self.DAC_prec)) / (
                2**self.__DAC_prec / (self.__DAC_prec)) * self.DAC_freq / self.__DAC_sample_rate

        self.DAC_area = self.__DAC_area
        
        if print_msg:
            logging.info('The %s-bit DAC power @%sGHz is %.2f mW', self.DAC_prec, self.DAC_freq, self.DAC_power)
            logging.info('The %s-bit DAC area is %.4f um^2', self.DAC_prec, self.DAC_area)
        
        
if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    test = DAC(choice=2)
    test.set_DAC_work_freq(4)
    test.set_DAC_work_prec(5)
    test.cal_DAC_param(print_msg=True)