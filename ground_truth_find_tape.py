import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import diff
import scipy

def get_key(file):
        # file = folder + '3,4spectralsig.csv'
        coordinates = os.path.basename(file).split('s')[0].split(',')
        test_list = list(map(int, coordinates))
        return test_list

class CalibrateDot():
    def __init__(self):
        self.calibration_folder = '12-15Start_95mA_2tenthsec_2tenthstepsize_4mm_xy/'
        self.green_spectra = np.genfromtxt('green.csv', delimiter=',')
        self.red_spectra = np.genfromtxt('red.csv', delimiter=',')
        self.step_size = .2
        # self.filepath = 

    def sort_files(self, folder):
        files = sorted(glob(folder + '*.csv'), key=get_key)
        for file in files:
            print(file)


if __name__ == "__main__":
    position = CalibrateDot()
    position.sort_files(position.calibration_folder)
    # get_key('12-15Start_95mA_2tenthsec_2tenthstepsize_4mm_xy/3,4spectralsig.csv')