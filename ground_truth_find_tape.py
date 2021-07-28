import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import diff
import scipy
import NNlib

def get_key(file):
        coordinates = os.path.basename(file).split('s')[0].split(',')
        test_list = list(map(int, coordinates))
        return test_list

class CalibrateDot():
    def __init__(self):
        self.calibration_folder = '12-15Start_95mA_2tenthsec_2tenthstepsize_4mm_xy/'
        self.green_spectra = np.genfromtxt('green.csv', delimiter=',')
        self.red_spectra = np.genfromtxt('red.csv', delimiter=',')
        self.step_size = .2

    def plot_spectra(self, filename):
        if filename != 'red.csv' and filename != 'green.csv':
            data = np.genfromtxt(self.calibration_folder + filename, delimiter=',')
        else:
            data = np.genfromtxt(filename, delimiter=',')
        plt.figure()
        plt.plot(data[:, 0], data[:, 1])
        plt.show()

    def normalize_spectra(self, in_spectra): 
        wavelength_500 = np.argmin(np.abs(in_spectra[:, 0] - 500))
        normalized_data = in_spectra[:, 1] / in_spectra[wavelength_500, 1]
        normalized_data = normalized_data.reshape(normalized_data.shape[0], 1)
        return normalized_data

    def sort_process_files(self, folder):
        files = sorted(glob(folder + '*.csv'), key=get_key)
        output_array = np.asarray(self.green_spectra[:, 0]).reshape(self.green_spectra.shape[0], 1)
        for file in files:
            data = np.genfromtxt(str(file), delimiter=',')
            normalized_data = self.normalize_spectra(data)
            output_array = np.append(output_array, normalized_data, axis=1)
        return output_array.shape
        # find wavelength band of dot, and filter for it, and select spectra with
        # biggest average intensity in that range, may need to make dot color other
        # than blue since 405nm issue is there, also re run on solid background rather
        # than boundary of green and red, for more formulaicism

        # may also want to try rms comparison to dot color spectra, and lowest is best


    


if __name__ == "__main__":
    position = CalibrateDot()
    print(position.sort_process_files(position.calibration_folder))
    # CalibrateDot().plot_spectra('green.csv')

    