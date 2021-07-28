import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import diff
import scipy

class NNlib():
    def __init__(self):
        self.directory = '5-26-21_TumorID_Data/'
        self.blue_spectra = np.genfromtxt(self.directory + 'blue.csv.csv', delimiter=',')
        self.red_spectra = np.genfromtxt(self.directory + 'red.csv.csv', delimiter=',')
        self.step_size = .01
        self.filepath = [self.directory + 'simstudy' + str(i) + '.csv.csv' for i in range(360)]
        

    def plot_spectra(self, filename):
        self.filename = filename
        data = np.genfromtxt(self.directory + self.filename, delimiter=',')
        plt.figure()
        plt.plot(data[:, 0], data[:, 1])
        plt.show()

    def normalize_spectra(self, in_spectra): # compare to wvlngth=500nm
        # in_spectra = np.genfromtxt(self.directory + str(in_spectra), delimiter=',')
        wavelength_500 = np.argmin(np.abs(in_spectra[:, 0] - 500)) # use index and divide intensity at index by spectra
        normalized_data = in_spectra[:, 1] / in_spectra[wavelength_500, 1]
        normalized_data = normalized_data.reshape(normalized_data.shape[0], 1)
        normalized_data = np.append(in_spectra[:, 0].reshape(in_spectra.shape[0], 1), normalized_data, axis=1)
        # plt.figure()
        # plt.plot(normalized_data[:, 0], normalized_data[:, 1]) # makes spectra even less normal since wv_500 is sub 1 intensity
        # plt.show()
        return normalized_data


    def read_files_diff(self): #optimize this to read files quicker, less ram/ memory from lists
        folderinfor1 = glob(self.directory + '*.csv')
        # folderinfor1.sort(key=lambda f: int(filter(str.isdigit, f)))
        # print(folderinfor1)
        s = '--12--'
        print(filter(str.isdigit, s))

    def read_files_averaged_and_filtered(self):
        output_array = []
        output_array = np.asarray(output_array)
        for file in self.filepath:
            data = np.genfromtxt(str(file), delimiter=',')
            data = self.normalize_spectra(data)
            filter_data = data[(data[:,0]>670) & (data[:,0]<700)][:, 1]
            average_data = np.average(filter_data) #instead of averaging try difference of spectra (like in TumorID), try normalizing first
            output_array = np.append(output_array, average_data)

        # d_output_array = np.diff(output_array)/step_size
        plt.figure()
        plt.scatter(np.arange(0, np.max(output_array.shape)/100, self.step_size), output_array, label='averaged and filtered data')
        # plt.scatter(np.arange(0, np.max(d_output_array.shape)/100, step_size), d_output_array, label='first derivative')
        plt.xlabel('Position (mm)')
        plt.ylabel('Intensity')
        plt.title('Average Intensity at a Given Position in the 670-700nm Band')
        plt.legend(loc='upper right')
        plt.show()
        
        return output_array

    def rms_plot_compare_to_red_blue_spectra(self): # rms error from red spectra plotted
        red_RMS_matrix = np.asarray([])
        blue_RMS_matrix = np.asarray([])
        normalized_blue = self.normalize_spectra(self.blue_spectra)
        normalized_red = self.normalize_spectra(self.red_spectra)
        for file in self.filepath:
            file = np.genfromtxt(str(file), delimiter=',')
            file = self.normalize_spectra(file)
            RMS_red = np.sqrt((np.sum(np.square(normalized_red[:, 1] - file[:, 1]))) / normalized_red.shape[0])
            RMS_blue = np.sqrt((np.sum(np.square(normalized_blue[:, 1] - file[:, 1]))) / normalized_blue.shape[0])
            red_RMS_matrix = np.append(red_RMS_matrix, RMS_red)
            blue_RMS_matrix = np.append(blue_RMS_matrix, RMS_blue)
        plt.figure()
        plt.scatter(np.arange(0, np.max(blue_RMS_matrix.shape)/100, self.step_size), blue_RMS_matrix, label='Blue RMS')
        plt.scatter(np.arange(0, np.max(red_RMS_matrix.shape)/100, self.step_size), red_RMS_matrix, label='Red RMS')
        plt.xlabel('Position (mm)')
        plt.ylabel('RMS Error')
        plt.title('RMS Error for Position Spectra When Compared with Baseline Blue and Red Spectra')
        plt.legend(loc='upper right')
        plt.show()

    def rms_plot_change_marker_color_multiclass_classification(self):
        RMS_matrix_blue = np.asarray([])
        RMS_matrix_red = np.copy(RMS_matrix_blue)
        normalized_blue = self.normalize_spectra(self.blue_spectra) # perhaps don't normalize since this looks bad and predicts badly
        normalized_red = self.normalize_spectra(self.red_spectra)
        for file in self.filepath:
            file = np.genfromtxt(str(file), delimiter=',')
            file = self.normalize_spectra(file)
            RMS_red = np.sqrt((np.sum(np.square(normalized_red[:, 1] - file[:, 1]))) / normalized_red.shape[0])
            RMS_blue = np.sqrt((np.sum(np.square(normalized_blue[:, 1] - file[:, 1]))) / normalized_blue.shape[0])
            # RMS_choice = min(RMS_red, RMS_blue)
            if RMS_red > RMS_blue:
                RMS_red = 0
            elif RMS_red < RMS_blue:
                RMS_blue = 0
            RMS_matrix_red = np.append(RMS_matrix_red, RMS_red)
            RMS_matrix_blue = np.append(RMS_matrix_blue, RMS_blue)
        boundary_bounds = [np.argmax(RMS_matrix_blue), np.argmax(RMS_matrix_red)]

        plt.figure()
        plt.scatter(np.arange(0, np.max(RMS_matrix_blue.shape)/100, self.step_size), RMS_matrix_blue)
        plt.scatter(np.arange(0, np.max(RMS_matrix_red.shape)/100, self.step_size), RMS_matrix_red)
        plt.xlabel('Position (mm)')
        plt.ylabel('RMS Error')
        plt.title('RMS Error for Position Spectra When Compared with Baseline Blue and Red Spectra and Lower Error Taken')
        plt.show()
        # rms error compare to all reference spectra and choose one with lowest error and change marker color to that reference color, and plot rms error for trust in prediction
        return np.arange(0, np.max(RMS_matrix_blue.shape)/100, self.step_size)[boundary_bounds]


    def plot_comparison_to_largest_wvlngth_intensity_diff(self):
        normalized_blue = self.normalize_spectra(self.blue_spectra)
        normalized_red = self.normalize_spectra(self.red_spectra)
        difference_in_spectra = normalized_blue[:, 1] - normalized_red[:, 1]
        min_intensity_index = np.argmin(difference_in_spectra)
        max_intensity_index = np.argmax(difference_in_spectra)
        wavelength_500_index = np.argmin(np.abs(self.blue_spectra[:, 0] - 500))

        max_spectral_ratio = np.asarray([])
        min_spectral_ratio = np.asarray([])
        
        for file in self.filepath:
            file = np.genfromtxt(str(file), delimiter=',')
            file = self.normalize_spectra(file)

            intensity_max = file[max_intensity_index, 1]
            intensity_min = file[min_intensity_index, 1]
            intensity_500 = file[wavelength_500_index, 1]

            intensity_max_ratio = intensity_500 / intensity_max
            intensity_min_ratio = intensity_500 / intensity_min

            max_spectral_ratio = np.append(max_spectral_ratio, intensity_max_ratio)
            min_spectral_ratio = np.append(min_spectral_ratio, intensity_min_ratio)    

        max_spectral_ratio = max_spectral_ratio - max_spectral_ratio[0]
        min_spectral_ratio = min_spectral_ratio - min_spectral_ratio[0]        
        
        plt.plot(np.arange(0, np.max(max_spectral_ratio.shape)/100, self.step_size), max_spectral_ratio, label='500nm/(Max B-R Intensity Difference Wavelength)')
        plt.plot(np.arange(0, np.max(min_spectral_ratio.shape)/100, self.step_size), min_spectral_ratio, label='500nm/(Min B-R Intensity Difference Wavelength)')

        plt.legend(loc='upper right')
        plt.xlabel('Position (mm)')
        plt.ylabel('Ratio of Classifier')
        plt.title('Classifiers Compared with Intensity at 500nm at Different Positions')
        plt.show()

        plt.figure()
        plt.plot(self.blue_spectra[:, 0], difference_in_spectra)
        plt.title('Spectral Signature Difference Between Red and Blue Spectra')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.show()
        # subtract red from blue spectra like in Sim, compare biggest differences, do plot of ratio of wvlngth to 500nm on x position spectra, biggest diff is red, smooth and predict with differentiation of curve

        dydx = np.diff(max_spectral_ratio)/ self.step_size #smooth it first
        plt.figure()
        plt.plot(np.arange(0, np.max(max_spectral_ratio.shape)/100, self.step_size), np.append(max_spectral_ratio[0], dydx))
        plt.show()


if __name__ == "__main__":
    # NNlib().plot_spectra('red.csv.csv')
    
    instance = NNlib()
    instance.read_files_averaged_and_filtered()
    # instance.plot_comparison_to_largest_wvlngth_intensity_diff()
    # instance.rms_plot_compare_to_red_blue_spectra()
    # print(instance.rms_plot_change_marker_color_multiclass_classification())
    
    # instance.plot_spectra('simstudy300.csv.csv')
    # print(instance.normalize_spectra('simstudy300.csv.csv'))