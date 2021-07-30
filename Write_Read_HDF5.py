import pandas as pd
import numpy as np
import glob
import h5py
import os
import matplotlib.pyplot as plt
import time
from NNlib import NNlib


def pandas_read_write():
    pass
    instance = NNlib()
    # instance.plot_spectra('blue.csv.csv')


    ## Pandas method of saving/reading hdf5 or h5 file
    data = pd.read_csv('12-15Start_95mA_2tenthsec_2tenthstepsize_4mm_xy/0,0spectralsig.csv', header=None)
    # data.laser_power_mA = .095
    # data.laser_nm = 405
    # data.int_time_s = .25
    # data.step_size_mm = .2
    # data.raster_path_length_mm = 4
    # data.date = '07-10-2021'


    data.to_hdf('data.h5', key='data')

    new_data = pd.read_hdf('data.h5')
    print(new_data.date)

## h5py method of saving/reading hdf5 or h5 files
## h5py better because of folders, attributes to save space, no pickling when using jagged data sizes, easier to work with since Dataframes get split into 4 parts
def write_hdf(csv_file, hdf_name, key_name):  # groups are like dictionaries, datasets like np arrays
    spectra = pd.read_csv(csv_file, header=None)
    # spectra = np.genfromtxt(csv_file, delimiter=',')
    f = h5py.File(hdf_name, 'w')
    dset = f.create_dataset(key_name, spectra.shape, data=spectra)
    f.close()

def append_hdf(csv_file, hdf_name):
    spectra = np.genfromtxt(csv_file, delimiter=',')
    f = h5py.File(hdf_name, 'a')
    dset2 = f.create_dataset('other_spectra/new_spectra', spectra.shape, data=spectra)
    dset2.attrs['laser_power_A'] = .095
    dset2.attrs['step_size_mm'] = .2
    dset2.attrs['laser_nm'] = 405
    dset2.attrs['int_time_s'] = .25
    dset2.attrs['raster_path_length_mm'] = 4
    dset2.attrs['date'] = '07-10-2021'

    # print(dset2.name)
    f.close()


def reading_hdf(filename):
    f = h5py.File(filename, 'r')
    def printname(name):
        print(name)
    print(f.visit(printname))

    dset = f[list(f.keys())[0]]
    dset2 = f['other_spectra/new_spectra']
    for key in dset2.attrs.keys():
        print(key, dset2.attrs[key])
    print(dset[:, 0].shape)
    new_data = np.copy(dset2)
    
    f.close()
    plt.figure()
    plt.plot(new_data[:, 0], new_data[:, 1])
    plt.show()
    return

def guangshen_method():
    data_pd = pd.DataFrame()
    path_use = "5-26-21_TumorID_Data/"
    folderinfor_spec = glob.glob(path_use + "*.csv")
    Nfiles = len(folderinfor_spec)

    for i in range(Nfiles):
        print("The current index is ", i)
        idx_wavelength = 2 * i
        idx_intensity = 2 * i + 1
        data_current_spec = np.asarray(np.genfromtxt(path_use + 'simstudy' + str(i) + '.csv.csv', delimiter=','))
        data_wavelegnth = np.asarray([data_current_spec[:,0]])
        data_intenstiy = np.asarray([data_current_spec[:,1]])
        data_pd[str(idx_wavelength)] = pd.Series( data_wavelegnth[0] )
        data_pd[str(idx_intensity)] = pd.Series( data_intenstiy[0] )
        if i == 10:
            break

    # Define the Laser Parameters
    data_pd['Laser Parameter 1'] = pd.Series([1])
    data_pd['Laser Parameter 2'] = pd.Series([2])
    data_pd['Laser Parameter 3'] = pd.Series([3])

    # Write the dataFrame
    # data_pd.to_excel("../BTL_ML/Data/DataFrame.xlsx")
    data_pd.to_hdf("DataFrame.h5", key = 'df')
    print('done')


class H5_Class(NNlib):
    def __init__(self):
        super(H5_Class, self).__init__()
        # self.directory = 'B2R_150mA_hundredth_step_3mm/'
        # self.step_size = .01
        # self.filepath = [self.directory + str(i) + 'spectralsig' + '.csv' for i in range(300)]

    def store_many_rasters(self, hdf_name):
        if os.path.exists(hdf_name):
            mode = 'a'
        else:
            mode = 'w'
        f = h5py.File(hdf_name, mode)
        first_group = f.create_group(self.directory)
        first_group.attrs['laser_power_A'] = .150
        first_group.attrs['step_size_mm'] = .01
        first_group.attrs['laser_nm'] = 405
        first_group.attrs['int_time_s'] = .25
        first_group.attrs['raster_path_length_mm'] = 3.0
        first_group.attrs['date'] = '07-10-2021'

        for file in self.filepath:
            csv_data = np.genfromtxt(str(file), delimiter=',')
            normalized_spectra = self.normalize_spectra(csv_data)
            dset = f.create_dataset(file, csv_data.shape, data=normalized_spectra)
        f.close()

    def read_raster_contents(self, hdf_name):
        t1 = time.time()
        f = h5py.File(hdf_name, 'r')
        t2 = time.time()
        print(t2-t1)
        def printname(name):
            print(name)
        print(f.visit(printname))
        for key in f.keys():
            for attr_key in f[key].attrs.keys():
                print(attr_key, f[key].attrs[attr_key])
            for dataset in f[key]:
                print(dataset, f[key][dataset])
                plt.figure()
                plt.plot(f[key][dataset][:, 0], f[key][dataset][:, 1])
                plt.show()
                break
        f.close()
    
    def delete_group(self, hdf_name, group_or_data_key):
        f = h5py.File(hdf_name, 'a')
        del f[group_or_data_key]
        f.close()


if __name__ == "__main__":
    # guangshen_method()
    # write_hdf('12-15Start_95mA_2tenthsec_2tenthstepsize_4mm_xy/0,0spectralsig.csv', 'foo.hdf5', 'dataset')
    # append_hdf('12-15Start_95mA_2tenthsec_2tenthstepsize_4mm_xy/10,10spectralsig.csv', 'foo.hdf5')
    # reading_hdf('foo.hdf5')
    H5_Class().store_many_rasters('Tape_Raster_Exp.hdf5')
    # H5_Class().delete_group('Tape_Raster_Exp.hdf5', '5-26-21_TumorID_Data')
    H5_Class().read_raster_contents('Tape_Raster_Exp.hdf5')