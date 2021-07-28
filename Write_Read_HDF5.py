import pandas as pd
import numpy as np
import glob
import h5py
import os
# import tables
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
def write_hdf(csv_file):
    pass
def read_hdf(filename):
    pass
    f = h5py.File(filename, 'r')
    print(list(f.keys()))
    dset = f['data']
    for key in dset.keys():
        print(dset[key])

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



if __name__ == "__main__":
    # guangshen_method()
    data_load = pd.read_hdf('DataFrame.h5', key='df')
    print(data_load['1'], data_load['Laser Parameter 1'])