import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import scipy

class NNlib():
    def __init__(self):
        self.filepath = '5-26-21_TumorID_Data'

    def read_files(self):
        folderinfor1 = glob(self.filepath + "*.csv")
