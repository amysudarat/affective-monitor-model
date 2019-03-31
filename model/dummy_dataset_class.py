# -*- coding: utf-8 -*-

import numpy as np
import torch
import math
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    """ Dummy Data for train """
    def __init__(self):
        self.dataframe = self.generate_data()
        
    def generate_data(self,):
        
    
        
    def generate_sine(self,fs=100,f):
        x = np.arange(fs)
        sin = math.sin(2*math.pi*f*x/fs)
        return sin
        
        