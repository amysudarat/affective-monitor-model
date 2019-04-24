# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    """ Dummy Data for train """
    def __init__(self,transform=None):
        random.seed(42)
        self.transform = transform
        self.dataframe = self.generate_data()
        return
        
    def generate_data(self):
        whole_dataframe = self.generate_sine()
        whole_dataframe = whole_dataframe.append(self.generate_flat())
        whole_dataframe = whole_dataframe.append(self.generate_linear())
#        whole_dataframe = whole_dataframe.append(self.generate_parabola())
        whole_dataframe = whole_dataframe.sample(frac=1).reset_index(drop=True)
        return whole_dataframe
        
        
    def generate_sine(self,length=100,n=850):
        frequency_random = [random.randrange(1,101) for _ in range(n)]
        x = np.arange(length)
        sins= []
        for freq in frequency_random:
            sins.append({'data': np.sin(2*np.pi*freq*(x/length)),'label':1})           
        return pd.DataFrame(sins)
    
    def generate_flat(self,length=100,n=850):
        magnitude_random = [random.randrange(-50,50) for _ in range(n)]
        flats = []
        for mag in magnitude_random:
            flats.append({'data':mag*np.ones(length),'label':2})
        return pd.DataFrame(flats)
    
    def generate_linear(self,length=100,n=850):
        slope_random = [random.randrange(-50,51) for _ in range(n)]
        bias_random = [random.randrange(-50,51) for _ in range(n)]
        x = np.arange(length)
        linears = []
        for slope_random,bias_random in zip(slope_random,bias_random):
            linears.append({'data':slope_random*x+bias_random,'label':3})
        return pd.DataFrame(linears)  
    
    def generate_parabola(self,length=100,n=850):
        """y =a*(x - h)2 + k."""
        a_random = [random.randrange(-50,51) for _ in range(n)]
        h_random = [random.randrange(-50,51) for _ in range(n)]
        k_random = [random.randrange(-50,51) for _ in range(n)]
        x = np.arange(length)
        paras = []
        for a, h, k in zip(a_random,h_random,k_random):
            paras.append({'data':a*(x-h)**2 + k,'label':4})
        return pd.DataFrame(paras)
    
    def __getitem__(self,idx):
        sample = self.dataframe.iloc[idx]
#        if len(sample) != 1:
#            print("can only get one sample at a time if ToTensor is used")
#            return
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return len(self.dataframe)
    
class ToTensor(object):
    """ convert to tensor object """
    def __call__(self,sample):
        transformed_sample = {'data': torch.tensor(sample['data'].astype('float')),
                              'label':torch.tensor(sample['label'])}
        return transformed_sample

if __name__ == "__main__":
    dummy_data = DummyDataset(transform=ToTensor())
    sample = dummy_data[8]

