# -*- coding: utf-8 -*-

import os
import ast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class AffectiveMonitorDataset(Dataset):
    """ Affective Monitor Dataset """
    
    def __init__(self,filepath,transform=None):
        """
        Args:
            filepath (string): Path to data input textfile
            transform (callable,optional): optional transform to be applied on a sample.
        """
        self.face_frame = self.load_dataframe(filepath)
        self.transform = transform

    def load_dataframe(self,path):
#        path = "C:\\Users\\DSPLab\\Research\\affective-monitor-model\\data\\test\\FacialPoints.txt"    
        face_df = pd.read_csv(path,header=2,delimiter=",",quotechar=";",index_col="PicIndex",skipinitialspace=True)
        # convert string to tuple
        for i in range(0,1347):
            face_df.iloc[:,i] = pd.Series([ast.literal_eval(x) for x in face_df.iloc[:,i]])         
        return face_df
   
    def __len__(self):
        return len(self.face_frame)
    
    def __getitem__(self,idx):
         return
        

