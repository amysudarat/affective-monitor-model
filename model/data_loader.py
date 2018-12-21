# -*- coding: utf-8 -*-

"""
specifies how the data should be fed to the network
"""
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_facial_graph():
    path = "E:\\Research\\affective-monitor-model\\data\\"
    filepaths = [os.path.join(path, "TestSubject"+str(i)+"\\FacialPoints.txt") for i in range(2,4)]
    input_samples = pd.DataFrame()
    for filepath in filepaths:
        face = pd.read_csv(filepath,header=2,parse_dates=True,delimiter=",")
        if input_samples.empty:
            input_samples = face
        else:
            input_samples = input_samples.append(face,ignore_index=True)
    
def load_facial_graph_test():
    path = "C:\\Users\\DSPLab\\Research\\affective-monitor-model\\data\\test\\FacialPoints.txt"    
    face = pd.read_csv(path,header=2,parse_dates=True,delimiter=",",quotechar=";")
    return face
    
        
    
    
if __name__ == "__main__":
#    testpath = "
#    load_facial_graph()
    face = load_facial_graph_test()
    
