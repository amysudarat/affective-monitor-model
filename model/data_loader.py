# -*- coding: utf-8 -*-

"""
specifies how the data should be fed to the network
"""

import pandas as pd
import numpy as np
import torch
from dataset_class import AffectiveMonitorDataset
import torch.utils.data.dataset as Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#
#def load_facial_graph():
#    path = "E:\\Research\\affective-monitor-model\\data\\"
#    filepaths = [os.path.join(path, "TestSubject"+str(i)+"\\FacialPoints.txt") for i in range(2,4)]
#    input_samples = pd.DataFrame()
#    for filepath in filepaths:
#        face = pd.read_csv(filepath,header=2,parse_dates=True,delimiter=",")
#        if input_samples.empty:
#            input_samples = face
#        else:
#            input_samples = input_samples.append(face,ignore_index=True)
#    
#def load_facial_graph_test():
#    path = "C:\\Users\\DSPLab\\Research\\affective-monitor-model\\data\\TestSubject2\\Data.txt"    
#    face_df = pd.read_csv(path,header=2,delimiter=",",quotechar=";",index_col="PicIndex",skipinitialspace=True)
#    # convert string to tuple
##    face_df.shape[1]-1
#    for i in range(0,1347):
#        face_df.iloc[:,i] = pd.Series([ast.literal_eval(x) for x in face_df.iloc[:,i]]) 
#    
#    return face_df
#
#def plot_face(face):
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    face = list(face.iloc[0:1347])
#    
#    ax.scatter(*zip(*face),c='r')   
#    ax.set_xlabel('X')
#    ax.set_ylabel('Y')
#    ax.set_zlabel('Z')
#   
#    plt.show()

def plot_face(face):
        """
        Args: one row of the sample
        ex: plot_face(face.iloc[0])
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        face = list(face.iloc[0:1347])
        
        ax.scatter(*zip(*face),c='r')   
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
       
        plt.show()

#def update_plot(i,data,scat)

if __name__ == "__main__":
#    testpath = "
#    load_facial_graph()
    
    face_dataset = AffectiveMonitorDataset("C:\\Users\\DSPLab\\Research\\affective-monitor-model\\data\\TestSubject2\\Data.txt")
    data = face_dataset.face_frame
    plot_face(face_dataset.face_frame.iloc[0])
    
