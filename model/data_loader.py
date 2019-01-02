# -*- coding: utf-8 -*-

"""
specifies how the data should be fed to the network
"""

from dataset_class import AffectiveMonitorDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    
#    face_dataset = AffectiveMonitorDataset("C:\\Users\\DSPLab\\Research\\affective-monitor-model\\data")
    face_dataset = AffectiveMonitorDataset("E:\\Research\\affective-monitor-model\\data")
    data = face_dataset[0]
#    plot_face(face_dataset.face_frame.iloc[0])
    
