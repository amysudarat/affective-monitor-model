# -*- coding: utf-8 -*-

"""
utility functions for handling hyperparams/logging/storing model
"""

import matplotlib.pyplot as plt

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