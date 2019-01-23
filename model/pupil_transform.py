# -*- coding: utf-8 -*-

import ast
import pandas as pd
from dataset_class import AffectiveMonitorDataset

#import matplotlib.pyplot as plt

def plot_pupil(data):
        """
        Args:
        data: list of pupil diameter (left,right)
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
#        face = list(face.iloc[0:1347])
        
#        ax.scatter(*zip(*face),c='r')   
        X = []
        Y=[]
        Z=[]
        for item in face:
            X.append(item[0])
            Y.append(item[1])
            Z.append(item[2])
        
        ax.scatter(X,Y,Z,c='r')
        # annotate each point
        if annotate:
            xyzn = zip(X,Y,Z)
            for j, xyz_ in enumerate(xyzn): 
                annotate3D(ax, s=str(j-1), xyz=xyz_, fontsize=10, xytext=(-3,3),
                textcoords='offset points', ha='right',va='bottom')   
        
        # label axis
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
       
        plt.show()
        
def pupil_to_tuple():    
    data = test_pupil()
    for i in range(1,71):
        test = data.loc[i,'PupilDiameter']
        try:
            data.loc[i,'PupilDiameter'] = pd.Series([ast.literal_eval(x) for x in test])
        except Exception as e:
            print(i)
            converted = []
            for j in range(len(test)):               
                try:
                    converted.append(ast.literal_eval(test.iloc[j])) 
                except:
                    a = (0,0)
                    converted.append(a)
            data.loc[i,'PupilDiameter'] = pd.Series(converted)


def test_pupil():
    # FAP is loaded by default
#    face_dataset = AffectiveMonitorDataset("C:\\Users\\DSPLab\\Research\\ExperimentData",subjects=[1])
    face_dataset = AffectiveMonitorDataset("E:\\Research\\ExperimentData",subjects=[1])
    samples = face_dataset[:]
    dataframe = face_dataset.face_df
    return samples, dataframe


if __name__ == "__main__":
    samples , dataframe = test_pupil()
   
                    
    
    
    
    
    
    
    
    
    
    
    
    
