# -*- coding: utf-8 -*-

import ast
import pandas as pd
from dataset_class import AffectiveMonitorDataset
import matplotlib.pyplot as plt

#import matplotlib.pyplot as plt

def plot_pupil(data):
    """
    Args:
    data: list of pupil diameter (left,right)
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212) 
    # Unpack tuple to two lists
    L = []
    R=[]       
    for item in data:
        L.append(item[0])
        R.append(item[1])
    # create time series
    t = [i for i in range(len(L))]
    
    # plot PD
    ax1.plot(t,L,'.-')
    ax1.set(xlabel='samples', ylabel='PD (Left)', title='Pupil Diameter')
    ax1.grid()
    ax2.plot(t,R,'.-')
    ax2.set(xlabel='samples', ylabel='PD (Right)')
    ax2.grid()
                      
#    plt.suptitle('Pupil Diameter')
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
    face_dataset = AffectiveMonitorDataset("C:\\Users\\DSPLab\\Research\\ExperimentData",subjects=[1])
#    face_dataset = AffectiveMonitorDataset("E:\\Research\\ExperimentData",subjects=[1])
    samples = face_dataset[:]
    dataframe = face_dataset.face_df
    return samples, dataframe


if __name__ == "__main__":
    samples , dataframe = test_pupil()
    plot_pupil(dataframe['PupilDiameter'])
   
                    
    
    
    
    
    
    
    
    
    
    
    
    
