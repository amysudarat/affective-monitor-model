#!/usr/bin/env python

import ast
import pandas as pd
import numpy as np
#import padasip as pa
from dataset_class import AffectiveMonitorDataset
import matplotlib.pyplot as plt

np.random.seed(52102)

#import matplotlib.pyplot as plt

def plot_pupil(PD,IL):
    """
    Args:
    data: list of pupil diameter (left,right)
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312) 
    ax3 = fig.add_subplot(313) 
    
    # Unpack tuple to two lists
    L = []
    R=[]       
    for item in PD:
        L.append(item[0])
        R.append(item[1])
    # create time series
    t = [i for i in range(len(L))]
    
    # plot PD
    ax1.plot(t,L,'-k')
    ax1.set(ylabel='PD (Left)', title='Pupil Diameter (Left)')
    ax1.grid()
    ax2.plot(t,R,'-k')
    ax2.set(ylabel='PD (Right)', title='Pupil Diameter (Right)')
    ax2.grid()
    ax3.plot(t,IL,'-')
    ax3.set(ylabel='Illum', title='Illuminance')
    ax3.grid()
                      
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
            
def tuple_to_list(pd_tuple):
    # Unpack tuple to two lists
    L = []
    R=[]       
    for item in pd_tuple:
        L.append(item[0])
        R.append(item[1])
    return L,R

def my_lms(d,r,L,mu):
    e = np.zeros(d.shape)
    y = np.zeros(r.shape)
    w = np.zeros(L)
    
    for k in range(L,len(r)):
        x = r[k-L:k]
        y[k] = np.dot(x,w)
        e[k] = d[k]-y[k]
        w_next = w + (2*mu*e[k])*x
        w = w_next   
    return y, e, w
    
def remove_PLR(pd,illum,n,mu):
    d = np.array(pd)
    d_norm = d / np.linalg.norm(d)
    illum_norm = illum / np.linalg.norm(illum)
    illum_norm = 1.2*illum_norm
    illum_norm = illum_norm - np.mean(illum_norm) + np.mean(d_norm)
    y, e, w = my_lms(d_norm,illum_norm,n,mu)
    
    # results
    plt.figure()    
    
    plt.plot(illum_norm, "c", label="reference signal")
    plt.plot(d_norm, "k", label="recorded signal")
    plt.plot(y, "--r", label="modified reference signal")
    plt.plot(e, "k", label="output signal")
    plt.grid()
    plt.title("mu = "+str(mu)+", L = "+str(n))
    plt.legend()
    plt.show()
    
    # plot separate graph
    fig, axes = plt.subplots(4,1)
    start = 2600
    end = 3300
    axes[0].plot(d_norm[start:end],'k')
    axes[0].set(title='Primary Input Signal')
    
    axes[1].plot(illum_norm[start:end],'k')
    axes[1].set(title='Reference Input Signal')
    axes[2].plot(y[start:end],'k')
    axes[2].set(title='Modified Reference Signal')
    axes[3].plot(e[start:end],'k')
    axes[3].set(title='Output Filtered Signal')
    
  
    
    
#    ax1 = fig.add_subplot(411)
#    ax2 = fig.add_subplot(412) 
#    ax3 = fig.add_subplot(413)
#    ax4 = fig.add_subplot(414)
#    ax1.plot(d_norm[start:end],'k')
#    ax1.set(title='Primary Input Signal')
#    ax2.plot(illum_norm[start:end],'k')
#    ax2.set(title='Reference Input Signal')
#    ax3.plot(y[start:end],'k')
#    ax3.set(title='Modified Reference Signal')
#    ax4.plot(e[start:end],'k')
#    ax4.set(title='Output Filtered Signal')
    plt.tight_layout()
    
    plt.show()
    
    return e, y, w
    
    
    
            
#def remove_PLR_padasip(pd,illum,n,mu,mu_start,mu_end,steps,epochs):    
#    d = np.array(pd)
#    d_norm = d / np.linalg.norm(d)
#    illum_norm = illum / np.linalg.norm(illum)
#    x_norm = pa.input_from_history(illum_norm,n)[:-1]
#    f = pa.filters.AdaptiveFilter(model="NLMS",n=n,mu=mu,w='zeros')
##    f = pa.filters.FilterLMS(n,mu=mu,w='zeros')
##    d = d[n:]
#    d_norm = d_norm[n:]
#    y, e, w = f.run(d_norm, x_norm)
#    
#    # results
#    plt.figure(figsize=(12.5,6))    
#    
#    plt.plot(illum_norm, "c", label="reference signal")
#    plt.plot(d_norm, "k", label="recorded signal")
#    plt.plot(y, "--r", label="modified reference signal")
#    plt.plot(e, "k", label="filtered signal")
#    plt.grid()
#    plt.title("mu = "+str(mu)+", n = "+str(n))
#    plt.legend()
#    plt.show()
#    
    #    fig = plt.figure()
#    ax1 = fig.add_subplot(411)
#    ax2 = fig.add_subplot(412) 
#    ax3 = fig.add_subplot(413)
#    ax4 = fig.add_subplot(414)
#    ax1.plot(d_norm,'k')
#    ax1.set(title='Primary Input')
#    ax2.plot(illum_norm,'k')
#    ax2.set(title='Reference Input')
#    ax3.plot(y,'k')
#    ax3.set(title='Output Filtered Signal')
#    ax4.plot(e,'k')
#    ax4.set(title='Error')
#    plt.tight_layout()
#    
#    plt.show()
    
    # Search for optimal learning rate
#    errors_e, mu_range = f.explore_learning(d_norm, x_norm,
#                mu_start=mu_start,
#                mu_end=mu_end,
#                steps=steps, ntrain=0.8, epochs=epochs,
#                criteria="MSE")
#    plt.figure()    
#    
#    plt.plot(mu_range,errors_e, "k")
#    plt.grid()
#    plt.title("Error VS Learning rate")
#    plt.show()
#   
#    return e, y, w


def test_pupil():    
    # FAP is loaded by default
    face_dataset = AffectiveMonitorDataset("C:\\Users\\dspcrew\\affective-monitor-model\\data",subjects=[1])
#    face_dataset = AffectiveMonitorDataset("C:\\Users\\DSPLab\\Research\\ExperimentData",subjects=[5])
#    face_dataset = AffectiveMonitorDataset("E:\\Research\\ExperimentData",subjects=[1])
    samples = face_dataset[:]
    dataframe = face_dataset.face_df
    return samples, dataframe


if __name__ == "__main__":
    samples , dataframe = test_pupil()
#    plot_pupil(dataframe['PupilDiameter'],dataframe['Illuminance'])
    pd_left, pd_right = tuple_to_list(dataframe['PupilDiameter'])
    plt.figure()
    plt.plot(pd_left,'k',label="pd_left")
    plt.plot(pd_right,'r--',label="pd_right")
    plt.legend()
    plt.show()
    illum = dataframe['Illuminance'].values
#    filtered_pupil_left, error, weight = remove_PLR_padasip(pd_left,illum,30,50,1,50,100,1)
    PAR, PLR, weight = remove_PLR(pd_left,illum,10,15)
    
    
    
    
    
    
    
    
    
    
    
    
