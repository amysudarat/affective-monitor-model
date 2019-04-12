# -*- coding: utf-8 -*-

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal


def detect_glitch(raw, threshold=1.5):
    
    # pass it to differentiator
    diff_raw = differentiator(raw)
    
    # detect where is the glitch
    glitch_index = [i for i in range(len(diff_raw)) if diff_raw[i] > threshold or diff_raw[i] < -1*threshold]
    
    # check if between indexes if the diff value is zero or not (no change in between)
    # then replace it with average
    i = 0
    output = raw
    while i+1 < len(glitch_index): # 2<3
        start = glitch_index[i]+1 
        stop = glitch_index[i+1]
        if sum(diff_raw[start:stop]) == 0:
            replace_value = (raw[start-1]+raw[stop+1])/2
            replace_list = [replace_value for i in range(stop-start+1)]
            output = output[:start]+replace_list+output[stop+1:]
            assert len(raw) == len(output), "fix slicing list"
        i = i+1 # 2
    

    return glitch_index, output




def differentiator(pd_signal):
    
    output = np.diff(pd_signal)
      
    return output

def gradient(pd_signal):
    
    output = np.gradient(pd_signal)
      
    return output

def savgol(pd_signal,window=15,polyorder=2):
    
    output = scipy.signal.savgol_filter(pd_signal,window,polyorder)
      
    return output


def plot_pd_before_after(sample,processed_pd=None,ax=None,adjust=True,glitch_index=None,):
    
    if ax is None:        
        ax = plt.axes()
        ax.set_title("black = original, red = processed signal")
        ax.grid(True)
    
#    pd_left = sample["PD_left_filtered"]
#    zero_line = [0 for i in range(len(pd_left))]
#    pd_right = sample["PD_right_filtered"]
    pd_merge = sample["PD_avg_filtered"]
    avg = np.average(pd_merge)
#    depth = sample["depth"]
    arousal = sample["arousal"]    
    ax.text(0, pd_merge[0], str(arousal), bbox=dict(facecolor='red', alpha=0.5))
    ax.plot(pd_merge,'k')
    if processed_pd is not None:
        if adjust:
            processed_pd = [i+avg for i in processed_pd]
        ax.plot(processed_pd,'--r',linewidth=2)
    if glitch_index is not None:
        x = glitch_index
        y = [pd_merge[i] for i in glitch_index]
        ax.plot(x,y,'bo')
#    ax.plot(pd_left,'--r')
#    ax.plot(pd_right,'--b')
#    ax.plot(zero_line,'y')
    
#    ax.plot(depth,'g')
    
#    if ax is None:   
        
#        plt.show()
        
#    else:
        # Turn off tick labels
#        ax.xaxis.set_visible(False)
#        ax.yaxis.set_visible(False)
    return