# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def smooth_differentiator(pd_signal):
    
    output = np.diff(pd_signal)
      
    return output


def plot_pd_before_after(sample,processed_pd=None,ax=None):
    
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
        processed_pd = [i+avg for i in processed_pd]
        ax.plot(processed_pd,'r',linewidth=2)
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