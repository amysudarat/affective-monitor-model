# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def median_filter(fap_signal,window=11):
    """ input should be numpy array and it will return numpy array """
    output = []
    for i in range(fap_signal.shape[1]):
        output.append(scipy.signal.medfilt(fap_signal[:,i]))
    return np.array(output)

def savgol_fap(fap_signal,window=15,polyorder=2):
    output = []
    for i in range(fap_signal.shape[1]):
        output.append(scipy.signal.savgol_filter(fap_signal[:,i],window,polyorder))   
    return np.array(output)

def plot_FAP_temporal(sample,sample_idx=None,processed_signal=None):
    valence = sample['valence']
    arousal = sample['arousal']
    sample = sample['faceFAP']
    sample = np.array(sample)
    FAP_index = ['l_i_eyebrow_y','r_i_eyebrow_y','l_o_eyebrow_y','r_o_eyebrow_y',
                 'l_i_eyebrow_x','r_i_eyebrow_x','t_l_eyelid_y','t_r_eyelid_y',
                 'l_cheeck_y','r_cheeck_y','l_nose_x','r_nose_x',
                 'l_o_cornerlip_y','r_o_cornerlip_y','l_o_cornerlip_x','r_o_cornerlip_x',
                 'l_b_midlip_y','l_t_midlip_y','open_jaw']
    
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(14, 12))
    
    for i, ax in enumerate(axes.flatten()):
        if i == 19:
            break
        ax.plot(sample[:,i])
        if processed_signal is not None:
            ax.plot(processed_signal[i,:],'--r')
        ax.set_title(FAP_index[i])
        
    if sample_idx is None:
        fig.suptitle("Arousal: "+ str(arousal) + " , Valence: "+ str(valence))
    else:
        fig.suptitle("Sample No.: "+ str(sample_idx)+" , Arousal: "
                     + str(arousal) + " , Valence: "+ str(valence))
#    plt.show()
    return fig