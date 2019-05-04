# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import utils
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def faps_preprocessing(faps_df,smooth=True,filter_miss=None,fix_scaler='standard',aoi=None):
    
    # reserve test subject idx
    total_sbj = int((faps_df.index.max()+1)/70)
    sbj_idx = [j for j in range(1,total_sbj+1) for i in range(70) ]
    faps_df['sbj_idx'] = sbj_idx
    
    if filter_miss is not None:
        faps_df['miss_ratio'] = filter_miss
        faps_df = faps_df[faps_df['miss_ratio'] <= 25]
        faps_df = faps_df.drop('miss_ratio',axis=1)
        
    if aoi is not None:
        cut = []
        for i in range(faps_df.shape[0]):
            faps = np.array(faps_df.iloc[i]['faps'])
            faps = faps[aoi[0]:aoi[1]]
            cut.append(faps)
        faps_df['tmp'] = cut
        faps_df = faps_df.drop('faps',axis=1)
        faps_df = faps_df[['tmp','ori_idx','sbj_idx']]
        faps_df.columns = ['faps','ori_idx','sbj_idx']
    
    if smooth:
        smoothed = []
        for i in range(faps_df.shape[0]):
            faps = np.array(faps_df.iloc[i]['faps'])
#            faps = scipy.signal.savgol_filter(faps,window_length=15,polyorder=2,axis=1)   
            for col in range(faps.shape[1]):
                faps[:,col] = scipy.signal.savgol_filter(faps[:,col],window_length=15,polyorder=2)   
            smoothed.append(faps)
        faps_df['tmp'] = smoothed
        faps_df = faps_df.drop('faps',axis=1)
        faps_df = faps_df[['tmp','ori_idx','sbj_idx']]
        faps_df.columns = ['faps','ori_idx','sbj_idx']
    
    if fix_scaler is not None:
        output_df = pd.DataFrame()
        for subject_idx in range(1,52):
            faps_per_sbj = faps_df[faps_df['sbj_idx']==subject_idx]
            faps_block = faps_per_sbj['faps'].values
            a_to_fit = faps_block[0]
            for i in range(1,faps_per_sbj.shape[0]):
                a_to_fit = np.concatenate([a_to_fit,faps_block[i]])
            if fix_scaler == 'minmax':
                sc = MinMaxScaler()
            else:
                sc = StandardScaler()
            sc.fit(a_to_fit)
            tmp_df = faps_per_sbj.copy()
            tmp_df['faps'] = faps_per_sbj['faps'].apply(lambda x:sc.transform(x))
            output_df = output_df.append(tmp_df)
        faps_df = output_df
    return faps_df


def get_faps_df(pickle_file="data_1_50_fixPD_Label_False.pkl"):
    face_dataset = utils.load_object(pickle_file)
    array_samples = []
    for i in range(len(face_dataset)):
        array_samples.append(face_dataset[i]['faceFAP'])
    array_samples = np.array(array_samples)
    # create dataframe from 3d numpy array (stacking each sample)
    faps_df = pd.DataFrame()
    faps_col = ['31','32','35','36','37','38','19','20',
                       '41','42','61','62','59','60','53','54','5','4','3']
    for slice_idx in range(1,array_samples.shape[0]+1):
        # slice array create view which is a shallow copy of array (different id)
        tmp_df = pd.DataFrame(array_samples[slice_idx-1,:,:])
        tmp_df.columns = faps_col
        tmp_df['index'] = [slice_idx for j in range(tmp_df.shape[0])]
        tmp_df = tmp_df.set_index('index')
        faps_df = faps_df.append(tmp_df)   
    return faps_df    

def get_faps_np_df(pickle_file='data_1_51.pkl'):
    face_dataset = utils.load_object(pickle_file)
    faps_df = pd.DataFrame(face_dataset[:]['faceFAP'])
    faps_df.columns = ['faps']
    faps_df['ori_idx'] = [i for i in range(len(face_dataset))]
    faps_df['faps'] = faps_df['faps'].apply(lambda x: np.array(x))
    return faps_df

def get_missing_percentage(faps_df):
    miss_list = []
    faps_df = faps_df.drop(columns=['ori_idx'])
    for smp_idx in range(faps_df.shape[0]):
        faps_np = np.array(faps_df.iloc[smp_idx]['faps'])
        tmp = np.diff(faps_np[:,0])
        miss_sum = 0
        for i in tmp:
            if i == 0:
                miss_sum+=1
        miss_list.append((miss_sum/faps_np.shape[0])*100)
    return miss_list
        
    
def savgol_filter(fap_signal,window=15,polyorder=2):
    """
        expect 2D array of faps shape (100,19)
    """
    output = []
    for i in range(fap_signal.shape[1]):
        output.append(scipy.signal.savgol_filter(fap_signal[:,i],window,polyorder))   
    return np.array(output).transpose()


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