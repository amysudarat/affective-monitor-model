# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal
from scipy.signal import peak_widths
import matplotlib.pyplot as plt
import utils
import peakutils
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def faps_slide_subplot(faps_feat_df,sbj,label=False):
    if sbj != 'all':
        faps_feat_df = faps_feat_df[faps_feat_df['sbj_idx']==sbj] 
    
    # prepare faps that will be plotted
    faps = faps_feat_df['faps'].tolist()
    if label:
        labels = faps_feat_df['label'].tolist()
    FAP_index = [i for i in range(19)]
    # slide show
    
    for i in range(len(faps)):
        fig, axes = plt.subplots(nrows=10,ncols=2,sharex=True,figsize=(18,16))
        for j, ax in enumerate(axes.flatten()):
            if j == 19:
                break
            ax.plot(faps[i][:,j])
            ax.set_ylabel(FAP_index[j])

        if label:
            fig.suptitle(str(labels[i]))
        plt.show()
        plt.waitforbuttonpress()
        plt.close()
    return

def faps_slide_plot(faps_feat_df,sbj,label=False,peak_plot=True):
    if sbj != 'all':
        faps_feat_df = faps_feat_df[faps_feat_df['sbj_idx']==sbj] 
    
    # prepare faps that will be plotted
    faps = faps_feat_df['faps'].tolist()
    if peak_plot:
        peaks = faps_feat_df['peak_pos'].tolist()
        try:
            p_selects = faps_feat_df['p_sel'].tolist()
            p_lbs = faps_feat_df['p_lb'].tolist()
            p_rbs = faps_feat_df['p_rb'].tolist()
        except:
            pass
    if label:
        labels = faps_feat_df['label'].tolist()
    # slide show
    for i in range(len(faps)):
        plt.figure(figsize=(10,8))
        try:
            for col in range(faps[i].shape[1]):
                plt.plot(faps[i][:,col])
        except:
            plt.plot(faps[i])
        if peak_plot:
            try:
                for p in peaks[i]:
                    plt.axvline(p,color='black',lw=1)
                plt.axvline(p_selects[i],color='black',lw=3)
                plt.axvline(p_lbs[i],color='black',lw=3)
                plt.axvline(p_rbs[i],color='black',lw=3)
            except:
                plt.axvline(peaks[i],color='black',lw=3)           
        if label:
            plt.title(str(labels[i]))
        FAP_index = ['l_i_eyebrow_y','r_i_eyebrow_y','l_o_eyebrow_y','r_o_eyebrow_y',
             'l_i_eyebrow_x','r_i_eyebrow_x','t_l_eyelid_y','t_r_eyelid_y',
             'l_cheeck_y','r_cheeck_y','l_nose_x','r_nose_x',
             'l_o_cornerlip_y','r_o_cornerlip_y','l_o_cornerlip_x','r_o_cornerlip_x',
             'l_b_midlip_y','l_t_midlip_y','open_jaw']
        plt.legend(FAP_index)
        plt.show()
        plt.waitforbuttonpress()
        plt.close()
    return

def dir_vector_slide_plot(faps_df,sbj,label=False):
    if sbj != 'all':
        faps_df = faps_df[faps_df['sbj_idx']==sbj]     
    # prepare faps that will be plotted
    sel_col = [str(i) for i in range(19)]
    faps = faps_df[sel_col].values
    if label:
        labels = faps_df['label'].tolist()
    # slide show
    i = 0
    for row in range(faps.shape[0]):
        plt.figure()
        plt.stem(faps[row])
        if label:
            plt.title(str(labels[i]))
        i += 1
        plt.show()
        plt.waitforbuttonpress()
        plt.close()
    return

def calm_detector(faps_df,thres=1,remove=True):
    
    def mask_gen(row,thres):
        fap = row['faps']
        col = [i for i in range(19)]
        # remove fap 6 and 7
        col.remove(6)
        col.remove(7)
        fap = fap[:,col]
        # absolute value
        fap = np.absolute(fap)
        # find peak for each traces
        p_collect = []
        for i in range(fap.shape[1]):
            p = peakutils.indexes(fap[:,i],min_dist=10,thres=0)
            if len(p) > 0:
                p_mag = [fap[p_pos,i] for p_pos in p]
                p_collect.append(np.max(p_mag))
        if len(p_collect) > 0:
            max_peak_avg = np.average(p_collect)
            if max_peak_avg < thres :
                row['calm_mask'] = True
            else:
                row['calm_mask'] = False
        else:
            row['calm_mask'] = True
        return row
    
    faps_df = faps_df.apply(mask_gen,thres=thres,axis=1)
    if remove:
        faps_df = faps_df[~faps_df['calm_mask']] 
    else:
        faps_df = faps_df[faps_df['calm_mask']]
    return faps_df.drop('calm_mask',axis=1)

def get_peak(faps_df,mode='peak',window_width=10,sliding_step=3,min_dist=10):
    
    def find_peak_cov(x,w):
        # change shape to (19,100) from (100,19)
        x = x.transpose()
        L = x.shape[1]     
        # find each cov for each sliding window
        diff_cov = []
        for i in range(w,L-w,sliding_step):
            x_w = x[:,i:i+w]
            cov_m = np.cov(x_w)
            # map the positive           
            pos = 0
            neg = 0
            for row in range(cov_m.shape[0]):
                for col in range(row+1,cov_m.shape[1]):
                    # normalize covarience by this formula 
                    # cov(x1,x2) / (std(x1)*std(x2))
                    cov_m[row,col] = cov_m[row,col]/(np.sqrt(cov_m[row,row])*np.sqrt(cov_m[col,col]))
                    if cov_m[row,col] >= 0:
                        pos = pos+cov_m[row,col]
                    else:
                        neg = neg+cov_m[row,col]
            diff_val = abs(pos) - abs(neg)
            diff_cov.append(diff_val)
        # peak should be at the maximum different + size of window
        peak_position = w + np.argmax(diff_cov)
        return [peak_position]
    
    def find_peak_peakutils(x,min_dist,thres,col=None):
        # use peak detection and find the maximum peak
        x = np.abs(x)
        x = np.sum(x,axis=1)
        if col is not None:
            p = peakutils.indexes(x[:,col],min_dist=min_dist,thres=thres)  
        else:
            p = peakutils.indexes(x,min_dist=min_dist,thres=thres)               
        return p
    
    # apply faps_df['faps'] with find peak function 
    if mode == 'cov':
        faps_df['peak_pos'] = faps_df['faps'].apply(find_peak_cov,w=window_width)
    elif mode == 'peak':
        # eye
        faps_df['p_eye'] = faps_df['faps'].apply(find_peak_peakutils,col=[0,1,2,3,4,5],min_dist=min_dist,thres=0)
        # eyelid
        faps_df['p_eyelid'] = faps_df['faps'].apply(find_peak_peakutils,col=[6,7],min_dist=min_dist,thres=0)
        # cheeck
        faps_df['p_cheeck'] = faps_df['faps'].apply(find_peak_peakutils,col=[8,9,10,11],min_dist=min_dist,thres=0)
        # mouth
        faps_df['p_mouth'] = faps_df['faps'].apply(find_peak_peakutils,col=[12,13,14,15,16,17,18],min_dist=min_dist,thres=0)
    
    return faps_df

def get_feature(faps_df):
    
    def get_peak_prop(row):
        fap = row['faps']
        peak = row['peak_pos']        
        fap = np.abs(fap)
        fap_sum = np.sum(fap,axis=1)
        
        # return value [width,height,left_ips,right_ips]
        p_width, p_height, p_lb, p_rb = peak_widths(fap_sum,peak,rel_height=0.75)
        
        # peak selection
        # criteria: width > 7
#        pop_idx = []
#        for i,w in enumerate(p_width):
#            if (w < 7 or w > 30) and len(p_width) > 2:
#                pop_idx.append(i)
#        pop_idx = sorted(pop_idx,reverse=True)
#        p_width = np.delete(p_width,pop_idx)
#        p_height = np.delete(p_height,pop_idx)
#        p_lb = np.delete(p_lb,pop_idx)
#        p_rb = np.delete(p_rb,pop_idx)
#        
        # criteria: big width, big height
#        criteria = np.add(p_width,p_height)
        criteria = np.divide(p_width,p_height)
        crit_idx = np.argmax(criteria)
        row['p_sel'] = peak[crit_idx]
        row['p_width'] = p_width[crit_idx]
        row['p_height'] = p_height[crit_idx]
        row['p_lb'] = p_lb[crit_idx]
        row['p_rb'] = p_rb[crit_idx]
        return row
    
    def get_dir_vector(row):
        fap = row['faps']
        p = row['p_sel']
        # find dir of each fap
        for col in range(fap.shape[1]):
            median = np.median(fap[:,col])
            if fap[p,col] > median:
                dir_val = 1
            else:
                dir_val = 0
            # create new column with name of index contain direction
            row[str(col)] = dir_val     
        return row
            
    faps_df = faps_df.apply(get_peak_prop,axis=1)
    faps_df = faps_df.apply(get_dir_vector,axis=1)
    return faps_df


def faps_preprocessing_samples(faps_df,smooth=True,fix_scaler='standard',aoi=None,sbj_num=88,fix_scaler_mode='sbj',sm_wid_len=10):
    
    # reserve test subject idx
    
    sbj_idx = [sbj_num for i in range(faps_df.shape[0])]
    faps_df['sbj_idx'] = sbj_idx
        
    if aoi is not None:
        faps_df['faps'] = faps_df['faps'].apply(lambda x:x[aoi[0]:aoi[1]])
    
    # absolute all the signal
    faps_df['faps'] = faps_df['faps'].apply(lambda x:np.absolute(x))

    if smooth:
        smoothed = []
        for i in range(faps_df.shape[0]):
            faps = np.array(faps_df.iloc[i]['faps'])
#            faps = scipy.signal.savgol_filter(faps,window_length=15,polyorder=2,axis=1)   
            for col in range(faps.shape[1]):
                faps[:,col] = scipy.signal.savgol_filter(faps[:,col],window_length=sm_wid_len,polyorder=2)   
            smoothed.append(faps)
        faps_df['tmp'] = smoothed
        faps_df = faps_df.drop('faps',axis=1)
        faps_df = faps_df[['tmp','ori_idx','sbj_idx']]
        faps_df.columns = ['faps','ori_idx','sbj_idx']
    
    if fix_scaler is not None:
        
        faps_block = faps_df['faps'].values
        a_to_fit = faps_block[0]
        for i in range(1,faps_df.shape[0]):
            a_to_fit = np.concatenate([a_to_fit,faps_block[i]])
        if fix_scaler == 'minmax':
            sc = MinMaxScaler()
        else:
            sc = StandardScaler()        
        if fix_scaler_mode == 'sbj':
            sc.fit(a_to_fit)
            faps_df['faps'] = faps_df['faps'].apply(lambda x:sc.transform(x))
        elif fix_scaler_mode == 'each':
            faps_df['faps'] = faps_df['faps'].apply(lambda x:sc.fit_transform(x))
#        # shift mean if use min max
#        if fix_scaler == 'minmax':
#            def shift_means(fap):
#                for col in range(fap.shape[1]):
#                    fap[:,col] = fap[:,col] - np.mean(fap[:,col])
#                return fap
#            faps_df['faps'] = faps_df['faps'].apply(shift_means)
        # set type of array
        faps_df['faps'] = faps_df['faps'].apply(lambda x:x.astype(np.float64))
    return faps_df

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
        faps_df['faps'] = faps_df['faps'].apply(lambda x:x[aoi[0]:aoi[1]])

    if smooth:
        smoothed = []
        for i in range(faps_df.shape[0]):
            faps = np.array(faps_df.iloc[i]['faps'])
#            faps = scipy.signal.savgol_filter(faps,window_length=15,polyorder=2,axis=1)   
            for col in range(faps.shape[1]):
                faps[:,col] = scipy.signal.savgol_filter(faps[:,col],window_length=21,polyorder=5)   
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
        # set type of array
        faps_df['faps'] = faps_df['faps'].apply(lambda x:x.astype(np.float64))
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