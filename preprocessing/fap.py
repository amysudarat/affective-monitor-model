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

def faps_slide_plot(faps_feat_df,sbj,label=False,peak_plot=None,plot_sig=None):
    if sbj != 'all':
        faps_feat_df = faps_feat_df[faps_feat_df['sbj_idx']==sbj] 
    if plot_sig is not None:
        faps_feat_df['faps'] = faps_feat_df['faps'].apply(lambda x:x[:,plot_sig])
    
    # prepare faps that will be plotted
    faps = faps_feat_df['faps'].tolist()
    if peak_plot is not None:
        peaks = faps_feat_df[peak_plot].tolist()
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
        plt.figure(figsize=(12,8))
        try:
            for col in range(faps[i].shape[1]):
                plt.plot(faps[i][:,col])
        except:
            plt.plot(faps[i])
        if peak_plot is not None:
            if len(peaks[i])>0:
                for p in peaks[i]:
                    plt.axvline(p,color='black',lw=1)
            try:                
                plt.axvline(p_selects[i],color='black',lw=3)
                plt.axvline(p_lbs[i],color='black',lw=3)
                plt.axvline(p_rbs[i],color='black',lw=3)
            except:
                pass
            
        if label:
            plt.title(str(labels[i]))
#        FAP_index = ['l_i_eyebrow_y','r_i_eyebrow_y','l_o_eyebrow_y','r_o_eyebrow_y',
#             'l_i_eyebrow_x','r_i_eyebrow_x','t_l_eyelid_y','t_r_eyelid_y',
#             'l_cheeck_y','r_cheeck_y','l_nose_x','r_nose_x',
#             'l_o_cornerlip_y','r_o_cornerlip_y','l_o_cornerlip_x','r_o_cornerlip_x',
#             'l_b_midlip_y','l_t_midlip_y','open_jaw']
        if plot_sig is not None:
            FAP_index = plot_sig
        else:
            FAP_index = [i for i in range(19)]
        plt.legend(FAP_index)
        plt.show()
        plt.waitforbuttonpress()
        plt.close()
    return

def dir_vector_slide_plot(faps_df,sbj,label=False):
    if sbj != 'all':
        faps_df = faps_df[faps_df['sbj_idx']==sbj]     
    # prepare faps that will be plotted
    au = faps_df['FAP'].tolist()
    if label:
        labels = faps_df['label'].tolist()
    # slide show
    i = 0
    for row in range(len(au)):
        plt.figure()
        x = [i for i in range(19)]
        plt.stem(x,au[row])
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

def get_peak(faps_df,mode='peak',window_width=10,sliding_step=3,min_dist=10,thres=0.6):
    
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
    
    def find_peak_peakutils(row,min_dist,thres):
        # detect peak based on min_dist and threshold
        x = row['faps']
        col_sel = [i for i in range(19)]
        col_sel.remove(6)
        col_sel.remove(7)
        col_sel.remove(14)
        col_sel.remove(15)
        x = x[:,col_sel]
        x = np.abs(x)            
        x = np.sum(x,axis=1)        
        p = peakutils.indexes(x,min_dist=min_dist,thres=thres)  
        row['peak_pos'] = p
        
        # select peak
        p_width, p_height, p_lb, p_rb = peak_widths(x,p,rel_height=1)
        
        # create array of peak properties
        # each column is one peak, delete column that p_width is less than 7
        p_prop_np = np.array([p_width,p_height,p_lb,p_rb,p])
        col_del = []
        for col in range(p_prop_np.shape[1]):
            if p_prop_np[0,col] < 7:
                col_del.append(col)
        col_del.sort(reverse=True)
        for c in col_del:
            p_prop_np = np.delete(p_prop_np,p_prop_np[:,c],1)
        
        # calculate p_width/p_height
        if len(p_prop_np.tolist()[0]) > 0:
            criteria = np.divide(p_prop_np[0],p_prop_np[1])
            crit_idx = np.argmax(criteria)
            p_sel = [p_prop_np[4,crit_idx]]            
        else:
            p_sel = []
        
        row['peak_sel'] = p_sel
        
        # get AU
        if len(p_sel)>0:
            # get window length of p_width
            x = row['faps']
            c = p_sel[0]
            L = 20
            L = int(round(L/2,0))
            pl = int(max(0,c-L))
            pr = int(min(x.shape[0],c+L))            
            x_win = x[pl:pr,:]
         
            FAP = []
            for col in range(x_win.shape[1]):
                trace = x_win[:,col]
                trace_abs = np.absolute(trace)
                p_trace = peakutils.indexes(trace_abs,thres=0.4,min_dist=10)
                if len(p_trace) == 0:
                    FAP.append(0)
                    continue
                else:
                    pp = p_trace[np.argmax([trace[i] for i in p_trace])]
                    slope_l = (trace[pp]-trace[0])/(pp)
                    slope_r = (trace[len(trace)-1]-trace[pp])/(19-pp)
                    if slope_l > 0 and slope_r < 0:
                        FAP.append(1)
                    elif slope_l < 0 and slope_r > 0:
                        FAP.append(-1)
                    else:
                        FAP.append(0)
        else:
            FAP = [0 for i in range(19)]
                
        # create column AU
        row['FAP'] = FAP
        
        # convert it to AU
        # AU1
        if FAP[0] == 1 and FAP[1] == 1:
            row['AU1'] = 1
        else:
            row['AU1'] = 0
        # AU2
        if FAP[2] == 1 and FAP[3] == 1:
            row['AU2'] = 1
        else:
            row['AU2'] = 0
        # AU4
        if FAP[0] == -1 and FAP[1] == -1 and FAP[4] == -1 and FAP[5] == -1:
            row['AU4'] = 1
        else:
            row['AU4'] = 0
        # AU5
        if FAP[6] == 1 and FAP[7] == 1 :
            row['AU5'] = 1
        else:
            row['AU5'] = 0
        # AU6
        if FAP[6] == -1 and FAP[7] == -1 and FAP[8] == -1 and FAP[9] == -1:
            row['AU6'] = 1
        else:
            row['AU6'] = 0
        # AU9
        if FAP[10] == -1 and FAP[11] == -1:
            row['AU9'] = 1
        else:
            row['AU9'] = 0
        # AU10
        if FAP[12] == -1 and FAP[13] == -1:
            row['AU10'] = 1
        else:
            row['AU10'] = 0
        # AU12
        if FAP[12] == -1 and FAP[13] == -1 and FAP[14] == 1 and FAP[15] == 1:
            row['AU12'] = 1
        else:
            row['AU12'] = 0
        # AU15
        if FAP[12] == 1 and FAP[13] == 1:
            row['AU15'] = 1
        else:
            row['AU15'] = 0
        # AU16
        if FAP[16] == -1 and FAP[18] == -1:
            row['AU16'] = 1
        else:
            row['AU16'] = 0
        # AU20
        if FAP[14] == 1 and FAP[15] == 1 and FAP[16] == -1:
            row['AU20'] = 1
        else:
            row['AU20'] = 0
        # AU23
        if FAP[14] == -1 and FAP[15] == -1:
            row['AU23'] = 1
        else:
            row['AU23'] = 0
        # AU26
        if FAP[18] == 1 and FAP[16] == 1:
            row['AU26'] = 1
        else:
            row['AU26'] = 0
                
        return row
    
    # apply faps_df['faps'] with find peak function 
    if mode == 'cov':
        faps_df['peak_pos'] = faps_df['faps'].apply(find_peak_cov,w=window_width)
    elif mode == 'peak':
        # find peak
        faps_df = faps_df.apply(find_peak_peakutils,min_dist=min_dist,thres=thres,axis=1)
    return faps_df


def faps_preprocessing_samples(faps_df,smooth=True,fix_scaler='standard',aoi=None,sbj_num=88,fix_scaler_mode='sbj',sm_wid_len=10,center_mean=False):
    
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
        
        if center_mean:
            faps_df['faps'] = faps_df['faps'].apply(lambda x:x-np.average(x))
#         set type of array
        faps_df['faps'] = faps_df['faps'].apply(lambda x:x.astype(np.float64))
    return faps_df

def faps_preprocessing(faps_df,smooth=True,filter_miss=None,fix_scaler='standard',aoi=None,sm_wid_len=10,center_mean=False):
    
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
        
        if center_mean:
            faps_df['faps'] = faps_df['faps'].apply(lambda x:x-np.average(x))
        
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