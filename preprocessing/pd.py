# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import savgol_filter
import preprocessing.pre_utils as pu
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import preprocessing.fap as pfap
import utils

#import warnings
#warnings.filterwarnings("error")


def plot_sample(sig,p,q,r,text):
    fig, axes = plt.subplots(nrows=1,ncols=1)
    axes.plot(sig)
    axes.grid(True)
    axes.plot(p,sig[p],'ro')
    axes.plot(q,sig[q],'go')  
    axes.plot(r,sig[r],'mo') 
    axes.legend(['sig','p','q','r'])
    fig.suptitle(text)
    plt.show()

def get_pqr(sig,smooth=False):
    # find peak
    p = np.argmax(sig[:5])
    if smooth:
        sig = savgol_filter(sig,19,3)
    # observe slope
    sig_diff = np.diff(sig)
    sig_diff = [1 if i>0 else -1 for i in sig_diff]
    # find r by the first index that the slope is positive
    for i in range(p+1,len(sig)):
        if sig_diff[i] > 0:
            q = i
            break
    # find r by just observe the ten samples behind q
    for i in range(q+1,len(sig)):
        if sig_diff[i] != sig_diff[q]:
            r = i
            break
        r = i
#    r = q+8
    
    # calculate delta_pq
    delta_pq = sig[p]-sig[q]
    delta_qr = sig[r]-sig[q]
    slope_qr = delta_qr/(r-q)
    return p,q,r,delta_pq,delta_qr,slope_qr

def get_pqr_feature(pd_df,smooth=False):
    pd_np = pd_df.drop('ori_idx',axis=1).values
    delta_pq_list = []
    delta_qr_list = []
    slope_qr_list = []
    for row in range(pd_np.shape[0]):        
        p,q,r,delta_pq,delta_qr,slope_qr = get_pqr(pd_np[row],smooth=smooth)
        # calculate delta_pq
        delta_pq_list.append(delta_pq)
        delta_qr_list.append(delta_qr)
        slope_qr_list.append(slope_qr)
        
    tmp_df = pd.DataFrame(pd_np)
    tmp_df['delta_pq'] = delta_pq_list
    tmp_df['delta_qr'] = delta_qr_list
    tmp_df['slope_qr'] = slope_qr_list
    tmp_df['ori_idx'] = pd_df['ori_idx'].reset_index(drop=True)
    tmp_df.index = pd_df.index
    return tmp_df

def plot_pqr_slideshow(pd_df,sbj,smooth=False,label=None):
    if label is not None:
        idx = pd_df.index
        pd_df = pd_df.reset_index(drop=True)
        pd_df['label'] = label
        pd_df.index = idx
    if sbj == 'all':
        pd_np = pd_df[[i for i in range(40)]].values
        label = label
    else:
        pd_np = pd_df[[i for i in range(40)]]
        pd_np = pd_np.loc[sbj].values  
        label = pd_df.loc[sbj]['label'].values
    for row in range(pd_np.shape[0]):
        p,q,r,delta_pq,delta_qr,slope_qr = get_pqr(pd_np[row],smooth=smooth)
#        text = "delta_pq: {:.2f},delta_qr: {:.2f} ,slope_qr: {:.2f}".format(delta_pq,delta_qr,slope_qr)
        text = str(label[row])
        plot_sample(pd_np[row],p,q,r,text)
        plt.waitforbuttonpress()
        plt.close()        

def preprocessing_pd(pd_df,aoi=40,loc_artf='diff',n_mad=16,diff_threshold=0.2,interpolate=True,miss_threshold=None,norm=False):
    
    # reserve test subject idx
    sbj_idx = [i for i in range(pd_df.shape[0])]   
    if aoi is not None:
        pd_df = pd_df.drop(columns=[i for i in range(aoi,100)])
    
    pd_df['ori_idx'] = sbj_idx
    
    if loc_artf is not None:
        if loc_artf == 'diff':
            pd_list = pd_df.drop('ori_idx',axis=1).values.tolist()
            pd_filtered_list = []
            pd_filtered_list, _ = remove_glitch(pd_list,threshold=diff_threshold)
#            for elem in pd_list:
#                pd_filtered, _ = remove_glitch(elem,threshold=0.2)
#                pd_filtered_list.append(pd_filtered)
            tmp_df = pd.DataFrame(np.array(pd_filtered_list))
            tmp_df['ori_idx'] = pd_df['ori_idx'].reset_index(drop=True)
            tmp_df.index = pd_df.index
            pd_df = tmp_df
            del tmp_df
        elif loc_artf == 'mad_filter':
            pd_df = identify_artifact(pd_df,n=n_mad,interpolate=True)
            
    if miss_threshold is not None:
        pd_np = pd_df.drop('ori_idx',axis=1).values
        miss = []
        for row in range(pd_np.shape[0]):
            pd_sg = pd_np[row]
            pd_sg = np.diff(pd_sg)
            count = 0
            for i in pd_sg:
                if i == 0:
                    count+=1
            count = count/len(pd_sg)
            miss.append(count)
        pd_df['miss'] = miss
        pd_df = pd_df[pd_df['miss']<miss_threshold]
        pd_df = pd_df.drop('miss',axis=1)
    
    if norm:        
        tmp_df = pd.DataFrame()
        sc = MinMaxScaler()
        for i in range(1,pd_df.index.max()+1):
            pd_np = pd_df.loc[i].drop('ori_idx',axis=1).values
            pd_np = sc.fit_transform(pd_np.transpose())
            tmp_df = tmp_df.append(pd.DataFrame(pd_np.transpose()))
        tmp_df = tmp_df.reset_index(drop=True)
        tmp_df['ori_idx'] = pd_df['ori_idx'].reset_index(drop=True)
        tmp_df.index = pd_df.index
        pd_df = tmp_df
    return pd_df

def generate_features_df(samples):
    """
    Imagine bell curve, skew left (tail to left,positive), skew right (tail to right,negative)
    mean follows tail, median stay with the bulk
    """
    ori_column = samples['ori_idx']
    samples = samples.drop(columns=['ori_idx'])
    samples['mean'] = samples.mean(axis=1)
    samples['median'] = samples.median(axis=1)
    samples['max'] = samples.max(axis=1)
    samples['min'] = samples.min(axis=1)
    samples['skew'] = samples.skew(axis=1)
    samples['std'] = samples.std(axis=1)
    samples['ori_idx'] = ori_column 
    return samples


def pd_plot_pause(pd_df,sbj,r=40,ylim=[1,4],label=None):
    pd_np = pd_df.loc[sbj].values
    pd_np = pd_np[:,:r]
    try:
        ori_idx = pd_df.loc[sbj]['ori_idx'].values.tolist()
    except:
        ori_idx = [i for i in range(pd_df.shape[0])]
    
    for i in range(pd_np.shape[0]): 
        m = np.argmin(pd_np[i][5:10])
        m = m+5
        plt.figure()
        plt.ylim(ylim[0],ylim[1])
        plt.plot(pd_np[i])
        plt.plot(m,pd_np[i,m],'ro')
        plt.title(str(ori_idx[i]))
        plt.show()
        plt.waitforbuttonpress()
        plt.close()
    return

def identify_artifact(pd_df,n,ignore=5,interpolate=True):
    pd_np = pd_df.drop('ori_idx',axis=1).values
    for row in range(pd_np.shape[0]):
        signal = pd_np[row]
        di = []        
        # calculate dilation speed
        for i in range(ignore,signal.shape[0]-1):
            di.append(max(np.abs(signal[i]-signal[i-1]),np.abs(signal[i+1]-signal[i])))
        # calculate MAD
        di_med = np.median(di)
        MAD = np.median(np.abs([i-di_med for i in di]))
        # calculate threshold
        threshold = np.median(di) + (n*MAD)
        
        # sample of di that is above the threshold is invalid
        for i,elem in enumerate(di):
            if elem > threshold:
                signal[i+ignore] = np.nan
        pd_np[row] = signal
    output_df = pd.DataFrame(pd_np)
    if interpolate:
        output_df[output_df.columns] = output_df[output_df.columns].astype(float).apply(lambda x:x.interpolate(method='index'),axis=1)
    output_df['ori_idx'] = pd_df['ori_idx'].reset_index(drop=True)
    output_df.index = pd_df.index
    output_df.columns = pd_df.columns

        
    return output_df


def select_and_clean(samples,norm=True,miss_percent=None,miss_threshold=0.4,sd_detect_remove=True,smooth=False,align=True,fix_depth=None,fix_illum=None,fix_illum_alt=None,alpha=0.03,beta=-5):
    """
        filter and transform samples based on the method parameter set, 
        return dataframe of output signals
    samples: 
        list of pd signals
    norm:
        (boolean)
    miss_percent:
        array of missing percentage 
    miss_threshold: 
        if missing_percent of that sample is larger than
        setting threshold, discard that sample
    label: 
        array of label
    sd_detect_remove: 
        discard the sample if one of their sequence deviate from 3 unit of std
    align : 
        shift the starting of the sample to the overall mean of each test subject
    """
    output_df = pd.DataFrame()
    for subject_idx in range(1,52):
        # [0,70,140,...]
        start_idx = ((subject_idx*70)-70)
        # [70,140,210,...]
        stop_idx = (subject_idx*70)
        
        # create dataframe per test subject
        subject = samples[start_idx:stop_idx]
        subject = np.array(subject)
                
        # drop sample with has missing percent more than 60%
        subject_df = pd.DataFrame(subject)
        subject_df['ori_idx'] = pd.Series([i for i in range(start_idx,stop_idx)])
        if miss_percent is not None:            
            miss_column = miss_percent[start_idx:stop_idx]
            subject_df['missing_percent'] = miss_column
            subject_df = subject_df[subject_df.missing_percent <= miss_threshold]
            subject_df = subject_df.drop(columns=['missing_percent'])
        if sd_detect_remove:
            # mean and std of the whole dataset
            df_mean = subject_df.drop(columns=['ori_idx']).values.mean()
            df_std = subject_df.drop(columns=['ori_idx']).values.std()
            upper_threshold = df_mean + 3*df_std
            lower_threshold = df_mean - 3*df_std
            subject_df = subject_df.reset_index(drop=True)
            def generate_mask(row,upper=upper_threshold,lower=lower_threshold):
                for i in row:
                    if i < lower or i > upper:
                        return False
                return True
            subject_df = subject_df[subject_df.drop(columns=['ori_idx']).apply(generate_mask,axis=1)]
        
        # align the starting point
        if align:
            df_mean = subject_df.drop(columns=['ori_idx']).values.mean()
            ori_idx_row_col = subject_df['ori_idx']
            pd_np = subject_df.drop(columns=['ori_idx']).values
            for i in range(pd_np.shape[0]):
                pd_np[i,:] = pd_np[i,:] + (df_mean-pd_np[i,0])
            subject_df = pd.DataFrame(pd_np)            
            subject_df['ori_idx'] = ori_idx_row_col.reset_index(drop=True)
        
        if smooth:
            ori_idx_list = subject_df['ori_idx'].tolist()
            pd_np = subject_df.drop('ori_idx',axis=1).values.transpose()
            pd_np = pfap.savgol_filter(pd_np,window=5,polyorder=3).transpose()
            tmp_df = pd.DataFrame(pd_np)
            tmp_df['ori_idx'] = subject_df['ori_idx']
            subject_df = tmp_df 
        
        if fix_depth is not None:
            ori_idx_list = subject_df['ori_idx'].tolist()
            depth_mean = fix_depth[fix_depth.index.isin(ori_idx_list)]['mean_per_frame'].values
            depth_min = fix_depth[fix_depth.index.isin(ori_idx_list)]['min'].values
            pd_np = subject_df.drop('ori_idx',axis=1).values
            for row in range(pd_np.shape[0]):
                pd_np[row] = pd_np[row]+(depth_mean[row]/depth_min[row])
            tmp_df = pd.DataFrame(pd_np)
            tmp_df['ori_idx'] = subject_df['ori_idx']
            subject_df = tmp_df
                        
        if fix_illum is not None:
            ori_idx_list = subject_df['ori_idx'].tolist()
            illum_mean = fix_illum[fix_illum.index.isin(ori_idx_list)]['mean_per_frame'].values
            illum_sbj_mean = fix_illum[fix_illum.index.isin(ori_idx_list)]['mean_per_subject'].values.tolist()[0]            
            pd_np = subject_df.drop('ori_idx',axis=1).values
            for row in range(pd_np.shape[0]):
                pd_np[row] = pd_np[row]+ (alpha*(illum_mean[row]-illum_sbj_mean))
            tmp_df = pd.DataFrame(pd_np)
            tmp_df['ori_idx'] = subject_df['ori_idx']
            subject_df = tmp_df
        
        if fix_illum_alt is not None:
            ori_idx_list = subject_df['ori_idx'].tolist()
            illum_rec = fix_illum_alt[fix_illum_alt.index.isin(ori_idx_list)]['illum_rec'].values
            pd_np = subject_df.drop('ori_idx',axis=1).values
            for row in range(pd_np.shape[0]):
                pd_np[row] = pd_np[row]+ (beta/max(illum_rec))*illum_rec[row]
            tmp_df = pd.DataFrame(pd_np)
            tmp_df['ori_idx'] = subject_df['ori_idx']
            subject_df = tmp_df
        
        # normalization mix max                
        if norm:            
            subject = subject_df.drop(columns=['ori_idx']).values
            min_val = subject.min()
            max_val = subject.max()
            subject = (subject-min_val)/(max_val-min_val)
       
        # convert numpy array to list and append it to output list
        
        subject = pd.DataFrame(subject)        
        subject['ori_idx'] = subject_df['ori_idx']
        subject['index'] = subject_idx
        subject = subject.set_index('index')
        output_df = output_df.append(subject)       
    return output_df


def get_missing_percentage(samples):
    missing_percentages = []
    for sample in samples:
        # get differentiation of sample
        diff_sample = differentiator(sample)
        # detect diff signal if it's zero
        count = 0
        for i in diff_sample:
            if i == 0:
                count+=1
        missing_percentages.append(count/len(diff_sample))
    return missing_percentages
    
    
def get_aoi_df(samples,start=20,stop=70):
    ori_column = samples['ori_idx']
    samples = samples.drop(columns=['ori_idx'])
    samples = samples.drop(columns=[i for i in range(stop,samples.shape[1])]) 
    samples = samples.drop(columns=[i for i in range(start)])
    samples['ori_idx'] = ori_column    
    return samples

def get_pds(pickle_file="data_1_50_fixPD_Label_False.pkl"):
    face_dataset = utils.load_object(pickle_file)
    array_samples = []
    for i in range(len(face_dataset)):
        array_samples.append(face_dataset[i]['PD_avg_filtered'])
    return array_samples

def get_raw_pd_df(samples,subjects):   
    output_df = pd.DataFrame()
    for subject_idx in range(1,52):
        # [0,70,140,...]
        start_idx = ((subject_idx*70)-70)
        # [70,140,210,...]
        stop_idx = (subject_idx*70)        
        # create dataframe per test subject
        pd_df = samples[start_idx:stop_idx]
        pd_df = np.array(pd_df)                
        pd_df = pd.DataFrame(pd_df)
        # prepare each sbj df and append to output df
        pd_df['index'] = subject_idx
        pd_df = pd_df.set_index('index')
        output_df = output_df.append(pd_df)
    return output_df

def my_lms(d,r,L,mu):
    e = np.zeros(d.shape)
    y = np.zeros(r.shape)
    w = np.zeros(L)
    
    for k in range(L,len(r)):
        x = r[k-L:k]
        y[k] = np.dot(x,w)
        e[k] = d[k]-y[k]
        try:
            w_next = w + (2*mu*e[k])*x
        except:
            print("here is when it fails")
        w = w_next   
    return y, e, w

def remove_PLR(pd_df,illums,n=10,mu=0.5,adjust=False,showFigures=None,arousal_col=True):
    """
        accept dataframe and return dataframe that PLR effect is removed along
        with the weights logs and modified reference signals
    pd:
        pupil diameter dataframe should contain column 'ori_idx_row' only aside
        from pupil diameter signals to refer to the corresponding illums order
        if column 'arousal' is attached then have to set arousal_col to True
    illums:
        accept list of illums list signal 
    n:
        length of adaptive window
    mu:
        learning rate
    """
    # preserve index col
    index_col = pd_df.index
    ori_idx_row = pd_df['ori_idx_row'].tolist()
    pd_np = pd_df.drop(columns=['ori_idx_row'])
    if arousal_col:
        arousal = pd_df['arousal']
        pd_np = pd_np.drop(columns=['arousal'])
    pd_np = pd_np.values
    original_pd = []
    processed_pd = []
    weights_log = []
    modified_r_signal = []
    for i in range(pd_np.shape[0]):
        d = np.array(pd_np[i,:])
        original_pd.append(d)
        illum = np.array(illums[ori_idx_row[i]])
        if adjust:
            d = d / np.linalg.norm(d)
            illum = illum / np.linalg.norm(illum)
            illum = 1.2*illum
            illum = illum - np.mean(illum) + np.mean(d)
        # call lms here
        y, e, w = my_lms(d,illum,n,mu)   
        processed_pd.append(e)
        weights_log.append(w)
        modified_r_signal.append(y)
    # create output dataframe with the original index based on test subject
    output_df = pd.DataFrame(processed_pd)
    if arousal_col:
        output_df['arousal'] = arousal.reset_index(drop=True)
    output_df = output_df.set_index(index_col)
    weight_log_df = pd.DataFrame(weights_log)
    weight_log_df = weight_log_df.set_index(index_col)
    modified_r_signal_df = pd.DataFrame(modified_r_signal)
    modified_r_signal_df = modified_r_signal_df.set_index(index_col)
    
    # plot if showFigure is True
    if showFigures is not None:        
        for sample_idx in showFigures:
            original_signal = original_pd[sample_idx]
            processed_signal = processed_pd[sample_idx]
            illum_signal = illums[ori_idx_row[sample_idx]]
            illum_signal_adjust = ((np.array(illum_signal) - min(illum_signal)) / (max(illum_signal)-min(illum_signal))).tolist()
#            illum_signal_adjust = illum_signal_adjust -np.mean(illum_signal_adjust) + np.mean(original_signal)
            modified_illum_signal = modified_r_signal[sample_idx]
            # first plot same graph
            plt.figure()
            fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(14, 12))
            axes.plot(original_signal,label='original pd')
            axes.plot(processed_signal,label='processed pd')
            axes.plot(illum_signal_adjust,label='original illum (adjust)')
            axes.plot(modified_illum_signal,label='modified illum')
            axes.grid(True)
            axes.legend()
            fig.suptitle("Sample No.: "+str(sample_idx))
            # second plot plot separately
            plt.figure()
            fig, axes = plt.subplots(nrows=4,ncols=1,figsize=(14, 12))
            axes[0].plot(original_signal,label='original pd')
            axes[0].set_ylabel("original pd")
            axes[0].grid(True)
            axes[1].plot(processed_signal,label='processed pd')
            axes[1].set_ylabel("processed pd")
            axes[1].grid(True)
            axes[2].plot(illum_signal,label='original illum')
            axes[2].set_ylabel("original illum")
            axes[2].grid(True)
            axes[3].plot(modified_illum_signal,label='modified illum')
            axes[3].set_ylabel("modified illum")
            axes[3].grid(True)
            fig.suptitle("Sample No.: "+str(sample_idx))
            # show plot
            plt.show()
    
    return output_df, weight_log_df, modified_r_signal_df

def detect_glitch(raw, threshold=0.3):
    
    # pass it to differentiator
    diff_raw = differentiator(raw)
    
    # detect where is the glitch
    glitch_index = [i for i in range(len(diff_raw)) if diff_raw[i] > threshold or diff_raw[i] < -1*threshold]
    
    # check if between indexes if the diff value is zero or not (no change in between)
    # then replace it with average
    i = 0
    output = raw
    missing_percentage = 0
    while i+1 < len(glitch_index): # 2<3
        start = glitch_index[i]+1 
        stop = glitch_index[i+1]
        if sum(diff_raw[start:stop]) == 0:
            replace_value = (raw[start-1]+raw[stop+1])/2
            replace_list = [replace_value for i in range(stop-start+1)]
            output = output[:start]+replace_list+output[stop+1:]
            assert len(raw) == len(output), "fix slicing list"
            missing_percentage = missing_percentage+len(replace_list)
        i = i+1 # 2
    missing_percentage = missing_percentage/len(raw)

    return output, missing_percentage, glitch_index


def remove_glitch(pd_signals,threshold=0.3):
    output = []
    miss_percent = []
    for elem in pd_signals:
        processed_signal, missing_percentage, _ = detect_glitch(elem,threshold=threshold)
        output.append(processed_signal)
        miss_percent.append(missing_percentage)
        
    return output, miss_percent


def differentiator(pd_signal):
    
    output = np.diff(pd_signal)
      
    return output

def gradient(pd_signal):
    
    output = np.gradient(pd_signal)
      
    return output

def savgol(pd_signal,window=15,polyorder=2):
    
    output = scipy.signal.savgol_filter(pd_signal,window,polyorder)
      
    return output


def plot_compare_sample(signal,processed_signal=None,ax=None,adjust=False,title=None):
    
    if ax is None:      
        plt.figure()
        ax = plt.axes()
        if title is not None:
            ax.set_title(title)
        ax.grid(True)
    
    avg = np.average(signal)    
    ax.plot(signal,'k')
    if processed_signal is not None:
        if adjust:
            processed_signal = [i+avg for i in processed_signal]
        ax.plot(processed_signal,'--r',linewidth=2)
    
    return

def plot_pd_before_after_df(ori,after,sample_idx=0):
    """
        plot original and processed signal overlapping for one sample
    ori:
        original signal
    after:
        processed signal
    sample_idx:
        index of sample you want to plot
    """
    primary_signal = ori.iloc[sample_idx].values
    secondary_signal = after.iloc[sample_idx].values
    plt.figure()
    fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(14, 12))
    axes.plot(primary_signal,label='original signal')
    axes.plot(secondary_signal,label='processed signal')
    axes.grid(True)
    axes.legend()
    fig.suptitle("Sample No.: "+str(sample_idx))
    return fig

def plot_pd_before_after(sample,processed_pd=None,ax=None,adjust=True,glitch_index=None,):
    
    if ax is None:      
        plt.figure()
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


def plot_pd_overlap(subjects=[1],fix_pd=True,threshold=0.3):
    face_dataset = utils.load_object("data_1_50_fixPD_Label_False.pkl")
    figs = []
    for subject_idx in subjects:
        # [0,70,140,...]
        start_idx = ((subject_idx*70)-70)
        # [69,139,209,...]
        stop_idx = (subject_idx*70)-1
        
        # prepare pd_signal numpy array
        pd_signals = []
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(14, 12))
        axes.grid(True)
        for i in range(start_idx,stop_idx+1):                         
            if fix_pd:
                output, _, _ = detect_glitch(face_dataset[i]['PD_avg_filtered'],threshold=threshold)                
            else:
                output = face_dataset[i]['PD_avg_filtered']
                 
            pd_signals.append(output) 
            axes.plot(output)
        
        fig.suptitle("Testsubject: " + str(subject_idx))
        figs.append(fig)
        print(subject_idx)
        
    return figs
        
def plot_pd_overlap_df(samples_df,subjects=[1,15,39]):   
    def color_label(label):
        color = 'black'
        if label == 5:
            color = 'yellow'
        elif label == 4:
            color = 'blue'
        elif label == 3:
            color = 'red'
        elif label == 2:
            color = 'green'
        elif label == 1:
            color = 'black'
        
#        scale = 1
#        target_scale = scale*((label-5)/4)
#        if -1.0 <= target_scale < -0.6:
#            color = 'black'
#        elif -0.6 <= target_scale < -0.2:
#            color = 'blue'
#        elif -0.2 <= target_scale < 0.2:
#            color = 'red'
#        elif 0.2 <= target_scale < 0.6:
#            color = 'green'
#        elif 0.6 <= target_scale <= 1:
#            color = 'yellow'
        return color
    
    figs = []
    for subject_idx in subjects:
        # get color from arousal
        try:
            arousal_pd = samples_df['arousal'].loc[subject_idx].apply(color_label)
        except:
            print(subject_idx)
        # cut samples per test subject to numpy array
        samples = samples_df.drop(columns=['arousal']).loc[subject_idx].values
        # get arousal list of color
        arousal_list = arousal_pd.tolist()
        # plotting
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(14, 12))
        axes.grid(True)
        try:
            for i in range(samples.shape[0]):    
                axes.plot(samples[i,:],color=arousal_list[i])
        except:
            print("array is  only one dimension")
            axes.plot(samples,color=arousal_list[0])
            
        fig.suptitle("Testsubject: " + str(subject_idx))
        figs.append(fig)
        print(subject_idx)
    return figs
        
        
        











