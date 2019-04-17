# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import utils

def generate_features_df(samples):
    """
    Imagine bell curve, skew left (tail to left,positive), skew right (tail to right,negative)
    mean follows tail, median stay with the bulk
    """
    if 'arousal' in samples.columns:
        arousal_col = samples['arousal']
        samples = samples.drop(columns=['arousal'])
    samples['mean'] = samples.mean(axis=1)
    samples['median'] = samples.median(axis=1)
    samples['max'] = samples.max(axis=1)
    samples['min'] = samples.min(axis=1)
    samples['skew'] = samples.skew(axis=1)
    if arousal_col is not None:
        samples['arousal'] = arousal_col  
    return samples

def select_and_clean(samples,norm=True,miss_percent=None,miss_threshold=0.4,label=None):
    """
        Select samples based on miss_percent with normalization as option.
    samples: 
        list of pd signals
    norm:
        (boolean)
    miss_percent:
        array of missing percentage 
    output_form : 
        'df' to return dataframe, otherwise will return list
    """
    output_df = pd.DataFrame()
    for subject_idx in range(1,51):
        # [0,70,140,...]
        start_idx = ((subject_idx*70)-70)
        # [70,140,210,...]
        stop_idx = (subject_idx*70)
        
        # create dataframe per test subject
        subject = samples[start_idx:stop_idx]
        subject = np.array(subject)
                
        # drop sample with has missing percent more than 60%
        subject_df = pd.DataFrame(subject)
        if label is not None:
            subject_df['arousal'] = label[start_idx:stop_idx] 
        if miss_percent is not None:            
            miss_column = miss_percent[start_idx:stop_idx]
            subject_df['missing_percent'] = miss_column
            subject_df = subject_df[subject_df.missing_percent <= miss_threshold]
            subject_df = subject_df.drop(columns=['missing_percent'])
        
        # normalization mix max
        
        if norm:
            if label is not None:
                arousal_col = subject_df['arousal']
                subject_df = subject_df.drop(columns=['arousal'])
            subject = subject_df.values
            min_val = subject.min()
            max_val = subject.max()
            subject = (subject-min_val)/(max_val-min_val)
            
        
        # convert numpy array to list and append it to output list
        
        subject = pd.DataFrame(subject)
        if label is not None:            
            subject['arousal'] = arousal_col.reset_index(drop=True)
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
    if 'arousal' in samples.columns:
        arousal_col = samples['arousal']
        samples = samples.drop(columns=['arousal'])
    samples = samples.drop(columns=[i for i in range(stop,samples.shape[1])]) 
    samples = samples.drop(columns=[i for i in range(start)])
    if arousal_col is not None:
        samples['arousal'] = arousal_col       
    return samples

def get_pds(pickle_file="data_1_50_fixPD_Label_False.pkl"):
    face_dataset = utils.load_object(pickle_file)
    array_samples = []
    for i in range(len(face_dataset)):
        array_samples.append(face_dataset[i]['PD_avg_filtered'])
    return array_samples

def get_depths(pickle_file="data_1_50_fixPD_Label_False.pkl"):
    face_dataset = utils.load_object(pickle_file)
    array_samples = []
    for i in range(len(face_dataset)):
        array_samples.append(face_dataset[i]['depth'])
    return array_samples

def get_illums(pickle_file="data_1_50_fixPD_Label_False.pkl"):
    face_dataset = utils.load_object(pickle_file)
    array_samples = []
    for i in range(len(face_dataset)):
        array_samples.append(face_dataset[i]['illuminance'])
    return array_samples

def get_arousal(pickle_file="data_1_50_fixPD_Label_False.pkl",fix=False):
    face_dataset = utils.load_object(pickle_file)
    array_samples = []
    
    def convert_to_label(SAM):
            scale = 1
            target_scale = scale*((SAM-5)/4)
            
            if -1.0 <= target_scale < -0.6:
                target_scale = 1
            elif -0.6 <= target_scale < -0.2:
                target_scale = 2
            elif -0.2 <= target_scale < 0.2:
                target_scale = 3
            elif 0.2 <= target_scale < 0.6:
                target_scale = 4
            elif 0.6 <= target_scale <= 1:
                target_scale = 5
            return target_scale
    for i in range(len(face_dataset)):
        sample = face_dataset[i]['arousal']
        if fix:
            sample = convert_to_label(sample)
        array_samples.append(sample)
    array_samples = np.array(array_samples)
    
    return array_samples

def get_valence(pickle_file="data_1_50_fixPD_Label_False.pkl"):
    face_dataset = utils.load_object(pickle_file)
    array_samples = []
    for i in range(len(face_dataset)):
        array_samples.append(face_dataset[i]['valence'])
    return array_samples



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

def remove_PLR(pd,illum,n,mu,norm=True):
    """
    n = length, mu = learning rate
    """
    d = np.array(pd)
    illum = np.array(illum)
    d_norm = d / np.linalg.norm(d)
    illum_norm = illum / np.linalg.norm(illum)
    illum_norm = 1.2*illum_norm
    illum_norm = illum_norm - np.mean(illum_norm) + np.mean(d_norm)
    if norm:
        y, e, w = my_lms(d_norm,illum_norm,n,mu)
    else:
        y, e, w = my_lms(d,illum,n,mu)
    return e


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
    figs = []
    for subject_idx in subjects:
        samples = samples_df.loc[subject_idx].values
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(14, 12))
        axes.grid(True)
        for i in range(samples.shape[0]):
            axes.plot(samples[i,:])
        fig.suptitle("Testsubject: " + str(subject_idx))
        figs.append(fig)
        print(subject_idx)
    return figs
        
        
        











