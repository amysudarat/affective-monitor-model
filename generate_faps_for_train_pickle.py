# -*- coding: utf-8 -*-
import utils
import pandas as pd
from preprocessing.iaps import iaps
import preprocessing.fap as pfap

#%% get data
#path = "C:\\Users\\DSPLab\\Research\\ExperimentData"
path = "E:\\Research\\ExperimentData"
n = 51
subjects = [i for i in range(1,n+1)]

#faps_df = pfap.get_faps_df(pickle_file='data_1_51.pkl')
faps_np_df = pfap.get_faps_np_df(pickle_file='data_1_51_fix_distance.pkl')

#%% find missing percentage
missing_percentage_list = pfap.get_missing_percentage(faps_np_df)
faps_filtered = pfap.faps_preprocessing(faps_np_df,
                                        aoi=[0,50],
                                        smooth=True,
                                        filter_miss=missing_percentage_list,
                                        fix_scaler='standard')

#%% get peak
faps_peak_df = pfap.get_peak(faps_filtered,
                             mode='peak')

#%% get feature
faps_peak_sel_df = pfap.get_feature(faps_peak_df)

#%% save to pickle
samples = faps_peak_sel_df
utils.save_object(samples,'faps_for_train.pkl')

#%% slide plot
import matplotlib.pyplot as plt

def faps_slide_plot(faps_feat_df,sbj,label=False):
    if sbj != 'all':
        faps_feat_df = faps_feat_df[faps_feat_df['sbj_idx']==sbj] 
    
    # prepare faps that will be plotted
    faps = faps_feat_df['faps'].tolist()
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
        plt.figure()
        try:
            for col in range(faps[i].shape[1]):
                plt.plot(faps[i][:,col])
        except:
            plt.plot(faps[i])
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
        plt.show()
        plt.waitforbuttonpress()
        plt.close()
    return

faps_slide_plot(faps_peak_df,31)  

#%% visualize sandbox
# generate picture id
iaps_class = iaps(r"C:\Users\DSPLab\Research\affective-monitor-model\preprocessing")
#iaps_class = iaps(r"E:\Research\affective-monitor-model\preprocessing")
samples_idx = iaps_class.get_sample_idx(2070)

# get samples based on pic_id
faps_selected = faps_filtered[faps_filtered['ori_idx'].isin(samples_idx)]


#%%
# Standard plotly imports
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import plotly.figure_factory as ff
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)


#%% check by visualize
plot_df = pd.DataFrame(faps_filtered.loc[3506]['faps'])
fig = plot_df.reset_index(drop=True).iplot(kind='scatter',mode='lines',
                                 title='FAPS',
                                 xTitle='frame', yTitle= 'FAP changes',
                                 asFigure=True)
plotly.offline.plot(fig)




