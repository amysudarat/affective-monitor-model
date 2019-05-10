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
faps_np_df = pfap.get_faps_np_df(pickle_file='data_1_51.pkl')

#%% find missing percentage
missing_percentage_list = pfap.get_missing_percentage(faps_np_df)
faps_filtered = pfap.faps_preprocessing(faps_np_df,
                                        aoi=[0,60],
                                        smooth=True,
                                        filter_miss=missing_percentage_list,
                                        fix_scaler='minmax')
samples = faps_filtered

#%% get peak
faps_feat_df = pfap.get_peak(faps_filtered,
                             window_width=10)

#%% get feature
faps_feat_df = pfap.get_feature(faps_feat_df)

#%% slide plot
import matplotlib.pyplot as plt

def faps_slide_plot(faps_feat_df,sbj):
    if sbj != 'all':
        faps_feat_df = faps_feat_df[faps_feat_df['sbj_idx']==sbj] 
    
    # prepare faps that will be plotted
    faps = faps_feat_df['faps'].tolist()
    peaks = faps_feat_df['peak_pos'].tolist()
    # slide show
    for fap, p in zip(faps,peaks):
        plt.figure()
        for col in range(fap.shape[1]):
            plt.plot(fap[:,col])
        plt.axvline(p,color='red',lw=1)
        plt.show()
        plt.waitforbuttonpress()
        plt.close()
    return

faps_slide_plot(faps_feat_df,6)

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

#%% save to pickle file
utils.save_object(samples,'faps_for_train.pkl')

