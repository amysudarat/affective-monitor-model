# -*- coding: utf-8 -*-
import preprocessing.pd as ppd
import utils
import pandas as pd
from preprocessing.iaps import iaps
import preprocessing.illum as pill
import preprocessing.depth as pdep
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

#%%
##iaps_class = iaps(r"C:\Users\DSPLab\Research\affective-monitor-model\preprocessing\IAPSinfoFile_Final.txt")
#iaps_class = iaps(r"E:\Research\affective-monitor-model\preprocessing\IAPSinfoFile_Final.txt")
#sample_list_from_pic_id = iaps_class.get_sample_idx(2141)

#%%
# get samples
pd_signals = ppd.get_pds()
illum_mean_df = utils.load_object('illum_mean.pkl')
depth_mean_df = utils.load_object('depth_mean.pkl')

##%% visualize illum and depth mean
#fig = depth_mean_df.reset_index(drop=True).iplot(kind='scatter',mode='lines',
#                                 title='depth_mean_df',
#                                 xTitle='sample', yTitle= 'mean of depth per frame and min per subject',
#                                 asFigure=True)
#plotly.offline.plot(fig)
#
#fig = illum_mean_df.reset_index(drop=True).iplot(kind='scatter',mode='lines',
#                                 title='illum_mean_df',
#                                 xTitle='sample', yTitle= 'mean of illum per frame and mean per subject',
#                                 asFigure=True)
#plotly.offline.plot(fig)
#
#
##%% visualize the scale of change
#
#depth_adjust = depth_mean_df['mean_per_frame']/depth_mean_df['min']
#alpha = 0.05
#illum_adjust = alpha*(illum_mean_df['mean_per_subject']-illum_mean_df['mean_per_frame'])
#
#fig = depth_adjust.reset_index(drop=True).iplot(kind='scatter',mode='lines',
#                                 title='depth_adjust',
#                                 xTitle='sample', yTitle= 'depth per frame / min depth per subject',
#                                 asFigure=True)
#plotly.offline.plot(fig)
#
#fig = illum_adjust.reset_index(drop=True).iplot(kind='scatter',mode='lines',
#                                 title='illum_adjust',
#                                 xTitle='sample', yTitle= 'mean illum per subject - illum per frame',
#                                 asFigure=True)
#plotly.offline.plot(fig)

#%%
# remove glitch
pd_signals, _ = ppd.remove_glitch(pd_signals,threshold=0.2)
# find missing percentage list
missing_percentage = ppd.get_missing_percentage(pd_signals)
selected_samples = ppd.select_and_clean(pd_signals,norm=True,
                                        miss_percent=missing_percentage,
                                        miss_threshold=0.25,
                                        sd_detect_remove=True,
                                        fix_depth=None,
                                        fix_illum=illum_mean_df,
                                        align=True,
                                        alpha=0.03)

#%%
## set that seems work: n=10, mu=0.00000085
## set that seems work: n=5, mu=0.00000095
#remove_PLR = False
#if remove_PLR:
#    illum_select_df = selected_samples.copy()
#    illum_select_df['idx'] = illum_select_df.reset_index(drop=True).index
#    illum_select_list = illum_select_df[illum_select_df['ori_idx'].isin(sample_list_from_pic_id)]['idx'].tolist()
#                            
#    final_samples, _, _ = ppd.remove_PLR(selected_samples,
#                                               illum_signals,
#                                               n=5,
#                                               mu=0.0000015,
##                                               showFigures=illum_select_list,
#                                               showFigures=None,
#                                               arousal_col=True)
#else:
#    final_samples = selected_samples
    
#%%
# slice to get area of interest
final_samples = selected_samples
samples_aoi = ppd.get_aoi_df(final_samples,start=20,stop=70)
#%% visualize pd
plot_df = samples_aoi.copy()
plot_df = plot_df.loc[1].reset_index(drop=True).drop('ori_idx',axis=1).transpose()
fig = plot_df.reset_index(drop=True).iplot(kind='scatter',mode='lines',
                                 title='depth_adjust',
                                 xTitle='sample', yTitle= 'depth per frame / min depth per subject',
                                 asFigure=True)
plotly.offline.plot(fig)

#%%
# find stat of aoi signals
samples = ppd.generate_features_df(samples_aoi)
print('Total amount of samples: '+str(samples.shape))
#%%
# save to pickle
utils.save_object(samples,'pd_for_train.pkl')
#%%