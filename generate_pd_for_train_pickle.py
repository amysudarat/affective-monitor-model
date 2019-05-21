# -*- coding: utf-8 -*-
import preprocessing.pd as ppd
import utils
#import pandas as pd
#from preprocessing.iaps import iaps
#import preprocessing.illum as pill
#import preprocessing.depth as pdep

#%% get samples
pd_signals = ppd.get_pds(pickle_file='data_1_51.pkl')
#illum_mean_df = utils.load_object('illum_mean.pkl')
#depth_mean_df = utils.load_object('depth_mean.pkl')
subjects = [i for i in range(1,52)]
pd_df = ppd.get_raw_pd_df(pd_signals,subjects)
#%%
#ppd.pd_plot_pause(pd_df,51,ylim=[1,4])
         
#%% identify PQR
pd_filt_df = ppd.preprocessing_pd(pd_df,
                             aoi=40,
                             loc_artf='mad_filter',
                             diff_threshold=0.1,
                             n_mad=2,
                             interpolate=True,
                             miss_threshold=0.25,
                             norm=True)

#%% plot slide show
#ppd.pd_plot_pause(pd_filt_df,51,ylim=[-1,2])
      
#%% illum compensation
import preprocessing.illum as pill
import preprocessing.pre_utils as pu
filepath = r"E:\Research\affective-monitor-model\preprocessing\lux_record_manual.csv"
#filepath = r"C:\Users\DSPLab\Research\affective-monitor-model\preprocessing\lux_record_manual.csv"
ill_list = pill.get_illum_lux_manual(filepath)
ill_list = ill_list['illum_gimp'].tolist()
ill_list = pu.match_illum_with_sample(pd_filt_df,ill_list)

#%% get PQR
pd_pqr_df = ppd.get_pqr_feature(pd_filt_df,
                                smooth=True,
                                filt_corrupt=False,
                                illum_comp=ill_list)

#%% get stat features
pd_pqr_df = ppd.generate_features_df(pd_pqr_df)
import matplotlib.pyplot as plt
plt.hist(pd_pqr_df['area_ql'])
#%% visualize pqr
#ppd.plot_pqr_slideshow(pd_pqr_df,42,smooth=True)

#%% data selection
#sel_pic_list = utils.load_object('selected_idx_list.pkl')
#pd_pqr_df = pd_pqr_df[pd_pqr_df['ori_idx'].isin(sel_pic_list)]

#%% data selection
import preprocessing.pre_utils as pu
import pandas as pd
arousals = utils.load_object('arousal.pkl')
arousals_list = arousals['arousal'].tolist()

pd_sel_df = pd_pqr_df.copy()
pd_sel_df = pu.match_label_with_sample(pd_sel_df,arousals_list)

pd_ar_df = pd_sel_df[((pd_sel_df['label']==1) & (pd_sel_df['area_ql']>20))]
pd_nar_df = pd_sel_df[((pd_sel_df['label']==2) & (pd_sel_df['area_ql']<=20))]
pd_nar_df = pd_nar_df.sample(pd_ar_df.shape[0])

samples = pd.concat([pd_ar_df,pd_nar_df],ignore_index=True)
samples = samples.sample(frac=1)

# plot slide show of data
#ppd.plot_pqr_slideshow(samples,'all',smooth=False,label=samples['label'].tolist())

# drop label before save it to pickle
samples = samples.drop('label',axis=1)

# save pickle
utils.save_object(samples,'pd_for_train.pkl')
#utils.save_object(pd_pqr_df,'pd_for_test.pkl')

#%%
utils.save_object(pd_pqr_df,'pd_for_train.pkl')

#%%
# Standard plotly imports
#import plotly
#import plotly.plotly as py
#import plotly.graph_objs as go
#from plotly.offline import iplot, init_notebook_mode
#import plotly.figure_factory as ff
## Using plotly + cufflinks in offline mode
#import cufflinks
#cufflinks.go_offline(connected=True)
#init_notebook_mode(connected=True)

#%%
# visualize pd
#fig = pd_df.loc[1].reset_index(drop=True).transpose().iplot(kind='scatter',mode='lines',
#                                 title='pd_df',
#                                 xTitle='frame', yTitle= 'pd',
#                                 asFigure=True)
#plotly.offline.plot(fig)


#%%
# remove glitch
#pd_signals, _ = ppd.remove_glitch(pd_signals,threshold=0.2)


#%% prepare data for dr.b
#dr_b_data_df = pd_filt_df.loc[51]
#dr_b_data_df.reset_index(drop=True).to_csv('pd_data_after_cleaning.csv',index=False,header=False)
#%%
# find missing percentage list
#missing_percentage = ppd.get_missing_percentage(pd_signals)
#selected_samples = ppd.select_and_clean(pd_signals,norm=True,
#                                        miss_percent=missing_percentage,
#                                        miss_threshold=0.25,
#                                        sd_detect_remove=True,
#                                        smooth=False,
#                                        fix_depth=None,
#                                        fix_illum=None,
#                                        fix_illum_alt=None,
#                                        align=True,
#                                        alpha=0.08,
#                                        beta=2)

#%% prepare data for dr b
#dr_b_data_df = selected_samples.drop('ori_idx',axis=1).loc[51]
#ppd.pd_plot_pause(selected_samples,51,ylim=[0,1])
#dr_b_data_df.reset_index(drop=True).to_csv('dr_b_pd_data_after_norm_remove_corrupted_samples.csv',index=False,header=False)

#%%
# slice to get area of interest
#final_samples = selected_samples
#samples_aoi = ppd.get_aoi_df(final_samples,start=20,stop=40)
#
## find stat of aoi signals
#samples = ppd.generate_features_df(samples_aoi)
#print('Total amount of samples: '+str(samples.shape))
#
## save to pickle
#utils.save_object(samples,'pd_for_train.pkl')
#%% visualize pd depth adjusted
#plot_df = samples_aoi.copy()
#plot_df = plot_df.loc[51].reset_index(drop=True).drop('ori_idx',axis=1).transpose()
#fig = plot_df.reset_index(drop=True).iplot(kind='scatter',mode='lines',
#                                 title='illum_adjust',
#                                 xTitle='sample', yTitle= 'pd (normalized scale)',
#                                 asFigure=True)
#plotly.offline.plot(fig)

#%% prepare data for dr.b
#pd_df = ppd.get_raw_pd_df(pd_signals,subjects)
#dr_b_data_df = pd_df.loc[51]
#dr_b_data_df.reset_index(drop=True).to_csv('dr_b_pd_data.csv',index=False,header=False)

#%% visualize illum and depth mean
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
#alpha = 0.1
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