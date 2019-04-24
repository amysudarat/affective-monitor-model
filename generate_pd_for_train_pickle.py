# -*- coding: utf-8 -*-
import preprocessing.pd as ppd
import utils
import pandas as pd
from preprocessing.iaps import iaps
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
iaps_class = iaps(r"C:\Users\DSPLab\Research\affective-monitor-model\preprocessing\IAPSinfoFile_Final.txt")
#iaps_class = iaps(r"E:\Research\affective-monitor-model\preprocessing\IAPSinfoFile_Final.txt")
sample_list_from_pic_id = iaps_class.get_sample_idx(2141)
#%%
# get samples
pd_signals = ppd.get_pds()
arousals = ppd.get_arousal(fix=True,class_mode='two')
illum_signals = ppd.get_illums()

#%%
# visualize arousal
data_df = pd.DataFrame(arousals)
data_df.columns = ['arousal']
fig = data_df['arousal'].iplot(kind='hist',histnorm='percent',
                                 title='arousal distribution',
                                 xTitle='Label', yTitle= '% of each group',
                                 asFigure=True)
plotly.offline.plot(fig)


#%%
# remove glitch
pd_signals, _ = ppd.remove_glitch(pd_signals,threshold=0.2)
# find missing percentage list
missing_percentage = ppd.get_missing_percentage(pd_signals)
selected_samples = ppd.select_and_clean(pd_signals,norm=True,
                                        miss_percent=missing_percentage,
                                        miss_threshold=0.25,
                                        label=arousals,
                                        sd_detect_remove=True,
                                        align=True)


#%%
# visualize arousal
fig = selected_samples['arousal'].iplot(kind='hist',histnorm='percent',
                                 title='arousal distribution',
                                 xTitle='Label', yTitle= '% of each group',
                                 asFigure=True)
plotly.offline.plot(fig)


#%%
# set that seems work: n=10, mu=0.00000085
# set that seems work: n=5, mu=0.00000095
remove_PLR = True
if remove_PLR:
    illum_select_df = selected_samples.copy()
    illum_select_df['idx'] = illum_select_df.reset_index(drop=True).index
    illum_select_list = illum_select_df[illum_select_df['ori_idx_row'].isin(sample_list_from_pic_id)]['idx'].tolist()
                            
    final_samples, _, _ = ppd.remove_PLR(selected_samples,
                                               illum_signals,
                                               n=5,
                                               mu=0.0000015,
#                                               showFigures=illum_select_list,
                                               showFigures=None,
                                               arousal_col=True)
else:
    final_samples = selected_samples.drop(columns=['ori_idx_row'])
    
#%%
# slice to get area of interest
samples_aoi = ppd.get_aoi_df(final_samples,start=20,stop=70)
#%%
# plot figures to pdf
figs = ppd.plot_pd_overlap_df(samples_aoi,subjects=[i for i in range(1,51)])

#%%
# find stat of aoi signals
samples = ppd.generate_features_df(samples_aoi)
print('Total amount of samples: '+str(samples.shape))
#%%
# save to pickle
utils.save_object(samples,'pd_for_train.pkl')
#%%