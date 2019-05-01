#%%
""" run pip install cufflinks plotly in anaconda prompt
run to see iplot option 'help(df.iplot)'
"""
import utils
import preprocessing.arousal as paro
import preprocessing.pre_utils as pu
from preprocessing.iaps import iaps
# Standard plotly imports
import plotly
import plotly.io as pio
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import plotly.figure_factory as ff
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

from sklearn.preprocessing import StandardScaler
#%% get data
data_df = utils.load_object('pd_for_train.pkl')
arousals = utils.load_object('arousal.pkl')

match_arousal_list = pu.match_with_sample(arousals,data_df['ori_idx'])
data_df = data_df.reset_index(drop=True)
data_df = data_df.drop(columns=['ori_idx'])
data_df['arousal'] = match_arousal_list

#%% box plot of mean grouped by arousal
fig = data_df.reset_index(drop=True).pivot(columns='arousal', values='mean').iplot(
        kind='box',
        title='illum alter and depth adjustment with beta=-3 , label=iaps',
        yTitle='mean',
        xTitle='label',
        asFigure=True)

plotly.offline.plot(fig)
#pio.write_image(fig,'fap_plot/test.pdf')
#%%
# box plot of mean grouped by median
fig = data_df.reset_index(drop=True).pivot(columns='arousal', values='median').iplot(
        kind='box',
        title='median',
        yTitle='median',
        xTitle='label',
        asFigure=True)

plotly.offline.plot(fig)
#%%
# box plot of mean grouped by std
fig = data_df.reset_index(drop=True).pivot(columns='arousal', values='std').iplot(
        kind='box',
        title='std',
        yTitle='std',
        xTitle='label',
        asFigure=True)

plotly.offline.plot(fig)
#%%
# box plot of mean grouped by max
fig = data_df.reset_index(drop=True).pivot(columns='arousal', values='max').iplot(
        kind='box',
        title='max',
        yTitle='max',
        xTitle='label',
        asFigure=True)

plotly.offline.plot(fig)
#%%
# box plot of mean grouped by min
fig = data_df.reset_index(drop=True).pivot(columns='arousal', values='min').iplot(
        kind='box',
        title='min',
        yTitle='min',
        xTitle='label',
        asFigure=True)

plotly.offline.plot(fig)
#%%
# box plot of mean grouped by skew
fig = data_df.reset_index(drop=True).pivot(columns='arousal', values='skew').iplot(
        kind='box',
        title='skew',
        yTitle='skew',
        xTitle='label',
        asFigure=True)

plotly.offline.plot(fig)
#%%
# scatter plot matrix
#fig = data_df[['mean','max','median','min','skew']].reset_index(drop=True).scatter_matrix(asFigure=True)
#plotly.offline.plot(fig)

fig = ff.create_scatterplotmatrix(
    data_df[['mean','max','median','min','skew','arousal']],
    diag='histogram',
    index='arousal',
    height=1000, width=1000)

plotly.offline.plot(fig)

#%%
iaps_class = iaps(r"C:\Users\DSPLab\Research\affective-monitor-model\preprocessing")
#iaps_class = iaps(r"E:\Research\affective-monitor-model\preprocessing\IAPSinfoFile_Final.txt")
iaps_df = iaps_class.iaps_df
pic_id_max_arousal = iaps_df.loc[iaps_df['arousal_m'].idxmax()]['pic_id']
pic_id_min_arousal = iaps_df.loc[iaps_df['arousal_m'].idxmin()]['pic_id']
list_max_idx = iaps_class.get_sample_idx(6550)
list_min_idx = iaps_class.get_sample_idx(1419)
final_list = list_max_idx + list_min_idx

# get samples
import preprocessing.pd as ppd
pd_signals = ppd.get_pds(pickle_file='data_1_51.pkl')
illum_mean_df = utils.load_object('illum_mean.pkl')
depth_mean_df = utils.load_object('depth_mean.pkl')

# remove glitch
pd_signals, _ = ppd.remove_glitch(pd_signals,threshold=0.2)
# find missing percentage list
missing_percentage = ppd.get_missing_percentage(pd_signals)
selected_samples = ppd.select_and_clean(pd_signals,norm=True,
                                        miss_percent=missing_percentage,
                                        miss_threshold=0.25,
                                        sd_detect_remove=True,
                                        smooth=False,
                                        fix_depth=None,
                                        fix_illum=None,
                                        fix_illum_alt=illum_mean_df,
                                        align=True,
                                        alpha=0.08,
                                        beta=4)

# slice to get area of interest
final_samples = selected_samples
samples_aoi = ppd.get_aoi_df(final_samples,start=20,stop=40)

# find stat of aoi signals
samples = ppd.generate_features_df(samples_aoi)
print('Total amount of samples: '+str(samples.shape))

# filter sample 
samples_filtered = samples[samples['ori_idx'].isin(final_list)]


arousals = utils.load_object('arousal.pkl')

match_arousal_list = pu.match_with_sample(arousals,samples_filtered['ori_idx'])
samples_filtered = samples_filtered.reset_index(drop=True)
samples_filtered = samples_filtered.drop(columns=['ori_idx'])
samples_filtered['arousal'] = match_arousal_list

fig = samples_filtered.reset_index(drop=True).pivot(columns='arousal', values='mean').iplot(
        kind='box',
        title='illum comp,Beta = 4, aoi=2-4, pic_id: 6550,1419, low illum,arou and high illum, arou',
        yTitle='mean',
        xTitle='label',
        asFigure=True)

plotly.offline.plot(fig)










