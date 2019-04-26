#%%
""" run pip install cufflinks plotly in anaconda prompt
run to see iplot option 'help(df.iplot)'
"""
import utils
import preprocessing.arousal as paro
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

from sklearn.preprocessing import StandardScaler
#%% get data
data_df = utils.load_object('pd_for_train.pkl')
arousals = utils.load_object('arousal.pkl')

match_arousal_list = paro.match_with_sample(arousals,data_df['ori_idx'])
data_df = data_df.reset_index(drop=True)
data_df = data_df.drop(columns=['ori_idx'])
data_df['arousal'] = match_arousal_list

#%% box plot of mean grouped by arousal
fig = data_df.reset_index(drop=True).pivot(columns='arousal', values='mean').iplot(
        kind='box',
        title='mean with subject rating and without remove PLR',
        yTitle='mean',
        xTitle='label',
        asFigure=True)

plotly.offline.plot(fig)
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














