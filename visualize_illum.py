# -*- coding: utf-8 -*-
#%%
import pandas as pd
import numpy as np
import preprocessing.illum as pill
import utils

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
"""
Read CSV file and convert it FAC unit
"""
#path = "C:\\Users\\DSPLab\\Research\\ExperimentData"
path = "E:\\Research\\ExperimentData"
n = 50
subjects = [i for i in range(1,n+1)]
#%%
illum_df = pill.get_illum_df(path,subjects)

#%% load from pickle
illum_df = utils.load_object('illum.pkl')

#%% get mean
illum_df = pill.get_mean(illum_df)

#%% plot one subject
fig = illum_df[[25,'D25']].reset_index(drop=True).iplot(kind='scatter',mode='lines',
                                 title='depth',
                                 xTitle='picIndex', yTitle= 'Depth',
                                 asFigure=True)
plotly.offline.plot(fig)

#%%
# plot all test subject
fig = illum_df.reset_index(drop=True).iplot(kind='scatter',mode='lines',
                                 title='depth',
                                 xTitle='picIndex', yTitle= 'Depth',
                                 asFigure=True)
plotly.offline.plot(fig)