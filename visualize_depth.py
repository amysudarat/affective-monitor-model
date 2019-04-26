# -*- coding: utf-8 -*-
#%%
import pandas as pd
import numpy as np
import preprocessing.depth as pdep
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
path = "C:\\Users\\DSPLab\\Research\\ExperimentData"
#path = "E:\\Research\\ExperimentData"
n = 50
subjects = [i for i in range(1,n+1)]
#%%
depth_df = pdep.get_depth_df(path,subjects)

#%%
# save to pickle
utils.save_object(depth_df,'depth.pkl')

#%% load object
depth_df = utils.load_object('depth.pkl')

#%% get mean columns
depth_df = pdep.get_mean(depth_df)

#%% plot one subject
fig = depth_df[[25,'D25']].reset_index(drop=True).iplot(kind='scatter',mode='lines',
                                 title='depth',
                                 xTitle='picIndex', yTitle= 'Depth',
                                 asFigure=True)
plotly.offline.plot(fig)

#%%
# plot all test subject
fig = depth_df.reset_index(drop=True).iplot(kind='scatter',mode='lines',
                                 title='depth',
                                 xTitle='picIndex', yTitle= 'Depth',
                                 asFigure=True)
plotly.offline.plot(fig)
#%%







































