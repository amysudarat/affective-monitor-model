# -*- coding: utf-8 -*-
import utils
import pandas as pd
from preprocessing.iaps import iaps
import preprocessing.fap as pfap
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
                                        smooth=True,
                                        filter_miss=missing_percentage_list)

#%% check by visualize
plot_df = pd.DataFrame(faps_filtered.iloc[0]['faps'])
fig = plot_df.reset_index(drop=True).iplot(kind='scatter',mode='lines',
                                 title='FAPS',
                                 xTitle='frame', yTitle= 'FAP changes',
                                 asFigure=True)
plotly.offline.plot(fig)

#%% cut area of interest
faps_aoi = 

#%% save to pickle file
utils.save_object(samples,'faps_for_train.pkl')
