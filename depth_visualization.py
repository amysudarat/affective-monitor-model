# -*- coding: utf-8 -*-
#%%
import os
import pandas as pd
import numpy as np
import preprocessing.pd as ppd

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
n = 50
subjects = [i for i in range(1,n+1)]
#%%
# Loop through each Testsubject folder
depth_df = pd.DataFrame()
for i,elem in enumerate(subjects):
    filepath = os.path.join(path, "TestSubject"+str(elem)+"\\FAP.txt")
    depth_df_raw = pd.read_csv(filepath,header=1,delimiter=",",
                              quotechar=";",
                              usecols=['Depth','PicIndex'],
#                              index_col="PicIndex",
                              skipinitialspace=True)
    depth_df_raw = depth_df_raw.set_index('PicIndex')
    if i==0:
        depth_df = depth_df_raw
        depth_df.columns = [elem]
    else:
        depth_df[elem] = depth_df_raw

#    # create face sample loop through each picture index
##            self.face_df = face_df
#    for i in range(1,max(depth_df_raw.index.values)+1):
#        # number of rows per sample
#        start = (i*100)-100# 0,100,200,...
#        end = (i*100)  # 100,200,300,...
#        # group sequence of face point
#        depth_per_picture = depth_df_raw.loc[i]
#        depth_df = depth_df.append(depth_per_picture)
    

#%%
# check depth
ppd.plot_compare_sample(depth_df[4].reset_index(drop=True),title='Depth')
#%%
# plot all test subject
fig = depth_df.reset_index(drop=True).iplot(kind='scatter',mode='lines',
                                 title='depth',
                                 xTitle='picIndex', yTitle= 'Depth',
                                 asFigure=True)
plotly.offline.plot(fig)









