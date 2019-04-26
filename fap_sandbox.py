#%%

import preprocessing.fap as pfap
import preprocessing.valence as pval
import utils
import numpy as np
import pandas as pd

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
from sklearn.preprocessing import StandardScaler

#%%
#path = "C:\\Users\\DSPLab\\Research\\ExperimentData"
path = "E:\\Research\\ExperimentData"
n = 50
subjects = [i for i in range(1,n+1)]

#%% get data
faps_df = pfap.get_faps()
valence_df = pval.get_valence_df(path,subjects,fix=True,class_mode='default')

#%%
# save to pickle
utils.save_object(faps_df,'fap.pkl')
utils.save_object(valence_df,'valence.pkl')

#%% in case we already save pickle load it from there
faps_df = utils.load_object('fap.pkl')
valence_df = utils.load_object('valence.pkl')

#%% standard normalization to put all units in the same scale
# after transformation: each column will have mean=0 and std=1 
# now we set with_std = False so we didn't scale std

sample = 4
# plot all test subject
fig = faps_df.loc[sample].reset_index(drop=True).iplot(kind='scatter',mode='lines',
                                 title='FAP through time before standard scaling',
                                 xTitle='picIndex', yTitle= 'Depth',
                                 asFigure=True)
plotly.offline.plot(fig)


scaler = StandardScaler(with_std=False)
scaler.fit(faps_df.loc[sample])
faps_scaled = scaler.transform(faps_df.loc[sample])

faps_scaled_df = pd.DataFrame(faps_scaled,columns=faps_df.columns)



# plot all test subject
fig = faps_scaled_df.iplot(kind='scatter',mode='lines',
                                 title='FAP through time after standard scaling',
                                 xTitle='picIndex', yTitle= 'Depth',
                                 asFigure=True)
plotly.offline.plot(fig)


#%% checkout feature scaling (minmax scalar and absscalar)



#%%
face_dataset = utils.load_object("data_1_50_fixPD_Label_False.pkl")
# 583 is a good one
#sample_idx = 583,640
sample_idx = 583
fap_signal = face_dataset[sample_idx]['faceFAP']

# convert to numpy
fap_np = np.array(fap_signal)

# median filter
processed_fap = pfap.median_filter(fap_np,window=50)
pfap.plot_FAP_temporal(face_dataset[sample_idx],sample_idx,processed_fap)

# smooth curve
processed_fap = pfap.savgol_fap(fap_np,31,5)
pfap.plot_FAP_temporal(face_dataset[sample_idx],sample_idx,processed_fap)