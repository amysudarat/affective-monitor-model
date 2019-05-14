# -*- coding: utf-8 -*-

from model.dataset_class import AffectiveMonitorDataset
import pandas as pd
import preprocessing.pd as ppd
import preprocessing.depth as pdep

import preprocessing.illum as pill
import preprocessing.pre_utils as pu
from preprocessing.iaps import iaps
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

#%% face_dataset pickle
n = 51
subjects = [i for i in range(1,n+1)]

##############--------- Generate face_dataset getting raw PD ----------##########
#face_dataset = AffectiveMonitorDataset("C:\\Users\\dspcrew\\affective-monitor-model\\data",
#                                           subjects=subjects,
#                                           transform=ToTensor())
    
face_dataset = AffectiveMonitorDataset("C:\\Users\\DSPLab\\Research\\ExperimentData",
                                       subjects=[88],
                                       fix_distance=True,
                                       fix_PD=False,
                                       convert_label=False)
#
#face_dataset = AffectiveMonitorDataset("E:\\Research\\ExperimentData",
#                                           subjects=subjects,
#                                           fix_distance=True,
#                                           fix_PD=False,
#                                           convert_label=False)

# raw face dataset sandbox
#face_df = face_dataset.face_df

# save to pickle file
utils.save_object(face_dataset, "data_88.pkl")

#%% create fap template for pickle
import preprocessing.fap as pfap
faps_np_df = pfap.get_faps_np_df(pickle_file='data_88.pkl')
# add label
label = ['surprise','calm1','smile1','laugh','sad1','disgust',
         'fear','angry','eye_widen','open_mouth1','very_sad',
         'move_eyebrow','calm2','move_forward','look_down',
         'nothing','turn_right','sad2', 'open_mouth2','smile2']

faps_np_df['label'] = label
#pfap.faps_slide_plot(faps_np_df,'all',peak_plot=False,label=True)
#pfap.faps_slide_subplot(faps_np_df,'all',label=True)
#%% preprocess
faps_tmp_df = pfap.faps_preprocessing_samples(faps_np_df,
                                              smooth=True,
                                              fix_scaler='standard',
                                              fix_scaler_mode='sbj',
                                              aoi=[0,100],
                                              sm_wid_len=21,
                                              sbj_num=88)
faps_tmp_df['label'] = label
#pfap.faps_slide_plot(faps_tmp_df,'all',peak_plot=False,label=True)
#pfap.faps_slide_subplot(faps_tmp_df,'all',label=True)
# add label
label = ['surprise','calm1','smile1','laugh','sad1','disgust',
         'fear','angry','eye_widen','open_mouth1','very_sad',
         'move_eyebrow','calm2','move_forward','look_down',
         'nothing','turn_right','sad2', 'open_mouth2','smile2']

faps_tmp_df['label'] = label



 #%% get features
faps_tmp_df = pfap.get_peak(faps_tmp_df,mode='peak')
faps_tmp_df = pfap.get_feature(faps_tmp_df)
pfap.faps_slide_plot(faps_tmp_df,'all',label=True)
pfap.dir_vector_slide_plot(faps_tmp_df,'all',label=True)
#%%  
# save template to pickle
utils.save_object(faps_tmp_df,'fap_template.pkl')

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
import pandas as pd

#%% check by visualize
title = 'laugh'
plot_df = pd.DataFrame(faps_tmp_df[faps_tmp_df['label']==title]['faps'].values.tolist()[0])
fig = plot_df.reset_index(drop=True).iplot(kind='scatter',mode='lines',
                                 title=title,
                                 xTitle='frame', yTitle= 'FAP changes',
                                 asFigure=True)
plotly.offline.plot(fig)

#%% arousal pickle
import preprocessing.arousal as paro
path = "C:\\Users\\DSPLab\\Research\\ExperimentData"
#path = "E:\\Research\\ExperimentData"
n = 51
subjects = [i for i in range(1,n+1)]
arousals = paro.get_arousal_df(path,subjects,source='subject_avg',fix=True,class_mode='two')

# save to pickle file
utils.save_object(arousals, "arousal.pkl")

#%% generate list of selected samples arousal
from sklearn.preprocessing import StandardScaler
path = "E:\\Research\\ExperimentData"
#path = "C:\\Users\\DSPLab\\Research\\ExperimentData"
n = 51
subjects = [i for i in range(1,n+1)]
arousals_iaps = paro.get_arousal_df(path,subjects,source='iaps',fix=False)
arousals_iaps_col = arousals_iaps.iloc[0:70].reset_index(drop=True)
arousals_avg = paro.get_arousal_df(path,subjects,source='subject_avg',fix=False,class_mode='default')
arousals_avg_col = arousals_avg.iloc[0:70].reset_index(drop=True)
# plot arousal from 
arousal_df = pd.DataFrame({'iaps':arousals_iaps_col['arousal'],'sbj_avg':arousals_avg_col['arousal']},
                            index=[i for i in range(70)])
arousal_df.index = [i for i in range(1,71)]
# find collide sample
arousal_df['diff'] = abs(arousal_df['iaps']-arousal_df['sbj_avg'])
# selection 1: high 7 samples, low 7 samples
#collide_df = arousal_df[((arousal_df['diff']<=0.35)&(arousal_df['sbj_avg']<4.1)) | ((arousal_df['diff']<=0.6)&(arousal_df['sbj_avg']>=6))]
collide_df = arousal_df[((arousal_df['diff']<=0.455)&(arousal_df['sbj_avg']<4)) | ((arousal_df['diff']<=0.6)&(arousal_df['sbj_avg']>=5.5))]

sc = StandardScaler()
collide_scaled = sc.fit_transform(collide_df[['iaps','sbj_avg']])
collide_scaled_df = pd.DataFrame(collide_scaled,columns=collide_df[['iaps','sbj_avg']].columns,index=collide_df.index)
text = collide_scaled_df.index.tolist()
text = [str(i) for i in text]

# plot arousal from 
fig = collide_scaled_df[['iaps','sbj_avg']].reset_index(drop=True).iplot(kind='scatter',mode='lines+markers+text',
                                 title='compare selected samples arousals',
                                 text=text,
                                 xTitle='picIndex', yTitle= 'arousal rating',
                                 asFigure=True)
plotly.offline.plot(fig,filename='selected_label.html')

# generate picture id
#iaps_class = iaps(r"C:\Users\DSPLab\Research\affective-monitor-model\preprocessing")
iaps_class = iaps(r"E:\Research\affective-monitor-model\preprocessing")
selected_sample_idx = collide_scaled_df.index.tolist()
selected_sample_idx = [i-1 for i in selected_sample_idx]
selected_sample_id = [iaps_class.get_pic_id(i) for i in selected_sample_idx]
# get sample idx from list of pic_id
sample_idx_list = []
for pic_id in selected_sample_id:
    sample_idx_list = sample_idx_list + iaps_class.get_sample_idx(pic_id)

utils.save_object(sample_idx_list,'selected_idx_list_arousal.pkl')

#%% valence pickle
import preprocessing.valence as pval
#path = "C:\\Users\\DSPLab\\Research\\ExperimentData"
path = "E:\\Research\\ExperimentData"
n = 51
subjects = [i for i in range(1,n+1)]

valence = pval.get_valence_df(path,subjects,source='subject_avg',fix=True,class_mode='two')

# save to pickle file
utils.save_object(valence, "valence.pkl")

#%% generate selected list for valence
from sklearn.preprocessing import StandardScaler
path = "E:\\Research\\ExperimentData"
#path = "C:\\Users\\DSPLab\\Research\\ExperimentData"
n = 51
subjects = [i for i in range(1,n+1)]
valence_iaps = pval.get_valence_df(path,subjects,source='iaps',fix=False)
valence_iaps_col = valence_iaps.iloc[0:70].reset_index(drop=True)
valence_avg = pval.get_valence_df(path,subjects,source='subject_avg',fix=False,class_mode='default')
valence_avg_col = valence_avg.iloc[0:70].reset_index(drop=True)
# plot arousal from 
valence_df = pd.DataFrame({'iaps':valence_iaps_col['valence'],'sbj_avg':valence_avg_col['valence']},
                            index=[i for i in range(70)])
valence_df.index = [i for i in range(1,71)]
# find collide sample
valence_df['diff'] = abs(valence_df['iaps']-valence_df['sbj_avg'])
# selection 1: high 7 samples, low 7 samples
#collide_df = arousal_df[((arousal_df['diff']<=0.35)&(arousal_df['sbj_avg']<4.1)) | ((arousal_df['diff']<=0.6)&(arousal_df['sbj_avg']>=6))]
collide_df = valence_df[((valence_df['diff']<=0.455)&(valence_df['sbj_avg']<4)) | ((valence_df['diff']<=0.6)&(valence_df['sbj_avg']>=5.5))]

sc = StandardScaler()
collide_scaled = sc.fit_transform(collide_df[['iaps','sbj_avg']])
collide_scaled_df = pd.DataFrame(collide_scaled,columns=collide_df[['iaps','sbj_avg']].columns,index=collide_df.index)
text = collide_scaled_df.index.tolist()
text = [str(i) for i in text]

# plot arousal from 
fig = collide_scaled_df[['iaps','sbj_avg']].reset_index(drop=True).iplot(kind='scatter',mode='lines+markers+text',
                                 title='compare selected samples valence',
                                 text=text,
                                 xTitle='picIndex', yTitle= 'arousal rating',
                                 asFigure=True)
plotly.offline.plot(fig,filename='selected_label.html')

# generate picture id
#iaps_class = iaps(r"C:\Users\DSPLab\Research\affective-monitor-model\preprocessing")
iaps_class = iaps(r"E:\Research\affective-monitor-model\preprocessing")
selected_sample_idx = collide_scaled_df.index.tolist()
selected_sample_idx = [i-1 for i in selected_sample_idx]
selected_sample_id = [iaps_class.get_pic_id(i) for i in selected_sample_idx]
# get sample idx from list of pic_id
sample_idx_list = []
for pic_id in selected_sample_id:
    sample_idx_list = sample_idx_list + iaps_class.get_sample_idx(pic_id)

utils.save_object(sample_idx_list,'selected_idx_list_valence.pkl')
#%% Depth
#path = "C:\\Users\\DSPLab\\Research\\ExperimentData"
##path = "E:\\Research\\ExperimentData"
#n = 51
#subjects = [i for i in range(1,n+1)]
#
#depth_df = pdep.get_depth_df(path,subjects)
#min_depth_df = pdep.get_min_depth(depth_df)
#depth_df = pdep.get_mean(depth_df)
#
## save to pickle
#utils.save_object(depth_df,'depth.pkl')
#
## generate mean columns for samples
#depth_df = depth_df.drop(columns=subjects)
#depth_df.columns = subjects
#depth_mean = []
#for col in range(1,len(depth_df.columns)+1):
#    for row in range(1,depth_df.index.max()+1):    
#        depth_mean.append(depth_df.loc[row][col].values.tolist()[0])
#depth_mean_df = pd.DataFrame(depth_mean)
#depth_mean_df.columns = ['mean_per_frame']
#depth_mean_df['min'] = min_depth_df
#
## save depth mean to pickle
#utils.save_object(depth_mean_df,'depth_mean.pkl')

#%% Illuminance
#path = "C:\\Users\\DSPLab\\Research\\ExperimentData"
##path = "E:\\Research\\ExperimentData"
#n = 51
#subjects = [i for i in range(1,n+1)]
#
#illum_df = pill.get_illum_df(path,subjects)
## save to pickle
#utils.save_object(illum_df,'illum.pkl')
#
## generate mean columns for samples
#illum_mean_subject_list = pill.get_mean_per_subject(illum_df)
#
#illum_df = pill.get_mean(illum_df)
#
#illum_df = illum_df.drop(columns=subjects)
#illum_df.columns = subjects
#illum_mean = []
#
## prepare illum recorded data
#filepath = r"C:\Users\DSPLab\Research\affective-monitor-model\preprocessing\lux_record_manual.csv"
##filepath = r"E:\Research\affective-monitor-model\preprocessing\lux_record_manual.csv"
#illum_manual_df = pill.get_illum_lux_manual(filepath)
#ill_lux_manual_list = illum_manual_df['illum_lux'].values.reshape(-1).tolist()
#ill_gimp_manual_list = illum_manual_df['illum_gimp'].values.reshape(-1).tolist()
#ill_lux_rec_col = []
#ill_gimp_rec_col = []
#for i in range(len(subjects)):
#    ill_lux_rec_col = ill_lux_rec_col+ill_lux_manual_list
#    ill_gimp_rec_col = ill_gimp_rec_col+ill_gimp_manual_list
#
#for col in range(1,len(illum_df.columns)+1):
#    for row in range(1,illum_df.index.max()+1):
#        illum_mean.append(illum_df.loc[row][col].values.tolist()[0])
#illum_mean_df = pd.DataFrame(illum_mean)
#illum_mean_df.columns = ['mean_per_frame']
#illum_mean_df['mean_per_subject'] = illum_mean_subject_list
#illum_mean_df['illum_rec'] = ill_lux_rec_col
#illum_mean_df['illum_gimp_rec'] = ill_gimp_rec_col
#
## save depth mean to pickle
#utils.save_object(illum_mean_df,'illum_mean.pkl')

#%%



