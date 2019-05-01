# -*- coding: utf-8 -*-

from model.dataset_class import AffectiveMonitorDataset
import pandas as pd
import preprocessing.pd as ppd
import preprocessing.depth as pdep
import preprocessing.valence as pval
import preprocessing.arousal as paro
import preprocessing.illum as pill
import preprocessing.pre_utils as pu
import utils

#%% face_dataset pickle
#n = 51
#subjects = [i for i in range(1,n+1)]
#
###############--------- Generate face_dataset getting raw PD ----------##########
##face_dataset = AffectiveMonitorDataset("C:\\Users\\dspcrew\\affective-monitor-model\\data",
##                                           subjects=subjects,
##                                           transform=ToTensor())
#    
#face_dataset = AffectiveMonitorDataset("C:\\Users\\DSPLab\\Research\\ExperimentData",
#                                       subjects=subjects,
#                                       fix_PD=False,
#                                       convert_label=False)
#    
##face_dataset = AffectiveMonitorDataset("E:\\Research\\ExperimentData",
##                                           subjects=subjects,
##                                           fix_PD=False,
##                                           convert_label=False)
#
## raw face dataset sandbox
##face_df = face_dataset.face_df
#
## save to pickle file
#utils.save_object(face_dataset, "data_1_51.pkl")

#%% arousal pickle
#path = "C:\\Users\\DSPLab\\Research\\ExperimentData"
path = "E:\\Research\\ExperimentData"
n = 51
subjects = [i for i in range(1,n+1)]
arousals = paro.get_arousal_df(path,subjects,source='subject_avg',fix=False,class_mode='default')

# save to pickle file
utils.save_object(arousals, "arousal.pkl")


#%% valence pickle
#path = "C:\\Users\\DSPLab\\Research\\ExperimentData"
##path = "E:\\Research\\ExperimentData"
#n = 51
#subjects = [i for i in range(1,n+1)]
#
#valence = pval.get_valence_df(path,subjects,source='iaps',fix=True,class_mode='three')
#
## save to pickle file
#utils.save_object(arousals, "valence.pkl")

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



