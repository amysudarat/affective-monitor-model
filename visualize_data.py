# -*- coding: utf-8 -*-
import utils
import numpy as np
import pandas as pd
import preprocessing.illum as pill
import preprocessing.arousal as paro
from preprocessing.iaps import iaps
from sklearn.preprocessing import StandardScaler

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
#filepath = r"E:\Research\affective-monitor-model\preprocessing\illum_lux_meter_record.txt"
filepath = r"C:\Users\DSPLab\Research\affective-monitor-model\preprocessing\illum_lux_meter_record.txt"
illum_lux_df = pill.get_illum_lux(filepath)
#filepath = r"E:\Research\affective-monitor-model\preprocessing\lux_record_manual.csv"
filepath = r"C:\Users\DSPLab\Research\affective-monitor-model\preprocessing\lux_record_manual.csv"
ill_lux_manual_df = pill.get_illum_lux_manual(filepath)

#%% find mean of illum
illum_lux_np = illum_lux_df.values
illum_lux_np = illum_lux_np[5:]
illum_lux_np = illum_lux_np[:1750]
mean_total=[]
mean_pic = []
mean_warn = []
for i in range(70):
    mean_warn.append(np.mean(illum_lux_np[:5]))
    mean_total.append(np.mean(illum_lux_np[:5]))
    illum_lux_np = illum_lux_np[5:]
    mean_pic.append(np.mean(illum_lux_np[:15]))
    mean_total.append(np.mean(illum_lux_np[:15]))
    illum_lux_np = illum_lux_np[15:]

#%% find std of average subject rating
#path = "E:\\Research\\ExperimentData"
path = "C:\\Users\\DSPLab\\Research\\ExperimentData"
n = 51
subjects = [i for i in range(1,n+1)]
arousals_iaps = paro.get_arousal_df(path,subjects,source='iaps',fix=False)
arousals_iaps_col = arousals_iaps.iloc[0:70].reset_index(drop=True)
arousals_avg = paro.get_arousal_df(path,subjects,source='subject_avg',fix=False,class_mode='default')
arousals_avg_col = arousals_avg.iloc[0:70].reset_index(drop=True)
arousals_df = paro.get_arousal_df(path,subjects,source='subject',fix=False,class_mode='default')
arousal_sbj_std = []
for sbj in range(1,71):
    arousal_sbj_std.append(round(arousals_df.loc[sbj].std().values.tolist()[0],2))

# plot arousal from 
arousal_df = pd.DataFrame({'iaps':arousals_iaps_col['arousal'],'sbj_avg':arousals_avg_col['arousal']},
                            index=[i for i in range(70)])
arousal_df.index = [i for i in range(1,71)]
fig = arousal_df.reset_index(drop=True).iplot(kind='scatter',mode='lines+markers',
                                 title='compare arousals',
                                 xTitle='picIndex', yTitle= 'arousal rating',
                                 asFigure=True)
plotly.offline.plot(fig)

#%% find collide sample
arousal_df['diff'] = abs(arousal_df['iaps']-arousal_df['sbj_avg'])
arousal_df['illum_gimp'] = ill_lux_manual_df['illum_gimp'].values.tolist()
arousal_df['illum_lux'] = ill_lux_manual_df['illum_lux'].values.tolist()
# selection 1: high 7 samples, low 7 samples
#collide_df = arousal_df[((arousal_df['diff']<=0.35)&(arousal_df['sbj_avg']<4.1)) | ((arousal_df['diff']<=0.6)&(arousal_df['sbj_avg']>=6))]
collide_df = arousal_df[((arousal_df['diff']<=0.35)&(arousal_df['sbj_avg']<4.1)) | ((arousal_df['diff']<=0.6)&(arousal_df['sbj_avg']>=6))]

sc = StandardScaler()
collide_scaled = sc.fit_transform(collide_df[['iaps','sbj_avg','illum_gimp']])
collide_scaled_df = pd.DataFrame(collide_scaled,columns=collide_df[['iaps','sbj_avg','illum_gimp']].columns,index=collide_df.index)
text = collide_scaled_df.index.tolist()
text = [str(i) for i in text]

# plot arousal from 
fig = collide_scaled_df[['iaps','sbj_avg','illum_gimp']].reset_index(drop=True).iplot(kind='scatter',mode='lines+markers+text',
                                 title='compare selected samples arousals',
                                 text=text,
                                 xTitle='picIndex', yTitle= 'arousal rating',
                                 asFigure=True)
plotly.offline.plot(fig)

# generate picture id
iaps_class = iaps(r"C:\Users\DSPLab\Research\affective-monitor-model\preprocessing")
selected_sample_idx = collide_scaled_df.index.tolist()
selected_sample_idx = [i-1 for i in selected_sample_idx]
selected_sample_id = [iaps_class.get_pic_id(i) for i in selected_sample_idx]
# get sample idx from list of pic_id
sample_idx_list = []
for pic_id in selected_sample_id:
    sample_idx_list = sample_idx_list + iaps_class.get_sample_idx(pic_id)

utils.save_object(sample_idx_list,'selected_idx_list.pkl')

#%% plot one subject
fig = illum_lux_df.reset_index(drop=True).iplot(kind='scatter',mode='lines',
                                 title='recorded illum',
                                 xTitle='picIndex', yTitle= 'illuminance (LUX)',
                                 asFigure=True)
plotly.offline.plot(fig)

#%%
#iaps_class = iaps(r"C:\Users\DSPLab\Research\affective-monitor-model\preprocessing")
iaps_class = iaps(r"E:\Research\affective-monitor-model\preprocessing")
iaps_df = iaps_class.iaps_df
report_df = iaps_df.drop(columns=['testsubject_idx','file_name'])
report_df = report_df.set_index('pic_idx')
report_df['illum_record'] = mean_pic
report_df['illum_record_manually'] = ill_lux_manual_df['illum_lux']
report_df['illum_from_gimp'] = ill_lux_manual_df['illum_gimp']
report_df['arousal_sbj_rate_avg'] = arousals_avg_col
report_df['arousal_sbj_rate_std'] = arousal_sbj_std
report_df.to_excel('pd_report.xlsx')

#%% plot one subject
import pandas as pd
sc = StandardScaler()
illum_scaled = sc.fit_transform(ill_lux_manual_df.values)
illum_scaled_df = pd.DataFrame(illum_scaled)
illum_scaled_df.columns = ['lux','gimp']
fig = illum_scaled_df.reset_index(drop=True).iplot(kind='scatter',mode='lines',
                                 title='recorded illum',
                                 xTitle='picIndex', yTitle= 'illuminance (LUX)',
                                 asFigure=True)
plotly.offline.plot(fig)

#%%
#pickle_file = "data_1_50_toTensor.pkl"
#
#face_dataset = utils.load_object(pickle_file)
#subject_ids = [i for i in range(1,51)]
#utils.plot_subjects(subject_ids,plot='PD')
#utils.plot_subjects(subject_ids,plot='FAP')

#############---------- plot by testsubject ID -------------##########
#utils.plot_subjects([5,8,15],plot='PD')
#utils.plot_sample(face_dataset[19])
#utils.plot_FAP(face_dataset[3])

#############---------- plot multisample -------------##########
#utils.plot_multi_samples(1,70,plot="PD")
#utils.plot_multi_samples(1,70,plot="FAP")

#############---------- plot one sample -------------##########
#utils.plot_sample(face_dataset[19])
#utils.plot_FAP(face_dataset[90])
#utils.plot_PD(face_dataset[90])
