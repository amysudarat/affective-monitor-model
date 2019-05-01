# -*- coding: utf-8 -*-
import utils
import numpy as np
import preprocessing.illum as pill
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
filepath = r"C:\Users\DSPLab\Research\affective-monitor-model\preprocessing\illum_lux_meter_record.txt"
illum_lux_df = pill.get_illum_lux(filepath)
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
    

#%% plot one subject
fig = illum_lux_df.reset_index(drop=True).iplot(kind='scatter',mode='lines',
                                 title='recorded illum',
                                 xTitle='picIndex', yTitle= 'illuminance (LUX)',
                                 asFigure=True)
plotly.offline.plot(fig)

#%%
iaps_class = iaps(r"C:\Users\DSPLab\Research\affective-monitor-model\preprocessing")
iaps_df = iaps_class.iaps_df
report_df = iaps_df.drop(columns=['testsubject_idx','file_name'])
report_df = report_df.set_index('pic_idx')
report_df['illum_record'] = mean_pic
report_df['illum_record_manually'] = ill_lux_manual_df['illum_lux']
report_df['illum_from_gimp'] = ill_lux_manual_df['illum_gimp']
report_df.to_excel('report_with_gimp.xlsx')

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
