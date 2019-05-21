#%% get data
import utils
import preprocessing.fap as pfap
#path = "C:\\Users\\DSPLab\\Research\\ExperimentData"
##path = "E:\\Research\\ExperimentData"
#n = 51
#subjects = [i for i in range(1,n+1)]

#faps_df = pfap.get_faps_df(pickle_file='data_1_51.pkl')
faps_np_df = pfap.get_faps_np_df(pickle_file='data_1_51.pkl')

#%% find missing percentage
missing_percentage_list = pfap.get_missing_percentage(faps_np_df)
faps_filtered = pfap.faps_preprocessing(faps_np_df,
                                        aoi=[5,65],
                                        smooth=True,
                                        filter_miss=missing_percentage_list,
                                        fix_scaler='standard',
                                        sm_wid_len=21,
                                        center_mean=True)

#%% remove calm
#faps_remove_calm = pfap.calm_detector(faps_filtered,thres=1,remove=True)
#pfap.faps_slide_plot(faps_remove_calm,51,label=False,peak_plot=False)

#%% get peak
faps_peak_df = pfap.get_peak(faps_filtered,
                             mode='peak',
                             min_dist=10,
                             thres=0.7)
#pfap.faps_slide_plot(faps_peak_df,49,label=False,peak_plot='peak_sel',plot_sig=None)
#pfap.dir_vector_slide_plot(faps_peak_df,51,label=False)

#%% prepare df for training
#import numpy as np
#import pandas as pd
#AU_list = faps_peak_df['AU'].tolist()
#AU_np = np.array(AU_list)
#AU_df = pd.DataFrame(AU_np)
#AU_df['ori_idx'] = faps_peak_df['ori_idx'].reset_index(drop=True)

#%% faps_au_df
import preprocessing.pre_utils as pu
faps_au_df = faps_peak_df[['AU1','AU2','AU4','AU5','AU6','AU9','AU10','AU12','AU15','AU16','AU20','AU23','AU26','ori_idx']]
valence = utils.load_object('valence.pkl')
valence_list = valence['valence'].tolist()
faps_au_df = pu.match_label_with_sample(faps_au_df,valence_list)
f = faps_au_df.copy()
f = f.reset_index(drop=True)
#%% 1 is pleasure 2 displeasure
import pandas as pd
# feeling observation
f1 = f[(f['AU1']==1) & (f['label']==1)]
f2 = f[(f['AU2']==1) & (f['label']==1)]
f4 = f[(f['AU4']==1) & (f['label']==2)]
f5 = f[(f['AU5']==1) & (f['label']==1)]
f6 = f[(f['AU6']==1) & (f['label']==1)]
f9 = f[(f['AU9']==1) & (f['label']==2)]
f12 = f[(f['AU12']==1) & (f['label']==2)]
f15 = f[(f['AU15']==1) & (f['label']==2)]
f16 = f[(f['AU16']==1) & (f['label']==2)]
f20 = f[(f['AU20']==1) & (f['label']==2)]

samples = pd.concat([f1,f2,f4,f5,f6,f9,f12,f15,f16,f20],ignore_index=True)
samples = samples.sample(frac=1)
samples = samples.drop('label',axis=1)
import matplotlib.pyplot as plt
plt.hist(f['label'])
#%% save to pickle
utils.save_object(samples,'faps_for_train.pkl')

#%% slide plot
#import matplotlib.pyplot as plt
#
#def faps_slide_plot(faps_feat_df,sbj,label=False):
#    if sbj != 'all':
#        faps_feat_df = faps_feat_df[faps_feat_df['sbj_idx']==sbj] 
#    
#    # prepare faps that will be plotted
#    faps = faps_feat_df['faps'].tolist()
#    peaks = faps_feat_df['peak_pos'].tolist()
#    try:
#        p_selects = faps_feat_df['p_sel'].tolist()
#        p_lbs = faps_feat_df['p_lb'].tolist()
#        p_rbs = faps_feat_df['p_rb'].tolist()
#    except:
#        pass
#    if label:
#        labels = faps_feat_df['label'].tolist()
#    # slide show
#    for i in range(len(faps)):
#        plt.figure()
#        try:
#            for col in range(faps[i].shape[1]):
#                plt.plot(faps[i][:,col])
#        except:
#            plt.plot(faps[i])
#        try:
#            for p in peaks[i]:
#                plt.axvline(p,color='black',lw=1)
#            plt.axvline(p_selects[i],color='black',lw=3)
#            plt.axvline(p_lbs[i],color='black',lw=3)
#            plt.axvline(p_rbs[i],color='black',lw=3)
#        except:
#            plt.axvline(peaks[i],color='black',lw=3)           
#        if label:
#            plt.title(str(labels[i]))
#        plt.show()
#        plt.waitforbuttonpress()
#        plt.close()
#    return
#
#faps_slide_plot(faps_peak_sel_df,42)  
#
##%% visualize sandbox
## generate picture id
#iaps_class = iaps(r"C:\Users\DSPLab\Research\affective-monitor-model\preprocessing")
##iaps_class = iaps(r"E:\Research\affective-monitor-model\preprocessing")
#samples_idx = iaps_class.get_sample_idx(2070)
#
## get samples based on pic_id
#faps_selected = faps_filtered[faps_filtered['ori_idx'].isin(samples_idx)]
#
#
##%%
## Standard plotly imports
#import plotly
#import plotly.plotly as py
#import plotly.graph_objs as go
#from plotly.offline import iplot, init_notebook_mode
#import plotly.figure_factory as ff
## Using plotly + cufflinks in offline mode
#import cufflinks
#cufflinks.go_offline(connected=True)
#init_notebook_mode(connected=True)
#
#
##%% check by visualize
#plot_df = pd.DataFrame(faps_filtered.loc[3506]['faps'])
#fig = plot_df.reset_index(drop=True).iplot(kind='scatter',mode='lines',
#                                 title='FAPS',
#                                 xTitle='frame', yTitle= 'FAP changes',
#                                 asFigure=True)
#plotly.offline.plot(fig)
#
#
#
#
