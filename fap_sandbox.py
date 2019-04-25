#%%

import preprocessing.fap as pfap
import preprocessing.valence as pval
import utils
import numpy as np

#%%
path = "C:\\Users\\DSPLab\\Research\\ExperimentData"
#path = "E:\\Research\\ExperimentData"
n = 50
subjects = [i for i in range(1,n+1)]

#%% load valence
valence_df = pval.get_valence_df(path,subjects,fix=True,class_mode='two')



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