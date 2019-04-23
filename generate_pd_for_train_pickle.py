# -*- coding: utf-8 -*-
import preprocessing.pd as ppd
import utils
from preprocessing.iaps import iaps
#iaps_class = iaps(r"C:\Users\DSPLab\Research\affective-monitor-model\preprocessing\IAPSinfoFile_Final.txt")
iaps_class = iaps(r"E:\Research\affective-monitor-model\preprocessing\IAPSinfoFile_Final.txt")

# get samples
pd_signals = ppd.get_pds()
arousals = ppd.get_arousal(fix=True)
illum_signals = ppd.get_illums()
sample_list_from_pic_id = iaps_class.get_sample_idx(2141)
# remove glitch
pd_signals, _ = ppd.remove_glitch(pd_signals,threshold=0.1)
# find missing percentage list
missing_percentage = ppd.get_missing_percentage(pd_signals)
selected_samples = ppd.select_and_clean(pd_signals,norm=True,
                                        miss_percent=missing_percentage,
                                        miss_threshold=0.25,
                                        label=arousals,
                                        sd_detect_remove=True,
                                        align=False)

# set that seems work: n=10, mu=0.00000085
# set that seems work: n=5, mu=0.00000095
remove_PLR = True
if remove_PLR:
    illum_select_df = selected_samples.copy()
    illum_select_df['idx'] = illum_select_df.reset_index(drop=True).index
    illum_select_list = illum_select_df[illum_select_df['ori_idx_row'].isin(sample_list_from_pic_id)]['idx'].tolist()
                            
    final_samples, _, _ = ppd.remove_PLR(selected_samples,
                                               illum_signals,
                                               n=5,
                                               mu=0.0000015,
                                               showFigures=illum_select_list,
                                               arousal_col=True)
else:
    final_samples = selected_samples

# slice to get area of interest
samples_aoi = ppd.get_aoi_df(final_samples,start=20,stop=50)

# find stat of aoi signals
samples = ppd.generate_features_df(samples_aoi)
print('Total amount of samples: '+str(samples.shape))

# save to pickle
utils.save_object(samples,'pd_for_train.pkl')