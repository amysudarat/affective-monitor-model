# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import preprocessing.pd as ppd
import utils
# get samples
pd_signals = ppd.get_pds()
arousals = ppd.get_arousal(fix=False)

# remove glitch
pd_signals, _ = ppd.remove_glitch(pd_signals,threshold=0.3)
# find missing percentage list
missing_percentage = ppd.get_missing_percentage(pd_signals)
# normalize and select samples
selected_samples = ppd.select_and_clean(pd_signals,norm=True,
                                        miss_percent=missing_percentage,
                                        miss_threshold=0.25,
                                        label=arousals,
                                        sd_detect_remove=True)

# slice to get area of interest
samples_aoi = ppd.get_aoi_df(selected_samples,start=20,stop=70)

# find stat of aoi signals
samples = ppd.generate_features_df(samples_aoi)
print('Total amount of samples: '+str(samples.shape))

# save to pickle
utils.save_object(samples,'pd_for_train.pkl')