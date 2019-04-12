# -*- coding: utf-8 -*-

import preprocessing.pd as ppd
import utils


face_dataset = utils.load_object("data_1_50_fixPD_Label_False.pkl")
# 583 is a good one
#sample_idx = 583
sample_idx = 583
pd_signal = face_dataset[sample_idx]['PD_avg_filtered']
processed_pd = ppd.smooth_differentiator(pd_signal)
ppd.plot_pd_before_after(face_dataset[sample_idx],processed_pd)