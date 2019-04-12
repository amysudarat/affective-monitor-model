# -*- coding: utf-8 -*-

import preprocessing.pd as ppd
import matplotlib.backends.backend_pdf
#import matplotlib.pyplot as plt
import utils


face_dataset = utils.load_object("data_1_50_fixPD_Label_False.pkl")
# 583 is a good one
#sample_idx = 583,640
sample_idx = 583
pd_signal = face_dataset[sample_idx]['PD_avg_filtered']

# np diff
processed_pd = ppd.differentiator(pd_signal)
# np gradient
processed_pd = ppd.gradient(pd_signal)
ppd.plot_pd_before_after(face_dataset[sample_idx],processed_pd)

# smooth curve
processed_pd = ppd.savgol(pd_signal,11,4)
ppd.plot_pd_before_after(face_dataset[sample_idx],processed_pd,adjust=False)

# detect glitch
glitch_index, processed_pd = ppd.detect_glitch(pd_signal,threshold=0.3)
ppd.plot_pd_before_after(face_dataset[sample_idx],processed_pd,adjust=False,glitch_index=glitch_index)

# plot overlap
#ppd.plot_pd_overlap([1,2])

#subjects = [i for i in range(1,51)]
# 
#figs = ppd.plot_pd_overlap(subjects,fix_pd=False)
#
## save figures to pdf file
#pdf = matplotlib.backends.backend_pdf.PdfPages("pdf/plot_pd_overlap_not_fix_pd.pdf")
#i = 0
#for fig in figs: ## will open an empty extra figure :(
#    pdf.savefig( fig )
#    i+=1
#    print("printing: "+str(i))
#pdf.close()  