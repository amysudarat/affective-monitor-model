# -*- coding: utf-8 -*-

import preprocessing.pd as ppd
#import matplotlib.backends.backend_pdf

# get samples
pd_signals = ppd.get_pds()

depth_signals = ppd.get_depths()
illum_signals = ppd.get_illums()
sample_idx = 1500

#ppd.plot_compare_sample(pd_signals[sample_idx],title='Pupil Diameter')
#ppd.plot_compare_sample(depth_signals[sample_idx],title='Depth')
#ppd.plot_compare_sample(illum_signals[sample_idx],title='Illuminance')
# remove glitch
pd_signals, missing_percentage = ppd.remove_glitch(pd_signals,threshold=0.2)

# select samples


## remove PLR
#pd_PLR_removed = []
#for pd, illum in zip(pd_signals,illum_signals):
#    pd_PLR_removed.append(ppd.remove_PLR(pd,illum,10,0.5,norm=False))


# visualize
ppd.plot_compare_sample(pd_signals[sample_idx],title='Pupil Diameter')
ppd.plot_compare_sample(pd_PLR_removed[sample_idx],title='remove PLR')



#face_dataset = utils.load_object("data_1_50_fixPD_Label_False.pkl")
## 583 is a good one
##sample_idx = 583,640
#sample_idx = 583
#pd_signal = face_dataset[sample_idx]['PD_avg_filtered']
#
## np diff
#processed_pd = ppd.differentiator(pd_signal)
## np gradient
#processed_pd = ppd.gradient(pd_signal)
#ppd.plot_pd_before_after(face_dataset[sample_idx],processed_pd)
#
## smooth curve
#processed_pd = ppd.savgol(pd_signal,11,4)
#ppd.plot_pd_before_after(face_dataset[sample_idx],processed_pd,adjust=False)
#
## detect glitch
#output, missing_percentage, glitch_index = ppd.detect_glitch(pd_signal,threshold=0.3)
#ppd.plot_pd_before_after(face_dataset[sample_idx],processed_pd,adjust=False,glitch_index=glitch_index)

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