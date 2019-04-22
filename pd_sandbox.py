# -*- coding: utf-8 -*-
import preprocessing.pd as ppd
import utils

# get samples
pd_signals = ppd.get_pds()

#depth_signals = ppd.get_depths()
illum_signals = ppd.get_illums()

#sample_idx = 9
#ppd.plot_compare_sample(pd_signals[sample_idx],title='Pupil Diameter')
#ppd.plot_compare_sample(depth_signals[sample_idx],title='Depth')
#ppd.plot_compare_sample(illum_signals[sample_idx],title='Illuminance')

# get arousal
arousals = ppd.get_arousal(fix=True)
# remove glitch
pd_signals, _ = ppd.remove_glitch(pd_signals,threshold=0.2)


# find missing percentage list
missing_percentage = ppd.get_missing_percentage(pd_signals)
# normalize and select samples
selected_samples = ppd.select_and_clean(pd_signals,norm=True,
                                        miss_percent=missing_percentage,
                                        miss_threshold=0.25,
                                        label=arousals,
                                        sd_detect_remove=True,
                                        align=True)

PLR_removed_samples, _, _ = ppd.remove_PLR(selected_samples,
                                           illum_signals,
                                           n=10,
                                           mu=0.000000005,
                                           showFigures=[3,1800,50],
                                           arousal_col=True)

# plot figures to pdf
#figs = ppd.plot_pd_overlap_df(selected_samples.drop(columns=['ori_idx_row']),subjects=[i for i in range(1,51)])
#utils.print_pdf(figs,"sd_remove")
#
# slice to get area of interest
samples_aoi = ppd.get_aoi_df(PLR_removed_samples,start=10,stop=50)
## plot figures to pdf
figs = ppd.plot_pd_overlap_df(samples_aoi,subjects=[i for i in range(1,51)])
#utils.print_pdf(figs,"everthing_0_40")


# find stat of aoi signals
samples = ppd.generate_features_df(samples_aoi)
print('Total amount of samples: '+str(samples.shape))

# save to pickle
utils.save_object(samples,'pd_for_train.pkl')

## remove PLR
#pd_PLR_removed = []
#for pd, illum in zip(pd_signals,illum_signals):
#    pd_PLR_removed.append(ppd.remove_PLR(pd,illum,10,0.5,norm=False))
#
#
## visualize
#ppd.plot_compare_sample(pd_signals[sample_idx],title='Pupil Diameter')
#ppd.plot_compare_sample(pd_PLR_removed[sample_idx],title='remove PLR')



#face_dataset = utils.load_object("data_1_50_fixPD_Label_False.pkl")
## 583 is a good one
##sample_idx = 583,640
#sample_idx = 1000
#pd_signal = face_dataset[sample_idx]['PD_avg_filtered']
##
## np diff
#processed_pd = ppd.differentiator(pd_signal)
### np gradient
##processed_pd = ppd.gradient(pd_signal)
#ppd.plot_pd_before_after(face_dataset[sample_idx],processed_pd,adjust=False)
##
## smooth curve
#processed_pd = ppd.savgol(pd_signal,11,4)
#ppd.plot_pd_before_after(face_dataset[sample_idx],processed_pd,adjust=False)
#
## detect glitch
#processed_pd, missing_percentage, glitch_index = ppd.detect_glitch(pd_signal,threshold=0.1)
#ppd.plot_pd_before_after(face_dataset[sample_idx],processed_pd,adjust=False,glitch_index=glitch_index)

# missing percentage
#missing_percentage = ppd.get_missing_percentage(pd_signal)

## plot overlap
#ppd.plot_pd_overlap([1,2])
#
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