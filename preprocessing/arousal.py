# -*- coding: utf-8 -*-
import os
#import utils
import numpy as np
import pandas as pd

 
def get_arousal_df(path,subjects,source='iaps',fix=False,class_mode='default'):    
    # Loop through each Testsubject folder
    arousal_df = pd.DataFrame()
    for i,elem in enumerate(subjects):
        filepath = os.path.join(path, "TestSubject"+str(elem)+"\\SAMrating.txt")
        arousal_df_raw = pd.read_csv(filepath,header=1,delimiter=",",
                                  quotechar=";",
                                  usecols=['Arousal_mean(IAPS)','Arousal(rating)','PictureIndex'],
    #                              index_col="PicIndex",
                                  skipinitialspace=True)
        arousal_df_raw = arousal_df_raw.set_index('PictureIndex')
        arousal_df_raw.columns = ['subject','iaps']        
        arousal_df = arousal_df.append(arousal_df_raw)
        
    # replace nan value with iaps value   
    arousal_df.subject.fillna(arousal_df.iaps,inplace=True)
    if source == 'iaps':
        arousal_df = arousal_df.drop(columns=['subject'])
    elif source == 'subject':
        arousal_df = arousal_df.drop(columns=['iaps'])
    elif source == 'subject_avg':
        arousal_df = arousal_df.drop(columns=['iaps'])
        # find mean of arousal per picture
        arousal_sbj_std = []
        for i in range(1,len(subjects)+1):
            arousal_sbj_std.append(arousal_df.loc[i].std().values.tolist()[0])
            arousal_df.loc[i] = arousal_df.loc[i].mean().values.tolist()[0]            
        for i in range(1,len(subjects)):
            arousal_sbj_std = arousal_sbj_std+arousal_sbj_std
    arousal_df.columns = ['arousal']
    if source == 'subject_avg':
        arousal_df['sbj_std'] = arousal_sbj_std
    # function to apply to each row if fix is True
    def convert_to_label(SAM,class_mode):
        # the first argument is an dataframe so make sure we choose column
        SAM = SAM['arousal']
        scale = 1
        target_scale = scale*((SAM-5)/4)
        if class_mode=='three':
            if target_scale < -0.15:
                target_scale = 3
            elif -0.15 <= target_scale <= 0.15:
                target_scale = 2
            elif 0.15 < target_scale:
                target_scale = 1
        elif class_mode=='two':
            if target_scale > 0:
                target_scale = 1
            else:
                target_scale = 2
        else:
            if -1.0 <= target_scale < -0.6:
                target_scale = 5
            elif -0.6 <= target_scale < -0.2:
                target_scale = 4
            elif -0.2 <= target_scale < 0.2:
                target_scale = 3
            elif 0.2 <= target_scale < 0.6:
                target_scale = 2
            elif 0.6 <= target_scale <= 1:
                target_scale = 1
        return target_scale   
    if fix:
        arousal_df['adjust'] = arousal_df.apply(convert_to_label,class_mode=class_mode,axis=1)
        arousal_df = arousal_df.drop(columns=['arousal'])
        arousal_df.columns = ['arousal']
   
    return arousal_df


    
    



