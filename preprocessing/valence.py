# -*- coding: utf-8 -*-
import os
#import utils
import numpy as np
import pandas as pd

def get_valence_df(path,subjects,source='iaps',fix=False,class_mode='default'):    
    # Loop through each Testsubject folder
    valence_df = pd.DataFrame()
    for i,elem in enumerate(subjects):
        filepath = os.path.join(path, "TestSubject"+str(elem)+"\\SAMrating.txt")
        valence_df_raw = pd.read_csv(filepath,header=1,delimiter=",",
                                  quotechar=";",
                                  usecols=['Valence_mean(IAPS)','Valence(rating)','PictureIndex'],
    #                              index_col="PicIndex",
                                  skipinitialspace=True)
        valence_df_raw = valence_df_raw.set_index('PictureIndex')
        valence_df_raw.columns = ['subject','iaps']        
        valence_df = valence_df.append(valence_df_raw)
        
    # replace nan value with iaps value   
    valence_df.subject.fillna(valence_df.iaps,inplace=True)
    if source == 'iaps':
        valence_df = valence_df.drop(columns=['subject'])
    elif source == 'subject':
        valence_df = valence_df.drop(columns=['iaps'])
    elif source == 'subject_avg':
        valence_df = valence_df.drop(columns=['iaps'])
        # find mean of arousal per picture
        for i in range(1,71):            
            valence_df.loc[i] = valence_df.loc[i].mean().values.tolist()[0]
    valence_df.columns = ['valence']
    # function to apply to each row if fix is True
    def convert_to_label(SAM,class_mode):
        # the first argument is an dataframe so make sure we choose column
        SAM = SAM['valence']
        scale = 1
        target_scale = scale*((SAM-5)/4)
        if class_mode=='three':
            if target_scale < -0.125:
                target_scale = 3
            elif -0.125 <= target_scale <= 0.125:
                target_scale = 2
            elif 0.125 < target_scale:
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
        valence_df['adjust'] = valence_df.apply(convert_to_label,class_mode=class_mode,axis=1)
        valence_df = valence_df.drop(columns=['valence'])
        valence_df.columns = ['valence']
    return valence_df
