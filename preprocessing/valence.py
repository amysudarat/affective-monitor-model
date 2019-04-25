# -*- coding: utf-8 -*-
import os
import utils
import numpy as np
import pandas as pd

def get_faps(path,subjects,source='iaps',fix=False,class_mode='default'):    
    # Loop through each Testsubject folder
    valence_df = pd.DataFrame()
    for i,elem in enumerate(subjects):
        filepath = os.path.join(path, "TestSubject"+str(elem)+"\\SAMrating.txt")
        valence_df_raw = pd.read_csv(filepath,header=1,delimiter=",",
                                  quotechar=";",
                                  usecols=['Valence_mean(IAPS)','Valence(rating)','PictureIndex'],
    #                              index_col="PicIndex",
                                  skipinitialspace=True)
        valence_df_raw = valence_df_raw.set_index('PicIndex')
        if i==0:
            valence_df = valence_df
            valence_df.columns = [elem]
        else:
            valence_df[elem] = valence_df_raw
    
    
    
    
    
    
    
    
    
    
    face_dataset = utils.load_object(pickle_file)
    
    array_samples = []
    
    def convert_to_label(SAM,three_class):
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
    if source == 'iaps':
        for i in range(len(face_dataset)):
            sample = face_dataset[i]['valence']
            if fix:
                sample = convert_to_label(sample,class_mode)
            array_samples.append(sample)
        array_samples = np.array(array_samples)
    else:
        pass
    
    return array_samples
