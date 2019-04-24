# -*- coding: utf-8 -*-
import utils
import numpy as np

def get_arousal(pickle_file="data_1_50_fixPD_Label_False.pkl",source='iaps',fix=False,class_mode='default'):
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
    if source == 'arousal':
        for i in range(len(face_dataset)):
            sample = face_dataset[i]['arousal']
            if fix:
                sample = convert_to_label(sample,class_mode)
            array_samples.append(sample)
        array_samples = np.array(array_samples)
    else:
        pass
    
    return array_samples
