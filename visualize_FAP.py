# -*- coding: utf-8 -*-

import utils
import numpy as np
import matplotlib.pyplot as plt

# load object if pickle file already exists
face_dataset = utils.load_object("data_1_50_fixPD_Label_False.pkl")

FAP_sample = face_dataset[5]
#FAP_sample_numpy = np.array(FAP_sample)

def plot_FAP_temporal(sample):
    valence = sample['valence']
    arousal = sample['arousal']
    sample = sample['faceFAP']
    sample = np.array(sample)
    FAP_index = ['l_i_eyebrow_y','r_i_eyebrow_y','l_o_eyebrow_y','r_o_eyebrow_y',
                 'l_i_eyebrow_x','r_i_eyebrow_x','t_l_eyelid_y','t_r_eyelid_y',
                 'l_cheeck_y','r_cheeck_y','l_nose_x','r_nose_x',
                 'l_o_cornerlip_y','r_o_cornerlip_y','l_o_cornerlip_x','r_o_cornerlip_x',
                 'l_b_midlip_y','l_t_midlip_y','open_jaw']
    
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(14, 12))
    
    for i, ax in enumerate(axes.flatten()):
        if i == 19:
            break
        ax.plot(sample[:,i])
        ax.set_title(FAP_index[i])
        
        
    fig.suptitle("Arousal: "+ str(arousal) + " , Valence: "+ str(valence))
    plt.show()

###### ---------- Plot one sample ---------############

plot_FAP_temporal(face_dataset[9])

##### ----------- Plot every samples ------###########

for i in range(len(face_dataset)):
    plot_FAP_temporal(face_dataset[i])