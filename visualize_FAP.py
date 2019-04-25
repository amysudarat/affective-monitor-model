#%%
import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import preprocessing.fap as pfap

#%% fap sandbox
faps_df = pfap.get_faps()

#%%
# save to pickle
utils.save_object(faps_df,'faps.pkl')

#%%

# load object if pickle file already exists
face_dataset = utils.load_object("data_1_50_fixPD_Label_False.pkl")

#FAP_sample = face_dataset[5]
#FAP_sample_numpy = np.array(FAP_sample)
#%%
def plot_FAP_temporal(sample,sample_idx=None):
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
        
    if sample_idx is None:
        fig.suptitle("Arousal: "+ str(arousal) + " , Valence: "+ str(valence))
    else:
        fig.suptitle("Sample No.: "+ str(sample_idx)+" , Arousal: "
                     + str(arousal) + " , Valence: "+ str(valence))
#    plt.show()
    return fig
#%%
###### ---------- Plot one sample ---------############
plot_FAP_temporal(face_dataset[9])
#%%
##### ----------- Plot every samples ------###########

figs = []
for i in range(1600,2400):
#for i in range(len(face_dataset)):    
    figs.append(plot_FAP_temporal(face_dataset[i],sample_idx=i))
    print(i)

# save figures to pdf file
pdf = matplotlib.backends.backend_pdf.PdfPages("pdf/plotFAP_1600_2399.pdf")
i = 0
for fig in figs: ## will open an empty extra figure :(
    pdf.savefig( fig )
    i+=1
    print("printing: "+str(i))
pdf.close()    
#%%    




 