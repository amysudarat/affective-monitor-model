#%%
import utils
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import preprocessing.valence as pval
import preprocessing.fap as pfap

#%%
# Standard plotly imports
import plotly
import plotly.plotly as py
import plotly.io as pio
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import plotly.figure_factory as ff
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

#%%
from sklearn.preprocessing import StandardScaler

#%%
#path = "C:\\Users\\DSPLab\\Research\\ExperimentData"
path = "E:\\Research\\ExperimentData"
n = 1
subjects = [i for i in range(1,n+1)]

#%% get data
faps_df = pfap.get_faps()
FAP_index = ['l_i_eyebrow_y','r_i_eyebrow_y','l_o_eyebrow_y','r_o_eyebrow_y',
                 'l_i_eyebrow_x','r_i_eyebrow_x','t_l_eyelid_y','t_r_eyelid_y',
                 'l_cheeck_y','r_cheeck_y','l_nose_x','r_nose_x',
                 'l_o_cornerlip_y','r_o_cornerlip_y','l_o_cornerlip_x','r_o_cornerlip_x',
                 'l_b_midlip_y','l_t_midlip_y','open_jaw']
faps_df.columns = FAP_index
valence_df = pval.get_valence_df(path,subjects,fix=True,class_mode='default')

#%%
# save to pickle
utils.save_object(faps_df,'fap.pkl')
utils.save_object(valence_df,'valence.pkl')

#%% in case we already save pickle load it from there
faps_df = utils.load_object('fap.pkl')
valence_df = utils.load_object('valence.pkl')

#%% Normalize scaler
scaler = StandardScaler(with_std=False)
scaler.fit(faps_df)
faps_scaled = scaler.transform(faps_df) 
FAP_index = ['l_i_eyebrow_y','r_i_eyebrow_y','l_o_eyebrow_y','r_o_eyebrow_y',
                 'l_i_eyebrow_x','r_i_eyebrow_x','t_l_eyelid_y','t_r_eyelid_y',
                 'l_cheeck_y','r_cheeck_y','l_nose_x','r_nose_x',
                 'l_o_cornerlip_y','r_o_cornerlip_y','l_o_cornerlip_x','r_o_cornerlip_x',
                 'l_b_midlip_y','l_t_midlip_y','open_jaw']
faps_scaled_df = pd.DataFrame(faps_scaled,columns=FAP_index,index=faps_df.index)


#%%
# plot all test subject
fig = faps_scaled_df.loc[1].reset_index(drop=True).iplot(kind='scatter',mode='lines',
                                 title='FAPS',
                                 xTitle='frame', yTitle= 'FAP changes',
                                 asFigure=True)
plotly.offline.plot(fig)
#pio.write_html(fig,'fap_plot/1_1.html')

#%% plot mulitple subplots to save in one html file
for sbj in subjects:
    traces = []
    # [0,70,140,...]
    start_idx = ((sbj*70)-70)+1
    # [70,140,210,...]
    stop_idx = (sbj*70)+1
    count=1
    for smp in range(start_idx,stop_idx):
        title = 'sample: '+str(smp)
        fig = faps_scaled_df.loc[smp].reset_index(drop=True).iplot(kind='scatter',mode='lines',
                                 title=title,
                                 xTitle='sequence', yTitle= 'FAP',
                                 asFigure=True)
        
        filename = 'sbj_'+str(sbj)+'_'+str(count)+'.pdf'
        pio.write_image(fig,'fap_plot/'+filename)
        count+=1
    utils.merge_pdf('fap_plot/all_sbj_'+str(sbj),'fap_plot/sbj_'+str(sbj)+'*.pdf')
    paths = glob.glob('fap_plot/sbj_'+str(sbj)+'*.pdf')    
    # remove all variable and then run this
    for elem in paths:
        os.remove(elem)
    

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




 