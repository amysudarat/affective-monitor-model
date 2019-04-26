# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

#%%
def get_illum_df(path,subjects):    
    # Loop through each Testsubject folder
    illum_df = pd.DataFrame()
    for i,elem in enumerate(subjects):
        filepath = os.path.join(path, "TestSubject"+str(elem)+"\\FAP.txt")
        illum_df_raw = pd.read_csv(filepath,header=1,delimiter=",",
                                  quotechar=";",
                                  usecols=['Illuminance','PicIndex'],
    #                              index_col="PicIndex",
                                  skipinitialspace=True)
        illum_df_raw = illum_df_raw.set_index('PicIndex')
        if i==0:
            illum_df = illum_df_raw
            illum_df.columns = [elem]
        else:
            illum_df[elem] = illum_df_raw
    return illum_df

#%%
def get_mean(illum_df):
    mean_df = pd.DataFrame()
    column_name = ['D'+str(i) for i in range(1,len(illum_df.columns)+1)]
    for i in range(1,illum_df.index.max()+1):    
        mean = illum_df.loc[i].mean().values
        sample_length = illum_df.loc[i].shape[0]
        # create temporary mean df to concat to the original df
        mean = np.tile(mean.transpose(),(sample_length,1))
        mean = pd.DataFrame(mean)
        mean.columns = column_name
        idx = [i for j in range(sample_length)]
        mean['index'] = idx
        mean = mean.set_index('index')
        if i == 1:
            mean_df = mean        
        else:
            mean_df = mean_df.append(mean)
    illum_df = pd.concat([illum_df,mean_df],axis=1)
    return illum_df