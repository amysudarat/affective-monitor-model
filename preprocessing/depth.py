# import
import os
import pandas as pd
import numpy as np

#%%
def get_depth_df(path,subjects):    
    # Loop through each Testsubject folder
    depth_df = pd.DataFrame()
    for i,elem in enumerate(subjects):
        filepath = os.path.join(path, "TestSubject"+str(elem)+"\\FAP.txt")
        depth_df_raw = pd.read_csv(filepath,header=1,delimiter=",",
                                  quotechar=";",
                                  usecols=['Depth','PicIndex'],
    #                              index_col="PicIndex",
                                  skipinitialspace=True)
        depth_df_raw = depth_df_raw.set_index('PicIndex')
        if i==0:
            depth_df = depth_df_raw
            depth_df.columns = [elem]
        else:
            depth_df[elem] = depth_df_raw
    return depth_df

#%%
def get_mean(depth_df):
    mean_df = pd.DataFrame()
    column_name = ['D'+str(i) for i in range(1,len(depth_df.columns)+1)]
    for i in range(1,depth_df.index.max()+1):    
        mean = depth_df.loc[i].mean().values
        sample_length = depth_df.loc[i].shape[0]
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
    depth_df = pd.concat([depth_df,mean_df],axis=1)
    return depth_df

#%%
def get_min_depth(depth_df):
    min_list = []    
    for col in range(1,len(depth_df.columns)+1):
        min_list = min_list + [depth_df[col].min() for i in range(70)]
    return min_list

























