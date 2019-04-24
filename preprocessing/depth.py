
#%%
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
