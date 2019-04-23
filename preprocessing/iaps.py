# -*- coding: utf-8 -*-
import pandas as pd

class iaps(object):
    
    def __init__(self,filepath):
        self.filepath = filepath
        self.iaps_df = self.get_iaps_data()
        self.iaps_df['pic_idx'] = self.iaps_df['pic_idx'].apply(lambda x:x-1)


    def get_iaps_data(self):
        filepath = self.filepath
        iaps_df = pd.read_csv(filepath,header=None)
        iaps_df.columns = ['pic_id','pic_idx','testsubject_idx','arousal_m','arousal_std','valence_m','valence_std','file_name']
        return iaps_df
    
    def get_pic_id(self,sample_idx):
        idx = sample_idx%70
        return self.iaps_df.loc[idx]['pic_id']
    
    def get_sample_idx(self,pic_id):
        idx = self.iaps_df[self.iaps_df['pic_id']==pic_id]['pic_idx'].values[0]
        return [i*70+idx for i in range(50)]
