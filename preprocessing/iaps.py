# -*- coding: utf-8 -*-
import pandas as pd
import preprocessing.illum as pill

class iaps(object):
    
    def __init__(self,filepath):
        self.filepath = filepath
        self.iaps_df = self.get_iaps_data()
        self.iaps_df['pic_idx'] = self.iaps_df['pic_idx'].apply(lambda x:x-1)

    def get_iaps_data(self):
        filepath = self.filepath+'\\IAPSinfoFile_Final.txt'
        iaps_df = pd.read_csv(filepath,header=None)
        iaps_df.columns = ['pic_id','pic_idx','testsubject_idx','valence_m','arousal_m','valence_std','arousal_std','file_name']
        return iaps_df
    
    def get_pic_id(self,sample_idx):
        idx = sample_idx%70
        return self.iaps_df.loc[idx]['pic_id']
    
    def get_sample_idx(self,pic_id):
        idx = self.iaps_df[self.iaps_df['pic_id']==pic_id]['pic_idx'].values[0]
        return [i*70+idx for i in range(51)]

    def get_feeling(self,feeling):
        """
            return sample index corresponding to the group of feeling
        """
        filepath = self.filepath+'\\IAPS_selectedList_Final.csv'
        feel_df = pd.read_csv(filepath,index_col=0)
        return feel_df
    
    

