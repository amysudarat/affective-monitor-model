# -*- coding: utf-8 -*-

def match_with_sample(original_df,filtered_idx):
    # check if index of arousal_df is in filtered_idx list or not
    original_df = original_df.reset_index(drop=True)
    output = original_df[original_df.index.isin(filtered_idx.tolist())]
    return output.reset_index(drop=True)

def match_illum_with_sample(original_df,illum_list):
    
    idx_col = original_df.index
    original_df = original_df.reset_index(drop=True)
    original_df['illum'] = original_df['ori_idx']
    original_df['illum'] = original_df['illum'].apply(lambda x:illum_list[x%70])
    original_df.index = idx_col
    
    return original_df
