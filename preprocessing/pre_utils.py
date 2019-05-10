# -*- coding: utf-8 -*-

def match_with_sample(original_df,filtered_idx):
    # check if index of arousal_df is in filtered_idx list or not
    original_df = original_df.reset_index(drop=True)
    output = original_df[original_df.index.isin(filtered_idx.tolist())]
    return output.reset_index(drop=True)

def match_label_with_sample(data_df,label_list,col_name='label'):
    ori_idx_col = data_df['ori_idx'].tolist()
    label_sel_list = []
    for idx in ori_idx_col:
        label_sel_list.append(label_list[idx])
    idx_col = data_df.index
    data_df = data_df.reset_index(drop=True)
    data_df[col_name] = label_sel_list
    data_df.index = idx_col
    
    return data_df

def match_illum_with_sample(original_df,illum_list):
    
#    idx_col = original_df.index
    original_df = original_df.reset_index(drop=True)
    original_df['illum'] = original_df['ori_idx']
    ill_list = original_df['ori_idx'].apply(lambda x:illum_list[x%70]).tolist()
#    original_df.index = idx_col
    
    return ill_list
